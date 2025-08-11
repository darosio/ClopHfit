"""
Clophfit: Fitting of Cl- binding and pH titration curves.

This module provides a comprehensive suite of tools for analyzing titration data,
particularly for chloride binding and pH titration experiments common in biochemistry,
such as those involving fluorescent probes.

Core Functionality:
-------------------
1.  **Data Modeling**: Implements a 1-site binding model suitable for both
    ligand concentration and pH titrations.

2.  **Spectral Data Processing**:
    -   Processes raw spectral data (e.g., from fluorescence spectroscopy).
    -   Offers two methods for data reduction:
        -   Singular Value Decomposition (SVD) to extract the most significant
          spectral component.
        -   Band integration over a specified wavelength range.

3.  **Curve Fitting**: Provides three distinct fitting backends to determine the
    dissociation constant (K) and other parameters:
    -   **Least-Squares (LM)**: Utilizes the `lmfit` library for robust non-linear
      least-squares minimization. Supports iterative reweighting and outlier
      removal.
    -   **Orthogonal Distance Regression (ODR)**: Employs `scipy.odr` to account
      for uncertainties in both x and y variables, which is crucial when x-values
      (e.g., pH measurements) have errors.
    -   **Bayesian Modeling (PyMC)**: Implements a hierarchical Bayesian model
      using `pymc`. This approach is powerful for:
        -   Quantifying parameter uncertainties as full posterior distributions.
        -   Modeling errors in x-values as latent variables.
        -   Sharing information between multiple experiments (hierarchical fitting)
          to obtain more robust parameter estimates.

4.  **Result Visualization**: Includes extensive plotting functions to visualize:
    -   Raw and processed spectra.
    -   Fitted curves with confidence intervals.
    -   Diagnostic plots for SVD and Bayesian analyses (e.g., corner plots).
"""


# --- Data Structures for Fit Results ---
# TODO: from gemini

from __future__ import annotations

import copy
import logging
import typing
from dataclasses import dataclass, field
from sys import float_info

import lmfit  # type: ignore[import-untyped]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lmfit import Parameters
from lmfit.minimizer import Minimizer, MinimizerResult  # type: ignore[import-untyped]
from matplotlib import figure
from scipy import stats

from clophfit.clophfit_types import ArrayF
from clophfit.fitting.data_structures import DataArray, Dataset
from clophfit.fitting.errors import InsufficientDataError
from clophfit.fitting.models import binding_1site
from clophfit.fitting.plotting import (
    COLOR_MAP,
    PlotParameters,
    _create_spectra_canvas,
    plot_autovalues,
    plot_autovectors,
    plot_emcee_k_on_ax,
    plot_fit,
    plot_pca,
    plot_spectra,
    plot_spectra_distributed,
)

if typing.TYPE_CHECKING:
    from clophfit.clophfit_types import ArrayDict, ArrayF, ArrayMask


# --- Globals ---
N_BOOT = 20  # To compute fill_between uncertainty.

EMCEE_STEPS = 1800

logger = logging.getLogger(__name__)


### helpers
def _binding_1site_models(params: Parameters, x: ArrayDict, is_ph: bool) -> ArrayDict:
    """Compute models for the given input data and parameters."""
    models = {}
    for lbl, x_data in x.items():
        models[lbl] = binding_1site(
            x_data,
            params["K"].value,
            params[f"S0_{lbl}"].value,
            params[f"S1_{lbl}"].value,
            is_ph,
        )
    return models


def _init_from_dataset(ds: Dataset) -> tuple[ArrayDict, ArrayDict, ArrayDict, bool]:
    x = {k: da.x for k, da in ds.items()}
    y = {k: da.y for k, da in ds.items()}
    w = {
        k: 1 / da.y_err if da.y_err.size > 0 else np.ones_like(da.y)
        for k, da in ds.items()
    }
    return x, y, w, ds.is_ph


def _binding_1site_residuals(params: Parameters, ds: Dataset) -> ArrayF:
    """Compute concatenated residuals (array) for multiple datasets [weight = 1/sigma]."""
    x, y, w, is_ph = _init_from_dataset(ds)
    models = _binding_1site_models(params, x, is_ph)
    residuals: ArrayF = np.concatenate([(w[lbl] * (y[lbl] - models[lbl])) for lbl in x])
    return residuals


def _build_params_1site(ds: Dataset) -> Parameters:
    """Initialize parameters for 1 site model based on the given dataset."""
    params = Parameters()
    if ds.is_ph:
        params.add("K", min=3, max=11)
    else:
        # epsilon avoids x/K raise x/0 error
        params.add("K", min=float_info.epsilon)
    k_initial = []
    for lbl, da in ds.items():
        params.add(f"S0_{lbl}", value=da.y[0])
        params.add(f"S1_{lbl}", value=da.y[-1])
        target_y = (da.y[0] + da.y[-1]) / 2
        k_initial.append(da.x[np.argmin(np.abs(da.y - target_y))])
    params["K"].value = np.mean(k_initial)
    return params


@dataclass
class FitResult:
    """Result container for a deterministic fit.

    Attributes
    ----------
    figure : matplotlib.figure.Figure | None
        Matplotlib figure containing the fit plot, if generated.
    result : lmfit.minimizer.MinimizerResult | None
        Result of the optimization produced by lmfit.
    mini : lmfit.minimizer.Minimizer | None
        Minimizer instance used to run the fit.
    dataset : Dataset | None
        Dataset used for the fit (typically a deep copy of the input dataset).
    """

    figure: figure.Figure | None = None
    result: MinimizerResult | None = None
    mini: Minimizer | None = None
    dataset: Dataset | None = None

    def is_valid(self) -> bool:
        """Whether figure, result, and minimizer exist."""
        return (
            self.figure is not None
            and self.result is not None
            and self.mini is not None
        )


@dataclass
class SpectraGlobResults:
    """A dataclass representing the results of both svd and bands fits.

    Attributes
    ----------
    svd : FitResult | None
        The `FitResult` object representing the outcome of the concatenated svd
        fit, or `None` if the svd fit was not performed.
    gsvd : FitResult | None
        The `FitResult` object representing the outcome of the svd fit, or
        `None` if the svd fit was not performed.
    bands : FitResult | None
        The `FitResult` object representing the outcome of the bands fit, or
        `None` if the bands fit was not performed.
    """

    svd: FitResult | None = field(default=None)
    gsvd: FitResult | None = field(default=None)
    bands: FitResult | None = field(default=None)


def analyze_spectra(
    spectra: pd.DataFrame, is_ph: bool, band: tuple[int, int] | None = None
) -> FitResult:
    """Analyze spectra titration, create and plot fit results.

    This function performs either Singular Value Decomposition (SVD) or
    integrates the spectral data over a specified band and fits the integrated
    data to a binding model.

    Parameters
    ----------
    spectra : pd.DataFrame
        The DataFrame containing spectra (one spectrum for each column).
    is_ph : bool
        Whether the x values should be interpreted as pH values rather than
        concentrations.
    band : tuple[int, int] | None
        The band to integrate over. If None (default), performs Singular Value
        Decomposition (SVD).

    Returns
    -------
    FitResult
        A FitResult object containing the figure, result, and mini.

    Raises
    ------
    ValueError
        If the band parameters are not in the spectra's index when the band
        method is used.

    Notes
    -----
    Creates plots of spectra, principal component vectors, singular values, fit
    of the first principal component and PCA for SVD; only of spectra and fit
    for Band method.
    """
    y_offset = 1.0
    x = spectra.columns.to_numpy()
    fig, (ax1, ax2, ax3, ax4, ax5) = _create_spectra_canvas()
    plot_spectra(ax1, spectra, PlotParameters(is_ph))
    if band is None:  # SVD
        threshold = int(len(spectra) * 0.5)
        spectra = spectra.dropna(axis=1, thresh=threshold)
        spectra = spectra.dropna(axis=0)
        ddf = spectra.sub(spectra.iloc[:, 0], axis=0)
        u, s, v = np.linalg.svd(ddf)
        ds = Dataset({"default": DataArray(x, v[0, :] + y_offset)}, is_ph)
        plot_autovectors(ax2, spectra.index, u)
        plot_autovalues(ax3, s[:])  # don't plot last auto-values?
        plot_pca(ax5, v, x, PlotParameters(is_ph))
        ylabel = "First Principal Component"
        ylabel_color = COLOR_MAP(0)
    else:  # Band integration
        ini, fin = band
        if ini not in spectra.index and fin not in spectra.index:
            msg = f"Band parameters ({ini}, {fin}) are not in the spectra's index."
            raise ValueError(msg)
        # columns index name are not necessarily unique
        y = np.array(
            [spectra.iloc[:, i].loc[ini:fin].sum() for i in range(spectra.shape[1])]
        )
        # rescale y
        y /= np.abs(y).max() / 10
        ds = Dataset.from_da(DataArray(x, y), is_ph)
        ylabel = "Integrated Band Fluorescence"
        ylabel_color = (0.0, 0.0, 0.0, 1.0)  # "k"
    weight_multi_ds_titration(ds)
    fit_result = fit_binding_glob(ds)
    result = fit_result.result
    mini = fit_result.mini
    params = result.params if result else Parameters()
    plot_fit(ax4, ds, params, nboot=N_BOOT, pp=PlotParameters(is_ph))
    ax4.set_ylabel(ylabel, color=ylabel_color)
    return FitResult(fig, result, mini, ds)


def fit_binding_glob(ds: Dataset, robust: bool = False) -> FitResult:
    """Analyze multi-label titration datasets and visualize the results."""
    params = _build_params_1site(ds)
    if len(params) > len(np.concatenate([da.y for da in ds.values()])):
        raise InsufficientDataError
    mini = Minimizer(_binding_1site_residuals, params, fcn_args=(ds,), scale_covar=True)
    if robust:
        # Use Huber loss for robust fitting to reduce the influence of outliers
        result = mini.minimize(method="least_squares", loss="huber")
    else:
        result = mini.minimize()
    fig = figure.Figure()
    ax = fig.add_subplot(111)
    plot_fit(ax, ds, result.params, nboot=N_BOOT, pp=PlotParameters(ds.is_ph))
    return FitResult(fig, result, mini, copy.deepcopy(ds))


def weight_da(da: DataArray, is_ph: bool) -> bool:
    """Assign weights to each label based on individual label fitting."""
    success = True
    ds = Dataset.from_da(da, is_ph=is_ph)
    params = _build_params_1site(ds)
    # Too few data points (compared to number of parameters).
    if len(params) > len(da.y):
        success = False
        sem = 1.0 * np.ones_like(da.xc)
    else:
        mr = lmfit.minimize(_binding_1site_residuals, params, args=(ds,))
        # Calculate residuals SEM
        sem = np.std(mr.residual, ddof=1) / np.sqrt(len(mr.residual))
    da.y_err = sem
    return success


def weight_multi_ds_titration(ds: Dataset) -> None:
    """Assign weights to each label based on individual label fitting."""
    failed_fit_labels = []
    for lbl, da in ds.items():
        if not weight_da(da, ds.is_ph):
            failed_fit_labels.append(lbl)
    if failed_fit_labels:
        max_yerr = -np.inf
        for lbl in ds.keys() - set(failed_fit_labels):
            max_yerr = max(np.max(ds[lbl].y_err).item(), max_yerr)
        for lbl in failed_fit_labels:
            ds[lbl].y_err = np.array([max_yerr * 10])


def analyze_spectra_glob(
    titration: dict[str, pd.DataFrame],
    ds: Dataset,
    dbands: dict[str, tuple[int, int]] | None = None,
) -> SpectraGlobResults:
    """Analyze multi-label spectra visualize the results."""
    _gap_ = 1
    dbands = dbands or {}
    labels_svd = titration.keys() - dbands.keys()
    labels_bands = titration.keys() - labels_svd
    if len(labels_svd) > 1:
        # Concatenate spectra.
        prev_max = 0
        adjusted_list = []
        for lbl in labels_svd:
            spectra_adjusted = titration[lbl].copy()  # Avoid modifying original data
            spectra_adjusted.index += prev_max - spectra_adjusted.index.min() + _gap_
            prev_max = spectra_adjusted.index.max()
            adjusted_list.append(spectra_adjusted)
        spectra_merged = pd.concat(adjusted_list)
        svd = analyze_spectra(spectra_merged, ds.is_ph)
        ds_svd = ds.copy(labels_svd)
        weight_multi_ds_titration(ds_svd)  # Fixed from ds
        f_res = fit_binding_glob(ds_svd)
        fig = _plot_spectra_glob_emcee(titration, ds_svd, f_res)
        gsvd = FitResult(fig, f_res.result, f_res.mini)
    else:
        svd, gsvd = None, None
    if len(labels_bands) > 1:
        ds_bands = ds.copy(labels_bands)
        # FIX: Weight the dataset we are about to fit
        weight_multi_ds_titration(ds_bands)
        f_res = fit_binding_glob(ds_bands)
        fig = _plot_spectra_glob_emcee(titration, ds_bands, f_res, dbands)
        bands = FitResult(fig, f_res.result, f_res.mini)
    else:
        bands = None
    return SpectraGlobResults(svd, gsvd, bands)


def _plot_spectra_glob_emcee(
    titration: dict[str, pd.DataFrame],
    ds: Dataset,
    f_res: FitResult,
    dbands: dict[str, tuple[int, int]] | None = None,
) -> figure.Figure:
    fig, (ax1, ax2, ax3, ax4, ax5) = _create_spectra_canvas()
    fig.delaxes(ax1)
    fig.delaxes(ax2)
    fig.delaxes(ax3)
    pparams = PlotParameters(ds.is_ph)
    tit_filtered = {k: spec for k, spec in titration.items() if k in ds}
    plot_spectra_distributed(fig, tit_filtered, pparams, dbands)
    params = f_res.result.params if f_res.result else Parameters()
    plot_fit(ax4, ds, params, nboot=N_BOOT, pp=pparams)
    if f_res.mini:
        result_emcee = f_res.mini.emcee(
            steps=EMCEE_STEPS * 3, workers=8, burn=100, nwalkers=30, progress=False
        )
        plot_emcee_k_on_ax(ax5, result_emcee)
    return fig


def outlier2(
    ds: Dataset, key: str = "", threshold: float = 3.0, plot_z_scores: bool = False
) -> FitResult:
    """Remove outliers and reassign weights."""
    # Re-weight dataset
    fr = fit_binding_glob(ds, robust=True)
    if not fr.result:
        return FitResult()
    weighted_residuals = fr.result.residual
    weights = np.concatenate([1.0 / da.y_err for da in ds.values()])
    residuals = weighted_residuals / weights
    reweighted_ds = copy.deepcopy(ds)
    start_idx = 0
    for da in reweighted_ds.values():
        end_idx = start_idx + len(da.y)
        reduced_residual = fr.result.residual[start_idx:end_idx]
        residual = np.abs(reduced_residual) * da.y_err
        sigma = np.mean(np.abs(residual))
        sigma = max(sigma, 1e-3)  # Avoid division by zero
        da.y_errc = sigma * np.ones_like(da.xc)

    # Find outliers
    fr = fit_binding_glob(reweighted_ds, robust=True)
    if not fr.result:
        return FitResult()
    weighted_residuals = fr.result.residual
    weights = np.concatenate([1.0 / da.y_err for da in reweighted_ds.values()])
    # Calculate the absolute residuals
    residuals = weighted_residuals / weights
    z_scores = stats.zscore(residuals)
    if plot_z_scores:
        plt.scatter(range(len(z_scores)), z_scores)
        plt.axhline(y=threshold, color="r", linestyle="-")
        plt.axhline(y=-threshold, color="r", linestyle="-")
        plt.title("Z-scores")
    mask = np.abs(z_scores) < threshold
    n_outliers = mask.tolist().count(False)
    if n_outliers > 0:
        reweighted_ds.apply_mask(mask)
        logger.warning(f"outlier in {key}: {mask.astype(int)}.")
    return fit_binding_glob(reweighted_ds, robust=False)


def _reweight_from_residuals(ds: Dataset, residuals: ArrayF) -> Dataset:
    """Update y_err from residuals mean."""
    updated_ds = copy.deepcopy(ds)
    for i, da in enumerate(updated_ds.values()):
        len_x = len(da.y)
        label_residuals = residuals[i * len_x : (i + 1) * len_x]
        # residuals not masked to reduce weight of dataset with outliers
        sigma = np.mean(np.abs(label_residuals))
        # Avoid division by zero if all residuals are 0
        sigma = max(sigma, 1e-3)
        da.y_err = np.full(da.y.shape, sigma)
    return updated_ds


def fit_binding_glob_reweighted(
    ds: Dataset, key: str, threshold: float = 2.05
) -> FitResult:
    """RLS and outlier removal for multi-label titration datasets."""
    # Initial fit
    r = fit_binding_glob(ds)
    if r.dataset and r.result:
        start_idx = 0
        for lbl, (da0, da) in enumerate(
            zip(ds.values(), r.dataset.values(), strict=True), start=1
        ):
            end_idx = start_idx + len(da.y)
            residual = r.result.residual[start_idx:end_idx]  # reduced residues
            da.y_errc[da.mask] = np.abs(residual) * da0.y_err
            mask = outlier_glob(residual, threshold=threshold)
            n_outliers = mask.tolist().count(True)
            if n_outliers == 1:
                logger.warning(f"{n_outliers} outlier in {key}:y{lbl}.")
            elif n_outliers > 1:
                logger.warning(f"{n_outliers} outliers in {key}:y{lbl}.")
            da.mask[da.mask] = ~mask
            start_idx = end_idx
        return fit_binding_glob(r.dataset)
    return FitResult()


def fit_binding_glob_recursive(
    ds: Dataset, max_iterations: int = 15, tol: float = 0.1
) -> FitResult:
    """Analyze multi-label titration datasets using ODR."""
    # Initial fit
    r = fit_binding_glob(ds)
    residual_variance = r.result.redchi if r.result else 0.0

    rn = r  # new
    for _ in range(max_iterations):
        if r.dataset and r.result:
            start_idx = 0
            for da in r.dataset.values():
                end_idx = start_idx + len(da.y)
                da.y_errc[da.mask] = np.maximum(
                    np.abs(r.result.residual[start_idx:end_idx]), 0.01
                )
                start_idx = end_idx
            rn = fit_binding_glob(r.dataset)
        else:  # new
            break
        if rn.mini and rn.mini.minimize().redchi == 0:
            rn = r
            break
        # Check convergence
        if rn.result and residual_variance - rn.result.redchi < tol:
            break
        residual_variance = r.result.redchi if r.result else 0.0
        r = rn
    return rn


def fit_binding_glob_recursive_outlier(
    ds: Dataset, tol: float = 0.01, threshold: float = 3.0
) -> FitResult:
    """Analyze multi-label titration datasets using IRLS."""
    # Initial fit
    r = fit_binding_glob_recursive(ds, tol=tol)
    if r.result:
        mask = outlier_glob(r.result.residual, threshold)
    while mask.any() and r.dataset:
        ds.apply_mask(~mask)
        r = fit_binding_glob_recursive(ds, tol=tol)
        if r.result:
            mask = outlier_glob(r.result.residual, threshold)
    return r


def outlier_glob(
    residuals: ArrayF, threshold: float = 2.0, plot_z_scores: bool = False
) -> ArrayMask:
    """Identify outliers."""
    z_scores = np.abs((residuals - np.mean(residuals)) / np.std(residuals))
    if plot_z_scores:
        plt.scatter(range(len(z_scores)), z_scores)
        plt.axhline(y=threshold, color="r", linestyle="-")
        plt.title("Z-scores")
    outliers: ArrayMask = z_scores > threshold
    return outliers


@dataclass
class _Result:
    params: Parameters
