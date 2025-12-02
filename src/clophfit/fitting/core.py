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
from sys import float_info

import lmfit  # type: ignore[import-untyped]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lmfit import Parameters
from lmfit.minimizer import Minimizer  # type: ignore[import-untyped]
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

from .data_structures import FitResult, SpectraGlobResults

if typing.TYPE_CHECKING:
    from clophfit.clophfit_types import ArrayF, ArrayMask


# --- Globals ---
N_BOOT = 20  # To compute fill_between uncertainty.

EMCEE_STEPS = 1800

logger = logging.getLogger(__name__)


# --- LMfit Backend: Helpers and Fitting Functions ---


def _binding_1site_residuals(params: Parameters, ds: Dataset) -> ArrayF:
    """Compute concatenated residuals for multiple datasets for lmfit.

    Optimized version with reduced dictionary lookups and vectorized operations.
    """
    is_ph = ds.is_ph
    K = params["K"].value  # Extract once to avoid repeated access  # noqa: N806

    # Pre-allocate lists for better performance
    residuals_list: list[ArrayF] = []

    for lbl, da in ds.items():
        # Get parameter values once per label
        S0 = params[f"S0_{lbl}"].value  # noqa: N806
        S1 = params[f"S1_{lbl}"].value  # noqa: N806

        # Compute model and residuals in one go
        model = binding_1site(da.x, K, S0, S1, is_ph=is_ph)
        weight = 1 / da.y_err if da.y_err.size > 0 else np.ones_like(da.y)
        residuals_list.append(weight * (da.y - model))

    return np.concatenate(residuals_list)


def _build_params_1site(ds: Dataset) -> Parameters:
    """Initialize lmfit Parameters for a 1-site model from a Dataset.

    Optimized version with reduced list operations and vectorized computations.
    """
    params = Parameters()
    if ds.is_ph:
        params.add("K", min=3, max=11)
    else:
        # epsilon avoids x/K raise x/0 error
        params.add("K", min=float_info.epsilon)

    # Pre-allocate numpy array for better performance
    k_initial_guesses = np.empty(len(ds))

    for i, (lbl, da) in enumerate(ds.items()):
        params.add(f"S0_{lbl}", value=da.y[0])
        params.add(f"S1_{lbl}", value=da.y[-1])
        # Vectorized computation of target and halfway point
        target_y = (da.y[0] + da.y[-1]) * 0.5
        halfway_idx = np.argmin(np.abs(da.y - target_y))
        k_initial_guesses[i] = da.x[halfway_idx]

    params["K"].value = np.mean(k_initial_guesses)
    return params


def weight_da(da: DataArray, *, is_ph: bool) -> bool:
    """Estimate initial weights for a DataArray by fitting it individually.

    The standard error of the residuals from this initial fit is used as
    the uncertainty (`y_err`) for subsequent weighted fits.

    Parameters
    ----------
    da : DataArray
        The data array to be weighted.
    is_ph : bool
        Whether the titration is pH-based.

    Returns
    -------
    bool
        True if the weighting fit was successful, False otherwise.
    """
    ds = Dataset.from_da(da, is_ph=is_ph)
    params = _build_params_1site(ds)
    if len(params) > len(da.y):
        # Not enough data points to fit; assign default error
        da.y_err = np.ones_like(da.xc)
        return False
    mr = lmfit.minimize(_binding_1site_residuals, params, args=(ds,))
    # Calculate residuals SEM
    sem = np.std(mr.residual, ddof=1) / np.sqrt(len(mr.residual))
    da.y_err = sem if np.isfinite(sem) and sem > 0 else np.array([1.0])
    return True


def weight_multi_ds_titration(ds: Dataset) -> None:
    """Assign weights to all DataArrays within a Dataset.

    Iterates through each `DataArray` in the `Dataset`, calling `weight_da`
    to estimate `y_err`. For any `DataArray` where weighting fails (e.g., due
    to insufficient data), a fallback error is assigned based on the errors
    from successfully fitted arrays.

    Optimized version with reduced set operations and memory allocations.
    """
    failed_labels = []
    successful_max_yerr = -np.inf

    # Single pass: weight and track failures/successes
    for lbl, da in ds.items():
        if not weight_da(da, is_ph=ds.is_ph):
            failed_labels.append(lbl)
        else:
            # Track maximum error from successful fits
            max_err = np.max(da.y_err).item()
            successful_max_yerr = max(successful_max_yerr, max_err)

    # Assign fallback error to failed labels
    if failed_labels and successful_max_yerr > -np.inf:
        fallback_error = successful_max_yerr * 10
        for lbl in failed_labels:
            ds[lbl].y_err = np.full_like(ds[lbl].xc, fallback_error)


# --- Spectral Data Processing ---
def analyze_spectra(
    spectra: pd.DataFrame, *, is_ph: bool, band: tuple[int, int] | None = None
) -> FitResult[Minimizer]:
    """Analyze spectra titration, fit the data, and plot the results.

    This function performs either Singular Value Decomposition (SVD) or
    integrates spectra over a specified band.

    Parameters
    ----------
    spectra : pd.DataFrame
        The DataFrame containing spectra (one spectrum for each column).
    is_ph : bool
        Whether the x-axis represents pH.
    band : tuple[int, int] | None
        If provided, use the 'band' integration method. Otherwise, use 'svd'.

    Returns
    -------
    FitResult[Minimizer]
        An object containing the fit results and the summary plot.

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
        ds = Dataset({"default": DataArray(x, v[0, :] + y_offset)}, is_ph=is_ph)
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
        y = np.array([
            spectra.iloc[:, i].loc[ini:fin].sum() for i in range(spectra.shape[1])
        ])
        # rescale y
        y /= np.abs(y).max() / 10
        ds = Dataset.from_da(DataArray(x, y), is_ph=is_ph)
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


def fit_binding_glob(ds: Dataset, *, robust: bool = False) -> FitResult[Minimizer]:
    """Analyze multi-label titration datasets and visualize the results.

    Parameters
    ----------
    ds : Dataset
        Input dataset with x, y, and y_err for each label.
    robust : bool
        If True, use Huber loss for robust fitting (reduces outlier influence).

    Returns
    -------
    FitResult[Minimizer]

    Raises
    ------
    InsufficientDataError
        If there are not enough data points for the number of parameters.

    Notes
    -----
    Parameter uncertainties are scaled by sqrt(reduced_chi_sq) via lmfit's
    Minimizer(scale_covar=True), which improves coverage when errors are
    underestimated.
    """
    params = _build_params_1site(ds)
    if len(params) > len(np.concatenate([da.y for da in ds.values()])):
        msg = "Not enough data points for the number of parameters."
        raise InsufficientDataError(msg)
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
        svd = analyze_spectra(spectra_merged, is_ph=ds.is_ph)
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
    f_res: FitResult[Minimizer],
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
    ds: Dataset,
    key: str = "",
    threshold: float = 3.0,
    *,
    plot_z_scores: bool = False,
    error_model: str = "uniform",
) -> FitResult[Minimizer]:
    """Remove outliers and reassign weights.

    Parameters
    ----------
    ds : Dataset
        Input dataset.
    key : str
        Identifier for logging.
    threshold : float
        Z-score threshold for outlier detection.
    plot_z_scores : bool
        Whether to plot z-scores.
    error_model : str
        Error reweighting model: "uniform" assigns uniform errors per label,
        "shot-noise" rescales physical errors preserving relative structure.

    Returns
    -------
    FitResult[Minimizer]

    """
    # Initial robust fit
    fr = fit_binding_glob(ds, robust=True)
    if not fr.result:
        return FitResult()

    # Reweight dataset based on error model
    reweighted_ds = copy.deepcopy(ds)
    start_idx = 0
    for da0, da in zip(ds.values(), reweighted_ds.values(), strict=True):
        end_idx = start_idx + len(da.y)
        reduced_residual = fr.result.residual[start_idx:end_idx]
        residual = np.abs(reduced_residual) * da0.y_err

        if error_model == "shot-noise":
            # Rescale original errors (preserve relative structure)
            scale = np.mean(np.abs(reduced_residual))
            scale = max(scale, 1e-3)
            da.y_errc = da0.y_errc * scale
        else:  # "uniform"
            sigma = np.mean(np.abs(residual))
            sigma = max(sigma, 1e-3)
            da.y_errc = sigma * np.ones_like(da.xc)
        start_idx = end_idx

    # Find outliers
    fr = fit_binding_glob(reweighted_ds, robust=True)
    if not fr.result:
        return FitResult()
    weighted_residuals = fr.result.residual
    # Recover raw residuals for z-score calculation
    # Using raw residuals avoids false positives on accurate curves where
    # weighted residuals are all small and z-scoring amplifies minor deviations
    weights = np.concatenate([1.0 / da.y_err for da in reweighted_ds.values()])
    raw_residuals = weighted_residuals / weights
    z_scores = stats.zscore(raw_residuals)

    if plot_z_scores:
        plt.scatter(range(len(z_scores)), z_scores)
        plt.axhline(y=threshold, color="r", linestyle="-")
        plt.axhline(y=-threshold, color="r", linestyle="-")
        plt.title("Z-scores")
    mask = np.abs(z_scores) < threshold
    n_outliers = mask.tolist().count(False)
    if n_outliers > 0:
        reweighted_ds.apply_mask(mask)
        logger.warning("outlier in %s: %s.", key, mask.astype(int))
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
) -> FitResult[Minimizer]:
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
                logger.warning("%s outlier in %s:y%s.", n_outliers, key, lbl)
            elif n_outliers > 1:
                logger.warning("%s outliers in %s:y%s.", n_outliers, key, lbl)
            da.mask[da.mask] = ~mask
            start_idx = end_idx
        return fit_binding_glob(r.dataset)
    return FitResult()


def fit_binding_glob_recursive(
    ds: Dataset, max_iterations: int = 15, tol: float = 0.1
) -> FitResult[Minimizer]:
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
) -> FitResult[Minimizer]:
    """Analyze multi-label titration datasets using IRLS."""
    # Initial fit
    r = fit_binding_glob_recursive(ds, tol=tol)
    if r.result:
        mask = outlier_glob(r.result.residual, threshold=threshold)
    while mask.any() and r.dataset:
        ds.apply_mask(~mask)
        r = fit_binding_glob_recursive(ds, tol=tol)
        if r.result:
            mask = outlier_glob(r.result.residual, threshold=threshold)
    return r


def outlier_glob(
    residuals: ArrayF, *, threshold: float = 2.0, plot_z_scores: bool = False
) -> ArrayMask:
    """Identify outliers."""
    z_scores = np.abs((residuals - np.mean(residuals)) / np.std(residuals))
    if plot_z_scores:
        plt.scatter(range(len(z_scores)), z_scores)
        plt.axhline(y=threshold, color="r", linestyle="-")
        plt.title("Z-scores")
    outliers: ArrayMask = z_scores > threshold
    return outliers
