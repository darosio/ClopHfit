"""Fit Cl binding and pH titration."""

from __future__ import annotations

import copy
import logging
import typing
from dataclasses import dataclass, field
from sys import float_info

import arviz
import arviz as az
import lmfit  # type: ignore[import-untyped]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm  # type: ignore[import-untyped]
from lmfit import Parameters
from lmfit.minimizer import Minimizer, MinimizerResult  # type: ignore[import-untyped]
from matplotlib import axes, figure
from pymc import math as pm_math
from pytensor.tensor import as_tensor_variable
from scipy import odr, optimize, stats  # type: ignore[import-untyped]
from uncertainties import ufloat  # type: ignore[import-untyped]

from clophfit.clophfit_types import ArrayF
from clophfit.fitting.data_structures import DataArray, Dataset
from clophfit.fitting.errors import InsufficientDataError
from clophfit.fitting.models import binding_1site
from clophfit.fitting.plotting import (
    COLOR_MAP,
    PlotParameters,
    _apply_common_plot_style,
    _create_spectra_canvas,
    plot_autovalues,
    plot_autovectors,
    plot_emcee_k_on_ax,
    plot_pca,
    plot_spectra,
    plot_spectra_distributed,
)

if typing.TYPE_CHECKING:
    from clophfit.clophfit_types import ArrayDict, ArrayF, ArrayMask, FloatFunc
    from clophfit.prtecan import PlateScheme


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
    """A dataclass representing the results of a fit.

    Attributes
    ----------
    figure : mpl.figure.Figure
        A matplotlib figure object representing the plot of the fit result.
    result : MinimizerResult
        The minimizer result object representing the outcome of the fit.
    mini : Minimizer
        The Minimizer object used for the fit.
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


def plot_fit(
    ax: axes.Axes,
    ds: Dataset,
    params: Parameters,
    nboot: int = 0,
    pp: PlotParameters | None = None,
) -> None:
    """Plot residuals for each dataset with uncertainty."""
    _stretch = 0.05
    xfit = {
        k: np.linspace(da.x.min() * (1 - _stretch), da.x.max() * (1 + _stretch), 100)
        for k, da in ds.items()
    }
    yfit = _binding_1site_models(params, xfit, ds.is_ph)
    # Create a color cycle
    colors = [COLOR_MAP(i) for i in range(len(ds))]
    for (lbl, da), clr in zip(ds.items(), colors, strict=False):
        # Make sure a label will be displayed.
        label = lbl if (da.y_err.size == 0 and nboot == 0) else None
        # Plot data.
        if pp:
            ax.scatter(
                da.x,
                da.y,
                c=list(da.x),
                s=99,
                edgecolors=clr,
                label=label,
                vmin=pp.hue_norm[0],
                vmax=pp.hue_norm[1],
                cmap=pp.palette,
            )
        else:
            ax.plot(da.x, da.y, "o", color=clr, label=label)
        # Plot fitting.
        ax.plot(xfit[lbl], yfit[lbl], "-", color="gray")
        # Display label in error bar plot.
        if da.y_err.size > 0:
            xe = da.x_err if da.x_err.size > 0 else None
            ax.errorbar(
                da.x,
                da.y,
                yerr=da.y_err,
                xerr=xe,
                fmt=".",  # alternative to "none"
                label=lbl,
                color=clr,
                alpha=0.4,
                capsize=3,
            )
        if nboot:
            # Calculate uncertainty using Monte Carlo method.
            y_samples = np.empty((nboot, len(xfit[lbl])))
            rng = np.random.default_rng()
            for i in range(nboot):
                p_sample = params.copy()
                for param in p_sample.values():
                    # Especially stderr can be None in case of critical fitting
                    if param.value and param.stderr:
                        param.value = rng.normal(param.value, param.stderr)
                y_samples[i, :] = _binding_1site_models(p_sample, xfit, ds.is_ph)[lbl]
            dy = y_samples.std(axis=0)
            # Plot uncertainty.
            # Display label in fill_between plot.
            ax.fill_between(
                xfit[lbl], yfit[lbl] - dy, yfit[lbl] + dy, alpha=0.1, color=clr
            )
    ax.legend()  # UserWarning: No artists... in tests
    if params["K"].stderr:  # Can be None in case of critical fitting
        k = ufloat(params["K"].value, params["K"].stderr)
    else:
        k = f"{params['K'].value:.3g}" if params["K"].value else None
    title = "=".join(["K", str(k).replace("+/-", "±")])
    xlabel = "pH" if ds.is_ph else "Cl"
    _apply_common_plot_style(ax, f"LM fit {title}", xlabel, "")


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


# ODR ########################################################################
def format_estimate(
    value: float, error: float, significant_digit_limit: int = 5
) -> str:
    """Format the value and its associated error into "{value} ± {error}" string."""
    large_number = np.exp(significant_digit_limit)
    small_number = np.exp(-significant_digit_limit)
    significant_digits = max(0, -int(np.floor(np.log10(abs(error)))))
    use_scientific = (
        significant_digits > significant_digit_limit
        or abs(value) > large_number
        or abs(value) < small_number
    )
    formatted_value = (
        f"{value:.{significant_digits + 1}e}"
        if use_scientific
        else f"{value:.{significant_digits + 1}f}"
    )
    formatted_error = (
        f"{error:.{significant_digits + 1}e}"
        if use_scientific
        else f"{error:.{significant_digits + 1}f}"
    )
    return f"{formatted_value} ± {formatted_error}"


def generalized_combined_model(
    pars: list[float], x: ArrayF, dataset_lengths: list[int]
) -> ArrayF:
    """Handle multiple datasets with different lengths and masks."""
    start_idx = 0
    results = []
    # The shared K parameter is always the first parameter
    K = pars[0]  # noqa: N806
    for i, length in enumerate(dataset_lengths):
        end_idx = start_idx + length
        current_x = x[start_idx:end_idx]
        S0 = pars[1 + 2 * i]  # S0 for dataset i # noqa: N806
        S1 = pars[2 + 2 * i]  # S1 for dataset i # noqa: N806
        model_output = binding_1site(current_x, K, S0, S1, is_ph=True)
        results.append(model_output)
        start_idx = end_idx
    return np.concatenate(results)


@dataclass
class _Result:
    params: Parameters


def fit_binding_odr(fr: FitResult) -> FitResult:
    """Analyze multi-label titration datasets using ODR."""
    if fr.result is None or fr.dataset is None:
        return FitResult()
    params = fr.result.params
    ds = copy.deepcopy(fr.dataset)
    for da in ds.values():
        # even if da.y_err is set to [1,1,..] array y_errc remains []
        shot_factor = 1 + np.sqrt(np.abs(da.yc))
        da.y_err = da.y_errc * shot_factor if da.y_errc.size > 0 else 1.0 * shot_factor
    # Collect dataset lengths
    dataset_lengths = [len(da.y) for da in ds.values()]
    x_data, y_data, x_err, y_err = ds.concatenate_data()
    data = odr.RealData(x_data, y_data, sx=x_err, sy=y_err)
    # Initial parameters setup
    initial_params = [params["K"].value]
    for lbl in ds:
        initial_params.extend([params[f"S0_{lbl}"].value, params[f"S1_{lbl}"].value])

    # Define the combined model
    def combined_model_odr(pars: list[float], x: ArrayF) -> ArrayF:
        return generalized_combined_model(pars, x, dataset_lengths)

    combined_model = odr.Model(combined_model_odr)
    odr_obj = odr.ODR(data, combined_model, beta0=initial_params)
    output = odr_obj.run()
    # reassign x_err and y_err to ds
    start_idx = 0
    for da in ds.values():
        end_idx = start_idx + len(da.y)
        da.x_errc[da.mask] = 2 * np.abs(output.delta[start_idx:end_idx])
        da.y_errc[da.mask] = 2 * np.abs(output.eps[start_idx:end_idx])
        start_idx = end_idx
    # Update the parameters with results from ODR
    p_names = ["K"]
    for lbl in ds:
        p_names.append(f"S0_{lbl}")
        p_names.append(f"S1_{lbl}")
    params = Parameters()
    for name, value, error in zip(p_names, output.beta, output.sd_beta, strict=True):
        params.add(name, value=value)
        params[name].stderr = error
    fig = figure.Figure()
    ax = fig.add_subplot(111)
    plot_fit(ax, ds, params, nboot=N_BOOT, pp=PlotParameters(ds.is_ph))
    return FitResult(fig, _Result(params), output, ds)


def fit_binding_odr_recursive(
    fr: FitResult, max_iterations: int = 15, tol: float = 0.1
) -> FitResult:
    """Analyze multi-label titration datasets using ODR."""
    result = copy.deepcopy(fr)
    if result.result is None or result.dataset is None:
        return FitResult()
    # Initial fit
    ro = fit_binding_odr(result)
    residual_variance = ro.mini.res_var if ro.mini else 0.0
    for _ in range(max_iterations):
        rn = fit_binding_odr(ro)
        if rn.mini and rn.mini.res_var == 0:
            rn = ro
            break
        # Check convergence
        if rn.mini and residual_variance - rn.mini.res_var < tol:
            break
        residual_variance = rn.mini.res_var if rn.mini else 0.0
        ro = rn
    return rn


def fit_binding_odr_recursive_outlier(
    fr: FitResult, tol: float = 0.5, threshold: float = 2.0
) -> FitResult:
    """Analyze multi-label titration datasets using ODR."""
    result = copy.deepcopy(fr)
    if result.result is None or result.dataset is None:
        return FitResult()
    # Initial fit
    ro = fit_binding_odr_recursive(result, tol=tol)
    omask = outlier(ro.mini, threshold)
    while omask.any() and ro.dataset:
        result.dataset.apply_mask(~omask)
        ro = fit_binding_odr_recursive(result, tol=tol)
        omask = outlier(ro.mini, 3.0)
    return ro


def outlier(
    output: odr.Output, threshold: float = 2.0, plot_z_scores: bool = False
) -> ArrayMask:
    """Identify outliers."""
    residuals_x = output.delta
    residuals_y = output.eps
    residuals = np.sqrt(residuals_x**2 + residuals_y**2)
    z_scores = np.abs((residuals - np.mean(residuals)) / np.std(residuals))
    if plot_z_scores:
        plt.scatter(range(len(z_scores)), z_scores)
        plt.axhline(y=threshold, color="r", linestyle="-")
        plt.title("Z-scores")
    outliers: ArrayMask = z_scores > threshold
    return outliers


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


def create_x_true(
    xc: ArrayF, x_errc: ArrayF, n_xerr: float, lower_nsd: float = 2.5
) -> ArrayF | pm.Deterministic:
    """Create x_true priors."""
    if n_xerr:
        x_errc_scaled = x_errc * n_xerr
        xd = -np.diff(xc)
        xd_err = np.sqrt(x_errc_scaled[:-1] ** 2 + x_errc_scaled[1:] ** 2)
        lower = xd.min() - lower_nsd * xd_err[np.argmin(xd)]
        logger.info(f"min pH distance: {lower}")
        x_diff = pm.TruncatedNormal(
            "x_diff", mu=xd, sigma=xd_err, lower=lower, shape=len(xc) - 1
        )
        x_start = pm.Normal("x_start", mu=xc[0], sigma=x_errc_scaled[0])
        x_cumsum = pm.math.cumsum(x_diff)
        return pm.Deterministic(
            "x_true", pm.math.concatenate([[x_start], x_start - x_cumsum])
        )
    return xc


def create_parameter_priors(
    params: Parameters, n_sd: float, key: str = "", ctr_name: str = ""
) -> dict[str, pm.Distribution]:
    """Create parameter priors."""
    priors = {}

    # Helper function for naming parameters
    def param_name(p_name: str) -> str:
        if key:
            return f"{p_name}_{key}"
        return p_name

    for name, p in params.items():
        sigma = max(p.stderr * n_sd, 1e-3) if p.stderr else 1e-3
        # Skip creating `K_{key}` if this belongs to a control group
        if ctr_name and name == "K":
            continue
        # Create prior distribution
        priors[param_name(name)] = pm.Normal(param_name(name), mu=p.value, sigma=sigma)
    return priors


def process_trace(
    trace: az.InferenceData, p_names: typing.KeysView[str], ds: Dataset, n_xerr: float
) -> FitResult:
    """Process the trace to extract parameter estimates and update datasets.

    Parameters
    ----------
    trace : az.InferenceData
        The posterior samples from PyMC sampling.
    p_names: typing.KeysView[str]
        Parameter names.
    ds : Dataset
        The dataset containing titration data.
    n_xerr : float
        Scaling factor for `x_errc`.

    Returns
    -------
    FitResult
        The updated fit result with extracted parameter values and datasets.
    """
    # Extract summary statistics for parameters
    rdf = az.summary(trace)
    rpars = Parameters()
    for name, row in rdf.iterrows():
        if name in p_names:
            rpars.add(name, value=row["mean"], min=row["hdi_3%"], max=row["hdi_97%"])
            rpars[name].stderr = row["sd"]
            rpars[name].init_value = row["r_hat"]
    # Process x_true and x_errc
    nxc = []  # New x_true values
    nx_errc = []  # New x_errc values
    for name, row in rdf.iterrows():
        if isinstance(name, str) and name.startswith("x_true"):
            nxc.append(row["mean"])
            nx_errc.append(row["sd"])
    for da in ds.values():
        da.xc = np.array(nxc)  # Update x_true values in the dataset
        da.x_errc = (
            np.array(nx_errc) * n_xerr
        )  # Scale the errors FIXME: n_xerr not needed
    # Extract magnitude for error scaling
    mag = rdf.loc["ye_mag", "mean"]  # type: ignore[index]
    mag = float(mag) if isinstance(mag, int | float) else 1.0
    # TODO: mag = 1
    for da in ds.values():
        da.y_errc *= mag  # Scale y errors by the magnitude
    # Create figure
    fig = figure.Figure()
    ax = fig.add_subplot(111)
    # FIXME: multi need this renaming quite surely
    rename_keys(rpars)
    plot_fit(ax, ds, rpars, nboot=N_BOOT, pp=PlotParameters(ds.is_ph))

    # Return the fit result
    return FitResult(fig, _Result(rpars), trace, ds)


def extract_fit(key: str, ctr: str, trace_df: pd.DataFrame, ds: Dataset) -> FitResult:
    """Compute individual dataset fit for a single key."""
    rpars = Parameters()
    rdf = trace_df[trace_df.index.str.endswith(key)]
    for name, row in rdf.iterrows():
        extracted_name = str(name).replace(f"_{key}", "")
        rpars.add(
            extracted_name, value=row["mean"], min=row["hdi_3%"], max=row["hdi_97%"]
        )
        rpars[extracted_name].stderr = row["sd"]
        rpars[extracted_name].init_value = row["r_hat"]
    if ctr:
        rdf = trace_df[trace_df.index.str.endswith(ctr)]
        for name, row in rdf.iterrows():
            extracted_name = str(name).replace(f"_{ctr}", "")
            rpars.add(
                extracted_name, value=row["mean"], min=row["hdi_3%"], max=row["hdi_97%"]
            )
            rpars[extracted_name].stderr = row["sd"]
            rpars[extracted_name].init_value = row["r_hat"]
    # Process x_true and x_errc
    nxc = []  # New x_true values
    nx_errc = []  # New x_errc values
    for name, row in trace_df.iterrows():
        if isinstance(name, str) and name.startswith("x_true"):
            nxc.append(row["mean"])
            nx_errc.append(row["sd"])
    for da in ds.values():
        da.xc = np.array(nxc)
        da.x_errc = np.array(nx_errc)
    # Create figure
    fig = figure.Figure()
    ax = fig.add_subplot(111)
    plot_fit(ax, ds, rpars, nboot=N_BOOT, pp=PlotParameters(ds.is_ph))
    return FitResult(fig, _Result(rpars), az.InferenceData(), ds)


def x_true_from_trace_df(trace_df: pd.DataFrame) -> DataArray:
    """Extract x_true from a trace dataframe."""
    nxc = []  # New x_true values
    nx_errc = []  # New x_errc values
    for name, row in trace_df.iterrows():
        if isinstance(name, str) and name.startswith("x_true"):
            nxc.append(row["mean"])
            nx_errc.append(row["sd"])
    return DataArray(xc=np.array(nxc), yc=np.ones_like(nxc), x_errc=np.array(nx_errc))


# TODO:
# 🧪 Test posterior integrity (e.g., credible intervals contain true Kd)
# 🧱 Replace repetitive for lbl in ds.items() logic using helper functions
# 🔁 Use pm.MutableData (in newer PyMC versions) to avoid model recompilation
def rename_keys(data: dict[str, typing.Any]) -> dict[str, typing.Any]:
    """Rename dictionary keys."""
    renamed_dict = {}
    for key, value in data.items():
        # Rule 1: Rename "K_*" → "K"
        if key.startswith("K_"):
            new_key = "K"
        # Rule 2: Remove "_{key}" suffix (keep only the first part before "_")
        elif key.rfind("_") > 1:
            idx = key.rfind("_")
            new_key = key[:idx]
        else:
            new_key = key
        renamed_dict[new_key] = value

    return renamed_dict


def fit_binding_pymc(
    fr: FitResult,
    n_sd: float = 10.0,
    n_xerr: float = 1.0,
    ye_scaling: float = 1.0,
    n_samples: int = 2000,
) -> FitResult:
    """Analyze multi-label titration datasets using pymc."""
    if fr.result is None or fr.dataset is None:
        return FitResult()
    params = fr.result.params
    ds = copy.deepcopy(fr.dataset)
    xc = next(iter(ds.values())).xc  # # TODO: move up out
    x_errc = next(iter(ds.values())).x_errc
    with pm.Model() as _:
        pars = create_parameter_priors(params, n_sd)
        x_true = create_x_true(xc, x_errc, n_xerr)
        # Add likelihoods for each dataset
        ye_mag = pm.HalfNormal("ye_mag", sigma=ye_scaling)
        for lbl, da in ds.items():
            y_model = binding_1site(
                x_true, pars["K"], pars[f"S0_{lbl}"], pars[f"S1_{lbl}"], ds.is_ph
            )
            pm.Normal(
                f"y_likelihood_{lbl}",
                mu=y_model[da.mask],
                sigma=ye_mag * da.y_err,
                observed=da.y,
            )
        # Inference
        tune = n_samples // 2
        trace = pm.sample(
            n_samples, tune=tune, target_accept=0.9, cores=4, return_inferencedata=True
        )
    return process_trace(trace, params.keys(), ds, n_xerr)


def fit_binding_pymc2(
    fr: FitResult,
    n_sd: float = 10.0,
    n_xerr: float = 1.0,
    n_samples: int = 2000,
) -> FitResult:
    """Analyze multi-label titration datasets using pymc."""
    if fr.result is None or fr.dataset is None:
        return FitResult()
    params = fr.result.params
    ds = copy.deepcopy(fr.dataset)
    xc = next(iter(ds.values())).xc  # # TODO: move up out
    x_errc = next(iter(ds.values())).x_errc
    with pm.Model() as _:
        pars = create_parameter_priors(params, n_sd)
        x_true = create_x_true(xc, x_errc, n_xerr)
        # Add likelihoods for each dataset
        ye_mag = {}
        ye_mag["y1"] = pm.HalfNormal("ye_mag1", sigma=100)
        ye_mag["y2"] = pm.HalfNormal("ye_mag2", sigma=10)
        for lbl, da in ds.items():
            y_model = binding_1site(
                x_true, pars["K"], pars[f"S0_{lbl}"], pars[f"S1_{lbl}"], ds.is_ph
            )
            pm.Normal(
                f"y_likelihood_{lbl}",
                mu=y_model[da.mask],
                sigma=ye_mag[lbl] * np.ones_like(da.y_err),
                observed=da.y,
            )
        # Inference
        tune = n_samples // 2
        trace = pm.sample(
            n_samples, tune=tune, target_accept=0.9, cores=4, return_inferencedata=True
        )
    return process_trace(trace, params.keys(), ds, n_xerr)


def fit_binding_pymc_compare(  # noqa: PLR0913
    fr: FitResult,
    buffer_sd: dict[str, float],
    learn_separate_y_mag: bool = False,
    n_sd: float = 10.0,
    n_xerr: float = 1.0,
    n_samples: int = 2000,
) -> az.InferenceData:
    """
    Fits a Bayesian binding model with two different noise models for comparison.

    Parameters
    ----------
    fr : FitResult
        The fit result from a previous run, providing initial parameters and dataset.
    buffer_sd : dict[str, float]
        bg_err
    learn_separate_y_mag : bool
        If True, learns a unique noise scaling factor for each dataset label.
        If False, learns a single scaling factor for all pre-weighted data.
    n_sd : float
        Prior width for parameters in create_parameter_priors.
    n_xerr : float
        Scaling factor for x_errc in create_x_true.
    n_samples : int
        Number of MCMC samples to draw.

    Returns
    -------
    az.InferenceData
        The posterior samples from PyMC for the specified noise model.
    """
    """
    if fr.result is None or fr.dataset is None:
        msg = "Input FitResult object must contain a result and a dataset."
        raise ValueError(msg)
    """
    if fr.result:
        params = fr.result.params
    if fr.dataset:
        ds = copy.deepcopy(fr.dataset)

    # Use the first dataset's x values. Assumes all datasets have same x points.
    xc = next(iter(ds.values())).xc
    x_errc = next(iter(ds.values())).x_errc

    with pm.Model():
        # Create priors for all parameters (K, S0_y1, S1_y1, etc.)
        pars = create_parameter_priors(params, n_sd)
        # Model the x-values with their uncertainties
        x_true = create_x_true(xc, x_errc, n_xerr)
        # ---------------------------------------------------------------------
        # Core conditional logic for the noise model

        if learn_separate_y_mag:
            # Model 1: Learn a unique noise scaling factor for each label
            # This is robust when you don't trust the initial y_err values
            ye_mag: dict[str | int, float] = {}
            true_buffer = {}
            for lbl, da in ds.items():
                ye_mag[lbl] = pm.HalfNormal(f"ye_mag_{lbl}", sigma=da.y_err.mean())
                y_model = binding_1site(
                    x_true, pars["K"], pars[f"S0_{lbl}"], pars[f"S1_{lbl}"], ds.is_ph
                )
                true_buffer[lbl] = pm.Normal(
                    f"true_buffer_{lbl}", mu=0, sigma=da.y_err.mean()
                )
                sigma = 10 * pm.math.sqrt(
                    (ye_mag[lbl] * np.ones_like(da.y_err)) ** 2 + buffer_sd[lbl] ** 2
                    # Alternatively use: ye_mag[lbl] ** 2 * da.y + buffer_sd[lbl] ** 2
                )

                pm.Normal(
                    f"y_likelihood_{lbl}",
                    mu=y_model[da.mask] + true_buffer[lbl],
                    sigma=sigma,
                    # Noise is learned from scratch and shot noise model
                    # Alternatively use: * np.ones_like(da.y_err),# Noise is learned from scratch
                    observed=da.y,
                )
        else:
            # Model 2: Learn a single noise scaling factor for all data
            # This is appropriate when you trust the relative y_err values
            ye_mag0 = pm.HalfNormal("ye_mag", sigma=10.0)
            for lbl, da in ds.items():
                y_model = binding_1site(
                    x_true, pars["K"], pars[f"S0_{lbl}"], pars[f"S1_{lbl}"], ds.is_ph
                )
                pm.Normal(
                    f"y_likelihood_{lbl}",
                    mu=y_model[da.mask],
                    sigma=ye_mag0 * da.y_err,  # Apply a single scaling factor
                    # Alternatively use:  sigma=da.y_err,  # Apply a single scaling factor
                    observed=da.y,
                )
        # ---------------------------------------------------------------------
        # Run MCMC sampling
        trace: az.InferenceData = pm.sample(
            n_samples, cores=4, return_inferencedata=True, target_accept=0.9
        )
    return trace


def closest_point_on_curve(f: FloatFunc, x_obs: float, y_obs: float) -> float:
    """Find the closest point on the model curve."""

    def objective(x_prime: float) -> float:
        return (x_obs - x_prime) ** 2 + (y_obs - f(x_prime)) ** 2

    result = optimize.minimize_scalar(objective)
    return float(result.x)


def fit_binding_pymc_odr(
    fr: FitResult,
    n_sd: float = 10.0,
    xe_scaling: float = 1.0,
    ye_scaling: float = 10.0,
    n_samples: int = 2000,
) -> arviz.InferenceData | pm.backends.base.MultiTrace:
    """Analyze multi-label titration datasets using pymc."""
    if fr.result is None or fr.dataset is None:
        return az.InferenceData()  # FitResult()
    params = fr.result.params
    ds = copy.deepcopy(fr.dataset)
    xc = next(iter(ds.values())).xc
    x_errc = next(iter(ds.values())).x_errc
    with pm.Model() as _:
        pars = create_parameter_priors(params, n_sd)
        # Add likelihoods for each dataset
        ye_mag = pm.HalfNormal("ye_mag", sigma=ye_scaling)
        xe_mag = pm.HalfNormal("xe_mag", sigma=xe_scaling)

        for lbl, da in ds.items():

            def _y_model(x: float, lbl: str = lbl) -> float:
                return binding_1site(
                    x, pars["K"], pars[f"S0_{lbl}"], pars[f"S1_{lbl}"], ds.is_ph
                )

            # Define symbolic closest points using PyMC-compatible operations
            x_prime = pm.Deterministic(
                f"x_prime_{lbl}",
                pm.math.stack(  # noqa: PD013
                    [
                        closest_point_on_curve(
                            lambda x_val: _y_model(x_val).eval(),  # type: ignore[attr-defined]
                            x_obs,
                            y_obs,
                        )
                        for x_obs, y_obs in zip(xc, da.y, strict=True)
                    ]
                ),
            )

            y_prime = pm.Deterministic(
                f"y_prime_{lbl}",
                pm.math.stack([_y_model(x) for x in x_prime.eval()]),  # noqa: PD013
            )
            y_model = pm.Deterministic(
                f"y_model_{lbl}",
                pm.math.stack([_y_model(x) for x in xc]),  # noqa: PD013
            )
            ## TODO:  y_model = as_tensor_variable([_y_model(x) for x in xc])

            mask = as_tensor_variable(da.mask)
            # Orthogonal distance likelihood
            distances = ((x_prime - xc) / (xe_mag * x_errc)) ** 2 + (
                (y_prime - y_model) / (ye_mag * da.y_err)
            ) ** 2
            pm.Normal(
                f"orthogonal_likelihood_{lbl}",
                mu=distances[mask],
                sigma=1,
                observed=np.zeros(len(distances[mask].eval())),
            )
        # Inference
        return pm.sample(n_samples, cores=4, return_inferencedata=True)
    ## TODO:  return process_trace(trace, params.keys(), ds, 0)


def fit_binding_pymc_multi(  # noqa: PLR0913
    results: dict[str, FitResult],
    scheme: PlateScheme,
    n_sd: float = 5.0,
    n_xerr: float = 1.0,
    ye_scaling: float = 1.0,
    n_samples: int = 2000,
) -> arviz.data.inference_data.InferenceData:
    """Analyze multiple titration datasets with shared parameters for controls."""
    # FIXME: pytensor.config.floatX = "float32"  # type: ignore[attr-defined]
    ds = next((result.dataset for result in results.values() if result.dataset), None)
    if ds is None:
        msg = "No valid dataset found in results."
        raise ValueError(msg)
    xc = next(iter(ds.values())).xc
    x_errc = next(iter(ds.values())).x_errc * n_xerr
    labels = ds.keys()

    values = {}
    stderr = {}
    for name, wells in scheme.names.items():
        values[name] = [
            v.result.params["K"].value
            for well, v in results.items()
            if v.result and well in wells
        ]
        stderr[name] = [
            v.result.params["K"].stderr
            for well, v in results.items()
            if v.result and well in wells
        ]
    ctr_ks = weighted_stats(values, stderr)

    with pm.Model() as _:
        ye_mag = {}
        for label in labels:
            ye_mag[label] = pm.HalfNormal(f"ye_mag_{label}", sigma=ye_scaling)
        x_true = create_x_true(xc, x_errc, n_xerr)

        # Create shared K parameters for each control group
        k_params = {
            control_name: pm.Normal(
                f"K_{control_name}",
                mu=ctr_ks[control_name][0],
                sigma=0.2,  # FIXME: use var
            )
            for control_name in scheme.names
        }

        for key, r in results.items():
            if r.result and r.dataset:
                ds = r.dataset
                # Determine if the well is associated with a control group
                ctr_name = next(
                    (name for name, wells in scheme.names.items() if key in wells), ""
                )
                pars = create_parameter_priors(r.result.params, n_sd, key, ctr_name)
                # Use shared K for control group wells or create a unique K otherwise
                K = k_params[ctr_name] if ctr_name else pars[f"K_{key}"]  # noqa: N806

                for lbl, da in ds.items():
                    y_model = binding_1site(
                        x_true,
                        K,
                        pars[f"S0_{lbl}_{key}"],
                        pars[f"S1_{lbl}_{key}"],
                        ds.is_ph,
                    )
                    pm.Normal(
                        f"y_likelihood_{lbl}_{key}",
                        mu=y_model[da.mask],
                        sigma=ye_mag[lbl] * da.y_err,
                        observed=da.y,
                    )

        trace: az.InferenceData = pm.sample(
            n_samples, target_accept=0.9, return_inferencedata=True
        )

    return trace


# ------------------------------------------------------------------
# Helper: weighted statistics
# ------------------------------------------------------------------
def weighted_stats(
    values: dict[str, list[float]], stderr: dict[str, list[float]]
) -> dict[str, tuple[float, float]]:
    """Return weighted mean ± sigma."""
    results = {}
    for sample in values:  # noqa:PLC0206
        x = np.array(values[sample])
        se = np.array(stderr[sample])
        weighted_mean = np.average(x, weights=1 / se**2)
        weighted_stderr = np.sqrt(1 / np.sum(1 / se**2))
        results[sample] = (weighted_mean, weighted_stderr)
    return results


def fit_binding_pymc_multi2(  # noqa: PLR0913
    results: dict[str, FitResult],
    scheme: PlateScheme,
    bg_err: dict[int, ArrayF],
    n_sd: float = 5.0,
    n_xerr: float = 1.0,
    # Ponder this: ye_scaling: float = 1.0, # This parameter is no longer needed in the same way
    n_samples: int = 2000,
) -> arviz.data.inference_data.InferenceData:
    """Analyze multiple titration datasets with shared parameters for controls."""
    ds_example = next(
        (result.dataset for result in results.values() if result.dataset), None
    )
    ds = next((result.dataset for result in results.values() if result.dataset), None)

    if ds_example is None:
        msg = "No valid dataset found in results."
        raise ValueError(msg)
    # Extract common data once
    xc = next(iter(ds_example.values())).xc
    x_errc = next(iter(ds_example.values())).x_errc * n_xerr
    labels = list(ds_example.keys())  # e.g., ['y1', 'y2']
    # --- Pre-calculate weighted stats for K priors (remains the same) ---
    values = {}
    stderr = {}
    for name, wells in scheme.names.items():
        values[name] = [
            v.result.params["K"].value
            for well, v in results.items()
            if v.result
            and well in wells  # and "K" in v.result.params._params # Check if K exists
        ]
        stderr[name] = [
            v.result.params["K"].stderr
            for well, v in results.items()
            if v.result and well in wells  #  and "K" in v.result.params._params
        ]
    ctr_ks = weighted_stats(values, stderr)
    logger.info(f"Weighted K stats for control groups: {ctr_ks}")

    with pm.Model() as _:
        # --- Common Priors / Variables for the entire model ---
        x_true = create_x_true(xc, x_errc, n_xerr)
        # Global scaling factors for the signal-dependent noise for each label (band)
        # `sigma` prior for HalfNormal should be set considering the expected scale of your signal values.
        # If your fluorescence values are thousands, a sigma of 10-100 might be reasonable for the scaling factor.
        sigma_signal_scale = {}
        for i, lbl in enumerate(labels, start=1):
            # Dynamically determine the variable name based on the label, e.g., 'sigma_signal_scale_y1'
            sigma_signal_scale[lbl] = pm.HalfNormal(
                f"sigma_signal_scale_{lbl}", sigma=10.0
            )  # TODO: Adjust sigma based on data scale
            # --- Model the true buffer mean (as a latent variable) ---
            true_buffer_mean = pm.Normal(
                f"true_buffer_{lbl}",
                mu=0,
                sigma=bg_err[i],
            )
            variance_buffer_contrib = bg_err[i] ** 2

        # Degrees of freedom for Student's T distribution (for robustness)
        # Can be shared or per-label. Shared is often fine for similar data types.
        # this was for student: nu_common = pm.Gamma("nu_common", alpha=2, beta=0.1)

        # Create shared K parameters for each control group
        k_params = {
            control_name: pm.Normal(
                f"K_{control_name}",
                mu=ctr_ks[control_name][0],
                # if ctr_ks[control_name][0] is not np.nan
                # else 7.0,  # Handle case where no K values found
                sigma=0.2,  # FIXME: consider using ctr_ks[control_name][1] for sigma
                # TODO: sigma=ctr_ks[control_name][1] if ctr_ks[control_name][1] is not np.nan else 0.5, # Default sigma for K if no stderr
            )
            for control_name in scheme.names
        }
        print(k_params)
        # --- Loop through each well (key) and its data ---
        for key, r in results.items():
            if r.result and r.dataset:
                ds = r.dataset
                # Determine if the well is associated with a control group
                ctr_name = next(
                    (name for name, wells in scheme.names.items() if key in wells), ""
                )
                # Parameters for S0, S1 (unique to each well and label)
                # `create_parameter_priors` should return a dict of PyMC distributions for S0, S1 for this key/well
                pars = create_parameter_priors(r.result.params, n_sd, key, ctr_name)
                # Use shared K for control group wells or create a unique K otherwise
                k_param_for_well = k_params[ctr_name] if ctr_name else pars[f"K_{key}"]
                # --- Loop through each fluorescence label (e.g., 'y1', 'y2') within the current well ---
                for lbl, da in ds.items():
                    # --- Predicted signal from the binding model ---
                    y_model_signal = binding_1site(
                        x_true,
                        k_param_for_well,
                        pars[f"S0_{lbl}_{key}"],
                        pars[f"S1_{lbl}_{key}"],
                        ds.is_ph,
                    )
                    # Predicted Total Fluorescence (Signal + Buffer) for the likelihood mu
                    mu_total_pred = pm.Deterministic(
                        f"mu_total_pred_{lbl}_{key}", y_model_signal + true_buffer_mean
                    )
                    # --- Model the Noise ---
                    # A common model for fluorescence noise is that standard deviation scales with sqrt(mean)
                    # or that variance scales linearly with mean (similar to Poisson, but continuous/scaled).
                    # Let's use a simpler and common power-law for noise (SD = a * mu^b)
                    # Here, we'll assume `b=0.5` (sqrt) and `a` is our `sigma_signal_scale`.
                    # Calculate the variance for the signal component (excluding buffer noise)
                    # Ensuring non-negativity for sqrt
                    # Variance from signal itself (heteroscedastic: proportional to predicted signal mean)
                    # Use pm_math.maximum to avoid issues with negative predicted signals for variance
                    variance_signal_contrib = sigma_signal_scale[lbl] * pm_math.maximum(
                        1e-6, y_model_signal
                    )
                    # Total variance is the sum of independent variances
                    total_variance_obs = pm.Deterministic(
                        f"total_variance_obs_{lbl}_{key}",
                        variance_buffer_contrib + variance_signal_contrib,
                    )
                    # Total standard deviation for the likelihood
                    sigma_obs = pm.Deterministic(
                        f"sigma_obs_{lbl}_{key}", pm_math.sqrt(total_variance_obs)
                    )
                    # --- Likelihood ---
                    # Use Student's T distribution for robustness against outliers
                    # Apply mask to observed data and corresponding mu/sigma
                    # This is the learned, heteroscedastic SD
                    pm.Normal(
                        f"y_likelihood_{lbl}_{key}",
                        # this was for student: nu=nu_common,  # Use the shared nu_common
                        mu=mu_total_pred[da.mask],
                        sigma=sigma_obs[da.mask],
                        observed=da.y,
                    )

        trace: az.InferenceData = pm.sample(
            n_samples, target_accept=0.9, return_inferencedata=True
        )

    return trace


# ------------------------------------------------------------------
# 2.3  Posterior-predictive helper - visualise one well at a time
# ------------------------------------------------------------------
def plot_ppc_well(
    trace: az.InferenceData,
    key: str,
    labels: list[str] | None = None,
    figsize: tuple[float, float] = (8, 4),
) -> figure.Figure:
    """Draw posterior predictive samples for a particular well (and all its labels).

    The returned figure can be displayed with matplotlib.

    Parameters
    ----------
    trace   : az.InferenceData
        Trace produced by ``fit_binding_pymc_advanced``.
    key     : str
        Well identifier (e.g. 'A01').
    labels  : list[str] | None
        Names of the bands to show.  If *None* the function will
        automatically look for all variables starting with
        ``'y_'`` that contain this key.
    figsize: tuple[float, float]
        size?

    Return
    ------
    figure.Figure
        Plot
    """
    if labels is None:
        labels = [
            var.split("_")[1]
            for var in trace.posterior.data_vars  # type: ignore[attr-defined]
            if f"{key}" in var and var.startswith("y_")
        ]

    fig, axes = plt.subplots(
        len(labels), 1, figsize=(figsize[0], figsize[1] * len(labels)), sharex=True
    )
    if len(labels) == 1:
        axes = [axes]

    for ax, lbl in zip(axes, labels, strict=True):
        var_name = f"y_{lbl}_{key}"
        az.plot_ppc(  # type: ignore[no-untyped-call]
            az.from_dict(
                {"posterior_predictive": trace.posterior_predictive[var_name]},  # type: ignore[attr-defined]
                coords={"y": [lbl]},
                dims={"y": 0},
            ),
            ax=ax,
        )
        ax.set_title(f"Well {key} - band {lbl}")
        ax.set_xlabel("Observed y")
        ax.set_ylabel("Posterior predictive")

    plt.tight_layout()
    return fig


# ------------------------------------------------------------------
# 2.4  Comparison of posteriors with deterministic fits
# ------------------------------------------------------------------
def compare_posteriors(trace: az.InferenceData, results: dict[str, FitResult]) -> None:
    """Print posterior mean ± 95 % C.I.

    For the K parameter for each well, and juxtapose it with the deterministic K
    (from fit_binding_pymc).

    Parameters
    ----------
    trace   : az.InferenceData
        Output of ``fit_binding_pymc_advanced``.
    results : dict[str, FitResult]
        Deterministic fits produced by the old pipeline.
    """
    # Summarise the trace
    summary = az.summary(trace, var_names=["K_*"], round_to=3)
    print("\nPosterior for K (averaged over all draws)")
    print(summary[["mean", "hdi_2.5%", "hdi_97.5%"]])

    # Add deterministic K to the table for easy comparison
    deterministic = {}
    for k, fr in results.items():
        if fr.result and "K" in fr.result.params:
            deterministic[k] = fr.result.params["K"].value

    print("\nDeterministic K  (fit_binding_pymc)")
    for k, v in deterministic.items():
        print(f"  {k:6s}  {v:0.3f}")

    # Align rows
    table = summary.join(pd.Series(deterministic, name="deterministic_K"))
    print("\nCombined table")
    print(table[["mean", "hdi_2.5%", "hdi_97.5%", "deterministic_K"]])
