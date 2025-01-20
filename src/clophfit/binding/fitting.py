"""Fit Cl binding and pH titration."""

from __future__ import annotations

import copy
import typing
from dataclasses import dataclass, field
from functools import partial
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
from pytensor.tensor import as_tensor_variable
from scipy import odr, optimize  # type: ignore[import-untyped]
from uncertainties import ufloat  # type: ignore[import-untyped]

from clophfit.binding.data import DataArray, Dataset
from clophfit.binding.plotting import (
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
from clophfit.logging_config import setup_logger

if typing.TYPE_CHECKING:
    from clophfit.clophfit_types import ArrayDict, ArrayF, ArrayMask, FloatFunc
    from clophfit.prtecan import PlateScheme, TitrationResults


N_BOOT = 20  # To compute fill_between uncertainty.

EMCEE_STEPS = 1800

logger = setup_logger(__name__)
logger.info("Plotting module started")


# fmt: off
@typing.overload
def binding_1site(
    x: float, K: float, S0: float, S1: float, is_ph: bool = False  # noqa: N803
) -> float: ...


@typing.overload
def binding_1site(
    x: ArrayF, K: float, S0: float, S1: float, is_ph: bool = False  # noqa: N803
) -> ArrayF: ...

# fmt: on
def binding_1site(
    x: float | ArrayF, K: float, S0: float, S1: float, is_ph: bool = False  # noqa: N803
) -> float | ArrayF:  # fmt: skip
    """Single site binding model function.

    Parameters
    ----------
    x : float | ArrayF
        Concentration values.
    K : float
        Dissociation constant.
    S0 : float
        Plateau value for the unbound state.
    S1 : float
        Plateau value for the bound state.
    is_ph : bool, optional
        If True, use the pH model for binding. Default is False.

    Returns
    -------
    float | ArrayF
        Modeled binding values.

    Notes
    -----
    The parameters K, S0 and S1 are in uppercase by convention as used in lmfit library.
    """
    if is_ph:
        return S0 + (S1 - S0) * 10 ** (K - x) / (1 + 10 ** (K - x))
    return S0 + (S1 - S0) * x / K / (1 + x / K)


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


def kd(kd1: float, pka: float, ph: ArrayF | float) -> ArrayF | float:
    """Infinite cooperativity model.

    It can describe pH-dependence for chloride dissociation constant.

    Parameters
    ----------
    kd1 : float
        Dissociation constant at pH <= 5.0 (fully protonated).
    pka : float
        Acid dissociation constant.
    ph : ArrayF | float
        pH value(s).

    Returns
    -------
    ArrayF | float
        Predicted Kd value(s).

    Examples
    --------
    >>> kd(10, 8.4, 7.4)
    11.0
    >>> import numpy as np
    >>> kd(10, 8.4, np.array([7.4, 8.4]))
    array([11., 20.])
    """
    return kd1 * (1 + 10 ** (pka - ph)) / 10 ** (pka - ph)


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


class InsufficientDataError(Exception):
    """Raised to prevent fitting failure for too few data points."""


def fit_binding_glob(ds: Dataset) -> FitResult:
    """Analyze multi-label titration datasets and visualize the results."""
    params = _build_params_1site(ds)
    if len(params) > len(np.concatenate([da.y for da in ds.values()])):
        raise InsufficientDataError
    mini = Minimizer(_binding_1site_residuals, params, fcn_args=(ds,))
    result = mini.minimize()
    fig = figure.Figure()
    ax = fig.add_subplot(111)
    plot_fit(ax, ds, result.params, nboot=N_BOOT, pp=PlotParameters(ds.is_ph))
    return FitResult(fig, result, mini, ds)


# TODO: remove the print statements use logging
def weight_da(da: DataArray, is_ph: bool) -> bool:
    """Assign weights to each label based on individual label fitting."""
    failed = False
    ds = Dataset.from_da(da, is_ph=is_ph)
    params = _build_params_1site(ds)
    if len(params) > len(da.y):
        print("failed")
        failed = True
        sem = 1.0 * np.ones_like(da.xc)
    else:
        res = lmfit.minimize(_binding_1site_residuals, params, args=(ds,))
        # Calculate residuals SEM
        sem = np.std(res.residual, ddof=1) / np.sqrt(len(res.residual))
    da.y_err = sem
    return failed


def weight_multi_ds_titration(ds: Dataset) -> None:
    """Assign weights to each label based on individual label fitting."""
    failed_fit_labels = []
    for lbl, da in ds.items():
        if weight_da(da, ds.is_ph):
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
        weight_multi_ds_titration(ds)
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
    rn.dataset = result.dataset
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
    ro.dataset = result.dataset
    return ro


def outlier(
    output: odr.Output, threshold: float = 2.0, plot_z_scores: bool = False
) -> ArrayMask:
    """Identify outliers."""
    residuals_x = output.delta
    residuals_y = output.eps
    residuals = np.sqrt(residuals_x**2 + residuals_y**2)
    residuals = np.sqrt(residuals_y**2)
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
        da.x_errc = np.array(nx_errc) * n_xerr  # Scale the errors
    # Extract magnitude for error scaling
    mag = rdf.loc["ye_mag", "mean"]  # type: ignore[index]
    mag = float(mag) if isinstance(mag, int | float) else 1.0
    for da in ds.values():
        da.y_errc *= mag  # Scale y errors by the magnitude
    # Create diagnostic plots
    fig = figure.Figure()
    ax = fig.add_subplot(111)
    # FIXME: multi need this renaming quite surely
    print(rpars)
    renamed = rename_keys(rpars)
    print(renamed)
    plot_fit(ax, ds, rpars, nboot=N_BOOT, pp=PlotParameters(ds.is_ph))

    # Return the fit result
    return FitResult(fig, _Result(rpars), trace, ds)


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
    ye_scaling: float = 10.0,
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
        print(pars)
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
    n_xerr: float = 5.0,
    ye_scaling: float = 10.0,
    n_samples: int = 2000,
) -> tuple[TitrationResults, arviz.data.inference_data.InferenceData]:
    """Analyze multiple titration datasets with shared parameters for controls."""
    from clophfit.prtecan import TitrationResults

    # FIXME: pytensor.config.floatX = "float32"  # type: ignore[attr-defined]
    ds = next((result.dataset for result in results.values() if result.dataset), None)
    if ds is None:
        msg = "No valid dataset found in results."
        raise ValueError(msg)
    xc = next(iter(ds.values())).xc
    x_errc = next(iter(ds.values())).x_errc * n_xerr

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
    print(ctr_ks)

    with pm.Model() as _:
        ye_mag = pm.HalfNormal("ye_mag", sigma=ye_scaling)
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
                        sigma=ye_mag * da.y_err,
                        observed=da.y,
                    )

        trace: az.InferenceData = pm.sample(
            n_samples, target_accept=0.9, cores=6, return_inferencedata=True
        )

    # Extract results into FitResult objects
    df_trace = typing.cast(pd.DataFrame, az.summary(trace, fmt="wide"))

    titration_results = TitrationResults(
        scheme=scheme,
        fit_keys=set(results.keys()),
        compute_func=partial(extract_params, df_trace=df_trace),
    )
    titration_results.compute_all()
    return titration_results, trace


def extract_params(key: str, df_trace: pd.DataFrame) -> FitResult:
    """Compute individual dataset fit for a single key."""
    rdf = df_trace[df_trace.index.str.endswith(key)]
    print(key, rdf)
    rpars = Parameters()
    for name, row in rdf.iterrows():
        rpars.add(name, value=row["mean"], min=row["hdi_3%"], max=row["hdi_97%"])
        rpars[name].stderr = row["sd"]
        rpars[name].init_value = row["r_hat"]
    print(name, rpars)
    return FitResult(result=_Result(rpars))


def weighted_stats(
    values: dict[str, list[float]], stderr: dict[str, list[float]]
) -> dict[str, tuple[float, float]]:
    """Weighted average."""
    results = {}
    for sample in values:  # noqa:PLC0206
        x = np.array(values[sample])
        se = np.array(stderr[sample])
        weighted_mean = np.average(x, weights=1 / se**2)
        weighted_stderr = np.sqrt(1 / np.sum(1 / se**2))
        results[sample] = (weighted_mean, weighted_stderr)
    return results
