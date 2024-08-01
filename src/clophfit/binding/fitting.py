"""Fit Cl binding and pH titration."""

from __future__ import annotations

import copy
import typing
import warnings
from dataclasses import dataclass, field
from sys import float_info

import lmfit  # type: ignore[import-untyped]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lmfit import Parameters
from lmfit.minimizer import Minimizer, MinimizerResult  # type: ignore[import-untyped]
from matplotlib import axes, figure
from scipy import odr  # type: ignore[import-untyped]
from uncertainties import ufloat  # type: ignore[import-untyped]

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
from clophfit.types import ArrayDict, ArrayF

N_BOOT = 20  # To compute fill_between uncertainty.
EMCEE_STEPS = 1800


@dataclass
class DataArrays:
    """A collection of matching x, y, and optional w data arrays."""

    x: ArrayF
    y: ArrayF
    w: ArrayF | None = None

    def __post_init__(self) -> None:
        """Ensure the x and y arrays are of equal length after initialization."""
        if len(self.x) != len(self.y):
            msg = "Length of 'x' and 'y' must be equal."
            raise ValueError(msg)
        if self.w is not None and len(self.x) != len(self.w):
            msg = "Length of 'x' and 'w' must be equal."
            raise ValueError(msg)


class Dataset(dict[str, DataArrays]):
    """Contain pairs of matching x and y data arrays, indexed by a string key.

    Initialization will remove any NaN values from the x and y arrays.

    Parameters
    ----------
    x : ArrayF | ArrayDict
        The x values of the dataset(s), either as a single ArrayF or as an ArrayDict
        if multiple datasets are provided.
    y : ArrayF | ArrayDict
        The y values of the dataset(s), either as a single ArrayF or as an ArrayDict
        if multiple datasets are provided.
    is_ph : bool
        Indicate if x values represent pH (default is False).
    w : ArrayF | ArrayDict | None
        The w values (weights) of the dataset(s), either as a single ArrayF or
        as an ArrayDict if multiple datasets are provided.

    Raises
    ------
    ValueError
        If x and y are both ArrayDict and their keys don't match.
    """

    is_ph: bool

    def __init__(
        self,
        x: ArrayF | ArrayDict,
        y: ArrayF | ArrayDict,
        is_ph: bool = False,
        w: ArrayF | ArrayDict | None = None,
    ) -> None:
        def filter_nan(arr: ArrayF, reference: ArrayF) -> ArrayF:
            return typing.cast(ArrayF, arr[~np.isnan(reference)])

        def create_data(x: ArrayF, y: ArrayF, w: ArrayF | None) -> DataArrays:
            x, y = filter_nan(x, y), filter_nan(y, y)
            w = filter_nan(w, y) if w is not None else None
            return DataArrays(x, y, w)

        # x:array, y:array
        if (
            isinstance(x, np.ndarray)
            and isinstance(y, np.ndarray)
            and not isinstance(w, dict)
        ):
            data = {"default": create_data(x, y, w)}
        # x:array, y:dict_of_arrays
        elif isinstance(x, np.ndarray) and isinstance(y, dict):
            data = {
                k: create_data(x, v, w[k] if isinstance(w, dict) else w)
                for k, v in y.items()
            }
        # x:dict_of_arrays, y:dict_of_arrays
        elif isinstance(x, dict) and isinstance(y, dict):
            if x.keys() != y.keys() or (isinstance(w, dict) and x.keys() != w.keys()):
                msg = "Keys of 'x', 'y', and 'w' (if w is a dict) must match."
                raise ValueError(msg)
            data = {
                k: create_data(x[k], y[k], w[k] if isinstance(w, dict) else w)
                for k in x
            }
        super().__init__(data)
        self.is_ph = is_ph

    def add_weights(self, w: ArrayF | ArrayDict) -> None:
        """Add weights to the dataset.

        Parameters
        ----------
        w : ArrayF | ArrayDict
            The weights to be added to the dataset.

        Raises
        ------
        ValueError
            If a key in the weights dictionary does not match any key in the
            current Dataset object.
        """
        if isinstance(w, np.ndarray):
            for da in self.values():
                da.w = w
        elif isinstance(w, dict):
            for k, weights in w.items():
                if k in self:
                    self[k].w = weights
                else:
                    msg = f"No matching found for key '{k}' in the current Dataset."
                    raise ValueError(msg)

    def copy(self, keys: set[str] | None = None) -> Dataset:
        """Return a copy of the Dataset.

        If keys are provided, only data associated with those keys are copied.

        Parameters
        ----------
        keys : set[str] | None, optional
            List of keys to include in the copied dataset. If None (default),
            copies all data.

        Returns
        -------
        Dataset
            A copy of the dataset.

        Raises
        ------
        KeyError
            If a provided key does not exist in the Dataset.
        """
        if keys is None:
            copied = copy.deepcopy(self)
        else:
            # If keys are specified, only copy those keys
            copied = Dataset({}, {}, is_ph=self.is_ph)
            for key in keys:
                if key in self:
                    copied[key] = copy.deepcopy(self[key])
                else:
                    msg = f"No such key: '{key}' in the Dataset."
                    raise KeyError(msg)
        return copied

    def clean_data(self, n_params: int) -> None:
        """Remove too small datasets."""
        for key in list(
            self.keys()
        ):  # list() is used to avoid modifying dict during iteration
            if n_params > len(self[key].y):
                warnings.warn(
                    f"Removing key '{key}' from Dataset: number of parameters "
                    f"({n_params}) exceeds number of data points ({len(self[key].y)}).",
                    stacklevel=2,
                )
                del self[key]


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
    w = {k: da.w if da.w is not None else np.ones_like(da.y) for k, da in ds.items()}
    return x, y, w, ds.is_ph


def _binding_1site_residuals(params: Parameters, ds: Dataset) -> ArrayF:
    """Compute concatenated residuals (array) for multiple datasets; weight = 1/std."""
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
    k_initial = []
    for lbl, da in ds.items():
        params.add(f"S0_{lbl}", value=da.y[0])
        params.add(f"S1_{lbl}", value=da.y[-1])
        target_y = (da.y[0] + da.y[-1]) / 2
        k_initial.append(da.x[np.argmin(np.abs(da.y - target_y))])
    if ds.is_ph:
        params.add("K", value=np.mean(k_initial), min=3, max=11)
    else:
        # epsilon avoids x/K raise x/0 error
        params.add("K", value=np.mean(k_initial), min=float_info.epsilon)
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
        ds = Dataset(x, v[0, :] + y_offset, is_ph)
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
        ds = Dataset(x, y, is_ph)
        ylabel = "Integrated Band Fluorescence"
        ylabel_color = (0.0, 0.0, 0.0, 1.0)  # "k"
    weight_multi_ds_titration(ds)
    fit_result = fit_binding_glob(ds)
    result = fit_result.result
    plot_fit(ax4, ds, result, nboot=N_BOOT, pp=PlotParameters(is_ph))
    ax4.set_ylabel(ylabel, color=ylabel_color)
    return FitResult(fig, result, fit_result.mini)


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
    plot_fit(ax, ds, result, nboot=N_BOOT, pp=PlotParameters(ds.is_ph))
    return FitResult(fig, result, mini)


def weight_multi_ds_titration(ds: Dataset) -> None:
    """Assign weights to each label based on individual label fitting."""
    wc: ArrayDict = {}
    labels_to_remove = []
    # Calculate standard deviations of residuals
    for label, da in ds.items():
        x = {label: da.x}
        y = {label: da.y}
        d = Dataset(x, y, ds.is_ph)
        params = _build_params_1site(d)
        # Mark for removal and skip minimization when n pars exceed data points in y
        if len(params) > len(da.y):
            warnings.warn(
                f"Marking dataset {label} for removal: insufficient data points.",
                stacklevel=2,
            )
            labels_to_remove.append(label)
            continue
        res = lmfit.minimize(_binding_1site_residuals, params, args=(d,))
        wc[label] = 1 / np.std(res.residual) * np.ones_like(da.x)
    # Remove marked datasets
    for label in labels_to_remove:
        del ds[label]
    if not ds:  # check if ds is now empty
        warnings.warn("No datasets left after cleaning. Exiting.", stacklevel=2)
    ds.add_weights(wc)


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
        weight_multi_ds_titration(ds)
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
    result: MinimizerResult,
    nboot: int = 0,
    pp: PlotParameters | None = None,
) -> None:
    """Plot residuals for each dataset with uncertainty."""
    _stretch = 0.05
    xfit = {
        k: np.linspace(da.x.min() * (1 - _stretch), da.x.max() * (1 + _stretch), 100)
        for k, da in ds.items()
    }
    yfit = _binding_1site_models(result.params, xfit, ds.is_ph)
    # Create a color cycle
    colors = [COLOR_MAP(i) for i in range(len(ds))]
    for (lbl, da), clr in zip(ds.items(), colors, strict=False):
        # Make sure a label will be displayed.
        label = lbl if (da.w is None and nboot == 0) else None
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
        if nboot:
            # Calculate uncertainty using Monte Carlo method.
            y_samples = np.empty((nboot, len(xfit[lbl])))
            rng = np.random.default_rng()
            for i in range(nboot):
                p_sample = result.params.copy()
                for param in p_sample.values():
                    # Especially stderr can be None in case of critical fitting
                    if param.value and param.stderr:
                        param.value = rng.normal(param.value, param.stderr)
                y_samples[i, :] = _binding_1site_models(p_sample, xfit, ds.is_ph)[lbl]
            dy = y_samples.std(axis=0)
            # Plot uncertainty.
            # Display label in fill_between plot.
            ax.fill_between(
                xfit[lbl],
                yfit[lbl] - dy,
                yfit[lbl] + dy,
                alpha=0.4,
                color=clr,
                label=lbl,
            )
            if da.w is not None:
                ye = 1 / da.w
                ax.errorbar(
                    da.x, da.y, yerr=ye, fmt="none", color="gray", alpha=0.4, capsize=3
                )
        elif da.w is not None:
            ye = 1 / da.w
            # Display label in error bar plot.
            ax.errorbar(
                da.x, da.y, yerr=ye, fmt=".", label=lbl, color=clr, alpha=0.4, capsize=3
            )
    ax.legend()
    if result.params["K"].stderr:  # Can be None in case of critical fitting
        k = ufloat(result.params["K"].value, result.params["K"].stderr)
    else:
        k = f'{result.params["K"].value:.3g}' if result.params["K"].value else None
    title = "=".join(["K", str(k).replace("+/-", "±")])
    xlabel = "pH" if ds.is_ph else "Cl"
    _apply_common_plot_style(ax, f"LM fit {title}", xlabel, "")


def plot_fit2(
    ds: Dataset,
    xerr: ArrayF,
    params: Parameters,
    nboot: int = 0,
    pp: PlotParameters | None = None,
) -> figure.Figure:
    """Plot residuals for each dataset with uncertainty."""
    _stretch = 0.05
    xfit = {
        k: np.linspace(da.x.min() * (1 - _stretch), da.x.max() * (1 + _stretch), 100)
        for k, da in ds.items()
    }
    yfit = _binding_1site_models(params, xfit, ds.is_ph)
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    # Create a color cycle
    colors = [COLOR_MAP(i) for i in range(len(ds))]
    ylim = (np.inf, -np.inf)
    for (lbl, da), clr in zip(ds.items(), colors, strict=False):
        # Plot data.
        if pp:
            ax.scatter(
                da.x,
                da.y,
                c=list(da.x),
                s=69,
                edgecolors=clr,
                label=lbl,
                vmin=pp.hue_norm[0],
                vmax=pp.hue_norm[1],
                cmap=pp.palette,
            )
        else:
            ax.plot(da.x, da.y, "o", color=clr, label=lbl)
        # Plot fitting.
        ax.plot(xfit[lbl], yfit[lbl], "-", color="gray")
        if da.w is not None:
            ye = 1 / da.w
            ax.errorbar(
                da.x,
                da.y,
                yerr=ye,
                xerr=xerr,
                fmt="none",
                color="k",
                alpha=0.2,
                capsize=3,
            )
        current_ylim = ax.get_ylim()
        ylim = (min(current_ylim[0], ylim[0]), max(current_ylim[1], ylim[1]))
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
                xfit[lbl], yfit[lbl] - dy, yfit[lbl] + dy, alpha=0.4, color=clr
            )
    ax.set_ylim(ylim)
    ax.legend()
    if params["K"].stderr:  # Can be None in case of critical fitting
        k = ufloat(params["K"].value, params["K"].stderr)
    else:
        k = f'{params["K"].value:.3g}' if params["K"].value else None
    title = "=".join(["K", str(k).replace("+/-", "±")])
    xlabel = "pH" if ds.is_ph else "Cl"
    _apply_common_plot_style(ax, f"LM fit {title}", xlabel, "")
    plt.close()
    return fig


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
    plot_fit(ax4, ds, f_res.result, nboot=N_BOOT, pp=pparams)
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


@dataclass
class _ODRResult:
    parameters: Parameters
    output: odr.Output


def odr_fitting(
    x: ArrayF,
    x_err: ArrayF,
    y_list: list[ArrayF],
    y_err_list: list[ArrayF],
    pars: Parameters,
) -> _ODRResult:
    """Fit ODR model on two datasets."""
    # Combine y1 and y2 data and errors
    n_dataset = len(y_list)
    y_data = np.concatenate(y_list)
    y_err = np.concatenate(y_err_list)
    x_data_combined = np.tile(x, n_dataset)
    x_err_combined = np.tile(x_err, n_dataset)

    data = odr.RealData(x_data_combined, y_data, sx=x_err_combined, sy=y_err)

    def model_function(pars: list[float], x: ArrayF) -> ArrayF:
        K, S0, S1 = pars  # noqa: N806
        return binding_1site(x, K, S0, S1, is_ph=True)

    def combined_model(pars: list[float], x: ArrayF) -> ArrayF:
        K, S0_1, S1_1, S0_2, S1_2 = pars  # noqa: N806
        y1_model = model_function([K, S0_1, S1_1], x[: len(x) // 2])
        y2_model = model_function([K, S0_2, S1_2], x[len(x) // 2 :])
        return np.concatenate((y1_model, y2_model))

    combined_model_odr = odr.Model(combined_model)
    # combined_model_odr = odr.Model(model_function)  # noqa: ERA001

    keys = ["K", "S0_y1", "S1_y1", "S0_y2", "S1_y2"]
    # keys = ["K", "S0_default", "S1_default"]  # noqa: ERA001
    initial_params = [pars[key].value for key in keys]

    # Create ODR objects
    odr_obj = odr.ODR(data, combined_model_odr, beta0=initial_params)
    # Run the fitting
    output = odr_obj.run()

    params = Parameters()
    for name, value, error in zip(keys, output.beta, output.sd_beta, strict=True):
        params.add(name, value=value)
        params[name].stderr = error
    return _ODRResult(params, output)
