"""Fit Cl binding and pH titration."""

from __future__ import annotations

import copy
import typing
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from sys import float_info

import arviz as az
import emcee  # type: ignore[import-untyped]
import lmfit  # type: ignore[import-untyped]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm  # type: ignore[import-untyped]
import pytensor
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

if typing.TYPE_CHECKING:
    from clophfit.prtecan import PlateScheme
    from clophfit.types import ArrayDict, ArrayF

ArrayMask = np.typing.NDArray[np.bool_]
N_BOOT = 20  # To compute fill_between uncertainty.
EMCEE_STEPS = 1800


@dataclass
class DataArray:
    """A collection of matching x, y, and optional w data arrays."""

    #: x at creation
    xc: ArrayF
    #: y at creation
    yc: ArrayF
    #: x_err at creation
    x_errc: ArrayF = field(init=True, default_factory=lambda: np.array([]))
    #: y_err at creation
    y_errc: ArrayF = field(init=True, default_factory=lambda: np.array([]))
    _mask: ArrayMask = field(init=False)

    def __post_init__(self) -> None:
        """Ensure the x and y arrays are of equal length after initialization."""
        self._validate_lengths()
        self._validate_yerrc_lengths()
        self._validate_xerrc_lengths()
        self._mask = ~np.isnan(self.yc)

    def _validate_lengths(self) -> None:
        """Validate that xc and yc have the same length."""
        if len(self.xc) != len(self.yc):
            msg = "Length of 'xc' and 'yc' must be equal."
            raise ValueError(msg)

    def _validate_yerrc_lengths(self) -> None:
        """Validate that xc and wc have the same length."""
        if self.y_errc.size > 0 and len(self.xc) != len(self.y_errc):
            msg = "Length of 'xc' and 'y_errc' must be equal."
            raise ValueError(msg)

    def _validate_xerrc_lengths(self) -> None:
        """Validate that xc and wc have the same length."""
        if self.x_errc.size > 0 and len(self.xc) != len(self.x_errc):
            msg = "Length of 'xc' and 'x_errc' must be equal."
            raise ValueError(msg)

    @property
    def mask(self) -> ArrayMask:
        """Mask."""
        return self._mask

    @mask.setter
    def mask(self, mask: ArrayMask) -> None:
        """Only boolean where yc is not nan are considered."""
        self._mask = mask & ~np.isnan(self.yc)

    @property
    def x(self) -> ArrayF:
        """Masked x."""
        return self.xc[self.mask]

    @property
    def y(self) -> ArrayF:
        """Masked y."""
        return self.yc[self.mask]

    @property
    def y_err(self) -> ArrayF:
        """Masked y_err."""
        if self.y_errc.size == 0:
            self.y_errc = np.ones_like(self.xc)
        return self.y_errc[self.mask]

    @y_err.setter
    def y_err(self, y_errc: ArrayF) -> None:
        """Set y_err and validate its length."""
        if y_errc.ndim == 0:
            y_errc = np.ones_like(self.xc) * y_errc
        self.y_errc = y_errc
        self._validate_yerrc_lengths()

    @property
    def x_err(self) -> ArrayF:
        """Masked x_err."""
        if self.x_errc.size == 0:
            self.x_errc = np.ones_like(self.xc)
        return self.x_errc[self.mask]

    @x_err.setter
    def x_err(self, x_errc: ArrayF) -> None:
        """Set x_err and validate its length."""
        if x_errc.ndim == 0:
            x_errc = np.ones_like(self.xc) * x_errc
        self.x_errc = x_errc
        self._validate_xerrc_lengths()


class Dataset(dict[str, DataArray]):
    """Holds datasets as key-value pairs of strings and `DataArray` objects.

    Parameters
    ----------
    data : dict[str, DataArray]
        Dictionary mapping string keys to `DataArray` instances.
    is_ph : bool, optional
        Indicate if x values represent pH (default is False).
    """

    is_ph: bool = False

    def __init__(self, data: dict[str, DataArray], is_ph: bool = False) -> None:
        super().__init__(data or {})
        self.is_ph = is_ph

    @classmethod
    def from_da(cls, da: DataArray | list[DataArray], is_ph: bool = False) -> Dataset:
        """Alternative constructor to create Dataset from a list of DataArray.

        Parameters
        ----------
        da : DataArray | list[DataArray]
            The DataArray objects to populate the dataset.
        is_ph : bool, optional
            Indicate if x values represent pH (default is False).

        Returns
        -------
        Dataset
            The constructed Dataset object.
        """
        if not da:
            return cls({})
        if isinstance(da, list):
            data = {f"y{i}": da_item for i, da_item in enumerate(da)}
        elif isinstance(da, DataArray):
            data = {"default": da}
        return cls(data, is_ph)

    def apply_mask(self, combined_mask: ArrayMask) -> None:
        """Correctly distribute and apply the combined mask across all DataArrays.

        Parameters
        ----------
        combined_mask : ArrayMask
            Boolean array where True keeps the data point, and False masks it out.

        Raises
        ------
        ValueError
            If the length of the combined_mask does not match the total number of data points.
        """
        if combined_mask.size != sum(len(da.y) for da in self.values()):
            msg = "Length of combined_mask must match the total number of data points."
            raise ValueError(msg)
        start_idx = 0
        for da in self.values():
            end_idx = start_idx + len(da.y)
            da.mask[da.mask] &= combined_mask[start_idx:end_idx]
            start_idx = end_idx

    def copy(self, keys: list[str] | set[str] | None = None) -> Dataset:
        """Return a copy of the Dataset.

        If keys are provided, only data associated with those keys are copied.

        Parameters
        ----------
        keys : list[str] | set[str] | None, optional
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
            return copy.deepcopy(self)
        copied = Dataset({}, is_ph=self.is_ph)
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

    def concatenate_data(self) -> tuple[ArrayF, ArrayF, ArrayF, ArrayF]:
        """Concatenate x, y, x_err, and y_err across all datasets."""
        x_data = np.concatenate([v.x for v in self.values()])
        y_data = np.concatenate([v.y for v in self.values()])
        x_err = np.concatenate([v.x_err for v in self.values()])
        y_err = np.concatenate([v.y_err for v in self.values()])
        return x_data, y_data, x_err, y_err

    def export(self, filep: str | Path) -> None:
        """Export this dataset into a csv file."""
        fp = Path(filep)
        for lbl, da in self.items():
            data: dict[str, ArrayF | ArrayMask] = {"xc": da.xc, "yc": da.yc}
            if da.x_errc.size > 0:
                data["x_errc"] = da.x_errc
            if da.y_errc.size > 0:
                data["y_errc"] = da.y_errc
            data["mask"] = da.mask
            pd.DataFrame(data).to_csv(fp.with_stem(f"{fp.stem}_{lbl}"), index=False)


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
        k = f'{params["K"].value:.3g}' if params["K"].value else None
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
    # # TODO: drop outlier     masks = [da._mask for da in ds.values()]
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


def fit_binding_emcee(fit_result: FitResult, n_sd: int = 10) -> FitResult:  # noqa: PLR0915
    """Analyze multi-label titration datasets using emcee."""
    if fit_result.result is None or fit_result.dataset is None:
        return FitResult()
    p = fit_result.result.params
    S0_y1_inf = p["S0_y0"].value - p["S0_y0"].stderr * n_sd  # noqa: N806
    S0_y1_sup = p["S0_y0"].value + p["S0_y0"].stderr * n_sd  # noqa: N806
    S1_y1_inf = p["S1_y0"].value - p["S1_y0"].stderr * n_sd  # noqa: N806
    S1_y1_sup = p["S1_y0"].value + p["S1_y0"].stderr * n_sd  # noqa: N806

    S0_y2_inf = p["S0_y1"].value - p["S0_y1"].stderr * n_sd  # noqa: N806
    S0_y2_sup = p["S0_y1"].value + p["S0_y1"].stderr * n_sd  # noqa: N806
    S1_y2_inf = p["S1_y1"].value - p["S1_y1"].stderr * n_sd  # noqa: N806
    S1_y2_sup = p["S1_y1"].value + p["S1_y1"].stderr * n_sd  # noqa: N806

    K_inf = p["K"].value - p["K"].stderr * n_sd  # noqa: N806
    K_sup = p["K"].value + p["K"].stderr * n_sd  # noqa: N806

    # Define the log-prior
    def log_prior(params: list[float]) -> float:
        K, S0_1, S1_1, S0_2, S1_2 = params  # noqa: N806
        if (
            S0_y1_inf < S0_1 < S0_y1_sup
            and S1_y1_inf < S1_1 < S1_y1_sup
            and S0_y2_inf < S0_2 < S0_y2_sup
            and S1_y2_inf < S1_2 < S1_y2_sup
            and K_inf < K < K_sup
        ):
            return 0.0
        return -np.inf

    ds = fit_result.dataset
    dataset_lengths = [len(da.y) for da in ds.values()]

    def log_likelihood(
        params: list[float], x: ArrayF, y: ArrayF, xerr: ArrayF, yerr: ArrayF
    ) -> float:
        # TODO: x_adjusted = x + np.random.normal(0, xerr)
        np.sum(xerr)
        model = generalized_combined_model(params, x, dataset_lengths)
        sigma2 = yerr**2
        log_ = -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))
        return float(log_)

    def log_posterior(
        params: list[float], x: ArrayF, y: ArrayF, xerr: ArrayF, yerr: ArrayF
    ) -> float:
        lp = log_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(params, x, y, xerr, yerr)

    # Initial parameters setup
    initial_params = [p["K"].value]
    for lbl in ds:
        initial_params.extend([p[f"S0_{lbl}"].value, p[f"S1_{lbl}"].value])
    # Number of dimensions, walkers, and steps
    ndim = len(initial_params)
    nwalkers = 50
    nsteps = 5000
    # Initialize the walkers
    rng = np.random.default_rng()
    pos = initial_params + 1e-4 * rng.standard_normal((nwalkers, ndim))

    args = ds.concatenate_data()  # (x, y, x_err, y_err)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=args)
    sampler.run_mcmc(pos, nsteps, progress=True)
    # Extract the samples
    samples = sampler.get_chain(discard=100, thin=10, flat=True)
    # Ensure that samples are shaped as (chain, draw)
    if samples.ndim == 2:  # noqa: PLR2004
        samples = np.expand_dims(samples, axis=0)
    # Calculate the HDI for each parameter
    hdi_intervals = az.hdi(samples, hdi_prob=0.94)  # type: ignore[no-untyped-call]
    keys = ["K"]
    for lbl in ds:
        keys.append(f"S0_{lbl}")
        keys.append(f"S1_{lbl}")
    pars = Parameters()
    for i, name in enumerate(keys):
        hdi_lower = hdi_intervals[i][0]  # Access HDI intervals by index
        hdi_upper = hdi_intervals[i][1]
        median_value = np.median(samples[:, :, i])  # Use appropriate indexing
        std_value = np.std(samples[:, :, i])  # Use appropriate indexing
        pars.add(name, value=median_value, min=hdi_lower, max=hdi_upper)
        pars[name].stderr = std_value
    fig = figure.Figure()
    ax = fig.add_subplot(111)
    plot_fit(ax, ds, pars, nboot=N_BOOT, pp=PlotParameters(ds.is_ph))
    return FitResult(fig, _Result(pars), samples, None)


def fit_binding_pymc(  # noqa: PLR0912,C901
    fr: FitResult, mth: str = "norm", n_sd: float = 10.0, n_xerr: float = 1.0
) -> FitResult:
    """Analyze multi-label titration datasets using emcee."""
    if fr.result is None or fr.dataset is None:
        return FitResult()
    params = fr.result.params
    ds = copy.deepcopy(fr.dataset)
    p_names = ["K"]
    for lbl in ds:
        p_names.append(f"S0_{lbl}")
        p_names.append(f"S1_{lbl}")
    init_pars = [params["K"].value]
    for lbl in ds:
        init_pars.extend([params[f"S0_{lbl}"].value, params[f"S1_{lbl}"].value])

    with pm.Model() as _:
        if mth == "norm":
            pars = {
                p.name: pm.Normal(p.name, mu=p.value, sigma=p.stderr * n_sd)
                for p in params.values()
            }
        elif mth == "cov":
            if hasattr(fr.result, "covar"):
                cov = fr.result.covar
            elif fr.mini:
                cov = fr.mini.cov_beta
            parameters = pm.MvNormal(
                "parameters", mu=init_pars, cov=cov * n_sd, shape=len(p_names)
            )
            pars = dict(zip(p_names, parameters, strict=True))

        # Loop over datasets and create the likelihood for each
        xc = next(iter(ds.values())).xc
        if n_xerr:
            x_errc = next(iter(ds.values())).x_errc * n_xerr
            # the first pH value is the less uncertain
            x_errc[0] = next(iter(ds.values())).x_errc[0] * n_xerr / 10
            x_true = pm.Normal("x_true", mu=xc, sigma=x_errc, shape=len(xc))
        else:
            x_true = xc

        ye_mag = pm.HalfNormal("ye_mag", sigma=10)
        for lbl, da in ds.items():
            y_model = binding_1site(
                x_true, pars["K"], pars[f"S0_{lbl}"], pars[f"S1_{lbl}"], ds.is_ph
            )
            pm.Normal(
                f"y_obs_{lbl}",
                mu=y_model[da.mask],
                sigma=ye_mag * da.y_err,
                observed=da.y,
            )
        # Inference
        trace: ArrayF = pm.sample(
            2000, tune=2000, target_accept=0.9, cores=4, return_inferencedata=True
        )

    rdf = az.summary(trace)
    rpars = Parameters()
    # Loop through each row in the dataframe and extract necessary values
    for name, row in rdf.iterrows():
        if name in p_names:
            rpars.add(name, value=row["mean"], min=row["hdi_3%"], max=row["hdi_97%"])
            rpars[name].stderr = row["sd"]
            rpars[name].init_value = row["r_hat"]
    if n_xerr:
        # Initialize lists to store xc and x_errc values
        nxc = []
        nx_errc = []
        # Loop through the dataframe rows corresponding to x_true[?]
        for name, row in rdf.iterrows():
            if isinstance(name, str) and name.startswith("x_true"):
                nxc.append(row["mean"])
                nx_errc.append(row["sd"])
        for da in ds.values():
            da.xc = np.array(nxc)
            da.x_errc = np.array(nx_errc)
    mag: float = rdf.loc["ye_mag", "mean"]  # type: ignore[assignment, index]
    for da in ds.values():
        da.y_errc *= mag

    fig = figure.Figure()
    ax = fig.add_subplot(111)
    if mth == "norm":
        plot_fit(ax, ds, rpars, nboot=N_BOOT, pp=PlotParameters(ds.is_ph))
    return FitResult(fig, _Result(rpars), trace, ds)


def fit_binding_pymc_many(
    result: dict[str, FitResult], n_sd: float = 5, n_xerr: float = 5
) -> ArrayF:
    """Analyze multi-label titration datasets using emcee."""
    ds = next(iter(result.values())).dataset
    while ds is None:
        ds = next(iter(result.values())).dataset
    xc = next(iter(ds.values())).xc
    x_errc = next(iter(ds.values())).x_errc * n_xerr
    # the first pH value is the less uncertain
    x_errc[0] = next(iter(ds.values())).x_errc[0] * n_xerr / 10

    with pm.Model() as _:
        x_true = pm.Normal("x_true", mu=xc, sigma=x_errc, shape=len(xc))
        ye_mag = pm.HalfNormal("ye_mag", sigma=10)

        for key, r in result.items():
            if r.result and r.dataset:
                ds = r.dataset
                da0 = ds["y0"]
                da1 = ds["y1"]
                pars = r.result.params
                K = pm.Normal(  # noqa: N806
                    f"K_{key}", mu=pars["K"].value, sigma=pars["K"].stderr * n_sd
                )
                S0_y0 = pm.Normal(  # noqa: N806
                    f"S0_y0_{key}",
                    mu=pars["S0_y0"].value,
                    sigma=pars["S0_y0"].stderr * n_sd,
                )
                S1_y0 = pm.Normal(  # noqa: N806
                    f"S1_y0_{key}",
                    mu=pars["S1_y0"].value,
                    sigma=pars["S1_y0"].stderr * n_sd,
                )
                S0_y1 = pm.Normal(  # noqa: N806
                    f"S0_y1_{key}",
                    mu=pars["S0_y1"].value,
                    sigma=pars["S0_y1"].stderr * n_sd,
                )
                S1_y1 = pm.Normal(  # noqa: N806
                    f"S1_y1_{key}",
                    mu=pars["S1_y1"].value,
                    sigma=pars["S1_y1"].stderr * n_sd,
                )

                # Model equations
                y0_model = binding_1site(x_true[da0.mask], K, S0_y0, S1_y0, ds.is_ph)
                y1_model = binding_1site(x_true[da1.mask], K, S0_y1, S1_y1, ds.is_ph)

                # Likelihood
                pm.Normal(
                    f"y0_likelihood_{key}",
                    mu=y0_model,
                    sigma=da0.y_err * ye_mag,
                    observed=da0.y,
                )
                pm.Normal(
                    f"y1_likelihood_{key}",
                    mu=y1_model,
                    sigma=da1.y_err * ye_mag,
                    observed=da1.y,
                )

        # Inference
        trace: ArrayF = pm.sample(2000, return_inferencedata=True)

    return trace


def fit_binding_pymc_many_scheme(
    result: dict[str, FitResult],
    scheme: PlateScheme,
    n_sd: float = 5.0,
    n_xerr: float = 5.0,
) -> ArrayF:
    """Analyze multi-label titration datasets using shared control parameters."""
    pytensor.config.floatX = "float32"  # type: ignore[attr-defined]
    ds = next(iter(result.values())).dataset
    while ds is None:
        ds = next(iter(result.values())).dataset
    xc = next(iter(ds.values())).xc
    x_errc = next(iter(ds.values())).x_errc * n_xerr
    # the first pH value is the less uncertain
    x_errc[0] = next(iter(ds.values())).x_errc[0] * n_xerr / 10

    values = {}
    stderr = {}
    for name, wells in scheme.names.items():
        values[name] = [
            v.result.params["K"].value
            for well, v in result.items()
            if v.result and well in wells
        ]
        stderr[name] = [
            v.result.params["K"].stderr
            for well, v in result.items()
            if v.result and well in wells
        ]
    ctr_ks = weighted_stats(values, stderr)
    print(ctr_ks)

    with pm.Model() as _:
        # Priors for global parameters
        x_true = pm.Normal("x_true", mu=xc, sigma=x_errc, shape=len(xc))
        ye_mag = pm.HalfNormal("ye_mag", sigma=10)

        # Create shared K parameters for each control group
        k_params = {
            ctr_name: pm.Normal(f"K_{ctr_name}", mu=ctr_ks[ctr_name][0], sigma=0.2)
            for ctr_name in scheme.names
        }

        for key, r in result.items():
            if r.result and r.dataset:
                ds = r.dataset
                da0 = ds["y0"]
                da1 = ds["y1"]
                pars = r.result.params

                # Determine if the well is associated with a control group
                ctr_name = next(
                    (name for name, wells in scheme.names.items() if key in wells), None
                )

                if ctr_name:
                    K = k_params[ctr_name]  # noqa: N806 # Shared K for this control group
                else:
                    # If not part of any control group, create an individual K
                    K = pm.Normal(  # noqa: N806
                        f"K_{key}", mu=pars["K"].value, sigma=pars["K"].stderr * n_sd
                    )

                # Per-well S0 and S1 parameters
                S0_y0 = pm.Normal(  # noqa: N806
                    f"S0_y0_{key}",
                    mu=pars["S0_y0"].value,
                    sigma=pars["S0_y0"].stderr * n_sd,
                )
                S1_y0 = pm.Normal(  # noqa: N806
                    f"S1_y0_{key}",
                    mu=pars["S1_y0"].value,
                    sigma=pars["S1_y0"].stderr * n_sd,
                )
                S0_y1 = pm.Normal(  # noqa: N806
                    f"S0_y1_{key}",
                    mu=pars["S0_y1"].value,
                    sigma=pars["S0_y1"].stderr * n_sd,
                )
                S1_y1 = pm.Normal(  # noqa: N806
                    f"S1_y1_{key}",
                    mu=pars["S1_y1"].value,
                    sigma=pars["S1_y1"].stderr * n_sd,
                )

                # Model equations
                y0_model = binding_1site(
                    x_true[da0.mask.astype(bool)], K, S0_y0, S1_y0, ds.is_ph
                )
                y1_model = binding_1site(
                    x_true[da1.mask.astype(bool)], K, S0_y1, S1_y1, ds.is_ph
                )

                # Likelihoods
                pm.Normal(
                    f"y0_likelihood_{key}",
                    mu=y0_model,
                    sigma=da0.y_err * ye_mag,
                    observed=da0.y,
                )
                pm.Normal(
                    f"y1_likelihood_{key}",
                    mu=y1_model,
                    sigma=da1.y_err * ye_mag,
                    observed=da1.y,
                )

        # Inference
        trace: ArrayF = pm.sample(
            2000,
            target_accept=0.9,
            cores=4,
            chains=6,
            return_inferencedata=True,
            init="adapt_diag",
            initvals={"ye_mag": 1.0},
        )

    return trace


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
