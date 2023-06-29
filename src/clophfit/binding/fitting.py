"""Fit Cl binding and pH titration."""
from __future__ import annotations

import typing
from dataclasses import InitVar, dataclass, field
from typing import Sequence

import arviz as az
import lmfit  # type: ignore
import matplotlib as mpl  # type: ignore
import numpy as np
import pandas as pd
import scipy  # type: ignore
import scipy.stats  # type: ignore
import seaborn as sb  # type: ignore # noqa: ICN001
from lmfit import Parameters
from lmfit.minimizer import Minimizer, MinimizerResult  # type: ignore
from uncertainties import ufloat  # type: ignore

from clophfit.types import ArrayDict, ArrayF

COLOR_MAP = mpl.cm.Paired  # for PCA components and LM fit


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


class Dataset(typing.Dict[str, DataArrays]):
    """A dataset containing pairs of matching x and y data arrays, indexed by a string key."""

    is_ph: bool

    def __init__(
        self,
        x: ArrayF | ArrayDict,
        y: ArrayF | ArrayDict,
        is_ph: bool = False,
        w: ArrayF | ArrayDict | None = None,
    ) -> None:
        """
        Initialize the Dataset object.

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
        w : ArrayF | ArrayDict, optional
            The w values (weights) of the dataset(s), either as a single ArrayF or an ArrayDict
            if multiple datasets are provided.

        Raises
        ------
        ValueError
            If x and y are both ArrayDict and their keys don't match.
        """
        self.is_ph = is_ph
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            weights = w if isinstance(w, np.ndarray) else None
            super().__init__({"default": DataArrays(x, y, weights)})
        elif isinstance(x, np.ndarray) and isinstance(y, dict):
            if isinstance(w, dict):
                super().__init__({k: DataArrays(x, v, w.get(k)) for k, v in y.items()})
            else:
                # this cover w is None or ArrayF
                super().__init__({k: DataArrays(x, v, w) for k, v in y.items()})
        elif isinstance(x, dict) and isinstance(y, dict):
            if x.keys() != y.keys() or (isinstance(w, dict) and x.keys() != w.keys()):
                msg = "Keys of 'x', 'y', and 'w' (if w is a dict) must match."
                raise ValueError(msg)
            if isinstance(w, dict):
                super().__init__({k: DataArrays(x[k], y[k], w.get(k)) for k in x})
            else:
                super().__init__({k: DataArrays(x[k], y[k], w) for k in x})

    def add_weights(self, w: ArrayF | ArrayDict | typing.Any) -> None:
        """Add weights to the dataset.

        Parameters
        ----------
        w : ArrayF | ArrayDict
            The weights to be added to the dataset.

        Raises
        ------
        ValueError
            If a key in the weights dictionary does not match any key in the current Dataset object.
        TypeError
            If the type of weights is not np.ndarray or dict.
        """
        if isinstance(w, np.ndarray):
            for da in self.values():
                da.w = w
        elif isinstance(w, dict):
            for k, weights in w.items():
                if k in self:
                    self[k].w = weights
                else:
                    msg = f"No matching dataset found for key '{k}' in the current Dataset object."
                    raise ValueError(msg)
        else:
            msg = "Invalid type for w. Expected np.ndarray or dict."
            raise TypeError(msg)


@typing.overload
def binding_1site(
    x: float, K: float, S0: float, S1: float, is_ph: bool = False  # noqa: N803
) -> float:
    ...


@typing.overload
def binding_1site(
    x: ArrayF, K: float, S0: float, S1: float, is_ph: bool = False  # noqa: N803
) -> ArrayF:
    ...


def binding_1site(
    x: float | ArrayF, K: float, S0: float, S1: float, is_ph: bool = False  # noqa: N803
) -> float | ArrayF:
    """Single site binding model function.

    Parameters
    ----------
    x : float | np.ndarray
        Concentration values.
    K : float
        Dissociation constant.
    S0 : float
        Plateau value for the unbound state.
    S1 : float
        Plateau value for the bound state.
    is_ph : bool
        If True, use the pH model for binding. Default is False.

    Returns
    -------
    float | np.ndarray
        Modeled binding values.

    Note:
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
    """Compute concatenated residuals (array) for multiple datasets; or model predictions (dict)."""
    x, y, w, is_ph = _init_from_dataset(ds)
    models = _binding_1site_models(params, x, is_ph)
    residuals: ArrayF = np.concatenate([(w[lbl] * (y[lbl] - models[lbl])) for lbl in x])
    return residuals


@typing.overload
def _binding_pk(x: float, K: float, S0: float, S1: float) -> float:  # noqa: N803
    ...


@typing.overload
def _binding_pk(x: ArrayF, K: float, S0: float, S1: float) -> ArrayF:  # noqa: N803
    ...


def _binding_pk(
    x: float | ArrayF, K: float, S0: float, S1: float  # noqa: N803
) -> float | ArrayF:
    return S0 + (S1 - S0) * 10 ** (K - x) / (1 + 10 ** (K - x))


def _binding_pkr(
    x: float | ArrayF, K: float, R: float, S1: float  # noqa: N803
) -> float | ArrayF:
    return S1 * (R + (1 - R) * 10 ** (K - x) / (1 + 10 ** (K - x)))


@typing.overload
def _binding_kd(x: float, K: float, S0: float, S1: float) -> float:  # noqa: N803
    ...


@typing.overload
def _binding_kd(x: ArrayF, K: float, S0: float, S1: float) -> ArrayF:  # noqa: N803
    ...


def _binding_kd(
    x: float | ArrayF, K: float, S0: float, S1: float  # noqa: N803
) -> float | ArrayF:
    return S1 + (S0 - S1) * x / K / (1 + x / K)


def kd(kd1: float, pka: float, ph: ArrayF | float) -> ArrayF | float:
    """Infinite cooperativity model.

    It can describe pH-dependence for chloride dissociation constant.

    Parameters
    ----------
    kd1 : float
        Dissociation constant at pH <= 5.0 (fully protonated).
    pka : float
        Acid dissociation constant.
    ph : Xtype
        pH value(s).

    Returns
    -------
    Xtype
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


# TODO other from datan
# TODO: use this like fz in prtecan
def fz_kd_singlesite(k: float, p: ArrayF | Sequence[float], x: ArrayF) -> ArrayF:
    """Fit function for Cl titration."""
    return (float(p[0]) + float(p[1]) * x / k) / (1 + x / k)


def fz_pk_singlesite(k: float, p: ArrayF | Sequence[float], x: ArrayF) -> ArrayF:
    """Fit function for pH titration."""
    return (float(p[1]) + float(p[0]) * 10 ** (k - x)) / (1 + 10 ** (k - x))


# `fit_titration` is exported and use in prtecan.
def fit_titration(  # noqa: PLR0913, PLR0915
    kind: str,
    x: Sequence[float],
    y: ArrayF,
    y2: ArrayF | None = None,
    residue: ArrayF | None = None,
    residue2: ArrayF | None = None,
    tval_conf: float = 0.95,
) -> pd.DataFrame:
    """Fit pH or Cl titration using a single-site binding model.

    Returns confidence interval (default=0.95) for fitting params (cov*tval), rather than
    standard error of the fit. Use scipy leastsq. Determine 3 fitting parameters:
    - binding constant *K*
    - and 2 plateau *SA* and *SB*.

    Parameters
    ----------
    kind : str
        Titration type {'pH'|'Cl'}
    x : Sequence[float]
        Dataset x-values.
    y : ArrayF
        Dataset y-values.
    y2 : ArrayF, optional
        Optional second dataset y-values (share x with main dataset).
    residue : ArrayF, optional
        Residues for main dataset.
    residue2 : ArrayF, optional
        Residues for second dataset.
    tval_conf : float
        Confidence level (default 0.95) for parameter estimations.

    Returns
    -------
    pd.DataFrame
        Fitting results.

    Raises
    ------
    NameError
        When kind is different than "pH" or "Cl".

    Examples
    --------
    >>> import numpy as np
    >>> fit_titration("Cl", np.array([1.0, 10, 30, 100, 200]), \
          np.array([10, 8, 5, 1, 0.1]))[["K", "sK"]]
               K         sK
    0  38.955406  30.201929

    """
    if kind == "pH":
        fz = fz_pk_singlesite
    elif kind == "Cl":
        fz = fz_kd_singlesite
    else:
        msg = "kind= pH or Cl"
        raise NameError(msg)

    def compute_p0(x: Sequence[float], y: ArrayF) -> ArrayF:
        data = pd.DataFrame({"x": x, "y": y})
        p0sa = data.y[data.x == min(data.x)].to_numpy()[0]
        p0sb = data.y[data.x == max(data.x)].to_numpy()[0]
        p0k = np.average([max(y), min(y)])
        try:
            x1, y1 = data[data["y"] >= p0k].to_numpy()[0]
        except IndexError:
            x1 = np.nan
            y1 = np.nan
        try:
            x2, y2 = data[data["y"] <= p0k].to_numpy()[0]
        except IndexError:
            x2 = np.nan
            y2 = np.nan
        p0k = (x2 - x1) / (y2 - y1) * (p0k - y1) + x1
        return np.array(np.r_[p0k, p0sa, p0sb])

    if y2 is None:

        def ssq1(p: ArrayF, x: ArrayF, y1: ArrayF) -> ArrayF:
            return np.array(np.r_[y1 - fz(p[0], p[1:3], x)])

        p0 = compute_p0(x, y)
        p, cov, info, msg, success = scipy.optimize.leastsq(
            ssq1, p0, args=(np.array(x), y), full_output=True, xtol=1e-11
        )
    else:

        def ssq2(  # noqa: PLR0913
            p: ArrayF, x: ArrayF, y1: ArrayF, y2: ArrayF, rd1: ArrayF, rd2: ArrayF
        ) -> ArrayF:
            return np.array(
                np.r_[
                    (y1 - fz(p[0], p[1:3], x)) / rd1**2,
                    (y2 - fz(p[0], p[3:5], x)) / rd2**2,
                ]
            )

        p1 = compute_p0(x, y)
        p2 = compute_p0(x, y2)
        ave = np.average([p1[0], p2[0]])
        p0 = np.r_[ave, p1[1], p1[2], p2[1], p2[2]]
        tmp = scipy.optimize.leastsq(
            ssq2,
            p0,
            full_output=True,
            xtol=1e-11,
            args=(np.array(x), y, y2, residue, residue2),
        )
        p, cov, info, msg, success = tmp
    res = pd.DataFrame({"ss": [success]})
    res["msg"] = msg
    if 1 <= success <= 4:  # noqa: PLR2004
        try:
            tval = (tval_conf + 1) / 2
            chisq = sum(info["fvec"] * info["fvec"])
            res["df"] = len(y) - len(p)
            res["tval"] = scipy.stats.distributions.t.ppf(tval, res.df)
            res["chisqr"] = chisq / res.df
            res["K"] = p[0]
            res["SA"] = p[1]
            res["SB"] = p[2]
            if y2 is not None:
                res["df"] += len(y2)
                res["tval"] = scipy.stats.distributions.t.ppf(tval, res.df)
                res["chisqr"] = chisq / res.df
                res["SA2"] = p[3]
                res["SB2"] = p[4]
                res["sSA2"] = np.sqrt(cov[3][3] * res.chisqr) * res.tval
                res["sSB2"] = np.sqrt(cov[4][4] * res.chisqr) * res.tval
            res["sK"] = np.sqrt(cov[0][0] * res.chisqr) * res.tval
            res["sSA"] = np.sqrt(cov[1][1] * res.chisqr) * res.tval
            res["sSB"] = np.sqrt(cov[2][2] * res.chisqr) * res.tval
        except TypeError:
            pass  # if some params are not successfully determined.
    return res


###############
def _build_params_1site(ds: Dataset) -> Parameters:
    """Initialize parameters for 1 site model based on the given dataset."""
    params = Parameters()
    k_initial = []
    for lbl, da in ds.items():
        params.add(f"S0_{lbl}", value=da.y[0], min=0)
        params.add(f"S1_{lbl}", value=da.y[-1])
        target_y = (da.y[0] + da.y[-1]) / 2
        k_initial.append(da.x[np.argmin(np.abs(da.y - target_y))])
    if ds.is_ph:
        params.add("K", value=np.mean(k_initial), min=3, max=11)
    else:
        params.add("K", value=np.mean(k_initial), min=0)
    return params


@dataclass
class PlotParameters:
    """Parameters for plotting, depending on whether the data is pH or Cl."""

    is_ph: InitVar[bool]
    hue_norm: tuple[float, float] = field(init=False, repr=True)
    palette: str = field(init=False, repr=True)
    kind: str = field(init=False, repr=True)

    def __post_init__(self, is_ph: bool) -> None:
        """Set attributes based on whether the data is pH or Cl."""
        if is_ph:
            self.hue_norm = (5.7, 8.7)
            self.palette = "vlag_r"
            self.kind = "pH"
        else:
            self.hue_norm = (0.0, 200.0)
            self.palette = "crest"
            self.kind = "Cl"


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

    figure: mpl.figure.Figure
    result: MinimizerResult
    mini: Minimizer


@dataclass
class SpectraGlobResults:
    """A dataclass representing the results of both svd and bands fits.

    Attributes
    ----------
    svd : FitResult | None
        The `FitResult` object representing the outcome of the svd fit, or `None` if the svd fit was not performed.
    bands : FitResult | None
        The `FitResult` object representing the outcome of the bands fit, or `None` if the bands fit was not performed.
    """

    svd: FitResult | None = field(default=None)
    bands: FitResult | None = field(default=None)


def _apply_common_plot_style(
    ax: mpl.axes.Axes, title: str, xlabel: str, ylabel: str
) -> None:
    """Apply grid style and add title and labels."""
    ax.grid(True)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def plot_spectra(f: pd.DataFrame, ax: mpl.axes.Axes, pp: PlotParameters) -> None:
    """Plot spectra.

    Parameters
    ----------
    f : pd.DataFrame
        The DataFrame containing spectral data.
    ax : mpl.axes.Axes
        The Axes object to which the plot should be added.
    pp : PlotParameters
        The PlotParameters object containing plot parameters.
    """
    color_map = mpl.cm.get_cmap(pp.palette)
    normalize = mpl.colors.Normalize(vmin=pp.hue_norm[0], vmax=pp.hue_norm[1])
    for i in range(len(f.columns)):
        ax.plot(f.index, f.iloc[:, i], color=color_map(normalize(f.columns[i])))
    _apply_common_plot_style(ax, "Spectra", "Wavelength", "Fluorescence")
    # Add a colorbar for reference
    sm = mpl.cm.ScalarMappable(cmap=color_map, norm=normalize)
    sm.set_array([])
    ax.figure.colorbar(sm, ax=ax, label=pp.kind)


def _plot_autovectors(wl: pd.Index, u: ArrayF, ax: mpl.axes.Axes) -> None:
    """Plot autovectors.

    Parameters
    ----------
    wl : pd.Index
        The index of spectra data frame.
    u : ArrayF
        The left singular vectors obtained from SVD.
    ax : mpl.axes.Axes
        The Axes object to which the plot should be added.
    """
    number_autovectors = 4
    for i in range(number_autovectors):
        ax.plot(wl, u[:, i], color=COLOR_MAP(i), lw=3 / (i + 1), alpha=(1 - 0.2 * i))
    _apply_common_plot_style(ax, "Autovectors", "Wavelength", "Magnitude")


def plot_autovalues(s: ArrayF, ax: mpl.axes.Axes) -> None:
    """Plot the singular values from SVD.

    Parameters
    ----------
    s : ArrayF
        The singular values from the SVD.
    ax : mpl.axes.Axes
        The axes on which to plot the singular values.
    """
    data = pd.DataFrame({"index": range(1, len(s) + 1), "singular_values": s})
    sb.scatterplot(
        x="index",
        y="singular_values",
        data=data,
        ax=ax,
        hue="index",
        s=99,
        legend=False,
        palette=COLOR_MAP.name,
    )
    _apply_common_plot_style(ax, "Singular Values from SVD", "Index", "Singular Value")
    ax.set(yscale="log")
    ax.set_xticks(np.arange(1, len(s) + 1))


def plot_fit(
    ax: mpl.axes.Axes,
    ds: Dataset,
    result: MinimizerResult,
    nboot: int = 0,
    pp: PlotParameters | None = None,
) -> None:
    """Plot residuals for each dataset with uncertainty."""
    _fraction = 0.05
    xfit = {
        k: np.linspace(
            da.x.min() - _fraction * (da.x.max() - da.x.min()),
            da.x.max() + _fraction * (da.x.max() - da.x.min()),
            100,
        )
        for k, da in ds.items()
    }
    yfit = _binding_1site_models(result.params, xfit, ds.is_ph)
    # Create a color cycle
    colors = [COLOR_MAP(i) for i in range(len(ds))]
    for (lbl, da), clr in zip(ds.items(), colors):
        # Make sure a label will be displayed
        label = lbl if (da.w is None and nboot == 0) else None
        # Plot data
        if pp:
            kws = {"vmin": pp.hue_norm[0], "vmax": pp.hue_norm[1], "cmap": pp.palette}
            ax.scatter(da.x, da.y, c=da.x, s=99, edgecolors="k", label=label, **kws)
        else:
            ax.plot(da.x, da.y, "o", color=clr, label=label)
        # Plot fitting
        ax.plot(xfit[lbl], yfit[lbl], "-", color="gray")
        kws_err = {"alpha": 0.3, "capsize": 3}
        if nboot:
            # Calculate uncertainty using Monte Carlo method
            y_samples = np.empty((nboot, len(xfit[lbl])))
            for i in range(nboot):
                p_sample = result.params.copy()
                for param in p_sample.values():
                    param.value = np.random.normal(param.value, param.stderr)
                y_samples[i, :] = _binding_1site_models(p_sample, xfit, ds.is_ph)[lbl]
            dy = y_samples.std(axis=0)
            # Plot uncertainty
            kws = {"alpha": 0.4, "color": clr}
            # Display label fill_between
            ax.fill_between(xfit[lbl], yfit[lbl] - dy, yfit[lbl] + dy, **kws, label=lbl)
            if da.w is not None:
                ye = np.sqrt(1 / da.w)
                ax.errorbar(da.x, da.y, yerr=ye, fmt="none", color="gray", **kws_err)
        elif da.w is not None:
            ye = np.sqrt(1 / da.w / result.nfree)
            # Display label errorbar
            ax.errorbar(da.x, da.y, yerr=ye, fmt=".", label=lbl, color=clr, **kws_err)
    ax.grid(True)
    ax.legend()


def plot_pca(v: ArrayF, ax: mpl.axes.Axes, conc: ArrayF, pp: PlotParameters) -> None:
    """Plot the first two principal components.

    Parameters
    ----------
    v : ArrayF
        The matrix containing the principal components.
    ax : mpl.axes.Axes
        The Axes object to which the plot should be added.
    conc : ArrayF
        The concentrations used for the titration.
    pp : PlotParameters
        The PlotParameters object containing plot parameters.
    """
    kws = {"vmin": pp.hue_norm[0], "vmax": pp.hue_norm[1], "cmap": pp.palette}
    ax.scatter(v[1], v[0], c=conc, s=99, edgecolors="k", **kws)
    _apply_common_plot_style(ax, "PCA plot", "", "")
    ax.set_ylabel("First Principal Component", color=COLOR_MAP(0))
    ax.set_xlabel("Second Principal Component", color=COLOR_MAP(1))
    # Add labels.
    for x, y, w in zip(v[1], v[0], conc):
        ax.text(x, y, w)


def analyze_spectra(
    spectra: pd.DataFrame, is_ph: bool, band: tuple[int, int] | None = None
) -> FitResult:
    """Analyze spectra titration, create and plot fit results.

    This function performs either Singular Value Decomposition (SVD) or integrates
    the spectral data over a specified band and fits the integrated data to a binding model.

    Parameters
    ----------
    spectra : pd.DataFrame
        The DataFrame containing spectra (one spectrum for each column).
    is_ph : bool
        Whether the x values should be interpreted as pH values rather than concentrations.
    band : Tuple[int, int] | None
        The band to integrate over. If None (default), performs Singular Value Decomposition (SVD).

    Returns
    -------
    FitResult
        A FitResult object containing the figure, result, and mini.

    Raises
    ------
    ValueError
        If the band parameters are not in the spectra's index when the band method is used.

    Notes
    -----
    Creates plots of spectra, principal component vectors, singular values, fit of the first
    principal component and PCA for SVD; only of spectra and fit for Band method.
    """
    y_offset = 1.0
    x = spectra.columns.to_numpy()
    fig = mpl.figure.Figure(figsize=(12, 8))
    ax1 = fig.add_axes([0.05, 0.65, 0.32, 0.31])
    ax2 = fig.add_axes([0.42, 0.65, 0.32, 0.31])
    ax3 = fig.add_axes([0.80, 0.65, 0.18, 0.31])
    ax4 = fig.add_axes([0.05, 0.08, 0.50, 0.50])
    ax5 = fig.add_axes([0.63, 0.08, 0.35, 0.50])
    plot_spectra(spectra, ax1, PlotParameters(is_ph))
    if band is None:  # SVD
        ddf = spectra.sub(spectra.iloc[:, 0], axis=0)
        u, s, v = np.linalg.svd(ddf)
        ds = Dataset(x, v[0, :] + y_offset, is_ph)
        _plot_autovectors(spectra.index, u, ax2)
        plot_autovalues(s[:], ax3)  # don't plot last auto-values?
        plot_pca(v, ax5, x, PlotParameters(is_ph))
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
        ylabel_color = "k"
    fit_result = fit_binding_glob(ds, True)
    result = fit_result.result
    plot_fit(ax4, ds, result, nboot=20, pp=PlotParameters(is_ph))
    k = ufloat(result.params["K"].value, result.params["K"].stderr)
    title = "=".join(["K", str(k).replace("+/-", "±")])
    xlabel = "pH" if is_ph else "Cl"
    _apply_common_plot_style(ax4, f"LM fit {title}", xlabel, "")
    ax4.set_ylabel(ylabel, color=ylabel_color)
    return FitResult(fig, result, fit_result.mini)


def fit_binding_glob(ds: Dataset, weighting: bool = False) -> FitResult:
    """Analyze multi-label binding datasets and visualize the results."""
    if weighting:
        wc: ArrayDict = {}
        # Calculate standard deviations of residuals
        for label, da in ds.items():
            x = {label: da.x}
            y = {label: da.y}
            d = Dataset(x, y, ds.is_ph)
            params = _build_params_1site(d)
            res = lmfit.minimize(_binding_1site_residuals, params, args=(d,))
            wc[label] = 1 / np.std(res.residual) * np.ones_like(da.x)
        ds.add_weights(wc)
    params = _build_params_1site(ds)
    mini = Minimizer(_binding_1site_residuals, params, fcn_args=(ds,))
    result = mini.minimize()
    fig = mpl.figure.Figure()
    ax = fig.add_subplot(111)
    plot_fit(ax, ds, result, nboot=20, pp=PlotParameters(ds.is_ph))
    return FitResult(fig, result, mini)


def analyze_spectra_glob(
    titration: dict[str, pd.DataFrame],
    ds: Dataset,
    dbands: dict[str, tuple[int, int]] | None = None,
) -> SpectraGlobResults:
    """Analyze multi-label spectra visualize the results."""
    _gap_ = 1
    dbands = dbands or {}
    labels_svd = titration.keys() - dbands.keys()
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
    else:
        svd = None
    bands = fit_binding_glob(ds, True) if len(dbands.keys()) > 1 else None
    return SpectraGlobResults(svd, bands)


def plot_emcee(result_emcee: MinimizerResult) -> None:
    """Plot emcee result."""
    samples = result_emcee.flatchain
    # Convert the dictionary of flatchains to an ArviZ InferenceData object
    samples_dict = {key: np.array(val) for key, val in samples.items()}
    idata = az.from_dict(posterior=samples_dict)
    az.plot_pair(idata, kind=["scatter", "kde"], scatter_kwargs={"marker": ".", "s": 1})
    az.plot_posterior(idata.posterior)  # type: ignore


def plot_emcee_on_ax(result_emcee: MinimizerResult, ax: mpl.axes.Axes) -> None:
    """Plot emcee result."""
    samples = result_emcee.flatchain
    # Convert the dictionary of flatchains to an ArviZ InferenceData object
    samples_dict = {key: np.array(val) for key, val in samples.items()}
    idata = az.from_dict(posterior=samples_dict)
    az.plot_posterior(idata.posterior["K"], ax=ax)  # type: ignore
