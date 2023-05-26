"""Fit Cl binding and pH titration."""
from __future__ import annotations

import typing
from collections import namedtuple
from typing import Sequence

import matplotlib as mpl  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd
import scipy  # type: ignore
import scipy.stats  # type: ignore
import seaborn as sb  # type: ignore # noqa: ICN001
from numpy.typing import NDArray
from scipy import optimize

# Binding equations."""
# TODO: use this like fz in prtecan

COLOR_MAP = plt.cm.Set2


def kd(
    kd1: float, pka: float, ph: NDArray[np.float_] | float
) -> NDArray[np.float_] | float:
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


def fz_kd_singlesite(
    k: float, p: NDArray[np.float_] | Sequence[float], x: NDArray[np.float_]
) -> NDArray[np.float_]:
    """Fit function for Cl titration."""
    return (float(p[0]) + float(p[1]) * x / k) / (1 + x / k)


def fz_pk_singlesite(
    k: float, p: NDArray[np.float_] | Sequence[float], x: NDArray[np.float_]
) -> NDArray[np.float_]:
    """Fit function for pH titration."""
    return (float(p[1]) + float(p[0]) * 10 ** (k - x)) / (1 + 10 ** (k - x))


def fit_titration(  # noqa: PLR0913, PLR0915
    kind: str,
    x: Sequence[float],
    y: NDArray[np.float_],
    y2: NDArray[np.float_] | None = None,
    residue: NDArray[np.float_] | None = None,
    residue2: NDArray[np.float_] | None = None,
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
    y : NDArray[np.float]
        Dataset y-values.
    y2 : NDArray[np.float], optional
        Optional second dataset y-values (share x with main dataset).
    residue : NDArray[np.float], optional
        Residues for main dataset.
    residue2 : NDArray[np.float], optional
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

    def compute_p0(x: Sequence[float], y: NDArray[np.float_]) -> NDArray[np.float_]:
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

        def ssq1(
            p: NDArray[np.float_], x: NDArray[np.float_], y1: NDArray[np.float_]
        ) -> NDArray[np.float_]:
            return np.array(np.r_[y1 - fz(p[0], p[1:3], x)])

        p0 = compute_p0(x, y)
        p, cov, info, msg, success = scipy.optimize.leastsq(
            ssq1, p0, args=(np.array(x), y), full_output=True, xtol=1e-11
        )
    else:

        def ssq2(  # noqa: PLR0913
            p: NDArray[np.float_],
            x: NDArray[np.float_],
            y1: NDArray[np.float_],
            y2: NDArray[np.float_],
            rd1: NDArray[np.float_],
            rd2: NDArray[np.float_],
        ) -> NDArray[np.float_]:
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


# Define the namedtuple outside the function
FitResult = namedtuple(
    "FitResult", ["success", "msg", "df", "chisqr", "K", "sK", "SA", "sSA", "SB", "sSB"]
)


def fit_titration_spectra(
    binding_model_func: typing.Callable[
        [float, list[float], NDArray[np.float_]], NDArray[np.float_]
    ],
    x: NDArray[np.float_],
    y: NDArray[np.float_],
    initial_parameters: list[float] | None = None,
) -> FitResult:
    """Fit a dataset (x, y) using a single-site binding model.

    The model is provided by the binding_model_func that defines a constant K and two plateaus SA and SB.

    Parameters
    ----------
    binding_model_func : typing.Callable[[float, list[float], NDArray], NDArray]
        The binding model function.
    x : NDArray[np.float]
        The x values of the dataset.
    y : NDArray[np.float]
        The y values of the dataset.
    initial_parameters : list[float], optional
        The initial parameters for the model, by default [7.1, None, None]

    Raises
    ------
    ValueError
        If the optimization fails.

    Returns
    -------
    FitResult
        A named tuple containing the least square results.
    """
    if initial_parameters is None:
        initial_parameters = [7.1, np.NaN, np.NaN]

    def ssq(
        p: list[float], x: NDArray[np.float_], y1: NDArray[np.float_]
    ) -> NDArray[np.float_]:
        return np.asarray(y1 - binding_model_func(p[0], p[1:3], x), dtype=np.float_)

    # Plateau calculation
    df = pd.DataFrame({"x": x, "y": y})  # noqa: PD901
    if np.isnan(initial_parameters[1]):
        initial_parameters[1] = df.y[df.x == min(df.x)].values[0]  # noqa: PD011
    if np.isnan(initial_parameters[2]):
        initial_parameters[2] = df.y[df.x == max(df.x)].values[0]  # noqa: PD011

    p, cov, info, msg, success = optimize.leastsq(
        ssq, initial_parameters, args=(x, y), full_output=True, xtol=1e-11
    )

    if not 1 <= success <= 4:  # noqa: PLR2004
        message = f"Optimization failed with message: {msg}"
        raise ValueError(message)

    degree_freedom = len(y) - len(p)
    chisqr = sum(info["fvec"] * info["fvec"]) / degree_freedom
    fit_result = FitResult(
        msg=msg,
        success=success,
        df=degree_freedom,
        chisqr=chisqr,
        K=p[0],
        sK=np.sqrt(cov[0][0]) * chisqr,
        SA=p[1],
        sSA=np.sqrt(cov[1][1]) * chisqr,
        SB=p[2],
        sSB=np.sqrt(cov[2][2]) * chisqr,
    )
    return fit_result


def apply_common_plot_style(
    ax: mpl.axes.Axes, title: str, xlabel: str, ylabel: str
) -> None:
    """Apply common plot style to a given Axes object.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Axes object to which the style should be applied.
    title : str
        The title to set for the plot.
    xlabel : str
        The label for the x-axis of the plot.
    ylabel : str
        The label for the y-axis of the plot.
    """
    ax.grid(True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def plot_spectra(
    f: pd.DataFrame,
    ax: mpl.axes.Axes,
    hue_norm: tuple[float, float],
    palette: str,
    kind: str,
) -> None:
    """
    Plot the spectral data.

    Parameters
    ----------
    f : pd.DataFrame
        The DataFrame containing spectral data.
    ax : mpl.axes.Axes
        The Axes object to which the plot should be added.
    hue_norm : Tuple[float, float]
        The normalization for the hue mapping, as a tuple of the form (min, max).
    palette : str
        The name of the palette to use for the hue mapping.
    kind : str
        The variable in `data` to map plot aspects to.

    """
    dat = f.copy()
    dat["lambda"] = dat.index
    sb.lineplot(
        data=dat.melt(id_vars="lambda", var_name=kind),
        x="lambda",
        y="value",
        hue=kind,
        hue_norm=hue_norm,
        palette=palette,
        ax=ax,
    )
    apply_common_plot_style(ax, "Spectra", "Wavelength", "Fluorescence")


def plot_autovectors(f: pd.DataFrame, u: NDArray[np.float_], ax: mpl.axes.Axes) -> None:
    """Plot autovectors.

    Parameters
    ----------
    f : pd.DataFrame
        The DataFrame containing spectral data.
    u : np.ndarray
        The left singular vectors obtained from SVD.
    ax : mpl.axes.Axes
        The Axes object to which the plot should be added.

    """
    apply_common_plot_style(ax, "Autovectors", "Wavelength", "Magnitude")
    for i in range(4):
        ax.plot(
            f.index,
            u[:, i],
            color=COLOR_MAP(i),
            lw=3 / (i + 1),
            alpha=max(0.3, 1 - 0.2 * i),
        )


def plot_fit(  # noqa: PLR0913
    conc: NDArray[np.float_],
    y: NDArray[np.float_],
    ax: mpl.axes.Axes,
    result: FitResult,
    fz: typing.Callable[[float, list[float], NDArray[np.float_]], NDArray[np.float_]],
    hue_norm: tuple[float, float],
    palette: str,
    kind: str,
) -> None:
    """Plot fit of first principal component against concentration variable.

    Parameters
    ----------
    conc : NDArray[np.float_]
        The concentrations used for the titration.
    y : NDArray[np.float_]
        The first principal component.
    ax : mpl.axes.Axes
        The Axes object to which the plot should be added.
    result : Any
        The fit results.
    fz : Callable[[float, List[float], np.ndarray], np.ndarray]
        The function used to fit the data.
    hue_norm : Tuple[float, float]
        The normalization for the hue mapping.
    palette : str
        The colormap to be used.
    kind : str
        The type of the titration.

    """
    # Create scatter plot
    ax.scatter(
        conc,
        y,
        c=conc,
        cmap=palette,
        vmin=hue_norm[0],
        vmax=hue_norm[1],
        s=200,
        edgecolors="k",
    )
    apply_common_plot_style(ax, "LM fit", "", "")
    ax.set_ylabel("First Principal Component", color=COLOR_MAP(0), fontsize=14)
    ax.set_xlabel(kind, fontsize=14)
    # Adjust x-axis limits
    xmin = conc.min()
    xmax = conc.max()
    xmax += (xmax - xmin) / 7
    xlin = np.linspace(xmin, xmax, 100)
    # Plot fit line
    ax.plot(xlin, fz(result.K, [result.SA, result.SB], xlin), "k--")
    # Add fit result to plot
    title = str(round(result.K, 2)) + " \u00B1 " + str(round(result.sK, 2))
    sb.mpl.pyplot.figtext(0.26, 0.54, title, size=20)


def plot_pca(
    v: NDArray[np.float_],
    ax: mpl.axes.Axes,
    conc: NDArray[np.float_],
    hue_norm: tuple[float, float],
    palette: str,
) -> None:
    """Plot the first two principal components.

    Parameters
    ----------
    v : NDArray[np.float_]
        The matrix containing the principal components.
    ax : mpl.axes.Axes
        The Axes object to which the plot should be added.
    conc : NDArray[np.float_]
        The concentrations used for the titration.
    hue_norm : Tuple[float, float]
        The normalization for the hue mapping.
    palette : str
        The colormap to be used.

    """
    ax.scatter(
        v[1],
        v[0],
        c=conc,
        cmap=palette,
        vmin=hue_norm[0],
        vmax=hue_norm[1],
        s=200,
        edgecolors="k",
    )
    apply_common_plot_style(ax, "PCA plot", "", "")
    ax.set_ylabel("First Principal Component", color=COLOR_MAP(0))
    ax.set_xlabel("Second Principal Component", color=COLOR_MAP(1))
    # Add labels.
    for x, y, w in zip(v[1], v[0], conc):
        ax.text(x, y, w)


def plot_autovalues(s: NDArray[np.float_], ax: mpl.axes.Axes) -> None:
    """Plot the singular values from SVD.

    Parameters
    ----------
    S : numpy.ndarray
        The singular values from the SVD.
    ax : mpl.axes.Axes
        The axes on which to plot the singular values.

    Returns
    -------
    None
    """
    data = pd.DataFrame({"index": range(1, len(s) + 1), "singular_values": s})
    sb.scatterplot(
        x="index",
        y="singular_values",
        data=data,
        ax=ax,
        hue="index",
        s=160,
        legend=False,
        palette="Set2",
    )
    apply_common_plot_style(ax, "Singular Values from SVD", "Index", "Singular Value")
    ax.set(yscale="log")
    ax.set_xticks(np.arange(1, len(s) + 1))


def f_svd(f: pd.DataFrame, kind: str) -> plt.Figure:
    """Perform singular value decomposition (SVD) and visualize results.

    Creates a figure with five subplots to visualize the results.

    Parameters
    ----------
    f : pd.DataFrame
        The input data for SVD.
    kind : str
        Specifies the type of data, which determines the colormap, hue normalization, and callable function used in the plots.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure containing the subplots.
    """
    if kind == "Cl":
        hue_norm = (0.0, 500.0)
        palette = sb.cm.icefire
        fz = fz_kd_singlesite
    else:
        hue_norm = (5.7, 8.7)
        palette = sb.cm.vlag_r
        fz = fz_pk_singlesite
    u, s, v = np.linalg.svd(f)
    conc = np.array(f.columns.to_list())
    y = v[0, :]
    result = fit_titration_spectra(fz, conc, y)
    sb.set_style("ticks")
    fig = sb.mpl.pyplot.figure(figsize=(12, 8))
    ax1 = fig.add_axes([0.05, 0.65, 0.32, 0.31])
    plot_spectra(f, ax1, hue_norm, palette, kind)

    ax2 = fig.add_axes([0.42, 0.65, 0.32, 0.31])
    plot_autovectors(f, u, ax2)
    ax3 = fig.add_axes([0.80, 0.65, 0.18, 0.31])
    plot_autovalues(s, ax3)

    ax4 = fig.add_axes([0.05, 0.08, 0.50, 0.50])
    plot_fit(conc, y, ax4, result, fz, hue_norm, palette, kind)
    ax5 = fig.add_axes([0.63, 0.08, 0.35, 0.50])
    plot_pca(v, ax5, conc, hue_norm, palette)
    return fig
