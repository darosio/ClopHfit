"""Fit Cl binding and pH titration."""
from __future__ import annotations

import typing
from typing import Sequence

import lmfit  # type: ignore
import matplotlib as mpl  # type: ignore
import matplotlib.colors as mcolors  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd
import scipy  # type: ignore
import scipy.stats  # type: ignore
import seaborn as sb  # type: ignore # noqa: ICN001
from numpy.typing import NDArray
from uncertainties import ufloat  # type: ignore

COLOR_MAP = plt.cm.Set1


@typing.overload
def _binding_pk(x: float, K: float, S0: float, S1: float) -> float:  # noqa: N803
    ...


@typing.overload
def _binding_pk(
    x: NDArray[np.float_], K: float, S0: float, S1: float  # noqa: N803
) -> NDArray[np.float_]:
    ...


def _binding_pk(
    x: float | NDArray[np.float_], K: float, S0: float, S1: float  # noqa: N803
) -> float | NDArray[np.float_]:
    return S1 + (S0 - S1) * 10 ** (K - x) / (1 + 10 ** (K - x))


@typing.overload
def _binding_kd(x: float, K: float, S0: float, S1: float) -> float:  # noqa: N803
    ...


@typing.overload
def _binding_kd(
    x: NDArray[np.float_], K: float, S0: float, S1: float  # noqa: N803
) -> NDArray[np.float_]:
    ...


def _binding_kd(
    x: float | NDArray[np.float_], K: float, S0: float, S1: float  # noqa: N803
) -> float | NDArray[np.float_]:
    return S1 + (S0 - S1) * x / K / (1 + x / K)


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
# TODO: use this like fz in prtecan
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


# `fit_titration` is exported and use in prtecan.
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


def fit_binding(
    x: NDArray[np.float_],
    y: NDArray[np.float_],
    model_func: typing.Callable[[float, float, float, float], float],
) -> lmfit.model.ModelResult:
    """Fit a dataset (x, y) using a single-site binding model with lmfit.

    The model is provided by the `model_func` argument, a function defining the
    model for fitting, which takes four arguments - x, K, S0 and S1, and returns
    a float.

    Parameters
    ----------
    x : NDArray[np.float]
        The x values of the dataset.
    y : NDArray[np.float]
        The y values of the dataset.
    model_func : Callable
        The function defining the model for fitting.

    Returns
    -------
    result : Minimizer
        lmfit's MinimizerResult object containing the fitting results.

    Raises
    ------
    ValueError
        If the optimization fails.
    """
    model = lmfit.Model(model_func)
    params = lmfit.Parameters()
    # Data-driven initialization
    xydata = pd.DataFrame({"x": x, "y": y})
    s1_initial = xydata.y[xydata.x == min(xydata.x)].to_numpy()[0]
    s0_initial = xydata.y[xydata.x == max(xydata.x)].to_numpy()[0]
    # Find the x value where y is closest to (S1_initial + S0_initial) / 2
    target_y = (s1_initial + s0_initial) / 2
    k_initial = xydata.loc[(xydata.y - target_y).abs().idxmin(), "x"]
    # Parameters initialization
    params.add("K", value=k_initial, min=2)
    params.add("S1", value=s1_initial)
    params.add("S0", value=s0_initial)
    result = model.fit(y, params, x=x)
    if result.success is not True:
        message = f"Optimization failed with message: {result.message}"
        raise ValueError(message)
    return result


def plot_lmfit(
    ax: mpl.axes.Axes,
    result: lmfit.model.ModelResult,
    hue_norm: tuple[float, float],
    palette: str,
) -> None:
    """Plot the results of the lmfit model on a given axes.

    This function creates a scatter plot of the data used for fitting, then
    overlays the best-fit curve and its uncertainty band. The data points are
    colored according to their x-value, and a text element is added to the plot
    to display the fitted parameter K with its uncertainty.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes on which to plot the lmfit results.
    result : lmfit.model.ModelResult
        The lmfit ModelResult object containing the fit results.
    hue_norm : tuple[float, float]
        A tuple of two floats specifying the normalization range for the colormap.
    palette : str
        The name of the colormap to use for coloring the data points.
    """
    x = result.userkws["x"]
    y = result.data
    n0 = hue_norm[0]
    n1 = hue_norm[1]
    ax.scatter(x, y, c=x, cmap=palette, vmin=n0, vmax=n1, s=99, edgecolors="k")
    # Adjust x-axis limits
    xmin = x.min()
    xmax = x.max()
    xmax += (xmax - xmin) / 7
    xfit = np.linspace(xmin, xmax, 100)
    # Plot fit line
    best_fit = result.eval(x=xfit)
    ax.plot(xfit, best_fit, "k--")
    dely = result.eval_uncertainty(x=xfit)
    ax.fill_between(xfit, best_fit - dely, best_fit + dely, color="#FEDCBA", alpha=0.5)
    # Add fit result to plot
    k = ufloat(result.params["K"].value, result.params["K"].stderr)
    title = "=".join(["K", str(k).replace("+/-", "±")])
    sb.mpl.pyplot.figtext(0.26, 0.54, title, size=20)


def plot_spectra(
    f: pd.DataFrame,
    ax: mpl.axes.Axes,
    hue_norm: tuple[float, float],
    palette: str,
    kind: str,
) -> None:
    """Plot spectra.

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
    color_map = plt.get_cmap(palette)
    normalize = mcolors.Normalize(vmin=hue_norm[0], vmax=hue_norm[1])
    for i in range(len(f.columns)):
        ax.plot(f.index, f.iloc[:, i], color=color_map(normalize(f.columns[i])))
    _apply_common_plot_style(ax, "Spectra", "Wavelength", "Fluorescence")
    # Add a colorbar for reference
    sm = plt.cm.ScalarMappable(cmap=color_map, norm=normalize)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label=kind)


def _apply_common_plot_style(
    ax: mpl.axes.Axes,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    """Apply grid style and add title and labels."""
    ax.grid(True)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def _plot_autovectors(wl: pd.Index, u: NDArray[np.float_], ax: mpl.axes.Axes) -> None:
    """Plot autovectors.

    Parameters
    ----------
    wl : pd.Index
        The index of spectra data frame.
    u : np.ndarray
        The left singular vectors obtained from SVD.
    ax : mpl.axes.Axes
        The Axes object to which the plot should be added.

    """
    number_autovectors = 4
    for i in range(number_autovectors):
        ax.plot(wl, u[:, i], color=COLOR_MAP(i), lw=3 / (i + 1), alpha=(1 - 0.2 * i))
    _apply_common_plot_style(ax, "Autovectors", "Wavelength", "Magnitude")


def plot_autovalues(s: NDArray[np.float_], ax: mpl.axes.Axes) -> None:
    """Plot the singular values from SVD.

    Parameters
    ----------
    s : numpy.ndarray
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
    n0 = hue_norm[0]
    n1 = hue_norm[1]
    ax.scatter(v[1], v[0], c=conc, cmap=palette, vmin=n0, vmax=n1, s=99, edgecolors="k")
    _apply_common_plot_style(ax, "PCA plot", "", "")
    ax.set_ylabel("First Principal Component", color=COLOR_MAP(0))
    ax.set_xlabel("Second Principal Component", color=COLOR_MAP(1))
    # Add labels.
    for x, y, w in zip(v[1], v[0], conc):
        ax.text(x, y, w)


def analyze_spectra_glob(
    titration: dict[str, pd.DataFrame],
    kind: str,
    dbands: dict[str, tuple[int, int]] | None = None,
    x_combined: dict[str, NDArray[np.float_]] | None = None,
    y_combined: dict[str, NDArray[np.float_]] | None = None,
) -> (
    tuple[
        plt.Figure | None,
        lmfit.model.ModelResult | None,
        plt.Figure,
        lmfit.model.MinimizerResult,
    ]
    | tuple[
        plt.Figure,
        lmfit.model.ModelResult,
        plt.Figure | None,
        lmfit.model.MinimizerResult | None,
    ]
):
    """Analyze multi-label spectra visualize the results."""
    _gap_ = 1
    dbands = dbands or {}
    x_combined = x_combined or {}
    y_combined = y_combined or {}
    labels_svd = titration.keys() - dbands.keys()
    if len(labels_svd) > 1:
        # Concatenate spectra.
        prev_max = 0
        adjusted_list = []
        for label in labels_svd:
            spectra_adjusted = titration[label].copy()  # Avoid modifying original data
            spectra_adjusted.index += prev_max - spectra_adjusted.index.min() + _gap_
            prev_max = spectra_adjusted.index.max()
            adjusted_list.append(spectra_adjusted)
        spectra_merged = pd.concat(adjusted_list)
        # Analyze concatenated spectra.
        figure_svd, result_svd = analyze_spectra(spectra_merged, kind)
    else:
        figure_svd, result_svd = (None, None)
    if len(dbands.keys()) > 1:
        params = lmfit.Parameters()
        params.add("K", value=7, min=0)
        i = 0
        xc = []
        yc = []
        for label in dbands:
            params.add(f"S0_{i+1}", value=y_combined[label][0])
            params.add(f"S1_{i+1}", value=y_combined[label][-1])
            i += 1
            xc.append(x_combined[label])
            yc.append(y_combined[label])
        ndata = len(xc)
        fz = _binding_kd if kind == "Cl" else _binding_pk
        result_bands = lmfit.minimize(_binding_residuals, params, args=(fz, xc, yc))
        figure_bands, ax = plt.subplots()
        xfit = [np.linspace(x.min(), x.max(), 100) for x in xc]
        yfit = _binding_residuals(result_bands.params, fz, xfit)
        for i in range(ndata):
            ax.plot(xc[i], yc[i], "o", xfit[i], yfit[i], "-")
        ax.grid(True)
    else:
        figure_bands, result_bands = (None, None)
    return figure_svd, result_svd, figure_bands, result_bands


def analyze_spectra(
    spectra: pd.DataFrame, kind: str, band: tuple[int, int] | None = None
) -> tuple[plt.Figure, lmfit.model.ModelResult]:
    """Analyze spectral data and visualize the results.

    Depending on the method chosen, this function either performs SVD on the input
    data, generating principal component spectra and plotting the resulting data,
    or integrates the spectral data over a specified band and fits the integrated
    data to a binding model.

    The 'kind' parameter determines the binding model used for fitting as well as
    the colormap and hue normalization for the plots.

    Parameters
    ----------
    spectra : pd.DataFrame
        A DataFrame where each column represents a spectrum at a different condition
        (for instance, a different pH or ligand concentration) and each row
        corresponds to a different wavelength.

    kind : str
        Specifies the type of data either 'pH' or 'Cl' titrations.

    band : tuple[int, int], optional
        The band to integrate over for the band method. If None (default), performs SVD.
        The tuple values correspond to the start and end wavelengths for the band.


    Returns
    -------
    fig : matplotlib.figure.Figure
        A Figure object with the plots visualizing the analysis results.

    result : lmfit.model.ModelResult
        The result of the lmfit model fitting.

    Raises
    ------
    ValueError
        If the band parameters are not in the spectra's index when the band method
        is used.

    Notes
    -----
    For the SVD method, the function plots the original spectra, the principal
    component vectors, the singular values, the first principal component, and
    the fitting results. For the band method, it plots the original spectra and
    the fitting results.
    """
    if kind == "Cl":
        hue_norm = (0.0, 200.0)
        palette = sb.cm.crest
        fz = _binding_kd
    else:
        hue_norm = (5.7, 8.7)
        palette = sb.cm.vlag_r
        fz = _binding_pk
    x = spectra.columns.to_numpy()
    fig = sb.mpl.pyplot.figure(figsize=(12, 8))
    ax1 = fig.add_axes([0.05, 0.65, 0.32, 0.31])
    plot_spectra(spectra, ax1, hue_norm, palette, kind)
    if band is None:  # SVD
        ddf = spectra.sub(spectra.iloc[:, 0], axis=0)
        u, s, v = np.linalg.svd(ddf)
        y = v[0, :] + 1
        result = fit_binding(x, y, fz)
        ax2 = fig.add_axes([0.42, 0.65, 0.32, 0.31])
        _plot_autovectors(spectra.index, u, ax2)
        ax3 = fig.add_axes([0.80, 0.65, 0.18, 0.31])
        plot_autovalues(s[:], ax3)  # do not plat last 2 autovalues
        ax5 = fig.add_axes([0.63, 0.08, 0.35, 0.50])
        plot_pca(v, ax5, x, hue_norm, palette)
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
        result = fit_binding(x, y, fz)
        ylabel = "Integrated Band Fluorescence"
        ylabel_color = "k"
    ax4 = fig.add_axes([0.05, 0.08, 0.50, 0.50])
    plot_lmfit(ax4, result, hue_norm, palette)
    _apply_common_plot_style(ax4, "LM fit", kind, "")
    ax4.set_ylabel(ylabel, color=ylabel_color)
    return fig, result


def _binding_residuals(
    params: lmfit.Parameter,
    model_func: typing.Callable[
        [NDArray[np.float_], float, float, float], NDArray[np.float_]
    ],
    x: list[NDArray[np.float_]],
    datasets: list[NDArray[np.float_]] | None = None,
) -> list[NDArray[np.float_]]:
    """Compute residuals for multiple datasets given a shared binding model function.

    Parameters
    ----------
    params : lmfit.Parameter
        Parameters for the model function.
    model_func : callable
        The model function used for fitting the data.
    x : list of np.ndarray
        Independent variables for each dataset.
    datasets : list of np.ndarray, optional
        The datasets to fit. If None, returns only the model predictions.

    Returns
    -------
    list of np.ndarray
        The residuals for each dataset or the model predictions if datasets is None.
    """
    # Number of datasets
    ndata = len(x)
    models = [
        model_func(x[i], params["K"], params[f"S0_{i+1}"], params[f"S1_{i+1}"])
        for i in range(ndata)
    ]
    # If no datasets are provided, return the model predictions
    if datasets is None:
        return models
    residuals = [datasets[i] - models[i] for i in range(ndata)]
    return residuals
