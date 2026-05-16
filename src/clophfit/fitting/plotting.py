"""Provide utilities for creating various types of plots used in this project.

Primary functions encompassed are:

    plot_spectra: Develops a plot for spectral data. Each line is colored based
    on a designated colormap.
    plot_autovectors: Plots the autovectors.
    plot_autovalues: Plots the singular values from SVD.
    plot_fit: Plots residuals for each dataset with uncertainty.
    plot_pca: Plots the first two principal components.
    plot_spectra_distributed: Plots spectra from titration distributing on the
    figure top.
    plot_emcee: Plots emcee result.
    plot_emcee_k_on_ax: Plots emcee result for a specific parameter on an axis.
    distribute_axes: Positions axes evenly along the horizontal axis of the figure.

The module uses several dependencies such as ArviZ, numpy, pandas, seaborn,
lmfit, matplotlib, and uncertainties. Moreover, it includes a range of internal
project modules and a specific color map for PCA components and LM fit.

Helper Functions:

    _apply_common_plot_style: Applies grid style, title, and labels to a plot.
    _create_spectra_canvas: Creates figure and axes for spectra plot.

Classes:

    PlotParameters: Parameters for plotting, depending on whether the data is pH or Cl.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from typing import TYPE_CHECKING

import arviz as az  # type: ignore[import-untyped]
import corner  # type: ignore[import-untyped]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  # type: ignore[import-untyped]
import xarray as xr
from matplotlib import cm, colormaps, colors
from matplotlib.figure import Figure
from uncertainties import ufloat  # type: ignore[import-untyped]

from clophfit.fitting.data_structures import Dataset
from clophfit.fitting.models import binding_1site

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lmfit import Parameters  # type: ignore[import-untyped]
    from lmfit.minimizer import MinimizerResult  # type: ignore[import-untyped]
    from matplotlib import axes
    from matplotlib.axes import Axes

    from clophfit.clophfit_types import ArrayF
    from clophfit.fitting.data_structures import FitResult, MiniT

    from .data_structures import Dataset

COLOR_MAP = colormaps["Set1"]  # To color PCA components and LM fit.
N_AUTOVALS = 4


__all__ = [
    "COLOR_MAP",
    "PlotParameters",
    "distribute_axes",
    "extract_sigma_df",
    "plot_autovalues",
    "plot_autovectors",
    "plot_emcee",
    "plot_emcee_k_on_ax",
    "plot_fit",
    "plot_fit_gemini",
    "plot_pca",
    "plot_qc_mean_vs_std",
    "plot_qc_span_vs_center",
    "plot_qc_span_vs_center_titration",
    "plot_spectra",
    "plot_spectra_distributed",
    "print_emcee",
]


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
            self.palette = "coolwarm_r"  # sb vlag_r
            self.kind = "pH"
        else:
            self.hue_norm = (0.0, 200.0)
            self.palette = "viridis_r"  # sb crest
            self.kind = "Cl"


def _apply_common_plot_style(ax: Axes, title: str, xlabel: str, ylabel: str) -> None:
    """Add grid, title and x_y_labels."""
    ax.grid(visible=True)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def _create_spectra_canvas() -> tuple[Figure, tuple[Axes, Axes, Axes, Axes, Axes]]:
    fig = Figure(figsize=(12, 8))
    ax1 = fig.add_axes((0.05, 0.65, 0.32, 0.31))
    ax2 = fig.add_axes((0.42, 0.65, 0.32, 0.31))
    ax3 = fig.add_axes((0.80, 0.65, 0.18, 0.31))
    ax4 = fig.add_axes((0.05, 0.08, 0.50, 0.50))
    ax5 = fig.add_axes((0.63, 0.08, 0.35, 0.50))
    return fig, (ax1, ax2, ax3, ax4, ax5)


def _make_empty_qc_figure(message: str) -> Figure:
    """Create a one-axis QC figure containing a centered message."""
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, message, ha="center")
    return fig


def _create_qc_subplots(
    labels: Sequence[str | int], figsize_per_label: tuple[float, float]
) -> tuple[Figure, np.ndarray]:
    """Create one-row QC subplots with one panel per label."""
    fig, axes = plt.subplots(
        1,
        len(labels),
        figsize=(figsize_per_label[0] * len(labels), figsize_per_label[1]),
        squeeze=False,
    )
    return fig, np.asarray(axes[0])


def _resolve_qc_bg_value(
    bg_noise: Mapping[str, float | ArrayF] | Mapping[int, float | ArrayF] | None,
    label: str | int,
) -> float | None:
    """Resolve a scalar background-noise reference for a QC label."""
    if bg_noise is None:
        return None

    raw: float | ArrayF | None = None
    for key in (label, str(label)):
        if key in bg_noise:
            raw = bg_noise[key]  # type: ignore[index]
            break

    if raw is None and isinstance(label, str):
        try:
            numeric_label = int(label)
        except ValueError:
            numeric_label = None
        if numeric_label is not None and numeric_label in bg_noise:
            raw = bg_noise[numeric_label]  # type: ignore[index]

    if raw is None:
        return None

    raw_array = np.asarray(raw)
    if raw_array.ndim == 0:
        return float(raw_array)
    return float(np.mean(raw_array))


def _annotate_qc_point(ax: Axes, x: float, y: float, well: str, color: str) -> None:
    """Annotate a QC point with consistent styling."""
    ax.annotate(
        well,
        (x, y),
        color=color,
        fontweight="bold",
        xytext=(4, 4),
        textcoords="offset points",
    )


def _render_qc_panel(  # noqa: PLR0913
    ax: Axes,
    panel_df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    z_threshold: float,
    bg_value: float | None = None,
    bg_multiplier: float = 4.0,
    annotate_wells: list[str] | None = None,
    loglog: bool = False,
) -> None:
    """Render one QC scatter panel with shared highlighting logic."""
    from clophfit.fitting.utils import (  # noqa: PLC0415
        fit_trendline,
        flag_trend_outliers,
    )

    x_val = panel_df[x_col]
    y_val = panel_df[y_col]
    outliers = flag_trend_outliers(x_val, y_val, threshold=z_threshold)

    bg_outliers = pd.Series(data=False, index=panel_df.index)
    if bg_value is not None:
        threshold_bg = bg_multiplier * bg_value
        bg_outliers = (x_val < threshold_bg) | (
            (y_val < threshold_bg) & (y_val.max() > 1e-6)  # noqa: PLR2004
        )

    m, c = fit_trendline(x_val, y_val)
    x_line = np.linspace(float(x_val.min()), float(x_val.max()), 100)
    ax.plot(x_line, m * x_line + c, "k--", alpha=0.5, label="Trendline")

    if bg_value is not None:
        ax.axvline(
            bg_multiplier * bg_value,
            color="cyan",
            linestyle=":",
            alpha=0.7,
        )

    good = panel_df[~(outliers | bg_outliers)]
    ax.scatter(
        good[x_col],
        good[y_col],
        alpha=0.7,
        color="indigo",
        edgecolors="none",
        label="Wells",
    )

    bad_bg = panel_df[bg_outliers]
    if not bad_bg.empty:
        ax.scatter(
            bad_bg[x_col],
            bad_bg[y_col],
            alpha=1.0,
            color="cyan",
            s=60,
            marker="s",
            label=f"BG < {bg_multiplier}x",
        )
        for _, row in bad_bg.iterrows():
            _annotate_qc_point(
                ax,
                float(row[x_col]),
                float(row[y_col]),
                str(row["well"]),
                "c",
            )

    bad = panel_df[outliers & ~bg_outliers]
    if not bad.empty:
        ax.scatter(
            bad[x_col],
            bad[y_col],
            alpha=1.0,
            color="orange",
            s=50,
            marker="X",
            label="Outliers",
        )
        for _, row in bad.iterrows():
            _annotate_qc_point(
                ax,
                float(row[x_col]),
                float(row[y_col]),
                str(row["well"]),
                "orange",
            )

    if annotate_wells:
        flagged_wells = set(panel_df.loc[outliers | bg_outliers, "well"])
        requested_wells = set(annotate_wells)
        for _, row in panel_df.iterrows():
            well = str(row["well"])
            if well in requested_wells and well not in flagged_wells:
                ax.scatter(
                    float(row[x_col]),
                    float(row[y_col]),
                    alpha=1.0,
                    color="red",
                    s=50,
                    marker="*",
                )
                _annotate_qc_point(
                    ax,
                    float(row[x_col]),
                    float(row[y_col]),
                    well,
                    "red",
                )

    if loglog:
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()


def distribute_axes(fig: Figure, num_axes: int) -> list[Axes]:
    """Position axes evenly along the horizontal axis of the figure.

    Parameters
    ----------
    fig : Figure
        The Figure object on which the Axes objects are drawn.
    num_axes : int
        The number of Axes objects to position.

    Returns
    -------
    list[Axes]
        A list of positioned Axes objects.
    """
    axs: list[Axes] = []
    for i in range(num_axes):
        if i == 0:
            ax = fig.add_subplot(1, num_axes, i + 1)
        else:
            ax = fig.add_subplot(1, num_axes, i + 1, sharey=axs[0])
            ax.tick_params(labelleft=False)
        left = 0.05 + i / num_axes
        bottom = 0.65
        width = 1 / num_axes - 0.0  # Subtract some amount to prevent overlap
        height = 0.31
        ax.set_position((left, bottom, width, height))
        axs.append(ax)
    return axs


def plot_autovalues(ax: Axes, s: ArrayF) -> None:
    """Plot the singular values from SVD.

    Parameters
    ----------
    ax : Axes
        The mpl.axes.axes on which to plot the singular values.
    s : ArrayF
        The singular values from the SVD.
    """
    index = range(1, N_AUTOVALS + 1)
    colors_autovalues = [COLOR_MAP(i - 1) for i in index]  # using direct indexing
    ax.scatter(index, s[:N_AUTOVALS], c=colors_autovalues, s=99)
    ax.axhline(y=s[-2], color="gray", linestyle="--")  # horizontal line at y=s[-2]
    _apply_common_plot_style(ax, "Singular Values from SVD", "Index", "Singular Value")
    ax.set(yscale="log")
    ax.set_xticks(np.arange(1, N_AUTOVALS + 1))


def plot_autovectors(ax: Axes, wl: pd.Index[int], u: ArrayF) -> None:
    """Plot autovectors.

    Parameters
    ----------
    ax : Axes
        The mpl.axes.Axes object to which the plot should be added.
    wl : pd.Index[int]
        The index of spectra data frame.
    u : ArrayF
        The left singular vectors obtained from SVD.
    """
    for i in range(N_AUTOVALS):
        ax.plot(wl, u[:, i], color=COLOR_MAP(i), lw=3 / (i + 1), alpha=1 - 0.2 * i)
    _apply_common_plot_style(ax, "Autovectors", "Wavelength", "Magnitude")


def plot_pca(ax: Axes, v: ArrayF, conc: ArrayF, pp: PlotParameters) -> None:
    """Plot the first two principal components.

    Parameters
    ----------
    ax : Axes
        The mpl.axes.Axes object to which the plot should be added.
    v : ArrayF
        The matrix containing the principal components.
    conc : ArrayF
        The concentrations used for the titration.
    pp : PlotParameters
        The PlotParameters object containing plot parameters.
    """
    ax.scatter(
        v[1],
        v[0],
        c=list(conc),
        s=99,
        edgecolors="k",
        vmin=pp.hue_norm[0],
        vmax=pp.hue_norm[1],
        cmap=pp.palette,
    )
    _apply_common_plot_style(ax, "PCA plot", "", "")
    ax.set_ylabel("First Principal Component", color=COLOR_MAP(0))
    ax.set_xlabel("Second Principal Component", color=COLOR_MAP(1))
    # Add labels.
    for x, y, w in zip(v[1], v[0], conc, strict=False):
        ax.text(x, y, str(w))


def plot_spectra(ax: Axes, spectra: pd.DataFrame, pp: PlotParameters) -> None:
    """Plot spectra.

    Parameters
    ----------
    ax : Axes
        The Mpl.Axes.Axes object to which the plot should be added.
    spectra : pd.DataFrame
        The DataFrame containing spectral data.
    pp : PlotParameters
        The PlotParameters object containing plot parameters.
    """
    color_map = colormaps[pp.palette]
    normalize = colors.Normalize(vmin=pp.hue_norm[0], vmax=pp.hue_norm[1])
    for i, col in enumerate(spectra.columns):
        ax.plot(spectra.index, spectra.iloc[:, i], color=color_map(normalize(col)))
    _apply_common_plot_style(ax, "Spectra", "Wavelength", "Fluorescence")
    # Add a colorbar for reference
    sm = cm.ScalarMappable(cmap=color_map, norm=normalize)
    sm.set_array([])
    if ax.figure is not None:
        ax.figure.colorbar(sm, ax=ax, label=pp.kind)


def plot_spectra_distributed(
    fig: Figure,
    titration: dict[str, pd.DataFrame],
    pp: PlotParameters,
    dbands: dict[str, tuple[int, int]] | None = None,
) -> None:
    """Plot spectra from titration distributing on the top of the figure top."""
    color_map = colormaps[pp.palette]
    normalize = colors.Normalize(vmin=pp.hue_norm[0], vmax=pp.hue_norm[1])
    axl = distribute_axes(fig, len(titration))
    for (j, ax), (lbl, spec) in zip(enumerate(axl), titration.items(), strict=False):
        # Calculate the average spectrum
        avg_spec = spec.mean(axis=1)
        for i, col in enumerate(spec.columns):
            if dbands is None:
                # Difference between current spectrum and the average spectrum
                diff = spec.iloc[:, i] - avg_spec
                ax.plot(spec.index, diff, color=color_map(normalize(col)))
            elif lbl in dbands:
                sp = spec.iloc[:, i]
                ax.plot(spec.index, sp, color=color_map(normalize(col)))
                xmin, xmax = dbands[lbl]
                ax.axvspan(xmin, xmax, color=COLOR_MAP(j), alpha=0.02)
        if j == 0:
            # x_label="" avoid overlapping with the title of plot_fit
            _apply_common_plot_style(ax, lbl, "", "Fluorescence")
        else:
            _apply_common_plot_style(ax, lbl, "Wavelength", "")
    # Add a colorbar for reference
    sm = cm.ScalarMappable(cmap=color_map, norm=normalize)
    sm.set_array([])
    fig.colorbar(sm, ax=axl, label=pp.kind)


def plot_qc_mean_vs_std(  # noqa: PLR0913, PLR0917
    trace: xr.DataTree,
    results: Mapping[str, FitResult[MiniT]] | None = None,
    figsize_per_label: tuple[float, float] = (5, 4),
    annotate_wells: list[str] | None = None,
    z_threshold: float = 3.0,
    bg_noise: Mapping[str, float] | None = None,
    bg_multiplier: float = 4.0,
) -> Figure:
    """Plot standard deviation versus mean of inferred sigma for quality control.

    Identifies "dead" or flat-line wells (e.g. caused by pipetting errors or
    missing fluorophore) which typically exhibit both low mean and low
    span (max - min) for the inferred standard deviation (sigma_obs) across titration steps.

    Parameters
    ----------
    trace : xr.DataTree
        The PyMC inference trace containing `sigma_obs` deterministic nodes.
    results : Mapping[str, FitResult[MiniT]] | None, optional
        The dictionary of well results to derive fallback sigma values.
    figsize_per_label : tuple[float, float], optional
        Figure size allocated for each spectral band (label). Default is (5, 4).
    annotate_wells : list[str] | None, optional
        A list of specific well IDs to annotate on the plot (e.g., ["B03", "C05"]).
        If None, no explicit labels are drawn.
    z_threshold : float
        Z-score threshold for identifying outliers.
    bg_noise : Mapping[str, float] | None, optional
        Background noise dictionary.
    bg_multiplier : float
        Multiplier for background noise.

    Returns
    -------
    Figure
        The generated QC matplotlib figure.
    """
    sigma_df = _extract_sigma_df(trace, results)
    if len(sigma_df) == 0:
        return _make_empty_qc_figure("No 'sigma_obs' or fallback data found.")

    agg_df = (
        sigma_df
        .groupby(["well", "label"])["mean"]
        .agg(mean="mean", span=np.ptp)
        .reset_index()
    )

    labels = list(agg_df["label"].unique())
    fig, axes = _create_qc_subplots(labels, figsize_per_label)

    for ax, lbl in zip(axes, labels, strict=False):
        d = agg_df[agg_df["label"] == lbl].copy()
        _render_qc_panel(
            ax,
            d,
            x_col="mean",
            y_col="span",
            title=f"QC: Span vs Mean of inferred $\\sigma$ ({lbl})",
            xlabel=r"Mean($\sigma_{obs}$)",
            ylabel=r"Span($\sigma_{obs}$) [max - min]",
            z_threshold=z_threshold,
            bg_value=_resolve_qc_bg_value(bg_noise, lbl),
            bg_multiplier=bg_multiplier,
            annotate_wells=annotate_wells,
        )

    fig.tight_layout()
    return fig


def plot_emcee(flatchain: pd.DataFrame) -> Figure:
    """Plot emcee result."""
    # Pass DataFrame directly to corner.corner.
    # ArviZ 1.x DataTree is not yet supported by corner 2.2.3.
    corner_fig: Figure = corner.corner(flatchain, divergences=False)
    return corner_fig


def plot_emcee_k_on_ax(ax: Axes, res_emcee: MinimizerResult, p_name: str = "K") -> None:
    """Plot emcee result."""
    samples = res_emcee.flatchain
    # Convert the dictionary of flatchains to an ArviZ InferenceData object
    # Reshape to (1, -1) to satisfy ArviZ 1.x requirement for 2 sample dims (chain, draw)
    param_samples = np.array(samples[p_name]).flatten()
    hdi_lo, hdi_hi = az.hdi(param_samples, prob=0.94)
    median = float(np.median(param_samples))
    ax.hist(param_samples, bins=30, density=True, color="steelblue", alpha=0.7)
    ax.axvline(median, color="black", linestyle="--", label=f"median={median:.3g}")
    ax.axvspan(hdi_lo, hdi_hi, alpha=0.2, color="orange", label="94% HDI")
    ax.set_xlabel(p_name)
    ax.legend(fontsize=8)


def plot_fit(
    ax: Axes,
    ds: Dataset,
    params: Parameters,
    nboot: int = 0,
    pp: PlotParameters | None = None,
) -> None:
    """Plot fitted curves and data points with uncertainty on a given Axes.

    Parameters
    ----------
    ax : Axes
        The matplotlib axis to plot on.
    ds : Dataset
        The dataset containing the data points.
    params : Parameters
        The fitted parameters from lmfit.
    nboot : int
        Number of bootstrap samples to generate confidence bands.
    pp : PlotParameters | None
        Plotting parameters for consistent styling.

    """
    stretch = 0.05
    colors = [COLOR_MAP(i) for i in range(len(ds))]

    xfit = {
        k: np.linspace(da.x.min() * (1 - stretch), da.x.max() * (1 + stretch), 100)
        for k, da in ds.items()
    }
    # Compute y-fit using the model directly to avoid circular imports
    yfit = {
        lbl: binding_1site(
            xfit[lbl],
            params["K"].value,
            params[f"S0_{lbl}"].value,
            params[f"S1_{lbl}"].value,
            is_ph=ds.is_ph,
        )
        for lbl in ds
    }
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
                y_samples[i, :] = binding_1site(
                    xfit[lbl],
                    p_sample["K"].value,
                    p_sample[f"S0_{lbl}"].value,
                    p_sample[f"S1_{lbl}"].value,
                    is_ph=ds.is_ph,
                )
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


# --- Plotting ---
def plot_fit_gemini(
    ax: axes.Axes,
    ds: Dataset,
    params: Parameters,
    nboot: int = 0,
    pp: PlotParameters | None = None,
) -> None:
    """
    Plot fitted curves and data points on a given axis.

    Parameters
    ----------
    ax : axes.Axes
        The matplotlib axis to plot on.
    ds : Dataset
        The dataset containing the data points.
    params : Parameters
        The fitted parameters from lmfit.
    nboot : int
        Number of bootstrap samples to generate confidence bands.
    pp : PlotParameters | None
        Plotting parameters for consistent styling.
    """
    stretch = 0.05
    colors = [COLOR_MAP(i) for i in range(len(ds))]

    for (lbl, da), clr in zip(ds.items(), colors, strict=False):
        # Generate smooth x-values for the fitted curve
        x_fit = np.linspace(da.x.min() * (1 - stretch), da.x.max() * (1 + stretch), 100)
        y_fit = binding_1site(
            x_fit,
            params["K"].value,
            params[f"S0_{lbl}"].value,
            params[f"S1_{lbl}"].value,
            is_ph=ds.is_ph,
        )

        # Plot original data points
        if pp:
            ax.scatter(
                da.x,
                da.y,
                c=list(da.x),
                s=99,
                edgecolors=clr,
                label=lbl,
                vmin=pp.hue_norm[0],
                vmax=pp.hue_norm[1],
                cmap=pp.palette,
            )
        else:
            ax.plot(da.x, da.y, "o", color=clr, label=lbl)

        # Plot the fitted curve
        ax.plot(x_fit, y_fit, "-", color="gray")

        # Plot error bars if available
        if da.y_err.size > 0:
            x_err = da.x_err if da.x_err.size > 0 else None
            ax.errorbar(
                da.x,
                da.y,
                yerr=da.y_err,
                xerr=x_err,
                fmt=".",
                color=clr,
                alpha=0.4,
                capsize=3,
            )

        # Plot bootstrap confidence interval
        if nboot > 0 and params["K"].stderr is not None:
            y_samples = np.empty((nboot, len(x_fit)))
            rng = np.random.default_rng()
            for i in range(nboot):
                sampled_params = params.copy()
                for p in sampled_params.values():
                    if p.stderr:
                        p.value = rng.normal(p.value, p.stderr)
                y_samples[i, :] = binding_1site(
                    x_fit,
                    sampled_params["K"].value,
                    sampled_params[f"S0_{lbl}"].value,
                    sampled_params[f"S1_{lbl}"].value,
                    is_ph=ds.is_ph,
                )
            dy = y_samples.std(axis=0)
            ax.fill_between(x_fit, y_fit - dy, y_fit + dy, alpha=0.1, color=clr)

    # --- Final Touches ---
    ax.legend()
    k_val = params["K"].value
    k_err = params["K"].stderr
    title = f"K = {ufloat(k_val, k_err):.2u}" if k_err else f"K = {k_val:.3g}"
    xlabel = "pH" if ds.is_ph else "Cl"
    _apply_common_plot_style(ax, f"Fit: {title}", xlabel, "")


# TODO: Complete print emcee
def print_emcee(result_emcee: MinimizerResult) -> None:
    """Print maximum likelihood estimation (MLE) results from an emcee fitting."""
    highest_prob = np.argmax(result_emcee.lnprob)
    hp_loc = np.unravel_index(highest_prob, result_emcee.lnprob.shape)
    mle_soln = result_emcee.chain[hp_loc]
    # Store the parameter names
    param_names = list(result_emcee.params.keys())
    # Construct a dictionary mapping parameter names to MLE solutions
    mle_dict = dict(zip(param_names, mle_soln, strict=False))
    header = "\nMaximum Likelihood Estimation from emcee"
    line = "-------------------------------------------------"
    format_string = "{:<5s} {:>11s} {:>11s} {:>11s}"
    print(f"{header}\n{line}")
    print(format_string.format("Parameter", "MLE Value", "Median Value", "Uncertainty"))
    for name, param in result_emcee.params.items():
        mle_value = f"{mle_dict[name]:.5f}" if name in mle_dict else "N/A"
        median_value = f"{param.value:.5f}"
        uncertainty = "None" if param.stderr is None else f"{param.stderr:.5f}"
        print(format_string.format(name, mle_value, median_value, uncertainty))

    print("\nError estimates from emcee:")
    print("------------------------------------------------------")
    print("Parameter  -2sigma  -1sigma   median  +1sigma  +2sigma")
    format_string = "  {:5s}   {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f}"
    for name in result_emcee.params:
        if name in result_emcee.flatchain:
            quantiles = np.percentile(
                result_emcee.flatchain[name], [2.275, 15.865, 50, 84.135, 97.275]
            )
            print(format_string.format(name, *quantiles))
        else:
            print(f"Key '{name}' not found in .flatchain.")

    print("\nMaximum Likelihood Estimation (MLE):")
    print("----------------------------------")
    for ix, param in enumerate(result_emcee.params):
        print(f"{param}: {mle_soln[ix]:.3f}")
    quantiles = np.percentile(result_emcee.flatchain["K"], [2.28, 15.9, 50, 84.2, 97.7])
    print(f"\n\n1 sigma spread = {0.5 * (quantiles[3] - quantiles[1]):.3f}")
    print(f"2 sigma spread = {0.5 * (quantiles[4] - quantiles[0]):.3f}")


def _sample_dims_from_posterior_var(da: xr.DataArray) -> list[str]:
    """Infer sample dimensions for a posterior variable."""
    sample_dims = [d for d in ("chain", "draw") if d in da.dims]
    if sample_dims:
        return sample_dims
    return [str(d) for d in da.dims[: min(2, da.ndim)]]


def _interval_bounds_from_var(
    da: xr.DataArray, sample_dims: Sequence[str]
) -> tuple[xr.DataArray, xr.DataArray]:
    """Return lower and upper credible bounds for a posterior variable."""
    if not sample_dims:
        return da, da
    quantiles = da.quantile([0.03, 0.97], dim=sample_dims)
    return quantiles.sel(quantile=0.03), quantiles.sel(quantile=0.97)


def _sigma_records_from_posterior(trace: xr.DataTree) -> list[dict[str, object]]:
    """Extract sigma_obs summaries directly from posterior variables."""
    if not hasattr(trace, "posterior"):
        return []

    records: list[dict[str, object]] = []
    pattern = r"sigma_obs_(?P<label>[A-Za-z0-9]+)_(?P<well>.+)"
    for name, da in trace.posterior.data_vars.items():
        var_name = str(name)
        match = pd.Series([var_name]).str.extract(pattern).iloc[0]
        if match.isna().any():
            continue

        sample_dims = _sample_dims_from_posterior_var(da)
        mean_da = da.mean(dim=sample_dims) if sample_dims else da
        sd_da = da.std(dim=sample_dims) if sample_dims else xr.zeros_like(da)
        lo_da, hi_da = _interval_bounds_from_var(da, sample_dims)

        index_dim = next((dim for dim in mean_da.dims), None)
        if index_dim is None:
            records.append({
                "mean": float(mean_da.values),
                "sd": float(sd_da.values),
                "hdi_3%": float(lo_da.values),
                "hdi_97%": float(hi_da.values),
                "label": str(match["label"]),
                "well": str(match["well"]),
                "idx": 0,
            })
            continue

        coord = mean_da.coords.get(index_dim)
        idx_values = (
            coord.to_numpy()
            if coord is not None
            else np.arange(mean_da.sizes[index_dim])
        )
        for pos, raw_idx in enumerate(np.asarray(idx_values).tolist(), start=0):
            try:
                idx = int(raw_idx)
            except (TypeError, ValueError):
                idx = pos
            records.append({
                "mean": float(mean_da.isel({index_dim: pos}).values),
                "sd": float(sd_da.isel({index_dim: pos}).values),
                "hdi_3%": float(lo_da.isel({index_dim: pos}).values),
                "hdi_97%": float(hi_da.isel({index_dim: pos}).values),
                "label": str(match["label"]),
                "well": str(match["well"]),
                "idx": idx,
            })
    return records


def _summary_stats_from_scalar_var(
    trace: xr.DataTree, var_name: str
) -> dict[str, float] | None:
    """Summarize a scalar posterior variable without relying on ``az.summary``."""
    if not hasattr(trace, "posterior") or var_name not in trace.posterior:
        return None
    da = trace.posterior[var_name]
    sample_dims = _sample_dims_from_posterior_var(da)
    mean_da = da.mean(dim=sample_dims) if sample_dims else da
    sd_da = da.std(dim=sample_dims) if sample_dims else xr.zeros_like(da)
    lo_da, hi_da = _interval_bounds_from_var(da, sample_dims)
    return {
        "mean": float(np.asarray(mean_da.values)),
        "sd": float(np.asarray(sd_da.values)),
        "hdi_3%": float(np.asarray(lo_da.values)),
        "hdi_97%": float(np.asarray(hi_da.values)),
    }


def _extract_sigma_df(
    trace: xr.DataTree, results: Mapping[str, FitResult[MiniT]] | None = None
) -> pd.DataFrame:
    """Extract heteroscedastic sigma_obs parameters from a PyMC trace into a DataFrame."""
    sigma_records = _sigma_records_from_posterior(trace)
    if sigma_records:
        return pd.DataFrame(sigma_records)

    # Fallback: if sigma_obs_ isn't in trace, reconstruct from ye_mag * y_err
    if results is None:
        return pd.DataFrame()

    records = []
    for well, fr in results.items():
        if fr.dataset is None:
            continue
        for lbl, da in fr.dataset.items():
            ye = _summary_stats_from_scalar_var(trace, f"ye_mag_{lbl}")
            if ye is None:
                ye = _summary_stats_from_scalar_var(trace, "ye_mag")
            if ye is None:
                continue

            for idx, y_err_val in enumerate(da.y_errc):
                records.append({
                    "mean": float(ye["mean"] * y_err_val),
                    "sd": float(ye["sd"] * y_err_val),
                    "hdi_3%": float(ye["hdi_3%"] * y_err_val),
                    "hdi_97%": float(ye["hdi_97%"] * y_err_val),
                    "label": lbl,
                    "well": well,
                    "idx": int(idx),
                })

    return pd.DataFrame(records)


def extract_sigma_df(
    trace: xr.DataTree, results: Mapping[str, FitResult[MiniT]] | None = None
) -> pd.DataFrame:
    """Extract heteroscedastic sigma summaries from a PyMC trace."""
    return _extract_sigma_df(trace, results)


def plot_noise_vs_index(
    trace: xr.DataTree,
    results: Mapping[str, FitResult[MiniT]] | None = None,
    wells: Sequence[str] | str | None = None,
    figsize_per_well: tuple[float, float] = (5, 4),
    max_cols: int = 4,
) -> Figure:
    """Plot inferred noise (sigma) across titration steps for specified wells.

    Parameters
    ----------
    trace : xr.DataTree
        The PyMC inference trace containing `sigma_obs` deterministic nodes.
    results : Mapping[str, FitResult[MiniT]] | None, optional
        The dictionary of well results to derive fallback sigma values.
    wells : Sequence[str] | str | None, optional
        A specific well ID (e.g., 'A01'), a list of well IDs, or None to plot all
        wells found in the trace. Default is None.
    figsize_per_well : tuple[float, float], optional
        The width and height allocated per well subplot.
    max_cols : int, optional
        Maximum number of columns in the subplot grid.

    Returns
    -------
    Figure
        The constructed matplotlib figure.
    """
    sigma_df = _extract_sigma_df(trace, results)
    if len(sigma_df) == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No 'sigma_obs' data found.", ha="center")
        return fig

    if wells is None:
        wells_to_plot = list(sigma_df["well"].unique())
    elif isinstance(wells, str):
        wells_to_plot = [wells]
    else:
        wells_to_plot = list(wells)

    n_wells = len(wells_to_plot)
    n_cols = min(n_wells, max_cols)
    n_rows = max(1, (n_wells - 1) // n_cols + 1)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * figsize_per_well[0], n_rows * figsize_per_well[1]),
        squeeze=False,
    )

    for i, well in enumerate(wells_to_plot):
        ax = axes.flat[i]
        well_data = sigma_df[sigma_df["well"] == well]
        if len(well_data) == 0:
            ax.set_title(f"Well {well} (No Data)")
            continue

        sns.lineplot(data=well_data, x="idx", y="mean", hue="label", marker="o", ax=ax)

        # Add error bands (94% HDI) for the noise estimate
        for lbl in well_data["label"].unique():
            lbl_data = well_data[well_data["label"] == lbl]
            ax.fill_between(
                lbl_data["idx"],
                lbl_data["hdi_3%"],
                lbl_data["hdi_97%"],
                alpha=0.2,
            )

        ax.set_title(rf"Inferred Noise ($\sigma_{{obs}}$) - Well {well}")
        ax.set_xlabel("Titration Step (Index)")
        ax.set_ylabel(r"Inferred $\sigma$ (RFU)")

    # Hide unused axes
    for j in range(len(wells_to_plot), len(axes.flat)):
        axes.flat[j].set_visible(False)

    plt.tight_layout()
    return fig


def plot_noise_vs_signal(
    trace: xr.DataTree,
    results: Mapping[str, FitResult[MiniT]],
    figsize_per_label: tuple[float, float] = (6, 5),
) -> Figure:
    """Plot inferred noise (sigma) versus observed signal across all wells.

    This function extracts the `sigma_obs_...` parameters from a heteroscedastic
    PyMC trace (like `fit_binding_pymc_multi`) and plots them against the observed
    fluorescence values `da.y` to visualize the noise-to-signal relationship.

    Parameters
    ----------
    trace : xr.DataTree
        The PyMC inference trace containing the `sigma_obs` deterministic nodes.
    results : Mapping[str, FitResult[MiniT]]
        The dictionary of well results containing datasets with `.y` arrays.
        Normally this is `tit.result_global.results`.
    figsize_per_label : tuple[float, float], optional
        The width and height to allocate per band/label in the final figure.

    Returns
    -------
    Figure
        The constructed matplotlib figure.
    """
    sigma_df = _extract_sigma_df(trace, results)
    if len(sigma_df) == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No 'sigma_obs' data found.", ha="center")
        return fig

    # Build a combined DataFrame of (Observed Y, Inferred Sigma) for all wells
    records = []
    for well, fr in results.items():
        if fr.dataset is not None:
            for lbl, da in fr.dataset.items():
                # Get the sigma_obs rows for this well and label, sorted by index
                mask_well_lbl = (sigma_df["well"] == well) & (sigma_df["label"] == lbl)
                well_lbl_sigmas = sigma_df[mask_well_lbl].sort_values("idx")

                # Ensure we have PyMC results for this well
                if len(well_lbl_sigmas) > 0 and hasattr(da, "mask"):
                    # Apply the dataset mask to the full sigma array to match valid da.y points
                    valid_sigmas = well_lbl_sigmas[da.mask]["mean"].to_numpy()
                    valid_y = da.y

                    for sig, y_val in zip(valid_sigmas, valid_y, strict=False):
                        records.append({
                            "well": well,
                            "label": lbl,
                            "sigma_mean": sig,
                            "y_obs": y_val,
                        })
    all_wells_df = pd.DataFrame(records)
    if len(all_wells_df) == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No 'sigma_obs' data found.", ha="center")
        return fig

    # Plotting
    labels_found = all_wells_df["label"].unique()
    fig, axes = plt.subplots(
        1,
        len(labels_found),
        figsize=(figsize_per_label[0] * len(labels_found), figsize_per_label[1]),
    )
    if len(labels_found) == 1:
        axes = [axes]

    for ax, lbl in zip(axes, labels_found, strict=False):
        data = all_wells_df[all_wells_df["label"] == lbl]

        # Plot all points as a semi-transparent scatter
        sns.scatterplot(
            data=data,
            x="y_obs",
            y="sigma_mean",
            alpha=0.3,
            edgecolor=None,
            color="indigo",
            s=20,
            ax=ax,
        )

        # Add a trendline to highlight the heteroscedastic curve shape
        sns.regplot(
            data=data,
            x="y_obs",
            y="sigma_mean",
            scatter=False,
            order=2,
            color="orange",
            line_kws={"linestyle": "--", "linewidth": 2},
            ax=ax,
        )

        ax.set_title(f"Band {lbl} (All Wells)")
        ax.set_xlabel("Observed Fluorescence ($y$)")
        ax.set_ylabel(r"Inferred $\sigma_{obs}$ (mean)")

    plt.suptitle("Noise vs Signal Intensity (All Wells)", y=1.05, fontsize=14)
    plt.tight_layout()
    return fig


def plot_ppc_well(
    trace: xr.DataTree,
    key: str,
    labels: list[str] | None = None,
    figsize: tuple[float, float] = (8, 4),
) -> Figure:
    """Draw posterior predictive samples for a particular well (and all its labels).

    The returned figure can be displayed with matplotlib.

    Parameters
    ----------
    trace   : xr.DataTree
        Trace produced by PyMC fitting with posterior predictive data included.
    key     : str
        Well identifier (e.g. 'A01').
    labels  : list[str] | None
        Names of the bands to show.  If *None* the function will
        automatically look for all variables starting with
        ``'y_likelihood'`` that contain this key.
    figsize: tuple[float, float]
        size?

    Returns
    -------
    Figure
        Plot

    Raises
    ------
    AttributeError
        If the trace does not contain `posterior_predictive` data.
    """
    if not hasattr(trace, "posterior_predictive"):
        msg = (
            "The InferenceData object does not contain 'posterior_predictive'. "
            "You must run pm.sample_posterior_predictive() inside the pm.Model "
            "context after sampling to generate this data."
        )
        raise AttributeError(msg)

    if labels is None:
        labels = [
            str(var).split("_")[2]
            for var in trace.posterior_predictive.data_vars
            if f"_{key}" in str(var) and str(var).startswith("y_likelihood")
        ]

    fig, axes = plt.subplots(
        len(labels), 1, figsize=(figsize[0], figsize[1] * len(labels)), sharex=True
    )
    if len(labels) == 1:
        axes = [axes]

    for ax, lbl in zip(axes, labels, strict=True):
        var_name = f"y_likelihood_{lbl}_{key}"

        # Prepare dictionary for az.from_dict
        trace_dict = {
            "posterior_predictive": {var_name: trace.posterior_predictive[var_name]}
        }
        if hasattr(trace, "observed_data") and var_name in trace.observed_data:
            trace_dict["observed_data"] = {var_name: trace.observed_data[var_name]}

        az.plot_ppc(
            az.from_dict(
                trace_dict,
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


def plot_qc_span_vs_center(  # noqa: PLR0913, PLR0917
    data: Mapping[int, Mapping[str, ArrayF]],
    center: str = "mean",
    figsize_per_label: tuple[float, float] = (5, 4),
    z_threshold: float = 3.0,
    bg_noise: Mapping[str, float | ArrayF] | Mapping[int, float | ArrayF] | None = None,
    bg_multiplier: float = 4.0,
    loglog: bool = False,  # noqa: FBT001, FBT002
    annotate_wells: list[str] | None = None,
) -> Figure:
    """Plot signal span versus center for quality control of titration wells.

    Identifies dead or flat wells (e.g. caused by pipetting errors or missing
    fluorophore) which show low dynamic range relative to their signal center.

    Parameters
    ----------
    data : Mapping[int, Mapping[str, ArrayF]]
        Raw or normalised data keyed by label index, then by well name.
    center : str
        How to compute the x-axis value per well: ``"mean"`` (default) or
        ``"max"`` (maximum absolute signal).
    figsize_per_label : tuple[float, float]
        Figure size allocated per spectral label.
    z_threshold : float
        Z-score threshold for trendline-based outlier detection.
    bg_noise : Mapping[str, float | ArrayF] | Mapping[int, float | ArrayF] | None
        Background noise reference per label (string or integer keys).  Used
        to draw a ``bg_multiplier * bg_noise`` reference line.
    bg_multiplier : float
        Multiplier applied to ``bg_noise`` for the low-signal reference.
    loglog : bool
        If True, use log-log axes.
    annotate_wells : list[str] | None
        Well IDs to annotate even if not flagged as outliers.

    Returns
    -------
    Figure
        The generated QC matplotlib figure.
    """
    labels = list(data.keys())
    fig, axes = _create_qc_subplots(labels, figsize_per_label)

    for ax, lbl in zip(axes, labels, strict=False):
        da_dict = data[lbl]
        panel_rows: list[dict[str, str | float]] = []
        for well, y_arr in da_dict.items():
            y = np.asarray(y_arr)
            valid = y[~np.isnan(y)]
            if len(valid) < 2:  # noqa: PLR2004
                continue
            center_value = (
                float(np.mean(np.abs(valid)))
                if center == "mean"
                else float(np.max(np.abs(valid)))
            )
            panel_rows.append({
                "well": well,
                "center": center_value,
                "span": float(np.max(valid) - np.min(valid)),
            })

        if not panel_rows:
            ax.set_title(f"QC: Span vs Center ({lbl}) — no data")
            continue

        panel_df = pd.DataFrame(panel_rows)
        xlabel = f"{'Mean' if center == 'mean' else 'Max'} signal (label {lbl})"
        _render_qc_panel(
            ax,
            panel_df,
            x_col="center",
            y_col="span",
            title=f"QC: Span vs {center.capitalize()} ({lbl})",
            xlabel=xlabel,
            ylabel="Span (max - min)",
            z_threshold=z_threshold,
            bg_value=_resolve_qc_bg_value(bg_noise, lbl),
            bg_multiplier=bg_multiplier,
            annotate_wells=annotate_wells,
            loglog=loglog,
        )

    fig.tight_layout()
    return fig


def plot_qc_span_vs_center_titration(  # noqa: PLR0913, PLR0917
    tit: object,
    center: str = "mean",
    figsize_per_label: tuple[float, float] = (5, 4),
    z_threshold: float = 3.0,
    bg_multiplier: float = 4.0,
    loglog: bool = False,  # noqa: FBT001, FBT002
    annotate_wells: list[str] | None = None,
) -> Figure:
    """Plot signal span versus center for quality control using a Titration object.

    Convenience wrapper around :func:`plot_qc_span_vs_center` that extracts
    data and background noise directly from a :class:`~clophfit.prtecan.Titration`.

    Parameters
    ----------
    tit : object
        The titration object.
    center : str
        How to compute the x-axis value per well: ``"mean"`` or ``"max"``.
    figsize_per_label : tuple[float, float]
        Figure size per spectral label.
    z_threshold : float
        Z-score threshold for trendline outlier detection.
    bg_multiplier : float
        Multiplier applied to the background noise reference.
    loglog : bool
        If True, use log-log axes.
    annotate_wells : list[str] | None
        Well IDs to annotate even if not flagged.

    Returns
    -------
    Figure
        The generated QC matplotlib figure.
    """
    if hasattr(tit, "_get_normalized_or_raw_data"):
        data = tit._get_normalized_or_raw_data()  # noqa: SLF001
    else:
        # Fallback if method is missing or not a prtecan Titration
        data = getattr(tit, "data", {})

    bg_noise: dict[int, float] | None = None
    if hasattr(tit, "bg_noise") and tit.bg_noise:
        bg_noise = {lbl: float(np.mean(v)) for lbl, v in tit.bg_noise.items()}
    return plot_qc_span_vs_center(
        data,
        center=center,
        figsize_per_label=figsize_per_label,
        z_threshold=z_threshold,
        bg_noise=bg_noise,
        bg_multiplier=bg_multiplier,
        loglog=loglog,
        annotate_wells=annotate_wells,
    )
