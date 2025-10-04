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

import arviz as az
import corner  # type: ignore[import-untyped]
import numpy as np
from matplotlib import cm, colormaps, colors
from matplotlib.figure import Figure
from uncertainties import ufloat  # type: ignore[import-untyped]

from clophfit.fitting.models import binding_1site

if TYPE_CHECKING:
    import pandas as pd
    from lmfit import Parameters  # type: ignore[import-untyped]
    from lmfit.minimizer import MinimizerResult  # type: ignore[import-untyped]
    from matplotlib import axes
    from matplotlib.axes import Axes

    from clophfit.clophfit_types import ArrayF

    from .data_structures import Dataset

COLOR_MAP = colormaps["Set1"]  # To color PCA components and LM fit.
N_AUTOVALS = 4


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


def plot_emcee(flatchain: pd.DataFrame) -> Figure:
    """Plot emcee result."""
    # Convert the dictionary of flatchains to an ArviZ InferenceData object
    samples_dict = {key: np.array(val) for key, val in flatchain.items()}
    idata = az.from_dict(posterior=samples_dict)
    corner_fig: Figure = corner.corner(idata, divergences=False)
    return corner_fig


def plot_emcee_k_on_ax(ax: Axes, res_emcee: MinimizerResult, p_name: str = "K") -> None:
    """Plot emcee result."""
    samples = res_emcee.flatchain
    # Convert the dictionary of flatchains to an ArviZ InferenceData object
    samples_dict = {key: np.array(val) for key, val in samples.items()}
    idata = az.from_dict(posterior=samples_dict)
    parameter_posterior = idata.posterior[p_name]  # pylint: disable=E1101
    az.plot_posterior(parameter_posterior, ax=ax)  # type: ignore[no-untyped-call]


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
