"""Prtecan/prtecan.py."""

from __future__ import annotations

import logging
import typing
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import odrpack
import pandas as pd
import seaborn as sns  # type: ignore[import-untyped]
from matplotlib import figure

from clophfit.fitting.data_structures import DataArray, Dataset, FitResult
from clophfit.fitting.odr import format_estimate
from clophfit.fitting.plotting import PlotParameters
from clophfit.fitting.utils import apply_outlier_mask, assign_error_model
from clophfit.utils import weights_from_sigma

if TYPE_CHECKING:
    from collections.abc import Callable

    from clophfit.clophfit_types import ArrayF

# TODO: Add tqdm progress bar
# TODO: sort before computing to have outlier output sorted
from .models import PlateScheme, Tecanfile, TecanfilesGroup
from .parsers import dilution_correction

# Constants for Tecan file parsing
#: Standard metadata line length for Tecan files
STD_MD_LINE_LENGTH = 2
#: Number of columns in a 96-well plate
NUM_COLS_96WELL = 12
#: Row names for 96-well plates
ROW_NAMES = tuple("ABCDEFGH")

logger = logging.getLogger(__name__)

# pH K bounds as used by _build_params_1site in fitting/core.py
_PH_K_MIN: float = 3.0
_PH_K_MAX: float = 11.0


@dataclass
class TitrationConfig:
    """Parameters defining the fitting data with callback support."""

    bg: bool = True
    bg_adj: bool = False
    dil: bool = True
    nrm: bool = True
    bg_mth: str = "mean"
    fit_method: str = "huber"
    outlier: str | None = None
    mcmc: str = "None"
    nuts_sampler: str = "default"
    n_mcmc_samples: int = 2000
    ctr_free_k: bool = False
    noise_alpha: tuple[float, ...] = ()
    """Proportional noise coefficients per label.

    When provided, adds a proportional term to the error estimate so that
    high-signal wells are appropriately down-weighted:
    `y_err^2 = gain * signal + bg_err^2 + (alpha * signal)^2`

    Values typically from MCMC multi-noise shared_noise_params.csv.
    Empty tuple disables the correction (legacy behaviour).
    """
    noise_gain: tuple[float, ...] = ()
    """Poisson gain coefficients per label.

    Replaces the hardcoded gain=1 in the shot-noise Poisson term:
    `y_err^2 = gain * signal + bg_err^2 + (alpha * signal)^2`

    Values typically from MCMC multi-noise shared_noise_params.csv.
    Empty tuple keeps gain=1 (legacy behaviour).
    """

    mask_outliers: bool = field(default=False)
    """Mask geometric outliers in each well's curve before fitting. Default is False."""
    outlier_threshold: float = field(default=0.2)
    """Threshold for geometric outlier scoring (0-1). Default is 0.2."""
    discard_bad_wells: bool = field(default=False)
    """Automatically detect and discard bad wells before fitting. Default is False."""

    _callback: Callable[[], None] | None = field(
        default=None, repr=False, compare=False
    )

    def set_callback(self, callback: Callable[[], None]) -> None:
        """Set the callback to be triggered on parameter change."""
        self._callback = callback

    def _trigger_callback(self) -> None:
        if self._callback is not None:
            self._callback()

    def __setattr__(
        self,
        name: str,
        value: bool | str | float | tuple[float, ...] | None,  # noqa: FBT001
    ) -> None:
        """Trigger callback when a tracked attribute value actually changes."""
        if name == "_callback":
            super().__setattr__(name, value)
        else:
            current_value = getattr(self, name, None)
            super().__setattr__(name, value)
            if current_value != value:
                self._trigger_callback()


@dataclass
class BufferFit:
    """Store (robust) linear fit result."""

    m: float = np.nan
    q: float = np.nan
    m_err: float = np.nan
    q_err: float = np.nan

    @property
    def empty(self) -> bool:
        """Return True if all attributes are NaN, emulating DataFrame's empty behavior."""
        return all(np.isnan(value) for value in vars(self).values())


@dataclass
class Buffer:
    """Buffer handling for a titration.

    Manages background correction and fitting for buffer wells.
    """

    tit: Titration

    _wells: list[str] = field(default_factory=list)
    _bg: dict[str, ArrayF] = field(init=False, default_factory=dict)
    _bg_err: dict[str, ArrayF] = field(init=False, default_factory=dict)

    fit_results: dict[str, BufferFit] = field(init=False, default_factory=dict)
    fit_results_nrm: dict[str, BufferFit] = field(init=False, default_factory=dict)

    @cached_property
    def dataframes(self) -> dict[str, pd.DataFrame]:
        # def dataframes(self) -> list[pd.DataFrame]:
        """Buffer dataframes with fit."""
        if not self.wells:
            return {}
        dfs = {
            label: pd.DataFrame({
                k: lbg.data[k] for k in self.wells if lbg.data and k in lbg.data
            })
            for label, lbg in self.tit.labelblocksgroups.items()
        }
        self.fit_results = self._fit_buffer(dfs)  # Perform fit
        return dfs

    @cached_property
    def dataframes_nrm(self) -> dict[str, pd.DataFrame]:
        # def dataframes_nrm(self) -> list[pd.DataFrame]:
        """Buffer normalized dataframes with fit."""
        if not self.wells:
            return {}
        dfs_nrm = {
            label: pd.DataFrame({k: lbg.data_nrm[k] for k in self.wells})
            for label, lbg in self.tit.labelblocksgroups.items()
        }
        self.fit_results_nrm = self._fit_buffer(dfs_nrm)  # Perform fit
        return dfs_nrm

    @property
    def wells(self) -> list[str]:
        """List of buffer wells."""
        return self._wells

    @wells.setter
    def wells(self, wells: list[str]) -> None:
        """Set the list of buffer wells and trigger recomputation."""
        self._wells = wells
        self._reset_cache()
        self.tit.clear_all_data_results()

    def _reset_cache(self) -> None:
        """Reset all cached properties."""
        for cached_attr in ["dataframes", "dataframes_nrm", "bg", "bg_err"]:
            if cached_attr in self.__dict__:
                del self.__dict__[cached_attr]

    @property
    def bg(self) -> dict[str, ArrayF]:
        """List of buffer values."""
        if not self._bg:
            self._bg, self._bg_err = self._compute_bg_and_sd()
        return self._bg

    @bg.setter
    def bg(self, value: dict[str, ArrayF]) -> None:
        """Set the buffer values and reset SEM."""
        self._bg = value

    @property
    def bg_err(self) -> dict[str, ArrayF]:
        """List of buffer SEM values."""
        if not self._bg_err:
            self._bg, self._bg_err = self._compute_bg_and_sd()
        return self._bg_err

    @bg_err.setter
    def bg_err(self, value: dict[str, ArrayF]) -> None:
        # def bg_err(self, value: list[ArrayF]) -> None:
        """Set the buffer SEM values manually."""
        self._bg_err = value

    @property
    def bg_noise(self) -> dict[str, float]:
        """Intrinsic well noise (RMSE/pooled SD) values."""
        buffers = self.dataframes_nrm if self.tit.params.nrm else self.dataframes
        noise = {}
        noise_col = "fit_noise" if self.tit.params.bg_mth == "fit" else "mean_noise"
        for label, bdf in buffers.items():
            if bdf.empty:
                noise[label] = 0.0
            else:
                # noise is a scalar per label for the whole dataset
                noise[label] = float(bdf[noise_col].iloc[0])
        return noise

    def _compute_bg_and_sd(self) -> tuple[dict[str, ArrayF], dict[str, ArrayF]]:
        """Compute and return buffer values and their SEM."""
        buffers = self.dataframes_nrm if self.tit.params.nrm else self.dataframes
        bg = {}
        bg_err = {}
        # Mapping methods to column names for clarity and reuse
        method_map = {
            "fit": ("fit", "fit_err"),
            "mean": ("mean", "sem"),
            "meansd": ("mean", "sem"),
        }
        if self.tit.params.bg_mth not in method_map:
            msg = f"Unknown bg_method: {self.tit.params.bg_mth}"
            raise ValueError(msg)
        value_col, error_col = method_map[self.tit.params.bg_mth]
        for label, bdf in buffers.items():
            if bdf.empty:
                bg[label] = np.array([])
                bg_err[label] = np.array([])
                continue
            bg[label] = bdf[value_col].to_numpy()
            if self.tit.params.bg_mth == "meansd":
                bg_err[label] = np.repeat(
                    np.nanpercentile(bdf[error_col], 50), len(bdf[error_col])
                )
            else:
                bg_err[label] = bdf[error_col].to_numpy()
        return bg, bg_err

    def _fit_buffer(self, dataframed: dict[str, pd.DataFrame]) -> dict[str, BufferFit]:
        """Fit buffers of all labelblocksgroups."""

        def linear_model(x: ArrayF, beta: ArrayF) -> ArrayF:
            """Define linear model function."""
            return typing.cast("ArrayF", beta[0] * x + beta[1])

        def fit_error(x: ArrayF, cov_matrix: ArrayF) -> ArrayF:
            x = x[:, np.newaxis]  # Ensure x is a 2D array
            jacobian = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
            fit_variance: ArrayF = np.einsum(
                "ij,jk,ik->i", jacobian, cov_matrix, jacobian
            )
            return np.sqrt(fit_variance)  # Standard error

        fit_resultd = {}
        for label, buf_df in dataframed.items():
            if buf_df.empty:
                fit_resultd[label] = BufferFit()
            else:
                y_obs = buf_df.to_numpy().astype(float)
                mean = buf_df.mean(axis=1).to_numpy().astype(float)
                sem = buf_df.sem(axis=1).to_numpy().astype(float)
                # y_err estimate is important when using 2 ds and x_err for ODR
                weight_x = weights_from_sigma(self.tit.x_err)
                weight_y = weights_from_sigma(sem)
                # Initial guess for slope and intercept
                output = odrpack.odr_fit(
                    linear_model,
                    self.tit.x,
                    mean,
                    beta0=[0.0, mean.mean()],
                    weight_x=weight_x,
                    weight_y=weight_y,
                )
                # Extract the best-fit parameters and their standard errors
                m_best, q_best = output.beta
                m_err, q_err = output.sd_beta
                cov_matrix = output.cov_beta
                fit_resultd[label] = BufferFit(
                    float(m_best), float(q_best), float(m_err), float(q_err)
                )

                # intrinsic well noise (RMSE of fit, or pooled SD of mean)
                y_fit = m_best * self.tit.x + q_best
                diffs = y_obs - y_fit[:, np.newaxis]
                n_obs = diffs.size
                sigma_res = (
                    np.sqrt(np.sum(diffs**2) / (n_obs - 2)) if n_obs > 2 else 0.0  # noqa: PLR2004
                )
                pooled_std = np.sqrt(buf_df.var(axis=1, ddof=1).mean())

                buf_df["Label"] = label
                buf_df["fit"] = y_fit
                buf_df["fit_err"] = fit_error(self.tit.x, cov_matrix)
                buf_df["fit_noise"] = sigma_res
                buf_df["mean"] = mean
                buf_df["sem"] = sem
                buf_df["mean_noise"] = pooled_std
        return fit_resultd

    def plot(self, *, nrm: bool = False, title: str | None = None) -> sns.FacetGrid:
        """Plot buffers of all labelblocksgroups."""
        dataframed = self.dataframes_nrm if nrm else self.dataframes
        fit_results = self.fit_results_nrm if nrm else self.fit_results
        if not dataframed or not self.wells:
            return sns.catplot()
        pp = PlotParameters(is_ph=self.tit.is_ph)
        melted_buffers = []
        wells_lbl = self.wells.copy()
        wells_lbl.extend(["Label"])
        for buf_df in dataframed.values():
            if not buf_df.empty:
                buffer = buf_df[wells_lbl].copy()
                buffer[pp.kind] = self.tit.x
                melted_buffers.append(
                    buffer.melt(
                        id_vars=[pp.kind, "Label"], var_name="well", value_name="F"
                    )
                )
        # Combine data from both buffers
        data = pd.concat(melted_buffers, ignore_index=True)
        g = sns.lmplot(
            data=data,
            y="F",
            x=pp.kind,
            ci=68,
            height=4,
            aspect=1.75,
            row="Label",
            x_estimator=np.median,
            markers="x",
            scatter=1,
            scatter_kws={"alpha": 0.33},
            facet_kws={"sharey": False},
        )
        # Determine the number of non-empty label groups
        num_labels = sum(not b_df.empty for b_df in dataframed.values())
        for label in dataframed:
            if not dataframed[label].empty:
                sns.scatterplot(
                    data=data[data.Label == label],
                    y="F",
                    x=pp.kind,
                    hue="well",
                    ax=g.axes_dict[label],
                    legend=label == str(num_labels),
                )
                g.axes_dict[label].errorbar(
                    x=self.tit.x,
                    y=dataframed[label]["fit"],
                    yerr=dataframed[label]["fit_err"],
                    xerr=self.tit.x_err,
                    fmt="",
                    color="r",
                    linewidth=2,
                    capsize=6,
                )
                # Extract the slope and intercept from BufferFit
                buffer_fit = fit_results[label]
                m = buffer_fit.m
                m_err = buffer_fit.m_err
                q = buffer_fit.q
                q_err = buffer_fit.q_err
                # Add slope and intercept as text annotation on the plot
                g.axes_dict[label].text(
                    0.05,
                    0.95,  # Position of the text (adjust as needed)
                    f"m = {m:.1f} ± {m_err:.1f}\nq = {q:.0f} ± {q_err:.1f}",
                    transform=g.axes_dict[label].transAxes,
                    fontsize=10,
                    verticalalignment="top",
                    bbox={
                        "boxstyle": "round,pad=0.3",
                        "edgecolor": "black",
                        "facecolor": "white",
                        "alpha": 0.7,
                    },
                )
        if title:
            plt.suptitle(title, fontsize=14, x=0.96, ha="right")
        plt.close()
        return g


@dataclass
class TitrationResults:
    """Manage titration results with optional lazy computation."""

    scheme: PlateScheme
    fit_keys: set[str]
    results: dict[str, FitResult[typing.Any]] = field(default_factory=dict)
    _dataframe: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def dataframe(self) -> pd.DataFrame:
        """Convert FitResult dictionary to a DataFrame."""
        if all(key in list(self._dataframe.index) for key in self.fit_keys):
            return self._dataframe

        data = []
        for lbl, fr in self.results.items():
            pars = fr.result.params if fr.result else None
            row = {"well": lbl}
            if pars is not None:
                for k in pars:
                    row[k] = pars[k].value
                    row[f"s{k}"] = pars[k].stderr
                    row[f"{k}hdi03"] = pars[k].min
                    row[f"{k}hdi97"] = pars[k].max
            data.append(row)
        self._dataframe = pd.DataFrame(data).set_index("well")
        return self._dataframe

    def __repr__(self) -> str:
        """Get or lazily compute a result for a given key."""
        return repr(self.results)

    def __getitem__(self, key: str) -> FitResult[typing.Any]:
        """Fetch result for a single key."""
        return self.results[key]

    def __bool__(self) -> bool:
        """Return True if there are any computed results, trigger full computation."""
        return bool(self.results)

    def __call__(self) -> None:
        """Call object to ensure all results are computed."""

    def __len__(self) -> int:
        """Ensure length is accurate after full computation."""
        return len(self.results)

    def compute_all(self) -> None:
        """Compute results for all keys."""

    def n_sd(self, par: str = "K", expected_sd: float = 0.15) -> float:
        """Compute median of K."""
        if not self.all_computed():
            self.compute_all()
        stderr_vals = [
            v.result.params[par].stderr
            for v in self.results.values()
            if v.result and v.result.params[par].stderr is not None
        ]
        if not stderr_vals:
            logger.warning("Unable to calculate n_sd; defaulting to 1.0")
            return 1.0
        try:
            n_sd: float = expected_sd / np.nanmedian(stderr_vals)
        except ZeroDivisionError:
            logger.warning("Unable to calculate n_sd; defaulting to 1.0")
            n_sd = 1.0
        return n_sd

    @staticmethod
    def all_computed() -> bool:
        """Check if all keys have been computed."""
        return True

    def export_pngs(self, folder: str | Path) -> None:
        """Export all fit result plots as PNG files."""
        path = Path(folder)
        path.mkdir(parents=True, exist_ok=True)
        for well, result in self.results.items():
            if result.figure:
                result.figure.savefig(path / f"{well}.png")

    def export_data(self, folder: str | Path) -> None:
        """Export all datasets as CSV files."""
        path = Path(folder) / "ds"
        path.mkdir(parents=True, exist_ok=True)
        for well, result in self.results.items():
            if result.dataset:
                result.dataset.export(path / f"{well}.csv")

    # MAYBE: Test plots
    def plot_k(
        self,
        xlim: tuple[float, float] | None = None,
        title: str = "",
    ) -> figure.Figure:
        """Plot K values as stripplot.

        Parameters
        ----------
        xlim : tuple[float, float] | None, optional
            Range.
        title : str, optional
            To name the plot.

        Returns
        -------
        figure.Figure
            The figure.
        """
        dataframe = self.dataframe
        with sns.plotting_context("paper"):  # axes_style("whitegrid"):
            fig = plt.figure(figsize=(12, 16))
            keys_unk = list(set(dataframe.index))
            if self.scheme.names:
                keys_unk = list(set(dataframe.index) - set(self.scheme.ctrl))
                df_ctr = dataframe.loc[dataframe.index.intersection(self.scheme.ctrl)]
                for name, wells in self.scheme.names.items():
                    for well in wells:
                        df_ctr.loc[well, "ctrl"] = name
                df_ctr = (
                    df_ctr
                    .assign(_well=df_ctr.index)
                    .sort_values(["ctrl", "_well"])
                    .drop(columns=["_well"])
                )
                ax1 = plt.subplot2grid((8, 1), loc=(0, 0))
                x, y, hue = (df_ctr["K"], df_ctr.index, df_ctr["ctrl"])
                sns.stripplot(x=x, y=y, size=8, orient="h", hue=hue, ax=ax1)
                ax1.errorbar(x, y, xerr=df_ctr["sK"], fmt=".", c="lightgray", lw=8)
                ax1.legend(loc="upper left", frameon=False)
                ax1.grid(visible=True, axis="both")
                ax1.set_xticklabels([])
                ax1.set_xlabel("")
            ax2 = plt.subplot2grid((8, 1), loc=(1, 0), rowspan=7)
            df_unk = dataframe.loc[keys_unk].sort_index(ascending=False)
            # Sort by 'K - 2 * sK'.
            df_unk["sort_val"] = df_unk["K"] - 2 * df_unk["sK"]
            df_unk = df_unk.sort_values(by="sort_val", ascending=True)
            x, y = df_unk["K"], df_unk.index
            sns.stripplot(x=x, y=y, size=9, orient="h", ax=ax2)
            ax2.errorbar(x, y, xerr=df_unk["sK"], fmt=".", c="gray", lw=2)
            ax2.grid(visible=True, axis="both")
            ax2.set_yticks(range(len(df_unk)))
            ax2.set_yticklabels([str(label) for label in df_unk.index])
            ax2.set_ylim(-1, len(df_unk))
            # Set x-limits
            xlim = xlim or self._determine_xlim(df_ctr, df_unk)
            if self.scheme.ctrl:
                ax1.set_xlim(xlim)
            ax2.set_xlim(xlim)
            # Set title
            fig.suptitle(title, fontsize=16)
            fig.tight_layout(pad=1.2, w_pad=0.1, h_pad=0.5, rect=(0, 0, 1, 0.97))
            # Close the figure after returning it to avoid memory issues
        plt.close(fig)
        return fig

    @staticmethod
    def _determine_xlim(
        df_ctr: pd.DataFrame, df_unk: pd.DataFrame
    ) -> tuple[float, float]:
        lower, upper = 0.99, 1.01
        xlim = (df_unk["K"].min(), df_unk["K"].max())
        if not df_ctr.empty:
            xlim = (min(df_ctr["K"].min(), xlim[0]), max(df_ctr["K"].max(), xlim[1]))
            xlim = (lower * xlim[0], upper * xlim[1])
        return xlim


@dataclass  # Too many public methods - acceptable for complex classes
class Titration(TecanfilesGroup):
    """Build titrations from grouped Tecanfiles and concentrations or pH values.

    Parameters
    ----------
    tecanfiles: list[Tecanfile]
        List of Tecanfiles.
    x : ArrayF
        Concentration or pH values.
    is_ph :
    x_err :

    Raises
    ------
    ValueError
        For unexpected file format, e.g. header `names`.
    """

    x: ArrayF
    is_ph: bool = False
    """Indicate if x values represent pH."""
    x_err: ArrayF = field(default_factory=lambda: np.array([]))
    """Uncertainties for x values (default is empty array)."""
    buffer: Buffer = field(init=False)
    """Buffer wells data and fit results. Set during initialization."""

    _params: TitrationConfig = field(init=False, default_factory=TitrationConfig)
    _additions: list[float] = field(init=False, default_factory=list)
    _scheme: PlateScheme = field(init=False, default_factory=PlateScheme)
    _bg: dict[str, ArrayF] = field(init=False, default_factory=dict)
    _bg_err: dict[str, ArrayF] = field(init=False, default_factory=dict)
    _data: dict[str, dict[str, ArrayF]] = field(init=False, default_factory=dict)

    _dil_corr: ArrayF = field(init=False, default_factory=lambda: np.array([]))
    _bad_wells_detected: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        """Create metadata and data."""
        self.buffer = Buffer(tit=self)
        self._params.set_callback(self._reset_data_results_and_bg)
        super().__post_init__()

    @cached_property
    def fit_keys(self) -> set[str]:
        """Set of wells to be fitted."""
        first_label = next(iter(self.labelblocksgroups.keys()))
        return (
            self.labelblocksgroups[first_label].data_nrm.keys() - self.scheme.nofit_keys
        )

    def detect_and_discard_bad_wells(  # noqa: C901, PLR0912
        self,
        smoothness_threshold: float | None = None,
        roughness_threshold: float | None = None,
        z_threshold: float | None = None,
        *,
        outlier_threshold: float | None = 0.2,
        bg_multiplier: float | None = 3.0,
    ) -> list[str]:
        """Detect and discard bad wells from masked per-label signal quality.

        By default, each well is converted to a per-label dataset with
        :meth:`_create_ds`, masked with :func:`apply_outlier_mask`, and then
        discarded when the mean masked signal of any label falls below a
        background-derived floor. The floor uses ``bg_err`` when available and
        falls back to ``bg_noise``.

        Legacy smoothness, roughness, and trendline criteria are still available
        as optional extra filters when their thresholds are provided explicitly.

        Parameters
        ----------
        smoothness_threshold : float | None
            Optional maximum allowed smoothness value.
        roughness_threshold : float | None
            Optional maximum allowed roughness value.
        z_threshold : float | None
            Optional trendline outlier threshold on max signal vs span.
        outlier_threshold : float | None
            Threshold passed to :func:`apply_outlier_mask` before computing
            per-label summary statistics. If ``None``, no masking is applied.
        bg_multiplier : float | None
            Discard a well when any masked per-label mean signal is below
            ``bg_multiplier * mean(background_floor)``. If ``None``, this check
            is disabled.

        Returns
        -------
        list[str]
            Newly discarded well keys.
        """
        from clophfit.fitting.utils import (  # noqa: PLC0415
            flag_trend_outliers,
            roughness,
            smoothness,
        )

        first_label = next(iter(self.labelblocksgroups.keys()), None)
        if first_label is None:
            return []

        candidate_wells = sorted(
            self.labelblocksgroups[first_label].data_nrm.keys() - self.scheme.nofit_keys
        )
        if not candidate_wells:
            return []

        label_ids = sorted(self.labelblocksgroups)
        new_discards: set[str] = set()
        trend_inputs: dict[str, dict[str, dict[str, float]]] = {
            label: {} for label in label_ids
        }

        for well in candidate_wells:
            discard_well = False
            for label in label_ids:
                ds = self.create_ds(well, label)
                if outlier_threshold is not None:
                    ds = apply_outlier_mask(ds, threshold=outlier_threshold)
                y = np.asarray(ds[str(label)].y, dtype=float)
                valid_y = y[~np.isnan(y)]
                if len(valid_y) < 2:  # noqa: PLR2004
                    discard_well = True
                    break

                if bg_multiplier is not None:
                    floor_values = self.bg_err.get(label)
                    if floor_values is None or np.asarray(floor_values).size == 0:
                        floor = self.bg_noise.get(label)
                    else:
                        floor = float(np.nanmean(np.asarray(floor_values, dtype=float)))

                    if floor is not None and (
                        np.isfinite(floor)
                        and float(np.nanmean(valid_y)) < bg_multiplier * floor
                    ):
                        discard_well = True

                if (
                    smoothness_threshold is not None
                    and smoothness(valid_y) > smoothness_threshold
                ):
                    discard_well = True

                if (
                    roughness_threshold is not None
                    and roughness(valid_y) > roughness_threshold
                ):
                    discard_well = True

                trend_inputs[label][well] = {
                    "max_sig": float(np.max(np.abs(valid_y))),
                    "span": float(np.max(valid_y) - np.min(valid_y)),
                }

            if discard_well:
                new_discards.add(well)

        if z_threshold is not None:
            for label in label_ids:
                label_metrics = trend_inputs[label]
                if not label_metrics:
                    continue
                x_series = pd.Series({
                    well: metrics["max_sig"] for well, metrics in label_metrics.items()
                })
                y_series = pd.Series({
                    well: metrics["span"] for well, metrics in label_metrics.items()
                })
                outliers = flag_trend_outliers(
                    x_series, y_series, threshold=z_threshold
                )
                new_discards.update(outliers[outliers].index)

        if new_discards:
            self.scheme.discard = list(set(self.scheme.discard) | new_discards)
            self.clear_all_data_results()

        return sorted(new_discards)

    def _ensure_bad_wells_detected(self) -> None:
        if not self.params.discard_bad_wells:
            return
        if self._bad_wells_detected:
            return
        self._bad_wells_detected = True
        self.detect_and_discard_bad_wells()

    def _reset_data_and_results(self) -> None:
        self._data = {}
        cached_results = (
            "results",
            "result_global",
            "result_odr",
            "result_mcmc",
            "result_multi_trace",
            "result_multi_trace2",
            "result_multi_noise",
            "result_multi_noise_xrw",
            "result_multi_mcmc",
            "result_multi_noise_mcmc",
            "result_multi_noise_xrw_mcmc",
            "fit_pipeline",
        )
        for attr in cached_results:
            self.__dict__.pop(attr, None)

    def _reset_data_results_and_bg(self) -> None:
        self._reset_data_and_results()
        self.bg = {}
        self.bg_err = {}

    def clear_all_data_results(self) -> None:
        """Clear fit keys, data, results and bg when buffer or scheme properties change."""
        self._reset_data_results_and_bg()
        if "fit_keys" in self.__dict__:
            del self.fit_keys

    @property
    def params(self) -> TitrationConfig:
        """Get the datafit parameters."""
        return self._params

    @params.setter
    def params(self, value: TitrationConfig) -> None:
        self._params = value
        self._reset_data_results_and_bg()

    @property
    def bg(self) -> dict[str, ArrayF]:
        # def bg(self) -> list[ArrayF]:
        """List of buffer values."""
        return self.buffer.bg

    @bg.setter
    def bg(self, value: dict[str, ArrayF]) -> None:
        # def bg(self, value: list[ArrayF]) -> None:
        self.buffer.bg = value
        self._reset_data_and_results()

    @property
    def bg_err(self) -> dict[str, ArrayF]:
        # def bg_err(self) -> list[ArrayF]:
        """List of buffer SEM values."""
        return self.buffer.bg_err

    @bg_err.setter
    def bg_err(self, value: dict[str, ArrayF]) -> None:
        # def bg_err(self, value: list[ArrayF]) -> None:
        self.buffer.bg_err = value
        self._reset_data_and_results()

    @property
    def bg_noise(self) -> dict[str, float]:
        """Intrinsic well noise (RMSE/pooled SD) values."""
        return self.buffer.bg_noise

    def __repr__(self) -> str:
        """Return a string representation of the instance."""
        return (
            f'Titration\n\tfiles=["{self.tecanfiles[0].path}", ...],\n'
            f"\tx={list(self.x)!r},\n"
            f"\tx_err={list(self.x_err)!r},\n"
            f"\tlabels={self.labelblocksgroups.keys()},\n"
            f"\tparams={self.params!r}"
            f"\tpH={self.is_ph}"
            f"\tadditions={self.additions}"
            f"\n\tscheme={self.scheme})"
        )

    @classmethod
    def fromlistfile(cls, list_file: Path | str, *, is_ph: bool) -> Titration:
        """Build `Titration` from a list[.pH|.Cl] file.

        Parameters
        ----------
        list_file : Path | str
            Path to the list file containing [filenames x x_err].
        is_ph : bool
            Whether x values represent pH (True) or concentrations (False).

        Returns
        -------
        Titration
            The constructed Titration object.
        """
        tecanfiles, x, x_err = cls._listfile(Path(list_file))
        return cls(tecanfiles, x, is_ph, x_err=x_err)

    @staticmethod
    def _listfile(listfile: Path) -> tuple[list[Tecanfile], ArrayF, ArrayF]:
        """Help construction from list file."""
        try:
            table = pd.read_csv(listfile, names=["filenames", "x", "x_err"])
        except FileNotFoundError as exc:
            msg = f"Cannot find: {listfile}"
            raise FileNotFoundError(msg) from exc
        # For unexpected file format, e.g. length of filename column differs
        # from length of x values.
        if table["filenames"].count() != table["x"].count():
            msg = f"Check format [filenames x x_err] for listfile: {listfile}"
            raise ValueError(msg)
        tecanfiles = [Tecanfile(listfile.parent / f) for f in table["filenames"]]
        x = table["x"].to_numpy().astype(float)
        x_err = table["x_err"].to_numpy().astype(float)
        return tecanfiles, x, x_err

    @property
    def additions(self) -> list[float] | None:
        """List of initial volume followed by additions."""
        return self._additions

    # MAYBE: Here there is not any check on the validity of additions (e.g. length).
    @additions.setter
    def additions(self, additions: list[float]) -> None:
        self._additions = additions
        self._dil_corr = dilution_correction(additions)
        self._data = {}

    def load_additions(self, additions_file: Path) -> None:
        """Load additions from file.

        Reads a CSV file with a single column 'add' containing addition volumes,
        and updates the Titration's additions property.

        Parameters
        ----------
        additions_file : Path
            Path to the additions CSV file.
        """
        additions = pd.read_csv(additions_file, names=["add"])
        self.additions = additions["add"].tolist()

    @property
    def data(self) -> dict[str, dict[str, ArrayF]]:
        # def data(self) -> list[dict[str, ArrayF]]:
        """Buffer subtracted and corrected for dilution data."""
        if not self._data:
            self._data = self._prepare_data()
        return self._data

    # def _prepare_data(self) -> list[dict[str, ArrayF]]:
    def _prepare_data(self) -> dict[str, dict[str, ArrayF]]:
        """Prepare and return the processed data."""
        # Step 1: Get raw or normalized data
        data = self._get_normalized_or_raw_data()
        # Step 2: Subtract background if enabled
        if self.params.bg and self.bg:
            data = self._subtract_background(data)
        # Step 3: Adjust for negative values if enabled
        if self.params.bg_adj:
            data = self._adjust_negative_values(data)
        # Step 4: Apply dilution correction if enabled
        if self.params.dil and self.additions:
            data = self._apply_dilution_correction(data)
        return data

    def _apply_dilution_correction(
        self, data: dict[str, dict[str, ArrayF]]
    ) -> dict[str, dict[str, ArrayF]]:
        """Apply dilution correction to the data (works with nan values)."""
        return {
            label: {k: v * self._dil_corr for k, v in dd.items()}
            for label, dd in data.items()
        }

    # def _get_normalized_or_raw_data(self) -> list[dict[str, ArrayF]]:
    def _get_normalized_or_raw_data(self) -> dict[str, dict[str, ArrayF]]:
        """Fetch raw or normalized data, transforming into arrays."""
        if self.params.nrm:
            return {
                label: {k: np.array(v) for k, v in lbg.data_nrm.items()}
                for label, lbg in self.labelblocksgroups.items()
            }
        return {
            label: {k: np.array(v) for k, v in lbg.data.items()} if lbg.data else {}
            for label, lbg in self.labelblocksgroups.items()
        }

    def _subtract_background(
        self, data: dict[str, dict[str, ArrayF]]
    ) -> dict[str, dict[str, ArrayF]]:
        """Subtract background from data."""
        return {
            label: {k: v - self.bg[label] for k, v in dd.items()}
            for label, dd in data.items()
        }

    def _adjust_negative_values(
        self, data: dict[str, dict[str, ArrayF]]
    ) -> dict[str, dict[str, ArrayF]]:
        """Adjust negative values in the data."""

        def _adjust_subtracted_data(
            key: str, y: ArrayF, sd: float, label: str, alpha: float = 1 / 10
        ) -> ArrayF:
            """Adjust negative values (alpha = F_bound/F_unbound)."""
            if y.min() < alpha * 0 * y.max():
                delta = alpha * (y.max() - y.min()) - y.min()
                logger.warning(
                    "Buffer for '%s:%s' was adjusted by %.2f SD.",
                    key,
                    label,
                    delta / sd,
                )
                return y + float(delta)
            return y  # never used if properly called?

        for i, dd in data.items():
            sd = self.bg_err[i].mean() if self.bg_err[i].size > 0 else np.nan
            for k in self.fit_keys:
                dd[k] = _adjust_subtracted_data(k, dd[k], sd, str(i))
        return data

    @property
    def scheme(self) -> PlateScheme:
        """Scheme for known samples like {'buffer', ['H12', 'H01'], 'ctrl'...}."""
        return self._scheme

    def load_scheme(self, schemefile: Path) -> None:
        """Load scheme from file and set buffer wells.

        Reads a scheme file to define buffer wells, known samples,
        and control wells, then updates the Titration's scheme and buffer wells.

        Parameters
        ----------
        schemefile : Path
            Path to the scheme CSV file.
        """
        self._scheme = PlateScheme(schemefile)
        self.buffer.wells = self._scheme.buffer

    def _create_data_array(self, key: str, label: str) -> DataArray:
        """Create a DataArray for a specific key and label with unit weights."""
        y = np.array(self.data[label][key])
        return DataArray(self.x, y, x_errc=self.x_err, y_errc=np.ones_like(y))

    def _apply_error_model(self, ds: Dataset) -> Dataset:
        """Apply the physical error model from TitrationConfig to the Dataset."""
        sigma_floor: dict[str, float | ArrayF] = {}
        gain: dict[str, float] = {}
        rel_error: dict[str, float] = {}

        labels = sorted(self.data.keys())
        for i, lbl in enumerate(labels):
            sigma_floor[lbl] = self.bg_err[lbl] if self.bg_err else 0.0

            gain[lbl] = (
                self.params.noise_gain[i]
                if self.params.noise_gain and i < len(self.params.noise_gain)
                else 1.0
            )
            rel_error[lbl] = (
                self.params.noise_alpha[i]
                if self.params.noise_alpha and i < len(self.params.noise_alpha)
                else 0.0
            )

        return assign_error_model(
            ds, sigma_floor=sigma_floor, gain=gain, rel_error=rel_error
        )

    def create_ds(self, key: str, label: str) -> Dataset:
        """Create a dataset for the given key."""
        da = self._create_data_array(key, label)
        ds = Dataset({label: da}, is_ph=self.is_ph)
        return self._apply_error_model(ds)

    def create_global_ds(self, key: str) -> Dataset:
        """Create a global dataset for the given key."""
        data_arrays_dict = {i: self._create_data_array(key, i) for i in self.data}
        ds = Dataset(data_arrays_dict, is_ph=self.is_ph)
        return self._apply_error_model(ds)

    def create_dataset_dict(self, label: str | None = None) -> dict[str, Dataset]:
        """Create a dictionary of datasets for all fit_keys, optionally masking outliers.

        Parameters
        ----------
        label : str | None, optional
            Specific label to extract. If None, creates global datasets containing
            all labels. Default is None.

        Returns
        -------
        dict[str, Dataset]
            A dictionary mapping well keys to their corresponding Datasets.
        """
        ds_dict = {}
        for key in sorted(self.fit_keys):
            if label is None:
                ds_dict[key] = self.create_global_ds(key)
            else:
                ds_dict[key] = self.create_ds(key, label)
        if self.params.mask_outliers:
            for key, ds in ds_dict.items():
                ds_dict[key] = apply_outlier_mask(
                    ds, threshold=self.params.outlier_threshold
                )
        return ds_dict

    def plot_temperature(self, title: str = "") -> figure.Figure:
        """Plot temperatures of all labelblocksgroups.

        Creates a line plot showing measured temperatures versus
        concentration/pH values, with statistics overlays.

        Parameters
        ----------
        title : str, optional
            Additional title text to append to the plot.

        Returns
        -------
        figure.Figure
            The matplotlib Figure object containing the plot.
        """
        temperatures: dict[str | int, list[float | int | str | None]] = {}
        for label_n, lbg in self.labelblocksgroups.items():
            temperatures[label_n] = [
                lb.metadata["Temperature"].value for lb in lbg.labelblocks
            ]
        pp = PlotParameters(is_ph=self.is_ph)
        temperatures[pp.kind] = [float(x) for x in self.x.ravel().tolist()]
        data = pd.DataFrame(temperatures)
        data = data.melt(id_vars=pp.kind, var_name="Label", value_name="Temperature")
        g = sns.lineplot(
            data=data,
            x=pp.kind,
            y="Temperature",
            hue="Label",
            palette="Set1",
            alpha=0.75,
            lw=3,
        )
        sns.scatterplot(
            data=data,
            x=pp.kind,
            y="Temperature",
            hue="Label",
            palette="Set1",
            alpha=0.6,
            legend=False,
            s=150,
        )
        ave = data["Temperature"].mean()
        std = data["Temperature"].std()
        lower, upper = ave - std, ave + std
        g.set_ylim(23, 27)
        g.axhline(ave, ls="--", lw=3)
        g.axhline(lower, ls="--", c="grey")
        g.axhline(upper, ls="--", c="grey")
        # Add titles and labels
        plt.title(f"Temperature = {format_estimate(ave, std)} °C {title}", fontsize=14)
        plt.xlabel(f"{pp.kind}", fontsize=14)
        plt.ylabel("Temperature (°C)", fontsize=14)
        plt.grid(visible=True, lw=0.33)
        # Add a legend
        plt.legend(title="Label")
        plt.close()
        return typing.cast("figure.Figure", g.get_figure())


@dataclass
class TecanConfig:
    """Group tecan cli options."""

    out_fp: Path
    comb: bool
    lim: tuple[float, float] | None
    title: str
    fit: bool
    png: bool
    detect_bad: bool = True
    """Run bad-well detection before fitting (pre-fit) and after (post-fit)."""
