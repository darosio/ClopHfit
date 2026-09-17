"""Prtecan/prtecan.py."""

from __future__ import annotations

import functools
import logging
import typing
from dataclasses import InitVar, dataclass, field
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np
import odrpack
import pandas as pd
import seaborn as sns  # type: ignore[import-untyped]
from matplotlib import figure

from clophfit.fitting.bayes import fit_binding_pymc
from clophfit.fitting.bayes_config import RobustConfig
from clophfit.fitting.core import fit_binding_glob
from clophfit.fitting.data_structures import (
    DataArray,
    Dataset,
    FitResult,
    NoiseModelParams,
    PlateNoiseModel,
    ResidualsMixin,
)
from clophfit.fitting.diagnostics import curve_turnover
from clophfit.fitting.errors import InsufficientDataError
from clophfit.fitting.model_validation import (
    RESIDUAL_TABLE_COLUMNS,
    residuals_from_fit_results,
)
from clophfit.fitting.models import binding_1site
from clophfit.fitting.noise_calibration import (
    _noise_params_converged,
    _plate_noise_model_from_nnls,
    compute_plate_slopes,
    fit_noise_model_nnls,
    fit_ph_slope_noise,
)
from clophfit.fitting.odr import fit_binding_odr, format_estimate
from clophfit.fitting.plotting import PlotParameters
from clophfit.fitting.utils import (
    apply_outlier_mask,
)
from clophfit.utils import weights_from_sigma

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterator, Mapping

    from clophfit.clophfit_types import ArrayF
    from clophfit.fitting.bayes_config import SamplerConfig

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


# Gain units per decade of amplification. Measured on this instrument: the
# buffer level tracks the reader Gain at r = 0.991 across eleven plates with a
# decade per 34.1 units, and the label-1 read noise independently follows the
# same law at a decade per 38.3 (r = 0.906). The two agree to 12%, which is why
# a floor quoted at one Gain can be moved to another at all.
_FLOOR_GAIN_DECADE = 34.1


_MIN_TITRATION_POINTS = 2
# A dim channel is exempted from failing when its curve turns over by no more
# than this fraction of its own range - i.e. when it is still monotone within
# noise. Measured on L8: the kept well H11 scores 0.000, the discarded E12 0.777.
_MONOTONE_TURNOVER = 0.2
# ... and it must actually move: the swing of a dim channel has to clear this
# many read-noise widths. L4 G12 swings 1.2 widths and is discarded, while the
# dimmest well kept anywhere swings 2.8.
_MIN_AMPLITUDE_RATIO = 2.0
# "Above background" is the wrong question when the background is 0.3 counts.
# On L5b, 3 x bg_noise for the 485 nm channel is 0.926, so a well reading 1.0
# passes while a healthy well on the same plate gives 97. A label whose swing is
# below this fraction of the plate's median swing is dim whatever the read noise
# says, and is then judged on its curve shape like any other dim label.
_MIN_PLATE_AMPLITUDE_RATIO = 0.03


def label_is_uninformative(  # ruff: ignore[too-many-arguments] - each threshold is an independent criterion
    x: ArrayF,
    y_masked: ArrayF,
    y_raw: ArrayF,
    *,
    floor: float,
    bg_multiplier: float,
    turnover_limit: float | None,
    amplitude: float | None = None,
    plate_amplitude: float | None = None,
    plate_amplitude_ratio: float = _MIN_PLATE_AMPLITUDE_RATIO,
) -> bool:
    """Whether one label of one well carries no usable titration.

    Dimness alone does not condemn a channel. Every well here has a dim 400 nm
    channel by construction, and a 485 nm channel can sit only a few counts
    above background and still trace a clean titration - L8 H11 runs 5.0 to -1.2
    and pins K to 0.193 pH. What makes a dim channel useless is its curve being
    noise-shaped: a single-pKa titration is monotone, so an interior turnover is
    evidence there is no titration to fit.

    Parameters
    ----------
    x : ArrayF
        Titrant axis, matching *y_raw*.
    y_masked : ArrayF
        Signal after outlier masking, used for the brightness test.
    y_raw : ArrayF
        Signal before masking, used for the shape test. The mask exists to
        protect the fit from a bad point, so screening the masked curve would
        delete the very evidence the screen is looking for: masking L8 E12
        removes the spike to 22.8 that proves its channel is noise.
    floor : float
        Read-noise scatter for this label, i.e. ``bg_noise``.
    bg_multiplier : float
        Multiples of *floor* the masked mean must reach.
    turnover_limit : float | None
        Largest turnover a dim channel may show and still count as informative.
        ``None`` disables the exemption, which is right for label 1: its acid
        turnover is a real feature of the fluorophore in roughly 60% of wells,
        so monotonicity says nothing there.
    amplitude : float | None
        This label's own swing, for the plate comparison below.
    plate_amplitude : float | None
        Median swing of this label across the plate. When both are given, a
        label swinging less than *plate_amplitude_ratio* of it counts as dim
        however it compares with the read noise - which is what catches a well
        reading 1.0 count where the plate median is 93.
    plate_amplitude_ratio : float
        Fraction of the plate median below which a label is dim.

    Returns
    -------
    bool
        True when the label carries no usable titration.
    """
    valid = y_masked[~np.isnan(y_masked)]
    if len(valid) < _MIN_TITRATION_POINTS:
        return True
    faint_for_plate = (
        amplitude is not None
        and plate_amplitude is not None
        and plate_amplitude > 0
        and amplitude < plate_amplitude_ratio * plate_amplitude
    )
    if float(np.nanmean(valid)) >= bg_multiplier * floor and not faint_for_plate:
        return False
    if turnover_limit is None:
        return True
    # Monotone is not enough: a drift smaller than the read noise is a random
    # walk, not a titration. L8 H11 swings 6.2 counts against a floor of 1.0 and
    # fits K; L4 G12 swings 3.2 against a floor of 4.0 and does not.
    finite = y_raw[~np.isnan(y_raw)]
    if len(finite) < _MIN_TITRATION_POINTS:
        return True
    amplitude = float(np.nanmax(finite) - np.nanmin(finite))
    if amplitude < _MIN_AMPLITUDE_RATIO * floor:
        return True
    return float(curve_turnover(x, y_raw)) > turnover_limit


def _interaction_rms(values: ArrayF) -> float:
    """Scatter left in a step-by-well matrix once both main effects are removed.

    A two-way layout with one observation per cell: subtracting the step mean
    and the well mean and adding back the grand mean leaves the interaction,
    which for buffer wells is the measurement scatter. A well reading
    consistently high is a well effect, not noise -- a per-well fit absorbs it
    into that well's plateaus -- and a background drifting across the titration
    is a step effect, not noise.

    Parameters
    ----------
    values : ArrayF
        Buffer readings, steps down the rows and wells across the columns.

    Returns
    -------
    float
        Root mean square interaction on ``(n_steps - 1) * (n_wells - 1)``
        degrees of freedom, or 0.0 when either dimension is too small to
        separate an interaction from the main effects.
    """
    arr = np.asarray(values, dtype=float)
    n_steps, n_wells = arr.shape
    if n_steps < 2 or n_wells < 2:  # ruff: ignore[magic-value-comparison]
        return 0.0
    interaction = (
        arr
        - arr.mean(axis=1, keepdims=True)
        - arr.mean(axis=0, keepdims=True)
        + arr.mean()
    )
    dof = (n_steps - 1) * (n_wells - 1)
    return float(np.sqrt(np.sum(interaction**2) / dof))


def _fit_datasets(
    datasets: dict[str, Dataset],
    method: str,
    **kwargs: typing.Any,  # ruff: ignore[any-type]
) -> dict[str, FitResult]:
    """Fit each dataset, turning per-well failures into empty results.

    Parameters
    ----------
    datasets : dict[str, Dataset]
        Mapping of well keys to `Dataset` objects.
    method : str
        'lm', 'huber', 'odr', 'mcmc', or any method accepted by
        :func:`clophfit.fitting.core.fit_binding_glob`.
    **kwargs : typing.Any
        Forwarded to the selected fitting function.

    Returns
    -------
    dict[str, FitResult]
        One entry per input well; failed wells hold an empty `FitResult`.
    """
    fitter: Callable[..., FitResult]
    if method == "odr":
        fitter, fit_kind = fit_binding_odr, "ODR fit"
    elif method == "mcmc":
        fitter, fit_kind = fit_binding_pymc, "MCMC fit"
    else:
        method = method or "lm"
        fitter = functools.partial(fit_binding_glob, method=method)
        fit_kind = "fit"

    results: dict[str, FitResult] = {}
    for well, ds in datasets.items():
        try:
            results[well] = fitter(ds, **kwargs)
        except InsufficientDataError:
            logger.warning("Skip %s for well %s.", fit_kind, well)
            results[well] = FitResult()
    return results


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
    noise_floor: tuple[float, ...] = ()
    """Read-noise floor per label, overriding the measured ``bg_noise``.

    ``bg_noise`` is whatever one plate's three to six buffer wells happened to
    scatter by, and it is estimated on far fewer degrees of freedom than a
    value pooled over a campaign. Supplying the floor lets a calibration fitted
    across plates be used instead.

    Empty tuple keeps ``bg_noise`` (legacy behaviour).
    """
    noise_floor_ref_gain: tuple[float, ...] = ()
    """Reader Gain each ``noise_floor`` was quoted at, per label.

    Read noise is amplified along with the signal, so a floor measured at one
    PMT setting means nothing at another until it is moved there. Given a
    reference, the floor is scaled by ``10 ** ((gain - ref) / 34.1)`` using the
    plate's own Gain metadata, which makes one calibration portable across
    plates that were read at different settings.

    ``0.0`` disables scaling for that label, which is the right choice where no
    Gain dependence was measured. Empty tuple disables it for every label.
    """

    mask_outliers: bool = field(default=False)
    """Mask geometric outliers in each well's curve before fitting. Default is False."""
    outlier_threshold: float = field(default=0.2)
    """Threshold for geometric outlier scoring (0-1). Default is 0.2."""

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
        value: bool | str | float | tuple[float, ...] | None,  # ruff: ignore[boolean-type-hint-positional-argument]
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
        """True when all attributes are NaN, emulating DataFrame's empty behavior."""
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

    @property
    def bg_read_noise(self) -> dict[str, float]:
        """Buffer measurement scatter, with fixed per-well offsets removed.

        ``bg_noise`` pools the scatter of the buffer wells about the background
        trend, and on these plates that pooled number is dominated by fixed
        positional differences between buffer wells rather than by measurement
        noise -- it runs 1.1x to 6.6x this value across the campaign. A
        positional offset is absorbed by a well's own plateaus in any per-well
        fit, so it does not show up as point-to-point residual scatter and does
        not belong in a noise-model floor. This is the part that does.

        Returns
        -------
        dict[str, float]
            Per-label RMS interaction of the buffer readings, 0.0 for a label
            with no buffer wells.
        """
        buffers = self.dataframes_nrm if self.tit.params.nrm else self.dataframes
        noise = {}
        for label, bdf in buffers.items():
            cols = [w for w in self.wells if w in bdf.columns]
            if bdf.empty or not cols:
                noise[label] = 0.0
            else:
                noise[label] = _interaction_rms(bdf[cols].to_numpy(dtype=float))
        return noise

    def _compute_bg_and_sd(self) -> tuple[dict[str, ArrayF], dict[str, ArrayF]]:
        """Compute and return buffer values and their SEM."""
        buffers = self.dataframes_nrm if self.tit.params.nrm else self.dataframes
        bg = {}
        bg_err = {}
        # Mapping methods to column names for clarity and reuse
        # Location estimator per method. `mean`/`median` keep one value per pH
        # point, which matters because the buffer is not flat across a titration
        # - on the real plates it climbs by several standard errors. The `*sd`
        # pair deliberately collapses that to a single pooled value.
        method_map = {
            "fit": ("fit", "fit_err"),
            "mean": ("mean", "sem"),
            "median": ("median", "sem"),
            "meansd": ("mean_all", "sem"),
            "mediansd": ("median_all", "sem"),
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
            if self.tit.params.bg_mth in {"meansd", "mediansd"}:
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
                    np.sqrt(np.sum(diffs**2) / (n_obs - 2)) if n_obs > 2 else 0.0  # ruff: ignore[magic-value-comparison]
                )
                pooled_std = np.sqrt(buf_df.var(axis=1, ddof=1).mean())

                buf_df["Label"] = label
                buf_df["fit"] = y_fit
                buf_df["fit_err"] = fit_error(self.tit.x, cov_matrix)
                buf_df["fit_noise"] = sigma_res
                buf_df["mean"] = mean
                buf_df["median"] = np.nanmedian(y_obs, axis=1)
                # Pooled over every replicate *and* every pH point, so these
                # are one number broadcast across the titration. `meansd` was
                # always meant to be this; it previously pooled only the error
                # and so returned `mean` under another name.
                buf_df["mean_all"] = float(np.nanmean(y_obs))
                buf_df["median_all"] = float(np.nanmedian(y_obs))
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
class TitrationResults(ResidualsMixin):
    """Manage titration results with optional lazy computation.

    Provide either the small ``scheme`` + ``fit_keys`` directly, or a
    ``titration`` keyword to snapshot both from a :class:`Titration`::

        TitrationResults(scheme=tit.scheme, fit_keys=tit.fit_keys, results=res)
        TitrationResults(results=res, titration=tit)  # equivalent, more concise

    ``titration`` is an ``InitVar``: only ``scheme`` and ``fit_keys`` are copied
    off it, so the (potentially large) raw plate data is never retained.

    ``noise_model`` carries the calibrated per-label noise model when the fit
    produced one (``fgls_fit_plate``); it is ``None`` for plain ``fit_plate``.
    """

    scheme: PlateScheme = field(default_factory=PlateScheme)
    fit_keys: set[str] = field(default_factory=set)
    results: dict[str, FitResult] = field(default_factory=dict)
    _dataframe: pd.DataFrame = field(default_factory=pd.DataFrame)
    noise_model: PlateNoiseModel | None = None
    titration: InitVar[Titration | None] = None

    def __post_init__(self, titration: Titration | None) -> None:
        """Snapshot ``scheme`` and ``fit_keys`` from *titration* when given."""
        if titration is not None:
            self.scheme = titration.scheme
            self.fit_keys = titration.fit_keys

    def residual_table(
        self,
        *,
        binding_function: Callable[..., object] | None = None,
        robust: bool | None = None,
        student_t_nu: float | None = None,
        outlier_threshold: float = 3.0,
    ) -> pd.DataFrame:
        """Compute the canonical plate-wide residual table.

        Delegates to each well's own :meth:`FitResult.residual_table`, so
        robustness is auto-detected per well from that well's own ``mini``
        (its lmfit ``Minimizer``, ODR output, or PyMC trace). A plate fitted
        with ``method="mcmc"`` therefore standardizes each well against its
        own trace, instead of being forced to Normal standardization.

        Parameters
        ----------
        binding_function : Callable[..., object] | None
            Model evaluated for ``yhat``; defaults to ``binding_1site``.
            Forwarded unchanged to each well.
        robust : bool | None
            Force the Student-t standardization of ``std_res``. ``None``
            auto-detects per well from that well's trace.
        student_t_nu : float | None
            Student-t degrees of freedom (``None`` uses detected/default).
        outlier_threshold : float
            Threshold for the ``is_residual_outlier`` flag.

        Returns
        -------
        pd.DataFrame
            The canonical residual table (see :attr:`residuals`), built by
            concatenating each well's own table. Wells whose fit failed carry
            no dataset or result and are skipped, so the table may cover
            fewer wells than ``fit_keys``.
        """
        tables = [
            fr.residual_table(
                well=well,
                binding_function=binding_function,
                robust=robust,
                student_t_nu=student_t_nu,
                outlier_threshold=outlier_threshold,
            )
            for well, fr in self.results.items()
            if fr.dataset is not None and fr.result is not None
        ]
        if not tables:
            return pd.DataFrame(columns=RESIDUAL_TABLE_COLUMNS)
        return pd.concat(tables, ignore_index=True)

    @property
    def dataframe(self) -> pd.DataFrame:
        """Convert FitResult dictionary to a DataFrame."""
        if all(key in list(self._dataframe.index) for key in self.fit_keys):
            return self._dataframe

        data = []
        for lbl, fr in self.results.items():
            pars = fr.result.params if fr.result else None
            # Derived from the fit itself rather than plumbed alongside it, so
            # it cannot drift out of step with what was actually fitted.
            row = {"well": lbl, "n_labels": len(fr.dataset) if fr.dataset else 0}
            if pars is not None:
                for k in pars:
                    row[k] = pars[k].value
                    row[f"s{k}"] = pars[k].stderr
                    # `min`/`max` carry the 94% HDI for a sampled fit, but the
                    # solver's bounds for a least-squares one - which exported
                    # K's [3, 11] pH box under an HDI column name. A bound is
                    # not an interval, so write these only when a sampler
                    # actually produced them.
                    lo, hi = pars[k].min, pars[k].max
                    finite = np.isfinite(lo) and np.isfinite(hi)
                    inside = finite and lo <= pars[k].value <= hi
                    from_sampler = inside and fr.trace is not None
                    row[f"{k}hdi03"] = lo if from_sampler else np.nan
                    row[f"{k}hdi97"] = hi if from_sampler else np.nan
            data.append(row)
        self._dataframe = pd.DataFrame(data).set_index("well")
        return self._dataframe

    def __repr__(self) -> str:
        """Get or lazily compute a result for a given key."""
        return repr(self.results)

    def __getitem__(self, key: str) -> FitResult:
        """Fetch result for a single key."""
        return self.results[key]

    def __iter__(self) -> Iterator[str]:
        """Iterate over well keys, like the mapping this wraps.

        Defining this stops Python falling back to the legacy iteration
        protocol, which would call ``__getitem__(0)`` and raise a confusing
        ``KeyError: 0`` on, for example, an attempt to tuple-unpack the result.
        """
        return iter(self.results)

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
        exclude: Mapping[str, Collection[str]] | None = None,
    ) -> figure.Figure:
        """Plot K values as stripplot.

        Wells fitted on fewer labels than the rest are marked with a trailing
        ``*``, and the title says how many there are. Their K rests on 3
        parameters over 7 points rather than 6 over 14 and loses the ratiometric
        cancellation, so it is systematically less certain than its neighbours -
        which a bare stripplot would otherwise hide.

        Parameters
        ----------
        xlim : tuple[float, float] | None, optional
            Range.
        title : str, optional
            To name the plot.
        exclude : Mapping[str, Collection[str]] | None, optional
            Wells to leave off the plot, keyed by the reason (e.g.
            ``"undetermined"``, ``"non-binding"``); the title says how many
            were left off for each.

        Returns
        -------
        figure.Figure
            The figure.
        """
        dataframe = self.dataframe
        # Left off, not drawn wide: one K of 14 +/- 9 sets the automatic
        # x-limits and squeezes every determined well into a sliver.
        omitted = {
            reason: dataframe.index.intersection(list(wells))
            for reason, wells in (exclude or {}).items()
        }
        for dropped in omitted.values():
            dataframe = dataframe.drop(index=dropped, errors="ignore")
        # A fit with no standard error reports sK as None, which matplotlib's
        # errorbar rejects ("'xerr' must not contain None"): one such well on a
        # chloride plate aborted the whole run from this diagnostic plot. As
        # NaN it is simply drawn without an error bar.
        if "sK" in dataframe.columns:
            dataframe = dataframe.assign(
                sK=pd.to_numeric(dataframe["sK"], errors="coerce")
            )
        # A well fitted on fewer labels than its neighbours carries a K that is
        # systematically less certain - 3 parameters over 7 points instead of 6
        # over 14, and no ratiometric cancellation. Mark those so nobody reads
        # them as equivalent points on the same plot.
        partial: set[str] = set()
        if "n_labels" in dataframe.columns:
            full = int(dataframe["n_labels"].max()) if len(dataframe) else 0
            partial = set(dataframe.index[dataframe["n_labels"] < full])
        with sns.plotting_context("paper"):  # axes_style("whitegrid"):
            fig = plt.figure(figsize=(12, 16))
            keys_unk = list(set(dataframe.index))
            # Bound unconditionally: a plate with no control groups still needs
            # x-limits, and this used to raise UnboundLocalError instead.
            df_ctr = dataframe.iloc[0:0]
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
                ax1.set_yticklabels([
                    f"{label} *" if str(label) in partial else str(label)
                    for label in df_ctr.index
                ])
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
            ax2.set_yticklabels([
                f"{label} *" if str(label) in partial else str(label)
                for label in df_unk.index
            ])
            ax2.set_ylim(-1, len(df_unk))
            # Set x-limits
            xlim = xlim or self._determine_xlim(df_ctr, df_unk)
            # With every well left off (a plate where no K is determined) there
            # is nothing to take limits from, and NaN limits abort the run;
            # matplotlib's own then stand.
            if np.isfinite(xlim).all():
                if self.scheme.ctrl:
                    ax1.set_xlim(xlim)
                ax2.set_xlim(xlim)
            # Set title
            note = f"  ({len(partial)} well(s) * = fewer labels)" if partial else ""
            for reason, dropped in omitted.items():
                if len(dropped):
                    note += f"  ({len(dropped)} {reason} well(s) not shown)"
            fig.suptitle(title + note, fontsize=16)
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


@dataclass  # ruff: ignore[too-many-public-methods] (acceptable for a complex class)
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
    # Well -> labels dropped by pre-fit detection. A well can lose one label and
    # keep the other; only losing every label discards the well.
    _excluded_labels: dict[str, set[str]] = field(init=False, default_factory=dict)
    _scheme: PlateScheme = field(init=False, default_factory=PlateScheme)
    _bg: dict[str, ArrayF] = field(init=False, default_factory=dict)
    _bg_err: dict[str, ArrayF] = field(init=False, default_factory=dict)
    _data: dict[str, dict[str, ArrayF]] = field(init=False, default_factory=dict)

    _dil_corr: ArrayF = field(init=False, default_factory=lambda: np.array([]))

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

    def detect_and_discard_bad_wells(  # ruff: ignore[complex-structure, too-many-branches]
        self,
        *,
        outlier_threshold: float | None = 0.2,
        bg_multiplier: float | None = 3.0,
        max_k_stderr: float | None = None,
        monotone_turnover: float | None = _MONOTONE_TURNOVER,
    ) -> list[str]:
        """Detect and discard bad wells from masked per-label signal quality.

        By default, each well is converted to a per-label dataset with
        :meth:`_create_ds`, masked with :func:`apply_outlier_mask`, and then
        discarded when the mean masked signal of any label falls below a
        background-derived floor. The floor uses ``bg_err`` when available and
        falls back to ``bg_noise``.

        The signal test asks two things of a dim label, not one: that it clears
        the background, and failing that, that its curve still moves further
        than the read noise and moves monotonically. The second question is what
        separates a faint titration from a drift - L8 H11 swings six noise
        widths and pins K to 0.193 pH, while L2 C05 swings 1.7 and puts K at
        7.138 +- 254 pH. A fit-quality test still runs alongside, for a well
        that is bright and uninformative, which no signal test can reach.

        The smoothness, roughness and trendline criteria that used to sit here
        were removed. They were disabled by default, no caller ever passed them,
        and at plausible thresholds (0.5/0.5/3.0) they discarded 90 of 90 wells
        on L2 - not a criterion, a bug in waiting.

        Parameters
        ----------
        outlier_threshold : float | None
            Threshold passed to :func:`apply_outlier_mask` before computing
            per-label summary statistics. If ``None``, no masking is applied.
        bg_multiplier : float | None
            Discard a well when any masked per-label mean signal is below
            ``bg_multiplier * mean(background_floor)``. If ``None``, this check
            is disabled.
        max_k_stderr : float | None
            Discard a well whose fitted K has a standard error above this, or a
            non-finite one. ``None`` uses the titration's own x span, which is
            the scale-free form of the rule: a well whose midpoint cannot be
            located inside the window actually titrated carries no information
            about K, however bright it is. Pass ``math.inf`` to disable. pH
            only: a chloride well whose Kd cannot be located may not bind, and
            is fitted and reported as such rather than discarded.
        monotone_turnover : float | None
            Largest turnover a dim label past the first may show and still count
            as informative. ``None`` disables the exemption, restoring the rule
            that any dim label fails.

        Returns
        -------
        list[str]
            Newly discarded well keys.
        """
        first_label = next(iter(self.labelblocksgroups.keys()), None)
        if first_label is None:
            return []

        candidate_wells = sorted(
            self.labelblocksgroups[first_label].data_nrm.keys() - self.scheme.nofit_keys
        )
        if not candidate_wells:
            return []

        label_ids = sorted(self.labelblocksgroups)
        # Median swing per label across the plate, so a well can be judged
        # against its neighbours and not only against the read noise.
        skip_keys = set(self.scheme.buffer) | set(self.scheme.nofit_keys)
        plate_amplitude: dict[typing.Any, float] = {}
        for label in label_ids:
            swings = [
                float(np.nanmax(arr) - np.nanmin(arr))
                for well_key, values in self.data[label].items()
                if well_key not in skip_keys
                and len(arr := np.asarray(values, dtype=float)) > 1
                and np.isfinite(arr).any()
            ]
            if swings:
                plate_amplitude[label] = float(np.median(swings))
        new_discards: set[str] = set()
        # None means the titration's own x span - see the docstring.
        k_stderr_limit = (
            float(np.nanmax(self.x) - np.nanmin(self.x))
            if max_k_stderr is None
            else float(max_k_stderr)
        )
        if not np.isfinite(k_stderr_limit):
            k_stderr_limit = None  # type: ignore[assignment]

        for well in candidate_wells:
            # Signal quality is a property of a label, not of a well. Judge each
            # label on its own and let a well keep the ones that are good: the
            # 400 nm channel is dim by construction, so condemning the well for
            # it would throw away usable titrations.
            failed_labels: set[str] = set()
            for label in label_ids:
                ds = self.create_ds(well, label)
                if outlier_threshold is not None:
                    ds = apply_outlier_mask(ds, threshold=outlier_threshold)
                y = np.asarray(ds[str(label)].y, dtype=float)
                if len(y[~np.isnan(y)]) < _MIN_TITRATION_POINTS:
                    failed_labels.add(str(label))
                    continue
                if bg_multiplier is None:
                    continue
                # The read scatter, not bg_err. With --bg-mth fit, bg_err is the
                # standard error of the fitted background - a precision of the
                # mean - and comparing a well's signal to three of those asks
                # the wrong question. It is also far smaller, which made this
                # rule discard nothing on any of eleven plates.
                floor = self.bg_noise.get(label)
                if floor is None or not np.isfinite(floor):
                    continue
                raw_y = np.asarray(self.data[label].get(well, y), dtype=float)
                own = (
                    float(np.nanmax(raw_y) - np.nanmin(raw_y))
                    if np.isfinite(raw_y).any()
                    else None
                )
                if label_is_uninformative(
                    np.asarray(self.x, dtype=float),
                    y,
                    raw_y,
                    floor=float(floor),
                    bg_multiplier=bg_multiplier,
                    amplitude=own,
                    plate_amplitude=plate_amplitude.get(label),
                    # Label 1 turns over in acid for real, so it gets no
                    # exemption; a dim label 2 that is still monotone does.
                    turnover_limit=(
                        None if str(label) == str(label_ids[0]) else monotone_turnover
                    ),
                ):
                    failed_labels.add(str(label))

            discard_well = failed_labels >= {str(i) for i in label_ids}
            if not discard_well and failed_labels:
                for label in sorted(failed_labels):
                    self.exclude_label(well, label)

            # pH only. A well whose Kd cannot be located may simply not bind,
            # which is a result: it is fitted and reported as "does not bind"
            # (see export.no_binding_k), not discarded - V224Q lost all its
            # wells on four chloride plates to this rule.
            if self.is_ph and not discard_well and k_stderr_limit is not None:
                discard_well = self._k_is_unconstrained(well, k_stderr_limit)

            if discard_well:
                new_discards.add(well)

        if new_discards:
            self.scheme.discard = list(set(self.scheme.discard) | new_discards)
            self.clear_all_data_results()

        return sorted(new_discards)

    def _k_is_unconstrained(self, well: str, limit: float) -> bool:
        """Whether a cheap global fit leaves K less certain than the x span.

        Parameters
        ----------
        well : str
            Well key to fit.
        limit : float
            Largest acceptable standard error on K.

        Returns
        -------
        bool
            True when K cannot be pinned - a non-finite standard error, a fit
            that fails outright, or one wider than ``limit``. A fit that will
            not converge is itself evidence the well carries no K.
        """
        try:
            res = fit_binding_glob(self.create_global_ds(well))
        except Exception:  # ruff: ignore[blind-except] - a well that cannot be fitted is a bad well
            return True
        params = getattr(getattr(res, "result", None), "params", None)
        if params is None or "K" not in params:
            return True
        stderr = params["K"].stderr
        return stderr is None or not np.isfinite(stderr) or float(stderr) > limit

    def _reset_data_and_results(self) -> None:
        """Discard derived data so the next access recomputes it."""
        self._data = {}

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
        """The datafit parameters."""
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

    @property
    def bg_read_noise(self) -> dict[str, float]:
        """Buffer measurement scatter, with fixed per-well offsets removed.

        See :attr:`Buffer.bg_read_noise`. Reported alongside :attr:`bg_noise`
        rather than replacing it: ``bg_noise`` is what ``y_err`` and the
        classical ``--plate-noise fixed`` path are built on and calibrated
        against, while this is the candidate for a structured-noise floor.

        Returns
        -------
        dict[str, float]
            Per-label measurement scatter of the buffer wells.
        """
        return self.buffer.bg_read_noise

    @property
    def sigma_floor(self) -> dict[str, float]:
        """The read-noise floor per label that the error models actually use.

        ``params.noise_floor`` when supplied, otherwise the measured
        :attr:`bg_read_noise`. A supplied floor carrying a non-zero
        ``params.noise_floor_ref_gain`` is scaled from that reference to this
        plate's own reader Gain, so one campaign-wide calibration serves plates
        read at different settings; a reference of ``0.0`` leaves the value
        alone, which is right for a label whose floor showed no Gain
        dependence.

        Returns
        -------
        dict[str, float]
            Per-label floor. Falls back to the measured value for any label the
            override does not cover.
        """
        labels = sorted(self.data.keys())
        measured = self.bg_read_noise
        floors: dict[str, float] = {}
        for i, lbl in enumerate(labels):
            key = str(lbl)
            if not self.params.noise_floor or i >= len(self.params.noise_floor):
                floors[key] = float(measured.get(key, 0.0))
                continue
            value = float(self.params.noise_floor[i])
            ref = (
                float(self.params.noise_floor_ref_gain[i])
                if self.params.noise_floor_ref_gain
                and i < len(self.params.noise_floor_ref_gain)
                else 0.0
            )
            if ref > 0.0:
                meta = getattr(self.labelblocksgroups.get(key), "metadata", {}) or {}
                gain = getattr(meta.get("Gain"), "value", None)
                if gain is not None:
                    value *= 10.0 ** ((float(gain) - ref) / _FLOOR_GAIN_DECADE)
            floors[key] = value
        return floors

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
    def fromlistfile(
        cls, list_file: Path | str, *, is_ph: bool, base_dir: Path | str | None = None
    ) -> Titration:
        """Build `Titration` from a list[.pH|.Cl] file.

        Parameters
        ----------
        list_file : Path | str
            Path to the list file containing [filenames x x_err].
        is_ph : bool
            Whether x values represent pH (True) or concentrations (False).
        base_dir : Path | str | None
            Directory holding the Tecan files. Relative filenames in the list
            file are resolved against it; defaults to the list file's own
            directory. Use it when list files and `.xls` files are kept in
            separate trees (e.g. `data/processed/` and `data/raw/`).

        Returns
        -------
        Titration
            The constructed Titration object.
        """
        tecanfiles, x, x_err = cls._listfile(
            Path(list_file), None if base_dir is None else Path(base_dir)
        )
        return cls(tecanfiles, x, is_ph, x_err=x_err)

    @staticmethod
    def _listfile(
        listfile: Path, base_dir: Path | None = None
    ) -> tuple[list[Tecanfile], ArrayF, ArrayF]:
        """Help construction from list file."""
        try:
            # Separator sniffed rather than assumed: the pH lists are csv with
            # three columns, the chloride ones tab-separated with two, and a
            # hardcoded comma silently folds filename and value into one field.
            table = pd.read_csv(
                listfile, names=["filenames", "x", "x_err"], sep=None, engine="python"
            )
        except FileNotFoundError as exc:
            msg = f"Cannot find: {listfile}"
            raise FileNotFoundError(msg) from exc
        # For unexpected file format, e.g. length of filename column differs
        # from length of x values.
        if table["filenames"].count() != table["x"].count():
            msg = f"Check format [filenames x x_err] for listfile: {listfile}"
            raise ValueError(msg)
        root = listfile.parent if base_dir is None else base_dir
        tecanfiles = [Tecanfile(root / f) for f in table["filenames"]]
        x = table["x"].to_numpy().astype(float)
        # A two-column list carries no uncertainty; that is zero, not unknown.
        x_err = table["x_err"].fillna(0.0).to_numpy().astype(float)
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
            """Lift a trace that dips below zero (alpha = F_bound/F_unbound).

            The test is simply whether the trace goes negative. It was written
            ``y.min() < alpha * 0 * y.max()``, which is the same thing with a
            multiplication by zero in the middle, so ``alpha`` looked like it
            set the threshold while only ever sizing the shift. Written plainly
            here; the behaviour is unchanged, and changing it to the
            ``alpha * y.max()`` the old expression resembles would adjust far
            more wells and move every existing ``--bg-adj`` result.
            """
            if y.min() < 0:
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
        labels = sorted(self.data.keys())
        noise_model = PlateNoiseModel()
        # An explicit --noise-floor governs here too, but the *default* stays
        # bg_noise rather than the read noise the noise-model paths use. With
        # no gain/alpha this y_err is homoscedastic, and the best single sigma
        # for that is the typical noise over the signal range, not the floor at
        # zero signal: bg_noise sits 4x below it on label 1 where bg_read_noise
        # sits 14x below. Least squares does not care -- a uniform scale
        # cancels -- but huber's transition point is absolute, so the smaller
        # value down-weights harder, and switching this default to read noise
        # left 29 of 731 well-fits unconstrained that had been fine.
        resolved = self.sigma_floor if self.params.noise_floor else self.bg_noise
        for i, lbl in enumerate(labels):
            floor = float(resolved.get(str(lbl), 0.0))

            gain = (
                self.params.noise_gain[i]
                if self.params.noise_gain and i < len(self.params.noise_gain)
                else 0.0
            )
            alpha = (
                self.params.noise_alpha[i]
                if self.params.noise_alpha and i < len(self.params.noise_alpha)
                else 0.0
            )

            noise_model[lbl] = NoiseModelParams(
                sigma_floor=floor, gain=gain, alpha=alpha
            )

        return noise_model.apply_to(ds)

    def create_ds(self, key: str, label: str) -> Dataset:
        """Create a dataset for the given key."""
        da = self._create_data_array(key, label)
        ds = Dataset({label: da}, is_ph=self.is_ph)
        return self._apply_error_model(ds)

    @property
    def excluded_labels(self) -> dict[str, set[str]]:
        """Labels dropped per well, keyed by well.

        Returns
        -------
        dict[str, set[str]]
            Well to the set of its excluded labels. Wells with every label
            excluded are discarded outright instead of appearing here.
        """
        return self._excluded_labels

    def exclude_label(self, key: str, label: str) -> None:
        """Drop one label of one well from the global fit.

        The 400 nm channel is dim by construction, so a well can fail a
        background test on one label while the other is perfectly usable.
        Discarding the well would throw away a good titration; dropping the
        label keeps it, at the cost of a less certain K - which is why the
        results table records how many labels each well was fitted on.

        Parameters
        ----------
        key : str
            Well identifier.
        label : str
            Label to drop for that well. Dropping every label discards the well.
        """
        self._excluded_labels.setdefault(key, set()).add(str(label))
        if self._excluded_labels[key] >= {str(i) for i in self.data}:
            self.scheme.discard = list({*self.scheme.discard, key})
        self.clear_all_data_results()

    def create_global_ds(self, key: str) -> Dataset:
        """Create a global dataset for the given key.

        Applies ``mask_outliers`` like :meth:`create_dataset_dict`. It used not
        to, and since ``ppr tecan`` builds its global datasets here, the flag
        never reached the global, ODR, MCMC or plate fits - accepted, echoed
        back in the run configuration, and silently confined to the per-label
        path. Two builders of the same dataset must not disagree about what a
        flag means.

        Parameters
        ----------
        key : str
            Well identifier.

        Returns
        -------
        Dataset
            The well's labels, masked and weighted as configured.
        """
        dropped = self._excluded_labels.get(key, set())
        data_arrays_dict = {
            i: self._create_data_array(key, i)
            for i in self.data
            if str(i) not in dropped
        }
        ds = Dataset(data_arrays_dict, is_ph=self.is_ph)
        if self.params.mask_outliers:
            ds = apply_outlier_mask(ds, threshold=self.params.outlier_threshold)
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
        if self.params.mask_outliers and label is not None:
            # The global branch masks inside create_global_ds; only the
            # per-label datasets still need it here.
            for key, ds in ds_dict.items():
                ds_dict[key] = apply_outlier_mask(
                    ds, threshold=self.params.outlier_threshold
                )
        return ds_dict

    def fit_plate(
        self,
        datasets: dict[str, Dataset] | None = None,
        method: str = "",
        *,
        label: str | None = None,
        **kwargs: typing.Any,  # ruff: ignore[any-type]
    ) -> TitrationResults:
        """Run a single-pass fit on an entire plate of datasets.

        Parameters
        ----------
        datasets : dict[str, Dataset] | None
            Mapping of well keys (e.g. 'A01') to `Dataset` objects. When
            ``None``, datasets are built with :meth:`create_dataset_dict`,
            which also applies outlier masking when ``params.mask_outliers``
            is set.
        method : str
            The fitting method: 'lm' (default), 'huber', 'odr', or 'mcmc'.
            Other methods supported by
            :func:`clophfit.fitting.core.fit_binding_glob` may also be used.
        label : str | None
            Build per-label datasets for this label instead of global ones.
            Only valid when *datasets* is ``None``.
        **kwargs : typing.Any
            Additional keyword arguments passed to the fitting function.

        Returns
        -------
        TitrationResults
            Plate results carrying this titration's ``scheme`` and ``fit_keys``.

        Raises
        ------
        ValueError
            If both *datasets* and *label* are given.
        """
        if datasets is not None and label is not None:
            msg = "Pass either `datasets` or `label`, not both."
            raise ValueError(msg)
        if datasets is None:
            datasets = self.create_dataset_dict(label)
        results = _fit_datasets(datasets, method, **kwargs)
        return TitrationResults(self.scheme, self.fit_keys, results)

    def fgls_fit_plate(  # ruff: ignore[too-many-arguments]
        self,
        datasets: dict[str, Dataset] | None = None,
        *,
        label: str | None = None,
        sigma_floor: dict[str, float] | None = None,
        first_pass_method: str = "huber",  # ruff: ignore[hardcoded-password-default]
        second_pass_method: str = "lm",  # ruff: ignore[hardcoded-password-default]
        max_iter: int = 3,
        tol: float = 1e-3,
    ) -> TitrationResults:
        """Run iterative Feasible Generalized Least Squares (FGLS) on the plate.

        Fits every well with *first_pass_method* using the existing ``y_errc``,
        calibrates a per-label noise model from the plate-wide residuals with
        the floor anchored to *sigma_floor*, re-applies the calibrated weights
        and re-fits with *second_pass_method*, iterating until gain and alpha
        converge or *max_iter* is reached.

        Parameters
        ----------
        datasets : dict[str, Dataset] | None
            Mapping of well keys to `Dataset` objects. When ``None``, datasets
            are built with :meth:`create_dataset_dict`.
        label : str | None
            Build per-label datasets for this label instead of global ones.
            Only valid when *datasets* is ``None``.
        sigma_floor : dict[str, float] | None
            Known read-noise floor per label. Defaults to :attr:`sigma_floor`.
        first_pass_method : str
            Method for the first-pass fit.
        second_pass_method : str
            Method for subsequent passes.
        max_iter : int
            Maximum FGLS iterations.
        tol : float
            Relative tolerance for gain/alpha convergence.

        Returns
        -------
        TitrationResults
            Plate results carrying this titration's ``scheme`` and
            ``fit_keys``, with ``noise_model`` set to the converged (or last)
            calibration.

        Raises
        ------
        ValueError
            If both *datasets* and *label* are given.
        """
        if datasets is not None and label is not None:
            msg = "Pass either `datasets` or `label`, not both."
            raise ValueError(msg)
        if datasets is None:
            datasets = self.create_dataset_dict(label)
        floors_in = dict(self.sigma_floor) if sigma_floor is None else dict(sigma_floor)

        noise_model: PlateNoiseModel | None = None
        results: dict[str, FitResult] = {}

        for iteration in range(max_iter):
            method = first_pass_method if iteration == 0 else second_pass_method
            if iteration == 0:
                current_ds = datasets
            else:
                current_ds = noise_model.apply_to_plate(  # type: ignore[union-attr]
                    datasets, compute_plate_slopes(results)
                )
            logger.info("FGLS iteration %d: %s fit", iteration + 1, method)

            results = {}
            for well, ds in current_ds.items():
                try:
                    results[well] = fit_binding_glob(ds, method=method)
                except InsufficientDataError:
                    logger.warning(
                        "Skip FGLS fit for well %s (iteration %d).",
                        well,
                        iteration + 1,
                    )
                    results[well] = FitResult()

            df_res = residuals_from_fit_results(
                results, trace_id="", binding_function=binding_1site
            )
            try:
                floors, gains, alphas = fit_noise_model_nnls(
                    df_res, sigma_floor_fixed=floors_in
                )
            except ValueError as e:
                logger.warning("FGLS calibration failed (%s).", e)
                gains = dict.fromkeys(floors_in, 0.0)
                alphas = dict.fromkeys(floors_in, 0.0)
                floors = dict(floors_in)

            plate_slopes = compute_plate_slopes(results)
            tmp_noise = _plate_noise_model_from_nnls(floors, gains, alphas)
            sigma_ph = fit_ph_slope_noise(df_res, tmp_noise, plate_slopes)
            new_noise = _plate_noise_model_from_nnls(floors, gains, alphas, sigma_ph)

            for lbl, params in new_noise.items():
                logger.info(
                    "Calibrated [%s] iter %d: sigma=%.2f, gain=%.3f, alpha=%.3f, "
                    "sigma_ph=%.4f",
                    lbl,
                    iteration + 1,
                    params.sigma_floor,
                    params.gain,
                    params.alpha,
                    params.sigma_ph,
                )

            converged = iteration > 0 and _noise_params_converged(
                noise_model,  # type: ignore[arg-type]
                new_noise,
                tol,
            )
            noise_model = new_noise
            if converged:
                logger.info("FGLS converged after %d iterations.", iteration + 1)
                break

        return TitrationResults(
            self.scheme, self.fit_keys, results, noise_model=noise_model
        )

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
    """Run bad-well detection before fitting, and act on undetermined K after it.

    Before: discard unusable wells (``Titration.detect_and_discard_bad_wells``).
    After: leave wells whose fitted K is undetermined (see ``max_k_se``) - and,
    for chloride, wells that do not bind - off the K plot, mark their per-well
    figure, and list them in ``discarded_wells.txt``. The ``undetermined`` (and
    ``no_binding``) columns of each ``ffit*.csv`` are written either way.
    """

    ctr_free_k: bool = False
    """Give every well its own K in --plate-fit, instead of pooling controls.

    Mirrors what the flag already did for ``--mcmc multi``; without it the two
    fitters answer different questions from the same command line.
    """

    plate_screen_z: float | None = None
    """Drop points whose calibrated |z| exceeds this, then refit, for --plate-fit.

    ``None`` fits once. 3.0 is the value measured to help (sum_log -5.92 against
    -5.49 unscreened, nine plates of eleven improved); 2.5 turns harmful, so
    this is not a knob to sweep casually.
    """

    plate_screen_frac: float | None = None
    """Also drop 400 nm points deviating fractionally by more than this.

    A z-score fails at both ends of a titration: sigma tracks the signal while
    model error tracks the curve, so a 5% miss at the dim end reads as 3.6 sigma
    while a 36% miss at the bright end reads as 2.7. ``|y - yhat| / yhat``
    separates reviewer keeps (0.097-0.111) from discards (0.126 up) over three
    plates. Points the 485 nm channel moves with are spared, since a shared
    multiplicative shift is what the ratiometric measurement cancels. ``None``
    leaves the z-screen alone.
    """

    plate_noise: str = "fixed"
    """How ``--plate-fit`` weights its points.

    ``"fixed"`` uses ``y_err`` as built - the ``bg_noise`` floor, plus any
    ``--noise-gain``/``--noise-alpha`` supplied. ``"calibrated"`` estimates gain
    and alpha per label from the fit's own residuals and refits under them.
    Calibration describes the residuals better and fits K worse, so it is not
    the default; see ``fit_plate_lm``. It sets the weights K is fitted with
    whether or not ``plate_screen_z`` is given. ``"gain"`` and ``"floor-gain"``
    weight by ``floor^2 + gain * yhat`` with alpha 0, the gain (and the floor)
    calibrated from dof-corrected residuals between refits; see
    :func:`~clophfit.fitting.gain_calibration.calibrate_plate_lm`.
    """

    plate_screen_noise: str = "calibrated"
    """The ruler ``--plate-screen-z`` judges points on.

    ``"calibrated"`` (the default) calibrates gain and alpha from the screening
    pass's own residuals so bright and dim points are judged on a comparable
    scale; ``"fixed"`` judges on the weights as built. Independent of
    ``plate_noise``: before the two were separate options, ``plate_noise`` meant
    the fit's weights without a screen but not after one.
    """

    max_k_se: float = 0.30
    """Largest standard error, in pH, at which a fitted pKa counts as determined.

    pH is an interval scale, so the cut is absolute; a ratio ``sK / K`` is
    ``sK / 7`` here and never fires. 0.30 is three times the 0.10 pH ROPE and
    four times the 0.074 pH replicate repeatability; on eleven library plates it
    catches 32 of 875 library wells and none of 123 controls. Chloride ignores
    it: Kd is a ratio scale, undetermined when ``sKd > Kd``, and a well whose
    94% lower bound is above the highest concentration does not bind.
    """

    fit_noise: str = "fixed"
    """How the per-well global lm/huber fit weights its points.

    ``"fixed"`` uses ``y_err`` as built. ``"gain"`` and ``"floor-gain"`` fit
    every well, pool all wells' dof-corrected residuals into one gain per label
    (and with ``"floor-gain"`` one floor), and refit until it settles; see
    :func:`~clophfit.fitting.gain_calibration.calibrate_single_well`.
    """


@dataclass(frozen=True)
class McmcSpec:
    """A per-well MCMC request, decided by the caller rather than parsed downstream.

    Parameters
    ----------
    model : Literal["single", "single-refit", "multi"]
        Which fit to run. ``"single"`` samples each well once;
        ``"single-refit"`` runs the robust screening pass then refits;
        ``"multi"`` fits every well jointly with control K shared across each
        control group.
    sampler : SamplerConfig
        NUTS controls forwarded to ``pm.sample``.
    structured_noise : bool
        Build the physical ``floor + gain * y + (alpha * y) ** 2`` observation
        noise instead of scaling ``y_err`` by a learned ``ye_mag`` multiplier.
    floor_mode : Literal["centered", "fixed"] | None
        Override the mode for the floor alone; ``None`` follows *noise_mode*.
    gain_mode : Literal["centered", "fixed"] | None
        Override the mode for gain alone; ``None`` follows *noise_mode*.
    alpha_mode : Literal["centered", "fixed"] | None
        Override the mode for alpha alone; ``None`` follows *noise_mode*.
    noise_mode : Literal["centered", "fixed"]
        How a supplied gain/alpha hint is treated when *structured_noise* is
        set: centred on (a hint the posterior may leave) or pinned to it. A
        parameter with no supplied value is always free.
    per_well_ye_mags : bool | None
        Whether the ye_mag multiplier is per well rather than per label.
        ``None`` leaves the library to resolve it from the noise family, which
        couples the two; pass a bool to keep them independent. Only meaningful
        for ``model="multi"``, since a single-well fit has one well.
    ye_mag_parameterization : Literal["centered", "hierarchical", "separable", "separable_step"]
        How per-well ye_mags are structured: independent per label
        (``"centered"``), a shared well factor with per-label deviations
        (``"hierarchical"``), or a per-label level plus one shared well factor
        (``"separable"``).
    robust : RobustConfig
        Likelihood family. The default is a plain Normal; a Student-t with
        ``nu=3`` is the arm that scored best on this campaign's plates.
    ctr_free_k : bool
        Give every well its own K instead of pooling each control group onto a
        shared one. Only meaningful for ``model="multi"``. Pooling buys no
        accuracy at the construct level and makes the stated interval too
        narrow, and the library wells have no group to pool with, so free K is
        the setting that matches what a plate is fitted for.
    x_error_model : Literal["deterministic", "per_well"]
        The latent pH axis of ``model="multi"``. ``"deterministic"`` is one
        pipetting walk shared by every well; ``"per_well"`` gives each well its
        own walk, with step SDs split from the measured pH errors into read
        noise plus accumulated pipetting variance. pH is measured in a few wells
        and their spread grows along a titration, so an unmeasured well's pH is
        uncertain in a way only the per-well axis carries into its K: on eleven
        plates it took free-K single-well bulk z-SD from ~1.65 to ~0.95 at
        unchanged accuracy. The default keeps the historical shared axis.
    x_start_between_sigma : float | None
        For ``x_error_model="per_well"``, the prior SD of each well's offset at
        the first step. A per-well offset is degenerate with that well's K, so
        this SD passes straight into K's interval; set it to the measured
        well-to-well spread at the first step rather than a round number.
        ``None`` keeps the library default.
    """

    model: Literal["single", "single-refit", "multi"]
    sampler: SamplerConfig
    structured_noise: bool = False
    noise_mode: Literal["centered", "fixed"] = "centered"
    # Per-term overrides. One mode for all three cannot separate the terms:
    # pinning alpha at zero also pins the floor, so sigma cannot rescale and the
    # cell measures that rather than the term it meant to isolate.
    floor_mode: Literal["centered", "fixed"] | None = None
    gain_mode: Literal["centered", "fixed"] | None = None
    alpha_mode: Literal["centered", "fixed"] | None = None
    per_well_ye_mags: bool | None = None
    noise_ye_mag: bool = False
    """Learn a ye_mag multiplier on top of a structured noise model.

    A structured model builds sigma from floor, gain and alpha and, by default,
    has no overall multiplier: the terms themselves carry the level. With this
    set, sigma is also scaled by a learned ye_mag - per label, or per well when
    ``per_well_ye_mags`` is set - which lets a calibrated shape keep its shape
    while the data set the level. Meaningless without ``structured_noise``.
    """
    ye_mag_parameterization: Literal[
        "centered", "hierarchical", "separable", "separable_step"
    ] = "centered"
    robust: RobustConfig = field(default_factory=RobustConfig)
    ctr_free_k: bool = False
    x_error_model: Literal["deterministic", "per_well"] = "deterministic"
    x_start_between_sigma: float | None = None
