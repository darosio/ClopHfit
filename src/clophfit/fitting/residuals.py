"""Residual extraction and analysis utilities for fit results.

This module provides tools to extract, analyze, and validate residuals from
fitting procedures. Useful for diagnostics, model validation, and comparing
different fitting methods.
"""

from __future__ import annotations

import typing
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from clophfit.clophfit_types import ArrayF
    from clophfit.fitting.data_structures import FitResult

# Statistical thresholds
OUTLIER_THRESHOLD_2SIGMA = 2.0
OUTLIER_THRESHOLD_3SIGMA = 3.0
OUTLIER_RATE_THRESHOLD = 0.05
BIAS_P_VALUE_THRESHOLD = 0.01
# Robust (modified) z-score cutoff, Iglewicz-Hoaglin convention. A threshold on
# the residuals' own median/MAD scale, so a few points cannot mask themselves by
# inflating the fitted noise scale (as a std-based flag on std_res can).
ROBUST_Z_THRESHOLD = 3.5
DW_LOWER_BOUND = 1.5
DW_UPPER_BOUND = 2.5

MIN_POINTS_FOR_TREND = 3  # Minimum points needed to detect trends
MIN_POINTS_FOR_BIN = 2  # Minimum points needed in a bin to include it
MIN_POINTS_FOR_BINNING = 10  # Minimum total points needed before binning
MAX_BINS = 15  # Maximum number of bins for binned diagnostics


@dataclass(frozen=True)
class ResidualPoint:
    """Single residual data point with metadata.

    Attributes
    ----------
    label : str
        Dataset label (e.g., '1', '2' for multi-label fits)
    x : float
        X-value (pH or ligand concentration)
    y : float
        Observed signal value.
    yhat : float
        Model-predicted signal value (``y - raw_res``).
    sigma : float
        Measurement uncertainty used during fitting.
    raw_res : float
        Raw residual: ``y - yhat``.
    likelihood_res : float
        Likelihood-scale residual: ``(y - yhat) / sigma``.
    std_res : float
        Normal-scale standardized residual (equal to *likelihood_res* for a
        Normal likelihood, as produced by LMFit/ODR fits).
    raw_i : int
        Index into the original (unmasked) arrays for this label (`DataArray.xc/yc`).

    Notes
    -----
    Field names follow the canonical residual-table schema shared with
    :data:`clophfit.fitting.model_validation.RESIDUAL_TABLE_COLUMNS`.
    """

    label: str
    x: float
    y: float
    yhat: float
    sigma: float
    raw_res: float
    likelihood_res: float
    std_res: float
    raw_i: int


def extract_residual_points(fr: FitResult) -> list[ResidualPoint]:
    """Extract residual points from a fit result.

    Parameters
    ----------
    fr : FitResult
        Fit result containing residuals and dataset

    Returns
    -------
    list[ResidualPoint]
        List of residual points with metadata for each observation

    Raises
    ------
    ValueError
        If residual length doesn't match dataset sizes

    Examples
    --------
    >>> from clophfit.fitting.core import fit_binding_glob
    >>> from clophfit.fitting.data_structures import Dataset, DataArray
    >>> import numpy as np
    >>> # Create test data
    >>> x = np.array([9.0, 8.0, 7.0, 6.0, 5.0])
    >>> y = 500 + 500 * 10 ** (7.0 - x) / (1 + 10 ** (7.0 - x))
    >>> da = DataArray(xc=x, yc=y, y_errc=np.ones_like(y) * 10)
    >>> dataset = Dataset({"1": da}, is_ph=True)
    >>> fr = fit_binding_glob(dataset)
    >>> residuals = extract_residual_points(fr)
    >>> len(residuals) > 0
    True
    >>> residuals[0].label
    '1'
    """
    if fr.result is None or fr.dataset is None:
        return []

    r = np.asarray(fr.result.residual, dtype=float)
    pts: list[ResidualPoint] = []
    masked_total = sum(len(da.y) for da in fr.dataset.values())
    raw_total = sum(len(da.yc) for da in fr.dataset.values())
    residuals_are_raw = len(r) == raw_total and len(r) != masked_total

    start = 0
    for lbl, da in fr.dataset.items():
        mask = np.asarray(da.mask, dtype=bool)
        if residuals_are_raw:
            n_raw = len(da.yc)
            rw_full = r[start : start + n_raw]
            rw = rw_full[mask]
            y_err = da.y_err
            xs = da.xc[mask]
            ys = da.yc[mask]
            raw_is = np.flatnonzero(mask)
            start += n_raw
        else:
            n = len(da.y)  # masked length
            rw = r[start : start + n]
            y_err = da.y_err
            xs = da.x
            ys = da.y
            raw_is = np.flatnonzero(mask)
            start += n

        rr = rw * y_err  # undo weighting: raw = weighted * y_err

        pts.extend(
            ResidualPoint(
                label=lbl,
                x=float(xs[i]),
                y=float(ys[i]),
                yhat=float(ys[i]) - float(rr[i]),
                sigma=float(y_err[i]),
                raw_res=float(rr[i]),
                likelihood_res=float(rw[i]),
                std_res=float(rw[i]),
                raw_i=int(raw_is[i]),
            )
            for i in range(len(rw))
        )

    if start != len(r):
        msg = f"Residual length mismatch: consumed {start}, residual has {len(r)}"
        raise ValueError(msg)

    return pts


def residual_dataframe(fr: FitResult) -> pd.DataFrame:
    """Convert fit result residuals to a DataFrame.

    Parameters
    ----------
    fr : FitResult
        Fit result to extract residuals from

    Returns
    -------
    pd.DataFrame
        DataFrame with the canonical residual columns: label, x, y, yhat,
        sigma, raw_res, likelihood_res, std_res, raw_i.

    Examples
    --------
    >>> from clophfit.fitting.core import fit_binding_glob
    >>> from clophfit.fitting.data_structures import Dataset, DataArray
    >>> import numpy as np
    >>> x = np.array([9.0, 8.0, 7.0, 6.0, 5.0])
    >>> y = 500 + 500 * 10 ** (7.0 - x) / (1 + 10 ** (7.0 - x))
    >>> da = DataArray(xc=x, yc=y, y_errc=np.ones_like(y) * 10)
    >>> dataset = Dataset({"1": da}, is_ph=True)
    >>> fr = fit_binding_glob(dataset)
    >>> df = residual_dataframe(fr)
    >>> "label" in df.columns and "x" in df.columns
    True
    """
    return pd.DataFrame([asdict(p) for p in extract_residual_points(fr)])


def residual_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute residual statistics by label.

    Parameters
    ----------
    df : pd.DataFrame
        Residual DataFrame (from residual_dataframe or residuals_from_fit_results)

    Returns
    -------
    pd.DataFrame
        Statistics by label: mean, std, median, mad, outlier_count,
        robust_outlier_count, n_points, outlier_rate, robust_outlier_rate.

        ``outlier_count`` thresholds the model-standardized ``std_res`` (``> 2``);
        a few points can hide by inflating the fitted scale.
        ``robust_outlier_count`` uses each label's own median/MAD scale
        (modified z-score ``> ROBUST_Z_THRESHOLD``), so masked points still
        surface.

    Examples
    --------
    >>> from clophfit.fitting.core import fit_binding_glob
    >>> from clophfit.fitting.data_structures import Dataset, DataArray
    >>> import numpy as np
    >>> x = np.array([9.0, 8.0, 7.0, 6.0, 5.0])
    >>> y = 500 + 500 * 10 ** (7.0 - x) / (1 + 10 ** (7.0 - x))
    >>> da = DataArray(xc=x, yc=y, y_errc=np.ones_like(y) * 10)
    >>> dataset = Dataset({"1": da}, is_ph=True)
    >>> fr = fit_binding_glob(dataset)
    >>> all_res = residual_dataframe(fr)
    >>> stats = residual_statistics(all_res)
    >>> "mean" in stats.columns
    True
    """

    def outlier_count(x: ArrayF) -> int:
        """Count points beyond ±2-sigma of the model-standardized residual."""
        return int((np.abs(x) > OUTLIER_THRESHOLD_2SIGMA).sum())

    def robust_outlier_count(x: ArrayF) -> int:
        """Count points beyond the robust (median/MAD) z-score cutoff.

        The modified z-score ``(x - median) / (1.4826 * MAD)`` uses the group's
        own scale, so points that inflate the fitted noise (and thus shrink
        ``std_res``) cannot mask themselves the way ``outlier_count`` allows.
        """
        arr = np.asarray(x, dtype=float)
        med = float(np.nanmedian(arr))
        robust_sigma = float(
            sp_stats.median_abs_deviation(arr, nan_policy="omit", scale="normal")
        )
        if not np.isfinite(robust_sigma) or robust_sigma <= 0.0:
            return 0
        z = (arr - med) / robust_sigma
        return int(np.nansum(np.abs(z) > ROBUST_Z_THRESHOLD))

    # Use dict format for agg to avoid mypy issues with tuple format
    summary = df.groupby("label")["std_res"].agg(
        mean="mean",
        std="std",
        median="median",
        mad=lambda x: sp_stats.median_abs_deviation(x, nan_policy="omit"),
        outlier_count=outlier_count,
        robust_outlier_count=robust_outlier_count,
        n_points="count",
    )

    # Convert to DataFrame (agg returns DataFrame when grouping by one column)
    summary_df = typing.cast("pd.DataFrame", summary)
    summary_df["outlier_rate"] = summary_df["outlier_count"] / summary_df["n_points"]
    summary_df["robust_outlier_rate"] = (
        summary_df["robust_outlier_count"] / summary_df["n_points"]
    )

    return summary_df


def detect_adjacent_correlation(
    all_res: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, ArrayF]]:
    """Detect correlation between adjacent (lag-1) residuals within wells.

    Tests whether adjacent points show systematic patterns (e.g. positive then
    negative), which can indicate x-value errors or model misspecification.

    Parameters
    ----------
    all_res : pd.DataFrame
        Residual table with ``label``, ``well``, ``x`` and ``std_res`` columns.

    Returns
    -------
    correlation_stats : pd.DataFrame
        Lag-1 correlation statistics per (label, well).
    correlations_by_label : dict[str, ArrayF]
        Array of lag-1 correlations for each label.
    """
    correlations = []
    correlations_by_label: dict[str, list[float]] = {}

    for (lbl, well), group in all_res.groupby(["label", "well"]):
        # Sort by x so adjacent points are sequential.
        g = group.sort_values("x")
        res = g["std_res"].to_numpy(dtype=float)

        if len(res) > 1:
            corr = np.corrcoef(res[:-1], res[1:])[0, 1]
            if not np.isnan(corr):
                correlations.append({
                    "label": lbl,
                    "well": well,
                    "lag1_corr": corr,
                    "n_points": len(res),
                })
                correlations_by_label.setdefault(str(lbl), []).append(float(corr))

    correlation_stats = pd.DataFrame(correlations)
    correlations_by_label_arr = {
        k: np.array(v) for k, v in correlations_by_label.items()
    }

    return correlation_stats, correlations_by_label_arr


def estimate_x_shift_statistics(
    all_res: pd.DataFrame,
    fit_results: dict[str, Any] | None = None,  # ruff: ignore[unused-function-argument]
) -> pd.DataFrame:
    """Estimate potential systematic x-shifts per well (heuristics).

    Analyzes residual-vs-x patterns to flag wells whose x-values (e.g. pH) may
    be systematically off: a linear trend of residuals against x, and asymmetry
    between positive and negative residuals.

    Parameters
    ----------
    all_res : pd.DataFrame
        Residual table with ``label``, ``well``, ``x`` and ``std_res`` columns.
    fit_results : dict[str, Any] | None
        Per-well fit results, reserved for future x-shift fitting; currently
        unused.

    Returns
    -------
    pd.DataFrame
        Per-well shift indicators: ``residual_slope``, ``residual_intercept``,
        ``trend_strength``, ``asymmetry`` and ``n_points``.
    """
    shift_stats = []

    for (lbl, well), group in all_res.groupby(["label", "well"]):
        sorted_group = group.sort_values("x")

        if len(sorted_group) > MIN_POINTS_FOR_TREND:
            x_vals = sorted_group["x"].to_numpy(dtype=float)
            res_vals = sorted_group["std_res"].to_numpy(dtype=float)

            # 1. Systematic linear trend of residuals against x.
            try:
                slope, intercept = np.polyfit(x_vals, res_vals, 1)
                trend_strength = np.abs(slope) * (x_vals.max() - x_vals.min())
            except (np.linalg.LinAlgError, ValueError):
                slope = intercept = trend_strength = np.nan

            # 2. Asymmetry of positive vs negative residuals (~0 if symmetric).
            pos_mean = res_vals[res_vals > 0].mean() if (res_vals > 0).any() else 0
            neg_mean = res_vals[res_vals < 0].mean() if (res_vals < 0).any() else 0
            asymmetry = pos_mean + neg_mean

            shift_stats.append({
                "label": lbl,
                "well": well,
                "residual_slope": slope,
                "residual_intercept": intercept,
                "trend_strength": trend_strength,
                "asymmetry": asymmetry,
                "n_points": len(sorted_group),
            })

    return pd.DataFrame(shift_stats)


# A point this far out is worth naming; below it the plot would be unreadable.
_OUTLIER_LABEL_SIGMA = 4.0
# Even above the threshold, only the worst few get named - a plate in trouble
# would otherwise paper the figure with text and hide the shape it exists to show.
_MAX_OUTLIER_LABELS = 8


def _annotate_strong_outliers(
    ax: Axes, grp: pd.DataFrame, pred: np.ndarray, std_res: np.ndarray
) -> None:
    """Name the worst residuals as ``well:step`` so they can be chased.

    An extreme point is the most useful thing on this plot and the hardest to
    act on: seeing a residual at 24 sigma says something is wrong, but not
    where. Labelling turns it into a well and a titration step one can open.

    Parameters
    ----------
    ax : Axes
        Axes to draw on.
    grp : pd.DataFrame
        Rows for this label, carrying ``well`` and ``step`` where available.
    pred : np.ndarray
        Predicted signal per row.
    std_res : np.ndarray
        Standardised residual per row.
    """
    if "well" not in grp.columns:
        return
    mag = np.abs(std_res)
    strong = np.flatnonzero(np.isfinite(mag) & (mag > _OUTLIER_LABEL_SIGMA))
    if strong.size == 0:
        return
    worst = strong[np.argsort(-mag[strong])][:_MAX_OUTLIER_LABELS]
    wells = grp["well"].to_numpy()
    steps = grp["step"].to_numpy() if "step" in grp.columns else None
    for i in worst:
        name = str(wells[i]) if steps is None else f"{wells[i]}:{steps[i]}"
        ax.annotate(
            name,
            (pred[i], mag[i]),
            fontsize=7,
            xytext=(4, 2),
            textcoords="offset points",
            color="C3",
        )


def plot_residual_vs_predicted(all_res: pd.DataFrame, title: str = "") -> Figure:
    r"""Plot \|standardized residual\| vs predicted signal per label.

    A flat trend at ~0.80 (expected \|N(0,1)\|) confirms the error model is
    correctly calibrated.  A rising trend indicates under-estimated errors
    at high signals (multiplicative noise).

    Parameters
    ----------
    all_res : pd.DataFrame
        Residual DataFrame from ``residuals_from_fit_results``.  Must contain
        columns ``label``, ``yhat``, and ``std_res``.
    title : str, optional
        Figure suptitle suffix.

    Returns
    -------
    Figure
        Matplotlib figure (one panel per label).
    """
    labels = sorted(all_res["label"].unique())
    fig, axes = plt.subplots(
        1, len(labels), figsize=(7 * len(labels), 5), squeeze=False
    )

    for ax, label in zip(axes[0], labels, strict=False):
        grp = all_res[all_res["label"] == label]
        pred = grp["yhat"].to_numpy(dtype=float)
        std_res = grp["std_res"].to_numpy(dtype=float)
        ax.scatter(pred, np.abs(std_res), s=8, alpha=0.3, color="C0")
        _annotate_strong_outliers(ax, grp, pred, std_res)

        valid = np.isfinite(pred) & np.isfinite(std_res)
        if valid.sum() > MIN_POINTS_FOR_BINNING:
            n_bins = min(MAX_BINS, valid.sum() // 5)
            bins = np.linspace(
                np.nanmin(pred[valid]), np.nanmax(pred[valid]), n_bins + 1
            )
            bin_centers, bin_means = [], []
            for j in range(len(bins) - 1):
                in_bin = valid & (pred >= bins[j]) & (pred < bins[j + 1])
                if in_bin.sum() > MIN_POINTS_FOR_BIN:
                    bin_centers.append((bins[j] + bins[j + 1]) / 2)
                    bin_means.append(float(np.mean(np.abs(std_res[in_bin]))))
            if bin_centers:
                ax.plot(bin_centers, bin_means, "r-o", lw=2, ms=5, label="Binned mean")
        ax.axhline(
            0.798, color="green", ls="--", lw=1.5, label=r"Expected \|N(0,1)\| = 0.80"
        )
        ax.set_xlabel("Predicted signal")
        ax.set_ylabel(r"\|Standardized residual\|")
        ax.set_title(f"Label {label}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    suptitle = "Residual vs predicted: flat = correct error model"
    if title:
        suptitle = f"{suptitle} — {title}"
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_residual_distribution(all_res: pd.DataFrame, title: str = "") -> Figure:
    """Histogram and Q-Q of the standardised residuals, per label.

    Summary statistics cannot tell a heavy tail from a bimodal spread from a
    shifted centre, and those call for different fixes: a heavy tail wants a
    robust likelihood, a shift wants the mean model looked at, a bimodal spread
    usually means two populations of wells. The histogram against the reference
    normal shows centre and width; the Q-Q plot shows the tails, which is where
    an error model is normally wrong.

    Parameters
    ----------
    all_res : pd.DataFrame
        Canonical residual table with ``label`` and ``std_res`` columns.
    title : str
        Figure title.

    Returns
    -------
    Figure
        Two rows - histogram and Q-Q - by one column per label.
    """
    labels = sorted(all_res["label"].astype(str).unique())
    fig, axes = plt.subplots(
        2, len(labels), figsize=(5.5 * len(labels), 8), squeeze=False
    )
    grid = np.linspace(-4.0, 4.0, 200)
    for col, label in enumerate(labels):
        v = all_res.loc[all_res["label"].astype(str) == label, "std_res"]
        v = np.asarray(v, dtype=float)
        v = v[np.isfinite(v)]
        ax_h = axes[0][col]
        if v.size:
            ax_h.hist(v, bins=40, density=True, alpha=0.6, color="C0")
            ax_h.plot(
                grid,
                np.exp(-0.5 * grid**2) / np.sqrt(2 * np.pi),
                "g--",
                label="N(0,1)",
            )
            ax_h.legend(fontsize=8)
        ax_h.set_title(f"Label {label}")
        ax_h.set_xlabel("Standardized residual")
        ax_h.set_ylabel("Density")
        ax_q = axes[1][col]
        if v.size > 1:
            sp_stats.probplot(v, dist="norm", plot=ax_q)
            ax_q.get_lines()[0].set_markersize(3)
            ax_q.set_title("")
        ax_q.set_xlabel("Theoretical quantiles")
        ax_q.set_ylabel("Ordered residuals")
    fig.suptitle(f"Residual distribution: departures from N(0,1) - {title}")
    fig.tight_layout()
    return fig


def plot_residual_vs_yerr(all_res: pd.DataFrame, title: str = "") -> Figure:
    """Plot raw residual² vs y_err² per label (error calibration check).

    Points should scatter around the y=x line if the assigned uncertainties
    match the actual scatter.  A slope < 1 means errors are over-estimated;
    slope > 1 means under-estimated.

    Parameters
    ----------
    all_res : pd.DataFrame
        Residual DataFrame from ``residuals_from_fit_results``.  Must contain
        columns ``label``, ``sigma``, and ``raw_res``.
    title : str, optional
        Figure suptitle suffix.

    Returns
    -------
    Figure
        Matplotlib figure (one panel per label).
    """
    labels = sorted(all_res["label"].unique())
    fig, axes = plt.subplots(
        1, len(labels), figsize=(7 * len(labels), 5), squeeze=False
    )

    for ax, label in zip(axes[0], labels, strict=False):
        grp = all_res[all_res["label"] == label]
        y_err = grp["sigma"].to_numpy(dtype=float)
        raw_res = grp["raw_res"].to_numpy(dtype=float)
        valid = np.isfinite(y_err) & np.isfinite(raw_res) & (y_err > 0)
        ye_v, rr_v = y_err[valid], raw_res[valid]

        ax.scatter(ye_v**2, rr_v**2, s=8, alpha=0.3, color="C0")

        n_bins = min(MAX_BINS, valid.sum() // 5)
        if n_bins > MIN_POINTS_FOR_BIN:
            bins = np.linspace(np.min(ye_v**2), np.max(ye_v**2), n_bins + 1)
            bc, bm = [], []
            for j in range(len(bins) - 1):
                in_bin = (ye_v**2 >= bins[j]) & (ye_v**2 < bins[j + 1])
                if in_bin.sum() > MIN_POINTS_FOR_BIN:
                    bc.append((bins[j] + bins[j + 1]) / 2)
                    bm.append(float(np.mean(rr_v[in_bin] ** 2)))
            if bc:
                ax.plot(bc, bm, "r-o", lw=2, ms=5, label="Binned mean(res²)")

        lim = max(float(np.max(ye_v**2)), float(np.max(rr_v**2)))
        ax.plot([0, lim], [0, lim], "g--", lw=1.5, label="Perfect: res² = y_err²")
        ax.set_xlabel("y_err² (model variance)")
        ax.set_ylabel("residual² (observed variance)")
        ax.set_title(f"Label {label}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    suptitle = "Error calibration: res² vs y_err²"
    if title:
        suptitle = f"{suptitle} — {title}"
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    plt.close(fig)
    return fig
