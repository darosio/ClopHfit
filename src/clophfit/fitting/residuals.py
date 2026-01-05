"""Residual extraction and analysis utilities for fit results.

This module provides tools to extract, analyze, and validate residuals from
fitting procedures. Useful for diagnostics, model validation, and comparing
different fitting methods.
"""

from __future__ import annotations

import typing
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

if TYPE_CHECKING:
    from clophfit.clophfit_types import ArrayF
    from clophfit.fitting.data_structures import FitResult

# Statistical thresholds
OUTLIER_THRESHOLD_2SIGMA = 2.0
OUTLIER_THRESHOLD_3SIGMA = 3.0
OUTLIER_RATE_THRESHOLD = 0.05
BIAS_P_VALUE_THRESHOLD = 0.01
DW_LOWER_BOUND = 1.5
DW_UPPER_BOUND = 2.5

MIN_POINTS_FOR_TREND = 3  # Minimum points needed to detect trends


@dataclass(frozen=True)
class ResidualPoint:
    """Single residual data point with metadata.

    Attributes
    ----------
    label : str
        Dataset label (e.g., 'y1', 'y2' for multi-label fits)
    x : float
        X-value (pH or ligand concentration)
    resid_weighted : float
        Weighted residual: (y - model) / y_err
    resid_raw : float
        Raw residual: (y - model)
    raw_i : int
        Index into the original (unmasked) arrays for this label (`DataArray.xc/yc`).
    """

    label: str
    x: float
    resid_weighted: float
    resid_raw: float
    raw_i: int


def extract_residual_points(fr: FitResult[Any]) -> list[ResidualPoint]:
    """Extract residual points from a fit result.

    Parameters
    ----------
    fr : FitResult[Any]
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
    >>> dataset = Dataset({"y1": da}, is_ph=True)
    >>> fr = fit_binding_glob(dataset)
    >>> residuals = extract_residual_points(fr)
    >>> len(residuals) > 0
    True
    >>> residuals[0].label
    'y1'
    """
    if fr.result is None or fr.dataset is None:
        return []

    r = np.asarray(fr.result.residual, dtype=float)
    pts: list[ResidualPoint] = []

    start = 0
    for lbl, da in fr.dataset.items():
        n = len(da.y)  # masked length
        rw = r[start : start + n]
        rr = rw * da.y_err  # undo weighting: raw = weighted * y_err
        xs = da.x
        raw_is = np.flatnonzero(da.mask)

        pts.extend(
            ResidualPoint(
                label=lbl,
                x=float(xs[i]),
                resid_weighted=float(rw[i]),
                resid_raw=float(rr[i]),
                raw_i=int(raw_is[i]),
            )
            for i in range(n)
        )
        start += n

    if start != len(r):
        msg = f"Residual length mismatch: consumed {start}, residual has {len(r)}"
        raise ValueError(msg)

    return pts


def residual_dataframe(fr: FitResult[Any]) -> pd.DataFrame:
    """Convert fit result residuals to a DataFrame.

    Parameters
    ----------
    fr : FitResult[Any]
        Fit result to extract residuals from

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: label, x, resid_weighted, resid_raw, raw_i

    Examples
    --------
    >>> from clophfit.fitting.core import fit_binding_glob
    >>> from clophfit.fitting.data_structures import Dataset, DataArray
    >>> import numpy as np
    >>> x = np.array([9.0, 8.0, 7.0, 6.0, 5.0])
    >>> y = 500 + 500 * 10 ** (7.0 - x) / (1 + 10 ** (7.0 - x))
    >>> da = DataArray(xc=x, yc=y, y_errc=np.ones_like(y) * 10)
    >>> dataset = Dataset({"y1": da}, is_ph=True)
    >>> fr = fit_binding_glob(dataset)
    >>> df = residual_dataframe(fr)
    >>> "label" in df.columns and "x" in df.columns
    True
    """
    return pd.DataFrame([asdict(p) for p in extract_residual_points(fr)])


def collect_multi_residuals(
    fit_results: dict[str, FitResult[Any]],
    round_x: int | None = 3,
) -> pd.DataFrame:
    """Collect residuals from multiple fit results into a single DataFrame.

    Parameters
    ----------
    fit_results : dict[str, FitResult[Any]]
        Dictionary mapping well/key identifiers to fit results
    round_x : int | None
        Number of decimals to round x values (avoids float drift).
        Set to None to disable rounding.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with columns: well, label, x, resid_weighted, resid_raw, raw_i

    Examples
    --------
    >>> from clophfit.fitting.core import fit_binding_glob
    >>> from clophfit.fitting.data_structures import Dataset, DataArray
    >>> import numpy as np
    >>> x = np.array([9.0, 8.0, 7.0, 6.0, 5.0])
    >>> y = 500 + 500 * 10 ** (7.0 - x) / (1 + 10 ** (7.0 - x))
    >>> da = DataArray(xc=x, yc=y, y_errc=np.ones_like(y) * 10)
    >>> dataset = Dataset({"y1": da}, is_ph=True)
    >>> results = {"A01": fit_binding_glob(dataset), "A02": fit_binding_glob(dataset)}
    >>> all_res = collect_multi_residuals(results)
    >>> "well" in all_res.columns
    True
    >>> len(all_res) == 10  # 2 wells * 5 points
    True
    """
    rows = []
    for well, fr in fit_results.items():
        df = residual_dataframe(fr).assign(well=well)
        rows.append(df)

    all_res = pd.concat(rows, ignore_index=True)

    if round_x is not None:
        all_res["x"] = all_res["x"].round(round_x)

    return all_res


def residual_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute residual statistics by label.

    Parameters
    ----------
    df : pd.DataFrame
        Residual DataFrame (from residual_dataframe or collect_multi_residuals)

    Returns
    -------
    pd.DataFrame
        Statistics by label: mean, std, median, mad, outlier_count

    Examples
    --------
    >>> from clophfit.fitting.core import fit_binding_glob
    >>> from clophfit.fitting.data_structures import Dataset, DataArray
    >>> import numpy as np
    >>> x = np.array([9.0, 8.0, 7.0, 6.0, 5.0])
    >>> y = 500 + 500 * 10 ** (7.0 - x) / (1 + 10 ** (7.0 - x))
    >>> da = DataArray(xc=x, yc=y, y_errc=np.ones_like(y) * 10)
    >>> dataset = Dataset({"y1": da}, is_ph=True)
    >>> results = {"A01": fit_binding_glob(dataset)}
    >>> all_res = collect_multi_residuals(results)
    >>> stats = residual_statistics(all_res)
    >>> "mean" in stats.columns
    True
    """

    def outlier_count(x: ArrayF) -> int:
        """Count points beyond ±2-sigma deviations."""
        return int((np.abs(x) > OUTLIER_THRESHOLD_2SIGMA).sum())

    # Use dict format for agg to avoid mypy issues with tuple format
    summary = df.groupby("label")["resid_weighted"].agg(
        mean="mean",
        std="std",
        median="median",
        mad=lambda x: sp_stats.median_abs_deviation(x, nan_policy="omit"),
        outlier_count=outlier_count,
        n_points="count",
    )

    # Convert to DataFrame (agg returns DataFrame when grouping by one column)
    summary_df = typing.cast("pd.DataFrame", summary)
    summary_df["outlier_rate"] = summary_df["outlier_count"] / summary_df["n_points"]

    return summary_df


def validate_residuals(fr: FitResult[Any], *, verbose: bool = True) -> dict[str, bool]:
    """Validate residual quality for a fit result.

    Checks for common issues:
    - Systematic bias (mean significantly different from 0)
    - Outliers (more than 5% beyond ±3-sigma)
    - Serial correlation (adjacent residuals)

    Parameters
    ----------
    fr : FitResult[Any]
        Fit result to validate
    verbose : bool
        Print warnings for failed checks

    Returns
    -------
    dict[str, bool]
        Dictionary of check results: {'bias_ok', 'outliers_ok', 'correlation_ok'}

    Examples
    --------
    >>> from clophfit.fitting.core import fit_binding_glob
    >>> from clophfit.fitting.data_structures import Dataset, DataArray
    >>> import numpy as np
    >>> x = np.array([9.0, 8.0, 7.0, 6.0, 5.0])
    >>> y = 500 + 500 * 10 ** (7.0 - x) / (1 + 10 ** (7.0 - x))
    >>> da = DataArray(xc=x, yc=y, y_errc=np.ones_like(y) * 10)
    >>> dataset = Dataset({"y1": da}, is_ph=True)
    >>> fr = fit_binding_glob(dataset)
    >>> checks = validate_residuals(fr, verbose=False)
    >>> isinstance(checks, dict) and "bias_ok" in checks
    True
    """
    checks = {"bias_ok": True, "outliers_ok": True, "correlation_ok": True}

    if fr.result is None or fr.result.residual is None:
        return checks

    r = fr.result.residual

    # Check 1: Systematic bias (t-test against 0)
    _t_stat, p_value = sp_stats.ttest_1samp(r, 0)
    checks["bias_ok"] = bool(p_value > BIAS_P_VALUE_THRESHOLD)  # 99% confidence
    if not checks["bias_ok"] and verbose:
        print(f"⚠️  Systematic bias detected (mean={r.mean():.3f}, p={p_value:.4f})")

    # Check 2: Outliers (more than 5% beyond ±2-sigma)
    outliers = np.abs(r) > OUTLIER_THRESHOLD_2SIGMA
    outlier_rate = outliers.mean()
    checks["outliers_ok"] = outlier_rate < OUTLIER_RATE_THRESHOLD
    if not checks["outliers_ok"] and verbose:
        print(f"⚠️  High outlier rate: {outlier_rate:.1%} beyond ±2-sigma")

    # Check 3: Serial correlation (Durbin-Watson test approximation)
    if len(r) > 1:
        diff = np.diff(r)
        dw_stat = np.sum(diff**2) / np.sum(r**2)
        # DW ~ 2 for no correlation, < 1.5 or > 2.5 suggests correlation
        checks["correlation_ok"] = DW_LOWER_BOUND < dw_stat < DW_UPPER_BOUND
        if not checks["correlation_ok"] and verbose:
            print(f"⚠️  Serial correlation detected (DW={dw_stat:.2f})")

    return checks


def compute_residual_covariance(
    all_res: pd.DataFrame, value_col: str = "resid_weighted"
) -> dict[str, pd.DataFrame]:
    """Compute covariance matrix of residuals for each label."""
    cov_by_label: dict[str, pd.DataFrame] = {}
    for lbl, g in all_res.groupby("label"):
        pivot_table = g.pivot_table(
            index="well", columns="x", values=value_col, aggfunc="mean"
        )
        # drop wells missing any x (to make a clean covariance across x points)
        pivot_table = pivot_table.dropna(axis=0, how="any")
        data = pivot_table.to_numpy(dtype=float)
        # covariance across x-points (features), so rowvar=False
        cov = np.cov(data, rowvar=False, ddof=1)
        cov_by_label[str(lbl)] = pd.DataFrame(
            cov,
            index=pivot_table.columns.to_list(),
            columns=pivot_table.columns.to_list(),
        )
    return cov_by_label


def compute_correlation_matrices(
    cov_by_label: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Convert covariance matrices to correlation matrices."""
    corr_by_label: dict[str, pd.DataFrame] = {}
    for lbl, cov_df in cov_by_label.items():
        cov = cov_df.to_numpy()
        std_outer = np.outer(np.sqrt(np.diag(cov)), np.sqrt(np.diag(cov)))
        corr = cov / std_outer
        corr_by_label[lbl] = pd.DataFrame(
            corr, index=cov_df.index, columns=cov_df.columns
        )
    return corr_by_label


def analyze_label_bias(
    all_res: pd.DataFrame, n_bins: int = 5
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Detect systematic bias by label and x-range."""
    outlier_threshold = 3.0  # Standard deviations
    strong_negative_threshold = -0.5

    all_res = all_res.copy()
    all_res["x_bin"] = pd.cut(all_res["x"], bins=n_bins)

    mean_resid = all_res["resid_weighted"].mean()
    std_resid = all_res["resid_weighted"].std()
    all_res["std_res"] = (all_res["resid_weighted"] - mean_resid) / std_resid

    bias_summary = all_res.groupby(["label", "x_bin"], observed=False).agg(
        mean_resid=("resid_weighted", "mean"),
        std_resid=("resid_weighted", "std"),
        count=("resid_weighted", "count"),
        outlier_rate=("std_res", lambda x: (np.abs(x) > outlier_threshold).mean()),
        mean_std_res=("std_res", "mean"),
    )

    label_bias = all_res.groupby("label", observed=False).agg(
        mean_resid=("resid_weighted", "mean"),
        std_resid=("resid_weighted", "std"),
        median_resid=("resid_weighted", "median"),
        outlier_rate=("std_res", lambda x: (np.abs(x) > outlier_threshold).mean()),
        negative_bias_frac=(
            "resid_weighted",
            lambda x: (x < strong_negative_threshold).mean(),
        ),
    )

    return bias_summary, label_bias


def detect_adjacent_correlation(
    all_res: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Detect correlation between adjacent residuals within wells."""
    correlations = []
    correlations_by_label: dict[str, list[float]] = {}

    for (lbl, well), group in all_res.groupby(["label", "well"]):
        g = group.sort_values("x")
        res = g["resid_weighted"].to_numpy()

        if len(res) > 1:
            corr = np.corrcoef(res[:-1], res[1:])[0, 1]
            if not np.isnan(corr):
                correlations.append({
                    "label": lbl,
                    "well": well,
                    "lag1_corr": corr,
                    "n_points": len(res),
                })
                correlations_by_label.setdefault(str(lbl), []).append(corr)

    correlation_stats = pd.DataFrame(correlations)
    correlations_by_label_arr = {
        k: np.array(v) for k, v in correlations_by_label.items()
    }

    return correlation_stats, correlations_by_label_arr


def estimate_x_shift_statistics(
    all_res: pd.DataFrame,
    fit_results: dict[str, Any],  # noqa: ARG001
) -> pd.DataFrame:
    """Estimate potential systematic x-shifts per well (heuristics)."""
    shift_stats = []

    for (lbl, well), group in all_res.groupby(["label", "well"]):
        sorted_group = group.sort_values("x")

        if len(sorted_group) > MIN_POINTS_FOR_TREND:
            x_vals = sorted_group["x"].to_numpy()
            res_vals = sorted_group["resid_weighted"].to_numpy()

            try:
                slope, intercept = np.polyfit(x_vals, res_vals, 1)
                trend_strength = np.abs(slope) * (x_vals.max() - x_vals.min())
            except (np.linalg.LinAlgError, ValueError):
                slope = intercept = trend_strength = np.nan

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
