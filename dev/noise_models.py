"""Noise model analysis utilities for prtecan data.

This module provides functions to characterize and simulate the noise structure
in fluorescence titration data, including:
- Residual covariance computation
- Systematic bias detection (e.g., label-dependent, pH-dependent)
- Adjacent point correlation analysis
- X-value (pH) uncertainty estimation

For general residual extraction, use `clophfit.fitting.residuals` instead.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from clophfit.fitting.residuals import ResidualPoint, extract_residual_points

# Re-export for backward compatibility
__all__ = ["ResidualPoint", "extract_residual_points"]

# Constants
MIN_POINTS_FOR_TREND = 3  # Minimum points needed to detect trends


def compute_residual_covariance(
    all_res: pd.DataFrame, value_col: str = "resid_weighted"
) -> dict[str, pd.DataFrame]:
    """Compute covariance matrix of residuals for each label.

    Parameters
    ----------
    all_res : pd.DataFrame
        DataFrame with columns: label, well, x, resid_weighted, etc.
    value_col : str
        Column name for residual values (default: "resid_weighted")

    Returns
    -------
    dict[str, pd.DataFrame]
        Covariance matrices indexed by label, with x values as index/columns
    """
    cov_by_label: dict[str, pd.DataFrame] = {}

    for lbl, g in all_res.groupby("label"):
        # Pivot: rows = wells/curves, cols = x points
        pivot_table = g.pivot_table(
            index="well", columns="x", values=value_col, aggfunc="mean"
        )
        # Drop wells missing any x (for clean covariance)
        pivot_table = pivot_table.dropna(axis=0, how="any")

        data = pivot_table.to_numpy(dtype=float)
        # Covariance across x-points (features), so rowvar=False
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
    """Convert covariance matrices to correlation matrices.

    Parameters
    ----------
    cov_by_label : dict[str, pd.DataFrame]
        Covariance matrices by label

    Returns
    -------
    dict[str, pd.DataFrame]
        Correlation matrices by label
    """
    corr_by_label = {}
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
    """Detect systematic bias by label and x-range.

    Issue 1: Identifies if certain labels (e.g., y1) have systematic bias
    at specific pH ranges (e.g., always negative at low pH).

    Parameters
    ----------
    all_res : pd.DataFrame
        DataFrame with columns: label, x, resid_weighted
        (from collect_multi_residuals or residual_dataframe)
    n_bins : int
        Number of x-value bins for analysis

    Returns
    -------
    bias_summary : pd.DataFrame
        Statistics by label and x-bin: mean, std, count, outlier_rate
    label_bias : pd.DataFrame
        Overall bias statistics by label

    Notes
    -----
    Standardized residuals (std_res) are computed as:
        std_res = (resid_weighted - mean) / std
    where mean and std are computed globally across all residuals.
    Outliers are defined as |std_res| > 3 (beyond ±3 sigma).
    Strong negative bias is defined as resid_weighted < -0.5.
    """
    # Thresholds
    outlier_threshold = 3.0  # Standard deviations
    strong_negative_threshold = -0.5

    # Create x-bins and compute standardized residuals
    all_res = all_res.copy()
    all_res["x_bin"] = pd.cut(all_res["x"], bins=n_bins)

    # Compute standardized residuals (z-scores)
    # Use global mean/std across all residuals
    mean_resid = all_res["resid_weighted"].mean()
    std_resid = all_res["resid_weighted"].std()
    all_res["std_res"] = (all_res["resid_weighted"] - mean_resid) / std_resid

    # Bias by label and x-bin
    bias_summary = all_res.groupby(["label", "x_bin"], observed=False).agg(
        mean_resid=("resid_weighted", "mean"),
        std_resid=("resid_weighted", "std"),
        count=("resid_weighted", "count"),
        outlier_rate=("std_res", lambda x: (np.abs(x) > outlier_threshold).mean()),
        mean_std_res=("std_res", "mean"),
    )

    # Overall bias by label
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
    """Detect correlation between adjacent residuals within wells.

    Issue 2: Tests if adjacent points show systematic patterns (positive then
    negative), which may indicate x-value errors or model misspecification.

    Parameters
    ----------
    all_res : pd.DataFrame
        DataFrame with columns: label, well, x, resid_weighted (sorted by x)

    Returns
    -------
    correlation_stats : pd.DataFrame
        Lag-1 correlation statistics by label and well
    correlations_by_label : dict[str, np.ndarray]
        Array of lag-1 correlations for each label
    """
    correlations = []
    correlations_by_label: dict[str, list[float]] = {}

    for (lbl, well), group in all_res.groupby(["label", "well"]):
        # Sort by x to ensure adjacent points are sequential
        g = group.sort_values("x")
        res = g["resid_weighted"].to_numpy()

        if len(res) > 1:
            # Lag-1 correlation
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

    # Convert lists to arrays
    correlations_by_label_arr = {
        k: np.array(v) for k, v in correlations_by_label.items()
    }

    return correlation_stats, correlations_by_label_arr


def estimate_x_shift_statistics(
    all_res: pd.DataFrame,
    fit_results: dict[str, Any],  # noqa: ARG001
) -> pd.DataFrame:
    """Estimate potential systematic pH shifts per well.

    Issue 3: Analyzes residual patterns to detect if x-values (pH) might be
    systematically wrong for individual wells or entire plates.

    Parameters
    ----------
    all_res : pd.DataFrame
        DataFrame with residuals by well
    fit_results : dict[str, Any]
        Dictionary of fit results by well (for future x-shift fitting)

    Returns
    -------
    pd.DataFrame
        Statistics suggesting potential x-shifts by well
    """
    shift_stats = []

    for (lbl, well), group in all_res.groupby(["label", "well"]):
        sorted_group = group.sort_values("x")

        # Simple indicators of x-shift:
        # 1. Systematic trend in residuals vs x
        if len(sorted_group) > MIN_POINTS_FOR_TREND:
            x_vals = sorted_group["x"].to_numpy()
            res_vals = sorted_group["resid_weighted"].to_numpy()

            # Linear trend coefficient
            try:
                slope, intercept = np.polyfit(x_vals, res_vals, 1)
                trend_strength = np.abs(slope) * (x_vals.max() - x_vals.min())
            except (np.linalg.LinAlgError, ValueError):
                slope = intercept = trend_strength = np.nan

            # 2. Asymmetry in positive vs negative residuals
            pos_mean = res_vals[res_vals > 0].mean() if (res_vals > 0).any() else 0
            neg_mean = res_vals[res_vals < 0].mean() if (res_vals < 0).any() else 0
            asymmetry = pos_mean + neg_mean  # Should be ~0 if symmetric

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


def simulate_correlated_noise(
    cov_matrix: np.ndarray,
    n_samples: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate correlated noise from covariance structure.

    Parameters
    ----------
    cov_matrix : np.ndarray
        Covariance matrix (n_features x n_features)
    n_samples : int
        Number of samples to generate
    rng : np.random.Generator | None
        Random number generator for reproducibility

    Returns
    -------
    np.ndarray
        Correlated noise samples (n_samples x n_features)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Ensure positive semi-definite
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)
    eigvals = np.maximum(eigvals, 0)  # Remove negative eigenvalues

    # Reconstruct covariance
    cov_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T

    # Generate samples
    return rng.multivariate_normal(
        mean=np.zeros(cov_matrix.shape[0]), cov=cov_psd, size=n_samples
    )


def export_noise_parameters(
    cov_by_label: dict[str, pd.DataFrame],
    label_bias: pd.DataFrame,
    correlation_stats: pd.DataFrame,
    output_dir: str = "dev",
) -> None:
    """Export noise model parameters for use in synthetic data generation.

    Parameters
    ----------
    cov_by_label : dict[str, pd.DataFrame]
        Covariance matrices by label
    label_bias : pd.DataFrame
        Bias statistics by label
    correlation_stats : pd.DataFrame
        Correlation statistics
    output_dir : str
        Directory to save outputs
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Save covariance matrices
    for label, cov_df in cov_by_label.items():
        cov_df.to_csv(output_path / f"cov_matrix_{label}.csv")

    # Save bias parameters
    label_bias.to_csv(output_path / "label_bias.csv")

    # Save correlation summary
    corr_summary = correlation_stats.groupby("label").agg(
        mean_lag1_corr=("lag1_corr", "mean"),
        median_lag1_corr=("lag1_corr", "median"),
        std_lag1_corr=("lag1_corr", "std"),
    )
    corr_summary.to_csv(output_path / "correlation_summary.csv")
