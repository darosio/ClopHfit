"""Functions for evaluating fit quality.

This module provides metrics for evaluating fitting performance, including:
1. Bias (accuracy)
2. Coverage (uncertainty quantification)
3. Residual distribution (goodness of fit)
4. Parameter error analysis
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from collections.abc import Sequence

    from clophfit.clophfit_types import ArrayF
    from clophfit.fitting.data_structures import FitResult, MiniT

SIGNIFICANCE_LEVEL = 0.05


def calculate_bias(estimated: np.ndarray, true_value: float) -> float:
    """Calculate the bias (mean error) of the estimates.

    Parameters
    ----------
    estimated : np.ndarray
        Array of estimated values.
    true_value : float
        The true value.

    Returns
    -------
    float
        The bias (mean of estimated - true_value).
    """
    valid = estimated[np.isfinite(estimated)]
    if len(valid) == 0:
        return np.nan
    return float(np.mean(valid - true_value))


def calculate_rmse(estimated: ArrayF, true_value: float) -> float:
    """Calculate the Root Mean Square Error (RMSE).

    Parameters
    ----------
    estimated : ArrayF
        Array of estimated values.
    true_value : float
        The true value.

    Returns
    -------
    float
        The RMSE.
    """
    valid = estimated[np.isfinite(estimated)]
    if len(valid) == 0:
        return np.nan
    return float(np.sqrt(np.mean((valid - true_value) ** 2)))


def calculate_coverage(
    estimated: ArrayF, errors: ArrayF, true_value: float, confidence: float = 0.95
) -> float:
    """Calculate the coverage probability of the confidence intervals.

    Parameters
    ----------
    estimated : ArrayF
        Array of estimated values.
    errors : ArrayF
        Array of standard errors (1 sigma).
    true_value : float
        The true value.
    confidence : float
        The desired confidence level (default: 0.95).

    Returns
    -------
    float
        The fraction of intervals that contain the true value.
    """
    valid_mask = np.isfinite(estimated) & np.isfinite(errors)
    est = estimated[valid_mask]
    err = errors[valid_mask]

    if len(est) == 0:
        return np.nan

    z = stats.norm.ppf((1 + confidence) / 2)
    lower = est - z * err
    upper = est + z * err

    covered = (true_value >= lower) & (true_value <= upper)
    return float(np.mean(covered))


def evaluate_residuals(residuals: np.ndarray) -> dict[str, float]:
    """Evaluate the normality of residuals.

    Parameters
    ----------
    residuals : np.ndarray
        Array of residuals.

    Returns
    -------
    dict[str, float]
        Dictionary containing:
        - 'shapiro_stat': Shapiro-Wilk test statistic
        - 'shapiro_p': Shapiro-Wilk p-value
        - 'mean': Mean of residuals
        - 'std': Standard deviation of residuals
    """
    min_shapiro_samples = 3  # Shapiro-Wilk requires at least 3 data points
    valid = residuals[np.isfinite(residuals)]
    if len(valid) < min_shapiro_samples:
        return {
            "shapiro_stat": np.nan,
            "shapiro_p": np.nan,
            "mean": np.nan,
            "std": np.nan,
        }

    stat, p = stats.shapiro(valid)
    return {
        "shapiro_stat": stat,
        "shapiro_p": p,
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid)),
    }


def extract_params(fr: FitResult[MiniT], param_name: str = "K") -> tuple[float, float]:
    """Extract parameter value and error from a FitResult.

    Parameters
    ----------
    fr : FitResult[MiniT]
        The fit result object.
    param_name : str
        The name of the parameter to extract (default: "K").

    Returns
    -------
    tuple[float, float]
        (value, error). Returns (np.nan, np.nan) if extraction fails.
    """
    if fr.result is None or not hasattr(fr.result, "params"):
        return np.nan, np.nan

    params = fr.result.params
    if param_name in params:
        val = params[param_name].value
        err = params[param_name].stderr
        # Handle case where stderr is None
        if err is None:
            err = np.nan
        return val, err

    return np.nan, np.nan


def load_real_data_paths() -> dict[str, Path]:
    """Find available real data directories.

    Returns
    -------
    dict[str, Path]
        Mapping of dataset name to path
    """
    base_paths = [
        Path("tests/Tecan"),
        Path("../tests/Tecan"),
    ]

    datasets = {}
    for base in base_paths:
        if not base.exists():
            continue

        for dataset_name in ["L1", "L2", "L4", "140220"]:
            dataset_path = base / dataset_name
            list_file = dataset_path / "list.pH.csv"

            if dataset_path.exists() and list_file.exists():
                datasets[dataset_name] = dataset_path

    return datasets


def compare_methods_statistical(
    method1_errors: Sequence[float],
    method2_errors: Sequence[float],
    method1_name: str = "Method 1",
    method2_name: str = "Method 2",
) -> dict[str, float | bool | str]:
    """Perform statistical comparison between two methods.

    Uses Mann-Whitney U test (non-parametric) for comparing absolute errors.

    Parameters
    ----------
    method1_errors : Sequence[float]
        Errors from method 1
    method2_errors : Sequence[float]
        Errors from method 2
    method1_name : str
        Name of method 1
    method2_name : str
        Name of method 2

    Returns
    -------
    dict[str, float | bool | str]
        Statistical comparison results. The dictionary includes:
        - 'test': Name of the statistical test.
        - 'statistic': Test statistic value.
        - 'p_value': Computed p-value.
        - 'significant': Whether the difference is significant at alpha=0.05.
        - 'better_method': Method with lower MAE (or 'Equivalent').
        - 'mae1': Mean absolute error for method1.
        - 'mae2': Mean absolute error for method2.
        - 'error': Only present when comparison cannot be performed.
    """
    m1 = np.asarray(method1_errors)
    m2 = np.asarray(method2_errors)

    m1_valid = m1[np.isfinite(m1)]
    m2_valid = m2[np.isfinite(m2)]

    if len(m1_valid) == 0 or len(m2_valid) == 0:
        print(f"\n--- Statistical Comparison: {method1_name} vs {method2_name} ---")
        print("  Insufficient data for comparison")
        return {
            "test": "mann_whitney_u",
            "statistic": np.nan,
            "p_value": np.nan,
            "significant": False,
            "better_method": "",
            "error": "Insufficient data",
        }

    # Use absolute errors for comparison
    m1_abs = np.abs(m1_valid)
    m2_abs = np.abs(m2_valid)

    stat, p_value = stats.mannwhitneyu(m1_abs, m2_abs, alternative="two-sided")

    mae1 = float(np.mean(m1_abs))
    mae2 = float(np.mean(m2_abs))

    significant: bool = float(p_value) < SIGNIFICANCE_LEVEL
    better_method: str = "Equivalent"
    if significant:
        better_method = method1_name if mae1 < mae2 else method2_name
    result: dict[str, str | bool | float] = {
        "test": "mann_whitney_u",
        "statistic": float(stat),
        "p_value": float(p_value),
        "significant": significant,
        "better_method": better_method,
        "mae1": mae1,
        "mae2": mae2,
    }

    print(f"\n--- Statistical Comparison: {method1_name} vs {method2_name} ---")
    print(f"  {method1_name}: MAE={mae1:.4f}, N={len(m1_abs)}")
    print(f"  {method2_name}: MAE={mae2:.4f}, N={len(m2_abs)}")
    print(f"  Mann-Whitney U: stat={stat:.1f}, p={p_value:.4f}")
    if significant:
        better = result["better_method"]
        print(f"  → Significant difference (p<0.05): {better} is better")
    else:
        print("  → No significant difference (p≥0.05)")

    return result
