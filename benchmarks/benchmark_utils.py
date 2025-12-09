"""Shared utilities for benchmark scripts.

Provides common functions for:
- Metric computation with NaN-safe operations
- Result provenance tracking
- Logging configuration
- Statistical comparisons
"""

from __future__ import annotations

import datetime
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import stats

if TYPE_CHECKING:
    from clophfit.fitting.data_structures import FitResult

logger = logging.getLogger(__name__)


def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """Configure logging for benchmark scripts."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)


@dataclass
class Provenance:
    """Track provenance information for benchmark results."""

    timestamp: str
    git_commit: str | None
    script_name: str
    n_trials: int
    extra_info: dict = field(default_factory=dict)

    @classmethod
    def capture(cls, script_name: str, n_trials: int, **extra) -> Provenance:
        """Capture current provenance information."""
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # Try to get git commit
        git_commit = None
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            if result.returncode == 0:
                git_commit = result.stdout.strip()
        except Exception:
            pass

        return cls(
            timestamp=timestamp,
            git_commit=git_commit,
            script_name=script_name,
            n_trials=n_trials,
            extra_info=extra,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "timestamp": self.timestamp,
            "git_commit": self.git_commit,
            "script_name": self.script_name,
            "n_trials": self.n_trials,
            **self.extra_info,
        }


def extract_K(fr: FitResult) -> tuple[float | None, float | None]:
    """Extract K estimate and uncertainty from FitResult.

    Parameters
    ----------
    fr : FitResult
        Fitting result object

    Returns
    -------
    tuple[float | None, float | None]
        (K_value, K_stderr) or (None, None) if extraction fails
    """
    if fr.result is None:
        return None, None

    if hasattr(fr.result, "params"):
        params = fr.result.params
        if "K" in params:
            K = params["K"].value
            K_err = params["K"].stderr if params["K"].stderr else None
            return K, K_err

    return None, None


def compute_coverage(
    K_values: np.ndarray,
    K_errors: np.ndarray,
    K_true: float,
    confidence: float = 0.95,
) -> float:
    """Compute coverage probability (fraction where true value is within CI).

    Parameters
    ----------
    K_values : np.ndarray
        Estimated K values
    K_errors : np.ndarray
        Standard errors for K estimates
    K_true : float
        True K value
    confidence : float
        Confidence level (default 0.95)

    Returns
    -------
    float
        Coverage probability (0-1)
    """
    if len(K_values) == 0:
        return np.nan

    z = stats.norm.ppf((1 + confidence) / 2)
    lower = K_values - z * K_errors
    upper = K_values + z * K_errors
    covered = (K_true >= lower) & (K_true <= upper)
    return float(np.nanmean(covered))


def compute_summary_stats(
    values: np.ndarray | list,
) -> dict[str, float]:
    """Compute NaN-safe summary statistics.

    Parameters
    ----------
    values : array-like
        Values to summarize

    Returns
    -------
    dict[str, float]
        Summary statistics: mean, median, std, min, max, count
    """
    arr = np.asarray(values)
    valid = arr[np.isfinite(arr)]

    if len(valid) == 0:
        return {
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "count": 0,
        }

    return {
        "mean": float(np.mean(valid)),
        "median": float(np.median(valid)),
        "std": float(np.std(valid)),
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "count": len(valid),
    }


def compute_rmse(errors: np.ndarray | list) -> float:
    """Compute root mean square error (NaN-safe).

    Parameters
    ----------
    errors : array-like
        Error values

    Returns
    -------
    float
        RMSE
    """
    arr = np.asarray(errors)
    valid = arr[np.isfinite(arr)]

    if len(valid) == 0:
        return np.nan

    return float(np.sqrt(np.mean(valid**2)))


def compare_methods_statistical(
    method1_errors: np.ndarray | list,
    method2_errors: np.ndarray | list,
    method1_name: str = "Method 1",
    method2_name: str = "Method 2",
) -> dict:
    """Perform statistical comparison between two methods.

    Uses Mann-Whitney U test (non-parametric) for comparing absolute errors.

    Parameters
    ----------
    method1_errors : array-like
        Errors from method 1
    method2_errors : array-like
        Errors from method 2
    method1_name : str
        Name of method 1
    method2_name : str
        Name of method 2

    Returns
    -------
    dict
        Statistical comparison results
    """
    m1 = np.asarray(method1_errors)
    m2 = np.asarray(method2_errors)

    m1_valid = m1[np.isfinite(m1)]
    m2_valid = m2[np.isfinite(m2)]

    if len(m1_valid) == 0 or len(m2_valid) == 0:
        print(
            f"\n--- Statistical Comparison: {method1_name} vs {method2_name} ---"
        )
        print("  Insufficient data for comparison")
        return {
            "test": "mann_whitney_u",
            "statistic": np.nan,
            "p_value": np.nan,
            "significant": False,
            "better_method": None,
            "error": "Insufficient data",
        }

    # Use absolute errors for comparison
    m1_abs = np.abs(m1_valid)
    m2_abs = np.abs(m2_valid)

    stat, p_value = stats.mannwhitneyu(m1_abs, m2_abs, alternative="two-sided")

    mae1 = float(np.mean(m1_abs))
    mae2 = float(np.mean(m2_abs))

    result = {
        "test": "mann_whitney_u",
        "statistic": float(stat),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
        "better_method": method1_name if mae1 < mae2 else method2_name,
        "mae1": mae1,
        "mae2": mae2,
    }

    print(f"\n--- Statistical Comparison: {method1_name} vs {method2_name} ---")
    print(f"  {method1_name}: MAE={mae1:.4f}, N={len(m1_abs)}")
    print(f"  {method2_name}: MAE={mae2:.4f}, N={len(m2_abs)}")
    print(f"  Mann-Whitney U: stat={stat:.1f}, p={p_value:.4f}")
    if p_value < 0.05:
        better = result["better_method"]
        print(f"  → Significant difference (p<0.05): {better} is better")
    else:
        print("  → No significant difference (p≥0.05)")

    return result


def save_results_with_provenance(
    df: pd.DataFrame,
    output_path: Path,
    provenance: Provenance,
) -> None:
    """Save results DataFrame with provenance metadata.

    Parameters
    ----------
    df : pd.DataFrame
        Results dataframe
    output_path : Path
        Output CSV path
    provenance : Provenance
        Provenance information
    """
    # Add provenance as metadata comment at top of CSV
    with open(output_path, "w") as f:
        f.write("# Benchmark Results\n")
        f.write(f"# Timestamp: {provenance.timestamp}\n")
        f.write(f"# Git commit: {provenance.git_commit or 'unknown'}\n")
        f.write(f"# Script: {provenance.script_name}\n")
        f.write(f"# N trials: {provenance.n_trials}\n")
        for key, val in provenance.extra_info.items():
            f.write(f"# {key}: {val}\n")
        f.write("#\n")

    # Append data
    df.to_csv(output_path, mode="a", index=False)
    logger.info(f"Saved results to {output_path}")


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
