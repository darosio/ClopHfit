"""Noise model analysis utilities for prtecan data.

Notebook/dev-oriented entry point.

General-purpose residual diagnostics are implemented in
`clophfit.fitting.residuals` and re-exported here for backward compatibility.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from clophfit.fitting.residuals import (
    ResidualPoint,
    analyze_label_bias,
    compute_correlation_matrices,
    compute_residual_covariance,
    detect_adjacent_correlation,
    estimate_x_shift_statistics,
    extract_residual_points,
)

__all__ = [
    "ResidualPoint",
    "analyze_label_bias",
    "compute_correlation_matrices",
    "compute_residual_covariance",
    "detect_adjacent_correlation",
    "estimate_x_shift_statistics",
    "export_noise_parameters",
    "extract_residual_points",
    "simulate_correlated_noise",
]


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
