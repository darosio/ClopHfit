"""Development utilities for ClopHfit.

This package contains experimental code and analysis tools that are not yet
part of the main package.

For general residual extraction, use `clophfit.fitting.residuals` instead.
The functions here are specific to noise model characterization.
"""

from .noise_models import (
    ResidualPoint,
    analyze_label_bias,
    compute_correlation_matrices,
    compute_residual_covariance,
    detect_adjacent_correlation,
    estimate_x_shift_statistics,
    export_noise_parameters,
    extract_residual_points,
    simulate_correlated_noise,
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
