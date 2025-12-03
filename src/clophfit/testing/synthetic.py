"""Synthetic data generation for testing and benchmarking.

This module provides a unified API for generating synthetic pH titration datasets
with characteristics matching real experimental data from Tecan plate readers.

Primary functions:
- make_dataset: Unified function for all synthetic data generation
- make_simple_dataset: Simplified interface for unit tests
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from clophfit.fitting.data_structures import DataArray, Dataset
from clophfit.fitting.models import binding_1site

# Default pH values from real L2 dataset (high to low)
L2_PH_VALUES = np.array([8.92, 8.31, 7.76, 7.04, 6.56, 5.98, 5.47])
# Default pH errors from real L2 dataset (variable, instrument-specific)
L2_PH_ERRORS = np.array([0.005, 0.045, 0.087, 0.026, 0.16, 0.157, 0.157])

# Real data distributions from L4 dataset (92 wells)
# Used for randomize_signals option
# Updated from analysis of 250 valid wells across L1-L4 datasets
REAL_DATA_STATS: dict[str, dict[str, float]] = {
    "K": {"mean": 6.8, "std": 1.35, "min": 5.3, "max": 8.2},
    "S0_y1": {"mean": 634.0, "std": 567.0 / 2, "min": 30.0, "max": 2700.0},
    "S1_y1": {"mean": 1024.0, "std": 989.0 / 2, "min": 36.0, "max": 5142.0},
    "S0_y2": {"mean": 903.0, "std": 740.0 / 2, "min": 136.0, "max": 4416.0},
    "S1_y2": {"mean": 177.0, "std": 198.0 / 2, "min": 6.0, "max": 1212.0},
}

# Log-space covariance matrix for correlated signal sampling
# Rows/cols: [log10(S0_y1), log10(S1_y1), log10(S0_y2), log10(S1_y2)]
# From real data: strong correlation between y1 and y2 signals (r~0.76-0.78)
REAL_DATA_LOG_MEAN = np.array([2.54, 2.70, 2.80, 1.95])  # log10 scale
REAL_DATA_LOG_COV = np.array([
    [0.329, 0.339, 0.188, 0.256],  # log_S0_y1
    [0.339, 0.408, 0.213, 0.305],  # log_S1_y1
    [0.188, 0.213, 0.185, 0.192],  # log_S0_y2
    [0.256, 0.305, 0.192, 0.372],  # log_S1_y2
])


@dataclass
class TruthParams:
    """Ground truth parameters for synthetic data."""

    K: float
    S0: dict[str, float]
    S1: dict[str, float]


def _sample_correlated_signals(rng: np.random.Generator) -> dict[str, float]:
    """Sample correlated S0/S1 signals for y1/y2 from real data distribution.

    Uses multivariate normal in log-space to capture correlations
    observed in real experimental data (r~0.76-0.78 between y1 and y2).
    """
    # Sample in log-space
    log_samples = rng.multivariate_normal(REAL_DATA_LOG_MEAN, REAL_DATA_LOG_COV)

    # Convert to linear scale with clipping
    signals = 10.0**log_samples
    return {
        "S0_y1": float(np.clip(signals[0], 30, 3000)),
        "S1_y1": float(np.clip(signals[1], 36, 5500)),
        "S0_y2": float(np.clip(signals[2], 100, 4500)),
        "S1_y2": float(np.clip(signals[3], 5, 1300)),
    }


def _sample_from_real(
    rng: np.random.Generator, param: str, clip_min: float | None = None
) -> float:
    """Sample a value from real data distribution (truncated normal)."""
    stats = REAL_DATA_STATS[param]
    val = rng.normal(stats["mean"], stats["std"])

    # Determine the actual lower bound for clipping
    # Use the larger of the user-provided clip_min or the stats' min
    lower_bound = max(stats["min"], clip_min if clip_min is not None else -np.inf)

    return float(np.clip(val, lower_bound, stats["max"]))


def make_dataset(  # noqa: PLR0913, PLR0912, PLR0915, C901
    k: float | None = None,
    s0: dict[str, float] | float | None = None,
    s1: dict[str, float] | float | None = None,
    *,
    is_ph: bool = True,
    seed: int | None = None,
    # Randomization option
    randomize_signals: bool = False,
    # Error model selection
    error_model: str = "realistic",
    # Simple error model: constant noise as fraction of dynamic range
    noise: float | dict[str, float] = 0.02,
    # Realistic error model: relative error with floor
    rel_error: float | dict[str, float] = 0.035,
    min_error: float | dict[str, float] = 1.0,
    # Physics error model: shot noise + buffer noise
    buffer_sd: float | dict[str, float] = 10.0,
    # Outlier injection
    outlier_prob: float = 0.0,
    outlier_sigma: float = 4.0,
    # Low-pH drop (acidic tail collapse)
    low_ph_drop: bool = False,
    low_ph_drop_magnitude: float = 0.4,
    low_ph_drop6_prob: float = 0.2,
    # Saturation/masking
    saturation_prob: float = 0.0,
    # X-axis perturbations
    x_error_large: float = 0.0,
    x_systematic_offset: float = 0.0,
    rel_x_err: float = 0.01,
) -> tuple[Dataset, TruthParams]:
    """Generate synthetic pH/Cl titration data with configurable complexity.

    This is the single unified function for all synthetic data generation.
    It supports per-label error scaling and optional randomization of signal
    parameters based on real experimental data distributions.

    Parameters
    ----------
    k : float | None
        Equilibrium constant (pKa for pH, Kd for Cl). If None and randomize_signals
        is True, sampled from real data distribution.
    s0 : dict[str, float] | float | None
        Signal at unbound state. Use dict for multiple labels: {"y1": 700, "y2": 1000}.
        If None and randomize_signals is True, sampled from real data distribution.
    s1 : dict[str, float] | float | None
        Signal at bound state. Use dict for multiple labels: {"y1": 1200, "y2": 200}.
        If None and randomize_signals is True, sampled from real data distribution.
    is_ph : bool
        True for pH titration, False for Cl titration.
    seed : int | None
        Random seed for reproducibility. If None (default), generates random data.
    randomize_signals : bool
        If True, randomize K, S0, S1 from real L4 data distributions when not provided.
        Creates y1/y2 dual-channel data with realistic signal magnitudes and ranges.
    error_model : str
        Error model to use:
        - "simple": Constant noise as fraction of dynamic range (uses `noise`).
        - "realistic": Relative error with floor (uses `rel_error`, `min_error`).
        - "physics": Shot noise + buffer noise (uses `buffer_sd`).
    noise : float | dict[str, float]
        For "simple" model: relative noise as fraction of dynamic range.
        Use dict for per-label: {"y1": 0.05, "y2": 0.02}.
    rel_error : float | dict[str, float]
        For "realistic" model: relative error as fraction of signal.
        Use dict for per-label: {"y1": 0.07, "y2": 0.025} for 3x y1/y2 ratio.
    min_error : float | dict[str, float]
        For "realistic" model: minimum error floor (instrument noise).
    buffer_sd : float | dict[str, float]
        For "physics" model: buffer SD where err = sqrt(signal + buffer_sd^2).
        Use dict for per-label: {"y1": 200, "y2": 40} for 5x y1/y2 ratio.
    outlier_prob : float
        Probability of random outlier per label (0-1).
    outlier_sigma : float
        Outlier magnitude in units of local error.
    low_ph_drop : bool
        Simulate acidic tail collapse at lowest pH.
    low_ph_drop_magnitude : float
        Fraction of signal to drop at lowest pH (0-1).
    low_ph_drop6_prob : float
        Probability of additional drop at pH ~6.0.
    saturation_prob : float
        Probability of masking points (saturation).
    x_error_large : float
        Additional random x-error (pH units).
    x_systematic_offset : float
        Systematic x-offset (pH units).
    rel_x_err : float
        Relative x-error for Cl titrations (ignored for pH).

    Returns
    -------
    Dataset
        Generated dataset with specified labels.
    TruthParams
        Ground truth parameters (K, S0, S1).

    Examples
    --------
    Simple single-channel for unit tests:

    >>> ds, truth = make_dataset(7.0, 100, 1000, error_model="simple", noise=0.02)

    Randomized dual-channel matching real data distributions:

    >>> ds, truth = make_dataset(randomize_signals=True, seed=42)

    Realistic dual-channel with differential errors (y1 3x noisier):

    >>> ds, truth = make_dataset(
    ...     k=7.0,
    ...     s0={"y1": 700, "y2": 1000},
    ...     s1={"y1": 1200, "y2": 200},
    ...     rel_error={"y1": 0.07, "y2": 0.025},
    ... )

    Stress test with outliers and high noise (increase rel_error directly):

    >>> ds, truth = make_dataset(
    ...     k=7.0,
    ...     s0=500,
    ...     s1=1000,
    ...     outlier_prob=0.3,
    ...     rel_error=0.10,
    ... )
    """
    rng = np.random.default_rng(seed)

    # Handle randomize_signals mode with correlated sampling
    if randomize_signals:
        if k is None:
            k = _sample_from_real(rng, "K", clip_min=5.0)
        if s0 is None or s1 is None:
            # Sample correlated signals for realistic y1/y2 relationship
            corr_signals = _sample_correlated_signals(rng)
            if s0 is None:
                s0 = {"y1": corr_signals["S0_y1"], "y2": corr_signals["S0_y2"]}
            if s1 is None:
                s1 = {"y1": corr_signals["S1_y1"], "y2": corr_signals["S1_y2"]}
    else:
        # Require k, s0, s1 when not randomizing
        if k is None:
            k = 7.0
        if s0 is None:
            s0 = {"y0": 500.0}
        if s1 is None:
            s1 = {"y0": 1000.0}

    # Convert scalar inputs to dicts
    if not isinstance(s0, dict):
        s0 = {"y0": float(s0)}
    if not isinstance(s1, dict):
        s1 = {"y0": float(s1)}

    # Determine labels from s0/s1
    all_labels = set(s0.keys()) | set(s1.keys())

    # Helper to expand scalar or partial dict to full per-label dict
    def to_label_dict(
        val: float | dict[str, float], default: float
    ) -> dict[str, float]:
        if isinstance(val, dict):
            fallback = next(iter(val.values()), default)
            return {lbl: val.get(lbl, fallback) for lbl in all_labels}
        return dict.fromkeys(all_labels, val)

    noise_d = to_label_dict(noise, 0.02)
    rel_error_d = to_label_dict(rel_error, 0.035)
    min_error_d = to_label_dict(min_error, 10.0)
    buffer_sd_d = to_label_dict(buffer_sd, 40.0)

    # Set up x values and errors based on titration type
    if is_ph:
        x = L2_PH_VALUES.copy()
        x_err = L2_PH_ERRORS.copy()
    else:
        x = np.array([0.01, 5.0, 10.0, 20.0, 40.0, 80.0, 150.0])
        x_err = np.where(x == 0, 0.01, np.maximum(0.01, rel_x_err * x))

    # Apply x-space perturbations
    x_measured = x.copy()
    if x_error_large > 0:
        x_measured += rng.normal(0, x_error_large, size=len(x))
    if x_systematic_offset != 0:
        x_measured += x_systematic_offset

    # Build dataset
    ds = Dataset({}, is_ph=is_ph)
    truth_s0: dict[str, float] = {}
    truth_s1: dict[str, float] = {}

    for label in sorted(all_labels):
        # Get truth parameters with fallback
        truth_s0[label] = s0.get(label, s0.get("y0", 1.0))
        truth_s1[label] = s1.get(label, s1.get("y0", 1.0))

        # Generate clean signal
        clean = binding_1site(x, k, truth_s0[label], truth_s1[label], is_ph=is_ph)

        # Compute errors based on error model
        if error_model == "simple":
            dy = noise_d[label] * (np.max(clean) - np.min(clean))
            # For simple model with zero noise, don't set y_err (use None)
            y_err = np.full_like(clean, dy) if dy > 0 else None
        elif error_model == "physics":
            y_err = np.sqrt(np.maximum(clean, 1.0) + buffer_sd_d[label] ** 2)
        else:  # realistic (default)
            y_err = np.maximum(rel_error_d[label] * np.abs(clean), min_error_d[label])

        # Add Gaussian noise (only if y_err is set)
        y = clean + rng.normal(0, y_err) if y_err is not None else clean.copy()

        # Inject outlier (per-point probability, only if y_err is set)
        if y_err is not None and outlier_prob > 0:
            # Each point has independent chance to be an outlier
            out_mask = rng.random(size=len(x)) < outlier_prob
            if np.any(out_mask):
                signs = rng.choice([-1, 1], size=len(x))
                y += out_mask * (outlier_sigma * y_err * signs)

        # Acidic tail collapse
        if is_ph and low_ph_drop:
            idx_low = int(np.argmin(x))
            y[idx_low] *= 1.0 - low_ph_drop_magnitude
            if low_ph_drop6_prob > 0 and rng.random() < low_ph_drop6_prob:
                idx6 = int(np.argmin(np.abs(x - 6.0)))
                y[idx6] *= 1.0 - low_ph_drop_magnitude * 0.6

        # Create DataArray (y_errc only if y_err is set)
        if y_err is not None:
            da = DataArray(xc=x_measured, yc=y, x_errc=x_err, y_errc=y_err)
        else:
            da = DataArray(xc=x_measured, yc=y, x_errc=x_err)

        # Apply saturation mask
        if saturation_prob > 0:
            mask = np.ones(len(x), dtype=bool)
            n_sat = max(1, int(len(x) * saturation_prob))
            mask[rng.choice(len(x), size=n_sat, replace=False)] = False
            da.mask = mask

        ds[label] = da

    return ds, TruthParams(K=k, S0=truth_s0, S1=truth_s1)


def make_simple_dataset(  # noqa: PLR0913
    k: float,
    s0: dict[str, float] | float,
    s1: dict[str, float] | float,
    *,
    is_ph: bool,
    noise: float = 0.02,
    seed: int | None = None,
    rel_x_err: float = 0.01,
) -> tuple[Dataset, TruthParams]:
    """Create a simple synthetic Dataset for unit tests.

    Uses fixed x-values and simple noise model for backward compatibility
    with existing tests. Does NOT set y_err when noise=0 to allow fitters
    to use default weighting.
    """
    if not isinstance(s0, dict):
        s0 = {"y0": float(s0)}
    if not isinstance(s1, dict):
        s1 = {"y0": float(s1)}

    rng = np.random.default_rng(seed)

    if is_ph:
        x = np.array([5.0, 5.8, 6.6, 7.0, 7.8, 8.2, 9.0])
        x_err = np.full_like(x, 0.05)
    else:
        x = np.array([0.01, 5.0, 10.0, 20.0, 40.0, 80.0, 150.0])
        x_err = np.where(x == 0, 0.01, np.maximum(0.01, rel_x_err * x))

    ds = Dataset({}, is_ph=is_ph)
    for lbl in sorted(s0.keys()):
        clean = binding_1site(x, k, s0[lbl], s1[lbl], is_ph=is_ph)
        dy = noise * (np.max(clean) - np.min(clean))
        y = clean + rng.normal(0.0, dy, size=x.shape)
        # Note: y_err is NOT set to allow fitters to use default weighting
        da = DataArray(xc=x, yc=y, x_errc=x_err)
        ds[lbl] = da

    return ds, TruthParams(K=k, S0=s0, S1=s1)


__all__ = [
    "L2_PH_ERRORS",
    "L2_PH_VALUES",
    "REAL_DATA_LOG_COV",
    "REAL_DATA_LOG_MEAN",
    "REAL_DATA_STATS",
    "TruthParams",
    "make_dataset",
    "make_simple_dataset",
]
