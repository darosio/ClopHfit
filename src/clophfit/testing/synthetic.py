"""Synthetic data generation for testing and benchmarking.

This module provides a unified API for generating synthetic pH titration datasets
with characteristics matching real experimental data from Tecan plate readers.

Primary function:
- make_dataset: Single unified function for all synthetic data generation

Backward-compatible aliases:
- make_simple_dataset, make_realistic_dataset, make_stress_dataset
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    cast,
)

import matplotlib.pyplot as plt
import numpy as np

from clophfit.fitting.data_structures import DataArray, Dataset
from clophfit.fitting.models import binding_1site

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

# Default pH values from real L2 dataset (high to low)
L2_PH_VALUES = np.array([8.92, 8.31, 7.76, 7.04, 6.56, 5.98, 5.47])
# Default pH errors from real L2 dataset (variable, instrument-specific)
L2_PH_ERRORS = np.array([0.005, 0.045, 0.087, 0.026, 0.16, 0.157, 0.157])

# Real data distributions from L4 dataset (92 wells)
# Used for randomize_signals option
# Updated from analysis of 250 valid wells across L1-L4 datasets
REAL_DATA_STATS: dict[str, dict[str, float]] = {
    "K": {"mean": 6.8, "std": 1.35, "min": 5.3, "max": 8.2},
    "S0_y1": {"mean": 634.0, "std": 567.0, "min": 30.0, "max": 2700.0},
    "S1_y1": {"mean": 1024.0, "std": 989.0, "min": 36.0, "max": 5142.0},
    "S0_y2": {"mean": 903.0, "std": 740.0, "min": 136.0, "max": 4416.0},
    "S1_y2": {"mean": 177.0, "std": 198.0, "min": 6.0, "max": 1212.0},
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


@dataclass
class StressScenario:
    """Configuration for stress test scenarios.

    Attributes
    ----------
    name : str
        Identifier for the scenario.
    description : str
        Human-readable description.
    outlier_prob : float
        Probability of adding outliers (0-1).
    outlier_magnitude : float
        Outlier magnitude in units of sigma.
    low_ph_drop_prob : float
        Probability of low-pH signal drop (acidic tail collapse).
    low_ph_drop_magnitude : float
        Fraction of signal to drop (0-1).
    noise_multiplier : float
        Multiplier for base noise level.
    saturation_prob : float
        Probability of masking points (saturation).
    x_error_large : float
        Additional random x-error (pH units).
    x_systematic_offset : float
        Systematic x-offset (pH units).
    seed : int
        Random seed for reproducibility.
    """

    name: str = "default"
    description: str = ""
    outlier_prob: float = 0.0
    outlier_magnitude: float = 3.0
    low_ph_drop_prob: float = 0.0
    low_ph_drop_magnitude: float = 0.6
    noise_multiplier: float = 1.0
    saturation_prob: float = 0.0
    x_error_large: float = 0.0
    x_systematic_offset: float = 0.0
    seed: int = 42


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
    rng: np.random.Generator, param: str, clip_min: float = 10.0
) -> float:
    """Sample a value from real data distribution (truncated normal)."""
    stats = REAL_DATA_STATS[param]
    val = rng.normal(stats["mean"], stats["std"])
    return float(np.clip(val, clip_min, stats["max"]))


def make_dataset(  # noqa: PLR0913, PLR0912, PLR0915, C901
    k: float | None = None,
    s0: dict[str, float] | float | None = None,
    s1: dict[str, float] | float | None = None,
    *,
    is_ph: bool = True,
    seed: int = 0,
    # Randomization option
    randomize_signals: bool = False,
    # Error model selection
    error_model: str = "realistic",
    # Simple error model: constant noise as fraction of dynamic range
    noise: float | dict[str, float] = 0.02,
    # Realistic error model: relative error with floor
    rel_error: float | dict[str, float] = 0.035,
    min_error: float | dict[str, float] = 10.0,
    # Physics error model: shot noise + buffer noise
    buffer_sd: float | dict[str, float] = 40.0,
    # Stress multiplier applied to all error models
    noise_multiplier: float = 1.0,
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
    seed : int
        Random seed for reproducibility.
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
    noise_multiplier : float
        Multiplier applied to all errors (stress factor).
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

    Stress test with outliers and high noise:

    >>> ds, truth = make_dataset(
    ...     k=7.0,
    ...     s0=500,
    ...     s1=1000,
    ...     outlier_prob=0.3,
    ...     noise_multiplier=3.0,
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

        # Apply noise multiplier (only if y_err is set)
        if y_err is not None:
            y_err *= noise_multiplier

        # Add Gaussian noise (only if y_err is set)
        y = clean + rng.normal(0, y_err) if y_err is not None else clean.copy()

        # Inject outlier (per-label probability, only if y_err is set)
        if y_err is not None and outlier_prob > 0 and rng.random() < outlier_prob:
            idx = int(rng.integers(len(x)))
            y[idx] += outlier_sigma * y_err[idx] * rng.choice([-1, 1])

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


# =============================================================================
# Backward-compatible wrapper functions
# =============================================================================


def make_simple_dataset(  # noqa: PLR0913
    k: float,
    s0: dict[str, float] | float,
    s1: dict[str, float] | float,
    *,
    is_ph: bool,
    noise: float = 0.02,
    seed: int = 0,
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


def make_realistic_dataset(  # noqa: PLR0913
    pka: float = 7.0,
    *,
    s0_y1: float = 700.0,
    s1_y1: float = 1200.0,
    s0_y2: float = 1000.0,
    s1_y2: float = 200.0,
    error_model: str = "realistic",
    rel_error: float | dict[str, float] = 0.035,
    min_error: float | dict[str, float] = 10.0,
    buffer_sd: float | dict[str, float] = 40.0,
    outlier_prob: float = 0.05,
    outlier_sigma: float = 4.0,
    low_ph_drop: bool = True,
    low_ph_drop_magnitude: float = 0.4,
    low_ph_drop6_prob: float = 0.2,
    seed: int | None = None,
) -> tuple[Dataset, TruthParams]:
    """Generate realistic synthetic pH titration data matching L2/L4/140220.

    Thin wrapper around make_dataset with realistic defaults for dual-channel data.
    """
    return make_dataset(
        k=pka,
        s0={"y1": s0_y1, "y2": s0_y2},
        s1={"y1": s1_y1, "y2": s1_y2},
        is_ph=True,
        seed=seed or 42,
        error_model=error_model,
        rel_error=rel_error,
        min_error=min_error,
        buffer_sd=buffer_sd,
        outlier_prob=outlier_prob,
        outlier_sigma=outlier_sigma,
        low_ph_drop=low_ph_drop,
        low_ph_drop_magnitude=low_ph_drop_magnitude,
        low_ph_drop6_prob=low_ph_drop6_prob,
    )


def make_stress_dataset(  # noqa: PLR0913, PLR0917
    scenario: StressScenario,
    pka: float = 7.0,
    s0_y1: float = 700.0,
    s1_y1: float = 1200.0,
    s0_y2: float = 1000.0,
    s1_y2: float = 200.0,
    *,
    error_model: str = "realistic",
    rel_error: float | dict[str, float] = 0.035,
    min_error: float | dict[str, float] = 10.0,
    buffer_sd: float | dict[str, float] = 40.0,
) -> tuple[Dataset, TruthParams]:
    """Generate synthetic data with configurable stress factors from a StressScenario.

    Thin wrapper around make_dataset that unpacks StressScenario fields.
    """
    return make_dataset(
        k=pka,
        s0={"y1": s0_y1, "y2": s0_y2},
        s1={"y1": s1_y1, "y2": s1_y2},
        is_ph=True,
        seed=scenario.seed,
        error_model=error_model,
        rel_error=rel_error,
        min_error=min_error,
        buffer_sd=buffer_sd,
        noise_multiplier=scenario.noise_multiplier,
        outlier_prob=scenario.outlier_prob,
        outlier_sigma=scenario.outlier_magnitude,
        low_ph_drop=scenario.low_ph_drop_prob > 0,
        low_ph_drop_magnitude=scenario.low_ph_drop_magnitude,
        saturation_prob=scenario.saturation_prob,
        x_error_large=scenario.x_error_large,
        x_systematic_offset=scenario.x_systematic_offset,
    )


# =============================================================================
# Pre-defined stress scenarios
# =============================================================================

STRESS_SCENARIOS = {
    "clean": StressScenario(
        name="Clean",
        description="Baseline: no stress factors",
    ),
    "high_noise": StressScenario(
        name="HighNoise",
        description="3x normal noise level",
        noise_multiplier=3.0,
    ),
    "outliers_10pct": StressScenario(
        name="Outliers-10%",
        description="10% outliers (3-sigma magnitude)",
        outlier_prob=0.10,
        outlier_magnitude=3.0,
    ),
    "outliers_30pct": StressScenario(
        name="Outliers-30%",
        description="30% outliers (3-sigma magnitude)",
        outlier_prob=0.30,
        outlier_magnitude=3.0,
    ),
    "ph_drop": StressScenario(
        name="pH-Drop",
        description="Low-pH signal drop (60% reduction)",
        low_ph_drop_prob=1.0,
        low_ph_drop_magnitude=0.6,
    ),
    "saturation": StressScenario(
        name="Saturation",
        description="20% saturated points (masked)",
        saturation_prob=0.20,
    ),
    "combined_moderate": StressScenario(
        name="Combined-Moderate",
        description="15% outliers + high noise + pH drop",
        outlier_prob=0.15,
        noise_multiplier=2.0,
        low_ph_drop_prob=0.5,
        low_ph_drop_magnitude=0.5,
    ),
    "combined_severe": StressScenario(
        name="Combined-Severe",
        description="30% outliers + very high noise + pH drop + saturation",
        outlier_prob=0.30,
        outlier_magnitude=4.0,
        noise_multiplier=4.0,
        low_ph_drop_prob=1.0,
        low_ph_drop_magnitude=0.7,
        saturation_prob=0.15,
    ),
    "x_error_large": StressScenario(
        name="X-Error-Large",
        description="Large x-errors (±0.3 pH units)",
        x_error_large=0.3,
    ),
    "x_systematic": StressScenario(
        name="X-Systematic",
        description="Systematic x-offset (+0.5 pH)",
        x_systematic_offset=0.5,
    ),
}


# =============================================================================
# Plotting utility
# =============================================================================


def plot_synthetic_dataset(
    ds: Dataset,
    truth: TruthParams | None = None,
    *,
    title: str | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Plot a synthetic dataset with optional true curves.

    Parameters
    ----------
    ds : Dataset
        The dataset to plot.
    truth : TruthParams | None
        Ground truth parameters. If provided, plots the true binding curves.
    title : str | None
        Plot title.
    ax : Axes | None
        Axes to plot on. If None, creates a new figure.

    Returns
    -------
    Figure
        The figure containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = cast("Figure", ax.get_figure())

    colors = {"y1": "tab:blue", "y2": "tab:orange", "y0": "tab:blue"}

    for label, da in ds.items():
        color = colors.get(label)
        ax.errorbar(
            da.x,
            da.y,
            yerr=da.y_err,
            xerr=da.x_err,
            fmt="o",
            label=f"{label} data",
            color=color,
            capsize=3,
            alpha=0.8,
        )
        # Plot masked points
        if not np.all(da.mask):
            ax.plot(
                da.xc[~da.mask],
                da.yc[~da.mask],
                "x",
                color=color,
                markersize=10,
                alpha=0.5,
                label=f"{label} masked",
            )

    if truth is not None:
        x_fine = np.linspace(
            ds[next(iter(ds.keys()))].x.min() - 0.5,
            ds[next(iter(ds.keys()))].x.max() + 0.5,
            100,
        )
        for label in ds:
            s0_val = truth.S0.get(label, truth.S0.get("y0"))
            s1_val = truth.S1.get(label, truth.S1.get("y0"))
            if s0_val is not None and s1_val is not None:
                y_true = binding_1site(x_fine, truth.K, s0_val, s1_val, is_ph=ds.is_ph)
                ax.plot(
                    x_fine,
                    y_true,
                    "--",
                    color=colors.get(label),
                    alpha=0.6,
                    label=f"{label} true (K={truth.K:.2f})",
                )

    ax.set_xlabel("pH" if ds.is_ph else "[Cl] (mM)")
    ax.set_ylabel("Signal")
    if title:
        ax.set_title(title)
    ax.legend(loc="best", fontsize="small")
    ax.grid(visible=True, alpha=0.3)

    return fig


__all__ = [
    "L2_PH_ERRORS",
    "L2_PH_VALUES",
    "REAL_DATA_LOG_COV",
    "REAL_DATA_LOG_MEAN",
    "REAL_DATA_STATS",
    "STRESS_SCENARIOS",
    "StressScenario",
    "TruthParams",
    "make_dataset",
    "make_realistic_dataset",
    "make_simple_dataset",
    "make_stress_dataset",
    "plot_synthetic_dataset",
]
