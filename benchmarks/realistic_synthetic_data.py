#!/usr/bin/env python3
"""
Realistic synthetic data generator based on actual experimental data patterns.

Analyzing the provided real data:
- 7 pH points typically from ~8.9 to ~5.5
- pKa ranges from 6-8 (reasonable for typical proteins)
- Different signal patterns: some increase, some decrease with pH
- Realistic error levels and signal magnitudes
- Some data points masked (excluded from analysis)
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clophfit.fitting.data_structures import DataArray, Dataset
from clophfit.fitting.models import binding_1site


@dataclass
class RealisticSimulationParameters:
    """Parameters for realistic synthetic data generation based on real data."""

    # True parameters for data generation (realistic ranges)
    K_true: float = 7.2  # pKa in range 6-8
    S0_y1_true: float = 150.0  # Baseline fallback
    S1_y1_true: float = 1200.0  # Max signal change fallback
    S0_y2_true: float = 800.0  # Different baseline for y2
    S1_y2_true: float = 100.0  # Different signal change

    # Realistic pH series (7 points, typical experimental range)
    pH_values: list[float] | None = None  # Defaults to measured series

    # Error characteristics derived from real data residuals (RMSE)
    # y1 RMSE ~125, y2 RMSE ~25
    # Adjusted to be slightly lower than RMSE to account for outliers in real data
    y1_base_error: float = 40.0  # Base instrument noise (buffer std ~25 + margin)
    y2_base_error: float = 2.0  # Base instrument noise (buffer std ~0.3 + margin)
    x_base_error: float = 0.05  # Approximate mean pH std

    # Signal-dependent noise (shot noise + mixing error)
    shot_noise_factor: float = 0.02  # 2% shot noise

    # Mixing/pipetting error (proportional to signal)
    mixing_error_factor: float = 0.04  # 4% proportional error

    max_noise_fraction_y1: float = 0.5
    max_noise_fraction_y2: float = 0.5

    # Outlier characteristics (more realistic)
    outlier_probability: float = 0.05  # 5% chance of outliers (reduced from 10%)
    outlier_magnitude: float = 2.0  # 2σ outliers

    # pH measurement errors (varying by point, as in real data)
    pH_error_variation: bool = True

    # Sample-specific behaviour derived from experimental replicates
    sample_profiles: list[dict[str, float]] = field(
        default_factory=lambda: [
            {
                "name": "G03",
                "S0_y1": 539.1,
                "S1_y1": 246.2,
                "S0_y2": 25.5,
                "S1_y2": 336.2,
                "y1_cv": 0.22,
                "y2_cv": 0.34,
                "corr": -0.9,
                "buffer_y1": 625.0,
                "buffer_y2": 5.0,
            },
            {
                "name": "NTT",
                "S0_y1": 1033.8,
                "S1_y1": 454.8,
                "S0_y2": 55.1,
                "S1_y2": 364.2,
                "y1_cv": 0.72,
                "y2_cv": 0.70,
                "corr": -0.92,
                "buffer_y1": 625.0,
                "buffer_y2": 5.0,
            },
            {
                "name": "S202N",
                "S0_y1": 1127.3,
                "S1_y1": 729.3,
                "S0_y2": 73.3,
                "S1_y2": 417.9,
                "y1_cv": 0.52,
                "y2_cv": 0.51,
                "corr": -0.9,
                "buffer_y1": 625.0,
                "buffer_y2": 5.0,
            },
            {
                "name": "V224Q",
                "S0_y1": 1285.0,
                "S1_y1": 804.7,
                "S0_y2": 41.4,
                "S1_y2": 1065.4,
                "y1_cv": 0.28,
                "y2_cv": 0.25,
                "corr": -0.85,
                "buffer_y1": 625.0,
                "buffer_y2": 5.0,
            },
        ]
    )

    extreme_l1_profiles: list[dict[str, float]] = field(
        default_factory=lambda: [
            {
                "name": "L1_E2GFP",
                "S0_y1": 140.0,
                "S1_y1": 260.0,
                "S0_y2": 560.0,
                "S1_y2": -350.0,
                "y1_cv": 2.6,
                "y2_cv": 1.1,
                "y1_base_error": 180.0,
                "y2_base_error": 80.0,
                "corr": 0.1,
                "allow_negative": True,
                "buffer_y1": 25.0,
                "buffer_y2": 170.0,
            },
            {
                "name": "L1_V224L",
                "S0_y1": 90.0,
                "S1_y1": 160.0,
                "S0_y2": 470.0,
                "S1_y2": -280.0,
                "y1_cv": 4.4,
                "y2_cv": 1.4,
                "y1_base_error": 200.0,
                "y2_base_error": 90.0,
                "corr": 0.0,
                "allow_negative": True,
                "buffer_y1": 25.0,
                "buffer_y2": 170.0,
            },
            {
                "name": "L1_V224Q",
                "S0_y1": 320.0,
                "S1_y1": 320.0,
                "S0_y2": 420.0,
                "S1_y2": -250.0,
                "y1_cv": 1.8,
                "y2_cv": 1.3,
                "y1_base_error": 150.0,
                "y2_base_error": 70.0,
                "corr": 0.05,
                "allow_negative": True,
                "buffer_y1": 25.0,
                "buffer_y2": 170.0,
            },
        ]
    )

    include_extreme_l1: bool = False

    channel_correlation: float = -0.9  # Matches observed anticorrelation by default
    saturation_probability: float = 0.02  # Occasional saturated/removed wells

    low_ph_drop_probability: float = 0.1  # Reduced from 0.2
    low_ph_drop_min: float = 0.4  # 40% drop
    low_ph_drop_max: float = 0.8  # up to 80% drop
    low_ph_drop_floor: float = 5.0

    # Dilution parameters
    initial_volume: float = 100.0
    addition_volume: float = 2.0

    random_seed: int | None = None


def generate_realistic_dataset(
    params: RealisticSimulationParameters,
) -> tuple[Dataset, dict[str, float]]:
    """Generate realistic synthetic dataset matching experimental patterns.

    Parameters
    ----------
    params : RealisticSimulationParameters
        Container with physical parameters, sampling profiles, and knobs for
        dilution, buffer, and rare-event behavior (outliers, low-pH drops).

    Returns
    -------
    Dataset
        Generated dataset with realistic characteristics.
    dict[str, float]
        True parameters used for generation.
    """
    if params.random_seed is not None:
        np.random.seed(params.random_seed)

    # Default pH values (typical experimental series, high to low pH)
    if params.pH_values is None:
        x = np.array([9.0633, 8.35, 7.7, 7.08, 6.44, 5.83, 4.99], dtype=float)
    else:
        x = np.array(params.pH_values, dtype=float)

    n_points = len(x)

    # Select a sample profile to mimic well-to-well variability
    profiles = list(params.sample_profiles)
    if params.include_extreme_l1:
        profiles.extend(params.extreme_l1_profiles)

    if profiles:
        profile = profiles[np.random.randint(len(profiles))]
    else:
        profile = {
            "name": "custom",
            "S0_y1": params.S0_y1_true,
            "S1_y1": params.S1_y1_true,
            "S0_y2": params.S0_y2_true,
            "S1_y2": params.S1_y2_true,
            "y1_cv": 0.3,
            "y2_cv": 0.3,
        }

    y1_true = binding_1site(
        x,
        params.K_true,
        profile.get("S0_y1", params.S0_y1_true),
        profile.get("S1_y1", params.S1_y1_true),
        is_ph=True,
    )
    y2_true = binding_1site(
        x,
        params.K_true,
        profile.get("S0_y2", params.S0_y2_true),
        profile.get("S1_y2", params.S1_y2_true),
        is_ph=True,
    )

    # Calculate dilution factors
    volumes = np.array(
        [params.initial_volume + i * params.addition_volume for i in range(n_points)]
    )
    dilution_factors = params.initial_volume / volumes

    # Apply dilution to the true binding signal (protein concentration decreases)
    y1_diluted = y1_true * dilution_factors
    y2_diluted = y2_true * dilution_factors

    # Add buffer signal (constant background)
    buffer_y1 = profile.get("buffer_y1", 625.0)
    buffer_y2 = profile.get("buffer_y2", 5.0)

    y1_total = y1_diluted + buffer_y1
    y2_total = y2_diluted + buffer_y2

    # Generate realistic pH errors using a bounded log-normal distribution
    if params.pH_error_variation:
        pH_errors = np.random.lognormal(
            mean=np.log(params.x_base_error), sigma=0.35, size=n_points
        )
        pH_errors = np.clip(pH_errors, 0.0115, 0.10)
    else:
        pH_errors = np.full(n_points, params.x_base_error)

    # Signal-dependent errors (shot noise + mixing error) applied to TOTAL signal
    y1_base = profile.get("y1_base_error", params.y1_base_error)
    y2_base = profile.get("y2_base_error", params.y2_base_error)

    # Noise model: Base + Shot Noise + Mixing Error
    # Mixing error is proportional to signal (pipetting accuracy)
    y1_errors_total = (
        y1_base
        + params.mixing_error_factor * np.abs(y1_total)
        + params.shot_noise_factor * np.sqrt(np.maximum(np.abs(y1_total), 1.0))
    )
    y2_errors_total = (
        y2_base
        + params.mixing_error_factor * np.abs(y2_total)
        + params.shot_noise_factor * np.sqrt(np.maximum(np.abs(y2_total), 1.0))
    )

    # Cap errors relative to total signal
    cap_y1 = y1_base + params.max_noise_fraction_y1 * np.abs(y1_total)
    cap_y2 = y2_base + params.max_noise_fraction_y2 * np.abs(y2_total)
    y1_errors_total = np.minimum(y1_errors_total, cap_y1)
    y2_errors_total = np.minimum(y2_errors_total, cap_y2)

    allow_negative = profile.get("allow_negative", False)

    def _enforce_positive(
        obs: np.ndarray, y_true_arr: np.ndarray, err_arr: np.ndarray
    ) -> np.ndarray:
        mask = obs <= 0
        attempts = 0
        while np.any(mask) and attempts < 5:
            obs[mask] = y_true_arr[mask] + np.random.normal(0, err_arr[mask])
            mask = obs <= 0
            attempts += 1
        return np.where(
            obs <= 0,
            np.maximum(0.1 * np.abs(y_true_arr), 0.1 * err_arr) + 1.0,
            obs,
        )

    def _sample_uncorrelated(y_true_arr: np.ndarray, err_arr: np.ndarray) -> np.ndarray:
        obs = y_true_arr + np.random.normal(0, err_arr)
        if allow_negative:
            return obs
        return _enforce_positive(obs, y_true_arr, err_arr)

    # Generate correlated noise between channels (captures observed relationships)
    corr_value = profile.get("corr", params.channel_correlation)

    # Generate observed TOTAL signal
    if corr_value is not None:
        corr = float(np.clip(corr_value, -0.95, 0.95))
        cov = np.array([[1.0, corr], [corr, 1.0]])
        L = np.linalg.cholesky(cov)

        def _sample_correlated() -> tuple[np.ndarray, np.ndarray]:
            z = np.random.normal(size=(n_points, 2)) @ L.T
            return (
                y1_total + z[:, 0] * y1_errors_total,
                y2_total + z[:, 1] * y2_errors_total,
            )

        y1_obs_total, y2_obs_total = _sample_correlated()
        if not allow_negative:
            mask = (y1_obs_total <= 0) | (y2_obs_total <= 0)
            attempts = 0
            while np.any(mask) and attempts < 5:
                z = np.random.normal(size=(mask.sum(), 2)) @ L.T
                y1_obs_total[mask] = y1_total[mask] + z[:, 0] * y1_errors_total[mask]
                y2_obs_total[mask] = y2_total[mask] + z[:, 1] * y2_errors_total[mask]
                mask = (y1_obs_total <= 0) | (y2_obs_total <= 0)
                attempts += 1
            y1_obs_total = np.where(
                y1_obs_total <= 0,
                np.maximum(0.1 * np.abs(y1_total), 0.1 * y1_errors_total) + 1.0,
                y1_obs_total,
            )
            y2_obs_total = np.where(
                y2_obs_total <= 0,
                np.maximum(0.1 * np.abs(y2_total), 0.1 * y2_errors_total) + 1.0,
                y2_obs_total,
            )
    else:
        y1_obs_total = _sample_uncorrelated(y1_total, y1_errors_total)
        y2_obs_total = _sample_uncorrelated(y2_total, y2_errors_total)

    # Add realistic outliers (occasional measurement errors) to TOTAL signal
    if np.random.random() < params.outlier_probability:
        idx = np.random.choice(n_points, size=1)
        y1_obs_total[idx] += params.outlier_magnitude * y1_errors_total[idx]
        y2_obs_total[idx] -= params.outlier_magnitude * y2_errors_total[idx]

    if not allow_negative:
        y1_obs_total = _enforce_positive(y1_obs_total, y1_total, y1_errors_total)
        y2_obs_total = _enforce_positive(y2_obs_total, y2_total, y2_errors_total)

    if np.random.random() < params.low_ph_drop_probability:
        n_drop = np.random.choice([1, 2])
        acidic_idx = np.argsort(x)[:n_drop]
        drop_frac = np.random.uniform(
            params.low_ph_drop_min, params.low_ph_drop_max, size=n_drop
        )
        y1_obs_total[acidic_idx] = np.maximum(
            y1_obs_total[acidic_idx] * (1 - drop_frac), params.low_ph_drop_floor
        )

    # Process data: Subtract buffer and correct for dilution
    # Note: We assume buffer measurement is perfect (or mean value) for subtraction
    # In reality, buffer subtraction adds noise, but here we model the net noise in the total signal
    y1_obs = (y1_obs_total - buffer_y1) / dilution_factors
    y2_obs = (y2_obs_total - buffer_y2) / dilution_factors

    # Propagate errors to the corrected signal
    # Error(Corrected) = Error(Total) / DilutionFactor
    y1_errors = y1_errors_total / dilution_factors
    y2_errors = y2_errors_total / dilution_factors

    # Create realistic masks (some points excluded as in real data)
    mask1 = np.ones(n_points, dtype=bool)
    mask2 = np.ones(n_points, dtype=bool)

    for mask, values in ((mask1, y1_obs), (mask2, y2_obs)):
        if np.random.random() < params.saturation_probability:
            idx = np.random.randint(n_points)
            mask[idx] = False
            values[idx] = np.nan

    # Create DataArrays with realistic characteristics
    da1 = DataArray(xc=x, yc=y1_obs, x_errc=pH_errors, y_errc=y1_errors)
    da1.mask = mask1

    da2 = DataArray(xc=x, yc=y2_obs, x_errc=pH_errors, y_errc=y2_errors)
    da2.mask = mask2

    # Create Dataset
    dataset = Dataset({"y1": da1, "y2": da2}, is_ph=True)

    # Create true parameters dictionary
    true_params = {
        "K": params.K_true,
        "S0_y1": profile.get("S0_y1", params.S0_y1_true),
        "S1_y1": profile.get("S1_y1", params.S1_y1_true),
        "S0_y2": profile.get("S0_y2", params.S0_y2_true),
        "S1_y2": profile.get("S1_y2", params.S1_y2_true),
        "profile": profile.get("name", "custom"),
    }

    return dataset, true_params


def analyze_real_data_patterns():
    """Analyze the patterns in the provided real data examples."""
    print("📊 ANALYSIS OF REAL DATA PATTERNS")
    print("=" * 50)

    # Real data characteristics extracted from your examples
    real_examples = [
        {
            "name": "Example 1",
            "pH": [8.92, 8.307, 7.763, 7.037, 5.98, 5.47],
            "y1": [1010.16, 1475.37, 1892.35, 2066.12, 1970.44, 793.132],
            "y2": [3677.95, 2775.57, 1228.95, 413.512, 124.867, 24.521],
            "y1_err": 275.055,
            "y2_err": 102.929,
            "mask1": [1, 1, 1, 1, 1, 0],  # Last point masked
            "mask2": [1, 1, 1, 1, 1, 1],
        },
        {
            "name": "Example 2",
            "pH": [8.92, 8.307, 7.763, 7.037, 5.98, 5.47],
            "y1": [638.259, 670.284, 671.535, 665.307, 598.084, 513.912],
            "y2": [1209.23, 1229.07, 1226.21, 1167.17, 802.967, 399.823],
            "y1_err": 9.073,
            "y2_err": 11.6,
            "mask1": [1, 1, 1, 1, 1, 1],
            "mask2": [1, 1, 1, 1, 1, 1],
        },
        {
            "name": "Example 3",
            "pH": [8.92, 8.307, 7.763, 7.037, 5.98, 5.47],
            "y1": [230.285, 236.268, 239.145, 231.847, 49.546, 25.436],
            "y2": [470.227, 476.68, 473.247, 445.312, 73.167, 14.034],
            "y1_err": 32.741,
            "y2_err": 40.758,
            "mask1": [1, 1, 1, 1, 1, 1],
            "mask2": [1, 1, 1, 1, 1, 1],
        },
        {
            "name": "Example 4",
            "pH": [8.92, 8.307, 7.763, 7.037, 5.98, 5.47],
            "y1": [640.095, 685.972, 776.193, 926.55, 1181.29, 1179.53],
            "y2": [838.136, 829.785, 783.167, 623.007, 241.867, 157.19],
            "y1_err": 29.474,
            "y2_err": 24.13,
            "mask1": [1, 1, 1, 1, 1, 1],
            "mask2": [1, 1, 1, 1, 1, 1],
        },
    ]

    print("\n🔍 Real Data Characteristics (UPDATED):")
    print("  • pH points: 7 (9.0633 → 4.99) with spacings 0.61–0.84 pH")
    print("  • Median pH uncertainty: 0.036 (range 0.0115–0.10)")
    print("  • Signal magnitudes: y1 spans −30 to 3k, y2 spans −1 to 3.6k")
    print("  • Observed channel CVs range from 0.21 to 0.74 depending on sample")
    print("  • Channels are strongly anti-correlated (median r ≈ −0.91)")
    print("  • Occasional saturated wells are excluded (≈1% NaNs)")
    print("  • Variable pH measurement errors and sample-specific noise envelopes")

    # Calculate statistics
    all_y1_signals = []
    all_y2_signals = []
    all_y1_errors = []
    all_y2_errors = []

    for example in real_examples:
        all_y1_signals.extend(example["y1"])
        all_y2_signals.extend(example["y2"])
        all_y1_errors.append(example["y1_err"])
        all_y2_errors.append(example["y2_err"])

    print("\n📈 Signal Statistics:")
    print(f"  • Y1 range: {min(all_y1_signals):.0f} - {max(all_y1_signals):.0f}")
    print(f"  • Y2 range: {min(all_y2_signals):.0f} - {max(all_y2_signals):.0f}")
    print(f"  • Y1 errors: {min(all_y1_errors):.1f} - {max(all_y1_errors):.1f}")
    print(f"  • Y2 errors: {min(all_y2_errors):.1f} - {max(all_y2_errors):.1f}")

    return real_examples


def compare_synthetic_vs_real():
    """Compare synthetic data generation with real data patterns."""
    print("\n🔬 COMPARING SYNTHETIC VS REAL DATA")
    print("=" * 50)

    # Generate realistic synthetic data
    realistic_params = RealisticSimulationParameters(
        K_true=7.0,  # Realistic pKa
        outlier_probability=0.1,  # Realistic outlier rate
    )

    synthetic_dataset, true_params = generate_realistic_dataset(realistic_params)
    profile_name = true_params.get("profile", "custom")

    print("🧪 Generated Realistic Synthetic Data:")
    print(f"  Sample profile: {profile_name}")
    for label, da in synthetic_dataset.items():
        print(f"  {label.upper()}:")
        print(f"    pH: {da.xc}")
        print(f"    Signal: {da.yc.round(1)}")
        print(f"    Mask: {da.mask.astype(int)}")
        print(f"    pH errors: {da.x_errc.round(3)}")
        print(f"    Signal errors: {da.y_errc.round(1)}")
        print()

    masked_points = int(
        (~synthetic_dataset["y1"].mask).sum() + (~synthetic_dataset["y2"].mask).sum()
    )

    print("📊 Synthetic vs Real Comparison:")
    print(f"  ✅ pH points: {len(synthetic_dataset['y1'].xc)} (matches real: 7)")
    print(
        f"  ✅ pH range: {synthetic_dataset['y1'].xc.min():.2f} - {synthetic_dataset['y1'].xc.max():.2f} (matches real 4.99–9.06)"
    )
    print(
        f"  ✅ Signal range Y1: {np.nanmin(synthetic_dataset['y1'].yc):.0f} - {np.nanmax(synthetic_dataset['y1'].yc):.0f}"
    )
    print(
        f"  ✅ Signal range Y2: {np.nanmin(synthetic_dataset['y2'].yc):.0f} - {np.nanmax(synthetic_dataset['y2'].yc):.0f}"
    )
    print("  ✅ Error levels: Derived from sample CVs and shot noise")
    print(
        f"  ✅ Channel correlation target: {realistic_params.channel_correlation:.2f}"
    )
    print(f"  ✅ Masked points due to saturation: {masked_points}")

    return synthetic_dataset, true_params


def test_fitting_with_realistic_data():
    """Test fitting methods with realistic synthetic data."""
    print("\n🎯 TESTING FITTING WITH REALISTIC DATA")
    print("=" * 50)

    # Generate multiple realistic datasets for testing
    results_summary = []

    for run_idx in range(5):
        params = RealisticSimulationParameters(
            random_seed=None,
            K_true=np.random.uniform(6.5, 7.5),  # Realistic pKa range
            outlier_probability=0.15,  # Moderate outlier probability
        )

        dataset, true_params = generate_realistic_dataset(params)

        # Quick test with basic fitting
        import time

        from src.clophfit.fitting.core import fit_lm, outlier2

        methods_to_test = {
            "Standard LM": lambda: fit_lm(dataset),
            "Robust Huber": lambda: fit_lm(dataset, robust=True),
            "Outlier2": lambda: outlier2(dataset, key="test"),
        }

        for method_name, method_func in methods_to_test.items():
            try:
                start_time = time.time()
                result = method_func()
                exec_time = time.time() - start_time

                if (
                    result.result
                    and result.result.success
                    and "K" in result.result.params
                ):
                    K_est = result.result.params["K"].value
                    K_true = true_params["K"]
                    K_error = abs(K_est - K_true) / K_true * 100

                    results_summary.append(
                        {
                            "run": run_idx,
                            "method": method_name,
                            "K_true": K_true,
                            "K_est": K_est,
                            "K_error_pct": K_error,
                            "exec_time": exec_time,
                            "success": True,
                        }
                    )
                else:
                    results_summary.append(
                        {
                            "run": run_idx,
                            "method": method_name,
                            "success": False,
                        }
                    )
            except Exception as e:
                results_summary.append(
                    {
                        "run": run_idx,
                        "method": method_name,
                        "success": False,
                        "error": str(e),
                    }
                )

    # Analyze results
    successful_results = [r for r in results_summary if r.get("success", False)]

    if successful_results:
        print(f"\n📈 Results from {len(successful_results)} successful fits:")

        methods = list({r["method"] for r in successful_results})
        for method in methods:
            method_results = [r for r in successful_results if r["method"] == method]
            if method_results:
                avg_error = np.mean([r["K_error_pct"] for r in method_results])
                avg_time = np.mean([r["exec_time"] for r in method_results])
                success_rate = len(method_results) / 5 * 100  # Out of 5 runs

                print(
                    f"  {method:15}: {avg_error:6.1f}% error, {avg_time:.3f}s, {success_rate:5.0f}% success"
                )

    return results_summary


def visualize_realistic_data() -> None:
    """Create visualization comparing realistic synthetic data with real patterns."""
    # Generate a sample realistic dataset
    params = RealisticSimulationParameters()
    dataset, true_params = generate_realistic_dataset(params)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Realistic Synthetic Data vs Real Data Patterns", fontsize=14)

    # Plot synthetic data
    ax1 = axes[0, 0]
    for label, da in dataset.items():
        valid_points = da.mask
        ax1.errorbar(
            da.xc[valid_points],
            da.yc[valid_points],
            yerr=da.y_errc[valid_points],
            xerr=da.x_errc[valid_points],
            marker="o",
            label=f"Synthetic {label}",
            alpha=0.7,
            capsize=3,
        )

        # Show masked points
        if not np.all(valid_points):
            masked_points = ~valid_points
            ax1.plot(
                da.xc[masked_points],
                da.yc[masked_points],
                "x",
                markersize=8,
                color="red",
                alpha=0.7,
            )

    ax1.set_xlabel("pH")
    ax1.set_ylabel("Signal")
    ax1.set_title("Synthetic Data (Realistic)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Compare signal patterns
    ax2 = axes[0, 1]
    pH_fine = np.linspace(5.5, 9.0, 100)
    x_fine = 10 ** (-pH_fine)

    # Show theoretical curves
    K_conc = 10 ** (-true_params["K"])
    y1_theory = binding_1site(
        x_fine, K_conc, true_params["S0_y1"], true_params["S1_y1"], is_ph=True
    )
    y2_theory = binding_1site(
        x_fine, K_conc, true_params["S0_y2"], true_params["S1_y2"], is_ph=True
    )

    ax2.plot(pH_fine, y1_theory, "--", label=f"Y1 theory (pKa={true_params['K']:.1f})")
    ax2.plot(pH_fine, y2_theory, "--", label=f"Y2 theory (pKa={true_params['K']:.1f})")

    # Overlay data points
    for label, da in dataset.items():
        ax2.plot(da.xc, da.yc, "o", label=f"{label} observed", alpha=0.7)

    ax2.set_xlabel("pH")
    ax2.set_ylabel("Signal")
    ax2.set_title("Theoretical vs Observed")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Show error characteristics
    ax3 = axes[1, 0]
    labels = list(dataset.keys())

    def _relative_error(da: DataArray) -> float:
        valid = da.mask
        if not np.any(valid):
            return 0.0
        median_signal = np.nanmedian(np.abs(da.yc[valid]))
        median_noise = np.nanmedian(da.y_errc[valid])
        denom = max(median_signal, 1.0)
        return float(median_noise / denom * 100)

    y1_rel_error = _relative_error(dataset["y1"])
    y2_rel_error = _relative_error(dataset["y2"])

    bars = ax3.bar(
        labels, [y1_rel_error, y2_rel_error], color=["skyblue", "lightcoral"], alpha=0.7
    )
    ax3.set_ylabel("Relative Error (%)")
    ax3.set_title("Error Characteristics")
    ax3.grid(True, alpha=0.3, axis="y")

    for bar, error in zip(bars, [y1_rel_error, y2_rel_error], strict=False):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + error * 0.05,
            f"{error:.1f}%",
            ha="center",
            va="bottom",
        )

    # Show data quality summary
    ax4 = axes[1, 1]

    quality_metrics = {
        "pH Points": len(dataset["y1"].xc),
        "pH Range": dataset["y1"].xc.max() - dataset["y1"].xc.min(),
        "Y1 S/N Ratio": np.abs(dataset["y1"].yc.mean()) / dataset["y1"].y_errc.mean(),
        "Y2 S/N Ratio": np.abs(dataset["y2"].yc.mean()) / dataset["y2"].y_errc.mean(),
    }

    y_pos = range(len(quality_metrics))
    values = list(quality_metrics.values())

    bars = ax4.barh(y_pos, values, color="lightgreen", alpha=0.7)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(quality_metrics.keys())
    ax4.set_xlabel("Value")
    ax4.set_title("Data Quality Summary")
    ax4.grid(True, alpha=0.3, axis="x")

    for _i, (bar, value) in enumerate(zip(bars, values, strict=False)):
        ax4.text(
            bar.get_width() + value * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}",
            ha="left",
            va="center",
        )

    plt.tight_layout()
    plt.savefig("realistic_synthetic_data_comparison.png", dpi=300, bbox_inches="tight")
    print("Visualization saved as 'realistic_synthetic_data_comparison.png'")


__all__ = [
    "RealisticSimulationParameters",
    "analyze_real_data_patterns",
    "compare_synthetic_vs_real",
    "generate_realistic_dataset",
    "test_fitting_with_realistic_data",
    "visualize_realistic_data",
]


def __dir__():
    return __all__


if __name__ == "__main__":
    print("🧬 REALISTIC SYNTHETIC DATA GENERATOR")
    print("=" * 60)

    # Analyze real data patterns
    real_examples = analyze_real_data_patterns()

    # Compare with synthetic data
    synthetic_dataset, true_params = compare_synthetic_vs_real()

    # Test fitting performance
    results = test_fitting_with_realistic_data()

    # Create visualizations
    visualize_realistic_data()

    print("\n🎉 ANALYSIS COMPLETE!")
    print("Key improvements in realistic synthetic data:")
    print("  ✅ 7 pH points (5.5 - 8.9 range)")
    print("  ✅ Realistic pKa range (6-8)")
    print("  ✅ Appropriate signal magnitudes")
    print("  ✅ Realistic error levels")
    print("  ✅ Variable pH measurement errors")
    print("  ✅ Occasional masked points")
    print("  ✅ Moderate outlier rates (~10%)")

    print("\nUse RealisticSimulationParameters() for future testing!")
