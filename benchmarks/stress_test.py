#!/usr/bin/env python3

"""Stress test all fitting methods with challenging synthetic scenarios.

Creates datasets with:
1. High outlier rates (10-30%)
2. Low-pH signal drops (acidic tail collapse)
3. High noise levels
4. Correlated channel errors
5. Missing/saturated points

These scenarios reveal which methods are truly robust vs. which only work on clean data.
"""

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from clophfit.fitting.bayes import fit_binding_pymc, fit_binding_pymc2
from clophfit.fitting.core import (
    fit_binding_glob,
    outlier2,
)
from clophfit.fitting.data_structures import DataArray, Dataset
from clophfit.fitting.models import binding_1site


@dataclass
class StressScenario:
    """Definition of a stress test scenario."""

    name: str
    description: str
    outlier_prob: float = 0.0
    outlier_magnitude: float = 3.0
    low_ph_drop_prob: float = 0.0
    low_ph_drop_magnitude: float = 0.6  # fraction to drop
    noise_multiplier: float = 1.0
    saturation_prob: float = 0.0
    x_error_large: float = 0.0  # Large random x-errors (e.g., 0.3 pH units)
    x_systematic_offset: float = 0.0  # Systematic x-offset (e.g., +0.5 pH)
    x_outlier_index: int = -1  # Index of x-value to make an outlier (-1 = none)
    seed: int = 42


def generate_stress_dataset(
    scenario: StressScenario,
    pka_true: float = 7.0,
    s0_y1: float = 500.0,
    s1_y1: float = 300.0,
    s0_y2: float = 50.0,
    s1_y2: float = 350.0,
) -> tuple[Dataset, dict]:
    """Generate a synthetic dataset with stress factors.

    Parameters
    ----------
    scenario : StressScenario
        Definition of outlier, saturation, and x-error behavior.
    pka_true : float, optional
        Ground-truth pKa (shared across channels).
    s0_y1 : float, optional
        Baseline fluorescence for y1.
    s1_y1 : float, optional
        Plateau fluorescence for y1.
    s0_y2 : float, optional
        Baseline fluorescence for y2.
    s1_y2 : float, optional
        Plateau fluorescence for y2.

    Returns
    -------
    Dataset
        The synthetic dataset with masks, errors, and perturbations applied.
    dict
        Ground-truth parameters used in the simulation.
    """
    rng = np.random.default_rng(scenario.seed)

    # Standard pH titration points (true values)
    x_true = np.array([9.0, 8.2, 7.8, 7.0, 6.6, 5.8, 5.0])

    # Apply x-space perturbations
    x_measured = x_true.copy()

    # Large random x-errors
    if scenario.x_error_large > 0:
        x_measured += rng.normal(0, scenario.x_error_large, size=x_true.shape)

    # Systematic x-offset
    if scenario.x_systematic_offset != 0:
        x_measured += scenario.x_systematic_offset

    # Single x-outlier
    if scenario.x_outlier_index >= 0 and scenario.x_outlier_index < len(x_measured):
        x_measured[scenario.x_outlier_index] += rng.choice(
            [
                -1.0,
                1.0,
            ]
        )  # ±1 pH unit off

    # Generate clean signal using TRUE x-values
    y1_clean = binding_1site(x_true, pka_true, s0_y1, s1_y1, is_ph=True)
    y2_clean = binding_1site(x_true, pka_true, s0_y2, s1_y2, is_ph=True)

    # Add base noise
    base_noise_y1 = 0.05 * (np.max(y1_clean) - np.min(y1_clean))
    base_noise_y2 = 0.05 * (np.max(y2_clean) - np.min(y2_clean))

    y1_noisy = y1_clean + rng.normal(
        0, base_noise_y1 * scenario.noise_multiplier, size=x_true.shape
    )
    y2_noisy = y2_clean + rng.normal(
        0, base_noise_y2 * scenario.noise_multiplier, size=x_true.shape
    )

    # Add outliers
    n_points = len(x_true)
    n_outliers = int(n_points * scenario.outlier_prob)
    if n_outliers > 0:
        outlier_indices = rng.choice(n_points, size=n_outliers, replace=False)
        for idx in outlier_indices:
            if rng.random() > 0.5:  # 50% chance for each channel
                y1_noisy[idx] += (
                    scenario.outlier_magnitude * base_noise_y1 * rng.choice([-1, 1])
                )
            else:
                y2_noisy[idx] += (
                    scenario.outlier_magnitude * base_noise_y2 * rng.choice([-1, 1])
                )

    # Add low-pH signal drop
    if scenario.low_ph_drop_prob > 0 and rng.random() < scenario.low_ph_drop_prob:
        # Drop signal at 1-2 most acidic points
        n_drop = rng.choice([1, 2])
        acidic_indices = np.argsort(x_true)[:n_drop]  # Lowest pH values
        for idx in acidic_indices:
            drop_factor = 1.0 - scenario.low_ph_drop_magnitude
            y1_noisy[idx] *= drop_factor
            # Optionally also affect y2
            if rng.random() > 0.3:
                y2_noisy[idx] *= drop_factor

    # Add saturation (mask points)
    mask = np.ones(n_points, dtype=bool)
    if scenario.saturation_prob > 0:
        n_saturated = int(n_points * scenario.saturation_prob)
        if n_saturated > 0:
            sat_indices = rng.choice(n_points, size=n_saturated, replace=False)
            mask[sat_indices] = False

    # Create dataset (using MEASURED x-values)
    ds = Dataset({}, is_ph=True)

    # Estimate errors
    y1_err = np.full_like(y1_noisy, base_noise_y1 * scenario.noise_multiplier)
    y2_err = np.full_like(y2_noisy, base_noise_y2 * scenario.noise_multiplier)

    # X-errors for Bayesian methods
    if scenario.x_error_large > 0:
        x_err = np.full_like(x_measured, scenario.x_error_large)
    elif scenario.x_systematic_offset != 0:
        x_err = np.full_like(
            x_measured, 0.1
        )  # Assume we don't know about systematic offset
    elif scenario.x_outlier_index >= 0:
        x_err = np.full_like(x_measured, 0.05)
    else:
        x_err = np.full_like(x_measured, 0.05)  # Default pH error

    da1 = DataArray(xc=x_measured, yc=y1_noisy, x_errc=x_err, y_errc=y1_err)
    da1.mask = mask
    da2 = DataArray(xc=x_measured, yc=y2_noisy, x_errc=x_err, y_errc=y2_err)
    da2.mask = mask

    ds["1"] = da1
    ds["2"] = da2

    truth = {
        "K": pka_true,
        "S0_1": s0_y1,
        "S1_1": s1_y1,
        "S0_2": s0_y2,
        "S1_2": s1_y2,
    }

    return ds, truth


def run_all_methods_on_scenario(
    scenario: StressScenario, n_replicates: int = 10
) -> pd.DataFrame:
    """Test all fitting methods on a stress scenario with multiple replicates.

    Parameters
    ----------
    scenario : StressScenario
        Scenario definition describing stressors to apply.
    n_replicates : int, optional
        Number of independently simulated datasets to evaluate per method.

    Returns
    -------
    pd.DataFrame
        Results with columns: method, success_rate, mean_error, std_error, mean_time.
    """

    # Define methods to test
    def run_bayesian_shared(ds):
        lm_result = fit_binding_glob(ds, robust=False)
        if lm_result.result and lm_result.result.success:
            return fit_binding_pymc(lm_result, n_samples=1000)
        return lm_result

    def run_bayesian_perlabel(ds):
        lm_result = fit_binding_glob(ds, robust=False)
        if lm_result.result and lm_result.result.success:
            return fit_binding_pymc2(lm_result, n_samples=1000)
        return lm_result

    methods = {
        "Standard LM": lambda ds: fit_binding_glob(ds, robust=False),
        "Robust Huber": lambda ds: fit_binding_glob(ds, robust=True),
        "Outlier2": lambda ds: outlier2(ds, key="stress"),
        "Bayesian-Shared": run_bayesian_shared,
        "Bayesian-PerLabel": run_bayesian_perlabel,
    }

    results = []

    for method_name, method_func in methods.items():
        successes = 0
        errors = []
        times = []

        for rep in range(n_replicates):
            # Generate new dataset for each replicate
            ds, truth = generate_stress_dataset(
                StressScenario(
                    name=scenario.name,
                    description=scenario.description,
                    outlier_prob=scenario.outlier_prob,
                    outlier_magnitude=scenario.outlier_magnitude,
                    low_ph_drop_prob=scenario.low_ph_drop_prob,
                    low_ph_drop_magnitude=scenario.low_ph_drop_magnitude,
                    noise_multiplier=scenario.noise_multiplier,
                    saturation_prob=scenario.saturation_prob,
                    seed=scenario.seed + rep,
                )
            )

            try:
                start = time.time()
                result = method_func(ds)
                elapsed = time.time() - start
                times.append(elapsed)

                # Extract K value (handle both LM and Bayesian results)
                K_fit = None
                if hasattr(result, "result") and result.result:
                    if hasattr(result.result, "success") and result.result.success:
                        if (
                            hasattr(result.result, "params")
                            and "K" in result.result.params
                        ):
                            K_fit = result.result.params["K"].value

                # For Bayesian, extract from idata
                if (
                    K_fit is None
                    and hasattr(result, "idata")
                    and result.idata is not None
                ):
                    try:
                        import arviz as az

                        summary = az.summary(result.idata, var_names=["K"])
                        if "mean" in summary.columns:
                            K_fit = float(summary["mean"].values[0])
                    except Exception:
                        pass

                if K_fit is not None:
                    K_true = truth["K"]
                    error = abs(K_fit - K_true) / K_true * 100  # Percent error
                    errors.append(error)
                    successes += 1
            except Exception:
                times.append(0.0)

        success_rate = successes / n_replicates * 100
        mean_error = np.mean(errors) if errors else np.nan
        std_error = np.std(errors) if errors else np.nan
        mean_time = np.mean(times)

        results.append(
            {
                "scenario": scenario.name,
                "method": method_name,
                "success_rate": success_rate,
                "mean_error": mean_error,
                "std_error": std_error,
                "mean_time": mean_time,
            }
        )

    return pd.DataFrame(results)


def main():
    """Run stress tests and compare methods."""
    print("=" * 80)
    print("STRESS TEST: FITTING METHODS ON CHALLENGING SYNTHETIC DATA")
    print("=" * 80)
    print()

    # Define stress scenarios
    scenarios = [
        StressScenario(
            name="Clean",
            description="Baseline: no stress factors",
            seed=42,
        ),
        StressScenario(
            name="HighNoise",
            description="3x normal noise level",
            noise_multiplier=3.0,
            seed=43,
        ),
        StressScenario(
            name="Outliers-10%",
            description="10% outliers (3σ magnitude)",
            outlier_prob=0.10,
            outlier_magnitude=3.0,
            seed=44,
        ),
        StressScenario(
            name="Outliers-30%",
            description="30% outliers (3σ magnitude)",
            outlier_prob=0.30,
            outlier_magnitude=3.0,
            seed=45,
        ),
        StressScenario(
            name="pH-Drop",
            description="Low-pH signal drop (60% reduction)",
            low_ph_drop_prob=1.0,  # Always happens
            low_ph_drop_magnitude=0.6,
            seed=46,
        ),
        StressScenario(
            name="Saturation",
            description="20% saturated points (masked)",
            saturation_prob=0.20,
            seed=47,
        ),
        StressScenario(
            name="Combined-Moderate",
            description="15% outliers + high noise + pH drop",
            outlier_prob=0.15,
            outlier_magnitude=3.0,
            low_ph_drop_prob=0.5,
            low_ph_drop_magnitude=0.5,
            noise_multiplier=2.0,
            seed=48,
        ),
        StressScenario(
            name="Combined-Severe",
            description="30% outliers + very high noise + pH drop + saturation",
            outlier_prob=0.30,
            outlier_magnitude=4.0,
            low_ph_drop_prob=1.0,
            low_ph_drop_magnitude=0.7,
            noise_multiplier=4.0,
            saturation_prob=0.15,
            seed=49,
        ),
        StressScenario(
            name="X-Error-Large",
            description="Large x-errors (±0.3 pH units) - Bayesian advantage",
            outlier_prob=0.0,
            noise_multiplier=1.0,
            x_error_large=0.3,
            seed=50,
        ),
        StressScenario(
            name="X-Error-Systematic",
            description="Systematic x-offset (+0.5 pH) - Bayesian advantage",
            outlier_prob=0.0,
            noise_multiplier=1.0,
            x_systematic_offset=0.5,
            seed=51,
        ),
        StressScenario(
            name="X-Outlier",
            description="One pH measurement way off (outlier in x-space)",
            outlier_prob=0.0,
            noise_multiplier=1.0,
            x_outlier_index=3,  # Middle pH point
            seed=52,
        ),
    ]

    all_results = []

    for scenario in scenarios:
        print(f"Testing scenario: {scenario.name}")
        print(f"  {scenario.description}")
        df = run_all_methods_on_scenario(scenario, n_replicates=20)
        all_results.append(df)

        # Print summary
        print(f"  {'Method':<25} {'Success%':<10} {'Error%':<12} {'Time(s)':<8}")
        print("  " + "-" * 70)
        for _, row in df.iterrows():
            if row["success_rate"] > 0:
                print(
                    f"  {row['method']:<25} {row['success_rate']:>7.0f}%   "
                    f"{row['mean_error']:>8.2f}±{row['std_error']:.2f}  {row['mean_time']:>7.3f}"
                )
            else:
                print(
                    f"  {row['method']:<25} {row['success_rate']:>7.0f}%   "
                    f"{'FAILED':<15}  {row['mean_time']:>7.3f}"
                )
        print()

    # Aggregate analysis
    full_df = pd.concat(all_results, ignore_index=True)

    print("=" * 80)
    print("AGGREGATE ANALYSIS")
    print("=" * 80)
    print()

    # Success rate by method across all scenarios
    print("OVERALL SUCCESS RATE BY METHOD:")
    overall_success = full_df.groupby("method")["success_rate"].mean()
    print(overall_success.sort_values(ascending=False).to_string())
    print()

    # Best method per scenario
    print("BEST METHOD PER SCENARIO (by success rate, then error):")
    for scenario in scenarios:
        scenario_df = full_df[full_df["scenario"] == scenario.name]
        # Sort by success rate (desc), then mean error (asc)
        scenario_df = scenario_df.sort_values(
            ["success_rate", "mean_error"], ascending=[False, True]
        )
        best = scenario_df.iloc[0]
        print(
            f"  {scenario.name:<20}: {best['method']:<25} "
            f"({best['success_rate']:.0f}% success, {best['mean_error']:.2f}% error)"
        )

    print()
    print("=" * 80)
    print("FINDINGS:")
    print(
        "Methods that maintain >95% success rate on severe scenarios are truly robust."
    )
    print("Methods that fail on moderate scenarios should be deprecated.")
    print("=" * 80)

    # Save results
    full_df.to_csv("stress_test_results.csv", index=False)
    print("\nResults saved to: stress_test_results.csv")


if __name__ == "__main__":
    main()
