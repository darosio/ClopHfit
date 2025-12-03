#!/usr/bin/env python3
"""Stress test all fitting methods with challenging synthetic scenarios.

Creates datasets with increasing difficulty:
1. Clean baseline
2. High noise (increased rel_error)
3. Outliers (10-30%)
4. Low-pH signal drops
5. Combined stress factors

These scenarios reveal which methods are truly robust vs. which only work on clean data.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from clophfit.fitting.bayes import fit_binding_pymc, fit_binding_pymc2
from clophfit.fitting.core import (
    fit_binding_glob,
    outlier2,
)
from clophfit.testing.synthetic import make_dataset


@dataclass
class ScenarioConfig:
    """Configuration for a stress test scenario."""

    name: str
    description: str
    rel_error: float = 0.035
    outlier_prob: float = 0.0
    outlier_sigma: float = 4.0
    low_ph_drop: bool = False
    low_ph_drop_magnitude: float = 0.4
    saturation_prob: float = 0.0
    x_error_large: float = 0.0


# Define stress scenarios with increasing difficulty
SCENARIOS = [
    ScenarioConfig(
        name="Clean",
        description="Baseline: no stress factors",
    ),
    ScenarioConfig(
        name="HighNoise",
        description="3x normal noise level",
        rel_error=0.105,  # 3x baseline
    ),
    ScenarioConfig(
        name="Outliers-10%",
        description="10% outliers (4-sigma magnitude)",
        outlier_prob=0.10,
    ),
    ScenarioConfig(
        name="Outliers-30%",
        description="30% outliers (4-sigma magnitude)",
        outlier_prob=0.30,
    ),
    ScenarioConfig(
        name="pH-Drop",
        description="Low-pH signal drop (40% reduction)",
        low_ph_drop=True,
        low_ph_drop_magnitude=0.4,
    ),
    ScenarioConfig(
        name="Saturation",
        description="20% saturated points (masked)",
        saturation_prob=0.20,
    ),
    ScenarioConfig(
        name="Combined-Moderate",
        description="15% outliers + 2x noise + pH drop",
        rel_error=0.07,
        outlier_prob=0.15,
        low_ph_drop=True,
        low_ph_drop_magnitude=0.3,
    ),
    ScenarioConfig(
        name="Combined-Severe",
        description="30% outliers + 4x noise + pH drop + saturation",
        rel_error=0.14,
        outlier_prob=0.30,
        low_ph_drop=True,
        low_ph_drop_magnitude=0.5,
        saturation_prob=0.15,
    ),
    ScenarioConfig(
        name="X-Error-Large",
        description="Large x-errors (±0.3 pH units)",
        x_error_large=0.3,
    ),
]


def run_all_methods_on_scenario(
    scenario: ScenarioConfig, n_replicates: int = 33
) -> pd.DataFrame:
    """Test all fitting methods on a stress scenario with multiple replicates.

    Parameters
    ----------
    scenario : ScenarioConfig
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
        # "Bayesian-Shared": run_bayesian_shared,
        # "Bayesian-PerLabel": run_bayesian_perlabel,
    }

    results = []

    for method_name, method_func in methods.items():
        successes = 0
        errors = []
        times = []

        for rep in range(n_replicates):
            # Generate dataset using make_dataset directly
            ds, truth = make_dataset(
                k=7.0,
                s0={"y1": 700.0, "y2": 1000.0},
                s1={"y1": 1200.0, "y2": 200.0},
                is_ph=True,
                seed=142 + rep,
                rel_error=scenario.rel_error,
                outlier_prob=scenario.outlier_prob,
                outlier_sigma=scenario.outlier_sigma,
                low_ph_drop=scenario.low_ph_drop,
                low_ph_drop_magnitude=scenario.low_ph_drop_magnitude,
                saturation_prob=scenario.saturation_prob,
                x_error_large=scenario.x_error_large,
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
                    K_true = truth.K
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

    all_results = []

    for scenario in SCENARIOS:
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
    for scenario in SCENARIOS:
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
