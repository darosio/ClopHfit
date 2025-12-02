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

import numpy as np
import pandas as pd

from clophfit.fitting.bayes import fit_binding_pymc, fit_binding_pymc2
from clophfit.fitting.core import (
    fit_binding_glob,
    outlier2,
)
from clophfit.testing.synthetic import (
    STRESS_SCENARIOS,
    StressScenario,
    make_stress_dataset,
)


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
            # Generate new dataset for each replicate using the new module
            rep_scenario = StressScenario(
                name=scenario.name,
                description=scenario.description,
                outlier_prob=scenario.outlier_prob,
                outlier_magnitude=scenario.outlier_magnitude,
                low_ph_drop_prob=scenario.low_ph_drop_prob,
                low_ph_drop_magnitude=scenario.low_ph_drop_magnitude,
                noise_multiplier=scenario.noise_multiplier,
                saturation_prob=scenario.saturation_prob,
                x_error_large=scenario.x_error_large,
                x_systematic_offset=scenario.x_systematic_offset,
                seed=scenario.seed + rep,
            )
            ds, truth = make_stress_dataset(rep_scenario)

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

    # Use predefined scenarios from the synthetic module
    scenarios = list(STRESS_SCENARIOS.values())

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
