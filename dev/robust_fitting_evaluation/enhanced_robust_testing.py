#!/usr/bin/env python3
"""
Enhanced robust fitting testing with realistic synthetic data.

This module combines:
1. The enhanced robust fitting function (IRLS + Huber loss)
2. Realistic synthetic data generation based on actual experimental patterns
3. Comprehensive comparison of fitting methods
4. Performance benchmarking and validation
"""

import operator
import time
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from enhanced_robust_fitting import fit_lm_robust_irls
from realistic_synthetic_data import (
    RealisticSimulationParameters,
    generate_realistic_dataset,
)

from src.clophfit.fitting.core import fit_binding_glob_reweighted, fit_lm


@dataclass
class TestingConfiguration:
    """Configuration for comprehensive testing of robust fitting methods."""

    # Number of test datasets to generate
    n_test_datasets: int = 20

    # pKa range for testing (realistic experimental range)
    pKa_range: tuple[float, float] = (6.0, 8.0)

    # Outlier probability range (moderate to high)
    outlier_prob_range: tuple[float, float] = (0.05, 0.25)

    # Noise level multiplier range
    noise_multiplier_range: tuple[float, float] = (0.5, 3.0)

    # Random seeds for reproducibility
    random_seeds: list[int] = None

    # Methods to test
    methods_to_test: dict[str, callable] = None

    # Tolerance for parameter estimation (percentage)
    tolerance_pct: float = 20.0

    def __post_init__(self):
        if self.random_seeds is None:
            self.random_seeds = list(range(42, 42 + self.n_test_datasets))

        if self.methods_to_test is None:
            self.methods_to_test = {
                "Standard LM": fit_lm,
                "Robust Huber": lambda ds: fit_lm(ds, robust=True),
                "IRLS Reweight": lambda ds: fit_binding_glob_reweighted(ds, key="test"),
                "Enhanced Robust": lambda ds: fit_lm_robust_irls(
                    ds, max_iterations=5, tolerance=1e-4, outlier_threshold=2.5
                ),
            }


def generate_test_suite(config: TestingConfiguration) -> list[dict[str, Any]]:
    """
    Generate a comprehensive test suite with varying conditions.

    Parameters
    ----------
    config : TestingConfiguration
        Configuration for test generation

    Returns
    -------
    List[Dict[str, Any]]
        List of test cases with datasets and metadata
    """
    test_cases = []

    for i, seed in enumerate(config.random_seeds[: config.n_test_datasets]):
        # Vary parameters across realistic ranges
        pKa_true = np.random.uniform(*config.pKa_range)
        outlier_prob = np.random.uniform(*config.outlier_prob_range)
        noise_mult = np.random.uniform(*config.noise_multiplier_range)

        # Create test-specific parameters
        base_params = RealisticSimulationParameters(
            random_seed=seed,
            K_true=pKa_true,
            outlier_probability=outlier_prob,
            y1_base_error=25.0 * noise_mult,
            y2_base_error=15.0 * noise_mult,
        )

        # Generate dataset
        dataset, true_params = generate_realistic_dataset(base_params)

        test_case = {
            "case_id": i,
            "seed": seed,
            "dataset": dataset,
            "true_params": true_params,
            "generation_params": base_params,
            "difficulty": "easy"
            if outlier_prob < 0.10
            else "medium"
            if outlier_prob < 0.20
            else "hard",
        }

        test_cases.append(test_case)

    return test_cases


def run_comprehensive_testing(config: TestingConfiguration) -> dict[str, Any]:
    """
    Run comprehensive testing of all fitting methods.

    Parameters
    ----------
    config : TestingConfiguration
        Configuration for testing

    Returns
    -------
    Dict[str, Any]
        Comprehensive test results
    """
    print("🧪 COMPREHENSIVE ROBUST FITTING TESTING")
    print("=" * 60)

    # Generate test suite
    print(f"Generating {config.n_test_datasets} test datasets...")
    test_cases = generate_test_suite(config)

    # Initialize results storage
    all_results = {method_name: [] for method_name in config.methods_to_test}
    timing_results = {method_name: [] for method_name in config.methods_to_test}

    # Run tests
    print("Running fitting tests...")
    for i, test_case in enumerate(test_cases):
        print(
            f"  Test {i + 1}/{len(test_cases)} (difficulty: {test_case['difficulty']})",
            end=" ",
        )

        for method_name, method_func in config.methods_to_test.items():
            try:
                # Time the fitting
                start_time = time.time()
                result = method_func(test_case["dataset"])
                exec_time = time.time() - start_time

                # Evaluate result
                if (
                    result.result
                    and result.result.success
                    and "K" in result.result.params
                ):
                    K_est = result.result.params["K"].value
                    K_true = test_case["true_params"]["K"]
                    K_error_pct = abs(K_est - K_true) / K_true * 100

                    # Check if within tolerance
                    success = K_error_pct <= config.tolerance_pct

                    test_result = {
                        "case_id": test_case["case_id"],
                        "K_true": K_true,
                        "K_est": K_est,
                        "K_error_pct": K_error_pct,
                        "success": success,
                        "converged": True,
                        "exec_time": exec_time,
                        "difficulty": test_case["difficulty"],
                        "outlier_prob": test_case[
                            "generation_params"
                        ].outlier_probability,
                        "chi2_reduced": result.result.redchi
                        if hasattr(result.result, "redchi")
                        else np.nan,
                    }
                else:
                    test_result = {
                        "case_id": test_case["case_id"],
                        "success": False,
                        "converged": False,
                        "exec_time": exec_time,
                        "difficulty": test_case["difficulty"],
                        "outlier_prob": test_case[
                            "generation_params"
                        ].outlier_probability,
                    }

            except Exception as e:
                test_result = {
                    "case_id": test_case["case_id"],
                    "success": False,
                    "converged": False,
                    "error": str(e),
                    "exec_time": 0.0,
                    "difficulty": test_case["difficulty"],
                    "outlier_prob": test_case["generation_params"].outlier_probability,
                }

            all_results[method_name].append(test_result)
            timing_results[method_name].append(test_result["exec_time"])

        print("✓")

    return {
        "test_cases": test_cases,
        "results": all_results,
        "timing": timing_results,
        "config": config,
    }


def analyze_test_results(test_results: dict[str, Any]) -> None:
    """
    Analyze and display comprehensive test results.

    Parameters
    ----------
    test_results : Dict[str, Any]
        Results from comprehensive testing
    """
    print("\n📊 COMPREHENSIVE TEST RESULTS ANALYSIS")
    print("=" * 60)

    results = test_results["results"]
    test_results["config"]

    # Overall performance summary
    print("🎯 OVERALL PERFORMANCE SUMMARY")
    print("-" * 40)

    method_stats = {}
    for method_name, method_results in results.items():
        successful_results = [r for r in method_results if r.get("success", False)]
        converged_results = [r for r in method_results if r.get("converged", False)]

        if successful_results:
            avg_error = np.mean([r["K_error_pct"] for r in successful_results])
            median_error = np.median([r["K_error_pct"] for r in successful_results])
            std_error = np.std([r["K_error_pct"] for r in successful_results])
            max_error = np.max([r["K_error_pct"] for r in successful_results])
        else:
            avg_error = median_error = std_error = max_error = np.inf

        success_rate = len(successful_results) / len(method_results) * 100
        convergence_rate = len(converged_results) / len(method_results) * 100
        avg_time = np.mean([r["exec_time"] for r in method_results])

        method_stats[method_name] = {
            "success_rate": success_rate,
            "convergence_rate": convergence_rate,
            "avg_error": avg_error,
            "median_error": median_error,
            "std_error": std_error,
            "max_error": max_error,
            "avg_time": avg_time,
            "successful_results": successful_results,
        }

        print(
            f"{method_name:15}: {success_rate:5.1f}% success, {avg_error:6.1f}% avg error, {avg_time:.3f}s avg time"
        )

    # Performance by difficulty
    print("\n🎖️ PERFORMANCE BY DIFFICULTY")
    print("-" * 40)

    difficulties = ["easy", "medium", "hard"]
    for difficulty in difficulties:
        print(f"\n{difficulty.upper()} Cases:")
        for method_name, method_results in results.items():
            diff_results = [
                r for r in method_results if r.get("difficulty") == difficulty
            ]
            successful_diff = [r for r in diff_results if r.get("success", False)]

            if diff_results:
                success_rate = len(successful_diff) / len(diff_results) * 100
                if successful_diff:
                    avg_error = np.mean([r["K_error_pct"] for r in successful_diff])
                    print(
                        f"  {method_name:15}: {success_rate:5.1f}% success, {avg_error:6.1f}% avg error"
                    )
                else:
                    print(
                        f"  {method_name:15}: {success_rate:5.1f}% success, no successful fits"
                    )

    # Outlier tolerance analysis
    print("\n🎯 OUTLIER TOLERANCE ANALYSIS")
    print("-" * 40)

    outlier_bins = [(0.0, 0.1), (0.1, 0.2), (0.2, 1.0)]
    for low, high in outlier_bins:
        print(f"\nOutlier probability {low:.1f}-{high:.1f}:")
        for method_name, method_results in results.items():
            outlier_results = [
                r for r in method_results if low <= r.get("outlier_prob", 0) < high
            ]
            successful_outlier = [r for r in outlier_results if r.get("success", False)]

            if outlier_results:
                success_rate = len(successful_outlier) / len(outlier_results) * 100
                if successful_outlier:
                    avg_error = np.mean([r["K_error_pct"] for r in successful_outlier])
                    print(
                        f"  {method_name:15}: {success_rate:5.1f}% success, {avg_error:6.1f}% avg error"
                    )
                else:
                    print(f"  {method_name:15}: {success_rate:5.1f}% success")

    # Best method recommendation
    print("\n🏆 RECOMMENDATIONS")
    print("-" * 40)

    # Rank methods by success rate + error performance
    method_scores = {}
    for method_name, stats in method_stats.items():
        # Weighted score: success rate (70%) + error performance (30%)
        error_score = 100 - min(stats["avg_error"], 100)  # Lower error = higher score
        combined_score = 0.7 * stats["success_rate"] + 0.3 * error_score
        method_scores[method_name] = combined_score

    ranked_methods = sorted(
        method_scores.items(), key=operator.itemgetter(1), reverse=True
    )

    print("Method ranking (success rate + accuracy):")
    for i, (method_name, score) in enumerate(ranked_methods):
        stats = method_stats[method_name]
        print(
            f"  {i + 1}. {method_name:15}: {score:5.1f} pts "
            f"({stats['success_rate']:4.1f}% success, {stats['avg_error']:5.1f}% error)"
        )

    best_method = ranked_methods[0][0]
    print(f"\n🎉 RECOMMENDED METHOD: {best_method}")

    best_stats = method_stats[best_method]
    print(f"   • Success rate: {best_stats['success_rate']:.1f}%")
    print(f"   • Average error: {best_stats['avg_error']:.1f}%")
    print(f"   • Execution time: {best_stats['avg_time']:.3f}s")


def visualize_test_results(test_results: dict[str, Any]) -> None:
    """Create comprehensive visualization of test results."""
    results = test_results["results"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Comprehensive Robust Fitting Method Comparison", fontsize=16)

    # 1. Success rates by method
    ax1 = axes[0, 0]
    methods = list(results.keys())
    success_rates = []
    for method in methods:
        successful = [r for r in results[method] if r.get("success", False)]
        success_rates.append(len(successful) / len(results[method]) * 100)

    bars = ax1.bar(
        methods, success_rates, color=["skyblue", "lightcoral", "lightgreen", "gold"]
    )
    ax1.set_ylabel("Success Rate (%)")
    ax1.set_title("Success Rate by Method")
    ax1.tick_params(axis="x", rotation=45)

    for bar, rate in zip(bars, success_rates, strict=False):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{rate:.1f}%",
            ha="center",
            va="bottom",
        )

    # 2. Error distributions
    ax2 = axes[0, 1]
    error_data = []
    method_labels = []
    colors = ["skyblue", "lightcoral", "lightgreen", "gold"]

    for i, method in enumerate(methods):
        successful = [r for r in results[method] if r.get("success", False)]
        if successful:
            errors = [r["K_error_pct"] for r in successful]
            error_data.append(errors)
            method_labels.append(method)

    if error_data:
        bp = ax2.boxplot(error_data, labels=method_labels, patch_artist=True)
        for patch, color in zip(bp["boxes"], colors[: len(error_data)], strict=False):
            patch.set_facecolor(color)
        ax2.set_ylabel("Parameter Error (%)")
        ax2.set_title("Error Distribution")
        ax2.tick_params(axis="x", rotation=45)

    # 3. Execution time comparison
    ax3 = axes[0, 2]
    exec_times = []
    for method in methods:
        times = [r["exec_time"] for r in results[method]]
        exec_times.append(np.mean(times))

    bars = ax3.bar(
        methods, exec_times, color=["skyblue", "lightcoral", "lightgreen", "gold"]
    )
    ax3.set_ylabel("Execution Time (s)")
    ax3.set_title("Average Execution Time")
    ax3.tick_params(axis="x", rotation=45)

    for bar, time in zip(bars, exec_times, strict=False):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + time * 0.05,
            f"{time:.3f}s",
            ha="center",
            va="bottom",
        )

    # 4. Performance vs outlier probability
    ax4 = axes[1, 0]
    outlier_probs = np.linspace(0.05, 0.25, 20)

    for method, color in zip(methods, ["blue", "red", "green", "orange"], strict=False):
        success_rates_outlier = []
        for prob in outlier_probs:
            prob_results = [
                r
                for r in results[method]
                if abs(r.get("outlier_prob", 0) - prob) < 0.02
            ]
            if prob_results:
                successful = [r for r in prob_results if r.get("success", False)]
                rate = len(successful) / len(prob_results) * 100
            else:
                rate = 0
            success_rates_outlier.append(rate)

        ax4.plot(
            outlier_probs,
            success_rates_outlier,
            "o-",
            label=method,
            color=color,
            alpha=0.7,
        )

    ax4.set_xlabel("Outlier Probability")
    ax4.set_ylabel("Success Rate (%)")
    ax4.set_title("Robustness to Outliers")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Performance by difficulty
    ax5 = axes[1, 1]
    difficulties = ["easy", "medium", "hard"]
    x_pos = np.arange(len(difficulties))
    width = 0.2

    for i, method in enumerate(methods):
        diff_success_rates = []
        for difficulty in difficulties:
            diff_results = [
                r for r in results[method] if r.get("difficulty") == difficulty
            ]
            successful = [r for r in diff_results if r.get("success", False)]
            rate = len(successful) / len(diff_results) * 100 if diff_results else 0
            diff_success_rates.append(rate)

        ax5.bar(
            x_pos + i * width,
            diff_success_rates,
            width,
            label=method,
            color=["skyblue", "lightcoral", "lightgreen", "gold"][i],
            alpha=0.7,
        )

    ax5.set_xlabel("Difficulty")
    ax5.set_ylabel("Success Rate (%)")
    ax5.set_title("Performance by Difficulty")
    ax5.set_xticks(x_pos + width * 1.5)
    ax5.set_xticklabels(difficulties)
    ax5.legend()

    # 6. Error vs execution time scatter
    ax6 = axes[1, 2]

    for method, color in zip(methods, ["blue", "red", "green", "orange"], strict=False):
        successful = [r for r in results[method] if r.get("success", False)]
        if successful:
            errors = [r["K_error_pct"] for r in successful]
            times = [r["exec_time"] for r in successful]
            ax6.scatter(times, errors, label=method, color=color, alpha=0.6, s=30)

    ax6.set_xlabel("Execution Time (s)")
    ax6.set_ylabel("Parameter Error (%)")
    ax6.set_title("Accuracy vs Speed Trade-off")
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("comprehensive_fitting_comparison.png", dpi=300, bbox_inches="tight")
    print("\nVisualization saved as 'comprehensive_fitting_comparison.png'")


def run_focused_comparison():
    """Run a focused comparison on particularly challenging datasets."""
    print("\n🔍 FOCUSED COMPARISON ON CHALLENGING DATASETS")
    print("=" * 60)

    # Create challenging scenarios
    challenging_scenarios = [
        {
            "name": "High Outliers",
            "params": RealisticSimulationParameters(
                random_seed=123,
                outlier_probability=0.3,  # 30% outliers
                outlier_magnitude=5.0,  # Strong outliers
            ),
        },
        {
            "name": "High Noise",
            "params": RealisticSimulationParameters(
                random_seed=456,
                y1_base_error=50.0,  # Double normal noise
                y2_base_error=30.0,
                shot_noise_factor=0.05,  # More shot noise
            ),
        },
        {
            "name": "Edge pKa",
            "params": RealisticSimulationParameters(
                random_seed=789,
                K_true=5.5,  # Edge of pH range
                outlier_probability=0.15,
            ),
        },
        {
            "name": "Combined Stress",
            "params": RealisticSimulationParameters(
                random_seed=999,
                outlier_probability=0.25,
                y1_base_error=40.0,
                K_true=7.8,  # High pKa
                mask_probability=0.1,  # Some masking
            ),
        },
    ]

    all_scenario_results = {}

    for scenario in challenging_scenarios:
        print(f"\nTesting scenario: {scenario['name']}")

        # Generate dataset
        dataset, true_params = generate_realistic_dataset(scenario["params"])

        # Test all methods
        methods = {
            "Standard LM": fit_lm,
            "Robust Huber": lambda ds: fit_lm(ds, robust=True),
            "IRLS": lambda ds: fit_binding_glob_reweighted(ds, key="test"),
            "Enhanced Robust": lambda ds: fit_lm_robust_irls(
                ds, max_iterations=5, tolerance=1e-4, outlier_threshold=2.5
            ),
        }

        # Use individual method testing since compare_robust_methods has different signature
        scenario_results = {}
        for method_name, method_func in methods.items():
            try:
                start_time = time.time()
                result = method_func(dataset)
                exec_time = time.time() - start_time

                if (
                    result.result
                    and result.result.success
                    and "K" in result.result.params
                ):
                    K_est = result.result.params["K"].value
                    K_true = true_params["K"]
                    K_error = abs(K_est - K_true) / K_true * 100

                    scenario_results[method_name] = {
                        "success": True,
                        "K_error": K_error,
                        "time": exec_time,
                        "chi2": result.result.redchi
                        if hasattr(result.result, "redchi")
                        else np.nan,
                    }
                else:
                    scenario_results[method_name] = {
                        "success": False,
                        "time": exec_time,
                    }
            except Exception as e:
                scenario_results[method_name] = {
                    "success": False,
                    "error": str(e),
                    "time": 0.0,
                }
        all_scenario_results[scenario["name"]] = scenario_results

        # Display results
        print("  Results:")
        for method, result in scenario_results.items():
            if result["success"]:
                print(
                    f"    {method:15}: ✓ {result['K_error']:.1f}% error, {result['time']:.3f}s"
                )
            else:
                print(f"    {method:15}: ✗ Failed")

    return all_scenario_results


if __name__ == "__main__":
    print("🚀 ENHANCED ROBUST FITTING - COMPREHENSIVE TESTING")
    print("=" * 70)

    # Configuration for testing
    config = TestingConfiguration(
        n_test_datasets=25,  # Reasonable number for comprehensive testing
        pKa_range=(6.0, 8.0),  # Realistic experimental range
        outlier_prob_range=(0.05, 0.25),  # Moderate to high outlier rates
        tolerance_pct=15.0,  # Reasonable tolerance for success
    )

    # Run comprehensive testing
    test_results = run_comprehensive_testing(config)

    # Analyze results
    analyze_test_results(test_results)

    # Create visualizations
    visualize_test_results(test_results)

    # Run focused comparison on challenging cases
    challenging_results = run_focused_comparison()

    print("\n🎊 COMPREHENSIVE TESTING COMPLETE!")
    print("Check the generated visualizations and results above.")
    print("The enhanced robust fitting method should show improved performance")
    print("especially on datasets with outliers and challenging conditions.")
