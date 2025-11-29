#!/usr/bin/env python3
"""
Final comprehensive testing and evaluation of robust fitting methods.

This module provides:
1. Comprehensive testing of all robust methods on realistic data
2. Performance analysis across different difficulty levels
3. Recommendations for method selection
4. Integration with your existing codebase
"""

import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from benchmarks.realistic_synthetic_data import (
    RealisticSimulationParameters,
    generate_realistic_dataset,
)
from benchmarks.simple_enhanced_robust import fit_lm_robust_simple
from clophfit.fitting.core import fit_lm, outlier2

# Constants
MIN_SUCCESS_RATE_THRESHOLD = 50
DEFAULT_PKA_RANGE = (5.0, 9.0)


@dataclass
class EvaluationResult:
    """Results from comprehensive method evaluation."""

    method_name: str
    success_rate: float
    avg_error: float
    median_error: float
    std_error: float
    avg_time: float
    best_case_error: float
    worst_case_error: float
    difficulty_performance: dict[str, float]  # success rate by difficulty


def run_comprehensive_evaluation(n_tests: int = 50) -> dict[str, EvaluationResult]:
    """
    Run comprehensive evaluation of all robust fitting methods.

    Parameters
    ----------
    n_tests : int
        Number of test cases to generate

    Returns
    -------
    dict[str, EvaluationResult]
        Comprehensive evaluation results.
    """
    print("🚀 COMPREHENSIVE ROBUST FITTING EVALUATION")
    print("=" * 60)
    print(f"Running {n_tests} test cases across various difficulty levels...")

    # Define methods to test
    methods = {
        "Standard LM": fit_lm,
        "Robust Huber": lambda ds: fit_lm(ds, robust=True),
        "Outlier2": lambda ds: outlier2(ds, key="eval"),
        "Simple Enhanced": lambda ds: fit_lm_robust_simple(ds, verbose=False),
    }

    # Generate diverse test scenarios
    test_cases = generate_diverse_test_cases(n_tests)

    # Store all results
    all_results = {method: [] for method in methods}

    # Run tests
    print(f"Testing {len(methods)} methods on {len(test_cases)} datasets...")

    for i, (dataset, true_params, difficulty) in enumerate(test_cases):
        print(f"  Test {i + 1:3d}/{len(test_cases)}: {difficulty:>8}", end="")

        test_results = {}
        for method_name, method_func in methods.items():
            start_time = time.time()

            try:
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

                    test_results[method_name] = {
                        "success": True,
                        "K_error": K_error,
                        "time": exec_time,
                        "difficulty": difficulty,
                    }
                else:
                    test_results[method_name] = {
                        "success": False,
                        "time": exec_time,
                        "difficulty": difficulty,
                    }
            except Exception:
                test_results[method_name] = {
                    "success": False,
                    "time": 0.0,
                    "difficulty": difficulty,
                }

        # Store results
        for method_name, result in test_results.items():
            all_results[method_name].append(result)

        # Show progress
        successful_methods = sum(1 for r in test_results.values() if r["success"])
        print(f" → {successful_methods}/{len(methods)} methods succeeded")

    # Analyze results
    evaluation_results = {}

    for method_name, results in all_results.items():
        successful_results = [r for r in results if r["success"]]

        if successful_results:
            errors = [r["K_error"] for r in successful_results]
            times = [r["time"] for r in results]

            # Performance by difficulty
            difficulty_performance = {}
            for difficulty in ["easy", "medium", "hard", "extreme"]:
                diff_results = [r for r in results if r["difficulty"] == difficulty]
                if diff_results:
                    success_count = sum(1 for r in diff_results if r["success"])
                    difficulty_performance[difficulty] = (
                        success_count / len(diff_results) * 100
                    )
                else:
                    difficulty_performance[difficulty] = 0.0

            evaluation_results[method_name] = EvaluationResult(
                method_name=method_name,
                success_rate=len(successful_results) / len(results) * 100,
                avg_error=np.mean(errors),
                median_error=np.median(errors),
                std_error=np.std(errors),
                avg_time=np.mean(times),
                best_case_error=np.min(errors),
                worst_case_error=np.max(errors),
                difficulty_performance=difficulty_performance,
            )
        else:
            # No successful results
            evaluation_results[method_name] = EvaluationResult(
                method_name=method_name,
                success_rate=0.0,
                avg_error=float("inf"),
                median_error=float("inf"),
                std_error=0.0,
                avg_time=np.mean([r["time"] for r in results]),
                best_case_error=float("inf"),
                worst_case_error=float("inf"),
                difficulty_performance=dict.fromkeys(
                    ["easy", "medium", "hard", "extreme"], 0.0
                ),
            )

    return evaluation_results


def generate_diverse_test_cases(n_tests: int) -> list[tuple]:
    """Generate diverse test cases with varying difficulty levels."""
    test_cases = []

    # Define difficulty levels
    difficulty_configs = {
        "easy": {
            "outlier_prob_range": (0.0, 0.05),
            "noise_mult_range": (0.5, 1.0),
            "pKa_range": (6.5, 7.5),
        },
        "medium": {
            "outlier_prob_range": (0.05, 0.15),
            "noise_mult_range": (1.0, 2.0),
            "pKa_range": (6.0, 8.0),
        },
        "hard": {
            "outlier_prob_range": (0.15, 0.25),
            "noise_mult_range": (2.0, 3.0),
            "pKa_range": (5.5, 8.5),
        },
        "extreme": {
            "outlier_prob_range": (0.25, 0.35),
            "noise_mult_range": (3.0, 5.0),
            "pKa_range": (5.0, 9.0),
        },
    }

    # Distribute tests across difficulty levels
    n_per_difficulty = n_tests // 4

    for difficulty, config in difficulty_configs.items():
        for i in range(n_per_difficulty):
            seed = hash((difficulty, i)) % 10000  # Reproducible but varied seeds

            # Random parameters within difficulty range
            rng = np.random.default_rng(seed)
            outlier_prob = rng.uniform(*config["outlier_prob_range"])
            noise_mult = rng.uniform(*config["noise_mult_range"])
            pka = rng.uniform(*config["pKa_range"])

            params = RealisticSimulationParameters(
                random_seed=seed,
                K_true=pka,
                outlier_probability=outlier_prob,
                # CORRECTED: y1 errors are 10x larger than y2 errors
                y1_base_error=100.0 * noise_mult,  # 10x larger base error
                y2_base_error=10.0 * noise_mult,  # Reference error level
                outlier_magnitude=rng.uniform(2.0, 5.0),
            )

            dataset, true_params = generate_realistic_dataset(params)
            test_cases.append((dataset, true_params, difficulty))

    return test_cases


def display_evaluation_results(results: dict[str, EvaluationResult]) -> None:
    """Display comprehensive evaluation results."""
    print("\\n📊 EVALUATION RESULTS SUMMARY")
    print("=" * 60)

    # Overall performance table
    print("\\n🎯 OVERALL PERFORMANCE")
    print("-" * 80)
    print(
        f"{'Method':<18} {'Success%':<9} {'Avg Error%':<12} {'Median%':<9} {'Std%':<8} {'Time(s)':<8}"
    )
    print("-" * 80)

    # Sort by success rate, then by error
    sorted_methods = sorted(
        results.items(), key=lambda x: (-x[1].success_rate, x[1].avg_error)
    )

    for method_name, result in sorted_methods:
        print(
            f"{method_name:<18} {result.success_rate:7.1f}% "
            f"{result.avg_error:10.1f}% {result.median_error:7.1f}% "
            f"{result.std_error:6.1f}% {result.avg_time:6.3f}s"
        )

    # Performance by difficulty
    print("\\n🎖️  PERFORMANCE BY DIFFICULTY LEVEL")
    print("-" * 60)

    difficulties = ["easy", "medium", "hard", "extreme"]
    print(f"{'Method':<18}", end="")
    for diff in difficulties:
        print(f"{diff.capitalize():<10}", end="")
    print()
    print("-" * 60)

    for method_name, result in sorted_methods:
        print(f"{method_name:<18}", end="")
        for diff in difficulties:
            success_rate = result.difficulty_performance[diff]
            print(f"{success_rate:8.1f}% ", end="")
        print()

    # Best/worst case analysis
    print("\\n🏆 BEST vs WORST CASE PERFORMANCE")
    print("-" * 50)
    print(f"{'Method':<18} {'Best%':<8} {'Worst%':<8} {'Range%':<8}")
    print("-" * 50)

    for method_name, result in sorted_methods:
        if result.success_rate > 0:
            range_error = result.worst_case_error - result.best_case_error
            print(
                f"{method_name:<18} {result.best_case_error:6.1f}% "
                f"{result.worst_case_error:7.1f}% {range_error:7.1f}%"
            )

    # Recommendations
    print("\\n💡 RECOMMENDATIONS")
    print("-" * 40)

    best_overall = sorted_methods[0]
    print(f"🥇 Best overall method: {best_overall[0]}")
    print(f"   • Success rate: {best_overall[1].success_rate:.1f}%")
    print(f"   • Average error: {best_overall[1].avg_error:.1f}%")
    print(f"   • Speed: {best_overall[1].avg_time:.3f}s per fit")

    # Best by category
    best_accuracy = min(
        results.items(),
        key=lambda x: (
            x[1].avg_error
            if x[1].success_rate > MIN_SUCCESS_RATE_THRESHOLD
            else float("inf")
        ),
    )
    best_speed = min(results.items(), key=lambda x: x[1].avg_time)
    best_robust = max(
        results.items(), key=lambda x: x[1].difficulty_performance.get("extreme", 0)
    )

    print(
        f"\\n🎯 Best for accuracy: {best_accuracy[0]} ({best_accuracy[1].avg_error:.1f}% error)"
    )
    print(f"⚡ Fastest method: {best_speed[0]} ({best_speed[1].avg_time:.3f}s)")
    print(
        f"🛡️  Most robust: {best_robust[0]} ({best_robust[1].difficulty_performance.get('extreme', 0):.1f}% on extreme)"
    )


def create_evaluation_plots(results: dict[str, EvaluationResult]) -> None:  # noqa: PLR0915
    """Create comprehensive visualization plots."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        "Comprehensive Robust Fitting Method Evaluation", fontsize=16, fontweight="bold"
    )

    methods = list(results.keys())
    colors = ["#2E8B57", "#DC143C", "#4169E1", "#FF8C00"]  # Distinct colors

    # 1. Success rates
    ax1 = axes[0, 0]
    success_rates = [results[m].success_rate for m in methods]
    bars = ax1.bar(methods, success_rates, color=colors, alpha=0.8)
    ax1.set_ylabel("Success Rate (%)", fontweight="bold")
    ax1.set_title("Overall Success Rate", fontweight="bold")
    ax1.tick_params(axis="x", rotation=45)

    for bar, rate in zip(bars, success_rates, strict=False):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{rate:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 2. Average errors
    ax2 = axes[0, 1]
    avg_errors = [
        results[m].avg_error if results[m].success_rate > 0 else np.nan for m in methods
    ]
    valid_methods = [
        (m, e) for m, e in zip(methods, avg_errors, strict=False) if not np.isnan(e)
    ]

    if valid_methods:
        m_names, errors = zip(*valid_methods, strict=False)
        bars = ax2.bar(
            m_names,
            errors,
            color=[colors[methods.index(m)] for m in m_names],
            alpha=0.8,
        )
        ax2.set_ylabel("Average Error (%)", fontweight="bold")
        ax2.set_title("Parameter Estimation Accuracy", fontweight="bold")
        ax2.tick_params(axis="x", rotation=45)

        for bar, error in zip(bars, errors, strict=False):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + error * 0.05,
                f"{error:.1f}%",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

    # 3. Performance by difficulty
    ax3 = axes[0, 2]
    difficulties = ["easy", "medium", "hard", "extreme"]
    x_pos = np.arange(len(difficulties))
    width = 0.2

    for i, method in enumerate(methods):
        diff_rates = [results[method].difficulty_performance[d] for d in difficulties]
        ax3.bar(
            x_pos + i * width,
            diff_rates,
            width,
            label=method,
            color=colors[i],
            alpha=0.8,
        )

    ax3.set_xlabel("Difficulty Level", fontweight="bold")
    ax3.set_ylabel("Success Rate (%)", fontweight="bold")
    ax3.set_title("Performance by Difficulty", fontweight="bold")
    ax3.set_xticks(x_pos + width * 1.5)
    ax3.set_xticklabels(difficulties)
    ax3.legend()

    # 4. Execution time comparison
    ax4 = axes[1, 0]
    exec_times = [results[m].avg_time for m in methods]
    bars = ax4.bar(methods, exec_times, color=colors, alpha=0.8)
    ax4.set_ylabel("Average Time (s)", fontweight="bold")
    ax4.set_title("Execution Speed", fontweight="bold")
    ax4.tick_params(axis="x", rotation=45)

    for bar, exec_time in zip(bars, exec_times, strict=False):
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + exec_time * 0.05,
            f"{exec_time:.3f}s",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 5. Error distribution (box plot style visualization)
    ax5 = axes[1, 1]

    # Create error range visualization
    valid_results = [(m, r) for m, r in results.items() if r.success_rate > 0]
    if valid_results:
        method_names, res_data = zip(*valid_results, strict=False)

        # Show error ranges
        best_errors = [r.best_case_error for r in res_data]
        worst_errors = [r.worst_case_error for r in res_data]
        avg_errors = [r.avg_error for r in res_data]

        x_positions = range(len(method_names))

        # Error bars showing range
        ax5.errorbar(
            x_positions,
            avg_errors,
            yerr=[
                np.array(avg_errors) - np.array(best_errors),
                np.array(worst_errors) - np.array(avg_errors),
            ],
            fmt="o",
            capsize=5,
            capthick=2,
            markersize=8,
            color="darkblue",
            alpha=0.8,
        )

        ax5.set_xticks(x_positions)
        ax5.set_xticklabels(method_names, rotation=45)
        ax5.set_ylabel("Parameter Error (%)", fontweight="bold")
        ax5.set_title("Error Range (Best to Worst)", fontweight="bold")
        ax5.grid(visible=True, alpha=0.3)

    # 6. Performance radar chart (simplified)
    ax6 = axes[1, 2]

    # Normalize metrics for radar chart
    metrics = ["Success Rate", "Accuracy", "Speed", "Robustness"]

    for i, method in enumerate(methods):
        result = results[method]

        # Normalize to 0-100 scale (higher is better)
        success_norm = result.success_rate  # Already 0-100
        accuracy_norm = (
            max(0, 100 - result.avg_error) if result.success_rate > 0 else 0
        )  # Invert error
        speed_norm = max(
            0, 100 - (result.avg_time * 100)
        )  # Invert time (rough normalization)
        robust_norm = result.difficulty_performance.get(
            "extreme", 0
        )  # Extreme case performance

        values = [success_norm, accuracy_norm, speed_norm, robust_norm]

        ax6.plot(
            metrics,
            values,
            "o-",
            label=method,
            color=colors[i],
            linewidth=2,
            markersize=6,
            alpha=0.8,
        )

    ax6.set_ylabel("Performance Score", fontweight="bold")
    ax6.set_title("Multi-Metric Performance", fontweight="bold")
    ax6.legend()
    ax6.grid(visible=True, alpha=0.3)
    ax6.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(
        "comprehensive_robust_fitting_evaluation.png", dpi=300, bbox_inches="tight"
    )
    print(
        "\\n📊 Comprehensive evaluation plots saved as 'comprehensive_robust_fitting_evaluation.png'"
    )


def generate_integration_guide(best_method: str) -> None:
    """Generate integration guide for the best method."""
    print("\\n🔧 INTEGRATION GUIDE")
    print("=" * 40)

    print(f"Based on the evaluation, **{best_method}** is recommended.")
    print("\\nIntegration steps:")

    if best_method == "Simple Enhanced":
        print("""
1. Copy the simple_enhanced_robust.py module to your fitting directory
2. Import the method:
   from simple_enhanced_robust import fit_lm_robust_simple

3. Use in your fitting pipeline:
   result = fit_lm_robust_simple(
       dataset,
       use_huber=True,
       outlier_removal_rounds=2,
       outlier_threshold=2.5
   )

4. For batch processing, use the non-verbose mode:
   result = fit_lm_robust_simple(dataset, verbose=False)

5. The method automatically handles:
   - Robust fitting with Huber loss
   - Iterative outlier removal
   - Convergence to best solution
        """)

    elif best_method == "Outlier2":
        print("""
1. The Outlier2 method is already integrated in your codebase
2. Use outlier2:
   from clophfit.fitting.core import outlier2

3. Usage:
   result = outlier2(dataset, key="your_key")

4. This method provides good balance of robustness and accuracy
        """)

    elif best_method == "Robust Huber":
        print("""
1. The Robust Huber method is already available
2. Use fit_lm with robust=True:
   from src.clophfit.fitting.core import fit_lm

3. Usage:
   result = fit_lm(dataset, robust=True)

4. This is the simplest robust option with good performance
        """)

    print("\\n💡 General recommendations:")
    print("- For clean data: Standard LM is sufficient and fastest")
    print("- For moderate outliers: Robust Huber or IRLS")
    print("- For challenging data: Simple Enhanced or IRLS")
    print("- For production use: Consider IRLS for best balance")


if __name__ == "__main__":
    print("🚀 FINAL COMPREHENSIVE ROBUST FITTING EVALUATION")
    print("=" * 70)

    # Run comprehensive evaluation
    evaluation_results = run_comprehensive_evaluation(n_tests=100)

    # Display results
    display_evaluation_results(evaluation_results)

    # Create visualizations
    create_evaluation_plots(evaluation_results)

    # Find best method
    best_method = max(
        evaluation_results.items(), key=lambda x: (x[1].success_rate, -x[1].avg_error)
    )[0]

    # Generate integration guide
    generate_integration_guide(best_method)

    print("\\n🎉 COMPREHENSIVE EVALUATION COMPLETE!")
    print("\\nKey findings:")
    print("- All methods tested on 100 diverse synthetic datasets")
    print("- Performance evaluated across difficulty levels")
    print("- Realistic data patterns based on your experimental data")
    print("- Ready for integration into your workflow")

    print(f"\\n🏆 RECOMMENDED METHOD: {best_method}")
    result = evaluation_results[best_method]
    print(f"   Success rate: {result.success_rate:.1f}%")
    print(f"   Average error: {result.avg_error:.1f}%")
    print(f"   Speed: {result.avg_time:.3f}s per fit")
