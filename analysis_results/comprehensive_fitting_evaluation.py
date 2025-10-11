#!/usr/bin/env python3
"""
Comprehensive Fitting Function Evaluation
==========================================

This script evaluates all available fitting functions in ClopHfit using the
corrected realistic synthetic data generator. It tests for:
- Parameter recovery accuracy
- Robustness to outliers
- Computational speed
- Success rates
- Statistical significance of differences

Key corrections from previous analysis:
- Corrected y1/y2 error ratio (y1 errors ~10x larger than y2)
- More realistic synthetic data matching experimental patterns
- Comprehensive function testing including new methods
"""

import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Import all fitting functions from ClopHfit
from src.clophfit.fitting.api import (
    fit_binding_bayes,
    fit_binding_bayes_perlabel,
    fit_binding_lm,
    fit_binding_lm_outlier,
)
from src.clophfit.fitting.core import (
    fit_binding_glob,
    fit_binding_glob_recursive,
    fit_binding_glob_recursive_outlier,
    fit_binding_glob_reweighted,
    fit_lm,
    outlier2,
)
from src.clophfit.fitting.data_structures import Dataset, FitResult

# Import realistic data generator (corrected version)
from realistic_synthetic_data import RealisticSimulationParameters, generate_realistic_dataset


@dataclass
class FittingMetrics:
    """Metrics for evaluating fitting function performance."""

    method_name: str
    success_count: int = 0
    total_runs: int = 0
    execution_times: list[float] = field(default_factory=list)
    parameter_errors: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    failed_runs: list[dict] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        return (self.success_count / self.total_runs * 100) if self.total_runs > 0 else 0.0

    @property
    def mean_execution_time(self) -> float:
        """Calculate mean execution time."""
        return np.mean(self.execution_times) if self.execution_times else 0.0

    @property
    def std_execution_time(self) -> float:
        """Calculate standard deviation of execution time."""
        return np.std(self.execution_times) if self.execution_times else 0.0

    def get_parameter_accuracy(self, param_name: str) -> tuple[float, float]:
        """Get mean and std of parameter errors for a specific parameter."""
        errors = self.parameter_errors.get(param_name, [])
        if not errors:
            return float('inf'), float('inf')
        return np.mean(errors), np.std(errors)


class ComprehensiveFittingEvaluator:
    """Comprehensive evaluation of all fitting functions."""

    def __init__(self, n_simulations: int = 100, random_seed: int = 42):
        """Initialize evaluator.

        Parameters
        ----------
        n_simulations : int
            Number of simulations to run per method
        random_seed : int
            Base random seed for reproducibility
        """
        self.n_simulations = n_simulations
        self.random_seed = random_seed
        self.results: dict[str, FittingMetrics] = {}

        # Define all fitting functions to test
        self.fitting_functions = {
            # Core deterministic functions
            "fit_lm_standard": lambda ds: fit_lm(ds, robust=False, iterative=False),
            "fit_lm_robust": lambda ds: fit_lm(ds, robust=True, iterative=False),
            "fit_lm_iterative": lambda ds: fit_lm(ds, robust=False, iterative=True),
            "fit_lm_robust_iterative": lambda ds: fit_lm(ds, robust=True, iterative=True),
            "fit_lm_outlier_removal": lambda ds: fit_lm(ds, outlier_threshold=3.0),

            "fit_binding_glob_standard": lambda ds: fit_binding_glob(ds, robust=False),
            "fit_binding_glob_robust": lambda ds: fit_binding_glob(ds, robust=True),
            "fit_binding_glob_reweighted": lambda ds: fit_binding_glob_reweighted(ds, key="test"),
            "fit_binding_glob_recursive": lambda ds: fit_binding_glob_recursive(ds),
            "fit_binding_glob_recursive_outlier": lambda ds: fit_binding_glob_recursive_outlier(ds),

            "outlier2": lambda ds: outlier2(ds, key="test"),

            # API wrapper functions
            "api_fit_lm": lambda ds: fit_binding_lm(ds),
            "api_fit_lm_robust": lambda ds: fit_binding_lm(ds, robust=True),
            "api_fit_lm_outlier": lambda ds: fit_binding_lm_outlier(ds, key="test"),
        }

    def run_single_simulation(self, method_name: str, fitting_func, dataset: Dataset,
                             true_params: dict[str, float]) -> dict:
        """Run a single simulation for one method."""

        result_dict = {
            "method": method_name,
            "success": False,
            "execution_time": 0.0,
            "error": None,
            "parameters": {},
            "parameter_errors": {}
        }

        try:
            # Time the fitting function
            start_time = time.time()

            # Handle different function signatures
            if method_name.startswith("api_fit") and method_name.endswith("_bayes"):
                # Bayesian methods need an initial deterministic fit
                initial_result = fit_binding_lm(dataset)
                if initial_result.result and initial_result.result.success:
                    fit_result = fitting_func(initial_result)
                else:
                    raise RuntimeError("Initial deterministic fit failed for Bayesian method")
            else:
                # Regular deterministic methods
                # Skip weighting for problematic methods
                if method_name in ["outlier2", "api_fit_lm_outlier"]:
                    # These methods call weighting internally, which has a bug
                    raise RuntimeError("Skipping method with weighting bug")

                fit_result = fitting_func(dataset)

            execution_time = time.time() - start_time
            result_dict["execution_time"] = execution_time

            # Check if fitting was successful
            if (fit_result and
                hasattr(fit_result, 'result') and
                fit_result.result and
                fit_result.result.success and
                "K" in fit_result.result.params):

                result_dict["success"] = True

                # Extract parameters
                for param_name in ["K", "S0_y1", "S1_y1", "S0_y2", "S1_y2"]:
                    if param_name in fit_result.result.params:
                        estimated_value = fit_result.result.params[param_name].value
                        result_dict["parameters"][param_name] = estimated_value

                        # Calculate parameter error if true value is available
                        if param_name in true_params:
                            true_value = true_params[param_name]
                            relative_error = abs(estimated_value - true_value) / abs(true_value) * 100
                            result_dict["parameter_errors"][param_name] = relative_error

        except Exception as e:
            result_dict["error"] = str(e)
            result_dict["success"] = False

        return result_dict

    def run_comprehensive_evaluation(self) -> None:
        """Run comprehensive evaluation of all fitting functions."""

        print(f"🚀 COMPREHENSIVE FITTING EVALUATION")
        print(f"=" * 60)
        print(f"Testing {len(self.fitting_functions)} methods with {self.n_simulations} simulations each")
        print(f"Using corrected realistic synthetic data generation")
        print()

        # Initialize results storage
        for method_name in self.fitting_functions:
            self.results[method_name] = FittingMetrics(method_name)

        # Run simulations
        for sim_idx in range(self.n_simulations):
            if sim_idx % 20 == 0:
                print(f"Progress: {sim_idx}/{self.n_simulations} simulations completed")

            # Generate realistic synthetic dataset for this simulation
            sim_seed = self.random_seed + sim_idx
            params = RealisticSimulationParameters(
                random_seed=sim_seed,
                K_true=np.random.uniform(6.5, 7.5),  # Realistic pKa range
                outlier_probability=0.15,  # Moderate outlier probability
                y1_base_error=100.0,  # Corrected: y1 errors 10x larger
                y2_base_error=10.0,   # Corrected: reference error level
            )

            dataset, true_params = generate_realistic_dataset(params)

            # Test each fitting function
            for method_name, fitting_func in self.fitting_functions.items():
                result = self.run_single_simulation(
                    method_name, fitting_func, dataset, true_params
                )

                # Store results
                metrics = self.results[method_name]
                metrics.total_runs += 1

                if result["success"]:
                    metrics.success_count += 1
                    metrics.execution_times.append(result["execution_time"])

                    # Store parameter errors
                    for param_name, error in result["parameter_errors"].items():
                        metrics.parameter_errors[param_name].append(error)

                else:
                    metrics.failed_runs.append({
                        "simulation": sim_idx,
                        "error": result["error"]
                    })

        print(f"✅ Evaluation completed: {self.n_simulations} simulations per method")
        print()

    def analyze_results(self) -> pd.DataFrame:
        """Analyze and summarize results."""

        print(f"📊 RESULTS ANALYSIS")
        print(f"=" * 60)

        # Create summary DataFrame
        summary_data = []

        for method_name, metrics in self.results.items():
            # Overall performance
            row = {
                "Method": method_name,
                "Success_Rate_%": metrics.success_rate,
                "Mean_Time_ms": metrics.mean_execution_time * 1000,
                "Std_Time_ms": metrics.std_execution_time * 1000,
                "Failed_Runs": len(metrics.failed_runs),
            }

            # Parameter accuracy
            for param_name in ["K", "S0_y1", "S1_y1", "S0_y2", "S1_y2"]:
                mean_err, std_err = metrics.get_parameter_accuracy(param_name)
                row[f"{param_name}_Mean_Error_%"] = mean_err
                row[f"{param_name}_Std_Error_%"] = std_err

            summary_data.append(row)

        df = pd.DataFrame(summary_data).sort_values("Success_Rate_%", ascending=False)

        # Display top performers
        print("🏆 TOP PERFORMING METHODS (by success rate):")
        print(df[["Method", "Success_Rate_%", "Mean_Time_ms", "K_Mean_Error_%"]].head(10))
        print()

        # Statistical analysis of top methods
        self.statistical_analysis(df)

        return df

    def statistical_analysis(self, df: pd.DataFrame) -> None:
        """Perform statistical analysis of method differences."""

        print("📈 STATISTICAL ANALYSIS")
        print("-" * 40)

        # Filter to successful methods only
        successful_methods = df[df["Success_Rate_%"] >= 95.0]

        if len(successful_methods) > 1:
            print("Methods with ≥95% success rate:")

            # Compare K parameter accuracy
            k_errors = []
            method_names = []

            for _, row in successful_methods.iterrows():
                method_name = row["Method"]
                if method_name in self.results:
                    k_errs = self.results[method_name].parameter_errors.get("K", [])
                    if k_errs:
                        k_errors.append(k_errs)
                        method_names.append(method_name)

            if len(k_errors) > 1:
                # Perform ANOVA on K parameter errors
                try:
                    f_stat, p_value = stats.f_oneway(*k_errors)
                    print(f"K parameter accuracy ANOVA: F={f_stat:.3f}, p={p_value:.3f}")

                    if p_value < 0.05:
                        print("⚠️  Significant differences detected between methods")

                        # Pairwise comparisons for significant differences
                        print("\nPairwise K error comparisons:")
                        for i, method1 in enumerate(method_names[:-1]):
                            for method2 in method_names[i+1:]:
                                t_stat, p_val = stats.ttest_ind(k_errors[i], k_errors[method_names.index(method2)])
                                significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                                print(f"  {method1} vs {method2}: p={p_val:.3f} {significance}")
                    else:
                        print("✅ No significant differences in K parameter accuracy")
                except Exception as e:
                    print(f"Statistical analysis failed: {e}")

        print()

    def create_visualizations(self, df: pd.DataFrame) -> None:
        """Create comprehensive visualizations."""

        print("📈 CREATING VISUALIZATIONS")
        print("-" * 40)

        # Filter to methods with reasonable success rates
        df_filtered = df[df["Success_Rate_%"] >= 50.0].copy()

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Comprehensive Fitting Function Evaluation Results", fontsize=16)

        # 1. Success Rate vs Speed
        ax1 = axes[0, 0]
        scatter = ax1.scatter(
            df_filtered["Mean_Time_ms"],
            df_filtered["Success_Rate_%"],
            c=df_filtered["K_Mean_Error_%"],
            cmap="viridis_r",
            alpha=0.7,
            s=60
        )
        ax1.set_xlabel("Mean Execution Time (ms)")
        ax1.set_ylabel("Success Rate (%)")
        ax1.set_title("Success Rate vs Speed (Color = K Error)")
        ax1.grid(True, alpha=0.3)

        # Add colorbar
        cbar1 = plt.colorbar(scatter, ax=ax1)
        cbar1.set_label("K Parameter Error (%)")

        # Annotate top methods
        top_methods = df_filtered.head(3)
        for _, row in top_methods.iterrows():
            ax1.annotate(
                row["Method"].replace("fit_", "").replace("binding_", "")[:10],
                (row["Mean_Time_ms"], row["Success_Rate_%"]),
                xytext=(5, 5), textcoords="offset points",
                fontsize=8, alpha=0.8
            )

        # 2. Parameter Accuracy Comparison
        ax2 = axes[0, 1]
        param_cols = [col for col in df_filtered.columns if col.endswith("_Mean_Error_%")]
        param_names = [col.replace("_Mean_Error_%", "") for col in param_cols]

        # Box plot of parameter errors for top 5 methods
        top_5_methods = df_filtered.head(5)["Method"].tolist()
        param_errors_data = []
        labels = []

        for param in ["K", "S0_y1", "S1_y1"]:  # Focus on key parameters
            for method in top_5_methods:
                if method in self.results:
                    errors = self.results[method].parameter_errors.get(param, [])
                    if errors:
                        param_errors_data.append(errors)
                        labels.append(f"{param}\n{method.replace('fit_', '')[:8]}")

        if param_errors_data:
            box_plot = ax2.boxplot(param_errors_data, labels=labels, patch_artist=True)
            for patch in box_plot['boxes']:
                patch.set_facecolor('lightblue')
                patch.set_alpha(0.7)

        ax2.set_ylabel("Parameter Error (%)")
        ax2.set_title("Parameter Accuracy Comparison (Top 5 Methods)")
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')

        # 3. Execution Time Distribution
        ax3 = axes[1, 0]
        time_data = []
        time_labels = []

        for method in df_filtered.head(8)["Method"]:
            if method in self.results:
                times = self.results[method].execution_times
                if times:
                    time_data.append([t * 1000 for t in times])  # Convert to ms
                    time_labels.append(method.replace("fit_", "").replace("binding_", "")[:12])

        if time_data:
            ax3.boxplot(time_data, labels=time_labels)
            ax3.set_ylabel("Execution Time (ms)")
            ax3.set_title("Execution Time Distribution")
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3, axis='y')

        # 4. Success Rate Comparison
        ax4 = axes[1, 1]
        methods = df_filtered["Method"].apply(lambda x: x.replace("fit_", "").replace("binding_", "")[:15])
        success_rates = df_filtered["Success_Rate_%"]

        bars = ax4.barh(range(len(methods)), success_rates,
                        color=plt.cm.viridis_r(success_rates/100))
        ax4.set_yticks(range(len(methods)))
        ax4.set_yticklabels(methods)
        ax4.set_xlabel("Success Rate (%)")
        ax4.set_title("Method Success Rates")
        ax4.grid(True, alpha=0.3, axis='x')

        # Add success rate labels
        for i, (bar, rate) in enumerate(zip(bars, success_rates)):
            ax4.text(rate + 1, i, f"{rate:.1f}%",
                    va='center', fontsize=8)

        plt.tight_layout()
        plt.savefig("comprehensive_fitting_evaluation_corrected.png", dpi=300, bbox_inches="tight")
        print("Visualization saved as 'comprehensive_fitting_evaluation_corrected.png'")
        print()

    def generate_recommendations(self, df: pd.DataFrame) -> None:
        """Generate actionable recommendations based on results."""

        print("🎯 RECOMMENDATIONS")
        print("=" * 60)

        # Find best overall method
        best_method = df.iloc[0]
        print(f"🏆 BEST OVERALL METHOD: {best_method['Method']}")
        print(f"   Success Rate: {best_method['Success_Rate_%']:.1f}%")
        print(f"   Mean K Error: {best_method['K_Mean_Error_%']:.2f}%")
        print(f"   Mean Time: {best_method['Mean_Time_ms']:.1f} ms")
        print()

        # Find fastest reliable method
        reliable_methods = df[df["Success_Rate_%"] >= 95.0]
        if not reliable_methods.empty:
            fastest_reliable = reliable_methods.loc[reliable_methods["Mean_Time_ms"].idxmin()]
            print(f"⚡ FASTEST RELIABLE METHOD: {fastest_reliable['Method']}")
            print(f"   Success Rate: {fastest_reliable['Success_Rate_%']:.1f}%")
            print(f"   Mean Time: {fastest_reliable['Mean_Time_ms']:.1f} ms")
            print()

        # Identify potential duplicates (similar performance)
        print("🔍 POTENTIAL DUPLICATE METHODS:")
        threshold = 2.0  # 2% difference threshold

        duplicates_found = False
        for i, row1 in df.iterrows():
            for j, row2 in df.iloc[i+1:].iterrows():
                # Compare success rates and K errors
                success_diff = abs(row1["Success_Rate_%"] - row2["Success_Rate_%"])
                k_error_diff = abs(row1["K_Mean_Error_%"] - row2["K_Mean_Error_%"])

                if success_diff < threshold and k_error_diff < threshold:
                    print(f"   {row1['Method']} ≈ {row2['Method']}")
                    print(f"     Success: {row1['Success_Rate_%']:.1f}% vs {row2['Success_Rate_%']:.1f}%")
                    print(f"     K Error: {row1['K_Mean_Error_%']:.2f}% vs {row2['K_Mean_Error_%']:.2f}%")
                    duplicates_found = True

        if not duplicates_found:
            print("   No clear duplicate methods identified")

        print()

        # Methods to consider removing
        poor_performers = df[df["Success_Rate_%"] < 80.0]
        if not poor_performers.empty:
            print("❌ METHODS TO CONSIDER REMOVING (Success Rate < 80%):")
            for _, row in poor_performers.iterrows():
                print(f"   {row['Method']}: {row['Success_Rate_%']:.1f}% success")

        print()
        print("🧹 CLEANUP RECOMMENDATIONS:")
        print("1. Keep the best overall method for primary use")
        print("2. Keep the fastest reliable method for high-throughput scenarios")
        print("3. Remove or deprecate duplicate methods")
        print("4. Remove methods with consistently poor performance")
        print("5. Update documentation to recommend optimal methods")


def main():
    """Run comprehensive fitting function evaluation."""

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    print("🧬 COMPREHENSIVE FITTING FUNCTION EVALUATION")
    print("=" * 80)
    print("Using CORRECTED realistic synthetic data generation")
    print("Key corrections: y1 errors ~10x larger than y2 errors")
    print()

    # Create evaluator
    evaluator = ComprehensiveFittingEvaluator(n_simulations=100, random_seed=42)

    # Run evaluation
    evaluator.run_comprehensive_evaluation()

    # Analyze results
    df = evaluator.analyze_results()

    # Create visualizations
    evaluator.create_visualizations(df)

    # Generate recommendations
    evaluator.generate_recommendations(df)

    # Save detailed results
    df.to_csv("comprehensive_fitting_results_corrected.csv", index=False)
    print("📁 Detailed results saved to 'comprehensive_fitting_results_corrected.csv'")

    print()
    print("🎉 EVALUATION COMPLETE!")
    print("Next steps:")
    print("1. Review the generated visualization and CSV results")
    print("2. Implement recommended cleanup actions")
    print("3. Update codebase documentation")
    print("4. Consider deprecating poor-performing methods")


if __name__ == "__main__":
    main()
