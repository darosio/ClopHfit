#!/usr/bin/env python3
"""
Advanced analysis of fitting function comparison results.

This script analyzes the comprehensive fitting function comparison results
to provide detailed insights and recommendations.
"""

from __future__ import annotations

import operator
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_and_validate_results(
    csv_file: str = "fitting_comparison_deterministic.csv",
) -> pd.DataFrame:
    """Load and validate the comparison results."""
    if not Path(csv_file).exists():
        # Try alternative file names
        alt_files = [
            "fitting_comparison_bayesian.csv",
            "fitting_comparison_results.csv",
        ]
        for alt_file in alt_files:
            if Path(alt_file).exists():
                csv_file = alt_file
                break
        else:
            msg = "No results file found. Run fitting_function_comparison.py first."
            raise FileNotFoundError(msg)

    df = pd.read_csv(csv_file)
    print(
        f"Loaded {len(df)} results from {df['function_name'].nunique()} functions across {df['simulation_id'].nunique()} simulations"
    )
    print(f"Using file: {csv_file}")
    return df


def identify_duplicate_functions(
    df: pd.DataFrame, tolerance: float = 1e-6
) -> dict[str, list[str]]:
    """Identify functions that produce nearly identical results."""
    duplicates = {}

    # Only compare successful fits with valid K estimates
    successful = df[df["success"] & df["K_estimate"].notna()].copy()

    if successful.empty:
        return duplicates

    # Group by simulation to compare results for the same input data
    function_pairs = []
    functions = successful["function_name"].unique()

    for func1 in functions:
        for func2 in functions:
            if func1 >= func2:  # Avoid duplicate comparisons
                continue

            # Get results for both functions
            f1_results = successful[successful["function_name"] == func1].set_index(
                "simulation_id"
            )["K_estimate"]
            f2_results = successful[successful["function_name"] == func2].set_index(
                "simulation_id"
            )["K_estimate"]

            # Find common simulations
            common_sims = f1_results.index.intersection(f2_results.index)

            if len(common_sims) < 10:  # Need sufficient overlap
                continue

            # Compare K estimates for common simulations
            f1_vals = f1_results.loc[common_sims]
            f2_vals = f2_results.loc[common_sims]

            # Calculate relative differences
            rel_diffs = np.abs(f1_vals - f2_vals) / np.maximum(np.abs(f1_vals), 1e-10)
            mean_rel_diff = rel_diffs.mean()

            if mean_rel_diff < tolerance:
                pair_key = f"{func1} ≈ {func2}"
                function_pairs.append((pair_key, mean_rel_diff, len(common_sims)))

    # Group similar functions
    if function_pairs:
        function_pairs.sort(key=operator.itemgetter(1))  # Sort by similarity
        duplicates = {
            pair[0]: {"mean_difference": pair[1], "comparisons": pair[2]}
            for pair in function_pairs[:10]
        }

    return duplicates


def calculate_robustness_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate robustness metrics for each function."""
    metrics = []

    for func_name in df["function_name"].unique():
        func_data = df[df["function_name"] == func_name]

        total_attempts = len(func_data)
        successful = func_data[func_data["success"]]

        if len(successful) == 0:
            metrics.append({
                "function_name": func_name,
                "success_rate": 0.0,
                "parameter_accuracy": float("inf"),
                "parameter_precision": float("inf"),
                "outlier_robustness": 0.0,
                "avg_execution_time": func_data["execution_time"].mean(),
                "stability_score": 0.0,
            })
            continue

        # Basic metrics
        success_rate = len(successful) / total_attempts
        avg_exec_time = successful["execution_time"].mean()

        # Parameter recovery accuracy (lower is better)
        valid_recovery = successful["parameter_recovery_error"].dropna()
        param_accuracy = (
            valid_recovery.mean() if not valid_recovery.empty else float("inf")
        )
        param_precision = (
            valid_recovery.std() if len(valid_recovery) > 1 else float("inf")
        )

        # Outlier robustness: success rate when outliers are present
        outlier_data = successful[successful["has_outliers"]]
        outlier_robustness = (
            len(outlier_data) / len(func_data[func_data["has_outliers"]])
            if len(func_data[func_data["has_outliers"]]) > 0
            else 0.0
        )

        # Overall stability score (combines success rate, accuracy, and robustness)
        # Higher is better, scaled 0-100
        if param_accuracy == float("inf"):
            stability_score = 0.0
        else:
            accuracy_score = max(
                0, 1 - param_accuracy
            )  # Convert to 0-1 where 1 is perfect
            stability_score = (
                success_rate * 0.4 + accuracy_score * 0.4 + outlier_robustness * 0.2
            ) * 100

        metrics.append({
            "function_name": func_name,
            "success_rate": success_rate,
            "parameter_accuracy": param_accuracy,
            "parameter_precision": param_precision,
            "outlier_robustness": outlier_robustness,
            "avg_execution_time": avg_exec_time,
            "stability_score": stability_score,
        })

    return pd.DataFrame(metrics).sort_values("stability_score", ascending=False)


def create_performance_plots(df: pd.DataFrame, metrics_df: pd.DataFrame) -> None:
    """Create comprehensive performance visualization plots."""
    plt.style.use("default")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(
        "Fitting Function Performance Comparison", fontsize=16, fontweight="bold"
    )

    # 1. Success Rate vs Execution Time
    ax1 = axes[0, 0]
    scatter = ax1.scatter(
        metrics_df["avg_execution_time"],
        metrics_df["success_rate"],
        c=metrics_df["stability_score"],
        cmap="RdYlGn",
        s=80,
        alpha=0.7,
    )
    ax1.set_xlabel("Average Execution Time (seconds)")
    ax1.set_ylabel("Success Rate")
    ax1.set_title("Success Rate vs Speed")
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label="Stability Score")

    # Add function name labels for top performers
    top_performers = metrics_df.nlargest(8, "stability_score")
    for _, row in top_performers.iterrows():
        ax1.annotate(
            row["function_name"],
            (row["avg_execution_time"], row["success_rate"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            alpha=0.8,
        )

    # 2. Parameter Recovery Accuracy Distribution
    ax2 = axes[0, 1]
    successful_fits = df[df["success"] & df["parameter_recovery_error"].notna()]
    if not successful_fits.empty:
        # Select top 10 functions by stability score for clarity
        top_functions = metrics_df.head(10)["function_name"].tolist()
        plot_data = successful_fits[
            successful_fits["function_name"].isin(top_functions)
        ]

        sns.boxplot(
            data=plot_data, y="function_name", x="parameter_recovery_error", ax=ax2
        )
        ax2.set_title("Parameter Recovery Error Distribution")
        ax2.set_xlabel("Parameter Recovery Error (lower is better)")
        ax2.set_xlim(0, plot_data["parameter_recovery_error"].quantile(0.95))

    # 3. Outlier Robustness Comparison
    ax3 = axes[1, 0]
    outlier_comparison = []
    for func in metrics_df["function_name"].unique():
        func_data = df[df["function_name"] == func]
        no_outliers = func_data[~func_data["has_outliers"]]
        with_outliers = func_data[func_data["has_outliers"]]

        if len(no_outliers) > 0 and len(with_outliers) > 0:
            success_no_outliers = no_outliers["success"].mean()
            success_with_outliers = with_outliers["success"].mean()
            robustness_drop = success_no_outliers - success_with_outliers

            outlier_comparison.append({
                "function_name": func,
                "success_no_outliers": success_no_outliers,
                "success_with_outliers": success_with_outliers,
                "robustness_drop": robustness_drop,
            })

    if outlier_comparison:
        outlier_df = pd.DataFrame(outlier_comparison).sort_values("robustness_drop")
        top_robust = outlier_df.head(12)  # Show top 12 most robust

        x = range(len(top_robust))
        ax3.bar(
            x,
            top_robust["success_no_outliers"],
            alpha=0.7,
            label="No Outliers",
            color="lightblue",
        )
        ax3.bar(
            x,
            top_robust["success_with_outliers"],
            alpha=0.7,
            label="With Outliers",
            color="lightcoral",
        )

        ax3.set_xlabel("Functions (ranked by robustness)")
        ax3.set_ylabel("Success Rate")
        ax3.set_title("Outlier Robustness Comparison")
        ax3.set_xticks(x)
        ax3.set_xticklabels(
            [
                name[:12] + "..." if len(name) > 12 else name
                for name in top_robust["function_name"]
            ],
            rotation=45,
            ha="right",
        )
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # 4. Overall Performance Ranking
    ax4 = axes[1, 1]
    top_15 = metrics_df.head(15)
    bars = ax4.barh(
        range(len(top_15)), top_15["stability_score"], color="skyblue", alpha=0.8
    )
    ax4.set_yticks(range(len(top_15)))
    ax4.set_yticklabels([
        name[:15] + "..." if len(name) > 15 else name
        for name in top_15["function_name"]
    ])
    ax4.set_xlabel("Stability Score (0-100)")
    ax4.set_title("Overall Function Ranking")
    ax4.grid(True, alpha=0.3, axis="x")

    # Add score labels on bars
    for bar in bars:
        width = bar.get_width()
        ax4.text(
            width + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.1f}",
            ha="left",
            va="center",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig("fitting_function_performance.png", dpi=300, bbox_inches="tight")
    print("Performance plots saved as 'fitting_function_performance.png'")


def generate_recommendations(metrics_df: pd.DataFrame, duplicates: dict) -> str:
    """Generate detailed recommendations based on analysis."""
    recommendations = []
    recommendations.extend((
        "# FITTING FUNCTION ANALYSIS RECOMMENDATIONS\n",
        "=" * 60 + "\n",
    ))

    # Top performers
    top_5 = metrics_df.head(5)
    recommendations.extend((
        "## 🏆 TOP PERFORMING FUNCTIONS\n",
        "Based on stability score (success rate + accuracy + outlier robustness):\n",
    ))
    for i, (_, row) in enumerate(top_5.iterrows(), 1):
        recommendations.extend((
            f"{i}. **{row['function_name']}**",
            f"   - Stability Score: {row['stability_score']:.1f}/100",
            f"   - Success Rate: {row['success_rate']:.1%}",
            f"   - Parameter Accuracy: {row['parameter_accuracy']:.3f}",
            f"   - Outlier Robustness: {row['outlier_robustness']:.1%}",
            f"   - Avg Execution Time: {row['avg_execution_time']:.3f}s\n",
        ))

    # Duplicates identification
    if duplicates:
        recommendations.extend((
            "## 🔍 DUPLICATE/EQUIVALENT FUNCTIONS\n",
            "These functions produce nearly identical results:\n",
        ))
        for dup_pair, info in duplicates.items():
            recommendations.extend((
                f"- **{dup_pair}**",
                f"  - Mean difference: {info['mean_difference']:.2e}",
                f"  - Based on {info['comparisons']} comparisons\n",
            ))
    else:
        recommendations.extend((
            "## 🔍 DUPLICATE/EQUIVALENT FUNCTIONS\n",
            "No exact duplicates found, but further investigation may reveal near-duplicates.\n",
        ))

    # Speed vs accuracy analysis
    fast_accurate = metrics_df[
        (metrics_df["avg_execution_time"] < metrics_df["avg_execution_time"].median())
        & (metrics_df["stability_score"] > 70)
    ]

    recommendations.extend((
        "## ⚡ FAST & RELIABLE FUNCTIONS\n",
        "Functions with below-median execution time and high stability:\n",
    ))
    if not fast_accurate.empty:
        for _, row in fast_accurate.head(5).iterrows():
            recommendations.append(
                f"- **{row['function_name']}** (Score: {row['stability_score']:.1f}, Time: {row['avg_execution_time']:.3f}s)"
            )
    else:
        recommendations.append(
            "No functions found that are both fast and highly stable.\n"
        )
    recommendations.append("")

    # Functions to avoid
    poor_performers = metrics_df[metrics_df["stability_score"] < 30]
    recommendations.extend((
        "## ⚠️  PROBLEMATIC FUNCTIONS\n",
        "Functions with low stability scores (<30):\n",
    ))
    if not poor_performers.empty:
        for _, row in poor_performers.iterrows():
            recommendations.append(
                f"- **{row['function_name']}** (Score: {row['stability_score']:.1f})"
            )
            if row["success_rate"] < 0.5:
                recommendations.append("  - Low success rate")
            if row["parameter_accuracy"] > 0.5:
                recommendations.append("  - Poor parameter recovery")
            if row["outlier_robustness"] < 0.3:
                recommendations.append("  - Not robust to outliers")
        recommendations.append("")
    else:
        recommendations.append("All functions show reasonable stability.\n")

    # Specific use case recommendations
    recommendations.append("## 🎯 USE CASE RECOMMENDATIONS\n")

    # For typical users
    general_use = metrics_df[
        (metrics_df["success_rate"] > 0.8)
        & (metrics_df["outlier_robustness"] > 0.6)
        & (metrics_df["stability_score"] > 60)
    ].head(3)

    recommendations.append(
        "### For General Use (high success rate + outlier robustness):"
    )
    for _, row in general_use.iterrows():
        recommendations.append(f"- `{row['function_name']}`")
    recommendations.append("")

    # For speed-critical applications
    fast_functions = metrics_df[
        (metrics_df["success_rate"] > 0.7) & (metrics_df["avg_execution_time"] < 0.1)
    ].head(3)

    recommendations.append("### For Speed-Critical Applications (fast + reliable):")
    if not fast_functions.empty:
        for _, row in fast_functions.iterrows():
            recommendations.append(
                f"- `{row['function_name']}` ({row['avg_execution_time']:.3f}s)"
            )
    else:
        recommendations.append(
            "- Consider the fastest functions from top performers list"
        )
    recommendations.append("")

    # For maximum accuracy
    accurate_functions = metrics_df[
        (metrics_df["success_rate"] > 0.7) & (metrics_df["parameter_accuracy"] < 0.1)
    ].head(3)

    recommendations.append("### For Maximum Parameter Accuracy:")
    if not accurate_functions.empty:
        for _, row in accurate_functions.iterrows():
            recommendations.append(
                f"- `{row['function_name']}` (accuracy: {row['parameter_accuracy']:.3f})"
            )
    else:
        recommendations.append("- Use top-ranked functions from stability score")
    recommendations.append("")

    # Cleanup recommendations
    recommendations.append("## 🧹 CLEANUP RECOMMENDATIONS\n")

    # Functions to deprecate/remove
    deprecated_candidates = metrics_df[
        (metrics_df["stability_score"] < 50) | (metrics_df["success_rate"] < 0.6)
    ]

    if not deprecated_candidates.empty:
        recommendations.append("### Functions to Consider for Deprecation:")
        for _, row in deprecated_candidates.iterrows():
            recommendations.append(
                f"- `{row['function_name']}` - Poor performance (score: {row['stability_score']:.1f})"
            )
        recommendations.append("")

    # API consolidation
    metrics_df[metrics_df["function_name"].str.contains("api_|fit_binding_")]
    metrics_df[
        metrics_df["function_name"].str.contains("core_|fit_lm|fit_binding_glob")
    ]

    recommendations.append("### API Consolidation:")
    recommendations.append(
        "- Keep the unified `fit_binding()` dispatcher as the primary interface"
    )
    recommendations.append(
        "- Maintain the best-performing direct functions for advanced users"
    )
    recommendations.append("- Remove deprecated shim functions after migration")
    recommendations.append("")

    recommendations.append("## 📋 MIGRATION PLAN\n")
    recommendations.append(
        "1. **Phase 1**: Update documentation to recommend top performers"
    )
    recommendations.append(
        "2. **Phase 2**: Add deprecation warnings to poor performers"
    )
    recommendations.append("3. **Phase 3**: Consolidate duplicate functions")
    recommendations.append(
        "4. **Phase 4**: Remove deprecated functions in next major version"
    )
    recommendations.append("")

    # Statistical significance note
    recommendations.append("## 📊 STATISTICAL NOTES\n")
    recommendations.append(
        f"- Analysis based on {len(metrics_df)} functions across 100 simulations"
    )
    recommendations.append(
        "- Results are specific to the tested conditions (2 DataArrays, outliers in last 2 pH points)"
    )
    recommendations.append(
        "- Consider running additional tests with different data characteristics"
    )
    recommendations.append(
        "- Parameter recovery error is normalized relative error across all parameters"
    )

    return "\n".join(recommendations)


def main() -> None:
    """Main analysis workflow."""
    print("Loading and analyzing fitting function comparison results...\n")

    # Load results
    try:
        df = load_and_validate_results()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run 'python fitting_function_comparison.py' first.")
        return

    # Calculate performance metrics
    print("Calculating performance metrics...")
    metrics_df = calculate_robustness_metrics(df)

    # Identify duplicates
    print("Identifying duplicate functions...")
    duplicates = identify_duplicate_functions(df)

    # Create visualizations
    print("Creating performance plots...")
    create_performance_plots(df, metrics_df)

    # Generate recommendations
    print("Generating recommendations...")
    recommendations = generate_recommendations(metrics_df, duplicates)

    # Save detailed results
    metrics_df.to_csv("fitting_function_metrics.csv", index=False)
    print("Detailed metrics saved to 'fitting_function_metrics.csv'")

    with Path("fitting_function_recommendations.md").open("w", encoding="utf-8") as f:
        f.write(recommendations)
    print("Recommendations saved to 'fitting_function_recommendations.md'")

    # Print summary to console
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\nTop 5 Functions by Stability Score:")
    for i, (_, row) in enumerate(metrics_df.head(5).iterrows(), 1):
        print(f"{i}. {row['function_name']} (Score: {row['stability_score']:.1f})")

    if duplicates:
        print(f"\nFound {len(duplicates)} potential duplicate function pairs")

    print("\nFull analysis saved to:")
    print("  - fitting_function_metrics.csv")
    print("  - fitting_function_recommendations.md")
    print("  - fitting_function_performance.png")


if __name__ == "__main__":
    main()
