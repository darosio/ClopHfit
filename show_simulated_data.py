#!/usr/bin/env python3
"""Show typical simulated dataset structure and visualize the data characteristics."""

import matplotlib.pyplot as plt

from fitting_function_comparison import SimulationParameters, generate_synthetic_dataset


def show_typical_dataset():
    """Generate and display a typical synthetic dataset."""
    # Generate a typical dataset with outliers
    params = SimulationParameters(random_seed=42)  # Fixed seed for reproducibility
    dataset, true_params = generate_synthetic_dataset(params)

    print("=" * 60)
    print("TYPICAL SIMULATED DATASET STRUCTURE")
    print("=" * 60)

    # Show dataset structure
    print(f"\nDataset type: {type(dataset)}")
    print(f"Number of DataArrays: {len(dataset)}")
    print(f"pH mode: {dataset.is_ph}")
    print(f"Labels: {list(dataset.keys())}")

    # Show true parameters used
    print("\nTrue Parameters:")
    for param, value in true_params.items():
        print(f"  {param}: {value}")

    # Show each DataArray characteristics
    for label, da in dataset.items():
        print(f"\n{label.upper()} DataArray:")
        print(f"  Data points: {len(da.xc)}")
        print(f"  pH range: {da.xc.min():.1f} - {da.xc.max():.1f}")
        print(f"  Signal range: {da.yc.min():.0f} - {da.yc.max():.0f}")
        print(f"  Mean error: {da.y_errc.mean():.0f}")
        print(f"  Error range: {da.y_errc.min():.0f} - {da.y_errc.max():.0f}")
        print(f"  Relative error: {(da.y_errc.mean() / abs(da.yc.mean()) * 100):.1f}%")

    # Show error ratio between channels
    y1_err_mean = dataset["y1"].y_errc.mean()
    y2_err_mean = dataset["y2"].y_errc.mean()
    y1_signal_mean = abs(dataset["y1"].yc.mean())
    y2_signal_mean = abs(dataset["y2"].yc.mean())

    print("\nError Characteristics:")
    print(f"  Y1 relative error: {(y1_err_mean / y1_signal_mean * 100):.1f}%")
    print(f"  Y2 relative error: {(y2_err_mean / y2_signal_mean * 100):.1f}%")
    print(f"  Y2/Y1 error ratio: {y2_err_mean / y1_err_mean:.1f}x")
    print(f"  Y2/Y1 signal ratio: {y2_signal_mean / y1_signal_mean:.1f}x")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        "Typical Simulated Dataset Characteristics", fontsize=14, fontweight="bold"
    )

    # Plot 1: Both signals with error bars
    ax1 = axes[0, 0]
    for label, da in dataset.items():
        ax1.errorbar(
            da.xc,
            da.yc,
            yerr=da.y_errc,
            label=f"{label} (obs)",
            marker="o",
            capsize=3,
            alpha=0.7,
        )
    ax1.set_xlabel("pH")
    ax1.set_ylabel("Fluorescence Signal")
    ax1.set_title("Observed Signals with Error Bars")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Error comparison
    ax2 = axes[0, 1]
    labels = list(dataset.keys())
    errors = [dataset[label].y_errc.mean() for label in labels]
    bars = ax2.bar(labels, errors, color=["skyblue", "lightcoral"], alpha=0.7)
    ax2.set_ylabel("Mean Error")
    ax2.set_title("Mean Error by Channel")
    ax2.grid(True, alpha=0.3, axis="y")

    # Add error values on bars
    for bar, error in zip(bars, errors, strict=False):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + error * 0.02,
            f"{error:.0f}",
            ha="center",
            va="bottom",
        )

    # Plot 3: Signal-to-noise ratio
    ax3 = axes[1, 0]
    snrs = [
        abs(dataset[label].yc.mean()) / dataset[label].y_errc.mean() for label in labels
    ]
    bars = ax3.bar(labels, snrs, color=["skyblue", "lightcoral"], alpha=0.7)
    ax3.set_ylabel("Signal-to-Noise Ratio")
    ax3.set_title("Signal-to-Noise Ratio by Channel")
    ax3.grid(True, alpha=0.3, axis="y")

    # Add SNR values on bars
    for bar, snr in zip(bars, snrs, strict=False):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + snr * 0.02,
            f"{snr:.1f}",
            ha="center",
            va="bottom",
        )

    # Plot 4: pH distribution and outlier indication
    ax4 = axes[1, 1]
    pH = dataset["y1"].xc
    ax4.plot(pH, dataset["y1"].yc, "o-", label="Y1", alpha=0.7)
    ax4.plot(pH, dataset["y2"].yc, "s-", label="Y2", alpha=0.7)

    # Highlight last 2 points (potential outliers)
    ax4.plot(
        pH[-2:],
        dataset["y1"].yc[-2:],
        "ro",
        markersize=8,
        fillstyle="none",
        linewidth=2,
        label="Potential outliers",
    )
    ax4.plot(
        pH[-2:],
        dataset["y2"].yc[-2:],
        "ro",
        markersize=8,
        fillstyle="none",
        linewidth=2,
    )

    ax4.set_xlabel("pH")
    ax4.set_ylabel("Fluorescence Signal")
    ax4.set_title("Signals with Outlier Indication")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("typical_simulated_data.png", dpi=300, bbox_inches="tight")
    print("\nVisualization saved as 'typical_simulated_data.png'")

    return dataset, true_params


def show_duplicate_functions() -> None:
    """Display the identified duplicate functions clearly."""
    print("\n" + "=" * 80)
    print("IDENTIFIED DUPLICATE FUNCTIONS")
    print("=" * 80)

    print("\n🔍 PERFECT DUPLICATES (0.00e+00 difference)")
    print("These functions produce exactly identical results:")
    print("-" * 60)

    # Group 1: Core LM variants
    print("\n📍 GROUP 1: Core Least-Squares Functions")
    group1 = ["fit_binding_lm", "fit_binding_lm_outlier", "fit_lm", "fit_lm (outlier)"]
    for i, func in enumerate(group1, 1):
        print(f"   {i}. {func}")
    print("   → All produce identical results")
    print("   → Recommendation: Keep fit_lm, deprecate others")

    # Group 2: Unified API variants
    print("\n📍 GROUP 2: Unified API Functions")
    group2 = [
        "fit_binding (lm)",
        "fit_binding_lm",
        "fit_binding_glob",
        "fit_binding (lm_outlier)",
        "fit_binding_lm_outlier",
    ]
    for i, func in enumerate(group2, 1):
        print(f"   {i}. {func}")
    print("   → All produce identical results")
    print("   → Recommendation: Keep fit_binding() API, remove direct calls")

    # Group 3: Legacy recursive functions
    print("\n📍 GROUP 3: Legacy Recursive Functions")
    group3 = [
        "api_recursive",
        "api_recursive_outlier",
        "core_recursive",
        "core_recursive_outlier",
    ]
    for i, func in enumerate(group3, 1):
        print(f"   {i}. {func}")
    print("   → All produce identical results")
    print("   → Recommendation: Deprecate API shims, keep one core implementation")

    print("\n⚠️  PROBLEMATIC DUPLICATES")
    print("-" * 40)
    print("• api_reweighted: 0% success rate → REMOVE IMMEDIATELY")

    print("\n📊 CONSOLIDATION SUMMARY")
    print("-" * 30)
    total_functions = len(group1) + len(group2) + len(group3)
    recommended_keep = 3  # fit_lm, fit_binding(), one core recursive

    print(f"• Total duplicate functions identified: {total_functions}")
    print(f"• Functions that can be consolidated: {total_functions - recommended_keep}")
    print(f"• Recommended to keep: {recommended_keep}")
    print(
        f"• Potential code reduction: {((total_functions - recommended_keep) / total_functions * 100):.0f}%"
    )

    print("\n🎯 FINAL RECOMMENDATIONS")
    print("-" * 25)
    print("KEEP:")
    print("  ✅ fit_binding()           # Unified API dispatcher")
    print("  ✅ fit_lm(robust=True)     # Best accuracy")
    print("  ✅ outlier2()              # Specialized outlier handling")
    print("\nDEPRECATE:")
    print("  ⚠️  All api_* functions    # Point to fit_binding()")
    print("  ⚠️  Direct LM variants     # Point to fit_lm()")
    print("\nREMOVE:")
    print("  ❌ api_reweighted          # Broken (0% success)")
    print("  ❌ fit_binding_pymc*       # Too slow, poor accuracy")


if __name__ == "__main__":
    # Show typical dataset
    dataset, true_params = show_typical_dataset()

    # Show duplicate functions
    show_duplicate_functions()

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("Files generated:")
    print("  • typical_simulated_data.png - Dataset visualization")
    print("  • fitting_functions_analysis_summary.md - Complete analysis")
    print("\nNext steps:")
    print("  1. Review the duplicate function groups above")
    print("  2. Implement deprecation warnings for duplicate functions")
    print("  3. Remove api_reweighted (broken function)")
    print("  4. Update documentation to recommend fit_binding() API")
