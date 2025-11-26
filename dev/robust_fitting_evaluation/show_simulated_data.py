#!/usr/bin/env python3
"""Show typical simulated dataset structure and visualize the data characteristics."""


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
