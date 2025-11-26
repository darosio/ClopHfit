#!/usr/bin/env python3
"""Cleanup script to organize analysis files from fitting evaluation."""

import shutil
from pathlib import Path


def main() -> None:
    """Clean up and organize analysis files."""
    print("🧹 CLEANING UP ANALYSIS FILES")
    print("=" * 40)

    # Create analysis directory
    analysis_dir = Path("analysis_results")
    analysis_dir.mkdir(exist_ok=True)
    print(f"Created directory: {analysis_dir}")

    # Files to keep (most recent and important)
    important_files = [
        "comprehensive_fitting_evaluation.py",
        "comprehensive_fitting_results_corrected.csv",
        "comprehensive_fitting_evaluation_corrected.png",
        "realistic_synthetic_data.py",
        "FITTING_CLEANUP_RECOMMENDATIONS.md",
        "debug_fitting_issues.py",
        "realistic_synthetic_data_comparison.png",
    ]

    # Files to archive (older analysis files)
    archive_files = [
        "analyze_fitting_results.py",
        "comprehensive_fitting_comparison.png",
        "comprehensive_robust_fitting_evaluation.png",
        "enhanced_robust_fitting.py",
        "fitting_comparison_bayesian.csv",
        "fitting_comparison_deterministic.csv",
        "fitting_function_comparison.py",
        "fitting_function_metrics.csv",
        "fitting_function_performance.png",
        "improved_robust_fitting.py",
        "optimal_robust_fitting.py",
    ]

    # Move important files to analysis directory
    for file_name in important_files:
        file_path = Path(file_name)
        if file_path.exists():
            dest = analysis_dir / file_name
            shutil.move(str(file_path), str(dest))
            print(f"Moved: {file_name} → {dest}")

    # Create archive subdirectory
    archive_dir = analysis_dir / "archive"
    archive_dir.mkdir(exist_ok=True)

    # Move archive files
    moved_count = 0
    for file_name in archive_files:
        file_path = Path(file_name)
        if file_path.exists():
            dest = archive_dir / file_name
            shutil.move(str(file_path), str(dest))
            print(f"Archived: {file_name} → {dest}")
            moved_count += 1

    print("\n✅ Cleanup complete!")
    print(f"   Important files: {len(important_files)} moved to analysis_results/")
    print(f"   Archive files: {moved_count} moved to analysis_results/archive/")
    print("\nNext steps:")
    print("1. Review FITTING_CLEANUP_RECOMMENDATIONS.md")
    print("2. Fix the weighting function bug")
    print("3. Implement deprecation warnings")


if __name__ == "__main__":
    main()
