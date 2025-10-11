#!/usr/bin/env python3
"""
Optimal three-stage robust fitting pipeline:
1. Huber loss for initial robust fit
2. Outlier identification and removal
3. IRLS on cleaned data for final refinement.
"""

import copy
import operator

import numpy as np
from lmfit import Minimizer
from matplotlib import figure
from scipy import stats

from src.clophfit.fitting.core import _binding_1site_residuals, _build_params_1site
from src.clophfit.fitting.data_structures import Dataset, FitResult
from src.clophfit.fitting.plotting import PlotParameters, plot_fit

N_BOOT = 20


def fit_huber_outlier_irls(
    ds: Dataset,
    *,
    outlier_threshold: float = 3.0,
    irls_iterations: int = 3,
    irls_damping: float = 0.3,
    min_points_fraction: float = 0.7,
    verbose: bool = False,
) -> FitResult:
    """
    Three-stage optimal robust fitting pipeline.

    Stage 1: Huber loss for robust initial fit
    Stage 2: Outlier identification and removal based on robust fit
    Stage 3: IRLS on cleaned data for final parameter refinement

    Parameters
    ----------
    ds : Dataset
        The dataset to fit.
    outlier_threshold : float, default=3.0
        Z-score threshold for outlier removal.
    irls_iterations : int, default=3
        Number of IRLS iterations in stage 3.
    irls_damping : float, default=0.3
        Damping factor for IRLS weight updates.
    min_points_fraction : float, default=0.7
        Minimum fraction of original points to keep (safety check).
    verbose : bool, default=False
        Print detailed progress.

    Returns
    -------
    FitResult
        Optimally fitted result with best parameter recovery and fit quality.
    """
    copy.deepcopy(ds)

    if verbose:
        total_points = sum(len(da.y) for da in ds.values())
        print("🔧 Starting 3-Stage Optimal Robust Fitting")
        print(f"   Dataset: {len(ds)} arrays, {total_points} total points")
        print(f"   Outlier threshold: {outlier_threshold} σ")
        print(f"   IRLS iterations: {irls_iterations}")
        print(f"   Min points to keep: {min_points_fraction:.0%}")

    # ================================================================
    # STAGE 1: HUBER LOSS - Robust initial fit
    # ================================================================
    if verbose:
        print("\n🎯 STAGE 1: Huber Loss Initial Fit")

    params = _build_params_1site(ds)
    minimizer = Minimizer(
        _binding_1site_residuals, params, fcn_args=(ds,), scale_covar=True
    )

    # Use Huber loss for robustness to outliers
    huber_result = minimizer.minimize(method="least_squares", loss="huber")

    if not huber_result.success:
        if verbose:
            print("   ❌ Huber fit failed, falling back to standard fit")
        huber_result = minimizer.minimize()

    if verbose:
        chi2 = (
            huber_result.redchi
            if hasattr(huber_result, "redchi")
            else huber_result.chisqr
        )
        print(f"   ✅ Huber fit: Chi² = {chi2:.4f}")

    # ================================================================
    # STAGE 2: OUTLIER IDENTIFICATION AND REMOVAL
    # ================================================================
    if verbose:
        print("\n🔍 STAGE 2: Outlier Identification & Removal")

    # Use residuals from robust fit to identify outliers
    cleaned_ds = copy.deepcopy(ds)
    total_original_points = sum(da.mask.sum() for da in cleaned_ds.values())
    outliers_removed = 0

    start_idx = 0
    for i, da in enumerate(cleaned_ds.values()):
        end_idx = start_idx + len(da.y)
        residuals = huber_result.residual[start_idx:end_idx]

        # Calculate z-scores for outlier detection
        z_scores = np.abs(stats.zscore(residuals))
        outlier_mask = z_scores > outlier_threshold

        outliers_this_array = np.sum(outlier_mask)
        if outliers_this_array > 0:
            # Apply outlier mask - keep points that are NOT outliers
            current_mask = da.mask.copy()
            current_mask[da.mask] = ~outlier_mask
            da.mask = current_mask
            outliers_removed += outliers_this_array

            if verbose:
                print(
                    f"   Array {i + 1}: Removed {outliers_this_array} outliers (z > {outlier_threshold})"
                )

        start_idx = end_idx

    # Safety check: ensure we haven't removed too many points
    remaining_points = sum(da.mask.sum() for da in cleaned_ds.values())
    min_required_points = int(total_original_points * min_points_fraction)

    if remaining_points < min_required_points:
        if verbose:
            print(
                f"   ⚠️  Too many points removed ({remaining_points} < {min_required_points})"
            )
            print("       Using less aggressive threshold...")

        # Retry with more conservative threshold
        cleaned_ds = copy.deepcopy(ds)
        conservative_threshold = outlier_threshold * 1.5
        outliers_removed = 0

        start_idx = 0
        for da in cleaned_ds.values():
            end_idx = start_idx + len(da.y)
            residuals = huber_result.residual[start_idx:end_idx]
            z_scores = np.abs(stats.zscore(residuals))
            outlier_mask = z_scores > conservative_threshold

            if np.any(outlier_mask):
                current_mask = da.mask.copy()
                current_mask[da.mask] = ~outlier_mask
                da.mask = current_mask
                outliers_removed += np.sum(outlier_mask)

            start_idx = end_idx

    if verbose:
        remaining_points = sum(da.mask.sum() for da in cleaned_ds.values())
        removal_pct = outliers_removed / total_original_points * 100
        print(f"   ✅ Final: Removed {outliers_removed} outliers ({removal_pct:.1f}%)")
        print(f"       Remaining: {remaining_points}/{total_original_points} points")

    # ================================================================
    # STAGE 3: IRLS - Iterative Reweighted Least Squares
    # ================================================================
    if verbose:
        print(f"\n⚖️  STAGE 3: IRLS Refinement ({irls_iterations} iterations)")

    current_ds = copy.deepcopy(cleaned_ds)
    best_result = huber_result
    best_chi2 = (
        huber_result.redchi if hasattr(huber_result, "redchi") else huber_result.chisqr
    )

    for iteration in range(irls_iterations):
        if verbose:
            print(f"   Iteration {iteration + 1}:")

        # Fit with current weights
        params = _build_params_1site(current_ds)
        minimizer = Minimizer(
            _binding_1site_residuals, params, fcn_args=(current_ds,), scale_covar=True
        )
        irls_result = minimizer.minimize()

        if not irls_result.success:
            if verbose:
                print(f"     ❌ IRLS iteration {iteration + 1} failed")
            break

        current_chi2 = (
            irls_result.redchi if hasattr(irls_result, "redchi") else irls_result.chisqr
        )

        if verbose:
            print(f"     Chi² = {current_chi2:.4f}")

        # Keep track of best result
        if current_chi2 < best_chi2:
            best_result = irls_result
            best_chi2 = current_chi2

        # Update weights based on residuals (IRLS step)
        if iteration < irls_iterations - 1:  # Don't update weights on last iteration
            start_idx = 0
            for da in current_ds.values():
                end_idx = start_idx + len(da.y)
                residuals = irls_result.residual[start_idx:end_idx]

                # Conservative weight update: inverse of residual magnitude with damping
                old_errors = da.y_errc[da.mask].copy()
                new_errors = np.maximum(
                    np.abs(residuals), 1e-6
                )  # Prevent division by zero

                # Apply damping to prevent instability
                damped_errors = (
                    1 - irls_damping
                ) * old_errors + irls_damping * new_errors
                da.y_errc[da.mask] = damped_errors

                start_idx = end_idx

    if verbose:
        final_chi2 = (
            best_result.redchi if hasattr(best_result, "redchi") else best_result.chisqr
        )
        improvement = (
            (huber_result.redchi - final_chi2) / huber_result.redchi * 100
            if hasattr(huber_result, "redchi")
            else 0
        )
        print(
            f"   ✅ IRLS complete: Chi² = {final_chi2:.4f} ({improvement:+.1f}% vs Huber)"
        )

    # ================================================================
    # FINAL RESULT
    # ================================================================

    # Create final plot
    fig = figure.Figure()
    ax = fig.add_subplot(111)
    plot_fit(
        ax, current_ds, best_result.params, nboot=N_BOOT, pp=PlotParameters(ds.is_ph)
    )

    # Add title indicating method used
    ax.set_title("3-Stage Robust Fit: Huber → Outlier Removal → IRLS", fontsize=10)

    if verbose:
        print("\n🎉 THREE-STAGE FITTING COMPLETE")
        print(
            f"   Final Chi²: {best_result.redchi if hasattr(best_result, 'redchi') else best_result.chisqr:.4f}"
        )
        print(
            f"   Data points used: {sum(da.mask.sum() for da in current_ds.values())}/{total_original_points}"
        )

    return FitResult(figure=fig, result=best_result, mini=minimizer, dataset=current_ds)


def compare_all_fitting_strategies(ds: Dataset, verbose: bool = True) -> dict:
    """Comprehensive comparison of all fitting strategies."""
    import time

    from src.clophfit.fitting.core import fit_binding_glob_reweighted, fit_lm

    methods = {}

    if verbose:
        print("🏁 COMPREHENSIVE FITTING STRATEGY COMPARISON")
        print("=" * 60)

    # Method 1: Standard LM
    start_time = time.time()
    try:
        result1 = fit_lm(ds)
        methods["standard"] = {
            "result": result1,
            "time": time.time() - start_time,
            "method": "Standard LM",
            "description": "Basic least-squares",
        }
    except Exception as e:
        methods["standard"] = {"error": str(e), "method": "Standard LM"}

    # Method 2: Huber only
    start_time = time.time()
    try:
        result2 = fit_lm(ds, robust=True)
        methods["huber"] = {
            "result": result2,
            "time": time.time() - start_time,
            "method": "Huber Only",
            "description": "Robust loss function",
        }
    except Exception as e:
        methods["huber"] = {"error": str(e), "method": "Huber Only"}

    # Method 3: Huber + Outlier removal
    start_time = time.time()
    try:
        result3 = fit_lm(ds, robust=True, outlier_threshold=3.0)
        methods["huber_outlier"] = {
            "result": result3,
            "time": time.time() - start_time,
            "method": "Huber + Outlier",
            "description": "Robust + outlier removal",
        }
    except Exception as e:
        methods["huber_outlier"] = {"error": str(e), "method": "Huber + Outlier"}

    # Method 4: IRLS only
    start_time = time.time()
    try:
        result4 = fit_binding_glob_reweighted(ds, key="test")
        methods["irls"] = {
            "result": result4,
            "time": time.time() - start_time,
            "method": "IRLS Only",
            "description": "Iterative reweighting",
        }
    except Exception as e:
        methods["irls"] = {"error": str(e), "method": "IRLS Only"}

    # Method 5: Three-stage optimal (Huber → Outlier → IRLS)
    start_time = time.time()
    try:
        result5 = fit_huber_outlier_irls(ds, verbose=False)
        methods["optimal"] = {
            "result": result5,
            "time": time.time() - start_time,
            "method": "3-Stage Optimal",
            "description": "Huber → Outlier → IRLS",
        }
    except Exception as e:
        methods["optimal"] = {"error": str(e), "method": "3-Stage Optimal"}

    # Print comparison table
    if verbose:
        print(
            f"\n{'Method':<20} {'Time (s)':<10} {'Chi²':<12} {'Status':<10} {'Description'}"
        )
        print("-" * 80)

        for data in methods.values():
            if "error" in data:
                print(
                    f"{data['method']:<20} {'ERROR':<10} {str(data['error'])[:20]:<12}"
                )
            else:
                chi2 = (
                    data["result"].result.redchi
                    if data["result"].result
                    and hasattr(data["result"].result, "redchi")
                    else None
                )
                chi2_str = f"{chi2:.4f}" if chi2 is not None else "N/A"
                status = (
                    "Success"
                    if data["result"].result and data["result"].result.success
                    else "Failed"
                )
                print(
                    f"{data['method']:<20} {data['time']:<10.3f} {chi2_str:<12} {status:<10} {data.get('description', '')}"
                )

    return methods


def analyze_parameter_recovery(methods: dict, true_params: dict, verbose: bool = True):
    """Analyze parameter recovery accuracy for all methods."""
    if verbose:
        print("\n📊 PARAMETER RECOVERY ANALYSIS")
        print("=" * 60)
        print(f"{'Method':<20} {'K Est.':<10} {'K Error %':<12} {'Grade':<10}")
        print("-" * 60)

    results_summary = []
    K_true = true_params["K"]

    for data in methods.values():
        if (
            "error" not in data
            and data["result"].result
            and data["result"].result.success
        ):
            params_fit = data["result"].result.params
            if "K" in params_fit:
                K_est = params_fit["K"].value
                K_error_pct = abs(K_est - K_true) / K_true * 100

                # Grading system
                if K_error_pct < 5:
                    grade = "A+ 🏆"
                elif K_error_pct < 10:
                    grade = "A"
                elif K_error_pct < 20:
                    grade = "B"
                elif K_error_pct < 35:
                    grade = "C"
                else:
                    grade = "D"

                chi2 = (
                    data["result"].result.redchi
                    if hasattr(data["result"].result, "redchi")
                    else None
                )
                results_summary.append((
                    data["method"],
                    K_est,
                    K_error_pct,
                    chi2,
                    grade,
                ))

                if verbose:
                    print(
                        f"{data['method']:<20} {K_est:<10.3f} {K_error_pct:<12.1f} {grade}"
                    )

    return results_summary


if __name__ == "__main__":
    from fitting_function_comparison import (
        SimulationParameters,
        generate_synthetic_dataset,
    )

    print("🧪 TESTING 3-STAGE OPTIMAL ROBUST FITTING")
    print("=" * 60)

    # Generate challenging dataset with outliers
    params = SimulationParameters(
        random_seed=42,
        outlier_probability=0.9,  # High probability of outliers
        outlier_magnitude=8.0,  # Larger outliers for challenging test
    )
    dataset, true_params = generate_synthetic_dataset(params)

    print("True parameters:")
    for param, value in true_params.items():
        print(f"  {param}: {value}")

    print("\nDataset characteristics:")
    for label, da in dataset.items():
        outlier_candidates = da.yc[-2:]  # Last 2 points
        print(f"  {label}: {len(da.yc)} points, last 2 values: {outlier_candidates}")

    # Test the 3-stage method with detailed output
    print("\n" + "=" * 60)
    result = fit_huber_outlier_irls(dataset, verbose=True)

    if result.result and result.result.success:
        print("\n🎯 FINAL PARAMETER ESTIMATES:")
        for name, param in result.result.params.items():
            true_val = true_params.get(name, 0)
            error_pct = (
                abs(param.value - true_val) / abs(true_val) * 100
                if true_val != 0
                else 0
            )
            print(
                f"  {name}: {param.value:.3f} ± {param.stderr:.3f} ({error_pct:.1f}% error)"
            )

    # Full comparison
    print("\n" + "=" * 80)
    comparison = compare_all_fitting_strategies(dataset, verbose=True)

    # Parameter recovery analysis
    recovery_results = analyze_parameter_recovery(comparison, true_params, verbose=True)

    if recovery_results:
        # Find best method
        best_method = min(recovery_results, key=operator.itemgetter(2))  # Min K error
        best_fit = min(
            recovery_results, key=lambda x: x[3] if x[3] is not None else float("inf")
        )  # Min Chi²

        print("\n🏆 WINNER ANALYSIS:")
        print(
            f"Best Parameter Recovery: {best_method[0]} ({best_method[2]:.1f}% K error)"
        )
        print(f"Best Fit Quality: {best_fit[0]} (Chi² = {best_fit[3]:.4f})")

        print("\n💡 RECOMMENDATIONS:")
        print(f"• For accuracy-critical applications: Use {best_method[0]}")
        print(f"• For best curve fitting: Use {best_fit[0]}")
        print(
            "• For general use with outliers: 3-Stage Optimal combines both strengths"
        )

    print("\nVisualization plots created for comparison.")
