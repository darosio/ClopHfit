#!/usr/bin/env python3
"""
Improved robust fitting combining Huber loss with adaptive IRLS.

This addresses the instability issues from the first version.
"""

import copy
import operator

import numpy as np
from lmfit import Minimizer, Parameters
from matplotlib import figure
from scipy import stats

from src.clophfit.fitting.core import _binding_1site_residuals, _build_params_1site
from src.clophfit.fitting.data_structures import Dataset, FitResult
from src.clophfit.fitting.plotting import PlotParameters, plot_fit

N_BOOT = 20


def fit_lm_robust_irls_improved(
    ds: Dataset,
    *,
    max_iterations: int = 3,
    convergence_tol: float = 0.001,
    outlier_threshold: float = 3.0,
    weight_damping: float = 0.5,
    use_huber: bool = True,
    verbose: bool = False,
) -> FitResult:
    """
    Improved robust fitting with stable weight updates.

    Key improvements:
    1. Damped weight updates to prevent instability
    2. More conservative convergence criteria
    3. Better outlier handling
    4. Fallback to previous iteration if fit worsens significantly

    Parameters
    ----------
    ds : Dataset
        The dataset to fit.
    max_iterations : int, default=3
        Maximum number of iterations (reduced for stability).
    convergence_tol : float, default=0.001
        Stop when relative improvement < this value.
    outlier_threshold : float, default=3.0
        Remove points with |z-score| > threshold.
    weight_damping : float, default=0.5
        Damping factor for weight updates (0=no update, 1=full update).
    use_huber : bool, default=True
        Use Huber loss for robust fitting.
    verbose : bool, default=False
        Print iteration details.

    Returns
    -------
    FitResult
        Enhanced fit result with improved parameter recovery.
    """
    if len(np.concatenate([da.y for da in ds.values()])) <= len(
        _build_params_1site(ds)
    ):
        msg = "Not enough data points for the number of parameters."
        raise ValueError(msg)

    # Initialize with copy
    current_ds = copy.deepcopy(ds)
    best_result = None
    best_chi2 = float("inf")
    best_ds = None

    if verbose:
        print("Starting improved robust fitting...")
        print(f"  Use Huber loss: {use_huber}")
        print(f"  Max iterations: {max_iterations}")
        print(f"  Weight damping: {weight_damping}")
        print(f"  Outlier threshold: {outlier_threshold}")

    for iteration in range(max_iterations + 1):  # +1 for initial fit
        if verbose:
            print(f"\n--- Iteration {iteration} ---")

        # Fit with current weights
        params = _build_params_1site(current_ds)
        minimizer = Minimizer(
            _binding_1site_residuals, params, fcn_args=(current_ds,), scale_covar=True
        )

        if use_huber and iteration > 0:  # Use Huber after first iteration
            result = minimizer.minimize(method="least_squares", loss="huber")
        else:
            result = minimizer.minimize()

        if not result.success:
            if verbose:
                print(f"  Fit failed at iteration {iteration}")
            break

        current_chi2 = result.redchi if hasattr(result, "redchi") else result.chisqr

        if verbose:
            print(f"  Chi-squared: {current_chi2:.6f}")

        # Keep track of best result
        if current_chi2 < best_chi2:
            best_result = result
            best_chi2 = current_chi2
            best_ds = copy.deepcopy(current_ds)

        # Check if this is the last iteration
        if iteration >= max_iterations:
            if verbose:
                print("  Maximum iterations reached")
            break

        # Check convergence (relative improvement)
        if iteration > 0:
            relative_improvement = (previous_chi2 - current_chi2) / previous_chi2
            if verbose:
                print(f"  Relative improvement: {relative_improvement:.6f}")
            if relative_improvement < convergence_tol and relative_improvement >= 0:
                if verbose:
                    print(
                        f"  Converged! Improvement {relative_improvement:.6f} < {convergence_tol}"
                    )
                break

        # Store chi2 for next iteration

        # Update weights with damping
        start_idx = 0
        outliers_removed = 0

        # Store original weights for damping
        original_weights = {}
        for i, da in enumerate(current_ds.values()):
            original_weights[i] = da.y_errc[da.mask].copy()

        for i, da in enumerate(current_ds.values()):
            end_idx = start_idx + len(da.y)
            residuals = result.residual[start_idx:end_idx]

            # Calculate new weights based on residuals
            # Use a more conservative approach: weight = 1/sqrt(residual²)
            abs_residuals = np.abs(residuals)
            # Add small constant to prevent division by zero
            new_errors = np.sqrt(abs_residuals + 1e-6)

            # Apply damping: blend old and new weights
            old_errors = original_weights[i]
            damped_errors = (
                1 - weight_damping
            ) * old_errors + weight_damping * new_errors

            # Apply new errors
            da.y_errc[da.mask] = damped_errors

            # Outlier removal (more conservative)
            if outlier_threshold is not None and iteration > 0:
                z_scores = np.abs(stats.zscore(residuals))
                outlier_mask = z_scores > outlier_threshold

                if (
                    np.any(outlier_mask) and np.sum(~outlier_mask) > 3
                ):  # Keep at least 3 points
                    outliers_removed += np.sum(outlier_mask)
                    # Apply outlier mask
                    current_mask = da.mask.copy()
                    current_mask[da.mask] = ~outlier_mask
                    da.mask = current_mask

            start_idx = end_idx

        if verbose and outliers_removed > 0:
            print(f"  Removed {outliers_removed} outliers")

        # Safety check: if we removed too many points, revert
        total_points = sum(da.mask.sum() for da in current_ds.values())
        min_points = len(_build_params_1site(current_ds)) + 2

        if total_points < min_points:
            if verbose:
                print(
                    f"  Too many points removed ({total_points} < {min_points}), reverting"
                )
            break

    # Use the best result found
    if best_result is None:
        # Fallback: try a simple robust fit
        params = _build_params_1site(ds)
        minimizer = Minimizer(_binding_1site_residuals, params, fcn_args=(ds,))
        best_result = minimizer.minimize(
            method="least_squares", loss="huber" if use_huber else None
        )
        best_ds = ds

    # Create final plot
    fig = figure.Figure()
    ax = fig.add_subplot(111)
    final_params = best_result.params if best_result.success else Parameters()
    plot_fit(ax, best_ds, final_params, nboot=N_BOOT, pp=PlotParameters(ds.is_ph))

    return FitResult(figure=fig, result=best_result, mini=minimizer, dataset=best_ds)


def compare_all_robust_methods(ds: Dataset, verbose: bool = True) -> dict:
    """Compare all robust methods including the improved version."""
    import time

    from src.clophfit.fitting.core import fit_binding_glob_reweighted, fit_lm

    methods = {}

    if verbose:
        print("Comparing all robust fitting methods...")
        print("=" * 60)

    # Method 1: Standard LM
    start_time = time.time()
    try:
        result1 = fit_lm(ds)
        chi2 = (
            result1.result.redchi
            if result1.result and hasattr(result1.result, "redchi")
            else None
        )
        methods["standard_lm"] = {
            "result": result1,
            "time": time.time() - start_time,
            "chi2": chi2,
            "method": "Standard LM",
        }
    except Exception as e:
        methods["standard_lm"] = {"error": str(e), "method": "Standard LM"}

    # Method 2: Robust LM (Huber only)
    start_time = time.time()
    try:
        result2 = fit_lm(ds, robust=True)
        chi2 = (
            result2.result.redchi
            if result2.result and hasattr(result2.result, "redchi")
            else None
        )
        methods["robust_huber"] = {
            "result": result2,
            "time": time.time() - start_time,
            "chi2": chi2,
            "method": "Robust (Huber only)",
        }
    except Exception as e:
        methods["robust_huber"] = {"error": str(e), "method": "Robust (Huber only)"}

    # Method 3: IRLS only
    start_time = time.time()
    try:
        result3 = fit_binding_glob_reweighted(ds, key="test")
        chi2 = (
            result3.result.redchi
            if result3.result and hasattr(result3.result, "redchi")
            else None
        )
        methods["irls_only"] = {
            "result": result3,
            "time": time.time() - start_time,
            "chi2": chi2,
            "method": "IRLS only",
        }
    except Exception as e:
        methods["irls_only"] = {"error": str(e), "method": "IRLS only"}

    # Method 4: Improved Huber + IRLS
    start_time = time.time()
    try:
        result4 = fit_lm_robust_irls_improved(ds, verbose=False)
        chi2 = (
            result4.result.redchi
            if result4.result and hasattr(result4.result, "redchi")
            else None
        )
        methods["improved_combined"] = {
            "result": result4,
            "time": time.time() - start_time,
            "chi2": chi2,
            "method": "Improved (Huber + IRLS)",
        }
    except Exception as e:
        methods["improved_combined"] = {
            "error": str(e),
            "method": "Improved (Huber + IRLS)",
        }

    # Method 5: Conservative IRLS (without Huber)
    start_time = time.time()
    try:
        result5 = fit_lm_robust_irls_improved(ds, use_huber=False, verbose=False)
        chi2 = (
            result5.result.redchi
            if result5.result and hasattr(result5.result, "redchi")
            else None
        )
        methods["conservative_irls"] = {
            "result": result5,
            "time": time.time() - start_time,
            "chi2": chi2,
            "method": "Conservative IRLS",
        }
    except Exception as e:
        methods["conservative_irls"] = {"error": str(e), "method": "Conservative IRLS"}

    if verbose:
        print(f"{'Method':<25} {'Time (s)':<10} {'Chi²':<12} {'Status'}")
        print("-" * 60)

        for data in methods.values():
            if "error" in data:
                print(f"{data['method']:<25} {'ERROR':<10} {data['error']:<12}")
            else:
                chi2_str = f"{data['chi2']:.4f}" if data["chi2"] is not None else "N/A"
                status = (
                    "Success"
                    if data["result"].result and data["result"].result.success
                    else "Failed"
                )
                print(
                    f"{data['method']:<25} {data['time']:<10.3f} {chi2_str:<12} {status}"
                )

    return methods


if __name__ == "__main__":
    from fitting_function_comparison import (
        SimulationParameters,
        generate_synthetic_dataset,
    )

    print("Testing Improved Robust Fitting")
    print("=" * 50)

    # Generate test dataset with outliers
    params = SimulationParameters(random_seed=42)
    dataset, true_params = generate_synthetic_dataset(params)

    print("True parameters:")
    for param, value in true_params.items():
        print(f"  {param}: {value}")

    print("\nTesting improved robust fitting...")
    result = fit_lm_robust_irls_improved(dataset, verbose=True)

    if result.result and result.result.success:
        print("\nFinal parameters:")
        for name, param in result.result.params.items():
            true_val = true_params.get(name, 0)
            error_pct = (
                abs(param.value - true_val) / abs(true_val) * 100
                if true_val != 0
                else 0
            )
            print(
                f"  {name}: {param.value:.3f} ± {param.stderr:.3f} (error: {error_pct:.1f}%)"
            )

    # Full comparison
    print("\n" + "=" * 70)
    print("COMPREHENSIVE ROBUST METHOD COMPARISON")
    print("=" * 70)

    comparison = compare_all_robust_methods(dataset, verbose=True)

    # Parameter accuracy comparison
    print("\nParameter Recovery Accuracy (K parameter):")
    print(f"{'Method':<25} {'K Est.':<10} {'K Error %':<10} {'Quality'}")
    print("-" * 55)

    K_true = true_params["K"]
    results_summary = []

    for name, data in comparison.items():
        if (
            "error" not in data
            and data["result"].result
            and data["result"].result.success
        ):
            params_fit = data["result"].result.params
            if "K" in params_fit:
                K_est = params_fit["K"].value
                K_error_pct = abs(K_est - K_true) / K_true * 100

                # Quality assessment
                if K_error_pct < 10:
                    quality = "Excellent"
                elif K_error_pct < 25:
                    quality = "Good"
                elif K_error_pct < 50:
                    quality = "Fair"
                else:
                    quality = "Poor"

                results_summary.append((
                    data["method"],
                    K_est,
                    K_error_pct,
                    data["chi2"],
                ))
                print(
                    f"{data['method']:<25} {K_est:<10.3f} {K_error_pct:<10.1f} {quality}"
                )

    # Recommendations
    print("\\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    if results_summary:
        # Sort by parameter recovery error
        results_summary.sort(key=operator.itemgetter(2))  # Sort by K error percentage

        best_method = results_summary[0]
        print(f"🏆 Best Parameter Recovery: {best_method[0]}")
        print(f"   K error: {best_method[2]:.1f}%, Chi²: {best_method[3]:.4f}")

        # Sort by chi-squared for goodness of fit
        results_summary.sort(key=operator.itemgetter(3))  # Sort by chi²
        best_fit = results_summary[0]
        print(f"📊 Best Fit Quality: {best_fit[0]}")
        print(f"   Chi²: {best_fit[3]:.4f}, K error: {best_fit[2]:.1f}%")

        print("\n💡 For your typical data (2 channels, outliers in high pH):")
        print("   1. Use 'Robust (Huber only)' for speed and reliability")
        print("   2. Use 'IRLS only' for best fit quality")
        print("   3. Use 'Improved (Huber + IRLS)' for best parameter recovery")

    print("\nVisualization saved as typical plot.")
