#!/usr/bin/env python3
"""
Enhanced robust fitting combining Huber loss with Iteratively Reweighted Least Squares.

This combines the best aspects of:
- fit_lm(robust=True): Huber loss for outlier resistance
- core_reweighted: IRLS for adaptive weighting
"""

import copy

import numpy as np
from lmfit import Minimizer, Parameters
from matplotlib import figure
from scipy import stats

from src.clophfit.fitting.core import (
    _binding_1site_residuals,
    _build_params_1site,
)
from src.clophfit.fitting.data_structures import Dataset, FitResult
from src.clophfit.fitting.plotting import PlotParameters, plot_fit

N_BOOT = 20  # From the original code


def fit_lm_robust_irls(
    ds: Dataset,
    *,
    max_iterations: int = 5,
    convergence_tol: float = 0.01,
    outlier_threshold: float = 3.0,
    min_weight: float = 1e-3,
    use_huber: bool = True,
    verbose: bool = False,
) -> FitResult:
    """
    Enhanced robust fitting combining Huber loss with IRLS.

    This method iteratively:
    1. Fits with Huber loss (if enabled) for robustness
    2. Updates weights based on residuals (IRLS)
    3. Removes extreme outliers
    4. Converges when improvement is below tolerance

    Parameters
    ----------
    ds : Dataset
        The dataset to fit.
    max_iterations : int, default=5
        Maximum number of IRLS iterations.
    convergence_tol : float, default=0.01
        Stop when improvement in chi-squared < this value.
    outlier_threshold : float, default=3.0
        Remove points with |z-score| > threshold.
    min_weight : float, default=1e-3
        Minimum weight to prevent division by zero.
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

    # Initialize with copy to avoid modifying original
    current_ds = copy.deepcopy(ds)

    # Track convergence
    previous_chi2 = float("inf")
    iteration_stats = []

    if verbose:
        print("Starting enhanced robust fitting...")
        print(f"  Use Huber loss: {use_huber}")
        print(f"  Max iterations: {max_iterations}")
        print(f"  Convergence tolerance: {convergence_tol}")
        print(f"  Outlier threshold: {outlier_threshold}")

    for iteration in range(max_iterations):
        if verbose:
            print(f"\n--- Iteration {iteration + 1} ---")

        # Step 1: Robust fit with current weights
        params = _build_params_1site(current_ds)
        minimizer = Minimizer(
            _binding_1site_residuals, params, fcn_args=(current_ds,), scale_covar=True
        )

        if use_huber:
            result = minimizer.minimize(method="least_squares", loss="huber")
        else:
            result = minimizer.minimize()

        if not result.success:
            if verbose:
                print(f"  Fit failed at iteration {iteration + 1}")
            break

        current_chi2 = result.redchi if hasattr(result, "redchi") else result.chisqr

        if verbose:
            print(f"  Chi-squared: {current_chi2:.6f}")
            print(f"  Success: {result.success}")

        # Step 2: Check convergence
        improvement = previous_chi2 - current_chi2
        if iteration > 0 and improvement < convergence_tol:
            if verbose:
                print(f"  Converged! Improvement {improvement:.6f} < {convergence_tol}")
            break

        # Step 3: Update weights based on residuals (IRLS)
        start_idx = 0
        outlier_count = 0

        for da in current_ds.values():
            end_idx = start_idx + len(da.y)
            residuals = result.residual[start_idx:end_idx]

            # Store original errors before modification
            original_errors = da.y_errc[da.mask].copy()

            # IRLS weight update: more conservative approach
            # Weight = 1 / (1 + |residual|/sigma)
            sigma_est = np.std(residuals)
            if sigma_est > 0:
                new_weights = 1.0 / (1.0 + np.abs(residuals) / sigma_est)
                # Scale by original errors to preserve error structure
                da.y_errc[da.mask] = original_errors / np.maximum(
                    new_weights, min_weight
                )

                # Keep weights reasonable (avoid extreme values)
                da.y_errc[da.mask] = np.clip(
                    da.y_errc[da.mask],
                    original_errors * 0.1,  # min 10% of original
                    original_errors * 10.0,
                )  # max 10x original

            # Step 4: Outlier detection and removal
            if outlier_threshold is not None:
                z_scores = np.abs(stats.zscore(residuals))
                outlier_mask = z_scores > outlier_threshold
                outlier_count += np.sum(outlier_mask)

                if np.any(outlier_mask):
                    # Apply outlier mask to dataset
                    current_mask = da.mask.copy()
                    current_mask[da.mask] = ~outlier_mask
                    da.mask = current_mask

            start_idx = end_idx

        if verbose and outlier_count > 0:
            print(f"  Removed {outlier_count} outliers")

        # Store iteration stats
        iteration_stats.append({
            "iteration": iteration + 1,
            "chi2": current_chi2,
            "improvement": improvement,
            "outliers_removed": outlier_count,
            "n_parameters": len(params),
            "n_data_points": len(result.residual),
        })

        previous_chi2 = current_chi2

    # Final plot
    fig = figure.Figure()
    ax = fig.add_subplot(111)
    final_params = result.params if result.success else Parameters()
    plot_fit(ax, current_ds, final_params, nboot=N_BOOT, pp=PlotParameters(ds.is_ph))

    # Store iteration info in result for analysis
    if hasattr(result, "userinfo"):
        result.userinfo = iteration_stats

    return FitResult(figure=fig, result=result, mini=minimizer, dataset=current_ds)


def compare_robust_methods(ds: Dataset, verbose: bool = True) -> dict:
    """
    Compare different robust fitting approaches on the same dataset.

    Returns
    -------
    dict
        Comparison results with timing and parameter recovery metrics.
    """
    import time

    from src.clophfit.fitting.core import fit_binding_glob_reweighted, fit_lm

    methods = {}

    if verbose:
        print("Comparing robust fitting methods...")
        print("=" * 50)

    # Method 1: Standard LM
    start_time = time.time()
    try:
        result1 = fit_lm(ds)
        methods["standard_lm"] = {
            "result": result1,
            "time": time.time() - start_time,
            "success": result1.result.success if result1.result else False,
            "chi2": result1.result.redchi
            if result1.result and hasattr(result1.result, "redchi")
            else None,
            "method": "Standard LM",
        }
    except Exception as e:
        methods["standard_lm"] = {"error": str(e), "method": "Standard LM"}

    # Method 2: Robust LM (Huber)
    start_time = time.time()
    try:
        result2 = fit_lm(ds, robust=True)
        methods["robust_lm"] = {
            "result": result2,
            "time": time.time() - start_time,
            "success": result2.result.success if result2.result else False,
            "chi2": result2.result.redchi
            if result2.result and hasattr(result2.result, "redchi")
            else None,
            "method": "Robust LM (Huber)",
        }
    except Exception as e:
        methods["robust_lm"] = {"error": str(e), "method": "Robust LM (Huber)"}

    # Method 3: Core Reweighted (IRLS)
    start_time = time.time()
    try:
        result3 = fit_binding_glob_reweighted(ds, key="test")
        methods["irls"] = {
            "result": result3,
            "time": time.time() - start_time,
            "success": result3.result.success if result3.result else False,
            "chi2": result3.result.redchi
            if result3.result and hasattr(result3.result, "redchi")
            else None,
            "method": "IRLS Reweighted",
        }
    except Exception as e:
        methods["irls"] = {"error": str(e), "method": "IRLS Reweighted"}

    # Method 4: Enhanced (Huber + IRLS)
    start_time = time.time()
    try:
        result4 = fit_lm_robust_irls(ds, verbose=False)
        methods["enhanced"] = {
            "result": result4,
            "time": time.time() - start_time,
            "success": result4.result.success if result4.result else False,
            "chi2": result4.result.redchi
            if result4.result and hasattr(result4.result, "redchi")
            else None,
            "method": "Enhanced (Huber + IRLS)",
        }
    except Exception as e:
        methods["enhanced"] = {"error": str(e), "method": "Enhanced (Huber + IRLS)"}

    if verbose:
        print(f"{'Method':<25} {'Time (s)':<10} {'Chi²':<12} {'Success':<8}")
        print("-" * 60)

        for data in methods.values():
            if "error" in data:
                print(f"{data['method']:<25} {'ERROR':<10} {data['error']:<12}")
            else:
                chi2_str = f"{data['chi2']:.4f}" if data["chi2"] is not None else "N/A"
                print(
                    f"{data['method']:<25} {data['time']:<10.3f} {chi2_str:<12} {data['success']}"
                )

    return methods


if __name__ == "__main__":
    # Test with the simulated dataset
    from fitting_function_comparison import (
        SimulationParameters,
        generate_synthetic_dataset,
    )

    print("Testing Enhanced Robust Fitting")
    print("=" * 40)

    # Generate test dataset
    params = SimulationParameters(random_seed=42)
    dataset, true_params = generate_synthetic_dataset(params)

    print("True parameters:")
    for param, value in true_params.items():
        print(f"  {param}: {value}")

    # Test the enhanced method
    print("\nTesting enhanced robust fitting...")
    result = fit_lm_robust_irls(dataset, verbose=True)

    if result.result and result.result.success:
        print("\nFinal parameters:")
        for name, param in result.result.params.items():
            true_val = true_params.get(name, "N/A")
            error = (
                abs(param.value - true_val) / abs(true_val) * 100
                if true_val != "N/A"
                else "N/A"
            )
            error_str = f"{error:.1f}%" if error != "N/A" else "N/A"
            print(
                f"  {name}: {param.value:.3f} ± {param.stderr:.3f} (true: {true_val}, error: {error_str})"
            )

    # Compare all methods
    print("\n" + "=" * 60)
    print("COMPARISON OF ALL ROBUST METHODS")
    print("=" * 60)

    comparison = compare_robust_methods(dataset, verbose=True)

    # Detailed parameter comparison for successful fits
    print("\nDetailed Parameter Comparison:")
    print(f"{'Method':<25} {'K estimate':<12} {'K error %':<10}")
    print("-" * 50)

    for name, data in comparison.items():
        if "error" not in data and data["success"] and data["result"].result:
            params_fit = data["result"].result.params
            if "K" in params_fit:
                K_est = params_fit["K"].value
                K_true = true_params["K"]
                K_error_pct = abs(K_est - K_true) / K_true * 100
                print(f"{data['method']:<25} {K_est:<12.3f} {K_error_pct:<10.1f}")
