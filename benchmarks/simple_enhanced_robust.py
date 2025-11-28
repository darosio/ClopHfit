#!/usr/bin/env python3
"""
Simplified enhanced robust fitting method focusing on core functionality.

This version combines robust fitting with iterative outlier removal in a simpler,
more stable approach.
"""

from __future__ import annotations

import copy
import time
from typing import TYPE_CHECKING

import numpy as np
from lmfit import Minimizer  # type: ignore[import-untyped]
from matplotlib import figure
from scipy import stats

from src.clophfit.fitting.core import (
    _binding_1site_residuals,
    _build_params_1site,
    fit_lm,
    outlier2,
)
from src.clophfit.fitting.data_structures import Dataset, FitResult
from src.clophfit.fitting.plotting import PlotParameters, plot_fit

if TYPE_CHECKING:
    from collections.abc import Callable

    from lmfit.minimizer import MinimizerResult  # type: ignore[import-untyped]

N_BOOT = 20
MIN_POINTS_FOR_ZSCORE = 3
MIN_POINTS_AFTER_REMOVAL = 2


def fit_lm_robust_simple(  # noqa: C901, PLR0912
    ds: Dataset,
    *,
    use_huber: bool = True,
    outlier_removal_rounds: int = 2,
    outlier_threshold: float = 2.5,
    verbose: bool = False,
) -> FitResult:
    """
    Simplified robust fitting with iterative outlier removal.

    This method:
    1. Fits with Huber loss (optional) for initial robustness
    2. Iteratively removes outliers and refits
    3. Returns the best result

    Parameters
    ----------
    ds : Dataset
        Dataset to fit.
    use_huber : bool, optional
        Use Huber loss for robust fitting.
    outlier_removal_rounds : int, optional
        Number of rounds of outlier removal.
    outlier_threshold : float, optional
        Remove points with |z-score| > threshold.
    verbose : bool, optional
        Print iteration details.

    Returns
    -------
    FitResult
        Enhanced fit result with improved parameter recovery.

    Raises
    ------
    ValueError
        If there are not enough data points to support the fit parameters.
    """
    if len(np.concatenate([da.y for da in ds.values()])) <= len(
        _build_params_1site(ds)
    ):
        msg = "Not enough data points for the number of parameters."
        raise ValueError(msg)

    # Initialize with copy to avoid modifying original
    current_ds = copy.deepcopy(ds)
    best_result = None
    best_chi2 = float("inf")
    best_ds = None

    if verbose:
        print("Starting simplified robust fitting...")
        print(f"  Use Huber loss: {use_huber}")
        print(f"  Outlier removal rounds: {outlier_removal_rounds}")
        print(f"  Outlier threshold: {outlier_threshold}")

    for round_num in range(outlier_removal_rounds + 1):  # +1 for initial fit
        if verbose:
            print(f"\\n--- Round {round_num + 1} ---")

        # Fit with current dataset
        params = _build_params_1site(current_ds)
        minimizer = Minimizer(
            _binding_1site_residuals, params, fcn_args=(current_ds,), scale_covar=True
        )

        if use_huber and round_num < outlier_removal_rounds:
            # Use Huber loss for initial fits
            result = minimizer.minimize(method="least_squares", loss="huber")
        else:
            # Use standard fit for final round
            result = minimizer.minimize()

        if not result.success:
            if verbose:
                print(f"  Fit failed at round {round_num + 1}")
            continue

        current_chi2 = result.redchi if hasattr(result, "redchi") else result.chisqr

        if verbose:
            print(f"  Chi-squared: {current_chi2:.6f}")
            print(f"  Success: {result.success}")

        # Keep track of best result
        if current_chi2 < best_chi2:
            best_chi2 = current_chi2
            best_result = result
            best_ds = copy.deepcopy(current_ds)

        # Skip outlier removal on last round
        if round_num >= outlier_removal_rounds:
            break

        # Remove outliers for next round
        outliers_removed = remove_outliers(
            current_ds, result, outlier_threshold, verbose
        )

        if outliers_removed == 0:
            if verbose:
                print("  No more outliers to remove, stopping early")
            break

    # Use best result
    if best_result is None:
        # Fallback: try without Huber loss
        params = _build_params_1site(ds)
        minimizer = Minimizer(
            _binding_1site_residuals, params, fcn_args=(ds,), scale_covar=True
        )
        best_result = minimizer.minimize()
        best_ds = ds

    # Final plot
    fig = figure.Figure()
    ax = fig.add_subplot(111)
    plot_fit(ax, best_ds, best_result.params, nboot=N_BOOT, pp=PlotParameters(ds.is_ph))

    return FitResult(figure=fig, result=best_result, mini=minimizer, dataset=best_ds)


def remove_outliers(
    ds: Dataset,
    result: MinimizerResult,
    threshold: float,
    *,
    verbose: bool = False,
) -> int:
    """
    Remove outliers from dataset based on fit residuals.

    Parameters
    ----------
    ds : Dataset
        Dataset to modify in-place
    result : MinimizerResult
        Fit result with residuals.
    threshold : float
        Z-score threshold for outlier removal
    verbose : bool
        Print details

    Returns
    -------
    int
        Number of outliers removed
    """
    total_removed = 0
    start_idx = 0

    for label, da in ds.items():
        end_idx = start_idx + np.sum(da.mask)  # Only count unmasked points
        residuals = result.residual[start_idx:end_idx]

        if len(residuals) > MIN_POINTS_FOR_ZSCORE:
            z_scores = np.abs(stats.zscore(residuals))
            outlier_indices = np.where(z_scores > threshold)[0]

            if (
                len(outlier_indices) > 0
                and len(residuals) - len(outlier_indices) >= MIN_POINTS_AFTER_REMOVAL
            ):
                # Apply outlier mask (convert from residual indices to original indices)
                current_mask_indices = np.where(da.mask)[0]
                outlier_original_indices = current_mask_indices[outlier_indices]

                for idx in outlier_original_indices:
                    da.mask[idx] = False
                    total_removed += 1

                if verbose:
                    print(f"    Removed {len(outlier_indices)} outliers from {label}")

        start_idx = end_idx

    return total_removed


def compare_simple_robust_methods(
    ds: Dataset,
    true_params: dict | None = None,
    *,
    verbose: bool = True,
) -> dict:
    """Compare simple robust fitting approaches."""
    methods: dict[str, Callable[[], FitResult]] = {
        "Standard LM": lambda: fit_lm(ds),
        "Robust Huber": lambda: fit_lm(ds, robust=True),
        "Outlier2": lambda: outlier2(ds, key="test"),
        "Simple Enhanced": lambda: fit_lm_robust_simple(ds, verbose=False),
    }

    results = {}

    if verbose:
        print("Comparing simple robust methods...")
        print("=" * 50)

    for method_name, method_func in methods.items():
        start_time = time.time()
        try:
            result = method_func()
            exec_time = time.time() - start_time

            if result.result and result.result.success and "K" in result.result.params:
                k_est = result.result.params["K"].value
                chi2 = (
                    result.result.redchi
                    if hasattr(result.result, "redchi")
                    else result.result.chisqr
                )

                method_result = {
                    "success": True,
                    "K_est": k_est,
                    "chi2": chi2,
                    "time": exec_time,
                }

                if true_params and "K" in true_params:
                    k_true = true_params["K"]
                    method_result["K_error"] = abs(k_est - k_true) / k_true * 100

            else:
                method_result = {"success": False, "time": exec_time}

        except Exception:  # noqa: BLE001
            method_result = {"success": False, "error": "Failed", "time": 0.0}

        results[method_name] = method_result

    if verbose:
        print(f"{'Method':<20} {'Success':<8} {'K Error %':<10} {'Time (s)':<10}")
        print("-" * 50)

        for method_name, result in results.items():
            if result["success"]:
                error_str = (
                    f"{result.get('K_error', 'N/A'):6.1f}"
                    if "K_error" in result
                    else "N/A"
                )
                print(
                    f"{method_name:<20} {'✓':<8} {error_str:<10} {result['time']:8.3f}"
                )
            else:
                print(f"{method_name:<20} {'✗':<8} {'N/A':<10} {result['time']:8.3f}")

    return results


if __name__ == "__main__":
    # Test with realistic synthetic data
    from realistic_synthetic_data import (
        RealisticSimulationParameters,
        generate_realistic_dataset,
    )

    print("🧪 TESTING SIMPLE ENHANCED ROBUST FITTING")
    print("=" * 50)

    # Test scenarios
    test_scenarios = [
        {
            "name": "Clean Data",
            "params": RealisticSimulationParameters(
                random_seed=42,
                outlier_probability=0.05,
                # CORRECTED: y1 errors 10x larger than y2
                y1_base_error=100.0,  # 10x larger
                y2_base_error=10.0,  # Reference
            ),
        },
        {
            "name": "Moderate Outliers",
            "params": RealisticSimulationParameters(
                random_seed=123,
                outlier_probability=0.15,
                outlier_magnitude=3.0,
            ),
        },
        {
            "name": "High Outliers",
            "params": RealisticSimulationParameters(
                random_seed=456,
                outlier_probability=0.25,
                outlier_magnitude=4.0,
            ),
        },
        {
            "name": "High Noise",
            "params": RealisticSimulationParameters(
                random_seed=789,
                outlier_probability=0.1,
                # CORRECTED: y1 errors 10x larger than y2
                y1_base_error=200.0,  # High noise, 10x larger
                y2_base_error=20.0,  # High noise reference
            ),
        },
    ]

    for scenario in test_scenarios:
        print(f"\\n🎯 Testing scenario: {scenario['name']}")
        print("-" * 40)

        dataset, true_params = generate_realistic_dataset(scenario["params"])
        results = compare_simple_robust_methods(dataset, true_params, verbose=True)

        # Find best method for this scenario
        successful_methods = {
            name: result for name, result in results.items() if result["success"]
        }
        if successful_methods:
            best_method = min(
                successful_methods.items(),
                key=lambda x: x[1].get("K_error", float("inf")),
            )
            print(
                f"\\n🏆 Best method: {best_method[0]} ({best_method[1]['K_error']:.1f}% error)"
            )

    print("\\n✨ Simple enhanced robust fitting test complete!")
