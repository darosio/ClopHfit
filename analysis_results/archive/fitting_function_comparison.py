#!/usr/bin/env python3
"""
Comprehensive fitting function comparison study.

This script generates synthetic datasets matching typical experimental conditions
and compares all available fitting functions to identify:
1. Which functions are duplicates/equivalent
2. Which perform best under different conditions
3. Recommendations for function consolidation and best practices
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

# Import all fitting functions for comparison
from src.clophfit.fitting.api import (
    FitMethod,
    fit_binding,
    fit_binding_bayes,
    fit_binding_bayes_perlabel,
    fit_binding_glob_recursive,
    fit_binding_glob_recursive_outlier,
    fit_binding_glob_reweighted,
    fit_binding_lm,
    fit_binding_lm_outlier,
)
from src.clophfit.fitting.bayes import (
    fit_binding_pymc,
    fit_binding_pymc2,
)
from src.clophfit.fitting.core import (
    fit_binding_glob,
    fit_binding_glob_recursive as core_recursive,
    fit_binding_glob_recursive_outlier as core_recursive_outlier,
    fit_binding_glob_reweighted as core_reweighted,
    fit_lm,
    outlier2,
)
from src.clophfit.fitting.data_structures import DataArray, Dataset, FitResult
from src.clophfit.fitting.models import binding_1site
from src.clophfit.fitting.odr import (
    fit_binding_odr,
    fit_binding_odr_recursive,
    fit_binding_odr_recursive_outlier,
)

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class SimulationParameters:
    """Parameters for synthetic data generation."""

    # True parameters for data generation
    K_true: float = 7.5  # pKd
    S0_y1_true: float = 10000.0  # High signal channel
    S1_y1_true: float = 50000.0
    S0_y2_true: float = 1000.0  # Low signal channel (10x smaller)
    S1_y2_true: float = 5000.0

    # Data generation settings
    pH_range: tuple[float, float] = (5.0, 10.0)
    n_points: int = 12
    base_error_y1: float = 500.0  # Base error for high signal
    base_error_y2: float = (
        200.0  # Base error for low signal (but relatively 10x larger)
    )
    outlier_probability: float = 0.8  # Probability of outliers in last 2 points
    outlier_magnitude: float = 5.0  # How many sigma for outliers
    random_seed: int | None = None


@dataclass
class FitResult_Comparison:
    """Results from a single fitting function test."""

    function_name: str
    success: bool
    execution_time: float
    K_estimate: float | None = None
    K_error: float | None = None
    S0_y1_estimate: float | None = None
    S1_y1_estimate: float | None = None
    S0_y2_estimate: float | None = None
    S1_y2_estimate: float | None = None
    parameter_recovery_error: float | None = None
    outlier_robustness: float | None = None
    error_message: str | None = None
    additional_metrics: dict[str, Any] = field(default_factory=dict)


def generate_synthetic_dataset(
    params: SimulationParameters,
) -> tuple[Dataset, dict[str, float]]:
    """
    Generate synthetic dataset matching typical experimental conditions.

    Returns
    -------
    Dataset
        Generated dataset with realistic noise and outliers
    dict[str, float]
        True parameters used for generation
    """
    if params.random_seed is not None:
        np.random.seed(params.random_seed)

    # Generate pH points
    pH = np.linspace(params.pH_range[0], params.pH_range[1], params.n_points)
    x = 10 ** (-pH)  # Convert pH to [H+]

    # Generate true signals
    y1_true = binding_1site(
        x, 10 ** (-params.K_true), params.S0_y1_true, params.S1_y1_true, is_ph=True
    )
    y2_true = binding_1site(
        x, 10 ** (-params.K_true), params.S0_y2_true, params.S1_y2_true, is_ph=True
    )

    # Add realistic noise
    y1_err = params.base_error_y1 + 0.01 * np.abs(y1_true)  # Shot noise component
    y2_err = params.base_error_y2 + 0.05 * np.abs(
        y2_true
    )  # Higher relative noise for low signal

    # Make y2 errors effectively 10x larger relative to signal
    y2_err *= 10

    # Generate noisy observations
    y1_obs = y1_true + np.random.normal(0, y1_err)
    y2_obs = y2_true + np.random.normal(0, y2_err)

    # Add outliers to last 2 points with given probability
    if np.random.random() < params.outlier_probability:
        # Add outliers to y2 (the noisier channel) in last 2 points
        y2_obs[-2:] += (
            np.random.choice([-1, 1], 2) * params.outlier_magnitude * y2_err[-2:]
        )

    # Create DataArrays
    da1 = DataArray(xc=pH, yc=y1_obs, x_errc=np.full_like(pH, 0.01), y_errc=y1_err)
    da2 = DataArray(xc=pH, yc=y2_obs, x_errc=np.full_like(pH, 0.01), y_errc=y2_err)

    # Create Dataset
    dataset = Dataset({"y1": da1, "y2": da2}, is_ph=True)

    true_params = {
        "K": params.K_true,
        "S0_y1": params.S0_y1_true,
        "S1_y1": params.S1_y1_true,
        "S0_y2": params.S0_y2_true,
        "S1_y2": params.S1_y2_true,
    }

    return dataset, true_params


def calculate_parameter_recovery_error(
    result: Any, true_params: dict[str, float]
) -> float:
    """Calculate normalized parameter recovery error."""
    if not hasattr(result, "params") or result.params is None:
        return float("inf")

    errors = []
    for param_name, true_value in true_params.items():
        if param_name in result.params:
            estimated_value = result.params[param_name].value
            relative_error = abs(estimated_value - true_value) / abs(true_value)
            errors.append(relative_error)

    return np.mean(errors) if errors else float("inf")


def safe_fit_function(
    fit_func: Callable, *args, **kwargs
) -> tuple[Any, float, str | None]:
    """Safely execute a fitting function with timing and error handling."""
    start_time = time.time()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress warnings during comparison
            result = fit_func(*args, **kwargs)
        execution_time = time.time() - start_time
        return result, execution_time, None
    except Exception as e:
        execution_time = time.time() - start_time
        return None, execution_time, str(e)


def test_fitting_function(
    func_name: str,
    fit_func: Callable,
    dataset: Dataset,
    true_params: dict[str, float],
    initial_fit: FitResult | None = None,
) -> FitResult_Comparison:
    """Test a single fitting function and collect metrics."""
    # Prepare arguments based on function requirements
    if func_name in {
        "fit_binding_bayes",
        "fit_binding_bayes_perlabel",
        "fit_binding_pymc",
        "fit_binding_pymc2",
    }:
        if initial_fit is None or initial_fit.result is None:
            return FitResult_Comparison(
                function_name=func_name,
                success=False,
                execution_time=0.0,
                error_message="Requires initial fit result",
            )
        result, exec_time, error = safe_fit_function(fit_func, initial_fit)
    elif func_name.startswith("fit_binding_odr"):
        if initial_fit is None:
            return FitResult_Comparison(
                function_name=func_name,
                success=False,
                execution_time=0.0,
                error_message="Requires initial fit result",
            )
        result, exec_time, error = safe_fit_function(fit_func, initial_fit)
    elif func_name == "outlier2" or func_name in {
        "fit_binding_glob_reweighted",
        "core_reweighted",
    }:
        result, exec_time, error = safe_fit_function(fit_func, dataset, key="test")
    else:
        # Standard functions that take just the dataset
        result, exec_time, error = safe_fit_function(fit_func, dataset)

    if error or result is None:
        return FitResult_Comparison(
            function_name=func_name,
            success=False,
            execution_time=exec_time,
            error_message=error,
        )

    # Extract parameters from result
    success = False
    K_est = S0_y1_est = S1_y1_est = S0_y2_est = S1_y2_est = None
    K_err = None
    param_recovery_error = float("inf")

    try:
        # Handle different result types
        if hasattr(result, "result") and result.result is not None:
            params = result.result.params
            success = True
        elif hasattr(result, "params"):
            params = result.params
            success = True
        else:
            params = None

        if params is not None:
            if "K" in params:
                K_est = params["K"].value
                K_err = params["K"].stderr
            if "S0_y1" in params:
                S0_y1_est = params["S0_y1"].value
            if "S1_y1" in params:
                S1_y1_est = params["S1_y1"].value
            if "S0_y2" in params:
                S0_y2_est = params["S0_y2"].value
            if "S1_y2" in params:
                S1_y2_est = params["S1_y2"].value

            param_recovery_error = calculate_parameter_recovery_error(
                result.result if hasattr(result, "result") else result, true_params
            )

    except Exception as e:
        error = f"Parameter extraction failed: {e!s}"

    return FitResult_Comparison(
        function_name=func_name,
        success=success,
        execution_time=exec_time,
        K_estimate=K_est,
        K_error=K_err,
        S0_y1_estimate=S0_y1_est,
        S1_y1_estimate=S1_y1_est,
        S0_y2_estimate=S0_y2_est,
        S1_y2_estimate=S1_y2_est,
        parameter_recovery_error=param_recovery_error,
        error_message=error,
    )


def run_comprehensive_comparison(
    n_simulations: int = 100, phase: str = "deterministic"
) -> pd.DataFrame:
    """Run comprehensive comparison of fitting functions in phases.

    Parameters
    ----------
    n_simulations : int
        Number of simulations to run
    phase : str
        "deterministic" for basic fitting functions, "bayesian" for advanced functions
    """
    if phase == "deterministic":
        # Phase 1: Test all deterministic fitting functions
        fitting_functions = {
            # API functions (recommended interface)
            "fit_binding_lm": fit_binding_lm,
            "fit_binding_lm_outlier": fit_binding_lm_outlier,
            "fit_binding (lm)": lambda ds: fit_binding(ds, method=FitMethod.LM),
            "fit_binding (lm_outlier)": lambda ds: fit_binding(
                ds, method=FitMethod.LM_OUTLIER, key="test"
            ),
            # Core functions (direct implementations)
            "fit_binding_glob": fit_binding_glob,
            "fit_lm": fit_lm,
            "fit_lm (robust)": lambda ds: fit_lm(ds, robust=True),
            "fit_lm (iterative)": lambda ds: fit_lm(ds, iterative=True),
            "fit_lm (outlier)": lambda ds: fit_lm(ds, outlier_threshold=3.0),
            "outlier2": outlier2,
            # Legacy core functions
            "core_recursive": core_recursive,
            "core_recursive_outlier": core_recursive_outlier,
            "core_reweighted": core_reweighted,
            # API legacy shims
            "api_recursive": fit_binding_glob_recursive,
            "api_recursive_outlier": fit_binding_glob_recursive_outlier,
            "api_reweighted": fit_binding_glob_reweighted,
        }

        results = []
        print(
            f"Phase 1: Testing {len(fitting_functions)} deterministic functions with {n_simulations} simulations..."
        )

        for sim_idx in range(n_simulations):
            if sim_idx % 10 == 0:
                print(f"  Simulation {sim_idx + 1}/{n_simulations}")

            # Generate synthetic dataset
            sim_params = SimulationParameters(random_seed=sim_idx)
            dataset, true_params = generate_synthetic_dataset(sim_params)

            # Test deterministic functions
            for func_name, fit_func in fitting_functions.items():
                result = test_fitting_function(
                    func_name, fit_func, dataset, true_params
                )
                result.additional_metrics["simulation_id"] = sim_idx
                result.additional_metrics["has_outliers"] = (
                    sim_params.outlier_probability > np.random.random()
                )
                result.additional_metrics["phase"] = "deterministic"
                results.append(result)

    elif phase == "bayesian":
        # Phase 2: Test advanced functions using best deterministic method as initial fit
        advanced_functions = {
            "fit_binding_bayes": fit_binding_bayes,
            "fit_binding_bayes_perlabel": fit_binding_bayes_perlabel,
            "fit_binding_pymc": fit_binding_pymc,
            "fit_binding_pymc2": fit_binding_pymc2,
            "fit_binding_odr": fit_binding_odr,
            "fit_binding_odr_recursive": fit_binding_odr_recursive,
            "fit_binding_odr_recursive_outlier": fit_binding_odr_recursive_outlier,
        }

        # Load deterministic results to find best method
        try:
            det_df = pd.read_csv("fitting_comparison_deterministic.csv")
            # Import calculate_robustness_metrics function from analysis script
            from analyze_fitting_results import calculate_robustness_metrics

            det_metrics = calculate_robustness_metrics(det_df)
            best_deterministic_func = det_metrics.iloc[0]["function_name"]
            print(
                f"Phase 2: Using '{best_deterministic_func}' as initial fit for advanced functions"
            )

            # Get the actual function for the best deterministic method
            all_det_functions = {
                "fit_binding_lm": fit_binding_lm,
                "fit_binding_lm_outlier": fit_binding_lm_outlier,
                "fit_binding (lm)": lambda ds: fit_binding(ds, method=FitMethod.LM),
                "fit_binding (lm_outlier)": lambda ds: fit_binding(
                    ds, method=FitMethod.LM_OUTLIER, key="test"
                ),
                "fit_binding_glob": fit_binding_glob,
                "fit_lm": fit_lm,
                "outlier2": outlier2,
            }
            best_det_func = all_det_functions.get(
                best_deterministic_func, fit_binding_glob
            )

        except FileNotFoundError:
            print(
                "Warning: Deterministic results not found, using fit_binding_glob as default"
            )
            best_det_func = fit_binding_glob

        results = []
        print(
            f"Phase 2: Testing {len(advanced_functions)} advanced functions with {n_simulations} simulations..."
        )

        for sim_idx in range(n_simulations):
            if sim_idx % 5 == 0:  # More frequent updates for slower functions
                print(f"  Simulation {sim_idx + 1}/{n_simulations}")

            # Generate synthetic dataset
            sim_params = SimulationParameters(random_seed=sim_idx)
            dataset, true_params = generate_synthetic_dataset(sim_params)

            # Get initial fit using best deterministic method
            initial_fit = None
            try:
                initial_fit = best_det_func(dataset)
            except Exception as e:
                print(f"  Warning: Initial fit failed for simulation {sim_idx}: {e}")
                continue

            # Test advanced functions if initial fit succeeded
            if initial_fit is not None and initial_fit.result is not None:
                for func_name, fit_func in advanced_functions.items():
                    result = test_fitting_function(
                        func_name, fit_func, dataset, true_params, initial_fit
                    )
                    result.additional_metrics["simulation_id"] = sim_idx
                    result.additional_metrics["has_outliers"] = (
                        sim_params.outlier_probability > np.random.random()
                    )
                    result.additional_metrics["phase"] = "bayesian"
                    results.append(result)

    else:
        msg = f"Unknown phase: {phase}. Use 'deterministic' or 'bayesian'"
        raise ValueError(msg)

    # Convert to DataFrame for analysis
    df_data = []
    for result in results:
        row = {
            "function_name": result.function_name,
            "simulation_id": result.additional_metrics.get("simulation_id", -1),
            "success": result.success,
            "execution_time": result.execution_time,
            "K_estimate": result.K_estimate,
            "K_error": result.K_error,
            "S0_y1_estimate": result.S0_y1_estimate,
            "S1_y1_estimate": result.S1_y1_estimate,
            "S0_y2_estimate": result.S0_y2_estimate,
            "S1_y2_estimate": result.S1_y2_estimate,
            "parameter_recovery_error": result.parameter_recovery_error,
            "error_message": result.error_message,
            "has_outliers": result.additional_metrics.get("has_outliers", False),
            "phase": result.additional_metrics.get("phase", "unknown"),
        }
        df_data.append(row)

    return pd.DataFrame(df_data)


if __name__ == "__main__":
    import sys

    # Parse command line arguments
    if len(sys.argv) > 1:
        phase = sys.argv[1]
        n_sims = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    else:
        phase = "deterministic"
        n_sims = 10

    print(f"Starting {phase} fitting function comparison with {n_sims} simulations...")

    # Run the comparison
    results_df = run_comprehensive_comparison(n_simulations=n_sims, phase=phase)

    # Save results
    output_file = f"fitting_comparison_{phase}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

    # Basic analysis
    print("\n" + "=" * 60)
    print(f"PRELIMINARY RESULTS - {phase.upper()} PHASE")
    print("=" * 60)

    success_rates = results_df.groupby("function_name")["success"].agg([
        "mean",
        "count",
    ])
    print("\nSuccess Rates:")
    print(success_rates.sort_values("mean", ascending=False))

    # Parameter recovery for successful fits
    successful_fits = results_df[
        results_df["success"] & results_df["parameter_recovery_error"].notna()
    ]
    if not successful_fits.empty:
        param_recovery = successful_fits.groupby("function_name")[
            "parameter_recovery_error"
        ].agg(["mean", "std", "count"])
        print("\nParameter Recovery Error (lower is better):")
        print(param_recovery.sort_values("mean"))

    # Execution time for successful fits
    if not successful_fits.empty:
        exec_times = successful_fits.groupby("function_name")["execution_time"].agg([
            "mean",
            "std",
        ])
        print("\nExecution Time (seconds):")
        print(exec_times.sort_values("mean"))

    # Phase-specific recommendations
    if phase == "deterministic":
        print("\n" + "=" * 60)
        print("NEXT STEPS")
        print("=" * 60)
        print(
            "1. Review the results above to identify the best deterministic functions"
        )
        print("2. Run the analysis script: python analyze_fitting_results.py")
        print(
            "3. Then test Bayesian functions: python fitting_function_comparison.py bayesian 10"
        )
    elif phase == "bayesian":
        print("\n" + "=" * 60)
        print("NEXT STEPS")
        print("=" * 60)
        print("1. Run the full analysis: python analyze_fitting_results.py")
        print("2. Review the comprehensive recommendations in the generated files")
