#!/usr/bin/env python3

"""Comprehensive comparison of ALL fitting methods using proper Titration pipeline.

Uses Titration to get properly corrected data (buffer subtraction + dilution).
Tests all available fitting functions:
- fit_binding_glob (LM variants)
- fit_binding_odr (ODR)
- fit_binding_pymc / pymc2 (Bayesian MCMC)
- outlier2

This reveals the true production behavior.
"""

import time
import warnings
from pathlib import Path

import pandas as pd

from clophfit.fitting.core import (
    fit_binding_glob,
    outlier2,
)
from clophfit.fitting.odr import (
    fit_binding_odr_recursive,
    fit_binding_odr_recursive_outlier,
)
from clophfit.prtecan import Titration


def load_titration(dataset_dir: Path) -> tuple[str, Titration | None]:
    """Load a Titration dataset."""
    list_file = dataset_dir / "list.pH.csv"
    scheme_file = dataset_dir / "scheme.txt"
    additions_file = dataset_dir / "additions.pH"

    if not list_file.exists() or not scheme_file.exists():
        return dataset_dir.name, None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tit = Titration.fromlistfile(list_file, is_ph=True)
            tit.load_scheme(scheme_file)
            if additions_file.exists():
                tit.load_additions(additions_file)
        return dataset_dir.name, tit
    except Exception as e:
        print(f"  Warning: Failed to load {dataset_dir.name}: {e}")
        return dataset_dir.name, None


def test_method(
    method_name: str, method_func, ds, key: str, needs_key: bool = False
) -> dict:
    """Test a single method on a dataset."""
    try:
        start = time.time()

        # For Bayesian methods that need a prior fit
        if method_name.startswith("Bayesian"):
            # Need to do LM fit first
            lm_result = fit_binding_glob(ds)
            if lm_result.result and lm_result.result.success:
                result = method_func(lm_result, n_samples=1000)  # Quick sampling
            else:
                msg = "LM fit failed, cannot run Bayesian"
                raise ValueError(msg)

        # For ODR methods that need a prior fit
        elif method_name.startswith("ODR") and method_name != "ODR-Base":
            lm_result = fit_binding_glob(ds)
            if lm_result.result and lm_result.result.success:
                result = method_func(lm_result)
            else:
                msg = "LM fit failed, cannot run ODR"
                raise ValueError(msg)

        # Standard methods
        elif needs_key:
            result = method_func(ds, key=key)
        else:
            result = method_func(ds)

        elapsed = time.time() - start

        # Extract K value
        K_val = None
        K_err = None

        if hasattr(result, "result") and result.result:
            if hasattr(result.result, "success") and result.result.success:
                if hasattr(result.result, "params") and "K" in result.result.params:
                    K_val = result.result.params["K"].value
                    K_err = result.result.params["K"].stderr or 0.0

        # For Bayesian, extract from trace
        if method_name.startswith("Bayesian") and K_val is None:
            if hasattr(result, "idata") and result.idata is not None:
                import arviz as az

                summary = az.summary(result.idata, var_names=["K"])
                if "mean" in summary.columns:
                    K_val = float(summary["mean"].values[0])
                    K_err = float(summary["sd"].values[0])

        if K_val is not None:
            return {
                "success": True,
                "K": K_val,
                "K_stderr": K_err or 0.0,
                "time_s": elapsed,
            }
        return {
            "success": False,
            "K": None,
            "K_stderr": None,
            "time_s": elapsed,
            "error": "No K parameter found",
        }

    except Exception as e:
        return {
            "success": False,
            "K": None,
            "K_stderr": None,
            "time_s": 0.0,
            "error": str(e)[:80],
        }


def test_all_methods(tit: Titration, well_key: str) -> dict[str, dict]:
    """Test all fitting methods on a well using proper Titration pipeline."""
    # Get properly corrected dataset using Titration's internal method
    ds = tit._create_global_ds(well_key)

    # Define all methods to test
    methods = [
        (
            "Standard LM",
            fit_binding_glob,
            lambda d: fit_binding_glob(d, robust=False),
            False,
        ),
        (
            "Robust Huber",
            fit_binding_glob,
            lambda d: fit_binding_glob(d, robust=True),
            False,
        ),
        ("Outlier2", outlier2, outlier2, True),
        ("ODR-Recursive", fit_binding_odr_recursive, fit_binding_odr_recursive, False),
        (
            "ODR-Recursive+Outlier",
            fit_binding_odr_recursive_outlier,
            fit_binding_odr_recursive_outlier,
            False,
        ),
    ]

    # Skip Bayesian for full-plate analysis (too slow: 4.5s × 300 wells = 22 minutes per method)
    # Bayesian will be tested separately on subset
    # methods.extend([
    #     ("Bayesian-Shared", fit_binding_pymc, fit_binding_pymc, False),
    #     ("Bayesian-PerLabel", fit_binding_pymc2, fit_binding_pymc2, False),
    # ])

    results = {}
    for method_name, _, method_func, needs_key in methods:
        results[method_name] = test_method(
            method_name, method_func, ds, well_key, needs_key
        )

    return results


def main():
    """Run comprehensive comparison on all Tecan datasets."""
    print("=" * 80)
    print("COMPREHENSIVE FITTING METHODS COMPARISON")
    print("Using proper Titration pipeline (buffer + dilution corrected)")
    print("=" * 80)
    print()

    # Find all Tecan datasets (now including L2!)
    tecan_dirs = [
        Path("tests/Tecan/140220"),
        Path("tests/Tecan/L1"),
        Path("tests/Tecan/L2"),
        Path("tests/Tecan/L4"),
    ]

    all_results = []

    for dataset_dir in tecan_dirs:
        if not dataset_dir.exists():
            continue

        print(f"Loading dataset: {dataset_dir.name}")
        dataset_name, tit = load_titration(dataset_dir)

        if tit is None:
            print("  Failed to load")
            continue

        # Get ALL wells (full plate analysis - critical for production evaluation)
        well_keys = list(tit.fit_keys)
        print(f"  Testing ALL {len(well_keys)} wells (full plate)")

        # Test each well
        for well_key in well_keys:
            print(f"  Testing well {well_key}", end=" ... ")

            results = test_all_methods(tit, well_key)

            # Count successes
            n_success = sum(1 for r in results.values() if r["success"])
            print(f"{n_success}/{len(results)} methods succeeded")

            # Store results
            for method, res in results.items():
                # Add sample name from scheme if available
                sample_name = None
                if hasattr(tit.scheme, "names"):
                    for sname, wells in tit.scheme.names.items():
                        if well_key in wells:
                            sample_name = sname
                            break

                all_results.append(
                    {
                        "dataset": dataset_name,
                        "well": well_key,
                        "sample": sample_name,
                        "method": method,
                        **res,
                    }
                )
        print()

    # Create dataframe
    df = pd.DataFrame(all_results)

    # Aggregate analysis
    print("=" * 80)
    print("AGGREGATE ANALYSIS")
    print("=" * 80)
    print()

    # Success rate by method
    print("SUCCESS RATE BY METHOD:")
    success_rate = df.groupby("method")["success"].mean() * 100
    print(success_rate.sort_values(ascending=False).to_string())
    print()

    # Speed comparison (successful fits only)
    print("AVERAGE EXECUTION TIME (seconds, successful fits):")
    df_success = df[df["success"]]
    if len(df_success) > 0:
        time_stats = df_success.groupby("method")["time_s"].mean().sort_values()
        print(time_stats.to_string())
        print()

    # pKa statistics
    print("pKa STATISTICS BY METHOD (successful fits):")
    if len(df_success) > 0:
        pka_stats = df_success.groupby("method")["K"].agg(["mean", "std", "count"])
        print(pka_stats.to_string())
        print()

    # ERROR BAR PRECISION - KEY METRIC from visual inspection
    print("ERROR BAR PRECISION (mean K_stderr - smaller is better):")
    if len(df_success) > 0 and "K_stderr" in df_success.columns:
        stderr_stats = df_success.groupby("method")["K_stderr"].agg(
            [
                "mean",
                "median",
                "std",
            ]
        )
        print(stderr_stats.sort_values("mean").to_string())
        print()

        # Flag methods with tighter error bars
        best_stderr = stderr_stats["mean"].min()
        print(
            f"Best precision (tightest error bars): {stderr_stats['mean'].idxmin()} ({best_stderr:.4f})"
        )
        print()

    # Per-dataset breakdown
    print("PERFORMANCE BY DATASET:")
    for dataset in df["dataset"].unique():
        df_ds = df[df["dataset"] == dataset]
        df_ds_success = df_ds[df_ds["success"]]
        n_total = len(df_ds)
        n_success = df_ds["success"].sum()

        print(f"\n{dataset}: {n_success}/{n_total} successful fits")

        # Success rate per method
        success_by_method = df_ds.groupby("method")["success"].mean() * 100
        print("  Success rates:")
        for method, rate in success_by_method.sort_values(ascending=False).items():
            print(f"    {method:<25}: {rate:>5.0f}%")

        # Error bar precision per dataset
        if len(df_ds_success) > 0 and "K_stderr" in df_ds_success.columns:
            print("\n  Mean K_stderr (precision):")
            stderr_by_method = (
                df_ds_success.groupby("method")["K_stderr"].mean().sort_values()
            )
            for method, stderr in stderr_by_method.items():
                print(f"    {method:<25}: {stderr:.4f}")

    # Identify redundant methods
    print("\n" + "=" * 80)
    print("IDENTIFYING REDUNDANT METHODS:")
    print("=" * 80)

    # Group by dataset+well, compare pKa values
    for (dataset, well), group in df_success.groupby(["dataset", "well"]):
        if len(group) < 2:
            continue

        # Find pairs with identical results
        methods = group["method"].values
        pkas = group["K"].values

        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                if abs(pkas[i] - pkas[j]) < 0.001:  # Identical to 3 decimals
                    print(
                        f"  {dataset}/{well}: {methods[i]} ≡ {methods[j]} (Δ={abs(pkas[i] - pkas[j]):.5f})"
                    )

    # Save results
    df.to_csv("comprehensive_fitting_comparison.csv", index=False)
    print("\n" + "=" * 80)
    print("Results saved to: comprehensive_fitting_comparison.csv")
    print("=" * 80)


if __name__ == "__main__":
    main()
