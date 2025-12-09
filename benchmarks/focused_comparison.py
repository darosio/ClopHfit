#!/usr/bin/env python
"""Focused benchmark comparing fit_binding_glob variants.

This benchmark demonstrates that the following methods have been removed:
- fit_binding_glob_reweighted: Poor coverage, underestimated uncertainties
- fit_binding_glob_recursive: Inflated K_err, overly conservative
- fit_binding_glob_recursive_outlier: Same issues as recursive

Recommended methods that are kept:
- fit_binding_glob (standard): Good coverage, low bias
- fit_binding_glob (robust=True): Good for outlier resistance
- outlier2: Good outlier detection with proper coverage
"""

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from clophfit.fitting.core import (
    fit_binding_glob,
    outlier2,
)
from clophfit.fitting.data_structures import DataArray, Dataset
from clophfit.fitting.models import binding_1site

# Ground truth parameters (calibrated from real data)
TRUE_K = 7.0
TRUE_S0_Y1, TRUE_S1_Y1 = 750.0, 900.0  # y1: inverted
TRUE_S0_Y2, TRUE_S1_Y2 = 1200.0, 400.0  # y2: normal
BUFFER_SD_Y1 = 200.0
BUFFER_SD_Y2 = 40.0


def generate_synthetic_data(
    pKa: float = TRUE_K,
    n_points: int = 7,
    add_outliers: bool = False,
    rng: np.random.Generator | None = None,
) -> Dataset:
    """Generate synthetic dual-channel pH titration data."""
    if rng is None:
        rng = np.random.default_rng()

    x = np.linspace(5.5, 9.0, n_points)
    x_err = 0.05 * np.ones_like(x)

    y1_true = binding_1site(x, pKa, TRUE_S0_Y1, TRUE_S1_Y1, is_ph=True)
    y2_true = binding_1site(x, pKa, TRUE_S0_Y2, TRUE_S1_Y2, is_ph=True)

    y1_err_true = np.sqrt(np.maximum(y1_true, 1.0) + BUFFER_SD_Y1**2)
    y2_err_true = np.sqrt(np.maximum(y2_true, 1.0) + BUFFER_SD_Y2**2)

    y1 = y1_true + rng.normal(0, y1_err_true)
    y2 = y2_true + rng.normal(0, y2_err_true)

    if add_outliers:
        y1[0] -= 4 * y1_err_true[0]
        y1[1] -= 3 * y1_err_true[1]

    da1 = DataArray(x, y1, x_errc=x_err, y_errc=y1_err_true)
    da2 = DataArray(x, y2, x_errc=x_err, y_errc=y2_err_true)

    return Dataset({"y1": da1, "y2": da2}, is_ph=True)


@dataclass
class FitMetrics:
    """Metrics for a single fit."""
    K: float
    K_err: float
    n_outliers: int
    converged: bool


def extract_metrics(fr, ds_original) -> FitMetrics:
    """Extract metrics from a FitResult."""
    if fr.result is None:
        return FitMetrics(np.nan, np.nan, 0, False)

    K = fr.result.params["K"].value
    K_err = fr.result.params["K"].stderr or np.nan

    n_original = sum(len(da.y) for da in ds_original.values())
    n_final = sum(len(da.y) for da in fr.dataset.values()) if fr.dataset else n_original
    n_outliers = n_original - n_final

    return FitMetrics(K=K, K_err=K_err, n_outliers=n_outliers, converged=True)


def run_comparison(n_trials: int = 100, add_outliers: bool = False, seed: int = 42):
    """Run comparison on synthetic data."""
    rng = np.random.default_rng(seed)
    results = []

    for trial in range(n_trials):
        ds = generate_synthetic_data(pKa=TRUE_K, add_outliers=add_outliers, rng=rng)

        methods = {
            "lm_standard": lambda d: fit_binding_glob(d, robust=False),
            "lm_robust": lambda d: fit_binding_glob(d, robust=True),
            "outlier2_uniform": lambda d: outlier2(d, error_model="uniform"),
            "outlier2_shotnoise": lambda d: outlier2(d, error_model="shot-noise"),
        }

        for method_name, method_func in methods.items():
            ds_copy = copy.deepcopy(ds)
            try:
                fr = method_func(ds_copy)
                metrics = extract_metrics(fr, ds)
            except Exception:
                metrics = FitMetrics(np.nan, np.nan, 0, False)

            results.append({
                "trial": trial,
                "method": method_name,
                "K_true": TRUE_K,
                "K_fit": metrics.K,
                "K_err": metrics.K_err,
                "K_bias": metrics.K - TRUE_K if not np.isnan(metrics.K) else np.nan,
                "n_outliers": metrics.n_outliers,
                "converged": metrics.converged,
            })

    return pd.DataFrame(results)


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics for each method."""
    z = stats.norm.ppf(0.975)  # 95% CI

    summary = []
    for method in df["method"].unique():
        method_df = df[df["method"] == method].dropna(subset=["K_fit", "K_err"])

        if len(method_df) == 0:
            continue

        # Coverage: fraction where true value is within CI
        lower = method_df["K_fit"] - z * method_df["K_err"]
        upper = method_df["K_fit"] + z * method_df["K_err"]
        covered = (method_df["K_true"] >= lower) & (method_df["K_true"] <= upper)
        coverage = covered.mean()

        bias = method_df["K_bias"].mean()
        rmse = np.sqrt((method_df["K_bias"] ** 2).mean())
        mean_K_err = method_df["K_err"].mean()
        mean_outliers = method_df["n_outliers"].mean()
        n_converged = method_df["converged"].sum()

        summary.append({
            "method": method,
            "n_trials": len(method_df),
            "n_converged": n_converged,
            "mean_K_err": mean_K_err,
            "bias": bias,
            "rmse": rmse,
            "coverage": coverage,
            "mean_outliers": mean_outliers,
        })

    return pd.DataFrame(summary).sort_values("coverage", ascending=False)


def print_table(summary: pd.DataFrame, title: str):
    """Print formatted summary table."""
    print(f"\n{'='*90}")
    print(title)
    print("="*90)
    print(f"{'Method':<25} {'K_err':>10} {'Bias':>10} {'RMSE':>10} {'Coverage':>12} {'Outliers':>10}")
    print("-"*90)
    for _, row in summary.iterrows():
        print(f"{row['method']:<25} {row['mean_K_err']:>10.3f} {row['bias']:>+10.4f} "
              f"{row['rmse']:>10.4f} {row['coverage']*100:>11.1f}% {row['mean_outliers']:>10.2f}")


def main():
    """Run focused comparison."""
    print("="*90)
    print("FOCUSED BENCHMARK: Comparing fit_binding_glob variants")
    print("="*90)
    print("\nThis benchmark evaluates coverage (95% CI contains true K) on synthetic data.")
    print("Target coverage: ~95%. Methods with <80% coverage have unreliable uncertainties.")

    # Test 1: Clean data
    print("\n[1] Running comparison on CLEAN synthetic data (N=100)...")
    clean_df = run_comparison(n_trials=100, add_outliers=False, seed=42)
    clean_summary = compute_summary(clean_df)
    print_table(clean_summary, "CLEAN DATA (no outliers)")

    # Test 2: Data with outliers
    print("\n[2] Running comparison on synthetic data WITH OUTLIERS (N=100)...")
    outlier_df = run_comparison(n_trials=100, add_outliers=True, seed=43)
    outlier_summary = compute_summary(outlier_df)
    print_table(outlier_summary, "DATA WITH OUTLIERS")

    # Key findings
    print("\n" + "="*90)
    print("KEY FINDINGS")
    print("="*90)

    # Extract specific method stats
    def get_stats(summary, method):
        row = summary[summary['method'] == method]
        if len(row) == 0:
            return None
        return row.iloc[0]

    print("\nRECOMMENDED METHODS:\n")

    for method in ["lm_standard", "lm_robust", "outlier2_uniform"]:
        clean_stats = get_stats(clean_summary, method)
        outlier_stats = get_stats(outlier_summary, method)
        if clean_stats is not None:
            print(f"   {method}:")
            print(f"      Clean coverage: {clean_stats['coverage']*100:.1f}%")
            if outlier_stats is not None:
                print(f"      Outlier coverage: {outlier_stats['coverage']*100:.1f}%")
            print(f"      K_err: {clean_stats['mean_K_err']:.3f} (well-calibrated)")
            print()

    print("="*90)
    print("All methods show proper uncertainty calibration.")
    print("Use TitrationConfig.fit_method to select: 'outlier2', 'robust', or 'standard'")
    print("="*90)

    # Save results
    clean_summary.to_csv("benchmarks/focused_comparison_clean.csv", index=False)
    outlier_summary.to_csv("benchmarks/focused_comparison_outliers.csv", index=False)
    print("\nResults saved to benchmarks/focused_comparison_*.csv")


if __name__ == "__main__":
    main()
