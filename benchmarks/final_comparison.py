#!/usr/bin/env python3
"""Final benchmark comparison of fitting methods.

Compares weight assignment, outlier handling, and robustness across all fitting
backends using synthetic data that mirrors real experimental conditions:
- y1 (FRET acceptor): 10X noisier, outlier-prone at low pH
- y2 (FRET donor): lower noise
- Physics error model: err = sqrt(signal + buffer_sd^2)

Outputs a CSV with per-method, per-condition metrics (coverage, bias, RMSE).
"""

from __future__ import annotations

import itertools
import sys
import warnings
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

# Append project root so the script works from benchmarks/ directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from clophfit.testing.evaluation import (
    calculate_bias,
    calculate_coverage,
    calculate_rmse,
    extract_params,
)
from clophfit.testing.fitter_test_utils import build_fitters
from clophfit.testing.synthetic import make_benchmark_dataset

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Configuration ────────────────────────────────────────────────────────

N_REPEATS = 100
TRUE_K_VALUES = [6.0, 6.5, 7.0, 7.5, 8.0]

SCENARIOS: list[dict[str, object]] = [
    {
        "name": "2label_heteroscedastic",
        "n_labels": 2,
        "error_ratio": 0.1,
        "add_outlier": False,
    },
    {
        "name": "2label_hetero_outlier",
        "n_labels": 2,
        "error_ratio": 0.1,
        "add_outlier": True,
        "outlier_label": "y1",
    },
    {
        "name": "1label_clean",
        "n_labels": 1,
        "error_ratio": 1.0,
        "add_outlier": False,
    },
    {
        "name": "1label_outlier",
        "n_labels": 1,
        "error_ratio": 1.0,
        "add_outlier": True,
        "outlier_label": "y1",
    },
]

# ── Benchmark runner ─────────────────────────────────────────────────────


def run_benchmark() -> pd.DataFrame:
    """Run the full benchmark grid and return results DataFrame."""
    fitters = build_fitters(include_odr=True)
    method_names = list(fitters.keys())

    rows: list[dict[str, object]] = []
    total = len(SCENARIOS) * len(TRUE_K_VALUES) * len(method_names)
    done = 0
    t0 = perf_counter()

    for scenario in SCENARIOS:
        sc_name = str(scenario["name"])
        sc_kwargs = {k: v for k, v in scenario.items() if k != "name"}

        for true_k in TRUE_K_VALUES:
            # Pre-generate all datasets for this (scenario, K) pair
            rng = np.random.default_rng(42)
            datasets = []
            for _ in range(N_REPEATS):
                ds, truth = make_benchmark_dataset(k=true_k, rng=rng, **sc_kwargs)
                datasets.append((ds, truth))

            for method_name in method_names:
                fitter = fitters[method_name]
                k_vals = np.full(N_REPEATS, np.nan)
                k_errs = np.full(N_REPEATS, np.nan)
                n_fail = 0

                for i, (ds, truth) in enumerate(datasets):
                    try:
                        fr = fitter(ds)
                        val, err = extract_params(fr, "K")
                        k_vals[i] = val
                        k_errs[i] = err
                    except Exception:
                        n_fail += 1

                # Calculate metrics
                valid = np.isfinite(k_vals)
                n_valid = int(valid.sum())

                row = {
                    "scenario": sc_name,
                    "true_K": true_k,
                    "method": method_name,
                    "n_valid": n_valid,
                    "n_fail": n_fail,
                    "bias": calculate_bias(k_vals, true_k),
                    "rmse": calculate_rmse(k_vals, true_k),
                    "coverage_95": calculate_coverage(k_vals, k_errs, true_k, 0.95),
                    "coverage_90": calculate_coverage(k_vals, k_errs, true_k, 0.90),
                    "mean_stderr": float(np.nanmean(k_errs)),
                    "empirical_sd": float(np.nanstd(k_vals[valid]))
                    if n_valid > 1
                    else np.nan,
                }
                rows.append(row)

                done += 1
                elapsed = perf_counter() - t0
                eta = elapsed / done * (total - done)
                print(
                    f"  [{done}/{total}] {sc_name:30s} K={true_k} "
                    f"{method_name:25s} cov95={row['coverage_95']:.0%} "
                    f"bias={row['bias']:+.4f}  (ETA {eta:.0f}s)",
                    flush=True,
                )

    return pd.DataFrame(rows)


# ── Summary table ────────────────────────────────────────────────────────


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Produce a summary table averaged over true_K values per (scenario, method)."""
    grouped = (
        df.groupby(["scenario", "method"])
        .agg(
            coverage_95=("coverage_95", "mean"),
            coverage_90=("coverage_90", "mean"),
            mean_bias=("bias", "mean"),
            abs_bias=("bias", lambda s: float(np.mean(np.abs(s)))),
            mean_rmse=("rmse", "mean"),
            mean_stderr=("mean_stderr", "mean"),
            empirical_sd=("empirical_sd", "mean"),
            n_fail=("n_fail", "sum"),
        )
        .reset_index()
    )
    # Stderr calibration: ratio of reported stderr to empirical SD
    # Ideal = 1.0; <1 means underestimation of uncertainty
    grouped["stderr_ratio"] = grouped["mean_stderr"] / grouped["empirical_sd"]
    return grouped.sort_values(["scenario", "coverage_95"], ascending=[True, False])


# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Running final benchmark: {N_REPEATS} repeats × "
          f"{len(TRUE_K_VALUES)} K values × {len(SCENARIOS)} scenarios × "
          f"{len(build_fitters())} methods")
    print("=" * 80)

    df = run_benchmark()

    out_dir = Path(__file__).resolve().parent
    csv_path = out_dir / "final_comparison_results.csv"
    df.to_csv(csv_path, index=False, float_format="%.5f")
    print(f"\nRaw results → {csv_path}")

    summary = summarize(df)
    summary_path = out_dir / "final_comparison_summary.csv"
    summary.to_csv(summary_path, index=False, float_format="%.4f")
    print(f"Summary     → {summary_path}")

    # Print nicely formatted summary
    print("\n" + "=" * 80)
    print("SUMMARY (averaged over K values)")
    print("=" * 80)
    for sc_name in summary["scenario"].unique():
        sc_df = summary[summary["scenario"] == sc_name]
        print(f"\n┌─ {sc_name} {'─' * (75 - len(sc_name))}")
        print(
            f"│ {'Method':25s} │ Cov95 │ Cov90 │  Bias   │  RMSE  │ "
            f"SE/SD  │ Fails"
        )
        print("│" + "─" * 78)
        for _, r in sc_df.iterrows():
            print(
                f"│ {r['method']:25s} │ {r['coverage_95']:5.1%} │ "
                f"{r['coverage_90']:5.1%} │ {r['mean_bias']:+7.4f} │ "
                f"{r['mean_rmse']:6.4f} │ {r['stderr_ratio']:5.2f}x │ "
                f"{int(r['n_fail']):5d}"
            )
        print("└" + "─" * 78)

    print(f"\nDone in {perf_counter():.0f}s total.")
