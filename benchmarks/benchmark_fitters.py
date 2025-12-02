#!/usr/bin/env python3
"""
Benchmark multiple fitters on synthetic datasets and compare accuracy.

Generates N datasets with randomized signal parameters matching real L4 data
distributions, and runs all fitters to compute statistically meaningful comparisons.

Key features:
- Uses make_dataset(randomize_signals=True) for realistic signal magnitudes
- Supports per-label error scaling (y1 vs y2 differential noise)
- Computes MAE, RMSE, median, and success rate per fitter
- Optionally runs statistical significance tests between fitters
"""
from __future__ import annotations

import math
import statistics as stats
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field

import numpy as np
from scipy import stats as scipy_stats

from clophfit.fitting.data_structures import Dataset, FitResult
from clophfit.testing.fitter_test_utils import build_fitters, k_from_result, s_from_result
from clophfit.testing.synthetic import TruthParams, make_dataset


def mae(values: list[float]) -> float:
    """Compute Mean Absolute Error."""
    return sum(abs(v) for v in values) / len(values) if values else float("nan")


def rmse(values: list[float]) -> float:
    """Compute Root Mean Square Error."""
    return (
        math.sqrt(sum(v * v for v in values) / len(values)) if values else float("nan")
    )


@dataclass
class FitterStats:
    """Statistics for a single fitter."""

    name: str
    k_errors: list[float] = field(default_factory=list)
    s0_errors: list[float] = field(default_factory=list)
    s1_errors: list[float] = field(default_factory=list)
    n_success: int = 0
    n_total: int = 0

    @property
    def success_rate(self) -> float:
        return 100.0 * self.n_success / self.n_total if self.n_total else 0.0

    @property
    def k_mae(self) -> float:
        return mae(self.k_errors)

    @property
    def k_rmse(self) -> float:
        return rmse(self.k_errors)

    @property
    def k_median(self) -> float:
        return stats.median(self.k_errors) if self.k_errors else float("nan")


def run_benchmark(
    n_repeats: int = 100,
    *,
    include_odr: bool = True,
    error_model: str = "realistic",
    rel_error: float | dict[str, float] = 0.035,
    outlier_prob: float = 0.0,
    noise_multiplier: float = 1.0,
    verbose: bool = True,
) -> dict[str, FitterStats]:
    """Run benchmark comparison of fitting methods.

    Parameters
    ----------
    n_repeats : int
        Number of synthetic datasets to generate.
    include_odr : bool
        Include ODR-based fitters (slower).
    error_model : str
        Error model: "simple", "realistic", or "physics".
    rel_error : float or dict
        Relative error. Use dict for per-label scaling, e.g., {"y1": 0.07, "y2": 0.025}.
    outlier_prob : float
        Probability of outlier per label (0-1).
    noise_multiplier : float
        Stress factor for noise.
    verbose : bool
        Print progress.

    Returns
    -------
    dict[str, FitterStats]
        Statistics per fitter.
    """
    fitters = build_fitters(include_odr=include_odr)
    results: dict[str, FitterStats] = {name: FitterStats(name=name) for name in fitters}

    for i in range(n_repeats):
        if verbose and (i + 1) % 20 == 0:
            print(f"  Progress: {i + 1}/{n_repeats}")

        # Generate dataset with randomized signals from real data distributions
        ds, truth = make_dataset(
            randomize_signals=True,
            seed=i,
            error_model=error_model,
            rel_error=rel_error,
            outlier_prob=outlier_prob,
            noise_multiplier=noise_multiplier,
        )

        for name, run in fitters.items():
            results[name].n_total += 1
            fr = None
            with suppress(Exception):
                fr = run(ds.copy())

            if fr is None:
                continue

            k_est, _ = k_from_result(fr)
            if k_est is not None and np.isfinite(k_est):
                results[name].n_success += 1
                results[name].k_errors.append(float(abs(k_est - truth.K)))

            # S0/S1 comparison
            s0_est = s_from_result(fr, "S0")
            s1_est = s_from_result(fr, "S1")
            if s0_est:
                truth_vals = [truth.S0[k] for k in sorted(truth.S0.keys())]
                est_vals = [v for _, v in sorted(s0_est.items())]
                if est_vals and len(est_vals) == len(truth_vals):
                    results[name].s0_errors.extend(
                        [abs(a - b) for a, b in zip(est_vals, truth_vals, strict=False)]
                    )
            if s1_est:
                truth_vals = [truth.S1[k] for k in sorted(truth.S1.keys())]
                est_vals = [v for _, v in sorted(s1_est.items())]
                if est_vals and len(est_vals) == len(truth_vals):
                    results[name].s1_errors.extend(
                        [abs(a - b) for a, b in zip(est_vals, truth_vals, strict=False)]
                    )

    return results


def print_results(results: dict[str, FitterStats], title: str = "") -> None:
    """Print benchmark results."""
    if title:
        print(f"\n{'=' * 70}")
        print(title)
        print("=" * 70)

    print(f"\n{'Fitter':<25} {'Success':>10} {'MAE(K)':>10} {'RMSE(K)':>10} {'Median':>10}")
    print("-" * 70)

    for name, st in sorted(results.items(), key=lambda x: x[1].k_mae):
        print(
            f"{name:<25} {st.success_rate:>9.1f}% {st.k_mae:>10.4f} "
            f"{st.k_rmse:>10.4f} {st.k_median:>10.4f}"
        )

    # S0/S1 errors if available
    has_s_errors = any(st.s0_errors or st.s1_errors for st in results.values())
    if has_s_errors:
        print(f"\n{'Fitter':<25} {'MAE(S0)':>12} {'MAE(S1)':>12}")
        print("-" * 50)
        for name, st in sorted(results.items(), key=lambda x: x[1].k_mae):
            s0_mae = mae(st.s0_errors) if st.s0_errors else float("nan")
            s1_mae = mae(st.s1_errors) if st.s1_errors else float("nan")
            print(f"{name:<25} {s0_mae:>12.2f} {s1_mae:>12.2f}")


def compare_fitters(
    results: dict[str, FitterStats],
    fitter_a: str,
    fitter_b: str,
) -> None:
    """Run statistical comparison between two fitters."""
    if fitter_a not in results or fitter_b not in results:
        print(f"Fitters not found: {fitter_a}, {fitter_b}")
        return

    a = results[fitter_a]
    b = results[fitter_b]

    if not a.k_errors or not b.k_errors:
        print("Insufficient data for comparison")
        return

    # Use Mann-Whitney U test (non-parametric) for K errors
    # since error distributions may not be normal
    stat, p_value = scipy_stats.mannwhitneyu(
        a.k_errors, b.k_errors, alternative="two-sided"
    )

    print(f"\n--- Statistical Comparison: {fitter_a} vs {fitter_b} ---")
    print(f"  {fitter_a}: MAE={a.k_mae:.4f}, N={len(a.k_errors)}")
    print(f"  {fitter_b}: MAE={b.k_mae:.4f}, N={len(b.k_errors)}")
    print(f"  Mann-Whitney U: stat={stat:.1f}, p={p_value:.4f}")

    if p_value < 0.05:
        better = fitter_a if a.k_mae < b.k_mae else fitter_b
        print(f"  → Significant difference (p<0.05): {better} is better")
    else:
        print("  → No significant difference (p≥0.05)")


def main() -> None:
    """Run the benchmark suite."""
    np.set_printoptions(precision=4, suppress=True)

    print("=" * 70)
    print("FITTER BENCHMARK: Synthetic Data with Randomized Signals")
    print("=" * 70)

    # Scenario 1: Clean data (baseline)
    print("\n[1] Clean data (no outliers, standard noise)")
    results_clean = run_benchmark(
        n_repeats=100,
        include_odr=True,
        error_model="realistic",
        rel_error=0.035,
        outlier_prob=0.0,
    )
    print_results(results_clean, "Clean Data Results")

    # Scenario 2: With outliers
    print("\n[2] Data with 10% outliers")
    results_outliers = run_benchmark(
        n_repeats=100,
        include_odr=True,
        error_model="realistic",
        rel_error=0.035,
        outlier_prob=0.1,
    )
    print_results(results_outliers, "With Outliers (10%)")

    # Scenario 3: Differential y1/y2 errors (y1 3x noisier)
    print("\n[3] Differential noise (y1 3x noisier than y2)")
    results_diff = run_benchmark(
        n_repeats=100,
        include_odr=True,
        error_model="realistic",
        rel_error={"y1": 0.07, "y2": 0.025},
        outlier_prob=0.05,
    )
    print_results(results_diff, "Differential Noise (y1 3x)")

    # Statistical comparisons
    print("\n" + "=" * 70)
    print("STATISTICAL COMPARISONS")
    print("=" * 70)

    compare_fitters(results_clean, "glob_ls", "glob_huber")
    compare_fitters(results_outliers, "glob_ls", "outlier2")
    compare_fitters(results_diff, "glob_huber", "outlier2")

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
