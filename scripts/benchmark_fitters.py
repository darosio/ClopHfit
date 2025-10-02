#!/usr/bin/env python3
"""
Benchmark multiple fitters on synthetic datasets and compare accuracy.

Generates 100 datasets with randomized configuration (is_ph, labels, noise)
and runs all fitters, summarizing K (and S0/S1 when available) accuracy.
"""
from __future__ import annotations

import math
import random
import statistics as stats
from contextlib import suppress

import numpy as np

from clophfit.fitting.data_structures import FitResult, MiniT
from clophfit.testing.fitter_test_utils import Truth, build_fitters, k_from_result, make_synthetic_ds, s_from_result


def mae(values: list[float]) -> float:
    """Compute MAE."""
    return sum(abs(v) for v in values) / len(values) if values else float("nan")


def rmse(values: list[float]) -> float:
    """Compute RMSE."""
    return (
        math.sqrt(sum(v * v for v in values) / len(values)) if values else float("nan")
    )


def main() -> None:
    """Run the simulation."""
    random.seed(123)
    np.set_printoptions(precision=4, suppress=True)

    n_repeats = 100
    fitters = build_fitters(include_odr=True)

    # Collect per-fitter stats
    per_fitter_errors_k: dict[str, list[float]] = {name: [] for name in fitters}
    per_fitter_success: dict[str, int] = dict.fromkeys(fitters, 0)
    per_fitter_total: dict[str, int] = dict.fromkeys(fitters, 0)

    per_fitter_errors_s0: dict[str, list[float]] = {name: [] for name in fitters}
    per_fitter_errors_s1: dict[str, list[float]] = {name: [] for name in fitters}

    for i in range(n_repeats):
        is_ph = bool(random.getrandbits(1))
        labels = random.choice([1, 2])  # noqa: S311
        noise = random.choice([0.0, 0.02, 0.05])  # noqa: S311
        seed = i
        k_true = 7.0 if is_ph else 10.0
        s0 = {
            f"y{j}": (2.0 + 0.2 * j) if is_ph else (1.5 + 0.2 * j)
            for j in range(labels)
        }
        s1 = {f"y{j}": (1.0 + 0.1 * j) if is_ph else (0.1 * j) for j in range(labels)}
        ds, truth = make_synthetic_ds(
            k_true, s0, s1, is_ph=is_ph, noise=noise, seed=seed
        )

        for name, run in fitters.items():
            per_fitter_total[name] += 1
            with suppress(Exception):
                fr = run(ds.copy())
            k_est, _ = k_from_result(fr)
            if k_est is not None and np.isfinite(k_est):
                per_fitter_success[name] += 1
                per_fitter_errors_k[name].append(float(abs(k_est - truth.K)))

            # S0/S1 comparison
            # Map truth keys to parameter keys if needed (S0_y0 etc.)
            # Accept either S0y0 or S0_y0; gather all numeric params and compare by order
            # Build sorted lists by label order to compute an aggregate error
            s0_est = s_from_result(fr, "S0")
            s1_est = s_from_result(fr, "S1")
            if s0_est:
                truth_vals = [truth.S0[k] for k in sorted(truth.S0.keys())]
                est_vals = [v for _, v in sorted(s0_est.items())]
                if est_vals and len(est_vals) == len(truth_vals):
                    per_fitter_errors_s0[name].extend(
                        [abs(a - b) for a, b in zip(est_vals, truth_vals, strict=False)]
                    )
            if s1_est:
                truth_vals = [truth.S1[k] for k in sorted(truth.S1.keys())]
                est_vals = [v for _, v in sorted(s1_est.items())]
                if est_vals and len(est_vals) == len(truth_vals):
                    per_fitter_errors_s1[name].extend(
                        [abs(a - b) for a, b in zip(est_vals, truth_vals, strict=False)]
                    )

    # Print summary
    print("Summary over", n_repeats, "datasets:\n")
    for name in fitters:
        total = per_fitter_total[name]
        succ = per_fitter_success[name]
        k_errs = per_fitter_errors_k[name]
        s0_errs = per_fitter_errors_s0[name]
        s1_errs = per_fitter_errors_s1[name]

        succ_rate = 100.0 * succ / total if total else 0.0
        mae_k = mae(k_errs)
        rmse_k = rmse(k_errs)
        p50_k = stats.median(k_errs) if k_errs else float("nan")

        print(f"- {name}")
        print(f"  success: {succ}/{total} ({succ_rate:.1f}%)")
        print(
            f"  K error:  MAE={mae_k:.4f}  RMSE={rmse_k:.4f}  median={p50_k:.4f}  N={len(k_errs)}"
        )
        if s0_errs:
            print(
                f"  S0 abs error: MAE={mae(s0_errs):.4f}  RMSE={rmse(s0_errs):.4f}  N={len(s0_errs)}"
            )
        else:
            print("  S0 abs error: (no params exposed)")
        if s1_errs:
            print(
                f"  S1 abs error: MAE={mae(s1_errs):.4f}  RMSE={rmse(s1_errs):.4f}  N={len(s1_errs)}"
            )
        else:
            print("  S1 abs error: (no params exposed)")
        print()


if __name__ == "__main__":
    main()
