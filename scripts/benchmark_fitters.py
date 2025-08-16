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
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from clophfit.fitting.core import (
    fit_binding_glob,
    fit_binding_glob_recursive_outlier,
    outlier2,
)
from clophfit.fitting.data_structures import DataArray, Dataset, FitResult, MiniT
from clophfit.fitting.models import binding_1site
from clophfit.fitting.odr import fit_binding_odr_recursive_outlier

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class Truth:
    """True parameters."""

    K: float
    S0: dict[str, float]
    S1: dict[str, float]


def make_synthetic_ds(  # noqa: PLR0913
    k: float,
    s0: dict[str, float] | float,
    s1: dict[str, float] | float,
    *,
    is_ph: bool,
    noise: float = 0.02,
    seed: int = 0,
) -> tuple[Dataset, Truth]:
    """Create a synthetic Dataset with optional Gaussian noise.

    noise is relative to dynamic range per label.
    Accepts scalar s0/s1 for convenience (single label).
    """
    if not isinstance(s0, dict):
        s0 = {"y0": float(s0)}
    if not isinstance(s1, dict):
        s1 = {"y0": float(s1)}
    is_ph = bool(is_ph)

    rng = np.random.default_rng(seed)
    if is_ph:
        x = np.array([5, 5.8, 6.6, 7.0, 7.8, 8.2, 9.0])
    else:
        x = np.array([0.01, 5, 10, 20, 40, 80, 150])

    ds = Dataset({}, is_ph=is_ph)
    for lbl in sorted(s0.keys()):
        clean = binding_1site(x, k, s0[lbl], s1[lbl], is_ph)
        dy = noise * (np.max(clean) - np.min(clean))
        y = clean + rng.normal(0.0, dy, size=x.shape)
        da = DataArray(xc=x, yc=y)
        # x uncertainty modeling per user guidance:
        # - at x == 0: absolute 0.01
        # - at x > 0: increased uncertainty reflecting serial additions (use relative component)
        if is_ph:
            x_err_arr = np.full_like(x, 0.05, dtype=float)
        else:
            rel = 0.01  # 3% relative uncertainty for concentrations
            x_err_arr = np.where(x == 0, 0.01, np.maximum(0.01, rel * x.astype(float)))
        da.x_err = x_err_arr
        ds[lbl] = da
    return ds, Truth(K=k, S0=s0, S1=s1)


def k_from_result(fr: FitResult[MiniT]) -> tuple[float | None, float | None]:
    """Resume helper."""
    if fr.result is None or not hasattr(fr.result, "params"):
        return None, None
    params = fr.result.params
    k = params["K"].value if "K" in params else None
    sk = params["K"].stderr if "K" in params else None
    return (float(k) if k is not None else None, float(sk) if sk is not None else None)


def s_from_result(fr: FitResult[MiniT], which: str) -> dict[str, float] | None:
    """Extract S0 or S1 values per label if present in params.

    Avoids broad exception catching by checking attribute presence and types explicitly.
    """
    if fr.result is None or not hasattr(fr.result, "params"):
        return None
    params = fr.result.params
    out: dict[str, float] = {}
    for key, p in params.items():
        if not key.startswith(which):
            continue
        # Safely access numeric value
        val = getattr(p, "value", None)
        if val is None:
            continue
        # Accept native floats/ints and numpy floating scalars
        if isinstance(val, (int | float | np.floating)):
            v = float(val)
            if np.isfinite(v):
                out[key] = v
    return out or None


def build_fitters() -> dict[str, Callable[[Dataset], FitResult[MiniT]]]:
    """Builder of fitters."""

    def _odr(ds: Dataset) -> FitResult[MiniT]:
        base = fit_binding_glob(ds)
        return fit_binding_odr_recursive_outlier(base)

    fitters: dict[str, Callable[[Dataset], FitResult[MiniT]]] = {
        "glob_ls": lambda ds: fit_binding_glob(ds),
        "glob_huber": lambda ds: fit_binding_glob(ds, robust=True),
        "glob_irls_outlier": lambda ds: fit_binding_glob_recursive_outlier(ds),
        "outlier2": lambda ds: outlier2(ds, "default"),
        "odr_recursive_outlier": _odr,
    }
    return fitters


def mae(values: list[float]) -> float:
    """Compute MAE."""
    return sum(abs(v) for v in values) / len(values) if values else float("nan")


def rmse(values: list[float]) -> float:
    """Compute RMSE."""
    return (
        math.sqrt(sum(v * v for v in values) / len(values)) if values else float("nan")
    )


def main() -> None:  # noqa: PLR0915
    """Run the simulation."""
    random.seed(123)
    np.set_printoptions(precision=4, suppress=True)

    n_repeats = 100
    fitters = build_fitters()

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
            # optional S0/S1 comparison (if params exposed)
            s0_est = s_from_result(fr, "S0")
            s1_est = s_from_result(fr, "S1")
            if s0_est:
                # Map truth keys to parameter keys if needed (S0_y0 etc.)
                # Accept either S0y0 or S0_y0; gather all numeric params and compare by order
                # Build sorted lists by label order to compute an aggregate error
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
