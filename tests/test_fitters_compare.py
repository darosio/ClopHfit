"""Differential tests comparing multiple fit_binding implementations.

This suite generates synthetic datasets with known ground truth parameters and
compares several fitting backends side-by-side on identical inputs. It asserts
that estimates are close to truth and broadly consistent across methods.
"""

from __future__ import annotations

import numpy as np
import pytest

from clophfit.testing.fitter_test_utils import (
    build_fitters,
    k_from_result,
    make_synthetic_ds,
)


@pytest.mark.parametrize("is_ph", [True, False])
@pytest.mark.parametrize("labels", [1, 2])
@pytest.mark.parametrize("noise", [0.0, 0.02, 0.05])
def test_fitters_converge_and_agree(*, is_ph: bool, labels: int, noise: float) -> None:
    """Test convergence and agreement to truth."""
    # Truth and per-label plateaus
    k_true = 7.0 if is_ph else 10.0
    s0 = {f"y{i}": (2.0 + 0.2 * i) if is_ph else (1.5 + 0.2 * i) for i in range(labels)}
    s1 = {f"y{i}": (1.0 + 0.1 * i) if is_ph else (0.1 * i) for i in range(labels)}
    ds, truth = make_synthetic_ds(
        k_true, s0, s1, is_ph=is_ph, noise=noise, seed=42, rel_x_err=0.03
    )

    fitters = build_fitters(include_odr=False)
    results: dict[str, tuple[float | None, float | None]] = {}
    for name, run in fitters.items():
        fr = run(ds.copy())  # pass a copy to avoid state coupling
        k, sk = k_from_result(fr)
        results[name] = (k, sk)

    # Ensure at least a couple of methods return a value
    available = {k: v for k, v in results.items() if v[0] is not None}
    assert len(available) >= 2, f"Insufficient successful fits: {results}"

    # Tolerances
    if is_ph:
        abs_tol = 0.15 if noise <= 0.02 else 0.25
        # check against truth
        for name, (k, _) in available.items():
            assert k is not None
            assert pytest.approx(truth.K, abs=abs_tol) == k, (
                f"{name}: K deviates from truth (got {k}, want {truth.K})"
            )
    else:
        rel_tol = 0.06 if noise <= 0.02 else 0.12
        for name, (k, _) in available.items():
            assert k is not None
            assert pytest.approx(truth.K, rel=rel_tol) == k, (
                f"{name}: K deviates from truth (got {k}, want {truth.K})"
            )

    # Pairwise agreement
    names = list(available.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            n1, n2 = names[i], names[j]
            k1, s1_ = available[n1]
            k2, s2_ = available[n2]
            assert k1 is not None
            assert k2 is not None
            if s1_ is not None and s2_ is not None and np.isfinite([s1_, s2_]).all():
                the = 3.0 * max(float(s1_), float(s2_))
                assert abs(k1 - k2) <= max(the, 1e-3)
                # fallback to basic tolerance
            elif is_ph:
                assert abs(k1 - k2) <= 0.3
            else:
                assert abs(k1 - k2) <= truth.K * 0.15
