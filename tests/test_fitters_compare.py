"""Differential tests comparing multiple fit_binding implementations.

This suite generates synthetic datasets with known ground truth parameters and
compares several fitting backends side-by-side on identical inputs. It asserts
that estimates are close to truth and broadly consistent across methods.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass

import numpy as np
import pytest

from clophfit.fitting.core import (
    fit_binding_glob,
    fit_binding_glob_recursive_outlier,
    outlier2,
)
from clophfit.fitting.data_structures import DataArray, Dataset, FitResult, MiniT
from clophfit.fitting.models import binding_1site
from clophfit.fitting.odr import fit_binding_odr_recursive_outlier

if typing.TYPE_CHECKING:
    from collections.abc import Callable

    from scipy import odr


@dataclass
class Truth:
    """True parameters."""

    K: float
    S0: dict[str, float]
    S1: dict[str, float]


def _make_synthetic_ds(  # noqa: PLR0913
    k: float,
    s0: dict[str, float],
    s1: dict[str, float],
    *,
    is_ph: bool,
    noise: float = 0.02,
    seed: int = 0,
) -> tuple[Dataset, Truth]:
    """Create a synthetic Dataset with optional Gaussian noise.

    noise is relative to dynamic range per label.
    """
    rng = np.random.default_rng(seed)
    if is_ph:
        # cover a reasonable pH range around K
        x = np.array([5, 5.8, 6.6, 7.0, 7.8, 8.2, 9.0])
    else:
        x = np.array([0.01, 5, 10, 20, 40, 80, 150])  # geomspace(0.1, 100.0, n)

    ds = Dataset({}, is_ph=is_ph)
    for lbl in sorted(s0.keys()):
        clean = binding_1site(x, k, s0[lbl], s1[lbl], is_ph)
        dy = noise * (np.max(clean) - np.min(clean))
        y = clean + rng.normal(0.0, dy, size=x.shape)
        da = DataArray(xc=x, yc=y)
        # small x error for ODR paths
        x_err = 0.05 if is_ph else 0.0
        da.x_err = np.ones_like(y) * x_err
        ds[lbl] = da
    return ds, Truth(K=k, S0=s0, S1=s1)


def _k_from_result(fr: FitResult[MiniT]) -> tuple[float | None, float | None]:
    if fr.result is None or not hasattr(fr.result, "params"):
        return None, None
    params = fr.result.params
    k = params["K"].value if "K" in params else None
    sk = params["K"].stderr if "K" in params else None
    return float(k) if k is not None else None, (float(sk) if sk is not None else None)


def _build_fitters() -> dict[str, Callable[[Dataset], FitResult[MiniT]]]:
    """Adapters that normalize various fitting backends."""

    def _odr(ds: Dataset) -> FitResult[odr.Output]:
        # ODR path expects an initial LS fit
        base = fit_binding_glob(ds)
        return fit_binding_odr_recursive_outlier(base)

    return {
        "glob_ls": lambda ds: fit_binding_glob(ds),
        "glob_huber": lambda ds: fit_binding_glob(ds, robust=True),
        "glob_irls_outlier": lambda ds: fit_binding_glob_recursive_outlier(ds),
        "outlier2": lambda ds: outlier2(ds, "default"),
        "odr_recursive_outlier": _odr,
    }


@pytest.mark.parametrize("is_ph", [True, False])
@pytest.mark.parametrize("labels", [1, 2])
@pytest.mark.parametrize("noise", [0.0, 0.02, 0.05])
def test_fitters_converge_and_agree(is_ph: bool, labels: int, noise: float) -> None:
    """Test convergence and agreement to truth."""
    # Truth and per-label plateaus
    k_true = 7.0 if is_ph else 10.0
    s0 = {f"y{i}": (2.0 + 0.2 * i) if is_ph else (1.5 + 0.2 * i) for i in range(labels)}
    s1 = {f"y{i}": (1.0 + 0.1 * i) if is_ph else (0.1 * i) for i in range(labels)}
    ds, truth = _make_synthetic_ds(k_true, s0, s1, is_ph=is_ph, noise=noise, seed=42)

    fitters = _build_fitters()
    results: dict[str, tuple[float | None, float | None]] = {}
    for name, run in fitters.items():
        fr = run(ds.copy())  # pass a copy to avoid state coupling
        k, sk = _k_from_result(fr)
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

    # Pairwise agreement: within 3 max reported stderrs or within tolerance
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
