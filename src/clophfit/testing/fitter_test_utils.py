"""Shared utilities for fitter comparison tests and benchmarks."""

from __future__ import annotations

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
    rel_x_err: float = 0.01,
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
        clean = binding_1site(x, k, s0[lbl], s1[lbl], is_ph=is_ph)
        dy = noise * (np.max(clean) - np.min(clean))
        y = clean + rng.normal(0.0, dy, size=x.shape)
        da = DataArray(xc=x, yc=y)
        # x uncertainty modeling per user guidance
        if is_ph:
            x_err_arr = np.full_like(x, 0.05, dtype=float)
        else:
            x_err_arr = np.where(
                x == 0, 0.01, np.maximum(0.01, rel_x_err * x.astype(float))
            )
        da.x_err = x_err_arr
        ds[lbl] = da
    return ds, Truth(K=k, S0=s0, S1=s1)


def k_from_result(fr: FitResult[MiniT]) -> tuple[float | None, float | None]:
    """Extract K value and stderr from fit result."""
    if fr.result is None or not hasattr(fr.result, "params"):
        return None, None
    params = fr.result.params
    k = params["K"].value if "K" in params else None
    sk = params["K"].stderr if "K" in params else None
    return (float(k) if k is not None else None, float(sk) if sk is not None else None)


def s_from_result(fr: FitResult[MiniT], which: str) -> dict[str, float] | None:
    """Extract S0 or S1 values per label if present in params."""
    if fr.result is None or not hasattr(fr.result, "params"):
        return None
    params = fr.result.params
    out: dict[str, float] = {}
    for key, p in params.items():
        if not key.startswith(which):
            continue
        val = getattr(p, "value", None)
        if val is None:
            continue
        if isinstance(val, (int | float | np.floating)):
            v = float(val)
            if np.isfinite(v):
                out[key] = v
    return out or None


def build_fitters(
    *,
    include_odr: bool = True,
) -> dict[str, Callable[[Dataset], FitResult[MiniT]]]:
    """Builder of fitters."""
    fitters: dict[str, Callable[[Dataset], FitResult[MiniT]]] = {
        "glob_ls": fit_binding_glob,
        "glob_huber": lambda ds: fit_binding_glob(ds, robust=True),
        "glob_irls_outlier": fit_binding_glob_recursive_outlier,
        "outlier2": lambda ds: outlier2(ds, "default"),
    }

    if include_odr:

        def _odr(ds: Dataset) -> FitResult[MiniT]:
            base = fit_binding_glob(ds)
            return fit_binding_odr_recursive_outlier(base)

        fitters["odr_recursive_outlier"] = _odr

    return fitters
