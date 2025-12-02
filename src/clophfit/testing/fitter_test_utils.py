"""Shared utilities for fitter comparison tests and benchmarks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from clophfit.fitting.core import (
    fit_binding_glob,
    fit_binding_glob_recursive_outlier,
    outlier2,
)
from clophfit.fitting.odr import fit_binding_odr_recursive_outlier
from clophfit.testing.synthetic import TruthParams, make_simple_dataset

if TYPE_CHECKING:
    from collections.abc import Callable

    from clophfit.fitting.data_structures import Dataset, FitResult, MiniT


# Re-export for backwards compatibility
Truth = TruthParams
make_synthetic_ds = make_simple_dataset


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
        "outlier2": lambda ds: outlier2(ds, key="default"),
    }

    if include_odr:

        def _odr(ds: Dataset) -> FitResult[MiniT]:
            base = fit_binding_glob(ds)
            return fit_binding_odr_recursive_outlier(base)

        fitters["odr_recursive_outlier"] = _odr

    return fitters
