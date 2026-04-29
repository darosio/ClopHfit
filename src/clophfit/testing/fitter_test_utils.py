"""Shared utilities for fitter comparison tests and benchmarks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from clophfit.fitting.core import fit_binding_glob
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
    """Build dictionary of fitting methods for benchmarking.

    Returns a registry of named fitters using the unified ``fit_binding_glob``
    API with different method/reweight/remove_outliers combinations.

    Parameters
    ----------
    include_odr : bool
        Whether to include ODR-based fitters (requires odrpack).

    Returns
    -------
    dict[str, Callable[[Dataset], FitResult[MiniT]]]
        Named fitters mapping.
    """
    fitters: dict[str, Callable[[Dataset], FitResult[MiniT]]] = {
        # --- Standard WLS ---
        "glob_ls": fit_binding_glob,
        # --- Huber robust ---
        "glob_huber": lambda ds: fit_binding_glob(ds, method="huber"),
        # --- Huber + outlier removal ---
        "glob_huber_outlier": lambda ds: fit_binding_glob(
            ds, method="huber", remove_outliers="zscore:2.5:5"
        ),
        # --- IRLS reweighting ---
        "glob_irls": lambda ds: fit_binding_glob(ds, reweight="irls"),
        # --- Iterative reweighting ---
        "glob_iterative": lambda ds: fit_binding_glob(ds, reweight="iterative"),
        # --- Iterative + outlier removal ---
        "glob_iterative_outlier": lambda ds: fit_binding_glob(
            ds, reweight="iterative", remove_outliers="zscore:3.0:5"
        ),
    }

    if include_odr:

        def _odr(ds: Dataset) -> FitResult[MiniT]:
            base = fit_binding_glob(ds)
            return fit_binding_odr_recursive_outlier(base)

        fitters["odr_recursive_outlier"] = _odr

    return fitters
