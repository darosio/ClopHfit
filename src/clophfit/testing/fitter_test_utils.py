"""Shared utilities for fitter comparison tests and benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import numpy as np

from clophfit.fitting.bayes import fit_binding_pymc
from clophfit.fitting.core import fit_binding_glob, weight_da, weight_multi_ds_titration
from clophfit.fitting.odr import fit_binding_odr, fit_binding_odr_recursive_outlier
from clophfit.testing.synthetic import TruthParams, make_simple_dataset

if TYPE_CHECKING:
    from collections.abc import Callable

    from clophfit.fitting.data_structures import Dataset, FitResult, MiniT


# Re-export for backwards compatibility
Truth = TruthParams
make_synthetic_ds = make_simple_dataset

TecanFitMethod = Literal["lm", "huber", "irls"]
TecanFinalStage = Literal[
    "lm",
    "huber",
    "irls",
    "odr",
    "mcmc_single",
    "mcmc_multi",
    "mcmc_multi-noise",
    "mcmc_multi-noise-xrw",
]


@dataclass(frozen=True)
class TecanFitCombination:
    """Declarative description of one Tecan fitting workflow."""

    name: str
    channels: tuple[str, ...]
    prefit: TecanFitMethod
    final_stage: TecanFinalStage
    outlier_handling: str | None = None
    weighting: Literal["auto", "none"] = "auto"


def _fit_binding_glob_for_method(
    ds: Dataset, method: TecanFitMethod
) -> FitResult[MiniT]:
    """Run the global fitter with the configured robustification mode."""
    method_map: dict[TecanFitMethod, tuple[str, str | None]] = {
        "lm": ("lm", None),
        "huber": ("huber", None),
        "irls": ("lm", "irls"),
    }
    fit_method, reweight = method_map[method]
    return fit_binding_glob(ds, method=fit_method, reweight=reweight)


def build_tecan_fit_combinations(
    *,
    base_method: TecanFitMethod = "huber",
    include_odr: bool = True,
    include_mcmc: bool = False,
    mcmc_modes: tuple[str, ...] = ("single",),
) -> dict[str, TecanFitCombination]:
    """Build named Tecan fit combinations for paired benchmark comparisons."""
    combinations = {
        "y1_huber": TecanFitCombination(
            name="y1_huber",
            channels=("y1",),
            prefit=base_method,
            final_stage=base_method,
        ),
        "y2_huber": TecanFitCombination(
            name="y2_huber",
            channels=("y2",),
            prefit=base_method,
            final_stage=base_method,
        ),
        "y1y2_huber": TecanFitCombination(
            name="y1y2_huber",
            channels=("y1", "y2"),
            prefit=base_method,
            final_stage=base_method,
        ),
    }
    if include_odr:
        combinations["y1y2_odr_from_huber"] = TecanFitCombination(
            name="y1y2_odr_from_huber",
            channels=("y1", "y2"),
            prefit=base_method,
            final_stage="odr",
        )
    if include_mcmc:
        stage_map: dict[str, TecanFinalStage] = {
            "single": "mcmc_single",
            "multi": "mcmc_multi",
            "multi-noise": "mcmc_multi-noise",
            "multi-noise-xrw": "mcmc_multi-noise-xrw",
        }
        for mode in mcmc_modes:
            final_stage = stage_map[mode]
            combinations[f"y1y2_{final_stage}_from_huber"] = TecanFitCombination(
                name=f"y1y2_{final_stage}_from_huber",
                channels=("y1", "y2"),
                prefit=base_method,
                final_stage=final_stage,
            )
    return combinations


def apply_tecan_combination(
    ds: Dataset, combination: TecanFitCombination
) -> FitResult[MiniT]:
    """Execute one Tecan fit combination on a fresh dataset copy."""
    work_ds = ds.copy(keys=list(combination.channels))
    if combination.weighting == "auto":
        if len(work_ds) == 1:
            da = next(iter(work_ds.values()))
            weight_da(da, is_ph=work_ds.is_ph)
        elif len(work_ds) > 1:
            weight_multi_ds_titration(work_ds)

    prefit_result = _fit_binding_glob_for_method(work_ds, combination.prefit)
    final_stage = combination.final_stage
    if final_stage in {"lm", "huber", "irls"}:
        return _fit_binding_glob_for_method(
            work_ds, cast("TecanFitMethod", final_stage)
        )
    if final_stage == "odr":
        return fit_binding_odr(prefit_result)
    if final_stage == "mcmc_single":
        return fit_binding_pymc(prefit_result, n_sd=5.0, n_xerr=1.0, n_samples=200)
    msg = (
        f"Final stage '{final_stage}' requires full TitrationAnalysis context and "
        "is not supported by dataset-only benchmark utilities."
    )
    raise NotImplementedError(msg)


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
