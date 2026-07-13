"""Control holdout / leave-one-control-out validation helpers."""

from __future__ import annotations

import copy
import typing as _t

import arviz as az  # type: ignore[import-untyped]
import numpy as np
import pandas as pd

from .model_validation import posterior_dataset


def ctr_param_name(group_name: str) -> str:
    """Return the shared-control K parameter name used by Bayesian multi-fit."""
    return f"K_ctr_{group_name}"


def free_ctr_param_name(group_name: str, well: str) -> str:
    """Return the free-control K parameter name used by Bayesian multi-fit."""
    return f"K_{group_name}_{well}"


def _well_k_draws(
    posterior: _t.Any, well: str, *, scalar_name: str
) -> tuple[np.ndarray, str] | None:
    """Return ``(draws, resolved_name)`` for a well's posterior K, or ``None``.

    Resolves K from either the legacy scalar variable (``scalar_name``, e.g.
    ``K_{well}`` or ``K_{group}_{well}``) or the vectorized ``K_free`` variable
    indexed by the ``free_well`` coordinate produced by the multi-well fit.
    Returns ``None`` when the well is present in neither.
    """
    if scalar_name in posterior:
        return np.asarray(posterior[scalar_name].values), scalar_name
    if "K_free" in posterior:
        k_free = posterior["K_free"]
        coord = k_free.coords.get("free_well")
        if coord is not None and well in {str(w) for w in coord.values}:
            return np.asarray(k_free.sel(free_well=well).values), f"K_free[{well}]"
    return None


def make_ctr_holdout_scheme(
    scheme: _t.Any, *, group_name: str, heldout_well: str
) -> _t.Any:
    """Return a PlateScheme copy with one control well removed.

    ``PlateScheme.names`` validates strictly as ``dict[str, set[str]]`` in
    ClopHfit, so this helper preserves that type.
    """
    new_scheme = copy.deepcopy(scheme)
    target_group = str(group_name)
    target_well = str(heldout_well)

    new_names: dict[str, set[str]] = {}
    for name, wells in scheme.names.items():
        name_s = str(name)
        kept = {
            str(w)
            for w in wells
            if not (name_s == target_group and str(w) == target_well)
        }
        if kept:
            new_names[name_s] = kept
    new_scheme.names = new_names
    return new_scheme


def iter_ctr_holdouts(
    scheme: _t.Any, *, min_remaining: int = 1
) -> _t.Iterator[dict[str, _t.Any]]:
    """Yield holdout tasks from all named control groups."""
    for group_name, wells in scheme.names.items():
        group_name = str(group_name)
        wells = sorted(str(w) for w in wells)
        if len(wells) <= min_remaining:
            continue
        for heldout_well in wells:
            remaining = [w for w in wells if w != heldout_well]
            yield {
                "ctr_group": group_name,
                "heldout_well": heldout_well,
                "n_remaining_ctr": len(remaining),
                "remaining_ctr_wells": remaining,
            }


def widen_heldout_k_prior(
    results: _t.Any,
    heldout_well: str,
    *,
    n_sd: float,
    prior_sigma: float = 0.6,
) -> _t.Any:
    """Return a deepcopy with the heldout well's K prior widened.

    ``fit_binding_pymc_multi`` uses ``p.stderr * n_sd`` as the prior sigma for
    unknown wells.  Increasing ``stderr`` avoids a control-LOO posterior that is
    dominated by a tiny preliminary-fit uncertainty.
    """
    try:
        fr = results[heldout_well]
    except (KeyError, TypeError, AttributeError):
        return results
    if not hasattr(fr, "result"):
        return results

    out = copy.deepcopy(results)
    fr = out[heldout_well]
    if fr.result is None or "K" not in fr.result.params:
        return out
    p = fr.result.params["K"]
    current = p.stderr if p.stderr is not None and np.isfinite(p.stderr) else 0.0
    p.stderr = max(float(current), float(prior_sigma) / float(n_sd))
    return out


def _hdi_array(x: np.ndarray, hdi_prob: float) -> tuple[float, float]:
    h = az.hdi(np.asarray(x, dtype=float), prob=hdi_prob)
    return float(h[0]), float(h[1])


def weighted_mean_reference(arrays: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Return an inverse-variance weighted posterior reference per draw."""
    flat = [np.asarray(arr, dtype=float).ravel() for arr in arrays]
    if not flat:
        msg = "No arrays provided for weighted reference."
        raise ValueError(msg)
    variances = np.array([np.nanvar(arr, ddof=1) for arr in flat], dtype=float)
    exact = np.isfinite(variances) & (variances == 0)
    if exact.any():
        weights = exact.astype(float)
    else:
        weights = np.divide(
            1.0,
            variances,
            out=np.full_like(variances, np.nan, dtype=float),
            where=np.isfinite(variances) & (variances > 0),
        )
        if not np.isfinite(weights).any() or float(np.nansum(weights)) <= 0:
            weights = np.ones(len(flat), dtype=float)
    weights /= float(np.nansum(weights))
    stacked = np.vstack(flat)
    return np.nansum(stacked * weights[:, None], axis=0), weights


def summarize_bayesian_ctr_holdout(  # noqa: PLR0913
    trace: _t.Any,
    *,
    trace_id: str,
    ctr_group: str,
    heldout_well: str,
    remaining_ctr_wells: list[str] | None = None,
    reference_mode: str = "shared",
    rope: float = 0.10,
) -> dict[str, _t.Any]:
    """Summarize posterior CTR holdout ``ΔK``.

    ``reference_mode="shared"`` compares heldout K to ``K_ctr_{group}``.
    ``reference_mode="weighted_mean"`` compares it to the inverse-variance
    weighted posterior mean of the remaining free-control K variables.
    """
    posterior = posterior_dataset(trace)
    heldout = _well_k_draws(posterior, heldout_well, scalar_name=f"K_{heldout_well}")
    if heldout is None:
        msg = f"Missing heldout variable {f'K_{heldout_well}'!r}"
        raise KeyError(msg)
    heldout_draws, heldout_var = heldout

    if reference_mode == "shared":
        reference_vars = [ctr_param_name(ctr_group)]
        if reference_vars[0] not in posterior:
            msg = f"Missing control variable {reference_vars[0]!r}"
            raise KeyError(msg)
        reference = posterior[reference_vars[0]].values.ravel()
        reference_weights = np.array([1.0])
    elif reference_mode == "weighted_mean":
        if remaining_ctr_wells is None:
            msg = "remaining_ctr_wells is required for weighted_mean reference."
            raise ValueError(msg)
        resolved = [
            (well, _well_k_draws(posterior, well, scalar_name=name))
            for well in remaining_ctr_wells
            for name in [free_ctr_param_name(ctr_group, well)]
        ]
        missing = [
            free_ctr_param_name(ctr_group, well)
            for well, draws in resolved
            if draws is None
        ]
        if missing:
            msg = f"Missing free-control variables {missing!r}"
            raise KeyError(msg)
        reference_arrays = [draws[0] for _, draws in resolved if draws is not None]
        reference_vars = [draws[1] for _, draws in resolved if draws is not None]
        reference, reference_weights = weighted_mean_reference(reference_arrays)
    else:
        msg = f"Unsupported CTR-LOO reference mode: {reference_mode!r}"
        raise ValueError(msg)

    diff = heldout_draws.ravel() - reference
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        msg = "No finite posterior ΔK draws."
        raise ValueError(msg)

    lo89, hi89 = _hdi_array(diff, 0.89)
    lo94, hi94 = _hdi_array(diff, 0.94)
    mean_diff = float(np.mean(diff))
    sd_diff = float(np.std(diff, ddof=1))

    return {
        "trace_id": trace_id,
        "ctr_group": ctr_group,
        "heldout_well": heldout_well,
        "heldout_var": heldout_var,
        "ctr_reference_mode": reference_mode,
        "ctr_reference_vars": ",".join(reference_vars),
        "ctr_reference_weights": ",".join(f"{w:.6g}" for w in reference_weights),
        "ctr_reference_n": len(reference_vars),
        "delta_k_mean": mean_diff,
        "delta_k_sd": sd_diff,
        "delta_k_abs_mean": abs(mean_diff),
        "delta_k_hdi89_low": lo89,
        "delta_k_hdi89_high": hi89,
        "delta_k_hdi94_low": lo94,
        "delta_k_hdi94_high": hi94,
        "delta_k_hdi89_contains_zero": bool(lo89 <= 0 <= hi89),
        "delta_k_hdi94_contains_zero": bool(lo94 <= 0 <= hi94),
        "p_abs_delta_k_lt_rope": float(np.mean(np.abs(diff) < rope)),
        "rope": rope,
        "z_delta_k": float(mean_diff / sd_diff) if sd_diff > 0 else np.nan,
    }


def summarize_ctr_loo_table(ctr_loo_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse individual holdout rows into one row per model condition."""
    return (
        ctr_loo_df
        .groupby("trace_id", observed=True)
        .agg(
            ctr_loo_n=("delta_k_mean", "size"),
            ctr_loo_bias_mean=("delta_k_mean", "mean"),
            ctr_loo_mae=("delta_k_abs_mean", "mean"),
            ctr_loo_rmse=(
                "delta_k_mean",
                lambda x: float(np.sqrt(np.mean(np.asarray(x) ** 2))),
            ),
            ctr_loo_max_abs_error=("delta_k_abs_mean", "max"),
            ctr_loo_mean_sd=("delta_k_sd", "mean"),
            ctr_loo_mean_abs_z=("z_delta_k", lambda x: float(np.nanmean(np.abs(x)))),
            ctr_loo_hdi89_coverage=("delta_k_hdi89_contains_zero", "mean"),
            ctr_loo_hdi94_coverage=("delta_k_hdi94_contains_zero", "mean"),
            ctr_loo_mean_p_rope=("p_abs_delta_k_lt_rope", "mean"),
        )
        .reset_index()
    )


def classical_ctr_holdout_rows(
    results: dict[str, _t.Any],
    scheme: _t.Any,
    *,
    trace_id: str,
    rope: float = 0.10,
) -> pd.DataFrame:
    """Post-hoc CTR holdout table for classical fits.

    For each control well, compare its fitted K to the inverse-variance weighted
    mean K of the remaining control wells in the same group.
    """
    rows: list[dict[str, _t.Any]] = []
    for task in iter_ctr_holdouts(scheme, min_remaining=1):
        group = task["ctr_group"]
        heldout = task["heldout_well"]
        remaining = task["remaining_ctr_wells"]

        if heldout not in results or results[heldout].result is None:
            continue
        k_h = results[heldout].result.params["K"].value
        se_h = results[heldout].result.params["K"].stderr

        vals = []
        ses = []
        for well in remaining:
            fr = results.get(well)
            if fr is None or fr.result is None or "K" not in fr.result.params:
                continue
            p = fr.result.params["K"]
            if p.stderr is None or not np.isfinite(p.stderr) or p.stderr <= 0:
                continue
            vals.append(float(p.value))
            ses.append(float(p.stderr))
        if not vals:
            continue
        vals_arr = np.asarray(vals, dtype=float)
        ses_arr = np.asarray(ses, dtype=float)
        weights = 1.0 / ses_arr**2
        k_ctr = float(np.average(vals_arr, weights=weights))
        se_ctr = float(np.sqrt(1.0 / weights.sum()))
        delta = float(k_h - k_ctr)
        se_delta = float(np.sqrt((se_h or 0.0) ** 2 + se_ctr**2))
        rows.append({
            "trace_id": trace_id,
            "ctr_group": group,
            "heldout_well": heldout,
            "remaining_ctr_wells": ",".join(remaining),
            "heldout_k": float(k_h),
            "remaining_k": k_ctr,
            "delta_k_mean": delta,
            "delta_k_sd": se_delta,
            "delta_k_abs_mean": abs(delta),
            "p_abs_delta_k_lt_rope": float(abs(delta) < rope),
            "rope": rope,
            "z_delta_k": float(delta / se_delta) if se_delta > 0 else np.nan,
        })
    return pd.DataFrame(rows)
