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
    results: dict[str, _t.Any],
    heldout_well: str,
    *,
    n_sd: float,
    prior_sigma: float = 0.6,
) -> dict[str, _t.Any]:
    """Return a deepcopy with the heldout well's K prior widened.

    ``fit_binding_pymc_multi`` uses ``p.stderr * n_sd`` as the prior sigma for
    unknown wells.  Increasing ``stderr`` avoids a control-LOO posterior that is
    dominated by a tiny preliminary-fit uncertainty.
    """
    out = copy.deepcopy(results)
    fr = out[heldout_well]
    if fr.result is None or "K" not in fr.result.params:
        return out
    p = fr.result.params["K"]
    current = p.stderr if p.stderr is not None and np.isfinite(p.stderr) else 0.0
    p.stderr = max(float(current), float(prior_sigma) / float(n_sd))
    return out


def _hdi_array(x: np.ndarray, hdi_prob: float) -> tuple[float, float]:
    h = az.hdi(np.asarray(x, dtype=float), hdi_prob=hdi_prob)
    return float(h[0]), float(h[1])


def summarize_bayesian_ctr_holdout(
    trace: _t.Any,
    *,
    trace_id: str,
    ctr_group: str,
    heldout_well: str,
    rope: float = 0.10,
) -> dict[str, _t.Any]:
    """Summarize posterior ``ΔK = K_heldout - K_ctr_group``."""
    posterior = posterior_dataset(trace)
    heldout_var = f"K_{heldout_well}"
    ctr_var = ctr_param_name(ctr_group)

    if heldout_var not in posterior:
        msg = f"Missing heldout variable {heldout_var!r}"
        raise KeyError(msg)
    if ctr_var not in posterior:
        msg = f"Missing control variable {ctr_var!r}"
        raise KeyError(msg)

    diff = (posterior[heldout_var] - posterior[ctr_var]).values.ravel()
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
        "ctr_var": ctr_var,
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
