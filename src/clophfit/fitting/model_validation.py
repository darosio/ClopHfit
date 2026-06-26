"""Reusable model-validation helpers for ClopHfit fitting workflows.

These utilities are designed to live in :mod:`clophfit.fitting` and be reused by
both package tests and manuscript-analysis scripts.  They intentionally avoid any
manuscript-specific paths, file formats, or plate names.
"""

from __future__ import annotations

import copy
import itertools
import typing as _t
import warnings

import arviz as az  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats as sp_stats

ArrayLike = _t.Any

STUDENT_T_NU = 3.0


def residual_normal_scores(
    likelihood_residual: ArrayLike,
    *,
    robust: bool = False,
    student_t_nu: float = STUDENT_T_NU,
) -> np.ndarray:
    """Map likelihood-scale residuals onto a Normal diagnostic scale.

    For Normal likelihoods this is the identity.  For Student-t likelihoods,
    ``(y - mu) / sigma`` follows a t distribution, so Normal QQ plots and
    ``abs(residual) > 2`` style diagnostics should use the probability integral
    transform to an equivalent standard-Normal score.
    """
    r = np.asarray(likelihood_residual, dtype=float)
    if not robust:
        return r

    probs = sp_stats.t.cdf(r, df=student_t_nu)
    probs = np.clip(probs, np.finfo(float).eps, 1.0 - np.finfo(float).eps)
    return np.asarray(sp_stats.norm.ppf(probs), dtype=float)


def robust_residual_outlier_mask(
    likelihood_residual: ArrayLike,
    *,
    threshold: float = 3.0,
    robust: bool = False,
    student_t_nu: float = STUDENT_T_NU,
) -> np.ndarray:
    """Flag observations by calibrated Normal-score residual magnitude."""
    z = residual_normal_scores(
        likelihood_residual, robust=robust, student_t_nu=student_t_nu
    )
    return np.asarray(np.isfinite(z) & (np.abs(z) > threshold), dtype=bool)


def excess_tail_outlier_mask(  # noqa: PLR0913
    likelihood_residual: ArrayLike,
    *,
    threshold: float = 3.0,
    allowed_tail_fraction: float = 0.01,
    min_allowed_tail_count: int = 1,
    robust: bool = False,
    student_t_nu: float = STUDENT_T_NU,
) -> np.ndarray:
    """Mask only residual outliers beyond an allowed tail fraction.

    The residuals are first mapped to the calibrated Normal diagnostic scale.
    Observations with ``abs(z) <= threshold`` are never removed.  If more than
    ``allowed_tail_fraction`` of finite observations exceed the threshold, only
    the largest excess observations are marked for removal.
    """
    z = residual_normal_scores(
        likelihood_residual, robust=robust, student_t_nu=student_t_nu
    )
    z = np.asarray(z, dtype=float)
    remove = np.zeros(z.shape, dtype=bool)
    finite = np.isfinite(z)
    n_finite = int(finite.sum())
    if n_finite == 0:
        return remove

    candidate_idx = np.flatnonzero(finite & (np.abs(z) > threshold))
    allowed = max(
        int(min_allowed_tail_count), int(np.floor(allowed_tail_fraction * n_finite))
    )
    n_remove = max(0, int(candidate_idx.size) - allowed)
    if n_remove == 0:
        return remove

    order = np.argsort(np.abs(z[candidate_idx]))[::-1]
    remove[candidate_idx[order[:n_remove]]] = True
    return remove


def mark_excess_residual_outliers(  # noqa: PLR0913
    residuals: pd.DataFrame,
    *,
    residual_col: str = "std_res",
    group_cols: tuple[str, ...] = ("trace_id", "label"),
    threshold: float = 3.0,
    allowed_tail_fraction: float = 0.01,
    min_allowed_tail_count: int = 1,
    exclude_col: str = "exclude_residual_outlier",
) -> pd.DataFrame:
    """Annotate residual rows to remove only excess calibrated tail outliers."""
    out = residuals.copy()
    out[exclude_col] = False
    out["residual_outlier_score"] = np.nan

    actual_group_cols = [col for col in group_cols if col in out.columns]
    grouped: _t.Iterable[tuple[object, pd.DataFrame]]
    if actual_group_cols:
        grouped = out.groupby(actual_group_cols, observed=True, sort=False)
    else:
        grouped = [(None, out)]

    for _key, group in grouped:
        values = group[residual_col].to_numpy(dtype=float)
        remove = excess_tail_outlier_mask(
            values,
            threshold=threshold,
            allowed_tail_fraction=allowed_tail_fraction,
            min_allowed_tail_count=min_allowed_tail_count,
        )
        out.loc[group.index, "residual_outlier_score"] = np.abs(values)
        out.loc[group.index[remove], exclude_col] = True

    return out


def masked_datasets_from_residual_outliers(
    results: _t.Mapping[str, _t.Any],
    residuals: pd.DataFrame,
    *,
    exclude_col: str = "exclude_residual_outlier",
    min_keep: int = 3,
) -> dict[str, _t.Any]:
    """Return datasets with residual rows marked by *exclude_col* masked out.

    This is intended for the second pass of a sensitivity analysis: fit once,
    compute residuals, annotate excess-tail outliers, mask those rows, then refit.
    """
    masked: dict[str, _t.Any] = {}
    for well, fr in results.items():
        if getattr(fr, "dataset", None) is not None:
            masked[str(well)] = copy.deepcopy(fr.dataset)

    if not masked or exclude_col not in residuals.columns:
        return masked

    drop_rows = residuals[residuals[exclude_col].astype(bool)].copy()
    if drop_rows.empty:
        return masked

    if "residual_outlier_score" not in drop_rows.columns:
        drop_rows["residual_outlier_score"] = np.abs(
            drop_rows.get("std_res", pd.Series(np.nan, index=drop_rows.index))
        )
    drop_rows = drop_rows.sort_values("residual_outlier_score", ascending=False)

    for row in drop_rows.itertuples(index=False):
        well = str(row.well)
        label = str(row.label)
        if well not in masked or label not in masked[well]:
            continue

        da = masked[well][label]
        if int(np.sum(da.mask)) <= min_keep:
            continue

        raw_i = getattr(row, "raw_i", None)
        if raw_i is None or pd.isna(raw_i):
            raw_i = getattr(row, "step", None)
        if raw_i is None or pd.isna(raw_i):
            continue

        idx = int(raw_i)
        if 0 <= idx < len(da.mask):
            da.mask[idx] = False

    return masked


def posterior_dataset(trace: _t.Any) -> _t.Any:
    """Return the posterior xarray Dataset from ArviZ InferenceData or DataTree.

    PyMC/ArviZ versions differ in whether returned objects are InferenceData-like
    or xarray DataTree-like.  This helper hides that difference for validation code.
    """
    if hasattr(trace, "posterior"):
        return trace.posterior
    node = trace["posterior"]
    return getattr(node, "ds", node)


def sample_stats_dataset(trace: _t.Any) -> _t.Any:
    """Return sample_stats Dataset from InferenceData or DataTree."""
    if hasattr(trace, "sample_stats"):
        return trace.sample_stats
    node = trace["sample_stats"]
    return getattr(node, "ds", node)


def _posterior_mean_scalar(
    posterior: _t.Any, var_name: str, default: float = 1.0
) -> float:
    if var_name not in posterior:
        return default
    return float(posterior[var_name].mean(("chain", "draw")).values)


def x_axis_sanity(trace: _t.Any) -> dict[str, _t.Any]:
    """Check pH/x-axis invariants for traces with ``x_per_well``.

    For per-well x models with a shared start pH, all wells at step 0 should be
    identical within each draw.  ``x_step0_max_abs_spread`` should therefore be
    close to zero.
    """
    out: dict[str, _t.Any] = {}
    try:
        posterior = posterior_dataset(trace)
        if "x_per_well" not in posterior:
            return out
        x = posterior["x_per_well"]
        if "step" not in x.dims or "well" not in x.dims:
            out["x_sanity_error"] = (
                f"x_per_well dims are {x.dims!r}, expected step/well"
            )
            return out
        x0 = x.isel(step=0)
        x0_ref = x0.isel(well=0)
        out["x_step0_max_abs_spread"] = float(np.abs(x0 - x0_ref).max().values)
        if "well" in x.coords:
            first_well = str(x.coords["well"].values[0])
            traj = x.sel(well=first_well).mean(("chain", "draw")).values
            out["x_first_well"] = first_well
            out["x_first_well_step0_mean"] = float(traj[0])
            out["x_first_well_last_mean"] = float(traj[-1])
    except Exception as e:
        out["x_sanity_error"] = repr(e)
    return out


def trace_diagnostics(
    trace: _t.Any,
    *,
    compute_loo: bool = False,
    summary_var_names: list[str] | None = None,
) -> dict[str, _t.Any]:
    """Collect basic MCMC and optional LOO diagnostics from a PyMC trace."""
    row: dict[str, _t.Any] = {}

    try:
        ss = sample_stats_dataset(trace)
        if "diverging" in ss:
            row["n_divergences"] = int(ss["diverging"].sum().values)
        if "tree_depth" in ss:
            row["tree_depth_max"] = int(ss["tree_depth"].max().values)
        if "reached_max_treedepth" in ss:
            row["n_reached_max_treedepth"] = int(
                ss["reached_max_treedepth"].sum().values
            )
        if "energy" in ss:
            energy = ss["energy"].values.ravel()
            row["energy_sd"] = float(np.nanstd(energy))
    except Exception as e:
        row["sample_stats_error"] = repr(e)

    if summary_var_names is None:
        summary_var_names = [
            "K",
            "K_ctr",
            "x_start",
            "x_true",
            "x_per_well",
            "acid_drop_global",
            "acid_drop_well",
            "x_step",
            "ye_mag",
            "rel_error",
            "floor",
            "gain",
        ]

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            summary = az.summary(trace, var_names=summary_var_names, filter_vars="like")
        if "r_hat" in summary:
            row["rhat_max"] = float(summary["r_hat"].max(skipna=True))
        if "ess_bulk" in summary:
            row["ess_bulk_min"] = float(summary["ess_bulk"].min(skipna=True))
        if "ess_tail" in summary:
            row["ess_tail_min"] = float(summary["ess_tail"].min(skipna=True))
        _add_warnings(row, "summary", caught)
    except Exception as e:
        row["summary_error"] = repr(e)

    if compute_loo:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                loo = az.loo(merge_log_likelihoods(trace), var_name="obs")
            row["elpd_loo"] = _loo_value(loo, "elpd_loo", "elpd")
            row["p_loo"] = _loo_value(loo, "p_loo", "p")
            row["loo_se"] = _loo_value(loo, "se", "elpd_loo_se", "elpd_se")
            if hasattr(loo, "pareto_k"):
                pk = np.asarray(loo.pareto_k).ravel()
                row["pareto_k_max"] = float(np.nanmax(pk))
                row["pareto_k_frac_gt_0p7"] = float(np.nanmean(pk > 0.7))
            _add_warnings(row, "loo", caught)
        except Exception as e:
            row["loo_error"] = repr(e)

    row.update(x_axis_sanity(trace))
    return row


def _add_warnings(
    row: dict[str, _t.Any], prefix: str, caught: list[warnings.WarningMessage]
) -> None:
    if not caught:
        return
    messages = [str(w.message) for w in caught]
    row[f"{prefix}_warning_count"] = len(messages)
    row[f"{prefix}_warnings"] = " | ".join(dict.fromkeys(messages))


def _loo_value(loo: _t.Any, *names: str) -> float:
    lookup_errors: list[str] = []
    for name in names:
        if hasattr(loo, name):
            return float(getattr(loo, name))
        try:
            return float(loo[name])
        except Exception as e:
            lookup_errors.append(f"{name}: {e!r}")
    msg = (
        f"LOO result has none of {names!r}; available fields: {dir(loo)!r}; "
        f"lookup errors: {lookup_errors!r}"
    )
    raise AttributeError(msg)


def merge_log_likelihoods(trace: _t.Any) -> _t.Any:
    """Merge multiple pointwise log-likelihood variables for ArviZ LOO/compare."""
    if not hasattr(trace, "log_likelihood"):
        return trace
    ll = trace.log_likelihood
    data_vars = getattr(ll, "data_vars", {})
    if len(data_vars) <= 1 and "obs" in data_vars:
        return trace

    data_list = []
    current_idx = 0
    for var_name in data_vars:
        data = ll[var_name]
        last_dim = data.dims[-1]
        n_obs = data.sizes[last_dim]
        data = data.rename({last_dim: "obs_id"})
        data = data.assign_coords(obs_id=np.arange(current_idx, current_idx + n_obs))
        data_list.append(data)
        current_idx += n_obs
    if not data_list:
        return trace

    groups = {
        "posterior": posterior_dataset(trace),
        "log_likelihood": xr.Dataset({"obs": xr.concat(data_list, dim="obs_id")}),
    }
    if hasattr(trace, "observed_data"):
        groups["observed_data"] = trace.observed_data
    return xr.DataTree.from_dict(groups)


def _sigma_for_label_well(
    trace: _t.Any, lbl: _t.Any, well: str, da: _t.Any, mask: np.ndarray
) -> np.ndarray:
    posterior = posterior_dataset(trace)

    sigma_var = f"sigma_obs_{lbl}"
    if sigma_var in posterior:
        arr = posterior[sigma_var]
        try:
            if "well" in arr.dims:
                sigma_full = arr.sel(well=well).mean(("chain", "draw")).values
            else:
                sigma_full = arr.mean(("chain", "draw")).values
        except Exception:
            sigma_full = arr.mean(("chain", "draw")).values
        sigma = np.asarray(sigma_full, dtype=float)
        if sigma.shape == mask.shape:
            sigma = sigma[mask]
        return np.where(np.isfinite(sigma) & (sigma > 0), sigma, np.nan)

    if hasattr(da, "y_err") and np.asarray(da.y_err).size == int(mask.sum()):
        sigma = np.asarray(da.y_err, dtype=float)
    elif hasattr(da, "y_errc") and np.asarray(da.y_errc).size == mask.size:
        sigma = np.asarray(da.y_errc, dtype=float)[mask]
    else:
        ye_var = f"ye_mag_{lbl}"
        ye_mag = _posterior_mean_scalar(posterior, ye_var, default=1.0)
        sigma = ye_mag * np.ones(int(mask.sum()), dtype=float)
    return np.where(np.isfinite(sigma) & (sigma > 0), sigma, np.nan)


def residuals_from_multifit(  # noqa: PLR0913
    multi: _t.Any,
    trace_id: str,
    binding_function: _t.Callable[..., ArrayLike],
    *,
    include_fit_params: bool = False,
    robust: bool = False,
    student_t_nu: float = STUDENT_T_NU,
    outlier_threshold: float = 3.0,
) -> pd.DataFrame:
    """Build a long calibrated-residual table from a MultiFitResult.

    ``likelihood_res`` is always ``(observed - predicted) / sigma``.  For
    Student-t robust fits, ``std_res`` is the equivalent standard-Normal score
    from the t CDF, suitable for Normal QQ plots and z-style outlier flags.
    """
    rows: list[dict[str, _t.Any]] = []

    for well, fr in multi.results.items():
        if fr.dataset is None or fr.result is None:
            continue
        ds = fr.dataset
        pars = fr.result.params

        for lbl, da in ds.items():
            mask = np.asarray(da.mask, dtype=bool)
            step = np.flatnonzero(mask)

            if hasattr(da, "xc") and np.asarray(da.xc).size == mask.size:
                x = np.asarray(da.xc, dtype=float)[mask]
            else:
                x = np.asarray(da.x, dtype=float)

            if hasattr(da, "yc") and np.asarray(da.yc).size == mask.size:
                y = np.asarray(da.yc, dtype=float)[mask]
            else:
                y = np.asarray(da.y, dtype=float)

            that = binding_function(
                x,
                pars["K"].value,
                pars[f"S0_{lbl}"].value,
                pars[f"S1_{lbl}"].value,
                is_ph=ds.is_ph,
            )
            sigma = _sigma_for_label_well(multi.trace, lbl, str(well), da, mask)
            likelihood_res = (y - that) / sigma
            std_res = residual_normal_scores(
                likelihood_res, robust=robust, student_t_nu=student_t_nu
            )
            outlier = robust_residual_outlier_mask(
                likelihood_res,
                threshold=outlier_threshold,
                robust=robust,
                student_t_nu=student_t_nu,
            )

            for j in range(len(y)):
                row = {
                    "trace_id": trace_id,
                    "well": str(well),
                    "label": str(lbl),
                    "step": int(step[j]),
                    "x": float(x[j]),
                    "y": float(y[j]),
                    "that": float(that[j]),
                    "sigma": float(sigma[j]),
                    "raw_res": float(y[j] - that[j]),
                    "likelihood_res": float(likelihood_res[j]),
                    "std_res": float(std_res[j]),
                    "residual_likelihood": "student_t" if robust else "normal",
                    "student_t_nu": float(student_t_nu) if robust else np.nan,
                    "is_residual_outlier": bool(outlier[j]),
                    "outlier_threshold": float(outlier_threshold),
                }
                if include_fit_params:
                    row["K"] = float(pars["K"].value)
                    row[f"S0_{lbl}"] = float(pars[f"S0_{lbl}"].value)
                    row[f"S1_{lbl}"] = float(pars[f"S1_{lbl}"].value)
                rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["std_res", "x", "step", "label"]
    )


def residuals_from_fit_results(  # noqa: PLR0913
    results: dict[str, _t.Any],
    trace_id: str,
    binding_function: _t.Callable[..., ArrayLike],
    *,
    include_fit_params: bool = False,
    robust: bool = False,
    student_t_nu: float = STUDENT_T_NU,
    outlier_threshold: float = 3.0,
) -> pd.DataFrame:
    """Build a long calibrated residual table from classical FitResult objects."""
    rows: list[dict[str, _t.Any]] = []
    for well, fr in results.items():
        if fr.dataset is None or fr.result is None:
            continue
        ds = fr.dataset
        pars = fr.result.params
        for lbl, da in ds.items():
            mask = np.asarray(da.mask, dtype=bool)
            step = np.flatnonzero(mask)
            x = np.asarray(da.x, dtype=float)
            y = np.asarray(da.y, dtype=float)
            that = binding_function(
                x,
                pars["K"].value,
                pars[f"S0_{lbl}"].value,
                pars[f"S1_{lbl}"].value,
                is_ph=ds.is_ph,
            )
            if hasattr(da, "y_err") and np.asarray(da.y_err).size == len(y):
                sigma = np.asarray(da.y_err, dtype=float)
            else:
                sigma = np.ones_like(y, dtype=float)
            sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, np.nan)
            likelihood_res = (y - that) / sigma
            std_res = residual_normal_scores(
                likelihood_res, robust=robust, student_t_nu=student_t_nu
            )
            outlier = robust_residual_outlier_mask(
                likelihood_res,
                threshold=outlier_threshold,
                robust=robust,
                student_t_nu=student_t_nu,
            )
            for j in range(len(y)):
                row = {
                    "trace_id": trace_id,
                    "well": str(well),
                    "label": str(lbl),
                    "step": int(step[j]) if j < len(step) else int(j),
                    "x": float(x[j]),
                    "y": float(y[j]),
                    "that": float(that[j]),
                    "sigma": float(sigma[j]),
                    "raw_res": float(y[j] - that[j]),
                    "likelihood_res": float(likelihood_res[j]),
                    "std_res": float(std_res[j]),
                    "residual_likelihood": "student_t" if robust else "normal",
                    "student_t_nu": float(student_t_nu) if robust else np.nan,
                    "is_residual_outlier": bool(outlier[j]),
                    "outlier_threshold": float(outlier_threshold),
                }
                if include_fit_params:
                    row["K"] = float(pars["K"].value)
                    row[f"S0_{lbl}"] = float(pars[f"S0_{lbl}"].value)
                    row[f"S1_{lbl}"] = float(pars[f"S1_{lbl}"].value)
                rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["std_res", "x", "step", "label"]
    )


def mad(x: pd.Series) -> float:
    arr = np.asarray(x, dtype=float)
    med = np.nanmedian(arr)
    return float(np.nanmedian(np.abs(arr - med)))


def residual_distribution_summary(res_df: pd.DataFrame) -> pd.DataFrame:
    return (
        res_df
        .groupby(["trace_id", "label"], observed=True)
        .agg(
            n=("std_res", "size"),
            mean_res=("std_res", "mean"),
            median_res=("std_res", "median"),
            sd_res=("std_res", "std"),
            mad_res=("std_res", mad),
            q05=("std_res", lambda x: float(np.nanquantile(x, 0.05))),
            q95=("std_res", lambda x: float(np.nanquantile(x, 0.95))),
            frac_abs_gt2=("std_res", lambda x: float(np.mean(np.abs(x) > 2))),
            frac_abs_gt3=("std_res", lambda x: float(np.mean(np.abs(x) > 3))),
            residual_outlier_frac=(
                "is_residual_outlier",
                lambda x: float(np.mean(x)) if len(x) else np.nan,
            )
            if "is_residual_outlier" in res_df.columns
            else (
                "std_res",
                lambda x: float(np.mean(np.abs(x) > 3)),
            ),
        )
        .reset_index()
    )


def residual_x_trend_summary(res_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_step = (
        res_df
        .groupby(["trace_id", "label", "step"], observed=True)
        .agg(
            x_mean=("x", "mean"),
            step_median_res=("std_res", "median"),
            step_mean_res=("std_res", "mean"),
        )
        .reset_index()
    )
    trend = (
        by_step
        .groupby(["trace_id", "label"], observed=True)
        .agg(
            x_median_rms=(
                "step_median_res",
                lambda x: float(np.sqrt(np.nanmean(np.asarray(x) ** 2))),
            ),
            x_median_maxabs=(
                "step_median_res",
                lambda x: float(np.nanmax(np.abs(np.asarray(x)))),
            ),
            x_mean_rms=(
                "step_mean_res",
                lambda x: float(np.sqrt(np.nanmean(np.asarray(x) ** 2))),
            ),
        )
        .reset_index()
    )
    return trend, by_step


def residual_x_correlation(res_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, _t.Any]] = []
    for (trace_id, label), g in res_df.groupby(["trace_id", "label"], observed=True):
        if len(g) < 4:
            continue
        rows.append({
            "trace_id": trace_id,
            "label": label,
            "pearson_res_x": g["std_res"].corr(g["x"], method="pearson"),
            "spearman_res_x": g["std_res"].corr(g["x"], method="spearman"),
        })
    return pd.DataFrame(rows)


def residual_lag1_autocorrelation(
    res_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, _t.Any]] = []
    for (trace_id, well, label), g in res_df.groupby(
        ["trace_id", "well", "label"], observed=True
    ):
        g = g.sort_values("step")
        r = g["std_res"].to_numpy(dtype=float)
        if len(r) < 3 or np.nanstd(r[:-1]) == 0 or np.nanstd(r[1:]) == 0:
            lag1 = np.nan
        else:
            lag1 = float(np.corrcoef(r[:-1], r[1:])[0, 1])
        rows.append({
            "trace_id": trace_id,
            "well": well,
            "label": label,
            "lag1_res_autocorr": lag1,
        })
    lag_df = pd.DataFrame(rows)
    if lag_df.empty:
        return lag_df, pd.DataFrame()
    summary = (
        lag_df
        .groupby(["trace_id", "label"], observed=True)
        .agg(
            lag1_mean=("lag1_res_autocorr", "mean"),
            lag1_median=("lag1_res_autocorr", "median"),
            lag1_abs_mean=("lag1_res_autocorr", lambda x: float(np.nanmean(np.abs(x)))),
        )
        .reset_index()
    )
    return lag_df, summary


def residual_cross_label_correlation(
    res_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    wide = res_df.pivot_table(
        index=["trace_id", "well", "step"],
        columns="label",
        values="std_res",
        aggfunc="mean",
    )
    rows: list[dict[str, _t.Any]] = []
    for trace_id, sub in wide.groupby(level=0):
        mat = sub.droplevel(0)
        labels = list(mat.columns)
        corr = mat.corr()
        for a, b in itertools.combinations(labels, 2):
            rows.append({
                "trace_id": trace_id,
                "label_a": str(a),
                "label_b": str(b),
                "cross_label_corr": float(np.asarray(corr.loc[a, b]).ravel()[0]),
            })
    corr_df = pd.DataFrame(rows)
    if corr_df.empty:
        return corr_df, pd.DataFrame()
    summary = (
        corr_df
        .groupby("trace_id", observed=True)
        .agg(
            cross_label_corr_abs_mean=(
                "cross_label_corr",
                lambda x: float(np.nanmean(np.abs(x))),
            ),
            cross_label_corr_abs_max=(
                "cross_label_corr",
                lambda x: float(np.nanmax(np.abs(x))),
            ),
        )
        .reset_index()
    )
    return corr_df, summary


def model_residual_score_table(
    res_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return model-level and detailed residual summary tables."""
    dist = residual_distribution_summary(res_df)
    trend, by_step = residual_x_trend_summary(res_df)
    corr_x = residual_x_correlation(res_df)
    lag_df, lag_summary = residual_lag1_autocorrelation(res_df)
    cross_df, cross_summary = residual_cross_label_correlation(res_df)

    per_label = dist.merge(trend, on=["trace_id", "label"], how="left")
    if not corr_x.empty:
        per_label = per_label.merge(corr_x, on=["trace_id", "label"], how="left")
    if not lag_summary.empty:
        per_label = per_label.merge(lag_summary, on=["trace_id", "label"], how="left")

    agg_spec: dict[str, tuple[str, _t.Any]] = {
        "residual_mean_abs": ("mean_res", lambda x: float(np.nanmean(np.abs(x)))),
        "residual_median_abs": ("median_res", lambda x: float(np.nanmean(np.abs(x)))),
        "residual_sd_mean": ("sd_res", "mean"),
        "residual_frac_abs_gt2": ("frac_abs_gt2", "mean"),
        "residual_frac_abs_gt3": ("frac_abs_gt3", "mean"),
        "residual_outlier_frac": ("residual_outlier_frac", "mean"),
        "residual_x_median_rms": ("x_median_rms", "mean"),
        "residual_x_median_maxabs": ("x_median_maxabs", "max"),
    }
    if "spearman_res_x" in per_label:
        agg_spec["residual_abs_spearman_x"] = (
            "spearman_res_x",
            lambda x: float(np.nanmean(np.abs(x))) if len(x) else np.nan,
        )
    if "lag1_abs_mean" in per_label:
        agg_spec["residual_lag1_abs_mean"] = ("lag1_abs_mean", "mean")

    model = per_label.groupby("trace_id", observed=True).agg(**agg_spec).reset_index()
    if not cross_summary.empty:
        model = model.merge(cross_summary, on="trace_id", how="left")
    return model, per_label, by_step, lag_df, cross_df
