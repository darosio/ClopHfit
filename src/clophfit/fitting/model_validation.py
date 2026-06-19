"""Reusable model-validation helpers for ClopHfit fitting workflows.

These utilities are designed to live in :mod:`clophfit.fitting` and be reused by
both package tests and manuscript-analysis scripts.  They intentionally avoid any
manuscript-specific paths, file formats, or plate names.
"""

from __future__ import annotations

import itertools
import typing as _t

import arviz as az  # type: ignore[import-untyped]
import numpy as np
import pandas as pd

ArrayLike = _t.Any


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
        summary = az.summary(trace, var_names=summary_var_names, filter_vars="like")
        if "r_hat" in summary:
            row["rhat_max"] = float(summary["r_hat"].max(skipna=True))
        if "ess_bulk" in summary:
            row["ess_bulk_min"] = float(summary["ess_bulk"].min(skipna=True))
        if "ess_tail" in summary:
            row["ess_tail_min"] = float(summary["ess_tail"].min(skipna=True))
    except Exception as e:
        row["summary_error"] = repr(e)

    if compute_loo:
        try:
            loo = az.loo(trace)
            row["elpd_loo"] = float(loo.elpd_loo)
            row["p_loo"] = float(loo.p_loo)
            row["loo_se"] = float(loo.se)
            if hasattr(loo, "pareto_k"):
                pk = np.asarray(loo.pareto_k).ravel()
                row["pareto_k_max"] = float(np.nanmax(pk))
                row["pareto_k_frac_gt_0p7"] = float(np.nanmean(pk > 0.7))
        except Exception as e:
            row["loo_error"] = repr(e)

    row.update(x_axis_sanity(trace))
    return row


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


def residuals_from_multifit(
    multi: _t.Any,
    trace_id: str,
    binding_function: _t.Callable[..., ArrayLike],
    *,
    include_fit_params: bool = False,
) -> pd.DataFrame:
    """Build a long standardized-residual table from a MultiFitResult."""
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
            std_res = (y - that) / sigma

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
                    "std_res": float(std_res[j]),
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


def residuals_from_fit_results(
    results: dict[str, _t.Any],
    trace_id: str,
    binding_function: _t.Callable[..., ArrayLike],
    *,
    include_fit_params: bool = False,
) -> pd.DataFrame:
    """Build a long residual table from classical per-well FitResult objects."""
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
            std_res = (y - that) / sigma
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
                    "std_res": float(std_res[j]),
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
