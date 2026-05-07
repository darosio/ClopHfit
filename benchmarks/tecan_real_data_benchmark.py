"""Real-data Tecan benchmark for comparing fitting workflows.

Quick start
-----------
Use this script when you want to compare fitting workflows on real wells,
not synthetic data. The most useful default comparison is dual-channel
`y1+y2`, because it lets you compare the joint fit against single-channel
alternatives with the same wells and samples.

Recommended progression:
1. Smoke run on a few wells with `--skip-loo` to confirm filters and outputs.
2. Small informative run with `--channel y1+y2` and LOO enabled.
3. Broader run across all realizations of the same sample so that paired,
   win-rate, and agreement summaries compare like with like.

Simplified output handling:
- pass `--output-dir some/folder` to write all CSV artifacts there;
- or override individual files with `--output-csv`, `--output-summary-csv`,
  etc. if you want custom names.

Suggested first informative command:
`uv run python benchmarks/tecan_real_data_benchmark.py \
  --list-file tests/Tecan/140220/list.pH.csv \
  --additions-file tests/Tecan/140220/additions.pH \
  --scheme-file tests/Tecan/140220/scheme.txt \
  --sample G03 \
  --channel y1+y2 \
  --final-stage huber \
  --final-stage odr \
  --loo-max-points 3 \
  --output-dir benchmarks/results/g03_y1y2`

If your goal is agreement across different realizations of the same sample,
filter to one sample at a time (for example `--sample G03`) and let the run
include all its wells. Then inspect:
- summary: overall fit quality and finite-fit rate
- paired/win-rate tables: matched well-by-well comparisons
- agreement: whether residual_std and loo_rmse pick the same winner
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any, NamedTuple

import click
import numpy as np
import pandas as pd

from clophfit.fitting.core import weight_multi_ds_calibrated
from clophfit.fitting.data_structures import Dataset, FitResult
from clophfit.fitting.models import binding_1site
from clophfit.prtecan.prtecan import Titration
from clophfit.testing.fitter_test_utils import (
    TecanFitCombination,
    TecanWeighting,
    apply_tecan_combination,
    build_factorized_tecan_fit_combinations,
)

FACTOR_COLUMNS = (
    "channels",
    "prefit",
    "final_stage",
    "weighting",
    "outlier_handling",
)
REAL_SUMMARY_METRIC_COLUMNS = (
    "success_rate",
    "finite_fit_rate",
    "mean_k_error",
    "mean_residual_mean",
    "mean_residual_std",
    "mean_shapiro_p",
    "mean_loo_rmse",
)
DEFAULT_INTERACTION_PAIRS = (
    ("channels", "final_stage"),
    ("weighting", "final_stage"),
)
DEFAULT_RANKING_METRICS = (
    "mean_loo_rmse",
    "mean_residual_std",
    "finite_fit_rate",
)
LOWER_IS_BETTER_PAIRED_METRICS = ("loo_rmse", "residual_std")
DEFAULT_OUTPUT_FILENAMES = {
    "output_csv": "tecan_real_data_results.csv",
    "output_summary_csv": "tecan_real_data_summary.csv",
    "output_factor_csv": "tecan_real_data_factor_effects.csv",
    "output_ranking_csv": "tecan_real_data_rankings.csv",
    "output_interaction_csv": "tecan_real_data_interactions.csv",
    "output_paired_csv": "tecan_real_data_paired.csv",
    "output_paired_winrate_csv": "tecan_real_data_paired_winrates.csv",
    "output_paired_by_sample_csv": "tecan_real_data_paired_by_sample.csv",
    "output_agreement_csv": "tecan_real_data_agreement.csv",
}


class RealCurve(NamedTuple):
    """Single real-data curve extracted from a Tecan titration plate."""

    well: str
    sample: str
    dataset: Dataset


def resolve_output_paths(output_dir: Path | None) -> dict[str, Path]:
    """Expand one output directory into the standard artifact file paths."""
    base_dir = output_dir or Path("benchmarks/results")
    return {key: base_dir / filename for key, filename in DEFAULT_OUTPUT_FILENAMES.items()}


def _normalize_channel_groups(channels: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    """Convert CLI channel strings like 'y1+y2' into channel tuples."""
    if not channels:
        return (("y1",), ("y2",), ("y1", "y2"))
    groups: list[tuple[str, ...]] = []
    for channel_group in channels:
        parsed = tuple(part.strip() for part in channel_group.split("+") if part.strip())
        if not parsed:
            continue
        groups.append(parsed)
    return tuple(groups)


def _result_diagnostics(fit_result: FitResult[Any]) -> tuple[float, float, float]:
    """Extract residual diagnostics from a fit result when available."""
    result = getattr(fit_result, "result", None)
    residuals = getattr(result, "residual", None)
    if residuals is None:
        return np.nan, np.nan, np.nan
    residual_array = np.asarray(residuals, dtype=float)
    finite = residual_array[np.isfinite(residual_array)]
    if finite.size == 0:
        return np.nan, np.nan, np.nan
    if finite.size < 3:
        return float(np.mean(finite)), float(np.std(finite)), np.nan
    from clophfit.testing.evaluation import evaluate_residuals

    stats = evaluate_residuals(finite)
    return stats["mean"], stats["std"], stats["shapiro_p"]


def _predict_label(
    fit_result: FitResult[Any], *, label: str, x_value: float, is_ph: bool
) -> float:
    """Predict one held-out y value from fitted parameters."""
    if fit_result.result is None:
        return np.nan
    params = fit_result.result.params
    if "K" not in params:
        return np.nan
    s0_name = f"S0_{label}"
    s1_name = f"S1_{label}"
    if s0_name not in params or s1_name not in params:
        return np.nan
    pred = binding_1site(
        np.array([x_value], dtype=float),
        params["K"].value,
        params[s0_name].value,
        params[s1_name].value,
        is_ph=is_ph,
    )
    return float(pred[0])


def leave_one_out_rmse(
    dataset: Dataset,
    combination: TecanFitCombination,
    *,
    max_points: int | None = None,
) -> float:
    """Compute leave-one-out RMSE for one fitting combination on one dataset."""
    errors: list[float] = []
    points_used = 0
    for label in combination.channels:
        if label not in dataset:
            continue
        original = dataset[label]
        for idx in range(len(original.xc)):
            if max_points is not None and points_used >= max_points:
                break
            if not original.mask[idx]:
                continue
            work_ds = dataset.copy(keys=list(dataset.keys()))
            holdout_mask = work_ds[label].mask.copy()
            holdout_mask[idx] = False
            work_ds[label].mask = holdout_mask
            if len(work_ds[label].y) < 4:
                continue
            try:
                fit_result = apply_tecan_combination(work_ds, combination)
            except Exception:
                continue
            predicted = _predict_label(
                fit_result,
                label=label,
                x_value=float(original.xc[idx]),
                is_ph=dataset.is_ph,
            )
            if not np.isfinite(predicted):
                continue
            held_out_y = float(original.yc[idx])
            errors.append((predicted - held_out_y) ** 2)
            points_used += 1
        if max_points is not None and points_used >= max_points:
            break
    if not errors:
        return np.nan
    return float(np.sqrt(np.mean(errors)))


def load_real_tecan_curves(
    *,
    list_file: Path,
    additions_file: Path,
    scheme_file: Path,
    is_ph: bool,
) -> list[RealCurve]:
    """Load per-well datasets from a real Tecan titration experiment."""
    tit = Titration.fromlistfile(list_file, is_ph=is_ph)
    tit.load_additions(additions_file)
    tit.load_scheme(scheme_file)
    curves: list[RealCurve] = []
    sample_by_well = {
        well: sample for sample, wells in tit.scheme.names.items() for well in wells
    }
    for well in sorted(tit.fit_keys):
        ds = tit._create_global_ds(well)  # noqa: SLF001
        curves.append(
            RealCurve(
                well=well,
                sample=sample_by_well.get(well, "unknown"),
                dataset=ds.copy(),
            )
        )
    return curves


def _load_real_titration(
    *,
    list_file: Path,
    additions_file: Path,
    scheme_file: Path,
    is_ph: bool,
) -> Titration:
    """Build a fresh Titration object for one real-data benchmark execution path."""
    tit = Titration.fromlistfile(list_file, is_ph=is_ph)
    tit.load_additions(additions_file)
    tit.load_scheme(scheme_file)
    return tit


def _resolve_titration_stage(final_stage: str) -> tuple[str, str]:
    """Map benchmark stage names to Titration config values and result accessors."""
    stage_map = {
        "huber": ("None", "result_global"),
        "lm": ("None", "result_global"),
        "irls": ("None", "result_global"),
        "odr": ("None", "result_odr"),
        "mcmc_single": ("single", "result_mcmc"),
        "mcmc_multi": ("multi", "result_multi_mcmc"),
        "mcmc_multi-noise": ("multi-noise", "result_multi_noise_mcmc"),
        "mcmc_multi-noise-xrw": ("multi-noise-xrw", "result_multi_noise_xrw_mcmc"),
    }
    return stage_map[final_stage]


def _combination_requires_titration_context(combination: TecanFitCombination) -> bool:
    """Return whether a benchmark combination needs full Titration context."""
    return (
        combination.weighting == "calibrated"
        or combination.final_stage in {"mcmc_multi", "mcmc_multi-noise", "mcmc_multi-noise-xrw"}
    )


def _fit_real_curve_with_titration(
    *,
    well: str,
    combination: TecanFitCombination,
    list_file: Path,
    additions_file: Path,
    scheme_file: Path,
    is_ph: bool,
) -> FitResult[Any]:
    """Run combinations requiring pooled real-data context on a fresh Titration."""
    tit = _load_real_titration(
        list_file=list_file,
        additions_file=additions_file,
        scheme_file=scheme_file,
        is_ph=is_ph,
    )
    requested_well = well
    tit.params.fit_method = combination.prefit
    tit.params.n_mcmc_samples = 200
    mcmc_mode, result_attr = _resolve_titration_stage(combination.final_stage)
    tit.params.mcmc = mcmc_mode

    if combination.weighting == "calibrated":
        datasets = {fit_well: tit._create_global_ds(fit_well) for fit_well in sorted(tit.fit_keys)}  # noqa: SLF001
        noise_params = weight_multi_ds_calibrated(datasets, is_ph=tit.is_ph)
        label_names = sorted(combination.channels)
        tit.params.noise_alpha = tuple(noise_params.get(label, (1.0, 0.0, 0.0))[2] for label in label_names)
        tit.params.noise_gain = tuple(noise_params.get(label, (1.0, 0.0, 0.0))[1] for label in label_names)

    result_set = getattr(tit, result_attr)
    return result_set[requested_well]


def run_real_data_benchmark(
    *,
    list_file: Path,
    additions_file: Path,
    scheme_file: Path,
    is_ph: bool,
    include_mcmc: bool,
    output_csv: Path | None,
    max_wells: int | None = None,
    samples: tuple[str, ...] = (),
    channels: tuple[str, ...] = (),
    final_stages: tuple[str, ...] = (),
    weightings: tuple[TecanWeighting, ...] = ("auto",),
    skip_loo: bool = False,
    loo_max_points: int | None = None,
) -> pd.DataFrame:
    """Run configured combinations on filtered real per-well curves."""
    curves = load_real_tecan_curves(
        list_file=list_file,
        additions_file=additions_file,
        scheme_file=scheme_file,
        is_ph=is_ph,
    )
    if samples:
        sample_filter = set(samples)
        curves = [curve for curve in curves if curve.sample in sample_filter]
    if max_wells is not None:
        curves = curves[:max_wells]

    combo_channels = _normalize_channel_groups(channels)
    combo_final_stages = (
        final_stages
        if final_stages
        else (("huber", "odr", "mcmc_single") if include_mcmc else ("huber", "odr"))
    )
    combinations_map = build_factorized_tecan_fit_combinations(
        channels=combo_channels,
        final_stages=combo_final_stages,
        weightings=weightings,
    )

    rows: list[dict[str, Any]] = []
    for curve in curves:
        for name, combination in combinations_map.items():
            try:
                if _combination_requires_titration_context(combination):
                    fit_result = _fit_real_curve_with_titration(
                        well=curve.well,
                        combination=combination,
                        list_file=list_file,
                        additions_file=additions_file,
                        scheme_file=scheme_file,
                        is_ph=is_ph,
                    )
                else:
                    ds_fresh = curve.dataset.copy()
                    fit_result = apply_tecan_combination(ds_fresh, combination)
                result = fit_result.result
                k_fit = np.nan
                k_err = np.nan
                if result is not None and hasattr(result, "params") and "K" in result.params:
                    k_val = result.params["K"].value
                    k_stderr = result.params["K"].stderr
                    k_fit = np.nan if k_val is None else float(k_val)
                    k_err = np.nan if k_stderr is None else float(k_stderr)
                residual_mean, residual_std, shapiro_p = _result_diagnostics(fit_result)
                loo_rmse = (
                    np.nan
                    if skip_loo
                    else leave_one_out_rmse(
                        curve.dataset.copy(),
                        combination,
                        max_points=loo_max_points,
                    )
                )
                success = result is not None and np.isfinite(k_fit)
            except NotImplementedError:
                success = False
                k_fit = np.nan
                k_err = np.nan
                residual_mean = np.nan
                residual_std = np.nan
                shapiro_p = np.nan
                loo_rmse = np.nan
            except Exception:
                success = False
                k_fit = np.nan
                k_err = np.nan
                residual_mean = np.nan
                residual_std = np.nan
                shapiro_p = np.nan
                loo_rmse = np.nan
            rows.append(
                {
                    "well": curve.well,
                    "sample": curve.sample,
                    "method": name,
                    "channels": "+".join(combination.channels),
                    "prefit": combination.prefit,
                    "final_stage": combination.final_stage,
                    "weighting": combination.weighting,
                    "outlier_handling": combination.outlier_handling or "none",
                    "estimated_k": k_fit,
                    "k_error": k_err,
                    "success": success,
                    "residual_mean": residual_mean,
                    "residual_std": residual_std,
                    "shapiro_p": shapiro_p,
                    "loo_rmse": loo_rmse,
                }
            )
    df = pd.DataFrame(rows)
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
    return df


def _finite_mean(values: pd.Series) -> float:
    """Return the mean of finite values, or NaN when none are finite."""
    arr = values.to_numpy(dtype=float)
    finite = arr[np.isfinite(arr)]
    return float(np.mean(finite)) if finite.size else np.nan


def _paired_merge(
    df: pd.DataFrame, method_a: str, method_b: str
) -> pd.DataFrame:
    """Return shared-well rows for two methods with aligned metric columns."""
    base_columns = ["well", "sample", "estimated_k", "residual_std", "loo_rmse"]
    subset_a = df.loc[df["method"] == method_a, base_columns].rename(
        columns={
            "estimated_k": "estimated_k_a",
            "residual_std": "residual_std_a",
            "loo_rmse": "loo_rmse_a",
        }
    )
    subset_b = df.loc[df["method"] == method_b, base_columns].rename(
        columns={
            "estimated_k": "estimated_k_b",
            "residual_std": "residual_std_b",
            "loo_rmse": "loo_rmse_b",
        }
    )
    return subset_a.merge(subset_b, on=["well", "sample"], how="inner")


def summarize_real_results(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize real-data benchmark results per method."""
    summary_rows: list[dict[str, Any]] = []
    factor_defaults = {
        "channels": "unknown",
        "prefit": "unknown",
        "final_stage": "unknown",
        "weighting": "unknown",
        "outlier_handling": "none",
    }
    for method, group in df.groupby("method", sort=True):
        successful = group[group["success"].astype(bool)]
        finite_mask = (
            successful["estimated_k"].notna()
            & successful["k_error"].notna()
            & np.isfinite(successful["estimated_k"])
            & np.isfinite(successful["k_error"])
        )
        finite = successful[finite_mask]
        row: dict[str, Any] = {
            "method": str(method),
            "n_curves": int(group.shape[0]),
            "success_rate": float(successful.shape[0] / group.shape[0]),
            "finite_fit_rate": float(finite.shape[0] / group.shape[0]),
            "median_k": (
                float(np.nanmedian(finite["estimated_k"])) if not finite.empty else np.nan
            ),
            "mean_k_error": (
                float(np.nanmean(finite["k_error"])) if not finite.empty else np.nan
            ),
            "mean_residual_mean": (
                float(np.nanmean(finite["residual_mean"]))
                if not finite.empty
                else np.nan
            ),
            "mean_residual_std": (
                float(np.nanmean(finite["residual_std"])) if not finite.empty else np.nan
            ),
            "mean_shapiro_p": (
                float(np.nanmean(finite["shapiro_p"])) if not finite.empty else np.nan
            ),
            "mean_loo_rmse": (
                _finite_mean(finite["loo_rmse"]) if not finite.empty else np.nan
            ),
        }
        for column, default in factor_defaults.items():
            row[column] = str(group[column].iloc[0]) if column in group.columns else default
        summary_rows.append(row)
    return pd.DataFrame(summary_rows).sort_values(
        ["success_rate", "finite_fit_rate", "mean_loo_rmse", "mean_residual_std"],
        ascending=[False, False, True, True],
    )


def summarize_real_factor_effects(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate real-data summaries by factor level."""
    effect_rows: list[dict[str, Any]] = []
    for factor in FACTOR_COLUMNS:
        if factor not in summary_df.columns:
            continue
        for level, group in summary_df.groupby(factor, dropna=False, sort=True):
            row: dict[str, Any] = {
                "factor": factor,
                "level": str(level),
                "n_methods": int(group.shape[0]),
            }
            for metric in REAL_SUMMARY_METRIC_COLUMNS:
                if metric not in group.columns:
                    row[f"mean_{metric}"] = np.nan
                    continue
                values = group[metric].to_numpy(dtype=float)
                finite_values = values[np.isfinite(values)]
                row[f"mean_{metric}"] = (
                    float(np.mean(finite_values)) if finite_values.size else np.nan
                )
            effect_rows.append(row)
    return pd.DataFrame(effect_rows).sort_values(["factor", "level"])


def rank_real_methods(summary_df: pd.DataFrame, *, metric: str) -> pd.DataFrame:
    """Rank real-data methods by a selected summary metric."""
    ascending = metric != "finite_fit_rate"
    ranked = summary_df.sort_values(metric, ascending=ascending, na_position="last").reset_index(
        drop=True
    )
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    ranked.insert(1, "ranking_metric", metric)
    return ranked


def summarize_real_interactions(
    summary_df: pd.DataFrame,
    *,
    factor_pairs: tuple[tuple[str, str], ...] = DEFAULT_INTERACTION_PAIRS,
) -> pd.DataFrame:
    """Aggregate pairwise interaction summaries across factor levels."""
    rows: list[dict[str, Any]] = []
    for factor_a, factor_b in factor_pairs:
        if factor_a not in summary_df.columns or factor_b not in summary_df.columns:
            continue
        grouped = summary_df.groupby([factor_a, factor_b], dropna=False, sort=True)
        for (level_a, level_b), group in grouped:
            row: dict[str, Any] = {
                "factor_a": factor_a,
                "level_a": str(level_a),
                "factor_b": factor_b,
                "level_b": str(level_b),
                "n_methods": int(group.shape[0]),
            }
            for metric in REAL_SUMMARY_METRIC_COLUMNS:
                if metric not in group.columns:
                    row[f"mean_{metric}"] = np.nan
                    continue
                values = group[metric].to_numpy(dtype=float)
                finite_values = values[np.isfinite(values)]
                row[f"mean_{metric}"] = (
                    float(np.mean(finite_values)) if finite_values.size else np.nan
                )
            rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["factor_a", "level_a", "factor_b", "level_b"]
    )


def paired_method_comparisons(
    df: pd.DataFrame,
    *,
    method_pairs: tuple[tuple[str, str], ...] | None = None,
) -> pd.DataFrame:
    """Compute paired method deltas on shared wells, which is the preferred comparison."""
    methods = sorted(df["method"].dropna().unique().tolist())
    pair_list = method_pairs or tuple(combinations(methods, 2))
    rows: list[dict[str, Any]] = []
    for method_a, method_b in pair_list:
        merged = _paired_merge(df, method_a, method_b)
        if merged.empty:
            continue
        delta_loo = merged["loo_rmse_a"] - merged["loo_rmse_b"]
        delta_resid = merged["residual_std_a"] - merged["residual_std_b"]
        delta_k = np.abs(merged["estimated_k_a"] - merged["estimated_k_b"])
        rows.append(
            {
                "method_a": method_a,
                "method_b": method_b,
                "n_pairs": int(merged.shape[0]),
                "mean_delta_loo_rmse": _finite_mean(delta_loo),
                "mean_delta_residual_std": _finite_mean(delta_resid),
                "mean_abs_delta_k": _finite_mean(delta_k),
            }
        )
    return pd.DataFrame(rows).sort_values(["method_a", "method_b"])


def paired_win_rates(
    df: pd.DataFrame,
    *,
    method_pairs: tuple[tuple[str, str], ...] | None = None,
) -> pd.DataFrame:
    """Report paired per-well win rates for metrics where lower values are better."""
    methods = sorted(df["method"].dropna().unique().tolist())
    pair_list = method_pairs or tuple(combinations(methods, 2))
    rows: list[dict[str, Any]] = []
    for method_a, method_b in pair_list:
        merged = _paired_merge(df, method_a, method_b)
        if merged.empty:
            continue
        row: dict[str, Any] = {
            "method_a": method_a,
            "method_b": method_b,
            "n_pairs": int(merged.shape[0]),
        }
        for metric in LOWER_IS_BETTER_PAIRED_METRICS:
            a_values = merged[f"{metric}_a"].to_numpy(dtype=float)
            b_values = merged[f"{metric}_b"].to_numpy(dtype=float)
            valid = np.isfinite(a_values) & np.isfinite(b_values)
            if not valid.any():
                row[f"win_rate_{metric}_a"] = np.nan
                row[f"win_rate_{metric}_b"] = np.nan
                row[f"tie_rate_{metric}"] = np.nan
                continue
            a_valid = a_values[valid]
            b_valid = b_values[valid]
            row[f"win_rate_{metric}_a"] = float(np.mean(a_valid < b_valid))
            row[f"win_rate_{metric}_b"] = float(np.mean(b_valid < a_valid))
            row[f"tie_rate_{metric}"] = float(np.mean(a_valid == b_valid))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["method_a", "method_b"])


def paired_method_comparisons_by_sample(
    df: pd.DataFrame,
    *,
    method_pairs: tuple[tuple[str, str], ...] | None = None,
) -> pd.DataFrame:
    """Compute paired comparisons separately for each sample on shared wells."""
    methods = sorted(df["method"].dropna().unique().tolist())
    pair_list = method_pairs or tuple(combinations(methods, 2))
    rows: list[dict[str, Any]] = []
    for sample, sample_df in df.groupby("sample", sort=True):
        for method_a, method_b in pair_list:
            merged = _paired_merge(sample_df, method_a, method_b)
            if merged.empty:
                continue
            delta_loo = merged["loo_rmse_a"] - merged["loo_rmse_b"]
            delta_resid = merged["residual_std_a"] - merged["residual_std_b"]
            delta_k = np.abs(merged["estimated_k_a"] - merged["estimated_k_b"])
            rows.append(
                {
                    "sample": str(sample),
                    "method_a": method_a,
                    "method_b": method_b,
                    "n_pairs": int(merged.shape[0]),
                    "mean_delta_loo_rmse": _finite_mean(delta_loo),
                    "mean_delta_residual_std": _finite_mean(delta_resid),
                    "mean_abs_delta_k": _finite_mean(delta_k),
                }
            )
    return pd.DataFrame(rows).sort_values(["sample", "method_a", "method_b"])


def paired_metric_agreement(
    df: pd.DataFrame,
    *,
    method_pairs: tuple[tuple[str, str], ...] | None = None,
) -> pd.DataFrame:
    """Quantify whether residual_std and LOO select the same winner on shared wells."""
    methods = sorted(df["method"].dropna().unique().tolist())
    pair_list = method_pairs or tuple(combinations(methods, 2))
    rows: list[dict[str, Any]] = []
    for method_a, method_b in pair_list:
        merged = _paired_merge(df, method_a, method_b)
        if merged.empty:
            continue
        resid_a = merged["residual_std_a"].to_numpy(dtype=float)
        resid_b = merged["residual_std_b"].to_numpy(dtype=float)
        loo_a = merged["loo_rmse_a"].to_numpy(dtype=float)
        loo_b = merged["loo_rmse_b"].to_numpy(dtype=float)
        valid = (
            np.isfinite(resid_a)
            & np.isfinite(resid_b)
            & np.isfinite(loo_a)
            & np.isfinite(loo_b)
        )
        if not valid.any():
            rows.append(
                {
                    "method_a": method_a,
                    "method_b": method_b,
                    "n_pairs": int(merged.shape[0]),
                    "n_valid_pairs": 0,
                    "agreement_rate": np.nan,
                    "both_prefer_a_rate": np.nan,
                    "both_prefer_b_rate": np.nan,
                    "disagreement_rate": np.nan,
                    "tie_in_either_rate": np.nan,
                }
            )
            continue
        resid_cmp = np.sign(resid_b[valid] - resid_a[valid])
        loo_cmp = np.sign(loo_b[valid] - loo_a[valid])
        agree = resid_cmp == loo_cmp
        both_a = (resid_cmp > 0) & (loo_cmp > 0)
        both_b = (resid_cmp < 0) & (loo_cmp < 0)
        disagreement = (resid_cmp != 0) & (loo_cmp != 0) & (resid_cmp != loo_cmp)
        tie_either = (resid_cmp == 0) | (loo_cmp == 0)
        n_valid = int(valid.sum())
        rows.append(
            {
                "method_a": method_a,
                "method_b": method_b,
                "n_pairs": int(merged.shape[0]),
                "n_valid_pairs": n_valid,
                "agreement_rate": float(np.mean(agree)),
                "both_prefer_a_rate": float(np.mean(both_a)),
                "both_prefer_b_rate": float(np.mean(both_b)),
                "disagreement_rate": float(np.mean(disagreement)),
                "tie_in_either_rate": float(np.mean(tie_either)),
            }
        )
    return pd.DataFrame(rows).sort_values(["method_a", "method_b"])


@click.command()
@click.option("--list-file", type=click.Path(path_type=Path, exists=True, dir_okay=False), required=True)
@click.option("--additions-file", type=click.Path(path_type=Path, exists=True, dir_okay=False), required=True)
@click.option("--scheme-file", type=click.Path(path_type=Path, exists=True, dir_okay=False), required=True)
@click.option("--is-ph/--no-is-ph", default=True, show_default=True)
@click.option("--include-mcmc/--no-include-mcmc", default=False, show_default=True)
@click.option("--max-wells", type=int, default=None)
@click.option("--sample", "samples", multiple=True)
@click.option("--channel", "channels", multiple=True, help="Repeatable. Use y1, y2, or y1+y2.")
@click.option("--final-stage", "final_stages", multiple=True)
@click.option("--weighting", "weightings", multiple=True)
@click.option("--skip-loo", is_flag=True, default=False)
@click.option("--loo-max-points", type=int, default=None)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=None,
    help="Directory where all standard CSV outputs are written.",
)
@click.option(
    "--output-csv",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
)
@click.option(
    "--output-summary-csv",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
)
@click.option(
    "--output-factor-csv",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
)
@click.option(
    "--output-ranking-csv",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
)
@click.option(
    "--output-interaction-csv",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
)
@click.option(
    "--output-paired-csv",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
)
@click.option(
    "--output-paired-winrate-csv",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
)
@click.option(
    "--output-paired-by-sample-csv",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
)
@click.option(
    "--output-agreement-csv",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
)
def main(
    list_file: Path,
    additions_file: Path,
    scheme_file: Path,
    is_ph: bool,
    include_mcmc: bool,
    max_wells: int | None,
    samples: tuple[str, ...],
    channels: tuple[str, ...],
    final_stages: tuple[str, ...],
    weightings: tuple[str, ...],
    skip_loo: bool,
    loo_max_points: int | None,
    output_dir: Path | None,
    output_csv: Path | None,
    output_summary_csv: Path | None,
    output_factor_csv: Path | None,
    output_ranking_csv: Path | None,
    output_interaction_csv: Path | None,
    output_paired_csv: Path | None,
    output_paired_winrate_csv: Path | None,
    output_paired_by_sample_csv: Path | None,
    output_agreement_csv: Path | None,
) -> None:
    """Run the real-data Tecan benchmark and save summaries suited for comparison."""
    output_paths = resolve_output_paths(output_dir)
    output_csv = output_csv or output_paths["output_csv"]
    output_summary_csv = output_summary_csv or output_paths["output_summary_csv"]
    output_factor_csv = output_factor_csv or output_paths["output_factor_csv"]
    output_ranking_csv = output_ranking_csv or output_paths["output_ranking_csv"]
    output_interaction_csv = output_interaction_csv or output_paths["output_interaction_csv"]
    output_paired_csv = output_paired_csv or output_paths["output_paired_csv"]
    output_paired_winrate_csv = (
        output_paired_winrate_csv or output_paths["output_paired_winrate_csv"]
    )
    output_paired_by_sample_csv = (
        output_paired_by_sample_csv or output_paths["output_paired_by_sample_csv"]
    )
    output_agreement_csv = output_agreement_csv or output_paths["output_agreement_csv"]
    df = run_real_data_benchmark(
        list_file=list_file,
        additions_file=additions_file,
        scheme_file=scheme_file,
        is_ph=is_ph,
        include_mcmc=include_mcmc,
        output_csv=output_csv,
        max_wells=max_wells,
        samples=samples,
        channels=channels,
        final_stages=final_stages,
        weightings=tuple(weightings or ("auto",)),
        skip_loo=skip_loo,
        loo_max_points=loo_max_points,
    )
    summary = summarize_real_results(df)
    factor_effects = summarize_real_factor_effects(summary)
    rankings = pd.concat(
        [rank_real_methods(summary, metric=metric) for metric in DEFAULT_RANKING_METRICS],
        ignore_index=True,
    )
    interactions = summarize_real_interactions(summary)
    paired = paired_method_comparisons(df)
    paired_winrate = paired_win_rates(df)
    paired_by_sample = paired_method_comparisons_by_sample(df)
    agreement = paired_metric_agreement(df)

    output_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_summary_csv, index=False)
    output_factor_csv.parent.mkdir(parents=True, exist_ok=True)
    factor_effects.to_csv(output_factor_csv, index=False)
    output_ranking_csv.parent.mkdir(parents=True, exist_ok=True)
    rankings.to_csv(output_ranking_csv, index=False)
    output_interaction_csv.parent.mkdir(parents=True, exist_ok=True)
    interactions.to_csv(output_interaction_csv, index=False)
    output_paired_csv.parent.mkdir(parents=True, exist_ok=True)
    paired.to_csv(output_paired_csv, index=False)
    output_paired_winrate_csv.parent.mkdir(parents=True, exist_ok=True)
    paired_winrate.to_csv(output_paired_winrate_csv, index=False)
    output_paired_by_sample_csv.parent.mkdir(parents=True, exist_ok=True)
    paired_by_sample.to_csv(output_paired_by_sample_csv, index=False)
    output_agreement_csv.parent.mkdir(parents=True, exist_ok=True)
    agreement.to_csv(output_agreement_csv, index=False)

    print(summary.to_string(index=False))
    print("\nFactor effects:")
    print(factor_effects.to_string(index=False))
    print("\nRankings:")
    print(rankings.to_string(index=False))
    print("\nInteractions:")
    print(interactions.to_string(index=False))
    print("\nPaired comparisons:")
    print(paired.to_string(index=False))
    print("\nPaired win rates:")
    print(paired_winrate.to_string(index=False))
    print("\nPaired comparisons by sample:")
    print(paired_by_sample.to_string(index=False))
    print("\nResidual-vs-LOO agreement:")
    print(agreement.to_string(index=False))
    print(f"\nSaved detailed real-data results to {output_csv}")
    print(f"Saved real-data summary to {output_summary_csv}")
    print(f"Saved factor effects to {output_factor_csv}")
    print(f"Saved rankings to {output_ranking_csv}")
    print(f"Saved interactions to {output_interaction_csv}")
    print(f"Saved paired comparisons to {output_paired_csv}")
    print(f"Saved paired win rates to {output_paired_winrate_csv}")
    print(f"Saved paired-by-sample comparisons to {output_paired_by_sample_csv}")
    print(f"Saved residual-vs-LOO agreement to {output_agreement_csv}")


if __name__ == "__main__":
    main()
