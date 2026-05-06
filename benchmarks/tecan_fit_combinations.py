#!/usr/bin/env python

"""Benchmark modular Tecan fit combinations on paired synthetic replicates."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import click
import numpy as np
import pandas as pd

from clophfit.testing.evaluation import calculate_bias, calculate_coverage, calculate_rmse
from clophfit.testing.fitter_test_utils import (
    apply_tecan_combination,
    build_factorized_tecan_fit_combinations,
)
from clophfit.testing.synthetic import make_dataset

FACTOR_COLUMNS = (
    "channels",
    "prefit",
    "final_stage",
    "weighting",
    "outlier_handling",
)
SUMMARY_METRIC_COLUMNS = (
    "success_rate",
    "finite_fit_rate",
    "mean_bias",
    "rmse",
    "coverage",
    "mean_k_error",
)


def _result_diagnostics(fit_result: object) -> tuple[float, float, float]:
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


def run_benchmark(
    *,
    n_replicates: int,
    seed: int,
    include_mcmc: bool,
    output_csv: Path | None,
) -> pd.DataFrame:
    """Run all configured combinations on the same synthetic replicates."""
    combinations = build_factorized_tecan_fit_combinations(
        final_stages=("huber", "odr", "mcmc_single") if include_mcmc else ("huber", "odr"),
    )
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []

    for replicate in range(n_replicates):
        replicate_seed = int(rng.integers(0, 2**32 - 1))
        ds, truth = make_dataset(
            randomize_signals=True,
            error_model="tecan",
            seed=replicate_seed,
        )
        for name, combination in combinations.items():
            ds_fresh = copy.deepcopy(ds)
            try:
                fit_result = apply_tecan_combination(ds_fresh, combination)
                result = fit_result.result
                k_fit = np.nan
                k_err = np.nan
                if result is not None and hasattr(result, "params") and "K" in result.params:
                    k_fit_value = result.params["K"].value
                    k_err_value = result.params["K"].stderr
                    k_fit = np.nan if k_fit_value is None else float(k_fit_value)
                    k_err = np.nan if k_err_value is None else float(k_err_value)
                residual_mean, residual_std, shapiro_p = _result_diagnostics(fit_result)
                success = result is not None and np.isfinite(k_fit)
            except NotImplementedError:
                success = False
                k_fit = np.nan
                k_err = np.nan
                residual_mean = np.nan
                residual_std = np.nan
                shapiro_p = np.nan
            except Exception:
                success = False
                k_fit = np.nan
                k_err = np.nan
                residual_mean = np.nan
                residual_std = np.nan
                shapiro_p = np.nan

            rows.append(
                {
                    "replicate": replicate,
                    "method": name,
                    "channels": "+".join(combination.channels),
                    "prefit": combination.prefit,
                    "final_stage": combination.final_stage,
                    "weighting": combination.weighting,
                    "outlier_handling": combination.outlier_handling or "none",
                    "truth_k": truth.K,
                    "estimated_k": k_fit,
                    "k_error": k_err,
                    "bias": k_fit - truth.K if np.isfinite(k_fit) else np.nan,
                    "success": success,
                    "residual_mean": residual_mean,
                    "residual_std": residual_std,
                    "shapiro_p": shapiro_p,
                }
            )

    df = pd.DataFrame(rows)
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
    return df


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize per-method benchmark metrics using finite successful fits."""
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
        truth_k = float(group["truth_k"].iloc[0])
        estimated = finite["estimated_k"].to_numpy(dtype=float)
        errors = finite["k_error"].to_numpy(dtype=float)
        residual_means = finite["residual_mean"].to_numpy(dtype=float)
        residual_stds = finite["residual_std"].to_numpy(dtype=float)
        shapiro_ps = finite["shapiro_p"].to_numpy(dtype=float)

        row: dict[str, Any] = {
            "method": str(method),
            "success_rate": float(successful.shape[0] / group.shape[0]),
            "finite_fit_rate": float(finite.shape[0] / group.shape[0]),
            "mean_bias": calculate_bias(estimated, truth_k),
            "rmse": calculate_rmse(estimated, truth_k),
            "coverage": calculate_coverage(estimated, errors, truth_k),
            "mean_k_error": float(np.mean(errors)) if errors.size else np.nan,
            "mean_residual_mean": (
                float(np.mean(residual_means[np.isfinite(residual_means)]))
                if np.isfinite(residual_means).any()
                else np.nan
            ),
            "mean_residual_std": (
                float(np.mean(residual_stds[np.isfinite(residual_stds)]))
                if np.isfinite(residual_stds).any()
                else np.nan
            ),
            "mean_shapiro_p": (
                float(np.mean(shapiro_ps[np.isfinite(shapiro_ps)]))
                if np.isfinite(shapiro_ps).any()
                else np.nan
            ),
        }
        for column, default in factor_defaults.items():
            row[column] = str(group[column].iloc[0]) if column in group.columns else default
        summary_rows.append(row)

    return pd.DataFrame(summary_rows).sort_values(
        ["success_rate", "finite_fit_rate", "rmse"],
        ascending=[False, False, True],
    )


def summarize_factor_effects(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate benchmark summaries by factor level for effect inspection."""
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
            for metric in SUMMARY_METRIC_COLUMNS:
                values = group[metric].to_numpy(dtype=float)
                finite_values = values[np.isfinite(values)]
                row[f"mean_{metric}"] = (
                    float(np.mean(finite_values)) if finite_values.size else np.nan
                )
            effect_rows.append(row)
    return pd.DataFrame(effect_rows).sort_values(["factor", "level"])


@click.command()
@click.option("--n-replicates", default=25, show_default=True, type=int)
@click.option("--seed", default=42, show_default=True, type=int)
@click.option("--include-mcmc/--no-include-mcmc", default=False, show_default=True)
@click.option(
    "--output-csv",
    type=click.Path(path_type=Path, dir_okay=False),
    default=Path("benchmarks/results/tecan_fit_combinations.csv"),
    show_default=True,
)
@click.option(
    "--output-factor-csv",
    type=click.Path(path_type=Path, dir_okay=False),
    default=Path("benchmarks/results/tecan_fit_factor_effects.csv"),
    show_default=True,
)
def main(
    n_replicates: int,
    seed: int,
    include_mcmc: bool,
    output_csv: Path,
    output_factor_csv: Path,
) -> None:
    """Run the Tecan fit-combination benchmark and print compact summaries."""
    df = run_benchmark(
        n_replicates=n_replicates,
        seed=seed,
        include_mcmc=include_mcmc,
        output_csv=output_csv,
    )
    summary = summarize_results(df)
    factor_effects = summarize_factor_effects(summary)
    output_factor_csv.parent.mkdir(parents=True, exist_ok=True)
    factor_effects.to_csv(output_factor_csv, index=False)
    print(summary.to_string(index=False))
    print("\nFactor effects:")
    print(factor_effects.to_string(index=False))
    print(f"\nSaved detailed results to {output_csv}")
    print(f"Saved factor effects to {output_factor_csv}")


if __name__ == "__main__":
    main()
