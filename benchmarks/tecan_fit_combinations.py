#!/usr/bin/env python

"""Benchmark modular Tecan fit combinations on paired synthetic replicates."""

from __future__ import annotations

import copy
from pathlib import Path

import click
import numpy as np
import pandas as pd

from clophfit.testing.evaluation import calculate_bias, calculate_coverage, calculate_rmse
from clophfit.testing.fitter_test_utils import (
    apply_tecan_combination,
    build_tecan_fit_combinations,
    k_from_result,
)
from clophfit.testing.synthetic import make_dataset


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
    combinations = build_tecan_fit_combinations(include_mcmc=include_mcmc)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | str | bool]] = []

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
                k_fit, k_err = k_from_result(fit_result)
                residual_mean, residual_std, shapiro_p = _result_diagnostics(fit_result)
                success = fit_result.result is not None and k_fit is not None
            except NotImplementedError:
                success = False
                k_fit = None
                k_err = None
                residual_mean = np.nan
                residual_std = np.nan
                shapiro_p = np.nan
            except Exception:
                success = False
                k_fit = None
                k_err = None
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
                    "truth_k": truth.K,
                    "estimated_k": np.nan if k_fit is None else float(k_fit),
                    "k_error": np.nan if k_err is None else float(k_err),
                    "bias": np.nan if k_fit is None else float(k_fit - truth.K),
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
    summary_rows: list[dict[str, float | str]] = []
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

        summary_rows.append(
            {
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
        )

    return pd.DataFrame(summary_rows).sort_values(
        ["success_rate", "finite_fit_rate", "rmse"],
        ascending=[False, False, True],
    )


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
def main(
    n_replicates: int,
    seed: int,
    include_mcmc: bool,
    output_csv: Path,
) -> None:
    """Run the Tecan fit-combination benchmark and print a compact summary."""
    df = run_benchmark(
        n_replicates=n_replicates,
        seed=seed,
        include_mcmc=include_mcmc,
        output_csv=output_csv,
    )
    summary = summarize_results(df)
    print(summary.to_string(index=False))
    print(f"\nSaved detailed results to {output_csv}")


if __name__ == "__main__":
    main()
