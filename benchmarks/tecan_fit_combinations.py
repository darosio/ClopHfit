#!/usr/bin/env python
#
"""Benchmark modular Tecan fit combinations on paired synthetic replicates."""

from __future__ import annotations

import copy
from pathlib import Path

import click
import numpy as np
import pandas as pd

from clophfit.testing.fitter_test_utils import (
    apply_tecan_combination,
    build_tecan_fit_combinations,
    k_from_result,
)
from clophfit.testing.synthetic import make_dataset


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
                success = fit_result.result is not None and k_fit is not None
            except NotImplementedError:
                success = False
                k_fit = None
                k_err = None
            except Exception:
                success = False
                k_fit = None
                k_err = None

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
                }
            )

    df = pd.DataFrame(rows)
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
    return df


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
    summary = (
        df.groupby("method", as_index=False)
        .agg(
            success_rate=("success", "mean"),
            mean_bias=("bias", "mean"),
            mean_k_error=("k_error", "mean"),
        )
        .sort_values(["success_rate", "mean_k_error"], ascending=[False, True])
    )
    print(summary.to_string(index=False))
    print(f"\nSaved detailed results to {output_csv}")


if __name__ == "__main__":
    main()
