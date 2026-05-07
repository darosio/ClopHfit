"""Tests for the real-data Tecan benchmark."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from clophfit.testing.fitter_test_utils import TecanFitCombination
from clophfit.testing.synthetic import make_dataset

MODULE_PATH = (
    Path(__file__).resolve().parent.parent
    / "benchmarks"
    / "tecan_real_data_benchmark.py"
)
SPEC = importlib.util.spec_from_file_location("tecan_real_data_benchmark", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

load_real_tecan_curves = MODULE.load_real_tecan_curves
leave_one_out_rmse = MODULE.leave_one_out_rmse
run_real_data_benchmark = MODULE.run_real_data_benchmark
summarize_real_results = MODULE.summarize_real_results
summarize_real_factor_effects = MODULE.summarize_real_factor_effects
rank_real_methods = MODULE.rank_real_methods
summarize_real_interactions = MODULE.summarize_real_interactions
paired_method_comparisons = MODULE.paired_method_comparisons
paired_win_rates = MODULE.paired_win_rates
paired_method_comparisons_by_sample = MODULE.paired_method_comparisons_by_sample
paired_metric_agreement = MODULE.paired_metric_agreement
resolve_output_paths = MODULE.resolve_output_paths


def test_load_real_tecan_curves_extracts_sample_metadata() -> None:
    """Real Tecan wells should retain their sample labels."""
    curves = load_real_tecan_curves(
        list_file=Path("tests/Tecan/140220/list.pH.csv"),
        additions_file=Path("tests/Tecan/140220/additions.pH"),
        scheme_file=Path("tests/Tecan/140220/scheme.txt"),
        is_ph=True,
    )

    assert curves
    assert all(curve.sample for curve in curves)
    assert any(curve.sample == "G03" for curve in curves)


def test_leave_one_out_rmse_returns_finite_value_for_realistic_synthetic_curve() -> (
    None
):
    """LOO RMSE should be finite for a realistic synthetic Tecan curve."""
    dataset, _truth = make_dataset(seed=42, randomize_signals=True, error_model="tecan")
    combination = TecanFitCombination(
        name="y2_huber_auto",
        channels=("y2",),
        prefit="huber",
        final_stage="huber",
        weighting="auto",
    )

    loo_rmse = leave_one_out_rmse(dataset, combination)

    assert loo_rmse == pytest.approx(loo_rmse)
    assert loo_rmse > 0


def test_leave_one_out_rmse_honors_max_points() -> None:
    """LOO RMSE should support bounded evaluation for faster smoke runs."""
    dataset, _truth = make_dataset(seed=7, randomize_signals=True, error_model="tecan")
    combination = TecanFitCombination(
        name="y2_huber_auto",
        channels=("y2",),
        prefit="huber",
        final_stage="huber",
        weighting="auto",
    )

    limited = leave_one_out_rmse(dataset, combination, max_points=2)
    full = leave_one_out_rmse(dataset, combination)

    assert limited == pytest.approx(limited)
    assert full == pytest.approx(full)
    assert limited > 0
    assert full > 0


def test_run_real_data_benchmark_supports_filters_and_skip_loo(tmp_path: Path) -> None:
    """Benchmark runner should support practical filtering and throttling."""
    output_csv = tmp_path / "results.csv"

    df = run_real_data_benchmark(
        list_file=Path("tests/Tecan/140220/list.pH.csv"),
        additions_file=Path("tests/Tecan/140220/additions.pH"),
        scheme_file=Path("tests/Tecan/140220/scheme.txt"),
        is_ph=True,
        include_mcmc=False,
        output_csv=output_csv,
        max_wells=2,
        samples=("G03",),
        channels=("y1+y2",),
        final_stages=("huber",),
        skip_loo=True,
    )

    assert output_csv.exists()
    assert not df.empty
    assert set(df["sample"]) == {"G03"}
    assert set(df["channels"]) == {"y1+y2"}
    assert set(df["final_stage"]) == {"huber"}
    assert df["well"].nunique() <= 2
    assert df["loo_rmse"].isna().all()


def test_run_real_data_benchmark_supports_multiple_weightings_and_mcmc_stages(
    tmp_path: Path,
) -> None:
    """Real benchmark should expose extra weighting schemes and multi-noise MCMC stages."""
    output_csv = tmp_path / "results.csv"

    df = run_real_data_benchmark(
        list_file=Path("tests/Tecan/140220/list.pH.csv"),
        additions_file=Path("tests/Tecan/140220/additions.pH"),
        scheme_file=Path("tests/Tecan/140220/scheme.txt"),
        is_ph=True,
        include_mcmc=True,
        output_csv=output_csv,
        max_wells=1,
        samples=("G03",),
        channels=("y2", "y1+y2"),
        final_stages=(
            "huber",
            "mcmc_single",
            "mcmc_multi-noise",
            "mcmc_multi-noise-xrw",
        ),
        weightings=("auto", "none", "calibrated"),
        skip_loo=True,
    )

    assert output_csv.exists()
    assert not df.empty
    assert {"auto", "none", "calibrated"} <= set(df["weighting"])
    assert {
        "huber",
        "mcmc_single",
        "mcmc_multi-noise",
        "mcmc_multi-noise-xrw",
    } <= set(df["final_stage"])
    calibrated_multi_noise = df.loc[
        (df["weighting"] == "calibrated") & (df["final_stage"] == "mcmc_multi-noise")
    ]
    assert not calibrated_multi_noise.empty
    assert (~calibrated_multi_noise["success"].astype(bool)).all()


def test_resolve_output_paths_creates_named_outputs_from_one_directory(
    tmp_path: Path,
) -> None:
    """One output directory should expand to all standard real-benchmark artifacts."""
    paths = resolve_output_paths(tmp_path / "real_benchmark")

    assert (
        paths["output_csv"]
        == tmp_path / "real_benchmark" / "tecan_real_data_results.csv"
    )
    assert (
        paths["output_summary_csv"]
        == tmp_path / "real_benchmark" / "tecan_real_data_summary.csv"
    )
    assert (
        paths["output_factor_csv"]
        == tmp_path / "real_benchmark" / "tecan_real_data_factor_effects.csv"
    )
    assert (
        paths["output_ranking_csv"]
        == tmp_path / "real_benchmark" / "tecan_real_data_rankings.csv"
    )
    assert (
        paths["output_interaction_csv"]
        == tmp_path / "real_benchmark" / "tecan_real_data_interactions.csv"
    )
    assert (
        paths["output_paired_csv"]
        == tmp_path / "real_benchmark" / "tecan_real_data_paired.csv"
    )
    assert (
        paths["output_paired_winrate_csv"]
        == tmp_path / "real_benchmark" / "tecan_real_data_paired_winrates.csv"
    )
    assert (
        paths["output_paired_by_sample_csv"]
        == tmp_path / "real_benchmark" / "tecan_real_data_paired_by_sample.csv"
    )
    assert (
        paths["output_agreement_csv"]
        == tmp_path / "real_benchmark" / "tecan_real_data_agreement.csv"
    )


def test_module_docstring_mentions_y1_y2_and_output_dir_quick_start() -> None:
    """README-style guidance should mention y1+y2 and simplified output-dir usage."""
    doc = MODULE.__doc__ or ""

    assert "y1+y2" in doc
    assert "--output-dir" in doc
    assert "agreement" in doc


def test_summarize_real_results_retains_loo_rmse_and_factor_columns() -> None:
    """Real-data summaries should preserve factor metadata and LOO RMSE."""
    df = pd.DataFrame({
        "method": ["m1", "m1", "m2"],
        "sample": ["G03", "G03", "G04"],
        "channels": ["y1", "y1", "y2"],
        "prefit": ["huber", "huber", "lm"],
        "final_stage": ["odr", "odr", "huber"],
        "weighting": ["auto", "auto", "none"],
        "outlier_handling": ["none", "none", "none"],
        "estimated_k": [7.1, 6.9, 7.2],
        "k_error": [0.2, 0.1, 0.3],
        "success": [True, True, True],
        "residual_mean": [0.2, 0.0, -0.1],
        "residual_std": [1.5, 1.0, 0.8],
        "shapiro_p": [0.5, 0.7, 0.6],
        "loo_rmse": [3.0, 2.0, 1.5],
    })

    summary = summarize_real_results(df)
    row_m1 = summary.loc[summary["method"] == "m1"].iloc[0]

    assert row_m1["channels"] == "y1"
    assert row_m1["prefit"] == "huber"
    assert row_m1["final_stage"] == "odr"
    assert row_m1["mean_loo_rmse"] == pytest.approx(2.5)
    assert row_m1["mean_residual_std"] == pytest.approx(1.25)


def test_summarize_real_factor_effects_aggregates_loo_rmse() -> None:
    """Factor summaries should average real-data metrics across levels."""
    summary_df = pd.DataFrame({
        "method": ["m1", "m2", "m3"],
        "channels": ["y1", "y1", "y2"],
        "prefit": ["huber", "lm", "huber"],
        "final_stage": ["odr", "huber", "odr"],
        "weighting": ["auto", "none", "auto"],
        "outlier_handling": ["none", "none", "none"],
        "success_rate": [1.0, 0.5, 1.0],
        "finite_fit_rate": [1.0, 0.5, 1.0],
        "mean_k_error": [0.1, 0.3, 0.2],
        "mean_residual_mean": [0.0, 0.1, -0.1],
        "mean_residual_std": [1.0, 1.5, 0.8],
        "mean_shapiro_p": [0.6, 0.4, 0.7],
        "mean_loo_rmse": [2.0, 3.0, 1.5],
    })

    effects = summarize_real_factor_effects(summary_df)
    channels_y1 = effects.loc[
        (effects["factor"] == "channels") & (effects["level"] == "y1")
    ].iloc[0]

    assert channels_y1["n_methods"] == 2
    assert channels_y1["mean_mean_loo_rmse"] == pytest.approx(2.5)
    assert channels_y1["mean_mean_residual_std"] == pytest.approx(1.25)


def test_rank_real_methods_orders_by_metric_priority() -> None:
    """Method ranking should sort by the chosen real-data metric ascending."""
    summary_df = pd.DataFrame({
        "method": ["m1", "m2", "m3"],
        "mean_loo_rmse": [3.0, 1.5, 2.0],
        "mean_residual_std": [1.2, 0.9, 1.1],
        "finite_fit_rate": [0.9, 0.8, 1.0],
        "channels": ["y1", "y1+y2", "y2"],
    })

    ranked = rank_real_methods(summary_df, metric="mean_loo_rmse")

    assert list(ranked["method"]) == ["m2", "m3", "m1"]
    assert list(ranked["rank"]) == [1, 2, 3]


def test_summarize_real_interactions_aggregates_factor_pairs() -> None:
    """Interaction summaries should aggregate pairwise factor combinations."""
    summary_df = pd.DataFrame({
        "method": ["m1", "m2", "m3", "m4"],
        "channels": ["y1", "y1", "y1+y2", "y1+y2"],
        "final_stage": ["huber", "odr", "huber", "odr"],
        "prefit": ["huber", "huber", "lm", "lm"],
        "weighting": ["auto", "auto", "none", "none"],
        "outlier_handling": ["none", "none", "none", "none"],
        "success_rate": [1.0, 0.5, 0.75, 0.25],
        "finite_fit_rate": [1.0, 0.5, 0.5, 0.25],
        "mean_loo_rmse": [2.0, 3.5, 1.8, 2.8],
        "mean_residual_std": [1.0, 1.4, 0.8, 1.1],
    })

    interactions = summarize_real_interactions(
        summary_df,
        factor_pairs=(("channels", "final_stage"),),
    )
    row = interactions.loc[
        (interactions["factor_a"] == "channels")
        & (interactions["level_a"] == "y1")
        & (interactions["factor_b"] == "final_stage")
        & (interactions["level_b"] == "odr")
    ].iloc[0]

    assert row["n_methods"] == 1
    assert row["mean_mean_loo_rmse"] == pytest.approx(3.5)
    assert row["mean_mean_residual_std"] == pytest.approx(1.4)


def test_paired_method_comparisons_use_same_wells_for_deltas() -> None:
    """Paired comparisons should compare methods on shared wells only."""
    df = pd.DataFrame({
        "well": ["A01", "A02", "A01", "A02", "A03"],
        "sample": ["G03", "G03", "G03", "G03", "G03"],
        "method": ["m1", "m1", "m2", "m2", "m2"],
        "estimated_k": [7.0, 7.4, 6.8, 7.0, 8.0],
        "k_error": [0.2, 0.3, 0.15, 0.25, 0.4],
        "success": [True, True, True, True, True],
        "residual_std": [1.2, 1.0, 0.9, 0.8, 0.7],
        "loo_rmse": [2.5, 2.0, 1.5, 1.8, 3.0],
    })

    paired = paired_method_comparisons(df, method_pairs=(("m1", "m2"),))
    row = paired.iloc[0]

    assert row["method_a"] == "m1"
    assert row["method_b"] == "m2"
    assert row["n_pairs"] == 2
    assert row["mean_delta_loo_rmse"] == pytest.approx(0.6)
    assert row["mean_delta_residual_std"] == pytest.approx(0.25)
    assert row["mean_abs_delta_k"] == pytest.approx(0.3)


def test_paired_win_rates_count_metric_wins_on_shared_wells() -> None:
    """Win-rate tables should count per-well paired wins for each metric."""
    df = pd.DataFrame({
        "well": ["A01", "A02", "A03", "A01", "A02", "A03"],
        "sample": ["G03", "G03", "G03", "G03", "G03", "G03"],
        "method": ["m1", "m1", "m1", "m2", "m2", "m2"],
        "estimated_k": [7.0, 7.1, 7.2, 6.9, 7.0, 7.1],
        "residual_std": [1.1, 0.7, 0.9, 1.0, 0.8, 1.2],
        "loo_rmse": [2.0, 1.5, 2.2, 2.5, 1.0, 2.0],
    })

    wins = paired_win_rates(df, method_pairs=(("m1", "m2"),))
    row = wins.iloc[0]

    assert row["method_a"] == "m1"
    assert row["method_b"] == "m2"
    assert row["n_pairs"] == 3
    assert row["win_rate_loo_rmse_a"] == pytest.approx(1 / 3)
    assert row["win_rate_residual_std_a"] == pytest.approx(2 / 3)


def test_paired_method_comparisons_by_sample_stratifies_shared_wells() -> None:
    """Sample-stratified paired summaries should split shared-well deltas by sample."""
    df = pd.DataFrame({
        "well": ["A01", "A02", "B01", "B02", "A01", "A02", "B01", "B02"],
        "sample": ["G03", "G03", "G04", "G04", "G03", "G03", "G04", "G04"],
        "method": ["m1", "m1", "m1", "m1", "m2", "m2", "m2", "m2"],
        "estimated_k": [7.0, 7.3, 6.8, 7.1, 6.9, 7.0, 7.2, 7.5],
        "residual_std": [1.0, 1.1, 0.7, 0.8, 0.9, 0.8, 1.0, 1.1],
        "loo_rmse": [2.1, 2.4, 1.2, 1.4, 1.5, 1.9, 1.8, 1.6],
    })

    by_sample = paired_method_comparisons_by_sample(df, method_pairs=(("m1", "m2"),))
    g03 = by_sample.loc[by_sample["sample"] == "G03"].iloc[0]
    g04 = by_sample.loc[by_sample["sample"] == "G04"].iloc[0]

    assert g03["n_pairs"] == 2
    assert g03["mean_delta_loo_rmse"] == pytest.approx(0.55)
    assert g04["n_pairs"] == 2
    assert g04["mean_delta_residual_std"] == pytest.approx(-0.3)


def test_paired_metric_agreement_reports_when_residual_and_loo_pick_same_winner() -> (
    None
):
    """Agreement summaries should quantify whether residual_std and LOO agree."""
    df = pd.DataFrame({
        "well": ["A01", "A02", "A03", "A04", "A01", "A02", "A03", "A04"],
        "sample": ["G03", "G03", "G03", "G03", "G03", "G03", "G03", "G03"],
        "method": ["m1", "m1", "m1", "m1", "m2", "m2", "m2", "m2"],
        "estimated_k": [7.0, 7.1, 7.2, 7.3, 6.9, 7.0, 7.1, 7.4],
        "residual_std": [1.0, 0.8, 1.1, 0.7, 1.2, 0.6, 0.9, 0.7],
        "loo_rmse": [1.5, 1.2, 2.0, 1.0, 1.8, 1.4, 1.8, 1.0],
    })

    agreement = paired_metric_agreement(df, method_pairs=(("m1", "m2"),))
    row = agreement.iloc[0]

    assert row["method_a"] == "m1"
    assert row["method_b"] == "m2"
    assert row["n_pairs"] == 4
    assert row["n_valid_pairs"] == 4
    assert row["agreement_rate"] == pytest.approx(0.75)
    assert row["both_prefer_a_rate"] == pytest.approx(0.25)
    assert row["both_prefer_b_rate"] == pytest.approx(0.25)
    assert row["disagreement_rate"] == pytest.approx(0.25)


def test_real_benchmark_module_exposes_no_simulated_preview_hook() -> None:
    """Real-data benchmark module should not keep synthetic preview helpers."""
    assert not hasattr(MODULE, "plot_simulated_examples")
