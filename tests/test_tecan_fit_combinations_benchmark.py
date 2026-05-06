"""Tests for the Tecan fit-combination benchmark runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

MODULE_PATH = (
    Path(__file__).resolve().parent.parent / "benchmarks" / "tecan_fit_combinations.py"
)
SPEC = importlib.util.spec_from_file_location("tecan_fit_combinations", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

summarize_results = MODULE.summarize_results
summarize_factor_effects = MODULE.summarize_factor_effects


def test_summarize_factor_effects_aggregates_by_factor_level() -> None:
    """Factor summaries should aggregate metrics across combinations per level."""
    df = pd.DataFrame({
        "method": ["m1", "m2", "m3", "m4"],
        "channels": ["y1", "y1", "y1+y2", "y1+y2"],
        "prefit": ["huber", "lm", "huber", "lm"],
        "final_stage": ["odr", "huber", "odr", "huber"],
        "weighting": ["auto", "none", "auto", "none"],
        "outlier_handling": ["none", "none", "zscore:2.5:5", "zscore:2.5:5"],
        "success_rate": [1.0, 0.5, 0.75, 0.25],
        "finite_fit_rate": [1.0, 0.5, 0.5, 0.25],
        "mean_bias": [0.1, 0.3, 0.2, 0.4],
        "rmse": [0.2, 0.4, 0.3, 0.5],
        "coverage": [0.9, 0.7, 0.8, 0.6],
        "mean_k_error": [0.1, 0.3, 0.2, 0.4],
    })

    effects = summarize_factor_effects(df)
    channels_y1 = effects.loc[
        (effects["factor"] == "channels") & (effects["level"] == "y1")
    ].iloc[0]
    stage_odr = effects.loc[
        (effects["factor"] == "final_stage") & (effects["level"] == "odr")
    ].iloc[0]

    assert channels_y1["n_methods"] == 2
    assert channels_y1["mean_success_rate"] == pytest.approx(0.75)
    assert channels_y1["mean_rmse"] == pytest.approx(0.3)
    assert stage_odr["n_methods"] == 2
    assert stage_odr["mean_coverage"] == pytest.approx(0.85)


def test_summarize_results_retains_factor_columns() -> None:
    """Summary rows should carry factor metadata for downstream analysis."""
    df = pd.DataFrame({
        "method": ["combo_a", "combo_a", "combo_b"],
        "channels": ["y1+y2", "y1+y2", "y1"],
        "prefit": ["huber", "huber", "lm"],
        "final_stage": ["odr", "odr", "lm"],
        "weighting": ["auto", "auto", "none"],
        "outlier_handling": ["zscore:2.5:5", "zscore:2.5:5", "none"],
        "truth_k": [7.0, 7.0, 7.0],
        "estimated_k": [7.1, 6.9, 7.2],
        "k_error": [0.2, 0.2, 0.1],
        "bias": [0.1, -0.1, 0.2],
        "success": [True, True, True],
        "residual_mean": [0.2, 0.0, 0.3],
        "residual_std": [1.2, 1.0, 0.8],
        "shapiro_p": [0.6, 0.4, 0.7],
    })

    summary = summarize_results(df)
    row_a = summary.loc[summary["method"] == "combo_a"].iloc[0]
    row_b = summary.loc[summary["method"] == "combo_b"].iloc[0]

    assert row_a["channels"] == "y1+y2"
    assert row_a["prefit"] == "huber"
    assert row_a["final_stage"] == "odr"
    assert row_a["weighting"] == "auto"
    assert row_a["outlier_handling"] == "zscore:2.5:5"
    assert row_b["channels"] == "y1"
    assert row_b["weighting"] == "none"


def test_summarize_results_filters_non_finite_fit_metrics() -> None:
    """Summary metrics should use only successful finite estimates/errors."""
    df = pd.DataFrame({
        "method": ["a", "a", "a", "b", "b"],
        "truth_k": [7.0, 7.0, 7.0, 7.0, 7.0],
        "estimated_k": [7.1, np.nan, 6.9, 7.2, 7.0],
        "k_error": [0.2, 0.3, np.inf, 0.1, 0.2],
        "bias": [0.1, np.nan, -0.1, 0.2, 0.0],
        "success": [True, False, True, True, True],
        "residual_mean": [0.0, np.nan, 0.2, 0.1, -0.1],
        "residual_std": [1.0, np.nan, 2.0, 0.5, 0.4],
        "shapiro_p": [0.9, np.nan, 0.3, 0.8, 0.7],
    })

    summary = summarize_results(df)
    row_a = summary.loc[summary["method"] == "a"].iloc[0]
    row_b = summary.loc[summary["method"] == "b"].iloc[0]

    assert row_a["success_rate"] == 2 / 3
    assert row_a["finite_fit_rate"] == 1 / 3
    assert row_a["mean_bias"] == pytest.approx(0.1)
    assert row_a["rmse"] == pytest.approx(0.1)
    assert row_a["coverage"] == 1.0

    assert row_b["success_rate"] == 1.0
    assert row_b["finite_fit_rate"] == 1.0
    assert row_b["mean_bias"] == pytest.approx(0.1)
    assert row_b["rmse"] == pytest.approx(np.sqrt((0.2**2 + 0.0**2) / 2))
    assert row_b["coverage"] == 0.5


def test_summarize_results_propagates_residual_summary() -> None:
    """Residual summary columns should average finite residual diagnostics."""
    df = pd.DataFrame({
        "method": ["combo", "combo", "combo"],
        "truth_k": [7.0, 7.0, 7.0],
        "estimated_k": [7.0, 7.1, 6.9],
        "k_error": [0.1, 0.2, 0.3],
        "bias": [0.0, 0.1, -0.1],
        "success": [True, True, True],
        "residual_mean": [0.2, np.nan, -0.1],
        "residual_std": [1.2, np.nan, 0.8],
        "shapiro_p": [0.6, np.nan, 0.4],
    })

    summary = summarize_results(df)
    row = summary.iloc[0]

    assert row["mean_residual_mean"] == 0.05
    assert row["mean_residual_std"] == 1.0
    assert row["mean_shapiro_p"] == 0.5
