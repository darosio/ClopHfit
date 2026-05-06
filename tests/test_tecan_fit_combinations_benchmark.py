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
