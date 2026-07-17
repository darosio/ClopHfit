"""Tests for benchmark evaluation helpers."""

from __future__ import annotations

import typing
from dataclasses import dataclass

import numpy as np
import pytest
from lmfit import Parameters  # type: ignore[import-untyped]

from clophfit.fitting.data_structures import FitResult
from clophfit.testing.evaluation import (
    calculate_bias,
    calculate_coverage,
    calculate_rmse,
    compare_methods_statistical,
    evaluate_residuals,
    extract_params,
    load_real_data_paths,
)


def test_calculate_bias_filters_non_finite_values() -> None:
    """Bias uses only finite estimates and returns nan when none remain."""
    estimates = np.array([7.1, np.nan, np.inf, 6.9])

    assert calculate_bias(estimates, true_value=7.0) == pytest.approx(0.0)
    assert np.isnan(calculate_bias(np.array([np.nan, np.inf]), true_value=7.0))


def test_calculate_rmse_filters_non_finite_values() -> None:
    """RMSE ignores failed estimates."""
    estimates = np.array([8.0, np.nan, 6.0, -np.inf])

    assert calculate_rmse(estimates, true_value=7.0) == pytest.approx(1.0)
    assert np.isnan(calculate_rmse(np.array([np.nan]), true_value=7.0))


def test_calculate_coverage_uses_confidence_interval_width() -> None:
    """Coverage counts intervals containing the true value."""
    estimates = np.array([7.0, 7.8, 6.2, np.nan])
    errors = np.array([0.1, 0.1, 1.0, 0.2])

    assert calculate_coverage(estimates, errors, true_value=7.0) == pytest.approx(2 / 3)
    assert calculate_coverage(
        estimates,
        errors,
        true_value=7.0,
        confidence=0.50,
    ) == pytest.approx(1 / 3)
    assert np.isnan(
        calculate_coverage(
            np.array([np.nan]),
            np.array([np.nan]),
            true_value=7.0,
        )
    )


def test_evaluate_residuals_handles_too_few_valid_points() -> None:
    """Shapiro-Wilk is skipped when fewer than three residuals remain."""
    result = evaluate_residuals(np.array([0.1, np.nan, 0.2]))

    assert set(result) == {"shapiro_stat", "shapiro_p", "mean", "std"}
    assert all(np.isnan(value) for value in result.values())


def test_evaluate_residuals_reports_summary_statistics() -> None:
    """Residual evaluation filters non-finite values before testing."""
    result = evaluate_residuals(np.array([-1.0, 0.0, 1.0, np.nan, np.inf]))

    assert 0.0 <= result["shapiro_p"] <= 1.0
    assert result["mean"] == pytest.approx(0.0)
    assert result["std"] == pytest.approx(np.sqrt(2 / 3))


@dataclass
class _ResultWithParams:
    params: Parameters


def test_extract_params_reads_lmfit_value_and_stderr() -> None:
    """Parameter extraction returns lmfit values with missing stderr as nan."""
    params = Parameters()
    params.add("K", value=7.2)
    params["K"].stderr = 0.15
    params.add("S0_a", value=100.0)

    fr: FitResult = FitResult(result=_ResultWithParams(params))

    assert extract_params(fr, "K") == (7.2, 0.15)
    val, err = extract_params(fr, "S0_a")
    assert val == 100.0
    assert np.isnan(err)


def test_extract_params_returns_nan_for_missing_result_or_parameter() -> None:
    """Extraction failures are represented as nan pairs."""
    params = Parameters()
    params.add("K", value=7.2)

    no_result: FitResult = FitResult()
    no_params: FitResult = FitResult(result=object())
    missing_param: FitResult = FitResult(result=_ResultWithParams(params))

    assert all(np.isnan(value) for value in extract_params(no_result, "K"))
    assert all(np.isnan(value) for value in extract_params(no_params, "K"))
    assert all(np.isnan(value) for value in extract_params(missing_param, "S1_a"))


def test_load_real_data_paths_finds_expected_test_datasets() -> None:
    """Real-data discovery finds checked-in Tecan fixtures from repo root."""
    datasets = load_real_data_paths()

    assert {"L1", "L2", "L4", "140220"}.issubset(datasets)
    for path in datasets.values():
        assert path.exists()
        assert (path / "list.pH.csv").exists()


def test_compare_methods_statistical_reports_insufficient_data(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Statistical comparison handles all-nan inputs without raising."""
    result = compare_methods_statistical(
        [np.nan],
        [1.0],
        method1_name="A",
        method2_name="B",
        verbose=True,
    )

    assert result["test"] == "mann_whitney_u"
    assert result["significant"] is False
    assert not result["better_method"]
    assert result["error"] == "Insufficient data"
    assert "Insufficient data" in capsys.readouterr().out


def test_compare_methods_statistical_can_skip_insufficient_data_output(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Quiet mode also applies to failed statistical comparisons."""
    result = compare_methods_statistical([np.nan], [1.0], verbose=False)

    assert result["error"] == "Insufficient data"
    assert not capsys.readouterr().out


def test_compare_methods_statistical_returns_mae_and_winner_without_printing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Method comparison uses finite absolute errors and honors verbosity."""
    result = compare_methods_statistical(
        [0.01, -0.02, np.nan, 0.03, -0.01, 0.02],
        [1.0, -1.1, 0.9, 1.2, -0.8],
        method1_name="tight",
        method2_name="loose",
        verbose=False,
    )

    assert result["test"] == "mann_whitney_u"
    assert result["mae1"] == pytest.approx(0.018)
    assert result["mae2"] == pytest.approx(1.0)
    assert typing.cast("float", result["p_value"]) < 0.05
    assert result["significant"] is True
    assert result["better_method"] == "tight"
    assert not capsys.readouterr().out


def test_compare_methods_statistical_verbose_significant_output(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verbose significant comparisons name the better method."""
    result = compare_methods_statistical(
        [1.0, -1.1, 0.9, 1.2, -0.8],
        [0.01, -0.02, 0.03, -0.01, 0.02],
        method1_name="loose",
        method2_name="tight",
        verbose=True,
    )

    assert result["significant"] is True
    assert result["better_method"] == "tight"
    output = capsys.readouterr().out
    assert "loose vs tight" in output
    assert "tight is better" in output


def test_compare_methods_statistical_marks_insignificant_as_equivalent() -> None:
    """Similar absolute-error distributions do not pick a winner."""
    result = compare_methods_statistical(
        [0.1, -0.2, 0.3],
        [0.11, -0.19, 0.31],
        verbose=True,
    )

    assert result["significant"] is False
    assert result["better_method"] == "Equivalent"
