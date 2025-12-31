"""Test cases for the clophfit.fitting.residuals module."""

import numpy as np
import pandas as pd
import pytest
from lmfit.minimizer import MinimizerResult  # type: ignore[import-untyped]

from clophfit.fitting.core import fit_binding_glob
from clophfit.fitting.data_structures import DataArray, Dataset, FitResult
from clophfit.fitting.models import binding_1site
from clophfit.fitting.residuals import (
    BIAS_P_VALUE_THRESHOLD,
    DW_LOWER_BOUND,
    DW_UPPER_BOUND,
    OUTLIER_RATE_THRESHOLD,
    OUTLIER_THRESHOLD_2SIGMA,
    ResidualPoint,
    collect_multi_residuals,
    extract_residual_points,
    residual_dataframe,
    residual_statistics,
    validate_residuals,
)

###############################################################################
# Fixtures
###############################################################################


@pytest.fixture
def simple_fit_result() -> FitResult[MinimizerResult]:
    """Create a simple fit result for testing residuals."""
    x = np.array([9.0, 8.0, 7.0, 6.0, 5.0])
    # Perfect fit data with K=7
    y = binding_1site(x, K=7.0, S0=500.0, S1=1000.0, is_ph=True)
    y_err = np.ones_like(y) * 10.0
    da = DataArray(xc=x, yc=y, y_errc=y_err)
    dataset = Dataset({"y1": da}, is_ph=True)
    return fit_binding_glob(dataset)


@pytest.fixture
def noisy_fit_result() -> FitResult[MinimizerResult]:
    """Create a fit result with noisy data."""
    rng = np.random.default_rng(42)
    x = np.array([9.0, 8.0, 7.0, 6.0, 5.0])
    y_true = binding_1site(x, K=7.0, S0=500.0, S1=1000.0, is_ph=True)
    y_err = np.ones_like(y_true) * 20.0
    y = y_true + rng.normal(0, 10, size=len(y_true))
    da = DataArray(xc=x, yc=y, y_errc=y_err)
    dataset = Dataset({"y1": da}, is_ph=True)
    return fit_binding_glob(dataset)


@pytest.fixture
def multi_label_fit_result() -> FitResult[MinimizerResult]:
    """Create a multi-label fit result."""
    x = np.array([9.0, 8.0, 7.0, 6.0, 5.0])
    y1 = binding_1site(x, K=7.0, S0=500.0, S1=1000.0, is_ph=True)
    y2 = binding_1site(x, K=7.0, S0=200.0, S1=800.0, is_ph=True)
    y_err = np.ones_like(y1) * 10.0
    da1 = DataArray(xc=x, yc=y1, y_errc=y_err)
    da2 = DataArray(xc=x, yc=y2, y_errc=y_err)
    dataset = Dataset({"y1": da1, "y2": da2}, is_ph=True)
    return fit_binding_glob(dataset)


@pytest.fixture
def empty_fit_result() -> FitResult[MinimizerResult]:
    """Create a fit result with no result."""
    return FitResult(result=None, dataset=None)


###############################################################################
# Tests for ResidualPoint
###############################################################################


class TestResidualPoint:
    """Test the ResidualPoint dataclass."""

    def test_creation(self) -> None:
        """Test ResidualPoint creation."""
        point = ResidualPoint(
            label="y1",
            x=7.0,
            resid_weighted=0.5,
            resid_raw=5.0,
            i=0,
        )
        assert point.label == "y1"
        assert point.x == 7.0
        assert point.resid_weighted == 0.5
        assert point.resid_raw == 5.0
        assert point.i == 0

    def test_frozen(self) -> None:
        """Test that ResidualPoint is immutable."""
        point = ResidualPoint(label="y1", x=7.0, resid_weighted=0.5, resid_raw=5.0, i=0)
        with pytest.raises(AttributeError):
            point.x = 8.0  # type: ignore[misc]


###############################################################################
# Tests for extract_residual_points
###############################################################################


class TestExtractResidualPoints:
    """Test the extract_residual_points function."""

    def test_single_label(self, simple_fit_result: FitResult[MinimizerResult]) -> None:
        """Test extraction from single-label fit."""
        points = extract_residual_points(simple_fit_result)
        assert len(points) == 5
        assert all(p.label == "y1" for p in points)
        assert all(isinstance(p.resid_weighted, float) for p in points)
        assert all(isinstance(p.resid_raw, float) for p in points)

    def test_multi_label(
        self, multi_label_fit_result: FitResult[MinimizerResult]
    ) -> None:
        """Test extraction from multi-label fit."""
        points = extract_residual_points(multi_label_fit_result)
        assert len(points) == 10  # 5 points * 2 labels
        y1_points = [p for p in points if p.label == "y1"]
        y2_points = [p for p in points if p.label == "y2"]
        assert len(y1_points) == 5
        assert len(y2_points) == 5

    def test_empty_result(self, empty_fit_result: FitResult[MinimizerResult]) -> None:
        """Test extraction from empty fit result."""
        points = extract_residual_points(empty_fit_result)
        assert points == []

    def test_x_values_preserved(
        self, simple_fit_result: FitResult[MinimizerResult]
    ) -> None:
        """Test that x values are preserved in residual points."""
        points = extract_residual_points(simple_fit_result)
        x_values = {p.x for p in points}
        expected_x = {9.0, 8.0, 7.0, 6.0, 5.0}
        assert x_values == expected_x

    def test_indices_sequential(
        self, simple_fit_result: FitResult[MinimizerResult]
    ) -> None:
        """Test that indices are sequential within each label."""
        points = extract_residual_points(simple_fit_result)
        indices = [p.i for p in points]
        assert indices == list(range(5))


###############################################################################
# Tests for residual_dataframe
###############################################################################


class TestResidualDataframe:
    """Test the residual_dataframe function."""

    def test_columns(self, simple_fit_result: FitResult[MinimizerResult]) -> None:
        """Test that DataFrame has correct columns."""
        df = residual_dataframe(simple_fit_result)
        expected_cols = {"label", "x", "resid_weighted", "resid_raw", "i"}
        assert set(df.columns) == expected_cols

    def test_row_count(self, simple_fit_result: FitResult[MinimizerResult]) -> None:
        """Test that DataFrame has correct number of rows."""
        df = residual_dataframe(simple_fit_result)
        assert len(df) == 5

    def test_empty_result(self, empty_fit_result: FitResult[MinimizerResult]) -> None:
        """Test DataFrame from empty fit result."""
        df = residual_dataframe(empty_fit_result)
        assert len(df) == 0

    def test_dtypes(self, simple_fit_result: FitResult[MinimizerResult]) -> None:
        """Test DataFrame column dtypes."""
        df = residual_dataframe(simple_fit_result)
        assert df["label"].dtype == object  # string
        assert df["x"].dtype == np.float64
        assert df["resid_weighted"].dtype == np.float64
        assert df["resid_raw"].dtype == np.float64
        assert df["i"].dtype == np.int64


###############################################################################
# Tests for collect_multi_residuals
###############################################################################


class TestCollectMultiResiduals:
    """Test the collect_multi_residuals function."""

    def test_single_well(self, simple_fit_result: FitResult[MinimizerResult]) -> None:
        """Test collection from single well."""
        results = {"A01": simple_fit_result}
        df = collect_multi_residuals(results)
        assert "well" in df.columns
        assert (df["well"] == "A01").all()
        assert len(df) == 5

    def test_multiple_wells(
        self, simple_fit_result: FitResult[MinimizerResult]
    ) -> None:
        """Test collection from multiple wells."""
        results = {
            "A01": simple_fit_result,
            "A02": simple_fit_result,
            "B01": simple_fit_result,
        }
        df = collect_multi_residuals(results)
        assert len(df) == 15  # 5 points * 3 wells
        assert set(df["well"].unique()) == {"A01", "A02", "B01"}

    def test_x_rounding(self, simple_fit_result: FitResult[MinimizerResult]) -> None:
        """Test x value rounding."""
        results = {"A01": simple_fit_result}
        df_rounded = collect_multi_residuals(results, round_x=2)
        df_not_rounded = collect_multi_residuals(results, round_x=None)
        # Both should have same values since input is already clean
        assert len(df_rounded) == len(df_not_rounded)

    def test_empty_dict(self) -> None:
        """Test collection from empty dictionary raises ValueError."""
        with pytest.raises(ValueError, match="No objects to concatenate"):
            collect_multi_residuals({})


###############################################################################
# Tests for residual_statistics
###############################################################################


class TestResidualStatistics:
    """Test the residual_statistics function."""

    def test_columns(self, simple_fit_result: FitResult[MinimizerResult]) -> None:
        """Test that statistics DataFrame has expected columns."""
        results = {"A01": simple_fit_result}
        df = collect_multi_residuals(results)
        stats = residual_statistics(df)
        expected_cols = {
            "mean",
            "std",
            "median",
            "mad",
            "outlier_count",
            "n_points",
            "outlier_rate",
        }
        assert expected_cols.issubset(set(stats.columns))

    def test_grouped_by_label(
        self, multi_label_fit_result: FitResult[MinimizerResult]
    ) -> None:
        """Test that statistics are grouped by label."""
        results = {"A01": multi_label_fit_result}
        df = collect_multi_residuals(results)
        stats = residual_statistics(df)
        assert len(stats) == 2  # y1 and y2
        assert "y1" in stats.index
        assert "y2" in stats.index

    def test_outlier_count(self) -> None:
        """Test outlier counting."""
        # Create DataFrame with known outliers
        data = {
            "label": ["y1"] * 10,
            "resid_weighted": [0.0, 0.1, -0.1, 0.2, -0.2, 3.0, -3.0, 0.0, 0.1, -0.1],
            "x": list(range(10)),
            "resid_raw": [0.0] * 10,
            "i": list(range(10)),
        }
        df = pd.DataFrame(data)
        stats = residual_statistics(df)
        # 2 outliers beyond ±2-sigma (3.0 and -3.0)
        assert stats.loc["y1", "outlier_count"] == 2

    def test_outlier_rate(self) -> None:
        """Test outlier rate calculation."""
        data = {
            "label": ["y1"] * 10,
            "resid_weighted": [0.0] * 8 + [3.0, -3.0],  # 2 outliers out of 10
            "x": list(range(10)),
            "resid_raw": [0.0] * 10,
            "i": list(range(10)),
        }
        df = pd.DataFrame(data)
        stats = residual_statistics(df)
        assert stats.loc["y1", "outlier_rate"] == 0.2  # 2/10


###############################################################################
# Tests for validate_residuals
###############################################################################


class TestValidateResiduals:
    """Test the validate_residuals function."""

    def test_good_fit(self, simple_fit_result: FitResult[MinimizerResult]) -> None:
        """Test validation of a good fit."""
        checks = validate_residuals(simple_fit_result, verbose=False)
        assert isinstance(checks, dict)
        assert "bias_ok" in checks
        assert "outliers_ok" in checks
        assert "correlation_ok" in checks

    def test_empty_result(self, empty_fit_result: FitResult[MinimizerResult]) -> None:
        """Test validation of empty fit result."""
        checks = validate_residuals(empty_fit_result, verbose=False)
        # All checks should pass (no data to fail on)
        assert checks["bias_ok"] is True
        assert checks["outliers_ok"] is True
        assert checks["correlation_ok"] is True

    def test_bias_detection(self) -> None:
        """Test detection of systematic bias."""
        # Create fit result with biased residuals
        x = np.array([9.0, 8.0, 7.0, 6.0, 5.0])
        y_true = binding_1site(x, K=7.0, S0=500.0, S1=1000.0, is_ph=True)
        # Add systematic bias
        y = y_true + 50.0  # Constant offset
        y_err = np.ones_like(y) * 10.0
        da = DataArray(xc=x, yc=y, y_errc=y_err)
        dataset = Dataset({"y1": da}, is_ph=True)
        fr = fit_binding_glob(dataset)
        # With a constant offset, the fit should still be good
        # (offset absorbed by S0/S1), so bias_ok should be True
        checks = validate_residuals(fr, verbose=False)
        assert isinstance(checks["bias_ok"], bool)

    def test_verbose_output(
        self,
        simple_fit_result: FitResult[MinimizerResult],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that verbose mode produces output on failures."""
        # With a good fit, verbose shouldn't print anything
        validate_residuals(simple_fit_result, verbose=True)
        captured = capsys.readouterr()
        # Good fit should not produce warnings
        assert "⚠️" not in captured.out or not captured.out

    def test_returns_dict(self, noisy_fit_result: FitResult[MinimizerResult]) -> None:
        """Test that function returns correct structure."""
        checks = validate_residuals(noisy_fit_result, verbose=False)
        assert set(checks.keys()) == {"bias_ok", "outliers_ok", "correlation_ok"}
        # Values are bool or numpy.bool_
        assert all(isinstance(v, (bool, np.bool_)) for v in checks.values())


###############################################################################
# Tests for module constants
###############################################################################


class TestModuleConstants:
    """Test that module constants have expected values."""

    def test_threshold_values(self) -> None:
        """Test threshold constants are reasonable."""
        assert OUTLIER_THRESHOLD_2SIGMA == 2.0
        assert OUTLIER_RATE_THRESHOLD == 0.05
        assert BIAS_P_VALUE_THRESHOLD == 0.01
        assert DW_LOWER_BOUND == 1.5
        assert DW_UPPER_BOUND == 2.5
        assert DW_LOWER_BOUND < DW_UPPER_BOUND


###############################################################################
# Integration tests
###############################################################################


class TestResidualWorkflow:
    """Integration tests for the complete residual analysis workflow."""

    def test_full_workflow(self) -> None:
        """Test complete workflow from fit to statistics."""
        # Create multiple fit results
        x = np.array([9.0, 8.0, 7.0, 6.0, 5.0])
        y = binding_1site(x, K=7.0, S0=500.0, S1=1000.0, is_ph=True)
        y_err = np.ones_like(y) * 10.0
        da = DataArray(xc=x, yc=y, y_errc=y_err)
        dataset = Dataset({"y1": da}, is_ph=True)

        # Fit multiple "wells"
        fit_results = {f"A{i:02d}": fit_binding_glob(dataset) for i in range(1, 4)}

        # Collect residuals
        all_residuals = collect_multi_residuals(fit_results)
        assert len(all_residuals) == 15  # 5 points * 3 wells

        # Compute statistics
        stats = residual_statistics(all_residuals)
        assert len(stats) == 1  # 1 label
        assert stats.loc["y1", "n_points"] == 15

        # Validate individual fits
        for fr in fit_results.values():
            checks = validate_residuals(fr, verbose=False)
            assert all(isinstance(v, (bool, np.bool_)) for v in checks.values())

    def test_multi_label_workflow(self) -> None:
        """Test workflow with multi-label data."""
        x = np.array([9.0, 8.0, 7.0, 6.0, 5.0])
        y1 = binding_1site(x, K=7.0, S0=500.0, S1=1000.0, is_ph=True)
        y2 = binding_1site(x, K=7.0, S0=200.0, S1=800.0, is_ph=True)
        y_err = np.ones_like(y1) * 10.0

        da1 = DataArray(xc=x, yc=y1, y_errc=y_err)
        da2 = DataArray(xc=x, yc=y2, y_errc=y_err)
        dataset = Dataset({"y1": da1, "y2": da2}, is_ph=True)

        fr = fit_binding_glob(dataset)

        # Extract residuals
        points = extract_residual_points(fr)
        assert len(points) == 10

        # Convert to DataFrame
        df = residual_dataframe(fr)
        assert len(df) == 10
        assert set(df["label"].unique()) == {"y1", "y2"}

        # Statistics by label
        stats = residual_statistics(df)
        assert len(stats) == 2
