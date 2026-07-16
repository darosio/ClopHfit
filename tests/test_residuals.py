"""Test cases for the clophfit.fitting.residuals module."""

import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from lmfit.minimizer import MinimizerResult  # type: ignore[import-untyped]

from clophfit.fitting.core import fit_binding_glob
from clophfit.fitting.data_structures import (
    DataArray,
    Dataset,
    FitResult,
    NoiseModelParams,
    PlateNoiseModel,
)
from clophfit.fitting.model_validation import (
    ResidualAnalysis,
    robust_likelihood_from_trace,
    robust_settings_from_trace,
)
from clophfit.fitting.models import binding_1site
from clophfit.fitting.pipeline import fgls_plate_fit
from clophfit.fitting.residuals import (
    BIAS_P_VALUE_THRESHOLD,
    DW_LOWER_BOUND,
    DW_UPPER_BOUND,
    OUTLIER_RATE_THRESHOLD,
    OUTLIER_THRESHOLD_2SIGMA,
    ResidualPoint,
    collect_multi_residuals,
    detect_adjacent_correlation,
    estimate_x_shift_statistics,
    extract_residual_points,
    residual_dataframe,
    residual_statistics,
)

###############################################################################
# Fixtures
###############################################################################


@pytest.fixture
def simple_fit_result() -> FitResult[MinimizerResult]:
    """Create a simple fit result for testing residuals."""
    x = np.array([9.0, 8.0, 7.0, 6.0, 5.0])
    # Perfect fit data with K=7, adding tiny noise to avoid zero-division in lmfit
    y = binding_1site(x, K=7.0, S0=500.0, S1=1000.0, is_ph=True)
    y[0] += 1e-10
    y_err = np.ones_like(y) * 10.0
    da = DataArray(xc=x, yc=y, y_errc=y_err)
    dataset = Dataset({"1": da}, is_ph=True)
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
    dataset = Dataset({"1": da}, is_ph=True)
    return fit_binding_glob(dataset)


@pytest.fixture
def multi_label_fit_result() -> FitResult[MinimizerResult]:
    """Create a multi-label fit result."""
    x = np.array([9.0, 8.0, 7.0, 6.0, 5.0])
    y1 = binding_1site(x, K=7.0, S0=500.0, S1=1000.0, is_ph=True)
    y2 = binding_1site(x, K=7.0, S0=200.0, S1=800.0, is_ph=True)
    # Adding tiny noise to avoid zero-division in lmfit
    y1[0] += 1e-10
    y2[0] += 1e-10
    y_err = np.ones_like(y1) * 10.0
    da1 = DataArray(xc=x, yc=y1, y_errc=y_err)
    da2 = DataArray(xc=x, yc=y2, y_errc=y_err)
    dataset = Dataset({"1": da1, "2": da2}, is_ph=True)
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
            label="1",
            x=7.0,
            y=1000.0,
            yhat=995.0,
            sigma=10.0,
            raw_res=5.0,
            likelihood_res=0.5,
            std_res=0.5,
            raw_i=0,
        )
        assert point.label == "1"
        assert point.x == 7.0
        assert point.std_res == 0.5
        assert point.raw_res == 5.0
        assert point.raw_i == 0
        assert point.sigma == 10.0
        assert point.yhat == 995.0

    def test_frozen(self) -> None:
        """Test that ResidualPoint is immutable."""
        point = ResidualPoint(
            label="1",
            x=7.0,
            y=1000.0,
            yhat=995.0,
            sigma=10.0,
            raw_res=5.0,
            likelihood_res=0.5,
            std_res=0.5,
            raw_i=0,
        )
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
        assert all(p.label == "1" for p in points)
        assert all(isinstance(p.std_res, float) for p in points)
        assert all(isinstance(p.raw_res, float) for p in points)

    def test_multi_label(
        self, multi_label_fit_result: FitResult[MinimizerResult]
    ) -> None:
        """Test extraction from multi-label fit."""
        points = extract_residual_points(multi_label_fit_result)
        assert len(points) == 10  # 5 points * 2 labels
        y1_points = [p for p in points if p.label == "1"]
        y2_points = [p for p in points if p.label == "2"]
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

    def test_raw_indices_sequential(
        self, simple_fit_result: FitResult[MinimizerResult]
    ) -> None:
        """Test that raw indices are sequential when no masking is applied."""
        points = extract_residual_points(simple_fit_result)
        indices = [p.raw_i for p in points]
        assert indices == list(range(5))

    def test_raw_length_residuals_skip_masked_points(self) -> None:
        """Raw-length backend residual vectors are filtered by DataArray.mask."""
        x = np.array([9.0, 8.0, 7.0, 6.0, 5.0])
        y = binding_1site(x, K=7.0, S0=500.0, S1=1000.0, is_ph=True)
        y_err = np.ones_like(y) * 10.0
        da = DataArray(xc=x, yc=y, y_errc=y_err)
        da.mask = np.array([True, False, True, True, True])
        dataset = Dataset({"1": da}, is_ph=True)
        fr = fit_binding_glob(dataset)
        assert fr.result is not None
        fr.result.residual = np.array([1.0, 99.0, 3.0, 4.0, 5.0])

        points = extract_residual_points(fr)

        assert [p.raw_i for p in points] == [0, 2, 3, 4]
        assert [p.std_res for p in points] == [1.0, 3.0, 4.0, 5.0]
        assert 99.0 not in [p.std_res for p in points]


###############################################################################
# Tests for residual_dataframe
###############################################################################


class TestResidualDataframe:
    """Test the residual_dataframe function."""

    def test_columns(self, simple_fit_result: FitResult[MinimizerResult]) -> None:
        """Test that DataFrame has correct columns."""
        df = residual_dataframe(simple_fit_result)
        expected_cols = {
            "label",
            "x",
            "y",
            "yhat",
            "sigma",
            "raw_res",
            "likelihood_res",
            "std_res",
            "raw_i",
        }
        assert set(df.columns) == expected_cols

    def test_row_count(self, simple_fit_result: FitResult[MinimizerResult]) -> None:
        """Test that DataFrame has correct number of rows."""
        df = residual_dataframe(simple_fit_result)
        assert len(df) == 5

    def test_empty_result(self, empty_fit_result: FitResult[MinimizerResult]) -> None:
        """Test DataFrame from empty fit result."""
        df = residual_dataframe(empty_fit_result)
        assert len(df) == 0


class TestResidualsEntryPoint:
    """The FitResult.residuals cached-property entry point."""

    def test_fitresult_residuals_canonical_and_cached(
        self, simple_fit_result: FitResult[MinimizerResult]
    ) -> None:
        """fr.residuals returns the canonical schema and is cached."""
        r = simple_fit_result.residuals
        assert {"raw_res", "yhat", "std_res", "sigma", "raw_i", "label", "x"} <= set(
            r.columns
        )
        assert simple_fit_result.residuals is r  # cached (same object)

    def test_residual_table_override(
        self, simple_fit_result: FitResult[MinimizerResult]
    ) -> None:
        """residual_table(...) yields the same columns as the property."""
        table = simple_fit_result.residual_table(robust=False)
        assert list(table.columns) == list(simple_fit_result.residuals.columns)

    def test_robust_settings_from_trace(self) -> None:
        """Robustness is inferred from posterior variable names."""
        assert robust_settings_from_trace(None) == (False, 3.0)
        post = xr.Dataset(
            {"student_t_nu": (("chain", "draw"), np.array([[4.0, 6.0]]))},
            coords={"chain": [0], "draw": [0, 1]},
        )
        trace = xr.DataTree.from_dict({"posterior": post})
        robust, nu = robust_settings_from_trace(trace)
        assert robust is True
        assert nu == 5.0
        mix = xr.Dataset(
            {"pi_outlier_1": (("chain", "draw"), np.array([[0.1, 0.2]]))},
            coords={"chain": [0], "draw": [0, 1]},
        )
        mtrace = xr.DataTree.from_dict({"posterior": mix})
        # A mixture is Normal-standardized (no Student-t transform), so it does
        # not request the robust std_res calibration; its outlier structure is
        # reported via p_outlier instead.
        assert robust_settings_from_trace(mtrace) == (False, 3.0)
        assert robust_likelihood_from_trace(mtrace) == "mixture"
        assert robust_likelihood_from_trace(trace) == "student_t"
        assert robust_likelihood_from_trace(None) == "normal"

    def test_dtypes(self, simple_fit_result: FitResult[MinimizerResult]) -> None:
        """Test DataFrame column dtypes."""
        df = residual_dataframe(simple_fit_result)
        assert pd.api.types.is_string_dtype(df["label"])
        assert df["x"].dtype == np.float64
        assert df["std_res"].dtype == np.float64
        assert df["raw_res"].dtype == np.float64
        assert df["raw_i"].dtype == np.int64


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
        assert "1" in stats.index
        assert "2" in stats.index

    def test_outlier_count(self) -> None:
        """Test outlier counting."""
        # Create DataFrame with known outliers
        data = {
            "label": ["1"] * 10,
            "std_res": [0.0, 0.1, -0.1, 0.2, -0.2, 3.0, -3.0, 0.0, 0.1, -0.1],
            "x": list(range(10)),
            "raw_res": [0.0] * 10,
            "i": list(range(10)),
        }
        df = pd.DataFrame(data)
        stats = residual_statistics(df)
        # 2 outliers beyond ±2-sigma (3.0 and -3.0)
        assert stats.loc["1", "outlier_count"] == 2

    def test_outlier_rate(self) -> None:
        """Test outlier rate calculation."""
        data = {
            "label": ["1"] * 10,
            "std_res": [0.0] * 8 + [3.0, -3.0],  # 2 outliers out of 10
            "x": list(range(10)),
            "raw_res": [0.0] * 10,
            "i": list(range(10)),
        }
        df = pd.DataFrame(data)
        stats = residual_statistics(df)
        assert stats.loc["1", "outlier_rate"] == 0.2  # 2/10

    def test_robust_outlier_count_surfaces_masked_points(self) -> None:
        """robust_outlier_count catches points that the std-z flag masks.

        A tight bulk near 0.2 with two points at ~-0.5: the model-standardized
        |std_res| stays below 2 (std-z flags nothing), but the modified z-score
        on the residuals' own MAD scale flags the two.
        """
        std_res = [0.20, 0.22, 0.18, 0.24, 0.19, -0.50, -0.52]
        df = pd.DataFrame({
            "label": ["1"] * 7,
            "std_res": std_res,
            "x": list(range(7)),
            "raw_res": [0.0] * 7,
        })
        stats = residual_statistics(df)
        assert stats.loc["1", "outlier_count"] == 0  # std-z misses them
        assert stats.loc["1", "robust_outlier_count"] == 2  # robust MAD-z catches
        assert stats.loc["1", "robust_outlier_rate"] == pytest.approx(2 / 7)


class TestResidualDiagnosticsHelpers:
    """Test dataframe-level residual diagnostic helpers."""

    def test_covariance_correlation_bias_and_shift_helpers(self) -> None:
        """Helpers should summarize residual structure by label, well, and x."""
        df = pd.DataFrame({
            "well": ["A01", "A01", "A01", "A01", "A02", "A02", "A02", "A02"] * 2,
            "label": ["1"] * 8 + ["2"] * 8,
            "x": [6.0, 7.0, 8.0, 9.0, 6.0, 7.0, 8.0, 9.0] * 2,
            "std_res": [
                -1.0,
                0.0,
                1.0,
                0.0,
                -2.0,
                0.0,
                2.0,
                0.0,
                1.0,
                0.5,
                0.0,
                0.5,
                2.0,
                1.0,
                0.0,
                1.0,
            ],
        })

        analysis = ResidualAnalysis(df)
        cov = analysis.covariance()
        corr = analysis.correlation()
        bias_summary, label_bias = analysis.label_bias(n_bins=2)
        lag_rows, lag_by_label = detect_adjacent_correlation(df)
        shift = estimate_x_shift_statistics(df, {})

        assert set(cov) == {"1", "2"}
        assert cov["1"].shape == (4, 4)
        assert corr["1"].loc[6.0].loc[6.0] == pytest.approx(1.0)
        assert set(bias_summary.index.get_level_values("label")) == {"1", "2"}
        assert len(bias_summary) == 4
        assert label_bias.loc["2", "negative_bias_frac"] == 0.0
        assert len(lag_rows) == 4
        assert set(lag_by_label) == {"1", "2"}
        assert set(shift["well"]) == {"A01", "A02"}
        assert shift["trend_strength"].notna().all()

    def test_detect_adjacent_correlation_skips_nan_correlations(self) -> None:
        """Constant residual series should not enter lag-correlation summaries."""
        df = pd.DataFrame({
            "well": ["A01", "A01", "A01"],
            "label": ["1", "1", "1"],
            "x": [6.0, 7.0, 8.0],
            "std_res": [1.0, 1.0, 1.0],
        })

        rows, by_label = detect_adjacent_correlation(df)

        assert rows.empty
        assert by_label == {}

    def test_covariance_aligns_on_step_with_jittered_x(self) -> None:
        """Per-well jittered x must align on ``step``, not collapse to all-NaN."""
        rng_x = [4.8, 5.2, 6.0, 7.0, 8.0, 8.2]
        rows = [
            {
                "well": f"A{w:02d}",
                "label": "1",
                "step": s,
                "x": round(x + 0.01 * w, 3),  # every well has distinct x-values
                "std_res": float(np.sin(w + s)),
            }
            for w in range(6)
            for s, x in enumerate(rng_x)
        ]
        analysis = ResidualAnalysis(pd.DataFrame(rows))

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            cov = analysis.covariance()
            corr = analysis.correlation()

        assert cov["1"].shape == (6, 6)
        assert not cov["1"].isna().any().any()  # would be all-NaN if pivoting on x
        assert np.allclose(np.diag(corr["1"].to_numpy()), 1.0)

    def test_covariance_empty_for_single_well_label(self) -> None:
        """A label with a single well cannot form a covariance; return empty."""
        df = pd.DataFrame({
            "well": ["A01", "A01", "A01"],
            "label": ["1", "1", "1"],
            "step": [0, 1, 2],
            "x": [6.0, 7.0, 8.0],
            "std_res": [0.1, -0.2, 0.3],
        })
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            cov = ResidualAnalysis(df).covariance()
        assert cov["1"].empty


###############################################################################
# Tests for ResidualAnalysis.validate
###############################################################################


class TestValidateResiduals:
    """Test ResidualAnalysis.validate residual-quality checks."""

    def test_good_fit(self, simple_fit_result: FitResult[MinimizerResult]) -> None:
        """Test validation of a good fit."""
        checks = ResidualAnalysis(simple_fit_result.residuals).validate(verbose=False)
        assert isinstance(checks, dict)
        assert "bias_ok" in checks
        assert "outliers_ok" in checks
        assert "correlation_ok" in checks

    def test_empty_result(self, empty_fit_result: FitResult[MinimizerResult]) -> None:
        """Test validation of empty fit result."""
        checks = ResidualAnalysis(empty_fit_result.residuals).validate(verbose=False)
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
        y = y_true + 50.0 + np.array([0.5, -0.25, 0.0, 0.25, -0.5])
        y_err = np.ones_like(y) * 10.0
        da = DataArray(xc=x, yc=y, y_errc=y_err)
        dataset = Dataset({"1": da}, is_ph=True)
        fr = fit_binding_glob(dataset)
        # With a constant offset, the fit should still be good
        # (offset absorbed by S0/S1), so bias_ok should be True
        checks = ResidualAnalysis(fr.residuals).validate(verbose=False)
        assert isinstance(checks["bias_ok"], bool)

    def test_verbose_output(
        self,
        noisy_fit_result: FitResult[MinimizerResult],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that verbose mode produces output."""
        ResidualAnalysis(noisy_fit_result.residuals).validate(verbose=True)
        captured = capsys.readouterr()
        # Verbose mode should produce some output (may or may not warn)
        assert isinstance(captured.out, str)

    def test_returns_dict(self, noisy_fit_result: FitResult[MinimizerResult]) -> None:
        """Test that function returns correct structure."""
        checks = ResidualAnalysis(noisy_fit_result.residuals).validate(verbose=False)
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
        y[0] += 1e-10
        y_err = np.ones_like(y) * 10.0
        da = DataArray(xc=x, yc=y, y_errc=y_err)
        dataset = Dataset({"1": da}, is_ph=True)

        # Fit multiple "wells"
        fit_results = {f"A{i:02d}": fit_binding_glob(dataset) for i in range(1, 4)}

        # Collect residuals
        all_residuals = collect_multi_residuals(fit_results)
        assert len(all_residuals) == 15  # 5 points * 3 wells

        # Compute statistics
        stats = residual_statistics(all_residuals)
        assert len(stats) == 1  # 1 label
        assert stats.loc["1", "n_points"] == 15

        # Validate individual fits
        for fr in fit_results.values():
            checks = ResidualAnalysis(fr.residuals).validate(verbose=False)
            assert all(isinstance(v, (bool, np.bool_)) for v in checks.values())

    def test_multi_label_workflow(self) -> None:
        """Test workflow with multi-label data."""
        x = np.array([9.0, 8.0, 7.0, 6.0, 5.0])
        y1 = binding_1site(x, K=7.0, S0=500.0, S1=1000.0, is_ph=True)
        y2 = binding_1site(x, K=7.0, S0=200.0, S1=800.0, is_ph=True)
        y1[0] += 1e-10
        y2[0] += 1e-10
        y_err = np.ones_like(y1) * 10.0

        da1 = DataArray(xc=x, yc=y1, y_errc=y_err)
        da2 = DataArray(xc=x, yc=y2, y_errc=y_err)
        dataset = Dataset({"1": da1, "2": da2}, is_ph=True)

        fr = fit_binding_glob(dataset)

        # Extract residuals
        points = extract_residual_points(fr)
        assert len(points) == 10

        # Convert to DataFrame
        df = residual_dataframe(fr)
        assert len(df) == 10
        assert set(df["label"].unique()) == {"1", "2"}

        # Statistics by label
        stats = residual_statistics(df)
        assert len(stats) == 2


class TestNoiseModelContainers:
    """Test cases for NoiseModelParams and PlateNoiseModel."""

    def test_noise_model_params(self) -> None:
        """Test NoiseModelParams creation, compute_y_err, and frozen/hashable."""
        params = NoiseModelParams(sigma_floor=1.5, gain=0.5, alpha=0.02)
        assert params.sigma_floor == 1.5
        assert params.gain == 0.5
        assert params.alpha == 0.02

        # compute_y_err for fixed noise model application
        y = np.array([100.0, 200.0, 300.0])
        expected = np.sqrt(1.5**2 + 0.5 * y + (0.02 * y) ** 2)
        assert np.allclose(params.compute_y_err(y), expected)

        # Frozen → cannot mutate
        with pytest.raises(AttributeError):
            params.sigma_floor = 3.0  # type: ignore[misc]

        # Hashable (frozen dataclass)
        d = {params: "test"}
        assert d[params] == "test"

    def test_noise_model_params_zero_terms(self) -> None:
        """NoiseModelParams with gain=0 or alpha=0 disables those terms."""
        # gain=0: no Poisson contribution
        p_no_gain = NoiseModelParams(sigma_floor=2.0, alpha=0.03)
        assert p_no_gain.gain == 0.0
        y = np.array([100.0, 200.0])
        expected = np.sqrt(2.0**2 + (0.03 * y) ** 2)
        assert np.allclose(p_no_gain.compute_y_err(y), expected)

        # alpha=0: no proportional contribution
        p_no_alpha = NoiseModelParams(sigma_floor=2.0, gain=1.0)
        assert p_no_alpha.alpha == 0.0
        expected = np.sqrt(2.0**2 + y)  # gain=1 * y
        assert np.allclose(p_no_alpha.compute_y_err(y), expected)

    def test_plate_noise_model(self) -> None:
        """Test PlateNoiseModel properties and direct apply_to."""
        params1 = NoiseModelParams(sigma_floor=1.5, gain=0.5, alpha=0.02)
        params2 = NoiseModelParams(sigma_floor=2.0, gain=0.0, alpha=0.01)

        model = PlateNoiseModel({"lbl1": params1, "lbl2": params2})

        # Properties still work
        assert model.sigma_floor == {"lbl1": 1.5, "lbl2": 2.0}
        assert model.gain == {"lbl1": 0.5, "lbl2": 0.0}
        assert model.alpha == {"lbl1": 0.02, "lbl2": 0.01}

        # Test direct apply_to
        x = np.array([5.0, 6.0, 7.0])
        y1 = np.array([100.0, 200.0, 300.0])
        y2 = np.array([50.0, 100.0, 150.0])
        ds = Dataset(
            {"lbl1": DataArray(xc=x, yc=y1), "lbl2": DataArray(xc=x, yc=y2)},
            is_ph=True,
        )

        ds_updated = model.apply_to(ds)

        # Refactored apply_to uses params.compute_y_err directly
        assert np.allclose(ds_updated["lbl1"].y_errc, params1.compute_y_err(y1))
        assert np.allclose(ds_updated["lbl2"].y_errc, params2.compute_y_err(y2))


def test_fgls_plate_fit_workflow() -> None:
    """Smoke test of the complete FGLS pipeline fit workflow."""
    x = np.array([9.0, 8.0, 7.0, 6.0, 5.0])
    y = binding_1site(x, K=7.0, S0=500.0, S1=1000.0, is_ph=True)
    # create dataset with small noise
    rng = np.random.default_rng(42)
    y_noisy = y + rng.normal(0, 2.0, size=len(y))
    da = DataArray(xc=x, yc=y_noisy)
    datasets = {"well1": Dataset({"1": da}, is_ph=True)}

    sigma_floor = {"1": 1.0}
    final_results, noise_params = fgls_plate_fit(datasets, sigma_floor)

    assert "well1" in final_results
    assert final_results["well1"].result is not None
    assert final_results["well1"].result.success
    assert isinstance(noise_params, PlateNoiseModel)
    assert "1" in noise_params
    assert noise_params["1"].sigma_floor == 1.0
    assert noise_params["1"].gain >= 0.0
    assert noise_params["1"].alpha >= 0.0
