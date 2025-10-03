"""Test the refactored fitter testing utilities."""

from __future__ import annotations

import typing
from dataclasses import dataclass

import numpy as np
import pytest
from lmfit import Parameters  # type: ignore[import-untyped]

from clophfit.fitting.data_structures import FitResult
from clophfit.fitting.models import binding_1site
from clophfit.testing.fitter_test_utils import (
    build_fitters,
    k_from_result,
    make_synthetic_ds,
    s_from_result,
)


class TestSyntheticDataGeneration:
    """Test synthetic dataset generation utilities."""

    def test_make_synthetic_ds_ph(self) -> None:
        """Test synthetic pH dataset generation."""
        k = 7.2
        s0 = {"y0": 2.0, "y1": 2.2}
        s1 = {"y0": 1.0, "y1": 1.1}

        ds, truth = make_synthetic_ds(k, s0, s1, is_ph=True, noise=0.01, seed=42)

        assert ds.is_ph
        assert len(ds) == 2
        assert "y0" in ds
        assert "y1" in ds
        assert k == truth.K
        assert s0 == truth.S0
        assert s1 == truth.S1

        # Check that data has expected structure
        for da in ds.values():
            assert len(da.x) == 7  # Default pH points
            assert len(da.y) == len(da.x)
            assert np.all(da.x_err > 0)  # Should have x errors

    def test_make_synthetic_ds_cl(self) -> None:
        """Test synthetic Cl dataset generation."""
        k = 15.0
        s0 = 3.0  # Single value should be converted to dict
        s1 = 0.5

        ds, truth = make_synthetic_ds(k, s0, s1, is_ph=False, noise=0.02, seed=123)

        assert not ds.is_ph
        assert len(ds) == 1
        assert "y0" in ds
        assert k == truth.K
        assert truth.S0 == {"y0": 3.0}
        assert truth.S1 == {"y0": 0.5}

        da = ds["y0"]
        assert len(da.x) == 7  # Default Cl points
        assert np.all(da.x >= 0.01)  # Cl concentrations should be positive

    def test_make_synthetic_ds_reproducible(self) -> None:
        """Test that synthetic data generation is reproducible."""
        k, s0, s1 = 7.0, 2.0, 1.0

        ds1, _ = make_synthetic_ds(k, s0, s1, is_ph=True, seed=42)
        ds2, _ = make_synthetic_ds(k, s0, s1, is_ph=True, seed=42)

        np.testing.assert_array_equal(ds1["y0"].y, ds2["y0"].y)

    def test_make_synthetic_ds_different_noise_levels(self) -> None:
        """Test different noise levels produce different standard deviations."""
        k, s0, s1 = 7.0, 2.0, 1.0

        # Use different seeds to avoid correlation
        ds_low, _ = make_synthetic_ds(k, s0, s1, is_ph=True, noise=0.001, seed=42)
        ds_high, _ = make_synthetic_ds(k, s0, s1, is_ph=True, noise=0.1, seed=43)

        # Compare standard deviation of residuals from clean model
        clean_y = binding_1site(ds_low["y0"].x, k, s0, s1, is_ph=True)

        residuals_low = np.abs(ds_low["y0"].y - clean_y)
        residuals_high = np.abs(ds_high["y0"].y - clean_y)

        std_low = np.std(residuals_low)
        std_high = np.std(residuals_high)

        # Higher noise should produce larger residuals
        assert (
            std_high > std_low * 5
        )  # Should be much larger given 100x noise difference


class TestResultExtraction:
    """Test parameter extraction from fit results."""

    def test_k_from_result_valid(self) -> None:
        """Test K extraction from valid result."""
        # Create a minimal valid FitResult

        params = Parameters()
        params.add("K", value=7.5)
        params["K"].stderr = 0.1

        # Create a mock result object
        @dataclass
        class MockResult:
            params: Parameters

        fr: FitResult[typing.Any] = FitResult(result=MockResult(params))
        k_val, k_err = k_from_result(fr)

        assert k_val == 7.5
        assert k_err == 0.1

    def test_k_from_result_no_stderr(self) -> None:
        """Test K extraction when stderr is None."""
        params = Parameters()
        params.add("K", value=8.2)
        # stderr defaults to None

        @dataclass
        class MockResult:
            params: Parameters

        fr: FitResult[typing.Any] = FitResult(result=MockResult(params))
        k_val, k_err = k_from_result(fr)

        assert k_val == 8.2
        assert k_err is None

    def test_k_from_result_invalid(self) -> None:
        """Test K extraction from invalid result."""
        fr: FitResult[typing.Any] = FitResult()  # Empty result
        k_val, k_err = k_from_result(fr)

        assert k_val is None
        assert k_err is None

    def test_s_from_result_multi_label(self) -> None:
        """Test S parameter extraction for multi-label datasets."""
        params = Parameters()
        params.add("S0_y0", value=2.0)
        params.add("S0_y1", value=2.2)
        params.add("S1_y0", value=1.0)
        params.add("K", value=7.0)

        params["S0_y0"].stderr = 0.05
        params["S0_y1"].stderr = 0.06
        params["S1_y0"].stderr = 0.02
        params["K"].stderr = 0.1

        @dataclass
        class MockResult:
            params: Parameters

        fr: FitResult[typing.Any] = FitResult(result=MockResult(params))
        s0_vals = s_from_result(fr, "S0")
        s1_vals = s_from_result(fr, "S1")

        assert s0_vals == {"S0_y0": 2.0, "S0_y1": 2.2}
        assert s1_vals == {"S1_y0": 1.0}


class TestFitterBuilding:
    """Test fitter construction utilities."""

    def test_build_fitters_default(self) -> None:
        """Test default fitter construction."""
        fitters = build_fitters(include_odr=False)

        expected_keys = {"glob_ls", "glob_huber", "glob_irls_outlier", "outlier2"}
        assert set(fitters.keys()) == expected_keys

        # All should be callable
        for name, fitter in fitters.items():
            assert callable(fitter), f"Fitter {name} is not callable"

    def test_build_fitters_with_odr(self) -> None:
        """Test fitter construction with ODR included."""
        fitters = build_fitters(include_odr=True)

        expected_keys = {
            "glob_ls",
            "glob_huber",
            "glob_irls_outlier",
            "outlier2",
            "odr_recursive_outlier",
        }
        assert set(fitters.keys()) == expected_keys

    def test_fitter_execution_smoke_test(self) -> None:
        """Smoke test that fitters can actually run on synthetic data."""
        fitters = build_fitters(include_odr=False)

        # Create simple synthetic dataset
        ds, _ = make_synthetic_ds(7.0, 2.0, 1.0, is_ph=True, noise=0.01, seed=42)

        # Test that each fitter runs without error
        for name, fitter in fitters.items():
            try:
                result = fitter(ds.copy())
                # Basic check that we got some result
                assert isinstance(result, FitResult)
            except (ValueError, TypeError, RuntimeError) as e:
                pytest.fail(f"Fitter {name} failed with error: {e}")


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_synthetic_data_empty_labels(self) -> None:
        """Test synthetic data generation with empty label dictionaries."""
        # Empty dictionaries should result in empty dataset
        ds, truth = make_synthetic_ds(7.0, {}, {}, is_ph=True)
        assert len(ds) == 0
        assert len(truth.S0) == 0
        assert len(truth.S1) == 0

    def test_parameter_extraction_with_inf_nan(self) -> None:
        """Test parameter extraction handles infinite/NaN values gracefully."""
        params = Parameters()
        params.add("S0_y0", value=np.inf)
        params.add("S1_y0", value=np.nan)
        params["S0_y0"].stderr = 0.05
        params["S1_y0"].stderr = 0.02

        @dataclass
        class MockResult:
            params: Parameters

        fr: FitResult[typing.Any] = FitResult(result=MockResult(params))
        s0_vals = s_from_result(fr, "S0")
        s1_vals = s_from_result(fr, "S1")

        # Should filter out non-finite values
        assert s0_vals is None  # inf should be filtered out
        assert s1_vals is None  # nan should be filtered out

    def test_synthetic_data_extreme_noise(self) -> None:
        """Test synthetic data generation with extreme noise levels."""
        # Very high noise
        ds_high, _ = make_synthetic_ds(7.0, 2.0, 1.0, is_ph=True, noise=10.0, seed=42)
        assert len(ds_high) > 0  # Should still create data

        # Zero noise
        ds_zero, _ = make_synthetic_ds(7.0, 2.0, 1.0, is_ph=True, noise=0.0, seed=42)
        assert len(ds_zero) > 0  # Should still create data
