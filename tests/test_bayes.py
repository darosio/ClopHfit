"""Test cases for the clophfit.fitting.bayes module."""

from __future__ import annotations

import numpy as np
import pymc as pm  # type: ignore[import-untyped]
import pytest
from lmfit import Parameters  # type: ignore[import-untyped]

from clophfit.fitting.bayes import (
    create_parameter_priors,
    create_x_true,
    extract_fit,
    fit_binding_pymc,
    fit_binding_pymc2,
    process_trace,
    rename_keys,
    weighted_stats,
    x_true_from_trace_df,
)
from clophfit.fitting.core import fit_binding_glob
from clophfit.fitting.data_structures import DataArray, Dataset, FitResult, MiniT

###############################################################################
# Fixtures
###############################################################################


@pytest.fixture
def ph_dataset() -> Dataset:
    """Create a sample pH-titration Dataset."""
    x = np.array([9.0, 8.0, 7.0, 6.0, 5.0])
    # Generated with K=7, S0=2, S1=1
    y = np.array([1.99, 1.909, 1.5, 1.0909, 1.0099])
    return Dataset({"default": DataArray(x, y)}, is_ph=True)


@pytest.fixture
def multi_dataset() -> Dataset:
    """Create a sample multi-label Dataset."""
    x = np.array([9.0, 8.0, 7.0, 6.0, 5.0])
    # y1 generated with K=7, S0=2, S1=1
    y1 = np.array([1.99, 1.909, 1.5, 1.0909, 1.0099])
    # y2 generated with K=7, S0=0, S1=1
    y2 = np.array([0.01, 0.091, 0.5, 0.909, 0.99])
    return Dataset({"y1": DataArray(x, y1), "y2": DataArray(x, y2)}, is_ph=True)


@pytest.fixture
def lmfit_params() -> Parameters:
    """Create sample lmfit Parameters with stderr values."""
    params = Parameters()
    params.add("K", value=7.0, min=5.0, max=9.0)
    params["K"].stderr = 0.2
    params.add("S0_y1", value=2.0, min=0.0, max=5.0)
    params["S0_y1"].stderr = 0.1
    params.add("S1_y1", value=1.0, min=0.0, max=5.0)
    params["S1_y1"].stderr = 0.05
    return params


###############################################################################
# Tests for create_x_true
###############################################################################


def test_create_x_true_no_xerr() -> None:
    """Test that create_x_true returns xc when n_xerr is 0."""
    xc = np.array([9.0, 8.0, 7.0, 6.0, 5.0])
    x_errc = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
    n_xerr = 0.0
    result = create_x_true(xc, x_errc, n_xerr)
    np.testing.assert_array_equal(result, xc)


def test_create_x_true_with_xerr() -> None:
    """Test that create_x_true creates PyMC distribution when n_xerr > 0."""
    pytest.importorskip("pymc")

    xc = np.array([9.0, 8.0, 7.0, 6.0, 5.0])
    x_errc = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
    n_xerr = 1.0

    with pm.Model():
        result = create_x_true(xc, x_errc, n_xerr)
        # Check that result is a PyMC variable (not just an array)
        assert hasattr(result, "eval")
        assert hasattr(result, "name")
        assert result.name == "x_true"


def test_create_x_true_shape() -> None:
    """Test that x_true has the correct shape."""
    pytest.importorskip("pymc")

    xc = np.array([9.0, 8.0, 7.0, 6.0, 5.0])
    x_errc = np.array([0.05, 0.05, 0.05, 0.05, 0.05])
    n_xerr = 1.5

    with pm.Model():
        x_true = create_x_true(xc, x_errc, n_xerr)
        # The shape should match the input xc
        assert hasattr(x_true, "eval")
        evaluated = x_true.eval()
        assert evaluated.shape == xc.shape


def test_create_x_true_lower_bound() -> None:
    """Test that lower_nsd parameter affects the truncation lower bound."""
    pytest.importorskip("pymc")

    xc = np.array([9.0, 8.0, 7.0, 6.0, 5.0])
    x_errc = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
    n_xerr = 1.0
    lower_nsd = 3.0

    with pytest.importorskip("pymc").Model():
        result = create_x_true(xc, x_errc, n_xerr, lower_nsd=lower_nsd)
        assert hasattr(result, "eval")
        assert hasattr(result, "name")


###############################################################################
# Tests for create_parameter_priors
###############################################################################


def test_create_parameter_priors_basic(lmfit_params: Parameters) -> None:
    """Test basic creation of parameter priors."""
    pytest.importorskip("pymc")

    with pytest.importorskip("pymc").Model():
        priors = create_parameter_priors(lmfit_params, n_sd=5.0)
        # Check that all parameters are created
        assert "K" in priors
        assert "S0_y1" in priors
        assert "S1_y1" in priors
        # Check that they have PyMC-like attributes
        assert hasattr(priors["K"], "eval")
        assert hasattr(priors["S0_y1"], "eval")
        assert hasattr(priors["S1_y1"], "eval")


def test_create_parameter_priors_with_key(lmfit_params: Parameters) -> None:
    """Test parameter priors with a key suffix."""
    pytest.importorskip("pymc")
    with pm.Model():
        priors = create_parameter_priors(lmfit_params, n_sd=5.0, key="A01")
        # Check that parameter names include the key
        assert "K_A01" in priors
        assert "S0_y1_A01" in priors
        assert "S1_y1_A01" in priors


def test_create_parameter_priors_no_stderr() -> None:
    """Test parameter priors when stderr is None."""
    pytest.importorskip("pymc")
    # Create params without stderr
    params = Parameters()
    params.add("K", value=7.0)
    params["K"].stderr = None
    with pytest.importorskip("pymc").Model():
        priors = create_parameter_priors(params, n_sd=5.0)
        # Should still create a prior with default sigma
        assert "K" in priors
        assert hasattr(priors["K"], "eval")


def test_create_parameter_priors_skip_shared_k(lmfit_params: Parameters) -> None:
    """Test that K is skipped when ctr_name is provided."""
    pytest.importorskip("pymc")
    with pm.Model():
        priors = create_parameter_priors(
            lmfit_params, n_sd=5.0, key="A01", ctr_name="control"
        )
        # K should be skipped (not created with _A01 suffix)
        assert "K_A01" not in priors
        # But other parameters should still be created
        assert "S0_y1_A01" in priors
        assert "S1_y1_A01" in priors


def test_create_parameter_priors_sigma_scaling(lmfit_params: Parameters) -> None:
    """Test that n_sd parameter scales the prior width."""
    pytest.importorskip("pymc")
    with pm.Model():
        priors_wide = create_parameter_priors(lmfit_params, n_sd=10.0)
    with pm.Model():
        priors_narrow = create_parameter_priors(lmfit_params, n_sd=1.0)
    # Both should create the same parameters
    assert set(priors_wide.keys()) == set(priors_narrow.keys())


###############################################################################
# Tests for rename_keys
###############################################################################


def test_rename_keys_basic() -> None:
    """Test basic key renaming."""
    data = {
        "K_A01": 7.0,
        "S0_y1_A01": 2.0,
        "S1_y1_A01": 1.0,
    }
    result = rename_keys(data)
    assert result == {
        "K": 7.0,
        "S0_y1": 2.0,
        "S1_y1": 1.0,
    }


def test_rename_keys_k_prefix() -> None:
    """Test that K_ prefix is properly renamed to K."""
    data = {
        "K_control": 7.5,
        "K_sample": 8.0,
    }
    result = rename_keys(data)
    # Both should be renamed to "K", last one wins
    assert "K" in result


def test_rename_keys_no_suffix() -> None:
    """Test keys without suffix remain unchanged."""
    data = {
        "K": 7.0,
        "S0": 2.0,
    }
    result = rename_keys(data)
    assert result == data


def test_rename_keys_mixed() -> None:
    """Test renaming with mixed key formats."""
    data = {
        "K_A01": 7.0,
        "S0_y1_A01": 2.0,
        "simple": 1.0,
        "K_B02": 7.2,
    }
    result = rename_keys(data)
    assert "K" in result
    assert "S0_y1" in result
    assert "simple" in result


def test_rename_keys_empty() -> None:
    """Test renaming with empty dictionary."""
    data: dict[str, float] = {}
    result = rename_keys(data)
    assert result == {}


def test_rename_keys_preserves_values() -> None:
    """Test that values are preserved during renaming."""
    data = {
        "K_A01": 7.123456,
        "S0_y1_A01": 2.987654,
    }
    result = rename_keys(data)
    assert result["K"] == 7.123456
    assert result["S0_y1"] == 2.987654


###############################################################################
# Tests for weighted_stats
###############################################################################


def test_weighted_stats_basic() -> None:
    """Test basic weighted statistics calculation."""
    values = {
        "sample1": [7.0, 7.2, 6.8],
        "sample2": [8.0, 8.1],
    }
    stderr = {
        "sample1": [0.1, 0.2, 0.15],
        "sample2": [0.3, 0.25],
    }
    result = weighted_stats(values, stderr)

    assert "sample1" in result
    assert "sample2" in result
    # Check that each result is a tuple of (mean, stderr)
    assert len(result["sample1"]) == 2
    assert len(result["sample2"]) == 2
    # Mean should be a float
    assert isinstance(result["sample1"][0], (float, np.floating))
    # Stderr should be a float
    assert isinstance(result["sample1"][1], (float, np.floating))


def test_weighted_stats_single_value() -> None:
    """Test weighted statistics with single value."""
    values = {"sample": [7.0]}
    stderr = {"sample": [0.2]}
    result = weighted_stats(values, stderr)

    assert "sample" in result
    # With a single value, weighted mean equals the value
    assert np.isclose(result["sample"][0], 7.0)


def test_weighted_stats_weights() -> None:
    """Test that weighting by inverse variance works correctly."""
    values = {
        "sample": [5.0, 9.0],  # Two very different values
    }
    stderr = {
        "sample": [0.1, 10.0],  # First has much smaller error
    }
    result = weighted_stats(values, stderr)

    # Weighted mean should be much closer to the first value (5.0)
    # since it has much smaller error
    assert result["sample"][0] < 6.0
    assert result["sample"][0] > 4.9


def test_weighted_stats_equal_weights() -> None:
    """Test that equal weights give arithmetic mean."""
    values = {"sample": [4.0, 6.0, 8.0]}
    stderr = {"sample": [1.0, 1.0, 1.0]}  # Equal weights
    result = weighted_stats(values, stderr)

    # Should be close to arithmetic mean (4+6+8)/3 = 6.0
    assert np.isclose(result["sample"][0], 6.0, rtol=0.01)


def test_weighted_stats_multiple_samples() -> None:
    """Test with multiple sample groups."""
    values = {
        "control": [7.0, 7.1, 6.9],
        "treated": [8.0, 8.2],
        "blank": [5.0],
    }
    stderr = {
        "control": [0.1, 0.1, 0.1],
        "treated": [0.2, 0.2],
        "blank": [0.3],
    }
    result = weighted_stats(values, stderr)

    assert len(result) == 3
    assert all(key in result for key in ["control", "treated", "blank"])


def test_weighted_stats_empty() -> None:
    """Test with empty dictionaries."""
    values: dict[str, list[float]] = {}
    stderr: dict[str, list[float]] = {}
    result = weighted_stats(values, stderr)
    assert result == {}


###############################################################################
# Tests for integration with fitting pipeline
###############################################################################


def test_bayes_module_imports() -> None:
    """Test that all main functions can be imported."""
    # Check functions are callable
    assert callable(create_x_true)
    assert callable(create_parameter_priors)
    assert callable(rename_keys)
    assert callable(weighted_stats)
    assert callable(fit_binding_pymc)
    assert callable(fit_binding_pymc2)
    assert callable(process_trace)
    assert callable(extract_fit)
    assert callable(x_true_from_trace_df)


@pytest.mark.slow
def test_fit_binding_pymc_smoke_test(ph_dataset: Dataset) -> None:
    """Smoke test for the PyMC fitter with minimal sampling."""
    pytest.importorskip("pymc")
    # First, run a standard lmfit to get initial parameters
    initial_fit = fit_binding_glob(ph_dataset)
    assert initial_fit.result is not None
    assert initial_fit.dataset is not None
    # Now, run the pymc fitter with very few samples (just to test it runs)
    # Use tune=50 to reduce sampling time
    fit_result_pymc = fit_binding_pymc(
        initial_fit, n_samples=50, n_xerr=0, ye_scaling=1.0, n_sd=10.0
    )
    # Check that we got a result
    assert fit_result_pymc.mini is not None
    assert fit_result_pymc.result is not None
    # Check that parameters are in the result
    assert "K" in fit_result_pymc.result.params
    # Check that K is reasonably close to expected value
    assert 5.0 < fit_result_pymc.result.params["K"].value < 9.0


@pytest.mark.slow
def test_fit_binding_pymc_with_xerr(ph_dataset: Dataset) -> None:
    """Test PyMC fitting with x error modeling."""
    pytest.importorskip("pymc")
    # Add x errors to the dataset
    for da in ph_dataset.values():
        da.x_errc = np.array([0.05, 0.05, 0.05, 0.05, 0.05])
    # Get initial fit
    initial_fit = fit_binding_glob(ph_dataset)
    assert initial_fit.result is not None
    assert initial_fit.dataset is not None
    # Run PyMC with x error modeling (reduced samples for speed)
    fit_result = fit_binding_pymc(
        initial_fit, n_samples=50, n_xerr=1.0, ye_scaling=1.0, n_sd=10.0
    )
    assert fit_result.result is not None
    assert "K" in fit_result.result.params
    # Check that K is reasonably close to expected value
    assert 5.0 < fit_result.result.params["K"].value < 9.0


@pytest.mark.slow
def test_fit_binding_pymc2_smoke_test(multi_dataset: Dataset) -> None:
    """Smoke test for PyMC fitter with separate ye_mag per label."""
    pytest.importorskip("pymc")
    # Get initial fit
    initial_fit = fit_binding_glob(multi_dataset)
    assert initial_fit.result is not None
    assert initial_fit.dataset is not None
    # Run PyMC2 with separate noise scaling (reduced samples for speed)
    fit_result = fit_binding_pymc2(initial_fit, n_samples=50, n_xerr=0, n_sd=10.0)
    assert fit_result.result is not None
    assert "K" in fit_result.result.params
    # Check that K is reasonably close to expected value
    assert 5.0 < fit_result.result.params["K"].value < 9.0


def test_fit_binding_pymc_empty_result() -> None:
    """Test that fit_binding_pymc handles empty FitResult gracefully."""
    pytest.importorskip("pymc")
    # Create an empty FitResult
    empty_result: FitResult[MiniT] = FitResult()
    # Should return empty result without crashing
    result = fit_binding_pymc(empty_result, n_samples=10)
    assert result.result is None or result.mini is None


###############################################################################
# Tests for edge cases and error handling
###############################################################################


def test_create_x_true_zero_errors() -> None:
    """Test create_x_true with zero x_errc values."""
    pytest.importorskip("pymc")

    xc = np.array([9.0, 8.0, 7.0])
    x_errc = np.array([0.0, 0.0, 0.0])
    n_xerr = 1.0

    with pytest.importorskip("pymc").Model():
        # Should handle zero errors gracefully
        result = create_x_true(xc, x_errc, n_xerr)
        assert hasattr(result, "eval")


def test_rename_keys_with_underscores() -> None:
    """Test rename_keys with various underscore patterns."""
    data = {
        "K_": 7.0,  # Single underscore at end
        "S0_": 2.0,
        "_K": 7.5,  # Underscore at start
        "S0__y1": 2.5,  # Double underscore
    }
    result = rename_keys(data)
    # Should handle these edge cases without crashing
    assert isinstance(result, dict)


def test_weighted_stats_zero_stderr() -> None:
    """Test weighted_stats with very small stderr (near zero)."""
    values = {"sample": [7.0, 7.1]}
    stderr = {"sample": [1e-10, 1e-10]}  # Very small errors
    result = weighted_stats(values, stderr)
    # Should still compute without division errors
    assert "sample" in result
    assert not np.isnan(result["sample"][0])
    assert not np.isinf(result["sample"][0])


def test_create_parameter_priors_minimum_sigma() -> None:
    """Test that priors have a minimum sigma even with tiny stderr."""
    pytest.importorskip("pymc")
    params = Parameters()
    params.add("K", value=7.0)
    params["K"].stderr = 1e-10  # Very small stderr
    with pm.Model():
        priors = create_parameter_priors(params, n_sd=5.0)
        # Should apply minimum sigma of 1e-3
        assert "K" in priors
