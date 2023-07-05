"""Test cases for the binding functions module."""

import re

import numpy as np
import pytest

from clophfit import binding
from clophfit.binding.fitting import Dataset, fit_binding_glob


def test_kd() -> None:
    """It returns int float np.ndarray[float]."""
    assert binding.kd(10, 7, 6.0) == 11
    assert binding.kd(10, 7, 7) == 20
    assert binding.kd(10, 7, 8) == 110.0
    np.testing.assert_allclose(
        binding.kd(10.0, 7.0, np.array([6.0, 8.0])), [11.0, 110.0]
    )
    np.testing.assert_allclose(binding.kd(10, 7, np.array([7, 6, 8.0])), [20, 11, 110])


def test_fit_titration() -> None:
    """It fits pH and Cl titrations."""
    x = [3.0, 5, 7, 9, 11.0]
    y = np.array([1.9991, 1.991, 1.5, 1.009, 1.0009])
    fit = binding.fit_titration("pH", x, y)
    assert abs(fit["K"][0] - 7) < 0.0000000001
    assert abs(fit["SA"][0] - 2) < 0.0001
    assert abs(fit["SB"][0] - 1) < 0.0001
    x = [0, 5.0, 10, 40, 160, 1000]
    y = np.array([2.0, 1.33333333, 1.0, 0.4, 0.11764706, 0.01980198])
    fit = binding.fit_titration("Cl", x, y)
    assert abs(fit["K"][0] - 10) < 0.0000001
    assert abs(fit["SA"][0] - 2) < 0.000000001
    assert abs(fit["SB"][0] - 0) < 0.00000001
    with pytest.raises(NameError, match="kind= pH or Cl"):
        binding.fit_titration("unk", x, y)


class TestFitBinding:
    """Test fit_binding()."""

    def test_fit_binding_glob(self) -> None:
        """Fit binding curves."""
        x = np.array([9.0, 8.0, 7.0, 6.0, 5.0])
        y = np.array(
            [1.99, 1.909, 1.5, 1.0909, 1.0099]
        )  # This is generated by dummy_model with K=7, S0=2, S1=1
        f_res = fit_binding_glob(Dataset(x, y, True))
        result = f_res.result
        assert result.success is True
        assert np.isclose(result.params["K"].value, 7, 1e-4)
        assert np.isclose(result.params["S0_default"].value, 2, 1e-4)
        assert np.isclose(result.params["S1_default"].value, 1, 1e-5)


def test_dataset_class() -> None:
    """Test Dataset and DataPair classes.

    This includes tests for:
    - Correct initialization with various input types (arrays and dictionaries).
    - Correct storage and retrieval of data pairs in the Dataset.
    - Appropriate error handling for mismatched keys and unequal data lengths.
    """
    # Define some test data
    x1 = np.array([5.5, 7.0, 8.5])
    x2 = np.array([6.8, 7.0, 7.2])
    y1 = np.array([2.1, 1.6, 1.1])
    y2 = np.array([1.8, 1.6, 1.4])
    # Test the case where x and y are both single ArrayF.
    # The key should default to "default".
    ds = Dataset(x1, y1)
    assert np.array_equal(ds["default"].x, x1)
    assert np.array_equal(ds["default"].y, y1)
    # Test the case where x is a single ArrayF and y is an ArrayDict.
    # The keys should come from y.
    ds = Dataset(x1, {"y1": y1, "y2": y2})
    assert np.array_equal(ds["y1"].x, x1)
    assert np.array_equal(ds["y2"].x, x1)
    assert np.array_equal(ds["y1"].y, y1)
    assert np.array_equal(ds["y2"].y, y2)
    # Test the case where x and y are both ArrayDict and keys match.
    # The keys should match between x and y.
    ds = Dataset({"1": x1, "2": x2}, {"1": y1, "2": y2})
    assert np.array_equal(ds["1"].x, x1)
    assert np.array_equal(ds["2"].x, x2)
    assert np.array_equal(ds["1"].y, y1)
    assert np.array_equal(ds["2"].y, y2)
    # Test the case where x and y are both ArrayDict and keys don't match.
    # This should raise a ValueError.
    with pytest.raises(
        ValueError,
        match=re.escape("Keys of 'x', 'y', and 'w' (if w is a dict) must match."),
    ):
        ds = Dataset({"x1": x1, "x2": x2}, {"y1": y1, "y2": y2})
    xx = np.array([6.8, 7.0, 7.2, 7.9])
    with pytest.raises(ValueError, match="Length of 'x' and 'y' must be equal."):
        ds = Dataset(xx, y1)
    ww = np.array([6.8, 7.0, 7.2, 7.9])
    with pytest.raises(ValueError, match="Length of 'x' and 'w' must be equal."):
        ds = Dataset(x1, y1, w=ww)


def test_dataset_copy() -> None:
    """Test deep copy."""
    x1 = np.array([5.5, 7.0, 8.5])
    y1 = np.array([2.1, 1.6, 1.1])
    y2 = np.array([1.8, 1.6, 1.4])
    ds = Dataset(x1, {"1": y1, "2": y2})
    # Test full copy
    ds_copy = ds.copy()
    assert ds_copy.is_ph == ds.is_ph
    assert "1" in ds_copy
    assert "2" in ds_copy
    # Test partial copy
    ds_copy = ds.copy(keys={"1"})
    assert ds_copy.is_ph == ds.is_ph
    assert "1" in ds_copy
    assert "2" not in ds_copy
    # Test KeyError
    with pytest.raises(KeyError):
        ds_copy = ds.copy(keys={"nonexistent"})
