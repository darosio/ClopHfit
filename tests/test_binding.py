"""Test cases for the binding functions module."""

import warnings

import numpy as np
import pytest

from clophfit import binding
from clophfit.binding.fitting import DataArray, Dataset, fit_binding_glob


def test_kd() -> None:
    """It returns int float np.ndarray[float]."""
    kd = binding.fitting.kd
    assert kd(10, 7, 6.0) == 11
    assert kd(10, 7, 7) == 20
    assert kd(10, 7, 8) == 110.0
    np.testing.assert_allclose(kd(10.0, 7.0, np.array([6.0, 8.0])), [11.0, 110.0])
    np.testing.assert_allclose(kd(10, 7, np.array([7, 6, 8.0])), [20, 11, 110])


def test_fit_binding_glob() -> None:
    """It fits pH and Cl individual titration curves."""
    # First pH titration
    x = np.array([9.0, 8.0, 7.0, 6.0, 5.0])
    y = np.array(
        [1.99, 1.909, 1.5, 1.0909, 1.0099]
    )  # This is generated by dummy_model with K=7, S0=2, S1=1
    f_res = fit_binding_glob(Dataset([DataArray(x, y)], is_ph=True))
    assert f_res is not None
    assert f_res.result is not None
    result = f_res.result
    assert result.success is True
    assert np.isclose(result.params["K"].value, 7, 1e-4)
    assert np.isclose(result.params["S0_default"].value, 2, 1e-4)
    assert np.isclose(result.params["S1_default"].value, 1, 1e-5)
    # pH
    x = np.array([3.0, 5, 7, 9, 11.0])
    y = np.array([1.9991, 1.991, 1.5, 1.009, 1.0009])
    f_res = fit_binding_glob(Dataset([DataArray(x, y)], is_ph=True))
    assert f_res is not None
    result = f_res.result
    assert result.success is True
    assert np.isclose(result.params["K"].value, 7, 1e-5)
    assert np.isclose(result.params["S0_default"].value, 1, 1e-4)
    assert np.isclose(result.params["S1_default"].value, 2, 1e-4)
    # Cl
    x = np.array([0, 5.0, 10, 40, 160, 1000])
    y = np.array([2.0, 1.33333333, 1.0, 0.4, 0.11764706, 0.01980198])
    f_res = fit_binding_glob(Dataset([DataArray(x, y)], is_ph=False))
    assert f_res is not None
    result = f_res.result
    assert result.success is True
    assert np.isclose(result.params["K"].value, 10, 1e-5)
    assert np.isclose(result.params["S0_default"].value, 2, 1e-4)
    assert np.isclose(result.params["S1_default"].value, 0, 1e-4)


def test_dataarray() -> None:
    """Mask nan values, keep values."""
    x = np.array([1, 2, 3, 4])
    y = np.array([5, 6, np.nan, 8])
    dataarray = DataArray(x, y)
    assert np.array_equal(dataarray.x, np.array([1, 2, 4]))
    dataarray.mask = np.array([0, 1, 1, 1], dtype="bool")
    # `nan` are anyway discarded
    assert np.array_equal(dataarray.x, np.array([2, 4]))


def test_dataset_single_array_no_nan() -> None:
    """Clean nan."""
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    dataarray = DataArray(x, y)
    dataset = Dataset([dataarray])
    assert len(dataset) == 1
    assert isinstance(dataset["default"], DataArray)
    assert np.array_equal(dataset["default"].x, x)
    assert np.array_equal(dataset["default"].y, y)


def test_dataset_single_array_with_nan() -> None:
    """Clean nan."""
    x = np.array([1, 2, 3, 4])
    y = np.array([5, 6, np.nan, 8])
    dataset = Dataset([DataArray(x, y)])
    assert len(dataset) == 1
    assert isinstance(dataset["default"], DataArray)
    assert np.array_equal(dataset["default"].x, np.array([1, 2, 4]))
    assert np.array_equal(dataset["default"].y, np.array([5, 6, 8]))


def test_dataset_dict_no_nan() -> None:
    """Clean nan."""
    x0, x1 = np.array([1, 2, 3]), np.array([4, 5, 6])
    y0, y1 = np.array([7, 8, 9]), np.array([10, 11, 12])
    da1 = DataArray(x0, y0)
    da2 = DataArray(x1, y1)
    dataset = Dataset([da1, da2])
    assert len(dataset) == 2
    assert np.array_equal(dataset["y0"].x, x0)
    assert np.array_equal(dataset["y0"].y, y0)
    assert np.array_equal(dataset["y1"].x, x1)
    assert np.array_equal(dataset["y1"].y, y1)


def test_dataset_dict_with_nan() -> None:
    """Clean nan."""
    x0, x1 = np.array([1, 2, 3]), np.array([4, 5, 6])
    y0, y1 = np.array([7, np.nan, 9]), np.array([10, 11, np.nan])
    da1 = DataArray(x0, y0)
    da2 = DataArray(x1, y1)
    dataset = Dataset([da1, da2])
    assert len(dataset) == 2
    assert np.array_equal(dataset["y0"].x, np.array([1, 3]))
    assert np.array_equal(dataset["y0"].y, np.array([7, 9]))
    assert np.array_equal(dataset["y1"].x, np.array([4, 5]))
    assert np.array_equal(dataset["y1"].y, np.array([10, 11]))


def test_dataset_single_x_dict_y_with_nan() -> None:
    """Clean nan."""
    x = np.array([1, 2, 3, 4])
    y0, y1 = np.array([5, 6, np.nan, 8]), np.array([9, 10, np.nan, 12])
    da1 = DataArray(x, y0)
    da2 = DataArray(x, y1)
    dataset = Dataset([da1, da2])
    assert len(dataset) == 2
    assert np.array_equal(dataset["y0"].x, np.array([1, 2, 4]))
    assert np.array_equal(dataset["y0"].y, np.array([5, 6, 8]))
    assert np.array_equal(dataset["y1"].x, np.array([1, 2, 4]))
    assert np.array_equal(dataset["y1"].y, np.array([9, 10, 12]))


def test_dataset_clean_data() -> None:
    """Clean too small datasets."""
    x = np.array([1, 2, 3])
    y = np.array([2, 3, 4])
    ds = Dataset([DataArray(x, y)], is_ph=True)
    # Check initial keys
    assert "default" in ds
    # Call clean_data and check for warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")  # Always record warnings
        ds.clean_data(4)
        assert len(w) == 1  # Check that one warning was issued
        assert str(w[0].message) == (
            "Removing key 'default' from Dataset: number of parameters (4) exceeds "
            "number of data points (3)."
        )
    # Check keys after clean_data
    assert "default" not in ds


def test_dataset_class() -> None:
    """Test Dataset and DataPair classes.

    This includes tests for:
    - Correct initialization with various input types (arrays and dictionaries).
    - Correct storage and retrieval of data pairs in the Dataset.
    - Appropriate error handling for mismatched keys and unequal data lengths.
    """
    # Define some test data
    x0 = np.array([5.5, 7.0, 8.5])
    x1 = np.array([6.8, 7.0, 7.2])
    y0 = np.array([2.1, 1.6, 1.1])
    y1 = np.array([1.8, 1.6, 1.4])
    # Test the case where x and y are both single ArrayF.
    # The key should default to "default".
    ds = Dataset([DataArray(x0, y0)])
    assert np.array_equal(ds["default"].x, x0)
    assert np.array_equal(ds["default"].y, y0)
    # Test the case where x is a single ArrayF and y is an ArrayDict.
    # The keys should come from y.
    ds = Dataset([DataArray(x0, y0), DataArray(x0, y1)])
    assert np.array_equal(ds["y0"].x, x0)
    assert np.array_equal(ds["y1"].x, x0)
    assert np.array_equal(ds["y0"].y, y0)
    assert np.array_equal(ds["y1"].y, y1)
    # Test the case where x and y are both ArrayDict and keys match.
    # The keys should match between x and y.
    ds = Dataset([DataArray(x0, y0), DataArray(x1, y1)])
    assert np.array_equal(ds["y0"].x, x0)
    assert np.array_equal(ds["y1"].x, x1)
    assert np.array_equal(ds["y0"].y, y0)
    assert np.array_equal(ds["y1"].y, y1)


def test_dataset_copy() -> None:
    """Test deep copy."""
    x0 = np.array([5.5, 7.0, 8.5])
    y0 = np.array([2.1, 1.6, 1.1])
    y1 = np.array([1.8, 1.6, 1.4])
    ds = Dataset([DataArray(x0, y0), DataArray(x0, y1)])
    # Test full copy
    ds_copy = ds.copy()
    assert ds_copy.is_ph == ds.is_ph
    assert "y0" in ds_copy
    assert "y1" in ds_copy
    # Test partial copy
    ds_copy = ds.copy(keys={"y1"})
    assert ds_copy.is_ph == ds.is_ph
    assert "y1" in ds_copy
    assert "y0" not in ds_copy
    # Test KeyError
    with pytest.raises(KeyError):
        ds_copy = ds.copy(keys={"nonexistent"})
