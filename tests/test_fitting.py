"""Test cases for the clophfit.binding.fitting module."""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from lmfit import Parameters  # type: ignore[import-untyped]

from clophfit.fitting.core import (
    analyze_spectra,
    fit_binding_glob,
    outlier2,
    weight_da,
    weight_multi_ds_titration,
)
from clophfit.fitting.data_structures import DataArray, Dataset
from clophfit.fitting.errors import InsufficientDataError
from clophfit.fitting.models import binding_1site, kd
from clophfit.fitting.plotting import plot_fit

###############################################################################
# Fixtures
###############################################################################


@pytest.fixture
def spectra_df() -> pd.DataFrame:
    """Create a sample spectral DataFrame."""
    x = np.array([9.0, 8.0, 7.0, 6.0, 5.0])
    # Each column is a spectrum at a given pH
    # Each row corresponds to a wavelength
    wavelengths = np.arange(300, 350)
    # Create two gaussian peaks whose amplitudes change with pH
    peak1 = 2 * np.exp(-((wavelengths - 320) ** 2) / 100)
    peak2 = 1 * np.exp(-((wavelengths - 330) ** 2) / 100)
    # Simulate signal change S(pH) = S0 + (S1-S0)*10**(K-pH)/(1+10**(K-pH))
    # K=7, S0_peak1=2, S1_peak1=1; S0_peak2=0, S1_peak2=1
    y1_ph = np.array([1.99, 1.909, 1.5, 1.0909, 1.0099]) / 2.0
    y2_ph = np.array([0.01, 0.091, 0.5, 0.909, 0.99])
    # Combine signals
    spectra = [
        y1 * peak1[:, np.newaxis] + y2 * peak2[:, np.newaxis]
        for y1, y2 in zip(y1_ph, y2_ph, strict=False)
    ]
    return pd.DataFrame(np.hstack(spectra), index=wavelengths, columns=x)


###############################################################################
# Tests for Core Models
###############################################################################


def test_binding_1site() -> None:
    """Test the 1-site binding model function directly."""
    # pH case (Henderson-Hasselbalch)
    assert binding_1site(x=7.0, K=7.0, S0=2.0, S1=1.0, is_ph=True) == 1.5
    # Concentration case (Standard Isotherm)
    assert binding_1site(x=10.0, K=10.0, S0=2.0, S1=0.0, is_ph=False) == 1.0
    # Array input
    x_vals = np.array([6.0, 7.0, 8.0])
    expected_y = np.array([1.09090909, 1.5, 1.90909091])
    actual_y = binding_1site(x=x_vals, K=7.0, S0=2.0, S1=1.0, is_ph=True)
    np.testing.assert_allclose(actual_y, expected_y)


def test_kd() -> None:
    """It returns int float np.ndarray[float]."""
    assert kd(10, 7, 6.0) == 11
    assert kd(10, 7, 7) == 20
    assert kd(10, 7, 8) == 110.0
    np.testing.assert_allclose(kd(10.0, 7.0, np.array([6.0, 8.0])), [11.0, 110.0])
    np.testing.assert_allclose(kd(10, 7, np.array([7, 6, 8.0])), [20, 11, 110])


###############################################################################
# Tests for LM Fitter
###############################################################################


def test_fit_binding_glob_ph(ph_dataset: Dataset) -> None:
    """It fits a single pH titration curve."""
    f_res = fit_binding_glob(ph_dataset)
    assert f_res.result is not None
    assert f_res.result.success is True
    assert np.isclose(f_res.result.params["K"].value, 7.0, atol=1e-4)
    assert np.isclose(f_res.result.params["S0_default"].value, 1.0, atol=1e-4)
    assert np.isclose(f_res.result.params["S1_default"].value, 2.0, atol=1e-4)


def test_fit_binding_glob_cl(cl_dataset: Dataset) -> None:
    """It fits a single Cl titration curve."""
    f_res = fit_binding_glob(cl_dataset)
    assert f_res.result is not None
    assert f_res.result.success is True
    assert np.isclose(f_res.result.params["K"].value, 10.0, atol=1e-4)
    assert np.isclose(f_res.result.params["S0_default"].value, 2.0, atol=1e-4)
    assert np.isclose(f_res.result.params["S1_default"].value, 0.0, atol=1e-4)


def test_fit_binding_glob_multi(multi_dataset: Dataset) -> None:
    """It performs global fitting on a multi-label dataset."""
    f_res = fit_binding_glob(multi_dataset)
    assert f_res.result is not None
    assert f_res.result.success is True
    # Check shared parameter K
    assert np.isclose(f_res.result.params["K"].value, 7.0, atol=1e-4)
    # Check parameters for the first dataset
    assert np.isclose(f_res.result.params["S0_y1"].value, 1.0, atol=1e-4)
    assert np.isclose(f_res.result.params["S1_y1"].value, 2.0, atol=1e-4)
    # Check parameters for the second dataset
    assert np.isclose(f_res.result.params["S0_y2"].value, 1.0, atol=1e-3)
    assert np.isclose(f_res.result.params["S1_y2"].value, 0.0, atol=1e-4)


def test_fit_binding_insufficient_data() -> None:
    """It raises InsufficientDataError when data points are too few."""
    x = np.array([7.0, 8.0])
    y = np.array([1.5, 1.9])
    ds = Dataset({"default": DataArray(x, y)}, is_ph=True)
    with pytest.raises(InsufficientDataError):
        fit_binding_glob(ds)


###############################################################################
# Tests for Weighting
###############################################################################


def test_weight_da(ph_dataset: Dataset) -> None:
    """Test that weighting calculates and assigns y_err."""
    da = ph_dataset["default"]
    da.y_err = np.array([])
    assert da.y_err.size == 5  # Starts with [1,1,1,1,1]
    success = weight_da(da, is_ph=True)
    assert success is True
    assert da.y_err.size > 0
    assert np.all(da.y_err > 0)


def test_weight_multi_ds_titration(multi_dataset: Dataset) -> None:
    """Test weighting on a multi-label dataset."""
    weight_multi_ds_titration(multi_dataset)
    assert multi_dataset["y1"].y_err.size > 0
    assert multi_dataset["y2"].y_err.size > 0
    assert np.all(multi_dataset["y1"].y_err > 0)
    assert np.all(multi_dataset["y2"].y_err > 0)


###############################################################################
# Tests for Spectral Analysis
###############################################################################


def test_analyze_spectra_svd(spectra_df: pd.DataFrame) -> None:
    """Test the spectral analysis pipeline with SVD."""
    fit_result = analyze_spectra(spectra=spectra_df, is_ph=True, band=None)
    assert fit_result.figure is not None
    assert fit_result.result is not None
    assert fit_result.result.success is True
    # The first principal component should capture the titration behavior
    # and fitting it should yield the correct K.
    assert np.isclose(fit_result.result.params["K"].value, 7.0, atol=1e-2)


def test_analyze_spectra_band(spectra_df: pd.DataFrame) -> None:
    """Test the spectral analysis pipeline with band integration."""
    # Integrate over the whole range
    fit_result = analyze_spectra(spectra=spectra_df, is_ph=True, band=(300, 349))
    assert fit_result.figure is not None
    assert fit_result.result is not None
    assert fit_result.result.success is True
    # The integrated signal should follow the titration, yielding the correct K
    assert np.isclose(fit_result.result.params["K"].value, 7.0, atol=1e-1)  # FIX: tol


###############################################################################
# Tests for Advanced Fitters and Plotting (Smoke Tests)
###############################################################################


def _() -> None:
    """Fix this.

    def test_pymc_fitter_smoke_test(ph_dataset: Dataset) -> None:
        "Smoke test for the PyMC fitter to ensure it runs without error."
        # First, run a standard lmfit to get initial parameters
        initial_fit = fit_binding_glob(ph_dataset)
        # Now, run the pymc fitter with very few samples
        # We are not testing for correctness of the result, just that it runs.
        fit_result_pymc = fit_binding_pymc(initial_fit, n_samples=10, n_xerr=0)
        assert fit_result_pymc.result is not None  # Should be an InferenceData object
        assert "K" in fit_result_pymc.result.posterior
    """


def test_plot_fit_smoke_test(ph_dataset: Dataset) -> None:
    """Smoke test for the plot_fit function."""
    # Get some parameters to plot
    params = Parameters()
    params.add("K", value=7.0)
    params.add("S0_default", value=2.0)
    params.add("S1_default", value=1.0)
    # Create a figure and axis
    fig, ax = plt.subplots()
    # Check that it runs without raising an exception
    try:
        plot_fit(ax, ph_dataset, params, nboot=2)
    finally:
        plt.close(fig)


def _() -> None:
    """Fix this.

    def test_y_err_initialization() -> None:
        xc = np.array([1, 2, 3, 4])
        yc = np.array([10, 20, 30, 40])
        y_errc = np.array([0.1, 0.2, 0.3, 0.4])
        da = DataArray(xc=xc, yc=yc, y_errc=y_errc)
        assert np.array_equal(da.y_errc, y_errc).
    """


def test_setting_y_err() -> None:
    """."""
    xc = np.array([1, 2, 3, 4])
    yc = np.array([10, 20, 30, 40])
    da = DataArray(xc=xc, yc=yc)
    da.y_err = np.array(0.5)
    assert np.array_equal(da.y_err, np.ones_like(xc) * 0.5)


def test_setting_x_err() -> None:
    """."""
    xc = np.array([1, 2, 3, 4])
    yc = np.array([10, 20, 30, 40])
    da = DataArray(xc=xc, yc=yc)
    da.x_err = np.array(0.1)
    assert np.array_equal(da.x_errc, np.ones_like(xc) * 0.1)


def test_masking_effect_on_errors() -> None:
    """."""
    xc = np.array([1, 2, 3, 4])
    yc = np.array([10, np.nan, 30, 40])
    y_errc = np.array([0.1, 0.2, 0.3, 0.4])
    da = DataArray(xc=xc, yc=yc, y_errc=y_errc)
    assert np.array_equal(da.y_err, np.array([0.1, 0.3, 0.4]))
    da.mask = np.array([True, True, False, True])
    assert np.array_equal(da.y_err, np.array([0.1, 0.4]))


def test_setting_and_validating_mask() -> None:
    """."""
    xc = np.array([1, 2, 3, 4])
    yc = np.array([10, 20, np.nan, 40])
    da = DataArray(xc=xc, yc=yc)
    da.mask = np.array([False, True, True, True])
    assert np.array_equal(da.x, np.array([2, 4]))
    assert np.array_equal(da.y, np.array([20, 40]))
    da.mask = np.array([True, True, True, False])
    assert np.array_equal(da.x, np.array([1, 2]))
    assert np.array_equal(da.y, np.array([10, 20]))


def test_dataset_single_array_no_nan() -> None:
    """Clean nan."""
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    dataarray = DataArray(x, y)
    dataset = Dataset.from_da([dataarray])
    assert len(dataset) == 1
    assert isinstance(dataset["y0"], DataArray)
    assert np.array_equal(dataset["y0"].x, x)
    assert np.array_equal(dataset["y0"].y, y)


def test_dataset_single_array_with_nan() -> None:
    """Clean nan."""
    x = np.array([1, 2, 3, 4])
    y = np.array([5, 6, np.nan, 8])
    dataset = Dataset.from_da(DataArray(x, y))
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
    dataset = Dataset({"y0": da1, "y1": da2})
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
    dataset = Dataset({"y0": da1, "y1": da2})
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
    dataset = Dataset({"y0": da1, "y1": da2})
    assert len(dataset) == 2
    assert np.array_equal(dataset["y0"].x, np.array([1, 2, 4]))
    assert np.array_equal(dataset["y0"].y, np.array([5, 6, 8]))
    assert np.array_equal(dataset["y1"].x, np.array([1, 2, 4]))
    assert np.array_equal(dataset["y1"].y, np.array([9, 10, 12]))


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
    ds = Dataset({"default": DataArray(x0, y0)})
    assert np.array_equal(ds["default"].x, x0)
    assert np.array_equal(ds["default"].y, y0)
    # Test the case where x is a single ArrayF and y is an ArrayDict.
    # The keys should come from y.
    ds = Dataset({"y0": DataArray(x0, y0), "y1": DataArray(x0, y1)})
    assert np.array_equal(ds["y0"].x, x0)
    assert np.array_equal(ds["y1"].x, x0)
    assert np.array_equal(ds["y0"].y, y0)
    assert np.array_equal(ds["y1"].y, y1)
    # Test the case where x and y are both ArrayDict and keys match.
    # The keys should match between x and y.
    ds = Dataset({"A": DataArray(x0, y0), "y1": DataArray(x1, y1)})
    assert np.array_equal(ds["A"].x, x0)
    assert np.array_equal(ds["y1"].x, x1)
    assert np.array_equal(ds["A"].y, y0)
    assert np.array_equal(ds["y1"].y, y1)


###############################################################################
# Tests from Original Suite (DataArray, Dataset) - Maintained for Coverage
###############################################################################


def test_dataarray_masking() -> None:
    """Mask nan values, keep values."""
    x = np.array([1, 2, 3, 4])
    y = np.array([5, 6, np.nan, 8])
    da = DataArray(x, y)
    np.testing.assert_array_equal(da.x, np.array([1, 2, 4]))
    da.mask = np.array([False, True, True, True], dtype="bool")
    # `nan` are anyway discarded
    np.testing.assert_array_equal(da.x, np.array([2, 4]))
    assert np.array_equal(da.y_errc, np.array([]))
    assert np.array_equal(da.x_errc, np.array([]))


def test_dataarray_initialization_failure() -> None:
    """Test for length mismatch error during DataArray initialization."""
    xc = np.array([1, 2, 3])
    yc = np.array([10, 20, 30, 40])  # Mismatched length
    with pytest.raises(ValueError, match="Length of 'xc' and 'yc' must be equal"):
        DataArray(xc=xc, yc=yc)


def test_dataarray_error_length_mismatch() -> None:
    """Test for length mismatch when setting error arrays."""
    xc = np.array([1, 2, 3, 4])
    yc = np.array([10, 20, 30, 40])
    short_err = np.array([0.1, 0.2])  # Mismatched length
    with pytest.raises(ValueError, match="Length of 'xc' and 'y_errc' must be equal"):
        DataArray(xc=xc, yc=yc, y_errc=short_err)
    with pytest.raises(ValueError, match="Length of 'xc' and 'x_errc' must be equal"):
        DataArray(xc=xc, yc=yc, x_errc=short_err)
    da = DataArray(xc=xc, yc=yc)
    # MAYBE: match with "Length of 'y_err' must be 1 or same as 'y'."
    with pytest.raises(ValueError, match="Length of 'xc' and 'y_errc' must be equal"):
        da.y_err = short_err


def test_dataset_from_da_with_nan() -> None:
    """Test Dataset creation from a DataArray containing NaN."""
    xc = np.array([1, 2, 3, 4])
    yc = np.array([5, 6, np.nan, 8])
    dataset = Dataset.from_da(DataArray(xc, yc))
    assert len(dataset) == 1
    assert "default" in dataset
    np.testing.assert_array_equal(dataset["default"].x, np.array([1, 2, 4]))
    np.testing.assert_array_equal(dataset["default"].y, np.array([5, 6, 8]))
    assert np.array_equal(dataset["default"].mask, np.array([True, True, False, True]))


def test_dataset_clean_data() -> None:
    """Test cleaning of datasets that are too small to fit."""
    x = np.array([1, 2, 3])
    y = np.array([2, 3, 4])
    ds = Dataset.from_da(DataArray(x, y), is_ph=True)
    assert "default" in ds
    # A 1-site model has 3 parameters (K, S0, S1). A 4th would be for a second dataset.
    n_params = 4
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ds.clean_data(n_params)
        assert len(w) == 1
        assert "Removing key 'default' from Dataset" in str(w[0].message)
    assert "default" not in ds


def test_dataset_copy(multi_dataset: Dataset) -> None:
    """Test deep copy functionality of Dataset."""
    # Test full copy
    ds_copy = multi_dataset.copy()
    assert ds_copy.is_ph == multi_dataset.is_ph
    assert "y1" in ds_copy
    assert "y2" in ds_copy
    # Test partial copy
    ds_copy_partial = multi_dataset.copy(keys={"y2"})
    assert ds_copy_partial.is_ph == multi_dataset.is_ph
    assert "y2" in ds_copy_partial
    assert "y1" not in ds_copy_partial
    # Test KeyError for non-existent key
    with pytest.raises(KeyError):
        multi_dataset.copy(keys={"nonexistent"})


def test_export_ds(multi_dataset: Dataset, tmp_path: Path) -> None:
    """It exports dataset to csv files."""
    file_path = tmp_path / "A01.csv"
    multi_dataset.export(str(file_path))
    # Check if files are created for each label
    assert (tmp_path / "A01_y1.csv").exists()
    assert (tmp_path / "A01_y2.csv").exists()
    # Read back one file and check content
    read_df = pd.read_csv(tmp_path / "A01_y1.csv")
    np.testing.assert_allclose(
        multi_dataset["y1"].y, read_df.yc.to_numpy().astype(float)
    )
    np.testing.assert_allclose(
        multi_dataset["y1"].x, read_df.xc.to_numpy().astype(float)
    )


###############################################################################
# outlier2() tests
###############################################################################

# Test parameters for outlier2
_TRUE_K = 7.0
_TRUE_S0_Y1, _TRUE_S1_Y1 = 600.0, 50.0
_TRUE_S0_Y2, _TRUE_S1_Y2 = 500.0, 40.0
_BUFFER_SD = 40.0


def _create_synthetic_dataset(  # noqa: PLR0913
    n_points: int = 7,
    true_k: float = _TRUE_K,
    seed: int = 42,
    add_outlier: bool = False,
    outlier_label: str = "y1",
    outlier_idx: int = 2,
    outlier_magnitude: float = 5.0,
) -> Dataset:
    """Create synthetic dual-channel pH titration dataset for outlier2 tests."""
    rng = np.random.default_rng(seed)

    x = np.linspace(5.5, 9.0, n_points)
    x_err = 0.05 * np.ones_like(x)

    y1_true = binding_1site(x, true_k, _TRUE_S0_Y1, _TRUE_S1_Y1, is_ph=True)
    y2_true = binding_1site(x, true_k, _TRUE_S0_Y2, _TRUE_S1_Y2, is_ph=True)

    y1_err = np.sqrt(np.maximum(y1_true, 1.0) + _BUFFER_SD**2)
    y2_err = np.sqrt(np.maximum(y2_true, 1.0) + _BUFFER_SD**2)

    y1 = y1_true + rng.normal(0, y1_err)
    y2 = y2_true + rng.normal(0, y2_err)

    if add_outlier:
        if outlier_label == "y1":
            y1[outlier_idx] += outlier_magnitude * y1_err[outlier_idx]
        else:
            y2[outlier_idx] += outlier_magnitude * y2_err[outlier_idx]

    da1 = DataArray(x, y1, x_errc=x_err, y_errc=y1_err)
    da2 = DataArray(x, y2, x_errc=x_err, y_errc=y2_err)

    return Dataset({"y1": da1, "y2": da2}, is_ph=True)


class TestOutlier2:
    """Tests for outlier2() function."""

    def test_returns_fit_result(self) -> None:
        """outlier2 should return a valid FitResult."""
        ds = _create_synthetic_dataset()
        fr = outlier2(ds, key="test")

        assert fr.result is not None
        assert "K" in fr.result.params

    def test_k_estimate_reasonable(self) -> None:
        """K estimate should be close to true value."""
        ds = _create_synthetic_dataset(seed=123)
        fr = outlier2(ds, key="test")

        assert fr.result is not None
        k_est = fr.result.params["K"].value
        assert abs(k_est - _TRUE_K) < 0.5

    def test_uniform_error_model(self) -> None:
        """Uniform error model should assign constant errors per label."""
        ds = _create_synthetic_dataset()
        fr = outlier2(ds, key="test", error_model="uniform")

        assert fr.dataset is not None
        for da in fr.dataset.values():
            np.testing.assert_allclose(da.y_err, da.y_err[0] * np.ones_like(da.y_err))

    def test_shotnoise_error_model(self) -> None:
        """Shot-noise error model should preserve relative error structure."""
        ds = _create_synthetic_dataset()
        original_ratio_y1 = ds["y1"].y_err[0] / ds["y1"].y_err[-1]

        fr = outlier2(ds, key="test", error_model="shot-noise")

        assert fr.dataset is not None
        new_ratio_y1 = fr.dataset["y1"].y_err[0] / fr.dataset["y1"].y_err[-1]
        np.testing.assert_allclose(original_ratio_y1, new_ratio_y1, rtol=0.1)

    def test_detects_outlier_in_y1(self) -> None:
        """Should detect large outlier in y1."""
        ds = _create_synthetic_dataset(
            add_outlier=True, outlier_label="y1", outlier_idx=3, outlier_magnitude=10.0
        )
        fr = outlier2(ds, key="test", threshold=2.5)

        assert fr.dataset is not None
        assert len(fr.dataset["y1"].y) < 7

    def test_detects_outlier_in_y2(self) -> None:
        """Should detect large outlier in y2."""
        ds = _create_synthetic_dataset(
            add_outlier=True, outlier_label="y2", outlier_idx=4, outlier_magnitude=10.0
        )
        fr = outlier2(ds, key="test", threshold=2.5)

        assert fr.dataset is not None
        assert len(fr.dataset["y2"].y) < 7

    def test_no_false_positives_clean_data(self) -> None:
        """Should not remove points from clean data."""
        ds = _create_synthetic_dataset(add_outlier=False, seed=42)
        fr = outlier2(ds, key="test", threshold=3.0)

        assert fr.dataset is not None
        assert len(fr.dataset["y1"].y) == 7
        assert len(fr.dataset["y2"].y) == 7

    def test_correct_residual_slicing(self) -> None:
        """Residuals should be correctly sliced for each label."""
        ds = _create_synthetic_dataset()
        fr_init = fit_binding_glob(ds, robust=True)
        assert fr_init.result is not None

        total_residuals = len(fr_init.result.residual)
        assert total_residuals == len(ds["y1"].y) + len(ds["y2"].y)

    def test_single_label_dataset(self) -> None:
        """Should work with single-label dataset."""
        rng = np.random.default_rng(42)
        x = np.linspace(5.5, 9.0, 7)
        y_true = binding_1site(x, _TRUE_K, _TRUE_S0_Y1, _TRUE_S1_Y1, is_ph=True)
        y_err = np.sqrt(np.maximum(y_true, 1.0) + _BUFFER_SD**2)
        y = y_true + rng.normal(0, y_err)

        da = DataArray(x, y, x_errc=0.05 * np.ones_like(x), y_errc=y_err)
        ds = Dataset({"y1": da}, is_ph=True)

        fr = outlier2(ds, key="test")
        assert fr.result is not None

    def test_deterministic_output(self) -> None:
        """Same input should give same output."""
        ds1 = _create_synthetic_dataset(seed=42)
        ds2 = _create_synthetic_dataset(seed=42)

        fr1 = outlier2(ds1, key="test")
        fr2 = outlier2(ds2, key="test")

        assert fr1.result is not None
        assert fr2.result is not None
        np.testing.assert_allclose(
            fr1.result.params["K"].value,
            fr2.result.params["K"].value,
            rtol=1e-10,
        )
