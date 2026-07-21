"""Test cases for the clophfit.binding.fitting module."""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from lmfit import Parameters  # type: ignore[import-untyped]
from lmfit.minimizer import MinimizerResult  # type: ignore[import-untyped]
from matplotlib.figure import Figure

from clophfit.fitting.bayes import (
    fit_binding_pymc,
    fit_binding_pymc_multi,
)
from clophfit.fitting.bayes_config import NoiseConfig, SamplerConfig
from clophfit.fitting.core import (
    analyze_spectra,
    fit_binding_glob,
    weight_da,
    weight_multi_ds_titration,
)
from clophfit.fitting.data_structures import (
    DataArray,
    Dataset,
    FitResult,
    NoiseModelParams,
    PlateNoiseModel,
)
from clophfit.fitting.errors import InsufficientDataError, InvalidDataError
from clophfit.fitting.models import binding_1site, kd
from clophfit.fitting.odr import (
    fit_binding_odr,
)
from clophfit.fitting.plotting import (
    extract_sigma_df,
    plot_fit,
    plot_noise_vs_signal,
    plot_qc_mean_vs_std,
    plot_qc_span_vs_center,
    plot_qc_span_vs_center_titration,
    qc_flag_bad_wells,
    qc_flag_bad_wells_titration,
)
from clophfit.fitting.utils import (
    add_robust_scores,
    bonferroni_threshold,
    cap_by_min_keep,
    identify_outliers_mad,
    parse_remove_outliers,
    robust_scale,
    robust_z_scores,
    studentized_scores,
)
from clophfit.prtecan import PlateScheme

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
    assert np.isclose(f_res.result.params["S0_1"].value, 1.0, atol=1e-4)
    assert np.isclose(f_res.result.params["S1_1"].value, 2.0, atol=1e-4)
    # Check parameters for the second dataset
    assert np.isclose(f_res.result.params["S0_2"].value, 1.0, atol=1e-3)
    assert np.isclose(f_res.result.params["S1_2"].value, 0.0, atol=1e-4)


def test_fitresult_routes_backend_to_named_field(ph_dataset: Dataset) -> None:
    """Each backend object lands in its own field, the others stay None."""
    lm = fit_binding_glob(ph_dataset, method="lm")
    assert lm.mini is not None
    assert lm.trace is None
    assert lm.odr is None

    odr = fit_binding_odr(ph_dataset)
    assert odr.odr is not None
    assert odr.mini is None
    assert odr.trace is None


def test_fitresult_is_valid_true_for_each_backend(ph_dataset: Dataset) -> None:
    """is_valid() no longer requires the lmfit-specific mini field."""
    assert fit_binding_glob(ph_dataset, method="lm").is_valid()
    assert not FitResult().is_valid()


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
    assert multi_dataset["1"].y_err.size > 0
    assert multi_dataset["2"].y_err.size > 0
    assert np.all(multi_dataset["1"].y_err > 0)
    assert np.all(multi_dataset["2"].y_err > 0)


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
        fit_result_pymc = fit_binding_pymc(
            initial_fit, n_xerr=0, sampler=SamplerConfig(n_samples=10)
        )
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
    assert isinstance(dataset["0"], DataArray)
    assert np.array_equal(dataset["0"].x, x)
    assert np.array_equal(dataset["0"].y, y)


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
    dataset = Dataset({"0": da1, "1": da2})
    assert len(dataset) == 2
    assert np.array_equal(dataset["0"].x, x0)
    assert np.array_equal(dataset["0"].y, y0)
    assert np.array_equal(dataset["1"].x, x1)
    assert np.array_equal(dataset["1"].y, y1)


def test_dataset_dict_with_nan() -> None:
    """Clean nan."""
    x0, x1 = np.array([1, 2, 3]), np.array([4, 5, 6])
    y0, y1 = np.array([7, np.nan, 9]), np.array([10, 11, np.nan])
    da1 = DataArray(x0, y0)
    da2 = DataArray(x1, y1)
    dataset = Dataset({"0": da1, "1": da2})
    assert len(dataset) == 2
    assert np.array_equal(dataset["0"].x, np.array([1, 3]))
    assert np.array_equal(dataset["0"].y, np.array([7, 9]))
    assert np.array_equal(dataset["1"].x, np.array([4, 5]))
    assert np.array_equal(dataset["1"].y, np.array([10, 11]))


def test_dataset_single_x_dict_y_with_nan() -> None:
    """Clean nan."""
    x = np.array([1, 2, 3, 4])
    y0, y1 = np.array([5, 6, np.nan, 8]), np.array([9, 10, np.nan, 12])
    da1 = DataArray(x, y0)
    da2 = DataArray(x, y1)
    dataset = Dataset({"0": da1, "1": da2})
    assert len(dataset) == 2
    assert np.array_equal(dataset["0"].x, np.array([1, 2, 4]))
    assert np.array_equal(dataset["0"].y, np.array([5, 6, 8]))
    assert np.array_equal(dataset["1"].x, np.array([1, 2, 4]))
    assert np.array_equal(dataset["1"].y, np.array([9, 10, 12]))


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
    ds = Dataset({"0": DataArray(x0, y0), "1": DataArray(x0, y1)})
    assert np.array_equal(ds["0"].x, x0)
    assert np.array_equal(ds["1"].x, x0)
    assert np.array_equal(ds["0"].y, y0)
    assert np.array_equal(ds["1"].y, y1)
    # Test the case where x and y are both ArrayDict and keys match.
    # The keys should match between x and y.
    ds = Dataset({"A": DataArray(x0, y0), "1": DataArray(x1, y1)})
    assert np.array_equal(ds["A"].x, x0)
    assert np.array_equal(ds["1"].x, x1)
    assert np.array_equal(ds["A"].y, y0)
    assert np.array_equal(ds["1"].y, y1)


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


def test_mask_label_steps_excludes_only_that_label() -> None:
    """mask_label_steps drops given steps for one label, leaving others intact."""
    x = np.array([9.0, 8.0, 7.0, 6.0, 5.0])
    y = np.array([2.0, 1.8, 1.5, 1.2, 1.0])
    ds = Dataset({"1": DataArray(x, y), "2": DataArray(x, y * 2)}, is_ph=True)

    ds.mask_label_steps("1", [0, 4])  # drop the pH extremes on label 1 only
    np.testing.assert_array_equal(ds["1"].x, np.array([8.0, 7.0, 6.0]))
    np.testing.assert_array_equal(ds["2"].x, x)  # label 2 untouched

    # out-of-range indices are ignored; missing label raises
    ds["1"].mask_steps([99, -1])
    np.testing.assert_array_equal(ds["1"].x, np.array([8.0, 7.0, 6.0]))
    with pytest.raises(KeyError, match="Label '3' not in dataset"):
        ds.mask_label_steps("3", [0])


def test_dataarray_initialization_failure() -> None:
    """Test for length mismatch error during DataArray initialization."""
    xc = np.array([1, 2, 3])
    yc = np.array([10, 20, 30, 40])  # Mismatched length
    with pytest.raises(InvalidDataError, match="Length of 'xc' and 'yc' must be equal"):
        DataArray(xc=xc, yc=yc)


def test_dataarray_error_length_mismatch() -> None:
    """Test for length mismatch when setting error arrays."""
    xc = np.array([1, 2, 3, 4])
    yc = np.array([10, 20, 30, 40])
    da = DataArray(xc=xc, yc=yc)

    with pytest.raises(InvalidDataError):
        da.y_err = np.array([0.1, 0.2])  # Wrong length

    with pytest.raises(InvalidDataError):
        da.x_err = np.array([0.1])  # Wrong length


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
    assert "1" in ds_copy
    assert "2" in ds_copy
    # Test partial copy
    ds_copy_partial = multi_dataset.copy(keys={"2"})
    assert ds_copy_partial.is_ph == multi_dataset.is_ph
    assert "2" in ds_copy_partial
    assert "1" not in ds_copy_partial
    # Test KeyError for non-existent key
    with pytest.raises(KeyError):
        multi_dataset.copy(keys={"nonexistent"})


def test_export_ds(multi_dataset: Dataset, tmp_path: Path) -> None:
    """It exports dataset to csv files."""
    file_path = tmp_path / "A01.csv"
    multi_dataset.export(str(file_path))
    # Check if files are created for each label
    assert (tmp_path / "A01_1.csv").exists()
    assert (tmp_path / "A01_2.csv").exists()
    # Read back one file and check content
    read_df = pd.read_csv(tmp_path / "A01_1.csv")
    np.testing.assert_allclose(
        multi_dataset["1"].y, read_df.yc.to_numpy().astype(float)
    )
    np.testing.assert_allclose(
        multi_dataset["1"].x, read_df.xc.to_numpy().astype(float)
    )


def test_dataset_plot(multi_dataset: Dataset) -> None:
    """Test Dataset.plot() method returns a figure."""
    fig = multi_dataset.plot()
    assert fig is not None
    assert isinstance(fig, Figure)
    # Check axes labels
    ax = fig.axes[0]
    assert ax.get_xlabel() == "pH"  # multi_dataset.is_ph is True
    assert ax.get_ylabel() == "Signal"
    plt.close(fig)


def test_dataset_plot_ph() -> None:
    """Test Dataset.plot() with pH data."""
    x = np.array([5.0, 6.0, 7.0, 8.0])
    y = np.array([100.0, 150.0, 180.0, 195.0])
    ds = Dataset({"0": DataArray(x, y)}, is_ph=True)
    fig = ds.plot()
    ax = fig.axes[0]
    assert ax.get_xlabel() == "pH"
    plt.close(fig)


def test_dataset_plot_with_errors() -> None:
    """Test Dataset.plot() with error bars."""
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([10.0, 20.0, 30.0, 40.0])
    x_err = np.array([0.1, 0.1, 0.1, 0.1])
    y_err = np.array([1.0, 2.0, 3.0, 4.0])
    ds = Dataset({"test": DataArray(x, y, x_errc=x_err, y_errc=y_err)})
    fig = ds.plot(title="Test Plot")
    ax = fig.axes[0]
    assert ax.get_title() == "Test Plot"
    plt.close(fig)


def test_dataset_plot_with_mask() -> None:
    """Test Dataset.plot() shows masked points."""
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([10.0, 20.0, 30.0, 40.0])
    da = DataArray(x, y)
    da.mask = np.array([True, True, False, True])  # mask out point 3
    ds = Dataset({"test": da})
    fig = ds.plot()
    # Should have plotted masked point with 'x' marker
    assert fig is not None
    # ensure that a masked point exists on the axes as a Line2D with marker 'x'
    lines = [line for line in fig.axes[0].get_lines() if line.get_marker() == "x"]
    assert lines, "Expected masked points plotted with marker x"
    plt.close(fig)


def test_dataset_plot_custom_ax() -> None:
    """Test Dataset.plot() with custom axes."""
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([10.0, 20.0, 30.0])
    ds = Dataset({"test": DataArray(x, y)})
    fig, ax = plt.subplots()
    result_fig = ds.plot(ax=ax)
    assert result_fig is fig
    plt.close(fig)

    # Additional
    # For DataArray
    # FIXME: complete tests
    """
    def test_dataarray_with_zero_length() -> None:
        "Test initialization with empty arrays."
        with pytest.raises(ValueError):
            DataArray(np.array([]), np.array([]))


    def test_dataarray_inf_values() -> None:
        "Test handling of infinite values."
        x = np.array([1, 2, np.inf])
        y = np.array([1, 2, 3])
        da = DataArray(x, y)
        assert not np.isinf(da.x).any()


    def test_serialization_roundtrip(tmp_path) -> None:
        "Test save/load roundtrip."
        original = Dataset(
            {
                "test1": DataArray(np.array([1, 2, 3]), np.array([4, 5, 6])),
                "test2": DataArray(np.array([1, 2, 3]), np.array([7, 8, 9])),
            }
        )
        # Save
        path = tmp_path / "test.h5"
        original.save(path)
        # Load
        loaded = Dataset.load(path)
        # Verify
        assert set(original.keys()) == set(loaded.keys())
        for k in original:
            assert np.array_equal(original[k].x, loaded[k].x)
            assert np.array_equal(original[k].y, loaded[k].y)

    """


# For Dataset
def test_dataset_empty() -> None:
    """Test empty dataset behavior."""
    ds = Dataset({})
    assert len(ds) == 0
    with pytest.raises(KeyError):
        _ = ds["nonexistent"]


# For fitting functions
def test_fit_binding_edge_cases() -> None:
    """Test fitting with edge case inputs."""
    # Flat line
    x = np.array([1, 2, 3])
    y = np.array([1, 1, 1])
    res = fit_binding_glob(Dataset({"flat": DataArray(x, y)}))
    assert res.result is not None
    assert res.result.success is True

    # Single point
    with pytest.raises(InsufficientDataError):
        fit_binding_glob(Dataset({"single": DataArray(np.array([1]), np.array([1]))}))


@pytest.fixture
def mock_fit_result() -> FitResult:
    """Mock a successful fit result."""
    result = MinimizerResult()
    result.success = True
    result.params = Parameters()
    result.params.add("K", value=7.0)
    result.params.add("S0_mock", value=1.0)
    result.params.add("S1_mock", value=2.0)
    return FitResult(result=result)


def test_plot_fit_with_mock(mock_fit_result: FitResult) -> None:
    """Test plotting with mocked fit result."""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ds = Dataset({"mock": DataArray(np.array([1, 2, 3]), np.array([1, 2, 3]))})
    if mock_fit_result.result:
        plot_fit(ax, ds, mock_fit_result.result.params)
    assert len(ax.lines) > 0  # Verify something was plotted


def test_error_messages() -> None:
    """Verify error messages are helpful."""
    with pytest.raises(InvalidDataError) as excinfo:
        DataArray(np.array([1, 2]), np.array([1]))  # Length mismatch
    assert "equal" in str(excinfo.value)
    tiny = Dataset({"tiny": DataArray(np.array([1]), np.array([1]))})
    with pytest.raises(InsufficientDataError) as excinfo2:
        fit_binding_glob(tiny)
    assert "Not enough" in str(excinfo2.value)


###############################################################################
# fit_binding_glob(..., remove_outliers=...) tests
###############################################################################

# Test parameters for outlier-removal coverage
_TRUE_K = 7.0
_TRUE_S0_Y1, _TRUE_S1_Y1 = 600.0, 50.0
_TRUE_S0_Y2, _TRUE_S1_Y2 = 500.0, 40.0
_BUFFER_SD = 40.0


def _create_synthetic_dataset(  # noqa: PLR0913
    n_points: int = 7,
    true_k: float = _TRUE_K,
    seed: int = 42,
    *,
    add_outlier: bool = False,
    outlier_label: str = "1",
    outlier_idx: int = 2,
    outlier_magnitude: float = 5.0,
) -> Dataset:
    """Create synthetic dual-channel pH titration dataset for outlier tests."""
    rng = np.random.default_rng(seed)

    x = np.linspace(5.5, 9.0, n_points)
    x_err = 0.05 * np.ones_like(x)

    ytrue_1 = binding_1site(x, true_k, _TRUE_S0_Y1, _TRUE_S1_Y1, is_ph=True)
    y2_true = binding_1site(x, true_k, _TRUE_S0_Y2, _TRUE_S1_Y2, is_ph=True)

    y1_err = np.sqrt(np.maximum(ytrue_1, 1.0) + _BUFFER_SD**2)
    y2_err = np.sqrt(np.maximum(y2_true, 1.0) + _BUFFER_SD**2)

    y1 = ytrue_1 + rng.normal(0, y1_err)
    y2 = y2_true + rng.normal(0, y2_err)

    if add_outlier:
        if outlier_label == "1":
            y1[outlier_idx] += outlier_magnitude * y1_err[outlier_idx]
        else:
            y2[outlier_idx] += outlier_magnitude * y2_err[outlier_idx]

    da1 = DataArray(x, y1, x_errc=x_err, y_errc=y1_err)
    da2 = DataArray(x, y2, x_errc=x_err, y_errc=y2_err)

    return Dataset({"1": da1, "2": da2}, is_ph=True)


def _drawable_excluded(da: DataArray) -> np.ndarray:
    """X of points a plot should mark: excluded from the fit but with finite y."""
    return da.xc[~da.mask & np.isfinite(da.yc)]


def _fit_binding_glob_huber_outlier(
    ds: Dataset, *, threshold: float = 2.5
) -> FitResult:
    """Run the supported robust fit with z-score outlier removal."""
    return fit_binding_glob(
        ds,
        method="huber",
        remove_outliers=f"mad:{threshold}:5",
    )


class TestFitBindingGlobOutlierRemoval:
    """Tests for the supported outlier-removal fit configuration."""

    def test_returns_fit_result(self) -> None:
        """The robust outlier-removal fit should return a valid FitResult."""
        ds = _create_synthetic_dataset()
        fr = _fit_binding_glob_huber_outlier(ds)

        assert fr.result is not None
        assert "K" in fr.result.params

    def test_k_estimate_reasonable(self) -> None:
        """K estimate should be close to true value."""
        ds = _create_synthetic_dataset(seed=123)
        fr = _fit_binding_glob_huber_outlier(ds)

        assert fr.result is not None
        k_est = fr.result.params["K"].value
        assert abs(k_est - _TRUE_K) < 0.5

    def test_detects_outlier_in_1(self) -> None:
        """Should detect large outlier in y1."""
        ds = _create_synthetic_dataset(
            add_outlier=True, outlier_label="1", outlier_idx=3, outlier_magnitude=10.0
        )
        fr = _fit_binding_glob_huber_outlier(ds, threshold=2.0)

        assert fr.dataset is not None
        assert len(fr.dataset["1"].y) < 7

    def test_detects_outlier_in_y2(self) -> None:
        """Should detect large outlier in y2."""
        ds = _create_synthetic_dataset(
            add_outlier=True, outlier_label="2", outlier_idx=4, outlier_magnitude=10.0
        )
        fr = _fit_binding_glob_huber_outlier(ds, threshold=2.0)

        assert fr.dataset is not None
        assert len(fr.dataset["2"].y) < 7

    def test_excluded_points_retained_for_plotting(self) -> None:
        """Points dropped from the fit stay in the arrays for plotting."""
        ds = _create_synthetic_dataset(
            add_outlier=True, outlier_label="1", outlier_idx=3, outlier_magnitude=10.0
        )
        fr = _fit_binding_glob_huber_outlier(ds, threshold=2.0)

        assert fr.dataset is not None
        da = fr.dataset["1"]
        # The excluded point is absent from the fit but kept in the arrays.
        x_exc = _drawable_excluded(da)
        assert x_exc.size == 7 - da.x.size > 0
        assert not np.intersect1d(da.x, x_exc).size
        # Original arrays are untouched; only the mask moved.
        assert da.xc.size == da.yc.size == 7

    def test_excluded_points_marked_in_figure(self) -> None:
        """The figure shows excluded points under a dedicated legend entry."""
        ds = _create_synthetic_dataset(
            add_outlier=True, outlier_label="1", outlier_idx=3, outlier_magnitude=10.0
        )
        fr = _fit_binding_glob_huber_outlier(ds, threshold=2.0)

        assert fr.figure is not None
        legend = fr.figure.axes[0].get_legend()
        assert legend is not None
        assert "excluded (not fitted)" in [t.get_text() for t in legend.get_texts()]

    def test_no_excluded_legend_on_clean_data(self) -> None:
        """Clean data produces no excluded marker or legend entry.

        Uses the studentized screen, the one that actually controls the
        false-positive rate; see `test_mad_over_flags_clean_data`.
        """
        ds = _create_synthetic_dataset(add_outlier=False, seed=42)
        fr = fit_binding_glob(ds, method="huber", remove_outliers="studentized:0.05:5")

        assert fr.dataset is not None
        assert _drawable_excluded(fr.dataset["1"]).size == 0
        assert fr.figure is not None
        legend = fr.figure.axes[0].get_legend()
        assert legend is not None
        assert "excluded (not fitted)" not in [t.get_text() for t in legend.get_texts()]

    def test_manually_masked_steps_are_shown(self) -> None:
        """Points excluded via `mask_steps` are drawn too, not just outliers."""
        ds = _create_synthetic_dataset(add_outlier=False, seed=42)
        ds["1"].mask_steps([0, 6])
        fr = fit_binding_glob(ds, method="huber")

        assert fr.dataset is not None
        assert _drawable_excluded(fr.dataset["1"]).size == 2
        assert fr.figure is not None
        legend = fr.figure.axes[0].get_legend()
        assert legend is not None
        assert "excluded (not fitted)" in [t.get_text() for t in legend.get_texts()]

    def test_nan_points_are_not_drawn(self) -> None:
        """NaN points are masked but have nothing to draw, so they stay hidden."""
        ds = _create_synthetic_dataset(add_outlier=False, seed=42)
        ds["1"].yc[2] = np.nan
        da = DataArray(ds["1"].xc, ds["1"].yc)

        assert not da.mask[2]
        assert _drawable_excluded(da).size == 0

    def test_no_false_positives_clean_data(self) -> None:
        """The studentized screen removes nothing from clean data."""
        ds = _create_synthetic_dataset(add_outlier=False, seed=42)
        fr = fit_binding_glob(ds, method="huber", remove_outliers="studentized:0.05:5")

        assert fr.dataset is not None
        assert len(fr.dataset["1"].y) == 7
        assert len(fr.dataset["2"].y) == 7

    def test_mad_over_flags_clean_data(self) -> None:
        """Known limitation: the MAD screen has a real false-positive rate.

        With only 7 points per label the MAD is a noisy scale estimate, and
        residuals are further deflated by leverage, so ordinary points score
        above the cutoff. Measured on the L2 + L4 plates this reaches ~10% of
        all points, against ~2.5% for the studentized screen.
        """
        ds = _create_synthetic_dataset(add_outlier=False, seed=42)
        fr = fit_binding_glob(ds, method="huber", remove_outliers="mad:3.5:5")

        assert fr.dataset is not None
        assert len(fr.dataset["2"].y) < 7  # points dropped though none are bad

    def test_correct_residual_slicing(self) -> None:
        """Residuals should be correctly sliced for each label."""
        ds = _create_synthetic_dataset()
        fr_init = fit_binding_glob(ds, method="huber")
        assert fr_init.result is not None

        total_residuals = len(fr_init.result.residual)
        assert total_residuals == len(ds["1"].y) + len(ds["2"].y)

    def test_single_label_dataset(self) -> None:
        """Should work with single-label dataset."""
        rng = np.random.default_rng(42)
        x = np.linspace(5.5, 9.0, 7)
        y_true = binding_1site(x, _TRUE_K, _TRUE_S0_Y1, _TRUE_S1_Y1, is_ph=True)
        y_err = np.sqrt(np.maximum(y_true, 1.0) + _BUFFER_SD**2)
        y = y_true + rng.normal(0, y_err)

        da = DataArray(x, y, x_errc=0.05 * np.ones_like(x), y_errc=y_err)
        ds = Dataset({"1": da}, is_ph=True)

        fr = _fit_binding_glob_huber_outlier(ds)
        assert fr.result is not None

    def test_deterministic_output(self) -> None:
        """Same input should give same output."""
        ds1 = _create_synthetic_dataset(seed=42)
        ds2 = _create_synthetic_dataset(seed=42)

        fr1 = _fit_binding_glob_huber_outlier(ds1)
        fr2 = _fit_binding_glob_huber_outlier(ds2)

        assert fr1.result is not None
        assert fr2.result is not None
        np.testing.assert_allclose(
            fr1.result.params["K"].value,
            fr2.result.params["K"].value,
            rtol=1e-10,
        )


###############################################################################
# Tests for ODR Fitter
###############################################################################


def test_fit_binding_odr_ph(ph_dataset: Dataset) -> None:
    """Test ODR fitting on pH dataset."""
    fr = fit_binding_odr(ph_dataset)
    assert fr.result is not None
    assert "K" in fr.result.params
    assert np.isclose(fr.result.params["K"].value, 7.0, atol=0.5)
    # Check that residuals are computed
    assert fr.result.residual is not None
    assert len(fr.result.residual) > 0


def test_fit_binding_odr_from_fitresult(ph_dataset: Dataset) -> None:
    """Test ODR fitting starting from a FitResult."""
    # First get initial fit
    fr_init = fit_binding_glob(ph_dataset)
    assert fr_init.result is not None

    # Then use ODR
    fr_odr = fit_binding_odr(fr_init)
    assert fr_odr.result is not None
    assert "K" in fr_odr.result.params
    assert np.isclose(fr_odr.result.params["K"].value, 7.0, atol=0.5)
    assert fr_odr.result.residual is not None


def test_fit_binding_odr_reweight_ph(ph_dataset: Dataset) -> None:
    """Test iterative ODR fitting."""
    fr = fit_binding_odr(ph_dataset, reweight=True, max_iter=3)
    assert fr.result is not None
    assert "K" in fr.result.params
    assert np.isclose(fr.result.params["K"].value, 7.0, atol=0.5)
    assert fr.result.residual is not None


def test_fit_binding_odr_outlier_ph(ph_dataset: Dataset) -> None:
    """Test ODR fitting with outlier removal."""
    fr = fit_binding_odr(ph_dataset, remove_outliers="mad:3.5")
    assert fr.result is not None
    assert "K" in fr.result.params


def test_fit_binding_odr_multi(multi_dataset: Dataset) -> None:
    """Test ODR fitting on multi-label dataset."""
    fr = fit_binding_odr(multi_dataset)
    assert fr.result is not None
    assert "K" in fr.result.params
    assert "S0_1" in fr.result.params
    assert "S1_1" in fr.result.params
    assert "S0_2" in fr.result.params
    assert "S1_2" in fr.result.params
    assert np.isclose(fr.result.params["K"].value, 7.0, atol=0.5)
    assert fr.result.residual is not None
    # Total residuals should match concatenated y data
    total_y_points = len(multi_dataset["1"].y) + len(multi_dataset["2"].y)
    assert len(fr.result.residual) == total_y_points


###############################################################################
# Tests for Bayesian Fitter
###############################################################################


def test_fit_binding_pymc_ph(ph_dataset: Dataset) -> None:
    """Test PyMC Bayesian fitting on pH dataset."""
    # n_tune must be set explicitly: with n_samples=100 the default (n_samples//2
    # = 50 warmup draws) under-tunes NUTS, biasing K high (~8, at the assertion
    # boundary) and making the test platform-flaky. A fixed seed keeps it
    # reproducible.
    fr = fit_binding_pymc(
        ph_dataset,
        n_sd=1.0,
        sampler=SamplerConfig(n_samples=200, n_tune=500, random_seed=42),
    )
    assert fr.result is not None
    assert "K" in fr.result.params
    # Check K value is reasonable
    assert 6.0 < fr.result.params["K"].value < 8.0
    # Check residuals are computed
    assert fr.result.residual is not None
    assert len(fr.result.residual) > 0


def test_fit_binding_pymc_avoids_log_likelihood_deprecation(
    ph_dataset: Dataset,
) -> None:
    """PyMC fitting should not emit the log_likelihood idata_kwargs deprecation."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fr = fit_binding_pymc(ph_dataset, n_sd=1.0, sampler=SamplerConfig(n_samples=20))

    assert fr.result is not None
    loglike_future_warnings = [
        warning
        for warning in caught
        if issubclass(warning.category, FutureWarning)
        and "log_likelihood" in str(warning.message)
    ]
    assert loglike_future_warnings == []


def test_fit_binding_pymc_from_fitresult(ph_dataset: Dataset) -> None:
    """Test PyMC starting from a FitResult."""
    fr_init = fit_binding_glob(ph_dataset)
    assert fr_init.result is not None

    fr_pymc = fit_binding_pymc(fr_init, n_sd=1.0, sampler=SamplerConfig(n_samples=50))
    assert fr_pymc.result is not None
    assert "K" in fr_pymc.result.params


def test_fit_binding_pymc_separate_ph(ph_dataset: Dataset) -> None:
    """Test PyMC with per-label noise floors."""
    noise_model = PlateNoiseModel({
        lbl: NoiseModelParams(sigma_floor=10.0) for lbl in ph_dataset
    })
    fr = fit_binding_pymc(
        ph_dataset,
        n_sd=1.0,
        noise=NoiseConfig.structured(noise_model=noise_model),
        sampler=SamplerConfig(n_samples=100),
    )
    assert fr.result is not None
    assert "K" in fr.result.params
    assert fr.result.residual is not None


def test_fit_binding_pymc_multi(multi_dataset: Dataset) -> None:
    """Test PyMC on multi-label dataset."""
    fr = fit_binding_pymc(multi_dataset, n_sd=1.0, sampler=SamplerConfig(n_samples=50))
    assert fr.result is not None
    assert "K" in fr.result.params
    assert "S0_1" in fr.result.params
    assert "S1_1" in fr.result.params
    assert fr.result.residual is not None


@pytest.mark.slow
def test_fit_binding_pymc_multi_noise(multi_dataset: Dataset) -> None:
    """Smoke test for multi-well noise-learning PyMC fit."""
    # Build two minimal wells sharing K via a control group
    fr_init = fit_binding_glob(multi_dataset)
    assert fr_init.result is not None

    results = {"A01": fr_init, "A02": fr_init}
    scheme = PlateScheme()
    scheme.names = {"ctrl": {"A01", "A02"}}

    np.random.default_rng(42)
    # Comprehensive noise: per-label floor + per-label gain + shared alpha
    noise_model = PlateNoiseModel({
        "1": NoiseModelParams(sigma_floor=250.0, gain=1.0, alpha=0.04),
        "2": NoiseModelParams(sigma_floor=5.0, gain=1.0, alpha=0.04),
    })

    multi = fit_binding_pymc_multi(
        results,
        scheme,
        n_sd=3.0,
        n_xerr=0.0,
        noise=NoiseConfig.structured(noise_model=noise_model, shared_alpha=True),
        sampler=SamplerConfig(n_samples=50, n_tune=25),
    )

    assert hasattr(multi.trace, "posterior")
    assert "rel_error" in multi.trace.posterior
    assert "gain_1" in multi.trace.posterior
    assert set(multi.results) == {"A01", "A02"}
    assert multi.results["A01"].result is not None


@pytest.mark.slow
def test_fit_binding_pymc_multi_noise_per_well(multi_dataset: Dataset) -> None:
    """Smoke test for multi-well noise+per-well pH PyMC fit."""
    fr_init = fit_binding_glob(multi_dataset)
    assert fr_init.result is not None

    results = {"A01": fr_init, "A02": fr_init}
    scheme = PlateScheme()
    scheme.names = {"ctrl": {"A01", "A02"}}

    np.random.default_rng(42)
    # Comprehensive noise with x_error_model="per_well"
    noise_model = PlateNoiseModel({
        "1": NoiseModelParams(sigma_floor=250.0, gain=1.0, alpha=0.04),
        "2": NoiseModelParams(sigma_floor=5.0, gain=1.0, alpha=0.04),
    })

    multi = fit_binding_pymc_multi(
        results,
        scheme,
        n_sd=3.0,
        n_xerr=1.0,
        x_error_model="per_well",
        noise=NoiseConfig.structured(noise_model=noise_model, shared_alpha=True),
        sampler=SamplerConfig(n_samples=50, n_tune=25),
    )

    assert hasattr(multi.trace, "posterior")
    assert "x_step" in multi.trace.posterior or "x_true" in multi.trace.posterior
    assert "rel_error" in multi.trace.posterior
    if "x_true" in multi.trace.posterior:
        # per-well x_true has dims (step, well)
        assert "step" in multi.trace.posterior["x_true"].dims
        assert "well" in multi.trace.posterior["x_true"].dims
    assert set(multi.results) == {"A01", "A02"}


def test_extract_sigma_df_from_datatree_posterior() -> None:
    """Sigma extraction should work directly from posterior variables."""
    posterior = xr.Dataset(
        data_vars={
            "sigma_obs_1_A01": (
                ("chain", "draw", "sigma_obs_1_A01_dim_0"),
                np.array([
                    [
                        [1.0, 2.0, 3.0],
                        [2.0, 3.0, 4.0],
                    ]
                ]),
            ),
            "sigma_obs_2_A01": (
                ("chain", "draw", "sigma_obs_2_A01_dim_0"),
                np.array([
                    [
                        [4.0, 5.0, 6.0],
                        [6.0, 7.0, 8.0],
                    ]
                ]),
            ),
        },
        coords={
            "chain": [0],
            "draw": [0, 1],
            "sigma_obs_1_A01_dim_0": [0, 1, 2],
            "sigma_obs_2_A01_dim_0": [0, 1, 2],
        },
    )
    trace = xr.DataTree.from_dict({"posterior": posterior})

    sigma_df = extract_sigma_df(trace)

    assert len(sigma_df) == 6
    assert set(sigma_df["label"]) == {"1", "2"}
    assert set(sigma_df["well"]) == {"A01"}
    np.testing.assert_array_equal(
        sigma_df[sigma_df["label"] == "1"].sort_values("idx")["idx"].to_numpy(),
        np.array([0, 1, 2]),
    )
    np.testing.assert_allclose(
        sigma_df[sigma_df["label"] == "1"].sort_values("idx")["mean"].to_numpy(),
        np.array([1.5, 2.5, 3.5]),
    )
    np.testing.assert_allclose(
        sigma_df[sigma_df["label"] == "2"].sort_values("idx")["mean"].to_numpy(),
        np.array([5.0, 6.0, 7.0]),
    )


def test_extract_sigma_df_falls_back_to_ye_mag_without_az_summary() -> None:
    """Fallback sigma extraction should use posterior ye_mag directly."""
    posterior = xr.Dataset(
        data_vars={
            "ye_mag_1": (("chain", "draw"), np.array([[2.0, 4.0]])),
        },
        coords={"chain": [0], "draw": [0, 1]},
    )
    trace = xr.DataTree.from_dict({"posterior": posterior})
    ds = Dataset(
        {
            "1": DataArray(
                np.array([6.0, 7.0]),
                np.array([1.0, 2.0]),
                y_errc=np.array([10.0, 20.0]),
            )
        },
        is_ph=True,
    )
    fr: FitResult = FitResult(dataset=ds)

    sigma_df = extract_sigma_df(trace, {"A01": fr})

    np.testing.assert_allclose(sigma_df["mean"].to_numpy(), np.array([30.0, 60.0]))
    np.testing.assert_allclose(sigma_df["sd"].to_numpy(), np.array([10.0, 20.0]))
    np.testing.assert_allclose(sigma_df["hdi_3%"].to_numpy(), np.array([20.6, 41.2]))
    np.testing.assert_allclose(sigma_df["hdi_97%"].to_numpy(), np.array([39.4, 78.8]))


def _make_direct_sigma_trace_and_results() -> tuple[xr.DataTree, dict[str, FitResult]]:
    posterior = xr.Dataset(
        data_vars={
            "sigma_obs_1_A01": (
                ("chain", "draw", "sigma_obs_1_A01_dim_0"),
                np.array([[[1.0, 1.1, 1.2], [1.3, 1.4, 1.5]]]),
            ),
            "sigma_obs_1_A02": (
                ("chain", "draw", "sigma_obs_1_A02_dim_0"),
                np.array([[[0.9, 1.0, 1.1], [1.2, 1.3, 1.4]]]),
            ),
        },
        coords={
            "chain": [0],
            "draw": [0, 1],
            "sigma_obs_1_A01_dim_0": [0, 1, 2],
            "sigma_obs_1_A02_dim_0": [0, 1, 2],
        },
    )
    trace = xr.DataTree.from_dict({"posterior": posterior})
    ds_a01 = Dataset(
        {"1": DataArray(np.array([6.0, 7.0, 8.0]), np.array([10.0, 11.0, 12.0]))},
        is_ph=True,
    )
    ds_a02 = Dataset(
        {"1": DataArray(np.array([6.0, 7.0, 8.0]), np.array([9.0, 10.0, 11.0]))},
        is_ph=True,
    )
    results: dict[str, FitResult] = {
        "A01": FitResult(dataset=ds_a01),
        "A02": FitResult(dataset=ds_a02),
    }
    return trace, results


def test_noise_and_qc_plots_accept_direct_posterior_sigma() -> None:
    """QC/noise plots should accept sigma extracted directly from posterior vars."""
    trace, results = _make_direct_sigma_trace_and_results()

    fig_noise = plot_noise_vs_signal(trace, results)
    fig_qc = plot_qc_mean_vs_std(trace, results)

    assert isinstance(fig_noise, Figure)
    assert isinstance(fig_qc, Figure)


def test_plot_qc_mean_vs_std_preserves_annotations_and_legend_semantics() -> None:
    """Mean-vs-std QC plot should keep legend and annotation behavior stable."""
    trace, results = _make_direct_sigma_trace_and_results()

    fig = plot_qc_mean_vs_std(
        trace,
        results,
        bg_noise={"1": 0.29},
        annotate_wells=["A01"],
    )

    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert ax.get_title() == r"QC: Span vs Mean of inferred $\sigma$ (1)"
    assert ax.get_xlabel() == r"Mean($\sigma_{obs}$)"
    assert ax.get_ylabel() == r"Span($\sigma_{obs}$) [max - min]"
    legend = ax.get_legend()
    assert legend is not None
    legend_labels = [text.get_text() for text in legend.get_texts()]
    assert "Trendline" in legend_labels
    assert "Wells" in legend_labels
    assert "BG < 4.0x" in legend_labels
    annotations = {text.get_text() for text in ax.texts}
    assert {"A01", "A02"}.issubset(annotations)


class _FakeTitration:
    def __init__(self) -> None:
        self.bg_noise = {"1": np.array([0.2, 0.25, 0.3])}

    def _get_normalized_or_raw_data(self) -> dict[str, dict[str, np.ndarray]]:
        return {
            "1": {
                "A01": np.array([0.05, 0.05, 0.06]),
                "A02": np.array([1.0, 1.4, 1.8]),
                "A03": np.array([1.1, 1.2, 1.25]),
            }
        }


def test_plot_qc_span_vs_center_titration_matches_mean_vs_std_style() -> None:
    """Titration QC wrapper should mirror mean-vs-std highlight/legend behavior."""
    fig = plot_qc_span_vs_center_titration(
        _FakeTitration(),
        annotate_wells=["A02"],
        loglog=True,
    )

    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert ax.get_xscale() == "log"
    assert ax.get_yscale() == "log"
    assert ax.get_title() == "QC: Span vs Q90 (1)"
    legend = ax.get_legend()
    assert legend is not None
    legend_labels = [text.get_text() for text in legend.get_texts()]
    assert "Trendline" in legend_labels
    assert "Wells" in legend_labels
    assert "BG < 4.0x" in legend_labels
    assert "4.0x bg" not in legend_labels
    annotations = {text.get_text() for text in ax.texts}
    assert {"A01", "A02"}.issubset(annotations)


def test_qc_flag_bad_wells_robust_to_spike() -> None:
    """Robust q90/inter-quantile QC flags a flat well that a lone spike hides."""
    x = np.linspace(5, 8, 8)
    data: dict[str, dict[str, np.ndarray]] = {"1": {}}
    for w in range(8):  # healthy sigmoidal wells
        rng = np.random.default_rng(w)
        data["1"][f"A{w:02d}"] = 200 + 800 / (1 + 10 ** (7 - x)) + rng.normal(0, 5, 8)
    data["1"]["H01"] = np.full(8, 50.0)  # dead/flat well
    spiky = np.full(8, 60.0)
    spiky[3] = 2000.0  # flat but one huge spike -> max() would look bright + dynamic
    data["1"]["H02"] = spiky

    flagged = qc_flag_bad_wells(data, bg_noise={"1": 40.0})
    assert "H01" in flagged["1"]
    assert "H02" in flagged["1"]  # robust measures see through the spike


def test_qc_flag_bad_wells_titration_excludes_buffer() -> None:
    """The titration wrapper drops buffer wells from the QC set."""

    class _Scheme:
        def __init__(self) -> None:
            self.buffer = ["A01"]

    class _Tit:
        def __init__(self) -> None:
            self.scheme = _Scheme()
            self.bg_noise = {"1": 40.0}

        def _get_normalized_or_raw_data(self) -> dict[str, dict[str, np.ndarray]]:
            x = np.linspace(5, 8, 8)
            out = {f"B{w:02d}": 200 + 800 / (1 + 10 ** (7 - x)) for w in range(6)}
            out["A01"] = np.full(8, 50.0)  # buffer well: flat, would flag if kept
            return {"1": out}

    flagged = qc_flag_bad_wells_titration(_Tit())
    assert "A01" not in flagged["1"]  # excluded as buffer, not reported


def test_qc_flag_bad_wells_combine_intersection_vs_union() -> None:
    """``combine`` aggregates per-label sets by intersection or union."""
    x = np.linspace(5, 8, 8)

    def _good(seed: int) -> np.ndarray:
        return (
            200
            + 800 / (1 + 10 ** (7 - x))
            + np.random.default_rng(seed).normal(0, 3, 8)
        )

    dead = np.full(8, 50.0)
    base = {f"A{w:02d}": _good(w) for w in range(6)}
    # H01 dead only in label 1; H02 dead only in label 2.
    data = {
        "1": {**base, "H01": dead.copy(), "H02": _good(101)},
        "2": {**base, "H01": _good(102), "H02": dead.copy()},
    }
    bg = {"1": 40.0, "2": 40.0}
    per_label = qc_flag_bad_wells(data, bg_noise=bg)
    sets = [set(v) for v in per_label.values()]
    union = qc_flag_bad_wells(data, bg_noise=bg, combine="union")
    inter = qc_flag_bad_wells(data, bg_noise=bg, combine="intersection")

    assert set(union) == set.union(*sets)
    assert set(inter) == set.intersection(*sets)
    assert union == sorted(union)
    assert inter == sorted(inter)
    assert set(inter) < set(union)  # single-label-dead wells drop from intersection


def test_qc_flag_bad_wells_excludes_control_wells() -> None:
    """Control wells (different span-vs-signal stats) are never flagged."""
    x = np.linspace(5, 8, 12)
    data: dict[str, dict[str, np.ndarray]] = {"1": {}}
    for w in range(10):  # samples: deep sigmoid, large span
        rng = np.random.default_rng(w)
        data["1"][f"A{w:02d}"] = 200 + 800 / (1 + 10 ** (7 - x)) + rng.normal(0, 5, 12)
    # a control: bright but shallow -> sits below the sample span/signal trend
    data["1"]["C01"] = 900 - 300 / (1 + 10 ** (7 - x))

    flagged_no_ctrl = qc_flag_bad_wells(data, bg_noise={"1": 40.0}, z_threshold=3.0)
    flagged_ctrl = qc_flag_bad_wells(
        data, bg_noise={"1": 40.0}, z_threshold=3.0, ctrl_wells=["C01"]
    )
    assert "C01" in flagged_no_ctrl["1"]  # flagged as a trend outlier when unmarked
    assert "C01" not in flagged_ctrl["1"]  # never flagged once marked as control


def test_qc_flag_bad_wells_control_below_background_still_flagged() -> None:
    """A control below the background floor is dead and stays flagged."""
    x = np.linspace(5, 8, 12)
    data: dict[str, dict[str, np.ndarray]] = {"1": {}}
    for w in range(10):
        rng = np.random.default_rng(w)
        data["1"][f"A{w:02d}"] = 200 + 800 / (1 + 10 ** (7 - x)) + rng.normal(0, 5, 12)
    data["1"]["C01"] = np.full(12, 30.0)  # control, but genuinely dead (below bg floor)

    flagged = qc_flag_bad_wells(
        data, bg_noise={"1": 40.0}, bg_multiplier=4.0, ctrl_wells=["C01"]
    )
    assert "C01" in flagged["1"]  # bg floor still applies to controls


def test_qc_plot_annotates_deviating_control_without_discarding() -> None:
    """A control off the sample trend is name-annotated but not discarded."""
    x = np.linspace(5, 8, 12)
    data: dict[str, dict[str, np.ndarray]] = {"1": {}}
    for w in range(10):
        rng = np.random.default_rng(w)
        data["1"][f"A{w:02d}"] = 200 + 800 / (1 + 10 ** (7 - x)) + rng.normal(0, 5, 12)
    data["1"]["C01"] = 900 - 300 / (1 + 10 ** (7 - x))  # bright, shallow: off-trend

    fig = plot_qc_span_vs_center(
        data,
        center=0.9,
        span_q=(0.1, 0.9),
        bg_noise={"1": 40.0},
        z_threshold=3.0,
        ctrl_wells=["C01"],
    )
    annotations = {t.get_text() for t in fig.axes[0].texts}
    flagged = qc_flag_bad_wells(
        data, bg_noise={"1": 40.0}, z_threshold=3.0, ctrl_wells=["C01"]
    )
    assert "C01" in annotations  # shown by name on the plot
    assert "C01" not in flagged["1"]  # but not in the discard set


###############################################################################
# Tests for Result Consistency
###############################################################################


def test_all_fitters_return_residuals(ph_dataset: Dataset) -> None:
    """All fitter functions should return residuals in result."""
    fr_lm = fit_binding_glob(ph_dataset)
    assert fr_lm.result is not None
    assert fr_lm.result.residual is not None
    assert len(fr_lm.result.residual) > 0

    fr_odr = fit_binding_odr(ph_dataset)
    assert fr_odr.result is not None
    assert fr_odr.result.residual is not None
    assert len(fr_odr.result.residual) > 0


def test_residual_computation_consistency(ph_dataset: Dataset) -> None:
    """Check that residuals are computed consistently across backends."""
    fr_lm = fit_binding_glob(ph_dataset)
    fr_odr = fit_binding_odr(ph_dataset)

    assert fr_lm.result is not None
    assert fr_odr.result is not None

    # Both should have same number of residuals
    assert len(fr_lm.result.residual) == len(fr_odr.result.residual)


def test_fit_result_attributes(ph_dataset: Dataset) -> None:
    """Test that FitResult has all expected attributes."""
    fr = fit_binding_glob(ph_dataset)

    # Should have all attributes
    assert fr.figure is not None
    assert fr.result is not None
    assert fr.mini is not None
    assert fr.dataset is not None

    # Result should have key attributes
    assert hasattr(fr.result, "params")
    assert hasattr(fr.result, "residual")

    # Parameters should be accessible
    assert "K" in fr.result.params
    assert fr.result.params["K"].value is not None


###############################################################################
# Robust (MAD) outlier identification
###############################################################################


class TestRobustOutlierIdentification:
    """Tests for the median/MAD robust z-score and its wiring."""

    def test_robust_score_beats_the_zscore_ceiling(self) -> None:
        """A mean/std z-score saturates at sqrt(n - 1); the robust one does not."""
        r = np.array([0.1, -0.2, 0.15, -0.05, 0.3, -0.1, 100.0])
        plain = np.abs((r - r.mean()) / r.std())
        assert plain.max() < np.sqrt(6)  # pinned under the ceiling
        assert robust_z_scores(r).max() > 100  # no ceiling
        assert identify_outliers_mad(r, threshold=3.5)[6]

    def test_robust_z_has_no_ceiling(self) -> None:
        """The robust score grows with the outlier instead of saturating."""
        r = np.array([0.0, 1.0, -1.0, 0.5, -0.5, 0.2, 100.0])
        assert robust_z_scores(r).max() > 50

    def test_robust_z_survives_zero_mad(self) -> None:
        """A collapsed MAD falls back to another scale instead of dividing by 0."""
        r = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 9.0])  # MAD == 0
        z = robust_z_scores(r)
        assert np.all(np.isfinite(z))
        assert z.argmax() == 5

    def test_robust_z_all_identical_flags_nothing(self) -> None:
        """No positive scale of any kind means no outliers, not a crash."""
        r = np.full(6, 2.5)
        assert np.all(robust_z_scores(r) == 0.0)
        assert not identify_outliers_mad(r).any()

    def test_robust_scale_is_normal_consistent(self) -> None:
        """The scale estimates sigma itself, not the bare (0.67x smaller) MAD."""
        rng = np.random.default_rng(0)
        assert abs(robust_scale(rng.normal(0.0, 5.0, 4000)) - 5.0) < 0.3

    def test_robust_z_accepts_a_pooled_scale(self) -> None:
        """A supplied sigma overrides the per-sample estimate."""
        r = np.array([0.0, 1.0, 2.0, 30.0])
        assert robust_z_scores(r, sigma=10.0)[3] < robust_z_scores(r)[3]

    def test_cap_by_min_keep_retains_worst(self) -> None:
        """Capping keeps the highest-scoring flags and honours min_keep."""
        flagged = np.array([True, True, True, False, False])
        scores = np.array([9.0, 3.0, 5.0, 0.1, 0.2])
        capped = cap_by_min_keep(flagged, scores, min_keep=4)
        assert capped.tolist() == [True, False, False, False, False]

    def test_cap_by_min_keep_noop_when_within_budget(self) -> None:
        """Flags below the budget pass through untouched."""
        flagged = np.array([True, False, False, False, False])
        scores = np.array([9.0, 0.1, 0.2, 0.3, 0.4])
        assert cap_by_min_keep(flagged, scores, min_keep=3).tolist() == flagged.tolist()

    def test_default_thresholds(self) -> None:
        """`mad` defaults to 3.5; `studentized` defaults to alpha = 0.05."""
        assert parse_remove_outliers("mad") == ("mad", 3.5, 1)
        assert parse_remove_outliers("studentized") == ("studentized", 0.05, 1)
        assert parse_remove_outliers("mad:4.5:5") == ("mad", 4.5, 5)

    def test_unknown_method_raises(self) -> None:
        """An unsupported method is an error, not a silent no-op."""
        ds = _create_synthetic_dataset()
        with pytest.raises(ValueError, match="Unknown outlier method"):
            fit_binding_glob(ds, remove_outliers="mixture:0.9")

    def test_mad_removes_planted_outlier(self) -> None:
        """The wired `mad` method drops a large planted outlier."""
        ds = _create_synthetic_dataset(
            add_outlier=True, outlier_label="1", outlier_idx=1, outlier_magnitude=10.0
        )
        fr = fit_binding_glob(ds, method="huber", remove_outliers="mad:3.0")
        assert fr.dataset is not None
        assert not fr.dataset["1"].mask[1]

    def test_robust_score_scales_where_zscore_saturates(self) -> None:
        """The robust score tracks outlier size; the z-score pins to its ceiling."""
        robust, plain = [], []
        for mag in (5.0, 10.0, 20.0):
            ds = _create_synthetic_dataset(
                add_outlier=True,
                outlier_label="1",
                outlier_idx=1,
                outlier_magnitude=mag,
            )
            fr = fit_binding_glob(ds, method="huber")
            assert fr.result is not None
            assert fr.dataset is not None
            da = fr.dataset["1"]
            pars = fr.result.params
            raw = da.y - binding_1site(
                da.x,
                pars["K"].value,
                pars["S0_1"].value,
                pars["S1_1"].value,
                is_ph=True,
            )
            n1 = len(da.y)
            w = fr.result.residual[:n1]
            robust.append(robust_z_scores(raw)[1])
            plain.append(float(np.abs((w - w.mean()) / w.std())[1]))
        # Robust score roughly doubles with the outlier; the z-score cannot.
        assert robust[2] > 2 * robust[1]
        assert robust[1] > 2 * robust[0]
        assert plain[2] < np.sqrt(7 - 1)  # pinned to the ceiling
        assert plain[2] - plain[1] < 0.2

    def test_mad_misses_high_leverage_midpoint_outlier(self) -> None:
        """Known limitation: a midpoint outlier corrupts the fit that exposes it.

        The midpoint carries most of the information about K, so an outlier
        there drags the whole curve; the remaining residuals grow, the MAD with
        them, and the outlier's own score stays under threshold. Neither this
        screen nor the plain z-score accounts for leverage.
        """
        ds = _create_synthetic_dataset(
            add_outlier=True, outlier_label="1", outlier_idx=3, outlier_magnitude=10.0
        )
        fr = fit_binding_glob(ds, method="huber", remove_outliers="mad:3.0")
        assert fr.dataset is not None
        assert fr.dataset["1"].mask[3]  # not removed

    def test_min_keep_is_enforced(self) -> None:
        """min_keep bounds how much of a label can be removed."""
        ds = _create_synthetic_dataset(n_points=7)
        # A threshold of 0 flags essentially everything.
        fr = fit_binding_glob(ds, method="huber", remove_outliers="mad:0.0:6")
        assert fr.dataset is not None
        for da in fr.dataset.values():
            assert da.x.size >= 6


class TestStudentizedOutlierIdentification:
    """Tests for leverage-aware studentized residuals with Bonferroni cutoff."""

    def test_leverage_shrinks_raw_residual(self) -> None:
        """A high-leverage point has its residual shrunk; studentizing undoes it."""
        rng = np.random.default_rng(0)
        n = 12
        x = np.concatenate([np.linspace(0, 1, n - 1), [8.0]])  # last = high leverage
        jac = np.column_stack([np.ones(n), x])
        r = rng.normal(0, 1, n)
        scores, dof = studentized_scores(r, jac)
        assert dof == n - 2 - 1
        q, _ = np.linalg.qr(jac)
        h = np.einsum("ij,ij->i", q, q)
        assert h[-1] > 0.8  # the design point really is high leverage
        assert scores[-1] > abs(r[-1]) / np.std(r)  # inflated back up

    def test_bonferroni_threshold_tightens_with_n(self) -> None:
        """Testing more points simultaneously demands a larger cutoff."""
        t10 = bonferroni_threshold(10, dof=20)
        t100 = bonferroni_threshold(100, dof=20)
        assert t100 > t10 > 2.0
        assert np.isinf(bonferroni_threshold(10, dof=0))

    def test_studentized_catches_midpoint_outlier_mad_misses(self) -> None:
        """The leverage-aware screen catches what the MAD screen cannot.

        This is the same midpoint case pinned in
        `test_mad_misses_high_leverage_midpoint_outlier`.
        """

        def build() -> Dataset:
            return _create_synthetic_dataset(
                add_outlier=True,
                outlier_label="1",
                outlier_idx=3,
                outlier_magnitude=10.0,
            )

        fr_mad = fit_binding_glob(build(), method="huber", remove_outliers="mad:3.5")
        fr_stu = fit_binding_glob(
            build(), method="huber", remove_outliers="studentized:0.05"
        )
        assert fr_mad.dataset is not None
        assert fr_stu.dataset is not None
        assert fr_mad.dataset["1"].mask[3]  # MAD misses it
        assert not fr_stu.dataset["1"].mask[3]  # studentized catches it

    def test_studentized_leaves_clean_data_alone(self) -> None:
        """Bonferroni control means clean plates are essentially untouched."""
        ds = _create_synthetic_dataset(add_outlier=False, seed=42)
        fr = fit_binding_glob(ds, method="huber", remove_outliers="studentized:0.05")
        assert fr.dataset is not None
        for da in fr.dataset.values():
            assert da.x.size == 7

    def test_studentized_respects_min_keep(self) -> None:
        """An absurd alpha still cannot strip a label past min_keep."""
        ds = _create_synthetic_dataset()
        fr = fit_binding_glob(ds, method="huber", remove_outliers="studentized:0.999:6")
        assert fr.dataset is not None
        for da in fr.dataset.values():
            assert da.x.size >= 6


class TestAddRobustScores:
    """Tests for multi-level robust scale/score annotation."""

    @staticmethod
    def _frame(n_wells: int = 6, seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        wells = [f"A{i:02d}" for i in range(n_wells)]
        df = pd.DataFrame({
            "well": np.repeat(wells, 14),
            "label": np.tile(np.repeat(["1", "2"], 7), n_wells),
            "raw_res": rng.normal(0, 1, 14 * n_wells),
        })
        df.loc[df.well == wells[0], "raw_res"] *= 5.0  # one noisy well
        return df

    def test_adds_all_levels(self) -> None:
        """Each requested level yields a sigma and a z column."""
        out = add_robust_scores(self._frame())
        for level in ("well_label", "well", "label", "global"):
            assert f"robust_sigma_{level}" in out.columns
            assert f"robust_z_{level}" in out.columns
        assert "ye_mag_est" in out.columns

    def test_ye_mag_est_tracks_noisy_well(self) -> None:
        """The scale ratio flags the well whose noise was inflated."""
        out = add_robust_scores(self._frame())
        per_well = out.groupby("well")["ye_mag_est"].first()
        assert per_well.idxmax() == "A00"
        assert per_well["A00"] > 1.5 * per_well.drop("A00").max()

    def test_pooled_scale_is_shared(self) -> None:
        """A pooled level gives one scale per group, not one per well."""
        out = add_robust_scores(self._frame())
        for _lbl, grp in out.groupby("label"):
            assert grp["robust_sigma_label"].std() == pytest.approx(0, abs=1e-9)
        assert out["robust_sigma_global"].std() == pytest.approx(0, abs=1e-9)
        for _key, grp in out.groupby(["well", "label"]):
            assert grp["robust_sigma_well_label"].std() == pytest.approx(0, abs=1e-9)

    def test_single_well_marks_degenerate_levels(self) -> None:
        """Unidentifiable levels are NaN and recorded, not silently duplicated."""
        one = self._frame()
        one = one[one.well == "A00"].copy()
        out = add_robust_scores(one)
        assert out.attrs["robust_score_degenerate_levels"] == ["label"]
        assert out["robust_sigma_label"].isna().all()
        assert out["robust_sigma_well_label"].notna().all()

    def test_well_level_mixes_unequal_label_scales(self) -> None:
        """Known caveat: pooling labels of unequal scale miscalibrates both.

        On real plates a bright and a dim band differ by an order of magnitude,
        so the pooled per-well scale sits between them -- too small for the
        noisy label, too large for the quiet one.
        """
        df = self._frame()
        df.loc[df.label == "1", "raw_res"] *= 20.0  # bright band
        out = add_robust_scores(df)
        per_label = out.groupby("label")[
            ["robust_sigma_well_label", "robust_sigma_well"]
        ].median()
        # The pooled scale is far too small for label 1 and far too big for 2.
        pooled = per_label["robust_sigma_well"].to_dict()
        own = per_label["robust_sigma_well_label"].to_dict()
        assert pooled["1"] < 0.75 * own["1"]
        assert pooled["2"] > 2 * own["2"]

    def test_single_label_marks_well_level_degenerate(self) -> None:
        """With one label, the per-well level collapses onto well_label."""
        df = self._frame()
        df = df[df.label == "1"].copy()
        out = add_robust_scores(df)
        assert "well" in out.attrs["robust_score_degenerate_levels"]
        assert out["robust_sigma_well"].isna().all()

    def test_rejects_unknown_level_and_column(self) -> None:
        """Bad input fails loudly rather than silently doing nothing."""
        df = self._frame()
        with pytest.raises(ValueError, match="Unknown level"):
            add_robust_scores(df, levels=("per_plate",))
        with pytest.raises(ValueError, match="no column"):
            add_robust_scores(df, residual_col="nope")
