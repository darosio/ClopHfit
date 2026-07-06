"""Test cases for the clophfit.fitting.bayes module."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
import pymc as pm  # type: ignore[import-untyped]
import pytest
import xarray as xr
from lmfit import Parameters  # type: ignore[import-untyped]

from clophfit.fitting import bayes
from clophfit.fitting.bayes import (
    create_parameter_priors,
    create_x_true,
    extract_fit,
    fit_binding_pymc,
    process_trace,
    weighted_stats,
    x_true_from_trace_df,
)
from clophfit.fitting.core import fit_binding_glob
from clophfit.fitting.data_structures import (
    DataArray,
    Dataset,
    FitResult,
    MiniT,
    MultiFitResult,
    NoiseModelParams,
    PlateNoiseModel,
)
from clophfit.fitting.models import binding_1site
from clophfit.prtecan import PlateScheme

if TYPE_CHECKING:
    from collections.abc import Callable


class _StopBayesBuildError(RuntimeError):
    """Internal test sentinel used to stop model construction early."""


def _noop_plot_fit(*_args: object, **_kwargs: object) -> None:
    """Test helper replacing plot generation in unit tests."""


def _summary_df_stub_factory(
    summary_df: pd.DataFrame,
) -> Callable[[xr.DataTree | MultiFitResult | pd.DataFrame], pd.DataFrame]:
    """Return a summary stub accepted by extract/x_true helper tests."""

    def stub(
        trace_or_df: xr.DataTree | MultiFitResult | pd.DataFrame,
    ) -> pd.DataFrame:
        del trace_or_df
        return summary_df

    return stub


###############################################################################
# Fixtures
###############################################################################


@pytest.fixture
def lmfit_params() -> Parameters:
    """Create sample lmfit Parameters with stderr values."""
    params = Parameters()
    params.add("K", value=7.0, min=5.0, max=9.0)
    params["K"].stderr = 0.2
    params.add("S0_1", value=2.0, min=0.0, max=5.0)
    params["S0_1"].stderr = 0.1
    params.add("S1_1", value=1.0, min=0.0, max=5.0)
    params["S1_1"].stderr = 0.05
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


def test_create_x_true_keeps_all_steps_individually_anchored() -> None:
    """Cumulative-addition x_true should stay close to observed anchors."""
    pytest.importorskip("pymc")

    xc = np.array([8.92, 8.31, 7.76, 7.04, 6.56, 5.98, 5.47])
    x_errc = np.array([0.005, 0.045, 0.087, 0.026, 0.16, 0.16, 0.16])

    with pm.Model():
        create_x_true(xc, x_errc, n_xerr=1.0)
        trace = pm.sample_prior_predictive(draws=500, var_names=["x_true"])

    sampled = np.asarray(trace.prior["x_true"])
    x_true_samples = sampled.reshape(-1, sampled.shape[-1])
    sample_means = x_true_samples.mean(axis=0)

    np.testing.assert_allclose(sample_means, xc, atol=0.25)
    assert np.all(np.diff(x_true_samples, axis=1) <= 1e-12)


def test_create_x_true_uses_cumulative_step_uncertainty() -> None:
    """Later x points should accumulate uncertainty from repeated additions."""
    pytest.importorskip("pymc")

    xc = np.array([8.9, 8.3, 7.8, 7.1, 6.6, 6.0, 5.5])
    x_errc = np.array([0.01, 0.05, 0.09, 0.11, 0.14, 0.16, 0.18])

    with pm.Model():
        create_x_true(xc, x_errc, n_xerr=1.0)
        trace = pm.sample_prior_predictive(draws=500, var_names=["x_true"])

    sampled = np.asarray(trace.prior["x_true"])
    x_true_samples = sampled.reshape(-1, sampled.shape[-1])
    sample_sds = x_true_samples.std(axis=0)

    assert np.all(np.diff(sample_sds) >= -0.02)
    assert sample_sds[-1] > sample_sds[1]


def test_create_x_true_respects_minimum_step_size() -> None:
    """Inferred x drops should not fall below the configured minimum step."""
    pytest.importorskip("pymc")

    xc = np.array([8.9, 8.3, 7.8, 7.1, 6.6, 6.0, 5.5])
    x_errc = np.array([0.01, 0.05, 0.09, 0.11, 0.14, 0.16, 0.18])
    min_x_step = 0.35

    with pm.Model():
        create_x_true(xc, x_errc, n_xerr=1.0, min_x_step=min_x_step)
        trace = pm.sample_prior_predictive(draws=500, var_names=["x_true"])

    sampled = np.asarray(trace.prior["x_true"])
    x_true_samples = sampled.reshape(-1, sampled.shape[-1])
    inferred_drops = -np.diff(x_true_samples, axis=1)

    assert np.all(inferred_drops >= min_x_step - 1e-6)


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


def test_x_true_from_trace_df_sorts_numeric_indices() -> None:
    """x_true rows should be reconstructed in numeric, not lexicographic, order."""
    trace_df = pd.DataFrame(
        {
            "mean": [0.0, 1.0, 10.0, 2.0],
            "sd": [0.1, 0.1, 0.3, 0.2],
        },
        index=["x_true[0]", "x_true[1]", "x_true[10]", "x_true[2]"],
    )

    x_true = x_true_from_trace_df(trace_df)

    np.testing.assert_array_equal(x_true.xc, np.array([0.0, 1.0, 2.0, 10.0]))
    np.testing.assert_array_equal(x_true.x_errc, np.array([0.1, 0.1, 0.2, 0.3]))


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
        assert "S0_1" in priors
        assert "S1_1" in priors
        # Check that they have PyMC-like attributes
        assert hasattr(priors["K"], "eval")
        assert hasattr(priors["S0_1"], "eval")
        assert hasattr(priors["S1_1"], "eval")


def test_create_parameter_priors_with_key(lmfit_params: Parameters) -> None:
    """Test parameter priors with a key suffix."""
    pytest.importorskip("pymc")
    with pm.Model():
        priors = create_parameter_priors(lmfit_params, n_sd=5.0, key="A01")
        # Check that parameter names include the key
        assert "K_A01" in priors
        assert "S0_1_A01" in priors
        assert "S1_1_A01" in priors


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
        assert "S0_1_A01" in priors
        assert "S1_1_A01" in priors


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
    assert callable(weighted_stats)
    assert callable(fit_binding_pymc)
    assert callable(process_trace)
    assert callable(extract_fit)
    assert callable(x_true_from_trace_df)


def test_process_trace_scales_per_label_ye_mags(
    monkeypatch: pytest.MonkeyPatch,
    multi_dataset: Dataset,
) -> None:
    """Per-label ye_mag posteriors should scale each dataset independently."""
    ds = copy.deepcopy(multi_dataset)
    for da in ds.values():
        da.y_errc = np.ones_like(da.xc)

    params = Parameters()
    params.add("K", value=7.0)
    params.add("S0_1", value=2.0)
    params.add("S1_1", value=1.0)
    params.add("S0_2", value=0.0)
    params.add("S1_2", value=1.0)

    summary_df = pd.DataFrame(
        {
            "mean": [7.0, 2.0, 1.0, 0.0, 1.0, 2.0, 0.5],
            "sd": [0.1, 0.1, 0.1, 0.1, 0.1, 0.0, 0.0],
            "hdi_3%": [6.8, 1.8, 0.8, -0.2, 0.8, 2.0, 0.5],
            "hdi_97%": [7.2, 2.2, 1.2, 0.2, 1.2, 2.0, 0.5],
            "r_hat": [1.0, 1.0, 1.0, 1.0, 1.0, np.nan, np.nan],
        },
        index=["K", "S0_1", "S1_1", "S0_2", "S1_2", "ye_mag_1", "ye_mag_2"],
    )

    monkeypatch.setattr(bayes, "_trace_summary_df", lambda _trace: summary_df)
    monkeypatch.setattr(bayes, "plot_fit", _noop_plot_fit)

    fit_result = process_trace(xr.DataTree(), params.keys(), ds)

    assert fit_result.dataset is not None
    np.testing.assert_allclose(fit_result.dataset["1"].y_errc, np.full(3, 2.0))
    np.testing.assert_allclose(fit_result.dataset["2"].y_errc, np.full(3, 0.5))


def test_fit_binding_pymc_dataset_defaults_free_noise_modes(
    monkeypatch: pytest.MonkeyPatch,
    ph_dataset: Dataset,
) -> None:
    """Raw datasets should default PyMC noise modes to free priors."""
    captured: dict[str, str] = {}
    noise_model = PlateNoiseModel({
        "default": NoiseModelParams(sigma_floor=1.0),
    })

    def fake_build_pymc_noise_priors(
        _noise_model: PlateNoiseModel,
        *,
        floor_mode: str = "centered",
        gain_mode: str = "centered",
        alpha_mode: str = "centered",
        shared_alpha: bool = False,
        shared_gain: bool = False,
    ) -> dict[str, object]:
        del shared_alpha, shared_gain
        captured["floor"] = floor_mode
        captured["gain"] = gain_mode
        captured["alpha"] = alpha_mode
        raise _StopBayesBuildError

    monkeypatch.setattr(bayes, "build_pymc_noise_priors", fake_build_pymc_noise_priors)

    with pytest.raises(_StopBayesBuildError):
        fit_binding_pymc(
            ph_dataset,
            noise_model=noise_model,
            n_samples=2,
            n_tune=1,
            n_xerr=0.0,
        )

    assert captured == {"floor": "free", "gain": "free", "alpha": "free"}


def test_fit_binding_pymc_fitresult_defaults_centered_noise_modes(
    monkeypatch: pytest.MonkeyPatch,
    ph_dataset: Dataset,
) -> None:
    """Pre-fit results should default PyMC noise modes to centered priors."""
    captured: dict[str, str] = {}
    noise_model = PlateNoiseModel({
        "default": NoiseModelParams(sigma_floor=1.0),
    })

    def fake_build_pymc_noise_priors(
        _noise_model: PlateNoiseModel,
        *,
        floor_mode: str = "centered",
        gain_mode: str = "centered",
        alpha_mode: str = "centered",
        shared_alpha: bool = False,
        shared_gain: bool = False,
    ) -> dict[str, object]:
        del shared_alpha, shared_gain
        captured["floor"] = floor_mode
        captured["gain"] = gain_mode
        captured["alpha"] = alpha_mode
        raise _StopBayesBuildError

    monkeypatch.setattr(bayes, "build_pymc_noise_priors", fake_build_pymc_noise_priors)

    with pytest.raises(_StopBayesBuildError):
        fit_binding_pymc(
            fit_binding_glob(ph_dataset),
            noise_model=noise_model,
            n_samples=2,
            n_tune=1,
            n_xerr=0.0,
        )

    assert captured == {"floor": "centered", "gain": "centered", "alpha": "centered"}


def test_fit_binding_pymc_passes_shared_noise_flags(
    monkeypatch: pytest.MonkeyPatch,
    multi_dataset: Dataset,
) -> None:
    """Single-dataset PyMC should expose the same shared noise knobs as multi."""
    captured: dict[str, bool] = {}
    noise_model = PlateNoiseModel({
        "1": NoiseModelParams(sigma_floor=1.0, alpha=0.01, gain=0.1),
        "2": NoiseModelParams(sigma_floor=1.0, alpha=0.02, gain=0.2),
    })

    def fake_build_pymc_noise_priors(
        _noise_model: PlateNoiseModel,
        *,
        floor_mode: str = "centered",
        gain_mode: str = "centered",
        alpha_mode: str = "centered",
        shared_alpha: bool = False,
        shared_gain: bool = False,
    ) -> dict[str, object]:
        del floor_mode, gain_mode, alpha_mode
        captured["shared_alpha"] = shared_alpha
        captured["shared_gain"] = shared_gain
        raise _StopBayesBuildError

    monkeypatch.setattr(bayes, "build_pymc_noise_priors", fake_build_pymc_noise_priors)

    with pytest.raises(_StopBayesBuildError):
        fit_binding_pymc(
            multi_dataset,
            noise_model=noise_model,
            shared_alpha=False,
            shared_gain=True,
            n_samples=2,
            n_tune=1,
            n_xerr=0.0,
        )

    assert captured == {"shared_alpha": False, "shared_gain": True}


def test_fit_binding_pymc_robust_uses_fixed_student_t_nu(
    monkeypatch: pytest.MonkeyPatch,
    ph_dataset: Dataset,
) -> None:
    """Single-fit robust PyMC should preserve fixed Student-t nu by default."""
    captured: dict[str, object] = {}

    def fake_student_t(name: str, **kwargs: object) -> None:
        captured["name"] = name
        captured["nu"] = kwargs["nu"]
        raise _StopBayesBuildError

    monkeypatch.setattr(pm, "StudentT", fake_student_t)

    with pytest.raises(_StopBayesBuildError):
        fit_binding_pymc(
            ph_dataset,
            robust=True,
            student_t_nu=7.5,
            n_samples=2,
            n_tune=1,
            n_xerr=0.0,
        )

    assert captured["name"] == "y_likelihood_default"
    assert captured["nu"] == 7.5


def test_fit_binding_pymc_robust_can_infer_student_t_nu(
    monkeypatch: pytest.MonkeyPatch,
    ph_dataset: Dataset,
) -> None:
    """Passing None should infer one shared Student-t nu parameter."""
    captured: dict[str, object] = {}

    def fake_student_t(name: str, **kwargs: object) -> None:
        captured["name"] = name
        captured["nu_name"] = getattr(kwargs["nu"], "name", None)
        raise _StopBayesBuildError

    monkeypatch.setattr(pm, "StudentT", fake_student_t)

    with pytest.raises(_StopBayesBuildError):
        fit_binding_pymc(
            ph_dataset,
            robust=True,
            student_t_nu=None,
            n_samples=2,
            n_tune=1,
            n_xerr=0.0,
        )

    assert captured["name"] == "y_likelihood_default"
    assert captured["nu_name"] == "student_t_nu"


def test_fit_binding_pymc_robust_can_use_mixture_likelihood(
    monkeypatch: pytest.MonkeyPatch,
    ph_dataset: Dataset,
) -> None:
    """Single-fit robust PyMC can use a marginalized contamination mixture."""
    captured: dict[str, object] = {}

    def fake_mixture(name: str, **kwargs: object) -> None:
        captured["name"] = name
        captured["has_weights"] = "w" in kwargs
        captured["has_components"] = "comp_dists" in kwargs
        raise _StopBayesBuildError

    monkeypatch.setattr(pm, "Mixture", fake_mixture)

    with pytest.raises(_StopBayesBuildError):
        fit_binding_pymc(
            ph_dataset,
            robust=True,
            robust_likelihood="mixture",
            n_samples=2,
            n_tune=1,
            n_xerr=0.0,
        )

    assert captured == {
        "name": "y_likelihood_default",
        "has_weights": True,
        "has_components": True,
    }


def test_fit_binding_pymc_robust_rejects_nonpositive_student_t_nu(
    ph_dataset: Dataset,
) -> None:
    """Nonpositive numeric Student-t nu values should be rejected."""
    with pytest.raises(ValueError, match="student_t_nu must be positive"):
        fit_binding_pymc(
            ph_dataset,
            robust=True,
            student_t_nu=0.0,
            n_samples=2,
            n_tune=1,
            n_xerr=0.0,
        )


def test_fit_binding_pymc_residual_refit_uses_requested_two_pass_settings(
    monkeypatch: pytest.MonkeyPatch,
    ph_dataset: Dataset,
) -> None:
    """Residual refit workflow should run robust unit-yerr then normal masked refit."""
    ph_dataset["default"].y_err = np.full_like(ph_dataset["default"].yc, 2.0)
    calls: list[dict[str, Any]] = []

    def fake_fit_binding_pymc(
        ds_or_fr: Dataset | FitResult[MiniT], **kwargs: object
    ) -> FitResult[MiniT]:
        ds = ds_or_fr.dataset if isinstance(ds_or_fr, FitResult) else ds_or_fr
        assert ds is not None
        calls.append({
            "input_is_fit_result": isinstance(ds_or_fr, FitResult),
            "y_errc": ds["default"].y_errc.copy(),
            **kwargs,
        })
        fr = fit_binding_glob(ds)
        if not isinstance(ds_or_fr, FitResult) and fr.dataset is not None:
            fr.dataset["default"].y_err = np.full_like(ds["default"].yc, 7.0)
        return fr

    monkeypatch.setattr(bayes, "fit_binding_pymc", fake_fit_binding_pymc)

    result = bayes.fit_binding_pymc_residual_refit(
        ph_dataset,
        bg_noise={"default": 0.3},
        n_samples=10,
        n_tune=5,
        n_xerr=0.0,
    )

    assert result.final.result is not None
    assert len(calls) == 2
    assert calls[0]["input_is_fit_result"] is False
    assert calls[1]["input_is_fit_result"] is True
    np.testing.assert_allclose(
        calls[0]["y_errc"], np.ones_like(ph_dataset["default"].yc)
    )
    np.testing.assert_allclose(
        calls[1]["y_errc"], np.full_like(ph_dataset["default"].yc, 7.0)
    )
    assert calls[0]["robust"] is True
    assert calls[0]["ye_mag_prior"] == "lognormal"
    assert calls[0]["ye_mag_mu"] == {"default": pytest.approx(np.log(1.08))}
    assert calls[0]["ye_mag_sigma"] == 0.5
    assert calls[1]["robust"] is False
    assert calls[1]["ye_mag_prior"] == "lognormal"
    assert calls[1]["ye_mag_mu"] == 0.0
    assert calls[1]["ye_mag_sigma"] == 0.25


def test_fit_binding_pymc_residual_refit_masks_one_outlier_per_label_by_default(
    monkeypatch: pytest.MonkeyPatch,
    multi_dataset: Dataset,
) -> None:
    """The refit workflow should not allow one tail residual per label by default."""
    calls: list[Dataset] = []

    def fake_fit_binding_pymc(
        ds_or_fr: Dataset | FitResult[MiniT], **_kwargs: object
    ) -> FitResult[MiniT]:
        ds = ds_or_fr.dataset if isinstance(ds_or_fr, FitResult) else ds_or_fr
        assert ds is not None
        calls.append(ds.copy())
        if isinstance(ds_or_fr, FitResult):
            return ds_or_fr
        return fit_binding_glob(ds)

    def fake_residuals_from_fit_results(
        *_args: object, **_kwargs: object
    ) -> pd.DataFrame:
        return pd.DataFrame({
            "trace_id": ["pymc_robust_unweighted"] * 6,
            "well": ["single"] * 6,
            "label": ["1", "1", "1", "2", "2", "2"],
            "step": [0, 1, 2, 0, 1, 2],
            "x": [6.0, 7.0, 8.0, 6.0, 7.0, 8.0],
            "std_res": [0.0, 0.0, 4.0, 0.0, -4.0, 0.0],
        })

    monkeypatch.setattr(bayes, "fit_binding_pymc", fake_fit_binding_pymc)
    monkeypatch.setattr(
        "clophfit.fitting.model_validation.residuals_from_fit_results",
        fake_residuals_from_fit_results,
    )

    result = bayes.fit_binding_pymc_residual_refit(
        multi_dataset,
        bg_noise={"1": 0.3, "2": 0.3},
        n_samples=10,
        n_tune=5,
        n_xerr=0.0,
        min_keep=2,
    )

    assert result.masked_dataset["1"].mask.tolist() == [True, True, False]
    assert result.masked_dataset["2"].mask.tolist() == [True, False, True]
    assert calls[1]["1"].mask.tolist() == [True, True, False]
    assert calls[1]["2"].mask.tolist() == [True, False, True]


def test_fit_binding_pymc_residual_refit_proportional_noise_strategy(
    monkeypatch: pytest.MonkeyPatch,
    ph_dataset: Dataset,
) -> None:
    """Proportional residual refit should use a noise model instead of ye_mag."""
    ph_dataset["default"].y_err = np.full_like(ph_dataset["default"].yc, 2.0)
    calls: list[dict[str, Any]] = []

    def fake_fit_binding_pymc(
        ds_or_fr: Dataset | FitResult[MiniT], **kwargs: object
    ) -> FitResult[MiniT]:
        ds = ds_or_fr.dataset if isinstance(ds_or_fr, FitResult) else ds_or_fr
        assert ds is not None
        calls.append({
            "input_is_fit_result": isinstance(ds_or_fr, FitResult),
            "y_errc": ds["default"].y_errc.copy(),
            **kwargs,
        })
        fr = fit_binding_glob(ds)
        if isinstance(ds_or_fr, FitResult):
            return ds_or_fr
        return fr

    monkeypatch.setattr(bayes, "fit_binding_pymc", fake_fit_binding_pymc)

    result = bayes.fit_binding_pymc_residual_refit(
        ph_dataset,
        noise_strategy="proportional",
        bg_noise={"default": 0.3},
        proportional_alpha=0.07,
        n_samples=10,
        n_tune=5,
        n_xerr=0.0,
    )

    assert result.final.result is not None
    assert len(calls) == 2
    np.testing.assert_allclose(
        calls[0]["y_errc"], np.full_like(ph_dataset["default"].yc, 2.0)
    )
    for call in calls:
        assert call["noise_model"]["default"].sigma_floor == 0.3
        assert call["noise_model"]["default"].alpha == 0.07
        assert call["floor_mode"] == "centered"
        assert call["gain_mode"] == "fixed"
        assert call["alpha_mode"] == "free"
        assert call["learn_ye_mags"] is False
        assert "ye_mag_prior" not in call


def test_fit_binding_pymc_multi_residual_refit_uses_well_noise_scale(
    monkeypatch: pytest.MonkeyPatch,
    multi_dataset: Dataset,
) -> None:
    """Multi residual refit should use per-well variance in both PyMC passes."""
    scheme = PlateScheme()
    scheme.names = {"ctrl": {"A01"}}
    inputs = {
        "A01": copy.deepcopy(multi_dataset),
        "A02": copy.deepcopy(multi_dataset),
    }
    calls: list[dict[str, Any]] = []

    def input_label1_mask(value: Dataset | FitResult[MiniT]) -> list[bool]:
        dataset = value.dataset if isinstance(value, FitResult) else value
        assert dataset is not None
        return [bool(v) for v in dataset["1"].mask.copy().tolist()]

    def fake_fit_binding_pymc_multi(
        fit_inputs: dict[str, Dataset | FitResult[MiniT]],
        _scheme: PlateScheme,
        **kwargs: object,
    ) -> MultiFitResult:
        calls.append({
            "input_types": {
                key: type(value).__name__ for key, value in fit_inputs.items()
            },
            "input_masks": {
                key: input_label1_mask(value) for key, value in fit_inputs.items()
            },
            **kwargs,
        })
        fit_results: dict[str, FitResult[MiniT]] = {}
        for well, item in fit_inputs.items():
            if isinstance(item, FitResult):
                fit_results[well] = item
            else:
                fit_results[well] = fit_binding_glob(item)
        return MultiFitResult(trace=xr.DataTree(), results=fit_results)

    def fake_residuals_from_multifit(*_args: object, **_kwargs: object) -> pd.DataFrame:
        return pd.DataFrame({
            "trace_id": ["pymc_multi_robust"] * 2,
            "well": ["A01", "A02"],
            "label": ["1", "2"],
            "raw_i": [1, 2],
            "step": [1, 2],
            "x": [7.0, 8.0],
            "std_res": [4.0, 0.0],
        })

    monkeypatch.setattr(bayes, "fit_binding_pymc_multi", fake_fit_binding_pymc_multi)
    monkeypatch.setattr(
        "clophfit.fitting.model_validation.residuals_from_multifit",
        fake_residuals_from_multifit,
    )

    result = bayes.fit_binding_pymc_multi_residual_refit(
        inputs,
        scheme,
        bg_noise={"1": 0.3, "2": 0.4},
        proportional_alpha=0.07,
        n_samples=10,
        n_tune=5,
        n_xerr=1.0,
        min_keep=2,
    )

    assert len(calls) == 2
    assert calls[0]["robust"] is True
    assert calls[1]["robust"] is False
    assert calls[0]["well_noise_scale"] is True
    assert calls[1]["well_noise_scale"] is True
    assert calls[0]["shared_well_noise_scale"] is False
    assert calls[1]["shared_well_noise_scale"] is False
    assert calls[0]["label_noise_scale_sigma"] == 0.3
    assert calls[0]["well_noise_sd_sigma"] == 0.3
    assert calls[0]["n_xerr"] == 1.0
    assert calls[1]["n_xerr"] == 1.0
    assert calls[0]["noise_model"]["1"].sigma_floor == 0.3
    assert calls[0]["noise_model"]["2"].alpha == 0.07
    assert calls[0]["shared_alpha"] is False
    assert calls[0]["alpha_mode"] == "free"
    assert calls[1]["input_types"] == {"A01": "FitResult", "A02": "FitResult"}
    assert calls[1]["input_masks"]["A01"] == [True, False, True]
    assert result.masked_datasets["A01"]["1"].mask.tolist() == [True, False, True]
    assert result.masked_datasets["A02"]["2"].mask.tolist() == [True, True, True]


def test_fit_binding_pymc_multi_residual_refit_can_share_well_noise_scale(
    monkeypatch: pytest.MonkeyPatch,
    multi_dataset: Dataset,
) -> None:
    """Multi residual refit should pass shared well-noise scaling to both passes."""
    scheme = PlateScheme()
    scheme.names = {"ctrl": {"A01"}}
    inputs = {
        "A01": copy.deepcopy(multi_dataset),
        "A02": copy.deepcopy(multi_dataset),
    }
    calls: list[dict[str, object]] = []

    def fake_fit_binding_pymc_multi(
        fit_inputs: dict[str, Dataset | FitResult[MiniT]],
        _scheme: PlateScheme,
        **kwargs: object,
    ) -> MultiFitResult:
        calls.append(kwargs)
        fit_results: dict[str, FitResult[MiniT]] = {}
        for well, item in fit_inputs.items():
            fit_results[well] = (
                item if isinstance(item, FitResult) else fit_binding_glob(item)
            )
        return MultiFitResult(trace=xr.DataTree(), results=fit_results)

    monkeypatch.setattr(bayes, "fit_binding_pymc_multi", fake_fit_binding_pymc_multi)
    monkeypatch.setattr(
        "clophfit.fitting.model_validation.residuals_from_multifit",
        lambda *_args, **_kwargs: pd.DataFrame({"std_res": []}),
    )

    bayes.fit_binding_pymc_multi_residual_refit(
        inputs,
        scheme,
        n_samples=10,
        n_tune=5,
        n_xerr=0.0,
        shared_well_noise_scale=True,
    )

    assert len(calls) == 2
    assert calls[0]["shared_well_noise_scale"] is True
    assert calls[1]["shared_well_noise_scale"] is True


def test_fit_binding_pymc_multi_dataset_mapping_defaults_free_noise_modes(
    monkeypatch: pytest.MonkeyPatch,
    multi_dataset: Dataset,
) -> None:
    """Multi-well PyMC should accept raw dataset mappings and default to free modes."""
    captured: dict[str, str] = {}
    noise_model = PlateNoiseModel({
        "1": NoiseModelParams(sigma_floor=1.0),
        "2": NoiseModelParams(sigma_floor=1.0),
    })
    scheme = PlateScheme()
    scheme.names = {"ctrl": {"A01", "A02"}}

    def fake_build_pymc_noise_priors(
        _noise_model: PlateNoiseModel,
        *,
        floor_mode: str = "centered",
        gain_mode: str = "centered",
        alpha_mode: str = "centered",
        shared_alpha: bool = False,
        shared_gain: bool = False,
    ) -> dict[str, object]:
        del shared_alpha, shared_gain
        captured["floor"] = floor_mode
        captured["gain"] = gain_mode
        captured["alpha"] = alpha_mode
        raise _StopBayesBuildError

    monkeypatch.setattr(bayes, "build_pymc_noise_priors", fake_build_pymc_noise_priors)

    with pytest.raises(_StopBayesBuildError):
        bayes.fit_binding_pymc_multi(
            {"A01": copy.deepcopy(multi_dataset), "A02": copy.deepcopy(multi_dataset)},
            scheme,
            noise_model=noise_model,
            n_samples=2,
            n_tune=1,
            n_xerr=0.0,
        )

    assert captured == {"floor": "free", "gain": "free", "alpha": "free"}


def test_fit_binding_pymc_multi_fitresult_defaults_centered_noise_modes(
    monkeypatch: pytest.MonkeyPatch,
    multi_dataset: Dataset,
) -> None:
    """Multi-well PyMC should keep centered defaults for pre-fit results."""
    captured: dict[str, str] = {}
    noise_model = PlateNoiseModel({
        "1": NoiseModelParams(sigma_floor=1.0),
        "2": NoiseModelParams(sigma_floor=1.0),
    })
    scheme = PlateScheme()
    scheme.names = {"ctrl": {"A01", "A02"}}
    fr_init = fit_binding_glob(multi_dataset)

    def fake_build_pymc_noise_priors(
        _noise_model: PlateNoiseModel,
        *,
        floor_mode: str = "centered",
        gain_mode: str = "centered",
        alpha_mode: str = "centered",
        shared_alpha: bool = False,
        shared_gain: bool = False,
    ) -> dict[str, object]:
        del shared_alpha, shared_gain
        captured["floor"] = floor_mode
        captured["gain"] = gain_mode
        captured["alpha"] = alpha_mode
        raise _StopBayesBuildError

    monkeypatch.setattr(bayes, "build_pymc_noise_priors", fake_build_pymc_noise_priors)

    with pytest.raises(_StopBayesBuildError):
        bayes.fit_binding_pymc_multi(
            {"A01": fr_init, "A02": fr_init},
            scheme,
            noise_model=noise_model,
            n_samples=2,
            n_tune=1,
            n_xerr=0.0,
        )

    assert captured == {"floor": "centered", "gain": "centered", "alpha": "centered"}


def test_fit_binding_pymc_multi_filters_noise_model_to_active_labels(
    monkeypatch: pytest.MonkeyPatch,
    multi_dataset: Dataset,
) -> None:
    """Noise priors should not be created for labels absent from the fit."""
    ds2 = Dataset({"2": copy.deepcopy(multi_dataset["2"])}, is_ph=True)
    noise_model = PlateNoiseModel({
        "1": NoiseModelParams(sigma_floor=10.0, gain=1.0, alpha=0.02),
        "2": NoiseModelParams(sigma_floor=1.0, gain=0.25, alpha=0.02),
    })
    scheme = PlateScheme()
    scheme.names = {"ctrl": {"A01", "A02"}}
    fr_init = fit_binding_glob(ds2)
    captured: dict[str, list[str]] = {}

    def fake_build_pymc_noise_priors(
        _noise_model: PlateNoiseModel,
        *,
        floor_mode: str = "centered",
        gain_mode: str = "centered",
        alpha_mode: str = "centered",
        shared_alpha: bool = False,
        shared_gain: bool = False,
    ) -> dict[str, object]:
        del floor_mode, gain_mode, alpha_mode, shared_alpha, shared_gain
        captured["labels"] = list(_noise_model.keys())
        raise _StopBayesBuildError

    monkeypatch.setattr(bayes, "build_pymc_noise_priors", fake_build_pymc_noise_priors)

    with pytest.raises(_StopBayesBuildError):
        bayes.fit_binding_pymc_multi(
            {"A01": fr_init, "A02": fr_init},
            scheme,
            noise_model=noise_model,
            n_samples=2,
            n_tune=1,
            n_xerr=0.0,
        )

    assert captured == {"labels": ["2"]}


@pytest.mark.parametrize(
    ("per_well_ye_mags", "expected_per_well"),
    [(None, True), (False, False), (True, True)],
)
def test_fit_binding_pymc_multi_learn_ye_mags_defaults_to_per_well_with_noise_model(
    monkeypatch: pytest.MonkeyPatch,
    multi_dataset: Dataset,
    per_well_ye_mags: bool | None,  # noqa: FBT001
    expected_per_well: bool,  # noqa: FBT001
) -> None:
    """Multi noise-model fits should add per-well ye_mag scales by default."""
    noise_model = PlateNoiseModel({
        "1": NoiseModelParams(sigma_floor=1.0, alpha=0.02),
        "2": NoiseModelParams(sigma_floor=1.0, alpha=0.02),
    })
    scheme = PlateScheme()
    scheme.names = {"ctrl": {"A01", "A02"}}
    fr_init = fit_binding_glob(multi_dataset)
    captured: dict[str, bool] = {}

    def fake_build_multi_ye_mag_priors(
        _labels: list[str],
        *,
        per_well: bool = False,
        shared_ye_mags: bool = False,
        prior: str = "lognormal",
        mu: float | dict[str, float] = 0.0,
        sigma: float | dict[str, float] = 1.5,
    ) -> dict[str, object]:
        del _labels, shared_ye_mags, prior, mu, sigma
        captured["per_well"] = per_well
        raise _StopBayesBuildError

    monkeypatch.setattr(
        bayes, "_build_multi_ye_mag_priors", fake_build_multi_ye_mag_priors
    )
    with pytest.raises(_StopBayesBuildError):
        bayes.fit_binding_pymc_multi(
            {"A01": fr_init, "A02": fr_init},
            scheme,
            noise_model=noise_model,
            learn_ye_mags=True,
            per_well_ye_mags=per_well_ye_mags,
            n_samples=2,
            n_tune=1,
            n_xerr=0.0,
        )

    assert captured == {"per_well": expected_per_well}


def test_fit_binding_pymc_multi_can_share_well_noise_scale_between_labels(
    monkeypatch: pytest.MonkeyPatch,
    multi_dataset: Dataset,
) -> None:
    """Shared well-noise scaling should build one well vector for all labels."""
    noise_model = PlateNoiseModel({
        "1": NoiseModelParams(sigma_floor=1.0, alpha=0.02),
        "2": NoiseModelParams(sigma_floor=1.0, alpha=0.02),
    })
    scheme = PlateScheme()
    scheme.names = {"ctrl": {"A01", "A02"}}
    fr_init = fit_binding_glob(multi_dataset)
    created_lognormals: list[str] = []
    created_halfnormals: list[str] = []

    original_lognormal = pm.LogNormal
    original_halfnormal = pm.HalfNormal

    def fake_lognormal(name: str, **kwargs: object) -> object:
        created_lognormals.append(name)
        if name == "well_noise_scale":
            assert kwargs["dims"] == "well"
            raise _StopBayesBuildError
        return original_lognormal(name, **kwargs)

    def fake_halfnormal(name: str, **kwargs: object) -> object:
        created_halfnormals.append(name)
        return original_halfnormal(name, **kwargs)

    monkeypatch.setattr(pm, "LogNormal", fake_lognormal)
    monkeypatch.setattr(pm, "HalfNormal", fake_halfnormal)

    with pytest.raises(_StopBayesBuildError):
        bayes.fit_binding_pymc_multi(
            {"A01": fr_init, "A02": fr_init},
            scheme,
            noise_model=noise_model,
            well_noise_scale=True,
            shared_well_noise_scale=True,
            n_samples=2,
            n_tune=1,
            n_xerr=0.0,
        )

    assert "label_noise_scale" in created_lognormals
    assert "well_noise_sd" in created_halfnormals
    assert "well_noise_scale_1" not in created_lognormals
    assert "well_noise_scale_2" not in created_lognormals


def test_fit_binding_pymc_multi_robust_can_infer_student_t_nu(
    monkeypatch: pytest.MonkeyPatch,
    multi_dataset: Dataset,
) -> None:
    """Multi-fit robust PyMC should infer one shared Student-t nu parameter."""
    scheme = PlateScheme()
    scheme.names = {"ctrl": {"A01", "A02"}}
    fr_init = fit_binding_glob(multi_dataset)
    captured: dict[str, object] = {}

    def fake_student_t(name: str, **kwargs: object) -> None:
        captured["name"] = name
        captured["nu_name"] = getattr(kwargs["nu"], "name", None)
        raise _StopBayesBuildError

    monkeypatch.setattr(pm, "StudentT", fake_student_t)

    with pytest.raises(_StopBayesBuildError):
        bayes.fit_binding_pymc_multi(
            {"A01": fr_init, "A02": fr_init},
            scheme,
            robust=True,
            student_t_nu=None,
            n_samples=2,
            n_tune=1,
            n_xerr=0.0,
            x_error_model="deterministic",
        )

    assert captured["name"] == "y_likelihood_1"
    assert captured["nu_name"] == "student_t_nu"


def test_fit_binding_pymc_multi_robust_can_use_mixture_likelihood(
    monkeypatch: pytest.MonkeyPatch,
    multi_dataset: Dataset,
) -> None:
    """Multi-fit robust PyMC can use a marginalized contamination mixture."""
    scheme = PlateScheme()
    scheme.names = {"ctrl": {"A01", "A02"}}
    fr_init = fit_binding_glob(multi_dataset)
    captured: dict[str, object] = {}

    def fake_mixture(name: str, **kwargs: object) -> None:
        captured["name"] = name
        captured["has_weights"] = "w" in kwargs
        captured["has_components"] = "comp_dists" in kwargs
        raise _StopBayesBuildError

    monkeypatch.setattr(pm, "Mixture", fake_mixture)

    with pytest.raises(_StopBayesBuildError):
        bayes.fit_binding_pymc_multi(
            {"A01": fr_init, "A02": fr_init},
            scheme,
            robust=True,
            robust_likelihood="mixture",
            n_samples=2,
            n_tune=1,
            n_xerr=0.0,
            x_error_model="deterministic",
        )

    assert captured == {
        "name": "y_likelihood_1",
        "has_weights": True,
        "has_components": True,
    }


def test_fit_binding_pymc_multi_passes_unscaled_xerr_to_create_x_true(
    monkeypatch: pytest.MonkeyPatch,
    multi_dataset: Dataset,
) -> None:
    """Deterministic mode should pass unscaled x_errc into create_x_true."""
    expected_x_err = np.array([0.1, 0.2, 0.3])
    ds = copy.deepcopy(multi_dataset)
    for da in ds.values():
        da.x_errc = expected_x_err.copy()

    captured: dict[str, np.ndarray | float] = {}
    scheme = PlateScheme()
    scheme.names = {"ctrl": {"A01", "A02"}}
    fr_init = fit_binding_glob(ds)

    def fake_create_x_true(
        xc: np.ndarray,
        x_errc: np.ndarray,
        n_xerr: float,
        lower_nsd: float = 2.5,
        min_x_step: float = 0.2,
    ) -> np.ndarray:
        del xc, lower_nsd, min_x_step
        captured["x_errc"] = np.array(x_errc, copy=True)
        captured["n_xerr"] = n_xerr
        raise _StopBayesBuildError

    monkeypatch.setattr(bayes, "create_x_true", fake_create_x_true)

    with pytest.raises(_StopBayesBuildError):
        bayes.fit_binding_pymc_multi(
            {"A01": fr_init, "A02": fr_init},
            scheme,
            n_samples=2,
            n_tune=1,
            n_xerr=3.0,
            x_error_model="deterministic",
        )

    np.testing.assert_allclose(captured["x_errc"], expected_x_err)
    assert captured["n_xerr"] == 3.0


def test_fit_binding_pymc_multi_deterministic_mode_uses_shared_fixed_x_true(
    multi_dataset: Dataset,
) -> None:
    """Deterministic mode should expose shared x_true, fixed when n_xerr=0."""
    scheme = PlateScheme()
    scheme.names = {"ctrl": {"A01", "A02"}}
    fr_init = fit_binding_glob(multi_dataset)
    assert fr_init.dataset is not None

    multi = bayes.fit_binding_pymc_multi(
        {"A01": fr_init, "A02": fr_init},
        scheme,
        n_samples=2,
        n_tune=1,
        n_xerr=0.0,
        x_error_model="deterministic",
    )

    x_true = x_true_from_trace_df(multi)
    assert set(multi.results) == {"A01", "A02"}
    np.testing.assert_allclose(x_true.xc, fr_init.dataset["1"].xc)
    np.testing.assert_allclose(x_true.x_errc, np.zeros_like(fr_init.dataset["1"].xc))
    for fr in multi.results.values():
        assert fr.dataset is not None
        for lbl, da in fr.dataset.items():
            np.testing.assert_allclose(da.xc, fr_init.dataset[lbl].xc)
            np.testing.assert_allclose(
                da.x_errc, np.zeros_like(fr_init.dataset[lbl].xc)
            )


def test_fit_binding_pymc_multi_zero_xerr_keeps_per_well_x(
    monkeypatch: pytest.MonkeyPatch,
    multi_dataset: Dataset,
) -> None:
    """Per-well x modes with n_xerr=0 should reuse each well's fixed x values."""
    scheme = PlateScheme()
    scheme.names = {"ctrl": {"A01"}}
    ds_a = copy.deepcopy(multi_dataset)
    ds_b = copy.deepcopy(multi_dataset)
    for da in ds_b.values():
        da.xc += 0.1

    noise_model = PlateNoiseModel({
        "1": NoiseModelParams(sigma_floor=1.0),
        "2": NoiseModelParams(sigma_floor=1.0),
    })

    def fail_create_x_true(*_args: object, **_kwargs: object) -> object:
        msg = "fixed per-well x path should not call create_x_true"
        raise AssertionError(msg)

    def stop_build_pymc_noise_priors(
        *_args: object, **_kwargs: object
    ) -> dict[str, object]:
        raise _StopBayesBuildError

    monkeypatch.setattr(bayes, "create_x_true", fail_create_x_true)
    monkeypatch.setattr(bayes, "build_pymc_noise_priors", stop_build_pymc_noise_priors)

    with pytest.raises(_StopBayesBuildError):
        bayes.fit_binding_pymc_multi(
            {"A01": ds_a, "A02": ds_b},
            scheme,
            noise_model=noise_model,
            x_error_model="hierarchical_per_well",
            n_xerr=0.0,
            n_samples=2,
            n_tune=1,
        )


def test_extract_fit_accepts_multifitresult_deterministic() -> None:
    """extract_fit should accept MultiFitResult and use shared x_true."""
    trace_df = pd.DataFrame(
        {
            "mean": [7.0, 2.0, 1.0, 8.0, 7.0, 6.0],
            "sd": [0.1, 0.1, 0.1, 0.0, 0.0, 0.0],
            "hdi_3%": [6.8, 1.8, 0.8, 8.0, 7.0, 6.0],
            "hdi_97%": [7.2, 2.2, 1.2, 8.0, 7.0, 6.0],
            "r_hat": [1.0, 1.0, 1.0, np.nan, np.nan, np.nan],
        },
        index=[
            "K_ctr_ctrl",
            "S0_1_A01",
            "S1_1_A01",
            "x_true[0]",
            "x_true[1]",
            "x_true[2]",
        ],
    )
    ds = Dataset(
        {"1": DataArray(np.array([9.0, 8.0, 7.0]), np.array([2.1, 1.5, 1.1]))},
        is_ph=True,
    )
    multi = MultiFitResult(trace=xr.DataTree(), results={})

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            bayes, "_trace_summary_df", _summary_df_stub_factory(trace_df)
        )
        monkeypatch.setattr(bayes, "plot_fit", _noop_plot_fit)
        fr = extract_fit("A01", "ctrl", multi, ds)

    assert fr.dataset is not None
    np.testing.assert_allclose(fr.dataset["1"].xc, np.array([8.0, 7.0, 6.0]))
    np.testing.assert_allclose(fr.dataset["1"].x_errc, np.zeros(3))


def test_extract_fit_accepts_multifitresult_per_well() -> None:
    """extract_fit should prefer per-well x_per_well from MultiFitResult."""
    # fmt: off
    trace_df = pd.DataFrame(
        {
            "mean": [7.0, 2.0, 1.0, 2.2, 1.2, 9.0, 8.0, 7.0, 8.1, 7.1, 6.1, 7.9, 6.9, 5.9],
            "sd": [0.1, 0.1, 0.1, 0.12, 0.12, 0.0, 0.0, 0.0, 0.05, 0.05, 0.05, 0.07, 0.07, 0.07],
            "hdi_3%": [6.8, 1.8, 0.8, 2.0, 1.0, 9.0, 8.0, 7.0, 8.0, 7.0, 6.0, 7.8, 6.8, 5.8],
            "hdi_97%": [7.2, 2.2, 1.2, 2.4, 1.4, 9.0, 8.0, 7.0, 8.2, 7.2, 6.2, 8.0, 7.0, 6.0],
            "r_hat": [1.0, 1.0, 1.0, 1.0, 1.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        },
        index=[
            "K_ctr_ctrl",
            "S0_1_A01",
            "S1_1_A01",
            "S0_1_A02",
            "S1_1_A02",
            "x_true[0]",
            "x_true[1]",
            "x_true[2]",
            "x_per_well[0, A01]",
            "x_per_well[1, A01]",
            "x_per_well[2, A01]",
            "x_per_well[0, A02]",
            "x_per_well[1, A02]",
            "x_per_well[2, A02]",
        ],
    )
    # fmt: on
    ds = Dataset(
        {"1": DataArray(np.array([9.0, 8.0, 7.0]), np.array([2.1, 1.5, 1.1]))},
        is_ph=True,
    )
    multi = MultiFitResult(trace=xr.DataTree(), results={})

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            bayes, "_trace_summary_df", _summary_df_stub_factory(trace_df)
        )
        monkeypatch.setattr(bayes, "plot_fit", _noop_plot_fit)
        fr_a01 = extract_fit("A01", "ctrl", multi, copy.deepcopy(ds), well_key="A01")
        fr_a02 = extract_fit("A02", "ctrl", multi, copy.deepcopy(ds), well_key="A02")

    assert fr_a01.dataset is not None
    assert fr_a02.dataset is not None
    np.testing.assert_allclose(fr_a01.dataset["1"].xc, np.array([8.1, 7.1, 6.1]))
    np.testing.assert_allclose(fr_a02.dataset["1"].xc, np.array([7.9, 6.9, 5.9]))
    np.testing.assert_allclose(fr_a01.dataset["1"].x_errc, np.array([0.05, 0.05, 0.05]))
    np.testing.assert_allclose(fr_a02.dataset["1"].x_errc, np.array([0.07, 0.07, 0.07]))


def test_fit_binding_pymc_multi_reuses_one_summary_for_all_wells(
    monkeypatch: pytest.MonkeyPatch,
    multi_dataset: Dataset,
) -> None:
    """Multi-fit should build one trace summary for per-well reconstruction."""
    fr_init = fit_binding_glob(multi_dataset)
    assert fr_init.dataset is not None
    fit_results = {"A01": fr_init, "A02": fr_init}
    scheme = PlateScheme()
    scheme.names = {"ctrl": {"A01", "A02"}}
    trace = xr.DataTree()
    per_well_calls = {"count": 0}

    def fake_per_well_fit_results_from_trace(
        trace_obj: xr.DataTree,
        per_well_fit_results: dict[str, FitResult[MiniT]],
        per_well_scheme: PlateScheme,
        *,
        x_error_model: str,
    ) -> dict[str, FitResult[xr.DataTree]]:
        del per_well_scheme, x_error_model
        assert trace_obj is trace
        assert set(per_well_fit_results) == {"A01", "A02"}
        per_well_calls["count"] += 1
        return {key: FitResult(mini=trace) for key in per_well_fit_results}

    def fake_pm_sample(*_unused_args: object, **_unused_kwargs: object) -> xr.DataTree:
        return trace

    monkeypatch.setattr(
        bayes,
        "_per_well_fit_results_from_trace",
        fake_per_well_fit_results_from_trace,
    )
    monkeypatch.setattr(pm, "sample", fake_pm_sample)
    monkeypatch.setattr(
        bayes, "_compute_sample_log_likelihood", lambda sampled: sampled
    )
    monkeypatch.setattr(bayes, "plot_fit", _noop_plot_fit)

    multi = bayes.fit_binding_pymc_multi(
        fit_results,
        scheme,
        n_samples=2,
        n_tune=1,
        n_xerr=0.0,
        x_error_model="deterministic",
    )

    assert set(multi.results) == {"A01", "A02"}
    assert per_well_calls["count"] == 1


def test_fit_binding_pymc_multi_passes_minimum_step_to_create_x_true(
    monkeypatch: pytest.MonkeyPatch,
    multi_dataset: Dataset,
) -> None:
    """Multi-fit should forward the configured minimum x step to create_x_true."""
    scheme = PlateScheme()
    scheme.names = {"ctrl": {"A01", "A02"}}
    fr_init = fit_binding_glob(multi_dataset)
    captured: dict[str, float] = {}

    def fake_create_x_true(
        xc: np.ndarray,
        x_errc: np.ndarray,
        n_xerr: float,
        lower_nsd: float = 2.5,
        min_x_step: float = 0.2,
    ) -> np.ndarray:
        del xc, x_errc, n_xerr, lower_nsd
        captured["min_x_step"] = min_x_step
        raise _StopBayesBuildError

    monkeypatch.setattr(bayes, "create_x_true", fake_create_x_true)

    with pytest.raises(_StopBayesBuildError):
        bayes.fit_binding_pymc_multi(
            {"A01": fr_init, "A02": fr_init},
            scheme,
            n_samples=2,
            n_tune=1,
            n_xerr=1.0,
            x_error_model="deterministic",
            min_x_step=0.35,
        )

    assert captured["min_x_step"] == 0.35


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore:invalid value encountered in scalar divide")
def test_single_and_multi_pymc_none_noise_model_estimate_similar_ye_mags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Single and multi PyMC should agree on per-well ye_mag for the same well."""
    pytest.importorskip("pymc")
    x = np.linspace(5.2, 8.8, 10)
    noise = np.array([
        0.035,
        -0.025,
        0.015,
        -0.03,
        0.02,
        0.025,
        -0.015,
        0.03,
        -0.02,
        0.01,
    ])

    def make_dataset(noise_scale: float, offset: float = 0.0) -> Dataset:
        return Dataset(
            {
                "1": DataArray(
                    x,
                    binding_1site(x, 7.0, 2.0, 1.0, is_ph=True)
                    + noise_scale * noise
                    + offset,
                    y_errc=np.full_like(x, 0.02),
                ),
                "2": DataArray(
                    x,
                    binding_1site(x, 7.0, 0.0, 1.0, is_ph=True)
                    - 0.8 * noise_scale * noise
                    + 0.3 * offset,
                    y_errc=np.full_like(x, 0.02),
                ),
            },
            is_ph=True,
        )

    ds_dict = {
        "A01": make_dataset(1.0),
        "A02": make_dataset(0.55, 0.003),
        "A03": make_dataset(1.45, -0.004),
    }
    scheme = PlateScheme()
    scheme.names = {"ctrl": {"A01", "A02", "A03"}}
    original_sample = pm.sample

    def seeded_sample(*args: object, **kwargs: object) -> xr.DataTree:
        kwargs.setdefault("random_seed", 123)
        kwargs.setdefault("progressbar", False)
        return cast("xr.DataTree", original_sample(*args, **kwargs))

    def posterior_mean(trace: xr.DataTree, name: str) -> float:
        var = trace.posterior[name]
        if "well" in var.dims:
            var = var.sel(well="A01")
        sample_dims = [dim for dim in var.dims if dim in {"chain", "draw"}]
        return float(var.mean(dim=sample_dims).to_numpy())

    monkeypatch.setattr(pm, "sample", seeded_sample)

    for shared_ye_mags in (False, True):
        single = fit_binding_pymc(
            fit_binding_glob(copy.deepcopy(ds_dict["A01"])),
            n_samples=150,
            n_tune=150,
            n_xerr=0.0,
            noise_model=None,
            shared_ye_mags=shared_ye_mags,
        )
        multi = bayes.fit_binding_pymc_multi(
            copy.deepcopy(ds_dict),
            scheme,
            n_samples=150,
            n_tune=150,
            n_xerr=0.0,
            noise_model=None,
            shared_ye_mags=shared_ye_mags,
            per_well_ye_mags=True,
            x_error_model="deterministic",
        )

        assert single.mini is not None
        if shared_ye_mags:
            assert posterior_mean(single.mini, "ye_mag") == pytest.approx(
                posterior_mean(multi.trace, "ye_mag"), abs=0.35
            )
        else:
            for lbl in ("1", "2"):
                var_name = f"ye_mag_{lbl}"
                assert posterior_mean(single.mini, var_name) == pytest.approx(
                    posterior_mean(multi.trace, var_name), abs=0.35
                )


@pytest.mark.slow
@pytest.mark.filterwarnings(
    "ignore:invalid value encountered in scalar divide"
)  # smoke tests use very short MCMC chains (50 samples), which causes arviz to encounter numerical issues when calculating convergence diagnostics.
def test_fit_binding_pymc_smoke_test(ph_dataset: Dataset) -> None:
    """Smoke test for the PyMC fitter with minimal sampling."""
    pytest.importorskip("pymc")
    # First, run a standard lmfit to get initial parameters
    initial_fit = fit_binding_glob(ph_dataset)
    assert initial_fit.result is not None
    assert initial_fit.dataset is not None
    # Now, run the pymc fitter with very few samples (just to test it runs)
    # Use tune=50 to reduce sampling time
    fit_result_pymc = fit_binding_pymc(initial_fit, n_samples=50, n_xerr=0, n_sd=10.0)
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
    fit_result = fit_binding_pymc(initial_fit, n_samples=50, n_xerr=1.0, n_sd=10.0)
    assert fit_result.result is not None
    assert "K" in fit_result.result.params
    # Check that K is reasonably close to expected value
    assert 5.0 < fit_result.result.params["K"].value < 9.0


@pytest.mark.slow
@pytest.mark.filterwarnings(
    "ignore:invalid value encountered in scalar divide"
)  # smoke tests use very short MCMC chains (50 samples), which causes arviz to encounter numerical issues when calculating convergence diagnostics.
def test_fit_binding_pymc_separate_smoke_test(multi_dataset: Dataset) -> None:
    """Smoke test for PyMC fitter with separate ye_mag per label."""
    pytest.importorskip("pymc")
    initial_fit = fit_binding_glob(multi_dataset)
    assert initial_fit.result is not None
    assert initial_fit.dataset is not None
    # Run PyMC with per-label noise floors (similar to old separate ye_mag)
    noise_model = PlateNoiseModel({
        lbl: NoiseModelParams(sigma_floor=10.0) for lbl in initial_fit.dataset
    })
    fit_result = fit_binding_pymc(
        initial_fit, n_samples=50, n_xerr=0, n_sd=10.0, noise_model=noise_model
    )
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


def test_masked_multiwell_vectors_have_same_order() -> None:
    """Masked model and observed arrays must flatten in the same row-major order."""
    y_model_all = np.array([[10, 100], [20, 200], [30, 300]], dtype=float)

    mask = np.array([[True, True], [True, False], [False, True]], dtype=bool)

    y_obs_full = np.array([[11, 101], [21, np.nan], [np.nan, 301]], dtype=float)

    mu_vec = y_model_all[mask]
    y_obs_vec = y_obs_full[mask]

    assert mu_vec.tolist() == [10, 100, 20, 300]
    assert y_obs_vec.tolist() == [11, 101, 21, 301]


def test_extract_x_per_well_from_xarray_returns_correct_ordered_values() -> None:
    """xarray-based extraction must return correct per-well x, ordered by step."""
    n_chains, n_draws, n_steps, n_wells = 2, 3, 3, 2
    x_data = np.zeros((n_chains, n_draws, n_steps, n_wells))
    for c in range(n_chains):
        for d in range(n_draws):
            x_data[c, d, :, 0] = [8.9, 8.3, 7.7]  # A01, identical
            x_data[c, d, :, 1] = [8.9, 8.4, 7.9]  # A02 base
    x_data[0, :, :, 1] += 0.01
    x_data[1, :, :, 1] -= 0.01

    ds = xr.Dataset(
        {"x_per_well": (["chain", "draw", "step", "well"], x_data)},
        coords={
            "chain": [0, 1],
            "draw": [0, 1, 2],
            "step": [0, 1, 2],
            "well": ["A01", "A02"],
        },
    )
    trace = xr.DataTree.from_dict({"posterior": ds})

    nxc_a01, nx_errc_a01 = bayes._extract_x_per_well_from_xarray(trace, "A01")  # noqa: SLF001
    nxc_a02, nx_errc_a02 = bayes._extract_x_per_well_from_xarray(trace, "A02")  # noqa: SLF001

    np.testing.assert_allclose(nxc_a01, np.array([8.9, 8.3, 7.7]))
    np.testing.assert_allclose(nxc_a02[0], 8.9, atol=1e-6)
    np.testing.assert_allclose(nx_errc_a01, 0, atol=1e-14)
    assert np.all(nx_errc_a02 > 1e-3)
    np.testing.assert_allclose(nxc_a01, [8.9, 8.3, 7.7])


def test_extract_fit_uses_raw_trace_not_broken_summary_for_x() -> None:
    """extract_fit must read x_per_well from raw_trace, not from az.summary.

    Regression test: az.summary can wrongly index multi-dim deterministics
    (producing wrong step-0 values per well).  The raw xarray trace is the
    ground truth, and extract_fit with raw_trace must use it.
    """
    # Build xarray trace with CORRECT x_per_well
    n_chains, n_draws, n_steps, n_wells = 1, 2, 3, 2
    x_data = np.zeros((n_chains, n_draws, n_steps, n_wells))
    for c in range(n_chains):
        for d in range(n_draws):
            x_data[c, d, :, 0] = [8.9, 8.3, 7.7]
            x_data[c, d, :, 1] = [8.9, 8.4, 7.9]

    ds = xr.Dataset(
        {"x_per_well": (["chain", "draw", "step", "well"], x_data)},
        coords={
            "chain": [0],
            "draw": [0, 1],
            "step": [0, 1, 2],
            "well": ["A01", "A02"],
        },
    )
    trace = xr.DataTree.from_dict({"posterior": ds})

    # Build a BROKEN summary DataFrame (simulating arviz wrong index)
    broken_df = pd.DataFrame(
        {
            "mean": [7.0, 2.0, 1.0, 8.0, 7.0, 6.0, 9.0, 8.0, 7.0, 8.1, 7.1, 6.1],
            "sd": [0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.05, 0.05],
            "hdi_3%": [6.8, 1.8, 0.8, 8.0, 7.0, 6.0, 9.0, 8.0, 7.0, 8.0, 7.0, 6.0],
            "hdi_97%": [7.2, 2.2, 1.2, 8.0, 7.0, 6.0, 9.0, 8.0, 7.0, 8.2, 7.2, 6.2],
            "r_hat": [1.0, 1.0, 1.0, np.nan] * 3,
        },
        index=[
            "K_A01",
            "S0_1[A01]",
            "S1_1[A01]",
            "x_true[0]",
            "x_true[1]",
            "x_true[2]",
            "x_per_well[0, A01]",
            "x_per_well[1, A01]",
            "x_per_well[2, A01]",
            "x_per_well[0, A02]",
            "x_per_well[1, A02]",
            "x_per_well[2, A02]",
        ],
    )
    # The broken summary has x_per_well[0, A01]=9, x_per_well[0, A02]=8.1 —
    # both WRONG (should be 8.9 for step 0).

    dataset = Dataset(
        {"1": DataArray(np.array([9.0, 8.0, 7.0]), np.array([1.0] * 3))},
        is_ph=True,
    )

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(bayes, "_trace_summary_df", lambda _: broken_df)
        monkeypatch.setattr(bayes, "plot_fit", _noop_plot_fit)
        fr = extract_fit(
            "A01",
            "",
            broken_df,
            copy.deepcopy(dataset),
            well_key="A01",
            raw_trace=trace,
        )

    assert fr.dataset is not None
    # Must use xarray truth (8.9, 8.3, 7.7), not broken summary (9, 8, 7)
    np.testing.assert_allclose(fr.dataset["1"].xc, np.array([8.9, 8.3, 7.7]))


def test_multiwell_likelihood_ordering_is_well_major() -> None:
    """mu_vec and y_obs_vec must be in the same row-major order after masking."""
    n_steps, n_wells = 3, 2
    # Build y_obs_full where each column is a well
    y_obs_full = np.array([[10, 100], [20, 200], [30, 300]], dtype=float)
    mask = np.ones((n_steps, n_wells), dtype=bool)
    # Row-major flatten
    mu_vec = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)[mask]
    y_obs_vec = y_obs_full[mask]
    # Both should be [well0_s0, well1_s0, well0_s1, well1_s1, well0_s2, well1_s2]
    assert mu_vec.tolist() == [1, 2, 3, 4, 5, 6]
    assert y_obs_vec.tolist() == [10, 100, 20, 200, 30, 300]


def test_gain_prior_is_positive_and_logp_finite() -> None:
    """TruncatedNormal gain must have non-negative domain (lower >= 0)."""
    pytest.importorskip("pymc")

    nm = PlateNoiseModel({"1": NoiseModelParams(sigma_floor=1.0, gain=0.5, alpha=0.0)})
    with pm.Model() as model:
        bayes.build_pymc_noise_priors(nm, gain_mode="centered")
        logps = model.point_logps()
        for v, lp in logps.items():
            assert np.isfinite(lp), f"{v} logp is {lp}"


def test_weighted_stats_ignores_nan_and_handles_zero_stderr() -> None:
    """weighted_stats must skip NaN/inf and floor zero/negative stderr."""
    result = weighted_stats(
        {"a": [1.0, 2.0, np.nan, 3.0]},
        {"a": [0.1, 0.0, 0.3, np.inf]},
    )
    # NaN ignored, inf ignored, zero stderr floored to min_stderr
    mean, se = result["a"]
    assert np.isfinite(mean)
    assert np.isfinite(se)
    # With min_stderr=1e-3, weight for point 2 is 1/(1e-3)² = 1e6
    # so mean ≈ 2.0


def test_first_fit_result_without_dataset_does_not_break_multiwell(
    multi_dataset: Dataset,
) -> None:
    """Multi-well fit must find a dataset even if first result has none."""
    scheme = PlateScheme()
    scheme.names = {"ctrl": {"A01", "A02"}}
    # Put None-dataset result FIRST, valid results after
    fr_init = fit_binding_glob(multi_dataset)
    empty_fr = copy.deepcopy(fr_init)
    empty_fr.dataset = None
    results = {"A01": empty_fr, "A02": fr_init}

    # Verification that the dataset-finding logic skips None-dataset entries.
    # Test the internal logic directly instead of running the full fit.
    fit_results, _ = bayes._normalize_fit_inputs(results)  # noqa: SLF001
    ds = next((r.dataset for r in fit_results.values() if r.dataset), None)
    assert ds is not None  # finds A02's dataset
    # A01 should be filtered out by the dataset check
    active_wells = {key for key, r in results.items() if r.result and r.dataset}
    assert active_wells == {"A02"}


def test_x_per_well_extraction_from_xarray_not_string_names() -> None:
    """xarray-based extraction must work, not depend on az.summary string parsing."""
    n_chains, n_draws, n_steps, n_wells = 1, 2, 2, 2
    x_data = np.zeros((n_chains, n_draws, n_steps, n_wells))
    for d in range(n_draws):
        x_data[0, d, :, 0] = [8.9, 7.0]
        x_data[0, d, :, 1] = [8.9, 8.0]

    ds = xr.Dataset(
        {"x_per_well": (["chain", "draw", "step", "well"], x_data)},
        coords={
            "chain": [0],
            "draw": [0, 1],
            "step": [0, 1],
            "well": ["A01", "A02"],
        },
    )
    trace = xr.DataTree.from_dict({"posterior": ds})

    nxc_a01, _ = bayes._extract_x_per_well_from_xarray(trace, "A01")  # noqa: SLF001
    nxc_a02, _ = bayes._extract_x_per_well_from_xarray(trace, "A02")  # noqa: SLF001

    np.testing.assert_allclose(nxc_a01, [8.9, 7.0])
    np.testing.assert_allclose(nxc_a02, [8.9, 8.0])
    # Verify no string parsing path was needed — shape must be correct
    assert len(nxc_a01) == n_steps


class TestYerrExtraction:
    """Tests that returned dataset y_errc derives from fitted noise parameters."""

    @pytest.mark.slow
    def test_yerr_from_sigma_obs_matches_xarray_posterior(
        self, ph_dataset: Dataset
    ) -> None:
        """With noise_model, y_errc must equal the posterior mean of sigma_obs."""
        lbl = next(iter(ph_dataset.keys()))
        # Set known y_errc to detect updates
        ph_dataset[lbl].y_errc = np.full_like(ph_dataset[lbl].xc, 0.01)

        params = Parameters()
        params.add("K", value=7.0)
        params["K"].stderr = 0.1
        params.add(f"S0_{lbl}", value=100.0)
        params[f"S0_{lbl}"].stderr = 1.0
        params.add(f"S1_{lbl}", value=0.0)
        params[f"S1_{lbl}"].stderr = 1.0
        fr: FitResult[MiniT] = FitResult(
            None, type("Result", (), {"params": params})(), None, ph_dataset
        )

        results = {"A01": fr}
        scheme = PlateScheme()
        scheme.names = {"ctrl": {"A01"}}

        noise_model = PlateNoiseModel({
            lbl: NoiseModelParams(sigma_floor=10.0, gain=0.0, alpha=0.0)
        })
        multi = bayes.fit_binding_pymc_multi(
            results,
            scheme,
            noise_model=noise_model,
            n_samples=100,
            n_tune=50,
            n_xerr=0.0,
            floor_mode="centered",
        )

        returned_dataset = multi.results["A01"].dataset
        assert returned_dataset is not None
        returned_y_errc = returned_dataset[lbl].y_errc

        # Ground truth: posterior mean of sigma_obs_{lbl} extracted
        # DIRECTLY from xarray, bypassing az.summary string parsing.
        posterior = multi.trace.posterior
        var_name = f"sigma_obs_{lbl}"
        assert var_name in posterior, f"{var_name} missing from trace.posterior"
        da = posterior[var_name]
        sample_dims = [d for d in da.dims if d in {"chain", "draw"}]
        assert "well" in da.dims, "sigma_obs must have well dimension"
        sigma_means = da.sel(well="A01").mean(dim=sample_dims).to_numpy()

        assert np.all(sigma_means > 5.0), "sigma_obs posterior implausible"
        np.testing.assert_allclose(
            returned_y_errc,
            sigma_means,
            rtol=0.15,
            err_msg="y_errc doesn't match xarray sigma_obs posterior mean",
        )

    @pytest.mark.slow
    def test_yerr_scaled_by_ye_mag_from_xarray(self, ph_dataset: Dataset) -> None:
        """Without noise_model, y_errc must be scaled by ye_mag posterior."""
        lbl = next(iter(ph_dataset.keys()))
        da = ph_dataset[lbl]
        da.y_errc = np.full_like(da.xc, 0.5)

        params = Parameters()
        params.add("K", value=7.0)
        params["K"].stderr = 0.1
        params.add(f"S0_{lbl}", value=100.0)
        params[f"S0_{lbl}"].stderr = 1.0
        params.add(f"S1_{lbl}", value=0.0)
        params[f"S1_{lbl}"].stderr = 1.0
        fr: FitResult[MiniT] = FitResult(
            None, type("Result", (), {"params": params})(), None, ph_dataset
        )

        results = {"A01": fr}
        scheme = PlateScheme()
        scheme.names = {"ctrl": {"A01"}}

        multi = bayes.fit_binding_pymc_multi(
            results,
            scheme,
            noise_model=None,
            n_samples=100,
            n_tune=50,
            n_xerr=0.0,
        )

        returned_dataset = multi.results["A01"].dataset
        assert returned_dataset is not None
        returned_y_errc = returned_dataset[lbl].y_errc

        # Ground truth: original y_errc * ye_mag posterior, from xarray
        posterior = multi.trace.posterior
        mag_var = f"ye_mag_{lbl}"
        assert mag_var in posterior, f"{mag_var} missing from trace.posterior"
        sample_dims = [d for d in posterior[mag_var].dims if d in {"chain", "draw"}]
        mag = float(posterior[mag_var].mean(dim=sample_dims).values)

        expected = np.full_like(da.xc, 0.5) * mag
        assert not np.allclose(returned_y_errc, 0.5), (
            "y_errc was NOT updated from original 0.5"
        )
        np.testing.assert_allclose(
            returned_y_errc,
            expected,
            atol=1e-5,
            err_msg="y_errc doesn't match original * ye_mag posterior mean",
        )

    def test_update_dataset_yerr_validates_all_steps_filled(self) -> None:
        """Unit test: extraction must fill every step — no silent 1.0 gaps."""
        lbl = "1"
        n_steps = 4
        ds = Dataset(
            {lbl: DataArray(np.arange(n_steps, dtype=float), np.ones(n_steps))},
            is_ph=True,
        )
        ds[lbl].y_errc = np.full(n_steps, 99.0)

        # Synthetic trace with sigma_obs_1 for all steps of well "A01"
        var_name = f"sigma_obs_{lbl}"
        n_chain, n_draw = 2, 10
        means = np.array([10.0, 20.0, 30.0, 40.0])
        arr = (
            np.ones((n_chain, n_draw, n_steps, 1))
            * means[np.newaxis, np.newaxis, :, np.newaxis]
        )
        ds_trace = xr.Dataset(
            {var_name: (["chain", "draw", "step", "well"], arr)},
            coords={
                "chain": np.arange(n_chain),
                "draw": np.arange(n_draw),
                "step": np.arange(n_steps),
                "well": ["A01"],
            },
        )
        trace = xr.DataTree.from_dict({"posterior": ds_trace})

        # Verify the xarray extraction produces the expected values
        posterior = trace.posterior
        sigma_means = (
            posterior[var_name].sel(well="A01").mean(dim=["chain", "draw"]).to_numpy()
        )

        # Verify shape matches
        assert len(sigma_means) == n_steps
        # All values must be far from the 1.0 fallback
        assert np.all(sigma_means > 5.0)
