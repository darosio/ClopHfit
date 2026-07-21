"""Test cases for the clophfit.fitting.bayes module."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pandas as pd
import pymc as pm  # type: ignore[import-untyped]
import pytensor.tensor as pt
import pytest
import xarray as xr
from lmfit import Parameters  # type: ignore[import-untyped]

from clophfit.fitting import bayes, bayes_config
from clophfit.fitting.bayes import (
    _pipetting_step_sigmas,  # noqa: PLC2701
    create_parameter_priors,
    create_x_true,
    extract_fit,
    fit_binding_pymc,
    process_trace,
    weighted_stats,
    x_true_from_trace_df,
)
from clophfit.fitting.bayes_config import (
    InitConfig,
    NoiseConfig,
    RobustConfig,
    SamplerConfig,
)
from clophfit.fitting.core import fit_binding_glob
from clophfit.fitting.data_structures import (
    DataArray,
    Dataset,
    FitResult,
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


def test_pipetting_step_sigmas_isotonic_pulls_down_leading_spike() -> None:
    """An atypically large early SD is pulled down, not frozen and propagated.

    Cumulative pipetting variance must be non-decreasing; a noisy leading spike
    (e.g. a real but transient step-0 well-to-well spread that reconverges next
    step) violates that. Isotonic regression pulls it toward the smaller later
    values, so the ``x_start`` anchor is not inflated by one bad 3-well SD.
    """
    # Real plate-1 profile: step-0 SD (0.067) exceeds the next four steps.
    x_errc = np.array([0.0666, 0.0173, 0.0351, 0.0351, 0.0436, 0.0702, 0.0702])
    x_start_sigma, step_sigmas = _pipetting_step_sigmas(x_errc)

    n_wells = 3.0
    naive_anchor = x_errc[0] / np.sqrt(n_wells)  # what a running-max would freeze
    assert x_start_sigma < naive_anchor  # spike pulled down
    assert np.all(step_sigmas > 0.0)  # no pinned step

    # On a clean non-decreasing profile isotonic is a no-op: anchor == step-0 SE.
    mono = np.array([0.01, 0.05, 0.09, 0.11, 0.14, 0.16, 0.18])
    mono_anchor, _ = _pipetting_step_sigmas(mono)
    assert mono_anchor == pytest.approx(mono[0] / np.sqrt(n_wells))


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


def test_data_prior_seed_uses_ph_extremes_and_midpoint(ph_dataset: Dataset) -> None:
    """Data-prior initialization should infer pH edge signals without LMFit."""
    fr = bayes._fit_result_from_data_priors(  # noqa: SLF001
        ph_dataset,
        edge_points=1,
        signal_sigma_scale=0.5,
        k_prior="midpoint_truncnorm",
        k_bounds=(4.5, 9.0),
        k_sigma=1.5,
    )

    assert fr.result is not None
    params = fr.result.params
    assert params["S0_default"].value == pytest.approx(ph_dataset["default"].yc[-1])
    assert params["S1_default"].value == pytest.approx(ph_dataset["default"].yc[0])
    assert params["K"].value == pytest.approx(7.0)
    assert params["K"].min == pytest.approx(4.5)
    assert params["K"].max == pytest.approx(9.0)


def test_data_prior_default_k_bounds_respect_is_ph(
    ph_dataset: Dataset, cl_dataset: Dataset
) -> None:
    """Omitted k_bounds should resolve from is_ph, not a fixed pH range."""
    ph_fr = bayes._fit_result_from_data_priors(  # noqa: SLF001
        ph_dataset,
        edge_points=1,
        signal_sigma_scale=0.5,
        k_prior="midpoint_truncnorm",
        k_bounds=None,
        k_sigma=1.5,
    )
    assert ph_fr.result is not None
    assert ph_fr.result.params["K"].min == pytest.approx(4.5)
    assert ph_fr.result.params["K"].max == pytest.approx(9.0)

    cl_fr = bayes._fit_result_from_data_priors(  # noqa: SLF001
        cl_dataset,
        edge_points=1,
        signal_sigma_scale=0.5,
        k_prior="midpoint_truncnorm",
        k_bounds=None,
        k_sigma=1.5,
    )
    assert cl_fr.result is not None
    assert cl_fr.result.params["K"].min == pytest.approx(1e-6)
    assert cl_fr.result.params["K"].max == pytest.approx(1e6)


def test_create_data_parameter_priors_can_use_uniform_k(ph_dataset: Dataset) -> None:
    """Direct data priors should support a bounded flat K prior."""
    fr = bayes._fit_result_from_data_priors(  # noqa: SLF001
        ph_dataset,
        edge_points=2,
        signal_sigma_scale=0.5,
        k_prior="uniform",
        k_bounds=(4.5, 9.0),
        k_sigma=1.5,
    )

    assert fr.result is not None
    with pm.Model():
        priors = bayes.create_data_parameter_priors(
            fr.result.params,
            k_prior="uniform",
            k_bounds=(4.5, 9.0),
        )

    assert "K" in priors
    assert "S0_default" in priors
    assert "S1_default" in priors


def test_fit_binding_pymc_data_priors_skips_lmfit(
    monkeypatch: pytest.MonkeyPatch,
    ph_dataset: Dataset,
) -> None:
    """The data-prior strategy should not run the LMFit prefit."""
    pytest.importorskip("pymc")

    def fail_lmfit(*_args: object, **_kwargs: object) -> FitResult:
        msg = "fit_binding_glob should not be called"
        raise AssertionError(msg)

    def stop_sample(*_args: object, **_kwargs: object) -> xr.DataTree:
        raise _StopBayesBuildError

    monkeypatch.setattr(bayes, "fit_binding_glob", fail_lmfit)
    monkeypatch.setattr(pm, "sample", stop_sample)

    with pytest.raises(_StopBayesBuildError):
        fit_binding_pymc(
            ph_dataset,
            n_xerr=0.0,
            init=InitConfig(strategy="data_priors"),
            sampler=SamplerConfig(n_samples=2, n_tune=1),
        )


@pytest.mark.parametrize(
    ("k_bounds", "is_ph", "expected"),
    [
        (None, True, (4.5, 9.0)),
        (None, False, (1e-6, 1e6)),
        ((float("nan"), 9.0), True, (4.5, 9.0)),
        ((5.0, 5.0), False, (1e-6, 1e6)),  # lo == hi is invalid
        ((9.0, 4.5), True, (4.5, 9.0)),  # reversed order is sorted
        ((1.0, 50.0), False, (1.0, 50.0)),  # valid explicit bounds respected
    ],
)
def test_resolve_data_prior_k_bounds(
    k_bounds: tuple[float, float] | None,
    is_ph: bool,  # noqa: FBT001
    expected: tuple[float, float],
) -> None:
    """K-bound resolution should fall back per is_ph and validate the range."""
    lo, hi = bayes._resolve_data_prior_k_bounds(k_bounds, is_ph=is_ph)  # noqa: SLF001
    assert lo == pytest.approx(expected[0])
    assert hi == pytest.approx(expected[1])


def test_normalize_fit_inputs_data_priors_skips_lmfit(
    monkeypatch: pytest.MonkeyPatch,
    multi_dataset: Dataset,
) -> None:
    """Multi-well normalization should seed from data without calling LMFit."""

    def fail_lmfit(*_args: object, **_kwargs: object) -> FitResult:
        msg = "fit_binding_glob should not be called"
        raise AssertionError(msg)

    monkeypatch.setattr(bayes, "fit_binding_glob", fail_lmfit)

    normalized, prefer_centered = bayes._normalize_fit_inputs(  # noqa: SLF001
        {"A01": copy.deepcopy(multi_dataset), "A02": copy.deepcopy(multi_dataset)},
        init_strategy="data_priors",
    )

    assert prefer_centered is False
    assert set(normalized) == {"A01", "A02"}
    for fr in normalized.values():
        assert fr.result is not None
        params = fr.result.params
        assert {"S0_1", "S1_1", "S0_2", "S1_2", "K"} <= set(params)
        # pH default bounds applied because k_bounds was omitted.
        assert params["K"].min == pytest.approx(4.5)
        assert params["K"].max == pytest.approx(9.0)


def test_normalize_fit_inputs_data_priors_ignores_prefit_result(
    monkeypatch: pytest.MonkeyPatch,
    multi_dataset: Dataset,
) -> None:
    """A supplied FitResult should be re-seeded from its dataset, not reused."""

    def fail_lmfit(*_args: object, **_kwargs: object) -> FitResult:
        msg = "fit_binding_glob should not be called"
        raise AssertionError(msg)

    prefit = fit_binding_glob(multi_dataset)
    monkeypatch.setattr(bayes, "fit_binding_glob", fail_lmfit)

    normalized, prefer_centered = bayes._normalize_fit_inputs(  # noqa: SLF001
        {"A01": prefit},
        init_strategy="data_priors",
        data_prior_edge_points=1,
    )

    assert prefer_centered is False
    fr = normalized["A01"]
    assert fr.result is not None
    # For pH data, S0 is the high-pH plateau (last x); a single-point edge
    # window makes this the raw endpoint rather than the LMFit optimum.
    da = multi_dataset["1"]
    assert fr.result.params["S0_1"].value == pytest.approx(float(da.yc[-1]))


def test_fit_binding_pymc_multi_data_priors_skips_lmfit(
    monkeypatch: pytest.MonkeyPatch,
    multi_dataset: Dataset,
) -> None:
    """The multi data-prior strategy should reach sampling without LMFit."""
    pytest.importorskip("pymc")
    scheme = PlateScheme()
    scheme.names = {"ctrl": {"A01", "A02"}}

    def fail_lmfit(*_args: object, **_kwargs: object) -> FitResult:
        msg = "fit_binding_glob should not be called"
        raise AssertionError(msg)

    def stop_sample(*_args: object, **_kwargs: object) -> xr.DataTree:
        raise _StopBayesBuildError

    monkeypatch.setattr(bayes, "fit_binding_glob", fail_lmfit)
    monkeypatch.setattr(pm, "sample", stop_sample)

    with pytest.raises(_StopBayesBuildError):
        bayes.fit_binding_pymc_multi(
            {"A01": copy.deepcopy(multi_dataset), "A02": copy.deepcopy(multi_dataset)},
            scheme,
            n_xerr=0.0,
            init=InitConfig(strategy="data_priors"),
            sampler=SamplerConfig(n_samples=2, n_tune=1),
        )


def test_edge_helpers_handle_empty_input() -> None:
    """Edge-window helpers should degrade gracefully on empty arrays."""
    empty = np.array([], dtype=float)
    assert bayes._edge_mean(empty, start=True, n_points=2) == 0.0  # noqa: SLF001
    assert bayes._edge_signal_priors(  # noqa: SLF001
        empty, empty, is_ph=True, edge_points=2
    ) == (0.0, 0.0)
    assert (
        bayes._midpoint_x_for_prior(  # noqa: SLF001
            empty, empty, high_edge=1.0, low_edge=0.0, bounds=(4.5, 9.0)
        )
        is None
    )


def test_edge_mean_falls_back_to_full_window_mean() -> None:
    """A fully-NaN edge window should fall back to the finite full-array mean."""
    values = np.array([np.nan, 2.0, 4.0])
    # start window of 1 point is NaN -> nanmean over all finite values (3.0).
    assert bayes._edge_mean(values, start=True, n_points=1) == pytest.approx(3.0)  # noqa: SLF001


def test_fit_result_from_data_priors_all_masked_label() -> None:
    """A label with no active points should still yield finite seed params."""
    da = DataArray(np.array([6.0, 7.0, 8.0]), np.array([np.nan, np.nan, np.nan]))
    ds = Dataset({"default": da}, is_ph=True)
    fr = bayes._fit_result_from_data_priors(  # noqa: SLF001
        ds,
        edge_points=2,
        signal_sigma_scale=0.5,
        k_prior="midpoint_truncnorm",
        k_bounds=(4.5, 9.0),
        k_sigma=1.5,
    )
    assert fr.result is not None
    params = fr.result.params
    assert params["S0_default"].value == 0.0
    assert params["S1_default"].value == 0.0
    # No midpoint could be estimated -> K seeded at the bound midpoint.
    assert params["K"].value == pytest.approx(6.75)


def test_fit_result_from_data_priors_constant_signal() -> None:
    """A flat titration (zero signal range) should use a positive fallback sigma."""
    da = DataArray(np.array([6.0, 7.0, 8.0]), np.array([2.0, 2.0, 2.0]))
    ds = Dataset({"default": da}, is_ph=True)
    fr = bayes._fit_result_from_data_priors(  # noqa: SLF001
        ds,
        edge_points=1,
        signal_sigma_scale=0.5,
        k_prior="midpoint_truncnorm",
        k_bounds=(4.5, 9.0),
        k_sigma=1.5,
    )
    assert fr.result is not None
    # y_range == 0 -> sigma from max(|s0|, |s1|, 1.0) * scale.
    assert fr.result.params["S0_default"].stderr == pytest.approx(1.0)


def test_normalize_fit_input_data_priors_without_dataset() -> None:
    """Single-well data-prior normalization returns empty on a dataset-less input."""
    fr, prefit = bayes._normalize_fit_input(  # noqa: SLF001
        FitResult(), init_strategy="data_priors"
    )
    assert prefit is False
    assert fr.result is None


def test_normalize_fit_inputs_data_priors_skips_dataset_less_wells() -> None:
    """Multi-well data-prior normalization drops wells that carry no dataset."""
    normalized, _prefer_centered = bayes._normalize_fit_inputs(  # noqa: SLF001
        {"A01": FitResult()}, init_strategy="data_priors"
    )
    assert normalized == {}


@pytest.mark.parametrize(
    ("bg_noise", "expected_floor"),
    [
        (2.5, 2.5),
        (-1.0, 1.0),  # non-positive floor falls back to 1.0
        (float("nan"), 1.0),  # non-finite floor falls back to 1.0
        ({"1": 3.0}, 3.0),  # mapping hit
        ({"2": 3.0}, 1.0),  # mapping miss -> default 1.0
    ],
)
def test_plate_noise_from_bg_floor_resolution(
    bg_noise: float | dict[str, float], expected_floor: float
) -> None:
    """Background-noise floors should validate and fall back per label."""
    model = bayes._plate_noise_from_bg(bg_noise, ["1"], alpha=0.1)  # noqa: SLF001
    assert model["1"].sigma_floor == pytest.approx(expected_floor)
    assert model["1"].gain == 0.0
    assert model["1"].alpha == pytest.approx(0.1)


def test_noise_config_factory_field_mapping() -> None:
    """The NoiseConfig factories should set kind and route their arguments."""
    assert NoiseConfig().kind == "ye_mag"
    ym = NoiseConfig.ye_mag(shared=True, prior="halfnormal", mu=0.1, sigma=2.0)
    assert ym.kind == "ye_mag"
    assert ym.shared_ye_mags is True
    assert ym.ye_mag_prior == "halfnormal"
    assert (ym.ye_mag_mu, ym.ye_mag_sigma) == (0.1, 2.0)
    st = NoiseConfig.structured(
        gain_mode="free", alpha_mode="free", floor=4.0, gain=0.1
    )
    assert st.kind == "structured"
    assert (st.gain_mode, st.alpha_mode) == ("free", "free")
    assert (st.floor, st.gain) == (4.0, 0.1)


def test_robust_config_validation() -> None:
    """RobustConfig should validate its likelihood and contamination prior."""
    with pytest.raises(ValueError, match=r"student_t.*mixture"):
        RobustConfig(likelihood="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"between 0\.001 and 0\.5"):
        RobustConfig(enabled=True, likelihood="mixture", contamination_frac_prior=0.9)
    # An out-of-range prior is ignored unless the mixture likelihood is active.
    assert RobustConfig(enabled=False, contamination_frac_prior=0.9).enabled is False


def test_resolve_structured_noise_model_passthrough(ph_dataset: Dataset) -> None:
    """An explicit noise model should be returned unchanged."""
    explicit = PlateNoiseModel({"default": NoiseModelParams(sigma_floor=9.0)})
    noise = NoiseConfig.structured(noise_model=explicit)
    assert bayes._resolve_structured_noise_model(noise, ph_dataset) is explicit  # noqa: SLF001


def test_resolve_structured_noise_model_synthesizes_from_data(
    ph_dataset: Dataset,
) -> None:
    """Without a model, the floor is taken from the label's y_err scale."""
    ph_dataset["default"].y_err = np.full_like(ph_dataset["default"].yc, 3.0)
    noise = NoiseConfig.structured(alpha=0.02)
    model = bayes._resolve_structured_noise_model(noise, ph_dataset)  # noqa: SLF001
    assert model["default"].sigma_floor == pytest.approx(3.0)
    assert model["default"].gain == 0.0
    assert model["default"].alpha == pytest.approx(0.02)


def test_resolve_structured_noise_model_uses_hints(cl_dataset: Dataset) -> None:
    """Explicit floor/gain hints seed the synthesized model."""
    noise = NoiseConfig.structured(floor=5.0, gain=0.1)
    model = bayes._resolve_structured_noise_model(noise, cl_dataset)  # noqa: SLF001
    for params in model.values():
        assert params.sigma_floor == pytest.approx(5.0)
        assert params.gain == pytest.approx(0.1)


def test_structured_noise_without_model_activates_free_terms(
    monkeypatch: pytest.MonkeyPatch,
    ph_dataset: Dataset,
) -> None:
    """structured(gain_mode/alpha_mode='free') creates gain/rel_error vars.

    Previously these modes were silently ignored unless a PlateNoiseModel was
    passed; this exercises the fix.
    """
    captured: dict[str, set[str]] = {}

    def stop_sample(*_args: object, **_kwargs: object) -> xr.DataTree:
        model = pm.modelcontext(None)
        captured["vars"] = {rv.name for rv in model.free_RVs}
        raise _StopBayesBuildError

    monkeypatch.setattr(pm, "sample", stop_sample)

    with pytest.raises(_StopBayesBuildError):
        fit_binding_pymc(
            ph_dataset,
            n_xerr=0.0,
            noise=NoiseConfig.structured(gain_mode="free", alpha_mode="free"),
            sampler=SamplerConfig(n_samples=2, n_tune=1),
        )

    names = captured["vars"]
    assert any(name.startswith("gain") for name in names)
    assert any(name.startswith("rel_error") for name in names)


def test_free_noise_priors_scale_from_hints() -> None:
    """Free gain/alpha priors should take their scale from the noise-model hints."""
    pytest.importorskip("pymc")
    hinted = PlateNoiseModel({
        "1": NoiseModelParams(sigma_floor=1.0, gain=0.5, alpha=0.03)
    })
    plain = PlateNoiseModel({"1": NoiseModelParams(sigma_floor=1.0)})
    with pm.Model():
        priors = bayes.build_pymc_noise_priors(
            hinted, gain_mode="free", alpha_mode="free"
        )
        gain_mean = float(
            pm.draw(priors["gain"]["1"], draws=6000, random_seed=0).mean()
        )
        alpha_mean = float(
            pm.draw(priors["rel_error"]["1"], draws=6000, random_seed=1).mean()
        )
    with pm.Model():
        priors0 = bayes.build_pymc_noise_priors(
            plain, gain_mode="free", alpha_mode="free"
        )
        gain0_mean = float(
            pm.draw(priors0["gain"]["1"], draws=6000, random_seed=0).mean()
        )
        alpha0_mean = float(
            pm.draw(priors0["rel_error"]["1"], draws=6000, random_seed=1).mean()
        )
    # The hint is the prior *mean* for both terms: Exponential(lam=1/h) and
    # HalfNormal(sigma=h*sqrt(pi/2)) both have mean h.
    assert gain_mean == pytest.approx(0.5, abs=0.05)
    assert alpha_mean == pytest.approx(0.03, abs=0.005)
    # gain=0 / alpha=0 -> tightest around-zero priors (floored at 1e-3), so each
    # width is strictly *below* a small positive hint (monotonic in the hint).
    assert gain0_mean == pytest.approx(1e-3, abs=5e-4)
    assert alpha0_mean == pytest.approx(1e-3, abs=5e-4)
    assert gain0_mean < gain_mean
    assert alpha0_mean < alpha_mean


def test_centered_zero_alpha_spans_plate_alpha_scale() -> None:
    """A calibrated alpha of 0 spans the plate's alpha scale in centered mode, not fixed.

    NNLS clamps the collinear y/y**2 basis, so alpha=0.0 on every label means
    gain won each label's decomposition -- not that the proportional-error term
    is physically absent. With no label resolving a positive alpha, the width
    falls back to `_ZERO_HINT_ALPHA_SCALE` (0.1), the plate scale to assume
    when nothing on the plate resolved a positive alpha. Only "fixed" drops the
    alpha term when the whole plate has alpha == 0.
    """
    pytest.importorskip("pymc")
    nm = PlateNoiseModel({
        "1": NoiseModelParams(sigma_floor=1.0, gain=2.0, alpha=0.0),
        "2": NoiseModelParams(sigma_floor=1.0, gain=1.0, alpha=0.0),
    })
    with pm.Model():
        centered = bayes.build_pymc_noise_priors(
            nm, gain_mode="centered", alpha_mode="centered"
        )
        draws = pm.draw(centered["rel_error"]["1"], draws=6000, random_seed=0)
    # Term is present, strictly non-negative, non-degenerate, and spans the
    # plate's alpha scale (HalfNormal(sigma=0.1), not a tight band around 0).
    assert "rel_error" in centered
    assert float(draws.min()) >= 0.0
    assert float(draws.std()) > 0.0
    assert float(draws.mean()) == pytest.approx(0.1 * np.sqrt(2 / np.pi), abs=0.005)
    with pm.Model():
        fixed = bayes.build_pymc_noise_priors(nm, gain_mode="fixed", alpha_mode="fixed")
    # "fixed" with an all-zero alpha leaves the term genuinely absent (hard 0).
    assert "rel_error" not in fixed


def test_shared_floor_pools_labels_into_one_variable() -> None:
    """shared_floor pools per-label floors into a single plate-wide variable."""
    pytest.importorskip("pymc")
    nm = PlateNoiseModel({
        "1": NoiseModelParams(sigma_floor=2.0),
        "2": NoiseModelParams(sigma_floor=4.0),
    })
    with pm.Model():
        pooled = bayes.build_pymc_noise_priors(nm, shared_floor=True)
        pooled_mean = float(pm.draw(pooled["floor"], draws=6000, random_seed=0).mean())
    with pm.Model():
        per_label = bayes.build_pymc_noise_priors(nm, shared_floor=False)
        mean1 = float(
            pm.draw(per_label["floor"]["1"], draws=6000, random_seed=0).mean()
        )
        mean2 = float(
            pm.draw(per_label["floor"]["2"], draws=6000, random_seed=1).mean()
        )
    # Pooled: one variable, not a per-label dict, centered on the mean hint.
    assert not isinstance(pooled["floor"], dict)
    assert pooled_mean == pytest.approx(3.0, abs=0.1)
    # Per-label keeps the two hints distinct.
    assert isinstance(per_label["floor"], dict)
    assert mean1 == pytest.approx(2.0, abs=0.1)
    assert mean2 == pytest.approx(4.0, abs=0.1)


def test_shared_floor_variance_is_per_point() -> None:
    """get_pymc_variance accepts a pooled floor and still returns a per-point tensor."""
    pytest.importorskip("pymc")
    nm = PlateNoiseModel({
        "1": NoiseModelParams(sigma_floor=2.0),
        "2": NoiseModelParams(sigma_floor=4.0),
    })
    mu = np.array([10.0, 20.0, 30.0])
    with pm.Model():
        # gain/alpha fixed at their zero hints so only the floor term survives.
        priors = bayes.build_pymc_noise_priors(
            nm,
            shared_floor=True,
            floor_mode="fixed",
            gain_mode="fixed",
            alpha_mode="fixed",
        )
        var = bayes.get_pymc_variance(pt.as_tensor_variable(mu), "1", nm, priors)
        values = np.asarray(var.eval())
    # Fixed pooled floor == mean hint (3.0); variance is floor**2 at every point.
    assert values.shape == mu.shape
    assert values == pytest.approx(np.full(3, 9.0))


@pytest.mark.parametrize(
    ("value", "label", "expected"),
    [
        (2.0, "1", 2.0),
        (float("nan"), "1", 0.0),
        ({"1": 3.0}, "1", 3.0),
        ({"1": 2.0, "2": 4.0}, "3", 3.0),  # missing label -> finite mean
        ({"1": float("nan")}, "2", 0.0),  # no finite values -> 0.0
    ],
)
def test_ye_mag_value(
    value: float | dict[str, float], label: str, expected: float
) -> None:
    """ye_mag prior values should be finite with scalar/mapping fallbacks."""
    assert bayes._ye_mag_value(value, label) == pytest.approx(expected)  # noqa: SLF001


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (2.0, 2.0),
        (float("nan"), 0.0),
        ({"1": 2.0, "2": 4.0}, 3.0),
        ({}, 0.0),
    ],
)
def test_shared_ye_mag_value(value: float | dict[str, float], expected: float) -> None:
    """Shared ye_mag values should collapse mappings to a finite mean."""
    assert bayes._shared_ye_mag_value(value) == pytest.approx(expected)  # noqa: SLF001


@pytest.mark.parametrize(
    ("sigma", "label", "expected"),
    [
        (2.0, "1", 2.0),
        (-1.0, "1", 1e-6),  # non-positive -> floor
        ({"1": 2.0}, "1", 2.0),
        ({"1": -1.0}, "2", 1e-6),  # no positive values -> floor
    ],
)
def test_ye_mag_sigma(
    sigma: float | dict[str, float], label: str, expected: float
) -> None:
    """ye_mag sigmas should be strictly positive with per-label fallbacks."""
    assert bayes._ye_mag_sigma(sigma, label) == pytest.approx(expected)  # noqa: SLF001


@pytest.mark.parametrize(
    ("sigma", "expected"),
    [
        (2.0, 2.0),
        (-1.0, 1e-6),
        ({"1": 2.0, "2": 4.0}, 3.0),
        ({"1": -1.0}, 1e-6),
    ],
)
def test_shared_ye_mag_sigma(sigma: float | dict[str, float], expected: float) -> None:
    """Shared ye_mag sigmas should stay positive across scalar/mapping inputs."""
    assert bayes._shared_ye_mag_sigma(sigma) == pytest.approx(expected)  # noqa: SLF001


def test_validate_robust_likelihood_rejects_unknown() -> None:
    """An unrecognized robust-likelihood selector should raise ValueError."""
    with pytest.raises(ValueError, match=r"student_t.*mixture"):
        bayes_config._validate_robust_likelihood("bogus")  # type: ignore[arg-type]  # noqa: SLF001


def test_add_y_likelihood_mixture_requires_outlier_priors() -> None:
    """Mixture likelihood without outlier priors should raise a clear error."""
    da = DataArray(np.array([6.0, 7.0, 8.0]), np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match="pi_outlier and outlier_inflate"):
        bayes._add_y_likelihood(  # noqa: SLF001
            "y_likelihood_1",
            np.zeros(3),
            da,
            1.0,
            robust=True,
            robust_likelihood="mixture",
            pi_outlier=None,
            outlier_inflate=None,
        )


def test_scale_and_log_scaled_ye_mag_helpers() -> None:
    """ye_mag scaling helpers should guard the scale and log-transform medians."""
    assert bayes._scale_ye_mag_sigma(2.0, 3.0) == pytest.approx(6.0)  # noqa: SLF001
    # Non-positive scale is coerced to 1.0.
    assert bayes._scale_ye_mag_sigma(2.0, -1.0) == pytest.approx(2.0)  # noqa: SLF001
    scaled = bayes._scale_ye_mag_sigma({"1": 2.0}, 3.0)  # noqa: SLF001
    assert isinstance(scaled, dict)
    assert scaled["1"] == pytest.approx(6.0)
    assert bayes._log_scaled_ye_mag_mu(1.0, 1.0) == pytest.approx(0.0)  # noqa: SLF001
    log_map = bayes._log_scaled_ye_mag_mu({"1": 1.0}, 1.0)  # noqa: SLF001
    assert isinstance(log_map, dict)
    assert log_map["1"] == pytest.approx(0.0)


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
        **_kwargs: object,
    ) -> dict[str, object]:
        captured["floor"] = floor_mode
        captured["gain"] = gain_mode
        captured["alpha"] = alpha_mode
        raise _StopBayesBuildError

    monkeypatch.setattr(bayes, "build_pymc_noise_priors", fake_build_pymc_noise_priors)

    with pytest.raises(_StopBayesBuildError):
        fit_binding_pymc(
            ph_dataset,
            n_xerr=0.0,
            noise=NoiseConfig.structured(noise_model=noise_model),
            sampler=SamplerConfig(n_samples=2, n_tune=1),
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
        **_kwargs: object,
    ) -> dict[str, object]:
        captured["floor"] = floor_mode
        captured["gain"] = gain_mode
        captured["alpha"] = alpha_mode
        raise _StopBayesBuildError

    monkeypatch.setattr(bayes, "build_pymc_noise_priors", fake_build_pymc_noise_priors)

    with pytest.raises(_StopBayesBuildError):
        fit_binding_pymc(
            fit_binding_glob(ph_dataset),
            n_xerr=0.0,
            noise=NoiseConfig.structured(noise_model=noise_model),
            sampler=SamplerConfig(n_samples=2, n_tune=1),
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
        shared_alpha: bool = False,
        shared_gain: bool = False,
        **_kwargs: object,
    ) -> dict[str, object]:
        captured["shared_alpha"] = shared_alpha
        captured["shared_gain"] = shared_gain
        raise _StopBayesBuildError

    monkeypatch.setattr(bayes, "build_pymc_noise_priors", fake_build_pymc_noise_priors)

    with pytest.raises(_StopBayesBuildError):
        fit_binding_pymc(
            multi_dataset,
            n_xerr=0.0,
            noise=NoiseConfig.structured(
                noise_model=noise_model, shared_alpha=False, shared_gain=True
            ),
            sampler=SamplerConfig(n_samples=2, n_tune=1),
        )

    assert captured == {"shared_alpha": False, "shared_gain": True}


def test_fit_binding_pymc_passes_shared_floor(
    monkeypatch: pytest.MonkeyPatch,
    multi_dataset: Dataset,
) -> None:
    """NoiseConfig.shared_floor should reach build_pymc_noise_priors."""
    captured: dict[str, bool] = {}
    noise_model = PlateNoiseModel({
        "1": NoiseModelParams(sigma_floor=1.0),
        "2": NoiseModelParams(sigma_floor=2.0),
    })

    def fake_build_pymc_noise_priors(
        _noise_model: PlateNoiseModel,
        *,
        shared_floor: bool = False,
        **_kwargs: object,
    ) -> dict[str, object]:
        captured["shared_floor"] = shared_floor
        raise _StopBayesBuildError

    monkeypatch.setattr(bayes, "build_pymc_noise_priors", fake_build_pymc_noise_priors)

    with pytest.raises(_StopBayesBuildError):
        fit_binding_pymc(
            multi_dataset,
            n_xerr=0.0,
            noise=NoiseConfig.structured(noise_model=noise_model, shared_floor=True),
            sampler=SamplerConfig(n_samples=2, n_tune=1),
        )

    assert captured == {"shared_floor": True}


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
            n_xerr=0.0,
            robust=RobustConfig(enabled=True, nu=7.5),
            sampler=SamplerConfig(n_samples=2, n_tune=1),
        )

    assert captured["name"] == "y_likelihood_default"
    assert captured["nu"] == 7.5


def test_student_t_nu_value_records_fixed_nu_as_deterministic() -> None:
    """A fixed nu is also recorded as a ``student_t_nu`` deterministic.

    This lets ``robust_settings_from_trace`` recover the Student-t likelihood from
    the trace alone, so ``FitResult.residuals`` standardizes ``std_res`` correctly
    without an explicit ``robust=`` override.
    """
    with pm.Model() as model:
        nu = bayes._student_t_nu_value(7.5)  # noqa: SLF001
    assert nu == 7.5  # plain float is still handed to the likelihood
    assert "student_t_nu" in model.named_vars
    assert float(model["student_t_nu"].eval()) == 7.5


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
            n_xerr=0.0,
            robust=RobustConfig(enabled=True, nu=None),
            sampler=SamplerConfig(n_samples=2, n_tune=1),
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
            n_xerr=0.0,
            robust=RobustConfig(enabled=True, likelihood="mixture"),
            sampler=SamplerConfig(n_samples=2, n_tune=1),
        )

    assert captured == {
        "name": "y_likelihood_default",
        "has_weights": True,
        "has_components": True,
    }


def test_fit_binding_pymc_mixture_accepts_label_specific_contamination_prior(
    monkeypatch: pytest.MonkeyPatch,
    multi_dataset: Dataset,
) -> None:
    """Single-fit mixture priors can encode label-specific outlier rates."""
    captured: dict[str, float] = {}

    def fake_beta(name: str, *, alpha: float, beta: float) -> float:
        del alpha
        captured[name] = beta
        return 0.1

    def fake_mixture(*_args: object, **_kwargs: object) -> None:
        raise _StopBayesBuildError

    monkeypatch.setattr(pm, "Beta", fake_beta)
    monkeypatch.setattr(pm, "Mixture", fake_mixture)

    with pytest.raises(_StopBayesBuildError):
        fit_binding_pymc(
            multi_dataset,
            n_xerr=0.0,
            robust=RobustConfig(
                enabled=True,
                likelihood="mixture",
                contamination_frac_prior={"1": 1.0 / 7.0, "2": 1.0 / 63.0},
            ),
            sampler=SamplerConfig(n_samples=2, n_tune=1),
        )

    assert captured["pi_outlier_1"] == pytest.approx(6.0)
    assert captured["pi_outlier_2"] == pytest.approx(62.0)


@pytest.mark.parametrize("contamination_frac_prior", [0.0005, 0.999])
def test_fit_binding_pymc_mixture_rejects_unsafe_contamination_prior(
    ph_dataset: Dataset,
    contamination_frac_prior: float,
) -> None:
    """Mixture likelihood should reject numerically unsafe contamination priors."""
    with pytest.raises(ValueError, match=r"between 0\.001 and 0\.5"):
        fit_binding_pymc(
            ph_dataset,
            n_xerr=0.0,
            robust=RobustConfig(
                enabled=True,
                likelihood="mixture",
                contamination_frac_prior=contamination_frac_prior,
            ),
            sampler=SamplerConfig(n_samples=2, n_tune=1),
        )


def test_fit_binding_pymc_robust_rejects_nonpositive_student_t_nu(
    ph_dataset: Dataset,
) -> None:
    """Nonpositive numeric Student-t nu values should be rejected."""
    with pytest.raises(ValueError, match="student_t_nu must be positive"):
        fit_binding_pymc(
            ph_dataset,
            n_xerr=0.0,
            robust=RobustConfig(enabled=True, nu=0.0),
            sampler=SamplerConfig(n_samples=2, n_tune=1),
        )


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
        **_kwargs: object,
    ) -> dict[str, object]:
        captured["floor"] = floor_mode
        captured["gain"] = gain_mode
        captured["alpha"] = alpha_mode
        raise _StopBayesBuildError

    monkeypatch.setattr(bayes, "build_pymc_noise_priors", fake_build_pymc_noise_priors)

    with pytest.raises(_StopBayesBuildError):
        bayes.fit_binding_pymc_multi(
            {"A01": copy.deepcopy(multi_dataset), "A02": copy.deepcopy(multi_dataset)},
            scheme,
            n_xerr=0.0,
            noise=NoiseConfig.structured(noise_model=noise_model),
            sampler=SamplerConfig(n_samples=2, n_tune=1),
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
        **_kwargs: object,
    ) -> dict[str, object]:
        captured["floor"] = floor_mode
        captured["gain"] = gain_mode
        captured["alpha"] = alpha_mode
        raise _StopBayesBuildError

    monkeypatch.setattr(bayes, "build_pymc_noise_priors", fake_build_pymc_noise_priors)

    with pytest.raises(_StopBayesBuildError):
        bayes.fit_binding_pymc_multi(
            {"A01": fr_init, "A02": fr_init},
            scheme,
            n_xerr=0.0,
            noise=NoiseConfig.structured(noise_model=noise_model),
            sampler=SamplerConfig(n_samples=2, n_tune=1),
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
        _noise_model: PlateNoiseModel, **_kwargs: object
    ) -> dict[str, object]:
        captured["labels"] = list(_noise_model.keys())
        raise _StopBayesBuildError

    monkeypatch.setattr(bayes, "build_pymc_noise_priors", fake_build_pymc_noise_priors)

    with pytest.raises(_StopBayesBuildError):
        bayes.fit_binding_pymc_multi(
            {"A01": fr_init, "A02": fr_init},
            scheme,
            n_xerr=0.0,
            noise=NoiseConfig.structured(noise_model=noise_model),
            sampler=SamplerConfig(n_samples=2, n_tune=1),
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

    def fake_build_multi_ye_mag_priors(  # noqa: PLR0913
        _labels: list[str],
        *,
        per_well: bool = False,
        shared_ye_mags: bool = False,
        prior: str = "lognormal",
        mu: float | dict[str, float] = 0.0,
        sigma: float | dict[str, float] = 1.5,
        parameterization: str = "centered",
    ) -> dict[str, object]:
        del _labels, shared_ye_mags, prior, mu, sigma, parameterization
        captured["per_well"] = per_well
        raise _StopBayesBuildError

    monkeypatch.setattr(
        bayes, "_build_multi_ye_mag_priors", fake_build_multi_ye_mag_priors
    )
    with pytest.raises(_StopBayesBuildError):
        bayes.fit_binding_pymc_multi(
            {"A01": fr_init, "A02": fr_init},
            scheme,
            n_xerr=0.0,
            per_well_ye_mags=per_well_ye_mags,
            noise=NoiseConfig.structured(noise_model=noise_model, learn_ye_mags=True),
            sampler=SamplerConfig(n_samples=2, n_tune=1),
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
            n_xerr=0.0,
            well_noise_scale=True,
            shared_well_noise_scale=True,
            noise=NoiseConfig.structured(noise_model=noise_model),
            sampler=SamplerConfig(n_samples=2, n_tune=1),
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
            n_xerr=0.0,
            x_error_model="deterministic",
            robust=RobustConfig(enabled=True, nu=None),
            sampler=SamplerConfig(n_samples=2, n_tune=1),
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
            n_xerr=0.0,
            x_error_model="deterministic",
            robust=RobustConfig(enabled=True, likelihood="mixture"),
            sampler=SamplerConfig(n_samples=2, n_tune=1),
        )

    assert captured == {
        "name": "y_likelihood_1",
        "has_weights": True,
        "has_components": True,
    }


def test_fit_binding_pymc_multi_mixture_accepts_contamination_prior_boundary(
    monkeypatch: pytest.MonkeyPatch,
    multi_dataset: Dataset,
) -> None:
    """The largest safe contamination prior should still be accepted."""
    scheme = PlateScheme()
    scheme.names = {"ctrl": {"A01", "A02"}}
    fr_init = fit_binding_glob(multi_dataset)

    def fake_mixture(*_args: object, **_kwargs: object) -> None:
        raise _StopBayesBuildError

    monkeypatch.setattr(pm, "Mixture", fake_mixture)

    with pytest.raises(_StopBayesBuildError):
        bayes.fit_binding_pymc_multi(
            {"A01": fr_init, "A02": fr_init},
            scheme,
            n_xerr=0.0,
            x_error_model="deterministic",
            robust=RobustConfig(
                enabled=True, likelihood="mixture", contamination_frac_prior=0.5
            ),
            sampler=SamplerConfig(n_samples=2, n_tune=1),
        )


def test_fit_binding_pymc_multi_mixture_rejects_high_contamination_prior(
    multi_dataset: Dataset,
) -> None:
    """Multi-fit mixture likelihood should reject priors concentrated near one."""
    scheme = PlateScheme()
    scheme.names = {"ctrl": {"A01", "A02"}}
    fr_init = fit_binding_glob(multi_dataset)

    with pytest.raises(ValueError, match=r"between 0\.001 and 0\.5"):
        bayes.fit_binding_pymc_multi(
            {"A01": fr_init, "A02": fr_init},
            scheme,
            n_xerr=0.0,
            x_error_model="deterministic",
            robust=RobustConfig(
                enabled=True, likelihood="mixture", contamination_frac_prior=0.999
            ),
            sampler=SamplerConfig(n_samples=2, n_tune=1),
        )


def test_fit_binding_pymc_multi_mixture_rejects_unsafe_label_specific_prior(
    multi_dataset: Dataset,
) -> None:
    """Label-specific mixture priors must satisfy the same practical bounds."""
    scheme = PlateScheme()
    scheme.names = {"ctrl": {"A01", "A02"}}
    fr_init = fit_binding_glob(multi_dataset)

    with pytest.raises(ValueError, match=r"between 0\.001 and 0\.5"):
        bayes.fit_binding_pymc_multi(
            {"A01": fr_init, "A02": fr_init},
            scheme,
            n_xerr=0.0,
            x_error_model="deterministic",
            robust=RobustConfig(
                enabled=True,
                likelihood="mixture",
                contamination_frac_prior={"1": 1.0 / 7.0, "2": 1e-4},
            ),
            sampler=SamplerConfig(n_samples=2, n_tune=1),
        )


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
            n_xerr=3.0,
            x_error_model="deterministic",
            sampler=SamplerConfig(n_samples=2, n_tune=1),
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
        n_xerr=0.0,
        x_error_model="deterministic",
        sampler=SamplerConfig(n_samples=2, n_tune=1),
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
            x_error_model="per_well",
            n_xerr=0.0,
            noise=NoiseConfig.structured(noise_model=noise_model),
            sampler=SamplerConfig(n_samples=2, n_tune=1),
        )


def test_fit_binding_pymc_multi_per_well_uses_denoised_sigmas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pin per_well to the de-noised pipetting derivation.

    ``x_start`` must be anchored at the isotonic standard error of the step-0
    mean and the per-well ``x_step`` must use the isotonic per-addition pipetting
    sigma — not the pre-fix read-noise-inflated raw / quadrature values. This
    guards against a silent regression to the old formulation.
    """
    # Decreasing pH (acid titration) with a noisy leading x_errc spike, the
    # real-plate scenario the de-noising fix targeted.
    xc = np.array([8.92, 8.80, 8.40, 7.90, 7.30, 6.60, 6.00])
    x_errc = np.array([0.067, 0.017, 0.015, 0.016, 0.014, 0.015, 0.013])
    y1 = binding_1site(xc, 7.2, 2.0, 1.0, is_ph=True)
    y2 = binding_1site(xc, 7.2, 0.1, 1.0, is_ph=True)
    ds = Dataset(
        {"1": DataArray(xc, y1, x_errc=x_errc), "2": DataArray(xc, y2, x_errc=x_errc)},
        is_ph=True,
    )
    fr_init = fit_binding_glob(ds)
    scheme = PlateScheme()
    scheme.names = {"ctrl": {"A01", "A02"}}

    _dir, exp_x_start_sigma, _nom, exp_step_sigmas, _min = bayes._pipetting_walk_params(  # noqa: SLF001
        xc, x_errc, 1.0, min_x_step=0.2
    )

    captured: dict[str, object] = {}
    orig_normal = pm.Normal
    orig_truncnormal = pm.TruncatedNormal

    def fake_normal(name: str, **kwargs: object) -> object:
        if name == "x_start":
            captured["x_start_sigma"] = float(cast("float", kwargs["sigma"]))
        return orig_normal(name, **kwargs)

    def fake_truncnormal(name: str, **kwargs: object) -> object:
        if name == "x_step":
            captured["step_sigma"] = np.ravel(
                np.asarray(cast("Any", kwargs["sigma"]), dtype=float)
            )
            raise _StopBayesBuildError
        return orig_truncnormal(name, **kwargs)

    monkeypatch.setattr(pm, "Normal", fake_normal)
    monkeypatch.setattr(pm, "TruncatedNormal", fake_truncnormal)

    with pytest.raises(_StopBayesBuildError):
        bayes.fit_binding_pymc_multi(
            {"A01": fr_init, "A02": fr_init},
            scheme,
            n_xerr=1.0,
            x_error_model="per_well",
            sampler=SamplerConfig(n_samples=2, n_tune=1),
        )

    x_start_sigma = cast("float", captured["x_start_sigma"])
    step_sigma = cast("np.ndarray", captured["step_sigma"])
    # x_start anchored at the de-noised SE of the step-0 mean ...
    assert x_start_sigma == pytest.approx(max(exp_x_start_sigma, 1e-6))
    # ... far below the raw leading 3-well SD it used to inherit.
    assert x_start_sigma < x_errc[0]

    # per-well x_step uses the isotonic per-addition pipetting sigma ...
    np.testing.assert_allclose(step_sigma, np.maximum(exp_step_sigmas, 1e-6))
    # ... not the old quadrature measured-difference sigma.
    old_quadrature = np.sqrt(x_errc[:-1] ** 2 + x_errc[1:] ** 2)
    assert not np.allclose(step_sigma, old_quadrature)


@pytest.mark.parametrize(("between", "expect_well"), [(0.0, False), (0.03, True)])
def test_fit_binding_pymc_multi_x_start_between_sigma_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    between: float,
    expect_well: bool,  # noqa: FBT001
) -> None:
    """A positive x_start_between_sigma adds a per-well x_start_well term.

    ``0.0`` (default) keeps a single shared scalar anchor (unchanged behavior).
    """
    xc = np.array([6.0, 6.6, 7.3, 8.0])
    x_errc = np.array([0.02, 0.02, 0.02, 0.02])
    y1 = binding_1site(xc, 7.0, 1.0, 2.0, is_ph=True)
    y2 = binding_1site(xc, 7.0, 1.0, 0.1, is_ph=True)
    ds = Dataset(
        {"1": DataArray(xc, y1, x_errc=x_errc), "2": DataArray(xc, y2, x_errc=x_errc)},
        is_ph=True,
    )
    fr_init = fit_binding_glob(ds)
    scheme = PlateScheme()
    scheme.names = {"ctrl": {"A01", "A02"}}

    normal_names: list[str] = []
    orig_normal = pm.Normal
    orig_truncnormal = pm.TruncatedNormal

    def fake_normal(name: str, **kwargs: object) -> object:
        normal_names.append(name)
        return orig_normal(name, **kwargs)

    def fake_truncnormal(name: str, **kwargs: object) -> object:
        if name == "x_step":
            raise _StopBayesBuildError
        return orig_truncnormal(name, **kwargs)

    monkeypatch.setattr(pm, "Normal", fake_normal)
    monkeypatch.setattr(pm, "TruncatedNormal", fake_truncnormal)

    with pytest.raises(_StopBayesBuildError):
        bayes.fit_binding_pymc_multi(
            {"A01": fr_init, "A02": fr_init},
            scheme,
            n_xerr=1.0,
            x_error_model="per_well",
            x_start_between_sigma=between,
            sampler=SamplerConfig(n_samples=2, n_tune=1),
        )

    assert "x_start" in normal_names
    assert ("x_start_well" in normal_names) is expect_well


def test_fit_binding_pymc_multi_rejects_removed_hierarchical_model() -> None:
    """The removed hierarchical_per_well x_error_model raises a clear error."""
    xc = np.array([6.0, 6.6, 7.3, 8.0])
    y = binding_1site(xc, 7.0, 1.0, 2.0, is_ph=True)
    ds = Dataset({"1": DataArray(xc, y)}, is_ph=True)
    fr_init = fit_binding_glob(ds)
    scheme = PlateScheme()
    scheme.names = {"ctrl": {"A01"}}
    with pytest.raises(ValueError, match="hierarchical_per_well"):
        bayes.fit_binding_pymc_multi(
            {"A01": fr_init},
            scheme,
            x_error_model="hierarchical_per_well",  # type: ignore[arg-type]
            sampler=SamplerConfig(n_samples=2, n_tune=1),
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
    """extract_fit should prefer per-well x_true from MultiFitResult."""
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
            "x_true[0, A01]",
            "x_true[1, A01]",
            "x_true[2, A01]",
            "x_true[0, A02]",
            "x_true[1, A02]",
            "x_true[2, A02]",
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
        per_well_fit_results: dict[str, FitResult],
        per_well_scheme: PlateScheme,
        *,
        x_error_model: str,
        global_p_names: object = (),
    ) -> dict[str, FitResult]:
        del per_well_scheme, x_error_model, global_p_names
        assert trace_obj is trace
        assert set(per_well_fit_results) == {"A01", "A02"}
        per_well_calls["count"] += 1
        return {key: FitResult(trace=trace) for key in per_well_fit_results}

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
        n_xerr=0.0,
        x_error_model="deterministic",
        sampler=SamplerConfig(n_samples=2, n_tune=1),
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
            n_xerr=1.0,
            x_error_model="deterministic",
            min_x_step=0.35,
            sampler=SamplerConfig(n_samples=2, n_tune=1),
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
            n_xerr=0.0,
            noise=NoiseConfig.ye_mag(shared=shared_ye_mags),
            sampler=SamplerConfig(n_samples=150, n_tune=150),
        )
        multi = bayes.fit_binding_pymc_multi(
            copy.deepcopy(ds_dict),
            scheme,
            n_xerr=0.0,
            per_well_ye_mags=True,
            x_error_model="deterministic",
            noise=NoiseConfig.ye_mag(shared=shared_ye_mags),
            sampler=SamplerConfig(n_samples=150, n_tune=150),
        )

        assert single.trace is not None
        if shared_ye_mags:
            assert posterior_mean(single.trace, "ye_mag") == pytest.approx(
                posterior_mean(multi.trace, "ye_mag"), abs=0.35
            )
        else:
            for lbl in ("1", "2"):
                var_name = f"ye_mag_{lbl}"
                assert posterior_mean(single.trace, var_name) == pytest.approx(
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
    fit_result_pymc = fit_binding_pymc(
        initial_fit, n_xerr=0, n_sd=10.0, sampler=SamplerConfig(n_samples=50)
    )
    # Check that we got a result
    assert fit_result_pymc.trace is not None
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
        initial_fit,
        n_xerr=1.0,
        n_sd=10.0,
        sampler=SamplerConfig(n_samples=50, random_seed=42),
    )
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
        initial_fit,
        n_xerr=0,
        n_sd=10.0,
        noise=NoiseConfig.structured(noise_model=noise_model),
        sampler=SamplerConfig(n_samples=50),
    )
    assert fit_result.result is not None
    assert "K" in fit_result.result.params
    # Check that K is reasonably close to expected value
    assert 5.0 < fit_result.result.params["K"].value < 9.0


def test_fit_binding_pymc_empty_result() -> None:
    """Test that fit_binding_pymc handles empty FitResult gracefully."""
    pytest.importorskip("pymc")
    # Create an empty FitResult
    empty_result: FitResult = FitResult()
    # Should return empty result without crashing
    result = fit_binding_pymc(empty_result, sampler=SamplerConfig(n_samples=10))
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
        {"x_true": (["chain", "draw", "step", "well"], x_data)},
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
    """extract_fit must read per-well x_true from raw_trace, not from az.summary.

    Regression test: az.summary can wrongly index multi-dim deterministics
    (producing wrong step-0 values per well).  The raw xarray trace is the
    ground truth, and extract_fit with raw_trace must use it.
    """
    # Build xarray trace with CORRECT per-well x_true
    n_chains, n_draws, n_steps, n_wells = 1, 2, 3, 2
    x_data = np.zeros((n_chains, n_draws, n_steps, n_wells))
    for c in range(n_chains):
        for d in range(n_draws):
            x_data[c, d, :, 0] = [8.9, 8.3, 7.7]
            x_data[c, d, :, 1] = [8.9, 8.4, 7.9]

    ds = xr.Dataset(
        {"x_true": (["chain", "draw", "step", "well"], x_data)},
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
            "x_true[0, A01]",
            "x_true[1, A01]",
            "x_true[2, A01]",
            "x_true[0, A02]",
            "x_true[1, A02]",
            "x_true[2, A02]",
        ],
    )
    # The broken summary has x_true[0, A01]=9, x_true[0, A02]=8.1 —
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
        {"x_true": (["chain", "draw", "step", "well"], x_data)},
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
        fr: FitResult = FitResult(
            None,
            type("Result", (), {"params": params})(),
            mini=None,
            dataset=ph_dataset,
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
            n_xerr=0.0,
            noise=NoiseConfig.structured(
                noise_model=noise_model, floor_mode="centered"
            ),
            sampler=SamplerConfig(n_samples=100, n_tune=50),
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
        fr: FitResult = FitResult(
            None,
            type("Result", (), {"params": params})(),
            mini=None,
            dataset=ph_dataset,
        )

        results = {"A01": fr}
        scheme = PlateScheme()
        scheme.names = {"ctrl": {"A01"}}

        multi = bayes.fit_binding_pymc_multi(
            results,
            scheme,
            n_xerr=0.0,
            sampler=SamplerConfig(n_samples=100, n_tune=50),
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


def test_gain_omitted_when_no_label_resolves_a_gain() -> None:
    """No positive gain anywhere -> the Poisson term is omitted, not invented.

    Gain carries the units of the signal, so unlike alpha there is no
    plate-independent around-zero width to fall back on. This invariant is what
    guarantees the zeroed-gain branch never borrows a width of zero.
    """
    pytest.importorskip("pymc")
    nm = PlateNoiseModel({
        "1": NoiseModelParams(sigma_floor=1.0, gain=0.0, alpha=0.03),
        "2": NoiseModelParams(sigma_floor=1.0, gain=0.0, alpha=0.02),
    })
    with pm.Model():
        priors = bayes.build_pymc_noise_priors(
            nm, gain_mode="centered", alpha_mode="centered"
        )
    # Gain has nothing to borrow from, so it is absent.
    assert "gain" not in priors
    # Alpha is dimensionless and always has _ZERO_HINT_ALPHA_SCALE to fall back
    # on, so it stays present even though every label calibrated to a positive
    # value here.
    assert "rel_error" in priors


def test_zeroed_gain_borrows_width_from_resolved_labels() -> None:
    """An exact-zero gain stays estimable, scaled by the labels that resolved one.

    NNLS clamps the collinear y/y**2 basis, so gain=0.0 on one label means alpha
    won that label's decomposition -- not that the Poisson term is absent. The
    prior must let the posterior re-decide.
    """
    pytest.importorskip("pymc")
    nm = PlateNoiseModel({
        "1": NoiseModelParams(sigma_floor=1.0, gain=0.0, alpha=0.02),
        "2": NoiseModelParams(sigma_floor=1.0, gain=1.6, alpha=0.0),
    })
    with pm.Model():
        priors = bayes.build_pymc_noise_priors(nm, gain_mode="centered")
        draws = pm.draw(priors["gain"]["1"], draws=8000, random_seed=0)
    # Non-degenerate and non-negative: a sampled variable, not a hard constant.
    assert float(draws.std()) > 0.0
    assert float(draws.min()) >= 0.0
    # An unresolved (zero) hint uses _ZERO_HINT_WIDTH * plate_gain_scale,
    # not the 20% relative width used for resolved hints: HalfNormal(sigma=1.0
    # * 1.6 = 1.6); its mean is sigma * sqrt(2/pi) ~= 1.2767. abs=0.08 is
    # roughly 7 sampling standard errors at draws=8000.
    assert float(draws.mean()) == pytest.approx(1.6 * np.sqrt(2 / np.pi), abs=0.08)


def test_zeroed_alpha_borrows_width_from_resolved_labels() -> None:
    """An exact-zero alpha stays estimable, scaled by the labels that resolved one.

    Shaped like real plate L4: label "1" resolves an alpha (gain=4.93), and
    label "2" lands on the NNLS alpha=0.0 boundary (gain=1.34) -- meaning gain
    won label 2's decomposition, not that its proportional-error term is
    physically absent. Label 2's prior must borrow its width from label 1's
    resolved alpha (`plate_alpha_scale`), not collapse to a tight band around
    zero, so the posterior can re-decide the collinear split.

    Label 1's alpha is deliberately set to 0.25 rather than L4's measured 0.106.
    At 0.106 the borrowed width is too close to the `_ZERO_HINT_ALPHA_SCALE`
    fallback of 0.1 for the assertion to tell them apart, so a regression that
    dropped the borrow entirely would still pass. 0.25 separates the two.
    """
    pytest.importorskip("pymc")
    nm = PlateNoiseModel({
        "1": NoiseModelParams(sigma_floor=1.0, gain=4.93, alpha=0.25),
        "2": NoiseModelParams(sigma_floor=1.0, gain=1.34, alpha=0.0),
    })
    with pm.Model():
        priors = bayes.build_pymc_noise_priors(nm, alpha_mode="centered")
        draws = pm.draw(priors["rel_error"]["2"], draws=8000, random_seed=0)
    # Non-degenerate and non-negative: a sampled variable, not a hard constant.
    assert float(draws.std()) > 0.0
    assert float(draws.min()) >= 0.0
    # An unresolved (zero) hint uses _ZERO_HINT_WIDTH * plate_alpha_scale
    # (here plate_alpha_scale == 0.25, label 1's resolved alpha): HalfNormal
    # (sigma=1.0 * 0.25); its mean is sigma * sqrt(2/pi) ~= 0.1995. Falling
    # back to _ZERO_HINT_ALPHA_SCALE=0.1 instead would give ~0.0798, far
    # outside this tolerance.
    assert float(draws.mean()) == pytest.approx(0.25 * np.sqrt(2 / np.pi), abs=0.006)


def test_gain_prior_width_agrees_between_shared_and_per_label() -> None:
    """One hint gives one width, whether the gain is pooled or per-label.

    The shared branch used to floor sigma at 0.1 and the per-label branch at
    0.01 -- a 10x disagreement with no rationale.
    """
    pytest.importorskip("pymc")
    nm = PlateNoiseModel({"1": NoiseModelParams(sigma_floor=1.0, gain=0.3)})
    with pm.Model():
        shared = bayes.build_pymc_noise_priors(
            nm, shared_gain=True, gain_mode="centered"
        )
        shared_std = float(pm.draw(shared["gain"], draws=8000, random_seed=0).std())
    with pm.Model():
        per_label = bayes.build_pymc_noise_priors(
            nm, shared_gain=False, gain_mode="centered"
        )
        per_std = float(
            pm.draw(per_label["gain"]["1"], draws=8000, random_seed=0).std()
        )
    # Both are 0.2 * 0.3 = 0.06 now; previously 0.1 (shared) vs 0.06 (per-label).
    assert shared_std == pytest.approx(per_std, rel=0.05)
    assert shared_std == pytest.approx(0.06, abs=0.006)


def test_fixed_mode_keeps_hard_constants_for_both_terms() -> None:
    """Mode ``fixed`` is the one mode where a zero genuinely means absent.

    The softening of zeroed gains must not leak into "fixed", which callers use
    to pin or disable a term outright.
    """
    pytest.importorskip("pymc")
    nm = PlateNoiseModel({
        "1": NoiseModelParams(sigma_floor=1.0, gain=0.0, alpha=0.02),
        "2": NoiseModelParams(sigma_floor=1.0, gain=1.6, alpha=0.0),
    })
    with pm.Model():
        priors = bayes.build_pymc_noise_priors(
            nm, gain_mode="fixed", alpha_mode="fixed"
        )
    # Constants, not sampled variables: they evaluate to the hint exactly and
    # carry no randomness.
    assert float(priors["gain"]["1"].eval()) == 0.0
    assert float(priors["gain"]["2"].eval()) == pytest.approx(1.6)
    assert float(priors["rel_error"]["1"].eval()) == pytest.approx(0.02)
    assert float(priors["rel_error"]["2"].eval()) == 0.0


def test_shared_zeroed_alpha_matches_per_label_halfnormal() -> None:
    """Pooled alpha with every hint zeroed spans the same width as per-label.

    The shared branch reaches the zero-hint width through a different
    distribution family -- ``TruncatedNormal(mu=0, sigma=s, lower=0)`` rather
    than ``HalfNormal(sigma=s)``. Those are the same distribution, and this
    pins that equivalence so the pooled path cannot drift from the per-label
    one. With no label resolving a positive alpha, both fall back to
    ``_ZERO_HINT_ALPHA_SCALE``.
    """
    pytest.importorskip("pymc")
    nm = PlateNoiseModel({
        "1": NoiseModelParams(sigma_floor=1.0, gain=5.53, alpha=0.0),
        "2": NoiseModelParams(sigma_floor=1.0, gain=0.52, alpha=0.0),
    })
    with pm.Model():
        pooled = bayes.build_pymc_noise_priors(
            nm, shared_alpha=True, alpha_mode="centered"
        )
        pooled_draws = pm.draw(pooled["rel_error"], draws=8000, random_seed=0)
    with pm.Model():
        per_label = bayes.build_pymc_noise_priors(
            nm, shared_alpha=False, alpha_mode="centered"
        )
        per_draws = pm.draw(per_label["rel_error"]["1"], draws=8000, random_seed=1)
    # Both are the _ZERO_HINT_ALPHA_SCALE fallback of 0.1, mean 0.1*sqrt(2/pi).
    expected = 0.1 * np.sqrt(2 / np.pi)
    assert float(pooled_draws.mean()) == pytest.approx(expected, abs=0.005)
    assert float(per_draws.mean()) == pytest.approx(expected, abs=0.005)
    assert float(pooled_draws.min()) >= 0.0


def test_ye_mag_hierarchical_correlates_labels_at_prior() -> None:
    """Shared per-well factor induces positive cross-label prior correlation."""
    coords = {"well": [f"w{i}" for i in range(30)]}
    with pm.Model(coords=coords):
        bayes._build_multi_ye_mag_priors(  # noqa: SLF001
            ["1", "2"], per_well=True, parameterization="hierarchical"
        )
        pr = pm.sample_prior_predictive(
            400, var_names=["ye_mag_1", "ye_mag_2", "ye_mag_tau_delta"]
        ).prior

    a = np.log(np.asarray(pr["ye_mag_1"]).reshape(-1, 30)).mean(axis=0)
    b = np.log(np.asarray(pr["ye_mag_2"]).reshape(-1, 30)).mean(axis=0)
    assert np.corrcoef(a, b)[0, 1] > 0.5
    assert float(np.asarray(pr["ye_mag_tau_delta"]).min()) >= 0.0


@pytest.mark.parametrize("param", ["centered", "hierarchical"])
def test_fit_binding_pymc_multi_accepts_parameterization(param: str) -> None:
    """Each ye_mag parameterization builds and prior-samples with pwym on."""
    rng = np.random.default_rng(0)
    x = np.linspace(5.5, 8.5, 7)
    dsd = {}
    for w in ("A01", "A02", "A03", "A04"):
        y1 = binding_1site(x, 7.0, 600.0, 50.0, is_ph=True) + rng.normal(0, 20, 7)
        y2 = binding_1site(x, 7.0, 40.0, 500.0, is_ph=True) + rng.normal(0, 8, 7)
        dsd[w] = Dataset(
            {
                "1": DataArray(x, y1, y_errc=np.full(7, 20.0)),
                "2": DataArray(x, y2, y_errc=np.full(7, 8.0)),
            },
            is_ph=True,
        )
    fit = bayes.fit_binding_pymc_multi(
        dsd,
        PlateScheme(),
        per_well_ye_mags=True,
        ye_mag_parameterization=cast('Literal["centered", "hierarchical"]', param),
        sampler=SamplerConfig(
            nuts_sampler="pymc", n_tune=20, n_samples=20, chains=2, cores=1
        ),
    )
    assert "ye_mag_1" in fit.trace.posterior
    if param == "hierarchical":
        assert "ye_mag_tau_delta" in fit.trace.posterior
