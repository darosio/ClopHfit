"""Tests for Bayesian handling of x-error adjustments."""

from __future__ import annotations

import arviz as az  # type: ignore[import-untyped]
import numpy as np
import pytest
from lmfit import Parameters  # type: ignore[import-untyped]

from clophfit.fitting import bayes
from clophfit.fitting.bayes_config import SamplerConfig
from clophfit.fitting.data_structures import (
    DataArray,
    Dataset,
    FitResult,
    _Result,  # ruff: ignore[import-private-name]
)


def test_fit_binding_pymc_x_error_adjustment() -> None:
    """Validate that Bayesian x-errors push ``x_true`` toward the latent value.

    We simulate a dataset where the "true" x is slightly offset from the observed
    x and then verify that the posterior for ``x_true`` moves in the correct
    direction once ``x_errc`` information is provided.
    """
    # 1. Simulate Data
    # True parameters
    k_true = 7.0
    s0_true = 100.0
    s1_true = 1000.0

    # Observed x (pH) - slightly wrong
    xc_obs = np.array([8.0, 7.5, 7.0, 6.5, 6.0])
    # True x - let's say the middle point is actually 7.1 instead of 7.0
    # This means at pH 7.0 (obs), the signal should be higher than expected for k=7
    xc_true = np.array([8.0, 7.5, 7.1, 6.5, 6.0])

    # Generate y based on xc_true
    def binding_model(x: np.ndarray, k: float, s0: float, s1: float) -> np.ndarray:
        return s0 + (s1 - s0) * (10 ** (x - k) / (1 + 10 ** (x - k)))

    y_true = binding_model(xc_true, k_true, s0_true, s1_true)
    # Add small noise to y
    rng = np.random.default_rng(42)
    y_obs = y_true + rng.normal(0, 5.0, size=len(y_true))

    # Define uncertainties — step 2 gets larger x_errc so the
    # cumulative-maximum step-sigma gives the model freedom to shift
    # x_true[2] toward 7.1.
    x_errc = np.array([0.05, 0.05, 0.15, 0.15, 0.15])
    y_err = np.array([5.0] * 5)

    # Create Dataset
    da = DataArray(xc=xc_obs, yc=y_obs, x_errc=x_errc, y_errc=y_err)
    ds = Dataset({"test_sample": da}, is_ph=True)

    # Initial Parameters (lmfit style)
    params = Parameters()
    params.add("K", value=k_true)
    params.add("S0_test_sample", value=s0_true)
    params.add("S1_test_sample", value=s1_true)

    fr: FitResult = FitResult(dataset=ds, result=_Result(params))

    # 2. Run Bayesian Fit with x-error modeling
    # n_xerr=1.0 enables x_true modeling. The x-true shift competes with the
    # per-sample y-error scale (``ye_mag``), so the effect on x_true[2] is small
    # (order 0.02 pH). Fix ``random_seed`` for reproducible draws, and use enough
    # of them that the Monte-Carlo error on the posterior mean (~sd/sqrt(ESS)) is
    # well below that shift; otherwise the boundary check below flips sign.
    fit_res = bayes.fit_binding_pymc(
        fr, n_xerr=1.0, sampler=SamplerConfig(n_samples=4000, random_seed=42)
    )
    assert fit_res.trace is not None
    assert hasattr(fit_res.trace, "posterior")
    trace = fit_res.trace

    # 3. Analyze Results
    # Keep full float precision: ``az.summary`` rounds to 2 decimals by default,
    # which collapses the small shift onto the 7.0 boundary.
    summary = az.summary(trace, round_to=8)
    means = {str(key): float(value) for key, value in summary["mean"].to_dict().items()}

    # Check x_true posterior for the 3rd point (index 2)
    # We expect it to shift from 7.0 (obs) towards 7.1 (true)
    # because y_obs corresponds to 7.1.
    x_true_2_mean = means["x_true[2]"]
    assert x_true_2_mean > xc_obs[2]

    # Also check K is reasonable
    k_mean = means["K"]
    assert 6.8 < k_mean < 7.2


def _walk_dataset(k: float, xc: np.ndarray) -> Dataset:
    """One well on a given pH grid, clean enough that the prior decides x."""
    y = 100.0 + 900.0 / (1.0 + 10.0 ** (xc - k))
    return Dataset(
        {
            "1": DataArray(
                xc, y, x_errc=np.full(len(xc), 0.05), y_errc=np.full(len(xc), 5.0)
            )
        },
        is_ph=True,
    )


def test_x_prior_rejects_degenerate_widths() -> None:
    """A zero or negative width is not a prior; it is a pin with no evidence."""
    from clophfit.fitting.bayes_config import (  # ruff: ignore[import-outside-top-level]
        XPrior,
    )

    ok = {"x_start_mu": 7.0, "step_mu": np.array([0.5]), "step_sigma": np.array([0.01])}
    XPrior(x_start_sigma=0.01, **ok)  # baseline: valid

    with pytest.raises(ValueError, match="x_start_sigma must be positive"):
        XPrior(x_start_sigma=0.0, **ok)
    with pytest.raises(ValueError, match="same shape"):
        XPrior(
            x_start_mu=7.0,
            x_start_sigma=0.01,
            step_mu=np.array([0.5, 0.5]),
            step_sigma=np.array([0.01]),
        )
    with pytest.raises(ValueError, match="step_sigma must be positive"):
        XPrior(
            x_start_mu=7.0,
            x_start_sigma=0.01,
            step_mu=np.array([0.5]),
            step_sigma=np.array([0.0]),
        )


def test_x_prior_from_trace_takes_only_plate_level_quantities() -> None:
    """A well's own deviation must not come back to it as its own prior.

    Stage one infers ``x_start_well`` for a well from that well's fluorescence.
    Returning it would put the same measurements into both prior and
    likelihood and shrink stage two's K interval below what the evidence
    supports. What transfers is the shared anchor and the across-well *spread*
    -- the spread widens the prior, it does not locate any single well.
    """
    import xarray as xr  # ruff: ignore[import-outside-top-level]

    rng = np.random.default_rng(0)
    chain, draw, n_well = 2, 400, 6
    # A shared anchor at 7.0, and per-well starts scattered widely around it.
    x_start = rng.normal(7.0, 0.01, (chain, draw))
    offsets = np.array([-0.5, -0.3, 0.0, 0.2, 0.4, 0.6])
    x_start_well = x_start[:, :, None] + offsets[None, None, :]
    x_step = rng.normal(0.5, 0.002, (chain, draw, 2, n_well))

    post = xr.Dataset({
        "x_start": (("chain", "draw"), x_start),
        "x_start_well": (("chain", "draw", "well"), x_start_well),
        "x_step": (("chain", "draw", "step_diff", "well"), x_step),
    })
    prior = bayes.x_prior_from_trace(xr.DataTree.from_dict({"posterior": post}))

    # The anchor is the shared one, not any well's own value.
    assert prior.x_start_mu == pytest.approx(7.0, abs=0.01)
    # The width absorbs the across-well spread (SD of the offsets, ~0.38),
    # so it is far wider than the shared anchor's own SD of ~0.01.
    assert prior.x_start_sigma > 0.3
    assert prior.step_mu.shape == (2,)
    assert np.all(prior.step_sigma > 0)


def test_x_prior_replaces_the_axis_moments_in_the_multi_fit() -> None:
    """x_prior must reach the multi model, not just the single-well one.

    A two-stage fit's second pass is the multi fit, so a prior that only
    threads into ``fit_binding_pymc`` cannot be used for it. Given a prior
    tight enough to dominate, ``x_start`` must sit where the prior says, not
    at the plate's recorded ``xc[0]``.
    """
    from clophfit.fitting.bayes_config import (  # ruff: ignore[import-outside-top-level]
        XPrior,
    )
    from clophfit.prtecan import PlateScheme  # ruff: ignore[import-outside-top-level]

    xc = np.array([8.0, 7.5, 7.0, 6.5, 6.0])
    datasets = {w: _walk_dataset(7.0, xc) for w in ("A01", "A02", "A03")}
    # Displaced from xc[0]=8.0 and very tight, so the prior, not the data,
    # decides where the axis starts.
    prior = XPrior(
        x_start_mu=8.4,
        x_start_sigma=0.005,
        step_mu=np.full(4, 0.5),
        step_sigma=np.full(4, 0.005),
    )
    sampler = SamplerConfig(
        nuts_sampler="pymc", n_samples=60, n_tune=60, chains=1, cores=1
    )
    fit = bayes.fit_binding_pymc_multi(
        datasets,
        PlateScheme(),
        n_xerr=1.0,
        x_error_model="per_well",
        x_prior=prior,
        sampler=sampler,
    )
    x_start = float(fit.trace["posterior"]["x_start"].mean())
    assert x_start == pytest.approx(8.4, abs=0.05), (
        f"x_start settled at {x_start}, so x_prior never reached the multi model"
    )
