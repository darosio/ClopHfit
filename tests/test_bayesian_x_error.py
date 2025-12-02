"""Tests for Bayesian handling of x-error adjustments."""

from __future__ import annotations

import arviz as az
import numpy as np
from lmfit import Parameters  # type: ignore[import-untyped]

from clophfit.fitting import bayes
from clophfit.fitting.data_structures import (
    DataArray,
    Dataset,
    FitResult,
    _Result,  # noqa: PLC2701
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

    # Define uncertainties
    x_errc = np.array([0.05] * 5)  # 0.05 pH unit uncertainty
    y_err = np.array([5.0] * 5)

    # Create Dataset
    da = DataArray(xc=xc_obs, yc=y_obs, x_errc=x_errc, y_errc=y_err)
    ds = Dataset({"test_sample": da}, is_ph=True)

    # Initial Parameters (lmfit style)
    params = Parameters()
    params.add("K", value=k_true)
    params.add("S0_test_sample", value=s0_true)
    params.add("S1_test_sample", value=s1_true)

    fr: FitResult[_Result] = FitResult(dataset=ds, result=_Result(params))

    # 2. Run Bayesian Fit with x-error modeling
    # n_xerr=1.0 enables x_true modeling
    fit_res = bayes.fit_binding_pymc(fr, n_samples=500, n_xerr=1.0)
    assert fit_res.mini is not None
    assert isinstance(fit_res.mini, az.InferenceData)
    trace = fit_res.mini

    # 3. Analyze Results
    summary = az.summary(trace)
    means = {str(key): float(value) for key, value in summary["mean"].to_dict().items()}

    # Check x_true posterior for the 3rd point (index 2)
    # We expect it to shift from 7.0 (obs) towards 7.1 (true)
    # because y_obs corresponds to 7.1.
    x_true_2_mean = means["x_true[2]"]
    assert x_true_2_mean > xc_obs[2]

    # Also check K is reasonable
    k_mean = means["K"]
    assert 6.8 < k_mean < 7.2
