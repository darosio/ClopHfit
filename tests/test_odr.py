"""Minimal unit tests for the ODR utilities."""

from __future__ import annotations

import types

import numpy as np
from matplotlib.figure import Figure

from clophfit.fitting.data_structures import DataArray, Dataset
from clophfit.fitting.fitting import fit_binding_glob
from clophfit.fitting.models import binding_1site
from clophfit.fitting.odr import (
    fit_binding_odr,
    format_estimate,
    generalized_combined_model,
    outlier,
)


def test_format_estimate_fixed_and_scientific() -> None:
    """Check format_estimate returns fixed or scientific format appropriately."""
    # Fixed format when values are moderate and error not tiny
    s = format_estimate(12.3456, 0.1234)
    assert "±" in s
    assert "e" not in s.lower()
    # Scientific format when values are extremely large
    s2 = format_estimate(1e9, 1e7)
    assert "e" in s2.lower()


def test_generalized_combined_model_mixed_lengths() -> None:
    """Validate concatenation across datasets of different lengths."""
    # Two datasets: lengths 3 and 2
    x = np.array([5.0, 6.0, 7.0, 1.0, 2.0])
    # pars are: [K, S0_1, S1_1, S0_2, S1_2]
    pars = [7.0, 2.0, 1.0, 0.0, 1.0]
    y = generalized_combined_model(pars, x, [3, 2])
    y1 = binding_1site(x[:3], 7.0, 2.0, 1.0, is_ph=True)
    y2 = binding_1site(x[3:], 7.0, 0.0, 1.0, is_ph=True)
    np.testing.assert_allclose(y, np.concatenate([y1, y2]))


def test_fit_binding_odr_tiny_dataset() -> None:
    """End-to-end smoke test for ODR on a tiny synthetic dataset."""
    # Small synthetic dataset (pH-like)
    x = np.array([9.0, 8.0, 7.0, 6.0, 5.0])
    y1 = binding_1site(x, 7.0, 2.0, 1.0, is_ph=True)
    y2 = binding_1site(x, 7.0, 0.0, 1.0, is_ph=True)
    ds = Dataset({"y1": DataArray(x, y1), "y2": DataArray(x, y2)}, is_ph=True)
    # Provide small errors so ODR has finite weights
    for da in ds.values():
        da.y_err = np.full_like(x, 0.05)
        da.x_err = np.full_like(x, 0.01)
    # Seed not needed; deterministic LM/ODR
    fr = fit_binding_glob(ds)
    assert fr.result is not None
    fr_odr = fit_binding_odr(fr)
    # Sanity checks
    assert fr_odr.result is not None
    assert fr_odr.figure is not None
    assert isinstance(fr_odr.figure, Figure)


def test_outlier_threshold_behavior() -> None:
    """Ensure outlier mask flags extreme residual and none of the small ones."""
    # Build a minimal odr.Output-like object with required attributes
    output = types.SimpleNamespace()
    # Mostly small residuals, one big outlier
    delta = np.array([0.0, 0.0, 0.0, 10.0])
    eps = np.array([0.0, 0.0, 0.0, 0.0])
    output.delta = delta
    output.eps = eps
    mask = outlier(output, threshold=1.5)
    assert mask.shape == delta.shape
    assert bool(mask[-1])
    assert mask[:3].sum() == 0
