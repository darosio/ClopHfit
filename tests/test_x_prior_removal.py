"""The two-stage ``XPrior`` machinery is gone from the fitting API.

``XPrior`` carried a plate-level pH-axis prior: a random-walk anchor
plus per-step moments, handed from a stage-one multi-well fit to a
stage-two one. Nothing in the library ever produced that prior - no
config, no CLI flag, no export path - so the only callers were the
tests that exercised it. A knob that is configured, named, and read by
nobody is the failure mode this codebase keeps rediscovering.

What replaced it is not a superset. The new per-point prior
(``x_sigma``) models an axis that is a *set* of independent nominal
targets - one prior per well, no walk between points. The old one
modelled a cumulative addition series. They are different geometries,
and the EnSpire plates need the new one. So the removal drops an
unwired capability rather than a working one; the per-point prior
stays, and this file pins that it still works.

The old names are kept as errors rather than silently accepted, so a
caller that reaches for them learns why they are gone.
"""

from __future__ import annotations

import numpy as np
import pytest

from clophfit.fitting import bayes
from clophfit.fitting.bayes_config import SamplerConfig
from clophfit.fitting.data_structures import DataArray, Dataset


def test_x_prior_is_gone_from_the_fitting_api() -> None:
    """The plate-level two-stage prior has no home in the library."""
    from clophfit.fitting import bayes_config  # ruff: ignore[import-outside-top-level]

    assert not hasattr(bayes, "x_prior_from_trace")
    assert not hasattr(bayes_config, "XPrior")


def test_the_old_prior_is_rejected_with_a_clear_message() -> None:
    """A caller reaching for the removed knob is told what replaced it."""
    ds = Dataset(
        {"1": DataArray(np.array([7.0, 8.0]), np.array([0.5, 0.2]))}, is_ph=True
    )
    with pytest.raises(TypeError, match="x_prior"):
        bayes.fit_binding_pymc(ds, x_prior=object())  # type: ignore[call-arg]  # the removed keyword


def test_the_per_point_prior_still_works() -> None:
    """``x_sigma`` is the live mechanism, and it is not the old one.

    A prior on each measured x independently, with no walk between the
    points: that is the model the EnSpire plates need, and it must
    survive the removal of the two-stage machinery untouched. The
    giveaway is that the axis becomes free - its posterior spread is
    non-zero - where the old walk would have tied the points together.
    """
    x = np.array([6.0, 7.0, 8.0, 9.0, 10.0])
    y = np.array([1.9, 1.5, 1.09, 1.01, 1.0])
    ds = Dataset({"1": DataArray(x, y)}, is_ph=True)
    fit = bayes.fit_binding_pymc(
        ds,
        n_xerr=0.0,
        x_sigma=0.1,
        sampler=SamplerConfig(
            nuts_sampler="pymc", n_samples=100, n_tune=100, chains=2, cores=1
        ),
    )
    assert fit.trace is not None
    x_true = fit.trace["posterior"]["x_true"]
    spread = float(x_true.std(dim=("chain", "draw")).mean())
    assert spread > 0.0, "the per-point prior must leave the axis free"


def test_a_fixed_axis_stays_fixed() -> None:
    """With neither mechanism in play, the axis stays pinned.

    This is what the ``else`` branch of the axis model looks like once
    the old prior is gone: the measured x is the true x, so the two
    mechanisms cannot be confused for one another.
    """
    x = np.array([6.0, 7.0, 8.0, 9.0, 10.0])
    y = np.array([1.9, 1.5, 1.09, 1.01, 1.0])
    ds = Dataset({"1": DataArray(x, y)}, is_ph=True)
    fit = bayes.fit_binding_pymc(
        ds,
        n_xerr=0.0,
        sampler=SamplerConfig(
            nuts_sampler="pymc", n_samples=50, n_tune=50, chains=1, cores=1
        ),
    )
    assert fit.trace is not None
    x_true = fit.trace["posterior"]["x_true"]
    spread = float(x_true.std(dim=("chain", "draw")).mean())
    assert spread == 0.0
