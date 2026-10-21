"""``fit_bands_bayes`` must sample once, not twice.

The bayes band fit ran ``fit_binding_pymc`` twice with the same
arguments: once to catch the underdetermined-band error, then again -
with ``x_sigma`` - for the result it actually returns. The prefit that
raises ``InsufficientDataError`` happens inside the call, before any
sampling, so the first call did the whole expensive fit and threw the
answer away.

Every band of every note paid for that: two full NUTS runs to keep
one. The fix is to catch the error around the single call that matters.
"""

from __future__ import annotations

import typing
from typing import TYPE_CHECKING

import numpy as np

from clophfit.fitting import bayes
from clophfit.fitting.data_structures import DataArray, Dataset
from clophfit.prenspire import bands
from clophfit.prenspire.bands import fit_bands_bayes

if TYPE_CHECKING:
    import pytest


def _band_dataset(n: int = 24) -> Dataset:
    """Build a clean sigmoid band on a pH grid."""
    rng = np.random.default_rng(4)
    x = np.linspace(5.0, 9.5, n)
    frac = 1.0 / (1.0 + np.power(10.0, 7.4 - x))
    return Dataset(
        {"exc_anionic": DataArray(x, 0.05 + 0.95 * frac + rng.normal(0, 0.02, n))},
        is_ph=True,
    )


def _count_calls(monkeypatch: pytest.MonkeyPatch) -> list[float | None]:
    """Record the x_sigma of every fit the band fit makes."""
    calls: list[float | None] = []
    real = bayes.fit_binding_pymc

    def counting_fit(ds_or_fr: object, **kwargs: object) -> object:
        calls.append(typing.cast("float | None", kwargs.get("x_sigma")))
        return real(ds_or_fr, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(bands, "fit_binding_pymc", counting_fit)
    return calls


def test_the_bayes_band_fit_samples_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One fit per band, whatever the prior on the axis."""
    calls = _count_calls(monkeypatch)
    fit_bands_bayes(_band_dataset(), seed=3, draws=200, chains=1)
    assert len(calls) == 1


def test_the_bayes_band_fit_samples_once_with_an_axis_prior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same holds when the per-point axis prior is in play."""
    calls = _count_calls(monkeypatch)
    fit_bands_bayes(_band_dataset(), seed=3, draws=200, chains=1, x_sigma=0.1)
    assert len(calls) == 1
    assert calls == [0.1]


def test_an_underdetermined_band_is_still_reported_as_unfitted() -> None:
    """Dropping the first call must not lose the guard it provided.

    Two points cannot carry the four parameters of a band, so the fit
    has to be reported as unfitted - a NaN K - not raise.
    """
    x = np.array([6.0, 7.0])
    y = np.array([0.9, 0.5])
    ds = Dataset({"exc_anionic": DataArray(x, y)}, is_ph=True)
    bb = fit_bands_bayes(ds, seed=1, draws=50, chains=1)
    assert np.isnan(bb.k)
    assert np.isnan(bb.hdi03)
    assert np.isnan(bb.hdi97)
