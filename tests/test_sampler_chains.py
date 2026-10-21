"""The chain count the caller asks for must reach the sampler.

Under pytest the sampler was forced to a single chain. That is not a
slower version of the diagnostic - it is the *absence* of it: R-hat is
undefined for one chain, so every fit in the suite reported "no
convergence information" and every multi-chain disagreement went
unnoticed. The suite could not see a failure it was built to catch.

The harness still keeps the chains in one process (no forking, which
is what the override was for), but it no longer collapses the chain
count.
"""

from __future__ import annotations

import typing
from typing import TYPE_CHECKING, Protocol

import numpy as np

from clophfit.fitting import bayes
from clophfit.fitting.data_structures import DataArray, Dataset
from clophfit.prenspire.bands import fit_bands_bayes

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pytest


class _Variable(Protocol):
    """The part of a posterior variable the chain count is read from."""

    sizes: Mapping[str, int]


def _chains(kwargs: dict[str, object]) -> int:
    """Read the chain count out of the sampling kwargs."""
    return typing.cast("int", kwargs["chains"])


def _n_chains(trace: object) -> int:
    """Read the chain count out of a fitted trace."""
    posterior = typing.cast("Mapping[str, Mapping[str, _Variable]]", trace)
    return int(posterior["posterior"]["K"].sizes["chain"])


def test_explicit_chain_count_survives_the_test_harness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A test that asks for two chains must get two chains.

    This is the blind spot itself: the caller asks for a multi-chain
    fit, the harness silently runs one, and every R-hat downstream
    comes back undefined.
    """
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "sentinel")
    kwargs = bayes._pymc_sample_parallel_args(  # ruff: ignore[private-member-access] - the seam under test
        "pymc", chains=2, cores=4
    )
    assert _chains(kwargs) == 2


def test_unspecified_chain_count_still_leaves_r_hat_defined(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fit that names no chain count must still have enough for R-hat.

    R-hat needs two or more chains. One chain gives NaN - undefined,
    not "converged" - so the default the harness picks has to leave the
    diagnostic defined.
    """
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "sentinel")
    kwargs = bayes._pymc_sample_parallel_args("pymc")  # ruff: ignore[private-member-access] - the seam under test
    assert _chains(kwargs) >= 2


def test_the_harness_keeps_the_chains_in_one_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sequential sampling stays: a test run must not fork workers.

    The chain count and the core count are separate knobs. Folding the
    core count down to one keeps the run in-process; it says nothing
    about how many chains are drawn, so the diagnostic stays intact.
    """
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "sentinel")
    kwargs = bayes._pymc_sample_parallel_args(  # ruff: ignore[private-member-access] - the seam under test
        "pymc", chains=2, cores=4
    )
    assert kwargs["cores"] == 1


def test_outside_the_harness_the_caller_config_is_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production keeps whatever the caller asked for."""
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    kwargs = bayes._pymc_sample_parallel_args(  # ruff: ignore[private-member-access] - the seam under test
        "pymc", chains=3, cores=2
    )
    assert _chains(kwargs) == 3
    assert kwargs["cores"] == 2


def _one_band_dataset(seed: int = 4) -> Dataset:
    """Build a clean sigmoid band on a pH grid, noisy but well behaved."""
    rng = np.random.default_rng(seed)
    x = np.linspace(5.0, 9.5, 24)
    frac = 1.0 / (1.0 + np.power(10.0, 7.4 - x))
    return Dataset(
        {
            "exc_anionic": DataArray(x, 0.05 + 0.95 * frac + rng.normal(0, 0.02, 24)),
        },
        is_ph=True,
    )


def test_a_two_chain_fit_reports_a_defined_r_hat() -> None:
    """The end-to-end consequence: a multi-chain fit has a real R-hat.

    The R-hat assertions elsewhere in the suite were guarded by
    ``if np.isfinite(...)`` and therefore never ran - one chain leaves
    the diagnostic undefined, and the guard turned that into a silent
    skip rather than a failure.
    """
    bb = fit_bands_bayes(_one_band_dataset(), seed=3, draws=400, chains=2)
    assert bb.trace is not None
    assert _n_chains(bb.trace) == 2
    assert np.isfinite(bb.rhat)
    assert bb.rhat < 1.1


def test_a_one_chain_fit_leaves_the_diagnostic_undefined() -> None:
    """Why the harness must not collapse the chains: one chain has no R-hat.

    This is the number the old harness produced for every fit in the
    suite - NaN, meaning undefined, not meaning converged. A caller
    that explicitly asks for one chain still gets one; it just does not
    get a convergence diagnostic, and this test says so plainly.
    """
    bb = fit_bands_bayes(_one_band_dataset(), seed=3, draws=400, chains=1)
    assert bb.trace is not None
    assert _n_chains(bb.trace) == 1
    assert np.isnan(bb.rhat)
