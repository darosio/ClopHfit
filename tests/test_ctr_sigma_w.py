"""A control group's replicates disagree by sigma_w, and the model can say by how much.

Sharing one K per control group asserts the replicates agree exactly; giving
each its own asserts nothing about the group. Neither is true on these plates -
controls miss their mates by more than their intervals allow - so
``ctr_sigma_w_prior`` puts a spread between the two and estimates it.
"""

from __future__ import annotations

import numpy as np
import pytest

from clophfit.fitting import bayes
from clophfit.fitting.bayes_config import SamplerConfig
from clophfit.fitting.data_structures import DataArray, Dataset, MultiFitResult
from clophfit.fitting.models import binding_1site
from clophfit.prtecan import PlateScheme

PH = np.linspace(5.0, 9.0, 7)
CTR_WELLS = ("A01", "A02", "A03", "A04", "A05", "A06")
SAMPLER = SamplerConfig(n_samples=400, n_tune=400, chains=2)


def _plate(spread: float, *, seed: int = 0) -> tuple[dict[str, Dataset], PlateScheme]:
    """Six controls whose true K differ by *spread* pH, plus two unknown wells."""
    rng = np.random.default_rng(seed)
    out: dict[str, Dataset] = {}
    for i, well in enumerate([*CTR_WELLS, "B01", "B02"]):
        k = 7.0 + (rng.normal(0.0, spread) if well in CTR_WELLS else 0.4 * i)
        arrays = {}
        for lbl, (s0, s1) in (("1", (200.0, 1000.0)), ("2", (1000.0, 200.0))):
            y = binding_1site(PH, k, s0, s1, is_ph=True) + rng.normal(0.0, 4.0, PH.size)
            arrays[lbl] = DataArray(PH, y, y_errc=np.full(PH.size, 4.0))
        out[well] = Dataset(arrays, is_ph=True)
    scheme = PlateScheme()
    scheme.names = {"ctrl": set(CTR_WELLS)}
    return out, scheme


def _sigma_w(spread: float, *, seed: int = 0) -> tuple[float, MultiFitResult]:
    """Posterior mean of K_sigma_w for a plate whose controls differ by *spread*."""
    dsd, scheme = _plate(spread, seed=seed)
    res = bayes.fit_binding_pymc_multi(
        dsd, scheme, ctr_sigma_w_prior=0.1, sampler=SAMPLER
    )
    assert res.trace is not None
    return float(res.trace.posterior["K_sigma_w"].mean()), res


def test_sigma_w_measures_how_far_the_replicates_really_sit_apart() -> None:
    """Controls planted 0.10 pH apart give a clearly positive sigma_w."""
    wide, res = _sigma_w(0.10, seed=1)
    tight, _ = _sigma_w(0.0, seed=2)
    assert wide > 0.04
    assert wide > 2 * tight  # identical controls leave little for sigma_w
    # Every well, control or not, still reports its own K under the old name.
    assert res.trace is not None
    k_free = res.trace.posterior["K_free"]
    assert set(CTR_WELLS) <= {str(w) for w in k_free.coords["free_well"].values}
    assert set(res.results) == set(_plate(0.10, seed=1)[0])


def test_the_replicates_are_pulled_together_not_onto_one_value() -> None:
    """Partial pooling: their spread shrinks against a free fit but does not vanish."""
    dsd, scheme = _plate(0.10, seed=3)
    free = bayes.fit_binding_pymc_multi(
        dsd, scheme, ctr_free_k=True, sampler=SAMPLER
    ).trace
    hier = bayes.fit_binding_pymc_multi(
        dsd, scheme, ctr_sigma_w_prior=0.1, sampler=SAMPLER
    ).trace
    assert free is not None
    assert hier is not None
    wells = list(CTR_WELLS)
    spread_free = float(
        free.posterior["K_free"].mean(("chain", "draw")).sel(free_well=wells).std()
    )
    spread_hier = float(
        hier.posterior["K_free"].mean(("chain", "draw")).sel(free_well=wells).std()
    )
    assert spread_hier < spread_free
    assert spread_hier > 0.2 * spread_free


def _cl_plate() -> tuple[dict[str, Dataset], PlateScheme]:
    """Build the same six controls as a chloride titration (is_ph False)."""
    rng = np.random.default_rng(5)
    cl = np.array([0.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0])
    out = {}
    for well in CTR_WELLS:
        arrays = {
            lbl: DataArray(
                cl,
                binding_1site(cl, 25.0, s0, s1, is_ph=False)
                + rng.normal(0.0, 4.0, cl.size),
                y_errc=np.full(cl.size, 4.0),
            )
            for lbl, (s0, s1) in (("1", (200.0, 1000.0)), ("2", (1000.0, 200.0)))
        }
        out[well] = Dataset(arrays, is_ph=False)
    scheme = PlateScheme()
    scheme.names = {"ctrl": set(CTR_WELLS)}
    return out, scheme


def test_a_kd_fit_has_no_group_spread_to_estimate() -> None:
    """The term is defined on the pH scale; a Kd fit refuses it rather than guessing."""
    dsd, scheme = _cl_plate()
    with pytest.raises(ValueError, match="pH-titration option"):
        bayes.fit_binding_pymc_multi(
            dsd, scheme, ctr_sigma_w_prior=0.1, sampler=SAMPLER
        )
