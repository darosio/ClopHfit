"""MCR-ALS against a hand-built two-species titration."""

from __future__ import annotations

import numpy as np
import pytest

from clophfit.prenspire.mcr import (
    component_pka,
    dead_components,
    mcr_als,
    mcr_titration,
    model_fraction,
)

KA = 405.0  # acidic species peak
KB = 485.0  # basic species peak
PKA_TRUE = 7.42


def _gauss(
    lam: np.ndarray, centre: float, height: float, width: float = 28.0
) -> np.ndarray:
    return height * np.exp(-0.5 * ((lam - centre) / width) ** 2)


def _synthetic() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lam = np.arange(380.0, 521.0, 2.0)
    x = np.linspace(5.0, 9.5, 24)
    frac = 1.0 / (1.0 + np.power(10.0, PKA_TRUE - x))  # deprotonated fraction
    s_acid = _gauss(lam, KA, 1.0) * (1.0 - 0.2)  # acidic species absorbs at 405
    s_base = _gauss(lam, KB, 1.6)
    c = np.column_stack([1.0 - frac, frac])
    d = c @ np.vstack([s_acid, s_base])
    rng = np.random.default_rng(3)
    return x, lam, d + rng.normal(0.0, 0.002, d.shape)


def test_resolves_a_known_mixture() -> None:
    """Two Gaussians mixed by a known pKa are recovered, spectra and all."""
    x, lam, d = _synthetic()
    fit = mcr_als(d, 2)
    assert fit.rel_err < 0.02  # noise-limited reconstruction

    # the closed profiles are the true fractions, up to column order;
    # with closure the columns are complements, so pick the one that RISES
    # with pH (signed correlation), not merely the more correlated one
    frac = 1.0 / (1.0 + np.power(10.0, PKA_TRUE - x))
    corr = [float(np.corrcoef(fit.c[:, i], frac)[0, 1]) for i in range(2)]
    order = int(np.argmax(corr))
    fit.c[:, order]
    assert corr[order] > 0.999

    # resolved spectra peak where the species were put (columns of s are species)
    peak = lam[np.argmax(fit.s[:, order])]
    assert abs(peak - KB) <= 6.0
    acid = lam[np.argmax(fit.s[:, 1 - order])]
    assert abs(acid - KA) <= 6.0


def test_pka_from_component_matches_truth() -> None:
    """The post-hoc Henderson-Hasselbalch on the tracking component gives K."""
    x, _, d = _synthetic()
    fit = mcr_als(d, 2)
    k, se, _f0, _f1, _comp, rho = component_pka(x, fit.c)
    assert abs(k - PKA_TRUE) < 0.05
    assert se < 0.05
    assert rho > 0.95


def test_closure_pins_rotation() -> None:
    """Closed profiles sum to 1 and stay non-negative by construction."""
    _x, _lam, d = _synthetic()
    fit = mcr_als(d, 2, closure=True)
    assert np.allclose(fit.c.sum(axis=1), 1.0)
    assert (fit.c >= -1e-12).all()
    assert (fit.s >= -1e-12).all()


def test_rejects_impossible_rank() -> None:
    """Asking for more species than the matrix has rows or columns must raise."""
    _x, _lam, d = _synthetic()
    try:
        mcr_als(d, 999)
    except ValueError:
        return
    msg = "rank above matrix size must raise, not converge"
    raise AssertionError(msg)


def test_langmuir_extraction_for_ligand_titrations() -> None:
    """On a Cl-type axis the tracking component is fitted with x/(Kd+x)."""
    rng = np.random.default_rng(5)
    x = np.linspace(0.0, 500.0, 24)  # mM
    kd = 40.0
    frac = x / (kd + x)
    c = np.column_stack([1.0 - frac, frac])
    c += rng.normal(0.0, 0.004, c.shape)
    c = np.clip(c, 0.0, None)
    k, se, _f0, _f1, _comp, rho = component_pka(x, c, is_ph=False)
    assert abs(k - kd) < 6.0
    assert 0 < se < 5.0
    assert rho > 0.9


def test_too_few_wells_is_reported_as_a_non_fit() -> None:
    """A component with fewer wells than parameters cannot carry a K.

    The curve has three free parameters (K, F0, F1), so two points are not
    enough to fit it. That must come back as a reported non-fit, not as an
    exception that aborts the whole run.
    """
    x = np.array([6.0, 7.0])
    c = np.column_stack([np.array([0.9, 0.5]), np.array([0.1, 0.5])])
    k, se, f0, f1, comp, rho = component_pka(x, c)
    assert np.isnan(k)
    assert np.isnan(se)
    assert np.isnan(f0)
    assert np.isnan(f1)
    assert comp in {0, 1}
    assert np.isfinite(rho)


def test_a_dead_component_is_reported_as_degenerate() -> None:
    """A component whose profile never moves carries no titration information.

    MCR-ALS is scale-degenerate: ``C <- aC, S <- S/a`` leaves the
    reconstruction unchanged, and a decomposition may park a spurious
    species at a huge scale while its concentration column sits at a
    constant. Such a column has a peak-to-peak of zero (up to round-off),
    so it cannot be fitted for a K - the curve would be fitted to noise and
    report a confident, meaningless number.

    ``mcr_als`` must flag that rather than hand back a factorisation whose
    dead column looks like any other.
    """
    rng = np.random.default_rng(21)
    x = np.linspace(5.5, 9.0, 16)
    frac = 1.0 / (1.0 + np.power(10.0, 7.2 - x))
    # two real species plus a third whose profile is constant
    c = np.column_stack([1.0 - frac, frac, np.full_like(frac, 0.5)])
    s = rng.uniform(0.5, 1.5, size=(3, 40))
    d = c @ s
    fit = mcr_als(d, 3)
    assert fit.degenerate
    assert fit.dead_components == (2,)


def test_a_healthy_decomposition_is_not_flagged_as_degenerate() -> None:
    """The gate must not fire on a factorisation that carries information."""
    _x, _lam, d = _synthetic()
    fit = mcr_als(d, 2)
    assert not fit.degenerate
    assert fit.dead_components == ()


def test_a_degenerate_decomposition_reports_no_k() -> None:
    """A degenerate decomposition must not report a K at all.

    The K would be fitted to a constant profile, so it would be a
    confident number describing nothing. The row has to come back as a
    non-fit, with the reason recorded.
    """
    rng = np.random.default_rng(33)
    x = np.linspace(5.5, 9.0, 16)
    frac = 1.0 / (1.0 + np.power(10.0, 7.2 - x))
    c = np.column_stack([1.0 - frac, frac, np.full_like(frac, 0.5)])
    s = rng.uniform(0.5, 1.5, size=(3, 40))
    d = c @ s
    table, _spectra, _profiles = mcr_titration(
        {"20": [("scan", (np.arange(40.0), d.T))]}, {"20": x}, n_species=3
    )
    row = table.iloc[0]
    assert np.isnan(row["K"])
    assert np.isnan(row["se"])
    assert row["degenerate"]
    assert "dead" in str(row["note"])


@pytest.mark.parametrize("seed", [33, 1, 7, 99, 1234])
def test_the_dead_component_is_caught_whatever_the_start(seed: int) -> None:
    """The gate must not depend on where ALS parks the redundant column.

    The scale degeneracy lets a redundant column settle at any scale, so
    its absolute variation moves over many orders of magnitude between
    runs. A gate keyed to an absolute threshold would therefore catch the
    decomposition on some starts and miss it on others - the exact
    behaviour that lets a meaningless K through.
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(5.5, 9.0, 16)
    frac = 1.0 / (1.0 + np.power(10.0, 7.2 - x))
    c = np.column_stack([1.0 - frac, frac, np.full_like(frac, 0.5)])
    s = rng.uniform(0.5, 1.5, size=(3, 40))
    d = c @ s
    fit = mcr_als(d, 3)
    assert fit.degenerate
    assert fit.dead_components


def test_a_three_species_system_is_not_flagged_as_degenerate() -> None:
    """The gate must not fire when every component really does vary.

    Three independent titrations along the same axis give three profiles
    that all move, so the decomposition cannot be collapsed to two. The
    relative peak-to-peak stays of the order of one, far above the
    threshold.
    """
    n = 16
    x = np.linspace(5.5, 9.0, n)
    f1 = 1.0 / (1.0 + np.power(10.0, 7.2 - x))
    f2 = 1.0 / (1.0 + np.power(10.0, 6.2 - x))
    f3 = 1.0 / (1.0 + np.power(10.0, 5.2 - x))
    c = np.column_stack([1.0 - f1, f1 - f2, f2 - f3])
    c /= c.sum(axis=1, keepdims=True)
    lam = np.arange(40.0)
    s = np.vstack([
        np.exp(-0.5 * ((lam - 14) / 3.5) ** 2),
        np.exp(-0.5 * ((lam - 25) / 3.5) ** 2),
        np.exp(-0.5 * ((lam - 35) / 3.5) ** 2),
    ])
    fit = mcr_als(c @ s, 3)
    assert not fit.degenerate
    assert fit.dead_components == ()


def test_the_model_curve_is_a_fraction() -> None:
    """The model curve must be the 0-1 fraction, on one scale for both axes.

    The closed profiles the figure plots are fractions summing to one, so
    the curve drawn over them has to be a fraction too. This is the seam
    the figure uses, so it is pinned here.
    """
    x = np.linspace(4.0, 9.5, 20)
    ph = model_fraction(x, 7.3, is_ph=True)
    assert ph.min() >= 0.0
    assert ph.max() <= 1.0
    assert ph[0] < 0.01  # low pH: fully protonated, fraction ~ 0
    assert ph[-1] > 0.99  # high pH: fully deprotonated, fraction ~ 1
    # the midpoint sits at the K (checked on the function, not the grid:
    # a coarse grid cannot resolve the crossing to better than its spacing)
    assert model_fraction(np.array([7.3]), 7.3, is_ph=True)[0] == pytest.approx(0.5)
    cl = model_fraction(np.array([0.0, 5.0, 40.0, 500.0]), 40.0, is_ph=False)
    assert cl.min() >= 0.0
    assert cl.max() <= 1.0
    assert cl[0] == 0.0
    assert cl[-1] == pytest.approx(500.0 / 540.0)


def test_a_decomposition_with_every_component_dead_is_flagged() -> None:
    """A gate that normalises by the most mobile column misses this case.

    When every column is dead the most mobile one is itself tiny, so a
    purely relative threshold collapses onto it and nothing is flagged.
    That is the shape the real plate gives at three components: all three
    profiles flat, the fit residual better than the sound two-component
    one, and a confident K reported from a profile that never moves.
    """
    c = np.array([
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ]) + np.array([0.0, 3.0e-10, 1.0e-12])
    dead = dead_components(c)
    assert dead == (0, 1, 2)
