"""MCR-ALS against a hand-built two-species titration."""

from __future__ import annotations

import numpy as np

from clophfit.prenspire.mcr import component_pka, mcr_als

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
