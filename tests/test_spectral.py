"""Whole-spectrum global fit (variable projection) and its EnSpire front end."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from clophfit.fitting.models import binding_1site
from clophfit.fitting.spectral import fit_spectra_global
from clophfit.prenspire import EnspireFile, Note, bands as bands_module
from clophfit.prenspire.spectral import fit_titrations_spectral

DATA = Path(__file__).parent / "EnSpire"
LAM = np.arange(400.0, 520.0)
X_PH = np.tile([5.6, 6.3, 6.9, 7.3, 7.9, 8.3, 8.7, 9.1], 3)


def _spectra(
    k: float, *, amount_sd: float, seed: int, x: np.ndarray = X_PH, is_ph: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    e0 = np.exp(-0.5 * ((LAM - 488) / 12) ** 2)
    e1 = 0.3 * np.exp(-0.5 * ((LAM - 400) / 15) ** 2)
    f = binding_1site(x, k, 0.0, 1.0, is_ph=is_ph)
    amount = rng.normal(1.0, amount_sd, x.size)
    y = (np.outer(e0, 1 - f) + np.outer(e1, f)) * amount
    return y + rng.normal(0, 0.005, y.shape), e0, e1


def test_recovers_k_and_species_spectra() -> None:
    """K and both species spectra come back from noisy two-state spectra."""
    y, e0, e1 = _spectra(7.4, amount_sd=0.0, seed=1)
    fit = fit_spectra_global(X_PH, {"exc": (LAM, y)}, jackknife=False)
    assert pytest.approx(7.4, abs=0.02) == fit.K
    species = fit.species["exc"]
    assert species[0] == pytest.approx(e0, abs=0.02)  # S0: deprotonated
    assert species[1] == pytest.approx(e1, abs=0.02)  # S1: protonated


def test_well_scale_absorbs_amount_error_and_whitens_residuals() -> None:
    """A per-well amplitude removes the rank-1 residual left by amount errors."""
    y, *_ = _spectra(7.4, amount_sd=0.05, seed=2)
    plain = fit_spectra_global(X_PH, {"exc": (LAM, y)})
    scaled = fit_spectra_global(X_PH, {"exc": (LAM, y)}, well_scale=True)
    assert abs(scaled.K - 7.4) < 3 * scaled.se
    assert (
        scaled.residual_sv[0] / scaled.residual_sv[1]
        < plain.residual_sv[0] / plain.residual_sv[1]
    )
    assert scaled.well_scale.mean() == pytest.approx(1.0)


def test_jackknife_error_exceeds_naive_when_wavelengths_share_errors() -> None:
    """Well-shared errors make the Jacobian error far too small."""
    y, *_ = _spectra(7.4, amount_sd=0.03, seed=3)
    fit = fit_spectra_global(X_PH, {"exc": (LAM, y)})
    assert fit.se > 3 * fit.se_naive


def test_labels_are_weighted_and_non_finite_wavelengths_dropped() -> None:
    """Each label gets its own weight; a NaN wavelength is dropped."""
    y, *_ = _spectra(7.0, amount_sd=0.0, seed=4)
    noisy = y * 50 + np.random.default_rng(5).normal(0, 1.0, y.shape)
    noisy[3, 0] = np.nan
    fit = fit_spectra_global(X_PH, {"a": (LAM, y), "b": (LAM, noisy)}, jackknife=False)
    assert fit.wavelengths["b"].size == LAM.size - 1
    assert fit.rms["b"] > 10 * fit.rms["a"]
    assert pytest.approx(7.0, abs=0.03) == fit.K


def test_ligand_titration() -> None:
    """A chloride-style binding titration recovers Kd."""
    x = np.tile([0.0, 5, 10, 20, 40, 80, 160, 320], 3)
    y, *_ = _spectra(30.0, amount_sd=0.0, seed=6, x=x, is_ph=False)
    fit = fit_spectra_global(x, {"exc": (LAM, y)}, is_ph=False, jackknife=False)
    assert pytest.approx(30.0, rel=0.05) == fit.K


def test_hill_slope() -> None:
    """The Hill slope is recovered when fitted."""
    rng = np.random.default_rng(7)
    f = binding_1site(X_PH, 7.2, 0.0, 1.0, is_ph=True, hill=0.6)
    y = np.outer(np.exp(-0.5 * ((LAM - 488) / 12) ** 2), 1 - f) + rng.normal(
        0, 0.003, (LAM.size, X_PH.size)
    )
    fit = fit_spectra_global(X_PH, {"exc": (LAM, y)}, fit_hill=True, jackknife=False)
    assert fit.hill == pytest.approx(0.6, abs=0.1)


def test_too_few_wells() -> None:
    """Fewer than four wells cannot determine K."""
    with pytest.raises(ValueError, match="four wells"):
        fit_spectra_global(X_PH[:3], {"exc": (LAM, np.ones((LAM.size, 3)))})


@pytest.mark.slow
def test_enspire_g10_agrees_with_bands() -> None:
    """On the G10 plate the global K agrees with the shared band K."""
    ef = EnspireFile(DATA / "G10.csv")
    note = Note(DATA / "NTT-G10_note.csv").note
    table, fits = fit_titrations_spectral(ef, note)
    bands = bands_module.fit_titrations(ef, note).table
    shared = bands[(bands.band == "all (shared K)") & (bands.subset == "screened")]
    whole = table[table.subset == "all"]
    for temp, k, se in zip(
        whole.temp.to_numpy(),
        whole.K.to_numpy(float),
        whole.se.to_numpy(float),
        strict=True,
    ):
        k_band = float(shared[shared.temp == temp].K.iloc[0])
        assert abs(k - k_band) < 0.1
        assert se > 0
    assert set(fits) == {"NTT-G10 20.0", "NTT-G10 37.0"}
    assert {"row D", "row E", "row F"} <= set(table.subset)


def test_acid_state_recovers_second_pk_and_rejects_ligand() -> None:
    """A dimmer acid state below the main transition is recovered with its own pK."""
    rng = np.random.default_rng(8)
    x = np.tile(np.linspace(4.0, 9.5, 12), 2)
    e0 = np.exp(-0.5 * ((LAM - 488) / 12) ** 2)
    e1 = 0.4 * np.exp(-0.5 * ((LAM - 420) / 15) ** 2)
    ln10 = np.log(10.0)
    logs = np.column_stack([
        np.zeros_like(x),
        (7.5 - x) * ln10,
        (7.5 - x + 5.0 - x) * ln10,
    ])
    frac = np.exp(logs - np.log(np.exp(logs).sum(axis=1, keepdims=True)))
    y = (
        np.outer(e0, frac[:, 0])
        + np.outer(e1, frac[:, 1])
        + np.outer(0.3 * e1, frac[:, 2])
    )
    y += rng.normal(0, 0.003, y.shape)
    fit = fit_spectra_global(x, {"exc": (LAM, y)}, acid_state=True, jackknife=False)
    assert pytest.approx(7.5, abs=0.05) == fit.K
    assert fit.k_acid == pytest.approx(5.0, abs=0.15)
    assert fit.species["exc"].shape == (3, LAM.size)
    with pytest.raises(ValueError, match="pH titrations only"):
        fit_spectra_global(x, {"exc": (LAM, y)}, is_ph=False, acid_state=True)
