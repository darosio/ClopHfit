"""Whole-spectrum global fit (variable projection) and its EnSpire front end."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from clophfit.fitting.models import binding_1site
from clophfit.fitting.spectral import fit_spectra_global, paired_delta
from clophfit.prenspire import EnspireFile, Note, bands as bands_module
from clophfit.prenspire.spectral import (
    fit_titrations_spectral,
    residual_table,
    scan_outliers,
)

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


def test_well_screen_drops_one_broken_spectrum() -> None:
    """A single corrupted well is flagged and the refit recovers K.

    The band path screens individual points; the whole-spectrum fit had no screen
    at all, so a plate with one broken well pulled K with nothing to say so. The
    well screen must find exactly that well, leave the others alone, and pull K
    back towards the truth.
    """
    rng = np.random.default_rng(17)
    x = np.linspace(5.5, 8.5, 7)
    k_true = 7.1
    frac = 1 / (1 + 10 ** (k_true - x))
    lam = np.arange(24, dtype=np.float64)
    e0 = np.linspace(0.2, 0.8, lam.size)
    e1 = np.linspace(0.9, 0.1, lam.size)
    y = (e0[:, None] * (1 - frac) + e1[:, None] * frac) * 0.95
    y += rng.normal(0.0, 0.003, y.shape)
    broken = 3
    # a tilt is not expressible as a mixture of the two species: a gain error or
    # a mean offset is, and the free per-well amplitude absorbs it, so the screen
    # would have nothing to find
    y[:, broken] *= 1.0 + 0.40 * np.linspace(-1.0, 1.0, lam.size)
    spectrum = {"exc": (lam, y)}

    raw = fit_spectra_global(x, spectrum, well_scale=True, jackknife=False)
    screened = fit_spectra_global(
        x, spectrum, well_scale=True, jackknife=False, screen=0.05
    )
    assert int((~screened.kept).sum()) == 1
    assert not screened.kept[broken]
    assert abs(screened.K - k_true) < abs(raw.K - k_true)
    assert screened.n_wells == x.size - 1
    # with nothing to screen, every well is kept
    y_clean = y.copy()
    y_clean[:, broken] /= 1.0 + 0.40 * np.linspace(-1.0, 1.0, lam.size)
    clean = fit_spectra_global(
        x, {"exc": (lam, y_clean)}, well_scale=True, jackknife=False, screen=0.05
    )
    assert clean.kept.all()


def test_residual_traces_are_per_scan_and_per_well() -> None:
    """R_s(pH) comes back per scan and per well, and is small for a clean fit."""
    y, *_ = _spectra(7.4, amount_sd=0.0, seed=11)
    green = y * 3.0
    fit = fit_spectra_global(
        X_PH, {"exc": (LAM, y), "em": (LAM + 100.0, green)}, jackknife=False
    )
    table = residual_table(fit, [f"W{i:02d}" for i in range(X_PH.size)], X_PH)
    assert set(table["scan"]) == {"exc", "em"}
    assert table.shape[0] == 2 * X_PH.size
    assert (table["rms"] < 0.02).all()
    # the scale of the residual follows each scan's own amplitude
    exc = table[table["scan"] == "exc"]["rms"].to_numpy()
    em = table[table["scan"] == "em"]["rms"].to_numpy()
    assert np.median(em) > 2 * np.median(exc)


def test_scan_outliers_flags_the_broken_spectrum_of_one_scan() -> None:
    """A well broken in one scan only is named there and not in the others."""
    y, *_ = _spectra(7.4, amount_sd=0.0, seed=12)
    good = y.copy()
    broken = y.copy()
    broken[:, 5] *= 1.0 + 0.5 * np.sin(np.linspace(0, 6, LAM.size))
    wells = [f"W{i:02d}" for i in range(X_PH.size)]
    table = scan_outliers({"exc": (LAM, good), "em": (LAM, broken)}, wells, X_PH, z=6.0)
    assert set(table.columns) == {
        "scan",
        "well",
        "x",
        "rms",
        "robust_z",
        "peak_nm",
        "flagged",
    }
    flagged = table[table["flagged"]]
    assert set(flagged["well"]) == {"W05"}
    assert set(flagged["scan"]) == {"em"}
    assert flagged["robust_z"].iloc[0] > 6.0
    # the residual of the broken well peaks inside the sine's swing
    assert LAM.min() <= flagged["peak_nm"].iloc[0] <= LAM.max()


def test_well_bootstrap_is_wider_than_the_naive_jacobian_error() -> None:
    """Resampling wells, not pixels, gives the interval the pixels cannot.

    A fit that treats every wavelength as independent reports an error far below
    the truth: 120 wavelengths of one spectrum are not 120 observations. The
    well-cluster bootstrap resamples whole spectra, so its interval reflects what
    would change if the experiment were repeated, which is the number to quote.
    """
    y, *_ = _spectra(7.4, amount_sd=0.05, seed=13)
    fit = fit_spectra_global(
        X_PH, {"exc": (LAM, y)}, n_boot=200, seed=0, jackknife=True
    )
    assert fit.se_boot > 5 * fit.se_naive
    lo, hi = fit.ci_boot
    assert lo < fit.K < hi
    assert 0 < (hi - lo) < 1.0
    # the jackknife and the bootstrap agree, as two well-level estimates should
    assert 0.4 < fit.se_boot / fit.se < 2.5


def test_well_bootstrap_strata_resamples_within_columns() -> None:
    """Stratifying by plate column keeps every replicate spanning the axis."""
    y, *_ = _spectra(7.4, amount_sd=0.02, seed=14)
    columns = [str(i % 8 + 1).zfill(2) for i in range(X_PH.size)]
    plain = fit_spectra_global(
        X_PH, {"exc": (LAM, y)}, n_boot=100, seed=1, jackknife=False
    )
    stratified = fit_spectra_global(
        X_PH,
        {"exc": (LAM, y)},
        n_boot=100,
        seed=1,
        jackknife=False,
        strata=columns,
    )
    assert np.isfinite(plain.se_boot)
    assert np.isfinite(stratified.se_boot)
    assert stratified.ci_boot[0] < stratified.K < stratified.ci_boot[1]


def test_paired_delta_recovers_a_known_correlation() -> None:
    """Two estimates from the same replicates: the paired error is the right one.

    The replicates are built with a known correlation, so the test can check the
    three things the estimator exists for: the paired standard error matches the
    analytic one, it is smaller than the quadrature sum by 1/sqrt(1-rho), and the
    correlation is recovered.
    """
    rng = np.random.default_rng(23)
    n = 4000
    common = rng.normal(0.0, 0.10, n)  # the shared well/preparation error
    a = 7.20 + common + rng.normal(0.0, 0.05, n)
    b = 7.15 + common + rng.normal(0.0, 0.05, n)
    out = paired_delta(7.20, 7.15, a, b)
    # rho follows from the construction: shared term over total variance
    rho_true = 0.10**2 / (0.10**2 + 0.05**2)
    analytic = np.sqrt(2 * (0.10**2 + 0.05**2) * (1 - rho_true))
    assert out["delta"] == pytest.approx(0.05, abs=1e-12)
    assert out["rho"] == pytest.approx(rho_true, abs=0.03)
    assert out["se_paired"] == pytest.approx(analytic, rel=0.06)
    # the quadrature sum is wider by 1/sqrt(1-rho)
    assert out["se_independent"] == pytest.approx(
        out["se_paired"] / np.sqrt(1 - rho_true), rel=0.06
    )
    assert out["ci_lo"] < out["delta"] < out["ci_hi"]
    # a short or mismatched series is refused rather than silently used
    assert np.isnan(paired_delta(7.2, 7.1, a[:2], b[:2])["se_paired"])
    with pytest.raises(ValueError, match="same length"):
        paired_delta(7.2, 7.1, a, b[:-1])
