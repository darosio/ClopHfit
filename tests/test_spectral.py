"""Whole-spectrum global fit (variable projection) and its EnSpire front end."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from clophfit.fitting.models import binding_1site
from clophfit.fitting.spectral import (
    fit_spectra_global,
    fit_spectra_global_bayes,
    observed_fraction,
    paired_delta,
)
from clophfit.prenspire import EnspireFile, Note, bands as bands_module
from clophfit.prenspire.spectral import (
    fit_titrations_spectral,
    isosbestic_scale,
    normalise_to_isosbestic,
    plot_global_panels,
    residual_table,
    sample_spectra,
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


def test_panels_keep_species_on_the_data_scale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A shared constant per panel: data, model and species on the same scale.

    Historically plot_global_panels picked a panel-specific reference for the
    data (value at the crossing, or a fallback) while drawing the species with
    yet another (their value at the shared crossing). An emission window whose
    species are ~0 there (its excitation is 35 nm below) turned that division
    into a ~1e4-1e5 blow-up that crushed the data curves on the same axes
    (arslanbaeva L8 NTT-A08 37.0). Since v0.15.4 sample_spectra returns the
    peak it divided each scan by, and the panel multiplies data, model and
    species by that one constant, so every panel shows raw / reference and the
    species sit on the data scale by construction.
    """
    iso = 455.6
    lam_em = np.arange(450.0, 650.0)
    e0_ex = np.exp(-0.5 * ((LAM - 488) / 12) ** 2)
    e1_ex = np.exp(-0.5 * ((LAM - 422) / 12) ** 2)
    # a true crossing: both excitation species take the same value at iso, so the
    # per-well normalisation there preserves the mixture fractions
    e1_ex *= float(np.interp(iso, LAM, e0_ex)) / float(np.interp(iso, LAM, e1_ex))
    e0_em = np.exp(-0.5 * ((lam_em - 510) / 14) ** 2)  # ~0 at iso: dark there
    e1_em = 0.6 * np.exp(-0.5 * ((lam_em - 590) / 16) ** 2)
    f = binding_1site(X_PH, 7.4, 0.0, 1.0, is_ph=True)
    y_ex = np.outer(e0_ex, 1 - f) + np.outer(e1_ex, f)
    y_em = np.outer(e0_em, 1 - f) + np.outer(e1_em, f)
    # sample_spectra's normalisation: each scan to its own peak (recorded), then
    # every scan by the excitation's per-well value at the crossing
    peak_ex = float(np.nanmax(np.abs(y_ex)))
    peak_em = float(np.nanmax(np.abs(y_em)))
    scale = np.array([np.interp(iso, LAM, y_ex[:, i]) for i in range(X_PH.size)])
    y_ex = y_ex / peak_ex / scale
    y_em = y_em / peak_em / scale
    spectra: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "excitation": (LAM, y_ex),
        "emission 420": (lam_em, y_em),
    }
    fit = fit_spectra_global(X_PH, spectra, is_ph=True, jackknife=False)
    assert pytest.approx(7.4, abs=0.05) == fit.K

    real_close = plt.close
    monkeypatch.setattr(plt, "close", lambda *_args, **_kwargs: None)
    plot_global_panels(
        tmp_path / "panels.pdf",
        spectra,
        X_PH,
        fit,
        title="synthetic",
        iso_nm=iso,
        peaks={"excitation": peak_ex, "emission 420": peak_em},
    )
    fig = plt.gcf()
    species_labels = {"basic (anionic) state", "acidic (protonated) state"}
    for ax in fig.get_axes():
        species = [
            float(np.nanmax(np.abs(np.asarray(line.get_ydata(), dtype=float))))
            for line in ax.get_lines()
            if str(line.get_label()) in species_labels
        ]
        if not species:
            continue  # the fraction panel and the colorbar draw no species
        data = [
            float(np.nanmax(np.abs(np.asarray(line.get_ydata(), dtype=float))))
            for line in ax.get_lines()
            if str(line.get_label()).startswith("_child")
        ]
        assert data, "the panel must draw the wells' spectra"
        assert max(species) < 50 * max(data), (
            f"species peak {max(species):.3g} is off the data scale {max(data):.3g}"
        )
    observed = [
        ax
        for num in plt.get_fignums()
        for ax in plt.figure(num).get_axes()
        for line in ax.get_lines()
        if str(line.get_label()) == "observed, one per well (all scans)"
    ]
    assert len(observed) == 1, "the fraction panel draws exactly one observed series"
    # these spectra carry the isosbestic reference, so the peak multiplies a ratio
    # back and the axis must not claim measured fluorescence
    ylabels = {ax.get_ylabel() for ax in fig.get_axes()}
    assert "signal / well reference" in ylabels, (
        f"normalised panels must say so; got {sorted(ylabels)}"
    )
    real_close(fig)

    # without the normalisation the same peak restores the recorded a.u., and the
    # axis says Fluorescence (a.u.) rather than hiding the true signal level
    raw = {"excitation": (LAM, (np.outer(e0_ex, 1 - f) + np.outer(e1_ex, f)) / peak_ex)}
    raw_fit = fit_spectra_global(X_PH, raw, is_ph=True, jackknife=False)
    plot_global_panels(
        tmp_path / "panels_raw.pdf",
        raw,
        X_PH,
        raw_fit,
        title="synthetic raw",
        peaks={"excitation": peak_ex},
    )
    raw_fig = plt.gcf()
    raw_labels = {ax.get_ylabel() for ax in raw_fig.get_axes()}
    assert "Fluorescence (a.u.)" in raw_labels, (
        f"a.u.-scale panels must say Fluorescence (a.u.); got {sorted(raw_labels)}"
    )
    real_close(raw_fig)


def test_sample_spectra_returns_the_peak_it_divides_by() -> None:
    """The 4th return value lets a caller restore raw/reference units.

    sample_spectra divides each scan by its peak (and, with isosbestic, by the
    excitation value at the crossing). Dropping the peak made plot_global_panels
    reinvent the scale per panel and disagree with the scale the species were fit
    on. The peak must come back.
    """
    eff = EnspireFile(DATA / "G10.csv")
    note = Note(DATA / "NTT-G10_note.csv").note
    one = note[(note["Name"] == "NTT-G10") & (note["Temp"] == 20.0)].drop_duplicates(
        "Well"
    )
    wells = one["Well"].tolist()
    x = one["pH"].to_numpy(float)
    spectra, _w, _x, peaks = sample_spectra(
        eff, wells, x, 20.0, is_ph=True, buffer_wells=[]
    )
    assert spectra
    assert set(peaks) == set(spectra)
    for name, (_lam, y) in spectra.items():
        assert peaks[name] > 0.0
        assert float(np.nanmax(np.abs(np.asarray(y, dtype=float)))) <= 1.0 + 1e-9


def test_one_fraction_per_well_across_scans() -> None:
    """Different scan amplitudes must not be mistaken for different fractions.

    A well's two scans differ in amount (0.35x) and shape; the old per-scan
    unmixing read one fraction per scan, trading the well's amplitude into each.
    With one fraction per well across both scans the amplitudes are absorbed by
    the per-scan scale and the true basic (anionic) fraction comes back.
    """
    x = np.tile(np.linspace(4.0, 9.5, 12), 2)
    # binding_1site's f is the PROTONATED share: 1 at low pH, ~0 at high pH
    f_neutral = np.asarray(binding_1site(x, 7.4, 0.0, 1.0, is_ph=True))
    anionic = np.exp(-0.5 * ((LAM - 490.0) / 15.0) ** 2)
    acid = 0.6 * np.exp(-0.5 * ((LAM - 430.0) / 14.0) ** 2)
    rng = np.random.default_rng(11)
    spectra: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, amount in (("excitation", 1.0), ("emission 420", 0.35)):
        y = np.outer(anionic, 1.0 - f_neutral) + np.outer(acid, f_neutral)
        y *= amount
        y += rng.normal(0.0, 5e-3 * amount, y.shape)
        spectra[name] = (LAM, y)
    fit = fit_spectra_global(x, spectra, is_ph=True, jackknife=False)
    assert pytest.approx(7.4, abs=0.1) == fit.K
    obs = observed_fraction(fit, spectra, is_ph=True)
    assert obs is not None
    assert float(obs.size) == x.size
    # the reported share is the anionic one, so it must RISE with pH
    assert np.corrcoef(x, obs)[0, 1] > 0.9, obs.round(3)
    assert np.allclose(obs, 1.0 - f_neutral, atol=0.05), (
        f"observed {obs.round(3)} vs true anionic {(1 - f_neutral).round(3)}"
    )


def test_observed_fraction_reports_the_state_the_titrant_grows() -> None:
    """PH reports the anionic share, a ligand titration the bound share, both rising.

    The fit's species[1] is the protonated state on a pH plate and the bound state
    on a ligand plate, so the same column means two different things and only the
    titration decides which share is "the product". Both must rise with the
    titrant: naming the protonated share "basic" is the f_basic/f_neutral swap.
    """
    # pH: the anionic share is small at low pH, large at high pH
    x_ph = np.linspace(5.5, 9.5, 9)
    f_neutral = np.asarray(binding_1site(x_ph, 7.4, 0.0, 1.0, is_ph=True))
    anionic = np.exp(-0.5 * ((LAM - 490.0) / 15.0) ** 2)
    neutral = 0.5 * np.exp(-0.5 * ((LAM - 430.0) / 14.0) ** 2)
    ph_spectra = {
        "excitation": (
            LAM,
            np.outer(anionic, 1 - f_neutral) + np.outer(neutral, f_neutral),
        )
    }
    ph_fit = fit_spectra_global(x_ph, ph_spectra, is_ph=True, jackknife=False)
    ph_obs = observed_fraction(ph_fit, ph_spectra, is_ph=True)
    assert ph_obs is not None
    assert np.allclose(ph_obs, 1.0 - f_neutral, atol=0.05), ph_obs.round(3)
    assert ph_obs[0] < 0.05, ph_obs.round(3)
    assert ph_obs[-1] > 0.95, ph_obs.round(3)

    # ligand: the bound share is small at low [Cl] and large at high [Cl]
    x_cl = np.array([0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0])
    f_bound = np.asarray(x_cl / (x_cl + 20.0))
    cl_spectra = {
        "excitation": (LAM, np.outer(anionic, 1 - f_bound) + np.outer(neutral, f_bound))
    }
    cl_fit = fit_spectra_global(x_cl, cl_spectra, is_ph=False, jackknife=False)
    cl_obs = observed_fraction(cl_fit, cl_spectra, is_ph=False)
    assert cl_obs is not None
    assert np.allclose(cl_obs, f_bound, atol=0.05), cl_obs.round(3)
    assert cl_obs[0] < 0.05, cl_obs.round(3)
    assert cl_obs[-1] > 0.85, cl_obs.round(3)


def test_observed_fraction_handles_a_mixed_invisible_scan() -> None:
    """A scan that cannot tell the states is not forced to decide.

    The second scan's two species are nearly identical, so its shape alone
    cannot pin the fraction; the estimator must read the fraction from the
    informative scan and not be dragged by the uninformative one's amplitude.
    (Both scans share the fit's wavelength grid, as real readouts of one
    plate do.)
    """
    x = np.linspace(5.5, 9.0, 8)
    fr = binding_1site(x, 7.4, 0.0, 1.0, is_ph=True)
    informative = np.exp(-0.5 * ((LAM - 470.0) / 12.0) ** 2)
    other = 0.8 * np.exp(-0.5 * ((LAM - 430.0) / 12.0) ** 2)
    flat = 0.5 * informative  # nearly degenerate with informative
    # same grid as the informative scan (the fit's per-scan basis is rank-1 here,
    # so the decomposition into two states is arbitrary - which is exactly the
    # "cannot tell the apart" case we want to exercise)
    spectra: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "informative": (
            LAM,
            np.outer(informative, 1.0 - fr) + np.outer(other, fr),
        ),
        "flat": (LAM, np.outer(flat, np.ones_like(fr))),
    }
    fit = fit_spectra_global(x, spectra, is_ph=True, jackknife=False)
    obs = observed_fraction(fit, spectra, is_ph=True)
    assert obs is not None
    assert np.all(obs == pytest.approx(1.0 - fr, abs=0.12))


def test_observed_fraction_returns_none_with_unknown_labels() -> None:
    """A spectra dict that matches no species asks for no marker."""
    spectra: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "other": (LAM, np.ones((LAM.size, 4))),
    }
    dummy = fit_spectra_global(
        X_PH,
        {
            "excitation": (
                LAM,
                np.outer(np.exp(-0.5 * ((LAM - 488) / 12) ** 2), X_PH / X_PH.max())
                + np.outer(
                    0.3 * np.exp(-0.5 * ((LAM - 400) / 15) ** 2),
                    1.0 - X_PH / X_PH.max(),
                ),
            ),
        },
        is_ph=True,
        jackknife=False,
    )
    assert observed_fraction(dummy, spectra, is_ph=True) is None
    del dummy


def test_isosbestic_reference_is_a_mean_over_the_band_never_a_sum() -> None:
    """The window averaged at the crossing is a *mean*, and it is one wavelength at band 0.

    A sum would scale with how bright the well is, so it would stop being a
    per-well scale: dividing by it would leave the brightness ratio between wells
    inside the spectra and the free per-well amplitude could not absorb it. The
    band is also what makes the reference a little less noisy, so its value must
    be the hand-computed average and not the centre point either.
    """
    lam = np.arange(400.0, 520.0)
    lam_iso = 455.0
    # curved in lambda so the band mean, the centre value and the sum all differ
    shape = 1.0 + 0.01 * (lam - lam_iso) ** 2
    amount = np.array([1.0e3, 2.0e3, 3.0e3])
    ya = np.outer(shape, amount)
    spectra = {"excitation": (lam, ya)}

    banded, usable = isosbestic_scale(spectra, lam_iso, band=3.0)
    assert usable.all()
    # |lam - 455| <= 1.5 keeps 454, 455, 456 on a 1 nm grid
    grid = 1.0 + 0.01 * (np.array([454.0, 455.0, 456.0]) - lam_iso) ** 2
    assert np.allclose(banded, float(np.mean(grid)) * amount, rtol=1e-12)

    exact, _ = isosbestic_scale(spectra, lam_iso, band=0.0)
    assert np.allclose(exact, 1.0 * amount, rtol=1e-12), "band 0 is the crossing itself"

    summed = float(np.sum(grid)) * amount
    assert not np.allclose(banded, summed, rtol=1e-6), "the reference must be a mean"


def test_isosbestic_band_averages_whatever_the_grid_puts_inside_it() -> None:
    """The band has no minimum point count: a coarse grid averages what it has.

    The L1-L4 plates are measured on a 2 nm grid, so a 3 nm band contains exactly
    two points and a 1 nm band contains none. Requiring three points silently made
    the window a no-op there - the reference never changed, which is how the flag
    came to look broken - so two points must be averaged like any other count.
    """
    lam_iso = 455.13
    shape = 1.0 + 0.01 * (np.arange(400.0, 520.0, 2.0) - lam_iso) ** 2
    amount = np.array([1.0e3, 2.0e3])
    spectra = {"excitation": (np.arange(400.0, 520.0, 2.0), np.outer(shape, amount))}
    banded, _ = isosbestic_scale(spectra, lam_iso, band=3.0)
    # 454 and 456 are the only grid points within 1.5 nm; both go into the mean
    at = {float(v): s for v, s in zip(np.arange(400.0, 520.0, 2.0), shape, strict=True)}
    assert np.allclose(
        banded, float(np.mean([at[454.0], at[456.0]])) * amount, rtol=1e-12
    )
    # a band that contains no grid point at all is the interpolated crossing
    empty, _ = isosbestic_scale(spectra, lam_iso, band=1.0)
    exact, _ = isosbestic_scale(spectra, lam_iso, band=0.0)
    assert np.allclose(empty, exact, rtol=1e-12)


def test_isosbestic_band_only_rescales_each_well_by_a_constant() -> None:
    """Widening the crossing band changes each well by a constant, never its shape.

    That constant is what the per-well amplitude in the global fit already fits,
    so the band is a readout-noise choice: it cannot move the titration. The test
    pins the structural half of that claim - the band-3 and band-0 normalised
    spectra differ by one constant per well and by nothing else. (A K-equality
    assertion would be vacuous on noiseless data: with a free amplitude the model
    reproduces any K exactly, so the objective is flat in K and the fitted value
    is decided by conditioning and by the noise.)
    """
    iso = 455.6
    x = np.tile([5.6, 6.3, 6.9, 7.3, 7.9, 8.3, 8.7, 9.1], 3)
    e0 = np.exp(-0.5 * ((LAM - 488) / 12) ** 2)
    e1 = np.exp(-0.5 * ((LAM - 422) / 12) ** 2)
    # a true crossing: both species equally bright at iso
    e1 *= float(np.interp(iso, LAM, e0)) / float(np.interp(iso, LAM, e1))
    f = binding_1site(x, 7.4, 0.0, 1.0, is_ph=True)
    bright = np.linspace(0.8, 1.2, x.size)
    ya = (np.outer(e0, 1.0 - f) + np.outer(e1, f)) * bright
    spectra = {"excitation": (LAM, ya)}

    narrow = normalise_to_isosbestic(spectra, iso, band=0.0)["excitation"][1]
    wide = normalise_to_isosbestic(spectra, iso, band=3.0)["excitation"][1]
    ratio = wide / narrow
    assert np.all(np.isfinite(ratio))
    # one constant per well across every wavelength: a pure per-well rescaling
    assert np.max(np.ptp(ratio, axis=0)) < 1e-12, (
        f"the band changed the shape of a spectrum: {np.max(np.ptp(ratio, axis=0)):.3e}"
    )
    # and it is not the identity: the two choices really do divide by different values
    assert not np.allclose(ratio[0], ratio[0, 0], rtol=1e-3)


def test_inspect_writes_a_sidecar_and_leaves_the_panels_alone(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``inspect=True`` dumps one text block per group and points at the file.

    The dump is the only place the per-well amplitudes and fractions are written
    out; the panel figures are unchanged by it, and the screen's verdict stays in
    the table rather than leaking into the figure.
    """
    eff = EnspireFile(DATA / "G10.csv")
    note = Note(DATA / "NTT-G10_note.csv").note
    note = note[note["Temp"] == 20.0]
    stem = tmp_path / "NTT-G10"
    table, fits = fit_titrations_spectral(
        eff,
        note,
        buffer=True,
        qc=6.0,
        inspect=True,
        figure_stem=stem,
    )
    assert not table.empty
    assert fits

    sidecar = tmp_path / "NTT-G10_global_inspect.txt"
    assert sidecar.exists(), sorted(p.name for p in tmp_path.iterdir())
    text = sidecar.read_text()
    assert "K = " in text
    assert "wells used:" in text
    assert "per-scan species peaks" in text
    # how far below zero each species dips belongs in the dump: the negative lobe
    # of a species that should be dark is invisible unless it is a number
    assert "pts < 0" in text
    assert "per-well basic (anionic) fraction" in text
    assert "f_basic = " in text
    # one block per (sample, temperature) group, and the amplitudes are present
    assert text.count("==") >= 2
    assert "a = " in text
    printed = capsys.readouterr().out
    assert f"inspect: wrote {sidecar}" in printed
    # the panel figure is still written, and no separate species figure appears
    assert sorted(p.name for p in tmp_path.glob("*_panels.pdf"))
    assert not sorted(tmp_path.glob("*_species.pdf"))


def test_bounded_two_state_solve_matches_scipy_nnls() -> None:
    """The closed-form bound is the same solution scipy's NNLS finds.

    Two states make the bounded least squares a 2-variable problem, so the
    optimum is either the unconstrained solution or the boundary of the one
    offending coefficient - no iteration. Pinning it against scipy keeps the
    shortcut honest if the species count or the design ever changes.
    """
    from scipy.optimize import nnls  # ruff: ignore[import-outside-top-level]

    from clophfit.fitting.spectral import _species_solve  # ruff: ignore[import-outside-top-level, import-private-name]

    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(35):
        n_wells = int(rng.integers(4, 8))
        c = np.column_stack([
            rng.uniform(0.1, 1.0, n_wells),
            rng.uniform(0.1, 1.0, n_wells),
        ])
        y = rng.normal(size=(10, n_wells)) * 100.0
        expected = np.array([nnls(c, y[j, :])[0] for j in range(y.shape[0])]).T
        got = _species_solve(c, y, 0.0)
        worst = max(worst, float(np.max(np.abs(got - expected))))
    assert worst < 1e-9, f"closed form and scipy nnls disagree by {worst:.2e}"


def test_species_floor_removes_the_negative_lobe_of_a_dark_species() -> None:
    """``species_min`` keeps a species that should be dark from going negative.

    On the G10 plate the emission-420 scan's anionic species dips below zero: for
    each wavelength the species are an independent linear solve, so where that
    species ought to be dark nothing holds it at zero and it becomes a free
    direction that absorbs the shape mismatch between the titrant ends. A floor of
    0 removes the lobe without materially moving K; the unconstrained solve stays
    the default, so the test also records that the lobe is really there.
    """
    eff = EnspireFile(DATA / "G10.csv")
    note = Note(DATA / "NTT-G10_note.csv").note
    one = note[(note["Name"] == "NTT-G10") & (note["Temp"] == 20.0)].drop_duplicates(
        "Well"
    )
    wells, x = one["Well"].tolist(), one["pH"].to_numpy(float)
    spectra, wells, x, _peaks = sample_spectra(
        eff,
        wells,
        x,
        20.0,
        is_ph=True,
        buffer_wells=note[note["Name"] == "buffer"]["Well"].tolist(),
        bands=("exc_anionic", "em_exc420", "em_exc278"),
    )
    free = fit_spectra_global(x, spectra, is_ph=True, well_scale=True, jackknife=False)
    bounded = fit_spectra_global(
        x, spectra, is_ph=True, well_scale=True, species_min=0.0, jackknife=False
    )
    dark = float(np.min(free.species["emission 420"][0]))
    assert dark < 0.0, f"the fixture no longer reproduces the lobe (min {dark:.3e})"
    for label, species in bounded.species.items():
        assert float(np.min(species)) >= -1e-12, f"{label} still dips below the floor"
    # the constraint is not allowed to move the titration appreciably
    assert pytest.approx(free.K, abs=0.05) == bounded.K, (
        f"floor moved K from {free.K:.4f} to {bounded.K:.4f}"
    )


def test_species_floor_does_not_change_which_wells_the_screen_drops() -> None:
    """The QC screen stays unconstrained, so the bound cannot move its verdict.

    ``--global-qc`` judges a standardized residual whose z was calibrated on the
    free model. A bounded solve cannot fit as closely, so screening the bounded
    fit would flag wells the calibration never intended to flag - on L8 A08 it
    dropped six high-pH wells that the free screen keeps. The screen is therefore
    always evaluated unconstrained and the bound applies to the reported fit
    only: same wells dropped, different species.
    """
    eff = EnspireFile(DATA / "G10.csv")
    note = Note(DATA / "NTT-G10_note.csv").note
    note = note[note["Temp"] == 20.0]
    scans = ("exc_anionic", "em_exc420", "em_exc278")

    def run(species_min: float | None) -> tuple[str, object]:
        table, fits = fit_titrations_spectral(
            eff, note, bands=scans, qc=6.0, species_min=species_min
        )
        row = table[table.subset == "all"].iloc[0]
        return str(row.qc_dropped), fits["NTT-G10 20.0"]

    dropped_free, free = run(None)
    dropped_bound, bounded = run(0.0)
    assert dropped_bound == dropped_free, (
        f"the bound changed the screen: {dropped_free!r} -> {dropped_bound!r}"
    )
    assert bounded.n_wells == free.n_wells
    # the bound did reach the reported fit
    assert float(np.min(free.species["emission 420"][0])) < 0.0
    for label, species in bounded.species.items():
        assert float(np.min(species)) >= -1e-12, f"{label} still dips below the floor"


def test_species_floor_censors_the_bayes_species_draws_only() -> None:
    """On the Bayes fitters the floor censors reported species, never the posterior.

    The species are nuisance parameters in the free-species posterior and are
    integrated out of the marginalized one, so a hard lower bound cannot be part
    of the likelihood without becoming a truncated model. The floor therefore runs
    after sampling: every reported species point is at or above it, while the
    draws of K are bit-identical to the unconstrained run given the same seed.
    """
    x = np.tile(np.linspace(5.6, 9.1, 8), 2)
    f_neutral = np.asarray(binding_1site(x, 7.4, 0.0, 1.0, is_ph=True))
    anionic = np.exp(-0.5 * ((LAM - 490.0) / 15.0) ** 2)
    neutral = 0.5 * np.exp(-0.5 * ((LAM - 430.0) / 14.0) ** 2)
    rng = np.random.default_rng(3)
    y = np.outer(anionic, 1.0 - f_neutral) + np.outer(neutral, f_neutral)
    # a dark species in a second, weakly informative scan is where the unconstrained
    # posterior goes negative without the floor
    dark = 0.02 * np.exp(-0.5 * ((LAM - 500.0) / 30.0) ** 2)
    y2 = np.outer(dark, 1.0 - f_neutral) + np.outer(neutral, f_neutral)
    y2 += rng.normal(0.0, 2e-3, y2.shape)
    spectra = {"excitation": (LAM, y), "emission 420": (LAM, y2)}

    kw = {
        "x_sigma": 0.08,
        "is_ph": True,
        "draws": 60,
        "chains": 2,
        "tune": 100,
        "seed": 0,
    }
    free = fit_spectra_global_bayes(x, spectra, **kw)
    bounded = fit_spectra_global_bayes(x, spectra, species_min=0.0, **kw)

    raw = np.asarray(free.posterior["E0_emission 420"].values)
    assert raw.min() < 0.0, f"the fixture no longer dips negative (min {raw.min():.3e})"
    for var in bounded.posterior.data_vars:
        if var.startswith("E") and var[1:2].isdigit():
            assert float(np.min(np.asarray(bounded.posterior[var].values))) >= 0.0, var
    # K's posterior must be untouched: the floor is applied after sampling
    assert np.array_equal(
        np.asarray(free.posterior["K"].values),
        np.asarray(bounded.posterior["K"].values),
    )
