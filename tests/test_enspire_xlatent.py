"""Per-well latent titrant axis prior of the EnSpire Bayesian methods.

The ``--x-sigma`` posterior treats the note's titrant as a measurement, not an
exact level: every Bayesian method (bands, whole-spectrum global, MCR and its
constrained variant) draws a latent ``x_true`` per well. These tests check the
prior resolution, the synthetic-data recovery of each model, the CLI outputs,
and the input validation.
"""

from __future__ import annotations

from pathlib import Path

import arviz as az  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from clophfit.__main__ import ppr
from clophfit.fitting.data_structures import DataArray, Dataset
from clophfit.fitting.models import binding_1site
from clophfit.fitting.spectral import (
    fit_spectra_global,
    fit_spectra_global_bayes,
    fit_spectra_global_bayes_marginal,
)
from clophfit.prenspire.bands import estimate_x_error_sigma, fit_bands_bayes
from clophfit.prenspire.mcr import fit_mcr_bayes, fit_mcr_constrained_bayes

tpath = Path(__file__).parent


@pytest.fixture(name="runner")
def runner_setup() -> CliRunner:
    """Fixture for invoking command-line interfaces."""
    return CliRunner()


def test_estimate_x_error_sigma_pools_replicate_columns() -> None:
    """Spread within an index column (same nominal target) sets the sigma."""
    wells = ["A01", "B01", "C01", "A02", "B02", "C02", "A03", "B03", "C03"]
    rng = np.random.default_rng(7)
    ph = [
        float(v) for target in (9.05, 8.22, 7.41) for v in rng.normal(target, 0.12, 3)
    ]
    est = estimate_x_error_sigma(wells, ph)
    assert est is not None
    # the per-well truth is 0.12 pH: the pooled estimate lands near it
    assert 0.02 < est < 0.30


def test_estimate_x_error_sigma_insufficient_returns_none() -> None:
    """A plate without replicate rows cannot separate axis error from target."""
    # three different index columns, one well each: no within-column spread
    assert estimate_x_error_sigma(["A01", "A02", "A03"], [8.0, 7.5, 7.0]) is None
    # one column, one well: no degrees of freedom at all
    assert estimate_x_error_sigma(["A01"], [8.0]) is None


def test_fit_bands_bayes_x_sigma_recovers_perturbed_k() -> None:
    """Measured axis carries N(0, 0.15) error; a matching prior recovers K."""
    rng = np.random.default_rng(11)
    k_true = 6.7
    x_true = np.array([5.0, 5.75, 6.5, 7.25, 8.0, 8.75])
    sigma = 0.15
    x_meas = x_true + rng.normal(0.0, sigma, size=x_true.size)
    y = binding_1site(x_meas, k_true, 0.35, 1.0, is_ph=True)
    y += rng.normal(0.0, 0.004, size=y.size)
    ds = Dataset({"exc_anionic": DataArray(x_meas, y)}, is_ph=True)
    bb = fit_bands_bayes(ds, seed=0, draws=400, chains=1, cores=1, x_sigma=sigma)
    assert bb.k is not None
    assert abs(bb.k - k_true) < 0.25
    assert bb.hdi03 < bb.k < bb.hdi97


def test_fit_bands_bayes_x_sigma_none_still_fits() -> None:
    """x_sigma=None is the previous fixed-axis behaviour and must keep working."""
    rng = np.random.default_rng(3)
    x = np.array([5.5, 6.2, 6.9, 7.6, 8.3])
    y = binding_1site(x, 6.9, 0.3, 1.0, is_ph=True)
    y += rng.normal(0.0, 0.003, size=y.size)
    ds = Dataset({"exc_anionic": DataArray(x, y)}, is_ph=True)
    bb = fit_bands_bayes(ds, seed=1, draws=200, chains=1, cores=1)
    assert abs(bb.k - 6.9) < 0.3


def test_global_bayes_rejects_invalid_inputs() -> None:
    """Bad inputs fail before any sampling runs."""
    lam = np.arange(10, dtype=np.float64)
    y = np.tile(np.linspace(0.2, 0.8, 10)[:, None], (1, 6))
    x = np.linspace(5.5, 8.5, 6)
    with pytest.raises(ValueError, match=">=4 wells"):
        fit_spectra_global_bayes(x[:3], {"exc": (lam, y[:, :3])}, x_sigma=0.1)
    with pytest.raises(ValueError, match="positive finite x_sigma"):
        fit_spectra_global_bayes(x, {"exc": (lam, y)}, x_sigma=0.0)
    with pytest.raises(ValueError, match="positive finite x_sigma"):
        fit_spectra_global_bayes(x, {"exc": (lam, y)}, x_sigma=np.nan)
    xb = x.copy()
    xb[1] = np.nan
    with pytest.raises(ValueError, match="finite titrant axis"):
        fit_spectra_global_bayes(xb, {"exc": (lam, y)}, x_sigma=0.1)
    yn = np.full_like(y, np.nan)
    with pytest.raises(ValueError, match="finite wavelengths"):
        fit_spectra_global_bayes(x, {"exc": (lam, yn)}, x_sigma=0.1)


def test_mcr_bayes_rejects_invalid_inputs() -> None:
    """Bad MCR inputs fail before any sampling runs."""
    rng = np.random.default_rng(2)
    x = np.linspace(5.5, 8.5, 6)
    d = rng.normal(0.5, 0.05, (6, 10))
    with pytest.raises(ValueError, match="one x value per well"):
        fit_mcr_bayes(d, x[:-1], x_sigma=0.1)
    with pytest.raises(ValueError, match=">=2 feasible species"):
        fit_mcr_bayes(d, x, x_sigma=0.1, n_species=1)
    with pytest.raises(ValueError, match=">=2 feasible species"):
        fit_mcr_bayes(d, x, x_sigma=0.1, n_species=11)
    with pytest.raises(ValueError, match="positive x_sigma"):
        fit_mcr_bayes(d, x, x_sigma=-0.1)
    db = d.copy()
    db[0, 0] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        fit_mcr_bayes(db, x, x_sigma=0.1)
    with pytest.raises(ValueError, match="every spectral feature"):
        fit_mcr_constrained_bayes(d, x, np.zeros(9, dtype=int), x_sigma=0.1)
    with pytest.raises(ValueError, match="contiguous"):
        fit_mcr_constrained_bayes(
            d, x, np.array([0, 0, 2] * 3 + [0], dtype=int), x_sigma=0.1
        )


def _two_state_spectra(
    *,
    n_lambda: int = 18,
    noise: float = 0.006,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Build a 16-well pH titration of two smooth species spectra.

    Returns (x, wavelengths, Y[n_lambda, n_wells], k_true). Species spectra
    follow :func:`~clophfit.fitting.models.binding_1site`'s ``is_ph`` ordering:
    state 0 carries the high-pH weight (``1 - fraction``), state 1 the low-pH
    one (``fraction = 1 / (1 + 10**(pH - K))``). The wells repeat each axis
    value twice with slightly different amplitudes, as a two-replicate plate
    row would.
    """
    x = np.tile(np.linspace(5.5, 8.5, 8), 2)
    k_true = 7.0
    frac = 1 / (1 + 10 ** (k_true - x))
    e0 = np.linspace(0.2, 0.8, n_lambda)
    e1 = np.linspace(0.9, 0.1, n_lambda)
    amp = np.repeat([0.95, 1.05], 8)
    y = (e0[:, None] * (1 - frac) + e1[:, None] * frac) * amp
    y += np.random.default_rng(24).normal(0, noise, y.shape)
    return x, np.arange(float(n_lambda)), y, k_true


@pytest.mark.slow
def test_global_bayes_recovers_synthetic_pka_and_latent_axis() -> None:
    """Whole-spectrum posterior: K, the latent axis, noise and state spectra."""
    x, lam, y, k_true = _two_state_spectra()
    trace = fit_spectra_global_bayes(
        x,
        {"exc": (lam, y)},
        x_sigma=0.05,
        seed=5,
        draws=300,
        chains=2,
        cores=1,
    )
    kdraw = np.asarray(trace.posterior["K"].values).reshape(-1)
    hdi = np.asarray(az.hdi(kdraw, prob=0.94), dtype=float)
    assert hdi[0] <= k_true <= hdi[1]
    rhat = float(az.rhat(trace, var_names=["K"])["K"].item())
    assert np.isfinite(rhat)
    assert rhat < 1.05
    # the scan residual scale is inferred, not guessed: it lands on the noise
    sig = float(np.median(np.asarray(trace.posterior["sigma_exc"].values).ravel()))
    assert 0.002 < sig < 0.012
    # the per-well amplitude keeps its mean-1 constraint and tracks the wells
    amp = np.asarray(trace.posterior["amplitude"].values).reshape(-1, x.size)
    med_amp = np.median(amp, axis=0)
    assert np.allclose(amp.mean((0, 1)), 1.0, atol=1e-6)
    assert np.abs(med_amp - np.repeat([0.95, 1.05], 8)).max() < 0.08
    # the latent axis follows the note pH (prior sd 0.05, data strong)
    xdraw = np.asarray(trace.posterior["x_true"].values).reshape(-1, x.size)
    med_x = np.median(xdraw, axis=0)
    assert np.abs(med_x - x).max() < 0.5
    assert float(np.corrcoef(x, med_x)[0, 1]) > 0.98

    # cross-check against the least-squares twin: same K, same state ordering
    lm = fit_spectra_global(x, {"exc": (lam, y)}, well_scale=True, jackknife=False)
    assert hdi[0] <= lm.K <= hdi[1]
    e0 = np.asarray(trace.posterior["E0_exc"].values).reshape(-1, lam.size).mean(axis=0)
    e1 = np.asarray(trace.posterior["E1_exc"].values).reshape(-1, lam.size).mean(axis=0)
    assert float(np.corrcoef(e0, lm.species["exc"][0])[0, 1]) > 0.99
    assert float(np.corrcoef(e1, lm.species["exc"][1])[0, 1]) > 0.99


@pytest.mark.slow
def test_global_bayes_ligand_titration() -> None:
    """The non-pH branch (Langmuir axis) recovers a synthetic Kd."""
    rng = np.random.default_rng(41)
    x = np.linspace(0.5, 30.0, 10)
    k_true = 6.0
    frac = x / (k_true + x)
    e0 = np.linspace(0.2, 0.8, 14)
    e1 = np.linspace(0.9, 0.1, 14)
    y = e0[:, None] * (1 - frac) + e1[:, None] * frac
    y += rng.normal(0, 0.005, y.shape)
    trace = fit_spectra_global_bayes(
        x,
        {"exc": (np.arange(y.shape[0], dtype=np.float64), y)},
        x_sigma=0.5,
        is_ph=False,
        seed=9,
        draws=300,
        chains=2,
        cores=1,
    )
    kdraw = np.asarray(trace.posterior["K"].values).reshape(-1)
    hdi = np.asarray(az.hdi(kdraw, prob=0.94), dtype=float)
    assert hdi[0] <= k_true <= hdi[1]


@pytest.mark.slow
def test_global_bayes_two_scans_share_k_with_own_noise() -> None:
    """Two scans of the same titration tie one K but keep separate sigmas."""
    rng = np.random.default_rng(52)
    x, _lam, y, k_true = _two_state_spectra(noise=0.004)
    e0b = np.linspace(0.7, 0.3, 10)
    e1b = np.linspace(0.15, 0.85, 10)
    frac = 1 / (1 + 10 ** (k_true - x))
    y2 = (e0b[:, None] * (1 - frac) + e1b[:, None] * frac) * 0.8
    y2 *= np.repeat([0.95, 1.05], 8)
    y2 += rng.normal(0, 0.01, y2.shape)
    trace = fit_spectra_global_bayes(
        x,
        {
            "exc": (np.arange(y.shape[0], dtype=np.float64), y),
            "em": (np.arange(y2.shape[0], dtype=np.float64), y2),
        },
        x_sigma=0.05,
        seed=2,
        draws=300,
        chains=2,
        cores=1,
    )
    kdraw = np.asarray(trace.posterior["K"].values).reshape(-1)
    hdi = np.asarray(az.hdi(kdraw, prob=0.94), dtype=float)
    assert hdi[0] <= k_true <= hdi[1]
    s1 = float(np.median(np.asarray(trace.posterior["sigma_exc"].values).ravel()))
    s2 = float(np.median(np.asarray(trace.posterior["sigma_em"].values).ravel()))
    assert 0.002 < s1 < 0.008
    assert s2 > s1


@pytest.mark.slow
def test_global_bayes_drops_nonfinite_wavelengths() -> None:
    """A wavelength non-finite in any well is dropped for every scan."""
    x, lam, y, _k_true = _two_state_spectra()
    y[[3, 7]] = np.nan
    y[5, 2] = np.nan  # only one well: the wavelength still drops
    trace = fit_spectra_global_bayes(
        x,
        {"exc": (lam, y)},
        x_sigma=0.05,
        seed=5,
        draws=100,
        chains=1,
        cores=1,
    )
    assert np.asarray(trace.posterior["E0_exc"].values).shape[-1] == lam.size - 3


@pytest.mark.slow
def test_global_bayes_seed_reproducible() -> None:
    """A fixed seed reproduces the posterior draws exactly."""
    x, lam, y, _k_true = _two_state_spectra()
    t1 = fit_spectra_global_bayes(
        x,
        {"exc": (lam, y)},
        x_sigma=0.05,
        seed=3,
        draws=100,
        chains=1,
        cores=1,
        tune=60,
    )
    t2 = fit_spectra_global_bayes(
        x,
        {"exc": (lam, y)},
        x_sigma=0.05,
        seed=3,
        draws=100,
        chains=1,
        cores=1,
        tune=60,
    )
    k1 = np.asarray(t1.posterior["K"].values).reshape(-1)
    k2 = np.asarray(t2.posterior["K"].values).reshape(-1)
    assert np.array_equal(k1, k2)


@pytest.mark.slow
def test_mcr_bayes_samples_closed_profiles_and_latent_ph() -> None:
    """Full Bayesian MCR returns closed and titrant-tracking profiles."""
    rng = np.random.default_rng(12)
    x = np.linspace(5.5, 8.5, 8)
    profile = 1 / (1 + 10 ** (7.0 - x))
    c = np.column_stack([1 - profile, profile])
    species = np.vstack([np.linspace(0.1, 0.8, 12), np.linspace(0.9, 0.2, 12)]).T
    d = c @ species.T + rng.normal(0, 0.008, (x.size, species.shape[0]))
    trace = fit_mcr_bayes(d, x, x_sigma=0.08, seed=5, draws=120, chains=2, cores=1)
    sampled_c = np.asarray(trace.posterior["profiles"].values)
    assert sampled_c.shape[-2:] == (x.size, 2)
    assert np.allclose(sampled_c.sum(axis=-1), 1.0, atol=1e-6)
    assert trace.posterior["x_true"].shape[-1] == x.size
    # one column tracks the titrant (up to the sign closure allows); the
    # softmax profile prior keeps a little slack around the logistic shape
    med = np.median(sampled_c.reshape(-1, x.size, 2), axis=0)
    rho = np.abs(np.corrcoef(x, med.T)[0, 1:])
    assert rho.max() > 0.85


@pytest.mark.slow
def test_constrained_mcr_bayes_recovers_pka_from_synthetic_spectra() -> None:
    """The joint MCR model estimates K through its HH-constrained profiles."""
    rng = np.random.default_rng(31)
    x = np.linspace(5.3, 8.7, 14)
    k_true = 7.1
    f = 1 / (1 + 10 ** (k_true - x))
    c = np.column_stack([f, 1 - f])
    species = np.vstack([np.linspace(0.1, 0.7, 18), np.linspace(0.8, 0.2, 18)]).T
    d = c @ species.T + rng.normal(0, 0.004, (x.size, species.shape[0]))
    trace = fit_mcr_constrained_bayes(
        d,
        x,
        np.zeros(d.shape[1], dtype=int),
        x_sigma=0.03,
        seed=8,
        draws=500,
        chains=2,
        cores=1,
    )
    kdraw = np.asarray(trace.posterior["K"].values).reshape(-1)
    hdi = np.asarray(az.hdi(kdraw, prob=0.94), dtype=float)
    assert hdi[0] <= k_true <= hdi[1]
    assert "profiles" in trace.posterior
    assert "x_true" in trace.posterior
    med = np.median(
        np.asarray(trace.posterior["profiles"].values).reshape(-1, x.size, 2), axis=0
    )
    assert np.allclose(med.sum(axis=1), 1.0, atol=1e-6)
    # state 0 = empty (deprotonated, rises with pH); state 1 = protonated
    assert float(np.corrcoef(x, med[:, 0])[0, 1]) > 0.9
    assert float(np.corrcoef(x, med[:, 1])[0, 1]) < -0.9


@pytest.mark.slow
def test_cli_constrained_mcr_bayes_writes_joint_outputs(
    tmp_path: Path, runner: CliRunner
) -> None:
    """The joint constrained-MCR CLI writes K, latent-x, trace, and figure."""
    data = tpath / "EnSpire"
    lines = (data / "NTT-G10_note.csv").read_text().splitlines()
    note = tmp_path / "NTT-G10_note.csv"
    note.write_text("\n".join([lines[0], *(ln for ln in lines[1:] if ",20.0," in ln)]))
    out = tmp_path / "mcr_constrained_bayes"
    out.mkdir()
    result = runner.invoke(
        ppr,
        [
            "--out",
            str(out),
            "enspire",
            str(data / "G10.csv"),
            str(note),
            "--method",
            "mcr-constrained-bayes",
            "--x-sigma",
            "auto",
            "--draws",
            "100",
            "--chains",
            "2",
            "--cores",
            "1",
            "--seed",
            "6",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out / "NTT-G10_K_mcr_constrained_bayes.csv").exists()
    assert (out / "NTT-G10_x_mcr_constrained_bayes.csv").exists()
    assert (out / "NTT-G10_mcr_constrained_bayes.pdf").stat().st_size > 1000
    assert list(out.glob("*_mcr_constrained_bayes.nc"))


@pytest.mark.slow
def test_cli_mcr_bayes_writes_posterior_outputs(
    tmp_path: Path, runner: CliRunner
) -> None:
    """The mcr-bayes CLI writes K, latent-x, trace, and figure outputs."""
    data = tpath / "EnSpire"
    lines = (data / "NTT-G10_note.csv").read_text().splitlines()
    note = tmp_path / "NTT-G10_note.csv"
    note.write_text("\n".join([lines[0], *(ln for ln in lines[1:] if ",20.0," in ln)]))
    out = tmp_path / "mcr_bayes"
    out.mkdir()
    result = runner.invoke(
        ppr,
        [
            "--out",
            str(out),
            "enspire",
            str(data / "G10.csv"),
            str(note),
            "--method",
            "mcr-bayes",
            "--x-sigma",
            "auto",
            "--draws",
            "100",
            "--chains",
            "2",
            "--cores",
            "1",
            "--seed",
            "4",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out / "NTT-G10_K_mcr_bayes.csv").exists()
    assert (out / "NTT-G10_x_mcr_bayes.csv").exists()
    assert (out / "NTT-G10_mcr_bayes.pdf").stat().st_size > 1000
    assert list(out.glob("*_mcr_bayes.nc"))


@pytest.mark.slow
def test_cli_global_bayes_writes_posterior_outputs(
    tmp_path: Path, runner: CliRunner
) -> None:
    """The global-bayes CLI writes K, latent-x, trace, and figure outputs."""
    data = tpath / "EnSpire"
    lines = (data / "NTT-G10_note.csv").read_text().splitlines()
    note = tmp_path / "NTT-G10_note.csv"
    note.write_text("\n".join([lines[0], *(ln for ln in lines[1:] if ",20.0," in ln)]))
    out = tmp_path / "global_bayes"
    out.mkdir()
    result = runner.invoke(
        ppr,
        [
            "--out",
            str(out),
            "enspire",
            str(data / "G10.csv"),
            str(note),
            "--method",
            "global-bayes",
            "--x-sigma",
            "auto",
            "--draws",
            "100",
            "--chains",
            "2",
            "--cores",
            "1",
            "--seed",
            "7",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out / "NTT-G10_K_global_bayes.csv").exists()
    assert (out / "NTT-G10_x_global_bayes.csv").exists()
    assert (out / "NTT-G10_global_bayes.pdf").stat().st_size > 1000
    assert list(out.glob("*_global_bayes.nc"))
    table = pd.read_csv(out / "NTT-G10_K_global_bayes.csv")
    assert table.K.notna().all()
    assert (table.hdi03 < table.hdi97).all()
    # no r_hat assertion: 100 smoke-test draws do not converge on a real plate
    # (with the default 2000 the r_hat lands well below 1.1)


@pytest.mark.slow
def test_global_bayes_acid_state_samples_delta_and_third_species() -> None:
    """The free-parameter Bayes supports the acid state as the LM twin does."""
    x, lam, y, _k_true = _two_state_spectra()
    trace = fit_spectra_global_bayes(
        x,
        {"exc": (lam, y)},
        x_sigma=0.05,
        acid_state=True,
        seed=13,
        draws=200,
        chains=1,
        cores=1,
    )
    assert {"delta", "E2_exc"} <= set(trace.posterior.data_vars)
    assert np.asarray(trace.posterior["E2_exc"].values).shape[-1] == lam.size
    kdraw = np.asarray(trace.posterior["K"].values).reshape(-1)
    # the forced third state soaks part of the transition, as it does in the
    # LM twin on the same two-state data: compare against that fit, not the
    # injected ground truth
    lm_acid = fit_spectra_global(x, {"exc": (lam, y)}, acid_state=True, jackknife=False)
    assert np.quantile(kdraw, 0.03) <= lm_acid.K <= np.quantile(kdraw, 0.97)
    delta = float(np.median(np.asarray(trace.posterior["delta"].values).ravel()))
    assert 0.3 < delta < 6.0
    # the acid pK sits below the main transition, as in the LM twin
    assert np.median(kdraw) - delta < np.median(kdraw)


def test_global_bayes_acid_state_is_ph_only() -> None:
    """A ligand titration has no acid state; it fails before any sampling."""
    lam = np.arange(10, dtype=np.float64)
    y = np.tile(np.linspace(0.2, 0.8, 10)[:, None], (1, 6))
    x = np.linspace(1.0, 20.0, 6)
    with pytest.raises(ValueError, match="pH titrations only"):
        fit_spectra_global_bayes(
            x, {"exc": (lam, y)}, x_sigma=0.5, is_ph=False, acid_state=True
        )


@pytest.mark.slow
def test_cli_global_bayes_acid_state_writes_third_state(
    tmp_path: Path, runner: CliRunner
) -> None:
    """--acid-state is accepted by the free-parameter whole-spectrum method."""
    data = tpath / "EnSpire"
    lines = (data / "NTT-G10_note.csv").read_text().splitlines()
    note = tmp_path / "NTT-G10_note.csv"
    note.write_text("\n".join([lines[0], *(ln for ln in lines[1:] if ",20.0," in ln)]))
    out = tmp_path / "global_bayes_acid"
    out.mkdir()
    result = runner.invoke(
        ppr,
        [
            "--out",
            str(out),
            "enspire",
            str(data / "G10.csv"),
            str(note),
            "--method",
            "global-bayes",
            "--x-sigma",
            "auto",
            "--acid-state",
            "--draws",
            "100",
            "--chains",
            "2",
            "--cores",
            "1",
            "--seed",
            "17",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out / "NTT-G10_K_global_bayes.csv").exists()
    ncs = list(out.glob("*_global_bayes.nc"))
    assert ncs
    trace = az.from_netcdf(ncs[0])
    assert "delta" in trace.posterior
    assert any(v.startswith("E2_") for v in trace.posterior.data_vars)


@pytest.mark.slow
def test_cli_marginal_bayes_acid_state_writes_third_state(
    tmp_path: Path, runner: CliRunner
) -> None:
    """--acid-state adds the third state and its pK to the marginal outputs."""
    data = tpath / "EnSpire"
    lines = (data / "NTT-G10_note.csv").read_text().splitlines()
    note = tmp_path / "NTT-G10_note.csv"
    note.write_text("\n".join([lines[0], *(ln for ln in lines[1:] if ",20.0," in ln)]))
    out = tmp_path / "global_bayes_marginal_acid"
    out.mkdir()
    result = runner.invoke(
        ppr,
        [
            "--out",
            str(out),
            "enspire",
            str(data / "G10.csv"),
            str(note),
            "--method",
            "global-bayes-marginal",
            "--x-sigma",
            "auto",
            "--acid-state",
            "--draws",
            "120",
            "--chains",
            "2",
            "--cores",
            "1",
            "--seed",
            "11",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out / "NTT-G10_K_global_bayes_marginal.csv").exists()
    assert (out / "NTT-G10_global_bayes_marginal.pdf").stat().st_size > 1000
    ncs = list(out.glob("*_global_bayes_marginal.nc"))
    assert ncs
    trace = az.from_netcdf(ncs[0])
    assert "delta" in trace.posterior
    assert any(v.startswith("E2_") for v in trace.posterior.data_vars)
    # the plot's fit panel and species panel both render the acid state: the
    # PDF page count doubles per group when the third state is drawn
    assert (out / "NTT-G10_global_bayes_marginal.pdf").stat().st_size > 1000


@pytest.mark.parametrize(
    "method",
    ["global-bayes", "global-bayes-marginal", "mcr-bayes", "mcr-constrained-bayes"],
)
def test_cli_whole_spectrum_bayes_requires_x_sigma(
    tmp_path: Path, runner: CliRunner, method: str
) -> None:
    """--x-sigma is required by the whole-spectrum Bayesian methods."""
    data = tpath / "EnSpire"
    result = runner.invoke(
        ppr,
        [
            "--out",
            str(tmp_path),
            "enspire",
            str(data / "G10.csv"),
            str(data / "NTT-G10_note.csv"),
            "--method",
            method,
        ],
    )
    assert result.exit_code != 0


def test_cli_bayes_x_sigma_auto_and_fixed(tmp_path: Path, runner: CliRunner) -> None:
    """--x-sigma auto resolves from the note's replicate rows; a number passes."""
    data = tpath / "EnSpire"
    for value, floor in (("auto", 0.02), ("0.15", 0.15)):
        out = tmp_path / f"xs_{value}"
        out.mkdir()
        result = runner.invoke(
            ppr,
            [
                "--out",
                str(out),
                "enspire",
                str(data / "G10.csv"),
                str(data / "NTT-G10_note.csv"),
                "--method",
                "bayes",
                "--draws",
                "120",
                "--chains",
                "1",
                "--cores",
                "1",
                "--seed",
                "3",
                "--x-sigma",
                value,
            ],
        )
        assert result.exit_code == 0, result.output
        table = pd.read_csv(out / "NTT-G10_K_bayes.csv")
        assert table["x_sigma"].notna().all()
        assert (table["x_sigma"] >= floor).all()
        assert table.K.between(7.0, 9.5).all()
        if value == "auto":
            # the G10 note has replicate rows: the pooled spread is real and small
            assert (table["x_sigma"] < 0.5).all()


@pytest.mark.parametrize("method", ["band", "svd", "global", "mcr", "vanthoff"])
def test_cli_rejects_x_sigma_without_bayes(
    tmp_path: Path, runner: CliRunner, method: str
) -> None:
    """--x-sigma is a posterior prior; with a non-Bayesian method it is a usage error."""
    data = tpath / "EnSpire"
    out = tmp_path / f"rej_{method}"
    out.mkdir()
    result = runner.invoke(
        ppr,
        [
            "--out",
            str(out),
            "enspire",
            str(data / "G10.csv"),
            str(data / "NTT-G10_note.csv"),
            "--method",
            method,
            "--x-sigma",
            "0.15",
        ],
    )
    assert result.exit_code != 0


# --- marginalized-species whole-spectrum fitter -------------------------------


def test_global_bayes_marginal_rejects_invalid_inputs() -> None:
    """Bad inputs fail before any sampling runs (same contract as the free fitter)."""
    lam = np.arange(10, dtype=np.float64)
    y = np.tile(np.linspace(0.2, 0.8, 10)[:, None], (1, 6))
    x = np.linspace(5.5, 8.5, 6)
    with pytest.raises(ValueError, match=">=4 wells"):
        fit_spectra_global_bayes_marginal(x[:3], {"exc": (lam, y[:, :3])}, x_sigma=0.1)
    with pytest.raises(ValueError, match="positive finite x_sigma"):
        fit_spectra_global_bayes_marginal(x, {"exc": (lam, y)}, x_sigma=0.0)
    with pytest.raises(ValueError, match="pH titrations only"):
        fit_spectra_global_bayes_marginal(
            x, {"exc": (lam, y)}, x_sigma=0.1, is_ph=False, acid_state=True
        )


@pytest.mark.slow
def test_global_bayes_marginal_recovers_synthetic_pka() -> None:
    """Marginalized posterior: K, latent axis, noise and conditional species draws."""
    x, lam, y, k_true = _two_state_spectra()
    trace = fit_spectra_global_bayes_marginal(
        x,
        {"exc": (lam, y)},
        x_sigma=0.05,
        seed=5,
        draws=500,
        chains=2,
        cores=1,
        target_accept=0.95,
    )
    kdraw = np.asarray(trace.posterior["K"].values).reshape(-1)
    hdi = np.asarray(az.hdi(kdraw, prob=0.94), dtype=float)
    assert hdi[0] <= k_true <= hdi[1]
    rhat = float(az.rhat(trace, var_names=["K"])["K"].item())
    assert np.isfinite(rhat)
    assert rhat < 1.05
    # the sampler holds the physical parameters only
    assert set(trace.posterior.data_vars) >= {
        "K",
        "x_true",
        "amplitude",
        "sigma_exc",
        "offset_exc",
        "species_scale_exc",
    }
    # the residual scales land near the injected noise
    sig = float(np.median(np.asarray(trace.posterior["sigma_exc"].values).ravel()))
    assert sig < 0.05
    # the per-well amplitude keeps its mean-1 constraint and tracks the wells
    amp = np.asarray(trace.posterior["amplitude"].values).reshape(-1, x.size)
    med_amp = np.median(amp, axis=0)
    assert np.allclose(amp.mean((0, 1)), 1.0, atol=1e-6)
    assert np.abs(med_amp - np.repeat([0.95, 1.05], 8)).max() < 0.08
    # the latent axis follows the note pH
    xdraw = np.asarray(trace.posterior["x_true"].values).reshape(-1, x.size)
    med_x = np.median(xdraw, axis=0)
    assert float(np.corrcoef(x, med_x)[0, 1]) > 0.98

    # the conditional species draws match the least-squares twin
    lm = fit_spectra_global(x, {"exc": (lam, y)}, well_scale=True, jackknife=False)
    assert hdi[0] <= lm.K <= hdi[1]
    e0 = np.asarray(trace.posterior["E0_exc"].values).reshape(-1, lam.size).mean(axis=0)
    e1 = np.asarray(trace.posterior["E1_exc"].values).reshape(-1, lam.size).mean(axis=0)
    assert float(np.corrcoef(e0, lm.species["exc"][0])[0, 1]) > 0.99
    assert float(np.corrcoef(e1, lm.species["exc"][1])[0, 1]) > 0.99


@pytest.mark.slow
def test_global_bayes_marginal_hill_and_acid_state() -> None:
    """fit_hill and acid_state work with the LM twin's conventions."""
    x, lam, y, k_true = _two_state_spectra()
    hill_tr = fit_spectra_global_bayes_marginal(
        x,
        {"exc": (lam, y)},
        x_sigma=0.05,
        seed=5,
        draws=120,
        chains=1,
        cores=1,
        fit_hill=True,
    )
    kd = np.asarray(hill_tr.posterior["K"].values).reshape(-1)
    h = np.asarray(hill_tr.posterior["hill"].values).reshape(-1)
    assert np.quantile(kd, 0.03) <= k_true <= np.quantile(kd, 0.97)
    assert np.median(h) == pytest.approx(1.0, abs=0.3)
    assert set(hill_tr.posterior.data_vars) >= {"hill", "K"}

    acid_tr = fit_spectra_global_bayes_marginal(
        x,
        {"exc": (lam, y)},
        x_sigma=0.05,
        seed=6,
        draws=120,
        chains=1,
        cores=1,
        acid_state=True,
    )
    kd_acid = np.asarray(acid_tr.posterior["K"].values).reshape(-1)
    # the forced third state soaks part of the transition, as it does in the LM
    # twin on the same two-state data: compare against that fit, not ground truth
    lm_acid = fit_spectra_global(x, {"exc": (lam, y)}, acid_state=True, jackknife=False)
    assert np.quantile(kd_acid, 0.03) <= lm_acid.K <= np.quantile(kd_acid, 0.97)
    assert np.asarray(acid_tr.posterior["E2_exc"].values).shape[-1] == lam.size


@pytest.mark.slow
def test_global_bayes_marginal_two_scans_share_k() -> None:
    """Two scans tie one K and keep separate sigma/offset scales."""
    rng = np.random.default_rng(52)
    x, _lam, y, k_true = _two_state_spectra(noise=0.004)
    e0b = np.linspace(0.7, 0.3, 10)
    e1b = np.linspace(0.15, 0.85, 10)
    frac = 1 / (1 + 10 ** (k_true - x))
    y2 = (e0b[:, None] * (1 - frac) + e1b[:, None] * frac) * 0.8
    y2 *= np.repeat([0.95, 1.05], 8)
    y2 += rng.normal(0, 0.01, y2.shape)
    trace = fit_spectra_global_bayes_marginal(
        x,
        {
            "exc": (np.arange(y.shape[0], dtype=np.float64), y),
            "em": (np.arange(y2.shape[0], dtype=np.float64), y2),
        },
        x_sigma=0.05,
        seed=2,
        draws=120,
        chains=1,
        cores=1,
    )
    kdraw = np.asarray(trace.posterior["K"].values).reshape(-1)
    assert np.quantile(kdraw, 0.03) <= k_true <= np.quantile(kdraw, 0.97)
    assert {"sigma_exc", "sigma_em", "offset_exc", "offset_em"} <= set(
        trace.posterior.data_vars
    )


@pytest.mark.slow
def test_cli_global_bayes_marginal_writes_posterior_outputs(
    tmp_path: Path, runner: CliRunner
) -> None:
    """The global-bayes-marginal CLI writes K, latent-x, trace, and figure outputs."""
    data = tpath / "EnSpire"
    lines = (data / "NTT-G10_note.csv").read_text().splitlines()
    note = tmp_path / "NTT-G10_note.csv"
    note.write_text("\n".join([lines[0], *(ln for ln in lines[1:] if ",20.0," in ln)]))
    out = tmp_path / "global_bayes_marginal"
    out.mkdir()
    result = runner.invoke(
        ppr,
        [
            "--out",
            str(out),
            "enspire",
            str(data / "G10.csv"),
            str(note),
            "--method",
            "global-bayes-marginal",
            "--x-sigma",
            "auto",
            "--draws",
            "100",
            "--chains",
            "2",
            "--cores",
            "1",
            "--seed",
            "7",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out / "NTT-G10_K_global_bayes_marginal.csv").exists()
    assert (out / "NTT-G10_x_global_bayes_marginal.csv").exists()
    assert (out / "NTT-G10_global_bayes_marginal.pdf").stat().st_size > 1000
    assert list(out.glob("*_global_bayes_marginal.nc"))
    table = pd.read_csv(out / "NTT-G10_K_global_bayes_marginal.csv")
    assert table.K.notna().all()
    assert (table.hdi03 < table.hdi97).all()
