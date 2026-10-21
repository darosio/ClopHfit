"""x-latent (per-well pH-axis prior) behaviour of the EnSpire band posterior."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from clophfit.__main__ import ppr
from clophfit.fitting.data_structures import DataArray, Dataset
from clophfit.fitting.models import binding_1site
from clophfit.prenspire.bands import estimate_x_error_sigma, fit_bands_bayes

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


def test_cli_rejects_x_sigma_without_bayes(tmp_path: Path, runner: CliRunner) -> None:
    """--x-sigma is a posterior prior; with an LM method it is a usage error."""
    data = tpath / "EnSpire"
    out = tmp_path / "rej"
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
            "band",
            "--x-sigma",
            "0.15",
        ],
    )
    assert result.exit_code != 0
