"""Band readouts of EnSpire spectra: note dialects, screens and the fitted K."""

from __future__ import annotations

from pathlib import Path

import arviz as az  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
import pytest

from clophfit.fitting.data_structures import DataArray, Dataset
from clophfit.prenspire import EnspireFile, Note, bands
from clophfit.prenspire.bands import (
    _pooled_hdi,  # ruff: ignore[import-private-name]
    band_rho,
    blank_rows,
    classify,
    fit_bands,
    fit_bands_bayes,
    fit_titrations,
    screen_bands,
)

DATA = Path(__file__).parent / "EnSpire"


@pytest.fixture(name="g10")
def fixture_g10() -> EnspireFile:
    """Return the G10 spectra shipped with the tests."""
    return EnspireFile(DATA / "G10.csv")


class TestNoteDialects:
    """Both note layouts have to end up with the same columns."""

    def test_comma_dialect_keeps_its_columns(self) -> None:
        """The 2013 layout already names everything."""
        note = Note(DATA / "NTT-G10_note.csv").note
        assert {"Well", "pH", "Cl", "Name", "Temp"} <= set(note.columns)
        assert note.loc[0, "Well"] == "D01"

    def test_tab_dialect_is_renamed_and_dated(self, tmp_path: Path) -> None:
        """well/Mutant become Well/Name and the file name carries the temperature."""
        path = tmp_path / "NTT-A06_20_note.csv"
        path.write_text(
            "well\tpH\tCl\tMutant\nA01\t9.15\t0\tNTT-A06\nA02\t8.83\t0\tNTT-A06\n"
        )
        note = Note(path).note
        assert list(note.columns[:4]) == ["Well", "pH", "Cl", "Name"]
        assert note.loc[0, "Name"] == "NTT-A06"
        assert note["Temp"].unique().tolist() == [20.0]

    def test_name_falls_back_to_the_file(self, tmp_path: Path) -> None:
        """A note with no sample column is still one sample."""
        path = tmp_path / "sampleX_note.csv"
        path.write_text("Well,pH,Cl\nA01,9.1,0\nA02,8.8,0\n")
        assert Note(path).note["Name"].unique().tolist() == ["sampleX"]


class TestBlankRows:
    """A buffer row is found by level, not by what the note calls it."""

    def test_finds_the_dark_row(self) -> None:
        """One row at the instrument's floor among three bright ones."""
        values = pd.Series({
            f"{row}{col:02d}": (2.0 if row == "D" else 1000.0 * (1 + 0.1 * col))
            for row in "ABCD"
            for col in range(1, 9)
        })
        assert blank_rows(values) == ["D"]

    def test_no_row_is_blank_when_all_are_bright(self) -> None:
        """Rows within a factor of two of each other are all samples."""
        values = pd.Series({
            f"{row}{col:02d}": 1000.0 + 100 * i
            for i, row in enumerate("ABC")
            for col in range(1, 9)
        })
        assert blank_rows(values) == []


class TestBandScreens:
    """Which bands carry a K, and which points inside them do."""

    def test_rho_is_taken_within_rows(self) -> None:
        """Two monotone rows at different levels still score ~1 together."""
        x = np.tile(np.arange(8, dtype=float), 2)
        wells = [f"A{i + 1:02d}" for i in range(8)] + [
            f"B{i + 1:02d}" for i in range(8)
        ]
        y = np.concatenate([np.arange(8, dtype=float), np.arange(8, dtype=float) + 20])
        assert band_rho(x, y, wells) == pytest.approx(1.0)

    def test_a_flat_band_is_rejected(self) -> None:
        """A band that does not move with the titrant carries no K."""
        rng = np.random.default_rng(3)
        x = np.linspace(5.0, 9.0, 10)
        wells = [f"A{i + 1:02d}" for i in range(10)]
        data = {
            "exc_anionic": (x, 1 / (1 + 10 ** (7.0 - x))),
            "exc_neutral": (x, rng.normal(1.0, 0.01, 10)),
        }
        keep, rejected = screen_bands(data, wells)
        assert keep == ["exc_anionic"]
        assert "does not titrate" in rejected["exc_neutral"]

    def test_indirect_band_is_reported_not_fitted(self) -> None:
        """em_exc278 titrates but is not a default member of the shared K."""
        x = np.linspace(5.0, 9.0, 10)
        wells = [f"A{i + 1:02d}" for i in range(10)]
        curve = 1 / (1 + 10 ** (7.0 - x))
        keep, rejected = screen_bands(
            {"exc_anionic": (x, curve), "em_exc278": (x, curve)}, wells
        )
        assert keep == ["exc_anionic"]
        assert "indirect readout" in rejected["em_exc278"]


class TestFitBands:
    """The shared-K fit and its point screen."""

    def test_recovers_a_known_pka(self) -> None:
        """Two bands moving in opposite directions pin one pKa."""
        x = np.linspace(5.0, 9.0, 12)
        frac = 1 / (1 + 10 ** (7.3 - x))
        fit = fit_bands({"a": (x, frac), "b": (x, 1 - frac)}, is_ph=True, screen=None)
        assert fit is not None
        assert fit.k == pytest.approx(7.3, abs=1e-3)

    def test_the_screen_removes_a_planted_outlier(self) -> None:
        """One point put where no titration would put it is dropped."""
        x = np.linspace(5.0, 9.0, 12)
        y = 1 / (1 + 10 ** (7.3 - x))
        y[3] = 0.95  # a well that did not receive its buffer
        with_screen = fit_bands({"a": (x, y)}, is_ph=True)
        without = fit_bands({"a": (x, y)}, is_ph=True, screen=None)
        assert with_screen is not None
        assert without is not None
        assert with_screen.n_dropped == 1
        assert without.n_dropped == 0
        assert abs(with_screen.k - 7.3) < abs(without.k - 7.3)

    def test_too_few_points_returns_none(self) -> None:
        """A band with fewer points than parameters is unfitted, not fatal.

        fit_binding_glob raises InsufficientDataError when n < parameters;
        one such band (a temperature with a nearly empty note) must not
        abort the folder's remaining fits.
        """
        x = np.array([6.0, 7.0])
        y = np.array([0.5, 0.6])
        assert fit_bands({"a": (x, y)}, is_ph=True, screen=None) is None
        xg = np.linspace(5.0, 9.0, 12)
        yg = 1 / (1 + 10 ** (7.3 - xg))
        good = fit_bands({"a": (xg, yg)}, is_ph=True, screen=None)
        assert good is not None  # a full band still fits


class TestTitrations:
    """The whole path, on the spectra shipped with the tests."""

    def test_fits_the_shipped_sample(self, g10: EnspireFile) -> None:
        """G10 titrates at both temperatures, with bands that agree."""
        note = Note(DATA / "NTT-G10_note.csv").note
        fits = fit_titrations(g10, note)
        head = fits.table[
            (fits.table.band == "all (shared K)") & (fits.table.subset == "screened")
        ]
        assert len(head) == 2  # 20 and 37 C
        assert head.K.between(7.0, 9.0).all()
        # the bands of one temperature have to agree; the two temperatures need not
        per_band = fits.table[fits.table.band.isin(["exc_anionic", "em_exc420"])]
        spread = per_band.groupby("temp").K.agg(lambda s: s.max() - s.min())
        assert (spread < 0.3).all()

    def test_classify_picks_the_anionic_window_by_titration_type(
        self, g10: EnspireFile
    ) -> None:
        """The anionic band uses a different window for pH and chloride.

        A pH titration's own reproducibility is best served by ANIONIC
        (480-495 nm); a chloride titration's actual signal - 98% of the
        fractional change between its extremes - sits at 426 nm, which
        ANIONIC barely reaches, so it uses the wider ANIONIC_CL instead.
        """
        ph_readouts = classify(g10, "20", is_ph=True)
        cl_readouts = classify(g10, "20", is_ph=False)
        anionic_ph = next(r for r in ph_readouts if r.name == "exc_anionic")
        anionic_cl = next(r for r in cl_readouts if r.name == "exc_anionic")
        assert (anionic_ph.lo, anionic_ph.hi) == bands.ANIONIC
        assert (anionic_cl.lo, anionic_cl.hi) == bands.ANIONIC_CL
        assert (anionic_ph.lo, anionic_ph.hi) != (anionic_cl.lo, anionic_cl.hi)

    def test_classify_reads_the_labels(self, g10: EnspireFile) -> None:
        """Every label is placed from its own metadata."""
        readouts = classify(g10, "20")
        assert {r.name for r in readouts} == {
            "exc_anionic",
            "exc_neutral",
            "em_exc420",
            "em_exc278",
        }


def test_fit_bands_bayes_recovers_k_with_hdi() -> None:
    """The seeded posterior covers the truth and mixes well (ESS, r_hat).

    r_hat needs two or more chains, which this fit has, so it is
    asserted outright: a guard that skipped the assertion whenever the
    diagnostic came back undefined would let a collapsed chain count
    pass unnoticed.
    """
    rng = np.random.default_rng(4)
    x = np.linspace(5.0, 9.5, 24)
    frac = 1.0 / (1.0 + np.power(10.0, 7.4 - x))
    ds = Dataset(
        {
            "exc_anionic": DataArray(x, 0.05 + 0.95 * frac + rng.normal(0, 0.02, 24)),
            "em_exc420": DataArray(x, 0.9 - 0.7 * frac + rng.normal(0, 0.02, 24)),
        },
        is_ph=True,
    )
    bb = fit_bands_bayes(ds, seed=3, draws=600, chains=2)
    assert abs(bb.k - 7.4) < 0.15
    assert bb.hdi03 < 7.4 < bb.hdi97
    assert bb.ess > 200
    assert np.isfinite(bb.rhat)
    assert bb.rhat < 1.1


def test_pooled_hdi_covers_every_chain() -> None:
    """The HDI of a multi-chain fit must span the pooled draws.

    ``az.hdi`` reduces over the last axis only, so a (chain, draw) array
    yields one interval per chain. Taking the first chain's interval
    describes a fraction of the posterior, while the r_hat printed beside it
    describes all of it. The test harness runs one chain under pytest, so
    this seam is exercised directly.
    """
    rng = np.random.default_rng(0)
    sampled = np.vstack([rng.normal(0.0, 1.0, 500), rng.normal(5.0, 1.0, 500)])
    lo, hi = _pooled_hdi(sampled, prob=0.94)
    pooled = np.asarray(az.hdi(sampled.ravel(), prob=0.94), dtype=float)
    assert (lo, hi) == pytest.approx((pooled[0], pooled[1]))
    first = np.asarray(az.hdi(sampled[0], prob=0.94), dtype=float)
    assert hi > first[1] + 1.0


def test_fit_bands_bayes_reports_an_underdetermined_band_as_unfitted() -> None:
    """A band with fewer points than parameters cannot carry a K.

    The lmfit pre-fit inside the sampler raises InsufficientDataError; it must
    be reported as an unfitted band instead of aborting the whole run.
    """
    x = np.array([6.0, 7.0])
    y = np.array([0.9, 0.5])
    ds = Dataset({"exc_anionic": DataArray(x, y)}, is_ph=True)
    bb = fit_bands_bayes(ds, seed=1, draws=50, chains=1)
    assert np.isnan(bb.k)
    assert np.isnan(bb.hdi03)
    assert np.isnan(bb.hdi97)
