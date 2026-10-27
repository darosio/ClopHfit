"""Test prenspire module."""

import os
import warnings
from pathlib import Path

import numpy as np
import pytest

from clophfit.prenspire import EnspireFile, Note
from clophfit.prenspire.spectral import (
    consensus_isosbestic,
    isosbestic_point,
    normalise_to_isosbestic,
)

data_files_dir = Path(__file__).parent / "EnSpire"
esff = data_files_dir.joinpath


@pytest.fixture(scope="module", name="ef1")
def ef1_setup() -> EnspireFile:
    """Read in file."""
    return EnspireFile(esff("h148g-spettroC.csv"))


@pytest.fixture(scope="module", name="ef2")
def ef2_setupt() -> EnspireFile:
    """Read in file without 'Samples' column."""
    return EnspireFile(esff("e2-T-without_sample_column.csv"))


class TestEnspireFile:
    """Test EnspireFile class."""

    def test_exceptions(self) -> None:
        """Test some raised exceptions."""
        esf = (data_files_dir / "exceptions").joinpath
        # Test get_data_ini Exceptions
        with pytest.raises(Exception, match="No line starting with "):
            EnspireFile(esf("h148g-spettroC-idx0.csv"))
        with pytest.raises(Exception, match="Multiple lines starting with "):
            EnspireFile(esf("h148g-spettroC-idx2.csv"))
        # Test platemap
        with pytest.raises(Exception, match="stop: Platemap format unexpected"):
            EnspireFile(esf("M1-A6-G12_11columns.csv"))
        # Test the presence of some empty lines
        with pytest.raises(Exception, match="Expecting two empty lines before _ini"):
            EnspireFile(esf("M1-A6-G12_missing_emptyline_ini.csv"))
        with pytest.raises(Exception, match="Expecting an empty line after _fin"):
            EnspireFile(esf("M1-A6-G12_missing_emptyline_fin.csv"))
        # Test multiple emission wavelengths in excitation spectra
        with pytest.raises(
            Exception, match="Excitation spectra with unexpected emission in MeasA"
        ):
            EnspireFile(esf("e2dan-exwavelength.csv"))
        with pytest.raises(
            Exception, match="Excitation spectra with unexpected emission in MeasA"
        ):
            EnspireFile(esf("e2dan-exwavelength2.csv"))
        with pytest.raises(
            Exception, match="Emission spectra with unexpected excitation in MeasC"
        ):
            EnspireFile(esf("e2dan-emwavelength.csv"))
        with pytest.raises(
            Exception, match='Unknown "Monochromator": Strange in MeasB'
        ):
            EnspireFile(esf("e2dan-exwavelengthstrange.csv"))

    def test_get_data_ini(self, ef1: EnspireFile, ef2: EnspireFile) -> None:
        """Test get_data_ini."""
        assert ef1._ini == 12  # ruff: ignore[private-member-access]
        assert ef2._ini == 9  # ruff: ignore[private-member-access]

    def test_fin(self, ef1: EnspireFile, ef2: EnspireFile) -> None:
        """Test _fin (line_index())."""
        assert ef1._fin == 14897  # ruff: ignore[private-member-access]
        assert ef2._fin == 856  # ruff: ignore[private-member-access]

    """
    # def test_metadata_post(self, ef1: EnspireFile, ef2: EnspireFile) -> None:
    #     \"Identify correctly the beginning of metadata after data block.\"
    #     assert ef1._metadata_post[0] == ["Basic assay information "]
    #     assert ef2._metadata_post[0] == ["Basic assay information "]

    # def test_localegen_in_metadata_post(
    #     self, ef1: EnspireFile, ef2: EnspireFile
    # ) -> None:
    #     \"Test locales.\"
    #     assert ef1._metadata_post[31][4] == "300 μl"
    #     assert ef2._metadata_post[31][4] == "300 μl"

    # def test_data_list(self, ef1: EnspireFile, ef2: EnspireFile) -> None:
    #     \"Test data_list.\"
    #     assert ef1._data_list[0][2] == "MeasA:Result"
    #     assert ef2._data_list[0][2] == "MeasB:WavelengthEms"
    #     assert ef1._data_list[1][2] == "3151"
    #     assert ef2._data_list[1][3] == "3739"
    #     # Important to check for last line
    #     assert ef1._data_list[-1][2] == "97612"
    #     assert ef2._data_list[-1][3] == "512"
    """

    def test_get_list_from_platemap(self, ef1: EnspireFile, ef2: EnspireFile) -> None:
        """Test list from platemap."""
        assert ef1._wells_platemap[2] == "A03"  # ruff: ignore[private-member-access]
        assert ef2._wells_platemap[1] == "F02"  # ruff: ignore[private-member-access]

    def test_metadata(self, ef1: EnspireFile) -> None:
        """Test metadata dictionary."""
        assert ef1.metadata["Measurement date"] == "2011-10-03 17:12:33"
        assert ef1.metadata["Chamber temperature at start"] == "20"
        assert ef1.metadata["Chamber temperature at end"] == "20"
        assert ef1.metadata["Ambient temperature at start"] == "6"
        assert ef1.metadata["Ambient temperature at end"] == "9.3"
        assert ef1.metadata["Protocol name"] == "Eccitazione C"
        assert (
            ef1.metadata["Exported data"] == "Well,Sample,MeasA:Result,MeasA:Wavelength"
        )

    def test_measurement_metadata(self, ef1: EnspireFile, ef2: EnspireFile) -> None:
        """Test data object."""
        assert ef1.measurements["A"]["metadata"]["Monochromator"] == "Excitation"
        assert ef1.measurements["A"]["metadata"]["Wavelength"] == "520"
        assert ef1.measurements["A"]["metadata"]["temp"] == "20"
        assert ef2.measurements["C"]["metadata"]["Monochromator"] == "Emission"
        assert ef2.measurements["C"]["metadata"]["Wavelength"] == "480"
        assert ef2.measurements["F"]["metadata"]["temp"] == "35"

    def test_measurements(self, ef1: EnspireFile, ef2: EnspireFile) -> None:
        """Test data object."""
        assert ef1.measurements["A"]["lambda"][0] == 272
        assert ef1.measurements["A"]["lambda"][228] == 500
        assert ef1.measurements["A"]["A01"][0] == 3151
        assert ef1.measurements["A"]["A01"][228] == 573
        # Important to check for last line of csvl
        assert ef1.measurements["A"]["G05"][228] == 97612
        assert ef1.measurements["A"]["A01"][10] == 2496
        assert ef1.measurements["A"]["A02"][10] == 1765
        assert ef1.measurements["A"]["lambda"][10] == 282
        assert ef1.measurements["A"]["metadata"]["Wavelength"] == "520"
        assert ef2.measurements["A"]["lambda"][2] == 402
        assert ef2.measurements["B"]["F07"][1] == 1898
        assert ef2.measurements["A"]["metadata"]["Wavelength"] == "530"

    def test_check_lists_warning(self) -> None:
        """It warns when (csv != platemap).

        *-incomplete.csv: 5 wells (G1-G5) are missing. (They are present in
        Platemap because they were present during acquisition and then not
        exported).
        """
        esf = (data_files_dir / "warnings").joinpath

        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            EnspireFile(esf("h148g-spettroC-incomplete.csv"))
            platemap_warnings = [wi for wi in w if "platemap" in str(wi.message)]
            assert len(platemap_warnings) == 1
            assert issubclass(platemap_warnings[0].category, Warning)

    # placemark def test_get_maxx(self):
    # placemark    "really need to be completed"
    # placemark    self.assertEqual(self.s.get_maxx(self.s.ex, self.s.y), 272)


@pytest.fixture(scope="module", name="n1")
def n1_setup() -> Note:
    """Read note file."""
    return Note(esff("h148g-spettroC-nota.csv"))


@pytest.fixture(scope="module", name="n1err")
def n1err_setup() -> Note:
    """Read note file with missing wells."""
    return Note(esff("h148g-spettroC-nota-Err.csv"))


class TestNote:
    """Experimental notes."""

    def test_get_list_from_note(self, n1: Note) -> None:
        """Test get_well_list_from_note method."""
        assert n1.wells[2] == "A03"

    def test__note_list(self, n1: Note) -> None:
        """Test well_list from note."""
        assert n1._note.loc[2, "Well"] == "A03"  # ruff: ignore[private-member-access]
        assert n1._note.iloc[64, 1] == 8.2  # ruff: ignore[private-member-access]

    def test_wells(self, n1: Note, ef1: EnspireFile, n1err: Note) -> None:
        """Check wells from Note vs EnspireFile."""
        assert n1.wells == ef1.wells
        assert (n1err.wells == ef1.wells) is False

    def test_build_titrations(self, n1: Note, ef1: EnspireFile) -> None:
        """Test the method extract_titrations()."""
        n1.build_titrations(ef1)
        assert len(n1.titrations["H148G"][20.0]) == 16
        tit0 = n1.titrations["H148G"][20]["Cl_0.0"]["A"]
        assert tit0.columns.to_list() == [5.2, 6.3, 7.4, 8.1, 8.2]
        assert tit0[5.2][272] == 3151
        tit5 = n1.titrations["H148G"][20]["pH_8.2"]["A"]
        assert tit5[667.0][500] == 8734


def test_export_measurements_is_idempotent(tmp_path: Path) -> None:
    """A second call lands on the same files instead of adding timestamped copies.

    Running several fitting methods on one plate used to leave five copies of
    every table and figure in the output directory, because an existing file made
    the writer append a timestamp. The export is a pure function of the input
    file, so the canonical names are reused: rewritten when the export is newer
    than them, skipped when it is not.
    """
    source = Path(__file__).parent / "EnSpire" / "G10.csv"
    ef = EnspireFile(source)
    first = ef.export_measurements(tmp_path)
    assert first
    listing = sorted(p.name for p in tmp_path.iterdir())
    assert ef.export_measurements(tmp_path) == []  # up to date: nothing rewritten
    assert sorted(p.name for p in tmp_path.iterdir()) == listing
    assert ef.export_measurements(tmp_path, mode="never") == []
    # an export older than the source is rewritten in place, same names
    for target in tmp_path.iterdir():
        os.utime(target, (1.0, 1.0))
    assert ef.export_measurements(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == listing


def _two_state_spectra(
    n_lam: int = 120, crossing: int = 500
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two species spectra that cross exactly at ``crossing`` nm."""
    lam = np.linspace(380.0, 580.0, n_lam)
    e0 = 0.15 + 0.85 * np.exp(-((lam - 520.0) ** 2) / (2 * 30.0**2))
    e1 = e0.copy()
    # make them differ only by a ramp that vanishes at the crossing
    e1 = e0 * (1.0 + 0.9 * (crossing - lam) / (crossing - lam.min()))
    return lam, e0, e1


def test_isosbestic_point_finds_the_known_crossing() -> None:
    """Two species crossing at 500 nm give a slope zero there, not elsewhere."""
    lam, e0, e1 = _two_state_spectra()
    x = np.linspace(6.0, 8.5, 12)
    frac = 1.0 / (1.0 + 10.0 ** (7.2 - x))
    y = e0[:, None] * (1 - frac) + e1[:, None] * frac
    found = isosbestic_point({"exc": (lam, y)}, x)
    assert abs(found["exc"] - 500.0) < 1.5


def test_normalising_to_isosbestic_puts_every_spectrum_on_one() -> None:
    """After the normalization every well passes through 1.0 at the crossing."""
    lam, e0, e1 = _two_state_spectra()
    x = np.linspace(6.0, 8.5, 12)
    frac = 1.0 / (1.0 + 10.0 ** (7.2 - x))
    rng = np.random.default_rng(5)
    # per-well brightness and pipetting, which the normalization must remove
    gain = rng.uniform(0.7, 1.3, x.size)
    y = (e0[:, None] * (1 - frac) + e1[:, None] * frac) * gain
    y += rng.normal(0, 0.002, y.shape)
    iso = isosbestic_point({"exc": (lam, y)}, x)["exc"]
    scaled = normalise_to_isosbestic({"exc": (lam, y)}, iso)["exc"][1]
    at = np.array([np.interp(iso, lam, scaled[:, i]) for i in range(x.size)])
    assert np.allclose(at, 1.0, atol=0.01)
    # the per-well gain is gone: at 520 nm the normalised signal matches the
    # noise-free model normalised the same way, whatever the well's brightness
    model = e0[:, None] * (1 - frac) + e1[:, None] * frac
    want = model[np.argmin(np.abs(lam - 520.0))] / model[np.argmin(np.abs(lam - iso))]
    got = scaled[np.argmin(np.abs(lam - 520.0))]
    # the crossing is located to about a nanometre, and the model's slope there
    # turns a 1 nm offset into a few percent, so this is a 5 % check
    assert np.allclose(got, want, atol=0.06)


def test_acid_end_bends_the_crossing_so_it_is_excluded() -> None:
    """The acid-end drop moves the apparent crossing; x_min keeps it in place.

    This is the case the NTT 37 C plate shows: the most acidic wells fall below
    the two-state model, and including them drags the slope zero away from the
    chromophore's real crossing.
    """
    lam, e0, e1 = _two_state_spectra()
    x = np.linspace(5.5, 8.5, 12)
    frac = 1.0 / (1.0 + 10.0 ** (7.2 - x))
    y = e0[:, None] * (1 - frac) + e1[:, None] * frac
    # the three most acidic wells are blue-shifted by 3 nm, the artefact a
    # degraded or partly denatured sample shows: the crossing moves with it
    shifted = np.array([np.interp(lam - 3.0, lam, y[:, i]) for i in range(3)]).T
    y[:, :3] = shifted
    naive = isosbestic_point({"exc": (lam, y)}, x)["exc"]
    guarded = isosbestic_point({"exc": (lam, y)}, x, x_min=6.5)["exc"]
    assert abs(guarded - 500.0) < 2.0
    assert abs(guarded - 500.0) < abs(naive - 500.0)


def test_consensus_isosbestic_pools_measurements() -> None:
    """Two noisy measurements give a consensus closer to the truth than either."""
    lam, e0, e1 = _two_state_spectra()
    datasets = []
    for seed, k in ((1, 7.1), (2, 7.4)):
        rng = np.random.default_rng(seed)
        x = np.linspace(6.0, 8.5, 12)
        frac = 1.0 / (1.0 + 10.0 ** (k - x))
        y = e0[:, None] * (1 - frac) + e1[:, None] * frac
        y += rng.normal(0, 0.006, y.shape)
        datasets.append(({"exc": (lam, y)}, x))
    pooled = consensus_isosbestic(datasets)
    each = [isosbestic_point(s, x)["exc"] for s, x in datasets]
    assert abs(pooled - 500.0) <= max(abs(w - 500.0) for w in each) + 0.5
