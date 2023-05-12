"""Test prenspire module."""
from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from clophfit.prenspire import EnspireFile
from clophfit.prenspire import ExpNote

data_files_dir = Path(__file__).parent / "EnSpire"
esff = data_files_dir.joinpath


@pytest.fixture(scope="module")
def ef1() -> EnspireFile:
    """Read in file."""
    ef = EnspireFile(esff("h148g-spettroC.csv"))
    ef.extract_measurements()
    return ef


@pytest.fixture(scope="module")
def ef2() -> EnspireFile:
    """Read in file without 'Samples' column."""
    ef = EnspireFile(esff("e2-T-without_sample_column.csv"))
    ef.extract_measurements()
    return ef


class TestEnspireFile:
    """Test EnspireFile class."""

    def test_exceptions(self) -> None:
        """Test some raised exceptions."""
        esf = (data_files_dir / "exceptions").joinpath
        # Test get_data_ini Exceptions
        with pytest.raises(Exception, match="No line starting with ."):
            EnspireFile(esf("h148g-spettroC-idx0.csv"))
        with pytest.raises(Exception, match="Multiple lines starting with ."):
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
            EnspireFile(esf("e2dan-exwavelength.csv")).extract_measurements()
        with pytest.raises(
            Exception, match="Excitation spectra with unexpected emission in MeasA"
        ):
            EnspireFile(esf("e2dan-exwavelength2.csv")).extract_measurements()
        with pytest.raises(
            Exception, match="Emission spectra with unexpected excitation in MeasC"
        ):
            EnspireFile(esf("e2dan-emwavelength.csv")).extract_measurements()
        with pytest.raises(
            Exception, match='Unknown "Monochromator": Strange in MeasB'
        ):
            EnspireFile(esf("e2dan-exwavelengthstrange.csv")).extract_measurements()

    def test_get_data_ini(self, ef1: EnspireFile, ef2: EnspireFile) -> None:
        """Test get_data_ini."""
        assert ef1._ini == 12
        assert ef2._ini == 9

    def test_fin(self, ef1: EnspireFile, ef2: EnspireFile) -> None:
        """Test _fin (line_index())."""
        assert ef1._fin == 14897
        assert ef2._fin == 856

    def test_metadata_post(self, ef1: EnspireFile, ef2: EnspireFile) -> None:
        """Identify correctly the beginning of metadata after data block."""
        assert ef1._metadata_post[0] == ["Basic assay information "]
        assert ef2._metadata_post[0] == ["Basic assay information "]

    def test_localegen_in_metadata_post(
        self, ef1: EnspireFile, ef2: EnspireFile
    ) -> None:
        """Test locales."""
        assert ef1._metadata_post[31][4] == "300 µl"
        assert ef2._metadata_post[31][4] == "300 µl"

    def test_data_list(self, ef1: EnspireFile, ef2: EnspireFile) -> None:
        """Test data_list."""
        assert ef1._data_list[0][2] == "MeasA:Result"
        assert ef2._data_list[0][2] == "MeasB:WavelengthEms"
        assert ef1._data_list[1][2] == "3151"
        assert ef2._data_list[1][3] == "3739"
        # Important to check for last line
        assert ef1._data_list[-1][2] == "97612"
        assert ef2._data_list[-1][3] == "512"

    def test_get_list_from_platemap(self, ef1: EnspireFile, ef2: EnspireFile) -> None:
        """Test list from platemap."""
        assert ef1._well_list_platemap[2] == "A03"
        assert ef2._well_list_platemap[1] == "F02"

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
            ef = EnspireFile(esf("h148g-spettroC-incomplete.csv"))
            ef.extract_measurements()
            assert len(w) == 2
            assert issubclass(w[-1].category, Warning)
            assert "platemap" in str(w[-1].message)

    # placemark def test_get_maxx(self):
    # placemark    "really need to be completed"
    # placemark    self.assertEqual(self.s.get_maxx(self.s.ex, self.s.y), 272)


@pytest.fixture(scope="module")
def en1() -> ExpNote:
    """Read note file."""
    return ExpNote(esff("h148g-spettroC-nota"))


@pytest.fixture(scope="module")
def en1err() -> ExpNote:
    """Read note file with missing wells."""
    return ExpNote(esff("h148g-spettroC-nota-Err"))


class TestExpNote:
    """Experimental notes."""

    def test_get_list_from_note(self, en1: ExpNote) -> None:
        """Test get_well_list_from_note method."""
        assert en1.wells[2] == "A03"

    def test_note_list(self, en1: ExpNote) -> None:
        """Test well_list from note."""
        assert en1.note_list[3][0] == "A03"
        assert en1.note_list[65][1] == "8.2"

    def test_check_list(self, en1: ExpNote, ef1: EnspireFile, en1err: ExpNote) -> None:
        """Test check list from note vs. Enspirefile."""
        assert en1.check_wells(ef1) is True
        assert en1err.check_wells(ef1) is False

    def test_build_titrations(self, en1: ExpNote, ef1: EnspireFile) -> None:
        """Test the method extract_titrations()."""
        en1.build_titrations(ef1)
        assert len(en1.titrations) == 6
        tit0 = en1.titrations[0]
        assert tit0.conc == [5.2, 6.3, 7.4, 8.1, 8.2]
        assert tit0.data["A"][(5.2, "A01")][272] == 3151
        tit5 = en1.titrations[5]
        assert tit5.data["A"][(667.0, "E11")][500] == 8734
