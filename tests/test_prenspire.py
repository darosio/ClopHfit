"""Test prenspire module."""

import warnings
from pathlib import Path

import pytest

from clophfit.prenspire import EnspireFile, Note

data_files_dir = Path(__file__).parent / "EnSpire"
esff = data_files_dir.joinpath


@pytest.fixture(scope="module")
def ef1() -> EnspireFile:
    """Read in file."""
    ef = EnspireFile(esff("h148g-spettroC.csv"))
    return ef


@pytest.fixture(scope="module")
def ef2() -> EnspireFile:
    """Read in file without 'Samples' column."""
    ef = EnspireFile(esff("e2-T-without_sample_column.csv"))
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
        assert ef1._ini == 12
        assert ef2._ini == 9

    def test_fin(self, ef1: EnspireFile, ef2: EnspireFile) -> None:
        """Test _fin (line_index())."""
        assert ef1._fin == 14897
        assert ef2._fin == 856

    """
    # def test_metadata_post(self, ef1: EnspireFile, ef2: EnspireFile) -> None:
    #     \"Identify correctly the beginning of metadata after data block.\"
    #     assert ef1._metadata_post[0] == ["Basic assay information "]
    #     assert ef2._metadata_post[0] == ["Basic assay information "]

    # def test_localegen_in_metadata_post(
    #     self, ef1: EnspireFile, ef2: EnspireFile
    # ) -> None:
    #     \"Test locales.\"
    #     assert ef1._metadata_post[31][4] == "300 µl"
    #     assert ef2._metadata_post[31][4] == "300 µl"

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
        assert ef1._wells_platemap[2] == "A03"
        assert ef2._wells_platemap[1] == "F02"

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
            assert len(w) == 2
            assert issubclass(w[-1].category, Warning)
            assert "platemap" in str(w[-1].message)

    # placemark def test_get_maxx(self):
    # placemark    "really need to be completed"
    # placemark    self.assertEqual(self.s.get_maxx(self.s.ex, self.s.y), 272)


@pytest.fixture(scope="module")
def n1() -> Note:
    """Read note file."""
    return Note(esff("h148g-spettroC-nota.csv"))


@pytest.fixture(scope="module")
def n1err() -> Note:
    """Read note file with missing wells."""
    return Note(esff("h148g-spettroC-nota-Err.csv"))


class TestNote:
    """Experimental notes."""

    def test_get_list_from_note(self, n1: Note) -> None:
        """Test get_well_list_from_note method."""
        assert n1.wells[2] == "A03"

    def test__note_list(self, n1: Note) -> None:
        """Test well_list from note."""
        assert n1._note.loc[2, "Well"] == "A03"
        assert n1._note.iloc[64, 1] == 8.2

    def test_wells(self, n1: Note, ef1: EnspireFile, n1err: Note) -> None:
        """Check wells from Note vs. EnspireFile."""
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
