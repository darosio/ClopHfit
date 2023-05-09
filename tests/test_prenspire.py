"""Test prenspire module."""
from __future__ import annotations

from pathlib import Path

import pytest

from clophfit.prenspire import EnspireFile
from clophfit.prenspire import ExpNote

data_files_dir = Path(__file__).parent / "EnSpire"
esff = data_files_dir.joinpath


class TestEnspireFile:
    """Test EnspireFile class."""

    @pytest.fixture(autouse=True)
    def _init(self) -> None:
        """Read in data files."""
        self.esf1 = EnspireFile(esff("h148g-spettroC.csv"))
        self.esf2 = EnspireFile(esff("M1-A6-G12.csv"))
        self.esf3 = EnspireFile(esff("S202N-E2_pHs.csv"))

    def test_exceptions(self) -> None:
        """Test some raised exceptions."""
        # Test get_data_ini Exceptions
        with pytest.raises(Exception, match="No line starting with ."):
            EnspireFile(esff("h148g-spettroC-idx0.csv"))
        with pytest.raises(Exception, match="Multiple lines starting with ."):
            EnspireFile(esff("h148g-spettroC-idx2.csv"))
        # Test platemap
        with pytest.raises(Exception, match="stop: Platemap format unexpected"):
            EnspireFile(esff("M1-A6-G12_11columns.csv"))
        # Test the presence of some empty lines
        with pytest.raises(Exception, match="Expecting two empty lines before _ini"):
            EnspireFile(esff("M1-A6-G12_missing_emptyline_ini.csv"))
        with pytest.raises(Exception, match="Expecting an empty line after _fin"):
            EnspireFile(esff("M1-A6-G12_missing_emptyline_fin.csv"))
        # Test multiple emission wavelebgths in excitation spectra
        with pytest.raises(
            Exception, match="Excitation spectra with unexpected emission in MeasA"
        ):
            EnspireFile(esff("e2dan-exwavelength.csv")).extract_measurements()
        with pytest.raises(
            Exception, match="Excitation spectra with unexpected emission in MeasA"
        ):
            EnspireFile(esff("e2dan-exwavelength2.csv")).extract_measurements()
        with pytest.raises(
            Exception, match="Emission spectra with unexpected excitation in MeasC"
        ):
            EnspireFile(esff("e2dan-emwavelength.csv")).extract_measurements()
        with pytest.raises(
            Exception, match='Unknown "Monochromator": Strange in MeasB'
        ):
            EnspireFile(esff("e2dan-exwavelengthstrange.csv")).extract_measurements()

    def test_get_data_ini(self) -> None:
        """Test get_data_ini."""
        assert self.esf1._ini == 12
        assert self.esf2._ini == 9
        assert self.esf3._ini == 9

    def test_fin(self) -> None:
        """Test _fin (line_index())."""
        # Remember that first line has index=0
        assert self.esf1._fin == 14897
        assert self.esf2._fin == 461
        assert self.esf3._fin == 7233

    def test_metadata_post(self) -> None:
        """Identify correctly the beginning of metadata after data block."""
        assert self.esf1._metadata_post[0] == ["Basic assay information "]
        assert self.esf2._metadata_post[0] == ["Basic assay information "]
        assert self.esf3._metadata_post[0] == ["Basic assay information "]

    def test_localegen_in_metadata_post(self) -> None:
        """Test locales."""
        assert self.esf1._metadata_post[31][4] == "300 µl"
        assert self.esf2._metadata_post[31][4] == "300 µl"
        assert self.esf3._metadata_post[31][4] == "300 µl"

    def test_data_list(self) -> None:
        """Test data_list."""
        assert self.esf1._data_list[0][2] == "MeasA:Result"
        assert self.esf2._data_list[0][2] == "MeasA:WavelengthExc"
        assert self.esf3._data_list[0][2] == "MeasA:Result"
        assert self.esf1._data_list[1][2] == "3151"
        assert self.esf2._data_list[1][4] == "66953"
        assert self.esf3._data_list[1][2] == "7504"
        # Important to check for last line
        assert self.esf1._data_list[-1][2] == "97612"
        assert self.esf2._data_list[-1][4] == "3993"
        assert self.esf3._data_list[-1][2] == "1785"

    def test_get_list_from_platemap(self) -> None:
        """Test list from platemap."""
        assert self.esf1._well_list_platemap[2] == "A03"
        assert self.esf2._well_list_platemap[1] == "H12"
        assert self.esf3._well_list_platemap[2] == "A03"

    def test_metadata(self) -> None:
        """Test metadata dictionary."""
        assert self.esf3.metadata["Measurement date"] == "2013-06-14 23:13:51"
        assert self.esf3.metadata["Chamber temperature at start"] == "19.55"
        assert self.esf3.metadata["Chamber temperature at end"] == "36.5"
        assert self.esf3.metadata["Ambient temperature at start"] == "4.7"
        assert self.esf3.metadata["Ambient temperature at end"] == "8.4"
        assert self.esf1.metadata["Protocol name"] == "Eccitazione C"
        assert (
            self.esf1.metadata["Exported data"]
            == "Well,Sample,MeasA:Result,MeasA:Wavelength"
        )

    def test_measurement_metadata(self) -> None:
        """Test data object."""
        self.esf3.extract_measurements()
        assert self.esf3.measurements["G"]["metadata"]["Monochromator"] == "Excitation"
        assert self.esf3.measurements["G"]["metadata"]["Wavelength"] == "535"
        assert self.esf3.measurements["G"]["metadata"]["temp"] == "37"
        assert self.esf3.measurements["A"]["metadata"]["Monochromator"] == "Emission"
        assert self.esf3.measurements["A"]["metadata"]["Wavelength"] == "278"
        assert self.esf3.measurements["A"]["metadata"]["temp"] == "20"

    def test_measurements(self) -> None:
        """Test data object."""
        self.esf1.extract_measurements()
        assert self.esf1.measurements["A"]["lambda"][0] == 272
        assert self.esf1.measurements["A"]["lambda"][228] == 500
        assert self.esf1.measurements["A"]["A01"][0] == 3151
        assert self.esf1.measurements["A"]["A01"][228] == 573
        # Important to check for last line of csvl
        assert self.esf1.measurements["A"]["G05"][228] == 97612
        assert self.esf1.measurements["A"]["A01"][10] == 2496
        assert self.esf1.measurements["A"]["A02"][10] == 1765
        assert self.esf1.measurements["A"]["lambda"][10] == 282
        assert self.esf1.measurements["A"]["metadata"]["Wavelength"] == "520"
        self.esf2.extract_measurements()
        assert self.esf2.measurements["A"]["lambda"][2] == 272
        assert self.esf2.measurements["A"]["G12"][0] == 66953
        assert self.esf2.measurements["A"]["metadata"]["Wavelength"] == "515"

    # @unittest.skip("demonstrating skipping")
    def test_check_lists_warning(self) -> None:
        """It warns when (csv != platemap).

        *-incomplete.csv: 5 wells (G1-G5) are missing. (They are present in
        Platemap because they were present during acquisition and then not
        exported).

        """
        import warnings

        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            ef = EnspireFile(esff("h148g-spettroC-incomplete.csv"))
            ef.extract_measurements()
            assert len(w) == 2
            assert issubclass(w[-1].category, Warning)
            assert "platemap" in str(w[-1].message)

    # placemark def test_get_maxx(self):
    # placemark    "really need to be completed"
    # placemark    self.assertEqual(self.s.get_maxx(self.s.ex, self.s.y), 272)


class TestExpNote:
    """Experimental notes."""

    @pytest.fixture(autouse=True)
    def _init(self) -> None:
        """Initialize test class."""
        self.ef1 = EnspireFile(esff("h148g-spettroC.csv"))
        self.ef1.extract_measurements()
        self.en1 = ExpNote(esff("h148g-spettroC-nota"))
        self.en2 = ExpNote(esff("M1-A6-G12-nota"))
        self.en3 = ExpNote(esff("S202N-E2_pHs-nota"))
        self.en1err = ExpNote(esff("h148g-spettroC-nota-Err"))

    def test_get_list_from_note(self) -> None:
        """Test get_well_list_from_note method."""
        assert self.en1.wells[2] == "A03"
        assert self.en2.wells[1] == "H12"
        assert self.en3.wells[2] == "A03"

    def test_note_list(self) -> None:
        """Test well_list from note."""
        assert self.en1.note_list[3][0] == "A03"
        assert self.en2.note_list[2][0] == "H12"
        assert self.en1.note_list[65][1] == "8.2"
        assert self.en2.note_list[2][1] == "9.36"

    def test_check_list(self) -> None:
        """Test check list from note vs. Enspirefile."""
        assert self.en1.check_wells(self.ef1) is True
        assert self.en1err.check_wells(self.ef1) is False

    def test_build_titrations(self) -> None:
        """Test the method extract_titrations()."""
        self.en1.build_titrations(self.ef1)
        assert len(self.en1.titrations) == 6
        tit = self.en1.titrations[0]
        assert tit.conc == [5.2, 6.3, 7.4, 8.1, 8.2]
        assert tit.data["A"][(5.2, "A01")][272] == 3151
        tit = self.en1.titrations[5]
        assert tit.data["A"][(667.0, "E11")][500] == 8734
