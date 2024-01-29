"""Test prtecan module."""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal

from clophfit import prtecan
from clophfit.binding.fitting import FitResult
from clophfit.prtecan import (
    Labelblock,
    LabelblocksGroup,
    PlateScheme,
    Tecanfile,
    TecanfilesGroup,
    TitrationAnalysis,
)

# By defining csvl, lb0, and lb1 as class attributes, they are created only once
# per test session. Use fixture to capture UserWarning "OVER"

data_tests = Path(__file__).parent / "Tecan"
pytestmark = pytest.mark.filterwarnings("ignore:OVER")


def test_lookup_listoflines() -> None:
    """It returns indexes for pattern match on col in list of lines."""
    csvl: list[list[str | int | float]] = [
        ["pp", "xy", 1, 2.0],
        ["pp", "xx", 1, 2],
        ["pp", 12, 1, 2],
        ["pp", "yy", 1, 2.0],
        ["a"],
        ["pp", "xy", 1, 2],
    ]
    assert prtecan.lookup_listoflines(csvl, pattern="pp") == [0, 1, 2, 3, 5]
    assert not prtecan.lookup_listoflines(csvl, pattern="xy")  # == []
    assert prtecan.lookup_listoflines(csvl, pattern="xy", col=1) == [0, 5]
    assert prtecan.lookup_listoflines(csvl, pattern="yy", col=1) == [3]
    assert not prtecan.lookup_listoflines(csvl, pattern="yy", col=2)  # == []


def test_strip_lines() -> None:
    """It strips empty fields."""
    lines: list[list[float | int | str]] = [
        ["Excitation Wavelength", "", "", "", 485.0, "nm", "", "", ""]
    ]
    stripped = prtecan.strip_lines(lines)
    assert stripped == [["Excitation Wavelength", 485.0, "nm"]]


def test_extract_metadata() -> None:
    """It extracts metadata correctly."""
    lines: list[list[Any]] = [
        ["Label: Label1", "", "", "", "", "", "", "", "", "", "", "", ""],
        ["Mode", "", "", "", "Fluorescence Top Reading", "", "", "", "", ""],
        ["Shaking (Linear) Amplitude:", "", "", "", 2, "mm", "", "", "", "", ""],
        ["Excitation Wavelength", "", "", "", 400, "nm", "", "unexpected", "", "", ""],
        ["", "Temperature: 26 °C", "", "", "", "", "", "", "", "", ""],
    ]
    expected_metadata = {
        "Label": prtecan.Metadata("Label1"),
        "Mode": prtecan.Metadata("Fluorescence Top Reading"),
        "Shaking (Linear) Amplitude:": prtecan.Metadata(2, ["mm"]),
        "Excitation Wavelength": prtecan.Metadata(400, ["nm", "unexpected"]),
        "Temperature": prtecan.Metadata(26.0, ["°C"]),
    }

    metadata = prtecan.extract_metadata(lines)
    assert metadata == expected_metadata


def test_merge_md() -> None:
    """Merge metadata of both labelblocks and tecanfiles."""
    md1 = {
        "Gain": prtecan.Metadata(93, ["Manual"]),
        "Shaking (Linear) Amplitude:": prtecan.Metadata(2, ["mm"]),
    }
    md2 = {
        "Gain": prtecan.Metadata(93, ["Optimal"]),
        "Shaking (Linear) Amplitude:": prtecan.Metadata(2, ["mm"]),
    }
    mmd = prtecan.merge_md([md1, md2])
    assert mmd["Gain"] == prtecan.Metadata(93)
    assert mmd["Shaking (Linear) Amplitude:"] == prtecan.Metadata(2, ["mm"])


def test_calculate_conc() -> None:
    """Calculates concentration values from Cl additions."""
    additions = [112, 2, 2, 2, 2, 2, 2, 6, 4]
    conc = prtecan.calculate_conc(additions, 1000)
    assert_almost_equal(
        conc, [0.0, 17.544, 34.483, 50.847, 66.667, 81.967, 96.774, 138.462, 164.179], 3
    )


@pytest.mark.filterwarnings("ignore:OVER")
class TestLabelblock:
    """Test labelblock class."""

    @staticmethod
    def _get_two_labelblocks() -> tuple[Labelblock, Labelblock]:
        """Simulate csvl with 2 labelblocks."""
        csvl = prtecan.read_xls(data_tests / "140220/pH6.5_200214.xls")
        idxs = prtecan.lookup_listoflines(csvl)
        # pylint: disable=W0201
        lb0 = Labelblock(csvl[idxs[0] : idxs[1]])
        lb1 = Labelblock(csvl[idxs[1] :])
        lb0.buffer_wells = ["D01", "D12", "E01", "E12"]
        lb1.buffer_wells = ["D01", "D12", "E01", "E12"]
        return lb0, lb1

    @pytest.fixture(scope="class")
    def labelblocks(self) -> tuple[Labelblock, Labelblock]:
        """Fixture that provides two labelblocks."""
        return self._get_two_labelblocks()

    def test_metadata(self, labelblocks: tuple[Labelblock, Labelblock]) -> None:
        """It parses "Temperature" metadata."""
        lb0, lb1 = labelblocks
        assert lb0.metadata["Temperature"].value == 25.6
        assert lb1.metadata["Temperature"].value == 25.3

    def test_data(self, labelblocks: tuple[Labelblock, Labelblock]) -> None:
        """It parses data values."""
        lb0, lb1 = labelblocks
        assert lb0.data["F06"] == 19551
        assert lb1.data["H12"] == 543

    def test_data_normalized(self, labelblocks: tuple[Labelblock, Labelblock]) -> None:
        """Normalize data using some metadata values."""
        lb0, lb1 = labelblocks
        assert lb0.data_norm["F06"] == pytest.approx(1051.1290323)
        assert lb1.data_norm["H12"] == pytest.approx(48.4821429)

    def test_data_buffersubtracted(
        self, labelblocks: tuple[Labelblock, Labelblock]
    ) -> None:
        """Calculate buffer value from average of buffer wells and subtract from data."""
        lb0, lb1 = labelblocks
        assert lb0.buffer == 11889.25
        assert lb1.buffer == 56.75
        assert lb0.buffer_sd == pytest.approx(450.2490)
        assert lb1.buffer_sd == pytest.approx(4.43706)
        assert lb0.data_buffersubtracted["F06"] == pytest.approx(7661.75)
        assert lb1.data_buffersubtracted["H12"] == pytest.approx(486.25)
        # Can also assign a buffer value.
        lb0.buffer = 1
        lb1.buffer = 2.9
        assert lb0.data_buffersubtracted["F06"] == 19550
        assert lb1.data_buffersubtracted["H12"] == 540.1

    def test_data_buffersubtracted_norm(
        self, labelblocks: tuple[Labelblock, Labelblock]
    ) -> None:
        """Calculate normalized buffer value from average of buffer wells and subtract from data."""
        lb0, lb1 = labelblocks
        assert lb0.buffer_norm == pytest.approx(639.20699)
        assert lb1.buffer_norm == pytest.approx(5.06696)
        assert lb0.buffer_norm_sd == pytest.approx(24.20694)
        assert lb1.buffer_norm_sd == pytest.approx(0.396166)
        assert lb0.data_buffersubtracted_norm["F06"] == pytest.approx(411.922)
        assert lb1.data_buffersubtracted_norm["H12"] == pytest.approx(43.4152)
        # Can also assign a buffer_norm value.
        lb0.buffer_norm = 1
        lb1.buffer_norm = 0.4821
        assert lb0.data_buffersubtracted_norm["F06"] == pytest.approx(1050.13)
        assert lb1.data_buffersubtracted_norm["H12"] == pytest.approx(48.0)

    def test_eq(self, labelblocks: tuple[Labelblock, Labelblock]) -> None:
        """Check if a Labelblock is equal to itself and not equal to a different Labelblock."""
        lb0, lb1 = labelblocks
        assert lb0 == lb0, "Labelblock is not equal to itself"  # noqa: PLR0124
        assert lb0 != lb1, "Different Labelblocks are incorrectly reported as equal"
        assert (
            lb0.__eq__(1) == NotImplemented
        ), "Equality check against non-Labelblock object did not return NotImplemented"

    def test_almost_eq(self, labelblocks: tuple[Labelblock, Labelblock]) -> None:
        """Test the __almost_eq__ method of the Labelblock class."""
        lb0, _ = labelblocks
        file_path1 = Path(data_tests) / "290513_7.2.xls"
        csvl1 = prtecan.read_xls(file_path1)  # Gain=98
        idxs1 = prtecan.lookup_listoflines(csvl1)
        lb11 = Labelblock(csvl1[idxs1[1] :])
        file_path2 = Path(data_tests) / "290513_8.8.xls"
        csvl2 = prtecan.read_xls(file_path2)  # Gain=99
        idxs2 = prtecan.lookup_listoflines(csvl2)
        lb12 = Labelblock(csvl2[idxs2[1] :])
        assert lb11 != lb12
        assert lb11.__almost_eq__(lb12)
        assert not lb11.__almost_eq__(lb0)

    def test_overvalue(self) -> None:
        """It detects saturated data ("OVER")."""
        csvl = prtecan.read_xls(data_tests / "140220/pH6.5_200214.xls")
        idxs = prtecan.lookup_listoflines(csvl)
        with pytest.warns(
            UserWarning, match=r"OVER\n Overvalue in Label1:A06 of tecanfile "
        ):
            lb = Labelblock(csvl[idxs[0] : idxs[1]])
        assert np.nansum(lb.data["A06"]) == np.nansum(np.nan)
        assert np.nansum(lb.data["H02"]) == np.nansum(np.nan)

    def test_raise_missing_column(self) -> None:
        """It raises Exception when a column is missing from the labelblock."""
        csvl = prtecan.read_xls(data_tests / "exceptions/88wells_290212_20.xlsx")
        idxs = prtecan.lookup_listoflines(csvl)
        with pytest.raises(ValueError, match=r"Cannot build Labelblock: not 96 wells?"):
            Labelblock(csvl[idxs[0] : len(csvl)])

    def test_raise_missing_row(self) -> None:
        """It raises Exception when a row is missing from the labelblock."""
        csvl = prtecan.read_xls(data_tests / "exceptions/84wells_290212_20.xlsx")
        idxs = prtecan.lookup_listoflines(csvl)
        with pytest.raises(
            ValueError, match="Cannot extract data in Labelblock: not 96 wells?"
        ):
            Labelblock(csvl[idxs[0] : len(csvl)])


class TestCsvlFunctions:
    """Test TecanFile reading and parsing functions."""

    csvl = prtecan.read_xls(data_tests / "140220/pH8.3_200214.xls")

    def test_read_xls(self) -> None:
        """The test reads the xls file using cls method."""
        assert len(self.csvl) == 74

    def test_lookup_listoflines(self) -> None:
        """It finds Label occurrences using module function."""
        assert prtecan.lookup_listoflines(self.csvl) == [14, 44]


class TestTecanfile:
    """Test TecanFile class."""

    tf = prtecan.Tecanfile(data_tests / "140220/pH8.3_200214.xls")

    def test_path(self) -> None:
        """It reads the file path."""
        assert self.tf.path == data_tests / "140220/pH8.3_200214.xls"

    def test_metadata(self) -> None:
        """It parses the Date."""
        assert self.tf.metadata["Date:"].value == "20/02/2014"

    def test_labelblocks(self) -> None:
        """It parses "Temperature" metadata and cell data from 2 labelblocks."""
        assert self.tf.labelblocks[0].metadata["Temperature"].value == 25.3
        assert self.tf.labelblocks[1].metadata["Temperature"].value == 25.7
        assert self.tf.labelblocks[0].data["A01"] == 17260
        assert self.tf.labelblocks[1].data["H12"] == 4196

    def test_eq(self) -> None:
        """Check if a Tecanfile is equal to itself and not equal to a different Tecanfile."""
        tf1 = prtecan.Tecanfile(data_tests / "140220/pH8.3_200214.xls")
        assert self.tf == tf1, "Tecanfile is not equal to itself"
        tf2 = prtecan.Tecanfile(data_tests / "140220/pH9.1_200214.xls")
        assert self.tf != tf2, "Different Tecanfiles are incorrectly reported as equal"

    def test_warn(self) -> None:
        """It warns when labelblocks are repeated in a Tf as it might compromise grouping."""
        with pytest.warns(UserWarning, match="Repeated labelblocks"):
            prtecan.Tecanfile(data_tests / "exceptions/290212_7.67_repeated_lb.xls")

    def test_filenotfound(self) -> None:
        """It raises FileNotFoundError when the file path does not exist."""
        with pytest.raises(FileNotFoundError):
            prtecan.Tecanfile(Path("pinocchio"))

    def test_missing_label(self) -> None:
        """It raises Exception when there is no Label pattern."""
        with pytest.raises(ValueError, match="No Labelblock found."):
            prtecan.Tecanfile(data_tests / "exceptions/0_Labelblocks_290513_5.5.xlsx")


class TestLabelblocksGroup:
    """Test LabelBlocksGroup class."""

    @pytest.fixture(autouse=True, scope="class")
    def tfs(self) -> list[Tecanfile]:
        """Set up list of Tecanfile."""
        return [
            Tecanfile(data_tests / "290513_5.5.xls"),
            Tecanfile(data_tests / "290513_7.2.xls"),
            Tecanfile(data_tests / "290513_8.8.xls"),
        ]

    @pytest.fixture(autouse=True, scope="class")
    def lbgs(self, tfs: list[Tecanfile]) -> tuple[LabelblocksGroup, LabelblocksGroup]:
        """Set up LabelblocksGroup 0 and 1."""
        lbg0 = LabelblocksGroup([tfs[0].labelblocks[0], tfs[1].labelblocks[0]])
        lbg1 = LabelblocksGroup([tfs[1].labelblocks[1], tfs[2].labelblocks[1]])
        lbg0.buffer_wells = ["C12", "D01", "D12", "E01", "E12", "F01"]
        lbg1.buffer_wells = ["C12", "D01", "D12", "E01", "E12", "F01"]
        return lbg0, lbg1

    def test_metadata(self, lbgs: tuple[LabelblocksGroup, LabelblocksGroup]) -> None:
        """Merge only shared metadata."""
        assert lbgs[0].metadata.get("Temperature") is None
        assert lbgs[1].metadata.get("Temperature") is None
        assert lbgs[1].metadata.get("Gain") is None
        assert lbgs[1].labelblocks[0].metadata["Gain"].value == 98
        assert lbgs[1].labelblocks[1].metadata["Gain"].value == 99
        # Common metadata.
        assert lbgs[0].metadata["Gain"].value == 94
        assert lbgs[0].metadata["Number of Flashes"].value == 10

    def test_data(self, lbgs: tuple[LabelblocksGroup, LabelblocksGroup]) -> None:
        """Merge data."""
        assert lbgs[0].data is not None
        assert lbgs[0].data["A01"] == [18713, 17088]
        assert lbgs[0].data["H12"] == [28596, 25771]
        assert lbgs[1].data is None

    def test_data_normalized(
        self, lbgs: tuple[LabelblocksGroup, LabelblocksGroup]
    ) -> None:
        """Merge data_normalized."""
        assert_almost_equal(lbgs[1].data_norm["H12"], [693.980, 714.495], 3)
        assert_almost_equal(lbgs[0].data_norm["A01"], [995.372, 908.936], 3)

    def test_data_buffersubtracted(
        self, lbgs: tuple[LabelblocksGroup, LabelblocksGroup]
    ) -> None:
        """Merge data_buffersubtracted."""
        assert lbgs[0].data_buffersubtracted is not None
        assert_almost_equal(lbgs[0].data_buffersubtracted["B07"], [7069, 5716.7], 1)
        assert lbgs[1].data_buffersubtracted is None

    def test_data_buffersubtracted_norm(
        self, lbgs: tuple[LabelblocksGroup, LabelblocksGroup]
    ) -> None:
        """Merge data_buffersubtracted."""
        assert_almost_equal(
            lbgs[0].data_buffersubtracted_norm["B07"], [376.01, 304.08], 2
        )
        assert_almost_equal(
            lbgs[1].data_buffersubtracted_norm["B07"], [355.16, 348.57], 2
        )

    def test_notequal_labelblocks(self, tfs: list[Tecanfile]) -> None:
        """Raise Exception when concatenating unequal labelblocks."""
        with pytest.raises(ValueError, match="Creation of labelblock group failed."):
            prtecan.LabelblocksGroup([tfs[1].labelblocks[0], tfs[2].labelblocks[1]])


class TestTecanfileGroup:
    """Group tecanfiles properly."""

    class TestAllEqLbgs:
        """Test TfG with 2 LbG in the same order."""

        @pytest.fixture(autouse=True, scope="class")
        def tfg(self) -> TecanfilesGroup:
            """Set up TecanfilesGroup."""
            filenames = ["290513_5.5.xls", "290513_7.2.xls"]
            tecanfiles = [Tecanfile(data_tests / f) for f in filenames]
            return TecanfilesGroup(tecanfiles)

        def test_metadata(self, tfg: TecanfilesGroup) -> None:
            """Parse general metadata."""
            assert tfg.metadata["Plate"].value == "PE 96 Flat Bottom White   [PE.pdfx]"
            assert tfg.metadata["System"].value == "TECANROBOT"

        def test_labelblocksgroups(self, tfg: TecanfilesGroup) -> None:
            """Generate 2 LbG with .data and .metadata."""
            lbg0 = tfg.labelblocksgroups[0]
            lbg1 = tfg.labelblocksgroups[1]
            # metadata
            assert lbg0.metadata["Number of Flashes"].value == 10.0
            assert lbg1.metadata["Gain"].value == 98.0
            # data normalized ... enough in lbg
            # data
            assert lbg0.data is not None
            assert lbg0.data["A01"] == [18713, 17088]
            assert lbg0.data["H12"] == [28596, 25771]
            assert lbg1.data is not None
            assert lbg1.data["A01"] == [7878, 8761]
            assert lbg1.data["H12"] == [14226, 13602]

    class TestAlmostEqLbgs:
        """Test TfG when 1 LbG equal and a second with almost equal labelblocks."""

        @pytest.fixture(autouse=True, scope="class")
        def tfg_warn(self) -> tuple[TecanfilesGroup, pytest.WarningsRecorder]:
            """Set up TecanfilesGroup with Warning."""
            filenames = [
                "290513_5.5.xls",  # Label1 and Label2
                "290513_7.2.xls",  # Label1 and Label2
                "290513_8.8.xls",  # Label1 and Label2 with different metadata
            ]
            tecanfiles = [Tecanfile(data_tests / f) for f in filenames]
            with pytest.warns(UserWarning) as record:
                tfg = TecanfilesGroup(tecanfiles)
            return tfg, record

        @pytest.fixture(autouse=True, scope="class")
        def tfg(
            self, tfg_warn: tuple[TecanfilesGroup, pytest.WarningsRecorder]
        ) -> TecanfilesGroup:
            """Extract TecanfilesGroup."""
            return tfg_warn[0]

        def test_warn(
            self, tfg_warn: tuple[TecanfilesGroup, pytest.WarningsRecorder]
        ) -> None:
            """Warn about labelblocks anomaly."""
            msg_str = str(tfg_warn[1][0].message)
            assert "Different LabelblocksGroup among filenames" in msg_str

        def test_labelblocksgroups(self, tfg: TecanfilesGroup) -> None:
            """Generate 1 LbG with .data and .metadata."""
            lbg0 = tfg.labelblocksgroups[0]
            # metadata
            assert lbg0.metadata["Number of Flashes"].value == 10.0
            assert lbg0.metadata["Gain"].value == 94
            # data
            assert lbg0.data is not None
            assert lbg0.data["A01"] == [18713.0, 17088.0, 17123.0]
            assert lbg0.data["H12"] == [28596.0, 25771.0, 28309.0]

        def test_mergeable_labelblocksgroups(self, tfg: TecanfilesGroup) -> None:
            """Generate 1 Lbg only with .data_normalized and only common .metadata."""
            lbg1 = tfg.labelblocksgroups[1]
            # metadata
            assert lbg1.metadata["Number of Flashes"].value == 10.0
            assert lbg1.metadata.get("Gain") is None
            assert lbg1.data is None
            # data_normalized
            assert_almost_equal(
                lbg1.data_norm["A01"], [401.9387755, 446.9897959, 450.0]
            )
            assert_almost_equal(
                lbg1.data_norm["H12"], [725.8163265, 693.9795918, 714.4949494]
            )

    class TestOnly1commonLbg:
        """Test TfG with different number of labelblocks, but mergeable."""

        @pytest.fixture(autouse=True, scope="class")
        def tfg_warn(self) -> tuple[TecanfilesGroup, pytest.WarningsRecorder]:
            """Set up TecanfilesGroup with Warning."""
            filenames = [
                "290212_5.78.xls",  # Label1 and Label2
                "290212_20.xls",  # Label2 only
                "290212_100.xls",  # Label2 only
            ]
            tecanfiles = [Tecanfile(data_tests / f) for f in filenames]
            with pytest.warns(UserWarning) as record:
                tfg = TecanfilesGroup(tecanfiles)
            return tfg, record

        @pytest.fixture(autouse=True, scope="class")
        def tfg(
            self, tfg_warn: tuple[TecanfilesGroup, pytest.WarningsRecorder]
        ) -> TecanfilesGroup:
            """Extract TecanfilesGroup."""
            return tfg_warn[0]

        def test_warn(
            self, tfg_warn: tuple[TecanfilesGroup, pytest.WarningsRecorder]
        ) -> None:
            """Warn about labelblocks anomaly."""
            msg_str = str(tfg_warn[1][0].message)
            assert "Different LabelblocksGroup among filenames" in msg_str

        def test_labelblocksgroups(self, tfg: TecanfilesGroup) -> None:
            """Generates 1 LbG with .data and .metadata."""
            lbg = tfg.labelblocksgroups[0]
            # metadata
            assert lbg.metadata["Number of Flashes"].value == 10.0
            assert lbg.metadata["Gain"].value == 93.0
            # data
            assert lbg.data is not None
            assert lbg.data["A01"] == [6289, 6462, 6465]
            assert lbg.data["H12"] == [4477, 4705, 4918]

    class TestFailToMerge:
        """Test TfG without mergeable labelblocks."""

        filenames: ClassVar[list[str]] = ["290513_5.5.xls", "290513_5.5_bad.xls"]
        tecanfiles: ClassVar[list[prtecan.Tecanfile]] = [
            prtecan.Tecanfile(data_tests / f) for f in filenames
        ]

        def test_raise_exception(self) -> None:
            """Raise Exception when there is no way to build labelblocksGroup."""
            with pytest.raises(
                ValueError,
                match=r"No common labelblock in filenames: .*290513_5.5_bad.xls",
            ):
                prtecan.TecanfilesGroup(self.tecanfiles)


class TestTitration:
    """Test Titration class."""

    tit_ph = prtecan.Titration.fromlistfile(data_tests / "list.pH", is_ph=True)
    tit_cl = prtecan.Titration.fromlistfile(data_tests / "list.cl20", is_ph=False)

    @pytest.mark.filterwarnings("ignore: Different LabelblocksGroup")
    def test_conc(self) -> None:
        """It reads pH values."""
        assert_array_equal(
            self.tit_ph.conc, [5.78, 6.38, 6.83, 7.24, 7.67, 8.23, 8.82, 9.31]
        )

    def test_labelblocksgroups(self) -> None:
        """It reads labelblocksgroups data and metadata."""
        lbg0 = self.tit_ph.labelblocksgroups[0]
        lbg1 = self.tit_ph.labelblocksgroups[1]
        # metadata
        assert lbg0.metadata["Number of Flashes"].value == 10.0
        # pH9.3 is 93 Optimal not Manual
        assert lbg1.metadata["Gain"] == prtecan.Metadata(93.0)
        # data
        assert lbg0.data is not None
        assert lbg1.data is not None
        assert lbg0.data["A01"][::2] == [30344, 31010, 33731, 37967]
        assert lbg1.data["A01"][1::2] == [9165, 15591, 20788, 22534]
        assert lbg0.data["H12"][1::2] == [20888, 21711, 23397, 25045]
        assert lbg1.data["H12"] == [4477, 5849, 7165, 8080, 8477, 8822, 9338, 9303]

    def test_labelblocksgroups_cl(self) -> None:
        """It reads labelblocksgroups data for Cl too."""
        lbg = self.tit_cl.labelblocksgroups[0]
        assert lbg.data is not None
        assert lbg.data["A01"] == [6462, 6390, 6465, 6774]
        assert lbg.data["H12"] == [4705, 4850, 4918, 5007]

    def test_export_data(self, tmp_path: Path) -> None:
        """It exports titrations data to files e.g. "A01.dat"."""
        self.tit_ph.export_data(tmp_path)
        a01 = pd.read_csv(tmp_path / "dat/A01.dat")
        h12 = pd.read_csv(tmp_path / "dat/H12.dat")
        assert a01["y1"].tolist()[1::2] == [30072, 32678, 36506, 37725]
        assert a01["y2"].tolist()[1::2] == [9165, 15591, 20788, 22534]
        assert h12["y1"].tolist()[1::2] == [20888, 21711, 23397, 25045]
        assert h12["y2"].tolist() == [4477, 5849, 7165, 8080, 8477, 8822, 9338, 9303]

    def test_raise_listfilenotfound(self) -> None:
        """It raises FileNotFoundError when list.xx file does not exist."""
        with pytest.raises(FileNotFoundError, match="Cannot find: aax"):
            prtecan.Titration.fromlistfile(Path("aax"), True)

    def test_bad_listfile(self) -> None:
        """It raises Exception when list.xx file is ill-shaped."""
        with pytest.raises(ValueError, match=r"Check format .* for listfile: .*"):
            prtecan.Titration.fromlistfile(data_tests / "list.pH2", True)


class TestPlateScheme:
    """Test PlateScheme."""

    @pytest.fixture(autouse=True, scope="class")
    def ps(self) -> PlateScheme:
        """Create a void PlateScheme."""
        return PlateScheme()

    def test_buffer(self, ps: PlateScheme) -> None:
        """Set buffer and test raise error."""
        ps.buffer = ["A1", "A2"]
        assert ps.buffer == ["A1", "A2"]
        with pytest.raises(TypeError):
            ps.buffer = [1, 2]  # type: ignore

    def test_ctrl(self, ps: PlateScheme) -> None:
        """Set ctrl and test raise error."""
        ps.ctrl = ["B1", "B2"]
        assert ps.ctrl == ["B1", "B2"]
        with pytest.raises(TypeError):
            ps.ctrl = [1, 2]  # type: ignore

    def test_names(self, ps: PlateScheme) -> None:
        """Set names and test raise error."""
        ps.names = {"name1": {"A1", "A2"}, "name2": {"B1", "B2"}}
        assert ps.names == {"name1": {"A1", "A2"}, "name2": {"B1", "B2"}}
        with pytest.raises(TypeError):
            ps.names = {"name1": [1, 2], "name2": [3, 4]}  # type: ignore

    def test_invalid_file(self) -> None:
        """Test providing an incorrect file."""
        with pytest.raises(FileNotFoundError):
            PlateScheme(file=Path("incorrect_file.csv"))


@pytest.mark.filterwarnings("ignore:OVER")
class TestTitrationAnalysis:
    """Test TitrationAnalysis class."""

    @pytest.fixture(autouse=True, scope="class")
    def titan(self) -> TitrationAnalysis:
        """Set up the TitrationAnalysis."""
        titan = prtecan.TitrationAnalysis.fromlistfile(
            data_tests / "140220/list.pH", is_ph=True
        )
        titan.load_additions(data_tests / "140220/additions.pH")
        titan.load_scheme(data_tests / "140220/scheme.txt")
        return titan

    def test_scheme(self, titan: TitrationAnalysis) -> None:
        """It finds well position for buffer samples."""
        assert titan.scheme.buffer == ["D01", "E01", "D12", "E12"]

    def test_raise_listfilenotfound(self, titan: TitrationAnalysis) -> None:
        """It raises OSError when scheme file does not exist."""
        with pytest.raises(
            FileNotFoundError, match=r"No such file or directory: 'aax'"
        ):
            titan.load_scheme(Path("aax"))

    def test_raise_listfile_exception(self, titan: TitrationAnalysis) -> None:
        """It raises AssertionError when scheme.txt file is ill-shaped."""
        bad_schemefile = data_tests / "140220/scheme0.txt"
        msg = f"Check format [well sample] for schemefile: {bad_schemefile}"
        with pytest.raises(ValueError, match=re.escape(msg)):
            titan.load_scheme(bad_schemefile)

    def test_subtract_bg(self, titan: TitrationAnalysis) -> None:
        """It subtracts buffer average values."""
        lbg0 = titan.labelblocksgroups[0]
        lbg1 = titan.labelblocksgroups[1]
        assert_almost_equal(
            lbg0.data_norm["E01"][::2], [601.72, 641.505, 674.355, 706.774], 3
        )
        assert lbg0.data is not None
        assert lbg0.data["E01"][::2] == [11192.0, 11932.0, 12543.0, 13146.0]
        assert lbg0.data_buffersubtracted is not None
        assert_array_equal(
            lbg0.data_buffersubtracted["A12"][::3], [8084.5, 16621.75, 13775.0]
        )
        assert lbg1.data_buffersubtracted is not None
        assert_array_equal(
            lbg1.data_buffersubtracted["A12"][::3], [9758.25, 1334.0, 283.5]
        )

    def test_dilution_correction(self, titan: TitrationAnalysis) -> None:
        """It applies dilution correction read from file listing additions."""
        assert titan.additions is not None
        assert_array_equal(titan.additions, [100, 2, 2, 2, 2, 2, 2])
        assert titan.data is not None
        assert titan.data[1] is not None
        assert_almost_equal(
            titan.data[1]["A12"],
            [9758.25, 7524.795, 3079.18, 1414.04, 641.79, 402.325, 317.52],
        )

    def test_data_nrm(self, titan: TitrationAnalysis) -> None:
        """It normalizes data."""
        assert titan.data_nrm is not None
        assert_almost_equal(
            titan.data_nrm[0]["A12"][::2],
            [434.65, 878.73, 975.58, 829.46],
            2,
        )
        assert_almost_equal(
            titan.data_nrm[1]["A12"][::2],
            [871.272, 274.927, 57.303, 28.35],
            3,
        )

    def test_keys(self, titan: TitrationAnalysis) -> None:
        """It gets well positions for ctrl and unknown samples."""
        assert set(titan.scheme.names) == {"NTT", "G03", "V224Q", "S202N"}
        x = {"B12", "H12", "F01", "C12", "F12", "C01", "H01", "G12", "B01", "G01"}
        assert set(titan.scheme.ctrl) - {"A01", "A12"} == x

    """
    @pytest.mark.skipif(sys.platform == "win32", reason="broken on windows")
    """

    def test_fit(self, titan: TitrationAnalysis) -> None:
        """It fits each label separately."""
        titan.fitdata_params = {"bg": True, "nrm": True, "dil": True}
        with warnings.catch_warnings():
            # Suppress the UserWarning related to dataset removal
            warnings.simplefilter("ignore", category=UserWarning)
            fres = titan.results
        # Check that the first fit result dictionary has 92 elements
        assert len(fres[0]) == 92
        # Check that the first fit result for 'H02' is None
        assert fres[0]["H02"] == FitResult(None, None, None)
        # Check that the second fit result for 'H02' is not None
        assert fres[1]["H02"].is_valid()
        # Check the value and standard error of the 'K' parameter for 'H02' in the second fit result
        assert fres[1]["H02"].result is not None
        k_h02 = fres[1]["H02"].result.params["K"]
        assert k_h02.value == pytest.approx(7.8904, abs=1e-4)
        assert k_h02.stderr == pytest.approx(0.0170, abs=1e-4)
        # Check the value and standard error of the 'K' parameter for 'H02' in the third fit result
        assert fres[2]["H02"].result is not None
        k_h02 = fres[2]["H02"].result.params["K"]
        assert k_h02.value == pytest.approx(7.8904, abs=1e-4)
        assert k_h02.stderr == pytest.approx(0.0169, abs=1e-4)
        # Check the value and standard error of the 'K' parameter for 'E02' in the second fit result
        assert fres[1]["E02"].result is not None
        k_e02 = fres[1]["E02"].result.params["K"]
        assert k_e02.value == pytest.approx(7.9771, abs=1e-4)
        assert k_e02.stderr == pytest.approx(0.0243, abs=1e-4)
        # Check the value and standard error of the 'K' parameter for 'E02' in the third fit result
        assert fres[2]["E02"].result is not None
        k_e02 = fres[2]["E02"].result.params["K"]
        assert k_e02.value == pytest.approx(7.9778, abs=1e-4)
        assert k_e02.stderr == pytest.approx(0.0235, abs=1e-4)
        # Fit up to the second-last data point
        titan.fitkws = TitrationAnalysis.FitKwargs(fin=-1)
        with warnings.catch_warnings():
            # Suppress the UserWarning related to dataset removal
            warnings.simplefilter("ignore", category=UserWarning)
            fres = titan.results
        # Check that the first fit result for 'H02' is still None
        assert fres[0]["H02"] == FitResult(None, None, None)
        # Check the value and standard error of the 'K' parameter for 'H02' in the second fit result, after fitting up to the second-last data point
        assert fres[1]["H02"].result is not None
        k_h02 = fres[1]["H02"].result.params["K"]
        assert k_h02.value == pytest.approx(7.8942, abs=1e-4)
        assert k_h02.stderr == pytest.approx(0.0195, abs=1e-4)
        # Check the value and standard error of the 'K' parameter for 'E02' in the second fit result, after fitting up to the second-last data point
        assert fres[1]["E02"].result is not None
        k_e02 = fres[1]["E02"].result.params["K"]
        assert k_e02.value == pytest.approx(7.9837, abs=1e-4)
        assert k_e02.stderr == pytest.approx(0.0267, abs=1e-4)
