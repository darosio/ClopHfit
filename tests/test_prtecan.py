"""Test prtecan module."""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from clophfit import prtecan


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
    np.testing.assert_almost_equal(
        conc,
        [0.0, 17.544, 34.483, 50.847, 66.667, 81.967, 96.774, 138.462, 164.179],
        3,
    )


def test_fit_titration() -> None:
    """It fits pH and Cl titrations."""
    x = [3.0, 5, 7, 9, 11.0]
    y = np.array([1.9991, 1.991, 1.5, 1.009, 1.0009])
    df = prtecan.fit_titration("pH", x, y)
    assert abs(df["K"][0] - 7) < 0.0000000001
    assert abs(df["SA"][0] - 2) < 0.0001
    assert abs(df["SB"][0] - 1) < 0.0001
    x = [0, 5.0, 10, 40, 160, 1000]
    y = np.array([2.0, 1.33333333, 1.0, 0.4, 0.11764706, 0.01980198])
    df = prtecan.fit_titration("Cl", x, y)
    assert abs(df["K"][0] - 10) < 0.0000001
    assert abs(df["SA"][0] - 2) < 0.000000001
    assert abs(df["SB"][0] - 0) < 0.00000001
    with pytest.raises(NameError, match="kind= pH or Cl"):
        prtecan.fit_titration("unk", x, y)


@pytest.mark.filterwarnings("ignore:OVER")
class TestLabelblock:
    """Test labelblock class."""

    @pytest.fixture(autouse=True)
    def _init(self) -> None:
        """Simulate csvl with 2 labelblocks."""
        csvl = prtecan.read_xls(data_tests / "140220/pH6.5_200214.xls")
        idxs = prtecan.lookup_listoflines(csvl)
        lb0 = prtecan.Labelblock(csvl[idxs[0] : idxs[1]])
        lb1 = prtecan.Labelblock(csvl[idxs[1] :])
        lb0.buffer_wells = ["D01", "D12", "E01", "E12"]
        lb1.buffer_wells = ["D01", "D12", "E01", "E12"]
        # pylint: disable=W0201
        self.lb0 = lb0
        self.lb1 = lb1

    def test_metadata(self) -> None:
        """It parses "Temperature" metadata."""
        assert self.lb0.metadata["Temperature"].value == 25.6
        assert self.lb1.metadata["Temperature"].value == 25.3

    def test_data(self) -> None:
        """It parses data values."""
        assert self.lb0.data["F06"] == 19551
        assert self.lb1.data["H12"] == 543

    def test_data_normalized(self) -> None:
        """Normalize data using some metadata values."""
        assert self.lb0.data_norm["F06"] == pytest.approx(1051.1290323)
        assert self.lb1.data_norm["H12"] == pytest.approx(48.4821429)

    def test_data_buffersubtracted(self) -> None:
        """Calculate buffer value from average of buffer wells and subtract from data."""
        assert self.lb0.buffer == 11889.25
        assert self.lb1.buffer == 56.75
        assert self.lb0.sd_buffer == pytest.approx(450.2490)
        assert self.lb1.sd_buffer == pytest.approx(4.43706)
        assert self.lb0.data_buffersubtracted["F06"] == pytest.approx(7661.75)
        assert self.lb1.data_buffersubtracted["H12"] == pytest.approx(486.25)
        # Can also assign a buffer value.
        self.lb0.buffer = 1
        self.lb1.buffer = 2.9
        assert self.lb0.data_buffersubtracted["F06"] == 19550
        assert self.lb1.data_buffersubtracted["H12"] == 540.1

    def test_data_buffersubtracted_norm(self) -> None:
        """Calculate normalized buffer value from average of buffer wells and subtract from data."""
        assert self.lb0.buffer_norm == pytest.approx(639.20699)
        assert self.lb1.buffer_norm == pytest.approx(5.06696)
        assert self.lb0.sd_buffer_norm == pytest.approx(24.20694)
        assert self.lb1.sd_buffer_norm == pytest.approx(0.396166)
        assert self.lb0.data_buffersubtracted_norm["F06"] == pytest.approx(411.922)
        assert self.lb1.data_buffersubtracted_norm["H12"] == pytest.approx(43.4152)
        # Can also assign a buffer_norm value.
        self.lb0.buffer_norm = 1
        self.lb1.buffer_norm = 0.4821
        assert self.lb0.data_buffersubtracted_norm["F06"] == pytest.approx(1050.13)
        assert self.lb1.data_buffersubtracted_norm["H12"] == pytest.approx(48.0)

    def test___eq__(self) -> None:
        """It is equal to itself. TODO and different from other."""
        assert self.lb0 == self.lb0
        assert self.lb0 != self.lb1
        assert self.lb0.__eq__(1) == NotImplemented

    def test___almost_eq__(self) -> None:
        """It is equal to itself. TODO and different from other."""
        csvl = prtecan.read_xls(data_tests / "290513_7.2.xls")  # Gain=98
        idxs = prtecan.lookup_listoflines(csvl)
        lb11 = prtecan.Labelblock(csvl[idxs[1] :])
        csvl = prtecan.read_xls(data_tests / "290513_8.8.xls")  # Gain=99
        idxs = prtecan.lookup_listoflines(csvl)
        lb12 = prtecan.Labelblock(csvl[idxs[1] :])
        assert lb11 != lb12
        assert lb11.__almost_eq__(lb12)
        assert not lb11.__almost_eq__(self.lb0)

    def test_overvalue(self) -> None:
        """It detects saturated data ("OVER")."""
        csvl = prtecan.read_xls(data_tests / "140220/pH6.5_200214.xls")
        idxs = prtecan.lookup_listoflines(csvl)
        with pytest.warns(
            UserWarning, match=r"OVER\n Overvalue in Label1:A06 of tecanfile "
        ):
            lb = prtecan.Labelblock(csvl[idxs[0] : idxs[1]])
        assert np.nansum(lb.data["A06"]) == np.nansum(np.nan)
        assert np.nansum(lb.data["H02"]) == np.nansum(np.nan)

    def test_raise_missing_column(self) -> None:
        """It raises Exception when a column is missing from the labelblock."""
        csvl = prtecan.read_xls(data_tests / "exceptions/88wells_290212_20.xlsx")
        idxs = prtecan.lookup_listoflines(csvl)
        with pytest.raises(ValueError, match=r"Cannot build Labelblock: not 96 wells?"):
            prtecan.Labelblock(csvl[idxs[0] : len(csvl)])

    def test_raise_missing_row(self) -> None:
        """It raises Exception when a row is missing from the labelblock."""
        csvl = prtecan.read_xls(data_tests / "exceptions/84wells_290212_20.xlsx")
        idxs = prtecan.lookup_listoflines(csvl)
        with pytest.raises(
            ValueError, match="Cannot extract data in Labelblock: not 96 wells?"
        ):
            prtecan.Labelblock(csvl[idxs[0] : len(csvl)])


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

    def test___eq__(self) -> None:
        """It is equal to itself. TODO and different from other."""
        assert self.tf == prtecan.Tecanfile(data_tests / "140220/pH8.3_200214.xls")
        tf2 = prtecan.Tecanfile(data_tests / "140220/pH9.1_200214.xls")
        assert tf2 != self.tf

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

    @pytest.fixture(name="tfs", scope="class")
    def fixture_tfs(self) -> list[prtecan.Tecanfile]:  # pragma: no cover
        """Return a list of `Tecan` files."""
        tf1 = prtecan.Tecanfile(data_tests / "290513_5.5.xls")
        tf2 = prtecan.Tecanfile(data_tests / "290513_7.2.xls")
        tf3 = prtecan.Tecanfile(data_tests / "290513_8.8.xls")
        return [tf1, tf2, tf3]

    @pytest.fixture(autouse=True)
    def _init(self, tfs: list[prtecan.Tecanfile]) -> None:  # pragma: no cover
        """Return 2 label-block groups."""
        lbg1 = prtecan.LabelblocksGroup([tfs[0].labelblocks[0], tfs[1].labelblocks[0]])
        lbg2 = prtecan.LabelblocksGroup([tfs[1].labelblocks[1], tfs[2].labelblocks[1]])
        lbg1.buffer_wells = ["C12", "D01", "D12", "E01", "E12", "F01"]
        lbg2.buffer_wells = ["C12", "D01", "D12", "E01", "E12", "F01"]
        # pylint: disable=W0201
        self.lbg0 = lbg1
        self.lbg1 = lbg2

    def test_metadata(self) -> None:
        """Merge only shared metadata."""
        assert self.lbg0.metadata.get("Temperature") is None
        assert self.lbg1.metadata.get("Temperature") is None
        assert self.lbg1.metadata.get("Gain") is None
        assert self.lbg1.labelblocks[0].metadata["Gain"].value == 98
        assert self.lbg1.labelblocks[1].metadata["Gain"].value == 99
        # Common metadata.
        assert self.lbg0.metadata["Gain"].value == 94
        assert self.lbg0.metadata["Number of Flashes"].value == 10

    def test_data(self) -> None:
        """Merge data."""
        assert self.lbg0.data["A01"] == [18713, 17088]  # type: ignore
        assert self.lbg0.data["H12"] == [28596, 25771]  # type: ignore
        assert self.lbg1.data is None

    def test_data_normalized(self) -> None:
        """Merge data_normalized."""
        np.testing.assert_almost_equal(
            self.lbg1.data_norm["H12"], [693.980, 714.495], 3
        )
        np.testing.assert_almost_equal(
            self.lbg0.data_norm["A01"], [995.372, 908.936], 3
        )

    def test_data_buffersubtracted(self) -> None:
        """Merge data_buffersubtracted."""
        np.testing.assert_almost_equal(
            self.lbg0.data_buffersubtracted["B07"], [7069, 5716.7], 1  # type: ignore
        )
        assert self.lbg1.data_buffersubtracted is None

    def test_data_buffersubtracted_norm(self) -> None:
        """Merge data_buffersubtracted."""
        np.testing.assert_almost_equal(
            self.lbg0.data_buffersubtracted_norm["B07"], [376.01, 304.08], 2
        )
        np.testing.assert_almost_equal(
            self.lbg1.data_buffersubtracted_norm["B07"], [355.16, 348.57], 2
        )

    def test_notequal_labelblocks(self, tfs: list[prtecan.Tecanfile]) -> None:
        """It raises Exception when concatenating unequal labelblocks."""
        with pytest.raises(ValueError, match="Creation of labelblock group failed."):
            prtecan.LabelblocksGroup([tfs[1].labelblocks[0], tfs[2].labelblocks[1]])


class TestTecanfileGroup:
    """Group tecanfiles properly."""

    class TestAllEqLbgs:
        """Test TfG with 2 LbG in the same order."""

        @pytest.fixture(name="tfg", scope="class")
        def fixture_tfg(self) -> prtecan.TecanfilesGroup:  # pragma: no cover
            """Initialize Tfg with 2 LbG in the same order."""
            filenames = ["290513_5.5.xls", "290513_7.2.xls"]
            tecanfiles = [prtecan.Tecanfile(data_tests / f) for f in filenames]
            return prtecan.TecanfilesGroup(tecanfiles)

        def test_metadata(self, tfg: prtecan.TecanfilesGroup) -> None:
            """Parse general metadata."""
            assert tfg.metadata["Plate"].value == "PE 96 Flat Bottom White   [PE.pdfx]"
            assert tfg.metadata["System"].value == "TECANROBOT"

        def test_labelblocksgroups(self, tfg: prtecan.TecanfilesGroup) -> None:
            """Generate 2 LbG with .data and .metadata."""
            lbg0 = tfg.labelblocksgroups[0]
            lbg1 = tfg.labelblocksgroups[1]
            # metadata
            assert lbg0.metadata["Number of Flashes"].value == 10.0
            assert lbg1.metadata["Gain"].value == 98.0
            # data normalized ... enough in lbg
            # data
            assert lbg0.data["A01"] == [18713, 17088]  # type: ignore
            assert lbg0.data["H12"] == [28596, 25771]  # type: ignore
            assert lbg1.data["A01"] == [7878, 8761]  # type: ignore
            assert lbg1.data["H12"] == [14226, 13602]  # type: ignore

    class TestAlmostEqLbgs:
        """Test TfG when 1 LbG equal and a second with almost equal labelblocks."""

        @pytest.fixture(autouse=True)
        def _init(self) -> None:  # pragma: no cover
            """Initialize Tfg when 1 LbG equal and a second with almost equal labelblocks."""
            filenames = [
                "290513_5.5.xls",  # Label1 and Label2
                "290513_7.2.xls",  # Label1 and Label2
                "290513_8.8.xls",  # Label1 and Label2 with different metadata
            ]
            tecanfiles = [prtecan.Tecanfile(data_tests / f) for f in filenames]
            # pylint: disable=W0201
            with pytest.warns(UserWarning) as self.record:
                self.group = prtecan.TecanfilesGroup(tecanfiles)

        def test_warn(self) -> None:
            """Warn about labelblocks anomaly."""
            assert "Different LabelblocksGroup among filenames:" in str(
                self.record[0].message
            )

        def test_labelblocksgroups(self) -> None:
            """Generate 1 LbG with .data and .metadata."""
            lbg0 = self.group.labelblocksgroups[0]
            # metadata
            assert lbg0.metadata["Number of Flashes"].value == 10.0
            assert lbg0.metadata["Gain"].value == 94
            # data
            assert lbg0.data["A01"] == [18713.0, 17088.0, 17123.0]  # type: ignore
            assert lbg0.data["H12"] == [28596.0, 25771.0, 28309.0]  # type: ignore

        def test_mergeable_labelblocksgroups(self) -> None:
            """Generate 1 Lbg only with .data_normalized and only common .metadata."""
            lbg1 = self.group.labelblocksgroups[1]
            # metadata
            assert lbg1.metadata["Number of Flashes"].value == 10.0
            assert lbg1.metadata.get("Gain") is None
            assert lbg1.data is None
            # data_normalized
            np.testing.assert_almost_equal(
                lbg1.data_norm["A01"], [401.9387755, 446.9897959, 450.0]
            )
            np.testing.assert_almost_equal(
                lbg1.data_norm["H12"], [725.8163265, 693.9795918, 714.4949494]
            )

    class TestOnly1commonLbg:
        """Test TfG with different number of labelblocks, but mergeable."""

        @pytest.fixture(autouse=True)
        def _init(self) -> None:
            """Initialize Tfg with different number of labelblocks, but mergeable."""
            filenames = [
                "290212_5.78.xls",  # Label1 and Label2
                "290212_20.xls",  # Label2 only
                "290212_100.xls",  # Label2 only
            ]
            tecanfiles = [prtecan.Tecanfile(data_tests / f) for f in filenames]
            # pylint: disable=W0201
            with pytest.warns(UserWarning) as self.record:
                self.group = prtecan.TecanfilesGroup(tecanfiles)

        def test_warn(self) -> None:
            """Warn about labelblocks anomaly."""
            assert "Different LabelblocksGroup among filenames" in str(
                self.record[0].message
            )

        def test_labelblocksgroups(self) -> None:
            """Generates 1 LbG with .data and .metadata."""
            lbg = self.group.labelblocksgroups[0]
            # metadata
            assert lbg.metadata["Number of Flashes"].value == 10.0
            assert lbg.metadata["Gain"].value == 93.0
            # data
            assert lbg.data["A01"] == [6289, 6462, 6465]  # type: ignore
            assert lbg.data["H12"] == [4477, 4705, 4918]  # type: ignore

    class TestFailToMerge:
        """Test TfG without mergeable labelblocks."""

        def test_raise_exception(self) -> None:
            """Raise Exception when there is no way to build labelblocksGroup."""
            filenames = ["290513_5.5.xls", "290513_5.5_bad.xls"]
            tecanfiles = [prtecan.Tecanfile(data_tests / f) for f in filenames]
            with pytest.raises(
                ValueError,
                match=r"No common labelblock in filenames: .*290513_5.5_bad.xls",
            ):
                prtecan.TecanfilesGroup(tecanfiles)


class TestTitration:
    """Test Titration class."""

    tit_ph = prtecan.Titration.fromlistfile(data_tests / "list.pH")
    tit_cl = prtecan.Titration.fromlistfile(data_tests / "list.cl20")

    @pytest.mark.filterwarnings("ignore: Different LabelblocksGroup")
    def test_conc(self) -> None:
        """It reads pH values."""
        assert self.tit_ph.conc == [5.78, 6.38, 6.83, 7.24, 7.67, 8.23, 8.82, 9.31]

    def test_labelblocksgroups(self) -> None:
        """It reads labelblocksgroups data and metadata."""
        lbg0 = self.tit_ph.labelblocksgroups[0]
        lbg1 = self.tit_ph.labelblocksgroups[1]
        # metadata
        assert lbg0.metadata["Number of Flashes"].value == 10.0
        # pH9.3 is 93 Optimal not Manual
        assert lbg1.metadata["Gain"] == prtecan.Metadata(93.0)
        # data
        assert lbg0.data["A01"][::2] == [30344, 31010, 33731, 37967]  # type: ignore
        assert lbg1.data["A01"][1::2] == [9165, 15591, 20788, 22534]  # type: ignore
        assert lbg0.data["H12"][1::2] == [20888, 21711, 23397, 25045]  # type: ignore
        assert lbg1.data["H12"] == [4477, 5849, 7165, 8080, 8477, 8822, 9338, 9303]  # type: ignore

    def test_labelblocksgroups_cl(self) -> None:
        """It reads labelblocksgroups data for Cl too."""
        lbg = self.tit_cl.labelblocksgroups[0]
        # assert lbg.data["A01"] == [6289, 6462, 6390, 6465, 6774]
        assert lbg.data["A01"] == [6462, 6390, 6465, 6774]  # type: ignore
        # assert lbg.data["H12"] == [4477, 4705, 4850, 4918, 5007]
        assert lbg.data["H12"] == [4705, 4850, 4918, 5007]  # type: ignore

    def test_export_data(self, tmp_path: Any) -> None:
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
            prtecan.Titration.fromlistfile(Path("aax"))

    def test_bad_listfile(self) -> None:
        """It raises Exception when list.xx file is ill-shaped."""
        with pytest.raises(ValueError, match=r"Check format .* for listfile: .*"):
            prtecan.Titration.fromlistfile(data_tests / "list.pH2")


@pytest.mark.filterwarnings("ignore:OVER")
class TestTitrationAnalysis:
    """Test TitrationAnalysis class."""

    @pytest.fixture(autouse=True)
    def _init(self) -> None:
        """Set up test class."""
        # Use fixture to capture UserWarning "OVER"
        # pylint: disable=W0201
        self.titan = prtecan.TitrationAnalysis.fromlistfile(
            data_tests / "140220/list.pH"
        )
        self.titan.load_additions(data_tests / "140220/additions.pH")
        self.titan.load_scheme(data_tests / "140220/scheme.txt")

    def test_scheme(self) -> None:
        """It finds well position for buffer samples."""
        assert self.titan.scheme.buffer == ["D01", "E01", "D12", "E12"]

    def test_raise_listfilenotfound(self) -> None:
        """It raises OSError when scheme file does not exist."""
        with pytest.raises(
            FileNotFoundError, match=r"No such file or directory: 'aax'"
        ):
            self.titan.load_scheme(Path("aax"))

    def test_raise_listfile_exception(self) -> None:
        """It raises AssertionError when scheme.txt file is ill-shaped."""
        bad_schemefile = data_tests / "140220/scheme0.txt"
        msg = f"Check format [well sample] for schemefile: {bad_schemefile}"
        with pytest.raises(ValueError, match=re.escape(msg)):
            self.titan.load_scheme(bad_schemefile)

    def test_subtract_bg(self) -> None:
        """It subtracts buffer average values."""
        lbg0 = self.titan.labelblocksgroups[0]
        lbg1 = self.titan.labelblocksgroups[1]
        np.testing.assert_almost_equal(
            lbg0.data_norm["E01"][::2],
            [601.72, 641.505, 674.355, 706.774],
            3,
        )
        if lbg0.data:
            assert lbg0.data["E01"][::2] == [11192.0, 11932.0, 12543.0, 13146.0]
        if lbg0.data_buffersubtracted:
            np.testing.assert_array_equal(
                lbg0.data_buffersubtracted["A12"][::3],
                [8084.5, 16621.75, 13775.0],
            )
        np.testing.assert_array_equal(
            lbg1.data_buffersubtracted["A12"][::3],  # type: ignore
            [9758.25, 1334.0, 283.5],
        )

    def test_dilution_correction(self) -> None:
        """It applies dilution correction read from file listing additions."""
        np.testing.assert_array_equal(self.titan.additions, [100, 2, 2, 2, 2, 2, 2])  # type: ignore
        np.testing.assert_almost_equal(
            self.titan.data_dilutioncorrected[1]["A12"],  # type: ignore
            [9758.25, 7524.795, 3079.18, 1414.04, 641.79, 402.325, 317.52],
        )

    def test_data_dilutioncorrected_norma(self) -> None:
        """It normalizes data."""
        np.testing.assert_almost_equal(
            self.titan.data_dilutioncorrected_norm[0]["A12"][::2],  # type: ignore
            [434.65, 878.73, 975.58, 829.46],
            2,
        )
        np.testing.assert_almost_equal(
            self.titan.data_dilutioncorrected_norm[1]["A12"][::2],  # type: ignore
            [871.272, 274.927, 57.303, 28.35],
            3,
        )

    def test_keys(self) -> None:
        """It gets well positions for ctrl and unknown samples."""
        assert set(self.titan.scheme.names) == {"NTT", "G03", "V224Q", "S202N"}
        x = {"B12", "H12", "F01", "C12", "F12", "C01", "H01", "G12", "B01", "G01"}
        assert set(self.titan.scheme.ctrl) - {"A01", "A12"} == x

    @pytest.mark.skipif(sys.platform == "win32", reason="broken on windows")
    def test_fit(self) -> None:
        """It fits each label separately."""
        self.titan.fit("pH", nrm=True, bg=True, dil=True)
        fit0 = self.titan.fittings[0].sort_index()
        fit1 = self.titan.fittings[1].sort_index()
        df0 = pd.read_csv(data_tests / "140220/fit0.csv", index_col=0)
        df1 = pd.read_csv(data_tests / "140220/fit1.csv", index_col=0)
        pd.testing.assert_frame_equal(df0.sort_index(), fit0, rtol=1e0)
        pd.testing.assert_frame_equal(
            df1,
            fit1,
            check_like=True,
            check_categorical=False,
            atol=1e-3,
        )
        # 0:-1
        self.titan.fit("pH", fin=-1, nrm=True, bg=True, dil=True)
        fit0 = self.titan.fittings[0].sort_index()
        fit1 = self.titan.fittings[1].sort_index()
        df0 = pd.read_csv(data_tests / "140220/fit0-1.csv", index_col=0)
        df1 = pd.read_csv(data_tests / "140220/fit1-1.csv", index_col=0)
        pd.testing.assert_frame_equal(
            df0.sort_index()[1:], fit0[1:], check_like=True, atol=1e-4
        )
        pd.testing.assert_frame_equal(df1.sort_index(), fit1, atol=1e-5)
