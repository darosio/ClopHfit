"""Test prtecan module."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from clophfit import prtecan


data_tests = Path(__file__).parent / "Tecan"


def test_strip_lines() -> None:
    """It strips empty fields."""
    lines = [["Excitation Wavelength", "", "", "", 485.0, "nm", "", "", ""]]
    stripped = prtecan.strip_lines(lines)
    assert stripped == [["Excitation Wavelength", 485.0, "nm"]]


def test_extract_metadata() -> None:
    """It extracts metadata correctly."""
    lines: list[list[Any]] = [
        ["Label: Label1", "", "", "", "", "", "", "", "", "", "", "", ""],
        ["Mode", "", "", "", "Fluorescence Top Reading", "", "", "", "", ""],
        ["Shaking (Linear) Amplitude:", "", "", "", 2, "mm", "", "", "", "", ""],
        ["Excitation Wavelength", "", "", "", 400, "nm", "", "", "", "", ""],
        ["", "Temperature: 26 °C", "", "", "", "", "", "", "", "", ""],
    ]
    expected_metadata = {
        "Shaking (Linear) Amplitude:": [2, "mm"],
        "Excitation Wavelength": [400, "nm"],
        "Temperature": [26.0],
        "Label": ["Label1"],
        "Mode": ["Fluorescence Top Reading"],
    }

    metadata = prtecan.extract_metadata(lines)
    assert metadata == expected_metadata


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


@pytest.mark.filterwarnings("ignore: OVER value")
class TestLabelblock:
    """Test labelblock class."""

    def setup_class(self) -> None:
        """Initialize a labelblock from an .xls file."""
        csvl = prtecan.Tecanfile.read_xls(data_tests / "140220/pH6.5_200214.xls")
        idxs = prtecan.Tecanfile.lookup_csv_lines(csvl)
        self.lb0 = prtecan.Labelblock(csvl[idxs[0] : idxs[1]])
        self.lb1 = prtecan.Labelblock(csvl[idxs[1] :])
        self.lb0.buffer_wells = ["D01", "D12", "E01", "E12"]
        self.lb1.buffer_wells = ["D01", "D12", "E01", "E12"]

    def test_metadata(self) -> None:
        """It parses "Temperature" metadata."""
        assert self.lb0.metadata["Temperature"] == [25.6]
        assert self.lb1.metadata["Temperature"] == [25.3]

    def test_data(self) -> None:
        """It parses data values."""
        assert self.lb0.data["F06"] == 19551
        assert self.lb1.data["H12"] == 543

    def test_data_normalized(self) -> None:
        """Normalize data using some metadata values."""
        assert self.lb0.data_normalized["F06"] == pytest.approx(1051.1290323)
        assert self.lb1.data_normalized["H12"] == pytest.approx(48.4821429)

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
        csvl = prtecan.Tecanfile.read_xls(data_tests / "140220/NaCl4_200214.xls")
        idxs = prtecan.Tecanfile.lookup_csv_lines(csvl)
        lb = prtecan.Labelblock(csvl[idxs[0] : idxs[1]])
        assert lb.__almost_eq__(self.lb0)
        assert not lb.__almost_eq__(self.lb1)

    def test_overvalue(self) -> None:
        """It detects saturated data ("OVER")."""
        csvl = prtecan.Tecanfile.read_xls(data_tests / "140220/pH6.5_200214.xls")
        idxs = prtecan.Tecanfile.lookup_csv_lines(csvl)
        with pytest.warns(
            UserWarning, match=r"OVER value in A06 well for \['Label1'\] of tecanfile: "
        ):
            lb = prtecan.Labelblock(csvl[idxs[0] : idxs[1]])
        assert np.nansum(lb.data["A06"]) == np.nansum(np.nan)
        assert np.nansum(lb.data["H02"]) == np.nansum(np.nan)

    def test_raise_missing_column(self) -> None:
        """It raises Exception when a column is missing from the labelblock."""
        csvl = prtecan.Tecanfile.read_xls(
            data_tests / "exceptions/88wells_290212_20.xlsx"
        )
        idxs = prtecan.Tecanfile.lookup_csv_lines(csvl)
        with pytest.raises(ValueError, match=r"Cannot build Labelblock: not 96 wells?"):
            prtecan.Labelblock(csvl[idxs[0] : len(csvl)])

    def test_raise_missing_row(self) -> None:
        """It raises Exception when a row is missing from the labelblock."""
        csvl = prtecan.Tecanfile.read_xls(
            data_tests / "exceptions/84wells_290212_20.xlsx"
        )
        idxs = prtecan.Tecanfile.lookup_csv_lines(csvl)
        with pytest.raises(
            ValueError, match="Cannot extract data in Labelblock: not 96 wells?"
        ):
            prtecan.Labelblock(csvl[idxs[0] : len(csvl)])


class TestTecanfile:
    """Test TecanFile class."""

    def setup_class(self) -> None:
        """Initialize a tecan file and read the corresponding xls file."""
        self.tf1 = prtecan.Tecanfile(data_tests / "290212_7.67.xls")
        self.csvl = prtecan.Tecanfile.read_xls(data_tests / "290212_7.67.xls")

    def test_warn(self) -> None:
        """It warns when labelblocks are repeated in a Tf as it might compromise grouping."""
        with pytest.warns(UserWarning, match="Repeated labelblocks"):
            prtecan.Tecanfile(data_tests / "290212_7.67_repeated_lb.xls")

    def test_path(self) -> None:
        """It reads the file path."""
        assert self.tf1.path == data_tests / "290212_7.67.xls"

    def test_metadata(self) -> None:
        """It parses the Date."""
        assert self.tf1.metadata["Date:"] == ["29/02/2012"]

    def test_read_xls(self) -> None:
        """The test reads the xls file using cls method."""
        assert len(self.csvl) == 77

    def test_lookup_csv_lines(self) -> None:
        """It finds Label occurrences using cls method."""
        assert prtecan.Tecanfile.lookup_csv_lines(self.csvl) == [17, 47]

    def test_labelblocks(self) -> None:
        """It parses "Temperature" metadata and cell data from 2 labelblocks."""
        assert self.tf1.labelblocks[0].metadata["Temperature"] == [25.9]
        assert self.tf1.labelblocks[1].metadata["Temperature"] == [25.7]
        assert self.tf1.labelblocks[0].data["A01"] == 33731
        assert self.tf1.labelblocks[1].data["H12"] == 8477

    def test___eq__(self) -> None:
        """It is equal to itself. TODO and different from other."""
        tf2 = prtecan.Tecanfile(data_tests / "290212_7.67.xls")
        assert tf2 == self.tf1

    def test_filenotfound(self) -> None:
        """It raises FileNotFoundError when the file path does not exist."""
        with pytest.raises(FileNotFoundError):
            prtecan.Tecanfile(Path("pinocchio"))

    def test_missing_label(self) -> None:
        """It raises Exception when there is no Label pattern."""
        with pytest.raises(ValueError, match="No Labelblock found."):
            prtecan.Tecanfile(data_tests / "0_Labelblocks_290513_5.5.xlsx")


class TestLabelblocksGroup:
    """Test LabelBlocksGroup class."""

    def setup_class(self) -> None:
        """Initialize a labelblocksgroup reading (and concatenating) 2 xls files."""
        self.tf1 = prtecan.Tecanfile(data_tests / "290212_5.78.xls")
        self.tf2 = prtecan.Tecanfile(data_tests / "290212_6.38.xls")
        self.metadata = self.tf2.labelblocks[0].metadata
        self.lb_grp = prtecan.LabelblocksGroup(
            [self.tf1.labelblocks[0], self.tf2.labelblocks[0]]
        )

    def test_labelblock__eq(self) -> None:
        """It groups Labelblocks with compatible metadata."""
        assert prtecan.Labelblock.__eq__(
            self.tf1.labelblocks[1], self.tf2.labelblocks[1]
        )

    def test_temperatures(self) -> None:
        """It reads Temperature metadata."""
        assert self.lb_grp.metadata["Temperature"] == [25.9, 26]

    def test_data(self) -> None:
        """It reads cell data."""
        assert self.lb_grp.data["A01"] == [30344, 30072]
        assert self.lb_grp.data["H12"] == [21287, 20888]

    def test_notequal_labelblocks(self) -> None:
        """It raises Exception when concatenating unequal labelblocks."""
        with pytest.raises(ValueError, match="Creation of labelblock group failed."):
            prtecan.LabelblocksGroup([self.tf1.labelblocks[0], self.tf2.labelblocks[1]])


class TestTecanfilesGroup1:
    """Test TecanfilesGroup class (2 labelblocksgroup in the same order)."""

    def setup_class(self) -> None:
        """Initialize file lists for pH and Cl."""
        filenames = ["290212_5.78.xls", "290212_6.38.xls", "290212_6.83.xls"]
        tecanfiles = [prtecan.Tecanfile(data_tests / f) for f in filenames]
        self.group = prtecan.TecanfilesGroup(tecanfiles)

    def test_metadata(self) -> None:
        """It parses general metadata."""
        assert self.group.metadata["Plate"] == ["PE 96 Flat Bottom White   [PE.pdfx]"]
        assert self.group.metadata["Shaking (Linear) Amplitude:"] == [2.0, "mm"]
        assert self.group.metadata["Shaking (Linear) Duration:"] == [50.0, "s"]
        assert self.group.metadata["System"] == ["TECANROBOT"]

    def test_labelblocksgroups(self) -> None:
        """It generates 2 labelblocksgroups for pH list.

        Test metadata and data.
        """
        lbg0 = self.group.labelblocksgroups[0]
        lbg1 = self.group.labelblocksgroups[1]
        # metadata
        assert lbg0.metadata["Number of Flashes"][0] == 10.0
        assert lbg1.metadata["Gain"][0] == 93.0
        # data
        assert lbg0.data["A01"] == [30344, 30072, 31010]
        assert lbg1.data["A01"] == [6289, 9165, 12326]
        assert lbg0.data["H12"] == [21287, 20888, 21209]
        assert lbg1.data["H12"] == [4477, 5849, 7165]


class TestTecanfilesGroup2:
    """Test TecanfilesGroup when one labelblocksgroup has only almost equal labelblocks."""

    def setup_class(self) -> None:
        """Initialize file lists for pH and Cl."""
        filenames = [
            "290513_5.5.xls",  # Label1 and Label2
            "290513_7.2.xls",  # Label1 and Label2
            "290513_8.8.xls",  # Label1 and Label2 with different metadata
        ]
        self.tecanfiles = [prtecan.Tecanfile(data_tests / f) for f in filenames]
        with pytest.warns(UserWarning) as self.record:
            self.group = prtecan.TecanfilesGroup(self.tecanfiles)

    def test_warn(self) -> None:
        """It warns about difference in labelblocks XXX."""
        assert "Different LabelblocksGroup among filenames:" in str(
            self.record[0].message
        )

    def test_labelblocksgroups(self) -> None:
        """It generates 1 std XXX labelblocksgroups."""
        lbg0 = self.group.labelblocksgroups[0]
        # metadata
        assert lbg0.metadata["Number of Flashes"][0] == 10.0
        assert lbg0.metadata["Gain"][0] == [94.0, "Manual"]
        # data
        assert lbg0.data["A01"] == [18713.0, 17088.0, 17123.0]
        assert lbg0.data["H12"] == [28596.0, 25771.0, 28309.0]

    def test_mergeable_labelblocksgroups(self) -> None:
        """It generates 1 std XXX labelblocksgroups."""
        lbg1 = self.group.labelblocksgroups[1]
        # metadata
        assert lbg1.metadata["Number of Flashes"][0] == 10.0
        assert lbg1.metadata["Gain"][0] == [98.0, "Manual"]
        # # data
        np.testing.assert_almost_equal(
            lbg1.data["A01"], [401.9387755, 446.9897959, 450.0]
        )
        np.testing.assert_almost_equal(
            lbg1.data["H12"], [725.8163265, 693.9795918, 714.4949494]
        )


class TestTecanfilesGroup3:
    """Test TecanfilesGroup with different number of labelblocks."""

    def setup_class(self) -> None:
        """Initialize file lists for pH and Cl."""
        filenames = [
            "290212_5.78.xls",  # Label1 and Label2
            "290212_20.xls",  # Label2 only
            "290212_100.xls",  # Label2 only
        ]
        self.tecanfiles = [prtecan.Tecanfile(data_tests / f) for f in filenames]
        with pytest.warns(UserWarning) as self.record:
            self.group = prtecan.TecanfilesGroup(self.tecanfiles)

    def test_warn(self) -> None:
        """It warns about difference in labelblocks order."""
        assert "Different LabelblocksGroup among filenames" in str(
            self.record[0].message
        )

    def test_labelblocksgroups(self) -> None:
        """It generates 1 labelblocksgroups for Cl list. It tests only data."""
        lbg = self.group.labelblocksgroups[0]
        # metadata
        assert lbg.metadata["Number of Flashes"][0] == 10.0
        assert lbg.metadata["Gain"][0] == 93.0
        # data
        assert lbg.data["A01"] == [6289, 6462, 6465]
        assert lbg.data["H12"] == [4477, 4705, 4918]


class TestTecanfilesGroup4Raise:
    """Test TecanfilesGroup without mergeable labelblocks."""

    def test_raise_exception(self) -> None:
        """It raises Exception when there is no way to build labelblocksGroup."""
        filenames = ["290212_5.78.xls", "290513_5.5_bad.xls"]
        tecanfiles = [prtecan.Tecanfile(data_tests / f) for f in filenames]
        with pytest.raises(
            ValueError,
            match=r"No common labelblock in filenames: .*290212_5.78.xls.*290513_5.5_bad.xls",
        ):
            prtecan.TecanfilesGroup(tecanfiles)


class TestTitration:
    """Test Titration class."""

    def setup_class(self) -> None:
        """Initialize pH and Cl titration from list.pH and list.cl files."""
        self.tit = prtecan.Titration(data_tests / "list.pH")
        self.tit_cl = prtecan.Titration(data_tests / "list.cl20")

    @pytest.mark.filterwarnings("ignore: Different LabelblocksGroup")
    def test_conc(self) -> None:
        """It reads pH values."""
        assert self.tit.conc == [5.78, 6.38, 6.83, 7.24, 7.67, 8.23, 8.82, 9.31]

    def test_labelblocksgroups(self) -> None:
        """It reads labelblocksgroups data and metadata."""
        lbg0 = self.tit.labelblocksgroups[0]
        lbg1 = self.tit.labelblocksgroups[1]
        # metadata
        assert lbg0.metadata["Number of Flashes"][0] == 10.0
        # pH9.3 is 93 Optimal not Manual
        assert lbg1.metadata["Gain"][0] == [93.0, "Manual"]
        # data
        assert lbg0.data["A01"] == [
            30344,
            30072,
            31010,
            32678,
            33731,
            36506,
            37967,
            37725,
        ]
        assert lbg1.data["A01"] == [
            6289,
            9165,
            12326,
            15591,
            17726,
            20788,
            21781,
            22534,
        ]
        assert lbg0.data["H12"] == [
            21287,
            20888,
            21209,
            21711,
            22625,
            23397,
            24791,
            25045,
        ]
        assert lbg1.data["H12"] == [4477, 5849, 7165, 8080, 8477, 8822, 9338, 9303]

    def test_labelblocksgroups_cl(self) -> None:
        """It reads labelblocksgroups data for Cl too."""
        lbg = self.tit_cl.labelblocksgroups[0]
        # assert lbg.data["A01"] == [6289, 6462, 6390, 6465, 6774]
        assert lbg.data["A01"] == [6462, 6390, 6465, 6774]
        # assert lbg.data["H12"] == [4477, 4705, 4850, 4918, 5007]
        assert lbg.data["H12"] == [4705, 4850, 4918, 5007]

    def test_export_dat(self, tmp_path: Any) -> None:
        """It exports titrations data to files e.g. "A01.dat"."""
        path = tmp_path / "dat"
        path.mkdir()
        self.tit.export_dat(path)
        a01 = pd.read_csv(path / "A01.dat")
        h12 = pd.read_csv(path / "H12.dat")
        assert a01["y1"].tolist() == [
            30344,
            30072,
            31010,
            32678,
            33731,
            36506,
            37967,
            37725,
        ]
        assert a01["y2"].tolist() == [
            6289,
            9165,
            12326,
            15591,
            17726,
            20788,
            21781,
            22534,
        ]
        assert h12["y1"].tolist() == [
            21287,
            20888,
            21209,
            21711,
            22625,
            23397,
            24791,
            25045,
        ]
        assert h12["y2"].tolist() == [4477, 5849, 7165, 8080, 8477, 8822, 9338, 9303]

    def test_raise_listfilenotfound(self) -> None:
        """It raises FileNotFoundError when list.xx file does not exist."""
        with pytest.raises(FileNotFoundError, match="Cannot find: aax"):
            prtecan.Titration(Path("aax"))

    def test_bad_listfile(self) -> None:
        """It raises Exception when list.xx file is ill-shaped."""
        with pytest.raises(ValueError, match=r"Check format .* for listfile: .*"):
            prtecan.Titration(data_tests / "list.pH2")


@pytest.mark.filterwarnings("ignore: OVER value")
class TestTitrationAnalysis:
    """Test TitrationAnalysis class."""

    def setup_class(self) -> None:
        """Initialize objects reading list.pH and scheme.txt."""
        self.tit = prtecan.Titration(data_tests / "140220/list.pH")
        self.tit_an = prtecan.TitrationAnalysis(
            self.tit, str(data_tests / "140220/scheme.txt")
        )
        self.lbg0 = self.tit_an.labelblocksgroups[0]
        self.lbg1 = self.tit_an.labelblocksgroups[1]

    def test_scheme(self) -> None:
        """It finds well position for buffer samples."""
        self.tit_an.scheme["buf"] = ["D01", "E01", "D12", "E12"]

    def test_raise_listfilenotfound(self) -> None:
        """It raises OSError when scheme file does not exist."""
        with pytest.raises(
            FileNotFoundError, match=r"No such file or directory: 'aax'"
        ):
            prtecan.TitrationAnalysis(self.tit, "aax")

    def test_raise_listfile_exception(self) -> None:
        """It raises AssertionError when scheme.txt file is ill-shaped."""
        bad_schemefile = str(data_tests / "140220/scheme0.txt")
        msg = f"Check format [well sample] for schemefile: {bad_schemefile}"
        with pytest.raises(ValueError, match=re.escape(msg)):
            prtecan.TitrationAnalysis(self.tit, bad_schemefile)

    def test_subtract_bg(self) -> None:
        """It subtracts buffer average values."""
        self.tit_an.subtract_bg()
        assert self.lbg0.buffer["E01"] == [  # type: ignore
            11192.0,
            12092.0,
            11932.0,
            12106.0,
            12543.0,
            12715.0,
            13146.0,
        ]
        np.testing.assert_array_equal(
            self.lbg0.data["A12"],
            [
                8084.5,
                11885.25,
                15715.75,
                16621.75,
                16801.75,
                16972.25,
                13775.0,
            ],
        )
        np.testing.assert_array_equal(
            self.lbg1.data["A12"],
            [
                9758.25,
                7377.25,
                2960.75,
                1334.0,
                594.25,
                365.75,
                283.5,
            ],
        )

    def test_dilution_correction(self) -> None:
        """It applies dilution correction read from file listing additions."""
        self.tit_an.dilution_correction(str(data_tests / "140220/additions.pH"))
        np.testing.assert_array_equal(self.tit_an.additions, [100, 2, 2, 2, 2, 2, 2])
        np.testing.assert_almost_equal(
            self.lbg1.data["A12"],
            [9758.25, 7524.795, 3079.18, 1414.04, 641.79, 402.325, 317.52],
        )

    def test_dilution_correction_warning(self) -> None:
        """It warns when dilution correction is repeated."""
        # run in previous test_dilution_correction
        with pytest.warns(
            UserWarning, match="Dilution correction was already applied."
        ):
            self.tit_an.dilution_correction(str(data_tests / "140220/additions.pH"))

    def test_metadata_normalization(self) -> None:
        """It normalizes data."""
        self.tit_an.metadata_normalization()
        np.testing.assert_almost_equal(
            self.lbg0.data["A12"],
            [
                434.65053763,
                651.77177419,
                878.73010753,
                947.26102151,
                975.58548387,
                1003.73521505,
                829.46236559,
            ],
        )
        np.testing.assert_almost_equal(
            self.lbg1.data["A12"],
            [
                871.27232143,
                671.85669643,
                274.92678571,
                126.25357143,
                57.30267857,
                35.921875,
                28.35,
            ],
        )

    def test_metadata_normalization_warning(self) -> None:
        """It warns when normalization is repeated."""
        with pytest.warns(
            UserWarning, match="Normalization using metadata was already applied."
        ):
            self.tit_an.metadata_normalization()

    def test_calculate_conc(self) -> None:
        """It calculates concentration values from Cl additions."""
        additions = [112, 2, 2, 2, 2, 2, 2, 6, 4]
        conc = prtecan.TitrationAnalysis.calculate_conc(additions, 1000)
        np.testing.assert_almost_equal(
            conc,
            [
                0.0,
                17.54385965,
                34.48275862,
                50.84745763,
                66.66666667,
                81.96721311,
                96.77419355,
                138.46153846,
                164.17910448,
            ],
        )

    def test__get_keys(self) -> None:
        """It gets well positions for ctrl and unknown samples."""
        self.tit_an.scheme.pop("buf")
        self.tit_an._get_keys()
        assert set(self.tit_an.names_ctrl) == {"NTT", "G03", "V224Q", "S202N"}
        assert self.tit_an.keys_ctrl == [
            "A01",
            "B12",
            "H12",
            "F01",
            "C12",
            "F12",
            "C01",
            "H01",
            "G12",
            "B01",
            "G01",
            "A12",
        ]
        assert set(self.tit_an.keys_unk) == {
            "A02",
            "A03",
            "A04",
            "A05",
            "A06",
            "A07",
            "A08",
            "A09",
            "A10",
            "A11",
            "B02",
            "B03",
            "B04",
            "B05",
            "B06",
            "B07",
            "B08",
            "B09",
            "B10",
            "B11",
            "C02",
            "C03",
            "C04",
            "C05",
            "C06",
            "C07",
            "C08",
            "C09",
            "C10",
            "C11",
            "D02",
            "D03",
            "D04",
            "D05",
            "D06",
            "D07",
            "D08",
            "D09",
            "D10",
            "D11",
            "E02",
            "E03",
            "E04",
            "E05",
            "E06",
            "E07",
            "E08",
            "E09",
            "E10",
            "E11",
            "F02",
            "F03",
            "F04",
            "F05",
            "F06",
            "F07",
            "F08",
            "F09",
            "F10",
            "F11",
            "G02",
            "G03",
            "G04",
            "G05",
            "G06",
            "G07",
            "G08",
            "G09",
            "G10",
            "G11",
            "H02",
            "H03",
            "H04",
            "H05",
            "H06",
            "H07",
            "H08",
            "H09",
            "H10",
            "H11",
        }

    def test_fit(self) -> None:
        """It fits each label separately."""
        self.tit_an.fit("pH")
        fit0 = self.tit_an.fittings[0].sort_index()
        fit1 = self.tit_an.fittings[1].sort_index()
        df0 = pd.read_csv(data_tests / "140220/fit0.csv", index_col=0)
        df1 = pd.read_csv(data_tests / "140220/fit1.csv", index_col=0)
        pd.testing.assert_frame_equal(df0.sort_index(), fit0, atol=1e-4)
        pd.testing.assert_frame_equal(df1.sort_index(), fit1, atol=1e-4)
        # 0:-1
        self.tit_an.fit("pH", fin=-1)
        fit0 = self.tit_an.fittings[0].sort_index()
        fit1 = self.tit_an.fittings[1].sort_index()
        df0 = pd.read_csv(data_tests / "140220/fit0-1.csv", index_col=0)
        df1 = pd.read_csv(data_tests / "140220/fit1-1.csv", index_col=0)
        pd.testing.assert_frame_equal(df0.sort_index(), fit0, atol=1e-5)
        pd.testing.assert_frame_equal(df1.sort_index(), fit1, atol=1e-5)
