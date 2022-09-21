"""Test prtecan module."""
from __future__ import annotations

import functools
import os
import os.path as path
from typing import Any

import numpy as np
import pandas as pd
import py
import pytest

from clophfit import prtecan


# Tecan test file folder __file__=this file
tests_dir = path.dirname(path.abspath(__file__))
ttff = functools.partial(path.join, tests_dir)


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


class TestLabelblock:
    """Test labelblock class."""

    def setup_class(self) -> None:
        """Initialize a labelblock from an .xls file."""
        csvl = prtecan.Tecanfile.read_xls(ttff("Tecan/290212_7.67.xls"))
        # tf = prtecan.Tecanfile(ttff("Tecan/290212_7.67.xls"))
        idxs = prtecan.Tecanfile.lookup_csv_lines(csvl)
        self.lb = prtecan.Labelblock(None, csvl[idxs[0] : idxs[1]])

    def test_metadata(self) -> None:
        """It parses "Temperature" metadata."""
        assert self.lb.metadata["Temperature"] == [25.9]

    def test_data(self) -> None:
        """It parses "A01" cell data."""
        assert self.lb.data["A01"] == 33731

    def test_overvalue(self) -> None:
        """It detects saturated data ("OVER")."""
        csvl = prtecan.Tecanfile.read_xls(ttff("Tecan/pH5.1_130913a-orig.xls"))
        idxs = prtecan.Tecanfile.lookup_csv_lines(csvl)
        with pytest.warns(UserWarning) as record:
            lb = prtecan.Labelblock(None, csvl[idxs[0] : idxs[1]])
        assert np.nansum(lb.data["A01"]) == np.nansum(np.nan)
        assert np.nansum(lb.data["F01"]) == np.nansum(np.nan)
        assert (
            str(record[0].message)
            == "OVER value in A01 well for ['Label1'] of tecanfile: "
        )

    def test_raise_missing_column(self) -> None:
        """It raises Exception when a column is missing from the labelblock."""
        csvl = prtecan.Tecanfile.read_xls(ttff("Tecan/88wells_290212_20.xlsx"))
        idxs = prtecan.Tecanfile.lookup_csv_lines(csvl)
        with pytest.raises(ValueError, match=r"Cannot build Labelblock: not 96 wells?"):
            self.lb = prtecan.Labelblock(None, csvl[idxs[0] : len(csvl)])

    def test_raise_missing_row(self) -> None:
        """It raises Exception when a row is missing from the labelblock."""
        csvl = prtecan.Tecanfile.read_xls(ttff("Tecan/84wells_290212_20.xlsx"))
        idxs = prtecan.Tecanfile.lookup_csv_lines(csvl)
        with pytest.raises(
            ValueError, match="Cannot extract data in Labelblock: not 96 wells?"
        ):
            self.lb = prtecan.Labelblock(None, csvl[idxs[0] : len(csvl)])


class TestTecanFile:
    """Test TecanFile class."""

    def setup_class(self) -> None:
        """Initialize a tecan file and read the corresponding xls file."""
        path = ttff("Tecan/290212_7.67.xls")
        self.tf1 = prtecan.Tecanfile(path)
        self.csvl = prtecan.Tecanfile.read_xls(path)

    def test_path(self) -> None:
        """It reads the file path."""
        assert self.tf1.path == ttff("Tecan/290212_7.67.xls")

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
        tf2 = prtecan.Tecanfile(ttff("Tecan/290212_7.67.xls"))
        assert tf2 == self.tf1

    def test___hash__(self) -> None:
        """It hashes its file path.

        ttff() uses absolute path.
        """
        assert self.tf1.__hash__() == hash(ttff("Tecan/290212_7.67.xls"))

    def test_filenotfound(self) -> None:
        """It raises FileNotFoundError when the file path does not exist."""
        with pytest.raises(FileNotFoundError):
            prtecan.Tecanfile("pinocchio")

    def test_missing_label(self) -> None:
        """It raises Exception when there is no Label pattern."""
        with pytest.raises(ValueError, match="No Labelblock found."):
            prtecan.Tecanfile(ttff("Tecan/0_Labelblocks_290513_5.5.xlsx"))


class TestLabelblocksGroup:
    """Test LabelBlocksGroup class."""

    def setup_class(self) -> None:
        """Initialize a labelblocksgroup reading (and concatenating) 2 xls files."""
        self.tf1 = prtecan.Tecanfile(ttff("Tecan/290212_5.78.xls"))
        self.tf2 = prtecan.Tecanfile(ttff("Tecan/290212_6.38.xls"))
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
        assert self.lb_grp.temperatures == [25.9, 26]

    def test_data(self) -> None:
        """It reads cell data."""
        assert self.lb_grp.data["A01"] == [30344, 30072]
        assert self.lb_grp.data["H12"] == [21287, 20888]

    def test_notequal_labelblocks(self) -> None:
        """It raises Exception when concatenating unequal labelblocks."""
        with pytest.raises(ValueError, match="Creation of labelblock group failed."):
            prtecan.LabelblocksGroup([self.tf1.labelblocks[0], self.tf2.labelblocks[1]])


# TODO change data and test any warning behavior separately
@pytest.mark.filterwarnings("ignore: Different LabelblocksGroup")
class TestTecanfilesGroup:
    """Test TecanfilesGroup class."""

    def setup_class(self) -> None:
        """Initialize file lists for pH and Cl."""
        filenames = [
            "290212_5.78.xls",
            "290212_6.38.xls",
            "290212_6.83.xls",
            "290212_7.24.xls",
            "290212_7.67.xls",
            "290212_8.23.xls",
            "290212_8.82.xls",
            "290212_9.31.xls",
        ]
        filenames_cl = [
            # "290212_5.78.xls",
            "290212_20.xls",
            "290212_50.xls",
            "290212_100.xls",
            "290212_150.xls",
        ]
        self.dir = os.getcwd()
        os.chdir(ttff("Tecan/"))
        self.group = prtecan.TecanfilesGroup(filenames)
        self.group_cl = prtecan.TecanfilesGroup(filenames_cl)

    def teardown_class(self) -> None:
        """Return to the initial folder."""
        os.chdir(self.dir)

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
        """It generates 2 labelblocksgroups for Cl list. It tests only data."""
        lbg = self.group_cl.labelblocksgroups[0]
        # assert lbg.data["A01"] == [6289, 6462, 6390, 6465, 6774]
        assert lbg.data["A01"] == [6462, 6390, 6465, 6774]
        # assert lbg.data["H12"] == [4477, 4705, 4850, 4918, 5007]
        assert lbg.data["H12"] == [4705, 4850, 4918, 5007]

    def test_raise_exception(self) -> None:
        """It raises Exception when labelblocks are different (acquisition day)."""
        filenames = ["290212_5.78.xls", "290513_5.5.xls"]
        with pytest.raises(ValueError, match=r"Creation of labelblock group failed."):
            prtecan.TecanfilesGroup(filenames)
        # assert "No common labelblock in filenames" in str(err.value)
        # assert "['290212_5.78.xls', '290513_5.5.xls']" in str(err.value)

    @pytest.mark.skip("raiseassertion in mixing tf with 1 and 2 lbks")
    def test_warn(self) -> None:
        """XXX must raise error."""
        # Different labelblocks or in different order.
        filenames = ["290212_5.78.xls", "290212_20.xls"]
        with pytest.warns(UserWarning) as record:
            prtecan.TecanfilesGroup(filenames)
        assert "Different LabelblocksGroup among filenames" in str(record[0].message)
        assert "['290212_5.78.xls', '290212_20.xls']" in str(record[0].message)


class TestTitration:
    """Test Titration class."""

    def setup_class(self) -> None:
        """Initialize pH and Cl titration from list.pH and list.cl files."""
        self.tit = prtecan.Titration(ttff("Tecan/list.pH"))
        self.tit_cl = prtecan.Titration(ttff("Tecan/list.cl20"))

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
        assert lbg1.metadata["Gain"][0] == 93.0
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

    def test_export_dat(self, tmpdir: py.path.local) -> None:
        """It exports titrations data to files e.g. "A01.dat"."""
        p = tmpdir.mkdir("export_test")
        path = str(p.join("dat"))  # to test also creation of folder
        self.tit.export_dat(path)
        a01 = pd.read_csv(p.join("dat", "A01.dat"))
        h12 = pd.read_csv(p.join("dat", "H12.dat"))
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
        with pytest.raises(FileNotFoundError) as err:
            prtecan.Titration(ttff("aax"))
        assert str(err.value) == "Cannot find: " + ttff("aax")

    def test_bad_listfile(self) -> None:
        """It raises Exception when list.xx file is ill-shaped."""
        name = ttff("Tecan/list.pH2")
        with pytest.raises(ValueError, match=r"Check format .* for listfile: (?s).*"):
            prtecan.Titration(name)


@pytest.mark.filterwarnings("ignore: OVER value")
class TestTitrationAnalysis:
    """Test TitrationAnalysis class."""

    def setup_class(self) -> None:
        """Initialize objects reading list.pH and scheme.txt."""
        self.tit = prtecan.Titration(ttff("Tecan/140220/list.pH"))
        self.tit_an = prtecan.TitrationAnalysis(
            self.tit, ttff("Tecan/140220/scheme.txt")
        )
        self.lbg0 = self.tit_an.labelblocksgroups[0]
        self.lbg1 = self.tit_an.labelblocksgroups[1]

    def test_scheme(self) -> None:
        """It finds well position for buffer samples."""
        self.tit_an.scheme["buf"] = ["D01", "E01", "D12", "E12"]

    def test_raise_listfilenotfound(self) -> None:
        """It raises OSError when sheme file does not exist."""
        with pytest.raises(OSError, match=r"No such file .*"):
            prtecan.TitrationAnalysis(self.tit, ttff("aax"))
        # assert "No such file" in str(err.value)

    def test_raise_listfile_exception(self) -> None:
        """It raises AssertionError when scheme.txt file is ill-shaped."""
        name = ttff("Tecan/140220/scheme0.txt")
        with pytest.raises(AssertionError) as err:
            prtecan.TitrationAnalysis(self.tit, name)
        assert str(err.value) == "Check format [well sample] for schemefile: " + name

    def test_subtract_bg(self) -> None:
        """It subtracts buffer average values."""
        self.tit_an.subtract_bg()
        assert self.lbg0.buffer["E01"] == [
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
        self.tit_an.dilution_correction(ttff("Tecan/140220/additions.pH"))
        np.testing.assert_array_equal(self.tit_an.additions, [100, 2, 2, 2, 2, 2, 2])
        np.testing.assert_almost_equal(
            self.lbg1.data["A12"],
            [9758.25, 7524.795, 3079.18, 1414.04, 641.79, 402.325, 317.52],
        )

    def test_dilution_correction_warning(self) -> None:
        """It warns when dilution correction is repeated."""
        # run in previous test_dilution_correction
        with pytest.warns(UserWarning) as record:
            self.tit_an.dilution_correction(ttff("Tecan/140220/additions.pH"))
        assert str(record[0].message) == "Dilution correction was already applied."

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
        with pytest.warns(UserWarning) as record:
            self.tit_an.metadata_normalization()
        assert (
            str(record[0].message)
            == "Normalization using metadata was already applied."
        )

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
        df0 = pd.read_csv(ttff("Tecan/140220/fit0.csv"), index_col=0)
        df1 = pd.read_csv(ttff("Tecan/140220/fit1.csv"), index_col=0)
        pd.testing.assert_frame_equal(df0.sort_index(), fit0)
        pd.testing.assert_frame_equal(df1.sort_index(), fit1)
        # 0:-1
        self.tit_an.fit("pH", fin=-1)
        fit0 = self.tit_an.fittings[0].sort_index()
        fit1 = self.tit_an.fittings[1].sort_index()
        df0 = pd.read_csv(ttff("Tecan/140220/fit0-1.csv"), index_col=0)
        df1 = pd.read_csv(ttff("Tecan/140220/fit1-1.csv"), index_col=0)
        pd.testing.assert_frame_equal(df0.sort_index(), fit0, atol=1e-5)
        pd.testing.assert_frame_equal(df1.sort_index(), fit1)


if __name__ == "__main__":
    pytest.main()
