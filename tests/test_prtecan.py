"""Comprehensive test suite for prtecan module."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest
import seaborn as sns  # type: ignore[import-untyped]
from numpy.testing import assert_allclose, assert_almost_equal, assert_array_equal

from clophfit import prtecan
from clophfit.fitting.data_structures import FitResult
from clophfit.prtecan import (
    Buffer,
    BufferFit,
    Labelblock,
    LabelblocksGroup,
    Metadata,
    PlateScheme,
    TecanConfig,
    Tecanfile,
    TecanfilesGroup,
    Titration,
    TitrationConfig,
    TitrationResults,
    calculate_conc,
    dilution_correction,
    extract_metadata,
    merge_md,
    strip_lines,
)

# Test data paths
data_tests = Path(__file__).parent / "Tecan"
pytestmark = pytest.mark.filterwarnings("ignore:OVER")


# =============================================================================
# Helper Functions
# =============================================================================


def create_test_metadata() -> list[list[Any]]:
    """Create sample metadata for testing."""
    return [
        ["Label: Label1", "", "", "", "", "", "", "", "", "", "", "", ""],
        ["Mode", "", "", "", "Fluorescence Top Reading", "", "", "", "", ""],
        ["Shaking (Linear) Amplitude:", "", "", "", 2, "mm", "", "", "", "", ""],
        ["Excitation Wavelength", "", "", "", 400, "nm", "", "unexpected", "", "", ""],
        ["", "Temperature: 26 °C", "", "", "", "", "", "", "", "", ""],
    ]


def create_sample_labelblock() -> Labelblock:
    """Create a sample Labelblock from test data."""
    csvl = prtecan.read_xls(data_tests / "140220/pH6.5_200214.xls")
    idxs = prtecan.lookup_listoflines(csvl)
    return Labelblock(csvl[idxs[0] : idxs[1]])


# =============================================================================
# Unit Tests
# =============================================================================


class TestLookupListOfLinesEdgeCases:
    """Test edge cases for lookup_listoflines function."""

    def test_empty_input(self) -> None:
        """It handles empty input gracefully."""
        assert prtecan.lookup_listoflines([], "pattern") == []

    def test_no_matches(self) -> None:
        """It returns empty list for no matches."""
        csvl: list[list[str | int | float]] = [["a", "b"], ["c", "d"]]
        assert prtecan.lookup_listoflines(csvl, "z") == []

    def test_partial_lines(self) -> None:
        """It handles lines with fewer columns than specified."""
        csvl: list[list[str | int | float]] = [["a"], ["b", "c"], ["d"]]
        assert prtecan.lookup_listoflines(csvl, "c", col=1) == [1]

    @pytest.mark.parametrize(
        ("pattern", "col", "expected"),
        [
            ("pp", 0, [0, 1, 2, 3, 5]),
            ("xy", 1, [0, 5]),
            ("yy", 1, [3]),
            ("zz", 1, []),
        ],
    )
    def test_lookup_listoflines(
        self, pattern: str, col: int, expected: list[int | float]
    ) -> None:
        """Parametrized test for lookup_listoflines with different patterns and columns."""
        csvl: list[list[str | int | float]] = [
            ["pp", "xy", 1, 2.0],
            ["pp", "xx", 1, 2],
            ["pp", 12, 1, 2],
            ["pp", "yy", 1, 2.0],
            ["a"],
            ["pp", "xy", 1, 2],
        ]
        assert prtecan.lookup_listoflines(csvl, pattern=pattern, col=col) == expected


class TestStripLines:
    """Test strip_lines function."""

    def test_strip_empty_fields(self) -> None:
        """It removes empty fields."""
        lines: list[list[str | int | float]] = [["Excitation", "", "", 485.0, "nm", ""]]
        stripped = strip_lines(lines)
        assert stripped == [["Excitation", 485.0, "nm"]]

    def test_empty_input(self) -> None:
        """It handles empty input."""
        assert strip_lines([]) == []


class TestExtractMetadata:
    """Test extract_metadata function."""

    def test_standard_metadata(self) -> None:
        """It extracts standard metadata."""
        lines = create_test_metadata()
        md = extract_metadata(lines)
        assert "Temperature" in md
        assert md["Temperature"].value == 26.0
        assert md["Temperature"].unit == ["°C"]

    def test_single_line_metadata(self) -> None:
        """It extracts single-line metadata."""
        lines: list[list[str | int | float]] = [["Mode", "Fluorescence Top Reading"]]
        md = extract_metadata(lines)
        assert md["Mode"].value == "Fluorescence Top Reading"


class TestMergeMD:
    """Test merge_md function."""

    def test_identical_metadata(self) -> None:
        """It merges identical metadata."""
        md1 = {"Gain": Metadata(93), "Temp": Metadata(25.0, ["°C"])}
        md2 = {"Gain": Metadata(93), "Temp": Metadata(25.0, ["°C"])}
        result = merge_md([md1, md2])
        assert result["Gain"].value == 93
        assert "Temp" in result

    @pytest.mark.parametrize(
        ("md1", "md2", "expected_keys"),
        [
            # Common key with same value
            (
                {"Gain": Metadata(93), "Temp": Metadata(25.0)},
                {"Gain": Metadata(93), "Mode": Metadata("Reading")},
                {"Gain"},
            ),
            # Different values for common key
            ({"Gain": Metadata(93)}, {"Gain": Metadata(94)}, set()),
            # Multiple keys with some missing
            (
                {"Gain": Metadata(93), "Temp": Metadata(25.0), "Extra1": Metadata(1)},
                {
                    "Gain": Metadata(93),
                    "Mode": Metadata("Reading"),
                    "Extra2": Metadata(2),
                },
                {"Gain"},
            ),
            # Empty input
            ([], [], set()),
            # Single metadata dictionary
            ([{"Key": Metadata("value")}], None, {"Key"}),
        ],
    )
    def test_merge_md(
        self,
        md1: dict[str, Metadata] | list[dict[str, Metadata]],
        md2: dict[str, Metadata] | None,
        expected_keys: set[str],
    ) -> None:
        """Test merge_md with various metadata combinations."""
        if isinstance(md1, list):
            mds = md1
        elif md2 is None:
            mds = [md1]
        else:
            mds = [md1, md2]

        result = merge_md(mds)

        # Check that only expected keys are present
        assert set(result.keys()) == expected_keys

        # Verify values for common keys are correct
        for key in expected_keys:
            if len(mds) > 1:
                assert result[key] == mds[0][key]


class TestCalculateConc:
    """Test calculate_conc function."""

    @pytest.mark.parametrize(
        ("additions", "stock", "expected"),
        [
            ([100, 20, 20], 1000, [0.0, 166.66666667, 285.71428571]),
            ([100, 50, 50], 500, [0.0, 166.66666667, 250.0]),
        ],
    )
    def test_calculation(
        self, additions: list[float], stock: float, expected: list[float]
    ) -> None:
        """It calculates concentrations correctly."""
        result = calculate_conc(additions, stock)
        assert_allclose(result, expected, rtol=1e-5)

    def test_zero_additions(self) -> None:
        """It handles empty additions."""
        with pytest.raises(IndexError):
            calculate_conc([], 1000)


class TestDilutionCorrection:
    """Test dilution_correction function."""

    @pytest.mark.parametrize(
        ("additions", "expected"),
        [
            ([100, 50, 50], [1.0, 1.5, 2.0]),
            ([200, 50, 50], [1.0, 1.25, 1.5]),
            ([100], [1.0]),
        ],
    )
    def test_correction_factors(
        self, additions: list[float], expected: list[float]
    ) -> None:
        """It calculates dilution correction factors."""
        result = dilution_correction(additions)
        assert_allclose(result, expected)


class TestLabelblock:
    """Test Labelblock class."""

    @pytest.fixture(scope="class")
    def labelblock(self) -> Labelblock:
        """Create sample labelblock fixture."""
        return create_sample_labelblock()

    def test_metadata_parsing(self, labelblock: Labelblock) -> None:
        """It parses metadata correctly."""
        assert "Temperature" in labelblock.metadata
        assert labelblock.metadata["Temperature"].value == 25.6

    def test_data_extraction(self, labelblock: Labelblock) -> None:
        """It extracts data values."""
        assert "A01" in labelblock.data
        assert isinstance(labelblock.data["A01"], float)

    def test_normalization(self, labelblock: Labelblock) -> None:
        """It normalizes data using metadata."""
        assert "A01" in labelblock.data_nrm
        assert labelblock.data_nrm["A01"] > 0

    def test_equality(self, labelblock: Labelblock) -> None:
        """It correctly implements equality checks."""
        assert labelblock == labelblock  # noqa: PLR0124

    def test_invalid_plate_format(self) -> None:
        """It raises ValueError for invalid plate formats."""
        invalid_lines: list[list[str | int | float]] = [["A01", 100], ["B01", 200]]
        with pytest.raises(ValueError, match="plate"):
            Labelblock(invalid_lines)

    def test_almost_eq(self) -> None:
        """Test the __almost_eq__ method of the Labelblock class."""
        lb0 = create_sample_labelblock()
        file_path1 = Path(data_tests) / "L1" / "290513_7.2.xls"
        csvl1 = prtecan.read_xls(file_path1)  # Gain=98
        idxs1 = prtecan.lookup_listoflines(csvl1)
        lb11 = Labelblock(csvl1[idxs1[1] :])
        file_path2 = Path(data_tests) / "L1" / "290513_8.8.xls"
        csvl2 = prtecan.read_xls(file_path2)  # Gain=99
        idxs2 = prtecan.lookup_listoflines(csvl2)
        lb12 = Labelblock(csvl2[idxs2[1] :])
        assert lb11 != lb12
        assert lb11.__almost_eq__(lb12)
        assert not lb11.__almost_eq__(lb0)

    def test_overvalue(self, caplog: pytest.LogCaptureFixture) -> None:
        """It detects saturated data ("OVER")."""
        csvl = prtecan.read_xls(data_tests / "140220" / "pH6.5_200214.xls")
        idxs = prtecan.lookup_listoflines(csvl)
        with caplog.at_level(logging.WARNING):
            lb = Labelblock(csvl[idxs[0] : idxs[1]])
        expected_messages = [
            " OVER value in Label1: A06 of tecanfile ",
            " OVER value in Label1: H02 of tecanfile ",
        ]
        for expected_message in expected_messages:
            assert any(log.message == expected_message for log in caplog.records), (
                f"Expected log message '{expected_message}' not found"
            )
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
        with pytest.raises(ValueError, match="Row 7 label mismatch: expected H, got "):
            Labelblock(csvl[idxs[0] : len(csvl)])


class TestTecanfile:
    """Test Tecanfile class."""

    @pytest.fixture(scope="class")
    def tecanfile(self) -> Tecanfile:
        """Create tecanfile fixture."""
        return Tecanfile(data_tests / "140220/pH8.3_200214.xls")

    def test_file_loading(self, tecanfile: Tecanfile) -> None:
        """It loads file metadata."""
        assert "Date:" in tecanfile.metadata
        assert tecanfile.metadata["Date:"].value == "20/02/2014"

    def test_labelblocks_parsing(self, tecanfile: Tecanfile) -> None:
        """It parses labelblocks."""
        assert len(tecanfile.labelblocks) >= 1
        assert "Temperature" in tecanfile.labelblocks[1].metadata

    def test_invalid_file(self) -> None:
        """It handles invalid files."""
        with pytest.raises(FileNotFoundError):
            Tecanfile(Path("nonexistent.xls"))

    def test_path(self, tecanfile: Tecanfile) -> None:
        """It reads the file path."""
        assert tecanfile.path == data_tests / "140220/pH8.3_200214.xls"

    def test_detailed_labelblocks(self, tecanfile: Tecanfile) -> None:
        """It parses Temperature metadata and cell data from labelblocks."""
        assert tecanfile.labelblocks[1].metadata["Temperature"].value == 25.3
        assert tecanfile.labelblocks[2].metadata["Temperature"].value == 25.7
        assert tecanfile.labelblocks[1].data["A01"] == 17260
        assert tecanfile.labelblocks[2].data["H12"] == 4196

    def test_eq(self, tecanfile: Tecanfile) -> None:
        """A Tecanfile is equal to itself and not equal to a different Tecanfile."""
        tf1 = prtecan.Tecanfile(data_tests / "140220/pH8.3_200214.xls")
        assert tecanfile == tf1, "Tecanfile is not equal to itself"
        tf2 = prtecan.Tecanfile(data_tests / "140220/pH9.1_200214.xls")
        assert tecanfile != tf2, (
            "Different Tecanfiles are incorrectly reported as equal"
        )

    def test_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        """Warn if labelblocks are repeated in a Tf as it might compromise grouping."""
        with caplog.at_level(logging.WARNING):
            prtecan.Tecanfile(data_tests / "exceptions/290212_7.67_repeated_lb.xls")
        assert any(
            "Repeated labelblocks" in record.message for record in caplog.records
        )

    def test_missing_label(self) -> None:
        """It raises Exception when there is no Label pattern."""
        with pytest.raises(ValueError, match="No Labelblock found"):
            prtecan.Tecanfile(data_tests / "exceptions/0_Labelblocks_290513_5.5.xlsx")


class TestLabelblocksGroup:
    """Test LabelblocksGroup class."""

    @pytest.fixture(scope="class")
    def labelblocks_group(self) -> LabelblocksGroup:
        """Create labelblocks group fixture."""
        tf1 = Tecanfile(data_tests / "L1/290513_5.5.xls")
        tf2 = Tecanfile(data_tests / "L1/290513_7.2.xls")
        return LabelblocksGroup([tf1.labelblocks[1], tf2.labelblocks[1]])

    def test_metadata_merging(self, labelblocks_group: LabelblocksGroup) -> None:
        """It merges metadata from labelblocks."""
        assert "Gain" in labelblocks_group.metadata
        assert labelblocks_group.metadata["Gain"].value == 94

    def test_data_aggregation(self, labelblocks_group: LabelblocksGroup) -> None:
        """It aggregates data from labelblocks."""
        assert "A01" in labelblocks_group.data
        assert len(labelblocks_group.data["A01"]) == 2

    def test_unequal_labelblocks(self) -> None:
        """It handles unequal labelblocks."""
        tf1 = Tecanfile(data_tests / "L1/290513_5.5.xls")
        tf2 = Tecanfile(data_tests / "L1/290513_8.8.xls")
        with pytest.raises(ValueError, match="Creation of labelblock group failed"):
            LabelblocksGroup([tf1.labelblocks[1], tf2.labelblocks[2]])

    def test_detailed_data(self) -> None:
        """Test detailed data aggregation."""
        tfs = [
            Tecanfile(data_tests / "L1" / "290513_5.5.xls"),
            Tecanfile(data_tests / "L1" / "290513_7.2.xls"),
        ]
        lbg = LabelblocksGroup([tfs[0].labelblocks[1], tfs[1].labelblocks[1]])
        assert lbg.data["A01"] == [18713, 17088]
        assert lbg.data["H12"] == [28596, 25771]
        assert_almost_equal(lbg.data_nrm["A01"], [995.372, 908.936], 3)

    def test_metadata_merging_detailed(self) -> None:
        """Test metadata merging with common and uncommon values."""
        tfs = [
            Tecanfile(data_tests / "L1" / "290513_7.2.xls"),
            Tecanfile(data_tests / "L1" / "290513_8.8.xls"),
        ]
        lbg = LabelblocksGroup([tfs[0].labelblocks[2], tfs[1].labelblocks[2]])
        # Gain values differ (98 vs 99), so Gain should not be in merged metadata
        assert lbg.metadata.get("Gain") is None
        assert lbg.labelblocks[0].metadata["Gain"].value == 98
        assert lbg.labelblocks[1].metadata["Gain"].value == 99
        # But normalized data should still be available
        assert_almost_equal(lbg.data_nrm["H12"], [693.980, 714.495], 3)


class TestPlateScheme:
    """Test PlateScheme class."""

    def test_buffer_wells(self) -> None:
        """It handles buffer wells."""
        ps = PlateScheme()
        ps.buffer = ["A01", "B01"]
        assert ps.buffer == ["A01", "B01"]

    def test_from_file(self) -> None:
        """It loads scheme from file."""
        ps = PlateScheme(data_tests / "140220/scheme.txt")
        assert "buffer" not in ps.names
        assert len(ps.buffer) > 0
        assert "G03" in ps.names
        assert "V224Q" in ps.names

    def test_buffer_validation(self) -> None:
        """Test buffer setter with type validation."""
        ps = PlateScheme()
        ps.buffer = ["A1", "A2"]
        assert ps.buffer == ["A1", "A2"]
        with pytest.raises(TypeError):
            ps.buffer = [1, 2]  # type: ignore[list-item]

    def test_ctrl_validation(self) -> None:
        """Test ctrl setter with type validation."""
        ps = PlateScheme()
        ps.ctrl = ["B1", "B2"]
        assert ps.ctrl == ["B1", "B2"]
        with pytest.raises(TypeError):
            ps.ctrl = [1, 2]  # type: ignore[list-item]

    def test_names_validation(self) -> None:
        """Test names setter with type validation."""
        ps = PlateScheme()
        ps.names = {"name1": {"A1", "A2"}, "name2": {"B1", "B2"}}
        assert ps.names == {"name1": {"A1", "A2"}, "name2": {"B1", "B2"}}
        with pytest.raises(TypeError):
            ps.names = {"name1": [1, 2], "name2": [3, 4]}  # type: ignore[dict-item]

    def test_invalid_file(self) -> None:
        """Test providing an incorrect file."""
        with pytest.raises(FileNotFoundError):
            PlateScheme(file=Path("incorrect_file.csv"))


class TestTitrationConfig:
    """Test TitrationConfig class."""

    def test_config_creation(self) -> None:
        """It creates configuration with default values."""
        config = TitrationConfig()
        assert config.bg is True
        assert config.bg_adj is False

    def test_callback(self) -> None:
        """It triggers callback on parameter change."""
        callback_called = False

        def callback() -> None:
            nonlocal callback_called
            callback_called = True

        config = TitrationConfig()
        config.set_callback(callback)
        config.bg = False
        assert callback_called


class TestBufferFit:
    """Test BufferFit class."""

    def test_empty_property(self) -> None:
        """It correctly identifies empty fits."""
        empty_fit = BufferFit()
        assert empty_fit.empty is True

        non_empty_fit = BufferFit(1.0, 0.0, 0.1, 0.2)
        assert non_empty_fit.empty is False


class TestBuffer:
    """Test Buffer class."""

    @pytest.fixture(scope="class")
    def titration(self) -> Titration:
        """Create titration fixture."""
        tf = Tecanfile(data_tests / "140220/pH6.5_200214.xls")
        return Titration([tf], x=np.array([6.5]), is_ph=True)

    @pytest.fixture
    def buffer(self, titration: Titration) -> Buffer:
        """Create buffer fixture from titration."""
        return titration.buffer

    def test_wells_setter(self, buffer: Buffer) -> None:
        """It sets buffer wells and clears cache."""
        buffer.wells = ["D01", "E01"]
        assert buffer.wells == ["D01", "E01"]

    def test_empty_wells(self, buffer: Buffer) -> None:
        """It handles empty buffer wells."""
        buffer.wells = []
        assert buffer.dataframes == {}


class TestTitrationResults:
    """Test TitrationResults class."""

    @pytest.fixture
    def titration_results(self) -> TitrationResults:
        """Create titration results fixture."""
        scheme = PlateScheme()
        scheme.names = {"sample1": {"A01"}}
        return TitrationResults(scheme, {"A01"}, lambda _: FitResult())

    def test_dataframe_property(self, titration_results: TitrationResults) -> None:
        """It creates a DataFrame from results."""
        df = titration_results.dataframe
        assert isinstance(df, pd.DataFrame)
        assert "A01" in df.index

    def test_lazy_computation(self, titration_results: TitrationResults) -> None:
        """It computes results lazily."""
        assert not titration_results.results  # Should be empty before access
        _ = titration_results["A01"]  # Trigger computation
        assert "A01" in titration_results.results


class TestTitration:
    """Test Titration class - basic functionality."""

    @pytest.fixture(scope="class")
    def titration(self) -> Titration:
        """Create titration fixture for testing."""
        tf = Tecanfile(data_tests / "140220/pH6.5_200214.xls")
        return Titration([tf], x=np.array([6.5]), is_ph=True)

    def test_from_listfile(self) -> None:
        """It creates titration from list file."""
        tit = Titration.fromlistfile(data_tests / "140220/list.pH.csv", is_ph=True)
        assert len(tit.x) > 0
        assert len(tit.tecanfiles) > 0

    def test_buffer_handling(self, titration: Titration) -> None:
        """It handles buffer wells."""
        titration.buffer.wells = ["D01", "E01"]
        assert titration.buffer.wells == ["D01", "E01"]

    def test_additions_loading(self, titration: Titration) -> None:
        """It loads additions from file."""
        titration.load_additions(data_tests / "140220/additions.pH")
        assert titration.additions is not None
        assert len(titration.additions) > 0

    def test_scheme_loading(self, titration: Titration) -> None:
        """It loads plate scheme from file."""
        titration.load_scheme(data_tests / "140220/scheme.txt")
        assert len(titration.scheme.buffer) > 0


class TestTecanConfig:
    """Test TecanConfig class."""

    def test_config_creation(self, tmp_path: Path) -> None:
        """It creates configuration with all parameters."""
        config = TecanConfig(
            out_fp=tmp_path, comb=True, lim=(0, 10), title="Test", fit=True, png=True
        )
        assert config.out_fp == tmp_path
        assert config.comb is True


# =============================================================================
# Integration Tests
# =============================================================================


def test_end_to_end_titration_processing(tmp_path: Path) -> None:
    """Test complete workflow from file loading to fitting."""
    # Load titration
    tit = Titration.fromlistfile(data_tests / "140220/list.pH.csv", is_ph=True)

    # Load additions and scheme
    tit.load_additions(data_tests / "140220/additions.pH")
    tit.load_scheme(data_tests / "140220/scheme.txt")

    # Configure
    tit.params.bg = True
    tit.params.dil = True
    tit.params.nrm = True

    # Export
    config = TecanConfig(
        out_fp=tmp_path, comb=False, lim=None, title="Test", fit=True, png=False
    )
    tit.export_data_fit(config)

    # Verify output files were created
    assert (tmp_path / "dat_bg_dil_nrm").exists()
    assert (tmp_path / "dat_bg_dil_nrm/fit").exists()


# =============================================================================
# Test Suite for Suggestions
# =============================================================================


class TestSuggestedTestCases:
    """Test cases based on the provided suggestions.py.txt."""

    def test_dilution_correction_standard(self) -> None:
        """Test dilution correction function returns cumulative volume ratio."""
        additions = [100.0, 50.0, 50.0]
        # volumes = [100,150,200] -> corrections = [1.0,1.5,2.0]
        corr = prtecan.dilution_correction(additions)
        assert_array_equal(corr, np.array([1.0, 1.5, 2.0]))

    def test_plate_scheme_discard_and_nofit_keys(self) -> None:
        """Test discard setter and nofit_keys property of PlateScheme."""
        ps = PlateScheme()
        ps.buffer = ["A01", "B01"]
        ps.discard = ["C01"]
        # nofit_keys is union of buffer and discard
        assert set(ps.nofit_keys) == {"A01", "B01", "C01"}

        with pytest.raises(TypeError):
            # non-str entries
            ps.discard = [1, 2]  # type: ignore[list-item]

    def test_titration_config_callback_trigger(self) -> None:
        """Test that TitrationConfig triggers callback on attribute change."""
        cfg = TitrationConfig()
        events = []
        cfg.set_callback(lambda: events.append(True))
        # change boolean attribute
        cfg.bg = not cfg.bg
        assert events == [True]
        # setting same value does not re-trigger
        cfg.bg = cfg.bg
        assert events == [True]
        # change string attribute
        cfg.bg_mth = "fit"
        assert len(events) == 2

    def test_bufferfit_empty_flag(self) -> None:
        """Test BufferFit.empty property for NaN and non-NaN values."""
        bf = BufferFit()
        assert bf.empty
        bf2 = BufferFit(m=1.0, q=0.0, m_err=0.1, q_err=0.2)
        assert not bf2.empty

    def test_generate_combinations_and_prepare_folder(self, tmp_path: Path) -> None:
        """Test generation of parameter combinations and output folder naming."""
        data_tests = Path(__file__).parent / "Tecan"
        # use existing list file for minimal Titration
        tit = Titration.fromlistfile(data_tests / "L1" / "list.pH.csv", is_ph=True)
        combos = tit._generate_combinations()  # noqa: SLF001
        # 2^4 boolean flags times 3 methods
        assert len(combos) == 16 * 3
        flags, method = combos[0]
        assert isinstance(flags, tuple)
        assert len(flags) == 4
        assert method in ("mean", "meansd", "fit")
        # test prepare output folder naming
        # set all flags to True and bg_mth to 'fit'
        tit.params.bg = True
        tit.params.bg_adj = True
        tit.params.dil = True
        tit.params.nrm = True
        tit.params.bg_mth = "fit"
        out = tit._prepare_output_folder(tmp_path)  # noqa: SLF001
        name = out.name
        assert "_bg" in name
        assert "_adj" in name
        assert "_dil" in name
        assert "_nrm" in name
        assert "_fit" in name

    def test_extract_xls_roundtrip(self, tmp_path: Path) -> None:
        """Test read_xls and strip_lines integration with a temporary CSV file."""
        # create a small Excel file
        test_df = pd.DataFrame(
            [[1, None, "a"], [None, 2, None]], columns=["x", "y", "z"]
        )
        path = tmp_path / "test.xls"
        test_df.to_excel(path, index=False)
        lines = prtecan.read_xls(path)
        # strip_lines should remove blanks
        stripped = prtecan.strip_lines(lines)
        # each line has no empty elements
        assert all(all(e != "" for e in row) for row in stripped)


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


def test_buffer_empty_wells() -> None:
    """Test Buffer Class with empty wells list."""
    tf = prtecan.Tecanfile(data_tests / "140220/pH6.5_200214.xls")
    tit = prtecan.Titration([tf], x=np.array([6.5]), is_ph=True)

    # Test with empty buffer wells
    tit.buffer.wells = []
    assert tit.buffer.dataframes == {}
    assert tit.buffer.dataframes_nrm == {}

    # Test plot with empty buffers
    g = tit.buffer.plot()
    assert isinstance(g, sns.FacetGrid)


def test_titration_results_empty() -> None:
    """Test TitrationResults with empty data."""
    scheme = prtecan.PlateScheme()
    empty_results = prtecan.TitrationResults(
        scheme=scheme, fit_keys=set(), compute_func=lambda _: FitResult()
    )

    assert len(empty_results) == 0
    assert empty_results.dataframe.empty

    # Test accessing non-existent key
    with pytest.raises(KeyError):
        _ = empty_results["A01"]


class TestCsvlFunctions:
    """Test TecanFile reading and parsing functions."""

    csvl = prtecan.read_xls(data_tests / "140220/pH8.3_200214.xls")

    def test_read_xls(self) -> None:
        """The test reads the xls file using cls method."""
        assert len(self.csvl) == 74

    def test_lookup_listoflines(self) -> None:
        """It finds Label occurrences using module function."""
        assert prtecan.lookup_listoflines(self.csvl) == [14, 44]


class TestTecanfilesGroup:
    """Group tecanfiles properly."""

    class TestAllEqLbgs:
        """Test TfG with 2 LbG in the same order."""

        @pytest.fixture(autouse=True, scope="class")
        def tfg(self) -> TecanfilesGroup:
            """Set up TecanfilesGroup."""
            filenames = ["290513_5.5.xls", "290513_7.2.xls"]
            tecanfiles = [Tecanfile(data_tests / "L1" / f) for f in filenames]
            return TecanfilesGroup(tecanfiles)

        def test_metadata(self, tfg: TecanfilesGroup) -> None:
            """Parse general metadata."""
            assert tfg.metadata["Plate"].value == "PE 96 Flat Bottom White   [PE.pdfx]"
            assert tfg.metadata["System"].value == "TECANROBOT"

        def test_labelblocksgroups(self, tfg: TecanfilesGroup) -> None:
            """Generate 2 LbG with .data and .metadata."""
            lbg1 = tfg.labelblocksgroups[1]
            lbg2 = tfg.labelblocksgroups[2]
            # metadata
            assert lbg1.metadata["Number of Flashes"].value == 10.0
            assert lbg2.metadata["Gain"].value == 98.0
            # data normalized ... enough in lbg
            # data
            assert lbg1.data is not None
            assert lbg1.data["A01"] == [18713, 17088]
            assert lbg1.data["H12"] == [28596, 25771]
            assert lbg2.data is not None
            assert lbg2.data["A01"] == [7878, 8761]
            assert lbg2.data["H12"] == [14226, 13602]

    class TestAlmostEqLbgs:
        """Test TfG when 1 LbG equal and a second with almost equal labelblocks."""

        @pytest.fixture
        def tfg_warn(
            self, caplog: pytest.LogCaptureFixture
        ) -> tuple[TecanfilesGroup, list[logging.LogRecord]]:
            """Set up TecanfilesGroup with Warning."""
            filenames = [
                "290513_5.5.xls",  # Label1 and Label2
                "290513_7.2.xls",  # Label1 and Label2
                "290513_8.8.xls",  # Label1 and Label2 with different metadata
            ]
            tecanfiles = [Tecanfile(data_tests / "L1" / f) for f in filenames]
            with caplog.at_level(logging.WARNING):
                tfg = TecanfilesGroup(tecanfiles)
            return tfg, caplog.records

        def test_log_warning(
            self, tfg_warn: tuple[TecanfilesGroup, list[logging.LogRecord]]
        ) -> None:
            """Warn about labelblocks anomaly."""
            _, records = tfg_warn
            # Check that the specific warning message was logged
            assert any(
                "Different LabelblocksGroup across files" in record.message
                for record in records
            )

        def test_labelblocksgroups(
            self, tfg_warn: tuple[TecanfilesGroup, list[logging.LogRecord]]
        ) -> None:
            """Generate 1 LbG with .data and .metadata."""
            lbg1 = tfg_warn[0].labelblocksgroups[1]
            # metadata
            assert lbg1.metadata["Number of Flashes"].value == 10.0
            assert lbg1.metadata["Gain"].value == 94
            # data
            assert lbg1.data is not None
            assert lbg1.data["A01"] == [18713.0, 17088.0, 17123.0]
            assert lbg1.data["H12"] == [28596.0, 25771.0, 28309.0]

        def test_mergeable_labelblocksgroups(
            self, tfg_warn: tuple[TecanfilesGroup, list[logging.LogRecord]]
        ) -> None:
            """Generate 1 Lbg only with .data_normalized and only common .metadata."""
            lbg2 = tfg_warn[0].labelblocksgroups[2]
            # metadata
            assert lbg2.metadata["Number of Flashes"].value == 10.0
            assert lbg2.metadata.get("Gain") is None
            assert lbg2.data == {}
            # data_normalized
            assert_almost_equal(lbg2.data_nrm["A01"], [401.9387755, 446.9897959, 450.0])
            assert_almost_equal(
                lbg2.data_nrm["H12"], [725.8163265, 693.9795918, 714.4949494]
            )

    class TestOnly1commonLbg:
        """Test TfG with different number of labelblocks, but mergeable."""

        @pytest.fixture
        def tfg_warn(
            self, caplog: pytest.LogCaptureFixture
        ) -> tuple[TecanfilesGroup, list[logging.LogRecord]]:
            """Set up TecanfilesGroup with Warning."""
            filenames = [
                "290212_5.78.xls",  # Label1 and Label2
                "290212_20.xls",  # Label2 only
                "290212_100.xls",  # Label2 only
            ]
            tecanfiles = [Tecanfile(data_tests / f) for f in filenames]
            with caplog.at_level(logging.WARNING):
                tfg = TecanfilesGroup(tecanfiles)
            return tfg, caplog.records

        def test_log_warning(
            self, tfg_warn: tuple[TecanfilesGroup, list[logging.LogRecord]]
        ) -> None:
            """Warn about labelblocks anomaly."""
            _, records = tfg_warn
            # Check that the specific warning message was logged
            assert any(
                "Different LabelblocksGroup across files" in record.message
                for record in records
            )

        def test_labelblocksgroups(
            self, tfg_warn: tuple[TecanfilesGroup, list[logging.LogRecord]]
        ) -> None:
            """Generates 1 LbG with .data and .metadata."""
            tfg, _ = tfg_warn
            lbg2 = tfg.labelblocksgroups[2]
            # metadata
            assert lbg2.metadata["Number of Flashes"].value == 10.0
            assert lbg2.metadata["Gain"].value == 93.0
            # data
            assert lbg2.data is not None
            assert lbg2.data["A01"] == [6289, 6462, 6465]
            assert lbg2.data["H12"] == [4477, 4705, 4918]

    class TestFailToMerge:
        """Test TfG without mergeable labelblocks."""

        filenames: ClassVar[list[str]] = ["290513_5.5.xls", "290513_5.5_bad.xls"]
        tecanfiles: ClassVar[list[prtecan.Tecanfile]] = [
            prtecan.Tecanfile(data_tests / f) for f in filenames
        ]

        def test_raise_exception(self) -> None:
            """Raise Exception when there is no way to build labelblocksGroup."""
            msg = r"No common labelblocks in files: ['290513_5.5.xls', '290513_5.5_bad.xls']."
            with pytest.raises(ValueError, match=re.escape(msg)):
                prtecan.TecanfilesGroup(self.tecanfiles)


class TestTitrationAdvanced:
    """Test Titration class - comprehensive advanced tests."""

    @pytest.fixture
    def tit(self) -> Titration:
        """Set up L1 pH titration: 1 lbg without scheme and additions."""
        return Titration.fromlistfile(data_tests / "L1" / "list.pH.csv", is_ph=True)

    @pytest.fixture(scope="class")
    def tit_ph(self) -> Titration:
        """Set up a pH titration."""
        return Titration.fromlistfile(data_tests / "140220" / "list.pH.csv", is_ph=True)

    @pytest.fixture(scope="class")
    def tit_cl(self) -> Titration:
        """Set up a Cl titration."""
        return Titration.fromlistfile(data_tests / "140220" / "list.cl.csv", is_ph=True)

    @pytest.fixture(scope="class")
    def tit1(self) -> Titration:
        """Set up a titration with a single Tecan file."""
        tf = prtecan.Tecanfile(data_tests / "140220" / "pH6.5_200214.xls")
        return prtecan.Titration([tf], x=np.array([6.5]), is_ph=True)

    def test_conc(self, tit_ph: Titration) -> None:
        """It reads pH values."""
        assert_array_equal(tit_ph.x, [9.0633, 8.35, 7.7, 7.08, 6.44, 5.83, 4.99])

    def test_labelblocksgroups(self, tit_ph: Titration) -> None:
        """It reads labelblocksgroups data and metadata."""
        lbg1 = tit_ph.labelblocksgroups[1]
        lbg2 = tit_ph.labelblocksgroups[2]
        # metadata
        assert lbg1.metadata["Number of Flashes"].value == 10.0
        assert lbg2.metadata["Gain"] == prtecan.Metadata(56.0)
        # data
        assert lbg1.data is not None
        assert lbg2.data is not None
        assert lbg1.data["A01"][::2] == [14798.0, 20142.0, 22915.0, 22060.0]
        assert lbg2.data["A01"][1::2] == [3761.0, 835.0, 347.0]
        assert lbg1.data["H12"][1::2] == [16345.0, 21719.0, 23532.0]
        assert lbg2.data["H12"] == [5372.0, 4196.0, 2390.0, 1031.0, 543.0, 427.0, 371.0]

    def test_export_data(self, tit_ph: Titration, tmp_path: Path) -> None:
        """It exports titrations data to files e.g. "A01.dat"."""
        tit_ph.params.bg = False
        tit_ph.params.dil = False
        tit_ph.params.nrm = False
        tecan_config = prtecan.TecanConfig(
            tmp_path,
            comb=False,
            lim=None,
            title="",
            fit=False,
            png=False,
        )
        tit_ph.export_data_fit(tecan_config)
        a01 = pd.read_csv(tmp_path / "dat" / "A01.dat")
        h12 = pd.read_csv(tmp_path / "dat" / "H12.dat")
        assert a01["y1"].tolist()[0::2] == [14798.0, 20142.0, 22915.0, 22060.0]
        assert a01["y2"].tolist()[1::2] == [3761.0, 835.0, 347.0]
        assert h12["y1"].tolist()[1::2] == [16345.0, 21719.0, 23532.0]
        assert h12["y2"].tolist()[1:] == [4196.0, 2390.0, 1031.0, 543.0, 427.0, 371.0]

    def test_data_buffersubtracted(self, tit: Titration) -> None:
        """Check data after normalization and bg subtraction."""
        tit.buffer.wells = ["C12", "D01", "D12", "E01", "E12", "F01"]
        tit.params.nrm = False
        assert tit.data[1]
        assert tit.data[2] == {}
        sliced_values = tit.data[1]["B07"][-1::-3][:2]
        assert_almost_equal(sliced_values, [7069, 5716.7], 1)
        # normalization
        tit.params.nrm = True
        sliced_values0 = tit.data[1]["B07"][-1::-3][:2]
        sliced_values1 = tit.data[2]["B07"][-4::-3]
        assert_almost_equal(sliced_values0, [376.01, 304.08], 2)
        assert_almost_equal(sliced_values1, [355.16, 348.57], 2)

    def test_labelblocksgroups_cl(self, tit_cl: Titration) -> None:
        """It reads labelblocksgroups data for Cl too."""
        lbg1 = tit_cl.labelblocksgroups[1]
        lbg2 = tit_cl.labelblocksgroups[2]
        assert lbg1.data is not None
        assert lbg1.data["A01"][1::2] == [16908.0, 14719.0, 14358.0, 14520.0]
        assert lbg2.data["A01"][1::2] == [167.0, 109.0, 87.0, 81.0]
        assert lbg2.data["H12"][1::2] == [223.0, 141.0, 120.0, 100.0]

    def test_raise_listfilenotfound(self) -> None:
        """It raises FileNotFoundError when list.xx file does not exist."""
        with pytest.raises(FileNotFoundError, match="Cannot find: aax"):
            Titration.fromlistfile(Path("aax"), True)

    def test_bad_listfile(self) -> None:
        """It raises Exception when list.xx file is ill-shaped."""
        with pytest.raises(ValueError, match=r"Check format .* for listfile: .*"):
            Titration.fromlistfile(data_tests / "140220" / "list.pH2.csv", True)

    def test_data_bg_and_nrm(self, tit1: Titration) -> None:
        """Calculate buffer value from average of buffer wells and subtract."""
        tit1.buffer.wells = ["D01", "D12", "E01", "E12"]
        tit1.params.nrm = False
        tit1.params.dil = False
        tit1.params.bg = False
        assert tit1.buffer.dataframes[1]["sem"][0] == pytest.approx(259.9514)
        assert tit1.buffer.dataframes[2]["sem"][0] == pytest.approx(2.561738)
        tit1.params.bg = True
        assert tit1.data[1]["F06"][0] == pytest.approx(7661.75)
        assert tit1.data[2]["H12"][0] == pytest.approx(486.25)
        # Can also assign a buffer value.
        tit1.bg = {1: np.array([1.0]), 2: np.array([2.9])}
        assert tit1.data[1]["F06"][0] == 19550
        assert tit1.data[2]["H12"][0] == 540.1
        # nrm
        assert tit1.buffer.dataframes_nrm[1]["fit"][0] == pytest.approx(639.20699)
        assert tit1.buffer.dataframes_nrm[1]["mean"][0] == pytest.approx(639.20699)
        assert tit1.buffer.dataframes_nrm[2]["fit"][0] == pytest.approx(5.06696)
        assert tit1.buffer.dataframes_nrm[1]["sem"][0] == pytest.approx(13.97588)
        assert tit1.buffer.dataframes_nrm[2]["sem"][0] == pytest.approx(0.2287266)
        # also bg duplicates data in buffers_nrm
        tit1.params.nrm = True
        tit1.buffer.wells = ["D01", "D12", "E01", "E12"]
        assert tit1.bg[1][0] == pytest.approx(639.20699)
        assert tit1.bg[2] == pytest.approx(5.06696)
        # nrm data
        assert tit1.data[1]["F06"] == pytest.approx(411.922)
        assert tit1.data[2]["H12"] == pytest.approx(43.4152)
        # Can also assign a buffer_norm value.
        tit1.bg = {1: np.array([1.0]), 2: np.array([0.4821])}
        assert tit1.data[1]["F06"] == pytest.approx(1050.13)
        assert tit1.data[2]["H12"] == pytest.approx(48.0)

    # Buffer
    def test_plot_buffer_1lbg(self, tit: Titration) -> None:
        """It plots buffers with 1 lbg and norm with 2 lbg because one lbg is mergeable."""
        tit.load_additions(data_tests / "L1/additions.pH")
        tit.load_scheme(data_tests / "L1/scheme.txt")
        g = tit.buffer.plot()
        assert isinstance(g, sns.FacetGrid)
        assert len(g.axes_dict) == 1
        g = tit.buffer.plot(nrm=True)
        assert isinstance(g, sns.FacetGrid)
        assert len(g.axes_dict) == 2

    def test_plot_buffer_empty_buffers(self, tit: Titration) -> None:
        """It handles empty buffers (before assignment of buffer_wells)."""
        g = tit.buffer.plot()
        assert isinstance(g, sns.FacetGrid)


# some:  @pytest.mark.skipif(sys.platform == "win32", reason="broken on windows")
class TestTitrationAnalysis:
    """Test TitrationAnalysis class."""

    @pytest.fixture(autouse=True, scope="class")
    def titan(self) -> Titration:
        """Set up TitrationAnalysis."""
        titan = Titration.fromlistfile(data_tests / "140220/list.pH.csv", is_ph=True)
        titan.load_additions(data_tests / "140220/additions.pH")
        titan.load_scheme(data_tests / "140220/scheme.txt")
        return titan

    def test_scheme(self, titan: Titration) -> None:
        """It finds well position for buffer samples."""
        assert titan.scheme.buffer == ["D01", "E01", "D12", "E12"]
        assert titan.buffer.wells == ["D01", "E01", "D12", "E12"]

    def test_raise_listfilenotfound(self, titan: Titration) -> None:
        """It raises OSError when scheme file does not exist."""
        with pytest.raises(
            FileNotFoundError, match=r"No such file or directory: 'aax'"
        ):
            titan.load_scheme(Path("aax"))

    def test_raise_listfile_exception(self, titan: Titration) -> None:
        """It raises AssertionError when scheme.txt file is ill-shaped."""
        bad_schemefile = data_tests / "140220/scheme0.txt"
        msg = f"Check format [well sample] for schemefile: {bad_schemefile}"
        with pytest.raises(ValueError, match=re.escape(msg)):
            titan.load_scheme(bad_schemefile)

    def test_subtract_bg(self, titan: Titration) -> None:
        """It subtracts buffer average values."""
        lbg0 = titan.labelblocksgroups[1]
        lbg1 = titan.labelblocksgroups[2]
        assert_almost_equal(
            lbg0.data_nrm["E01"][::2], [601.72, 641.505, 674.355, 706.774], 3
        )
        assert lbg0.data is not None
        assert lbg0.data["E01"][::2] == [11192.0, 11932.0, 12543.0, 13146.0]
        assert type(lbg1) is LabelblocksGroup
        titan.params.bg = True
        titan.params.nrm = False
        titan.params.dil = False
        assert_array_equal(titan.data[1]["A12"][::3], [8084.5, 16621.75, 13775.0])
        assert lbg1.data is not None
        assert_array_equal(titan.data[2]["A12"][::3], [9758.25, 1334.0, 283.5])

    def test_dilution_correction(self, titan: Titration) -> None:
        """It applies dilution correction read from file listing additions."""
        assert titan.additions is not None
        assert_array_equal(titan.additions, [100, 2, 2, 2, 2, 2, 2])
        titan.params.nrm = False
        titan.params.dil = True
        assert titan.data is not None
        assert titan.data[1] is not None
        assert_almost_equal(
            titan.data[2]["A12"],
            [9758.25, 7524.795, 3079.18, 1414.04, 641.79, 402.325, 317.52],
        )

    def test_data_nrm(self, titan: Titration) -> None:
        """It normalizes data."""
        titan.params.nrm = True
        titan.params.bg = True
        titan.params.dil = True

        assert_almost_equal(
            titan.data[1]["A12"][::2],
            [434.65, 878.73, 975.58, 829.46],
            2,
        )
        assert_almost_equal(
            titan.data[2]["A12"][::2],
            [871.272, 274.927, 57.303, 28.35],
            3,
        )

    def test_keys(self, titan: Titration) -> None:
        """It gets well positions for ctrl and unknown samples."""
        assert set(titan.scheme.names) == {"NTT", "G03", "V224Q", "S202N"}
        x = {"B12", "H12", "F01", "C12", "F12", "C01", "H01", "G12", "B01", "G01"}
        assert set(titan.scheme.ctrl) - {"A01", "A12"} == x

    def test_fit(self, titan: Titration) -> None:
        """It fits each label separately."""
        fres = titan.results
        # Check that the first fit result dictionary has 92 elements
        fres[1].compute_all()
        assert len(fres[1]) == 92
        # Check that the first fit result for 'H02' is None
        assert fres[1]["H02"] == FitResult(None, None, None)
        # Check that the second fit result for 'H02' is not None
        assert fres[2]["H02"].is_valid()
        # Check 'K' and std error for 'H02' in the second fit result
        assert fres[2]["H02"].result is not None
        k_h02 = fres[2]["H02"].result.params["K"]
        assert k_h02.value == pytest.approx(7.898, abs=1e-3)
        assert k_h02.stderr == pytest.approx(0.026, abs=1e-3)
        # Check 'K' and std error for 'H02' in the third fit result
        assert titan.result_global["H02"].result is not None
        k_h02 = titan.result_global["H02"].result.params["K"]
        assert k_h02.value == pytest.approx(7.890, abs=1e-3)
        assert k_h02.stderr == pytest.approx(0.017, abs=1e-3)
        # Check 'K' and std error for 'E02' in the second fit result
        assert fres[2]["E02"].result is not None
        k_e02 = fres[2]["E02"].result.params["K"]
        assert k_e02.value == pytest.approx(7.978, abs=1e-3)
        assert k_e02.stderr == pytest.approx(0.038, abs=1e-3)
        # Check 'K' and std error for 'E02' in the third fit result
        assert titan.result_global["E02"].result is not None
        k_e02 = titan.result_global["E02"].result.params["K"]
        assert k_e02.value == pytest.approx(7.978, abs=1e-3)
        assert k_e02.stderr == pytest.approx(0.017, abs=1e-3)
        # Fit up to the second-last data point

    def test_plot_buffer_with_title(self, titan: Titration) -> None:
        """It plots buffers for 2 lbg with title."""
        g = titan.buffer.plot(title="Test Title")
        assert isinstance(g, sns.FacetGrid)
        assert len(g.axes_dict) == 2
        assert g.fig._suptitle.get_text() == "Test Title"  # noqa: SLF001

    def test_plot_buffer_normalized(self, titan: Titration) -> None:
        """It plots buffers_norm for 2 lbg."""
        g = titan.buffer.plot(nrm=True)
        assert isinstance(g, sns.FacetGrid)
        assert len(g.axes_dict) == 2


#########
# Extra #
#########
def test_dilution_correction_standard() -> None:
    """Test dilution correction function returns cumulative volume ratio."""
    additions = [100.0, 50.0, 50.0]
    # volumes = [100,150,200] -> corrections = [1.0,1.5,2.0]
    corr = prtecan.dilution_correction(additions)
    assert_array_equal(corr, np.array([1.0, 1.5, 2.0]))


def test_plate_scheme_discard_and_nofit_keys() -> None:
    """Test discard setter and nofit_keys property of PlateScheme."""
    ps = prtecan.PlateScheme()
    ps.buffer = ["A01", "B01"]
    ps.discard = ["C01"]
    # nofit_keys is union of buffer and discard
    assert set(ps.nofit_keys) == {"A01", "B01", "C01"}
    with pytest.raises(TypeError):
        #  non-str entries
        ps.discard = [1, 2]  # type: ignore[list-item]


def test_titration_config_callback_trigger() -> None:
    """Test that TitrationConfig triggers callback on attribute change."""
    cfg = prtecan.TitrationConfig()
    events = []
    cfg.set_callback(lambda: events.append(True))
    # change boolean attribute
    cfg.bg = not cfg.bg
    assert events == [True]
    # setting same value does not re-trigger
    cfg.bg = cfg.bg
    assert events == [True]
    # change string attribute
    cfg.bg_mth = "fit"
    assert len(events) == 2


def test_bufferfit_empty_flag() -> None:
    """Test BufferFit.empty property for NaN and non-NaN values."""
    bf = BufferFit()
    assert bf.empty
    bf2 = prtecan.BufferFit(m=1.0, q=0.0, m_err=0.1, q_err=0.2)
    assert not bf2.empty


def test_generate_combinations_and_prepare_folder(tmp_path: Path) -> None:
    """Test generation of parameter combinations and output folder naming."""
    data_tests = Path(__file__).parent / "Tecan"
    # use existing list file for minimal Titration
    tit = prtecan.Titration.fromlistfile(data_tests / "L1" / "list.pH.csv", is_ph=True)
    combos = tit._generate_combinations()  # noqa: SLF001
    # 2^4 boolean flags times 3 methods
    assert len(combos) == 16 * 3
    flags, method = combos[0]
    assert isinstance(flags, tuple)
    assert len(flags) == 4
    assert method in ("mean", "meansd", "fit")
    # test prepare output folder naming
    # set all flags to True and bg_mth to 'fit'
    tit.params.bg = True
    tit.params.bg_adj = True
    tit.params.dil = True
    tit.params.nrm = True
    tit.params.bg_mth = "fit"
    out = tit._prepare_output_folder(tmp_path)  # noqa: SLF001
    name = out.name
    assert "_bg" in name
    assert "_adj" in name
    assert "_dil" in name
    assert "_nrm" in name
    assert "_fit" in name


def test_extract_xls_roundtrip(tmp_path: Path) -> None:
    """Test read_xls and strip_lines integration with a temporary CSV file."""
    # create a small Excel file
    test_df = pd.DataFrame([[1, None, "a"], [None, 2, None]], columns=["x", "y", "z"])
    path = tmp_path / "test.xls"
    test_df.to_excel(path, index=False)
    lines = prtecan.read_xls(path)
    # strip_lines should remove blanks
    stripped = prtecan.strip_lines(lines)
    # each line has no empty elements
    assert all(all(e != "" for e in row) for row in stripped)
