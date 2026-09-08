"""Comprehensive test suite for prtecan module."""

from __future__ import annotations

import dataclasses
import inspect
import logging
import math
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pymc as pm
import pytest
import seaborn as sns  # type: ignore[import-untyped]
from lmfit import Parameters
from numpy.testing import assert_allclose, assert_almost_equal, assert_array_equal

from clophfit import prtecan
from clophfit.fitting import bayes
from clophfit.fitting.bayes_config import NoiseConfig, SamplerConfig
from clophfit.fitting.core import fit_binding_glob
from clophfit.fitting.data_structures import (
    DataArray,
    Dataset,
    FitResult,
    PlateNoiseModel,
)
from clophfit.fitting.model_validation import RESIDUAL_TABLE_COLUMNS, RobustZMad
from clophfit.fitting.models import binding_1site
from clophfit.fitting.plate_lm import fit_plate_lm
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
    export,
    extract_metadata,
    merge_md,
    strip_lines,
    titration as titration_module,
)
from clophfit.prtecan.export import (
    export_data_fit,
    export_trace_summary,
    generate_combinations,
    prepare_output_folder,
)
from clophfit.prtecan.titration import McmcSpec


class _StopBayesBuildError(Exception):
    """Stop a fit once the call has been inspected, without sampling."""


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

    def test_dilution_correction_edge_cases(self) -> None:
        """Test dilution correction with edge cases."""
        # Empty list returns empty array
        result = dilution_correction([])
        assert len(result) == 0
        # Single element
        result = dilution_correction([100.0])
        assert_allclose(result, [1.0])
        # Zero initial volume raises ValueError
        with pytest.raises(ValueError, match=r"Initial volume .* cannot be zero"):
            dilution_correction([0.0, 100.0])


class TestLabelblock:
    """Test Labelblock class."""

    @staticmethod
    def _get_two_labelblocks() -> tuple[Labelblock, Labelblock]:
        """Simulate csvl with 2 labelblocks."""
        csvl = prtecan.read_xls(data_tests / "140220/pH6.5_200214.xls")
        idxs = prtecan.lookup_listoflines(csvl)
        lb0 = Labelblock(csvl[idxs[0] : idxs[1]])
        lb1 = Labelblock(csvl[idxs[1] :])
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
        """Normalize data using key metadata values."""
        lb0, lb1 = labelblocks
        assert lb0.data_nrm["F06"] == pytest.approx(1051.1290323)
        assert lb1.data_nrm["H12"] == pytest.approx(48.4821429)

    def test_eq(self, labelblocks: tuple[Labelblock, Labelblock]) -> None:
        """A Labelblock is equal to itself and not equal to a different Labelblock."""
        lb0, lb1 = labelblocks
        assert lb0 == lb0  # ruff: ignore[comparison-with-itself]
        assert lb0 is not lb1
        with pytest.raises(TypeError):
            assert lb0 == 1

    def test_invalid_plate_format(self) -> None:
        """It raises ValueError for invalid plate formats."""
        invalid_lines: list[list[str | int | float]] = [["A01", 100], ["B01", 200]]
        with pytest.raises(ValueError, match="plate"):
            Labelblock(invalid_lines)

    def test_almost_eq(self, labelblocks: tuple[Labelblock, Labelblock]) -> None:
        """Test the almost_equal method of the Labelblock class."""
        lb0, _ = labelblocks
        file_path1 = Path(data_tests) / "L1" / "290513_7.2.xls"
        csvl1 = prtecan.read_xls(file_path1)  # Gain=98
        idxs1 = prtecan.lookup_listoflines(csvl1)
        lb11 = Labelblock(csvl1[idxs1[1] :])
        file_path2 = Path(data_tests) / "L1" / "290513_8.8.xls"
        csvl2 = prtecan.read_xls(file_path2)  # Gain=99
        idxs2 = prtecan.lookup_listoflines(csvl2)
        lb12 = Labelblock(csvl2[idxs2[1] :])
        assert lb11 != lb12
        assert lb11.almost_equal(lb12)
        assert not lb11.almost_equal(lb0)

    def test_overvalue(self, caplog: pytest.LogCaptureFixture) -> None:
        """It detects saturated data ("OVER")."""
        csvl = prtecan.read_xls(data_tests / "140220" / "pH6.5_200214.xls")
        idxs = prtecan.lookup_listoflines(csvl)
        with caplog.at_level(logging.WARNING):
            lb = Labelblock(csvl[idxs[0] : idxs[1]])
            # Print out the captured logs for debugging
        for log in caplog.records:
            print(log.message)
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
        assert "Temperature" in tecanfile.labelblocks["1"].metadata

    def test_invalid_file(self) -> None:
        """It handles invalid files."""
        with pytest.raises(FileNotFoundError):
            Tecanfile(Path("nonexistent.xls"))

    def test_path(self, tecanfile: Tecanfile) -> None:
        """It reads the file path."""
        assert tecanfile.path == data_tests / "140220/pH8.3_200214.xls"

    def test_detailed_labelblocks(self, tecanfile: Tecanfile) -> None:
        """It parses Temperature metadata and cell data from labelblocks."""
        assert tecanfile.labelblocks["1"].metadata["Temperature"].value == 25.3
        assert tecanfile.labelblocks["2"].metadata["Temperature"].value == 25.7
        assert tecanfile.labelblocks["1"].data["A01"] == 17260
        assert tecanfile.labelblocks["2"].data["H12"] == 4196

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
        return LabelblocksGroup([tf1.labelblocks["1"], tf2.labelblocks["1"]])

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
            LabelblocksGroup([tf1.labelblocks["1"], tf2.labelblocks["2"]])

    def test_detailed_data(self) -> None:
        """Test detailed data aggregation."""
        tfs = [
            Tecanfile(data_tests / "L1" / "290513_5.5.xls"),
            Tecanfile(data_tests / "L1" / "290513_7.2.xls"),
        ]
        lbg = LabelblocksGroup([tfs[0].labelblocks["1"], tfs[1].labelblocks["1"]])
        assert lbg.data["A01"] == [18713, 17088]
        assert lbg.data["H12"] == [28596, 25771]
        assert_almost_equal(lbg.data_nrm["A01"], [995.372, 908.936], 3)

    def test_metadata_merging_detailed(self) -> None:
        """Test metadata merging with common and uncommon values."""
        tfs = [
            Tecanfile(data_tests / "L1" / "290513_7.2.xls"),
            Tecanfile(data_tests / "L1" / "290513_8.8.xls"),
        ]
        lbg = LabelblocksGroup([tfs[0].labelblocks["2"], tfs[1].labelblocks["2"]])
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

    def test_discard_validation_and_nofit_keys(self) -> None:
        """Test discard setter and nofit_keys property of PlateScheme."""
        ps = prtecan.PlateScheme()
        ps.buffer = ["A01", "B01"]
        ps.discard = ["C01"]
        # nofit_keys is union of buffer and discard
        assert set(ps.nofit_keys) == {"A01", "B01", "C01"}
        with pytest.raises(TypeError):
            #  non-str entries
            ps.discard = [1, 2]  # type: ignore[list-item]


class TestTitrationConfig:
    """Test TitrationConfig class."""

    def test_config_creation(self) -> None:
        """It creates configuration with default values."""
        config = TitrationConfig()
        assert config.bg is True
        assert config.bg_adj is False
        assert config.dil is True
        assert config.nrm is True
        assert config.bg_mth == "mean"
        assert config.fit_method == "huber"
        assert config.outlier is None

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

    def test_callback_trigger(self) -> None:
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


class TestBufferFit:
    """Test BufferFit class."""

    def test_empty_property(self) -> None:
        """It correctly identifies empty fits."""
        empty_fit = BufferFit()
        assert empty_fit.empty is True
        non_empty_fit = BufferFit(m=1.0, q=0.0, m_err=0.1, q_err=0.2)
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
        return TitrationResults(scheme, {"A01"}, {"A01": FitResult()})

    def test_dataframe_property(self, titration_results: TitrationResults) -> None:
        """It creates a DataFrame from results."""
        df = titration_results.dataframe
        assert isinstance(df, pd.DataFrame)
        assert "A01" in df.index

    def test_titration_initvar_snapshots_without_retaining(self) -> None:
        """``titration=`` copies scheme+fit_keys and does not retain the object."""
        scheme = PlateScheme()
        scheme.names = {"sample1": {"A01"}}
        fake_tit = SimpleNamespace(scheme=scheme, fit_keys={"A01", "A02"})
        tr = TitrationResults(
            results={"A01": FitResult()},
            titration=fake_tit,  # type: ignore[arg-type]
        )
        assert tr.scheme is scheme
        assert tr.fit_keys == {"A01", "A02"}
        # InitVar must not be stored on the instance (no raw-data retention).
        assert "titration" not in tr.__dict__


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

    def test_from_listfile_tab_separated_without_x_err(self, tmp_path: Path) -> None:
        """A tab-separated list with no x_err column loads.

        The pH lists in this project are comma-separated with three columns, but
        the chloride ones are tab-separated with two: filename and a placeholder
        concentration, because the real concentrations are derived from the
        addition volumes afterwards. Hardcoding the comma collapsed filename and
        value into a single column, so every chloride list failed to parse.
        """
        src = data_tests / "140220"
        listfile = tmp_path / "list.cl"
        listfile.write_text("NaCl1_200214.xls\t0\nNaCl2_200214.xls\t0\n")

        tit = Titration.fromlistfile(listfile, is_ph=False, base_dir=src)

        assert len(tit.tecanfiles) == 2
        assert list(tit.x) == [0.0, 0.0]
        assert list(tit.x_err) == [0.0, 0.0]  # absent means zero, not NaN

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

    def test_generate_combinations_and_prepare_folder(self, tmp_path: Path) -> None:
        """Test generation of parameter combinations and output folder naming."""
        # use existing list file for minimal Titration
        tit = Titration.fromlistfile(data_tests / "L1" / "list.pH.csv", is_ph=True)
        combos = generate_combinations()
        # 2^4 boolean flags times 3 methods
        assert len(combos) == 16 * 3
        flags, method = combos[0]
        assert isinstance(flags, tuple)
        assert len(flags) == 4
        assert method in {"mean", "meansd", "fit"}
        # test prepare output folder naming
        # set all flags to True and bg_mth to 'fit'
        tit.params.bg = True
        tit.params.bg_adj = True
        tit.params.dil = True
        tit.params.nrm = True
        tit.params.bg_mth = "fit"
        out = prepare_output_folder(tit, tmp_path)
        name = out.name
        assert "_bg" in name
        assert "_adj" in name
        assert "_dil" in name
        assert "_nrm" in name
        assert "_fit" in name


class TestTecanConfig:
    """Test TecanConfig class."""

    def test_config_creation(self, tmp_path: Path) -> None:
        """It creates configuration with all parameters."""
        config = TecanConfig(
            out_fp=tmp_path, comb=True, lim=(0, 10), title="Test", fit=True, png=True
        )
        assert config.out_fp == tmp_path
        assert config.comb is True
        assert config.detect_bad is True  # default

    def test_config_detect_bad_false(self, tmp_path: Path) -> None:
        """detect_bad=False disables bad-well detection."""
        config = TecanConfig(
            out_fp=tmp_path,
            comb=False,
            lim=None,
            title="",
            fit=True,
            png=False,
            detect_bad=False,
        )
        assert config.detect_bad is False


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
    export_data_fit(tit, config)

    # Verify output files were created
    assert (tmp_path / "dat_bg_dil_nrm").exists()
    assert (tmp_path / "dat_bg_dil_nrm/fit").exists()
    # Residual CSVs should be written for each result slot
    fit_dir = tmp_path / "dat_bg_dil_nrm/fit"
    assert any(fit_dir.glob("residuals_*.csv")), "residuals CSV missing"
    assert any(fit_dir.glob("residual_stats_*.csv")), "residual_stats CSV missing"
    # Bad-well CSV written by default
    assert (fit_dir / "bad_wells.csv").exists(), "bad_wells.csv missing"


def test_end_to_end_no_detect_bad(tmp_path: Path) -> None:
    """When detect_bad=False, bad_wells.csv and discarded_wells.txt must not be written."""
    tit = Titration.fromlistfile(data_tests / "140220/list.pH.csv", is_ph=True)
    tit.load_additions(data_tests / "140220/additions.pH")
    tit.load_scheme(data_tests / "140220/scheme.txt")
    tit.params.bg = True
    tit.params.dil = True
    tit.params.nrm = True

    config = TecanConfig(
        out_fp=tmp_path,
        comb=False,
        lim=None,
        title="",
        fit=True,
        png=False,
        detect_bad=False,
    )
    export_data_fit(tit, config)

    fit_dir = tmp_path / "dat_bg_dil_nrm/fit"
    assert not (fit_dir / "bad_wells.csv").exists(), (
        "bad_wells.csv must not be written when detect_bad=False"
    )
    subfolder = tmp_path / "dat_bg_dil_nrm"
    assert not (subfolder / "discarded_wells.txt").exists(), (
        "discarded_wells.txt must not be written when detect_bad=False"
    )


# =============================================================================
# Test Suite for Suggestions
# =============================================================================


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
    empty_results = prtecan.TitrationResults(scheme=scheme, fit_keys=set(), results={})
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
        assert all(all(e for e in row) for row in stripped)


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
            lbg1 = tfg.labelblocksgroups["1"]
            lbg2 = tfg.labelblocksgroups["2"]
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
            lbg1 = tfg_warn[0].labelblocksgroups["1"]
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
            lbg2 = tfg_warn[0].labelblocksgroups["2"]
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
            lbg2 = tfg.labelblocksgroups["2"]
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
        lbg1 = tit_ph.labelblocksgroups["1"]
        lbg2 = tit_ph.labelblocksgroups["2"]
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
        export_data_fit(tit_ph, tecan_config)
        a01 = pd.read_csv(tmp_path / "dat" / "A01.dat")
        h12 = pd.read_csv(tmp_path / "dat" / "H12.dat")
        assert a01["1"].tolist()[0::2] == [14798.0, 20142.0, 22915.0, 22060.0]
        assert a01["2"].tolist()[1::2] == [3761.0, 835.0, 347.0]
        assert h12["1"].tolist()[1::2] == [16345.0, 21719.0, 23532.0]
        assert h12["2"].tolist()[1:] == [4196.0, 2390.0, 1031.0, 543.0, 427.0, 371.0]

    def test_data_buffersubtracted(self, tit: Titration) -> None:
        """Check data after normalization and bg subtraction."""
        tit.buffer.wells = ["C12", "D01", "D12", "E01", "E12", "F01"]
        tit.params.nrm = False
        assert tit.data["1"]
        assert tit.data["2"] == {}
        sliced_values = tit.data["1"]["B07"][-1::-3][:2]
        assert_almost_equal(sliced_values, [7069, 5716.7], 1)
        # normalization
        tit.params.nrm = True
        sliced_values0 = tit.data["1"]["B07"][-1::-3][:2]
        sliced_values1 = tit.data["2"]["B07"][-4::-3]
        assert_almost_equal(sliced_values0, [376.01, 304.08], 2)
        assert_almost_equal(sliced_values1, [355.16, 348.57], 2)

    def test_labelblocksgroups_cl(self, tit_cl: Titration) -> None:
        """It reads labelblocksgroups data for Cl too."""
        lbg1 = tit_cl.labelblocksgroups["1"]
        lbg2 = tit_cl.labelblocksgroups["2"]
        assert lbg1.data is not None
        assert lbg1.data["A01"][1::2] == [16908.0, 14719.0, 14358.0, 14520.0]
        assert lbg2.data["A01"][1::2] == [167.0, 109.0, 87.0, 81.0]
        assert lbg2.data["H12"][1::2] == [223.0, 141.0, 120.0, 100.0]

    def test_raise_listfilenotfound(self) -> None:
        """It raises FileNotFoundError when list.xx file does not exist."""
        with pytest.raises(FileNotFoundError, match="Cannot find: aax"):
            Titration.fromlistfile(Path("aax"), is_ph=True)

    def test_fromlistfile_base_dir(self, tmp_path: Path) -> None:
        """It resolves Tecan files against base_dir when the list file is elsewhere."""
        data_dir = data_tests / "140220"
        listfile = tmp_path / "list.pH.csv"
        listfile.write_text((data_dir / "list.pH.csv").read_text())
        tit = Titration.fromlistfile(listfile, is_ph=True, base_dir=data_dir)
        assert [tf.path.parent for tf in tit.tecanfiles] == [data_dir] * 7

    def test_fromlistfile_base_dir_missing_file(self, tmp_path: Path) -> None:
        """It names the missing Tecan file, not the list file, when base_dir is wrong."""
        listfile = tmp_path / "list.pH.csv"
        listfile.write_text((data_tests / "140220" / "list.pH.csv").read_text())
        with pytest.raises(FileNotFoundError, match=r"pH9\.1_200214\.xls"):
            Titration.fromlistfile(listfile, is_ph=True, base_dir=tmp_path / "nowhere")

    def test_bad_listfile(self) -> None:
        """It raises Exception when list.xx file is ill-shaped."""
        with pytest.raises(ValueError, match=r"Check format .* for listfile: .*"):
            Titration.fromlistfile(data_tests / "140220" / "list.pH2.csv", is_ph=True)

    def test_data_bg_and_nrm(self, tit1: Titration) -> None:
        """Calculate buffer value from average of buffer wells and subtract."""
        tit1.buffer.wells = ["D01", "D12", "E01", "E12"]
        tit1.params.nrm = False
        tit1.params.dil = False
        tit1.params.bg = False
        assert tit1.buffer.dataframes["1"]["sem"][0] == pytest.approx(259.9514)
        assert tit1.buffer.dataframes["2"]["sem"][0] == pytest.approx(2.561738)
        tit1.params.bg = True
        assert tit1.data["1"]["F06"][0] == pytest.approx(7661.75)
        assert tit1.data["2"]["H12"][0] == pytest.approx(486.25)
        # Can also assign a buffer value.
        tit1.bg = {"1": np.array([1.0]), "2": np.array([2.9])}
        assert tit1.data["1"]["F06"][0] == 19550
        assert tit1.data["2"]["H12"][0] == 540.1
        # nrm
        assert tit1.buffer.dataframes_nrm["1"]["fit"][0] == pytest.approx(639.20699)
        assert tit1.buffer.dataframes_nrm["1"]["mean"][0] == pytest.approx(639.20699)
        assert tit1.buffer.dataframes_nrm["2"]["fit"][0] == pytest.approx(5.06696)
        assert tit1.buffer.dataframes_nrm["1"]["sem"][0] == pytest.approx(13.97588)
        assert tit1.buffer.dataframes_nrm["2"]["sem"][0] == pytest.approx(0.2287266)
        # also bg duplicates data in buffers_nrm
        tit1.params.nrm = True
        tit1.buffer.wells = ["D01", "D12", "E01", "E12"]
        assert tit1.bg["1"][0] == pytest.approx(639.20699)
        assert tit1.bg["2"] == pytest.approx(5.06696)
        # nrm data
        assert tit1.data["1"]["F06"] == pytest.approx(411.922)
        assert tit1.data["2"]["H12"] == pytest.approx(43.4152)
        # Can also assign a buffer_norm value.
        tit1.bg = {"1": np.array([1.0]), "2": np.array([0.4821])}
        assert tit1.data["1"]["F06"] == pytest.approx(1050.13)
        assert tit1.data["2"]["H12"] == pytest.approx(48.0)

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
    def tit(self) -> Titration:
        """Set up TitrationAnalysis."""
        tit = Titration.fromlistfile(data_tests / "140220/list.pH.csv", is_ph=True)
        tit.load_additions(data_tests / "140220/additions.pH")
        tit.load_scheme(data_tests / "140220/scheme.txt")
        return tit

    def test_fit_pipeline_order_default(self, tit: Titration) -> None:
        """It finds well position for buffer samples."""
        assert tit.scheme.buffer == ["D01", "E01", "D12", "E12"]
        assert tit.buffer.wells == ["D01", "E01", "D12", "E12"]

    def test_raise_listfilenotfound(self, tit: Titration) -> None:
        """It raises OSError when scheme file does not exist."""
        with pytest.raises(
            FileNotFoundError, match=r"No such file or directory: 'aax'"
        ):
            tit.load_scheme(Path("aax"))

    def test_raise_listfile_exception(self, tit: Titration) -> None:
        """It raises AssertionError when scheme.txt file is ill-shaped."""
        bad_schemefile = data_tests / "140220/scheme0.txt"
        msg = f"Check format [well sample] for schemefile: {bad_schemefile}"
        with pytest.raises(ValueError, match=re.escape(msg)):
            tit.load_scheme(bad_schemefile)

    def test_subtract_bg(self, tit: Titration) -> None:
        """It subtracts buffer average values."""
        lbg0 = tit.labelblocksgroups["1"]
        lbg1 = tit.labelblocksgroups["2"]
        assert_almost_equal(
            lbg0.data_nrm["E01"][::2], [601.72, 641.505, 674.355, 706.774], 3
        )
        assert lbg0.data is not None
        assert lbg0.data["E01"][::2] == [11192.0, 11932.0, 12543.0, 13146.0]
        assert type(lbg1) is LabelblocksGroup
        tit.params.bg = True
        tit.params.nrm = False
        tit.params.dil = False
        assert_array_equal(tit.data["1"]["A12"][::3], [8084.5, 16621.75, 13775.0])
        assert lbg1.data is not None
        assert_array_equal(tit.data["2"]["A12"][::3], [9758.25, 1334.0, 283.5])

    def test_dilution_correction(self, tit: Titration) -> None:
        """It applies dilution correction read from file listing additions."""
        assert tit.additions is not None
        assert_array_equal(tit.additions, [100, 2, 2, 2, 2, 2, 2])
        tit.params.nrm = False
        tit.params.dil = True
        assert tit.data is not None
        assert tit.data["1"] is not None
        assert_almost_equal(
            tit.data["2"]["A12"],
            [9758.25, 7524.795, 3079.18, 1414.04, 641.79, 402.325, 317.52],
        )

    def test_data_nrm(self, tit: Titration) -> None:
        """It normalizes data."""
        tit.params.nrm = True
        tit.params.bg = True
        tit.params.dil = True

        assert_almost_equal(
            tit.data["1"]["A12"][::2],
            [434.65, 878.73, 975.58, 829.46],
            2,
        )
        assert_almost_equal(
            tit.data["2"]["A12"][::2],
            [871.272, 274.927, 57.303, 28.35],
            3,
        )

    def test_keys(self, tit: Titration) -> None:
        """It gets well positions for ctrl and unknown samples."""
        assert set(tit.scheme.names) == {"NTT", "G03", "V224Q", "S202N"}
        x = {"B12", "H12", "F01", "C12", "F12", "C01", "H01", "G12", "B01", "G01"}
        assert set(tit.scheme.ctrl) - {"A01", "A12"} == x

    def test_fit(self, tit: Titration) -> None:
        """It fits each label separately."""
        # Test Label 1 for H02 (should be skipped because of OVER value or insufficient points)
        ds1 = {k: tit.create_ds(k, label="1") for k in tit.fit_keys}
        res1 = tit.fit_plate(ds1, method="huber")
        assert not res1["H02"].is_valid()

        # Test Label 2 for H02
        ds2 = {k: tit.create_ds(k, label="2") for k in tit.fit_keys}
        res2 = tit.fit_plate(ds2, method="huber")
        assert res2["H02"].is_valid()

        # Check 'K' and std error for 'H02' in the second fit result
        assert res2["H02"].result is not None
        k_h02 = res2["H02"].result.params["K"]
        assert k_h02.value == pytest.approx(7.899, abs=1e-3)
        assert k_h02.stderr == pytest.approx(0.026, abs=1e-3)

        # Check 'K' and std error for 'H02' in global fit
        ds_global = {k: tit.create_global_ds(k) for k in tit.fit_keys}
        res_global = tit.fit_plate(ds_global, method="huber")
        assert res_global["H02"].result is not None
        k_h02_glob = res_global["H02"].result.params["K"]
        assert k_h02_glob.value == pytest.approx(7.899, abs=1e-3)
        assert k_h02_glob.stderr == pytest.approx(0.026, abs=1e-3)

        # Check 'K' and std error for 'E02' in the second fit result.
        # 8.000 -> 8.002 when compute_noise_variance stopped clipping the
        # variance at 1.0: this plate's second-label floor is 0.424, so that
        # label's sigma had been inflated 2.4x and its points under-weighted.
        assert res2["E02"].result is not None
        k_e02 = res2["E02"].result.params["K"]
        assert k_e02.value == pytest.approx(8.002, abs=1e-3)
        assert k_e02.stderr == pytest.approx(0.041, abs=1e-3)

        # Check 'K' and std error for 'E02' in the third fit result.
        # Moved with the second fit, and for the same reason.
        assert res_global["E02"].result is not None
        k_e02_glob = res_global["E02"].result.params["K"]
        assert k_e02_glob.value == pytest.approx(8.002, abs=1e-3)
        assert k_e02_glob.stderr == pytest.approx(0.028, abs=1e-3)

    def test_titration_results_residuals(self, tit: Titration) -> None:
        """Plate results expose the canonical residual table."""
        ds = {k: tit.create_ds(k, label="2") for k in tit.fit_keys}
        res = tit.fit_plate(ds, method="huber")

        table = res.residuals

        assert list(table.columns) == RESIDUAL_TABLE_COLUMNS
        assert not table.empty
        # H02 fits on label 2; wells whose fit failed are skipped, not raised on.
        assert "H02" in set(table["well"])
        assert res.residuals is table  # cached

    def test_titration_results_residuals_classical_plate(self, tit: Titration) -> None:
        """A classical (non-MCMC) plate fit standardizes every well as Normal."""
        res = tit.fit_plate(method="huber")

        table = res.residuals

        assert list(table.columns) == RESIDUAL_TABLE_COLUMNS
        assert set(table["well"]) == tit.fit_keys
        assert (table["residual_likelihood"] == "normal").all()

    def test_fgls_fit_plate_returns_titration_results_with_noise_model(
        self, tit: Titration
    ) -> None:
        """FGLS returns a TitrationResults carrying its calibrated noise model."""
        res = tit.fgls_fit_plate(max_iter=1)

        assert isinstance(res, TitrationResults)
        assert isinstance(res.noise_model, PlateNoiseModel)
        # Plate metadata is populated from the titration, like fit_plate.
        assert res.scheme is tit.scheme
        assert res.fit_keys == tit.fit_keys
        # The unified residual accessor works, which the tuple return could not offer.
        assert not res.residuals.empty

    def test_plot_buffer_with_title(self, tit: Titration) -> None:
        """It plots buffers for 2 lbg with title."""
        g = tit.buffer.plot(title="Test Title")
        assert isinstance(g, sns.FacetGrid)
        assert len(g.axes_dict) == 2
        assert g.fig._suptitle.get_text() == "Test Title"  # ruff: ignore[private-member-access]

    def test_plot_buffer_normalized(self, tit: Titration) -> None:
        """It plots buffers_norm for 2 lbg."""
        g = tit.buffer.plot(nrm=True)
        assert isinstance(g, sns.FacetGrid)
        assert len(g.axes_dict) == 2

    @pytest.mark.parametrize(
        ("folder", "expected"),
        [
            ("140220", []),
            # L2: B03 fails label 1 only, so it is kept single-label.
            ("L2", []),
            # L4: G12 fails every label, so there is nothing left to fit.
            ("L4", ["G12"]),
        ],
    )
    def test_detect_and_discard_bad_wells(
        self, folder: str, expected: list[str]
    ) -> None:
        """Low-signal wells are discarded, or lose only the label that failed.

        Signal quality belongs to a label, not a well: the 400 nm channel is dim
        by construction, so a well failing on label 1 alone keeps label 2 and is
        fitted single-label rather than thrown away. Only a well that fails
        every label is discarded outright, so ``expected`` now names the wells
        with no usable label at all, and the rest appear in ``excluded_labels``.
        """
        titan = Titration.fromlistfile(data_tests / f"{folder}/list.pH.csv", is_ph=True)
        titan.load_additions(data_tests / f"{folder}/additions.pH")
        titan.load_scheme(data_tests / f"{folder}/scheme.txt")
        titan.scheme.discard = []

        discards = titan.detect_and_discard_bad_wells(
            outlier_threshold=0.2,
            bg_multiplier=3.0,
            max_k_stderr=math.inf,
        )

        assert discards == expected
        assert set(expected).issubset(set(titan.scheme.discard))
        # Nothing is silently half-kept: a well is either discarded, fully
        # fitted, or recorded as having lost specific labels.
        for well, labels in titan.excluded_labels.items():
            assert well not in discards
            assert labels

    def test_single_refit_two_pass_contract(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tit: Titration,
        tmp_path: Path,
    ) -> None:
        """--mcmc single-refit runs a robust pass then an unrobust refit."""
        # Capture the real signature before monkeypatching replaces
        # bayes.fit_binding_pymc below.
        real_sig = inspect.signature(bayes.fit_binding_pymc)
        calls: list[dict[str, Any]] = []

        def fake_fit_binding_pymc(
            ds_or_fr: Dataset | FitResult, **kwargs: object
        ) -> FitResult:
            calls.append({"_input": ds_or_fr, **kwargs})
            ds_for_fit = (
                ds_or_fr.dataset if isinstance(ds_or_fr, FitResult) else ds_or_fr
            )
            assert ds_for_fit is not None
            return fit_binding_glob(ds_for_fit)

        def fake_residuals_from_fit_results(
            *_args: object, **_kwargs: object
        ) -> pd.DataFrame:
            # A single clear outlier at label "1"'s last titration point, so
            # mask propagation into the refit is observable below.
            return pd.DataFrame({
                "trace_id": ["pymc_robust_unweighted"] * 14,
                "well": ["single"] * 14,
                "label": ["1"] * 7 + ["2"] * 7,
                "step": [*range(7), *range(7)],
                "x": [*tit.x, *tit.x],
                "std_res": [0.0] * 6 + [50.0] + [0.0] * 7,
            })

        # Patch both lookup sites: the call happens inside export post-inlining,
        # so patching export is what matters; the bayes patch is kept for
        # provenance (it was needed pre-refactor) and is a harmless no-op now.
        # This is a post-refactor regression test, not a before/after-inlining
        # characterization test.
        monkeypatch.setattr(bayes, "fit_binding_pymc", fake_fit_binding_pymc)
        monkeypatch.setattr(export, "fit_binding_pymc", fake_fit_binding_pymc)
        # residuals_from_fit_results is imported by name into export.py (not
        # accessed as `model_validation.<attr>`), so it must be patched there.
        monkeypatch.setattr(
            export, "residuals_from_fit_results", fake_residuals_from_fit_results
        )

        spec = prtecan.McmcSpec(
            model="single-refit", sampler=SamplerConfig(n_samples=7)
        )
        res = export.fit_single_mcmc(
            tit,
            {"A01": tit.create_global_ds("A01")},
            tmp_path,
            spec,
        )

        assert res is not None
        assert len(calls) == 2
        first, second = calls
        # Pass 1 is robust; pass 2 is not.
        assert first["robust"].enabled is True
        assert second["robust"].enabled is False
        # Both use the ye_mag noise strategy, unshared, lognormal.
        assert first["noise"].kind == "ye_mag"
        assert second["noise"].kind == "ye_mag"
        assert first["noise"].shared_ye_mags is False
        assert first["noise"].ye_mag_prior == "lognormal"
        # The screening prior is centred on log(bg_noise * 3.6) per label.
        expected_mu = {
            label: pytest.approx(np.log(bg * 3.6)) for label, bg in tit.bg_noise.items()
        }
        assert first["noise"].ye_mag_mu == expected_mu
        assert first["noise"].ye_mag_sigma == 0.5
        # The refit's ye_mag prior is recentred on 0 with a tighter sigma.
        assert second["noise"].ye_mag_mu == 0.0
        assert second["noise"].ye_mag_sigma == 0.25
        # Sampler settings come from spec.sampler in both passes.
        assert first["sampler"].n_samples == 7
        assert second["sampler"].n_samples == 7

        # The deleted fit_binding_pymc_residual_refit() passed n_sd=10.0,
        # n_xerr=1.0, min_x_step=0.2 explicitly; the inlined helper relies on
        # these being fit_binding_pymc's defaults instead, so neither call
        # captures them as explicit kwargs. Read the effective defaults from
        # the real (pre-monkeypatch) signature so a future default change in
        # bayes.py is what makes this assertion fail.
        default_n_sd = real_sig.parameters["n_sd"].default
        default_n_xerr = real_sig.parameters["n_xerr"].default
        default_min_x_step = real_sig.parameters["min_x_step"].default
        for call in (first, second):
            assert call.get("n_sd", default_n_sd) == default_n_sd == 10.0
            assert call.get("n_xerr", default_n_xerr) == default_n_xerr == 1.0
            assert (
                call.get("min_x_step", default_min_x_step) == default_min_x_step == 0.2
            )

        # The screening pass receives a plain dataset whose y_err has been
        # reset to one (required for the ye_mag multiplier to learn scale).
        first_input = first["_input"]
        assert isinstance(first_input, Dataset)
        for da in first_input.values():
            np.testing.assert_allclose(da.y_errc, np.ones_like(da.yc))

        # The refit pass receives a FitResult (not a raw Dataset) whose
        # dataset carries the outlier mask computed from the screening pass's
        # residuals: label "1"'s last point was flagged and must be masked
        # out before the refit; label "2" is untouched.
        second_input = second["_input"]
        assert isinstance(second_input, FitResult)
        assert second_input.dataset is not None
        assert second_input.dataset["1"].mask.tolist() == [
            True,
            True,
            True,
            True,
            True,
            True,
            False,
        ]
        assert second_input.dataset["2"].mask.tolist() == [True] * 7


def test_single_refit_two_pass_masks_clear_outlier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The screening pass's flagged outlier is masked out before the refit.

    This restores the coverage retired with
    ``fit_binding_pymc_residual_refit``'s deleted tests: it drives
    ``_single_refit_two_pass``'s ``min_keep=3``/``allowed_tail_fraction=0.0``
    literals through an actual masking decision, rather than merely checking
    that the composition runs.
    """
    x = np.array([6.0, 7.0, 8.0, 9.0])
    y = np.array([1.90909091, 1.5, 1.09090909, 1.00990099])
    ds = Dataset({"default": DataArray(x, y)}, is_ph=True)

    calls: list[object] = []

    def fake_fit_binding_pymc(
        ds_or_fr: Dataset | FitResult, **_kwargs: object
    ) -> FitResult:
        calls.append(ds_or_fr)
        ds_for_fit = ds_or_fr.dataset if isinstance(ds_or_fr, FitResult) else ds_or_fr
        assert ds_for_fit is not None
        return fit_binding_glob(ds_for_fit)

    def fake_residuals_from_fit_results(
        *_args: object, **_kwargs: object
    ) -> pd.DataFrame:
        # Four points, one clear outlier (step 3): with min_keep=3, exactly
        # one point may be dropped from this label (4 kept -> 3 kept).
        return pd.DataFrame({
            "trace_id": ["pymc_robust_unweighted"] * 4,
            "well": ["single"] * 4,
            "label": ["default"] * 4,
            "step": [0, 1, 2, 3],
            "x": x,
            "std_res": [0.0, 0.0, 0.0, 50.0],
        })

    monkeypatch.setattr(export, "fit_binding_pymc", fake_fit_binding_pymc)
    monkeypatch.setattr(
        export, "residuals_from_fit_results", fake_residuals_from_fit_results
    )

    sampler = SamplerConfig(n_samples=5, nuts_sampler="pymc")
    final, _residuals = export._single_refit_two_pass(  # ruff: ignore[private-member-access]
        ds,
        screening_noise=export._ye_mag_screening_noise(0.1),  # ruff: ignore[private-member-access]
        refit_noise=NoiseConfig.ye_mag(
            shared=False, prior="lognormal", mu=0.0, sigma=0.25
        ),
        sampler=sampler,
    )

    assert final is not None
    assert len(calls) == 2
    seeded = calls[1]
    assert isinstance(seeded, FitResult)
    assert seeded.dataset is not None
    assert seeded.dataset["default"].mask.tolist() == [True, True, True, False]


def test_single_refit_two_pass_screens_with_the_given_criterion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A supplied criterion decides the exclusions, not the default tail rule.

    The residual table is built so the default ``ResidualTail`` on ``std_res``
    flags nothing, while ``RobustZMad`` on ``raw_res`` flags step 3. Only a
    criterion that actually reaches the screening step can change the mask.
    """
    x = np.array([6.0, 7.0, 8.0, 9.0])
    y = np.array([1.90909091, 1.5, 1.09090909, 1.00990099])
    ds = Dataset({"default": DataArray(x, y)}, is_ph=True)

    calls: list[object] = []

    def fake_fit_binding_pymc(
        ds_or_fr: Dataset | FitResult, **_kwargs: object
    ) -> FitResult:
        calls.append(ds_or_fr)
        ds_for_fit = ds_or_fr.dataset if isinstance(ds_or_fr, FitResult) else ds_or_fr
        assert ds_for_fit is not None
        return fit_binding_glob(ds_for_fit)

    def fake_residuals_from_fit_results(
        *_args: object, **_kwargs: object
    ) -> pd.DataFrame:
        return pd.DataFrame({
            "trace_id": ["pymc_robust_unweighted"] * 4,
            "well": ["single"] * 4,
            "label": ["default"] * 4,
            "step": [0, 1, 2, 3],
            "x": x,
            # Below the default 3.0 cutoff, so ResidualTail flags nothing.
            "std_res": [0.1, -0.2, 0.15, 0.4],
            # Far outside the group's MAD scale, so RobustZMad flags step 3.
            "raw_res": [0.1, -0.2, 0.15, 3.0],
        })

    monkeypatch.setattr(export, "fit_binding_pymc", fake_fit_binding_pymc)
    monkeypatch.setattr(
        export, "residuals_from_fit_results", fake_residuals_from_fit_results
    )

    final, _residuals = export._single_refit_two_pass(  # ruff: ignore[private-member-access]
        ds,
        screening_noise=export._ye_mag_screening_noise(0.1),  # ruff: ignore[private-member-access]
        refit_noise=NoiseConfig.ye_mag(
            shared=False, prior="lognormal", mu=0.0, sigma=0.25
        ),
        sampler=SamplerConfig(n_samples=5, nuts_sampler="pymc"),
        criterion=RobustZMad(threshold=3.5),
    )

    assert final is not None
    seeded = calls[1]
    assert isinstance(seeded, FitResult)
    assert seeded.dataset is not None
    assert seeded.dataset["default"].mask.tolist() == [True, True, True, False]


def test_titration_results_noise_model_defaults_to_none() -> None:
    """The noise_model field is optional and absent for a plain plate fit."""
    assert TitrationResults().noise_model is None


def test_titration_results_noise_model_is_last_positional() -> None:
    """Appending noise_model must not shift any existing positional argument."""
    scheme = PlateScheme()
    fit_keys = {"A01"}
    results: dict[str, FitResult] = {}
    tr = TitrationResults(scheme, fit_keys, results)
    # The three historical positional args still bind to their own fields.
    assert tr.scheme is scheme
    assert tr.fit_keys == fit_keys
    assert tr.results is results
    assert tr.noise_model is None
    # And it is settable by keyword.
    nm = PlateNoiseModel()
    assert TitrationResults(scheme, fit_keys, results, noise_model=nm).noise_model is nm


class TestBufferReadNoise:
    """``bg_read_noise``: buffer scatter with fixed well offsets taken out.

    ``bg_noise`` pools two things a titration fit treats very differently. A
    buffer well sitting consistently high is a positional offset, and in a
    per-well fit it is absorbed by that well's ``S0``/``S1`` plateaus, so it
    contributes nothing to point-to-point residual scatter. Only what is left
    once step and well effects are removed is measurement noise, and only that
    belongs in a noise-model floor.
    """

    @staticmethod
    def _titration() -> Titration:
        titan = Titration.fromlistfile(data_tests / "140220/list.pH.csv", is_ph=True)
        titan.load_scheme(data_tests / "140220/scheme.txt")
        return titan

    def test_pure_well_offsets_are_not_read_noise(self) -> None:
        """Wells that differ only by a constant offset carry no read noise.

        This is the whole point of the statistic: three buffer wells reading a
        flat background at 100, 110 and 120 have a large pooled SD and zero
        measurement scatter.
        """
        values = np.array([[100.0, 110.0, 120.0]] * 5)
        rms = titration_module._interaction_rms(values)  # ruff: ignore[private-member-access]
        assert rms == pytest.approx(0.0)

    def test_pure_step_drift_is_not_read_noise(self) -> None:
        """A background that drifts identically in every well is not noise."""
        drift = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        values = np.tile(drift[:, None], (1, 4))
        rms = titration_module._interaction_rms(values)  # ruff: ignore[private-member-access]
        assert rms == pytest.approx(0.0)

    def test_recovers_injected_scatter(self) -> None:
        """With offsets and drift on top, the injected sigma is what comes back."""
        rng = np.random.default_rng(20260908)
        n_steps, n_wells, sigma = 40, 8, 3.0
        offsets = np.array([0.0, 5.0, -5.0, 12.0, -8.0, 3.0, -2.0, 9.0])
        drift = np.linspace(200.0, 260.0, n_steps)
        values = (
            drift[:, None]
            + offsets[None, :]
            + rng.normal(0.0, sigma, size=(n_steps, n_wells))
        )
        rms = titration_module._interaction_rms(values)  # ruff: ignore[private-member-access]
        assert rms == pytest.approx(sigma, rel=0.1)

    def test_bg_read_noise_matches_bg_noise_labels(self) -> None:
        """The new statistic is reported for exactly the labels bg_noise is."""
        titan = self._titration()
        assert set(titan.bg_read_noise) == set(titan.bg_noise)

    def test_bg_read_noise_is_below_bg_noise_on_real_buffers(self) -> None:
        """Real buffer wells carry positional offsets, so removing them lowers it.

        Guards the direction of the correction: a ``bg_read_noise`` that came
        back equal to ``bg_noise`` would mean the well effect was never removed.
        """
        titan = self._titration()
        for label, read in titan.bg_read_noise.items():
            assert 0.0 < read < titan.bg_noise[label]


class TestSigmaFloorOverride:
    """``--noise-floor``: supply the read-noise floor instead of measuring it.

    ``bg_noise`` is what the buffer wells happen to scatter by on one plate.
    Across eleven plates the label-1 floor tracks the reader Gain at r = 0.906,
    with a slope of a decade per 38 gain units against the 34.1 predicted by the
    amplification law -- so one floor quoted at a reference Gain describes every
    plate, and is a better estimate for any single plate than its own four
    buffer wells.
    """

    @staticmethod
    def _titration() -> Titration:
        titan = Titration.fromlistfile(data_tests / "140220/list.pH.csv", is_ph=True)
        titan.load_scheme(data_tests / "140220/scheme.txt")
        return titan

    def test_unset_floor_falls_back_to_measured_read_noise(self) -> None:
        """With no override the floor is the measured read noise, not the pooled SD.

        ``bg_noise`` pools the buffer wells' fixed positional offsets in with
        their scatter, and those offsets are absorbed by each well's own
        plateaus, so they are not floor. Against the pooled calibration the
        read-noise figure is the right order -- 0.81x for label 1, 2.28x for
        label 2 -- where ``bg_noise`` is 4.5x too large for label 1.
        """
        titan = self._titration()
        assert titan.sigma_floor == pytest.approx(titan.bg_read_noise)
        assert titan.sigma_floor != pytest.approx(titan.bg_noise)

    def test_supplied_floor_is_used_verbatim_without_a_reference_gain(self) -> None:
        """A floor with no reference Gain is an absolute value, not a hint."""
        titan = self._titration()
        labels = sorted(titan.data)
        titan.params.noise_floor = (3.59, 0.42)
        assert titan.sigma_floor == pytest.approx({labels[0]: 3.59, labels[1]: 0.42})

    def test_reference_gain_scales_the_floor_by_the_amplification_law(self) -> None:
        """A floor quoted at a reference Gain is scaled to the plate's own Gain.

        Read noise is amplified with the signal, so a floor measured at one PMT
        setting has to be moved to another before it means anything.
        """
        titan = self._titration()
        labels = sorted(titan.data)
        gain = float(titan.labelblocksgroups[labels[0]].metadata["Gain"].value)
        decade = titration_module._FLOOR_GAIN_DECADE  # ruff: ignore[private-member-access]
        titan.params.noise_floor = (3.59, 0.42)
        titan.params.noise_floor_ref_gain = (gain - decade, 0.0)
        # One decade of gain units above the reference is exactly 10x.
        assert titan.sigma_floor[labels[0]] == pytest.approx(35.9)

    def test_zero_reference_gain_disables_scaling_for_that_label(self) -> None:
        """Label 2's floor showed no Gain dependence, so it must be able to opt out."""
        titan = self._titration()
        labels = sorted(titan.data)
        titan.params.noise_floor = (3.59, 0.42)
        titan.params.noise_floor_ref_gain = (0.0, 0.0)
        assert titan.sigma_floor == pytest.approx({labels[0]: 3.59, labels[1]: 0.42})

    def test_an_explicit_override_reaches_y_err(self) -> None:
        """An explicit floor governs the classical weighting, not only the sampler."""
        titan = self._titration()
        labels = sorted(titan.data)
        well = min(titan.data[labels[0]])
        titan.params.noise_floor = (500.0, 500.0)
        y_err = np.asarray(titan.create_ds(well, labels[0])[labels[0]].y_err)
        # A floor far above every other term dominates the error model.
        assert float(np.min(y_err)) >= 500.0

    def test_y_err_defaults_to_bg_noise_not_the_read_noise(self) -> None:
        """Unset, y_err keeps the pooled figure the classical path was tuned on.

        With no gain or alpha this y_err is homoscedastic, and the best single
        sigma for that is the typical noise over the signal range, not the
        floor at zero signal: ``bg_noise`` sits 4x below it on label 1 where
        ``bg_read_noise`` sits 14x below. Least squares does not care, since a
        uniform scale cancels, but huber's transition point is absolute -- and
        defaulting this to the read noise left 29 of 731 well-fits with an
        unconstrained K that had been fine.
        """
        titan = self._titration()
        labels = sorted(titan.data)
        well = min(titan.data[labels[0]])
        y_err = np.asarray(titan.create_ds(well, labels[0])[labels[0]].y_err)
        assert float(np.median(y_err)) == pytest.approx(
            titan.bg_noise[labels[0]], rel=1e-6
        )
        assert titan.bg_noise[labels[0]] > titan.bg_read_noise[labels[0]]

    def test_the_override_reaches_the_structured_mcmc_noise(self) -> None:
        """And the sampler's floor hint, which is the other consumer."""
        titan = self._titration()
        labels = sorted(titan.data)
        titan.params.noise_floor = (3.59, 0.42)
        noise = export._structured_noise(  # ruff: ignore[private-member-access]
            titan, noise_mode="fixed"
        )
        assert noise.floor == pytest.approx({labels[0]: 3.59, labels[1]: 0.42})


class TestStructuredMcmcNoise:
    """CLI-selectable structured noise for ``--mcmc single-refit``."""

    @staticmethod
    def _titration() -> Titration:
        titan = Titration.fromlistfile(data_tests / "140220/list.pH.csv", is_ph=True)
        titan.load_scheme(data_tests / "140220/scheme.txt")
        return titan

    def test_no_hints_leaves_gain_and_alpha_free(self) -> None:
        """With no --noise-gain/--noise-alpha, both terms are learned.

        There is no hint to centre on or pin, so "free" is the only meaningful
        mode; floors still come from the measured background noise.
        """
        titan = self._titration()
        noise = export._structured_noise(  # ruff: ignore[private-member-access]
            titan, noise_mode="centered"
        )
        assert noise.kind == "structured"
        assert noise.gain_mode == "free"
        assert noise.alpha_mode == "free"
        # The floor hint is always measured, so it takes the configured mode
        # even when gain and alpha have nothing to centre on. Left "free" it
        # drifts well above bg_noise and flattens the structured model.
        assert noise.floor_mode == "centered"
        assert isinstance(noise.floor, dict)
        assert set(noise.floor) == set(titan.data)

    def test_supplied_hints_take_the_configured_mode(self) -> None:
        """A supplied value becomes a per-label hint under ``noise_mode``."""
        titan = self._titration()
        labels = sorted(titan.data)
        titan.params.noise_gain = (4.93, 1.34)
        titan.params.noise_alpha = (0.106, 0.0)
        noise = export._structured_noise(  # ruff: ignore[private-member-access]
            titan, noise_mode="centered"
        )
        assert noise.gain_mode == "centered"
        assert noise.alpha_mode == "centered"
        assert noise.gain == {labels[0]: 4.93, labels[1]: 1.34}
        # Label 2's exact 0.0 is the NNLS-boundary case: it is kept as a hint,
        # not dropped, so the prior can span the plate's alpha scale.
        assert noise.alpha == {labels[0]: 0.106, labels[1]: 0.0}

    def test_fixed_mode_is_honoured(self) -> None:
        """``noise_mode="fixed"`` pins supplied hints instead of centring them."""
        titan = self._titration()
        titan.params.noise_alpha = (0.05, 0.02)
        noise = export._structured_noise(  # ruff: ignore[private-member-access]
            titan, noise_mode="fixed"
        )
        assert noise.alpha_mode == "fixed"
        # Gain got no value, so it stays free regardless of noise_mode.
        assert noise.gain_mode == "free"
        # The floor is measured, so "fixed" pins it to bg_noise.
        assert noise.floor_mode == "fixed"

    def test_structured_noise_takes_mode_as_argument(self) -> None:
        """The mode is passed in, not read off titration.params.

        ``gain_mode`` and ``alpha_mode`` are independent ternaries in
        ``_structured_noise``; ``test_fixed_mode_is_honoured`` above only
        supplies ``noise_alpha`` and so only pins ``alpha_mode``, leaving
        ``gain_mode`` at its untouched "free" default. This test supplies
        ``noise_gain`` instead, so a bug that corrupts ``gain_mode`` alone
        (e.g. hardcoding it, or crossing the ``gains``/``alphas`` variables)
        would be caught here.
        """
        titan = self._titration()
        titan.params.noise_gain = (1.0, 1.0)
        noise = export._structured_noise(  # ruff: ignore[private-member-access]
            titan, noise_mode="fixed"
        )
        assert noise.gain_mode == "fixed"


def test_params_change_resets_derived_data() -> None:
    """Setting any surviving params field discards the derived data cache.

    Asserts the effect, not a list of attribute names: a name list is exactly
    what rotted into twelve dead entries after 01735f12.
    """
    tit = prtecan.Titration.fromlistfile(data_tests / "140220/list.pH.csv", is_ph=True)
    tit.load_scheme(data_tests / "140220" / "scheme.txt")
    assert tit.data  # populate the lazily-built cache
    assert tit._data != {}  # ruff: ignore[private-member-access]

    tit.params.nrm = not tit.params.nrm

    # Do not touch tit.data here — reading it would refill the cache.
    assert tit._data == {}  # ruff: ignore[private-member-access]


def test_mcmc_spec_defaults() -> None:
    """McmcSpec carries the sampling decision without touching TitrationConfig."""
    spec = prtecan.McmcSpec(model="single", sampler=SamplerConfig(n_samples=8))
    assert spec.model == "single"
    assert spec.sampler.n_samples == 8
    assert spec.structured_noise is False
    assert spec.noise_mode == "centered"


def test_fit_single_mcmc_returns_none_without_spec(tmp_path: Path) -> None:
    """No spec means no sampling, regardless of what params says."""
    from clophfit.prtecan.export import (  # ruff: ignore[import-outside-top-level]
        fit_single_mcmc,
    )

    tit = prtecan.Titration.fromlistfile(data_tests / "140220/list.pH.csv", is_ph=True)
    assert fit_single_mcmc(tit, {}, tmp_path, None) is None


def test_fit_single_mcmc_multi_fits_wells_jointly(tmp_path: Path) -> None:
    """model="multi" routes to the joint fit and returns per-well results.

    The CLI could previously express only per-well sampling, so a plate whose
    controls share a K had no route through it at all. This pins the branch: a
    multi spec must come back with one result per fitted well, not None and not
    a single-well fit.
    """
    from clophfit.fitting.data_structures import (  # ruff: ignore[import-outside-top-level]
        DataArray,
        Dataset,
    )
    from clophfit.prtecan.export import (  # ruff: ignore[import-outside-top-level]
        fit_single_mcmc,
    )

    tit = prtecan.Titration.fromlistfile(data_tests / "140220/list.pH.csv", is_ph=True)
    x = np.linspace(5.5, 8.5, 7)
    rng = np.random.default_rng(0)
    datasets = {
        w: Dataset(
            {"1": DataArray(x, 100 + 50 * x + rng.normal(0, 1, 7), y_errc=np.ones(7))},
            is_ph=True,
        )
        for w in ("A01", "A02")
    }
    spec = prtecan.McmcSpec(
        model="multi",
        sampler=SamplerConfig(
            nuts_sampler="pymc", n_tune=10, n_samples=10, chains=1, cores=1
        ),
    )
    res = fit_single_mcmc(tit, datasets, tmp_path, spec)

    assert res is not None
    assert set(datasets) <= set(res.results)


@pytest.mark.parametrize("comb", [False, True])
def test_export_data_fit_with_mcmc_spec_samples(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, comb: bool
) -> None:
    """A real McmcSpec threaded into export_data_fit reaches the sampler.

    Pins the seam one hop above ``test_fit_single_mcmc_returns_none_without_spec``:
    that test shows ``mcmc=None`` performs no sampling at ``fit_single_mcmc``;
    this one shows a non-``None`` spec passed to ``export_data_fit`` actually
    drives a call into the sampler, through ``export_fit``. The sampler is
    patched so the assertion is "was it called", not a real MCMC run.

    Parametrised over ``comb`` because ``export_data_fit`` has two call sites
    for ``export_fit`` -- the ``comb=False`` ``else`` branch and the
    ``comb=True`` branch backing ``--all`` -- and only the former was
    previously pinned; a dropped ``spec`` argument on the ``comb=True``
    branch alone was invisible to the suite. ``generate_combinations`` is
    trimmed to a single combination for ``comb=True`` so the (unpatched)
    classical fits it also triggers stay cheap; the seam under test does not
    depend on which combination runs.
    """
    calls: list[Dataset] = []

    def fake_fit_binding_pymc(ds: Dataset, **_kwargs: object) -> FitResult:
        calls.append(ds)
        return fit_binding_glob(ds)

    monkeypatch.setattr(export, "fit_binding_pymc", fake_fit_binding_pymc)
    if comb:
        monkeypatch.setattr(
            export,
            "generate_combinations",
            lambda: [((False, False, False, False), "fit")],
        )

    tit = Titration.fromlistfile(data_tests / "140220/list.pH.csv", is_ph=True)
    tit.load_additions(data_tests / "140220/additions.pH")
    tit.load_scheme(data_tests / "140220/scheme.txt")

    config = TecanConfig(
        out_fp=tmp_path,
        comb=comb,
        lim=None,
        title="Test",
        fit=True,
        png=False,
        detect_bad=False,
    )
    spec = prtecan.McmcSpec(model="single", sampler=SamplerConfig(n_samples=5))

    export_data_fit(tit, config, spec)

    assert calls, "export_data_fit(..., mcmc=spec) never reached the sampler"


def test_titration_config_carries_no_sampler_fields() -> None:
    """Sampling choices belong to the call that samples, not to the plate."""
    names = {f.name for f in dataclasses.fields(prtecan.TitrationConfig)}
    retired = {
        "mcmc",
        "nuts_sampler",
        "n_mcmc_samples",
        "ctr_free_k",
        "mcmc_noise",
        "noise_mode",
    }
    assert names & retired == set()
    assert {"noise_alpha", "noise_gain"} <= names


def test_mcmc_spec_knobs_reach_the_multi_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Every knob McmcSpec declares must arrive at ``fit_binding_pymc_multi``.

    ``fit_single_mcmc`` forwarded only the sampler and the ye_mag settings, so
    the likelihood family and the control-K parameterization were unreachable
    from the CLI: ``--mcmc multi`` always fitted a Normal with pooled control K
    whatever was asked for. That is the same class of failure the module
    docstring above records for ``--mcmc multi`` itself, and it is silent -- the
    run succeeds, having fitted a different model.

    The sampler is patched, so this asserts the call's arguments rather than a
    fit.
    """
    from clophfit.fitting.bayes_config import (  # ruff: ignore[import-outside-top-level]
        RobustConfig,
    )
    from clophfit.prtecan import export  # ruff: ignore[import-outside-top-level]

    seen: dict[str, object] = {}

    def fake_multi(*_args: object, **kwargs: object) -> object:
        seen.update(kwargs)
        return SimpleNamespace(results={})

    monkeypatch.setattr(export, "fit_binding_pymc_multi", fake_multi)
    tit = prtecan.Titration.fromlistfile(data_tests / "140220/list.pH.csv", is_ph=True)
    spec = prtecan.McmcSpec(
        model="multi",
        sampler=SamplerConfig(n_tune=4000),
        robust=RobustConfig(enabled=True, likelihood="student_t", nu=3.0),
        ctr_free_k=True,
    )

    export.fit_single_mcmc(tit, {}, tmp_path, spec)

    assert seen["robust"] == spec.robust, "robust likelihood never reached the model"
    assert seen["ctr_free_k"] is True, "ctr_free_k never reached the model"
    assert seen["sampler"].n_tune == 4000  # type: ignore[attr-defined]


def test_export_plate_fit_writes_k_per_well(tmp_path: Path) -> None:
    """``--plate-fit`` must produce a K per well, with control groups pooled.

    The plate-wide classical fitters were developed outside this package and
    had no route through the CLI at all. This pins the seam: a plate fit writes
    one row per well, marks which wells were pooled, and gives the pooled ones
    the identical K -- pooling that happens inside the least-squares problem,
    not by averaging separate fits afterwards.
    """
    from clophfit.fitting.data_structures import (  # ruff: ignore[import-outside-top-level]
        DataArray,
        Dataset,
    )
    from clophfit.prtecan.export import (  # ruff: ignore[import-outside-top-level]
        export_plate_fit,
    )

    x = np.array([5.0, 6.0, 6.5, 7.0, 7.5, 8.0, 9.0])

    def curve(k: float) -> np.ndarray:
        return 100.0 + 900.0 / (1.0 + 10.0 ** (x - k))

    wells = {"A01": 7.0, "A12": 7.0, "B01": 6.2}
    datasets = {
        w: Dataset({"1": DataArray(x, curve(k), y_errc=np.full(7, 5.0))}, is_ph=True)
        for w, k in wells.items()
    }
    scheme = SimpleNamespace(names={"CTR": ["A01", "A12"]})
    tit = SimpleNamespace(scheme=scheme, x_err=None)

    out = export_plate_fit(tit, datasets, tmp_path, "lm")  # type: ignore[arg-type]  # SimpleNamespace test double

    assert out is not None
    table = pd.read_csv(out).set_index("well")
    assert set(table.index) == set(wells)
    assert table.loc["A01", "K"] == pytest.approx(table.loc["A12", "K"])
    assert bool(table.loc["A01", "k_shared"])
    assert not bool(table.loc["B01", "k_shared"])
    assert table.loc["B01", "K"] == pytest.approx(6.2, abs=0.05)
    assert (tmp_path / "plate_lm_ye_mag.csv").exists()


def test_export_plate_fit_returns_none_without_wells(tmp_path: Path) -> None:
    """No wells means no file, rather than an empty CSV that reads as a result."""
    from clophfit.prtecan.export import (  # ruff: ignore[import-outside-top-level]
        export_plate_fit,
    )

    tit = SimpleNamespace(scheme=SimpleNamespace(names={}), x_err=None)
    assert export_plate_fit(tit, {}, tmp_path, "lm") is None  # type: ignore[arg-type]  # SimpleNamespace test double


class TestBufferEstimators:
    """Buffer location estimators: per-pH ``mean``/``median``, pooled ``*sd``.

    ``meansd`` was intended to pool the buffer into a single value over every
    replicate and every pH point, but only ever pooled the *error*, leaving the
    value identical to ``mean``. Every config selecting it therefore ran the
    same fit as ``mean`` under a different name.
    """

    @staticmethod
    def _tit() -> Titration:
        tit = Titration.fromlistfile(data_tests / "L1" / "list.pH.csv", is_ph=True)
        tit.buffer.wells = ["D01", "D12", "E01", "E12"]
        return tit

    def _bg(self, method: str) -> np.ndarray:
        tit = self._tit()
        tit.params.bg_mth = method
        return np.asarray(tit.bg["1"], dtype=float)

    def test_pooled_estimators_are_flat_across_ph(self) -> None:
        """``meansd`` and ``mediansd`` collapse the buffer to one number."""
        for method in ("meansd", "mediansd"):
            bg = self._bg(method)
            assert len(bg) > 1
            assert np.allclose(bg, bg[0]), f"{method} should be constant across pH"

    def test_per_ph_estimators_track_the_buffer_trend(self) -> None:
        """``mean`` and ``median`` keep one value per pH point.

        The buffer is not flat across a titration - on the real plates it rises
        by several standard errors - so these must not be collapsed.
        """
        for method in ("mean", "median"):
            bg = self._bg(method)
            assert not np.allclose(bg, bg[0]), f"{method} should vary with pH"

    def test_meansd_no_longer_duplicates_mean(self) -> None:
        """The bug: these two were bit-identical, so the knob did nothing."""
        assert not np.allclose(self._bg("mean"), self._bg("meansd"))

    def test_pooled_values_pool_over_reps_and_ph(self) -> None:
        """The pooled value is taken over every replicate at every pH point."""
        tit = self._tit()
        wells = tit.buffer.wells
        # bg reads the normalised buffers whenever nrm is on, which is default.
        obs = tit.buffer.dataframes_nrm["1"][wells].to_numpy(dtype=float)
        assert self._bg("meansd")[0] == pytest.approx(np.nanmean(obs))
        assert self._bg("mediansd")[0] == pytest.approx(np.nanmedian(obs))

    def test_median_resists_one_bad_buffer_well(self) -> None:
        """Why ``median`` earns its place: bad buffer wells are not hypothetical.

        Detection is disabled in most of this project's configs, so an outlying
        buffer well reaches the mean and shifts the background for every well on
        the plate. The median barely moves.
        """
        tit = self._tit()
        wells = tit.buffer.wells
        obs = tit.buffer.dataframes["1"][wells].to_numpy(dtype=float)
        spiked = obs.copy()
        spiked[:, 0] *= 50.0
        mean_shift = abs(np.nanmean(spiked, axis=1) - np.nanmean(obs, axis=1)).max()
        med_shift = abs(np.nanmedian(spiked, axis=1) - np.nanmedian(obs, axis=1)).max()
        assert med_shift < mean_shift / 10.0


def test_output_folder_is_distinct_for_every_bg_method(tmp_path: Path) -> None:
    """Each buffer method needs its own folder, or results overwrite each other.

    The name was built from two special cases (`fit`, `meansd`), so any method
    outside that pair fell through to the bare `dat...` name and would quietly
    land on top of `mean`'s output. Historical names must not move, since
    existing result trees are addressed by them.
    """
    tit = Titration.fromlistfile(data_tests / "L1" / "list.pH.csv", is_ph=True)
    tit.params.bg = True
    tit.params.nrm = True
    tit.params.dil = False
    tit.params.bg_adj = False
    names = {}
    for method in ("mean", "median", "meansd", "mediansd", "fit"):
        tit.params.bg_mth = method
        names[method] = prepare_output_folder(tit, tmp_path).name
    assert len(set(names.values())) == len(names), f"folder collision: {names}"
    # The three that already existed keep the names result trees refer to.
    assert names["mean"] == "dat_bg_nrm"
    assert names["fit"] == "dat_bg_nrm_fit"
    assert names["meansd"] == "dat_bg_nrm_1sd"


class TestKStderrDetection:
    """Pre-fit detection must catch a well that is bright but uninformative."""

    @staticmethod
    def _tit() -> Titration:
        tit = Titration.fromlistfile(data_tests / "L1" / "list.pH.csv", is_ph=True)
        tit.load_additions(data_tests / "L1" / "additions.pH")
        tit.load_scheme(data_tests / "L1" / "scheme.txt")
        tit.scheme.discard = []
        return tit

    def test_raising_bg_multiplier_only_ever_adds_wells(self) -> None:
        """The signal rule is monotone in its threshold, and only about brightness.

        Raising the multiplier raises the bar, so the discarded set can only
        grow. What it cannot do at any setting is notice a well that is bright
        and still uninformative: on L2, C05 has K = 7.138 +- 426 pH and survives
        every multiplier from 2.0 to 4.0, all of which discard the same single
        well. That is what the fit-quality criterion is for.
        """
        sets = {
            m: set(
                self._tit().detect_and_discard_bad_wells(
                    bg_multiplier=m, max_k_stderr=math.inf
                )
            )
            for m in (2.0, 3.0, 4.0)
        }
        assert sets[2.0] <= sets[3.0] <= sets[4.0]

    def test_k_stderr_criterion_discards_an_unconstrained_well(self) -> None:
        """A well whose K error exceeds the pH range measured says nothing about K.

        That is the rule, and it is scale-free: if one cannot locate the midpoint
        inside the window one actually titrated, the well carries no information
        about K no matter how bright it is.
        """
        tit = self._tit()
        span = float(np.nanmax(tit.x) - np.nanmin(tit.x))
        loose = set(tit.detect_and_discard_bad_wells(max_k_stderr=math.inf))
        strict_tit = self._tit()
        strict = set(strict_tit.detect_and_discard_bad_wells(max_k_stderr=span))
        # The criterion can only ever add wells, never rescue one.
        assert loose.issubset(strict)

    def test_k_stderr_flags_exactly_what_the_limit_says(self) -> None:
        """The rule is a property of each well's fit, not of a plate's quality.

        Asserting "few wells are discarded" would only measure how good the
        fixture plate happens to be - the bundled L1 is a poor one, where most
        wells genuinely cannot pin K. So check the rule itself: every well whose
        fitted K standard error exceeds the limit is discarded, and no well
        below it is discarded by this criterion.

        On the real L2 plate the limit behaves as intended - K stderr has median
        0.024 and 98th percentile 0.144, C05 sits at 426, and the span limit of
        3.39 discards exactly that one well of ninety.
        """
        tit = self._tit()
        limit = float(np.nanmax(tit.x) - np.nanmin(tit.x))
        wells = sorted(
            set(tit.labelblocksgroups[next(iter(tit.labelblocksgroups))].data_nrm)
            - tit.scheme.nofit_keys
        )
        expected_unconstrained = {
            w
            for w in wells
            if tit._k_is_unconstrained(w, limit)  # ruff: ignore[private-member-access]
        }
        baseline = set(self._tit().detect_and_discard_bad_wells(max_k_stderr=math.inf))
        discards = set(self._tit().detect_and_discard_bad_wells(max_k_stderr=limit))
        assert expected_unconstrained - baseline <= discards
        assert discards <= baseline | expected_unconstrained


class TestTraceSummaryExport:
    """A sampled model must leave inspectable posterior statistics behind."""

    def test_summary_and_trace_are_written(self, tmp_path: Path) -> None:
        """Running MCMC and keeping nothing from it is not a usable result.

        `--mcmc multi` sampled x_true, the per-well K, and the ye_mag family
        (including the new per-step terms) and then discarded the trace, so a
        user could run the model but not see what it had inferred.
        """
        rng = np.random.default_rng(0)
        x = np.linspace(5.0, 9.0, 7)
        with pm.Model() as model:
            pm.Normal("K", 7.0, 0.1)
            pm.HalfNormal("ye_mag_1", 1.0)
            trace = pm.sample(60, tune=60, chains=2, progressbar=False, random_seed=1)
        assert model is not None
        assert rng is not None
        assert x is not None
        export_trace_summary(trace, tmp_path, "multi")
        summary = tmp_path / "trace_summary_multi.csv"
        assert summary.exists()
        text = summary.read_text()
        assert "K" in text
        assert "ye_mag_1" in text
        # r_hat and ess are the numbers that say whether to trust the rest.
        assert "r_hat" in text
        assert "ess_bulk" in text
        assert (tmp_path / "trace_multi.nc").exists()

    def test_summary_export_never_breaks_a_run(self, tmp_path: Path) -> None:
        """A diagnostic must not be able to destroy the fit it describes.

        The summary is written after sampling has already succeeded, so a
        failure here would throw away a completed run for the sake of a report.
        """
        export_trace_summary(object(), tmp_path, "multi")
        assert not (tmp_path / "trace_summary_multi.csv").exists()


class TestSingleLabelWells:
    """A well may lose one label and keep the other, and must say so."""

    @staticmethod
    def _tit() -> Titration:
        tit = Titration.fromlistfile(data_tests / "L1" / "list.pH.csv", is_ph=True)
        tit.load_additions(data_tests / "L1" / "additions.pH")
        tit.load_scheme(data_tests / "L1" / "scheme.txt")
        tit.scheme.discard = []
        return tit

    def test_one_dim_label_does_not_condemn_the_whole_well(self) -> None:
        """Losing a label is not the same as losing a well.

        The 400 nm channel is dim by construction, so a well can fail the
        background test on label 1 while label 2 is perfectly good. Discarding
        the well throws away a usable titration; dropping the label keeps it.
        """
        tit = self._tit()
        labels = sorted(tit.labelblocksgroups)
        well = min(
            set(tit.labelblocksgroups[labels[0]].data_nrm) - tit.scheme.nofit_keys
        )
        # Make one label of one well look like background, and nothing else.
        tit.exclude_label(well, labels[0])
        assert tit.excluded_labels[well] == {labels[0]}
        ds = tit.create_global_ds(well)
        assert labels[0] not in ds
        assert labels[1] in ds

    def test_a_well_failing_every_label_is_still_discarded(self) -> None:
        """Excluding all labels leaves nothing to fit, so the well must go."""
        tit = self._tit()
        labels = sorted(tit.labelblocksgroups)
        well = min(
            set(tit.labelblocksgroups[labels[0]].data_nrm) - tit.scheme.nofit_keys
        )
        for lbl in labels:
            tit.exclude_label(well, lbl)
        assert well in tit.scheme.discard
        assert well not in tit.fit_keys

    def test_results_record_how_many_labels_each_well_was_fitted_on(self) -> None:
        """The number is derived from the fit itself, so it cannot drift.

        A single-label K rests on 3 parameters over 7 points instead of 6 over
        14, and loses the ratiometric cancellation, so it is systematically less
        certain than its neighbours. A results table that does not say which
        wells those are invites exactly the silent heterogeneity this project
        has been bitten by.
        """
        tit = self._tit()
        labels = sorted(tit.labelblocksgroups)
        well = min(
            set(tit.labelblocksgroups[labels[0]].data_nrm) - tit.scheme.nofit_keys
        )
        tit.exclude_label(well, labels[0])
        datasets = {k: tit.create_global_ds(k) for k in tit.fit_keys}
        res = tit.fit_plate(datasets, method="lm")
        df = res.dataframe
        assert "n_labels" in df.columns
        assert int(df.loc[well, "n_labels"]) == len(labels) - 1
        others = df.drop(index=well)["n_labels"]
        assert (others == len(labels)).all()


def test_ctr_loo_summary_uses_the_column_the_holdout_actually_writes(
    tmp_path: Path,
) -> None:
    """The summary reads ``delta_k_mean``; a rename must fail here, not silently.

    First attempt guessed ``delta`` and the summary simply never appeared - the
    rows were written, no error was raised, and the missing file was the only
    clue. Pin the contract so the next rename breaks a test instead.
    """
    rows = pd.DataFrame([
        {"heldout_well": "A01", "delta_k_mean": 0.10, "p_abs_delta_k_lt_rope": 1.0},
        {"heldout_well": "A02", "delta_k_mean": -0.30, "p_abs_delta_k_lt_rope": 0.0},
    ])
    rows.to_csv(tmp_path / "ctr_loo_lm.csv", index=False)
    df = pd.read_csv(tmp_path / "ctr_loo_lm.csv")
    assert "delta_k_mean" in df.columns
    col = df["delta_k_mean"].astype(float)
    assert float(np.nanmedian(np.abs(col))) == pytest.approx(0.20)
    assert float(np.sqrt(np.nanmean(col**2))) == pytest.approx(0.2236, abs=1e-3)


def test_hdi_columns_are_empty_when_the_fit_has_no_credible_interval() -> None:
    """A least-squares bound must not be exported under an HDI column name.

    ``ffit2.csv`` carried ``Khdi03=3, Khdi97=11`` - the pH box the solver was
    bounded to, in columns a reader takes for a 94% interval. A bound is not an
    interval, and one dressed as the other is worse than an absent value, so
    these are written only when the fit actually produced them.
    """
    pars = Parameters()
    pars.add("K", value=7.0, min=3.0, max=11.0)  # a bounded least-squares fit
    pars["K"].stderr = 0.05
    res = TitrationResults(
        scheme=PlateScheme(),
        fit_keys={"A01"},
        results={"A01": FitResult(result=SimpleNamespace(params=pars), dataset=None)},
    )
    row = res.dataframe.loc["A01"]
    assert row["K"] == pytest.approx(7.0)
    assert row["sK"] == pytest.approx(0.05)
    assert pd.isna(row["Khdi03"])
    assert pd.isna(row["Khdi97"])


def test_plate_fit_results_carry_a_figure_per_well() -> None:
    """The winning fitter must produce the same per-well figures as the others.

    A plate-wide solve returns parameter vectors, not the plotted fit objects
    the export path draws from, so ``--plate-fit`` produced a CSV and nothing to
    look at. Since this is the method the comparison selected, parity matters:
    a well whose K looks wrong has to be inspectable.
    """
    rng = np.random.default_rng(0)
    x = np.linspace(5.0, 9.0, 7)
    datasets = {}
    for i, well in enumerate(("A01", "A02", "A03")):
        arrays = {}
        for lbl, (s0, s1) in (("1", (200.0, 1000.0)), ("2", (1000.0, 200.0))):
            y = binding_1site(x, 6.8 + 0.1 * i, s0, s1, is_ph=True) + rng.normal(
                0.0, 5.0, len(x)
            )
            arrays[lbl] = DataArray(x, y, y_errc=np.ones(len(x)))
        datasets[well] = Dataset(arrays, is_ph=True)
    result = fit_plate_lm(datasets, groups={})
    tit = Titration.fromlistfile(data_tests / "L1" / "list.pH.csv", is_ph=True)
    res = export._plate_fit_results(  # ruff: ignore[private-member-access]
        datasets, result, tit, with_figures=True
    )
    assert set(res.fit_keys) == set(datasets)
    for well in datasets:
        fr = res[well]
        assert fr.figure is not None, f"{well} has no figure to inspect"
        assert fr.result.params["K"].value == pytest.approx(result.k[well], abs=1e-9)


def test_plate_fit_figures_report_k_and_residual_stats() -> None:
    """A fit figure with no numbers on it cannot be judged, only admired.

    The per-well plate-fit images carried the curve and nothing else: no pKa,
    no interval, no indication of whether the residuals were plausibly normal.
    Those are what tell a reader whether to believe the curve, and for an
    edge-of-range well - where the midpoint sits outside the titrated span -
    they are the only warning.
    """
    rng = np.random.default_rng(0)
    x = np.linspace(5.0, 9.0, 7)
    datasets = {}
    for well in ("A01", "A02"):
        arrays = {}
        for lbl, (s0, s1) in (("1", (200.0, 1000.0)), ("2", (1000.0, 200.0))):
            y = binding_1site(x, 6.9, s0, s1, is_ph=True) + rng.normal(0.0, 5.0, len(x))
            arrays[lbl] = DataArray(x, y, y_errc=np.ones(len(x)))
        datasets[well] = Dataset(arrays, is_ph=True)
    result = fit_plate_lm(datasets, groups={})
    tit = Titration.fromlistfile(data_tests / "L1" / "list.pH.csv", is_ph=True)
    res = export._plate_fit_results(  # ruff: ignore[private-member-access]
        datasets, result, tit, with_figures=True
    )
    text = (
        " ".join(t.get_text() for t in res["A01"].figure.axes[0].texts)
        + res["A01"].figure.axes[0].get_title()
    )
    assert "pK" in text, "the fitted pKa is not on the figure"
    assert "±" in text or "+/-" in text, "no interval shown"
    assert "RMS" in text or "z" in text, "no residual statistic shown"


def test_bg_adj_lifts_only_traces_that_go_negative() -> None:
    """The adjustment triggers on a negative minimum, and nothing else.

    The condition read ``y.min() < alpha * 0 * y.max()``, which is ``y.min() <
    0`` with a multiplication by zero sitting in the middle of it - so ``alpha``
    appeared to set the threshold while having no effect on it, and only sized
    the shift. Pin the behaviour so the simplification cannot drift into the
    ``alpha * y.max()`` the expression looks like it wanted.
    """
    tit = Titration.fromlistfile(data_tests / "L1" / "list.pH.csv", is_ph=True)
    tit.load_scheme(data_tests / "L1" / "scheme.txt")
    tit.params.bg = True
    tit.params.nrm = True
    tit.params.bg_adj = True
    label = next(iter(tit.labelblocksgroups))
    adjust = tit._adjust_negative_values  # ruff: ignore[private-member-access]

    def one(trace: list[float]) -> np.ndarray:
        """Run the adjustment over every fit key, and read back the first."""
        keys = sorted(tit.fit_keys)
        data = {label: dict.fromkeys(keys, np.array(trace))}
        data[label] = {k: np.array(trace) for k in keys}
        return np.asarray(adjust(data)[label][keys[0]], dtype=float)

    # Wholly positive, however close to zero relative to its range: untouched.
    assert np.allclose(one([1.0, 20.0, 100.0]), [1.0, 20.0, 100.0])

    # Dipping below zero: lifted so the minimum sits a tenth of the range up.
    out = one([-10.0, 40.0, 90.0])
    assert out.min() == pytest.approx(0.1 * 100.0)
    assert np.allclose(np.diff(out), np.diff([-10.0, 40.0, 90.0]))


def test_structured_noise_reaches_the_multi_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`--mcmc-noise structured` must actually configure the multi-well fit.

    The multi branch called `fit_binding_pymc_multi` without passing `noise` at
    all, so `--mcmc-noise structured`, `--noise-gain` and `--noise-alpha` were
    accepted, echoed in the run's configuration, and silently ignored - a whole
    noise family that could be selected but never took effect.
    """
    captured: dict[str, object] = {}

    def fake_multi(*_args: object, **kwargs: object) -> object:
        captured.update(kwargs)
        raise _StopBayesBuildError

    monkeypatch.setattr(export, "fit_binding_pymc_multi", fake_multi)
    tit = Titration.fromlistfile(data_tests / "L1" / "list.pH.csv", is_ph=True)
    tit.load_scheme(data_tests / "L1" / "scheme.txt")
    spec = McmcSpec(
        model="multi",
        sampler=SamplerConfig(n_samples=10, n_tune=10),
        structured_noise=True,
    )
    datasets = {k: tit.create_global_ds(k) for k in list(tit.fit_keys)[:2]}
    with pytest.raises(_StopBayesBuildError):
        export.fit_single_mcmc(tit, datasets, tmp_path, spec)
    noise = captured.get("noise")
    assert noise is not None, "noise was never passed to the multi model"
    assert getattr(noise, "kind", None) == "structured"


def test_create_global_ds_honours_mask_outliers() -> None:
    """One flag, one meaning, whichever entry point builds the dataset.

    ``create_dataset_dict`` applied ``mask_outliers``; ``create_global_ds`` did
    not. Since ``ppr tecan`` builds its global datasets with the latter, the
    flag never reached the global, ODR, MCMC or plate fits at all - it was
    accepted, echoed back, and silently confined to the per-label path. Two
    builders of the same dataset must not disagree about what the flag means.
    """
    tit = Titration.fromlistfile(data_tests / "L1" / "list.pH.csv", is_ph=True)
    tit.load_scheme(data_tests / "L1" / "scheme.txt")
    tit.params.bg = False
    tit.params.nrm = True
    well = min(tit.fit_keys)

    tit.params.mask_outliers = False
    tit.params.outlier_threshold = 0.2
    unmasked = tit.create_global_ds(well)
    n_off = sum(int(np.asarray(da.mask).sum()) for da in unmasked.values())

    tit.params.mask_outliers = True
    masked = tit.create_global_ds(well)
    n_on = sum(int(np.asarray(da.mask).sum()) for da in masked.values())

    # Same route as create_dataset_dict takes, so the two must agree.
    via_dict = tit.create_dataset_dict()[well]
    n_dict = sum(int(np.asarray(da.mask).sum()) for da in via_dict.values())
    assert n_on == n_dict, "the two builders disagree about masking"
    assert n_on <= n_off
