"""Prtecan/prtecan.py."""

from __future__ import annotations

import copy
import itertools
import logging
import typing
import warnings
from dataclasses import InitVar, dataclass, field
from functools import cached_property, partial
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  # type: ignore[import-untyped]
from matplotlib import figure
from scipy.odr import ODR, Model, RealData  # type: ignore[import-untyped]

from clophfit.fitting.bayes import (
    extract_fit,
    fit_binding_pymc,
    fit_binding_pymc_multi,
    fit_binding_pymc_multi2,
    x_true_from_trace_df,
)
from clophfit.fitting.core import (
    FitResult,
    fit_binding_glob,
    outlier2,
    weight_da,
    weight_multi_ds_titration,
)
from clophfit.fitting.data_structures import DataArray, Dataset
from clophfit.fitting.errors import InsufficientDataError
from clophfit.fitting.odr import fit_binding_odr_recursive_outlier, format_estimate
from clophfit.fitting.plotting import PlotParameters

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from clophfit.clophfit_types import ArrayF

# TODO: Add tqdm progress bar

# list_of_lines
# after set([type(x) for l in csvl for x in l]) = float | int | str
STD_MD_LINE_LENGTH = 2
NUM_COLS_96WELL = 12
ROW_NAMES = tuple("ABCDEFGH")

logger = logging.getLogger(__name__)


def read_xls(path: Path) -> list[list[str | int | float]]:
    """Read first sheet of an xls file.

    Parameters
    ----------
    path : Path
        Path to `.xls` file.

    Returns
    -------
    list[list[str | int | float]]
        Lines as list_of_lines.
    """
    sheet = pd.read_excel(path, dtype=object)  # Keep original types
    # Add empty row and replace NaN
    n0 = pd.DataFrame([[np.nan] * len(sheet.columns)], columns=sheet.columns)
    sheet = pd.concat([n0, sheet], ignore_index=True).fillna("")
    return typing.cast("list[list[str | int | float]]", sheet.to_numpy().tolist())


@typing.overload
def lookup_listoflines(csvl: list[list[str]], pattern: str, col: int) -> list[int]: ...


@typing.overload
def lookup_listoflines(
    csvl: list[list[str | int | float]], pattern: str = "Label: Label", col: int = 0
) -> list[int]: ...


def lookup_listoflines(
    csvl: list[list[str | int | float]] | list[list[str]],
    pattern: str = "Label: Label",
    col: int = 0,
) -> list[int]:
    """Lookup line numbers (row index) where given pattern occurs.

    Parameters
    ----------
    csvl : list[list[str | int | float]] | list[list[str]]
        Lines (list_of_lines) of a csv/xls file.
    pattern : str
        Pattern to be searched (default="Label: Label").
    col : int
        Column to search (default=0).

    Returns
    -------
    list[int]
        Row/line index for all occurrences of pattern. Empty list for no occurrences.
    """
    indexes = []
    for i, line in enumerate(csvl):
        if isinstance(line, list) and col < len(line):
            element = line[col]
            if isinstance(element, str | int | float) and pattern in str(element):
                indexes.append(i)
    return indexes


def strip_lines(lines: list[list[str | int | float]]) -> list[list[str | int | float]]:
    """Remove empty fields/cells from lines read from a csv file.

    Parameters
    ----------
    lines : list[list[str | int | float]]
        Lines (list_of_lines) that are a list of fields, typically from a csv/xls file.

    Returns
    -------
    list[list[str | int | float]]
        Lines (list_of_lines) removed from blank cells.

    Examples
    --------
    >>> lines = [
    ...     ["Shaking (Linear) Amplitude:", "", "", "", 2, "mm", "", "", "", "", ""]
    ... ]
    >>> strip_lines(lines)
    [['Shaking (Linear) Amplitude:', 2, 'mm']]
    """
    return [[e for e in line if e] for line in lines]


# MAYBE: with a filter ectract_metadata with a map


@dataclass(frozen=False)
class Metadata:
    """Represents the value of a metadata dictionary.

    Parameters
    ----------
    value : int | str | float | None
        The value for the dictionary key.
    unit : Sequence[str | float | int] | None, optional
        The first element represents the unit, while the following elements are
        only listed.
    """

    value: int | str | float | None
    unit: Sequence[str | float | int] | None = None


def extract_metadata(
    lines: list[list[str | int | float]],
) -> dict[str, Metadata]:
    """Extract metadata into both Tecanfile and Labelblock.

    From a list of stripped lines takes the first field as the **key** of the
    metadata dictionary, remaining fields goes into a list of values with the
    exception of Label ([str]) and Temperature ([float]).

    Parameters
    ----------
    lines : list[list[str | int | float]]
        Lines (list_of_lines) that are a list of fields, typically from a
        csv/xls file.

    Returns
    -------
    dict[str, Metadata]
        Metadata for Tecanfile or Labelblock.

    Examples
    --------
    >>> lines = [
    ...     ["Shaking (Linear) Amplitude:", "", "", "", 2, "mm", "", "", "", "", ""]
    ... ]
    >>> extract_metadata(lines)
    {'Shaking (Linear) Amplitude:': Metadata(value=2, unit=['mm'])}

    >>> lines = [["", "Temperature: 26 °C", "", "", "", "", "", "", "", "", ""]]
    >>> extract_metadata(lines)
    {'Temperature': Metadata(value=26.0, unit=['°C'])}

    >>> lines = [["Excitation Wavelength", "", "", "", 400, "nm", "", "", "", "", ""]]
    >>> extract_metadata(lines)
    {'Excitation Wavelength': Metadata(value=400, unit=['nm'])}

    >>> lines = [["Label: Label1", "", "", "", "", "", "", "", "", "", "", "", ""]]
    >>> extract_metadata(lines)
    {'Label': Metadata(value='Label1', unit=None)}

    >>> lines = [["Mode", "", "", "", "Fluorescence Top Reading", "", "", "", "", ""]]
    >>> extract_metadata(lines)["Mode"].value
    'Fluorescence Top Reading'
    """
    md: dict[str, Metadata] = {}

    for line in strip_lines(lines):
        if len(line) > STD_MD_LINE_LENGTH:
            md.update({str(line[0]): Metadata(line[1], line[2:])})
        elif len(line) == STD_MD_LINE_LENGTH:
            md.update({str(line[0]): Metadata(line[1])})
        elif len(line) == 1 and isinstance(line[0], str) and ":" in line[0]:
            k, v = line[0].split(":")
            vals: list[str] = v.split()
            val: float | str
            try:
                val = float(vals[0])
            except ValueError:
                val = vals[0]
            if len(vals) == 1:
                md.update({k: Metadata(val)})
            else:
                md.update({k: Metadata(val, vals[1:])})
        elif line:
            md.update({str(line[0]): Metadata(line[0])})

    return md


def merge_md(mds: list[dict[str, Metadata]]) -> dict[str, Metadata]:
    """Merge a list of metadata dict if the key value is the same in the list."""
    mmd = {k: v for k, v in mds[0].items() if all(v == md[k] for md in mds[1:])}

    # To account for the case 93"Optimal" and 93"Manual" in lb metadata
    def all_same_gain(mds: list[dict[str, Metadata]]) -> bool:
        return all(md["Gain"].value == mds[0]["Gain"].value for md in mds[1:])

    if (
        mmd.get("Gain") is None
        and mds[0].get("Gain") is not None
        and all_same_gain(mds)
    ):
        mmd["Gain"] = Metadata(mds[0]["Gain"].value)

    return mmd


def calculate_conc(
    additions: Sequence[float], conc_stock: float, conc_ini: float = 0.0
) -> ArrayF:
    """Calculate concentration values.

    additions[0]=vol_ini; Stock concentration is a parameter.

    Parameters
    ----------
    additions : Sequence[float]
        Initial volume and all subsequent additions.
    conc_stock : float
        Concentration of the stock used for additions.
    conc_ini : float
        Initial concentration (default=0).

    Returns
    -------
    ArrayF
        Concentrations as vector.
    """
    vol_tot = np.cumsum(additions)
    concs = np.ones(len(additions))
    concs[0] = conc_ini
    for i, add in enumerate(additions[1:], start=1):
        concs[i] = (concs[i - 1] * vol_tot[i - 1] + conc_stock * float(add)) / vol_tot[
            i
        ]
    return concs  # , vol_tot


def dilution_correction(additions: list[float]) -> ArrayF:
    """Apply dilution correction.

    Parameters
    ----------
    additions: list[float]
        List of initial volume (index=0) followed by all additions.

    Returns
    -------
    ArrayF
        Dilution correction vector.
    """
    volumes = np.cumsum(additions)
    corrections: ArrayF = volumes / volumes[0]
    return corrections


@dataclass
class Labelblock:
    """Parse a label block.

    Parameters
    ----------
    _lines : list[list[str | int | float]]
        Lines to create this Labelblock.

    Raises
    ------
    Exception
        When data do not correspond to a complete 96-well plate.
    ValueError
        When something went wrong. Possibly because not 96-well.
    TypeError
        When normalization parameters are not numerical.

    Logging
    -------
    Logs a warning
        When it replaces "OVER" with ``np.nan`` for saturated values.
    """

    _lines: InitVar[list[list[str | int | float]]]
    #: Path of the corresponding Tecan file.
    filename: str = ""
    #: Metadata specific for this Labelblock.
    metadata: dict[str, Metadata] = field(init=False, repr=True)
    #: Plate data values as {'well_name', value}.
    data: dict[str, float] = field(init=False, repr=True)
    #: Plate data values normalized as {'well_name', value}.
    data_nrm: dict[str, float] = field(init=False, repr=True)
    _KEYS: typing.ClassVar[list[str]] = [
        "Emission Bandwidth",
        "Emission Wavelength",
        "Excitation Bandwidth",
        "Excitation Wavelength",
        "Mode",
        "Integration Time",
        "Number of Flashes",
    ]
    _NORM_KEYS: typing.ClassVar[list[str]] = [
        "Integration Time",
        "Number of Flashes",
        "Gain",
    ]

    def __post_init__(self, lines: list[list[str | int | float]]) -> None:
        """Create metadata and data."""
        self._validate_lines(lines)
        stripped = strip_lines(lines)
        stripped[14:23] = []
        self.metadata: dict[str, Metadata] = extract_metadata(stripped)
        self.data: dict[str, float] = self._extract_data(lines[15:23])
        self._normalize_data()

    def _validate_lines(self, lines: list[list[str | int | float]]) -> None:
        """Validate if input lines correspond to a 96-well plate."""
        if not (lines[14][0] == "<>" and lines[23] == lines[24] == [""] * 13):
            msg = "Cannot build Labelblock: not 96 wells?"
            raise ValueError(msg)

    def _extract_data(self, lines: list[list[str | int | float]]) -> dict[str, float]:
        """Extract data from a list of lines into a dictionary.

        This function validates the 96-well format, processes each cell, and
        converts its content to a float value. Cells with invalid or saturated
        signals are replaced with `np.nan`, and a warning is logged.

        Parameters
        ----------
        lines : list[list[str | int | float]]
            Input data read from an xls file, organized as rows and columns.

        Returns
        -------
        dict[str, float]
            A dictionary where keys are well identifiers (e.g., 'A01', 'B02')
            and values are the corresponding numeric data. Invalid or saturated
            signals are stored as `np.nan` and a warn is logged.

        Notes
        -----
        - Assumes the input `lines` corresponds to a 96-well plate format.
        - Rows are labeled 'A' through 'H'; columns are numbered 1 through 12.
        """
        data = {}
        self._validate_96_well_format(lines)
        for i, row in enumerate(ROW_NAMES):
            for col in range(1, NUM_COLS_96WELL + 1):
                well = f"{row}{col:0>2}"
                try:
                    data[well] = float(lines[i][col])
                except ValueError:
                    data[well] = np.nan
                    label = self.metadata.get("Label")
                    if label is not None and hasattr(label, "value"):
                        lbl = label.value
                    else:
                        lbl = "Unknown"
                    msg = f" OVER value in {lbl}: {well} of tecanfile {self.filename}"
                    logger.warning(msg)
        return data

    def _validate_96_well_format(self, lines: list[list[str | int | float]]) -> None:
        """Validate 96-well plate data format."""
        for i, row in enumerate(ROW_NAMES):
            if lines[i][0] != row:
                msg = f"Row {i} label mismatch: expected {row}, got {lines[i][0]}"
                raise ValueError(msg)

    def _normalize_data(self) -> None:
        """Normalize the data against specific metadata."""
        try:
            norm = 1000.0
            for k in Labelblock._NORM_KEYS:
                val = self.metadata[k].value
                if isinstance(val, float | int):
                    norm /= val
        except TypeError as err:
            msg = "Could not normalize for non numerical Gain, Number of Flashes or Integration time."
            raise TypeError(msg) from err
        self.data_nrm = {k: v * norm for k, v in self.data.items()}

    def __eq__(self, other: object) -> bool:
        """Check if two Labelblocks are equal (same key metadata)."""
        if not isinstance(other, Labelblock):
            msg = f"Cannot compare Labelblock object with {type(other).__name__}"
            raise TypeError(msg)
        return (
            all(self.metadata[k] == other.metadata[k] for k in Labelblock._KEYS)
            and self.metadata["Gain"].value == other.metadata["Gain"].value
        )

    def __almost_eq__(self, other: Labelblock) -> bool:
        """Check if two Labelblocks are almost equal (same excitation and emission)."""
        return all(self.metadata[k] == other.metadata[k] for k in Labelblock._KEYS[:5])

    def __hash__(self) -> int:
        """Return a hash value based Labelblock metadata."""
        return hash(self.metadata)


@dataclass
class Tecanfile:
    """Parse a Tecan .xls file.

    Parameters
    ----------
    path: Path
        Path to `.xls` file.

    Raises
    ------
    FileNotFoundError
        When path does not exist.
    Exception
        When no Labelblock is found.
    """

    path: Path
    #: General metadata for Tecanfile, like `Date` and `Shaking Duration`.
    metadata: dict[str, Metadata] = field(init=False, repr=False)
    #: All labelblocks contained in this file.
    labelblocks: dict[int, Labelblock] = field(
        init=False, repr=False, default_factory=dict
    )

    def __post_init__(self) -> None:
        """Initialize."""
        csvl = read_xls(self.path)
        idxs = lookup_listoflines(csvl, pattern="Label: Label", col=0)
        if not idxs:
            msg = "No Labelblock found."
            raise ValueError(msg)
        self.metadata = extract_metadata(csvl[: idxs[0]])
        n_labelblocks = len(idxs)
        idxs.append(len(csvl))
        labelblocks = {
            i + 1: Labelblock(csvl[idxs[i] : idxs[i + 1]], str(self.path))
            for i in range(n_labelblocks)
        }
        self.labelblocks = labelblocks
        if self._has_repeated_labelblocks():
            logger.warning("Repeated labelblocks")

    def _has_repeated_labelblocks(self) -> bool:
        """Check for repeated labelblocks."""
        return any(
            self.labelblocks[i] == self.labelblocks[j]
            for i, j in itertools.combinations(self.labelblocks.keys(), 2)
        )


@dataclass
class LabelblocksGroup:
    """Group labelblocks with compatible metadata.

    Parameters
    ----------
    labelblocks: list[Labelblock]
        Labelblocks to be grouped.
    allequal: bool
        True if labelblocks already tested equal.

    Raises
    ------
    Exception
        When labelblocks are neither equal nor almost equal.
    """

    labelblocks: list[Labelblock] = field(repr=False)
    allequal: bool = False
    #: Metadata shared by all labelblocks.
    metadata: dict[str, Metadata] = field(init=False, repr=True)

    @cached_property
    def data(self) -> dict[str, list[float]]:
        """Grouped data if labelblocks are equal, otherwise empty."""
        if not self.allequal:
            return {}
        return self._collect_data("data")

    @cached_property
    def data_nrm(self) -> dict[str, list[float]]:
        """Normalized data by number of flashes, integration time and gain."""
        return self._collect_data("data_nrm")

    def __post_init__(self) -> None:
        """Initialize common metadata and validate labelblocks."""
        self._validate_labelblocks()
        self.metadata = merge_md([lb.metadata for lb in self.labelblocks])

    def _validate_labelblocks(self) -> None:
        """Validate labelblocks for equality or near-equality."""
        labelblocks = self.labelblocks
        # Check if all labelblocks are exactly equal
        if self.allequal or all(labelblocks[0] == lb for lb in labelblocks[1:]):
            self.allequal = True
        # Check if all labelblocks are almost equal (requires normalization)
        elif all(labelblocks[0].__almost_eq__(lb) for lb in labelblocks[1:]):
            self.allequal = False
        else:
            msg = "Creation of labelblock group failed. Labelblocks are neither equal nor almost equal."
            raise ValueError(msg)

    def _collect_data(self, attribute: str) -> dict[str, list[float]]:
        """Collect data from labelblocks for a given attribute."""
        return {
            key: [getattr(lb, attribute)[key] for lb in self.labelblocks]
            for key in getattr(self.labelblocks[0], attribute)
        }


@dataclass
class TecanfilesGroup:
    """Group of Tecanfiles containing at least one common Labelblock.

    Parameters
    ----------
    tecanfiles: list[Tecanfile]
        List of Tecanfiles.

    Raises
    ------
    ValueError
        When no common Labelblock is found across all Tecanfiles.

    Logging
    -------
    Logs a warning
        If the Tecanfiles do not contain the same number of Labelblocks that
        can be merged in the same order, a warning is logged. In such cases,
        fewer LabelblocksGroup may be created.
    """

    tecanfiles: list[Tecanfile]

    #: Each group contains its own data like a titration. ??
    labelblocksgroups: dict[int, LabelblocksGroup] = field(
        init=False, default_factory=dict
    )
    #: Metadata shared by all tecanfiles.
    metadata: dict[str, Metadata] = field(init=False, repr=True)

    def __post_init__(self) -> None:
        """Initialize metadata and labelblocksgroups."""
        if self._all_labelblocks_equal():
            self._create_equal_groups()
        else:
            self._create_almostequal_groups()
        self.metadata = merge_md([tf.metadata for tf in self.tecanfiles])

    def _all_labelblocks_equal(self) -> bool:
        """Check if all Tecanfiles have the same Labelblocks."""
        tf0 = self.tecanfiles[0]
        return all(tf0.labelblocks == tf.labelblocks for tf in self.tecanfiles[1:])

    def _create_equal_groups(self) -> None:
        """Create LabelblocksGroup when all Labelblocks are equal."""
        tf0 = self.tecanfiles[0]
        self.labelblocksgroups = {
            i: LabelblocksGroup(
                [tf.labelblocks[i] for tf in self.tecanfiles], allequal=True
            )
            for i in tf0.labelblocks
        }

    def _create_almostequal_groups(self) -> None:
        """Create as many LabelblocksGroups as possible when not all Labelblocks are equal."""
        rngs = tuple(tf.labelblocks.keys() for tf in self.tecanfiles)
        for idx in itertools.product(*rngs):
            try:
                gr = LabelblocksGroup(
                    [tf.labelblocks[idx[i]] for i, tf in enumerate(self.tecanfiles)]
                )
            except ValueError:
                continue
            else:
                self.labelblocksgroups[idx[0]] = gr
        if not self.labelblocksgroups:
            msg = f"No common labelblocks in files: {[tf.path.name for tf in self.tecanfiles]}."
            raise ValueError(msg)
        logger.warning(
            f"Different LabelblocksGroup across files: {[str(tf.path) for tf in self.tecanfiles]}."
        )


@dataclass
class PlateScheme:
    """Define buffer, discard, ctrl and unk wells, and ctrl names.

    Parameters
    ----------
    file: Path
        File path to the scheme file [<well Id, sample name>].
    """

    file: Path | None = None
    _buffer: list[str] = field(default_factory=list, init=False)
    _discard: list[str] = field(default_factory=list, init=False)
    _ctrl: list[str] = field(default_factory=list, init=False)
    _names: dict[str, set[str]] = field(default_factory=dict, init=False)

    @cached_property
    def nofit_keys(self) -> set[str]:
        """Buffer and discarded wells."""
        return set(self.buffer) | set(self.discard)

    @property
    def buffer(self) -> list[str]:
        """List of buffer wells."""
        return self._buffer

    @buffer.setter
    def buffer(self, value: list[str]) -> None:
        if not all(isinstance(item, str) for item in value):
            msg = "Buffer wells must be a list of strings"
            raise TypeError(msg)
        self._buffer = value

    @property
    def discard(self) -> list[str]:
        """List of discard wells."""
        return self._discard

    @discard.setter
    def discard(self, value: list[str]) -> None:
        if not all(isinstance(item, str) for item in value):
            msg = "Discard wells must be a list of strings"
            raise TypeError(msg)
        self._discard = value

    @property
    def ctrl(self) -> list[str]:
        """List of CTR wells."""
        return self._ctrl

    @ctrl.setter
    def ctrl(self, value: list[str]) -> None:
        if not all(isinstance(item, str) for item in value):
            msg = "Ctrl wells must be a list of strings"
            raise TypeError(msg)
        self._ctrl = value

    @property
    def names(self) -> dict[str, set[str]]:
        """A dictionary mapping sample names to their associated list of wells."""
        return self._names

    @names.setter
    def names(self, value: dict[str, set[str]]) -> None:
        msg = "Names must be a dictionary mapping strings to sets of strings"
        if not isinstance(value, dict):
            raise TypeError(msg)
        for k, v in value.items():
            if not isinstance(k, str):
                raise TypeError(msg)
            if not isinstance(v, set):
                raise TypeError(msg)
            if not all(isinstance(item, str) for item in v):
                raise TypeError(msg)
        self._names = value

    def __post_init__(self) -> None:
        """Complete initialization."""
        if self.file:
            table = pd.read_csv(self.file, sep="\t")
            if (
                table.columns.tolist() != ["well", "sample"]
                or table["well"].count() != table["sample"].count()
            ):
                msg = f"Check format [well sample] for schemefile: {self.file}"
                raise ValueError(msg)
            scheme = table.groupby("sample")["well"].unique()
            self.buffer = list(scheme.get("buffer", []))
            self.discard = list(scheme.get("discard", []))
            self.ctrl = list(
                {well for sample in scheme.tolist() for well in sample}
                - set(self.buffer)
                - set(self.discard)
            )
            self.names = {
                str(k): set(v)
                for k, v in scheme.items()
                if k not in ("buffer", "discard")
            }


@dataclass
class TitrationConfig:
    """Parameters defining the fitting data with callback support."""

    bg: bool = True
    bg_adj: bool = False
    dil: bool = True
    nrm: bool = True
    bg_mth: str = "mean"
    mcmc: str = "None"

    _callback: Callable[[], None] | None = field(
        default=None, repr=False, compare=False
    )

    def set_callback(self, callback: Callable[[], None]) -> None:
        """Set the callback to be triggered on parameter change."""
        self._callback = callback

    def _trigger_callback(self) -> None:
        if self._callback is not None:
            self._callback()

    def __setattr__(self, name: str, value: bool | str) -> None:
        """Trigger callback when a tracked attribute value actually changes."""
        if name == "_callback":
            super().__setattr__(name, value)
        else:
            current_value = getattr(self, name, None)
            super().__setattr__(name, value)
            if current_value != value:
                self._trigger_callback()


@dataclass
class BufferFit:
    """Store (robust) linear fit result."""

    m: float = np.nan
    q: float = np.nan
    m_err: float = np.nan
    q_err: float = np.nan

    @property
    def empty(self) -> bool:
        """Return True if all attributes are NaN, emulating DataFrame's empty behavior."""
        return all(np.isnan(value) for value in vars(self).values())


@dataclass
class Buffer:
    """Buffer handling for a titration."""

    tit: Titration

    _wells: list[str] = field(default_factory=list)
    _bg: dict[int, ArrayF] = field(init=False, default_factory=dict)
    _bg_err: dict[int, ArrayF] = field(init=False, default_factory=dict)

    fit_results: dict[int, BufferFit] = field(init=False, default_factory=dict)
    fit_results_nrm: dict[int, BufferFit] = field(init=False, default_factory=dict)

    @cached_property
    def dataframes(self) -> dict[int, pd.DataFrame]:
        # def dataframes(self) -> list[pd.DataFrame]:
        """Buffer dataframes with fit."""
        if not self.wells:
            return {}
        dfs = {
            label: pd.DataFrame(
                {k: lbg.data[k] for k in self.wells if lbg.data and k in lbg.data}
            )
            for label, lbg in self.tit.labelblocksgroups.items()
        }
        self.fit_results = self._fit_buffer(dfs)  # Perform fit
        return dfs

    @cached_property
    def dataframes_nrm(self) -> dict[int, pd.DataFrame]:
        # def dataframes_nrm(self) -> list[pd.DataFrame]:
        """Buffer normalized dataframes with fit."""
        if not self.wells:
            return {}
        dfs_nrm = {
            label: pd.DataFrame({k: lbg.data_nrm[k] for k in self.wells})
            for label, lbg in self.tit.labelblocksgroups.items()
        }
        self.fit_results_nrm = self._fit_buffer(dfs_nrm)  # Perform fit
        return dfs_nrm

    @property
    def wells(self) -> list[str]:
        """List of buffer wells."""
        return self._wells

    @wells.setter
    def wells(self, wells: list[str]) -> None:
        """Set the list of buffer wells and trigger recomputation."""
        self._wells = wells
        self._reset_cache()
        self.tit.clear_all_data_results()

    def _reset_cache(self) -> None:
        """Reset all cached properties."""
        for cached_attr in ["dataframes", "dataframes_nrm", "bg", "bg_err"]:
            if cached_attr in self.__dict__:
                del self.__dict__[cached_attr]

    @property
    def bg(self) -> dict[int, ArrayF]:
        """List of buffer values."""
        if not self._bg:
            self._bg, self._bg_err = self._compute_bg_and_sd()
        return self._bg

    @bg.setter
    def bg(self, value: dict[int, ArrayF]) -> None:
        """Set the buffer values and reset SEM."""
        self._bg = value

    @property
    def bg_err(self) -> dict[int, ArrayF]:
        """List of buffer SEM values."""
        if not self._bg_err:
            self._bg, self._bg_err = self._compute_bg_and_sd()
        return self._bg_err

    @bg_err.setter
    def bg_err(self, value: dict[int, ArrayF]) -> None:
        # def bg_err(self, value: list[ArrayF]) -> None:
        """Set the buffer SEM values manually."""
        self._bg_err = value

    def _compute_bg_and_sd(self) -> tuple[dict[int, ArrayF], dict[int, ArrayF]]:
        """Compute and return buffer values and their SEM."""
        buffers = self.dataframes_nrm if self.tit.params.nrm else self.dataframes
        bg = {}
        bg_err = {}
        # Mapping methods to column names for clarity and reuse
        method_map = {
            "fit": ("fit", "fit_err"),
            "mean": ("mean", "sem"),
            "meansd": ("mean", "sem"),
        }
        if self.tit.params.bg_mth not in method_map:
            msg = f"Unknown bg_method: {self.tit.params.bg_mth}"
            raise ValueError(msg)
        value_col, error_col = method_map[self.tit.params.bg_mth]
        for label, bdf in buffers.items():
            if bdf.empty:
                bg[label] = np.array([])
                bg_err[label] = np.array([])
                continue
            bg[label] = bdf[value_col].to_numpy()
            if self.tit.params.bg_mth == "meansd":
                bg_err[label] = np.repeat(
                    np.nanpercentile(bdf[error_col], 50), len(bdf[error_col])
                )
            else:
                bg_err[label] = bdf[error_col].to_numpy()
        return bg, bg_err

    def _fit_buffer(self, dataframed: dict[int, pd.DataFrame]) -> dict[int, BufferFit]:
        """Fit buffers of all labelblocksgroups."""

        def linear_model(pars: list[float], x: ArrayF) -> ArrayF:
            """Define linear model function."""
            return pars[0] * x + pars[1]

        def fit_error(x: ArrayF, cov_matrix: ArrayF) -> ArrayF:
            x = x[:, np.newaxis]  # Ensure x is a 2D array
            jacobian = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
            fit_variance: ArrayF = np.einsum(
                "ij,jk,ik->i", jacobian, cov_matrix, jacobian
            )
            return np.sqrt(fit_variance)  # Standard error

        fit_resultd = {}
        for label, buf_df in dataframed.items():
            if buf_df.empty:
                fit_resultd[label] = BufferFit()
            else:
                mean = buf_df.mean(axis=1).to_numpy()
                sem = buf_df.sem(axis=1).to_numpy()
                # y_err estimate is important when using 2 ds and x_err for ODR
                data = RealData(self.tit.x, mean, sy=sem, sx=self.tit.x_err)
                model = Model(linear_model)
                # Initial guess for slope and intercept
                odr = ODR(data, model, beta0=[0.0, mean.mean()])
                output = odr.run()
                # Extract the best-fit parameters and their standard errors
                m_best, q_best = output.beta
                m_err, q_err = output.sd_beta
                cov_matrix = output.cov_beta
                fit_resultd[label] = BufferFit(m_best, q_best, m_err, q_err)
                buf_df["Label"] = label
                buf_df["fit"] = m_best * self.tit.x + q_best
                buf_df["fit_err"] = fit_error(self.tit.x, cov_matrix)
                buf_df["mean"] = mean
                buf_df["sem"] = sem
        return fit_resultd

    def plot(self, nrm: bool = False, title: str | None = None) -> sns.FacetGrid:
        """Plot buffers of all labelblocksgroups."""
        dataframed = self.dataframes_nrm if nrm else self.dataframes
        fit_results = self.fit_results_nrm if nrm else self.fit_results
        if not dataframed or not self.wells:
            return sns.catplot()
        pp = PlotParameters(is_ph=self.tit.is_ph)
        melted_buffers = []
        wells_lbl = self.wells.copy()
        wells_lbl.extend(["Label"])
        for buf_df in dataframed.values():
            if not buf_df.empty:
                buffer = buf_df[wells_lbl].copy()
                buffer[pp.kind] = self.tit.x
                melted_buffers.append(
                    buffer.melt(
                        id_vars=[pp.kind, "Label"], var_name="well", value_name="F"
                    )
                )
        # Combine data from both buffers
        data = pd.concat(melted_buffers, ignore_index=True)
        g = sns.lmplot(
            data=data,
            y="F",
            x=pp.kind,
            ci=68,
            height=4,
            aspect=1.75,
            row="Label",
            x_estimator=np.median,
            markers="x",
            scatter=1,
            scatter_kws={"alpha": 0.33},
            facet_kws={"sharey": False},
        )
        # Determine the number of non-empty label groups
        num_labels = sum(not b_df.empty for b_df in dataframed.values())
        for label in dataframed:
            if not dataframed[label].empty:
                sns.scatterplot(
                    data=data[data.Label == label],
                    y="F",
                    x=pp.kind,
                    hue="well",
                    ax=g.axes_dict[label],
                    legend=label == num_labels,
                )
                g.axes_dict[label].errorbar(
                    x=self.tit.x,
                    y=dataframed[label]["fit"],
                    yerr=dataframed[label]["fit_err"],
                    xerr=self.tit.x_err,
                    fmt="",
                    color="r",
                    linewidth=2,
                    capsize=6,
                )
                # Extract the slope and intercept from BufferFit
                buffer_fit = fit_results[label]
                m = buffer_fit.m
                m_err = buffer_fit.m_err
                q = buffer_fit.q
                q_err = buffer_fit.q_err
                # Add slope and intercept as text annotation on the plot
                g.axes_dict[label].text(
                    0.05,
                    0.95,  # Position of the text (adjust as needed)
                    f"m = {m:.1f} ± {m_err:.1f}\nq = {q:.0f} ± {q_err:.1f}",
                    transform=g.axes_dict[label].transAxes,
                    fontsize=10,
                    verticalalignment="top",
                    bbox={
                        "boxstyle": "round,pad=0.3",
                        "edgecolor": "black",
                        "facecolor": "white",
                        "alpha": 0.7,
                    },
                )
        if title:
            plt.suptitle(title, fontsize=14, x=0.96, ha="right")
        plt.close()
        return g


@dataclass
class TitrationResults:
    """Manage titration results with optional lazy computation."""

    scheme: PlateScheme
    fit_keys: set[str]
    compute_func: Callable[[str], FitResult]
    results: dict[str, FitResult] = field(default_factory=dict)
    _dataframe: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def dataframe(self) -> pd.DataFrame:
        """Convert FitResult dictionary to a DataFrame."""
        if all(key in list(self._dataframe.index) for key in self.fit_keys):
            return self._dataframe
        if not self.results:
            self.compute_all()
        data = []
        for lbl, fr in self.results.items():
            pars = fr.result.params if fr.result else None
            row = {"well": lbl}
            if pars is not None:
                for k in pars:
                    row[k] = pars[k].value
                    row[f"s{k}"] = pars[k].stderr
                    row[f"{k}hdi03"] = pars[k].min
                    row[f"{k}hdi97"] = pars[k].max
            data.append(row)
        self._dataframe = pd.DataFrame(data).set_index("well")
        return self._dataframe

    def __repr__(self) -> str:
        """Get or lazily compute a result for a given key."""
        return repr(self.results)

    def __getitem__(self, key: str) -> FitResult:
        """Get or lazily compute a result for a given key."""
        if key not in self.results:
            if key not in self.fit_keys:
                msg = f"Key '{key}' is not a valid fit key."
                raise KeyError(msg)
            self.results[key] = self.compute_func(key)
        return self.results[key]

    def __bool__(self) -> bool:
        """Return True if there are any computed results, trigger full computation."""
        return bool(self.results)

    def __call__(self) -> None:
        """Ensure all results are computed when called."""
        self.compute_all()

    def __len__(self) -> int:
        """Ensure length is accurate after full computation."""
        return len(self.results)

    def compute_all(self) -> None:
        """Compute results for all keys."""
        for key in self.fit_keys:
            # Access each key to trigger computation
            self[key]

    def n_sd(self, par: str = "K", expected_sd: float = 0.15) -> float:
        """Compute median of K."""
        if not self.all_computed():
            self.compute_all()
        try:
            n_sd: float = expected_sd / np.nanmedian(
                [v.result.params[par].stderr for v in self.results.values() if v.result]
            )
        except ZeroDivisionError:
            logger.warning("Unable to calculate n_sd; defaulting to 1.0")
            n_sd = 1.0  # Fallback if stderr values are missing
        return n_sd

    def all_computed(self) -> bool:
        """Check if all keys have been computed."""
        return all(key in self.results for key in self.fit_keys)

    def export_pngs(self, folder: str | Path) -> None:
        """Export all fit result plots as PNG files."""
        path = Path(folder)
        path.mkdir(parents=True, exist_ok=True)
        for well, result in self.results.items():
            if result.figure:
                result.figure.savefig(path / f"{well}.png")

    def export_data(self, folder: str | Path) -> None:
        """Export all datasets as CSV files."""
        path = Path(folder) / "ds"
        path.mkdir(parents=True, exist_ok=True)
        for well, result in self.results.items():
            if result.dataset:
                result.dataset.export(path / f"{well}.csv")

    # MAYBE: Test plots
    def plot_k(
        self,
        xlim: tuple[float, float] | None = None,
        title: str = "",
    ) -> figure.Figure:
        """Plot K values as stripplot.

        Parameters
        ----------
        xlim : tuple[float, float] | None, optional
            Range.
        title : str, optional
            To name the plot.

        Returns
        -------
        figure.Figure
            The figure.
        """
        dataframe = self.dataframe
        with sns.plotting_context("paper"):  # axes_style("whitegrid"):
            fig = plt.figure(figsize=(12, 16))
            keys_unk = list(set(dataframe.index))
            if self.scheme.names:
                keys_unk = list(set(dataframe.index) - set(self.scheme.ctrl))
                df_ctr = dataframe.loc[dataframe.index.intersection(self.scheme.ctrl)]
                for name, wells in self.scheme.names.items():
                    for well in wells:
                        df_ctr.loc[well, "ctrl"] = name
                df_ctr = df_ctr.sort_values("ctrl")
                ax1 = plt.subplot2grid((8, 1), loc=(0, 0))
                x, y, hue = (df_ctr["K"], df_ctr.index, df_ctr["ctrl"])
                sns.stripplot(x=x, y=y, size=8, orient="h", hue=hue, ax=ax1)
                ax1.errorbar(x, y, xerr=df_ctr["sK"], fmt=".", c="lightgray", lw=8)
                ax1.legend(loc="upper left", frameon=False)
                ax1.grid(True, axis="both")
                ax1.set_xticklabels([])
                ax1.set_xlabel("")
            ax2 = plt.subplot2grid((8, 1), loc=(1, 0), rowspan=7)
            df_unk = dataframe.loc[keys_unk].sort_index(ascending=False)
            # Sort by 'K - 2 * sK'.
            df_unk["sort_val"] = df_unk["K"] - 2 * df_unk["sK"]
            df_unk = df_unk.sort_values(by="sort_val", ascending=True)
            x, y = df_unk["K"], df_unk.index
            sns.stripplot(x=x, y=y, size=9, orient="h", ax=ax2)
            ax2.errorbar(x, y, xerr=df_unk["sK"], fmt=".", c="gray", lw=2)
            ax2.grid(True, axis="both")
            ax2.set_yticks(range(len(df_unk)))
            ax2.set_yticklabels([str(label) for label in df_unk.index])
            ax2.set_ylim(-1, len(df_unk))
            # Set x-limits
            xlim = xlim if xlim else self._determine_xlim(df_ctr, df_unk)
            if self.scheme.ctrl:
                ax1.set_xlim(xlim)
            ax2.set_xlim(xlim)
            # Set title
            fig.suptitle(title, fontsize=16)
            fig.tight_layout(pad=1.2, w_pad=0.1, h_pad=0.5, rect=(0, 0, 1, 0.97))
            # Close the figure after returning it to avoid memory issues
        plt.close(fig)
        return fig

    def _determine_xlim(
        self, df_ctr: pd.DataFrame, df_unk: pd.DataFrame
    ) -> tuple[float, float]:
        lower, upper = 0.99, 1.01
        xlim = (df_unk["K"].min(), df_unk["K"].max())
        if not df_ctr.empty:
            xlim = (min(df_ctr["K"].min(), xlim[0]), max(df_ctr["K"].max(), xlim[1]))
            xlim = (lower * xlim[0], upper * xlim[1])
        return xlim


@dataclass
class Titration(TecanfilesGroup):
    """Build titrations from grouped Tecanfiles and concentrations or pH values.

    Parameters
    ----------
    tecanfiles: list[Tecanfile]
        List of Tecanfiles.
    x : ArrayF
        Concentration or pH values.
    is_ph : bool
        Indicate if x values represent pH (default is False).
    x_err : ArrayF | None
        Uncertainties of concentration or pH values.

    Raises
    ------
    ValueError
        For unexpected file format, e.g. header `names`.
    """

    x: ArrayF
    is_ph: bool
    x_err: ArrayF = field(default_factory=lambda: np.array([]))
    buffer: Buffer = field(init=False)

    _params: TitrationConfig = field(init=False, default_factory=TitrationConfig)
    _additions: list[float] = field(init=False, default_factory=list)
    _scheme: PlateScheme = field(init=False, default_factory=PlateScheme)
    _bg: dict[int, ArrayF] = field(init=False, default_factory=dict)
    _bg_err: dict[int, ArrayF] = field(init=False, default_factory=dict)
    _data: dict[int, dict[str, ArrayF]] = field(init=False, default_factory=dict)

    #: A list of wells containing samples that are neither buffer nor CTR samples.
    _dil_corr: ArrayF = field(init=False, default_factory=lambda: np.array([]))

    def __post_init__(self) -> None:
        """Create metadata and data."""
        self.buffer = Buffer(tit=self)
        self._params.set_callback(self._reset_data_results_and_bg)
        super().__post_init__()

    @cached_property
    def fit_keys(self) -> set[str]:
        """Set of wells to be fitted."""
        first_label = next(iter(self.labelblocksgroups.keys()))
        return (
            self.labelblocksgroups[first_label].data_nrm.keys() - self.scheme.nofit_keys
        )

    def _reset_data_and_results(self) -> None:
        self._data = {}
        if "results" in self.__dict__:
            del self.results
        if "result_global" in self.__dict__:
            del self.result_global
        if "result_odr" in self.__dict__:
            del self.result_odr
        if "result_mcmc" in self.__dict__:
            del self.result_mcmc

    def _reset_data_results_and_bg(self) -> None:
        self._reset_data_and_results()
        self.bg = {}
        self.bg_err = {}

    def clear_all_data_results(self) -> None:
        """Clear fit keys, data, results and bg when buffer or scheme properties change."""
        self._reset_data_results_and_bg()
        if "fit_keys" in self.__dict__:
            del self.fit_keys

    @property
    def params(self) -> TitrationConfig:
        """Get the datafit parameters."""
        return self._params

    @params.setter
    def params(self, value: TitrationConfig) -> None:
        self._params = value
        self._reset_data_results_and_bg()

    @property
    def bg(self) -> dict[int, ArrayF]:
        # def bg(self) -> list[ArrayF]:
        """List of buffer values."""
        return self.buffer.bg

    @bg.setter
    def bg(self, value: dict[int, ArrayF]) -> None:
        # def bg(self, value: list[ArrayF]) -> None:
        self.buffer.bg = value
        self._reset_data_and_results()

    @property
    def bg_err(self) -> dict[int, ArrayF]:
        # def bg_err(self) -> list[ArrayF]:
        """List of buffer SEM values."""
        return self.buffer.bg_err

    @bg_err.setter
    def bg_err(self, value: dict[int, ArrayF]) -> None:
        # def bg_err(self, value: list[ArrayF]) -> None:
        self.buffer.bg_err = value
        self._reset_data_and_results()

    def __repr__(self) -> str:
        """Return a string representation of the instance."""
        return (
            f'Titration\n\tfiles=["{self.tecanfiles[0].path}", ...],\n'
            f"\tx={list(self.x)!r},\n"
            f"\tx_err={list(self.x_err)!r},\n"
            f"\tlabels={self.labelblocksgroups.keys()},\n"
            f"\tparams={self.params!r}"
            f"\tpH={self.is_ph}"
            f"\tadditions={self.additions}"
            f"\n\tscheme={self.scheme})"
        )

    @classmethod
    def fromlistfile(cls, list_file: Path | str, is_ph: bool) -> Titration:
        """Build `Titration` from a list[.pH|.Cl] file."""
        tecanfiles, x, x_err = cls._listfile(Path(list_file))
        return cls(tecanfiles, x, is_ph, x_err=x_err)

    @staticmethod
    def _listfile(listfile: Path) -> tuple[list[Tecanfile], ArrayF, ArrayF]:
        """Help construction from list file."""
        try:
            table = pd.read_csv(listfile, names=["filenames", "x", "x_err"])
        except FileNotFoundError as exc:
            msg = f"Cannot find: {listfile}"
            raise FileNotFoundError(msg) from exc
        # For unexpected file format, e.g. length of filename column differs
        # from length of x values.
        if table["filenames"].count() != table["x"].count():
            msg = f"Check format [filenames x x_err] for listfile: {listfile}"
            raise ValueError(msg)
        tecanfiles = [Tecanfile(listfile.parent / f) for f in table["filenames"]]
        x = table["x"].to_numpy()
        x_err = table["x_err"].to_numpy()
        return tecanfiles, x, x_err

    @property
    def additions(self) -> list[float] | None:
        """List of initial volume followed by additions."""
        return self._additions

    # MAYBE: Here there is not any check on the validity of additions (e.g. length).
    @additions.setter
    def additions(self, additions: list[float]) -> None:
        self._additions = additions
        self._dil_corr = dilution_correction(additions)
        self._data = {}

    def load_additions(self, additions_file: Path) -> None:
        """Load additions from file."""
        additions = pd.read_csv(additions_file, names=["add"])
        self.additions = additions["add"].tolist()

    @property
    def data(self) -> dict[int, dict[str, ArrayF]]:
        # def data(self) -> list[dict[str, ArrayF]]:
        """Buffer subtracted and corrected for dilution data."""
        if not self._data:
            self._data = self._prepare_data()
        return self._data

    # def _prepare_data(self) -> list[dict[str, ArrayF]]:
    def _prepare_data(self) -> dict[int, dict[str, ArrayF]]:
        """Prepare and return the processed data."""
        # Step 1: Get raw or normalized data
        data = self._get_normalized_or_raw_data()
        # Step 2: Subtract background if enabled
        if self.params.bg and self.bg:
            data = self._subtract_background(data)
        # Step 3: Adjust for negative values if enabled
        if self.params.bg_adj:
            data = self._adjust_negative_values(data)
        # Step 4: Apply dilution correction if enabled
        if self.params.dil and self.additions:
            data = self._apply_dilution_correction(data)
        return data

    def _apply_dilution_correction(
        self, data: dict[int, dict[str, ArrayF]]
    ) -> dict[int, dict[str, ArrayF]]:
        """Apply dilution correction to the data (works with nan values)."""
        return {
            label: {k: v * self._dil_corr for k, v in dd.items()}
            for label, dd in data.items()
        }

    # def _get_normalized_or_raw_data(self) -> list[dict[str, ArrayF]]:
    def _get_normalized_or_raw_data(self) -> dict[int, dict[str, ArrayF]]:
        """Fetch raw or normalized data, transforming into arrays."""
        if self.params.nrm:
            return {
                label: {k: np.array(v) for k, v in lbg.data_nrm.items()}
                for label, lbg in self.labelblocksgroups.items()
            }
        return {
            label: {k: np.array(v) for k, v in lbg.data.items()} if lbg.data else {}
            for label, lbg in self.labelblocksgroups.items()
        }

    def _subtract_background(
        self, data: dict[int, dict[str, ArrayF]]
    ) -> dict[int, dict[str, ArrayF]]:
        """Subtract background from data."""
        return {
            label: {k: v - self.bg[label] for k, v in dd.items()}
            for label, dd in data.items()
        }

    def _adjust_negative_values(
        self, data: dict[int, dict[str, ArrayF]]
    ) -> dict[int, dict[str, ArrayF]]:
        """Adjust negative values in the data."""

        def _adjust_subtracted_data(
            key: str, y: ArrayF, sd: float, label: str, alpha: float = 1 / 10
        ) -> ArrayF:
            """Adjust negative values (alpha = F_bound/F_unbound)."""
            if y.min() < alpha * 0 * y.max():
                delta = alpha * (y.max() - y.min()) - y.min()
                logger.warning(
                    f"Buffer for '{key}:{label}' was adjusted by {delta / sd:.2f} SD."
                )
                return y + float(delta)
            return y  # never used if properly called?

        for i, dd in data.items():
            sd = self.bg_err[i].mean() if self.bg_err[i].size > 0 else np.nan
            for k in self.fit_keys:
                dd[k] = _adjust_subtracted_data(k, dd[k], sd, str(i))
        return data

    @property
    def scheme(self) -> PlateScheme:
        """Scheme for known samples like {'buffer', ['H12', 'H01'], 'ctrl'...}."""
        return self._scheme

    def load_scheme(self, schemefile: Path) -> None:
        """Load scheme from file. Set buffer_wells."""
        self._scheme = PlateScheme(schemefile)
        self.buffer.wells = self._scheme.buffer

    def _generate_combinations(self) -> list[tuple[tuple[bool, ...], str]]:
        """Generate parameter combinations for export and fitting."""
        bool_iter = itertools.product([False, True], repeat=4)
        return [
            (tuple(bool_combo), method)
            for bool_combo in bool_iter
            for method in ["mean", "meansd", "fit"]
        ]

    def _apply_combination(self, combination: tuple[tuple[bool, ...], str]) -> None:
        """Apply a combination of parameters to the Titration."""
        (bg, adj, dil, nrm), method = combination
        logger.info(f"Params are: ........... {(bg, adj, dil, nrm), method}")
        self.params.bg = bg
        self.params.bg_adj = adj
        self.params.dil = dil
        self.params.nrm = nrm
        self.params.bg_mth = method

    def _prepare_output_folder(self, base_path: Path) -> Path:
        """Prepare the output folder for a given combination of parameters."""
        p = self.params
        sbg = "_bg" if p.bg else ""
        sadj = "_adj" if p.bg_adj else ""
        sdil = "_dil" if p.dil else ""
        snrm = "_nrm" if p.nrm else ""
        sfit = "_fit" if p.bg_mth == "fit" else ""
        smeansd = "_1sd" if p.bg_mth == "meansd" else ""
        subfolder_name = "dat" + sbg + sadj + sdil + snrm + sfit + smeansd
        subfolder = base_path / subfolder_name
        subfolder.mkdir(parents=True, exist_ok=True)
        return subfolder

    def _export_fit(self, subfolder: Path, config: TecanConfig) -> None:
        outfit = subfolder / "fit"
        outfit.mkdir(parents=True, exist_ok=True)
        export_list = [
            *list(self.results.values()),
            self.result_global,
            self.result_odr,
        ]
        if self.params.mcmc == "single":
            export_list.append(self.result_mcmc)
        elif self.params.mcmc == "multi":
            export_list.append(self.result_multi_mcmc)
        for i, results in enumerate(export_list):
            results.compute_all()
            fit = results.dataframe
            # CSV tables
            fit.sort_index().to_csv(outfit / Path("ffit" + str(i) + ".csv"))
            if config.png:
                results.export_pngs(outfit / f"lb{i}")
                results.export_data(outfit / f"lb{i}")
            title = config.title + f"lb:{i}"
            f = results.plot_k(xlim=config.lim, title=title)
            f.savefig(outfit / f"K{i}.png")

    def export_data_fit(self, tecan_config: TecanConfig) -> None:
        """Export dat files [x,y1,..,yN] from copy of self.data."""

        def write(
            x: ArrayF, data: dict[int, dict[str, ArrayF]], out_folder: Path
        ) -> None:
            """Write data."""
            if any(data):
                out_folder.mkdir(parents=True, exist_ok=True)
                columns = ["x"] + [f"y{i}" for i in data]
                first_label = next(iter(self.labelblocksgroups.keys()))
                for key in data[first_label]:
                    dat = np.vstack((x, [data[i][key] for i in data]))
                    datxy = pd.DataFrame(dat.T, columns=columns)
                    datxy.to_csv(out_folder / f"{key}.dat", index=False)

        if tecan_config.comb:
            saved_p = copy.copy(self.params)
            combinations = self._generate_combinations()
            for combination in combinations:
                self._apply_combination(combination)
                subfolder = self._prepare_output_folder(tecan_config.out_fp)
                write(self.x, self.data, subfolder)
                if tecan_config.fit:
                    self._export_fit(subfolder, tecan_config)
            self.params = saved_p
        else:
            subfolder = self._prepare_output_folder(tecan_config.out_fp)
            write(self.x, self.data, subfolder)
            if tecan_config.fit:
                self._export_fit(subfolder, tecan_config)

    # TODO: test cases are:
    # 1) len(w)>1 from buffer
    # 2) len(w)=1 from weight_multi_ds_titration() with/out masked da
    """
    def export_data_fit(self, tecan_config: TecanConfig) -> None:
        ""\"Export data files [x, y1, ..., yN] and optionally perform fit exports.""\"
        def write_data(
            x: ArrayF, datasets: list[dict[str, ArrayF]], output_folder: Path
        ) -> None:
            "\""Write datasets to `.dat` files in the specified output folder.""\"
            if not datasets:
                return  # No data to write
            output_folder.mkdir(parents=True, exist_ok=True)
            columns = ["x"] + [f"y{i}" for i in range(1, len(datasets) + 1)]
            for key in datasets[0]:
                # Stack x and y datasets vertically and save as DataFrame
                data_matrix = np.vstack((x, [dataset[key] for dataset in datasets]))
                data_df = pd.DataFrame(data_matrix.T, columns=columns)
                data_df.to_csv(output_folder / f"{key}.dat", index=False)
        def process_combination(combination: dict, base_output_path: Path) -> None:
            ""\"Apply a combination and export its data and fit results.""\"
            self._apply_combination(combination)
            output_folder = self._prepare_output_folder(base_output_path)
            write_data(self.x, [dataset for dataset in self.data if dataset], output_folder)
            if tecan_config.fit:
                self._export_fit(output_folder, tecan_config)
        # Export combinations or default data
        if tecan_config.comb:
            saved_params = copy.copy(self.params)  # Save current parameters
            combinations = self._generate_combinations()  # Generate all combinations
            for combination in combinations:
                process_combination(combination, tecan_config.out_fp)
            self.params = saved_params  # Restore saved parameters
        else:
            output_folder = self._prepare_output_folder(tecan_config.out_fp)
            write_data(self.x, [dataset for dataset in self.data if dataset], output_folder)
            if tecan_config.fit:
                self._export_fit(output_folder, tecan_config)
    """

    @cached_property
    def results(self) -> dict[int, TitrationResults]:
        """Fit results for all single titration dataset."""
        fittings = {}
        for label, dat in enumerate(self.data, start=1):
            if dat:
                fit = TitrationResults(
                    self.scheme,
                    self.fit_keys,
                    partial(self._compute_fit, label=label),
                )
                fittings[label] = fit
        return fittings

    @cached_property
    def result_global(self) -> TitrationResults:
        """Perform global fitting lazily."""
        return TitrationResults(self.scheme, self.fit_keys, self._compute_global_fit)

    @cached_property
    def result_odr(self) -> TitrationResults:
        """Perform global ODR fitting."""
        return TitrationResults(self.scheme, self.fit_keys, self._compute_odr_fit)

    def _compute_fit(self, key: str, label: int) -> FitResult:
        """Compute individual dataset fit for a single key."""
        try:
            ds = self._create_ds(key, label)
            return fit_binding_glob(ds)
        except InsufficientDataError:
            logger.warning(f"Skip fit for well {key} for Label:{label}.")
            return FitResult()

    def _compute_global_fit(self, key: str) -> FitResult:
        """Compute global fit for a single key."""
        try:
            ds = self._create_global_ds(key)
            # FIXME: return fit_binding_glob_reweighted(ds, key)
            return outlier2(ds, key)
        except InsufficientDataError:
            logger.warning(f"Skipping global fit for well {key}.")
            return FitResult()

    def _create_data_array(self, key: str, label: int) -> DataArray:
        """Create a DataArray for a specific key and label."""
        y = np.array(self.data[label][key])
        alpha = 1
        beta = 1
        signal = np.maximum(1.0, alpha * y**beta)  # avoid Sqrt of negative values
        if self.bg_err:
            y_errc = np.sqrt(signal + self.bg_err[label] ** 2)
        else:
            y_errc = np.sqrt(signal)
        return DataArray(self.x, y, x_errc=self.x_err, y_errc=y_errc)

    def _create_ds(self, key: str, label: int) -> Dataset:
        """Create a dataset for the given key."""
        da = self._create_data_array(key, label)
        ds = Dataset({f"{label}": da}, is_ph=self.is_ph)
        # Apply weighting if bg_err is not provided
        if not self.bg_err:
            weight_da(da, ds.is_ph)
        return ds

    def _create_global_ds(self, key: str) -> Dataset:
        """Create a global dataset for the given key."""
        data_arrays_dict = {f"y{i}": self._create_data_array(key, i) for i in self.data}
        ds = Dataset(data_arrays_dict, is_ph=self.is_ph)
        # Apply multi-dataset weighting if bg_err is not provided
        if not self.bg_err:
            weight_multi_ds_titration(ds)
        return ds

    def _compute_odr_fit(self, key: str) -> FitResult:
        """Compute global ODR fit for a single key.

        if not self.result_global[key]:
            logger.warning(
                f"Global fitting results for {key} are empty. ODR fitting skipped."
            )
        """
        result_glob = self.result_global[key]
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", RuntimeWarning)  # Catch RuntimeWarnings
            try:
                result_odr = fit_binding_odr_recursive_outlier(
                    result_glob, threshold=2.5
                )
            except Exception:
                logger.exception(f"Error during ODR fitting for well '{key}'")
                result_odr = FitResult()
            # Log any warnings captured during the process
            for warn in caught_warnings:
                if issubclass(warn.category, RuntimeWarning):
                    logger.warning(f"Warning for well '{key}': {warn.message}")
        return result_odr

    @cached_property
    def result_mcmc(self) -> TitrationResults:
        """Perform global MCMC fitting."""
        # FIXME: 0.15 vs. 0.05
        n_sd = self.result_global.n_sd(par="K", expected_sd=0.15)
        logger.info(f"n_sd[Global] estimated for MCMC fitting: {n_sd:.3f}")
        return TitrationResults(
            self.scheme, self.fit_keys, partial(self._compute_mcmc_fit, n_sd=n_sd)
        )

    def _compute_mcmc_fit(self, key: str, n_sd: float) -> FitResult:
        """Compute global MCMC fit for a single key."""
        # Calculate n_sd from the previous global fitting results
        logger.info(f"Starting PyMC sampling for key: {key}")
        try:  # FIXME: Global vs. ODR
            result_pymc = fit_binding_pymc(self.result_global[key], n_sd=n_sd, n_xerr=1)
        except Exception:
            logger.exception(f"Error during MCMC sampling for key: {key}")
            result_pymc = FitResult()  # empty result
        finally:
            logger.info(f"MCMC fitting completed for well: {key}")
        return result_pymc

    @cached_property
    def result_multi_trace(self) -> tuple[az.InferenceData, pd.DataFrame]:
        """Perform global MCMC fitting and x_true."""
        n_sd = self.result_global.n_sd(par="K", expected_sd=0.15)
        logger.info(f"n_sd[Global] estimated for MCMC fitting: {n_sd:.3f}")
        results = self.result_global.results
        trace = fit_binding_pymc_multi(results, self.scheme, n_sd=n_sd)
        trace_df = typing.cast("pd.DataFrame", az.summary(trace, fmt="wide"))
        da_true = x_true_from_trace_df(trace_df)
        filenames = [tf.path.stem + tf.path.suffix for tf in self.tecanfiles]
        pd.DataFrame(
            {"filenames": filenames, "x": da_true.x, "x_err": da_true.x_err}
        ).to_csv("list_x_true.csv", index=False, header=False)
        return trace, trace_df

    @cached_property
    def result_multi_trace2(self) -> tuple[az.InferenceData, pd.DataFrame]:
        """Perform global MCMC fitting and x_true."""
        n_sd = self.result_global.n_sd(par="K", expected_sd=0.15)
        logger.info(f"n_sd[Global] estimated for MCMC fitting: {n_sd:.3f}")
        results = self.result_global.results
        trace = fit_binding_pymc_multi2(
            results, self.scheme, self.buffer.bg_err, n_sd=n_sd
        )
        trace_df = typing.cast("pd.DataFrame", az.summary(trace, fmt="wide"))
        da_true = x_true_from_trace_df(trace_df)
        filenames = [tf.path.stem + tf.path.suffix for tf in self.tecanfiles]
        pd.DataFrame(
            {"filenames": filenames, "x": da_true.x, "x_err": da_true.x_err}
        ).to_csv("list_x_true.csv", index=False, header=False)
        return trace, trace_df

    @cached_property
    def result_multi_mcmc(self) -> TitrationResults:
        """Perform global MCMC fitting and x_true."""
        return TitrationResults(
            scheme=self.scheme,
            fit_keys=self.fit_keys,
            compute_func=partial(self._compute_multi_mcmc_fit),
        )

    def _compute_multi_mcmc_fit(self, key: str) -> FitResult:
        """Compute individual dataset fit for a single key."""
        ctr = self.get_scheme_name(key, self.scheme.names)
        ds = self.result_global[key].dataset
        if ds:
            return extract_fit(key, ctr, self.result_multi_trace[1], ds)
        return FitResult()

    def get_scheme_name(self, key: str, scheme_map: dict[str, set[str]]) -> str:
        """Extract ctr name."""
        for scheme, keys in scheme_map.items():
            if key in keys:
                return scheme
        return ""

    def plot_temperature(self, title: str = "") -> figure.Figure:
        """Plot temperatures of all labelblocksgroups."""
        temperatures: dict[str | int, list[float | int | str | None]] = {}
        for label_n, lbg in self.labelblocksgroups.items():
            temperatures[label_n] = [
                lb.metadata["Temperature"].value for lb in lbg.labelblocks
            ]
        pp = PlotParameters(is_ph=self.is_ph)
        temperatures[pp.kind] = [float(x) for x in self.x.ravel().tolist()]
        data = pd.DataFrame(temperatures)
        data = data.melt(id_vars=pp.kind, var_name="Label", value_name="Temperature")
        g = sns.lineplot(
            data=data,
            x=pp.kind,
            y="Temperature",
            hue="Label",
            palette="Set1",
            alpha=0.75,
            lw=3,
        )
        sns.scatterplot(
            data=data,
            x=pp.kind,
            y="Temperature",
            hue="Label",
            palette="Set1",
            alpha=0.6,
            legend=False,
            s=150,
        )
        ave = data["Temperature"].mean()
        std = data["Temperature"].std()
        lower, upper = ave - std, ave + std
        g.set_ylim(23, 27)
        g.axhline(ave, ls="--", lw=3)
        g.axhline(lower, ls="--", c="grey")
        g.axhline(upper, ls="--", c="grey")
        # Add titles and labels
        plt.title(f"Temperature = {format_estimate(ave, std)} °C {title}", fontsize=14)
        plt.xlabel(f"{pp.kind}", fontsize=14)
        plt.ylabel("Temperature (°C)", fontsize=14)
        plt.grid(lw=0.33)
        # Add a legend
        plt.legend(title="Label")
        plt.close()
        return typing.cast("figure.Figure", g.get_figure())


@dataclass
class TecanConfig:
    """Group tecan cli options."""

    out_fp: Path
    comb: bool
    lim: tuple[float, float] | None
    title: str
    fit: bool
    png: bool
