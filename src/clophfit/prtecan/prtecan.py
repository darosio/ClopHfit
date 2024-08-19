"""Prtecan/prtecan.py."""

from __future__ import annotations

import copy
import hashlib
import itertools
import logging
import pprint
import typing
from contextlib import suppress
from dataclasses import InitVar, dataclass, field
from pathlib import Path

import lmfit  # type: ignore[import-untyped]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  # type: ignore[import-untyped]
from matplotlib import figure
from matplotlib.backends.backend_pdf import PdfPages
from scipy.odr import ODR, Model, RealData  # type: ignore[import-untyped]
from uncertainties import ufloat  # type: ignore[import-untyped]

from clophfit.binding.fitting import (
    Dataset,
    FitResult,
    InsufficientDataError,
    fit_binding_glob,
    format_estimate,
)
from clophfit.binding.plotting import PlotParameters

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from clophfit.types import ArrayF

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# list_of_lines
# after set([type(x) for l in csvl for x in l]) = float | int | str
STD_MD_LINE_LENGTH = 2
NUM_COLS_96WELL = 12
ROW_NAMES = tuple("ABCDEFGH")


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
    sheet = pd.read_excel(path)
    n0 = pd.DataFrame([[np.nan] * len(sheet.columns)], columns=sheet.columns)
    sheet = pd.concat([n0, sheet], ignore_index=True)
    sheet = sheet.fillna("")
    return list(sheet.to_numpy().tolist())


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
        """Extract data into a dictionary.

        {'A01' : value}
        :
        {'H12' : value}

        Parameters
        ----------
        lines : list[list[str | int | float]]
            xls file read into lines (list_of_lines).

        Returns
        -------
        dict[str, float]
            Data from a label block.

        Warns
        -----
            When a cell contains saturated signal (converted into np.nan).
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
    labelblocks: list[Labelblock] = field(init=False, repr=False)

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
        labelblocks = [
            Labelblock(csvl[idxs[i] : idxs[i + 1]], str(self.path))
            for i in range(n_labelblocks)
        ]
        self.labelblocks = labelblocks
        if self._has_repeated_labelblocks():
            logger.warning("Repeated labelblocks")

    def _has_repeated_labelblocks(self) -> bool:
        """Check for repeated labelblocks."""
        n_labelblocks = len(self.labelblocks)
        return any(
            self.labelblocks[i] == self.labelblocks[j]
            for i, j in itertools.combinations(range(n_labelblocks), 2)
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
    _data: dict[str, list[float]] = field(init=False, default_factory=dict)
    _data_nrm: dict[str, list[float]] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        """Create common metadata and data."""
        labelblocks = self.labelblocks
        allequal = self.allequal
        if not allequal:
            allequal = all(labelblocks[0] == lb for lb in labelblocks[1:])
        if allequal:
            for key in labelblocks[0].data:
                self._data[key] = [lb.data[key] for lb in labelblocks]
        # labelblocks that can be merged only after normalization
        elif all(labelblocks[0].__almost_eq__(lb) for lb in labelblocks[1:]):
            for key in labelblocks[0].data:
                self._data_nrm[key] = [lb.data_nrm[key] for lb in labelblocks]
        else:
            msg = "Creation of labelblock group failed."
            raise ValueError(msg)
        self.labelblocks = labelblocks
        self.metadata = merge_md([lb.metadata for lb in labelblocks])

    @property
    def data(self) -> dict[str, list[float]]:
        """Return None or data."""
        return self._data

    @property
    def data_nrm(self) -> dict[str, list[float]]:
        """Normalize data by number of flashes, integration time and gain."""
        if not self._data_nrm:
            for key in self.labelblocks[0].data:
                self._data_nrm[key] = [lb.data_nrm[key] for lb in self.labelblocks]
        return self._data_nrm


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
    labelblocksgroups: list[LabelblocksGroup] = field(init=False, default_factory=list)
    #: Metadata shared by all tecanfiles.
    metadata: dict[str, Metadata] = field(init=False, repr=True)
    #: Number of merged Labelblocks groups.
    n_labels: int = field(init=False, repr=True)

    def __post_init__(self) -> None:
        """Initialize metadata and labelblocksgroups."""
        if self._all_labelblocks_equal():
            self._create_equal_groups()
        else:
            self._create_almostequal_groups()
        self.metadata = merge_md([tf.metadata for tf in self.tecanfiles])
        self.n_labels = len(self.labelblocksgroups)

    def _all_labelblocks_equal(self) -> bool:
        """Check if all Tecanfiles have the same Labelblocks."""
        tf0 = self.tecanfiles[0]
        return all(tf0.labelblocks == tf.labelblocks for tf in self.tecanfiles[1:])

    def _create_equal_groups(self) -> None:
        """Create LabelblocksGroup when all Labelblocks are equal."""
        tf0 = self.tecanfiles[0]
        self.labelblocksgroups = [
            LabelblocksGroup(
                [tf.labelblocks[i] for tf in self.tecanfiles], allequal=True
            )
            for i in range(len(tf0.labelblocks))
        ]

    def _create_almostequal_groups(self) -> None:
        """Create as many LabelblocksGroups as possible when not all Labelblocks are equal."""
        n_labelblocks = [len(tf.labelblocks) for tf in self.tecanfiles]
        rngs = tuple(range(n) for n in n_labelblocks)
        for idx in itertools.product(*rngs):
            try:
                gr = LabelblocksGroup(
                    [tf.labelblocks[idx[i]] for i, tf in enumerate(self.tecanfiles)]
                )
            except ValueError:
                continue
            else:
                self.labelblocksgroups.append(gr)
        if not self.labelblocksgroups:
            msg = f"No common labelblocks in files: {[tf.path.name for tf in self.tecanfiles]}."
            raise ValueError(msg)
        logger.warning(
            f"Different LabelblocksGroup across files: {[tf.path for tf in self.tecanfiles]}."
        )


@dataclass
class PlateScheme:
    """Define buffer, ctrl and unk wells, and ctrl names.

    Parameters
    ----------
    file: Path
        File path to the scheme file [<well Id, sample name>].
    """

    file: Path | None = None
    _buffer: list[str] = field(default_factory=list, init=False)
    _ctrl: list[str] = field(default_factory=list, init=False)
    _names: dict[str, set[str]] = field(default_factory=dict, init=False)

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
            self.buffer = list(scheme["buffer"])
            self.ctrl = list(
                {well for sample in scheme.tolist() for well in sample}
                - set(self.buffer)
            )
            self.names = {str(k): set(v) for k, v in scheme.items() if k != "buffer"}


@dataclass
class TitrationConfig:
    """Parameters defining the fitting data with callback support."""

    bg: bool = True
    bg_adj: bool = False
    dil: bool = True
    nrm: bool = True
    bg_mth: str = "mean"

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
        """Override attribute setting to trigger callback on specific changes."""
        current_value = getattr(self, name)
        if current_value != value:
            super().__setattr__(name, value)
            self._trigger_callback()


@dataclass
class Buffer:
    """Buffer handling for a titration."""

    tit: Titration
    _wells: list[str] = field(default_factory=list)

    _dataframes: list[pd.DataFrame] = field(init=False, default_factory=list)
    _dataframes_nrm: list[pd.DataFrame] = field(init=False, default_factory=list)
    _bg: list[ArrayF] = field(init=False, default_factory=list)
    _bg_sd: list[ArrayF] = field(init=False, default_factory=list)

    @property
    def wells(self) -> list[str]:
        """List of buffer wells."""
        return self._wells

    @wells.setter
    def wells(self, wells: list[str]) -> None:
        """Set the list of buffer wells and trigger recomputation."""
        self._wells = wells
        self._reset_buffer()
        self.tit.update_fit_keys(wells)

    def _reset_buffer(self) -> None:
        """Reset buffer data."""
        self._dataframes = []
        self._dataframes_nrm = []
        self._bg = []
        self._bg_sd = []

    @property
    def dataframes(self) -> list[pd.DataFrame]:
        """Buffer dataframes with fit."""
        if not self._dataframes and self.wells:
            self._dataframes = [
                pd.DataFrame(
                    {k: lbg.data[k] for k in self.wells if lbg.data and k in lbg.data}
                )
                for lbg in self.tit.labelblocksgroups
            ]
            self._fit_buffer(self._dataframes)  # fit
        return self._dataframes

    @property
    def dataframes_nrm(self) -> list[pd.DataFrame]:
        """Buffer normalized dataframes with fit."""
        if not self._dataframes_nrm and self.wells:
            self._dataframes_nrm = [
                pd.DataFrame({k: lbg.data_nrm[k] for k in self.wells})
                for lbg in self.tit.labelblocksgroups
            ]
            self._fit_buffer(self._dataframes_nrm)  # fit
        return self._dataframes_nrm

    @property
    def bg(self) -> list[ArrayF]:
        """List of buffer values."""
        if not self._bg:
            self._bg, self._bg_sd = self._compute_bg_and_sd()
        return self._bg

    @bg.setter
    def bg(self, value: list[ArrayF]) -> None:
        """Set the buffer values and reset SEM."""
        self._bg = value
        self._bg_sd = []

    @property
    def bg_sd(self) -> list[ArrayF]:
        """List of buffer SEM values."""
        if not self._bg_sd:
            self._bg, self._bg_sd = self._compute_bg_and_sd()
        return self._bg_sd

    def _compute_bg_and_sd(self) -> tuple[list[ArrayF], list[ArrayF]]:
        """Compute and return buffer values and their SEM."""
        buffers = self.dataframes_nrm if self.tit.params.nrm else self.dataframes
        if self.tit.params.bg_mth == "fit":
            bg = [
                bdf["fit"].to_numpy() if not bdf.empty else np.array([])
                for bdf in buffers
            ]
            bg_sd = [
                bdf["fit_err"].to_numpy() if not bdf.empty else np.array([])
                for bdf in buffers
            ]
        elif self.tit.params.bg_mth == "mean":
            bg = [
                bdf["mean"].to_numpy() if not bdf.empty else np.array([])
                for bdf in buffers
            ]
            bg_sd = [
                bdf["sem"].to_numpy() if not bdf.empty else np.array([])
                for bdf in buffers
            ]
        else:
            msg = f"Unknown bg_method: {self.tit.params.bg_mth}"
            raise ValueError(msg)
        return bg, bg_sd

    def _fit_buffer(self, dfs: list[pd.DataFrame]) -> None:
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

        for lbl_n, buf_df in enumerate(dfs, start=1):
            if not buf_df.empty:
                mean = buf_df.mean(axis=1).to_numpy()
                sem = buf_df.sem(axis=1).to_numpy()
                data = RealData(self.tit.conc, mean, sy=sem)
                model = Model(linear_model)
                # Initial guess for slope and intercept
                odr = ODR(data, model, beta0=[0.0, mean.mean()])
                output = odr.run()
                # Extract the best-fit parameters and their standard errors
                m_best, b_best = output.beta
                m_err, b_err = output.sd_beta
                cov_matrix = output.cov_beta
                print(
                    f"Best fit: m = {m_best:.3f} ± {m_err:.3f}, b = {b_best:.3f} ± {b_err:.3f}"
                )
                buf_df["Label"] = lbl_n
                buf_df["fit"] = m_best * self.tit.conc + b_best
                buf_df["fit_err"] = fit_error(self.tit.conc, cov_matrix)
                buf_df["mean"] = mean
                buf_df["sem"] = sem

    def plot(self, nrm: bool = False, title: str | None = None) -> sns.FacetGrid:
        """Plot buffers of all labelblocksgroups."""
        buffer_dfs = self.dataframes_nrm if nrm else self.dataframes
        if not buffer_dfs or not self.wells:
            return sns.catplot()
        pp = PlotParameters(is_ph=self.tit.is_ph)
        melted_buffers = []
        wells_lbl = self.wells.copy()
        wells_lbl.extend(["Label"])
        for buf_df in buffer_dfs:
            if not buf_df.empty:
                buffer = buf_df[wells_lbl].copy()
                buffer[pp.kind] = self.tit.conc
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
        num_labels = np.sum([not b_df.empty for b_df in buffer_dfs])
        for label_n in range(1, num_labels + 1):
            sns.scatterplot(
                data=data[data.Label == label_n],
                y="F",
                x=pp.kind,
                hue="well",
                ax=g.axes_dict[label_n],
                legend=label_n == num_labels,
            )
            g.axes_dict[label_n].errorbar(
                x=self.tit.conc,
                y=buffer_dfs[label_n - 1]["fit"],
                yerr=buffer_dfs[label_n - 1]["fit_err"],
                xerr=0.1,
                fmt="",
                color="r",
                linewidth=2,
                capsize=6,
            )
        if title:
            plt.suptitle(title, fontsize=14, x=0.96, ha="right")
        plt.close()
        return g


@dataclass
class Titration(TecanfilesGroup):
    """Build titrations from grouped Tecanfiles and concentrations or pH values.

    Parameters
    ----------
    tecanfiles: list[Tecanfile]
        List of Tecanfiles.
    conc : ArrayF
        Concentration or pH values.
    is_ph : bool
        Indicate if x values represent pH (default is False).

    Raises
    ------
    ValueError
        For unexpected file format, e.g. header `names`.
    """

    conc: ArrayF
    is_ph: bool
    buffer: Buffer = field(init=False)

    _params: TitrationConfig = field(init=False, default_factory=TitrationConfig)
    _additions: list[float] = field(init=False, default_factory=list)
    _fit_keys: set[str] = field(init=False, default_factory=set)
    _bg: list[ArrayF] = field(init=False, default_factory=list)
    _bg_sd: list[ArrayF] = field(init=False, default_factory=list)
    _data: list[dict[str, ArrayF]] = field(init=False, default_factory=list)
    _dil_corr: ArrayF = field(init=False, default_factory=lambda: np.array([]))
    _scheme: PlateScheme = field(init=False, default_factory=PlateScheme)

    #: A list of wells containing samples that are neither buffer nor CTR samples.
    keys_unk: list[str] = field(init=False, default_factory=list)
    _results: list[dict[str, FitResult]] = field(init=False, default_factory=list)
    _result_dfs: list[pd.DataFrame] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        """Create metadata and data."""
        self.buffer = Buffer(tit=self)
        self._params.set_callback(self._reset_data_results_and_bg)
        super().__post_init__()

    def _reset_data_and_results(self) -> None:
        self._data = []
        self._results = []
        self._result_dfs = []

    def _reset_data_results_and_bg(self) -> None:
        self._reset_data_and_results()
        self.bg = []
        self._bg_sd = []

    @property
    def params(self) -> TitrationConfig:
        """Get the datafit parameters."""
        return self._params

    @params.setter
    def params(self, value: TitrationConfig) -> None:
        self._params = value
        self._reset_data_results_and_bg()

    @property
    def fit_keys(self) -> set[str]:
        """List data wells that are not currently assigned to a buffer."""
        if not self._fit_keys:
            self._fit_keys = self.labelblocksgroups[0].data_nrm.keys() - set(
                self.buffer.wells
            )
        return self._fit_keys

    def update_fit_keys(self, buffer_wells: list[str]) -> None:
        """Public method to update fit keys based on buffer wells."""
        self._fit_keys = self.labelblocksgroups[0].data_nrm.keys() - set(buffer_wells)
        self._reset_data_results_and_bg()

    @property
    def bg(self) -> list[ArrayF]:
        """List of buffer values."""
        return self.buffer.bg

    @bg.setter
    def bg(self, value: list[ArrayF]) -> None:
        self.buffer.bg = value
        self._reset_data_and_results()

    @property
    def bg_sd(self) -> list[ArrayF]:
        """List of buffer SEM values."""
        return self.buffer.bg_sd

    def __repr__(self) -> str:
        """Return a string representation of the instance."""
        return (
            f'Titration\n\tfiles=["{self.tecanfiles[0].path}", ...],\n'
            f"\tconc={list(self.conc)!r},\n"
            f"\tnumber of labels={self.n_labels},\n"
            f"\tparams={self.params!r}"
        )

    @classmethod
    def fromlistfile(cls, list_file: Path | str, is_ph: bool) -> Titration:
        """Build `Titration` from a list[.pH|.Cl] file.

        Parameters
        ----------
        list_file: Path | str
            File path to the listfile ([fpath conc]).
        is_ph : bool
            Indicate if x values represent pH.

        Returns
        -------
        Titration
        """
        tecanfiles, conc = Titration._listfile(Path(list_file))
        return cls(tecanfiles, conc, is_ph)

    @staticmethod
    def _listfile(listfile: Path) -> tuple[list[Tecanfile], ArrayF]:
        """Help construction from file.

        Parameters
        ----------
        listfile: Path
            File path to the listfile ([fpath conc]).

        Returns
        -------
        tecanfiles: list[Tecanfile]
            List of tecanfiles.
        conc: ArrayF
            Concentration array.

        Raises
        ------
        FileNotFoundError
            When cannot access `list_file`.
        ValueError
            For unexpected file format, e.g. length of filename column differs from
            length of conc values.
        """
        try:
            table = pd.read_csv(listfile, sep="\t", names=["filenames", "conc"])
        except FileNotFoundError as exc:
            msg = f"Cannot find: {listfile}"
            raise FileNotFoundError(msg) from exc
        if table["filenames"].count() != table["conc"].count():
            msg = f"Check format [filenames conc] for listfile: {listfile}"
            raise ValueError(msg)
        conc = table["conc"].to_numpy()
        tecanfiles = [Tecanfile(listfile.parent / f) for f in table["filenames"]]
        return tecanfiles, conc

    @property
    def additions(self) -> list[float] | None:
        """List of initial volume followed by additions."""
        return self._additions

    # MAYBE: Here there is not any check on the validity of additions (e.g. length).
    @additions.setter
    def additions(self, additions: list[float]) -> None:
        self._additions = additions
        self._dil_corr = dilution_correction(additions)
        self._data = []

    def load_additions(self, additions_file: Path) -> None:
        """Load additions from file."""
        additions = pd.read_csv(additions_file, names=["add"])
        self.additions = additions["add"].tolist()

    @property
    def data(self) -> list[dict[str, ArrayF]]:
        """Buffer subtracted and corrected for dilution data."""

        def _adjust_subtracted_data(
            key: str, y: ArrayF, sd: float, label: str, alpha: float = 1 / 30
        ) -> ArrayF:
            """Adjust negative values."""
            # alpha is a tolerance threshold and estimate of fluorescence ratio
            # between bound and unbound states. Values shift up so that min=alpha.
            if y.min() < alpha * y.max():
                delta = alpha * (y.max() - y.min()) - y.min()
                msg = f"Buffer for '{key}:{label}' was adjusted by {delta/sd:.2f} SD."
                logger.warning(msg)
                return y + float(delta)
            return y  # never used if properly called

        if not self._data:
            # nrm
            if self.params.nrm:
                lbs_data = [lbg.data_nrm for lbg in self.labelblocksgroups]
            else:
                lbs_data = [
                    lbg.data if lbg.data else {} for lbg in self.labelblocksgroups
                ]
            # Transform values of non-empty dict into arrays
            data = [{k: np.array(v) for k, v in dd.items()} for dd in lbs_data]

            # bg
            if self.params.bg:
                data = [
                    {k: v - bg for k, v in dd.items()}
                    for dd, bg in zip(data, self.bg, strict=True)
                ]
                if self.params.bg_adj:
                    for i in range(self.n_labels):
                        label_str = self.labelblocksgroups[i].metadata["Label"].value
                        lbl_s = str(label_str)
                        sd = self.bg_sd[i].mean()
                        for k, v in data[i].items():
                            data[i][k] = _adjust_subtracted_data(k, v, sd, lbl_s)
            # dil
            if self.params.dil and self.additions:
                data = [{k: v * self._dil_corr for k, v in dd.items()} for dd in data]
            self._data = data
        return self._data

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
            for method in ["mean", "fit"]
        ]

    def _apply_combination(self, combination: tuple[tuple[bool, ...], str]) -> None:
        """Apply a combination of parameters to the Titration."""
        (bg, adj, dil, nrm), method = combination
        print(f"Params are: ........... {(bg, adj, dil, nrm), method}")
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
        subfolder_name = "dat" + sbg + sadj + sdil + snrm + sfit
        subfolder = base_path / subfolder_name
        subfolder.mkdir(parents=True, exist_ok=True)
        return subfolder

    def _export_fit(self, subfolder: Path, config: TecanConfig) -> None:
        outfit = subfolder / "1fit"
        outfit.mkdir(parents=True, exist_ok=True)

        for i, fit in enumerate(self.result_dfs):
            if config.verbose:
                try:
                    print(fit)
                    print(config.lim)
                    meta = self.labelblocksgroups[i].metadata
                    print("-" * 79)
                    print(f"\nlabel{i:d}")
                    pprint.pprint(meta)
                except IndexError:
                    print("-" * 79)
                    print("\nGlobal on both labels")
                self.print_fitting(i)
            # CSV tables
            fit.sort_index().to_csv(outfit / Path("ffit" + str(i) + ".csv"))
            if "S1_y1" in fit.columns:
                order = ["ctrl", "K", "sK", "S0_y0", "sS0_y0", "S1_y0"]
                order.extend(["sS1_y0", "S0_y1", "sS0_y1", "S1_y1", "sS1_y1"])
                ebar_y, ebar_yerr = "S1_y1", "sS1_y1"
            else:
                order = ["ctrl", "K", "sK", "S0_default", "sS0_default"]
                order.extend(["S1_default", "sS1_default"])
                ebar_y, ebar_yerr = "S1_default", "sS1_default"
            out_df = fit.reindex(order, axis=1).sort_index()
            out_df.to_csv(outfit / f"fit{i}.csv", float_format="%.3g")
            # Plots
            plotter = TitrationPlotter(self)
            f = plotter.plot_k(i, hue_column=ebar_y, xlim=config.lim, title="title")
            f.savefig(outfit / f"K{i}.png")
            f = plotter.plot_ebar(i, ebar_y, ebar_yerr, title="title")
            f.savefig(outfit / f"ebar{i}.png")
            if config.sel and config.sel[0] and config.sel[1]:
                xmin = float(config.sel[0]) if self.is_ph else None
                xmax = float(config.sel[0]) if not self.is_ph else None
                ymin = float(config.sel[1])
                f = plotter.plot_ebar(
                    i,
                    ebar_y,
                    ebar_yerr,
                    xmin=xmin,
                    xmax=xmax,
                    ymin=ymin,
                    title=config.title,
                )
                f.savefig(outfit / f"ebar{i}_sel{xmin},{ymin}.png")
            if config.png:
                self.export_png(i, outfit)
        if config.pdf:
            # Export pdf for tentatively global result
            plotter.plot_all_wells(-1, outfit / "all_wells.pdf")

    def export_data_fit(self, tecan_config: TecanConfig) -> None:
        """Export dat files [x,y1,..,yN] from copy of self.data."""

        def write(
            conc: ArrayF, data: list[dict[str, ArrayF]], out_folder: Path
        ) -> None:
            """Write data."""
            if any(data):
                out_folder.mkdir(parents=True, exist_ok=True)
                columns = ["x"] + [f"y{i}" for i in range(1, len(data) + 1)]
                for key in data[0]:
                    dat = np.vstack((conc, [dt[key] for dt in data]))
                    datxy = pd.DataFrame(dat.T, columns=columns)
                    datxy.to_csv(out_folder / f"{key}.dat", index=False)

        if tecan_config.comb:
            saved_p = copy.copy(self.params)
            combinations = self._generate_combinations()
            for combination in combinations:
                self._apply_combination(combination)
                subfolder = self._prepare_output_folder(tecan_config.out_fp)
                write(self.conc, [dd for dd in self.data if dd], subfolder)
                if tecan_config.fit:
                    self._export_fit(subfolder, tecan_config)
            self.params = saved_p
        else:
            subfolder = self._prepare_output_folder(tecan_config.out_fp)
            write(self.conc, [dd for dd in self.data if dd], subfolder)
            if tecan_config.fit:
                self._export_fit(subfolder, tecan_config)

    @property
    def results(self) -> list[dict[str, FitResult]]:
        """Result dataframes."""
        if not self._results:
            self._results = self.fit()
        return self._results

    @property
    def result_dfs(self) -> list[pd.DataFrame]:
        """Result dataframes."""
        if not self._result_dfs:
            for fitresult in self.results:
                data = []
                for lbl, fr in fitresult.items():
                    pars = fr.result.params if fr.result else None
                    row = {"well": lbl}
                    if pars is not None:
                        for k in pars:
                            row[k] = pars[k].value
                            row[f"s{k}"] = pars[k].stderr
                    data.append(row)
                    df0 = pd.DataFrame(data).set_index("well")
                    # ctrl
                    for ctrl_name, wells in self.scheme.names.items():
                        for well in wells:
                            df0.loc[well, "ctrl"] = ctrl_name
                self._result_dfs.append(df0)
        return self._result_dfs

    def fit(self) -> list[dict[str, FitResult]]:
        """Fit titrations.

        The fitting process uses the initial point (`ini`), the final point
        (`fin`), and weighting (`weight`) parameters defined in the `FitKwargs`
        instance (accessible through `self.fitkws`).

        To perform a fit, you would first define the fit parameters and then
        call the fit method: titan.fitkws = TitrationAnalysis.FitKwargs(ini=0,
        fin=None, weight=True)

        Returns
        -------
        list[dict[str, FitResult]]
            A list of dictionaries with fitting results.

        Notes
        -----
        This method is less general and is designed for two label blocks.
        """
        x = self.conc
        fittings = []
        # Any lbg at least contains normalized data.
        self.keys_unk = list(self.fit_keys - set(self.scheme.ctrl))

        # TODO: Use sd array after proper masking
        # NEXT: Fix bg_sd when scheme is None (add a test).
        weights = [1 / sd.mean() if sd.size > 0 else sd for sd in self.bg_sd]
        print(f"weights: {weights}")
        for lbl_n, dat in enumerate(self.data, start=1):
            fitting = {}
            if dat:
                for k in self.fit_keys:
                    ds = Dataset(x, np.array(dat[k]), is_ph=self.is_ph)
                    ds.add_weights(np.array(weights[lbl_n - 1]))
                    # Alternatively weight_multi_ds_titration(ds)
                    try:
                        fitting[k] = fit_binding_glob(ds)
                    except InsufficientDataError:
                        print(f"Skip {k} for Label{lbl_n}.")
                        fitting[k] = FitResult(None, None, None)
                fittings.append(fitting)
        # Global weighted on relative residues of single fittings.
        if self.data[0] and self.data[1]:
            fitting = {}
            for k in self.fit_keys:
                y0 = np.array(self.data[0][k])
                y1 = np.array(self.data[1][k])
                ds = Dataset(x, {"y0": y0, "y1": y1}, is_ph=self.is_ph)
                # Alternatively weight_multi_ds_titration(ds)
                # NEXT: use correction for dilution imply masked weights * dil_corr
                # NEXT: list.pH with xerr
                # TODO: dilution corr must be masked where y is NaN
                # for the moment use np broadcasting from 1D array of len=1
                ds.add_weights({"y0": np.array(weights[0]), "y1": np.array(weights[1])})
                try:
                    fitting[k] = fit_binding_glob(ds)
                except InsufficientDataError:
                    print(f"Skip {k} for global fit.")
                    fitting[k] = FitResult(None, None, None)
            fittings.append(fitting)
        return fittings

    def print_fitting(self, lb: int) -> None:
        """Print fitting parameters for the whole plate."""

        def format_row(index: str, row: pd.Series[float], keys: list[str]) -> str:
            formatted_values = []
            for key in keys:
                val = ufloat(row[key], row["s" + key])
                # Format the values with 4 significant figures
                nominal = f"{val.nominal_value:.3g}"
                std_dev = f"{val.std_dev:.2g}"
                # Ensure each value occupies a fixed space
                formatted_values.extend([f"{nominal:>7s}", f"{std_dev:>7s}"])
            return f"{index:s} {' '.join(formatted_values)}"

        fit_df = self.result_dfs[lb]
        out_keys = ["K"] + [
            col
            for col in fit_df.columns
            if col not in ["ctrl", "K", "sK"] and not col.startswith("s")
        ]
        header = "    " + " ".join([f"{x:>7s} s{x:>7s}" for x in out_keys])
        if len(self.scheme.ctrl) > 0:
            res_ctrl = fit_df.loc[self.scheme.ctrl]
            groups = res_ctrl.groupby("ctrl")
            print(header)
            for name, group in groups:
                print(f" {name}")
                for i, r in group.iterrows():
                    print(format_row(str(i), r, out_keys))
        res_unk = fit_df.loc[self.keys_unk]
        # Compute 'K - 2*sK' for each row in res_unk
        res_unk["sort_val"] = res_unk["K"] - 2 * res_unk["sK"]
        # Sort the DataFrame by this computed value in descending order
        res_unk_sorted = res_unk.sort_values(by="sort_val", ascending=False)
        # # TODO: in case: del res_unk["sort_val"] # if you want to remove the
        # # temporary sorting column
        print("\n" + header)
        print("  UNK")
        # for i, r in res_unk.sort_index().iterrows():
        for i, r in res_unk_sorted.iterrows():
            print(format_row(str(i), r, out_keys))

    def plot_temperature(self, title: str = "") -> figure.Figure:
        """Plot temperatures of all labelblocksgroups."""
        temperatures: dict[str | int, list[float | int | str | None]] = {}
        for label_n, lbg in enumerate(self.labelblocksgroups, start=1):
            temperatures[label_n] = [
                lb.metadata["Temperature"].value for lb in lbg.labelblocks
            ]
        pp = PlotParameters(is_ph=self.is_ph)
        temperatures[pp.kind] = self.conc.tolist()
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
        return typing.cast(figure.Figure, g.get_figure())

    def export_png(self, lb: int, path: str | Path) -> None:
        """Export png like lb1/ lb2/ lb1_lb2/."""
        # Make sure the directory exists
        folder = Path(path) / f"lb{lb}"
        folder.mkdir(parents=True, exist_ok=True)
        for k, v in self.results[lb].items():
            if v.figure:
                v.figure.savefig(folder / f"{k}.png")


# TODO: refactor plots
# TODO: Test plots
@dataclass
class TitrationPlotter:
    """Class responsible for plotting Titration data."""

    tit: Titration

    def plot_k(
        self,
        lb: int,
        hue_column: str,
        xlim: tuple[float, float] | None = None,
        title: str = "",
    ) -> figure.Figure:
        """Plot K values as stripplot.

        Parameters
        ----------
        lb: int
            Labelblock index.
        hue_column: str
            Column in `fitresults_df` used for color-coding data points in the
            stripplot.
        xlim : tuple[float, float] | None, optional
            Range.
        title : str, optional
            To name the plot.

        Returns
        -------
        figure.Figure
            The figure.
        """
        sns.set(style="whitegrid")
        f = plt.figure(figsize=(12, 16))
        # Ctrl
        ax1 = plt.subplot2grid((8, 1), loc=(0, 0))
        if len(self.tit.scheme.ctrl) > 0:
            res_ctrl = (
                self.tit.result_dfs[lb].loc[self.tit.scheme.ctrl].sort_values("ctrl")
            )
            sns.stripplot(
                x=res_ctrl["K"],
                y=res_ctrl.index,
                size=8,
                orient="h",
                hue=res_ctrl.ctrl,
                ax=ax1,
            )
            plt.legend(loc="upper left", frameon=False)
            plt.errorbar(
                res_ctrl.K,
                range(len(res_ctrl)),
                xerr=res_ctrl["sK"],
                fmt=".",
                c="lightgray",
                lw=8,
            )
            plt.grid(True, axis="both")
        # Unk
        res_unk = (
            self.tit.result_dfs[lb].loc[self.tit.keys_unk].sort_index(ascending=False)
        )
        # Compute 'K - 2*sK' for each row in res_unk
        res_unk["sort_val"] = res_unk["K"] - 2 * res_unk["sK"]
        # Sort the DataFrame by this computed value in descending order
        res_unk = res_unk.sort_values(by="sort_val", ascending=True)
        ax2 = plt.subplot2grid((8, 1), loc=(1, 0), rowspan=7)
        sns.stripplot(
            x=res_unk["K"],
            y=res_unk.index,
            size=12,
            orient="h",
            palette="Blues",
            hue=res_unk[hue_column],
            ax=ax2,
        )
        plt.legend(loc="upper left", frameon=False)
        plt.errorbar(
            res_unk["K"],
            range(len(res_unk)),
            xerr=res_unk["sK"],
            fmt=".",
            c="gray",
            lw=2,
        )
        ytick_labels = [str(label) for label in res_unk.index]
        plt.yticks(range(len(res_unk)), ytick_labels)
        plt.ylim(-1, len(res_unk))
        plt.grid(True, axis="both")
        xlim = self._determine_xlim(res_ctrl, res_unk, xlim)
        # Configure axis and title
        ax1.set_xlim(xlim)
        ax2.set_xlim(xlim)
        ax1.set_xticklabels([])
        ax1.set_xlabel("")
        title += "  label:" + str(lb)
        f.suptitle(title, fontsize=16)
        f.tight_layout(pad=1.2, w_pad=0.1, h_pad=0.5, rect=(0, 0, 1, 0.97))
        return f

    def _determine_xlim(
        self,
        res_ctrl: pd.DataFrame | None,
        res_unk: pd.DataFrame,
        xlim: tuple[float, float] | None,
    ) -> tuple[float, float]:
        if not xlim:
            xlim = (res_unk["K"].min(), res_unk["K"].max())
            if res_ctrl is not None:
                xlim = (
                    0.99 * min(res_ctrl["K"].min(), xlim[0]),
                    1.01 * max(res_ctrl["K"].max(), xlim[1]),
                )
                xlim = (0.99 * xlim[0], 1.01 * xlim[1])
        return xlim

    def plot_all_wells(self, lb: int, path: str | Path) -> None:
        """Plot all wells into a pdf."""
        # Create a PdfPages object
        pdf_pages = PdfPages(Path(path).with_suffix(".pdf"))  # type: ignore[no-untyped-call]
        # TODO: Order.
        """
        for k in self.fitresults[0].loc[self.scheme.ctrl].sort_values("ctrl").index:
            out.savefig(self.plot_well(str(k)))
        for k in self.fitresults[0].loc[self.keys_unk].sort_index().index:
            out.savefig(self.plot_well(str(k)))
        """
        for lbl, fr in self.tit.results[lb].items():
            fig = fr.figure
            if fig is not None:
                # A4 size in inches. You can adjust this as per your need.
                fig.set_size_inches((8.27, 11.69))
                # Get the first axes in the figure to adjust the positions
                ax = fig.get_axes()[0]
                # Adjust position as needed. Values are [left, bottom, width, height]
                ax.set_position((0.1, 0.5, 0.8, 0.4))
                # Create a new axes for the text
                text_ax = fig.add_axes((0.1, 0.05, 0.8, 0.35))  # Adjust as needed
                text = lmfit.printfuncs.fit_report(self.tit.results[lb][lbl].result)
                text_ax.text(0.0, 0.0, text, fontsize=14)
                text_ax.axis("off")  # Hide the axes for the text
                # Save the figure into the PDF
                pdf_pages.savefig(fig, bbox_inches="tight")  # type: ignore[no-untyped-call]
        # Close the PdfPages object
        pdf_pages.close()  # type: ignore[no-untyped-call]
        # Close all the figures
        plt.close("all")

    def plot_ebar(  # noqa: PLR0913
        self,
        lb: int,
        y: str,
        yerr: str,
        x: str = "K",
        xerr: str = "sK",
        xmin: float | None = None,
        ymin: float | None = None,
        xmax: float | None = None,
        title: str | None = None,
    ) -> figure.Figure:
        """Plot SA vs.

        K with errorbar for the whole plate.
        """
        fit_df = self.tit.result_dfs[lb]
        with plt.style.context("fivethirtyeight"):
            f = plt.figure(figsize=(10, 10))
            if xmin:
                fit_df = fit_df[fit_df[x] > xmin]
            if xmax:
                fit_df = fit_df[fit_df[x] < xmax]
            if ymin:
                fit_df = fit_df[fit_df[y] > ymin]
            with suppress(ValueError):
                plt.errorbar(
                    fit_df[x],
                    fit_df[y],
                    xerr=fit_df[xerr],
                    yerr=fit_df[yerr],
                    fmt="o",
                    elinewidth=1,
                    markersize=10,
                    alpha=0.7,
                )
            if "ctrl" not in fit_df:
                fit_df["ctrl"] = 0
            fit_df = fit_df[~np.isnan(fit_df[x])]
            fit_df = fit_df[~np.isnan(fit_df[y])]
            for idx, xv, yv, l in zip(  # noqa: E741
                fit_df.index, fit_df[x], fit_df[y], fit_df["ctrl"], strict=False
            ):
                # x or y do not exist.# try:
                if isinstance(l, str):
                    color = "#" + hashlib.sha224(l.encode()).hexdigest()[2:8]
                    plt.text(xv, yv, l, fontsize=13, color=color)
                else:
                    plt.text(xv, yv, idx, fontsize=12)
                # x or y do not exist.# except:
                # x or y do not exist.# continue
            plt.yscale("log")
            # min(x) can be = NaN
            min_x = min(max([0.01, fit_df[x].min()]), 14)
            min_y = min(max([0.01, fit_df[y].min()]), 5000)
            plt.xlim(0.99 * min_x, 1.01 * fit_df[x].max())
            plt.ylim(0.90 * min_y, 1.10 * fit_df[y].max())
            plt.grid(True, axis="both")
            plt.ylabel(y)
            plt.xlabel(x)
            title = title if title else ""
            title += "  label:" + str(lb)
            plt.title(title, fontsize=15)
            return f


@dataclass
class TecanConfig:
    """Group tecan cli options."""

    out_fp: Path
    verbose: int
    comb: bool
    lim: tuple[float, float] | None
    sel: tuple[float, float] | None
    title: str
    fit: bool
    png: bool
    pdf: bool
