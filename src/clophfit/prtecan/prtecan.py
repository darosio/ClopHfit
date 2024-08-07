"""Prtecan/prtecan.py."""

from __future__ import annotations

import hashlib
import itertools
import logging
import typing
import warnings
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
    from collections.abc import Sequence

    from clophfit.types import ArrayF

# Set up logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# list_of_lines
# after set([type(x) for l in csvl for x in l]) = float | int | str
DAT = "dat"
DAT_NRM = "dat_nrm"
DAT_BG = "dat_bg"
DAT_BG_NRM = "dat_bg_nrm"
DAT_BG_DIL = "dat_bg_dil"
DAT_BG_DIL_NRM = "dat_bg_dil_nrm"
STD_MD_LINE_LENGTH = 2
NUM_ROWS_96WELL = 8


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


class BufferWellsMixin:
    """Mixin class for handling buffer wells.

    This mixin adds a property for buffer wells. It's intended to be used in
    classes that deal with collections of wells, where some of the wells are
    designated as buffer wells.
    """

    _buffer_wells: list[str] | None = None

    @property
    def buffer_wells(self) -> list[str] | None:
        """List of buffer wells."""
        return self._buffer_wells

    @buffer_wells.setter
    def buffer_wells(self, value: list[str]) -> None:
        self._buffer_wells = value
        self._on_buffer_wells_set(value)

    def _on_buffer_wells_set(self, value: list[str]) -> None:
        """Provide a hook for subclasses to add behavior when buffer_wells is set."""


@dataclass
class Labelblock(BufferWellsMixin):
    """Parse a label block.

    Parameters
    ----------
    _lines : list[list[str | int | float]]
        Lines to create this Labelblock.
    path : Path, optional
        Path to the containing file, if it exists.

    Raises
    ------
    Exception
        When data do not correspond to a complete 96-well plate.

    Warns
    -----
    Warning
        When it replaces "OVER" with ``np.nan`` for saturated values.
    """

    _lines: InitVar[list[list[str | int | float]]]
    path: Path | None = None
    #: Metadata specific for this Labelblock.
    metadata: dict[str, Metadata] = field(init=False, repr=True)
    #: The 96 data values as {'well_name', value}.
    data: dict[str, float] = field(init=False, repr=True)
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
        """Create metadata and data; initialize labelblock's properties."""
        self._buffer_wells: list[str] | None = None
        self._buffer: float | None = None
        self._buffer_norm: float | None = None
        self._buffer_sd: float | None = None
        self._buffer_norm_sd: float | None = None
        self._data_norm: dict[str, float] | None = None
        self._data_buffersubtracted: dict[str, float] | None = None
        self._data_buffersubtracted_norm: dict[str, float] | None = None
        if lines[14][0] == "<>" and lines[23] == lines[24] == [""] * 13:
            stripped = strip_lines(lines)
            stripped[14:23] = []
            self.metadata: dict[str, Metadata] = extract_metadata(stripped)
            self.data: dict[str, float] = self._extract_data(lines[15:23])
        else:
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

        Raises
        ------
        ValueError
            When something went wrong. Possibly because not 96-well.

        Warns
        -----
            When a cell contains saturated signal (converted into np.nan).
        """
        rownames = tuple("ABCDEFGH")
        data = {}
        try:
            assert len(lines) == NUM_ROWS_96WELL  # noqa: S101 # nosec B101
            for i, row in enumerate(rownames):
                assert lines[i][0] == row  # nosec B101 # noqa: S101 # e.g. "A" == "A"
                for col in range(1, 13):
                    try:
                        data[row + f"{col:0>2}"] = float(lines[i][col])
                    except ValueError:
                        data[row + f"{col:0>2}"] = np.nan
                        warnings.warn(
                            f"OVER\n Overvalue in {self.metadata['Label'].value}:"
                            f"{row}{col:0>2} of tecanfile {self.path}",
                            stacklevel=2,
                        )
        except AssertionError as exc:
            msg = "Cannot extract data in Labelblock: not 96 wells?"
            raise ValueError(msg) from exc
        return data

    def _on_buffer_wells_set(self, value: list[str]) -> None:
        """Update related attributes upon setting 'buffer_wells' in Labelblock class.

        Parameters
        ----------
        value: list[str]
            The new value of 'buffer_wells'
        """
        self._buffer = float(np.average([self.data[k] for k in value]))
        self._buffer_sd = float(np.std([self.data[k] for k in value]))
        self._buffer_norm = float(np.average([self.data_norm[k] for k in value]))
        self._buffer_norm_sd = float(np.std([self.data_norm[k] for k in value]))
        self._data_buffersubtracted = None
        self._data_buffersubtracted_norm = None

    @property
    def buffer(self) -> float | None:
        """Background value to be subtracted before dilution correction."""
        return self._buffer

    @buffer.setter
    def buffer(self, value: float) -> None:
        if self._buffer == value:
            return
        self._data_buffersubtracted = None
        self._buffer = value

    @property
    def buffer_norm(self) -> float | None:
        """Background value to be subtracted before dilution correction."""
        return self._buffer_norm

    @buffer_norm.setter
    def buffer_norm(self, value: float) -> None:
        if self._buffer_norm == value:
            return
        self._data_buffersubtracted_norm = None
        self._buffer_norm = value

    @property
    def buffer_sd(self) -> float | None:
        """Get standard deviation of buffer_wells values."""
        return self._buffer_sd

    @property
    def buffer_norm_sd(self) -> float | None:
        """Get standard deviation of normalized buffer_wells values."""
        return self._buffer_norm_sd

    @property
    def data_norm(self) -> dict[str, float]:
        """Normalize data by number of flashes, integration time and gain."""
        if self._data_norm is None:
            try:
                norm = 1000.0
                for k in Labelblock._NORM_KEYS:
                    val = self.metadata[k].value
                    if isinstance(val, float | int):
                        norm /= val
            except TypeError:
                warnings.warn(
                    "Could not normalize for non numerical Gain, "
                    "Number of Flashes or Integration time.",
                    stacklevel=2,
                )  # pragma: no cover
            self._data_norm = {k: v * norm for k, v in self.data.items()}
        return self._data_norm

    @property
    def data_buffersubtracted(self) -> dict[str, float]:
        """Buffer subtracted data."""
        if self._data_buffersubtracted is None:
            if self.buffer:
                self._data_buffersubtracted = {
                    k: v - self.buffer for k, v in self.data.items()
                }
            else:
                self._data_buffersubtracted = {}
        return self._data_buffersubtracted

    @property
    def data_buffersubtracted_norm(self) -> dict[str, float]:
        """Normalize buffer-subtracted data."""
        if self._data_buffersubtracted_norm is None:
            if self.buffer_norm:
                self._data_buffersubtracted_norm = {
                    k: v - self.buffer_norm for k, v in self.data_norm.items()
                }
            else:
                self._data_buffersubtracted_norm = {}
        return self._data_buffersubtracted_norm

    def __eq__(self, other: object) -> bool:
        """Two labelblocks are equal when metadata KEYS are identical."""
        if not isinstance(other, Labelblock):
            msg = f"Cannot compare Labelblock object with {type(other).__name__}"
            raise TypeError(msg)
        eq: bool = True
        for k in Labelblock._KEYS:
            eq &= self.metadata[k] == other.metadata[k]
        # 'Gain': [81.0, 'Manual'] = 'Gain': [81.0, 'Optimal'] They are equal
        eq &= self.metadata["Gain"].value == other.metadata["Gain"].value
        return eq

    def __almost_eq__(self, other: Labelblock) -> bool:
        """Labelblocks are almost equal if they could be merged after normalization."""
        eq: bool = True
        # Integration Time, Number of Flashes and Gain can differ.
        for k in Labelblock._KEYS[:5]:
            eq &= self.metadata[k] == other.metadata[k]
        return eq


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
        if len(idxs) == 0:
            msg = "No Labelblock found."
            raise ValueError(msg)
        self.metadata = extract_metadata(csvl[: idxs[0]])
        n_labelblocks = len(idxs)
        idxs.append(len(csvl))
        labelblocks = [
            Labelblock(csvl[idxs[i] : idxs[i + 1]], self.path)
            for i in range(n_labelblocks)
        ]
        if any(
            labelblocks[i] == labelblocks[j]
            for i, j in itertools.combinations(range(n_labelblocks), 2)
        ):
            warnings.warn("Repeated labelblocks", stacklevel=2)
        self.labelblocks = labelblocks


@dataclass
class LabelblocksGroup(BufferWellsMixin):
    """Group labelblocks with compatible metadata.

    `data_norm` always exist.

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

    def __post_init__(self) -> None:
        """Create common metadata and data."""
        labelblocks = self.labelblocks
        allequal = self.allequal
        self._buffer_wells: list[str] | None = None
        self._data: dict[str, list[float]] | None = None
        self._data_norm: dict[str, list[float]] | None = None
        self._data_buffersubtracted: dict[str, list[float]] | None = None
        self._data_buffersubtracted_norm: dict[str, list[float]] | None = None
        if not allequal:
            allequal = all(labelblocks[0] == lb for lb in labelblocks[1:])
        if allequal:
            self._data = {}
            for key in labelblocks[0].data:
                self._data[key] = [lb.data[key] for lb in labelblocks]
        # labelblocks that can be merged only after normalization
        elif all(labelblocks[0].__almost_eq__(lb) for lb in labelblocks[1:]):
            self._data_norm = {}
            for key in labelblocks[0].data:
                self._data_norm[key] = [lb.data_norm[key] for lb in labelblocks]
        else:
            msg = "Creation of labelblock group failed."
            raise ValueError(msg)
        self.labelblocks = labelblocks
        self.metadata = merge_md([lb.metadata for lb in labelblocks])

    @property
    def data(self) -> dict[str, list[float]] | None:
        """Return None or data."""
        return self._data

    @property
    def data_norm(self) -> dict[str, list[float]]:
        """Normalize data by number of flashes, integration time and gain."""
        if self._data_norm is None:
            self._data_norm = {}
            for key in self.labelblocks[0].data:
                self._data_norm[key] = [lb.data_norm[key] for lb in self.labelblocks]
        return self._data_norm

    def _on_buffer_wells_set(self, value: list[str]) -> None:
        """Update related attributes upon setting 'buffer_wells' in Labelblock class.

        Parameters
        ----------
        value: list[str]
            The new value of 'buffer_wells'
        """
        for lb in self.labelblocks:
            lb.buffer_wells = value
        self._data_buffersubtracted = None
        self._data_buffersubtracted_norm = None

    def _calculate_subtracted_data(self, norm: bool = False) -> dict[str, list[float]]:
        """Calculate buffer subtracted data avoiding negative values."""
        subtracted_data = (
            {
                key: [
                    lb.data_buffersubtracted_norm[key]
                    if norm
                    else lb.data_buffersubtracted[key]
                    for lb in self.labelblocks
                ]
                for key in self.labelblocks[0].data
            }
            if self.buffer_wells
            else {}
        )

        # Adjust negative values.
        def _new_values(
            key: str, norm: bool, labelblocks: list[Labelblock], factor: float
        ) -> list[float]:
            return [
                (
                    lb.data_norm[key]
                    - (lb.buffer_norm or 0)
                    + factor * (lb.buffer_norm_sd or 0)
                    if norm
                    else lb.data[key] - (lb.buffer or 0) + factor * (lb.buffer_sd or 0)
                )
                for lb in labelblocks
            ]

        # Adjust negative values.
        for key, y_values in subtracted_data.items():
            # tolerance threshold =  max_val / 50
            if np.min(y_values) < 0.02 * np.max(y_values):
                label_str = self.metadata["Label"].value
                msg = f"Buffer for '{key}:{label_str}' was adjusted."
                factor = 1.0
                new_values = _new_values(key, norm, self.labelblocks, factor)
                logger.warning(msg)
                while np.min(new_values) < 0:
                    factor *= 1.1  # Increase the factor
                    new_values = _new_values(key, norm, self.labelblocks, factor)
                    logger.warning(
                        msg.rstrip(".") + f" with factor increased to {factor:.2f}."
                    )
                subtracted_data[key] = new_values
        return subtracted_data

    @property
    def data_buffersubtracted(self) -> dict[str, list[float]] | None:
        """Buffer subtracted data."""
        if self.data is None or self._data_buffersubtracted is not None:
            return self._data_buffersubtracted
        self._data_buffersubtracted = self._calculate_subtracted_data()
        return self._data_buffersubtracted

    @property
    def data_buffersubtracted_norm(self) -> dict[str, list[float]]:
        """Buffer subtracted normalized data."""
        if self._data_buffersubtracted_norm is None:
            self._data_buffersubtracted_norm = self._calculate_subtracted_data(
                norm=True
            )
        return self._data_buffersubtracted_norm


@dataclass
class TecanfilesGroup:
    """Group of Tecanfiles containing at least one common Labelblock.

    Parameters
    ----------
    tecanfiles: list[Tecanfile]
        List of Tecanfiles.

    Raises
    ------
    Exception
        When all Labelblocks are not at least almost equal.

    Warns
    -----
    Warning
        The Tecanfiles listed in `tecanfiles` are expected to contain the
        "same" list (of length N) of Labelblocks. Normally, N labelblocksgroups
        will be created. However, if not all Tecanfiles contain the same number
        of Labelblocks that can be merged ('equal' mergeable) in the same order,
        then a warning will be raised. In this case, a number M < N of groups
        can be built.
    """

    tecanfiles: list[Tecanfile]

    #: Each group contains its own data like a titration. ??
    labelblocksgroups: list[LabelblocksGroup] = field(init=False, default_factory=list)
    #: Metadata shared by all tecanfiles.
    metadata: dict[str, Metadata] = field(init=False, repr=True)

    def __post_init__(self) -> None:
        """Create metadata and labelblocksgroups."""
        n_labelblocks = [len(tf.labelblocks) for tf in self.tecanfiles]
        tf0 = self.tecanfiles[0]
        if all(tf0.labelblocks == tf.labelblocks for tf in self.tecanfiles[1:]):
            # Same number and order of labelblocks
            for i, _lb in enumerate(tf0.labelblocks):
                self.labelblocksgroups.append(
                    LabelblocksGroup(
                        [tf.labelblocks[i] for tf in self.tecanfiles], allequal=True
                    )
                )
        else:
            # Create as many as possible groups of labelblocks
            rngs = tuple(range(n) for n in n_labelblocks)
            for idx in itertools.product(*rngs):
                try:
                    gr = LabelblocksGroup(
                        [tf.labelblocks[idx[i]] for i, tf in enumerate(self.tecanfiles)]
                    )
                except ValueError:
                    continue
                # if labelblocks are all 'equal'
                else:
                    self.labelblocksgroups.append(gr)
            files = [tf.path for tf in self.tecanfiles]
            if len(self.labelblocksgroups) == 0:  # == []
                msg = f"No common labelblock in filenames: {files}."
                raise ValueError(msg)
            warnings.warn(
                f"Different LabelblocksGroup among filenames: {files}.", stacklevel=2
            )
        self.metadata = merge_md([tf.metadata for tf in self.tecanfiles])


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
class Titration(TecanfilesGroup, BufferWellsMixin):
    """Build titrations from grouped Tecanfiles and concentrations or pH values.

    Parameters
    ----------
    tecanfiles : list[Tecanfile]
        Tecanfiles to be grouped.
    conc : ArrayF
        Concentration or pH values.
    is_ph : bool
        Indicate if x values represent pH (default is False).
    """

    tecanfiles: list[Tecanfile]
    conc: ArrayF
    is_ph: bool

    _additions: list[float] = field(init=False, default_factory=list)
    _buffer_wells: list[str] = field(init=False, default_factory=list)
    _data: list[dict[str, list[float]] | None] = field(init=False, default_factory=list)
    _data_nrm: list[dict[str, list[float]]] = field(init=False, default_factory=list)
    _dil_corr: ArrayF | None = field(init=False, default=None)
    _scheme: PlateScheme = field(init=False, default_factory=PlateScheme)

    def __post_init__(self) -> None:  # pylint: disable=W0246
        """Set up the initial values for the properties."""
        super().__post_init__()

    def __repr__(self) -> str:
        """Return a string representation of the instance."""
        return (
            f'Titration(files=["{self.tecanfiles[0].path}", '
            f'"{self.tecanfiles[1].path}", ...], '
            f"conc={self.conc!r}, data_size={len(self.data) if self.data else 0})"
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
        self._data_nrm = []

    def load_additions(self, additions_file: Path) -> None:
        """Load additions from file."""
        additions = pd.read_csv(additions_file, names=["add"])
        self.additions = additions["add"].tolist()

    def _on_buffer_wells_set(self, value: list[str]) -> None:
        """Update related attributes upon setting 'buffer_wells' in Labelblock class.

        Parameters
        ----------
        value: list[str]
            The new value of 'buffer_wells'
        """
        for lbg in self.labelblocksgroups:
            lbg.buffer_wells = value
        self._data = []
        self._data_nrm = []

    @property
    def data(self) -> list[dict[str, list[float]] | None] | None:
        """Buffer subtracted and corrected for dilution data."""
        if not self._data and self.additions:
            self._data = [
                (
                    {
                        k: (np.array(v) * self._dil_corr).tolist()
                        for k, v in lbg.data_buffersubtracted.items()
                    }
                    if lbg.data_buffersubtracted
                    else None
                )
                for lbg in self.labelblocksgroups
            ]
        return self._data

    @property
    def data_nrm(self) -> list[dict[str, list[float]]] | None:
        """Buffer subtracted, corrected for dilution and normalized data."""
        if not self._data_nrm and self.additions:
            self._data_nrm = [
                {
                    k: (np.array(v) * self._dil_corr).tolist()
                    for k, v in lbg.data_buffersubtracted_norm.items()
                }
                for lbg in self.labelblocksgroups
            ]
        return self._data_nrm

    @property
    def scheme(self) -> PlateScheme:
        """Scheme for known samples like {'buffer', ['H12', 'H01'], 'ctrl'...}."""
        return self._scheme

    def load_scheme(self, schemefile: Path) -> None:
        """Load scheme from file. Set buffer_wells."""
        self._scheme = PlateScheme(schemefile)
        self.buffer_wells = self._scheme.buffer

    def export_data(self, out_folder: Path) -> None:
        """Export dat files [x,y1,..,yN] from labelblocksgroups.

        Remember that a Titration has at least 1 normalized Lbg dataset `dat_nrm`.

        dat:            [d1, None] | [d1, d2]
        dat_bg:         [{}, None] | [d1, None] | [{}, {}] | [d1, d2]
        dat_bg_dil:     [{}, None] | [d1, None] | [{}, {}] | [d1, d2]
        dat_nrm:        [d1,d2]
        dat_bg_nrm:     [{}, {}] | [d1, d2]
        dat_bg_dil_nrm: [{}, {}] | [d1, d2]

        Parameters
        ----------
        out_folder : Path
            Path to output folder.
        """
        out_folder.mkdir(parents=True, exist_ok=True)

        def write(
            conc: ArrayF, data: list[dict[str, list[float]]], out_folder: Path
        ) -> None:
            """Write data."""
            if any(data):
                out_folder.mkdir(parents=True, exist_ok=True)
                columns = ["x"] + [f"y{i}" for i in range(1, len(data) + 1)]
                for key in data[0]:
                    dat = np.vstack((conc, [dt[key] for dt in data]))
                    datxy = pd.DataFrame(dat.T, columns=columns)
                    datxy.to_csv(
                        out_folder / Path(key).with_suffix(".dat"), index=False
                    )

        write(
            self.conc,
            [lbg.data for lbg in self.labelblocksgroups if lbg.data],
            out_folder / DAT,
        )
        write(
            self.conc,
            [lbg.data_norm for lbg in self.labelblocksgroups],
            out_folder / DAT_NRM,
        )
        write(
            self.conc,
            [
                lbg.data_buffersubtracted
                for lbg in self.labelblocksgroups
                if lbg.data_buffersubtracted
            ],
            out_folder / DAT_BG,
        )
        write(
            self.conc,
            [lbg.data_buffersubtracted_norm for lbg in self.labelblocksgroups],
            out_folder / DAT_BG_NRM,
        )
        if self.data:
            write(
                self.conc,
                [e for e in self.data if e],
                out_folder / DAT_BG_DIL,
            )
        if self.data_nrm:
            write(
                self.conc,
                self.data_nrm,
                out_folder / DAT_BG_DIL_NRM,
            )


@dataclass
class FitdataParams:
    """Parameters defining the fitting data."""

    nrm: bool = True
    bg: bool = True
    dil: bool = True


# TODO: Substitute warning with logging
@dataclass
class TitrationAnalysis(Titration):
    """Perform analysis of a titration.

    Raises
    ------
    ValueError
        For unexpected file format, e.g. header `names`.
    """

    #: A list of wells containing samples that are neither buffer nor CTR samples.
    keys_unk: list[str] = field(init=False, default_factory=list)
    _fitdata: Sequence[dict[str, list[float]] | None] = field(
        init=False, default_factory=list
    )
    _fitdata_params: FitdataParams = field(init=False, default_factory=FitdataParams)
    _results: list[dict[str, FitResult]] = field(init=False, default_factory=list)
    _result_dfs: list[pd.DataFrame] = field(init=False, default_factory=list)
    _buffers: list[pd.DataFrame] = field(init=False, default_factory=list)
    _buffers_norm: list[pd.DataFrame] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:  # pylint: disable=W0246
        """Set up the initial values of inherited class properties."""
        super().__post_init__()

    def __repr__(self) -> str:
        """Return a string representation of the instance."""
        return (
            f"TitrationAnalysis({super().__repr__()!r}\n"
            f"   (preprocess)    fitdata_params={self.fitdata_params!r}\n"
        )

    @classmethod
    def fromlistfile(cls, list_file: Path | str, is_ph: bool) -> TitrationAnalysis:
        """Build `TitrationAnalysis` from a list[.pH|.Cl] file."""
        titration = super().fromlistfile(list_file, is_ph)
        return cls(titration.tecanfiles, titration.conc, titration.is_ph)

    @property
    def fitdata(self) -> Sequence[dict[str, list[float]] | None]:
        """Data used for fitting."""
        if not self._fitdata:
            self._results = []
            if self.fitdata_params.dil:
                # maybe need also bool(any([{}, {}])) or np.sum([bool(e) for e
                # in [{}, {}]]) i.e. DDD if nrm and self.data_nrm and
                # any(self.data_nrm):
                if self.fitdata_params.nrm and self.data_nrm:
                    self._fitdata = self.data_nrm
                elif self.data:
                    self._fitdata = self.data
                else:  # back up to dat_nrm
                    warnings.warn(
                        "No dilution corrected data found; use normalized data.",
                        stacklevel=2,
                    )
                    self._fitdata = [lbg.data_norm for lbg in self.labelblocksgroups]
            elif self.fitdata_params.bg:
                if self.fitdata_params.nrm:
                    self._fitdata = [
                        lbg.data_buffersubtracted_norm for lbg in self.labelblocksgroups
                    ]
                else:
                    self._fitdata = [
                        lbg.data_buffersubtracted for lbg in self.labelblocksgroups
                    ]
            elif self.fitdata_params.nrm:
                self._fitdata = [lbg.data_norm for lbg in self.labelblocksgroups]
            else:
                self._fitdata = [lbg.data for lbg in self.labelblocksgroups]
        return self._fitdata

    @property
    def fitdata_params(self) -> FitdataParams:
        """Get the datafit parameters."""
        return self._fitdata_params

    @fitdata_params.setter
    def fitdata_params(self, fitdata_params: FitdataParams) -> None:
        """Set the datafit parameters."""
        self._fitdata_params = fitdata_params
        self._fitdata = []
        self._results = []
        self._result_dfs = []

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

    @property
    def buffers(self) -> list[pd.DataFrame]:
        """Buffer dataframes with fit."""
        if not self._buffers and self.buffer_wells:
            self._buffers = [
                pd.DataFrame(
                    {
                        k: lbg.data[k]
                        for k in self.buffer_wells
                        if isinstance(lbg.data, dict)
                        and k in lbg.data
                        and lbg.data[k] is not None
                    }
                )
                for lbg in self.labelblocksgroups
            ]
            self._fit_buffer(self._buffers)  # fit
        return self._buffers

    @property
    def buffers_norm(self) -> list[pd.DataFrame]:
        """Buffer normalized dataframes with fit."""
        if not self._buffers_norm and self.buffer_wells:
            self._buffers_norm = [
                pd.DataFrame({k: lbg.data_norm[k] for k in self.buffer_wells})
                for lbg in self.labelblocksgroups
            ]
            self._fit_buffer(self._buffers_norm)  # fit
        return self._buffers_norm

    def _fit_buffer(self, buffer_dfs: list[pd.DataFrame]) -> None:
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

        for lbl_n, buf_df in enumerate(buffer_dfs, start=1):
            if not buf_df.empty:
                mean = buf_df.mean(axis=1).to_numpy()
                sem = buf_df.sem(axis=1).to_numpy()
                data = RealData(self.conc, mean, sy=sem)
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
                buf_df["fit"] = m_best * self.conc + b_best
                buf_df["fit_err"] = fit_error(self.conc, cov_matrix)

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
        x = np.array(self.conc)  # # TODO: remove array here
        fittings = []
        # Any Lbg at least contains normalized data.
        keys_fit = self.labelblocksgroups[0].data_norm.keys() - set(self.scheme.buffer)
        self.keys_unk = list(keys_fit - set(self.scheme.ctrl))

        buffer_dfs = self.buffers_norm if self.fitdata_params.nrm else self.buffers
        weights = [1 / np.array(buf_df["fit_err"].mean()) for buf_df in buffer_dfs]
        print(weights)
        for lbl_n, dat in enumerate(self.fitdata, start=1):
            fitting = {}
            if dat:
                for k in keys_fit:
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
        if self.fitdata[0] and self.fitdata[1]:
            fitting = {}
            for k in keys_fit:
                y0 = np.array(self.fitdata[0][k])
                y1 = np.array(self.fitdata[1][k])
                ds = Dataset(x, {"y0": y0, "y1": y1}, is_ph=self.is_ph)
                # Alternatively weight_multi_ds_titration(ds)
                # NEXT: use correction for dilution imply masked weights
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

    def plot_k(
        self,
        lb: int,
        hue_column: str,
        xlim: tuple[float, float] | None = None,
        title: str | None = None,
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
        title : str | None, optional
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
        if len(self.scheme.ctrl) > 0:
            res_ctrl = self.result_dfs[lb].loc[self.scheme.ctrl].sort_values("ctrl")
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
        res_unk = self.result_dfs[lb].loc[self.keys_unk].sort_index(ascending=False)
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
        if not xlim:
            xlim = (res_unk["K"].min(), res_unk["K"].max())
            if len(self.scheme.ctrl) > 0:
                xlim = (
                    0.99 * min(res_ctrl["K"].min(), xlim[0]),
                    1.01 * max(res_ctrl["K"].max(), xlim[1]),
                )
            xlim = (0.99 * xlim[0], 1.01 * xlim[1])
        ax1.set_xlim(xlim)
        ax2.set_xlim(xlim)
        ax1.set_xticklabels([])
        ax1.set_xlabel("")
        title = title if title else ""
        title += "  label:" + str(lb)
        f.suptitle(title, fontsize=16)
        f.tight_layout(pad=1.2, w_pad=0.1, h_pad=0.5, rect=(0, 0, 1, 0.97))
        return f

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
        for lbl, fr in self.results[lb].items():
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
                text = lmfit.printfuncs.fit_report(self.results[lb][lbl].result)
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
        fit_df = self.result_dfs[lb]
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
        print(type(pp.kind))
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

    # NEXT: use fitted buffer values datafit
    def plot_buffer(self, nrm: bool = False, title: str | None = None) -> sns.FacetGrid:
        """Plot buffers of all labelblocksgroups."""
        buffer_dfs = self.buffers_norm if nrm else self.buffers
        if not buffer_dfs or not self.buffer_wells:
            return sns.catplot()
        pp = PlotParameters(is_ph=self.is_ph)
        melted_buffers = []
        wells_lbl = self.buffer_wells.copy()
        wells_lbl.extend(["Label"])
        for buf_df in buffer_dfs:
            if not buf_df.empty:
                buffer = buf_df[wells_lbl].copy()
                buffer[pp.kind] = self.conc
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
                x=self.conc,
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

    def export_png(self, lb: int, path: str | Path) -> None:
        """Export png like lb1/ lb2/ lb1_lb2/."""
        # Make sure the directory exists
        folder = Path(path) / f"lb{lb}"
        folder.mkdir(parents=True, exist_ok=True)
        for k, v in self.results[lb].items():
            if v.figure:
                v.figure.savefig(folder / f"{k}.png")
