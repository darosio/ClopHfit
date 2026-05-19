"""Prtecan/prtecan.py."""

from __future__ import annotations

import itertools
import logging
import re
import typing
from dataclasses import InitVar, dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path


# TODO: Add tqdm progress bar
# TODO: sort before computing to have outlier output sorted
from .parsers import (
    Metadata,
    extract_metadata,
    lookup_listoflines,
    merge_md,
    read_xls,
    strip_lines,
)

# Constants for Tecan file parsing
#: Standard metadata line length for Tecan files
STD_MD_LINE_LENGTH = 2
#: Number of columns in a 96-well plate
NUM_COLS_96WELL = 12
#: Row names for 96-well plates
ROW_NAMES = tuple("ABCDEFGH")

logger = logging.getLogger(__name__)

# pH K bounds as used by _build_params_1site in fitting/core.py
_PH_K_MIN: float = 3.0
_PH_K_MAX: float = 11.0


def _extract_ctrl_cols(scheme: PlateScheme) -> list[int] | None:
    """Extract unique 1-based column numbers from ctrl well IDs.

    Parameters
    ----------
    scheme : PlateScheme
        Plate scheme containing ctrl well identifiers (e.g. ``["A01", "H12"]``).

    Returns
    -------
    list[int] | None
        Sorted column numbers, or ``None`` if no ctrl wells are defined.
    """
    if not scheme.ctrl:
        return None
    cols: set[int] = set()
    for well in scheme.ctrl:
        m = re.search(r"(\d+)$", well)
        if m:
            cols.add(int(m.group(1)))
    return sorted(cols) or None


@dataclass
class Labelblock:
    """Parse a label block.

    Parameters
    ----------
    lines :

    Raises
    ------
    ValueError
        When data do not correspond to a complete 96-well plate.
    TypeError
        When normalization parameters are not numerical.

    Notes
    -----
    Logs a warning when it replaces "OVER" with ``np.nan`` for saturated values.
    """

    lines: InitVar[list[list[str | int | float]]]
    """Lines to create this Labelblock."""
    filename: str = ""
    """Path of the corresponding Tecan file."""
    metadata: dict[str, Metadata] = field(init=False, repr=True)
    """Metadata specific for this Labelblock."""
    data: dict[str, float] = field(init=False, repr=True)
    """Plate data values as {'well_name', value}."""
    data_nrm: dict[str, float] = field(init=False, repr=True)
    """Plate data values normalized as {'well_name', value}."""
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

    @staticmethod
    def _validate_lines(lines: list[list[str | int | float]]) -> None:
        """Validate if input lines correspond to a 96-well plate."""
        first_block_line = 25
        if (
            len(lines) < first_block_line
            or len(lines[14]) == 0
            or lines[14][0] != "<>"
            or lines[23] != lines[24]
            or lines[23] != [""] * 13
        ):
            msg = "Cannot build Labelblock: not 96 wells plate format"
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
                except (ValueError, IndexError):
                    data[well] = np.nan
                    label = self.metadata.get("Label")
                    if label is not None and hasattr(label, "value"):
                        lbl = label.value
                    else:
                        lbl = "Unknown"
                    msg = f" OVER value in {lbl}: {well} of tecanfile {self.filename}"
                    logger.warning(msg)
        return data

    @staticmethod
    def _validate_96_well_format(lines: list[list[str | int | float]]) -> None:
        """Validate 96-well plate data format."""
        if len(lines) < len(ROW_NAMES):
            msg = f"Insufficient rows: expected {len(ROW_NAMES)}, got {len(lines)}"
            raise ValueError(msg)
        for i, row in enumerate(ROW_NAMES):
            if len(lines[i]) == 0:
                msg = f"Row {i} is empty"
                raise ValueError(msg)
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

    def almost_equal(self, other: Labelblock) -> bool:
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
    ValueError
        When no Labelblock is found or file format is invalid.
    """

    path: Path
    metadata: dict[str, Metadata] = field(init=False, repr=False)
    """General metadata for Tecanfile, like `Date` and `Shaking Duration`."""
    labelblocks: dict[int, Labelblock] = field(
        init=False, repr=False, default_factory=dict
    )
    """All labelblocks contained in this file."""

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
    allequal:

    Raises
    ------
    Exception
        When labelblocks are neither equal nor almost equal.
    """

    labelblocks: list[Labelblock] = field(repr=False)
    allequal: bool = False
    """True if labelblocks already tested equal."""
    metadata: dict[str, Metadata] = field(init=False, repr=True)
    """Metadata shared by all labelblocks."""

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
        elif all(labelblocks[0].almost_equal(lb) for lb in labelblocks[1:]):
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

    Notes
    -----
    Logs a warning if the Tecanfiles do not contain the same number of
    Labelblocks that can be merged in the same order. In such cases,
    fewer LabelblocksGroup may be created.
    """

    tecanfiles: list[Tecanfile]
    labelblocksgroups: dict[int, LabelblocksGroup] = field(
        init=False, default_factory=dict
    )
    """Each group contains its own data like a titration."""
    metadata: dict[str, Metadata] = field(init=False, repr=True)
    """Metadata shared by all tecanfiles."""

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
                gr = LabelblocksGroup([
                    tf.labelblocks[idx[i]] for i, tf in enumerate(self.tecanfiles)
                ])
            except ValueError:
                continue
            else:
                self.labelblocksgroups[idx[0]] = gr
        if not self.labelblocksgroups:
            msg = f"No common labelblocks in files: {[tf.path.name for tf in self.tecanfiles]}."
            raise ValueError(msg)
        logger.warning(
            "Different LabelblocksGroup across files: %s.",
            [str(tf.path) for tf in self.tecanfiles],
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
        self.__dict__.pop("nofit_keys", None)

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
        self.__dict__.pop("nofit_keys", None)

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
                if k not in {"buffer", "discard"}
            }
