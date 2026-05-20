"""Prtecan/prtecan.py."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from clophfit.utils import strip_lines

if TYPE_CHECKING:
    from collections.abc import Sequence

    from clophfit.clophfit_types import ArrayF

# TODO: Add tqdm progress bar
# TODO: sort before computing to have outlier output sorted

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


# MAYBE: with a filter ectract_metadata with a map


@dataclass(frozen=False)
class Metadata:
    """Represents the value of a metadata dictionary.

    Parameters
    ----------
    value : int | str | float | None
        The value for the dictionary key.
    """

    value: int | str | float | None
    unit: Sequence[str | float | int] | None = None
    """The first element represents the unit, while the following elements are only listed."""


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
    if not mds:
        return {}

    mmd = {
        k: v for k, v in mds[0].items() if all(k in md and v == md[k] for md in mds[1:])
    }

    # To account for the case 93"Optimal" and 93"Manual" in lb metadata
    def all_same_gain(mds: list[dict[str, Metadata]]) -> bool:
        return all(
            "Gain" in md and md["Gain"].value == mds[0]["Gain"].value for md in mds[1:]
        )

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

    Raises
    ------
    ValueError
        If additions list is empty or if initial volume is zero.
    """
    if len(additions) == 0:
        return np.array([])

    volumes = np.cumsum(additions)

    if volumes[0] == 0:
        msg = "Initial volume (first addition) cannot be zero"
        raise ValueError(msg)

    corrections: ArrayF = volumes / volumes[0]
    return corrections
