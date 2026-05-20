"""General utility helpers for clophfit."""

from __future__ import annotations

import typing
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path


def weights_from_sigma(sigma: np.ndarray) -> np.ndarray | None:
    """Convert standard deviations to ODR weights (1 / sigma**2).

    Returns None when sigma is empty to use odrpack defaults.
    """
    if sigma.size == 0:
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = np.where(sigma > 0, 1.0 / (sigma**2), 0.0)
    weights[~np.isfinite(weights)] = 0.0
    return weights


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
            if isinstance(element, (str, int, float)) and pattern in str(element):
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
    # Use generator expression for memory efficiency
    return [[e for e in line if e] for line in lines]
