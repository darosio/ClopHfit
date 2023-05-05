"""Generic functions for parsing Plate Reader data files."""
from __future__ import annotations

import csv
from pathlib import Path


class ManyLinesFoundError(Exception):
    """Raised when there are multiple lines containing a specified search string."""

    pass


def load_csv(fname: str, dialect: str | None = None) -> list[list[str]]:
    """Read a csv file.

    Returns a list of lines from a (csv) file using encoding **iso-8859-1**.

    Parameters
    ----------
    fname : str
        File name.
    dialect : str
        The csv dialect.

    Returns
    -------
    list[list[str]]

    Examples
    --------
    >>> from prenspire.prparser import load_csv
    >>> lines = load_csv("tests/EnSpire/h148g-spettroC-nota", "excel-tab")
    >>> lines[0:2]
    [['Well', 'pH', 'Chloride'], ['A01', '5.2', '0']]

    """
    with Path(fname).open(encoding="iso-8859-1") as f:
        return list(
            csv.reader(f, dialect=dialect if dialect is not None else csv.excel)
        )


def line_index(lines: list[str], search: str) -> int:
    """Index function checking the existence of a unique match.

    It behaves like list.index() but raise an Exception if the search string
    occurs multiple times. Returns 0 otherwise.

    Parameters
    ----------
    lines : list[str]
        List of lines.
    search : str
        The search term.

    Returns
    -------
    int
        The position index of search within lines.

    Raises
    ------
    Exception
        If search term is present 2 or more times.

    """
    count: int = lines.count(search)
    if count <= 1:
        return lines.index(search)  # [].index ValueError if count == 0
    else:
        raise ManyLinesFoundError("Many " + str(search) + " lines.")
