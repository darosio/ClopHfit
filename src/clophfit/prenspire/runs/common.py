"""Shared types and file bookkeeping for every EnSpire run."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    import pandas as pd


class EnspireInputError(ValueError):
    """An option the input data cannot satisfy; the CLI reports it as a usage error."""


class EnspireRunError(RuntimeError):
    """A run that cannot finish on this data; the CLI reports it as an error."""


class EnspireMessage(NamedTuple):
    """A note or warning printed in the console, in the order it happened."""

    level: str
    text: str


@dataclass(frozen=True)
class EnspireRunResult[SummaryT: NamedTuple]:
    """What one EnSpire run wrote and found.

    ``written`` lists every file produced, in the order they were written.
    """

    written: tuple[Path, ...]
    table: pd.DataFrame
    summaries: tuple[SummaryT, ...]
    messages: tuple[EnspireMessage, ...] = ()


def _snapshot(folder: Path) -> dict[Path, tuple[int, int]]:
    """Modification time and size of every file directly inside ``folder``."""
    if not folder.is_dir():
        return {}
    return {
        p: (st.st_mtime_ns, st.st_size)
        for p in folder.iterdir()
        if p.is_file()
        for st in [p.stat()]
    }


def _written_since(
    folder: Path, before: Mapping[Path, tuple[int, int]]
) -> tuple[Path, ...]:
    """Files created or rewritten in ``folder`` since ``before`` was taken."""
    after = _snapshot(folder)
    return tuple(p for p in sorted(after) if before.get(p) != after[p])
