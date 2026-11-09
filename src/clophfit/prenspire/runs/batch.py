"""Batch runs over a folder tree of EnSpire exports (``enspire-batch``)."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import pandas as pd

from clophfit.prenspire import (
    EnspireFile,
    bands as bands_module,
)

if TYPE_CHECKING:
    from pathlib import Path


from clophfit.prenspire.runs.band import run_bands
from clophfit.prenspire.runs.common import EnspireMessage, EnspireRunResult


def _data_for(note_fp: Path) -> Path | None:
    """Return the EnSpire csv a note belongs to, or None when ambiguous."""
    exact = note_fp.with_name(f"{note_fp.name.removesuffix('_note.csv')}.csv")
    if exact.exists():
        return exact
    others = [
        p for p in note_fp.parent.glob("*.csv") if not p.name.endswith("_note.csv")
    ]
    return others[0] if len(others) == 1 else None


def run_batch(
    root_dir: Path,
    out: Path,
    *,
    buffer: bool,
    screen: str | None,
    min_rho: float,
) -> EnspireRunResult[NamedTuple]:
    """Fit the band run of every note under ``root_dir``, one folder at a time.

    One row per fit lands in ``enspire_batch_K.csv`` and one per folder that
    could not be paired or fitted in ``enspire_batch_failures.csv``: a folder
    that fails is part of the result, not a silent gap.
    """
    out.mkdir(parents=True, exist_ok=True)
    every = pd.DataFrame()
    collected: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []
    written: list[Path] = []
    messages: list[EnspireMessage] = []
    for note_fp in sorted(root_dir.rglob("*_note.csv")):
        data = _data_for(note_fp)
        if data is None:
            failures.append({"note": str(note_fp), "why": "no unambiguous data csv"})
            continue
        tag = note_fp.parent.relative_to(root_dir).as_posix().replace("/", "_")
        try:
            band_run = run_bands(
                EnspireFile(data),
                note_fp,
                out / tag,
                buffer=buffer,
                screen=screen,
                min_rho=min_rho,
                wanted=bands_module.DIRECT_BANDS,
            )
        except Exception as exc:  # ruff: ignore[blind-except] - a bad folder must not stop the walk
            failures.append({
                "note": str(note_fp),
                "why": f"{type(exc).__name__}: {exc}",
            })
            continue
        messages.extend(band_run.messages)
        written.extend(band_run.written)
        table = band_run.table
        table.insert(0, "folder", note_fp.parent.relative_to(root_dir).as_posix())
        table.insert(1, "file", data.name)
        collected.append(table)
    if collected:
        every = pd.concat(collected, ignore_index=True)
        every.to_csv(out / "enspire_batch_K.csv", index=False)
        written.append(out / "enspire_batch_K.csv")
        messages.append(
            EnspireMessage(
                "info",
                f"{len(collected)} files fitted -> {out / 'enspire_batch_K.csv'}",
            )
        )
    if failures:
        pd.DataFrame(failures).to_csv(out / "enspire_batch_failures.csv", index=False)
        written.append(out / "enspire_batch_failures.csv")
        messages.append(
            EnspireMessage(
                "info",
                f"{len(failures)} not fitted -> {out / 'enspire_batch_failures.csv'}",
            )
        )
    return EnspireRunResult(
        written=tuple(written), table=every, summaries=(), messages=tuple(messages)
    )
