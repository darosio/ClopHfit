"""Reading an EnSpire export and its note, joining split titrations, and the isosbestic table."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from clophfit.prenspire import (
    EnspireFile,
    Note,
    bands as bands_module,
    merge_enspire_files,
)
from clophfit.prenspire.spectral import (
    isosbestic_by_group,
    plate_isosbestic,
    sample_spectra,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


from clophfit.prenspire.runs.common import EnspireMessage


@dataclass(frozen=True)
class EnspireInputs:
    """The export to fit and the note to fit it with, after any ``--join``.

    ``note`` is None when no note was given, in which case nothing is fitted.
    """

    ef: EnspireFile
    note: Path | None
    messages: tuple[EnspireMessage, ...] = ()


def prepare_inputs(  # ruff: ignore[too-many-arguments]
    csv_f: Path,
    note_f: Path | None,
    joins: Sequence[tuple[Path, Path]],
    out: Path,
    *,
    export: str,
    verbose: int,
) -> EnspireInputs:
    """Write the measurement tables and join the notes of a split titration.

    Each ``joins`` pair is another export and its note, for a titration split
    over several EnSpire files; the exports must carry disjoint labels. The
    joined note is a derived input, so it is written next to the outputs and
    ``data/`` is never touched.
    """
    exports = [EnspireFile(csv_f, verbose=verbose)]
    exports.extend(EnspireFile(csv, verbose=verbose) for csv, _ in joins)
    notes = [] if note_f is None else [note_f]
    notes.extend(note for _, note in joins)
    ef = merge_enspire_files(exports) if len(exports) > 1 else exports[0]
    written: list[str] = []
    for one in exports:
        written += one.export_measurements(out, mode=export)
    messages: list[EnspireMessage] = []
    note_path = notes[0] if notes else None
    if len(notes) > 1 and note_path is not None:
        frames = [pd.read_csv(path) for path in notes]
        joined = pd.concat(frames, ignore_index=True)
        note_fp = out / f"{note_path.name.removesuffix('_note.csv')}_joined_note.csv"
        joined.to_csv(note_fp, index=False)
        provenance = {
            "inputs": [str(path) for path in [csv_f, *(csv for csv, _ in joins)]],
            "notes": [str(path) for path in notes],
            "labels": sorted(ef.measurements),
            "temperatures": sorted({
                str(m["metadata"].get("temp")) for m in ef.measurements.values()
            }),
        }
        note_fp.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n")
        note_path = note_fp
        if verbose:
            messages.append(
                EnspireMessage("info", f"joined {len(notes)} notes into {note_fp}")
            )
    if verbose and not written:
        messages.append(
            EnspireMessage(
                "info",
                f"measurement tables already up to date in {out}; nothing rewritten",
            )
        )
    return EnspireInputs(ef=ef, note=note_path, messages=tuple(messages))


def write_isosbestic_table(
    ef: EnspireFile,
    note_fp: Path,
    out_dir: Path,
    *,
    isosbestic: float | str,
) -> pd.DataFrame:
    """Record the crossing each (sample, temperature) was normalised at.

    The wavelength is a fitted-by-search number, not a constant of the method, so
    it is written beside the K tables instead of being implied by them: a
    normalised K can only be reproduced with the crossing that produced it.

    Parameters
    ----------
    ef : EnspireFile
        The parsed EnSpire export.
    note_fp : Path
        The note describing wells, titrant and samples.
    out_dir : Path
        Where ``<stem>_isosbestic.csv`` goes.
    isosbestic : float | str
        A wavelength [nm] or ``"auto"``.

    Returns
    -------
    pd.DataFrame
        One row per (sample, temperature): the wavelength used (``nan`` when the
        group has no crossing, which is the chloride case) and the wells read.
    """
    note = Note(note_fp).note
    samples = note[note["Name"] != "buffer"]
    buffers = note[note["Name"] == "buffer"]
    if isinstance(isosbestic, str) and isosbestic == "auto-scale":
        shared = isosbestic_by_group(ef, note)
        shared_rows = [
            {
                "sample": name,
                "temp": temp,
                "isosbestic_nm": shared.get((str(name), float(str(temp))), np.nan),
                "n_wells": int((samples["Name"] == name).sum()),
            }
            for (name, temp) in samples[["Name", "Temp"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        ]
        table = pd.DataFrame(
            shared_rows, columns=["sample", "temp", "isosbestic_nm", "n_wells"]
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = note_fp.name.removesuffix("_note.csv").removesuffix("_joined_note")
        table.to_csv(out_dir / f"{stem}_isosbestic.csv", index=False)
        return table
    rows: list[dict[str, object]] = []
    for (name, temp), group in samples.groupby(
        ["Name", "Temp"], sort=True, dropna=False
    ):
        one = group.drop_duplicates("Well")
        is_ph = bool(one["pH"].nunique() > one["Cl"].nunique())
        wells = one["Well"].tolist()
        x = np.asarray(one["pH" if is_ph else "Cl"], dtype=np.float64)
        wanted_temp = float(str(temp))
        bufw = buffers[np.isclose(buffers["Temp"].astype(float), wanted_temp)][
            "Well"
        ].tolist()
        spectra, used, used_x, _peaks = sample_spectra(
            ef,
            wells,
            x,
            str(temp),
            is_ph=is_ph,
            buffer_wells=bufw,
            bands=("exc_anionic",),
        )
        # sample_spectra drops blank plate rows, so the crossing is located on
        # the axis it returns rather than on the note's
        lam_iso = (
            plate_isosbestic(
                spectra, used_x, x_min=bands_module.PH_ACID_END if is_ph else None
            )
            if isinstance(isosbestic, str)
            else isosbestic
        )
        rows.append({
            "sample": name,
            "temp": temp,
            "isosbestic_nm": lam_iso if lam_iso is not None else np.nan,
            "n_wells": len(used),
        })
    table = pd.DataFrame(rows, columns=["sample", "temp", "isosbestic_nm", "n_wells"])
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = note_fp.name.removesuffix("_note.csv").removesuffix("_joined_note")
    table.to_csv(out_dir / f"{stem}_isosbestic.csv", index=False)
    return table
