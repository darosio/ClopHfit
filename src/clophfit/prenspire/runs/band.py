"""Band runs: fit a note's titrations from named spectral windows and share K."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

import matplotlib.pyplot as plt

from clophfit.prenspire import (
    EnspireFile,
    Note,
    bands as bands_module,
)
from clophfit.prenspire.provenance import write_provenance

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path


from clophfit.prenspire.runs.common import EnspireMessage, EnspireRunResult


class BandSummary(NamedTuple):
    """The shared-K line reported for one sample at one temperature."""

    sample: Any
    temp: Any
    K: Any
    lo: Any
    hi: Any
    n: Any
    dropped: Any


def _band_line(r: BandSummary) -> str:
    """Format the console line for one sample's shared K in a band run."""
    return (
        f"{r.sample} at {r.temp}: K {r.K:.3f} [{r.lo:.3f}, {r.hi:.3f}] "
        f"on {r.n} points, {r.dropped} screened out"
    )


def run_bands(  # ruff: ignore[too-many-arguments]
    ef: EnspireFile,
    note_fp: Path,
    out_dir: Path,
    *,
    buffer: bool,
    screen: str | None,
    min_rho: float,
    wanted: Sequence[str],
    isosbestic: float | str | None = None,
    provenance: Mapping[str, object] | None = None,
) -> EnspireRunResult[BandSummary]:
    """Fit a note's titrations from named bands and write the tables and figure.

    Parameters
    ----------
    ef : EnspireFile
        The parsed EnSpire export.
    note_fp : Path
        The note describing wells, titrant and samples.
    out_dir : Path
        Where the tables and the figure go.
    buffer : bool
        Subtract the buffer well of the same plate column.
    screen : str | None
        Point screen, e.g. ``studentized:0.05:5``; None keeps every point.
    min_rho : float
        Monotonicity gate below which a band carries no K.
    wanted : Sequence[str]
        Bands allowed into the shared K.

    isosbestic : float | str | None
        Normalise every well by its signal at the isosbestic point before
        fitting: a wavelength [nm], ``"auto"`` to locate it per group, or
        ``None`` (see :func:`~clophfit.prenspire.spectral.plate_isosbestic`).

    provenance : Mapping[str, object] | None
        The options this run was asked for, written beside the tables as JSON
        together with the library version and the input files.

    Returns
    -------
    EnspireRunResult[BandSummary]
        The tables and figure written, the shared-K summaries, and the table.
    """
    note = Note(note_fp)
    fits = bands_module.fit_titrations(
        ef,
        note.note,
        buffer=buffer,
        wanted=wanted,
        min_rho=min_rho,
        screen=screen,
        isosbestic=isosbestic,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = note_fp.name.removesuffix("_note.csv")
    fits.table.to_csv(out_dir / f"{stem}_K.csv", index=False)
    if not fits.screened.empty:
        fits.screened.to_csv(out_dir / f"{stem}_screened.csv", index=False)
    if fits.samples:
        fig, axes = plt.subplots(
            1,
            len(fits.samples),
            figsize=(5.4 * len(fits.samples), 4.3),
            constrained_layout=True,
            squeeze=False,
        )
        for ax, (title, sample) in zip(axes[0], fits.samples.items(), strict=True):
            bands_module.plot_sample(ax, sample, title)
        fig.savefig(out_dir / f"{stem}_bands.pdf", bbox_inches="tight")
        plt.close(fig)
    json_path = write_provenance(
        out_dir / f"{stem}_K.json",
        inputs={"data": str(ef.file), "note": str(note_fp)},
        options=provenance,
        outputs=[f"{stem}_K.csv", f"{stem}_bands.pdf"],
    )
    head = fits.table[
        (fits.table.band == "all (shared K)") & (fits.table.subset == "screened")
    ]
    summaries = tuple(
        BandSummary(
            sample=r.sample,
            temp=r.temp,
            K=r.K,
            lo=r.lo,
            hi=r.hi,
            n=r.n,
            dropped=r.dropped,
        )
        for r in head.itertuples()
    )
    written = [out_dir / f"{stem}_K.csv"]
    if not fits.screened.empty:
        written.append(out_dir / f"{stem}_screened.csv")
    if fits.samples:
        written.append(out_dir / f"{stem}_bands.pdf")
    written.append(json_path)
    return EnspireRunResult(
        written=tuple(written),
        table=fits.table,
        summaries=summaries,
        messages=tuple(EnspireMessage("info", _band_line(s)) for s in summaries),
    )
