"""Whole-spectrum least-squares runs (``--method global``)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

from clophfit.prenspire import (
    EnspireFile,
    Note,
)
from clophfit.prenspire.figures import plot_global_panels
from clophfit.prenspire.provenance import write_provenance
from clophfit.prenspire.spectral import fit_titrations_spectral

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


from clophfit.prenspire.runs.common import (
    EnspireMessage,
    EnspireRunResult,
    _snapshot,
    _written_since,
)


class GlobalSummary(NamedTuple):
    """The jackknife line reported for one sample at one temperature."""

    sample: Any
    temp: Any
    K: Any
    se: Any
    n: Any
    sv_ratio: Any


def run_global(  # ruff: ignore[too-many-arguments]
    ef: EnspireFile,
    note_fp: Path,
    out_dir: Path,
    *,
    buffer: bool,
    scans: tuple[str, ...] = (),
    screen: float | None = None,
    isosbestic: float | str | None = None,
    iso_band: float = 3.0,
    species_min: float | None = None,
    n_boot: int = 0,
    strata: str = "none",
    qc: float | None = None,
    inspect: bool = False,
    provenance: Mapping[str, object] | None = None,
) -> EnspireRunResult[GlobalSummary]:
    """Fit a note's titrations from whole spectra and write the table and its panel figures.

    Parameters
    ----------
    ef : EnspireFile
        The parsed EnSpire export.
    note_fp : Path
        The note describing wells, titrant and samples.
    out_dir : Path
        Where the table and the figure go.
    buffer : bool
        Subtract the buffer well of the same plate column.
    scans : tuple[str, ...]
        Band names whose scans enter whole; empty means the excitation scan
        only (the measured optimum; see :mod:`clophfit.prenspire.spectral`).
    screen : float | None
        Family-wise level of the well screen (e.g. ``0.05``): wells whose
        standardized whole-spectrum residual is beyond the chi-square cutoff are
        dropped and the fit repeated. ``None`` keeps every well, which is what
        this method did before; the band path screens single points by default.

    isosbestic : float | str | None
        Normalise every well by its signal at the isosbestic point before
        fitting: a wavelength [nm], ``"auto"`` to locate it per group, or
        ``None`` (see :func:`~clophfit.prenspire.spectral.plate_isosbestic`).
    iso_band : float
        Full width [nm] of the window averaged at the isosbestic crossing to form
        the per-well reference (a mean, never a sum); ``0`` uses the exact
        interpolated value at the crossing (one wavelength). It changes only a
        per-well constant, which the free well amplitude already fits, so it
        moves K by far less than its error bar.
    species_min : float | None
        Lower bound on every species spectrum, per wavelength (see
        :func:`~clophfit.fitting.spectral.fit_spectra_global`); ``None`` keeps
        the unconstrained linear solve.
    n_boot : int
        Well-cluster bootstrap replicates behind the reported interval (0 leaves
        it to the jackknife alone). Whole wells are resampled, all scans and all
        wavelengths together, because the wavelengths of one spectrum are not
        independent observations.
    strata : str
        ``"column"`` resamples wells within each plate column so every replicate
        spans the titration axis; ``"none"`` resamples from one pool.
    qc : float | None
        Robust cut of the per-scan screen, run before the fit (see
        :func:`~clophfit.prenspire.spectral.scan_outliers`); ``None`` fits every
        well.
    inspect : bool
        Also write ``<stem>_global_inspect.txt`` - a per-(sample, temperature)
        text dump of K and its errors, the wells used and dropped, each scan's
        species peaks and residual RMS, the 2-state residual sv, and every
        well's basic fraction and amplitude. The panel figure and the
        ``--global-qc`` output are unchanged.
    provenance : Mapping[str, object] | None
        The options this run was asked for, written beside the table as
        ``<stem>_K_global.json`` together with the library version and the input
        files. The table's own columns already carry the options that move the
        numbers; this records the whole invocation so a folder of results can be
        traced back to the command that made it.

    Returns
    -------
    EnspireRunResult[GlobalSummary]
        The table and the files written, the jackknife summaries, and the messages.
    """
    before = _snapshot(out_dir)
    messages: list[EnspireMessage] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    stem_guess = note_fp.name.removesuffix("_note.csv")
    if qc is not None and species_min is not None:
        # the screen judges standardized residuals and its cutoff was calibrated on
        # the unconstrained model, so it is always evaluated unconstrained; say so
        # once, because the two fits then differ in the species but not in which
        # wells survive
        messages.append(
            EnspireMessage(
                "note",
                f"note: --global-qc {qc:g} screens the unconstrained model even with "
                f"--species-min {species_min:g}, so the cutoff keeps its calibration; "
                "the bound applies to the reported fit only",
            )
        )
    run = fit_titrations_spectral(
        ef,
        Note(note_fp).note,
        buffer=buffer,
        screen=screen,
        bands=tuple(scans) if scans else ("exc_anionic",),
        isosbestic=isosbestic,
        iso_band=iso_band,
        species_min=species_min,
        n_boot=n_boot,
        strata=strata,
        qc=qc,
        inspect=inspect,
    )
    table = run.table
    for panel in run.panels.values():
        plot_global_panels(
            out_dir / f"{stem_guess}_{panel.file_key}_panels.pdf",
            panel.spectra,
            panel.x,
            panel.fit,
            title=panel.title,
            iso_nm=panel.iso_nm,
            peaks=panel.peaks,
        )
    if inspect:
        insp_fp = out_dir / f"{stem_guess}_global_inspect.txt"
        insp_fp.write_text(run.inspect_text)
        messages.append(EnspireMessage("info", f"inspect: wrote {insp_fp}"))
    stem = note_fp.name.removesuffix("_note.csv")
    table.to_csv(out_dir / f"{stem}_K_global.csv", index=False)
    write_provenance(
        out_dir / f"{stem}_K_global.json",
        inputs={"data": str(ef.file), "note": str(note_fp)},
        options=provenance,
        outputs=[f"{stem}_K_global.csv"],
    )
    if "qc_dropped" in table:
        for row in table[table.subset == "all"].itertuples():
            gone = str(row.qc_dropped)
            if gone and gone != "-":
                messages.append(
                    EnspireMessage(
                        "info", f"qc: {row.sample} at {row.temp}: dropped {gone}"
                    )
                )
    summaries = tuple(
        GlobalSummary(
            sample=r["sample"],
            temp=r["temp"],
            K=r["K"],
            se=r["se"],
            n=r["n"],
            sv_ratio=r["sv_ratio"],
        )
        for r in table[table.subset == "all"].to_dict("records")
    )
    return EnspireRunResult(
        written=_written_since(out_dir, before),
        table=table,
        summaries=summaries,
        messages=tuple(messages),
    )
