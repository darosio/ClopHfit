"""EnSpire runs: fit a note's titrations and write what each method produces.

The functions here do the fitting and the writing and return what happened;
the CLI parses options and prints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

import lmfit  # type: ignore[import-untyped]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from clophfit.fitting.core import analyze_spectra, analyze_spectra_glob
from clophfit.fitting.data_structures import DataArray, Dataset
from clophfit.prenspire import EnspireFile, Note, bands as bands_module
from clophfit.prenspire.provenance import write_provenance
from clophfit.prenspire.spectral import fit_titrations_spectral

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from clophfit.fitting.data_structures import FitResult


class EnspireMessage(NamedTuple):
    """A note or warning printed in the console, in the order it happened."""

    level: str
    text: str


class GlobalSummary(NamedTuple):
    """The jackknife line reported for one sample at one temperature."""

    sample: Any
    temp: Any
    K: Any
    se: Any
    n: Any
    sv_ratio: Any


class BandSummary(NamedTuple):
    """The shared-K line reported for one sample at one temperature."""

    sample: Any
    temp: Any
    K: Any
    lo: Any
    hi: Any
    n: Any
    dropped: Any


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
        written=tuple(written), table=fits.table, summaries=summaries
    )


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
    table = fit_titrations_spectral(
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
        figure_stem=out_dir / stem_guess,
    )[0]
    if inspect:
        insp_fp = out_dir / f"{stem_guess}_global_inspect.txt"
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


def _covariance_report(fit_result: FitResult, *, ndigits: int = 2) -> str:
    """Build the lmfit confidence-interval layout from the covariance instead.

    The legacy svd path prints a table of the K parameter at the 3, 2 and 1 sigma
    levels on each side; anything downstream reads that layout. This builds the
    same seven columns from the fitted standard error, so a run whose profile
    interval cannot be bracketed still reports a K with an interval.
    """
    params = fit_result.result.params if fit_result.result is not None else None
    if params is None or "K" not in params:
        return ""
    k = float(params["K"].value)
    se = float(params["K"].stderr or np.nan)
    grid = np.array([-3, -2, -1, 0, 1, 2, 3], dtype=float)
    values = k + grid * se
    return "K         : " + " ".join(f"{v:.{ndigits}f}" for v in values)


def result_report(fit_result: FitResult, pdf_file: Path, band_str: str) -> str:
    """Render the console block for one spectral fit: its figure, its K and interval."""
    try:
        ci = lmfit.conf_interval(fit_result.mini, fit_result.result)
    except Exception as err:  # ruff: ignore[blind-except] - lmfit raises MinimizerException here
        # The profile interval needs the objective to rise on both sides of the
        # best fit, and near a flat or bounded parameter it does not, so lmfit
        # cannot bracket it. The covariance is then the only interval there is:
        # report it in the same layout rather than losing the K line entirely.
        body = (
            f"# profile confidence interval unavailable ({err}); using covariance\n"
            f"{_covariance_report(fit_result, ndigits=2)}"
        )
    else:
        body = lmfit.ci_report(ci, ndigits=2, with_offset=False)
    return (
        f"{pdf_file}\nBest fit using '{band_str}' band:\n\n{body}\n\n"
        f" Plot saved in '{pdf_file}'.\n"
    )


def run_svd(  # ruff: ignore[too-many-arguments, complex-structure, too-many-branches, too-many-statements]
    ef: EnspireFile,
    note_fp: Path,
    out_dir: Path,
    bands: list[tuple[str, int, int]],
    verbose: int,
    *,
    isosbestic: float | str | None = None,
    qc: float | None = None,
    provenance: Mapping[str, object] | None = None,
) -> EnspireRunResult[NamedTuple]:
    """Fit every titration by projecting its whole spectra on the first principal component.

    ``isosbestic`` normalises every well by its signal at the isosbestic point
    before the projection, so the legacy svd path is normalised like the band and
    whole-spectrum ones (see :meth:`Note.build_titrations`).

    Parameters
    ----------
    ef : EnspireFile
        The parsed EnSpire export.
    note_fp : Path
        The note describing wells, titrant and samples.
    out_dir : Path
        Where the figures and the provenance record go.
    bands : list[tuple[str, int, int]]
        Named integration windows ``(label, first, last)`` for the band fits.
    verbose : int
        Verbosity passed to :class:`~clophfit.prenspire.Note`.
    isosbestic : float | str | None
        Isosbestic normalisation; see above.
    qc : float | None
        Robust cut of the per-scan screen, run before the projection; ``None``
        fits every well.
    provenance : Mapping[str, object] | None
        The options this run was asked for, written beside the figures.

    Returns
    -------
    EnspireRunResult[NamedTuple]
        The figures and the provenance record written, and one message per fit
        (the lmfit report) and per screened well group. It has no K table or
        summaries: the legacy path reports its K only in those messages.

    Raises
    ------
    ValueError
        If a titration is neither a pH nor a Cl series.
    """
    before = _snapshot(out_dir)
    messages: list[EnspireMessage] = []
    note = Note(note_fp, verbose=verbose)
    drop_map: dict[tuple[str, str], list[str]] = {}
    if qc is not None:
        # the detector is the whole-spectrum fit; the estimator stays this legacy
        # projection. One screen per (sample, temperature), over the same scans the
        # projection pools.
        screen_table, _fits = fit_titrations_spectral(
            ef,
            Note(note_fp).note,
            bands=("exc_anionic", "em_exc420", "em_exc278"),
            qc=qc,
        )
        for row in screen_table[screen_table.subset == "all"].itertuples():
            gone = [w for w in str(row.qc_dropped).split(";") if w and w != "-"]
            if gone:
                drop_map[str(row.sample), str(row.temp)] = gone
        for (sample, temp), gone in drop_map.items():
            messages.append(
                EnspireMessage(
                    "info", f"qc: {sample} at {temp}: dropped {', '.join(gone)}"
                )
            )
    note.build_titrations(ef, isosbestic=isosbestic, drop_wells=drop_map)
    dbands = {label: (ini, fin) for label, ini, fin in bands} if bands else {}
    ds_data: dict[str, DataArray] = {}
    for name, d_name in note.titrations.items():
        for temp, d_temp in d_name.items():
            for tit, d_tit in d_temp.items():
                if tit.split("_")[0] == "pH":
                    is_ph = False
                elif tit.split("_")[0] == "Cl":
                    is_ph = True
                else:
                    msg = "Unknown titration type."
                    raise ValueError(msg)
                for label, data in d_tit.items():
                    band = dbands.get(label)
                    fit_result = analyze_spectra(data, is_ph=is_ph, band=band)
                    if fit_result.is_valid() and fit_result.mini:
                        userargs = fit_result.mini.userargs[0]["default"]
                        ds_data[label] = DataArray(userargs.x, userargs.y)
                        pdf_file = out_dir / f"{name}_{temp}_{label}_{tit}_{band}.pdf"
                        if fit_result.figure:
                            fit_result.figure.savefig(pdf_file)
                    messages.append(
                        EnspireMessage(
                            "info", result_report(fit_result, pdf_file, str(band))
                        )
                    )
                # Global spectra analysis with more than 1 label.
                if (
                    len(d_tit.keys() - dbands.keys()) > 1  # svd > 1
                    or len(dbands.keys() & d_tit.keys()) > 1  # bands > 1
                ):
                    ds = Dataset(ds_data, is_ph=is_ph)
                    spectra_gres = analyze_spectra_glob(d_tit, ds, dbands)
                    if spectra_gres.svd and spectra_gres.svd.is_valid():
                        pdf_file = out_dir / f"{name}_{temp}_all_{tit}_SVD.pdf"
                        if spectra_gres.svd.figure:
                            spectra_gres.svd.figure.savefig(pdf_file)
                        messages.append(
                            EnspireMessage(
                                "info", result_report(spectra_gres.svd, pdf_file, "")
                            )
                        )
                    if spectra_gres.gsvd and spectra_gres.gsvd.is_valid():
                        pdf_file = out_dir / f"{name}_{temp}_g_{tit}_SVD.pdf"
                        if spectra_gres.gsvd.figure:
                            spectra_gres.gsvd.figure.savefig(pdf_file)
                        messages.append(
                            EnspireMessage(
                                "info", result_report(spectra_gres.gsvd, pdf_file, "")
                            )
                        )
                    if spectra_gres.bands and spectra_gres.bands.is_valid():
                        keys = dbands.keys() & d_tit.keys()
                        lname = [f"{k}({dbands[k][0]},{dbands[k][1]})" for k in keys]
                        bands_str = "".join(lname)
                        pdf_file = out_dir / f"{name}_{temp}_all_{tit}_{bands_str}.pdf"
                        if spectra_gres.bands.figure:
                            spectra_gres.bands.figure.savefig(pdf_file)
                        messages.append(
                            EnspireMessage(
                                "info",
                                result_report(spectra_gres.bands, pdf_file, bands_str),
                            )
                        )
    # the legacy path writes one figure per (sample, temperature, label) and keeps
    # no K table of its own, so the record names the figures as its outputs
    stem = note_fp.name.removesuffix("_note.csv")
    write_provenance(
        out_dir / f"{stem}_svd.json",
        inputs={"data": str(ef.file), "note": str(note_fp)},
        options=provenance,
        outputs=[f"{stem}_<sample>_<temp>_*.pdf"],
    )
    return EnspireRunResult(
        written=_written_since(out_dir, before),
        table=pd.DataFrame(),
        summaries=(),
        messages=tuple(messages),
    )
