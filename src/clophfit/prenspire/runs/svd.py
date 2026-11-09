"""Legacy SVD runs (``--method svd``) and the console report shared with ``fit-titration``."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import lmfit  # type: ignore[import-untyped]
import numpy as np
import pandas as pd

from clophfit.fitting.core import analyze_spectra, analyze_spectra_glob
from clophfit.fitting.data_structures import DataArray, Dataset
from clophfit.prenspire import (
    EnspireFile,
    Note,
)
from clophfit.prenspire.provenance import write_provenance
from clophfit.prenspire.spectral import (
    fit_titrations_spectral,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from clophfit.fitting.data_structures import FitResult

from clophfit.prenspire.runs.common import (
    EnspireMessage,
    EnspireRunResult,
    _snapshot,
    _written_since,
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
