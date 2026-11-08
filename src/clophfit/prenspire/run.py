"""EnSpire runs: fit a note's titrations and write what each method produces.

The functions here do the fitting and the writing and return what happened;
the CLI parses options and prints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import arviz as az  # type: ignore[import-untyped]
import lmfit  # type: ignore[import-untyped]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from clophfit.fitting.core import analyze_spectra, analyze_spectra_glob
from clophfit.fitting.data_structures import DataArray, Dataset
from clophfit.fitting.model_validation import trace_diagnostics
from clophfit.fitting.models import binding_1site
from clophfit.prenspire import EnspireFile, Note, bands as bands_module
from clophfit.prenspire.provenance import write_provenance
from clophfit.prenspire.spectral import (
    fit_titrations_spectral,
    isosbestic_by_group,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from clophfit.fitting.data_structures import FitResult


class EnspireInputError(ValueError):
    """An option the input data cannot satisfy; the CLI reports it as a usage error."""


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


#: Screening floors for a posterior, not convergence criteria: they catch a
#: posterior that is visibly unhealthy, and passing them proves nothing.
_RHAT_SCREEN = 1.01
_ESS_BULK_SCREEN = 400.0
_BFMI_SCREEN = 0.3
#: share of trajectories hitting the depth cap before the geometry is suspect
_TREEDEPTH_SCREEN = 0.2


# Prior SD (pH) floor for --x-sigma: strictly positive, and small enough to
# keep the posterior honest instead of smearing every point onto its plateau.
X_SIGMA_MIN_PH = 0.01
_MIN_BAYES_WELLS_AFTER_SCREEN = 6  # below this the shared latent axis is not identified


def _diag_number(diag: Mapping[str, object], key: str) -> float | None:
    """Read one screening number, or ``None`` when the sampler did not record it."""
    value = diag.get(key)
    if value is None:
        return None
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def posterior_screen(diag: Mapping[str, object]) -> str:
    """Say whether a posterior passed the sampler's screening floors.

    The floors are screening values, not convergence criteria: no divergences,
    rank r-hat below 1.01, bulk ESS above 400, energy BFMI above 0.3. Passing them
    still does not establish that the model is identified or the priors sensible,
    and the line stays silent when all four pass.
    """
    problems: list[str] = []
    divergences = _diag_number(diag, "n_divergences")
    if divergences:
        problems.append(f"{divergences:.0f} divergences")
    rhat = _diag_number(diag, "rhat_max")
    if rhat is not None and rhat > _RHAT_SCREEN:
        problems.append(f"r-hat {rhat:.3f}")
    ess = _diag_number(diag, "ess_bulk_min")
    if ess is not None and ess < _ESS_BULK_SCREEN:
        problems.append(f"bulk ESS {ess:.0f}")
    bfmi = _diag_number(diag, "bfmi")
    if bfmi is not None and bfmi < _BFMI_SCREEN:
        problems.append(f"BFMI {bfmi:.2f}")
    depth = _diag_number(diag, "reached_max_treedepth_frac")
    if depth is not None and depth > _TREEDEPTH_SCREEN:
        problems.append(f"{100 * depth:.0f}% at max tree depth")
    if not problems:
        return ""
    return "  ! screen: " + ", ".join(problems) + " (screening only, not proof)"


def posterior_columns(diag: Mapping[str, object]) -> dict[str, object]:
    """Collect the screening numbers as table columns beside the estimate."""
    return {
        "n_divergences": diag.get("n_divergences", np.nan),
        "rhat_max": diag.get("rhat_max", np.nan),
        "ess_bulk_min": diag.get("ess_bulk_min", np.nan),
        "ess_tail_min": diag.get("ess_tail_min", np.nan),
        "bfmi": diag.get("bfmi", np.nan),
        "max_treedepth": diag.get("max_treedepth", np.nan),
        "n_reached_max_treedepth": diag.get("n_reached_max_treedepth", np.nan),
        "reached_max_treedepth_frac": diag.get("reached_max_treedepth_frac", np.nan),
        "sampler": diag.get("sampler", "unknown"),
    }


def _post_samples(trace: object, name: str) -> np.ndarray:
    """Flattened posterior draws of one variable from an ArviZ trace."""
    return np.asarray(
        az.extract(trace, var_names=[name]).to_numpy(), dtype=float
    ).ravel()


def plot_bayes(
    figdata: list[
        tuple[
            str,
            str,
            dict[str, tuple[np.ndarray, np.ndarray]],
            bands_module.BandBayes,
            bool,
        ]
    ],
    pdf_path: Path,
) -> None:
    """Per sample: bands with the posterior curve band, and the K posterior.

    A sample whose sampler produced nothing carries a null trace; it is
    skipped rather than drawn, so one unfitted sample does not cost the
    others their figure.
    """
    drawable = [row for row in figdata if row[3].trace is not None]
    if not drawable:
        return
    n = len(drawable)
    fig, axes = plt.subplots(
        n, 2, figsize=(10.8, 4.0 * n), squeeze=False, constrained_layout=True
    )
    for row, (sample, t, data, bb, is_ph) in enumerate(drawable):
        left, right = axes[row]
        ks = _post_samples(bb.trace, "K")
        pick = ks[:: max(1, len(ks) // 200)]
        for i, (nm, (x, y)) in enumerate(sorted(data.items())):
            left.plot(x, y, "o", ms=4, color=f"C{i}", alpha=0.85, label=nm)
            s0 = float(np.median(_post_samples(bb.trace, f"S0_{nm}")))
            s1 = float(np.median(_post_samples(bb.trace, f"S1_{nm}")))
            grid = np.linspace(float(x.min()), float(x.max()), 200)
            # same binding_1site the likelihood used, so the drawn band is the
            # fitted one (an independently-written fraction used here used to
            # invert the curve relative to the data)
            curves = np.array([
                binding_1site(grid, kk, s0, s1, is_ph=is_ph) for kk in pick
            ])
            left.fill_between(
                grid,
                np.quantile(curves, 0.03, axis=0),
                np.quantile(curves, 0.97, axis=0),
                color=f"C{i}",
                alpha=0.15,
            )
            left.plot(
                grid, np.quantile(curves, 0.5, axis=0), "-", color=f"C{i}", lw=1.2
            )
        left.legend(fontsize=7)
        left.set_title(
            f"{sample} {t} C: K {bb.k:.2f} [{bb.hdi03:.2f}, {bb.hdi97:.2f}] 94% HDI"
        )
        left.set_xlabel("pH" if is_ph else "ligand (mM)")
        left.set_ylabel("normalized signal")
        right.hist(ks, bins=40, color="0.75")
        right.axvline(bb.k, color="k", lw=1)
        right.axvspan(bb.hdi03, bb.hdi97, color="orange", alpha=0.25)
        right.set_title(f"posterior K, 94% HDI, rhat {bb.rhat:.3f}")
        right.set_xlabel("K")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def shared_isosbestic(
    ef: EnspireFile,
    note: pd.DataFrame,
    isosbestic: float | str | None,
    *,
    buffer: bool,
) -> dict[tuple[str, float], float | None] | None:
    """Resolve ``auto-scale`` into one crossing per (sample, temperature).

    Parameters
    ----------
    ef : EnspireFile
        The parsed EnSpire export.
    note : pd.DataFrame
        The normalised note.
    isosbestic : float | str | None
        The option as parsed: a wavelength, ``"auto"``, ``"auto-scale"`` or
        ``None``.
    buffer : bool
        Subtract the buffer wells before locating the crossings.

    Returns
    -------
    dict[tuple[str, float], float | None] | None
        The per-sample crossings, or ``None`` when the caller should let each
        group locate its own.
    """
    if isinstance(isosbestic, str) and isosbestic == "auto-scale":
        return isosbestic_by_group(ef, note, buffer=buffer)
    return None


def lam_for(
    isosbestic: float | str | None,
    lam_map: dict[tuple[str, float], float | None] | None,
    name: object,
    temp: object,
) -> float | str | None:
    """Pick the crossing to hand to one (sample, temperature) group."""
    if lam_map is None:
        return isosbestic
    return lam_map.get((str(name), float(str(temp))))


def resolve_x_sigma(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
    x_sigma: str | None,
    is_ph: bool,  # ruff: ignore[boolean-type-hint-positional-argument]
    one: object,
    name: str,
    t: str,
    messages: list[EnspireMessage],
) -> float | None:
    """Resolve ``--x-sigma`` for one (sample, temperature) bayes fit.

    ``None`` leaves the axis fixed at the note values. ``auto`` pools the
    note's replicate-row spread per index column (rows A-C and D-F are the
    same nominal targets at their temperature) into the prior sigma - on the
    pH axis for a pH titration, on the Cl column (mM) for a ligand one; a
    number passes through after bounds checking, in the axis's units.
    """
    if x_sigma is None:
        return None
    one = cast("pd.DataFrame", one)
    xcol = "pH" if is_ph else "Cl"
    if x_sigma.strip().lower() == "auto":
        sigma = bands_module.estimate_x_error_sigma(one["Well"], one[xcol])
        if sigma is None:
            messages.append(
                EnspireMessage(
                    "warning",
                    f"warning: {name} at {t}: fewer than two index columns with two "
                    "or more replicates; using a fixed axis (note values)",
                )
            )
            return None
        return sigma
    try:
        sigma = float(x_sigma)
    except ValueError:
        msg = "--x-sigma must be 'auto' or a number, e.g. 0.15"
        raise EnspireInputError(msg) from None
    if sigma < X_SIGMA_MIN_PH:
        units = "pH" if is_ph else "mM"
        msg = f"--x-sigma must be >= {X_SIGMA_MIN_PH} {units} (or 'auto')"
        raise EnspireInputError(msg)
    return sigma


def run_bayes(  # ruff: ignore[too-many-arguments, too-many-branches, too-many-statements, complex-structure]
    ef: EnspireFile,
    note_fp: Path,
    out_dir: Path,
    *,
    buffer: bool,
    band_names: tuple[str, ...],
    seed: int,
    draws: int,
    chains: int,
    cores: int | None,
    target_accept: float | None,
    x_sigma: str | None,
    robust: bool = True,
    screen: str | None = None,
    isosbestic: float | str | None = None,
    provenance: Mapping[str, object] | None = None,
) -> EnspireRunResult[NamedTuple]:
    """Seeded posterior of the shared K per sample/temperature.

    Parameters
    ----------
    ef : EnspireFile
        The parsed EnSpire export.
    note_fp : Path
        The note describing wells, titrant and samples.
    out_dir : Path
        Where the K table and the posterior figure go.
    buffer : bool
        Subtract the buffer wells of the same plate column.
    band_names : tuple[str, ...]
        Bands entering the shared-K posterior; default the direct readouts.
    seed : int
        NUTS seed, so the HDI reproduces.
    draws : int
        Posterior draws per chain.
    chains : int
        MCMC chains; reproducible for a fixed seed in either core mode.
    cores : int | None
        CPU cores for chain parallelism; None lets PyMC parallelise.
    target_accept : float | None
        NUTS target acceptance; None keeps the sampler default.
    x_sigma : str | None
        "auto" or a prior SD in pH on each well's true axis value
        (x_true ~ N(note pH, sigma), independently); None fixes the axis at
        the note values.
    robust : bool
        Student-t (nu = 3) likelihood when True, plain Normal when False.
    screen : str | None
        Point screen applied to the readouts before sampling, in the ``--screen``
        form. A well flagged in any band is dropped from **every** band: the
        posterior carries one latent axis per well shared by the bands, so the
        bands must keep the same wells. ``None`` keeps every point.

    isosbestic : float | str | None
        Normalise every well by its signal at the isosbestic point before
        fitting: a wavelength [nm], ``"auto"`` to locate it per group, or
        ``None`` (see :func:`~clophfit.prenspire.spectral.plate_isosbestic`).

    provenance : Mapping[str, object] | None
        The options this run was asked for, written beside the tables as JSON
        together with the library version and the input files.

    Returns
    -------
    EnspireRunResult[NamedTuple]
        The table, the files written, and the messages: one line per
        (sample, temperature) and any warnings, in the order they happened.
    """
    before = _snapshot(out_dir)
    messages: list[EnspireMessage] = []
    note_df = Note(note_fp).note
    samples = note_df[note_df["Name"] != "buffer"]
    buffers = note_df[note_df["Name"] == "buffer"]
    wanted = tuple(band_names) if band_names else bands_module.DIRECT_BANDS
    lam_map = shared_isosbestic(ef, note_df, isosbestic, buffer=buffer)
    rows: list[dict[str, object]] = []
    figdata: list[
        tuple[
            str,
            str,
            dict[str, tuple[np.ndarray, np.ndarray]],
            bands_module.BandBayes,
            bool,
        ]
    ] = []
    for (name, temp), group in samples.groupby(
        ["Name", "Temp"], sort=True, dropna=False
    ):
        t = str(temp).split(".")[0]
        one = group.drop_duplicates("Well")
        wells = one["Well"].tolist()
        is_ph = bool(one["pH"].nunique() > one["Cl"].nunique())
        titrant = one["pH"] if is_ph else one["Cl"]
        x = np.array(titrant.astype(float), dtype=np.float64)
        bufw = (
            buffers[np.isclose(buffers["Temp"].astype(float), float(t))][
                "Well"
            ].tolist()
            if len(buffers)
            else []
        )
        readouts = bands_module.classify(ef, t, is_ph=is_ph)
        scale = None
        lam = lam_for(isosbestic, lam_map, name, temp)
        if lam is not None:
            excitation = next((r for r in readouts if r.name == "exc_anionic"), None)
            if excitation is not None:
                _lam_iso, scale = bands_module.isosbestic_readout_scale(
                    ef,
                    wells,
                    x,
                    excitation,
                    bufw if buffer else (),
                    isosbestic=lam,
                    x_min=bands_module.PH_ACID_END if is_ph else None,
                )
        data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for r in readouts:
            if r.name not in wanted:
                continue
            y = bands_module.band_values(ef, r, wells)
            if bufw and buffer:
                y = bands_module.subtract_buffer(
                    y, bands_module.band_values(ef, r, bufw)
                )
            arr = np.array(y, dtype=np.float64)
            if scale is not None:
                arr /= scale
            top = float(np.nanmax(np.abs(arr)))
            if not np.isfinite(top) or top <= 0.0:
                continue
            data[r.name] = (x, arr / top)
        if not data:
            continue
        n_kept = x.size
        if screen is not None:
            # One mask for every band: a well flagged in any band leaves all of
            # them, because the posterior shares one latent axis across bands.
            screened = bands_module.fit_bands(data, is_ph=is_ph, screen=screen)
            if screened is not None:
                keep = np.ones(x.size, dtype=bool)
                for mask in screened.masks.values():
                    flags = np.asarray(mask, dtype=bool)
                    if flags.size == x.size:
                        keep &= flags
                if keep.sum() >= _MIN_BAYES_WELLS_AFTER_SCREEN:
                    data = {nm: (xx[keep], yy[keep]) for nm, (xx, yy) in data.items()}
                    x = x[keep]
                    wells = [w for w, k in zip(wells, keep, strict=True) if k]
                    n_kept = int(keep.sum())
                else:  # a screen that eats the plate is not a screen
                    messages.append(
                        EnspireMessage(
                            "warning",
                            f"warning: {name} at {t}: screen would leave "
                            f"{int(keep.sum())} of {x.size} wells; keeping every well",
                        )
                    )
        ds = Dataset(
            {nm: DataArray(xx, yy) for nm, (xx, yy) in data.items()}, is_ph=is_ph
        )
        x_res = resolve_x_sigma(x_sigma, is_ph, one, str(name), t, messages)
        bb = bands_module.fit_bands_bayes(
            ds,
            seed=seed,
            draws=draws,
            chains=chains,
            cores=cores,
            target_accept=target_accept,
            robust=robust,
            x_sigma=x_res,
        )
        label = cast("str", name)
        diag = trace_diagnostics(bb.trace, summary_var_names=["K"])
        rows.append({
            "sample": label,
            "temp": t,
            "K": bb.k,
            "hdi03": bb.hdi03,
            "hdi97": bb.hdi97,
            "r_hat": bb.rhat,
            "ess_bulk": bb.ess,
            "n": n_kept,
            "x_sigma": x_res,
            **posterior_columns(diag),
        })
        figdata.append((label, t, data, bb, is_ph))
    table = pd.DataFrame(
        rows,
        columns=[
            "sample",
            "temp",
            "K",
            "hdi03",
            "hdi97",
            "r_hat",
            "ess_bulk",
            "n",
            "x_sigma",
            "n_divergences",
            "rhat_max",
            "ess_bulk_min",
            "ess_tail_min",
            "bfmi",
            "max_treedepth",
            "n_reached_max_treedepth",
            "reached_max_treedepth_frac",
            "sampler",
        ],
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = note_fp.name.removesuffix("_note.csv")
    table.to_csv(out_dir / f"{stem}_K_bayes.csv", index=False)
    write_provenance(
        out_dir / f"{stem}_K_bayes.json",
        inputs={"data": str(ef.file), "note": str(note_fp)},
        options=provenance,
        outputs=[f"{stem}_K_bayes.csv", f"{stem}_K_bayes.pdf"],
    )
    if figdata:
        plot_bayes(figdata, out_dir / f"{stem}_K_bayes.pdf")
    for row in table[table["K"].notna()].itertuples():
        cell = str(row.x_sigma)
        xsig = f" x_sigma {float(cell):.3f}" if cell not in {"nan", "None"} else ""
        line = (
            f"{row.sample} at {row.temp}: K {row.K:.3f} HDI94 [{row.hdi03:.3f}, {row.hdi97:.3f}] "
            f"rhat {row.r_hat:.3f} ess {row.ess_bulk:.0f} on {row.n} points{xsig}"
            + posterior_screen({
                "n_divergences": row.n_divergences,
                "rhat_max": row.rhat_max,
                "ess_bulk_min": row.ess_bulk_min,
                "bfmi": row.bfmi,
            })
        )
        messages.append(EnspireMessage("info", line))
    return EnspireRunResult(
        written=_written_since(out_dir, before),
        table=table,
        summaries=(),
        messages=tuple(messages),
    )
