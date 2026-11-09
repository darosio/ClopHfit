"""EnSpire runs: fit a note's titrations and write what each method produces.

The functions here do the fitting and the writing and return what happened;
the CLI parses options and prints.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import arviz as az  # type: ignore[import-untyped]
import lmfit  # type: ignore[import-untyped]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from clophfit.fitting.bayes import column_axis_tie, well_groups
from clophfit.fitting.core import analyze_spectra, analyze_spectra_glob
from clophfit.fitting.data_structures import DataArray, Dataset
from clophfit.fitting.model_validation import trace_diagnostics
from clophfit.fitting.models import binding_1site
from clophfit.fitting.spectral import fit_spectra_global, posterior_predictive_check
from clophfit.prenspire import (
    EnspireFile,
    Note,
    bands as bands_module,
    merge_enspire_files,
)
from clophfit.prenspire.bands import ZERO_AT
from clophfit.prenspire.provenance import write_provenance
from clophfit.prenspire.spectral import (
    BayesPanelGroup,
    fit_titrations_spectral,
    isosbestic_by_group,
    model_outliers,
    plate_isosbestic,
    plot_global_bayes_panels,
    sample_spectra,
)
from clophfit.prenspire.thermo import (
    ZERO_C,
    fit_vanthoff_model_bayes,
    vanthoff_curves,
    vanthoff_titration,
)
from clophfit.prenspire.thermo_plot import (
    vanthoff_axes,
    vanthoff_axes_for,
    vanthoff_curve,
    vanthoff_enthalpy,
    vanthoff_right_line,
    vanthoff_right_point,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    import xarray as xr

    from clophfit.fitting.data_structures import FitResult
    from clophfit.prenspire.thermo import VanthoffModel


class EnspireInputError(ValueError):
    """An option the input data cannot satisfy; the CLI reports it as a usage error."""


class EnspireRunError(RuntimeError):
    """A run that cannot finish on this data; the CLI reports it as an error."""


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


def _axis_scale(trace: object, name: str) -> float:
    """Posterior median of an inferred axis scale, or NaN when the model has none.

    ``free`` has no group or well scale, so the column is NaN rather than a
    number that would look like a fitted quantity.
    """
    post = getattr(trace, "posterior", None)
    if post is None or name not in post:
        return float("nan")
    return float(np.median(np.asarray(post[name].values)))


def _flag(p_two_sided: float) -> bool:
    """Whether a two-sided Bayesian p-value is small enough to print."""
    return p_two_sided < 0.05  # ruff: ignore[magic-value-comparison]


def _write_ppc(
    figdata: Sequence[Any],
    out_dir: Path,
    stem: str,
    suffix: str,
    messages: list[EnspireMessage],
) -> pd.DataFrame:
    """Run and write the posterior predictive check for every fitted group.

    ``figdata`` carries the trace and the spectra each group was fitted on, so the
    check needs nothing else. One row per (group, scan, statistic); the console
    line names the statistics whose observed value falls outside the replicate
    interval, because those are the ones that say the model is misspecified rather
    than merely noisy.
    """
    tables: list[pd.DataFrame] = []
    # the tuple carries the scan peaks as an eighth field since the panel figure
    # needed them to restore measured units; the check ignores that field
    for sample, temp, is_ph, _wells, _x, spectra, trace, _peaks in figdata:
        table = posterior_predictive_check(trace, spectra, is_ph=bool(is_ph), draws=200)
        table.insert(0, "temp", temp)
        table.insert(0, "sample", sample)
        tables.append(table)
        messages.extend(
            EnspireMessage(
                "info",
                f"ppc: {sample} at {temp} {rec['scan']} {rec['stat']} "
                f"observed {float(rec['observed']):.3g} vs replicates "
                f"[{float(rec['rep_lo']):.3g}, {float(rec['rep_hi']):.3g}] "
                f"p {float(rec['p_two_sided']):.3f}",
            )
            for rec in table.to_dict("records")
            if _flag(float(rec["p_two_sided"]))
        )
    frame = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    if not frame.empty:
        frame.to_csv(out_dir / f"{stem}_ppc.csv", index=False)
        messages.append(
            EnspireMessage("info", f"ppc: wrote {out_dir / f'{stem}_ppc.csv'}")
        )
    del suffix
    return frame


def _plot_global_bayes(
    figdata: list[
        tuple[
            str,
            str,
            bool,
            list[str],
            np.ndarray,
            dict[str, tuple[np.ndarray, np.ndarray]],
            Any,
            dict[str, float],
        ]
    ],
    pdf_path: Path,
) -> None:
    """Write the whole-spectrum Bayesian figure: data, two fractions, latent axis.

    A thin adapter: the panels live in
    :func:`clophfit.prenspire.spectral.plot_global_bayes_panels`, so the figure
    and the fit cannot drift apart about which wells, scans or peaks were used.
    """
    groups = [
        BayesPanelGroup(
            label=f"{sample} at {temp}",
            is_ph=bool(is_ph),
            wells=wells,
            x_note=np.asarray(x, dtype=np.float64),
            spectra=spectra,
            trace=trace,
            peaks=peaks,
        )
        for sample, temp, is_ph, wells, x, spectra, trace, peaks in figdata
    ]
    plot_global_bayes_panels(pdf_path, groups)


def run_global_bayes(  # ruff: ignore[too-many-arguments, too-many-statements]
    ef: EnspireFile,
    note_fp: Path,
    out_dir: Path,
    *,
    buffer: bool,
    scans: tuple[str, ...],
    x_sigma: str,
    seed: int,
    draws: int,
    chains: int,
    cores: int | None,
    target_accept: float | None,
    fitter: Callable[..., Any],
    acid_state: bool = False,
    suffix: str,
    label: str,
    isosbestic: float | str | None = None,
    iso_band: float = 3.0,
    qc: float | None = None,
    species_min: float | None = None,
    x_pool: str = "partial",
    x_groups: str = "column",
    provenance: Mapping[str, object] | None = None,
    ppc: bool = False,
) -> EnspireRunResult[NamedTuple]:
    """Shared whole-spectrum Bayesian pipeline behind the two fitters.

    Same generative model and reporting for free-species
    (:func:`fit_spectra_global_bayes`) and marginalized-species
    (:func:`fit_spectra_global_bayes_marginal`) inference; ``suffix`` names the
    output files, ``label`` the console line.
    """
    before = _snapshot(out_dir)
    messages: list[EnspireMessage] = []
    note = Note(note_fp).note
    samples = note[note["Name"] != "buffer"]
    buffers = note[note["Name"] == "buffer"]
    wanted = scans or ("exc_anionic",)
    lam_map = shared_isosbestic(ef, note, isosbestic, buffer=buffer)
    rows: list[dict[str, object]] = []
    x_rows: list[dict[str, object]] = []
    figdata: list[
        tuple[
            str,
            str,
            bool,
            list[str],
            np.ndarray,
            dict[str, tuple[np.ndarray, np.ndarray]],
            Any,
            dict[str, float],
        ]
    ] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for (name, temp), group in samples.groupby(["Name", "Temp"], sort=True):
        one = group.drop_duplicates("Well")
        is_ph = bool(one["pH"].nunique() > one["Cl"].nunique())
        xcol = "pH" if is_ph else "Cl"
        wells = one["Well"].tolist()
        x = np.asarray(one[xcol], dtype=np.float64)
        bwells = buffers[buffers["Temp"] == temp]["Well"].tolist() if buffer else []
        lam = lam_for(isosbestic, lam_map, name, temp)
        if isinstance(lam, str):
            # resolve "auto" here, as the least-squares path does, so the crossing
            # that was actually used can go into the table instead of the word
            raw, _raw_w, raw_x, _raw_peaks = sample_spectra(
                ef, wells, x, str(temp), is_ph=is_ph, buffer_wells=bwells, bands=wanted
            )
            lam = (
                plate_isosbestic(
                    raw,
                    raw_x,
                    x_min=bands_module.PH_ACID_END if is_ph else None,
                    is_ph=is_ph,
                )
                if raw
                else None
            )
        spectra, wells, x, peaks = sample_spectra(
            ef,
            wells,
            x,
            str(temp),
            is_ph=is_ph,
            buffer_wells=bwells,
            bands=wanted,
            isosbestic=lam,
            iso_band=iso_band,
        )
        if not spectra:
            continue
        qc_dropped: list[str] = []
        if qc is not None:
            screen_fit = fit_spectra_global(
                x, spectra, is_ph=is_ph, well_scale=True, jackknife=False
            )
            table = model_outliers(screen_fit, spectra, wells, z=qc)
            qc_dropped = list(table.attrs.get("flagged_wells", []))
            if qc_dropped:
                gone = set(qc_dropped)
                mask = np.array([w not in gone for w in wells], dtype=bool)
                spectra = {
                    key: (lam_arr, y[:, mask]) for key, (lam_arr, y) in spectra.items()
                }
                wells = [w for w, k in zip(wells, mask, strict=True) if k]
                x = x[mask]
        # the replicate groups must follow the wells that survived the correction
        # and the QC screen: sample_spectra drops the wells it cannot measure for
        # this temperature, so groups built from the note's wells are too long and
        # create_x_true_pooled refuses them (32 groups for 24 wells on a two
        # temperature note)
        groups = well_groups(wells) if x_groups == "column" else None
        sigma = resolve_x_sigma(x_sigma, is_ph, one, str(name), str(temp), messages)
        if sigma is None:
            msg = f"--method {suffix.removeprefix('global_').replace('_', '-')} requires a pH titration and resolvable x-sigma"
            raise EnspireRunError(msg)
        trace = fitter(
            x,
            spectra,
            x_sigma=sigma,
            is_ph=is_ph,
            seed=seed,
            draws=draws,
            chains=chains,
            cores=cores,
            target_accept=target_accept or 0.9,
            species_min=species_min,
            x_pool=cast("Any", x_pool),
            x_groups=groups,
            **({"acid_state": acid_state} if acid_state else {}),
        )
        kdraw = np.asarray(trace.posterior["K"].values).reshape(-1)
        hdi = np.asarray(az.hdi(kdraw, prob=0.94), dtype=float)
        lo, hi = float(hdi[0]), float(hdi[1])
        key = f"{name}_{str(temp).replace('.', 'p')}"
        trace.to_netcdf(out_dir / f"{key}_{suffix}.nc")
        diag = trace_diagnostics(trace, summary_var_names=["K"])
        rows.append({
            "sample": name,
            "temp": temp,
            "K": float(np.median(kdraw)),
            "hdi03": lo,
            "hdi97": hi,
            "r_hat": float(az.rhat(trace, var_names=["K"])["K"].item()),
            "qc_dropped": ";".join(qc_dropped) or "-",
            "ess_bulk": float(az.ess(trace, var_names=["K"])["K"].item()),
            "n": len(wells),
            "x_sigma": sigma,
            **posterior_columns(diag),
            # the options that shape the numbers, as the least-squares global
            # table records them
            "scans": ";".join(spectra),
            "isosbestic_nm": np.nan if lam is None else float(lam),
            "iso_window_nm": np.nan if lam is None else float(iso_band),
            "species_min": np.nan if species_min is None else float(species_min),
            "x_pool": x_pool,
            "x_axis_group_sd": _axis_scale(trace, "x_true_group_sd"),
            "x_axis_well_sd": _axis_scale(trace, "x_true_well_sd"),
        })
        xdraw = np.asarray(trace.posterior["x_true"].values).reshape(-1, len(wells))
        figdata.append((str(name), str(temp), is_ph, wells, x, spectra, trace, peaks))
        for i, well in enumerate(wells):
            qlo, median, qhi = np.quantile(xdraw[:, i], [0.03, 0.5, 0.97])
            x_rows.append({
                "sample": name,
                "temp": temp,
                "well": well,
                "x_note": x[i],
                "x_shift": float(median) - float(x[i]),
                "x_median": median,
                "hdi03": qlo,
                "hdi97": qhi,
            })
        messages.append(
            EnspireMessage(
                "info",
                f"{name} at {temp}: {label} K {np.median(kdraw):.3f} "
                f"94% HDI [{lo:.3f}, {hi:.3f}] x_sigma {sigma:.3f}"
                + posterior_screen(diag),
            )
        )
    stem = note_fp.name.removesuffix("_note.csv")
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / f"{stem}_K_{suffix}.csv", index=False)
    write_provenance(
        out_dir / f"{stem}_K_{suffix}.json",
        inputs={"data": str(ef.file), "note": str(note_fp)},
        options=provenance,
        outputs=[
            f"{stem}_K_{suffix}.csv",
            f"{stem}_x_{suffix}.csv",
            f"{stem}_{suffix}.pdf",
            f"<stem>_<sample>_<temp>_{suffix}.nc",
        ],
    )
    pd.DataFrame(x_rows).to_csv(out_dir / f"{stem}_x_{suffix}.csv", index=False)
    _plot_global_bayes(figdata, out_dir / f"{stem}_{suffix}.pdf")
    if ppc:
        _write_ppc(figdata, out_dir, stem, suffix, messages)
    return EnspireRunResult(
        written=_written_since(out_dir, before),
        table=table,
        summaries=(),
        messages=tuple(messages),
    )


_MIN_VANTHOFF_TEMPS = 2  # one enthalpy needs a pair of temperatures


def _vanthoff_axis_tie(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
    note: pd.DataFrame,
    name: str,
    wells: Mapping[str, Sequence[str]],
    temps: Sequence[str],
    dph_buffer: str | None,
    messages: list[EnspireMessage],
) -> dict[str, tuple[np.ndarray, np.ndarray]] | None:
    """Tie each temperature's axis to the lowest one, well by column.

    A column is one nominal target and the buffer's pH moves with temperature the
    same way for every well in it, so only the reference temperature's latent axes
    need to be free: the others follow from them plus a measured offset. That
    removes the between-temperature half of the K-x ridge.

    Parameters
    ----------
    note : pd.DataFrame
        The normalised note, carrying ``Well`` and ``pH``.
    name : str
        Sample name, to pick its wells out of the note.
    wells : Mapping[str, Sequence[str]]
        Wells used at each temperature.
    temps : Sequence[str]
        The temperatures, ascending; the first is the reference.
    dph_buffer : str | None
        ``"auto"`` reads each column's offset from the note, a number uses it for
        every column, ``None`` ties nothing.
    messages : list[EnspireMessage]
        Receives the notes explaining a tie that could not be made.

    Returns
    -------
    dict[str, tuple[np.ndarray, np.ndarray]] | None
        Per tied temperature, the index into the reference wells and the offset, or
        ``None`` when nothing is tied - including when the wells do not pair by
        column, which is reported rather than silently ignored.
    """
    if dph_buffer is None:
        return None
    subset = note[note["Name"].astype(str) == name]
    # keyed by temperature AND well: these notes reuse the same well labels at both
    # temperatures, so a well-only key silently returns one temperature's pH for
    # both and every offset comes out zero
    recorded = {
        (float(temp), str(well)): float(ph)
        for temp, well, ph in zip(
            subset["Temp"], subset["Well"], subset["pH"], strict=True
        )
    }
    reference = temps[0]
    ph = {
        t: np.array(
            [recorded.get((float(t), str(w)), np.nan) for w in wells[t]],
            dtype=np.float64,
        )
        for t in temps
    }
    auto = dph_buffer.strip().lower() == "auto"
    if not auto:
        try:
            delta = float(dph_buffer)
        except ValueError:
            messages.append(
                EnspireMessage(
                    "note",
                    f"note: --dpH-buffer expects 'auto' or a pH offset, got {dph_buffer!r}; "
                    "the axes stay independent",
                )
            )
            return None
    tie: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for t in temps[1:]:
        got = column_axis_tie(
            list(wells[reference]), ph[reference], list(wells[t]), ph[t]
        )
        if got is None:
            messages.append(
                EnspireMessage(
                    "note",
                    f"note: --dpH-buffer could not pair the {t} wells with the "
                    f"{reference} wells by column; the axes stay independent",
                )
            )
            return None
        pair, offsets = got
        tie[t] = (pair, offsets if auto else np.full_like(offsets, delta))
    return tie or None


def _plot_vanthoff(models: dict[str, VanthoffModel], pdf_path: Path) -> None:
    """Per sample: the banded curves with the joint model, and the constant vs 1/T."""
    n = len(models)
    fig, axes = plt.subplots(
        n, 2, figsize=(10.8, 4.2 * n), squeeze=False, constrained_layout=True
    )
    for row, (sample, info) in enumerate(sorted(models.items())):
        left, right = axes[row]
        temps = sorted(info.k_by_temp)
        ax_labels = vanthoff_axes(info)
        colmap = {t: plt.get_cmap("tab10")(i / 10) for i, t in enumerate(temps)}
        for (band, t), (x, y) in sorted(info.curves.items()):
            if ax_labels.left_log:
                # a zero concentration has no place on a log axis: it is
                # drawn at the axis floor, as the band figure does
                px = np.where(x == 0, ZERO_AT, x)
                grid = np.logspace(
                    np.log10(ZERO_AT), np.log10(max(float(np.max(x)), 1.0) * 1.5), 200
                )
            else:
                px = x
                grid = np.linspace(float(np.min(x)), float(np.max(x)), 200)
            left.plot(px, y, "o", ms=3.5, color=colmap[t], alpha=0.85)
            left.plot(
                grid,
                vanthoff_curve(info, t, grid, info.plateaus[band, t]),
                "-",
                lw=1.1,
                color=colmap[t],
            )
        for t in temps:
            left.plot([], [], "o", color=colmap[t], label=f"{t} C")
        left.legend(fontsize=8)
        left.set_title(f"{sample}: bands and joint model")
        left.set_xlabel(ax_labels.left_x)
        left.set_ylabel(ax_labels.left_y)
        if ax_labels.left_log:
            left.set_xscale("log")
        xs = np.array([info.inv[t] * 1000 for t in temps])
        points = [vanthoff_right_point(info, t) for t in temps]
        ys = np.array([p[0] for p in points])
        ses = np.array([p[1] for p in points])
        right.errorbar(xs, ys, yerr=ses, fmt="o", color="k")
        line = np.linspace(float(xs.min()), float(xs.max()), 50)
        right.plot(
            line,
            vanthoff_right_line(info, line / 1000),
            "-k",
            lw=1,
        )
        for i, t in enumerate(temps):
            right.annotate(
                f"{t} C",
                (xs[i], ys[i]),
                textcoords="offset points",
                xytext=(6, -12),
                fontsize=8,
            )
        right.set_title(f"van't Hoff: dH = {vanthoff_enthalpy(info) / 1000:.1f} kJ/mol")
        right.set_xlabel("1000 / T  (1/K)")
        right.set_ylabel(ax_labels.right_y)
        if ax_labels.right_log:
            right.set_yscale("log")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def run_vanthoff(  # ruff: ignore[too-many-arguments]
    ef: EnspireFile,
    note_fp: Path,
    out_dir: Path,
    *,
    buffer: bool,
    isosbestic: float | str | None = None,
    provenance: Mapping[str, object] | None = None,
) -> EnspireRunResult[NamedTuple]:
    """Fit a note jointly across temperatures by van't Hoff and write the table.

    Parameters
    ----------
    ef : EnspireFile
        The parsed EnSpire export.
    note_fp : Path
        The note describing wells, titrant and samples.
    out_dir : Path
        Where the table goes.
    buffer : bool
        Subtract the buffer well of the same plate column.

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
        One row per (sample, temperature), with the shared ``A``, ``B``,
        ``dH_J_mol`` and ``dS_J_molK`` repeated on each row; the files written;
        and the messages.
    """
    before = _snapshot(out_dir)
    messages: list[EnspireMessage] = []
    note_df = Note(note_fp).note
    table, models = vanthoff_titration(
        ef, note_df, buffer=buffer, isosbestic=isosbestic
    )
    if not models:
        seen = sorted({str(t).split(".")[0] for t in note_df["Temp"].dropna()})
        messages.append(
            EnspireMessage(
                "note",
                f"vanthoff: the note carries temperature(s) {seen}; a joint fit "
                "needs at least two - nothing written, use --method band or global",
            )
        )
        return EnspireRunResult(
            written=_written_since(out_dir, before),
            table=pd.DataFrame(),
            summaries=(),
            messages=tuple(messages),
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = note_fp.name.removesuffix("_note.csv")
    table.to_csv(out_dir / f"{stem}_K_vanthoff.csv", index=False)
    write_provenance(
        out_dir / f"{stem}_K_vanthoff.json",
        inputs={"data": str(ef.file), "note": str(note_fp)},
        options=provenance,
        outputs=[f"{stem}_K_vanthoff.csv", f"{stem}_K_vanthoff.pdf"],
    )
    for r in table[table["K"].notna()].itertuples():
        kind, unit = ("pKa", "") if models[str(r.sample)].is_ph else ("Kd", " mM")
        messages.append(
            EnspireMessage(
                "info",
                f"{r.sample} at {r.temp}: {kind} {r.K:.3f}{unit} +/- {r.se:.3f} "
                f"(dH {cast('float', r.dH_J_mol) / 1000:.1f} kJ/mol) over {r.n} points",
            )
        )
    _plot_vanthoff(models, out_dir / f"{stem}_K_vanthoff.pdf")
    return EnspireRunResult(
        written=_written_since(out_dir, before),
        table=table,
        summaries=(),
        messages=tuple(messages),
    )


def _trace_draws(posterior: xr.Dataset, var: str) -> np.ndarray:
    """Flatten one trace variable to (draws, ...), whatever the chain layout."""
    arr = np.asarray(posterior[var].values)
    return arr.reshape(-1, *arr.shape[2:])


def _plot_vanthoff_bayes(  # ruff: ignore[too-many-statements]
    figdata: list[
        tuple[
            str, bool, dict[tuple[str, str], tuple[np.ndarray, np.ndarray]], Any, float
        ]
    ],
    pdf_path: Path,
) -> None:
    """Per sample: bands with the posterior-median model and its 94 % band, and K vs 1/T."""
    if not figdata:
        return
    n = len(figdata)
    fig, axes = plt.subplots(
        n, 2, figsize=(10.8, 4.2 * n), squeeze=False, constrained_layout=True
    )
    for row, (sample, is_ph, curves, trace, m) in enumerate(figdata):
        left, right = axes[row]
        ax_labels = vanthoff_axes_for(is_ph=is_ph)
        a_draws = _trace_draws(trace.posterior, "A")
        b_draws = _trace_draws(trace.posterior, "B")
        keep = np.linspace(0, a_draws.size - 1, min(200, a_draws.size)).astype(int)
        temps = sorted({t for _band, t in curves})
        colmap = {t: plt.get_cmap("tab10")(i / 10) for i, t in enumerate(temps)}
        for (band, t), (x, y) in sorted(curves.items()):
            if ax_labels.left_log:
                # a zero concentration has no place on a log axis: it is drawn
                # at the axis floor, as the least-squares figure does
                px = np.where(x == 0, ZERO_AT, x)
                grid = np.logspace(
                    np.log10(ZERO_AT), np.log10(max(float(np.max(x)), 1.0) * 1.5), 200
                )
            else:
                px = x
                grid = np.linspace(float(np.min(x)), float(np.max(x)), 200)
            left.plot(px, y, "o", ms=3.5, color=colmap[t], alpha=0.85)
            pk = a_draws + b_draws * (1.0 / (float(t) + ZERO_C) - m)
            kd = np.power(10.0, pk)[:, None]
            frac = (
                1.0 / (1.0 + np.power(10.0, pk[:, None] - grid[None, :]))
                if is_ph
                else grid[None, :] / (grid[None, :] + kd)
            )
            s0 = _trace_draws(trace.posterior, f"s0_{band}_{t}")
            s1 = _trace_draws(trace.posterior, f"s1_{band}_{t}")
            drawn = s0[:, None] + (s1 - s0)[:, None] * frac
            lo, hi = np.quantile(drawn[keep], [0.03, 0.97], axis=0)
            left.fill_between(grid, lo, hi, color=colmap[t], alpha=0.15, lw=0)
            left.plot(grid, np.median(drawn, axis=0), "-", lw=1.1, color=colmap[t])
        for t in temps:
            left.plot([], [], "o", color=colmap[t], label=f"{t} C")
        left.legend(fontsize=8)
        left.set_title(f"{sample}: bands and posterior-median joint model")
        left.set_xlabel(ax_labels.left_x)
        left.set_ylabel(ax_labels.left_y)
        if ax_labels.left_log:
            left.set_xscale("log")

        xs = np.array([1000.0 / (float(t) + ZERO_C) for t in temps])
        med = np.array([
            float(np.median(_trace_draws(trace.posterior, f"K_{t}"))) for t in temps
        ])
        q = np.array([
            np.quantile(_trace_draws(trace.posterior, f"K_{t}"), [0.03, 0.97])
            for t in temps
        ])
        right.errorbar(
            xs,
            med,
            yerr=np.vstack([med - q[:, 0], q[:, 1] - med]),
            fmt="o",
            color="k",
            capsize=2,
        )
        # the line of the joint model, with the credible band of the pair
        grid_inv = np.linspace(
            1.0 / (float(temps[-1]) + ZERO_C), 1.0 / (float(temps[0]) + ZERO_C), 100
        )
        fitted = a_draws[keep, None] + b_draws[keep, None] * (grid_inv[None, :] - m)
        plotted = fitted if is_ph else np.power(10.0, fitted)
        right.fill_between(
            grid_inv * 1000,
            *np.quantile(plotted, [0.03, 0.97], axis=0),
            color="0.6",
            alpha=0.3,
            lw=0,
        )
        right.plot(grid_inv * 1000, np.median(plotted, axis=0), "-k", lw=1)
        for i, t in enumerate(temps):
            right.annotate(
                f"{t} C",
                (xs[i], med[i]),
                textcoords="offset points",
                xytext=(6, -12),
                fontsize=8,
            )
        dh = _trace_draws(trace.posterior, "dH_J_mol") / 1000.0
        dh_lo, dh_hi = np.quantile(dh, [0.03, 0.97])
        right.set_title(
            f"van't Hoff: dH = {np.median(dh):.1f} [{dh_lo:.1f}, {dh_hi:.1f}] kJ/mol"
        )
        right.set_xlabel("1000 / T  (1/K)")
        right.set_ylabel(ax_labels.right_y)
        if ax_labels.right_log:
            right.set_yscale("log")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def run_vanthoff_bayes(  # ruff: ignore[too-many-arguments]
    ef: EnspireFile,
    note_fp: Path,
    out_dir: Path,
    *,
    buffer: bool,
    x_sigma: str | None,
    seed: int,
    draws: int,
    chains: int,
    cores: int | None,
    target_accept: float | None,
    isosbestic: float | str | None = None,
    provenance: Mapping[str, object] | None = None,
    dph_buffer: str | None = None,
) -> EnspireRunResult[NamedTuple]:
    """Sample the joint van't Hoff fit and write its tables and figure.

    One posterior per sample carries every temperature at once, as the
    least-squares joint fit does: ``A`` and ``B`` of ``pK(T) = A + B (1/T - m)``
    are shared by all bands and temperatures, the plateaus stay free per
    ``(band, temperature)``. The K table keeps the columns of the Bayesian
    tables elsewhere (``K``, ``hdi03``, ``hdi97``, ``r_hat``, ``ess_bulk``,
    ``n``, ``x_sigma``) and adds ``dH_J_mol``/``dS_J_molK``, the posterior
    medians of the thermodynamic pair that the least-squares table reports
    with propagated, not sampled, errors.

    Parameters
    ----------
    ef : EnspireFile
        The parsed EnSpire export.
    note_fp : Path
        The note describing wells, titrant and samples.
    out_dir : Path
        Where the tables, traces and figure go.
    buffer : bool
        Subtract the buffer well of the same plate column.
    x_sigma : str | None
        ``--x-sigma`` as given: None keeps the axis fixed at the note values,
        ``auto`` pools the note's replicate spread, a number fixes the prior
        width in the titrant's own units.
    seed : int
        NUTS seed, so the posterior reproduces.
    draws : int
        Posterior draws per chain.
    chains : int
        MCMC chains.
    cores : int | None
        CPU cores for chain parallelism; None lets PyMC parallelise.
    target_accept : float | None
        NUTS target acceptance; None keeps the sampler default.

    isosbestic : float | str | None
        Normalise every well by its signal at the isosbestic point before
        fitting: a wavelength [nm], ``"auto"`` to locate it per group, or
        ``None`` (see :func:`~clophfit.prenspire.spectral.plate_isosbestic`).

    provenance : Mapping[str, object] | None
        The options this run was asked for, written beside the tables as JSON
        together with the library version and the input files.
    dph_buffer : str | None
        ``"auto"`` ties each temperature's latent axis to the lowest one plus that
        column's recorded pH difference; a number uses it for every column; ``None``
        leaves the axes independent.

    Returns
    -------
    EnspireRunResult[NamedTuple]
        One row per (sample, temperature); the files written; and the messages.

    Raises
    ------
    EnspireRunError
        When ``--x-sigma`` was given but cannot be resolved for a sample.
    """
    before = _snapshot(out_dir)
    messages: list[EnspireMessage] = []
    note = Note(note_fp).note
    stem = note_fp.name.removesuffix("_note.csv")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    x_rows: list[dict[str, object]] = []
    figdata: list[
        tuple[
            str, bool, dict[tuple[str, str], tuple[np.ndarray, np.ndarray]], Any, float
        ]
    ] = []
    for name, info in vanthoff_curves(
        ef, note, buffer=buffer, isosbestic=isosbestic
    ).items():
        curves = info.curves
        temps = sorted({t for _band, t in curves})
        if len(temps) < _MIN_VANTHOFF_TEMPS:
            messages.append(
                EnspireMessage(
                    "note",
                    f"vanthoff-bayes: {name} carries temperature(s) {temps}; a joint "
                    "posterior needs at least two - skipped",
                )
            )
            continue
        # the axis prior is one number for the sample: 'auto' pools the
        # replicate spread of every temperature's wells, which is the same
        # error model the per-temperature fitters use
        one = note[note["Name"] == name]
        sigma = resolve_x_sigma(x_sigma, info.is_ph, one, str(name), temps[0], messages)
        if x_sigma is not None and sigma is None:
            msg = "--method vanthoff-bayes could not resolve --x-sigma for this note"
            raise EnspireRunError(msg)
        axis_tie = _vanthoff_axis_tie(
            note, str(name), info.wells, temps, dph_buffer, messages
        )
        trace = fit_vanthoff_model_bayes(
            curves,
            is_ph=info.is_ph,
            x_sigma=sigma,
            axis_tie=axis_tie,
            seed=seed,
            draws=draws,
            chains=chains,
            cores=cores,
            target_accept=target_accept or 0.9,
        )
        trace.to_netcdf(out_dir / f"{stem}_{name}_vanthoff_bayes.nc")
        n = int(sum(np.asarray(pair[0]).size for pair in curves.values()))
        dh_draws = np.asarray(trace.posterior["dH_J_mol"].values).reshape(-1)
        ds_draws = np.asarray(trace.posterior["dS_J_molK"].values).reshape(-1)
        for t in temps:
            kdraw = np.asarray(trace.posterior[f"K_{t}"].values).reshape(-1)
            lo, hi = (float(v) for v in az.hdi(kdraw, prob=0.94))
            median = float(np.median(kdraw))
            diag = trace_diagnostics(trace, summary_var_names=[f"K_{t}"])
            rows.append({
                "sample": name,
                "temp": t,
                "K": median,
                "hdi03": lo,
                "hdi97": hi,
                "r_hat": float(az.rhat(trace, var_names=[f"K_{t}"])[f"K_{t}"].item()),
                "ess_bulk": float(az.ess(trace, var_names=[f"K_{t}"])[f"K_{t}"].item()),
                "dH_J_mol": float(np.median(dh_draws)),
                "dS_J_molK": float(np.median(ds_draws)),
                "n": n,
                "x_sigma": np.nan if sigma is None else sigma,
                **posterior_columns(diag),
            })
            kind, unit = ("pKa", "") if info.is_ph else ("Kd", " mM")
            messages.append(
                EnspireMessage(
                    "info",
                    f"{name} at {t}: {kind} {median:.3f}{unit} 94% HDI [{lo:.3f}, {hi:.3f}]"
                    + posterior_screen(diag),
                )
            )
            if sigma is None:
                continue
            xdraws = np.asarray(trace.posterior[f"x_true_{t}"].values)
            draws_x = xdraws.reshape(-1, xdraws.shape[-1])
            # the wells that survived the blank and monotonicity gates, in the
            # order the sampler carries them
            for i, well in enumerate(info.wells[t]):
                qlo, xmed, qhi = np.quantile(draws_x[:, i], [0.03, 0.5, 0.97])
                x_rows.append({
                    "sample": name,
                    "temp": t,
                    "well": well,
                    "x_median": float(xmed),
                    "hdi03": float(qlo),
                    "hdi97": float(qhi),
                })
        m = float(np.mean([1.0 / (float(t) + ZERO_C) for t in temps]))
        figdata.append((str(name), info.is_ph, curves, trace, m))
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / f"{stem}_K_vanthoff_bayes.csv", index=False)
    write_provenance(
        out_dir / f"{stem}_K_vanthoff_bayes.json",
        inputs={"data": str(ef.file), "note": str(note_fp)},
        options=provenance,
        outputs=[f"{stem}_K_vanthoff_bayes.csv", f"{stem}_K_vanthoff_bayes.pdf"],
    )
    if x_rows:
        pd.DataFrame(x_rows).to_csv(
            out_dir / f"{stem}_x_vanthoff_bayes.csv", index=False
        )
    _plot_vanthoff_bayes(figdata, out_dir / f"{stem}_K_vanthoff_bayes.pdf")
    return EnspireRunResult(
        written=_written_since(out_dir, before),
        table=table,
        summaries=(),
        messages=tuple(messages),
    )


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
