"""Whole-spectrum Bayesian runs (``--method global-bayes`` and ``-marginal``)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple, cast

import arviz as az  # type: ignore[import-untyped]
import numpy as np
import pandas as pd

from clophfit.fitting.bayes import well_groups
from clophfit.fitting.model_validation import trace_diagnostics
from clophfit.fitting.spectral import fit_spectra_global, posterior_predictive_check
from clophfit.prenspire import (
    EnspireFile,
    Note,
    bands as bands_module,
)
from clophfit.prenspire.figures import BayesPanelGroup, plot_global_bayes_panels
from clophfit.prenspire.provenance import write_provenance
from clophfit.prenspire.spectral import (
    model_outliers,
    plate_isosbestic,
    sample_spectra,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path


from clophfit.prenspire.runs.bayes_common import (
    lam_for,
    posterior_columns,
    posterior_screen,
    resolve_x_sigma,
    shared_isosbestic,
)
from clophfit.prenspire.runs.common import (
    EnspireMessage,
    EnspireRunError,
    EnspireRunResult,
    _snapshot,
    _written_since,
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
