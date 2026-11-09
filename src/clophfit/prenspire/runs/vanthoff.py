"""van't Hoff runs: least-squares and Bayesian joint fits across temperatures."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple, cast

import arviz as az  # type: ignore[import-untyped]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from clophfit.fitting.bayes import column_axis_tie
from clophfit.fitting.model_validation import trace_diagnostics
from clophfit.prenspire import (
    EnspireFile,
    Note,
)
from clophfit.prenspire.bands import ZERO_AT
from clophfit.prenspire.provenance import write_provenance
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
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    import xarray as xr

    from clophfit.prenspire.thermo import VanthoffModel

from clophfit.prenspire.runs.bayes_common import (
    posterior_columns,
    posterior_screen,
    resolve_x_sigma,
)
from clophfit.prenspire.runs.common import (
    EnspireMessage,
    EnspireRunError,
    EnspireRunResult,
    _snapshot,
    _written_since,
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
