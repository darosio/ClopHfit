"""Band Bayesian runs (``--method bayes``) and their posterior figure."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, cast

import arviz as az  # type: ignore[import-untyped]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from clophfit.fitting.data_structures import DataArray, Dataset
from clophfit.fitting.model_validation import trace_diagnostics
from clophfit.fitting.models import binding_1site
from clophfit.prenspire import (
    EnspireFile,
    Note,
    bands as bands_module,
)
from clophfit.prenspire.provenance import write_provenance

if TYPE_CHECKING:
    from collections.abc import Mapping
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
    EnspireRunResult,
    _snapshot,
    _written_since,
)

_MIN_BAYES_WELLS_AFTER_SCREEN = 6  # below this the shared latent axis is not identified


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
