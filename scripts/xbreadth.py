"""Wide verification of --x-sigma: robustness, reproducibility, residuals.

Fits every (sample, temperature) block of a (plate, note) pair with the
shared-K band posterior under several x_sigma settings and two seeds, then
reports per block:
- posterior K / HDI94 / r_hat / ess (convergence);
- residual diagnostics at the posterior mean per band (RMSE, |z| in units of
  the posterior noise scale, trend vs x, lag-1 autocorrelation, sign-flip
  ratio vs the random expectation of 0.5);
- reproducibility: |delta K| and HDI overlap between seed 7 and seed 42;
- LM band fit (point estimate + SE) as the cross-method reference.

Usage
-----
uv run python scripts/xbreadth.py PLATE.csv NOTE.csv [NOTE.csv ...] \
    --out DIR [--draws 4000] [--chains 4] [--cores 4]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from clophfit.__main__ import _resolve_enspire_x_sigma
from clophfit.fitting.data_structures import DataArray, Dataset
from clophfit.fitting.models import binding_1site
from clophfit.prenspire import Note, bands
from clophfit.prenspire.prenspire import EnspireFile

REPRO_SETTINGS = ("fixed", "auto")
SEEDS = (7, 42)


def _blocks(
    ef: EnspireFile, note_fp: Path, buffer: bool = True
) -> list[dict[str, object]]:
    """One record per (sample, temperature) block: x, normalized band data.

    Mirrors the block construction of ``fit_enspire_bayes`` (same CLI
    pipeline) so the numbers are the CLI's numbers.
    """
    note_df = Note(note_fp).note
    samples = note_df[note_df["Name"] != "buffer"]
    buffers = note_df[note_df["Name"] == "buffer"]
    wanted = bands.DIRECT_BANDS
    out: list[dict[str, object]] = []
    for (name, temp), group in samples.groupby(
        ["Name", "Temp"], sort=True, dropna=False
    ):
        t = str(temp).split(".")[0]
        one = group.drop_duplicates("Well")
        wells = one["Well"].tolist()
        is_ph = bool(one["pH"].nunique() > one["Cl"].nunique())
        x = np.array((one["pH"] if is_ph else one["Cl"]).astype(float), dtype=np.float64)
        bufw = (
            buffers[np.isclose(buffers["Temp"].astype(float), float(t))]["Well"].tolist()
            if len(buffers)
            else []
        )
        readouts = bands.classify(ef, t, is_ph=is_ph)
        data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for r in readouts:
            if r.name not in wanted:
                continue
            y = bands.band_values(ef, r, wells)
            if bufw and buffer:
                y = bands.subtract_buffer(y, bands.band_values(ef, r, bufw))
            arr = np.array(y, dtype=np.float64)
            top = float(np.nanmax(np.abs(arr)))
            if not np.isfinite(top) or top <= 0.0:
                continue
            data[r.name] = (x, arr / top)
        if not data:
            continue
        out.append({
            "name": str(name),
            "temp": t,
            "is_ph": is_ph,
            "n": int(x.size),
            "x": x,
            "one": one,
            "data": data,
        })
    return out


def _lm_ref(block: dict[str, object]) -> dict[str, float | None]:
    data = block["data"]
    try:
        fit = bands.fit_bands(
            {nm: (xx, yy) for nm, (xx, yy) in data.items()},
            is_ph=bool(block["is_ph"]),
        )
    except Exception:  # ruff: ignore[blind-except] - LM may reject sparse blocks
        return {"k_lm": None, "se_lm": None, "rms_lm": None}
    if fit is None:
        return {"k_lm": None, "se_lm": None, "rms_lm": None}
    return {"k_lm": fit.k, "se_lm": fit.se, "rms_lm": fit.rms}


def _residuals(
    block: dict[str, object], bb: object, has_x_true: bool
) -> dict[str, float | None]:
    """Deterministic residual diagnostics at the posterior mean.

    y_pred uses the same ``binding_1site`` call the model itself uses, with
    the posterior medians of K, S0, S1, and (if latent) the median of
    x_true; the noise scale is the median of ``ye_mag_<band>``.
    """
    trace = bb.trace  # az.InferenceData / DataTree
    post = trace.posterior
    K = float(np.median(post["K"].values))
    x_true = (
        np.median(post["x_true"].values, axis=(0, 1)) if has_x_true else None
    )
    data = block["data"]
    per: list[dict[str, float]] = []
    for nm, (xx, yy) in data.items():
        s0 = float(np.median(post[f"S0_{nm}"].values))
        s1 = float(np.median(post[f"S1_{nm}"].values))
        x_used = np.asarray(x_true, dtype=np.float64) if has_x_true else np.asarray(xx, dtype=np.float64)
        ye_var = post[f"ye_mag_{nm}"].values
        if ye_var.ndim == 3:  # per-well scale (chain, draw, n)
            scale_arr = np.maximum(np.abs(np.median(ye_var, axis=(0, 1))), 1e-9)
        else:  # shared scalar scale (chain, draw)
            scale_arr = np.full(x_used.shape, max(float(np.median(ye_var)), 1e-9))
        pred = binding_1site(x_used, K, s0, s1, is_ph=bool(block["is_ph"]))
        r = np.asarray(yy, dtype=np.float64) - np.asarray(pred, dtype=np.float64)
        order = np.argsort(x_used)
        rr = r[order]
        flips = float(np.mean(np.sign(rr[1:]) != np.sign(rr[:-1])))
        lag1 = float(pearsonr(rr[1:], rr[:-1])[0]) if np.std(rr) > 0 else float("nan")
        try:
            rho = float(spearmanr(x_used, r)[0])
        except ValueError:
            rho = float("nan")
        per.append({
            "band": nm,
            "rmse": float(np.sqrt(np.mean(r**2))),
            "mean_abs_z": float(np.mean(np.abs(r) / scale_arr)),
            "max_abs_z": float(np.max(np.abs(r) / scale_arr)),
            "trend_rho": rho,
            "lag1": lag1,
            "flip_ratio": flips,
        })
    def agg(key: str) -> float:
        vals = [p[key] for p in per if np.isfinite(p[key])]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "rmse": agg("rmse"),
        "mean_abs_z": agg("mean_abs_z"),
        "max_abs_z": agg("max_abs_z"),
        "trend_rho": agg("trend_rho"),
        "lag1": agg("lag1"),
        "flip_ratio": agg("flip_ratio"),
    }


def _hdi_overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    width = min(a[1] - a[0], b[1] - b[0])
    if width <= 0:
        return float("nan")
    return max(0.0, hi - lo) / width


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("plate", type=Path)
    ap.add_argument("notes", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--draws", type=int, default=4000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--cores", type=int, default=4)
    ap.add_argument(
        "--settings",
        default="fixed,sig05,auto,sig15",
        help="Comma-separated subset of settings to run.",
    )
    args = ap.parse_args()

    ef = EnspireFile(args.plate)
    rows: list[dict[str, object]] = []
    per_block_lm: dict[tuple[str, str], dict[str, float | None]] = {}

    settings_all = [
        ("fixed", None),
        ("sig05", 0.05),
        ("auto", "auto"),
        ("sig15", 0.15),
    ]
    wanted = {s.strip() for s in args.settings.split(",") if s.strip()}
    settings = [s for s in settings_all if s[0] in wanted]
    for note_fp in args.notes:
        for block in _blocks(ef, note_fp):
            key = (str(block["name"]), str(block["temp"]))
            if key not in per_block_lm:
                per_block_lm[key] = _lm_ref(block)
            for label, xs in settings:
                for seed in (SEEDS if label in REPRO_SETTINGS else (SEEDS[0],)):
                    x_res = _resolve_enspire_x_sigma(
                        xs if isinstance(xs, str) else None,
                        bool(block["is_ph"]),
                        block["one"],  # type: ignore[arg-type]
                        str(block["name"]),
                        str(block["temp"]),
                    )
                    if xs is not None and not isinstance(xs, str) and x_res is None:
                        x_res = float(xs)
                    ds = Dataset(
                        {
                            nm: DataArray(xx, yy)
                            for nm, (xx, yy) in block["data"].items()
                        },
                        is_ph=bool(block["is_ph"]),
                    )
                    bb = bands.fit_bands_bayes(
                        ds,
                        seed=seed,
                        draws=args.draws,
                        chains=args.chains,
                        cores=args.cores,
                        x_sigma=x_res,
                    )
                    res = _residuals(block, bb, x_res is not None)
                    row = {
                        "note": note_fp.name,
                        "block": key[0],
                        "temp": key[1],
                        "is_ph": block["is_ph"],
                        "n": block["n"],
                        "setting": label,
                        "seed": seed,
                        "x_sigma": x_res,
                        "K": float(bb.k),
                        "hdi03": float(bb.hdi03),
                        "hdi97": float(bb.hdi97),
                        "r_hat": float(bb.rhat),
                        "ess": float(bb.ess),
                        "converged": bool(bb.rhat <= 1.01 and bb.ess >= 100.0),
                    }
                    row.update({f"res_{k}": v for k, v in res.items()})
                    rows.append(row)
                    print(
                        f"{key[0]}@{key[1]} {label:6s} seed={seed} "
                        f"x_sigma={x_res!s:<6} K={float(bb.k):.4f} "
                        f"[{float(bb.hdi03):.4f}, {float(bb.hdi97):.4f}] "
                        f"rhat={float(bb.rhat):.4f} ess={float(bb.ess):.0f} "
                        f"rmse={res['rmse']:.4f} |z|={res['mean_abs_z']:.3f} "
                        f"trend={res['trend_rho']:.3f} lag1={res['lag1']:.3f} "
                        f"flip={res['flip_ratio']:.3f}",
                        flush=True,
                    )

    table = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    tag = re.sub(r"[^A-Za-z0-9]+", "_", args.plate.stem)
    for (label, lm) in per_block_lm.items():
        table.loc[
            (table.block == label[0]) & (table.temp == label[1]),
            ["k_lm", "se_lm", "rms_lm"],
        ] = lm
    table.to_csv(args.out / f"xbreadth_{tag}.csv", index=False)

    # Reproducibility pairs (fixed/auto, seed 7 vs 42).
    repro_rows = []
    for (blabel, btemp), g in table.groupby(["block", "temp"], sort=True):
        if g.shape[0] == 0:
            continue
        for setting in REPRO_SETTINGS:
            sub = g[g.setting == setting]
            if sub.seed.nunique() < 2:
                continue
            a = sub[sub.seed == SEEDS[0]].iloc[0]
            b = sub[sub.seed == SEEDS[1]].iloc[0]
            width = float(max(b.hdi97 - b.hdi03, 1e-9))
            repro_rows.append({
                "block": blabel,
                "temp": btemp,
                "setting": setting,
                "K_7": a.K,
                "K_42": b.K,
                "deltaK": float(a.K - b.K),
                "deltaK_over_width": float(a.K - b.K) / width,
                "hdi_overlap": _hdi_overlap(
                    (a.hdi03, a.hdi97), (b.hdi03, b.hdi97)
                ),
            })
    repro = pd.DataFrame(repro_rows)
    if not repro.empty:
        repro.to_csv(args.out / f"xbreadth_{tag}_repro.csv", index=False)

    print(f"\nWROTE {args.out / f'xbreadth_{tag}.csv'} rows={len(table)} "
          f"repro_rows={len(repro)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
