#!/usr/bin/env python
"""Parameter recovery for the latent titrant axis: free, shared or pooled.

The whole-spectrum Bayesian fitters put a latent ``x_true`` on every well. When
the wells of one replicate group are one solution, the recorded axis differing
between them is *axis error*, not truth - so the model can either give every well
its own truth (``free``), give the group one truth (``shared``), or share a group
truth and let a well deviate with an inferred scale (``partial``).

This script measures what those choices cost. Synthetic plates carry three
replicate rows of one ladder sharing one fraction, with a per-row offset in the
*recorded* axis (``--shift``), and a known K. It reports, per cell, the bias, its
standard error over datasets, the RMSE, the 94 % interval coverage, the
divergences, the median bulk ESS and the worst rank r-hat.

Why a study rather than one fit: a single synthetic plate cannot separate a
0.02 pH effect from Monte Carlo error. The numbers quoted in ``ppr enspire
--help`` (pooling cuts the poisoned-axis bias from 0.028 to 0.022 and the clean
axis's 0.005 to 0.0001, with poor mixing for the pooled fits) came from 45
datasets per cell at 4 chains x 2000 draws. Small runs are for smoke tests only.

Usage
-----
    # reproduce the published table (long: ~1 h per structure)
    uv run python benchmarks/axis_pooling_recovery.py --datasets 45 \
        --shift 0.0 0.3 --structures free shared partial --out recovery.csv

    # a quick health check that the machinery still runs
    uv run python benchmarks/axis_pooling_recovery.py --datasets 3 --draws 300 \
        --tune 300 --structures free --shift 0.3
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

from clophfit.fitting.bayes import well_groups
from clophfit.fitting.models import binding_1site
from clophfit.fitting.spectral import fit_spectra_global_bayes

#: Truth every replicate of the synthetic plate is generated with.
K_TRUE = 7.400
#: Prior SD of the recorded axis, in pH: the model's measurement-error scale.
X_SIGMA = 0.12
#: Ladder of one replicate row; three rows are generated from it.
LADDER = np.array([6.0, 6.5, 7.0, 7.4, 7.8, 8.3, 8.8, 9.2])
#: Wavelength grid; coarser than a real scan to keep a study affordable.
LAM = np.arange(400.0, 520.0, 1.2)
_ANIONIC = np.exp(-0.5 * ((LAM - 490.0) / 15.0) ** 2)
_NEUTRAL = 0.5 * np.exp(-0.5 * ((LAM - 430.0) / 14.0) ** 2)


def make_plate(seed: int, shift: float, noise: float) -> tuple[
    np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]], np.ndarray
]:
    """One synthetic plate whose recorded axis is offset per replicate row.

    Parameters
    ----------
    seed : int
        Noise seed; the same seed gives the same plate at either shift.
    shift : float
        Recorded-axis offset, ``-shift`` on row A, ``0`` on row B and ``+shift``
        on row C, in pH. ``0.0`` is the clean-axis control.
    noise : float
        White noise SD on every wavelength, in the peak-normalised units the
        whole-spectrum fit consumes.

    Returns
    -------
    tuple
        The recorded axis, the spectra mapping and the replicate-group index.
    """
    x_true = np.tile(LADDER, 3)
    x_recorded = x_true + np.tile([-shift, 0.0, shift], LADDER.size)
    frac = np.asarray(binding_1site(x_true, K_TRUE, 0.0, 1.0, is_ph=True))
    rng = np.random.default_rng(seed)
    y = np.outer(_ANIONIC, 1.0 - frac) + np.outer(_NEUTRAL, frac)
    y = y + rng.normal(0.0, noise, y.shape)
    wells = [f"{row}0{i + 1}" for row in "ABC" for i in range(LADDER.size)]
    return x_recorded, {"excitation": (LAM, y)}, well_groups(wells)


def run(args: argparse.Namespace) -> pd.DataFrame:
    """Sample every (shift, structure, seed) cell and return the raw rows."""
    rows: list[dict[str, object]] = []
    for shift in args.shift:
        for seed in range(args.datasets):
            x_rec, spectra, groups = make_plate(seed, shift, args.noise)
            for pool in args.structures:
                started = time.time()
                idata = fit_spectra_global_bayes(
                    x_rec,
                    spectra,
                    x_sigma=args.x_sigma,
                    is_ph=True,
                    x_pool=pool,
                    x_groups=None if pool == "free" else groups,
                    species_init=args.inits,
                    species_prior_scale=args.prior_scale,
                    draws=args.draws,
                    chains=args.chains,
                    cores=args.cores,
                    tune=args.tune,
                    seed=seed,
                )
                k = np.asarray(idata.posterior["K"].values).reshape(-1)
                lo, hi = np.quantile(k, [0.03, 0.97])
                row = {
                    "shift": shift,
                    "seed": seed,
                    "pool": pool,
                    "init": args.inits,
                    "prior_scale": args.prior_scale,
                    "k_median": float(np.median(k)),
                    "lo": float(lo),
                    "hi": float(hi),
                    "div": int(np.asarray(idata.sample_stats["diverging"].values).sum()),
                    "ess": float(np.asarray(az.ess(idata, var_names=["K"])["K"])),
                    "rhat": float(np.asarray(az.rhat(idata, var_names=["K"])["K"])),
                    "secs": round(time.time() - started, 1),
                }
                rows.append(row)
                print(
                    f"shift {shift:.2f} seed {seed:3d} {pool:8s} "
                    f"K {row['k_median']:.4f} div {row['div']:3d} "
                    f"ess {row['ess']:.0f} rhat {row['rhat']:.3f}",
                    flush=True,
                )
    return pd.DataFrame(rows)


def summarise(frame: pd.DataFrame) -> pd.DataFrame:
    """Bias with its standard error, RMSE, coverage and sampler health per cell."""
    frame = frame.assign(
        bias=frame["k_median"] - K_TRUE,
        covered=(frame["lo"] <= K_TRUE) & (K_TRUE <= frame["hi"]),
    )
    return frame.groupby(["shift", "pool", "init", "prior_scale"]).agg(
        n=("bias", "size"),
        bias=("bias", "mean"),
        se=("bias", lambda s: s.std(ddof=1) / np.sqrt(len(s))),
        rmse=("bias", lambda s: float(np.sqrt((s**2).mean()))),
        coverage=("covered", "mean"),
        div=("div", "mean"),
        ess=("ess", "median"),
        rhat=("rhat", "max"),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=int, default=5, help="noise seeds per shift")
    parser.add_argument(
        "--shift", type=float, nargs="+", default=[0.0, 0.3], help="recorded-axis offsets"
    )
    parser.add_argument(
        "--structures",
        nargs="+",
        default=["free", "shared", "partial"],
        choices=["free", "shared", "partial"],
    )
    parser.add_argument(
        "--inits", default="prefit", choices=["prefit", "edges", "iterated"],
        help="species prior centre",
    )
    parser.add_argument(
        "--prior-scale", type=float, default=5.0, help="species prior width, x resid rms"
    )
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--x-sigma", type=float, default=X_SIGMA)
    parser.add_argument("--noise", type=float, default=3e-3)
    parser.add_argument("--out", type=Path, default=None, help="append raw rows here")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the study and print the summary table."""
    warnings.filterwarnings("ignore")
    args = parse_args(argv)
    frame = run(args)
    if args.out is not None:
        frame.to_csv(args.out, index=False)
        print(f"wrote {args.out}")
    print()
    print(summarise(frame).round(4).to_string())


if __name__ == "__main__":
    main()
