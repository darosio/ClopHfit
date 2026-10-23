"""Scientific evaluation of fit_mcr_constrained_bayes vs its alternatives.

Decision evidence for keeping ``--method mcr-constrained-bayes`` alongside
``global-bayes``: run ``uv run python benchmarks/mcr_bayes_constrained_eval.py``
and read the printed table (also stored in ``mcr_bayes_constrained_eval.csv``).

Data-generating process (the null the methods claim to fit):
    y(w, lam) = amp_w * [(1-f_w) E0(lam) + f_w E1(lam)] + b_w   (per-well offset)
              + AR(1) noise along wavelength + white noise
    f(x) = 1 / (1 + 10**(K - x)),   x_note = x_true + N(0, x_sigma)

The swept nuisance factors:
- relative noise scale: 0.02 / 0.06 / 0.12 (of the signal max),
- a per-well baseline offset (normalized data dips below 0, which
  non-negative species spectra cannot represent), tested at noise 0.06.

Methods compared on every dataset (20 seeds per cell, draws=250 chains=2):
- fit_spectra_global (LM, variable projection) + jackknife SE -> 95% CI,
- fit_spectra_global_bayes (free species, LM-seeded priors, adaptive sigma),
- fit_spectra_global_bayes_marginal (species integrated out; smooth prior,
  exact joint marginal via the matrix determinant lemma),
- fit_mcr_constrained_bayes (ALS-seeded truncated species, per-well
  wavelength-correlated MvStudent-t likelihood).

Metrics per cell: bias, RMSE, 95%-coverage of K_true (the project's top
criterion), and mean interval width. The raw CSV also carries each Bayesian
fit's K R-hat and bulk ESS so sampler health is part of the record.

Recorded results (run 2026-10-23, re-run after the constrained model's
sampler fixes; CSVs in this directory): the DGP shares one systematic
negative bias (about -0.07 to -0.12) with every method - the fit receives a
x jittered by X_SIGMA/3 plus an x_sigma prior of X_SIGMA, and that
attenuation drags K for the LM, the free-species Bayes, the marginal and the
constrained alike - so low-noise cells cannot show nominal 95 % coverage for
anyone.

- ``mcr_bayes_constrained_eval.csv`` (LM, ``global-bayes``,
  ``global-bayes-marginal``, ``mcr-constrained-bayes``): under this
  AR(1)-correlated wavelength noise the whole-spectrum ``global-bayes``
  undercovers (0.30-0.85 observed 95 % coverage), its per-point sigma prior
  being anchored at the LM rms and unable to absorb correlated residuals;
  ``mcr-constrained-bayes`` has the smallest bias and RMSE in the
  noise>=0.06 cells and holds the nominal side of the shared-bias ceiling
  (0.80-0.95 coverage from noise 0.06 up) with widths in the family's range
  (0.24-0.39 at noise 0.06-0.12). At noise=0.02 it undercovers (0.15) like
  everything else - its honest width (0.16) cannot cover the -0.11 shared
  bias. Sampler quality is recorded run-by-run in the raw CSV (K R-hat and
  bulk ESS): at this deliberately cheap shared budget (draws=250, chains=2,
  same for every method) the medians sit at 1.004-1.063 and nothing reaches
  a lost or label-flipped chain (max 1.36) - against 2-3 for the pre-fix
  model on real plates. The production defaults (4 chains, 4000 draws,
  low-rank mass adaptation) mix far harder than these floor settings.
- ``marginal_bayes_coverage_eval.csv`` (``global-bayes-marginal`` with the
  per-well smooth-noise block vs ``mcr-constrained-bayes``): the
  marginalized fitter improves on its LM-mean-only predecessor yet still
  undercovers (0.25-0.70) with intervals that are too narrow (width
  0.14-0.27) and the same small negative bias. Until that improves,
  ``mcr-constrained-bayes`` remains the recommended method when
  wavelength-correlated noise dominates.
- Sampler-fix history: the previously recorded constrained rows here
  (coverage 0.85-0.95, widths 0.37-0.43) came from the pre-fix model whose
  unseeded species prior let chains split between mirrored label modes and
  whose iid Normal likelihood froze chains on hair-thin K slices (R-hat up
  to 2.9 on real plates); the "conservative widths" were a lost-sampler
  artifact, not a property of the posterior. The re-run above is the
  decision-relevant record. On the real G10 plate the fixed model reports
  K 7.90 (R-hat 1.02, no divergences; 7.905/7.908 across independent runs
  with low-rank mass adaptation), with the whole-spectrum methods around
  7.5-7.6 - a systematic note-axis-vs-curve tension of a few tenths the
  sampler now settles reproducibly, where the pre-fix model reported a
  label-flipped mode at 8.7 with R-hat 2.9.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import arviz as az  # type: ignore[import-untyped]

from clophfit.fitting.spectral import (
    fit_spectra_global,
    fit_spectra_global_bayes,
    fit_spectra_global_bayes_marginal,
)
from clophfit.prenspire.mcr import fit_mcr_constrained_bayes

N_WELLS = 10
N_LAMBDA = 30
K_TRUE = 7.2
X_SIGMA = 0.08
AMPS = np.linspace(1.18, 0.86, N_WELLS)  # protein drift across the plate
TUNE = 100
DRAWS = 250

_E_GRID = np.linspace(-2, 2, N_LAMBDA)
E0 = np.exp(-0.5 * ((_E_GRID - 0.8) / 0.9) ** 2) * 0.85
E1 = np.exp(-0.5 * ((_E_GRID + 0.9) / 1.1) ** 2) * 0.95 + 0.08

# switch off the marginalized model on machines without the extra minutes
RUN_MARGINAL = True


def one_plate(seed: int, noise: float, offset: float) -> tuple[np.ndarray, np.ndarray]:
    """Build (x_note, Y[n_lambda, n_wells]) for one simulated plate row."""
    rng = np.random.default_rng(seed)
    x_true = np.linspace(5.5, 8.6, N_WELLS)
    x_note = x_true + rng.normal(0.0, X_SIGMA, N_WELLS)
    f = 1 / (1 + 10 ** (K_TRUE - x_true))
    signal = (1 - f)[:, None] * E0 + f[:, None] * E1
    signal *= AMPS[:, None]
    # smooth AR(1) noise along wavelength: neighbouring wavelengths' errors share
    ar = np.empty_like(signal)
    for w in range(signal.shape[0]):
        base = rng.normal(0, noise, N_LAMBDA)
        for i in range(1, N_LAMBDA):
            base[i] = 0.55 * base[i - 1] + np.sqrt(1 - 0.55**2) * rng.normal(0, noise)
        ar[w] = base
    data = (
        signal * (1 + ar)
        + offset * rng.normal(0, 1, (N_WELLS, 1))
        + rng.normal(0, noise / 3, signal.shape)
    )
    return x_note, data.T  # Y[n_lambda, n_wells], the package convention


def run() -> pd.DataFrame:
    rows = []
    cells = [*((n, 0.0) for n in (0.02, 0.06, 0.12)), (0.06, 0.05)]
    for noise, offset in cells:
        tag = f"noise={noise:g}{'+offset' if offset else ''}"
        print(tag, flush=True)
        for seed in range(20):
            x, y = one_plate(1000 + seed, noise, offset)
            top = max(float(np.max(np.abs(y))), 1e-9)
            d = y.T / top  # D[well, feature] for the MCR factorization
            rngm = np.random.default_rng(seed)
            x_jit = x + rngm.normal(0, X_SIGMA / 3, x.size)
            spec = {"exc": (np.arange(N_LAMBDA, dtype=np.float64), y)}

            def row(
                method: str, k: float, lo: float, hi: float,
                r_hat: float | None = None, ess: float | None = None,
            ) -> dict:
                return {
                    "cell": tag, "seed": seed, "method": method,
                    "K": float(k), "lo": float(lo), "hi": float(hi),
                    "r_hat": float(r_hat) if r_hat is not None else np.nan,
                    "ess_bulk": float(ess) if ess is not None else np.nan,
                }

            lm = fit_spectra_global(x_jit, spec, jackknife=True)
            rows.append(row("global-lm", lm.K, lm.K - 1.96 * lm.se, lm.K + 1.96 * lm.se))

            gb = fit_spectra_global_bayes(
                x_jit, spec, x_sigma=X_SIGMA, seed=seed, draws=DRAWS, chains=2, cores=1,
                tune=TUNE,
            )
            kgb = np.asarray(gb.posterior["K"].values).reshape(-1)
            lo, hi = np.quantile(kgb, [0.025, 0.975])
            rows.append(row(
                "global-bayes", np.median(kgb), lo, hi,
                r_hat=az.rhat(gb, var_names=["K"])["K"].item(),
                ess=az.ess(gb, var_names=["K"])["K"].item(),
            ))
            del gb, kgb

            if RUN_MARGINAL:
                mg = fit_spectra_global_bayes_marginal(
                    x_jit, spec, x_sigma=X_SIGMA, seed=seed, draws=DRAWS, chains=2,
                    cores=1, tune=TUNE,
                )
                kmg = np.asarray(mg.posterior["K"].values).reshape(-1)
                lo, hi = np.quantile(kmg, [0.025, 0.975])
                rows.append(row(
                    "global-bayes-marginal", np.median(kmg), lo, hi,
                    r_hat=az.rhat(mg, var_names=["K"])["K"].item(),
                    ess=az.ess(mg, var_names=["K"])["K"].item(),
                ))
                del mg, kmg

            cb = fit_mcr_constrained_bayes(
                d, x_jit, np.zeros(N_LAMBDA, dtype=int), x_sigma=X_SIGMA,
                seed=seed, draws=DRAWS, chains=2, cores=1,
            )
            kcb = np.asarray(cb.posterior["K"].values).reshape(-1)
            lo, hi = np.quantile(kcb, [0.025, 0.975])
            rows.append(row(
                "mcr-constrained-bayes", np.median(kcb), lo, hi,
                r_hat=az.rhat(cb, var_names=["K"])["K"].item(),
                ess=az.ess(cb, var_names=["K"])["K"].item(),
            ))
            del cb, kcb, d
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (cell, method), g in df.groupby(["cell", "method"], observed=True):
        inside = ((g.lo <= K_TRUE) & (g.hi >= K_TRUE)).mean()
        out.append({
            "cell": cell, "method": method,
            "bias": (g.K - K_TRUE).mean(),
            "rmse": float(np.sqrt(((g.K - K_TRUE) ** 2).mean())),
            "coverage": float(inside),
            "width": (g.hi - g.lo).mean(),
        })
    return pd.DataFrame(out)


if __name__ == "__main__":
    raw = run()
    raw.to_csv("mcr_bayes_constrained_eval_raw.csv", index=False)
    summary = summarize(raw)
    summary.to_csv("mcr_bayes_constrained_eval.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
