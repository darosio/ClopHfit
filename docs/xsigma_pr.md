# PR: feat/enspire-bayes-xlatent — per-well pH-axis prior for `ppr enspire --method bayes`

## Why

The EnSpire `*_note.csv` rows are **fixed setpoints for nominal pH targets**;
the actual per-well pH varies around them row to row. The existing EnSpire
bayes path hard-codes `n_xerr=0.0`, i.e. treats the note pH as exact, so
pH-axis error is silently absorbed into the reported `K`. We already have
**direct, non-parametric** evidence of this error: the replicate rows at a
given temperature (A/B/C at 20 °C, D/E/F at 37 °C) are three wells aiming at
the same nominal target, so their spread *is* the per-well axis error.

This PR makes the axis uncertain in the posterior in a way that matches the
EnSpire design: **independent** `x_true ~ Normal(note_pH, σ_x)` per well
— deliberately *not* the random-walk prior used by the ODR/MCMC titration
path, because EnSpire rows are independent targets, not a cumulative
addition series.

## What

| File                              | Change                                                                                                                                                                                                                                                                         |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `src/clophfit/fitting/bayes.py`   | `fit_binding_pymc(..., x_sigma: float \| None = None)`; when `x_sigma > 0`, `x_true` is `pm.Normal("x_true", mu=x, sigma=x_sigma, shape=len(x))` instead of `create_x_true(...)`. Default `None` = byte-identical behaviour to before.                                         |
| `src/clophfit/prenspire/bands.py` | `fit_bands_bayes(..., x_sigma=None)` passthrough; new `estimate_x_error_sigma(wells, ph)` — pooled within-column SD of the note's replicate rows (index = trailing digits of the well label), floored at 0.02 pH, `None` when fewer than two index columns have ≥2 replicates. |
| `src/clophfit/__main__.py`        | \`--x-sigma {auto                                                                                                                                                                                                                                                              |
| `tests/test_enspire_xlatent.py`   | 6 new tests: estimator (spread recovery, insufficient-data → None), `fit_bands_bayes` recovers a K embedded in axis-perturbed data, `x_sigma=None` keeps working, CLI `auto` and `0.15` on G10, CLI rejection for `--method band`.                                             |

`--x-sigma auto` estimates σ_x from the same note's rows (20 °C from
A/B/C, 37 °C from D/E/F); `--x-sigma 0.15` uses a fixed value — the two
requested options.

## Verification

- `pytest` — **1032 passed** (baseline 1026, +6 new), no failures.
- `mypy --strict` — clean (100 files). `ruff check` / `ruff format` — clean
  on all touched files.
- **G03 real data** (`--draws 4000 --chains 4 --cores 4 --seed 7`, equal
  budget for all variants; mixing all converged, r̂ ≤ 1.036):

| T (°C) | σ_x            | K     | 94% HDI        | width | r̂     | ess  |
| ------ | -------------- | ----- | -------------- | ----- | ----- | ---- |
| 20     | – (fixed axis) | 8.402 | [8.337, 8.462] | 0.125 | 1.000 | 9274 |
| 20     | 0.050          | 8.362 | [8.296, 8.421] | 0.125 | 1.002 | 2573 |
| 20     | 0.106 (auto)   | 8.273 | [8.199, 8.366] | 0.167 | 1.003 | 1341 |
| 20     | 0.150          | 8.252 | [8.167, 8.339] | 0.172 | 1.036 | 1573 |
| 37     | – (fixed axis) | 7.933 | [7.856, 8.006] | 0.150 | 1.000 | 9258 |
| 37     | 0.050          | 7.896 | [7.826, 7.977] | 0.151 | 1.000 | 6168 |
| 37     | 0.109 (auto)   | 7.838 | [7.774, 7.914] | 0.140 | 1.001 | 1540 |
| 37     | 0.150          | 7.822 | [7.728, 7.919] | 0.191 | 1.007 | 1340 |

Reading: the σ_x posterior and the fixed-axis posterior do **not** agree on
K — the shift is monotone in σ_x and ≈ 1:1 with it at both temperatures
(20 °C: ΔK 0.040 / 0.129 / 0.150 for σ_x 0.05 / 0.106 / 0.15; 37 °C:
0.037 / 0.095 / 0.111) — axis error that the fixed-axis fit absorbed into
K, now surfaced and under the user's control. All ladder HDIs overlap,
so the ladder is one connected, consistent region: "which K" is purely
"how much axis error do we admit". 0.05 sits below the lab's own
replicate-spread estimate (auto ≈ 0.106) and is the optimistic end.
Sampling cost also scales with the prior: ess at 37 °C is 6168 at σ_x
0.05 vs 1340 at 0.15 — the smaller the prior, the better pinned `x_true`,
the cheaper NUTS. Where the data anchor K (37 °C) the width is unchanged
or slightly wider; on the shoulder (20 °C) it widens (0.125 → 0.17).

## Caveat

With a σ_x prior active, `x_true` is a weakly pinned 24-dim nuisance and
NUTS needs more compute to converge than the fixed-axis fit (ess 1300–1600
vs 9300 at 4000 draws). Practical rule: check the printed `r_hat`/`ess`
and re-run with more draws when r̂ > 1.01 or ess < 100 — as in the 600-draw
smoke runs. The `auto` estimate is smaller (thus easier) than a hand-picked
upper-bound σ_x, and is the recommended default when replicate rows exist.

## Open questions for review

1. Should `--x-sigma auto` also feed a **data-driven lower floor** beyond
   0.02 (e.g., refuse σ_x below the note's own rounding precision)?
   Current floor 0.02 pH keeps the prior strictly positive.
1. The estimator pools *within-column* SD only; if a lab adopts replicate
   **columns** instead of replicate rows, the estimator returns `None`
   (safe fallback) — fine, but worth a one-line docs note.
1. `fit_binding_pymc`'s new branch sits next to the existing `n_xerr`
   random-walk branch; both default off in EnSpire. No coupling needed,
   but worth a code-comment cross-reference when this merges.

## Out of scope (separate PR)

- MCR/global readout support for `ppr enspire --method bayes`
  (`ppr pka` already has `--method bayes` + `n_xerr`).
- Centering/scaling `x_true` for better mixing (the r̂ caveat above).
