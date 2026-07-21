# Noise-model & screening comparison on real plates — design

Date: 2026-07-21

## Motivation

A systematic comparison of plate-fitting pipelines on the real L2/L3/L4 plates
established that a hierarchical MCMC noise model beats every classical fitter and
every point-screening rule on residual Gaussianity (Anderson A²), and that
outlier screening becomes actively harmful once the noise model is right. Three
questions remain open and are worth resolving before settling on a production
pipeline.

1. **Does `per_well_ye_mags=True` (pwym) actually help, or does it just fail to
   sample?** Every centered pwym=1 run in the ν-grid diverged (r̂ 1.5–3.6,
   ESS ≈ 5). Its A²/ELPD numbers are therefore uninterpretable. The structured
   noise model showed pwym=1 *improving* A² (2.80 vs 4.36) — but with no
   convergence check, so that result may be another non-converged chain.

1. **Is the pwym benefit real once sampling geometry is fixed?** The standard
   remedy for the funnel is a non-centered reparameterization of `ye_mag`, which
   keeps the exact same prior and parameter count and only changes geometry.

1. **Do the two heavy-tail choices (student-t ν=1 vs contamination mixture) and
   the two noise shapes (homoscedastic `ye_mag` vs signal-proportional
   `structured`) interact with pwym and with screening?** Observed A² shows the
   pwym effect *reversing* with the noise model, which needs a clean, converged
   comparison.

Separately, the user's own QQ diagnostics motivate two new calibration metrics
(see "Metrics"). A Normal-likelihood fit after a pooled-per-label MAD removal
showed a flat residual bulk with heavy tails — the signature of σ inflated by
the outliers a pooled screen fails to catch in quiet wells — while a ν=1 fit
with *no* removal showed near-Normal residuals with slightly light tails (mild
over-robustness). Distinguishing these failure modes per config requires
measuring the residual bulk width and any structural (step-dependent) misfit,
not just A².

## Goals

- Determine whether non-centered `ye_mag` makes pwym=1 converge, and whether
  pwym then helps, per noise model.
- Test whether a **hierarchical, partially-pooled** per-well multiplier
  (`log ye_mag_lbl[w] = log w_shared[w] + log δ_lbl[w]`, δ-scale learned) — which
  encodes the observed ye_mag₁/ye_mag₂ correlation (r ≈ 0.5) — improves
  calibration or convergence over the independent per-well form.
- Compare student-t ν=1 vs mixture, and `ye_mag` vs `structured`, on equal,
  convergence-gated footing.
- Systematically characterize a MAD **pre-screen** (per-well×label vs pooled;
  z>5 vs z>7) before spending sampler time on it, and test the surviving rule as
  a screening arm.
- Add per-config metrics that separate σ-scale miscalibration from likelihood
  shape and from structural model misfit.

## Non-goals (YAGNI)

- A ν sweep inside the grid. ν=1 is the representative robust config; the
  converged ν ∈ {0.5,1,1.5,2,3} results already exist and stand.
- Any plate beyond L2/L3/L4.
- Changing the default behaviour of `fit_binding_pymc_multi`. The new flag is
  opt-in and default-off.

## Piece 1 — `ye_mag` parameterizations (code change in `bayes.py`)

Add one opt-in keyword to `fit_binding_pymc_multi`,
`ye_mag_parameterization: Literal["centered", "noncentered", "hierarchical"] = "centered"`, subsuming both new model forms. It applies to the per-well `ye_mag`
multiplier (`per_well_ye_mags=True`) in **both** noise modes; for the shared or
scalar (non-per-well) paths it is a no-op falling back to the current centered
construction. `"centered"` reproduces today's behaviour exactly.

To make the parameterization well-defined and confound-free under
`structured` noise, the structured `learn_ye_mags` multiplier — currently
`HalfNormal(sigma=5.0)` (`bayes.py:2936`) — is switched to the same per-well
**LogNormal** `ye_mag` builder used by `ye_mag`-noise mode, for **all three**
parameterizations including `"centered"`. This changes the structured
`learn_ye_mags` prior (a research-harness path, not a shipped default), but keeps
`structured`-centered vs `structured`-noncentered differing only in geometry,
not prior — which is required for the pwym comparison to be valid.

**`"noncentered"`** — same prior, same parameter count, different geometry:

```python
z = pm.Normal(f"ye_mag_z_{lbl}", 0.0, 1.0, dims="well")
ye_mag = pm.Deterministic(f"ye_mag_{lbl}", pm.math.exp(mu + sigma * z), dims="well")
```

`mu`/`sigma` are the existing prior parameters (`sigma=1.5` default), so the
prior on `ye_mag` is unchanged. This is a pure convergence test of the pwym=1
funnel.

**`"hierarchical"`** — a shared per-well factor times a label-specific
deviation, so the two labels' multipliers partially pool (encoding the observed
r ≈ 0.5) rather than being independent. Deviation scale is **learned**, and the
whole thing is non-centered:

```python
tau_delta = pm.HalfNormal("ye_mag_tau_delta", sigma=0.5)          # learned shrinkage
z_w = pm.Normal("ye_mag_z_well", 0.0, 1.0, dims="well")
log_w = mu + sigma * z_w                                          # shared per-well factor, sigma=1.5
for lbl in labels:
    z_d = pm.Normal(f"ye_mag_z_delta_{lbl}", 0.0, 1.0, dims="well")
    log_delta = tau_delta * z_d                                   # label deviation
    ye_mag[lbl] = pm.Deterministic(
        f"ye_mag_{lbl}", pm.math.exp(log_w + log_delta), dims="well"
    )
```

Identifiability: the common level lives in `w_shared`; each `log_delta_lbl` has a
mean-zero prior, so the level does not float between them. With two labels this
is softly identified by the priors, which suffices for a Bayesian fit.

**Fallback.** If the learned-`tau_delta` hierarchical model fails to converge
(r̂ > 1.05) on a plate, re-run that config with `tau_delta` **fixed** at 0.5 and
record which variant produced the reported row. The fixed variant is a
contingency, not a separate grid cell.

**Tests.**

- Prior-predictive equivalence: `ye_mag` marginals from `"centered"` and
  `"noncentered"` match (KS or moment check).
- Hierarchical prior-predictive: `ye_mag_lbl[w]` marginals are sensible
  (positive, right order of magnitude) and the two labels are positively
  correlated across wells at the prior.
- Convergence smoke: a small multi-well `pwym=1` fit attains r̂ ≤ 1.1 under
  `"noncentered"` where `"centered"` does not.

## Piece 2 — MAD pre-screen characterization (classical, no MCMC)

For each of L2/L3/L4: lm-fit every well, take the raw (unweighted) residuals,
and compute the robust z **two ways** —

- **per well×label**: `robust_scale` within each well and label separately
  (local σ; a quiet well's outlier is judged against its own ≈5, not a pooled 43);
- **per label (pooled)**: one `robust_scale` over all wells of a label.

Tabulate the number of points flagged at |z|>5 and |z|>7 under each grouping,
per plate and label, and plot the flagged points on their curves for a handful
of wells — including at least one quiet well and one noisy well, to make the
pooled-vs-local difference visible. This is expected to reproduce the pooled
pathology: pooled flags concentrate in high-σ wells and miss proportionally
large outliers in quiet ones.

**Deliverable:** a chosen pre-screen rule (grouping + threshold) for the piece-3
MAD arm. Default expectation, to be confirmed by the table: per-well×label,
|z|>5, `min_keep=5`, two-sided.

## Piece 3 — Reduced comparison grid (MCMC)

Production sampler throughout: `nutpie`, `target_accept=0.98`, `n_tune=1000`,
`n_samples=10000`, `compute_log_likelihood=True`. Model kwargs match the
established setup: `n_sd=7, n_xerr=1.0, x_error_model="per_well", ctr_free_k=True`, `INIT` = the data-priors init.

### Base configs (13)

Notation `likelihood | noise | pwym·parameterization`.

| #   | likelihood    | noise      | pwym | parameterization       |
| --- | ------------- | ---------- | ---- | ---------------------- |
| 1   | student_t ν=1 | ye_mag     | 0    | —                      |
| 2   | student_t ν=1 | ye_mag     | 1    | non-centered           |
| 3   | student_t ν=1 | structured | 0    | —                      |
| 4   | student_t ν=1 | structured | 1    | non-centered           |
| 5   | mixture       | ye_mag     | 0    | —                      |
| 6   | mixture       | ye_mag     | 1    | non-centered           |
| 7   | mixture       | structured | 0    | —                      |
| 8   | mixture       | structured | 1    | non-centered           |
| 9   | student_t ν=1 | ye_mag     | 1    | **centered (control)** |
| 10  | student_t ν=1 | ye_mag     | 1    | hierarchical           |
| 11  | student_t ν=1 | structured | 1    | hierarchical           |
| 12  | mixture       | ye_mag     | 1    | hierarchical           |
| 13  | mixture       | structured | 1    | hierarchical           |

Config 9 is the mechanism control: paired with config 2 it demonstrates directly
whether non-centering fixes the pwym=1 funnel. Configs 10–13 test whether
partial pooling of the per-well multipliers beats the independent form (configs
2, 4, 6, 8).

Mixture configs are `RobustConfig(enabled=True, likelihood="mixture", contamination_frac_prior={"1":0.15,"2":0.015})`; structured is
`NoiseConfig.structured(floor=bg_noise, gain=0.5, alpha=0.02, floor_mode="centered", gain_mode="free", alpha_mode="free", learn_ye_mags=<pwym>)`;
ye_mag is `NoiseConfig.ye_mag()`.

### Screening arms

After the base run, rank the base configs that **converged** (r̂ ≤ 1.05) by
`anderson_A2_pooled` ascending and take the **top 7**. For each, add two arms:

- `likres>4`: mark on `|likelihood_res|>4` via `mark_outliers(ResidualTail(...))`,
  `apply_exclusions(min_keep=5)`, refit **non-robust** (Normal), same noise/pwym.
- `mad_prescreen`: apply the piece-2 rule to the *input* datasets before fitting,
  then fit with the same likelihood/noise/pwym as the base config.

Non-converged base configs are ineligible (screening a divergent fit is
meaningless). If fewer than 7 converge, screen all that did.

Total: 13 base + 14 screening = **27 configs × 3 plates ≈ 81 runs**, ~13.5 h.

### Selection is automated

The harness runs the 13 base configs, computes the ranking, auto-selects the top
7 converged, and runs their screening arms — one launch, no manual checkpoint.
Partial results are written after every config.

## Metrics (cross-cutting additions to `score_pipeline`)

Existing per-run metrics are retained: r̂ (max, counts >1.01/>1.05, worst var),
ESS min, ELPD-LOO (+SE, per label), p_loo, Pareto-k (count > good_k, max),
Anderson A² (pooled + per label), Shapiro p (recorded, not used — rejects at
n>1200), control ΔK / ROPE / |z| / within-group SD, n_excluded, medK, iqrK,
med_sK, wall time.

Two additions:

- **`bulk_sd_{level}`** — robust SD of the residual bulk, `IQR(std_res)/1.349`,
  computed pooled and per label. Target ≈ 1. `bulk_sd < 1` = σ over-inflated for
  the bulk (scale miscalibration); `bulk_sd ≈ 1` with heavy A² tails = shape
  (over/under-robust ν). This separates the two QQ failure modes per config.
- **`resid_step_corr_{label}`** — Spearman correlation (rho and p) between
  `std_res` and titration `step`, per label. A non-trivial correlation flags a
  structural, step-dependent model misfit — the benign alternative explanation
  for a flat residual bulk — which the noise metrics cannot see.

## Sequencing & dependencies

1. **Piece 1** (code + tests). Gate: prior-predictive equivalence passes;
   `make lint` and `make type` clean; convergence smoke shows non-centered
   pwym=1 mixing.
1. **Piece 2** (classical characterization). Gate: user reviews the flagged-point
   table and curves, picks the pre-screen rule.
1. **Piece 3** (overnight MCMC run) — depends on Piece 1 (non-centered arm) and
   Piece 2 (MAD arm rule). Then ranked analysis.

The two fast pieces gate the long run so nothing expensive starts on an
unverified assumption.

## Success criteria

- Piece 1: non-centered pwym=1 converges (r̂ ≤ 1.05) where centered pwym=1 (config
  9\) does not; prior-predictive equivalence holds; the hierarchical model samples
  (learned or, failing that, fixed `tau_delta`) and its prior-predictive checks
  pass.
- Piece 2: flag-count table under both groupings and both thresholds, with the
  pooled pathology visible on example curves; a recorded rule decision.
- Piece 3: 81 runs complete with the full metric set; a convergence-gated ranking
  that identifies the best pipeline on A², `bulk_sd`, ELPD, and control
  consistency, and states whether pwym (non-centered), hierarchical pooling, and
  screening help or not, per noise model.

## Files touched

- `src/clophfit/fitting/bayes.py` — `ye_mag_parameterization` flag
  (`centered`/`noncentered`/`hierarchical`) on the per-well LogNormal path.
- `tests/test_bayes.py` (or the existing bayes test module) — prior-predictive
  equivalence + convergence smoke.
- `scripts/compare_plate_pipelines.py` — `bulk_sd` and `resid_step_corr` in
  `score_pipeline`; the piece-2 characterization routine; the base+top-7
  screening grid with automated selection.
- `pipeline_comparison/` — output CSVs and figures (untracked artifacts).
