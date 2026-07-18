# Plate refit consolidation: one return shape, composable stages

- **Date:** 2026-07-18
- **Status:** approved, pending implementation plan
- **Scope:** `clophfit.fitting.pipeline` (dissolved), `clophfit.fitting.utils`,
  `clophfit.fitting.bayes`, `clophfit.fitting.model_validation`,
  `clophfit.prtecan.titration`, `clophfit.prtecan.export`

## Problem

Four plate-level entry points each end in a different return type, and only one
of them is `TitrationResults`:

| Function                                                  | Returns                                        |
| --------------------------------------------------------- | ---------------------------------------------- |
| `Titration.fit_plate` (`titration.py:1193`)               | `TitrationResults`                             |
| `fgls_plate_fit` (`pipeline.py:118`)                      | `tuple[dict[str, FitResult], PlateNoiseModel]` |
| `fit_binding_pymc_residual_refit` (`bayes.py:2196`)       | `PymcResidualRefitResult`                      |
| `fit_binding_pymc_multi_residual_refit` (`bayes.py:2378`) | `PymcMultiResidualRefitResult`                 |

The plate-fit-residuals spec (`2026-07-17-plate-fit-residuals-design.md`)
deferred this explicitly as follow-ups (b) and (c): "reconciling the two return
shapes is unresolved design work, not scoped here."

All four share one shape: **fit → derive plate-wide residuals → use those
residuals to define outliers and/or a noise model → refit.** `fgls_plate_fit`
uses residuals for the noise model only; both refit functions use them for
outliers only with a hardcoded noise strategy. Nothing does both.

### The refit functions are frozen compositions, not capability

Every stage already exists as a public primitive:

| Stage                         | Existing primitive                                                                   |
| ----------------------------- | ------------------------------------------------------------------------------------ |
| Fit plate                     | `Titration.fit_plate(method=...)`                                                    |
| Plate residuals               | `TitrationResults.residuals` / `.residual_table()`                                   |
| Screen by robust residual     | `mark_excess_residual_outliers` (`model_validation.py:788`)                          |
| Screen by outlier probability | `mark_outlier_probability_outliers` (`model_validation.py:880`)                      |
| Apply mask                    | `masked_datasets_from_outlier_probabilities` (`model_validation.py:931`)             |
| Calibrate noise               | `calibrate_noise_robust` (`pipeline.py:64`), `fit_noise_model_nnls` (`utils.py:446`) |
| Apply noise                   | `PlateNoiseModel.apply_to_plate` (`data_structures.py:865`)                          |

Because the refit functions freeze one composition, every knob of every stage
has to be re-exported through the top. `fit_binding_pymc_multi_residual_refit`
carries 28 parameters and still cannot express a `student_t` screening pass with
a per-label `contamination_frac_prior`: it predates the config objects and takes
none of `InitConfig` / `RobustConfig` / `NoiseConfig` / `SamplerConfig`, and its
`noise_strategy` is a closed `Literal["ye_mag", "proportional"]`.

Usage confirms they are not earning their surface:

- `fit_binding_pymc_multi_residual_refit` — **zero** production callers; only
  tests and `__all__`.
- `fit_binding_pymc_residual_refit` — **one** caller, `export.py:148`. It passes
  4 of 22 parameters, discards `initial` and `masked_dataset`, keeps only
  `final`, and hand-wraps into `TitrationResults` itself.

### Layering blocks the obvious fix

The dependency direction is strictly `prtecan → fitting`. `fitting/bayes.py:58`
imports `PlateScheme` under `TYPE_CHECKING` only, to avoid a runtime cycle.
`TitrationResults` is defined in `prtecan/titration.py:467`, so `fgls_plate_fit`
cannot return one from inside `fitting/` without inverting that dependency.

### `pipeline.py` loses its reason to exist

Its docstring is "Pipeline orchestrators for fitting multistage workflows
(e.g., FGLS)". Once the orchestrator moves out, the module orchestrates nothing.
It is not exported from any `__init__`; only tests import the path.

## Design

### 1. FGLS moves onto `Titration`, split by layer

`Titration.fgls_fit_plate(...)` holds the converge-loop and returns a single
`TitrationResults`, mirroring `fit_plate` and obtaining `scheme` / `fit_keys`
for free. The pure, dataset-level calibration helpers stay in `fitting/`.

The FGLS algorithm itself is unchanged: same first/second-pass methods, same
gain/alpha convergence test, same calibration-failure fallback.

### 2. `TitrationResults` carries the noise model

One new field, appended **after** `_dataframe` (`titration.py:483`):

```python
noise_model: PlateNoiseModel | None = None
```

Field order matters here. `TitrationResults` is constructed positionally in
production — `TitrationResults(titration.scheme, titration.fit_keys, mcmc_fits)`
at `export.py:140` and `:161` — so inserting the field anywhere before
`_dataframe` would silently rebind an existing positional argument, the same
trap the fitresult-backend-fields spec hit when new fields landed before
`dataset`. Appending last leaves every existing positional call correct and
requires no call-site changes.

Populated by `fgls_fit_plate`; left `None` by plain `fit_plate`. This matches
the framing that plate residuals yield both outliers and a noise model — the
calibration is plate-level fit output and belongs with the fits.

### 3. `pipeline.py` is dissolved into a focused `noise_calibration.py`

New module `fitting/noise_calibration.py` holds the whole noise-model family:

- from `pipeline.py`: `calibrate_noise_robust`, `_plate_noise_model_from_nnls`,
  `_noise_params_converged`
- from `utils.py`: `fit_rel_error_from_residuals` (337),
  `fit_gain_from_residuals` (391), `fit_noise_model_nnls` (446),
  `compute_binding_slope` (532), `compute_plate_slopes` (548),
  `fit_ph_slope_noise` (583)

`compute_binding_slope` moves with `compute_plate_slopes` because the latter is
its per-plate wrapper; splitting them would leave a cross-module call for no
gain.

`pipeline.py` is deleted. `utils.py` drops to genuinely miscellaneous helpers
(outlier detection, curve shape). Blast radius is two test files —
`test_outlier_scores.py` and `test_pipeline.py` — since no other source module
imports the moved functions.

### 4. Both refit functions are deleted

Removed: `fit_binding_pymc_residual_refit`, `fit_binding_pymc_multi_residual_refit`,
`PymcResidualRefitResult` (`bayes.py:1436`), `PymcMultiResidualRefitResult`
(`bayes.py:1446`), and the two `__all__` entries (`bayes.py:310-311`).

`--mcmc single-refit` keeps working: `export.py` inlines the same two-pass
sequence from existing primitives (robust screening fit → residual table →
`mark_excess_residual_outliers` → `masked_datasets_from_residual_outliers` →
unrobust refit), preserving today's behaviour including the
`single_refit_initial_residual_outliers.csv` diagnostic.

Two-pass pipelines otherwise become explicit composition at the call site
(benchmarks, notebooks), which is what makes arbitrary config combinations
expressible.

### 5. Conjunction screening extends the existing primitive

`mark_outlier_probability_outliers` gains two keyword parameters:

```python
residual_threshold: float | None = None
residual_col: str = "std_res"
```

When `residual_threshold` is set, a row is marked only if it exceeds **both**
the probability cutoff and the absolute-residual cutoff, expressing
`p_outlier > 0.7 AND |std_res| > 3`. Default `None` preserves today's
probability-only behaviour exactly.

AND only. No configurable combine rule until something needs OR.

## Out of scope

- **No outlier removal inside the FGLS loop.** The plate-fit-residuals spec
  notes this gap; adding it is a behavioural change the benchmark should
  justify, not a refactor.
- **No general iterative orchestrator.** FGLS is the only true converge-loop;
  the two-pass flows are screen-once-and-stop. Merging those control flows
  would recreate the parameter sprawl one level up.
- **No default threshold changes** anywhere, including whether `single-refit`
  should adopt the `student_t` + conjunction-screen pipeline. That is the
  benchmark's decision.
- **Replacing `ye_mag` with a structured noise model.** `ye_mag` is
  homoscedastic once `y_err` is flat, which misdescribes detector noise that
  grows with signal (`floor² + gain·y + (alpha·y)²`). The expected replacement
  for both the screening and refit passes is a heteroscedastic config such as
  `NoiseConfig.structured(floor=tit.bg_noise, gain=0.5, alpha=0.02, floor_mode="centered", gain_mode="free", alpha_mode="free", shared_alpha=False, shared_gain=False, learn_ye_mags=False)` — optionally with
  `shared_floor=True`. Gated on the benchmark; `export._single_refit_two_pass`
  is parameterized so the swap is a call-site edit.
- **The benchmark harness itself** — its own spec, built on this interface.
- **`pymc_multi` / `MultiFitResult` return-shape reconciliation** — still
  unresolved, still deferred.

## Testing

1. **Characterization first.** Pin `export.py`'s `single-refit` output *before*
   deleting `fit_binding_pymc_residual_refit`, so the inlining is provably
   behaviour-preserving.
1. `fgls_fit_plate` returns `TitrationResults` with `noise_model` populated and
   a working `.residuals`; port `test_fgls_plate_fit_workflow`
   (`test_residuals.py:684`) and `test_pipeline.py`'s two FGLS tests to the new
   entry point.
1. Conjunction screener: marks only rows meeting both criteria; a row over the
   probability cutoff but under the residual cutoff is *not* marked; default
   (`residual_threshold=None`) behaviour is unchanged.
1. Deletion completeness: `__all__` has no dangling names; no import of
   `clophfit.fitting.pipeline` survives.
1. Moved functions keep their existing tests, with imports updated
   (`test_outlier_scores.py`, `test_pipeline.py`).
1. Delete the five refit tests in `test_bayes.py` (1366, 1419, 1469, 1520, 1611)
   along with the functions they cover.

## Error handling

`fgls_fit_plate` keeps the current calibration-failure fallback (log a warning,
fall back to zero gain/alpha with the supplied floors) and the per-well
`InsufficientDataError` skip that yields an empty `FitResult`.
`TitrationResults.noise_model` is `None` whenever no calibration ran, so
consumers must treat it as optional.

## Migration notes

`fgls_plate_fit`, both refit functions, and their two result dataclasses are
removed outright rather than deprecated, consistent with the hard-cut precedent
set for `.mini` in the fitresult-backend-fields spec. `pipeline.py` is not
public API (absent from every `__init__`), so its removal is not a breaking
import change for packaged consumers.
