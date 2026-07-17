# Plate-level fit results with unified residual access

- **Date:** 2026-07-17
- **Status:** approved, pending implementation plan
- **Scope:** `clophfit.fitting.pipeline`, `clophfit.fitting.data_structures`, `clophfit.prtecan.titration`, `clophfit.prtecan.export`

## Problem

`fit_plate` returns a bare `dict[str, FitResult]`, so a plate fit has no residual
accessor:

```python
res_glob = pipeline.fit_plate(ds_dict, "huber")
res_glob.residuals          # AttributeError
```

Callers must instead reach for the builder by hand, restating what the fit
already knows:

```python
all_res = residuals_from_fit_results(res_glob, trace_id="", binding_function=binding_1site)
```

This is inconsistent with the rest of the package, where `FitResult.residuals`
(`data_structures.py:525`) and `MultiFitResult.residuals` (`data_structures.py:613`)
both exist as cached properties over the same canonical schema.

Four further problems cluster around the same code:

1. **Repeated fit-then-wrap.** Every production call site of `fit_plate` immediately
   wraps the result with plate metadata it already has —
   `export.py:178→184`, `194→200`, `203→209`:

   ```python
   global_fits = fit_plate(datasets, method=method, ...)
   global_res = TitrationResults(titration.scheme, titration.fit_keys, global_fits)
   ```

1. **Duplicated residual-settings logic.** `FitResult.residual_table`
   (`data_structures.py:537-591`) and `MultiFitResult.residual_table`
   (`data_structures.py:624-674`) are near-identical: the same
   `robust_likelihood_from_trace` → `robust_settings_from_trace` → `STUDENT_T_NU`
   fallback, differing only in the trace source and the terminal builder call.
   Adding a third plate-level accessor would be a third copy.

1. **`fit_plate` lives in the wrong module.** `pipeline.py` is documented as
   "Pipeline orchestrators for fitting multistage workflows (e.g., FGLS)". That
   describes `fgls_plate_fit` / `calibrate_noise_robust`, but `fit_plate` is a
   single-pass dispatch loop, not a multistage workflow.

1. **Stray debug output.** `pipeline.py:272` contains a bare `print(method)` that
   fires on every `lm`/`huber`/`irls` plate fit, including from `export.py`.

## Design

### 1. `Titration.fit_plate()` replaces `pipeline.fit_plate()`

`fit_plate` moves to `clophfit.prtecan.titration` as a method on `Titration`, and
returns `TitrationResults` directly.

```python
def fit_plate(
    self,
    datasets: dict[str, Dataset] | None = None,
    method: str = "",
    *,
    label: str | None = None,
    **kwargs: typing.Any,
) -> TitrationResults:
```

**Rationale.** Both dataset dicts in `export.py` are pure functions of the
titration — `{k: titration.create_global_ds(k) for k in titration.fit_keys}`
(line 169) and `{k: titration.create_ds(k, label=label) for k in titration.fit_keys}`
(line 176) — so a method on `Titration` already holds everything it needs.
`Titration` owns `fit_keys` (`titration.py:644`), `create_ds` (1055), and
`create_global_ds` (1061); `fit_plate` joins that family. `TitrationResults` is
defined at line 414, before `Titration` at 602, so the return annotation needs no
forward reference. `titration.py` already imports `clophfit.fitting` at module
level (lines 19-28), so no dependency boundary is crossed: `prtecan → fitting` is
the established direction, and `fitting → prtecan` exists only under
`TYPE_CHECKING` (`bayes.py:55-59`).

**Semantics.**

- `datasets` stays positional so existing calls survive verbatim:
  `tit.fit_plate(ds_dict, "huber")`.
- `datasets=None` → build via the **existing** `self.create_dataset_dict(label)`
  (`titration.py:1067`), which already switches on `label` (global when `None`,
  per-label otherwise) and additionally applies `apply_outlier_mask` when
  `self.params.mask_outliers` is set. Reusing it keeps this DRY and avoids
  re-deriving the dict comprehension.
- Note the resulting asymmetry, which is intentional: `export.py` passes its own
  explicitly-built dicts (`export.py:169,176`) that skip `mask_outliers`, so its
  behaviour is unchanged by this refactor; only the new `datasets=None`
  convenience path picks masking up.
- Passing both `datasets` and `label` raises `ValueError`. The two are
  alternative ways to specify the same input; silently preferring one would be
  implicit behaviour.
- Returns `TitrationResults(self.scheme, self.fit_keys, results)`.
- Dispatch, the `InsufficientDataError` → empty `FitResult()` handling, and
  `**kwargs` pass-through (`remove_outliers`, `reweight`, …) are carried over
  from the current implementation unchanged.

Explicit `datasets` must remain supported: `prtecan_devel.ipynb` mutates datasets
before fitting (`ds['1'].mask[-1] = False`, `ds["1"].y_err /= ...`), so building
them internally cannot be mandatory.

`fit_binding_pymc` is imported at **module level**, not lazily — see §5.

### 2. `ResidualsMixin` in `data_structures.py`

A mixin holds the two things that are genuinely shared:

- `residuals` — `cached_property` returning `self.residual_table()`, with the
  canonical docstring.
- `_resolve_residual_settings(trace, *, binding_function, robust, student_t_nu)` —
  the auto-detection currently duplicated in both `residual_table` bodies,
  returning a `NamedTuple` of `(binding_function, robust, student_t_nu, residual_likelihood)`.

Each class keeps its **own** `residual_table` with its own precise signature,
reduced to roughly six lines: resolve settings, then call its builder. This is
deliberate — `FitResult.residual_table` takes a `well: str = ""` parameter that
the others do not, and it is exercised by
`test_model_validation_smoke.py:839,860,888`. Pushing a single shared
`residual_table` signature onto all three would either drop `well` or force
loosely-typed `**kwargs`, both worse than a six-line method per class.

The mixin is applied to `FitResult`, `MultiFitResult`, and `TitrationResults`.

### 3. `TitrationResults` gains `.residuals`

`TitrationResults` (`prtecan/titration.py:414`) already carries
`results: dict[str, FitResult]`, `scheme`, `fit_keys`, `dataframe`, `n_sd`, and
`plot_k` (line 522). It gains `residuals` from the mixin and implements its own
`residual_table()` in the §2 shape: resolve settings via
`_resolve_residual_settings(trace=None, ...)` — classical plate fits have no
trace, so this resolves to Normal standardization — then delegate to
`residuals_from_fit_results(self.results, trace_id="", ...)`.

**Failed wells are skipped, not raised on.** `residuals_from_fit_results` already
drops entries where `fr.dataset is None or fr.result is None`
(`model_validation.py:1779-1780`), which is exactly the empty `FitResult()`
produced by the `InsufficientDataError` path. A plate with failed wells therefore
yields a quietly shorter table; this matches existing behaviour and is documented
in the property docstring rather than changed.

After this, `fr.residuals`, `res_glob.residuals`, `multi.residuals`, and
`tit_res.residuals` all return the same `RESIDUAL_TABLE_COLUMNS` schema and feed
`ResidualDiagnostics`, `residual_statistics`, and the `plot_residual_vs_*`
helpers interchangeably.

### 4. `MultiFitResult` stays as-is

It is **not** merged into `TitrationResults`. It is a trace proxy, not a plate
container: `__getattr__` delegates to `self.trace`, and `plotting.py` treats it
polymorphically with the raw trace across six signatures
(`trace: xr.DataTree | MultiFitResult`) plus an `isinstance` branch at line 1145.
Its per-well results are reconstructed from one shared hierarchical posterior —
a different object from a plate of independent fits. Merging would drag a PyMC
trace and trace-delegation into the plate-metadata class.

What is unified is the **schema and the accessor**, not the type.

### 5. Imports become module-level

Every lazy import in the affected code is removed, and the one originally planned
for this work is not introduced. Each was verified unnecessary rather than
assumed so.

- **`data_structures.py:568-569`** (`from clophfit.fitting import model_validation`,
  `from clophfit.fitting.models import binding_1site`). These are lazy because
  they use the **package-attribute form**, which requires `fitting/__init__.py`
  to have finished binding the attribute — and `__init__` eagerly imports `bayes`
  (line 7) → … → `data_structures`, so at module level that form would read off a
  partially-initialized package. The **submodule-direct form**
  (`from clophfit.fitting.model_validation import residuals_from_fit_results`)
  sidesteps this by importing the submodule instead of an attribute of the parent.
  Safe because the graph is acyclic: `model_validation` → `residuals` (which
  imports nothing from `clophfit`), `models` → `clophfit_types`, and
  `model_validation` never references `data_structures`.
  *Verified empirically*: patched to module-level, both `import clophfit.fitting`
  (worst case, through the eager `__init__`) and a cold submodule-first import
  succeeded; then reverted.

- **`titration.py:692`** (`flag_trend_outliers`, `roughness`, `smoothness` from
  `clophfit.fitting.utils`). Redundant: that module is already imported at
  **line 28** (`from clophfit.fitting.utils import apply_outlier_mask`). Folds
  into line 28's import list.

- **The planned lazy `fit_binding_pymc` import is not needed.** *Measured*:
  `import clophfit.prtecan` already takes 0.79 s and already has `pymc` in
  `sys.modules`, because `titration.py:19` → `clophfit.fitting.__init__` →
  `bayes` (line 7) → PyMC. A module-level import in `titration.py` therefore
  costs nothing new.

Their `# noqa: PLC0415` suppressions are dropped with them.

### 6. `pipeline.py` keeps FGLS only

With `fit_plate` gone, `pipeline.py` contains `fgls_plate_fit`,
`calibrate_noise_robust`, `_noise_params_converged`, and
`_plate_noise_model_from_nnls` — which is precisely what its docstring already
claims. The stray `print(method)` (line 272) is deleted along with the function
that hosts it. Module-level imports that become unused after the move
(`fit_binding_pymc`, `fit_binding_odr`, `fit_binding_glob`,
`InsufficientDataError`) are pruned if no FGLS code path uses them.

FGLS itself is unchanged. Its only callers are the test suite and
`prtecan_devel.ipynb`; it is experimental, not dead, and out of scope here.

## Migration

| Call site       | Before                                                             | After                                              |
| --------------- | ------------------------------------------------------------------ | -------------------------------------------------- |
| `export.py:178` | `fit_plate(ds_single, ...)` + `TitrationResults(...)`              | `titration.fit_plate(ds_single, ...)`              |
| `export.py:194` | `fit_plate(datasets, ...)` + `TitrationResults(...)`               | `titration.fit_plate(datasets, ...)`               |
| `export.py:203` | `fit_plate(datasets, method="odr", ...)` + `TitrationResults(...)` | `titration.fit_plate(datasets, method="odr", ...)` |

`export.py` drops its `from clophfit.fitting.pipeline import fit_plate` import
(line 19). The three `export_list.append(...)` wraps collapse to single calls.

`tests/test_pipeline.py` currently monkeypatches
`clophfit.fitting.pipeline.fit_binding_glob` / `fit_binding_odr` /
`fit_binding_pymc` (lines 88-90, 110-112). The `fit_plate` tests
(`test_fit_plate_routes_methods_and_preserves_well_keys`,
`test_fit_plate_turns_insufficient_data_into_empty_result`) move to the prtecan
test module and repatch against `clophfit.prtecan.titration`. The FGLS tests stay.
`tests/test_prtecan.py:1290,1295,1306` switch to the method form.

Notebooks (`docs/tutorials/prtecan.ipynb:263-275,332`, `prtecan_devel.ipynb`) drop
the `pipeline` import and call `tit.fit_plate(...)`.

**No deprecation shim.** The plate-fit entry point becomes prtecan-only; there is
no generic "fit a dict of datasets" API left in `fitting/`. This is accepted:
every production caller is in `prtecan`, and the only metadata-free callers are
synthetic unit tests.

## Testing

- `Titration.fit_plate` returns `TitrationResults` with `scheme`/`fit_keys`
  populated from the titration.
- Dataset construction: `datasets=None` builds global; `label="1"` builds
  per-label; both together raise `ValueError`.
- Method routing (`lm`, `huber`, `odr`, `mcmc`) and well-key preservation —
  ported from `test_pipeline.py:66-98`.
- `InsufficientDataError` still yields an empty `FitResult()` for that well —
  ported from `test_pipeline.py:100-116`.
- `TitrationResults.residuals` returns `RESIDUAL_TABLE_COLUMNS`, populates `well`
  per row, and skips failed wells.
- Parity: `TitrationResults.residuals` and `MultiFitResult.residuals` agree on
  columns and dtypes.
- `FitResult.residual_table(well=...)` keeps working after the mixin refactor
  (`test_model_validation_smoke.py:839,860,888` must pass unchanged).
- An import test asserting no lazy-import regression: `import clophfit.prtecan`
  and `import clophfit.fitting` both succeed cold.

## Risks and limitations

- **`cached_property` staleness.** `TitrationResults.residuals` caches on first
  access; mutating `results` afterwards yields a stale table. This already applies
  to `FitResult.residuals` and `MultiFitResult.residuals`, and `TitrationResults`
  already hand-rolls a similar cache for `dataframe` (with an index check). Accepted
  as consistent with existing behaviour; `residual_table()` remains the uncached
  escape hatch.
- **`titration.py` growth.** Currently ~1160 lines; this adds ~50. Acceptable for
  now, but the file is a future split candidate (`Titration`, `TitrationResults`,
  and `Buffer`/`BufferFit` are separable).
- **Robustness auto-detection is PyMC-only.** `robust_settings_from_trace` infers
  Student-t standardization from a trace; classical fits (`lm`, `huber`, `odr`)
  have no trace and correctly resolve to `robust=False`, i.e. Normal
  standardization. This is intended — `robust` here means specifically "apply the
  Student-t probability-integral transform", which is valid only for a Student-t
  likelihood, not for the Huber M-estimator's least-favorable density. Behaviour
  is unchanged by this spec; noted because "huber ⇒ robust=True" is an inviting
  and wrong inference.

## Out of scope

- FGLS behaviour changes.
- Splitting `titration.py`.
- Method-aware robust standardization for classical fits.
- Any change to `RESIDUAL_TABLE_COLUMNS` or the residual builders themselves.
