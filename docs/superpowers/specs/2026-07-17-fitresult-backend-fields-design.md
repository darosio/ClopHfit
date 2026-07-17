# Split `FitResult.mini` into `.mini` / `.trace` / `.odr`

- **Date:** 2026-07-17
- **Status:** approved, pending implementation plan
- **Scope:** `clophfit.clophfit_types`, `clophfit.fitting.data_structures`, `clophfit.fitting.core`, `clophfit.fitting.odr`, `clophfit.fitting.bayes`, `clophfit.__main__`, and the `FitResult[...]` annotations across `fitting/` and `testing/`
- **Depends on:** PR #1398 (`feat/plate-fit-residuals`) merged to `main`. This spec edits `ResidualsMixin.residual_table`, which #1398 introduced. Branch off `main` after that merge.

## Problem

`FitResult` carries one field, `mini`, typed as a three-way union:

```python
# clophfit_types.py:20
MiniT = Minimizer | odrpack.OdrResult | xr.DataTree
```

so a single attribute holds whichever backend produced the fit — an lmfit
`Minimizer`, an `odrpack.OdrResult`, or a PyMC `xr.DataTree`. Every consumer
must already know which one it holds and narrows by faith:

- lmfit reads: `__main__.py:544` (`.mini.userargs`), `__main__.py:580`
  (`conf_interval(mini, result)`), `__main__.py:684` / `core.py:345`
  (`.mini.emcee(...)`).
- ODR reads: `odr.py:233-257` (`.mini.res_var`, `outlier(.mini, ...)`).
- Trace reads: `data_structures.py:637,649` — `ResidualsMixin.residual_table`
  passes `self.mini` into `_resolve_residual_settings(trace=...)` and
  `residuals_from_fit_results(trace=...)`, because for a PyMC fit `mini` *is*
  the trace. `bayes.py:1960` assigns a trace *to* `.mini`
  (`per_well_results[key].mini = trace`).

Three problems follow:

1. **The union is a footgun.** The field name says nothing about which backend
   is present; correctness depends on each call site's unstated assumption. The
   sharpest instance is `residual_table` passing a `Minimizer` into a parameter
   literally named `trace`, relying on `_posterior_dataset_or_none` to reject
   non-traces at runtime.

1. **The generic machinery earns nothing.** `FitResult[MiniType: MiniProtocol]`
   is parameterized over the backend, but **no code anywhere binds on
   `MiniType` or `MiniProtocol`** (verified: zero uses outside
   `data_structures.py`'s own definitions). All 81 `FitResult[...]` annotations
   are documentation-only — the type parameter encodes which backend a result
   carries, information that belongs in named fields.

1. **`is_valid()` is backend-blind in the wrong direction.** It requires
   `self.mini is not None` (`data_structures.py:602`), which is correct for
   lmfit and ODR but would be *wrong* the moment the trace lived anywhere but
   `mini`.

## Design

### 1. Three concrete fields, generic dropped

```python
@dataclass
class FitResult:
    figure: Figure | None = None
    result: MinimizerResult | _Result | None = None
    mini:  Minimizer | None = None      # lmfit Minimizer (conf_interval, emcee, userargs)
    trace: xr.DataTree | None = None    # PyMC posterior trace
    odr:   OdrResult | None = None      # odrpack output (res_var, outlier)
    dataset: Dataset | None = None
```

`mini` narrows from the union to the lmfit `Minimizer` only. `trace` and `odr`
are new. Note `mini` is the `Minimizer` object, **not** the
`MinimizerResult` — that already lives in `result`.

The new fields are inserted **after** `mini` and **before** `dataset`. This
changes the positional signature, so every construction site moves to keyword
arguments (§2) — a positional `FitResult(fig, result, mini, ds)` would
otherwise bind `ds` to `trace`.

The generic parameter is removed: `class FitResult:` (no `[MiniType: ...]`).

### 2. Construction-site migration (6 sites)

| Site            | Now                                                       | After                                                                  |
| --------------- | --------------------------------------------------------- | ---------------------------------------------------------------------- |
| `core.py:285`   | `FitResult(fig, result, mini, ds)`                        | `FitResult(fig, result, mini=mini, dataset=ds)`                        |
| `core.py:312`   | `FitResult(fig, f_res.result, f_res.mini)`                | `FitResult(fig, f_res.result, mini=f_res.mini)`                        |
| `core.py:321`   | `FitResult(fig, f_res.result, f_res.mini)`                | `FitResult(fig, f_res.result, mini=f_res.mini)`                        |
| `core.py:493`   | `FitResult(fig, result, mini, copy.deepcopy(ds_working))` | `FitResult(fig, result, mini=mini, dataset=copy.deepcopy(ds_working))` |
| `odr.py:217`    | `FitResult(fig, _Result(...), output, ds)`                | `FitResult(fig, _Result(...), odr=output, dataset=ds)`                 |
| `bayes.py:1651` | `FitResult(fig, _Result(...), mini, ds)`                  | `FitResult(fig, _Result(...), trace=mini, dataset=ds)`                 |

Plus the post-construction assignment:

| Site            | Now                                  | After                                 |
| --------------- | ------------------------------------ | ------------------------------------- |
| `bayes.py:1960` | `per_well_results[key].mini = trace` | `per_well_results[key].trace = trace` |

At `bayes.py:1651` the local is named `mini` but holds a trace; rename the local
to `trace` for clarity while migrating, or pass `trace=mini` — implementer's
choice, but the field it lands in must be `trace`.

### 3. Read-site migration (by backend)

- **lmfit — stays `.mini`:** `core.py:281` (`mini = fit_result.mini`, later
  `.emcee`), `core.py:343-345` (`f_res.mini.emcee`), `__main__.py:543-544`
  (`is_valid()` + `.mini.userargs`), `__main__.py:580`
  (`conf_interval(fit_result.mini, ...)`), `__main__.py:682-684`
  (`f_res.mini.emcee`).
- **ODR — becomes `.odr`:** `odr.py:233` (`ro.odr.res_var`), `odr.py:236-237`
  (`ro.odr`, `outlier(ro.odr, ...)`), `odr.py:243,247,252,257`
  (`rn.odr.res_var`, `outlier(rn.odr, ...)`).
- **Trace — becomes `.trace`:** `ResidualsMixin.residual_table`
  (`data_structures.py:637,649`) resolves and passes `self.trace` instead of
  `self.mini`.

### 4. `residual_table` becomes honest (behaviour-preserving)

Before: `_resolve_residual_settings(self.mini, ...)` and
`residuals_from_fit_results(..., trace=self.mini, ...)`. After: both use
`self.trace`. For a classical fit `trace` is `None`, so
`_resolve_residual_settings(None)` resolves to Normal standardization —
identical to today, where a `Minimizer`/`OdrResult` in `mini` failed
`_posterior_dataset_or_none` and degraded to `None`. For a PyMC fit `trace`
is the `xr.DataTree`, exactly as `mini` was. **No behaviour change** — the
parameter named `trace` now actually receives a trace.

`MultiFitResult.residual_table` already uses `self.trace` (its own field) and
is untouched.

### 5. `is_valid()` becomes backend-agnostic

```python
def is_valid(self) -> bool:
    """Whether a figure, a result, and at least one backend object exist."""
    return (
        self.figure is not None
        and self.result is not None
        and (self.mini is not None or self.trace is not None or self.odr is not None)
    )
```

This preserves the current contract for lmfit/ODR (which set `mini`/`odr`) and
fixes it for PyMC (which now sets `trace`, where before it set `mini` and so
happened to pass). Update the docstring from "minimizer exist" accordingly.

### 6. Type-machinery cleanup

- Delete `MiniT` from `clophfit_types.py:20` and its six imports
  (`__main__.py:52`, `odr.py:20`, `testing/evaluation.py:22`,
  `fitting/plotting.py:61`, `testing/fitter_test_utils.py:24`, and the
  re-export at `data_structures.py:27`).
- Delete `MiniProtocol` and `MiniType` from `data_structures.py`.
- Rewrite all 81 `FitResult[...]` annotations to plain `FitResult`:
  `FitResult[MiniT]` (50), `FitResult[xr.DataTree]` (16), `FitResult[typing.Any]`
  (10), `FitResult[Minimizer]` (8), `FitResult[Any]` (4),
  `FitResult[odrpack.OdrResult]` (3). This spans `data_structures.py`,
  `core.py`, `odr.py`, `bayes.py`, `plotting.py`, `titration.py`, `export.py`,
  `__main__.py`, `testing/evaluation.py`, and `testing/fitter_test_utils.py`.

Deleting `MiniT` also dissolves the import-cycle concern that put it in the
leaf `clophfit_types` module in the first place — with the union gone, there is
no cross-module alias to protect.

`odrpack` / `xr` / `Minimizer` imports that existed only to build `MiniT` in
`clophfit_types.py` are pruned if nothing else there uses them (ruff F401 is the
arbiter). `data_structures.py` gains a `Minimizer` type-import and keeps its
`xr` / `OdrResult` type-imports for the new field annotations.

### 7. Bulk-edit strategy

The 81 annotation rewrites are the bulk of the diff and are mechanical. Do them
as a distinct step from the field/logic changes so the semantic changes (§1-§5)
are reviewable in isolation from the mechanical churn (§6). `mypy` is the
safety net: after the field split lands, every stale `FitResult[...]` is a type
error, so the rewrite is complete exactly when `mypy` is clean.

## Testing

- **Regression:** the full suite (`uv run pytest tests/`) must stay green. This
  is a behaviour-preserving refactor.
- **Field routing (new):** an lmfit fit populates `.mini` and leaves
  `.trace`/`.odr` `None`; an ODR fit populates `.odr` and leaves `.mini`/`.trace`
  `None`; a PyMC fit populates `.trace` and leaves `.mini`/`.odr` `None`.
- **`is_valid()` (new):** true for a PyMC fit (the case the split fixes),
  true for lmfit and ODR, false for an empty `FitResult()`.
- **Residuals unchanged:** `FitResult.residuals` for a classical fit is
  identical before and after (the `mini`→`trace` swap is a no-op there);
  for an MCMC fit `residual_likelihood` still resolves from the trace.
- **`mypy`** clean across `src/clophfit` — the completeness check for §6.

## Risks and limitations

- **External callers of `.mini`.** A hard cut (no shim) means any downstream
  code — notebooks, scripts — that read a trace or ODR result out of `.mini`
  breaks until migrated. Accepted: all in-repo callers are migrated here, and
  `prtecan_devel.ipynb` (out of scope, owner-owned) is the only known external
  reader; its `.mini` uses, if any, are the owner's to update.
- **Annotation churn size.** 81 edits is a large diff, but purely mechanical and
  mypy-verified. Splitting the PR further (fields vs annotations) is possible
  but the two are coupled through `mypy` — the annotations cannot compile until
  the field lands, so they ship together, sequenced per §7.

## Out of scope

- `shared_floor` for the PyMC noise priors — its own spec and PR (Part B).
- Any change to `MultiFitResult`, which already has a first-class `trace` field.
- Folding `pymc_multi` or FGLS into `Titration.fit_plate` (recorded in the
  plate-fit-residuals spec's follow-ups).
- A deprecation shim on `.mini` — explicitly rejected in favour of a hard cut.
