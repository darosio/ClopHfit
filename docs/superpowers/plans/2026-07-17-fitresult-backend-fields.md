# FitResult Backend Fields Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split `FitResult.mini` (a `Minimizer | OdrResult | xr.DataTree` union) into three concrete fields `.mini` / `.trace` / `.odr`, and drop the annotation-only generic parameter.

**Architecture:** Two tasks. Task 1 is the semantic change: add `trace`/`odr` fields, route each backend to its own field, fix `is_valid` and `residual_table`, all while **keeping** the generic parameter so the tree stays green and reviewable. Task 2 is the mechanical sweep: drop the generic, narrow `mini`'s type, rewrite every `FitResult[...]` annotation (src **and** tests) to plain `FitResult`, and delete the `MiniT`/`MiniProtocol`/`MiniType` machinery.

**Tech Stack:** Python 3.12+ (PEP 695 generics), pandas, lmfit, PyMC, odrpack, pytest, ruff, mypy, uv.

**Spec:** `docs/superpowers/specs/2026-07-17-fitresult-backend-fields-design.md`

## Global Constraints

- Run every command through `uv run` (e.g. `uv run pytest`). Never bare `pip`/`python`.
- Type hints on all public functions/methods; must pass `uv run mypy src/clophfit`.
- Numpy-style docstrings on all public API (ruff `convention = "numpy"`).
- Lint/format with ruff ONLY: `uv run ruff check` / `uv run ruff format`. Never black or isort.
- `line-length = 88`, `target-version = "py312"`.
- Do not reformat code unrelated to the task at hand.
- Do not modify `pyproject.toml`.
- **Do not touch `prtecan_devel.ipynb`** — it has uncommitted owner changes and is ruff/docs-excluded. Any `.mini` uses in it are the owner's to migrate.
- ruff's `PLC0415` (no function-scoped imports) and `D401` (imperative docstrings) are enforced; `tests/*` is NOT exempted.
- Branch is `feat/fitresult-backend-fields`, already off merged `main`.

______________________________________________________________________

### Task 1: Add `trace`/`odr` fields and route each backend (generic retained)

Adds the two new fields, migrates every construction and read site so PyMC uses `.trace`, ODR uses `.odr`, and lmfit keeps `.mini`, and fixes `is_valid`/`residual_table`. The generic parameter and all `FitResult[...]` annotations stay untouched here, so the tree remains green — this task is the reviewable semantic diff. `mini` still routes lmfit; for PyMC/ODR fits `mini` simply becomes `None` (their objects now live in `trace`/`odr`).

**Files:**

- Modify: `src/clophfit/fitting/data_structures.py` (fields ~579-583; `is_valid` 597-603; `residual_table` 636-651; add `OdrResult` type-import ~44-47)
- Modify: `src/clophfit/fitting/core.py` (construction 285, 312, 321, 493)
- Modify: `src/clophfit/fitting/odr.py` (construction 217-219; reads 233, 236-237, 243, 247, 252, 257)
- Modify: `src/clophfit/fitting/bayes.py` (construction 1651; assignment 1960)
- Modify: `src/clophfit/__main__.py` (no change — its reads are lmfit, stay `.mini`; listed only to confirm)
- Test: `tests/test_fitting.py` (field-routing + is_valid tests)

**Interfaces:**

- Consumes: nothing new.

- Produces: `FitResult` gains `trace: xr.DataTree | None = None` and `odr: OdrResult | None = None`, ordered `mini, trace, odr, dataset`. `is_valid()` returns true if figure, result, and any one of mini/trace/odr are present. Task 2 relies on these fields existing.

- [ ] **Step 1: Write the failing field-routing test**

Add to `tests/test_fitting.py`. Note this file does **not** have `from __future__ import annotations`, so annotations are evaluated at import — but these tests use no `FitResult[...]` subscripts, and `Dataset` is already imported (line 28) and `fit_binding_glob`/`fit_binding_odr` are used elsewhere in the file, so real-object annotations are safe. The `ph_dataset` fixture (from `tests/conftest.py:39`) is already used throughout this file; annotate it `Dataset` to match the file's convention. `fit_binding_odr` may need adding to the existing `from clophfit.fitting.odr import ...` block if not already imported — check and add if missing.

```python
def test_fitresult_routes_backend_to_named_field(ph_dataset: Dataset) -> None:
    """Each backend object lands in its own field, the others stay None."""
    lm = fit_binding_glob(ph_dataset, method="lm")
    assert lm.mini is not None
    assert lm.trace is None
    assert lm.odr is None

    odr = fit_binding_odr(ph_dataset)
    assert odr.odr is not None
    assert odr.mini is None
    assert odr.trace is None


def test_fitresult_is_valid_true_for_each_backend(ph_dataset: Dataset) -> None:
    """is_valid() no longer requires the lmfit-specific mini field."""
    assert fit_binding_glob(ph_dataset, method="lm").is_valid()
    assert not FitResult().is_valid()
```

`fit_binding_glob` is imported at the top of the file (line 20 block); `FitResult` and `fit_binding_odr` must be importable — add them to the existing top-of-file import blocks if absent (module-level, not inside the test — ruff PLC0415).

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_fitting.py -k "routes_backend or is_valid_true" -v`
Expected: FAIL — `AttributeError: 'FitResult' object has no attribute 'trace'` (and `.odr`), and the ODR routing assertion fails because `fit_binding_odr` still populates `mini`.

- [ ] **Step 3: Add the two fields and the OdrResult import**

In `src/clophfit/fitting/data_structures.py`, extend the `TYPE_CHECKING` block (after the lmfit import at lines 44-47) with:

```python
    from odrpack import OdrResult
```

`xarray as xr` is already imported there (line 42), and `Minimizer`/`MinimizerResult` at 44-47.

Change the `mini` field block (579-581) to add the two new fields directly after it, before `dataset`:

```python
    mini: MiniType | None = None
    """The lmfit ``Minimizer`` object (for ``conf_interval``, ``emcee``,
    ``userargs``). ``None`` for non-lmfit backends."""
    trace: xr.DataTree | None = None
    """The PyMC posterior trace. ``None`` for non-PyMC backends."""
    odr: OdrResult | None = None
    """The odrpack output (``res_var``, outlier detection). ``None`` for
    non-ODR backends."""
    dataset: Dataset | None = None
```

Leave `mini`'s type as `MiniType | None` for now — Task 2 narrows it. Keep the `class FitResult[MiniType: MiniProtocol](ResidualsMixin):` line unchanged.

- [ ] **Step 4: Fix `is_valid`**

Replace `is_valid` (597-603) with:

```python
    def is_valid(self) -> bool:
        """Whether a figure, a result, and at least one backend object exist."""
        return (
            self.figure is not None
            and self.result is not None
            and (
                self.mini is not None
                or self.trace is not None
                or self.odr is not None
            )
        )
```

- [ ] **Step 5: Point `residual_table` at `self.trace`**

In `residual_table` (636-651), change the two `self.mini` references to `self.trace`:

```python
        settings = self._resolve_residual_settings(
            self.trace,
            binding_function=binding_function,
            robust=robust,
            student_t_nu=student_t_nu,
        )
        return residuals_from_fit_results(
            {well: self},
            trace_id="",
            binding_function=settings.binding_function,
            robust=settings.robust,
            student_t_nu=settings.student_t_nu,
            outlier_threshold=outlier_threshold,
            trace=self.trace,
            residual_likelihood=settings.residual_likelihood,
        )
```

This is behaviour-preserving: for a classical fit `self.trace` is `None` (Normal standardization, same as before, when a `Minimizer` in `mini` failed `_posterior_dataset_or_none`); for a PyMC fit `self.trace` now holds the `xr.DataTree` that `self.mini` used to.

- [ ] **Step 6: Migrate the PyMC construction and assignment (`bayes.py`)**

At `bayes.py:1650-1651`, the local `mini` holds a trace. Rename it and route to the `trace` field:

```python
    trace = trace_obj if isinstance(trace_obj, xr.DataTree) else xr.DataTree()
    return FitResult(fig, _Result(rpars, residual=residuals), trace=trace, dataset=ds)
```

At `bayes.py:1960`, change the per-well assignment:

```python
        per_well_results[key].trace = trace
```

- [ ] **Step 7: Migrate the ODR construction and reads (`odr.py`)**

At `odr.py:217-219`, route the output to `odr=`:

```python
        return FitResult(
            fig,
            _Result(params, residual=residuals, redchi=output.res_var),
            odr=output,
            dataset=ds,
        )
```

Change every `.mini` read in this file to `.odr` (lines 233, 236, 237, 243, 247, 252, 257):

```python
    residual_variance = ro.odr.res_var if ro.odr else 0.0
    ...
        if remove_outliers and ro.odr:
            omask = outlier(ro.odr, threshold=threshold)
    ...
        if rn.odr and rn.odr.res_var == 0:
    ...
        if rn.odr and residual_variance - rn.odr.res_var < tol:
    ...
                omask_new = outlier(rn.odr, threshold=threshold)
    ...
        residual_variance = rn.odr.res_var if rn.odr else 0.0
```

- [ ] **Step 8: Migrate the lmfit construction sites to keyword (`core.py`)**

The new fields sit before `dataset`, so positional `FitResult(fig, result, mini, ds)` would now bind `ds` to `trace`. Convert the four lmfit sites to keyword form (they keep `mini=`):

- Line 285: `return FitResult(fig, result, mini=mini, dataset=ds)`
- Line 312: `gsvd = FitResult(fig, f_res.result, mini=f_res.mini)`
- Line 321: `bands = FitResult(fig, f_res.result, mini=f_res.mini)`
- Line 493: `return FitResult(fig, result, mini=mini, dataset=copy.deepcopy(ds_working))`

The `.mini` reads in this file (281, 343, 345) are lmfit and stay `.mini` — do not change them.

- [ ] **Step 9: Run the routing tests + regression suites**

Run: `uv run pytest tests/test_fitting.py -k "routes_backend or is_valid_true" -v`
Expected: PASS.

Run: `uv run pytest tests/test_odr.py tests/test_residuals.py tests/test_prtecan.py tests/test_bayes.py -q`
Expected: all pass. `test_odr.py` proves the `.odr` read migration; `test_residuals.py`/`test_prtecan.py` prove `residual_table` still works; `test_bayes.py` proves PyMC now populates `.trace`.

- [ ] **Step 10: Confirm MCMC residuals still resolve from the trace**

This is the behaviour-preservation check for the `mini`→`trace` swap. Run:

```
uv run python -c "
import numpy as np
from clophfit.fitting.bayes import fit_binding_pymc
from clophfit.fitting.data_structures import DataArray, Dataset
x = np.array([5.5,6.5,7.0,7.5,8.5]); y = np.array([1.,3.,5.,7.,9.])
ds = Dataset({'1': DataArray(x, y, y_errc=np.ones_like(y))}, is_ph=True)
fr = fit_binding_pymc(ds, n_samples=200, n_tune=200)
print('trace set:', fr.trace is not None, '| mini None:', fr.mini is None)
print('residual_likelihood:', fr.residuals['residual_likelihood'].iloc[0])
"
```

Expected: `trace set: True | mini None: True`, and a `residual_likelihood` that is `student_t`/`mixture`/`normal` per the fit's likelihood — NOT forced to `normal`. (Exact family depends on the default likelihood; the point is it is resolved from the trace, not hard-Normal.)

- [ ] **Step 11: Lint and typecheck**

Run: `uv run ruff check src/clophfit/fitting/data_structures.py src/clophfit/fitting/core.py src/clophfit/fitting/odr.py src/clophfit/fitting/bayes.py tests/test_fitting.py && uv run mypy src/clophfit`
Expected: clean. mypy still sees the generic; `mini`/`trace`/`odr` typecheck against their concrete field types.

- [ ] **Step 12: Commit**

```bash
git add src/clophfit/fitting/data_structures.py src/clophfit/fitting/core.py src/clophfit/fitting/odr.py src/clophfit/fitting/bayes.py tests/test_fitting.py
git commit -m "feat(data-structures): route backends to FitResult.mini/.trace/.odr"
```

______________________________________________________________________

### Task 2: Drop the generic and sweep all `FitResult[...]` annotations

Removes the now-purposeless generic parameter, narrows `mini` to `Minimizer`, deletes the `MiniT`/`MiniProtocol`/`MiniType` machinery, and rewrites every `FitResult[...]` annotation in src **and** tests to plain `FitResult`. This is atomic: the class-definition change breaks every subscription at once, so mypy + full test collection are the completeness net.

**Files:**

- Modify: `src/clophfit/fitting/data_structures.py` (class line 570; `mini` type 579; delete `MiniProtocol` 489-491, `MiniType` 494, `MiniT` re-export 18)
- Modify: `src/clophfit/clophfit_types.py` (delete `MiniT` 20 + now-unused imports)
- Modify: `src/clophfit/fitting/bayes.py` (runtime subscript 2320; `cast` string 592; annotations)
- Modify: `src/clophfit/testing/fitter_test_utils.py` (`cast` string 230; MiniT import 24; annotations)
- Modify: `src/clophfit/__main__.py`, `src/clophfit/fitting/odr.py`, `src/clophfit/fitting/plotting.py`, `src/clophfit/testing/evaluation.py`, `src/clophfit/fitting/core.py`, `src/clophfit/prtecan/titration.py`, `src/clophfit/prtecan/export.py` (MiniT imports where present + annotations)
- Modify: all `tests/*.py` carrying `FitResult[...]` (annotation sweep; three lack `from __future__ import annotations` and break at import: `tests/test_production_methods.py`, `tests/test_residuals.py`, `tests/test_fitting.py`)

**Interfaces:**

- Consumes: the three fields from Task 1.

- Produces: `class FitResult(ResidualsMixin):` (non-generic); `MiniT`/`MiniProtocol`/`MiniType` gone.

- [ ] **Step 1: Capture the failing state (why this task is atomic)**

Run: `grep -rEo "FitResult\[[A-Za-z_.]*\]" src tests --include='*.py' | wc -l`
Expected: a large count (~168). Every one of these must become plain `FitResult` in this task. This step is just to see the scope; there is no isolated unit test — the completeness gate is Step 6 (mypy clean + full suite passes).

- [ ] **Step 2: Drop the generic and narrow `mini`**

In `src/clophfit/fitting/data_structures.py`:

Change the class line (570) from:

```python
class FitResult[MiniType: MiniProtocol](ResidualsMixin):
```

to:

```python
class FitResult(ResidualsMixin):
```

Change the `mini` field type (579) from `mini: MiniType | None = None` to:

```python
    mini: Minimizer | None = None
```

Delete `MiniProtocol` (lines 489-491) and the `MiniType = TypeVar(...)` line (494). Delete the `from clophfit.clophfit_types import MiniT as MiniT  # noqa: PLC0414` re-export (line 18). Remove `Protocol`, `TypeVar`, and `runtime_checkable` from the `typing` import on line 9 **only if** nothing else in the file uses them — run ruff after and let F401 confirm.

- [ ] **Step 3: Fix the two runtime subscriptions in bayes.py**

`bayes.py:2320` is a runtime constructor call, not an annotation — it raises `TypeError` once the generic is gone. Change:

```python
    holder = FitResult(dataset=copy.deepcopy(mask_source))
```

`bayes.py:592` is a `cast` with a string first argument — change `"FitResult[MiniT]"` to `"FitResult"`.

- [ ] **Step 4: Fix the cast string in fitter_test_utils.py**

`src/clophfit/testing/fitter_test_utils.py:230` — change `cast("FitResult[MiniT]", ...)` to `cast("FitResult", ...)`.

- [ ] **Step 5: Sweep all `FitResult[...]` annotations to plain `FitResult`**

Every remaining `FitResult[X]` is a simple single-token subscript (`Any`, `typing.Any`, `MiniT`, `Minimizer`, `MinimizerResult`, `_Result`, `xr.DataTree`, `odrpack.OdrResult`). Rewrite them all to `FitResult` across src and tests. A safe global replace (the pattern never contains nested brackets):

```bash
grep -rlE "FitResult\[[A-Za-z_.]*\]" src tests --include='*.py' \
  | xargs sed -i -E 's/FitResult\[[A-Za-z_.]*\]/FitResult/g'
```

Then delete now-orphaned `MiniT` imports. These lines import `MiniT` (some alongside other names — remove only `MiniT`, keep the rest):

- `src/clophfit/__main__.py:52`, `src/clophfit/fitting/odr.py:20`, `src/clophfit/testing/evaluation.py:22`, `src/clophfit/fitting/plotting.py:61`, `src/clophfit/testing/fitter_test_utils.py:24`, and any `tests/*.py` that imported `MiniT`.

Do NOT hand-edit each annotation; run the sed, then let ruff/mypy find leftovers.

- [ ] **Step 6: Delete `MiniT` and prune its imports in clophfit_types.py**

In `src/clophfit/clophfit_types.py`, delete the `MiniT = Minimizer | odrpack.OdrResult | xr.DataTree` line (20) and its explanatory comment (15-19). Then remove the imports that existed only for it — `import odrpack` (6), `import xarray as xr` (7), `from lmfit.minimizer import Minimizer` (8) — **only if** unused elsewhere in the file. Run `uv run ruff check src/clophfit/clophfit_types.py` and delete exactly what F401 flags.

- [ ] **Step 7: Typecheck — the completeness gate**

Run: `uv run mypy src/clophfit`
Expected: clean. Any remaining `FitResult[...]` in src surfaces here as "FitResult is not generic" / "not subscriptable". Fix until clean.

- [ ] **Step 8: Full test suite — the runtime completeness gate**

Run: `uv run pytest tests/ -q -m "not slow"`
Expected: all pass. The three non-`__future__` test files (`test_production_methods.py`, `test_residuals.py`, `test_fitting.py`) would fail at **collection** with `TypeError: type 'FitResult' is not subscriptable` if any `FitResult[...]` remained in a function signature there — a clean collection proves the sweep reached them.

- [ ] **Step 9: Lint**

Run: `uv run ruff check src tests`
Expected: clean apart from the repo's known pre-existing baseline findings (verify any residual error predates this branch via `git stash` if unsure). No new F401 from orphaned `MiniT`/`Protocol`/`TypeVar` imports.

- [ ] **Step 10: Commit**

```bash
git add -A
git commit -m "refactor(types): drop FitResult generic, delete MiniT/MiniProtocol/MiniType"
```

______________________________________________________________________

## Final verification

- [ ] Full suite incl. slow: `uv run pytest tests/ -q`
- [ ] `uv run mypy src/clophfit` clean
- [ ] Grep proves the union is gone: `grep -rn "MiniT\b\|MiniProtocol\|MiniType" src tests --include='*.py'` returns nothing.
- [ ] Grep proves no stale subscripts: `grep -rEn "FitResult\[" src tests --include='*.py'` returns nothing.
- [ ] End-to-end field routing in a REPL:

```python
import numpy as np
from clophfit.fitting.core import fit_binding_glob
from clophfit.fitting.odr import fit_binding_odr
from clophfit.fitting.data_structures import DataArray, Dataset
x = np.array([5.5,6.5,7.0,7.5,8.5]); y = np.array([1.,3.,5.,7.,9.])
ds = Dataset({"1": DataArray(x, y, y_errc=np.ones_like(y))}, is_ph=True)
lm = fit_binding_glob(ds, method="lm")
print("lmfit:", lm.mini is not None, lm.trace is None, lm.odr is None)   # True True True
odr = fit_binding_odr(ds)
print("odr:", odr.odr is not None, odr.mini is None, odr.trace is None)   # True True True
```

## Known follow-ups (out of scope)

- `shared_floor` for PyMC noise priors — Part B, its own spec/plan/PR.
- `prtecan_devel.ipynb` may read `.mini` for a trace/ODR result; owner migrates it.
