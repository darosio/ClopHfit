# Plate Refit Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unify the plate-level return shape on `TitrationResults`, dissolve `pipeline.py` into a focused `noise_calibration.py`, delete both PyMC residual-refit functions in favour of explicit composition, and add conjunction outlier screening.

**Architecture:** Six tasks ordered by dependency. Task 1 (screener) is independent and lands first. Task 2 creates `noise_calibration.py` by moving functions verbatim — no behaviour change. Task 3 adds the `TitrationResults.noise_model` field that Task 4 needs. Task 4 moves FGLS onto `Titration` and deletes `pipeline.py`. Tasks 5 and 6 delete the refit functions, with Task 6 pinning `export.py` behaviour with a characterization test *before* touching it.

**Tech Stack:** Python 3.12+, pandas, numpy, lmfit, PyMC, pytest, ruff, mypy, uv.

**Spec:** `docs/superpowers/specs/2026-07-18-plate-refit-consolidation-design.md`

## Global Constraints

- Run every command through `uv run` (e.g. `uv run pytest`). Never bare `pip`/`python`.
- **Lint with `uv run ruff check --no-fix`.** `pyproject.toml` sets `fix = true` and `unsafe-fixes = true` (lines 164, 168), so a bare `ruff check` silently rewrites every `# noqa:` comment in the file to ruff 0.15's `# ruff:ignore[...]` syntax. That is unrelated churn — do not commit it.
- The repo has a **pre-existing baseline of `noqa-comments` findings** (80 across the three files touched by the last change). New `# noqa:` comments matching the file's existing idiom are correct; do not migrate them.
- `mdformat` runs as a pre-commit hook and will reformat markdown tables, failing the first commit attempt. Re-`git add` and re-commit.
- Type hints on all public functions/methods; must pass `uv run mypy src/clophfit`.
- Numpy-style docstrings on all public API (ruff `convention = "numpy"`).
- `line-length = 88`, `target-version = "py312"`. Format with `uv run ruff format`.
- Do not reformat code unrelated to the task at hand. Do not modify `pyproject.toml`.
- **Do not touch `prtecan_devel.ipynb`** — it has uncommitted owner changes.
- ruff's `PLC0415` (no function-scoped imports) and `D401` are enforced; `tests/*` is NOT exempted.
- The dependency direction is strictly `prtecan → fitting`. Nothing under `src/clophfit/fitting/` may import from `src/clophfit/prtecan/` at runtime.

______________________________________________________________________

### Task 1: Conjunction screening in `mark_outlier_probability_outliers`

Adds the `p_outlier > X AND |std_res| > Y` rule as two optional keywords on the existing primitive. Fully independent of every other task.

**Files:**

- Modify: `src/clophfit/fitting/model_validation.py:880-928`
- Test: `tests/test_residuals.py`

**Interfaces:**

- Consumes: nothing.

- Produces: `mark_outlier_probability_outliers(residuals, *, probability_col="p_outlier", threshold=0.9, exclude_col="exclude_outlier_probability", residual_threshold: float | None = None, residual_col: str = "std_res") -> pd.DataFrame`. No later task depends on this.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_residuals.py`. Check the existing top-of-file imports and add `mark_outlier_probability_outliers` to the `clophfit.fitting.model_validation` import block if absent (module-level — ruff `PLC0415`).

```python
def test_mark_outlier_probability_requires_both_criteria() -> None:
    """With residual_threshold set, a row must exceed probability AND residual."""
    df = pd.DataFrame({
        "label": ["1", "1", "1", "1"],
        "p_outlier": [0.9, 0.9, 0.2, 0.2],
        "std_res": [5.0, 1.0, 5.0, 1.0],
    })
    out = mark_outlier_probability_outliers(
        df, threshold=0.7, residual_threshold=3.0
    )
    # Only the first row clears both cutoffs.
    assert out["exclude_outlier_probability"].tolist() == [True, False, False, False]


def test_mark_outlier_probability_default_is_probability_only() -> None:
    """Omitting residual_threshold preserves the existing probability-only rule."""
    df = pd.DataFrame({
        "label": ["1", "1"],
        "p_outlier": [0.9, 0.2],
        "std_res": [0.1, 9.0],
    })
    out = mark_outlier_probability_outliers(df, threshold=0.7)
    # std_res is ignored entirely when residual_threshold is None.
    assert out["exclude_outlier_probability"].tolist() == [True, False]


def test_mark_outlier_probability_missing_residual_column_marks_nothing() -> None:
    """A requested residual column that is absent excludes no rows."""
    df = pd.DataFrame({"label": ["1"], "p_outlier": [0.99]})
    out = mark_outlier_probability_outliers(
        df, threshold=0.7, residual_threshold=3.0
    )
    assert out["exclude_outlier_probability"].tolist() == [False]
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_residuals.py -k "mark_outlier_probability" -v`
Expected: FAIL — `TypeError: mark_outlier_probability_outliers() got an unexpected keyword argument 'residual_threshold'` for the first and third tests. The second test PASSES already (it pins existing behaviour) — that is intended, it is the regression guard.

- [ ] **Step 3: Add the two keywords**

In `src/clophfit/fitting/model_validation.py`, change the signature (880-886) to:

```python
def mark_outlier_probability_outliers(
    residuals: _t.Any,
    *,
    probability_col: str = "p_outlier",
    threshold: float = 0.9,
    exclude_col: str = "exclude_outlier_probability",
    residual_threshold: float | None = None,
    residual_col: str = "std_res",
) -> pd.DataFrame:
```

Add to the docstring's Parameters section, after `exclude_col`:

```
    residual_threshold : float | None, optional
        When set, a row is marked only if it exceeds both *threshold* and this
        absolute-residual cutoff on *residual_col*. ``None`` (the default)
        applies the probability criterion alone.
    residual_col : str, optional
        Column holding the standardized residual compared against
        *residual_threshold*.
```

Replace the final assignment block (the last three lines of the body, currently
`out[exclude_col] = probabilities > threshold` onward) with:

```python
    marked = probabilities > threshold
    if residual_threshold is not None:
        residual_values = (
            out[residual_col]
            if residual_col in out.columns
            else pd.Series(np.nan, index=out.index)
        )
        residual_magnitude = pd.to_numeric(residual_values, errors="coerce").abs()
        marked &= residual_magnitude > residual_threshold
    out[exclude_col] = marked
    out["residual_outlier_score"] = probabilities
    return out
```

`np` and `pd` are already imported in this module. A missing or non-numeric
`residual_col` yields `NaN`, and `NaN > threshold` is `False`, so nothing is
marked — which is what the third test asserts.

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_residuals.py -k "mark_outlier_probability" -v`
Expected: all three PASS.

- [ ] **Step 5: Run the regression suites**

Run: `uv run pytest tests/test_residuals.py tests/test_bayes.py -q`
Expected: all pass. Existing callers pass no `residual_threshold`, so behaviour is unchanged.

- [ ] **Step 6: Lint, typecheck, commit**

```bash
uv run ruff format src/clophfit/fitting/model_validation.py tests/test_residuals.py
uv run ruff check --no-fix src/clophfit/fitting/model_validation.py tests/test_residuals.py
uv run mypy src/clophfit
git add src/clophfit/fitting/model_validation.py tests/test_residuals.py
git commit -m "feat(model-validation): add conjunction screening to outlier probability marking"
```

______________________________________________________________________

### Task 2: Create `noise_calibration.py` and move the noise-model family

A pure move: no signature or behaviour changes. `pipeline.py` survives this task holding only `fgls_plate_fit`, which now imports its helpers from the new module. Task 4 deletes it.

**Files:**

- Create: `src/clophfit/fitting/noise_calibration.py`
- Modify: `src/clophfit/fitting/utils.py` (remove lines 337-634: `fit_rel_error_from_residuals`, `fit_gain_from_residuals`, `fit_noise_model_nnls`, `compute_binding_slope`, `compute_plate_slopes`, `fit_ph_slope_noise`)
- Modify: `src/clophfit/fitting/pipeline.py` (remove `_noise_params_converged` 28-43, `_plate_noise_model_from_nnls` 46-61, `calibrate_noise_robust` 64-115; update imports)
- Modify: `tests/test_outlier_scores.py`, `tests/test_pipeline.py` (import paths)

**Interfaces:**

- Consumes: nothing.

- Produces: module `clophfit.fitting.noise_calibration` exporting the public
  `calibrate_noise_robust`, `fit_rel_error_from_residuals`,
  `fit_gain_from_residuals`, `fit_noise_model_nnls`, `compute_binding_slope`,
  `compute_plate_slopes`, `fit_ph_slope_noise`, plus the private
  `_plate_noise_model_from_nnls` and `_noise_params_converged`.

  Task 4 imports exactly five of these: `_plate_noise_model_from_nnls`,
  `_noise_params_converged`, `fit_noise_model_nnls`, `compute_plate_slopes`,
  `fit_ph_slope_noise`. `calibrate_noise_robust` is **not** used by FGLS — it
  is the PyMC-side calibrator, kept public for benchmark and notebook use.

- [ ] **Step 1: Verify the current baseline is green**

Run: `uv run pytest tests/test_pipeline.py tests/test_outlier_scores.py tests/test_residuals.py -q`
Expected: all pass. This is a refactor with no new tests; the existing suite is the safety net, so confirm it is green *before* moving anything.

- [ ] **Step 2: Create the new module by moving the code verbatim**

Run this script. It extracts the exact line ranges so no body is retyped:

```bash
uv run python - <<'PYEOF'
from pathlib import Path

utils = Path("src/clophfit/fitting/utils.py").read_text().splitlines(keepends=True)
pipeline = Path("src/clophfit/fitting/pipeline.py").read_text().splitlines(keepends=True)

# utils.py lines 337..end  -> the six noise/slope functions (1-indexed, inclusive)
moved_utils = "".join(utils[336:])
# pipeline.py lines 28..115 -> the three calibration helpers
moved_pipeline = "".join(pipeline[27:115])

header = '''"""Noise-model calibration from fit residuals.

Estimators that turn a canonical residual table into a
:class:`~clophfit.fitting.data_structures.PlateNoiseModel`: per-label floor,
photon gain, and proportional error, plus the plate slope helpers used to
propagate x-axis noise.
"""

import logging
import typing

import numpy as np
import pandas as pd
from scipy import optimize, stats

from clophfit.fitting.data_structures import (
    Dataset,
    NoiseModelParams,
    PlateNoiseModel,
    compute_noise_variance,
)
from clophfit.fitting.utils import fit_trendline

logger = logging.getLogger(__name__)


'''

Path("src/clophfit/fitting/noise_calibration.py").write_text(
    header + moved_pipeline + "\n" + moved_utils
)

Path("src/clophfit/fitting/utils.py").write_text("".join(utils[:336]))
Path("src/clophfit/fitting/pipeline.py").write_text(
    "".join(pipeline[:27]) + "".join(pipeline[115:])
)
print("moved")
PYEOF
```

- [ ] **Step 3: Fix imports in all three modules**

`noise_calibration.py`: the header above imports a superset. Run
`uv run ruff check --no-fix src/clophfit/fitting/noise_calibration.py` and delete
exactly what `F401` flags as unused. If any moved function references a name not
in the header (for example `ArrayMask`, `copy`, or `binding_1site`), `F821`
will flag it — add that import rather than guessing.

`utils.py`: remove now-unused imports the same way (`F401` will flag e.g.
`optimize`, `stats`, `PlateNoiseModel`, `compute_noise_variance` if nothing
remaining uses them). Keep `fit_trendline` in `utils.py` — `noise_calibration`
imports it, and moving it would create a cycle.

`pipeline.py`: replace its import block so it pulls the helpers from the new
module. Its remaining content is only `fgls_plate_fit`:

```python
"""Iterative FGLS plate fitting."""

import logging

from clophfit.fitting.core import fit_binding_glob
from clophfit.fitting.data_structures import Dataset, FitResult, PlateNoiseModel
from clophfit.fitting.errors import InsufficientDataError
from clophfit.fitting.model_validation import residuals_from_fit_results
from clophfit.fitting.models import binding_1site
from clophfit.fitting.noise_calibration import (
    _noise_params_converged,
    _plate_noise_model_from_nnls,
    compute_plate_slopes,
    fit_noise_model_nnls,
    fit_ph_slope_noise,
)

logger = logging.getLogger(__name__)
```

- [ ] **Step 4: Update the two test files' imports**

In `tests/test_outlier_scores.py` and `tests/test_pipeline.py`, repoint imports of
the six moved `utils` functions and the three moved `pipeline` helpers to
`clophfit.fitting.noise_calibration`.

`tests/test_pipeline.py` also monkeypatches by string path (lines 88, 92, 95,
150, 154, 157). Those patch targets must follow the lookup, not the definition:
`pipeline.py` imports these names into its own namespace, so the correct target
remains `clophfit.fitting.pipeline.<name>`. **Leave those strings unchanged.**
Only `clophfit.fitting.pipeline.calibrate_noise_robust`, if patched anywhere,
would need repointing — grep to confirm.

Run: `grep -rn "clophfit.fitting.pipeline\.\|clophfit.fitting.utils import" tests/`

- [ ] **Step 5: Verify nothing broke**

Run: `uv run pytest tests/ -q -m "not slow"`
Expected: all pass, same count as Step 1 plus Task 1's three new tests.

Run: `uv run mypy src/clophfit`
Expected: clean.

- [ ] **Step 6: Confirm no import cycle was introduced**

Run: `uv run python -c "import clophfit.fitting.noise_calibration, clophfit.fitting.utils, clophfit.fitting.pipeline; print('ok')"`
Expected: `ok`. A cycle would raise `ImportError: cannot import name ... (most likely due to a circular import)`.

- [ ] **Step 7: Lint, format, commit**

```bash
uv run ruff format src/clophfit/fitting/noise_calibration.py src/clophfit/fitting/utils.py src/clophfit/fitting/pipeline.py tests/test_outlier_scores.py tests/test_pipeline.py
uv run ruff check --no-fix src/clophfit/fitting tests/test_outlier_scores.py tests/test_pipeline.py
git add -A src/clophfit/fitting tests/test_outlier_scores.py tests/test_pipeline.py
git commit -m "refactor(fitting): extract noise_calibration module from utils and pipeline"
```

______________________________________________________________________

### Task 3: `TitrationResults.noise_model` field

Adds the optional field, appended last so no existing positional construction shifts.

**Files:**

- Modify: `src/clophfit/prtecan/titration.py:480-484`
- Test: `tests/test_prtecan.py`

**Interfaces:**

- Consumes: nothing.

- Produces: `TitrationResults.noise_model: PlateNoiseModel | None = None`, declared **after** `_dataframe`. Task 4 sets it via the keyword `noise_model=`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_prtecan.py`. `TitrationResults` and `PlateNoiseModel` must be importable — add to the existing top-of-file import blocks if absent.

```python
def test_titration_results_noise_model_defaults_to_none() -> None:
    """The noise_model field is optional and absent for a plain plate fit."""
    assert TitrationResults().noise_model is None


def test_titration_results_noise_model_is_last_positional() -> None:
    """Appending noise_model must not shift any existing positional argument."""
    scheme = PlateScheme()
    fit_keys = {"A01"}
    results: dict[str, FitResult] = {}
    tr = TitrationResults(scheme, fit_keys, results)
    # The three historical positional args still bind to their own fields.
    assert tr.scheme is scheme
    assert tr.fit_keys == fit_keys
    assert tr.results is results
    assert tr.noise_model is None
    # And it is settable by keyword.
    nm = PlateNoiseModel()
    assert TitrationResults(scheme, fit_keys, results, noise_model=nm).noise_model is nm
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_prtecan.py -k "noise_model" -v`
Expected: FAIL — `AttributeError: 'TitrationResults' object has no attribute 'noise_model'`, and `TypeError: __init__() got an unexpected keyword argument 'noise_model'`.

- [ ] **Step 3: Add the field**

In `src/clophfit/prtecan/titration.py`, the field block currently reads:

```python
    scheme: PlateScheme = field(default_factory=PlateScheme)
    fit_keys: set[str] = field(default_factory=set)
    results: dict[str, FitResult] = field(default_factory=dict)
    _dataframe: pd.DataFrame = field(default_factory=pd.DataFrame)
    titration: InitVar[Titration | None] = None
```

Insert `noise_model` after `_dataframe` and before the `InitVar`:

```python
    scheme: PlateScheme = field(default_factory=PlateScheme)
    fit_keys: set[str] = field(default_factory=set)
    results: dict[str, FitResult] = field(default_factory=dict)
    _dataframe: pd.DataFrame = field(default_factory=pd.DataFrame)
    noise_model: PlateNoiseModel | None = None
    titration: InitVar[Titration | None] = None
```

Add to the class docstring, after the existing body:

```
    ``noise_model`` carries the calibrated per-label noise model when the fit
    produced one (``fgls_fit_plate``); it is ``None`` for plain ``fit_plate``.
```

`PlateNoiseModel` is already imported at `titration.py:22-29`.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_prtecan.py -k "noise_model" -v`
Expected: PASS.

- [ ] **Step 5: Regression**

Run: `uv run pytest tests/ -q -m "not slow"`
Expected: all pass. In particular `tests/test_prtecan.py` and `tests/test_residuals.py` exercise positional `TitrationResults(...)` construction.

- [ ] **Step 6: Lint, typecheck, commit**

```bash
uv run ruff format src/clophfit/prtecan/titration.py tests/test_prtecan.py
uv run ruff check --no-fix src/clophfit/prtecan/titration.py tests/test_prtecan.py
uv run mypy src/clophfit
git add src/clophfit/prtecan/titration.py tests/test_prtecan.py
git commit -m "feat(prtecan): add optional noise_model field to TitrationResults"
```

______________________________________________________________________

### Task 4: `Titration.fgls_fit_plate` and delete `pipeline.py`

Moves the FGLS converge-loop onto `Titration` so it can return `TitrationResults`, resolving the `prtecan → fitting` layering constraint. Algorithm unchanged.

**Files:**

- Modify: `src/clophfit/prtecan/titration.py` (add method after `fit_plate`, which ends at line 1236; add imports)
- Delete: `src/clophfit/fitting/pipeline.py`
- Modify: `tests/test_pipeline.py` (port the two FGLS tests), `tests/test_residuals.py:684-700` (port `test_fgls_plate_fit_workflow`)

**Interfaces:**

- Consumes: `TitrationResults.noise_model` (Task 3); `_noise_params_converged`, `_plate_noise_model_from_nnls`, `fit_noise_model_nnls`, `compute_plate_slopes`, `fit_ph_slope_noise` from `clophfit.fitting.noise_calibration` (Task 2).

- Produces: `Titration.fgls_fit_plate(datasets=None, *, label=None, sigma_floor=None, first_pass_method="huber", second_pass_method="lm", max_iter=3, tol=1e-3) -> TitrationResults`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_prtecan.py`. It needs a `Titration`; reuse whatever fixture that file already uses for `fit_plate` tests (grep for `fit_plate(` in `tests/test_prtecan.py` to find it) and follow that pattern.

```python
def test_fgls_fit_plate_returns_titration_results_with_noise_model(
    titration_fixture: Titration,
) -> None:
    """FGLS returns a TitrationResults carrying its calibrated noise model."""
    res = titration_fixture.fgls_fit_plate(max_iter=1)

    assert isinstance(res, TitrationResults)
    assert isinstance(res.noise_model, PlateNoiseModel)
    # Plate metadata is populated from the titration, like fit_plate.
    assert res.scheme is titration_fixture.scheme
    assert res.fit_keys == titration_fixture.fit_keys
    # The unified residual accessor works, which the tuple return could not offer.
    assert not res.residuals.empty
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_prtecan.py -k "fgls_fit_plate" -v`
Expected: FAIL — `AttributeError: 'Titration' object has no attribute 'fgls_fit_plate'`.

- [ ] **Step 3: Add the imports to `titration.py`**

Extend the existing `clophfit.fitting` import block (around lines 18-38) with:

```python
from clophfit.fitting.model_validation import (
    RESIDUAL_TABLE_COLUMNS,
    residuals_from_fit_results,
)
from clophfit.fitting.models import binding_1site
from clophfit.fitting.noise_calibration import (
    _noise_params_converged,  # noqa: PLC2701
    _plate_noise_model_from_nnls,  # noqa: PLC2701
    compute_plate_slopes,
    fit_noise_model_nnls,
    fit_ph_slope_noise,
)
```

`RESIDUAL_TABLE_COLUMNS` is already imported at line 31 — merge, do not duplicate.
`PLC2701` (private-name import) is suppressed because these two helpers stay
private to `fitting`; if ruff does not flag it, drop the comments.

- [ ] **Step 4: Add the method**

Insert into `class Titration`, immediately after `fit_plate` ends (line 1236, just before `plot_temperature`):

```python
    def fgls_fit_plate(  # noqa: PLR0913
        self,
        datasets: dict[str, Dataset] | None = None,
        *,
        label: str | None = None,
        sigma_floor: dict[str, float] | None = None,
        first_pass_method: str = "huber",  # noqa: S107
        second_pass_method: str = "lm",  # noqa: S107
        max_iter: int = 3,
        tol: float = 1e-3,
    ) -> TitrationResults:
        """Run iterative Feasible Generalized Least Squares (FGLS) on the plate.

        Fits every well with *first_pass_method* using the existing ``y_errc``,
        calibrates a per-label noise model from the plate-wide residuals with
        the floor anchored to *sigma_floor*, re-applies the calibrated weights
        and re-fits with *second_pass_method*, iterating until gain and alpha
        converge or *max_iter* is reached.

        Parameters
        ----------
        datasets : dict[str, Dataset] | None
            Mapping of well keys to `Dataset` objects. When ``None``, datasets
            are built with :meth:`create_dataset_dict`.
        label : str | None
            Build per-label datasets for this label instead of global ones.
            Only valid when *datasets* is ``None``.
        sigma_floor : dict[str, float] | None
            Known read-noise floor per label. Defaults to :attr:`bg_noise`.
        first_pass_method : str
            Method for the first-pass fit.
        second_pass_method : str
            Method for subsequent passes.
        max_iter : int
            Maximum FGLS iterations.
        tol : float
            Relative tolerance for gain/alpha convergence.

        Returns
        -------
        TitrationResults
            Plate results carrying this titration's ``scheme`` and
            ``fit_keys``, with ``noise_model`` set to the converged (or last)
            calibration.

        Raises
        ------
        ValueError
            If both *datasets* and *label* are given.
        """
        if datasets is not None and label is not None:
            msg = "Pass either `datasets` or `label`, not both."
            raise ValueError(msg)
        if datasets is None:
            datasets = self.create_dataset_dict(label)
        floors_in = dict(self.bg_noise) if sigma_floor is None else dict(sigma_floor)

        noise_model: PlateNoiseModel | None = None
        results: dict[str, FitResult] = {}

        for iteration in range(max_iter):
            method = first_pass_method if iteration == 0 else second_pass_method
            if iteration == 0:
                current_ds = datasets
            else:
                current_ds = noise_model.apply_to_plate(  # type: ignore[union-attr]
                    datasets, compute_plate_slopes(results)
                )
            logger.info("FGLS iteration %d: %s fit", iteration + 1, method)

            results = {}
            for well, ds in current_ds.items():
                try:
                    results[well] = fit_binding_glob(ds, method=method)
                except InsufficientDataError:
                    logger.warning(
                        "Skip FGLS fit for well %s (iteration %d).",
                        well,
                        iteration + 1,
                    )
                    results[well] = FitResult()

            df_res = residuals_from_fit_results(
                results, trace_id="", binding_function=binding_1site
            )
            try:
                floors, gains, alphas = fit_noise_model_nnls(
                    df_res, sigma_floor_fixed=floors_in
                )
            except ValueError as e:
                logger.warning("FGLS calibration failed (%s).", e)
                gains = dict.fromkeys(floors_in, 0.0)
                alphas = dict.fromkeys(floors_in, 0.0)
                floors = dict(floors_in)

            plate_slopes = compute_plate_slopes(results)
            tmp_noise = _plate_noise_model_from_nnls(floors, gains, alphas)
            sigma_ph = fit_ph_slope_noise(df_res, tmp_noise, plate_slopes)
            new_noise = _plate_noise_model_from_nnls(floors, gains, alphas, sigma_ph)

            for lbl, params in new_noise.items():
                logger.info(
                    "Calibrated [%s] iter %d: sigma=%.2f, gain=%.3f, alpha=%.3f, "
                    "sigma_ph=%.4f",
                    lbl,
                    iteration + 1,
                    params.sigma_floor,
                    params.gain,
                    params.alpha,
                    params.sigma_ph,
                )

            converged = iteration > 0 and _noise_params_converged(
                noise_model,  # type: ignore[arg-type]
                new_noise,
                tol,
            )
            noise_model = new_noise
            if converged:
                logger.info("FGLS converged after %d iterations.", iteration + 1)
                break

        return TitrationResults(
            self.scheme, self.fit_keys, results, noise_model=noise_model
        )
```

Note the convergence block is restructured versus the original: the original
assigned `noise_model = new_noise` in both branches. This version assigns once
then breaks, which is equivalent and avoids the duplicated assignment.

- [ ] **Step 5: Run to verify it passes**

Run: `uv run pytest tests/test_prtecan.py -k "fgls_fit_plate" -v`
Expected: PASS.

- [ ] **Step 6: Port the existing FGLS tests, then delete `pipeline.py`**

Rewrite the two tests in `tests/test_pipeline.py` (`test_fgls_plate_fit_uses_calibration_fallback_and_second_pass` at line 59, `test_fgls_plate_fit_converges_and_handles_failed_well` at line 116) and `test_fgls_plate_fit_workflow` in `tests/test_residuals.py:684` to drive `Titration.fgls_fit_plate` and assert on `TitrationResults`. Their monkeypatch string targets move from `clophfit.fitting.pipeline.<name>` to `clophfit.prtecan.titration.<name>`, because `titration.py` is now the module that looks those names up.

Update the `test_residuals.py` assertions from:

```python
    final_results, noise_params = fgls_plate_fit(datasets, sigma_floor)
    assert "well1" in final_results
    assert final_results["well1"].result is not None
    assert final_results["well1"].result.success
    assert isinstance(noise_params, PlateNoiseModel)
```

to:

```python
    res = titration.fgls_fit_plate(datasets, sigma_floor=sigma_floor)
    assert "well1" in res.results
    assert res.results["well1"].result is not None
    assert res.results["well1"].result.success
    assert isinstance(res.noise_model, PlateNoiseModel)
```

Then remove the file and its stale import:

```bash
git rm src/clophfit/fitting/pipeline.py
```

Remove `from clophfit.fitting.pipeline import fgls_plate_fit` from
`tests/test_residuals.py:26`. If `tests/test_pipeline.py` now only holds the
`calibrate_noise_robust` tests (lines 168, 189), rename it to
`tests/test_noise_calibration.py` with `git mv`.

- [ ] **Step 7: Verify nothing references the deleted module**

Run: `grep -rn "fgls_plate_fit\|fitting.pipeline\|fitting import pipeline" src tests --include='*.py'`
Expected: no output.

Run: `uv run pytest tests/ -q -m "not slow"`
Expected: all pass.

- [ ] **Step 8: Lint, typecheck, commit**

```bash
uv run ruff format src/clophfit/prtecan/titration.py tests/
uv run ruff check --no-fix src/clophfit tests
uv run mypy src/clophfit
git add -A
git commit -m "refactor(prtecan): move FGLS onto Titration.fgls_fit_plate returning TitrationResults"
```

______________________________________________________________________

### Task 5: Delete `fit_binding_pymc_multi_residual_refit`

Zero production callers — a pure deletion.

**Files:**

- Modify: `src/clophfit/fitting/bayes.py` (delete `PymcMultiResidualRefitResult` 1445-1452, `fit_binding_pymc_multi_residual_refit` 2378 through the end of its body, `__all__` entry line 310)
- Modify: `tests/test_bayes.py` (delete the two tests at lines 1520 and 1611)

**Interfaces:**

- Consumes: nothing.

- Produces: nothing. Purely subtractive.

- [ ] **Step 1: Confirm there are still no production callers**

Run: `grep -rn "fit_binding_pymc_multi_residual_refit\|PymcMultiResidualRefitResult" src tests --include='*.py'`
Expected: only `src/clophfit/fitting/bayes.py` (definition, `__all__`) and `tests/test_bayes.py` (two tests). If anything else appears, STOP and report — the spec's premise no longer holds.

- [ ] **Step 2: Delete the function, its result dataclass, and its tests**

Remove from `src/clophfit/fitting/bayes.py`:

- the `@dataclass` + `class PymcMultiResidualRefitResult` block (1445-1452)
- the whole `def fit_binding_pymc_multi_residual_refit(...)` function starting at 2378, up to but not including the next top-level `def`/`@` at column 0
- the `"fit_binding_pymc_multi_residual_refit",` line from `__all__` (line 310)

Remove from `tests/test_bayes.py` the two whole test functions
`test_fit_binding_pymc_multi_residual_refit_uses_well_noise_scale` (1520) and
`test_fit_binding_pymc_multi_residual_refit_can_share_well_noise_scale` (1611),
including their local monkeypatch stubs.

- [ ] **Step 3: Verify the deletion is complete and nothing else broke**

Run: `grep -rn "fit_binding_pymc_multi_residual_refit\|PymcMultiResidualRefitResult" src tests --include='*.py'`
Expected: no output.

Run: `uv run pytest tests/ -q -m "not slow"`
Expected: all pass.

Run: `uv run python -c "from clophfit.fitting import bayes; print(bayes.__all__)"`
Expected: prints the list with no dangling name; an entry naming a deleted object would make `from clophfit.fitting.bayes import *` raise `AttributeError`.

- [ ] **Step 4: Lint, typecheck, commit**

```bash
uv run ruff check --no-fix src/clophfit/fitting/bayes.py tests/test_bayes.py
uv run mypy src/clophfit
git add src/clophfit/fitting/bayes.py tests/test_bayes.py
git commit -m "refactor(bayes): remove unused fit_binding_pymc_multi_residual_refit"
```

______________________________________________________________________

### Task 6: Pin `export.py` behaviour, inline the two-pass, delete the single refit

The characterization test comes first and must pass **before and after** the change — that is the entire proof that `--mcmc single-refit` still behaves identically.

**Files:**

- Test: `tests/test_prtecan.py` (characterization test, written first)
- Modify: `src/clophfit/prtecan/export.py` (imports line 13; `fit_single_mcmc` 105-161)
- Modify: `src/clophfit/fitting/bayes.py` (delete `PymcResidualRefitResult` 1435-1442, `fit_binding_pymc_residual_refit` 2196-2375, `__all__` entry line 311)
- Modify: `tests/test_bayes.py` (delete the three tests at 1366, 1419, 1469)

**Interfaces:**

- Consumes: nothing from earlier tasks.
- Produces: private `export._single_refit_two_pass(ds, bg_noise, sampler) -> tuple[FitResult, pd.DataFrame]` returning `(final_fit, initial_residuals)`.

**Note — deviation from the spec.** The spec estimated the inlining at "~6 lines". It is closer to 35, because the `ye_mag` strategy computes per-label log-scale priors before the first pass. Rather than inflate `fit_single_mcmc`, extract it as the private helper above, local to `export.py`. This still removes the frozen public API and keeps the composition explicit at the call site.

- [ ] **Step 1: Write the characterization test**

Add to `tests/test_prtecan.py`. It records the two `fit_binding_pymc` calls that
`--mcmc single-refit` makes, so it pins the two-pass contract without sampling.

```python
def test_single_refit_two_pass_contract(
    monkeypatch: pytest.MonkeyPatch,
    titration_fixture: Titration,
    tmp_path: Path,
) -> None:
    """--mcmc single-refit runs a robust pass then an unrobust refit."""
    calls: list[dict[str, object]] = []

    def fake_fit_binding_pymc(ds_or_fr: object, **kwargs: object) -> FitResult:
        calls.append(kwargs)
        return fit_binding_glob(titration_fixture.create_global_ds("A01"))

    # Patch BOTH lookup sites so this test is valid before and after the
    # inlining: today the call happens inside bayes, afterwards inside export.
    monkeypatch.setattr(bayes, "fit_binding_pymc", fake_fit_binding_pymc)
    monkeypatch.setattr(export, "fit_binding_pymc", fake_fit_binding_pymc)

    titration_fixture.params.mcmc = "single-refit"
    titration_fixture.params.n_mcmc_samples = 7
    res = export.fit_single_mcmc(
        titration_fixture,
        {"A01": titration_fixture.create_global_ds("A01")},
        tmp_path,
    )

    assert res is not None
    assert len(calls) == 2
    first, second = calls
    # Pass 1 is robust; pass 2 is not.
    assert first["robust"].enabled is True
    assert second["robust"].enabled is False
    # Both use the ye_mag noise strategy, unshared, lognormal.
    assert first["noise"].kind == "ye_mag"
    assert second["noise"].kind == "ye_mag"
    assert first["noise"].shared_ye_mags is False
    assert first["noise"].ye_mag_prior == "lognormal"
    # The refit's ye_mag prior is recentred on 0 with a tighter sigma.
    assert second["noise"].ye_mag_mu == 0.0
    assert second["noise"].ye_mag_sigma == 0.25
    # Sampler settings come from titration params in both passes.
    assert first["sampler"].n_samples == 7
    assert second["sampler"].n_samples == 7
```

Add `pytest`, `Path`, `bayes`, `export`, `fit_binding_glob`, `FitResult` to the
file's imports if absent (module level — ruff `PLC0415`).

`titration_fixture` is a placeholder for whatever `Titration` fixture this file
already uses — run `grep -n "fit_plate(" tests/test_prtecan.py` to find the
established one and follow that pattern, as in Task 4. The fixture must have at
least one well in `fit_keys`; adjust the `"A01"` key to match it.

The assertions deliberately check individual kwargs rather than comparing the
whole dict, because today's call also passes `n_sd`, `n_xerr` and `min_x_step`
explicitly while the inlined version relies on their defaults. Asserting dict
equality would make this test fail after Step 5 for a reason that is not a
behaviour change.

- [ ] **Step 2: Run it against the CURRENT code**

Run: `uv run pytest tests/test_prtecan.py -k "single_refit_two_pass_contract" -v`
Expected: **PASS**. This is the point of a characterization test — it must pass
before the refactor. If it fails, the assertions do not match today's behaviour;
fix the test, not the source, until it passes.

- [ ] **Step 3: Commit the pin on its own**

```bash
git add tests/test_prtecan.py
git commit -m "test(prtecan): pin --mcmc single-refit two-pass contract"
```

Committing separately means the next commit's diff shows the behaviour was
already locked before anything moved.

- [ ] **Step 4: Add the private helper to `export.py`**

Insert above `fit_single_mcmc` (line 105):

```python
def _single_refit_two_pass(
    ds: Dataset,
    bg_noise: Mapping[str, float] | float,
    sampler: SamplerConfig,
) -> tuple[FitResult, pd.DataFrame]:
    """Screen residual outliers with a robust PyMC pass, then refit unrobustly.

    Parameters
    ----------
    ds : Dataset
        Single-well multi-label titration dataset.
    bg_noise : Mapping[str, float] | float
        Per-label background-noise hints seeding the screening ``ye_mag`` prior.
    sampler : SamplerConfig
        Sampling controls used for both passes.

    Returns
    -------
    tuple[FitResult, pd.DataFrame]
        The refit result and the screening pass's residual table.
    """
    if isinstance(bg_noise, MappingABC):
        first_mu: float | dict[str, float] = {
            str(label): float(np.log(max(float(value) * 3.6, 1e-6)))
            for label, value in bg_noise.items()
        }
    else:
        first_mu = float(np.log(max(float(bg_noise) * 3.6, 1e-6)))

    initial = fit_binding_pymc(
        dataset_with_unit_yerr(ds),
        robust=RobustConfig(enabled=True),
        noise=NoiseConfig.ye_mag(
            shared=False, prior="lognormal", mu=first_mu, sigma=0.5
        ),
        sampler=sampler,
    )
    residuals = residuals_from_fit_results(
        {"single": initial},
        "pymc_robust_unweighted",
        binding_1site,
        robust=True,
        outlier_threshold=3.0,
    )
    residuals = mark_excess_residual_outliers(
        residuals,
        threshold=3.0,
        allowed_tail_fraction=0.0,
        min_allowed_tail_count=0,
    )
    mask_source = initial.dataset if initial.dataset is not None else ds
    holder = FitResult(dataset=copy.deepcopy(mask_source))
    masked = masked_datasets_from_residual_outliers(
        {"single": holder}, residuals, min_keep=3
    ).get("single", copy.deepcopy(mask_source))
    seeded = copy.deepcopy(initial)
    seeded.dataset = masked

    final = fit_binding_pymc(
        seeded,
        robust=RobustConfig(enabled=False),
        noise=NoiseConfig.ye_mag(
            shared=False, prior="lognormal", mu=0.0, sigma=0.25
        ),
        sampler=sampler,
    )
    return final, residuals
```

The deleted function passed `n_sd=10.0`, `n_xerr=1.0` and `min_x_step=0.2`
explicitly; those are already `fit_binding_pymc`'s defaults, so they are omitted
here. Verify against `bayes.py:1996-1998` before dropping them.

Update `export.py`'s imports (replacing line 13):

```python
from collections.abc import Mapping
from collections.abc import Mapping as MappingABC

from clophfit.fitting.bayes import dataset_with_unit_yerr, fit_binding_pymc
from clophfit.fitting.bayes_config import NoiseConfig, RobustConfig, SamplerConfig
from clophfit.fitting.data_structures import Dataset, FitResult
from clophfit.fitting.model_validation import (
    mark_excess_residual_outliers,
    masked_datasets_from_residual_outliers,
    residuals_from_fit_results,
)
```

Merge with the existing `FitResult`, `residuals_from_fit_results`, `binding_1site`
and `SamplerConfig` imports rather than duplicating them.

- [ ] **Step 5: Rewrite the `single-refit` branch to use the helper**

Replace the body from `mcmc_fits = {}` (line 144) through the `return` at 161:

```python
    sampler = SamplerConfig(
        n_samples=titration.params.n_mcmc_samples,
        nuts_sampler=titration.params.nuts_sampler,
    )
    mcmc_fits = {}
    residual_rows = []
    for key, ds in datasets.items():
        final, residuals = _single_refit_two_pass(ds, titration.bg_noise, sampler)
        mcmc_fits[key] = final
        if not residuals.empty:
            residual_rows.append(residuals.assign(well=key))
    if residual_rows:
        pd.concat(residual_rows, ignore_index=True).to_csv(
            outfit / "single_refit_initial_residual_outliers.csv", index=False
        )
    return TitrationResults(titration.scheme, titration.fit_keys, mcmc_fits)
```

- [ ] **Step 6: Run the characterization test — it must STILL pass**

Run: `uv run pytest tests/test_prtecan.py -k "single_refit_two_pass_contract" -v`
Expected: PASS, unchanged from Step 2. A failure here means the inlining altered
behaviour; fix the helper, do not weaken the test.

- [ ] **Step 7: Delete the single refit function and its tests**

Remove from `src/clophfit/fitting/bayes.py`:

- the `@dataclass` + `class PymcResidualRefitResult` block (1435-1442)
- the whole `def fit_binding_pymc_residual_refit(...)` (2196-2375)
- the `"fit_binding_pymc_residual_refit",` line from `__all__` (line 311)

Remove from `tests/test_bayes.py` the three tests at lines 1366, 1419 and 1469.

- [ ] **Step 8: Verify complete removal**

Run: `grep -rn "fit_binding_pymc_residual_refit\|PymcResidualRefitResult" src tests --include='*.py'`
Expected: no output.

Run: `uv run python -c "from clophfit.fitting import bayes; print(bayes.__all__)"`
Expected: `['fit_binding_pymc', 'fit_binding_pymc_multi', 'process_trace']`.

Run: `uv run pytest tests/ -q -m "not slow"`
Expected: all pass, including the characterization test.

- [ ] **Step 9: Lint, typecheck, commit**

```bash
uv run ruff format src/clophfit/prtecan/export.py src/clophfit/fitting/bayes.py tests/test_bayes.py
uv run ruff check --no-fix src/clophfit tests
uv run mypy src/clophfit
git add -A
git commit -m "refactor(prtecan): inline single-refit two-pass and drop fit_binding_pymc_residual_refit"
```

______________________________________________________________________

## Final verification

- [ ] Full suite incl. slow: `uv run pytest tests/ -q`
- [ ] `uv run mypy src/clophfit` clean
- [ ] `uv run ruff check --no-fix src tests` shows only the pre-existing `noqa-comments` baseline
- [ ] No dangling references:
  `grep -rn "fgls_plate_fit\|fitting.pipeline\|residual_refit\|PymcResidualRefitResult\|PymcMultiResidualRefitResult" src tests --include='*.py'`
  returns nothing.
- [ ] `src/clophfit/fitting/pipeline.py` no longer exists.
- [ ] The user's empirical pipeline is expressible end-to-end:

```python
res = tit.fit_plate(method="mcmc", init=..., robust=..., noise=..., sampler=...)
marked = mark_outlier_probability_outliers(
    res.residuals, threshold=0.7, residual_threshold=3.0
)
masked = masked_datasets_from_outlier_probabilities(res.results, marked)
final = tit.fit_plate(masked, method="mcmc", robust=RobustConfig(enabled=False), ...)
```

## Known follow-ups (out of scope)

- Outlier removal inside the FGLS loop — the benchmark decides.
- Whether `--mcmc single-refit` should adopt the `student_t` + conjunction pipeline.
- The multi-plate benchmark harness — its own spec, built on this interface.
- `pymc_multi` / `MultiFitResult` return-shape reconciliation.
