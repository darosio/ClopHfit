# Plate-Fit Residuals Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give plate-level fits a `.residuals` accessor matching `FitResult.residuals` and `MultiFitResult.residuals`, by moving `fit_plate` onto `Titration` and returning `TitrationResults`.

**Architecture:** `fit_plate` moves from `clophfit.fitting.pipeline` (a module for multistage FGLS orchestration, which it never belonged to) onto `Titration` as a method returning `TitrationResults` — legal because `prtecan → fitting` is the established dependency direction. The residual-settings auto-detection currently duplicated in two `residual_table` bodies is extracted into a `ResidualsMixin`, which then serves `FitResult`, `MultiFitResult`, and `TitrationResults`. `MultiFitResult` is untouched structurally: it is a trace proxy, not a plate container.

**Tech Stack:** Python 3.12+ (PEP 695 generics), pandas, lmfit, PyMC, pytest, ruff, mypy, uv.

**Spec:** `docs/superpowers/specs/2026-07-17-plate-fit-residuals-design.md`

## Global Constraints

- Run every command through `uv run` (e.g. `uv run pytest`). Never call bare `pip`/`python`.
- Type hints on all public functions and methods; must pass `uv run mypy`.
- Numpy-style docstrings on all public API (`ruff` is configured with `convention = "numpy"`).
- Format and lint with `ruff` only: `uv run ruff format` / `uv run ruff check`. Never `black` or `isort`.
- `line-length = 88`, `target-version = "py312"`.
- Do not reformat code unrelated to the task at hand.
- Do not modify `pyproject.toml`.
- **Do not touch `prtecan_devel.ipynb`.** It has uncommitted user changes and is excluded from both ruff (`extend-exclude`) and the docs build. Its `fit_plate` calls are the user's to migrate.
- `docs/tutorials/prtecan.ipynb` **is** linted by ruff (`extend-include = ["*.ipynb"]`) and **is** executed by the docs build (`nb_execution_allow_errors = False`). It must be migrated before `pipeline.fit_plate` is deleted, or CI fails.
- Never edit `docs/jupyter_execute/**` or `docs/tutorials/.virtual_documents/**` — generated artifacts, excluded via `conf.py`.

______________________________________________________________________

### Task 1: Module-level imports in `data_structures.py`

Removes the lazy imports inside `FitResult.residual_table`. They exist only because they used the **package-attribute form** (`from clophfit.fitting import model_validation`), which needs `fitting/__init__.py` to have finished binding that attribute — and `__init__` eagerly imports `bayes` → … → `data_structures`. The **submodule-direct form** avoids this. The graph is acyclic: `model_validation` → `residuals` (imports nothing from `clophfit`), `models` → `clophfit_types`, and `model_validation` never references `data_structures`.

**Files:**

- Modify: `src/clophfit/fitting/data_structures.py:27-29` (imports), `:568-591` (`residual_table` body)
- Create: `tests/test_imports.py`

**Interfaces:**

- Consumes: nothing.

- Produces: module-level names in `data_structures`: `residuals_from_fit_results`, `robust_likelihood_from_trace`, `robust_settings_from_trace`, `STUDENT_T_NU`, `binding_1site`. Task 2 relies on these being importable at module scope.

- [ ] **Step 1: Write the failing test**

Create `tests/test_imports.py`:

```python
"""Cold-import regression tests.

Each import runs in a fresh interpreter so partially-initialized-package
regressions surface instead of being masked by an already-populated
``sys.modules``.
"""

from __future__ import annotations

import subprocess
import sys


def _cold_import(statement: str) -> None:
    subprocess.run([sys.executable, "-c", statement], check=True)


def test_cold_import_fitting_package() -> None:
    """Importing the package triggers the eager ``__init__`` chain."""
    _cold_import("import clophfit.fitting")


def test_cold_import_data_structures_first() -> None:
    """Importing the submodule first must not hit a partial parent package."""
    _cold_import("import clophfit.fitting.data_structures")


def test_cold_import_prtecan() -> None:
    """prtecan imports fitting at module level; it must stay importable."""
    _cold_import("import clophfit.prtecan")
```

- [ ] **Step 2: Run the test to verify it passes on the current tree**

Run: `uv run pytest tests/test_imports.py -v`
Expected: 3 passed. This is a characterization test — it pins current behaviour so Step 3 cannot silently break imports.

- [ ] **Step 3: Add the module-level imports**

In `src/clophfit/fitting/data_structures.py`, after the existing line 29 (`from .errors import InvalidDataError`), add:

```python
from clophfit.fitting.model_validation import (
    STUDENT_T_NU,
    residuals_from_fit_results,
    residuals_from_multifit,
    robust_likelihood_from_trace,
    robust_settings_from_trace,
)
from clophfit.fitting.models import binding_1site
```

Use the absolute submodule-direct form exactly as written — `from clophfit.fitting import model_validation` will fail here, and that failure is the reason the lazy import existed.

- [ ] **Step 4: Delete the lazy imports and rewrite the call**

In `FitResult.residual_table`, delete these two lines (currently 568-569):

```python
        from clophfit.fitting import model_validation  # noqa: PLC0415
        from clophfit.fitting.models import binding_1site  # noqa: PLC0415
```

Then drop the `model_validation.` prefix from the five call sites in that method body, so it reads:

```python
        bfunc = binding_1site if binding_function is None else binding_function
        residual_likelihood: str | None = None
        if robust is None:
            residual_likelihood = robust_likelihood_from_trace(self.mini)
            robust, detected_nu = robust_settings_from_trace(self.mini)
            if student_t_nu is None:
                student_t_nu = detected_nu
        if student_t_nu is None:
            student_t_nu = STUDENT_T_NU
        return residuals_from_fit_results(
            {well: self},
            trace_id="",
            binding_function=bfunc,
            robust=robust,
            student_t_nu=student_t_nu,
            outlier_threshold=outlier_threshold,
            trace=self.mini,
            residual_likelihood=residual_likelihood,
        )
```

Apply the same treatment to `MultiFitResult.residual_table` (currently 650-674): delete its two lazy import lines and unqualify, keeping its terminal call as `residuals_from_multifit(self, ...)`.

- [ ] **Step 5: Run the tests to verify nothing broke**

Run: `uv run pytest tests/test_imports.py tests/test_residuals.py tests/test_model_validation_smoke.py -q`
Expected: all pass. `test_model_validation_smoke.py::...` exercises `residual_table(well="A01")` at lines 839, 860, 888.

- [ ] **Step 6: Lint and typecheck**

Run: `uv run ruff check src/clophfit/fitting/data_structures.py tests/test_imports.py && uv run mypy src/clophfit/fitting/data_structures.py`
Expected: `All checks passed!` and no mypy errors.

- [ ] **Step 7: Commit**

```bash
git add src/clophfit/fitting/data_structures.py tests/test_imports.py
git commit -m "refactor(imports): hoist data_structures residual imports to module level"
```

______________________________________________________________________

### Task 2: `ResidualsMixin` for `FitResult` and `MultiFitResult`

Extracts the ~12-line auto-detect dance duplicated across `FitResult.residual_table` and `MultiFitResult.residual_table`. Each class keeps its **own** `residual_table` signature — `FitResult` takes `well: str = ""` (used by `test_model_validation_smoke.py:839,860,888`) and the others do not, so a single shared signature would either drop `well` or force untyped `**kwargs`.

**Files:**

- Modify: `src/clophfit/fitting/data_structures.py` (add mixin above `FitResult`; edit `FitResult` and `MultiFitResult`)
- Test: `tests/test_residuals.py`

**Interfaces:**

- Consumes: module-level `robust_likelihood_from_trace`, `robust_settings_from_trace`, `STUDENT_T_NU`, `binding_1site` from Task 1.

- Produces: `ResidualsMixin` with `residuals` (cached_property) and `_resolve_residual_settings(trace, *, binding_function=None, robust=None, student_t_nu=None) -> _ResidualSettings`, where `_ResidualSettings` is a `NamedTuple(binding_function, robust, student_t_nu, residual_likelihood)`. Task 3 subclasses this mixin.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_residuals.py`:

```python
def test_residuals_mixin_resolves_classical_fit_as_normal() -> None:
    """A classical fit has no trace, so it standardizes as Normal."""
    from clophfit.fitting.data_structures import ResidualsMixin

    settings = ResidualsMixin._resolve_residual_settings(None)  # noqa: SLF001

    assert settings.robust is False
    assert settings.student_t_nu == 3.0
    assert settings.residual_likelihood == "normal"


def test_residuals_mixin_honours_explicit_robust() -> None:
    """An explicit robust flag skips auto-detection and its likelihood label."""
    from clophfit.fitting.data_structures import ResidualsMixin

    settings = ResidualsMixin._resolve_residual_settings(  # noqa: SLF001
        None, robust=True, student_t_nu=5.0
    )

    assert settings.robust is True
    assert settings.student_t_nu == 5.0
    assert settings.residual_likelihood is None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_residuals.py -k residuals_mixin -v`
Expected: FAIL with `ImportError: cannot import name 'ResidualsMixin'`.

- [ ] **Step 3: Add the mixin**

In `src/clophfit/fitting/data_structures.py`, immediately above `@dataclass class FitResult[MiniType: MiniProtocol]:` (currently line 488), add:

```python
class _ResidualSettings(NamedTuple):
    """Resolved settings for building a canonical residual table."""

    binding_function: Callable[..., object]
    robust: bool
    student_t_nu: float
    residual_likelihood: str | None


class ResidualsMixin:
    """Shared canonical-residual accessor for fit-result containers.

    Subclasses must implement :meth:`residual_table`. They may add extra
    keyword-only parameters to it (``FitResult`` adds ``well``).
    """

    @cached_property
    def residuals(self) -> pd.DataFrame:
        """Canonical per-observation residual table for this fit.

        Lazily computed, cached, and returned in the schema shared across the
        package (``clophfit.fitting.model_validation.RESIDUAL_TABLE_COLUMNS``:
        ``raw_res``, ``yhat``, ``sigma``, ``std_res``, …). Robustness is
        auto-detected. Use :meth:`residual_table` to override the model or the
        robust settings.
        """
        return self.residual_table()

    def residual_table(self) -> pd.DataFrame:
        """Compute the canonical residual table. Implemented by subclasses."""
        raise NotImplementedError

    @staticmethod
    def _resolve_residual_settings(
        trace: object,
        *,
        binding_function: Callable[..., object] | None = None,
        robust: bool | None = None,
        student_t_nu: float | None = None,
    ) -> _ResidualSettings:
        """Resolve model and standardization settings for a residual table.

        Parameters
        ----------
        trace : object
            PyMC trace to auto-detect robustness from, or ``None`` for a
            classical fit (which resolves to Normal standardization).
        binding_function : Callable[..., object] | None
            Model evaluated for ``yhat``; defaults to ``binding_1site``.
        robust : bool | None
            Force the Student-t standardization of ``std_res``. ``None``
            auto-detects from *trace*.
        student_t_nu : float | None
            Student-t degrees of freedom. ``None`` uses the detected/default.

        Returns
        -------
        _ResidualSettings
            The resolved binding function, robust flag, nu, and likelihood label.
        """
        bfunc = binding_1site if binding_function is None else binding_function
        residual_likelihood: str | None = None
        if robust is None:
            residual_likelihood = robust_likelihood_from_trace(trace)
            robust, detected_nu = robust_settings_from_trace(trace)
            if student_t_nu is None:
                student_t_nu = detected_nu
        if student_t_nu is None:
            student_t_nu = STUDENT_T_NU
        return _ResidualSettings(bfunc, robust, student_t_nu, residual_likelihood)
```

Add `NamedTuple` to the `typing` import on line 18, so it reads:

```python
from typing import TYPE_CHECKING, NamedTuple, Protocol, TypeVar, cast, runtime_checkable
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_residuals.py -k residuals_mixin -v`
Expected: 2 passed.

- [ ] **Step 5: Apply the mixin to both classes**

Change the `FitResult` declaration (currently line 489) to:

```python
class FitResult[MiniType: MiniProtocol](ResidualsMixin):
```

Delete its `residuals` cached_property (currently 525-535) — it now comes from the mixin — and replace the body of `residual_table` (keeping its existing docstring and signature, including `well: str = ""`) with:

```python
        settings = self._resolve_residual_settings(
            self.mini,
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
            trace=self.mini,
            residual_likelihood=settings.residual_likelihood,
        )
```

Change the `MultiFitResult` declaration (currently line 595) to:

```python
class MultiFitResult(ResidualsMixin):
```

Delete its `residuals` cached_property (currently 613-622) and replace its `residual_table` body (keeping signature and docstring) with:

```python
        settings = self._resolve_residual_settings(
            self.trace,
            binding_function=binding_function,
            robust=robust,
            student_t_nu=student_t_nu,
        )
        return residuals_from_multifit(
            self,
            trace_id="",
            binding_function=settings.binding_function,
            robust=settings.robust,
            student_t_nu=settings.student_t_nu,
            outlier_threshold=outlier_threshold,
            residual_likelihood=settings.residual_likelihood,
        )
```

- [ ] **Step 6: Run the full residual test suites**

Run: `uv run pytest tests/test_residuals.py tests/test_model_validation_smoke.py tests/test_imports.py -q`
Expected: all pass, including the pre-existing `residual_table(well="A01")` tests.

- [ ] **Step 7: Lint and typecheck**

Run: `uv run ruff check src/clophfit/fitting/data_structures.py tests/test_residuals.py && uv run mypy src/clophfit/fitting/data_structures.py`
Expected: clean.

- [ ] **Step 8: Commit**

```bash
git add src/clophfit/fitting/data_structures.py tests/test_residuals.py
git commit -m "refactor(residuals): extract ResidualsMixin shared by FitResult and MultiFitResult"
```

______________________________________________________________________

### Task 3: `TitrationResults.residuals`

**Files:**

- Modify: `src/clophfit/prtecan/titration.py:19-25` (imports), `:413-437` (`TitrationResults` declaration)
- Test: `tests/test_prtecan.py`

**Interfaces:**

- Consumes: `ResidualsMixin` from Task 2.

- Produces: `TitrationResults.residuals` (`pd.DataFrame`) and `TitrationResults.residual_table(*, binding_function=None, robust=None, student_t_nu=None, outlier_threshold=3.0)`. Task 4 returns instances carrying these.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_prtecan.py`, inside the class that already provides the `tit` fixture (the one at line 1207, alongside `test_fit`):

```python
    def test_titration_results_residuals(self, tit: Titration) -> None:
        """Plate results expose the canonical residual table."""
        from clophfit.fitting.model_validation import RESIDUAL_TABLE_COLUMNS

        ds = {k: tit.create_ds(k, label="2") for k in tit.fit_keys}
        res = TitrationResults(tit.scheme, tit.fit_keys, fit_plate(ds, method="huber"))

        table = res.residuals

        assert list(table.columns) == RESIDUAL_TABLE_COLUMNS
        assert not table.empty
        # H02 fits on label 2; wells whose fit failed are skipped, not raised on.
        assert "H02" in set(table["well"])
        assert res.residuals is table  # cached
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_prtecan.py -k titration_results_residuals -v`
Expected: FAIL with `AttributeError: 'TitrationResults' object has no attribute 'residuals'`.

- [ ] **Step 3: Import the mixin and builder**

In `src/clophfit/prtecan/titration.py`, extend the existing import block at lines 19-25 to:

```python
from clophfit.fitting.data_structures import (
    DataArray,
    Dataset,
    FitResult,
    NoiseModelParams,
    PlateNoiseModel,
    ResidualsMixin,
)
from clophfit.fitting.model_validation import residuals_from_fit_results
```

- [ ] **Step 4: Add the accessor**

Change the `TitrationResults` declaration (currently line 414) to:

```python
class TitrationResults(ResidualsMixin):
```

and add this method to the class, directly after `__post_init__`:

```python
    def residual_table(
        self,
        *,
        binding_function: Callable[..., object] | None = None,
        robust: bool | None = None,
        student_t_nu: float | None = None,
        outlier_threshold: float = 3.0,
    ) -> pd.DataFrame:
        """Compute the canonical plate-wide residual table.

        Parameters
        ----------
        binding_function : Callable[..., object] | None
            Model evaluated for ``yhat``; defaults to ``binding_1site``.
        robust : bool | None
            Force the Student-t standardization of ``std_res``. ``None``
            auto-detects, which for a classical plate fit means Normal.
        student_t_nu : float | None
            Student-t degrees of freedom (``None`` uses detected/default).
        outlier_threshold : float
            Threshold for the ``is_residual_outlier`` flag.

        Returns
        -------
        pd.DataFrame
            The canonical residual table (see :attr:`residuals`). Wells whose
            fit failed carry no dataset or result and are skipped, so the table
            may cover fewer wells than ``fit_keys``.
        """
        settings = self._resolve_residual_settings(
            None,
            binding_function=binding_function,
            robust=robust,
            student_t_nu=student_t_nu,
        )
        return residuals_from_fit_results(
            self.results,
            trace_id="",
            binding_function=settings.binding_function,
            robust=settings.robust,
            student_t_nu=settings.student_t_nu,
            outlier_threshold=outlier_threshold,
            residual_likelihood=settings.residual_likelihood,
        )
```

`Callable` is already available under `TYPE_CHECKING` (line 32), and the module has `from __future__ import annotations` (line 3), so the annotation resolves.

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest tests/test_prtecan.py -k titration_results_residuals -v`
Expected: 1 passed.

- [ ] **Step 6: Lint and typecheck**

Run: `uv run ruff check src/clophfit/prtecan/titration.py && uv run mypy src/clophfit/prtecan/titration.py`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add src/clophfit/prtecan/titration.py tests/test_prtecan.py
git commit -m "feat(prtecan): add residuals accessor to TitrationResults"
```

______________________________________________________________________

### Task 4: `Titration.fit_plate`

**Files:**

- Modify: `src/clophfit/prtecan/titration.py` (imports; add `_fit_datasets` module-private helper and `Titration.fit_plate`)
- Create: `tests/test_titration_fit_plate.py`

**Interfaces:**

- Consumes: `TitrationResults` (Task 3), `create_dataset_dict` (`titration.py:1067`).

- Produces: `Titration.fit_plate(datasets=None, method="", *, label=None, **kwargs) -> TitrationResults` and module-private `_fit_datasets(datasets, method, **kwargs) -> dict[str, FitResult[Any]]`. Task 5 migrates callers onto `fit_plate`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_titration_fit_plate.py`:

```python
"""Tests for Titration.fit_plate dispatch and dataset construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from clophfit.fitting.data_structures import DataArray, Dataset, FitResult
from clophfit.fitting.errors import InsufficientDataError
from clophfit.prtecan.titration import Titration, TitrationResults

data_tests = Path(__file__).parent / "Tecan"


@dataclass
class _Result:
    residual: np.ndarray
    success: bool = True


@pytest.fixture(scope="module")
def tit() -> Titration:
    """A small pH titration; the fits themselves are monkeypatched away."""
    return Titration.fromlistfile(data_tests / "L1" / "list.pH.csv", is_ph=True)


def _dataset() -> Dataset:
    x = np.array([6.0, 7.0, 8.0])
    y = np.array([1.0, 2.0, 3.0])
    return Dataset({"1": DataArray(x, y, y_errc=np.ones_like(y))}, is_ph=True)


@pytest.mark.parametrize(
    ("method", "target"),
    [("", "lm"), ("huber", "huber"), ("odr", "odr"), ("mcmc", "mcmc")],
)
def test_fit_plate_routes_methods_and_preserves_well_keys(
    monkeypatch: pytest.MonkeyPatch, tit: Titration, method: str, target: str
) -> None:
    """Plate fitting should dispatch each method to the intended backend."""
    calls: list[tuple[str, str]] = []

    def fake_glob(
        ds: Dataset, *, method: str = "lm", **_kwargs: object
    ) -> FitResult[Any]:
        calls.append(("glob", method))
        return FitResult(result=_Result(np.zeros(len(ds["1"].y))), dataset=ds)

    def fake_odr(ds: Dataset, **_kwargs: object) -> FitResult[Any]:
        calls.append(("odr", "odr"))
        return FitResult(result=_Result(np.zeros(len(ds["1"].y))), dataset=ds)

    def fake_mcmc(ds: Dataset, **_kwargs: object) -> FitResult[Any]:
        calls.append(("mcmc", "mcmc"))
        return FitResult(result=_Result(np.zeros(len(ds["1"].y))), dataset=ds)

    monkeypatch.setattr("clophfit.prtecan.titration.fit_binding_glob", fake_glob)
    monkeypatch.setattr("clophfit.prtecan.titration.fit_binding_odr", fake_odr)
    monkeypatch.setattr("clophfit.prtecan.titration.fit_binding_pymc", fake_mcmc)

    results = tit.fit_plate({"A01": _dataset(), "A02": _dataset()}, method=method)

    assert isinstance(results, TitrationResults)
    assert set(results.results) == {"A01", "A02"}
    assert all(fr.result is not None for fr in results.results.values())
    assert calls == [(target if target in {"odr", "mcmc"} else "glob", target)] * 2


@pytest.mark.parametrize("method", ["lm", "odr", "mcmc"])
def test_fit_plate_turns_insufficient_data_into_empty_result(
    monkeypatch: pytest.MonkeyPatch, tit: Titration, method: str
) -> None:
    """Per-well insufficient-data failures should not abort the whole plate."""

    def fail(*_args: object, **_kwargs: object) -> FitResult[Any]:
        msg = "too few points"
        raise InsufficientDataError(msg)

    monkeypatch.setattr("clophfit.prtecan.titration.fit_binding_glob", fail)
    monkeypatch.setattr("clophfit.prtecan.titration.fit_binding_odr", fail)
    monkeypatch.setattr("clophfit.prtecan.titration.fit_binding_pymc", fail)

    results = tit.fit_plate({"A01": _dataset()}, method=method)

    assert results["A01"].result is None


def test_fit_plate_carries_plate_metadata(
    monkeypatch: pytest.MonkeyPatch, tit: Titration
) -> None:
    """The returned container carries this titration's scheme and fit_keys."""

    def fake_glob(ds: Dataset, **_kwargs: object) -> FitResult[Any]:
        return FitResult(result=_Result(np.zeros(len(ds["1"].y))), dataset=ds)

    monkeypatch.setattr("clophfit.prtecan.titration.fit_binding_glob", fake_glob)

    results = tit.fit_plate({"A01": _dataset()})

    assert results.scheme is tit.scheme
    assert results.fit_keys == tit.fit_keys


def test_fit_plate_builds_datasets_when_none_given(
    monkeypatch: pytest.MonkeyPatch, tit: Titration
) -> None:
    """Omitting datasets delegates construction to create_dataset_dict."""
    seen: list[str | None] = []
    original = tit.create_dataset_dict

    def spy(label: str | None = None) -> dict[str, Dataset]:
        seen.append(label)
        return original(label)

    monkeypatch.setattr(tit, "create_dataset_dict", spy)
    monkeypatch.setattr(
        "clophfit.prtecan.titration.fit_binding_glob",
        lambda ds, **_kw: FitResult(
            result=_Result(np.zeros(len(next(iter(ds.values())).y))), dataset=ds
        ),
    )

    tit.fit_plate(label="1")

    assert seen == ["1"]


def test_fit_plate_rejects_datasets_and_label_together(tit: Titration) -> None:
    """datasets and label are alternative spellings of the same input."""
    with pytest.raises(ValueError, match="not both"):
        tit.fit_plate({"A01": _dataset()}, label="1")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_titration_fit_plate.py -v`
Expected: FAIL with `AttributeError: 'Titration' object has no attribute 'fit_plate'`.

- [ ] **Step 3: Import the fitting backends**

In `src/clophfit/prtecan/titration.py`, add to the module-level imports (after the `clophfit.fitting.data_structures` block edited in Task 3):

```python
from clophfit.fitting.bayes import fit_binding_pymc
from clophfit.fitting.core import fit_binding_glob
from clophfit.fitting.errors import InsufficientDataError
from clophfit.fitting.odr import fit_binding_odr
```

These are module-level, not lazy. Measured: `import clophfit.prtecan` already pulls PyMC via `titration.py:19` → `clophfit/fitting/__init__.py:7` → `bayes`, so this costs nothing new.

- [ ] **Step 4: Add the dispatch helper**

Add near the top of `src/clophfit/prtecan/titration.py`, after the `logger` assignment (line 49):

```python
def _fit_datasets(
    datasets: dict[str, Dataset],
    method: str,
    **kwargs: typing.Any,  # noqa: ANN401
) -> dict[str, FitResult[typing.Any]]:
    """Fit each dataset, turning per-well failures into empty results.

    Parameters
    ----------
    datasets : dict[str, Dataset]
        Mapping of well keys to `Dataset` objects.
    method : str
        'lm', 'huber', 'odr', 'mcmc', or any method accepted by
        :func:`clophfit.fitting.core.fit_binding_glob`.
    **kwargs : typing.Any
        Forwarded to the selected fitting function.

    Returns
    -------
    dict[str, FitResult[typing.Any]]
        One entry per input well; failed wells hold an empty `FitResult`.
    """
    fitter: Callable[..., FitResult[typing.Any]]
    if method == "odr":
        fitter, label = fit_binding_odr, "ODR fit"
    elif method == "mcmc":
        fitter, label = fit_binding_pymc, "MCMC fit"
    else:
        method = method or "lm"
        fitter = functools.partial(fit_binding_glob, method=method)
        label = "fit"

    results: dict[str, FitResult[typing.Any]] = {}
    for well, ds in datasets.items():
        try:
            results[well] = fitter(ds, **kwargs)
        except InsufficientDataError:
            logger.warning("Skip %s for well %s.", label, well)
            results[well] = FitResult()
    return results
```

Add `import functools` to the stdlib imports at the top of the module (after `import logging`, line 5).

This collapses the original's three near-identical loops (`pipeline.py:255-278`) into one, and drops the stray `print(method)` (`pipeline.py:272`) — note `ruff` never caught it because `T20` is in the project's `ignore` list.

- [ ] **Step 5: Add the method**

Add to the `Titration` class, immediately after `create_dataset_dict` (which currently ends at line 1092):

```python
    def fit_plate(
        self,
        datasets: dict[str, Dataset] | None = None,
        method: str = "",
        *,
        label: str | None = None,
        **kwargs: typing.Any,  # noqa: ANN401
    ) -> TitrationResults:
        """Run a single-pass fit on an entire plate of datasets.

        Parameters
        ----------
        datasets : dict[str, Dataset] | None
            Mapping of well keys (e.g. 'A01') to `Dataset` objects. When
            ``None``, datasets are built with :meth:`create_dataset_dict`,
            which also applies outlier masking when ``params.mask_outliers``
            is set.
        method : str
            The fitting method: 'lm' (default), 'huber', 'odr', or 'mcmc'.
            Other methods supported by
            :func:`clophfit.fitting.core.fit_binding_glob` may also be used.
        label : str | None
            Build per-label datasets for this label instead of global ones.
            Only valid when *datasets* is ``None``.
        **kwargs : typing.Any
            Additional keyword arguments passed to the fitting function.

        Returns
        -------
        TitrationResults
            Plate results carrying this titration's ``scheme`` and ``fit_keys``.

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
        results = _fit_datasets(datasets, method, **kwargs)
        return TitrationResults(self.scheme, self.fit_keys, results)
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_titration_fit_plate.py -v`
Expected: 10 passed — 4 routing (parametrized) + 3 insufficient-data (parametrized) + metadata + builds-datasets + rejects-both.

- [ ] **Step 7: Lint and typecheck**

Run: `uv run ruff check src/clophfit/prtecan/titration.py tests/test_titration_fit_plate.py && uv run mypy src/clophfit/prtecan/titration.py`
Expected: clean.

- [ ] **Step 8: Commit**

```bash
git add src/clophfit/prtecan/titration.py tests/test_titration_fit_plate.py
git commit -m "feat(prtecan): add Titration.fit_plate returning TitrationResults"
```

______________________________________________________________________

### Task 5: Migrate `export.py` and `test_prtecan.py`

Collapses the three fit-then-wrap pairs. Each call keeps passing its explicitly-built dataset dict, so behaviour is unchanged — those dicts (`export.py:169,176`) deliberately skip the `mask_outliers` step that `create_dataset_dict` applies.

**Files:**

- Modify: `src/clophfit/prtecan/export.py:19` (import), `:178-185`, `:194-201`, `:203-210`
- Modify: `tests/test_prtecan.py:1286-1310`

**Interfaces:**

- Consumes: `Titration.fit_plate` (Task 4).

- Produces: nothing new.

- [ ] **Step 1: Migrate the three call sites**

In `src/clophfit/prtecan/export.py`, replace the per-label block (currently 178-185):

```python
            export_list.append(
                titration.fit_plate(
                    ds_single,
                    method=titration.params.fit_method,
                    remove_outliers=titration.params.outlier,
                )
            )
```

Replace the global block (currently 194-201):

```python
    global_res = titration.fit_plate(
        datasets,
        method=method,
        reweight=reweight,
        remove_outliers=titration.params.outlier,
    )
    export_list.append(global_res)
```

Replace the ODR block (currently 203-210):

```python
    odr_res = titration.fit_plate(
        datasets,
        method="odr",
        remove_outliers=titration.params.outlier,
        reweight=reweight,
    )
    export_list.append(odr_res)
```

Delete the now-unused import on line 19:

```python
from clophfit.fitting.pipeline import fit_plate
```

Leave `TitrationResults` imported (line 25) — `export_bad_wells` still annotates with it.

- [ ] **Step 2: Migrate the prtecan fit test**

In `tests/test_prtecan.py`, rewrite `test_fit` (currently 1286-1310) to use the method. `TitrationResults.__getitem__` means the `res[...]` accesses are unchanged:

```python
    def test_fit(self, tit: Titration) -> None:
        """It fits each label separately."""
        # Test Label 1 for H02 (should be skipped because of OVER value or insufficient points)
        ds1 = {k: tit.create_ds(k, label="1") for k in tit.fit_keys}
        res1 = tit.fit_plate(ds1, method="huber")
        assert not res1["H02"].is_valid()

        # Test Label 2 for H02
        ds2 = {k: tit.create_ds(k, label="2") for k in tit.fit_keys}
        res2 = tit.fit_plate(ds2, method="huber")
        assert res2["H02"].is_valid()

        # Check 'K' and std error for 'H02' in the second fit result
        assert res2["H02"].result is not None
        k_h02 = res2["H02"].result.params["K"]
        assert k_h02.value == pytest.approx(7.899, abs=1e-3)
        assert k_h02.stderr == pytest.approx(0.026, abs=1e-3)

        # Check 'K' and std error for 'H02' in global fit
        ds_global = {k: tit.create_global_ds(k) for k in tit.fit_keys}
        res_global = tit.fit_plate(ds_global, method="huber")
        assert res_global["H02"].result is not None
        k_h02_glob = res_global["H02"].result.params["K"]
        assert k_h02_glob.value == pytest.approx(7.899, abs=1e-3)
        assert k_h02_glob.stderr == pytest.approx(0.026, abs=1e-3)
```

Then update the Task 3 test you added (`test_titration_results_residuals`) to build via the method:

```python
        res = tit.fit_plate({k: tit.create_ds(k, label="2") for k in tit.fit_keys}, method="huber")
```

Delete the now-unused `from clophfit.fitting.pipeline import fit_plate` import at `tests/test_prtecan.py:19`.

- [ ] **Step 2b: Run to verify**

Run: `uv run pytest tests/test_prtecan.py -k "fit or residuals" -q`
Expected: all pass, with the same `K` values (7.899 ± 0.026) as before — proving the move preserved fitting behaviour.

- [ ] **Step 3: Run the CLI/export suites**

Run: `uv run pytest tests/test_cli.py tests/test_prtecan.py -q`
Expected: all pass. `export_fit` is the main consumer of the migrated code.

- [ ] **Step 4: Lint and typecheck**

Run: `uv run ruff check src/clophfit/prtecan/export.py tests/test_prtecan.py && uv run mypy src/clophfit/prtecan/export.py`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add src/clophfit/prtecan/export.py tests/test_prtecan.py
git commit -m "refactor(prtecan): use Titration.fit_plate in export and tests"
```

______________________________________________________________________

### Task 6: Migrate the tutorial notebook

Must land **before** Task 7. The docs build executes this notebook with `nb_execution_allow_errors = False`, so a dangling `pipeline.fit_plate` import would fail CI's `docs_build` job.

**Files:**

- Modify: `docs/tutorials/prtecan.ipynb` — the fit cell, the MCMC cell, the residuals cell, and the `cl_an` cell

Locate cells by **content**, not line number: `docs/tutorials/.virtual_documents/prtecan.ipynb` is a stale mirror (it still shows the `collect_multi_residuals` call removed in `e906ec9`), so its line numbers do not match the real notebook.

Verified facts about this notebook: `tit` and `cl_an` are both `Titration` instances; `ds_dict1`, `ds_dict2`, `ds_dict_glob`, and `res1`…`res_odr` are used **only** to build `tr1`…`tr_odr`; but `tr1`, `tr2`, `tr_glob`, and `tr_odr` are used throughout later cells (well comparison, `.dataframe`, `.figure`, the residuals cell) and **must keep their names**.

**Interfaces:**

- Consumes: `Titration.fit_plate` (Task 4).

- Produces: nothing.

- [ ] **Step 1: Collapse the main fit cell**

Replace the whole cell that currently reads:

```python
from clophfit.fitting.pipeline import fit_plate
from clophfit.prtecan import TitrationResults

# Create dataset dicts
ds_dict1 = tit.create_dataset_dict("1")
ds_dict2 = tit.create_dataset_dict("2")
ds_dict_glob = tit.create_dataset_dict()

# Compute fits
res1 = fit_plate(ds_dict1, method="lm")
res2 = fit_plate(ds_dict2, method="lm")
res_glob = fit_plate(ds_dict_glob, method="lm")
res_odr = fit_plate(ds_dict_glob, method="odr")

# Create TitrationResults wrappers for dataframe access
tr1 = TitrationResults(tit.scheme, tit.fit_keys, res1)
tr2 = TitrationResults(tit.scheme, tit.fit_keys, res2)
tr_glob = TitrationResults(tit.scheme, tit.fit_keys, res_glob)
tr_odr = TitrationResults(tit.scheme, tit.fit_keys, res_odr)
```

with:

```python
# Fit each label and the global dataset; plate metadata comes from the titration
tr1 = tit.fit_plate(label="1", method="lm")
tr2 = tit.fit_plate(label="2", method="lm")
tr_glob = tit.fit_plate(method="lm")
tr_odr = tit.fit_plate(method="odr")
```

This is behaviour-preserving: `fit_plate(label="1")` builds its datasets with the same `tit.create_dataset_dict("1")` the cell called explicitly.

- [ ] **Step 2: Update the MCMC cell**

`ds_mcmc` is a hand-built single-well dict, so it stays and is passed explicitly. Replace:

```python
res_mcmc = fit_plate(ds_mcmc, method="mcmc")
```

with:

```python
res_mcmc = tit.fit_plate(ds_mcmc, method="mcmc")
```

- [ ] **Step 3: Update the `cl_an` cell**

Replace:

```python
ds_glob_cl = cl_an.create_dataset_dict()
tr_glob_cl = prtecan.TitrationResults(
    cl_an.scheme, cl_an.fit_keys, fit_plate(ds_glob_cl, method="lm")
)
```

with:

```python
tr_glob_cl = cl_an.fit_plate(method="lm")
```

- [ ] **Step 4: Simplify the residuals cell**

Replace:

```python
all_res = residuals_from_fit_results(
    {w: tr_glob[w] for w in tit.fit_keys},
    trace_id="",
    binding_function=binding_1site,
)
diag = ResidualDiagnostics(all_res)
```

with:

```python
diag = ResidualDiagnostics(tr_glob.residuals)
```

This cell is the whole point of the feature — it should demonstrate the accessor. Adjust that cell's imports so only `ResidualDiagnostics` remains, dropping `residuals_from_fit_results` and `binding_1site` if no other cell uses them.

- [ ] **Step 5: Prune dead imports**

`from clophfit.fitting.pipeline import fit_plate` is gone. `TitrationResults` / `prtecan.TitrationResults` may now be unused — Step 6's ruff run reports it as `F401`; delete only what ruff flags.

- [ ] **Step 6: Verify the notebook executes**

Run: `uv run jupyter nbconvert --to notebook --execute --stdout docs/tutorials/prtecan.ipynb > /dev/null`
Expected: completes without error. This is what CI's `docs_build` does; `nb_execution_allow_errors = False` means any traceback fails the job.

- [ ] **Step 7: Lint the notebook**

Run: `uv run ruff check docs/tutorials/prtecan.ipynb`
Expected: `All checks passed!` (ruff lints notebooks here via `extend-include`).

- [ ] **Step 8: Commit**

```bash
git add docs/tutorials/prtecan.ipynb
git commit -m "docs(tutorial): use Titration.fit_plate and TitrationResults.residuals"
```

______________________________________________________________________

### Task 7: Delete `pipeline.fit_plate` and prune

**Files:**

- Modify: `src/clophfit/fitting/pipeline.py:1-26` (imports), delete `:230-280` (`fit_plate`)
- Modify: `tests/test_pipeline.py` — delete the two `fit_plate` tests (`:57-116`) and the now-unused helpers

**Interfaces:**

- Consumes: nothing.

- Produces: nothing. `pipeline` now exports only `calibrate_noise_robust` and `fgls_plate_fit`.

- [ ] **Step 1: Confirm no consumers remain**

Run: `grep -rn "pipeline import fit_plate\|pipeline.fit_plate" src/ tests/ docs/tutorials/ --include='*.py' --include='*.ipynb'`
Expected: no output. If anything matches (other than `prtecan_devel.ipynb`, which is out of scope), migrate it before continuing.

- [ ] **Step 2: Delete the function**

Delete `src/clophfit/fitting/pipeline.py:230-280` in full — the whole `fit_plate` definition, including the stray `print(method)` at line 272.

- [ ] **Step 3: Prune newly-unused imports**

Run: `uv run ruff check src/clophfit/fitting/pipeline.py`

Ruff's `F401` reports which of `fit_binding_pymc` (line 8), `fit_binding_glob` (9), `fit_binding_odr` (19), and `InsufficientDataError` (16) are now unused. Delete exactly those it flags — `fgls_plate_fit` still uses some of them, so do not guess. Then re-run until clean.

- [ ] **Step 4: Delete the migrated tests**

In `tests/test_pipeline.py`, delete `test_fit_plate_routes_methods_and_preserves_well_keys` (with its `@pytest.mark.parametrize`, lines 57-96) and `test_fit_plate_turns_insufficient_data_into_empty_result` (99-116). They now live in `tests/test_titration_fit_plate.py`.

Keep `_dataset`, `_Result`, and the `InsufficientDataError` import only if the remaining FGLS tests still reference them — run ruff to confirm and delete whatever it flags as unused.

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest tests/ -q -m "not slow"`
Expected: all pass. This is the first point where every consumer must be migrated, so a failure here means a missed call site.

- [ ] **Step 6: Lint and typecheck the whole package**

Run: `uv run ruff check src/ tests/ && uv run mypy src/clophfit`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add src/clophfit/fitting/pipeline.py tests/test_pipeline.py
git commit -m "refactor(pipeline): drop fit_plate, leaving only FGLS orchestration"
```

______________________________________________________________________

### Task 8: Hoist the `titration.py` lazy import

Independent cleanup. `clophfit.fitting.utils` is already imported at module level on line 28, so this laziness buys nothing.

**Files:**

- Modify: `src/clophfit/prtecan/titration.py:28` (import), `:692-696` (delete lazy import)

**Interfaces:**

- Consumes: nothing.

- Produces: nothing.

- [ ] **Step 1: Extend the module-level import**

In `src/clophfit/prtecan/titration.py`, change line 28 to:

```python
from clophfit.fitting.utils import (
    apply_outlier_mask,
    flag_trend_outliers,
    roughness,
    smoothness,
)
```

- [ ] **Step 2: Delete the lazy import**

Delete these lines from the method body (currently 692-696):

```python
        from clophfit.fitting.utils import (  # noqa: PLC0415
            flag_trend_outliers,
            roughness,
            smoothness,
        )
```

- [ ] **Step 3: Run the tests**

Run: `uv run pytest tests/test_prtecan.py tests/test_imports.py -q`
Expected: all pass.

- [ ] **Step 4: Lint and typecheck**

Run: `uv run ruff check src/clophfit/prtecan/titration.py && uv run mypy src/clophfit/prtecan/titration.py`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add src/clophfit/prtecan/titration.py
git commit -m "refactor(imports): hoist titration outlier-helper import to module level"
```

______________________________________________________________________

## Final verification

- [ ] Run the full suite including slow tests: `uv run pytest tests/ -q`
- [ ] Build the docs the way CI does: `NB_EXECUTION_MODE=cache uv run make docs`
- [ ] Confirm the feature works end-to-end in a REPL:

```python
from clophfit.prtecan.titration import Titration

tit = Titration.fromlistfile("tests/Tecan/140220/list.pH.csv", is_ph=True)
res_glob = tit.fit_plate(method="huber")
print(res_glob.residuals.head())   # the accessor this plan exists to add
print(res_glob.plot_k())           # plate metadata still available
```

## Known follow-ups (out of scope)

- `prtecan_devel.ipynb` has six `fit_plate` calls and uncommitted user changes; the user migrates it.
- `titration.py` is ~1200 lines after this work and remains a split candidate.
- FGLS (`fgls_plate_fit`, `calibrate_noise_robust`) has no production callers — only tests and the devel notebook.
