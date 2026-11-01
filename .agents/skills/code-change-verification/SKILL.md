______________________________________________________________________

## name: code-change-verification description: "Gate every refactoring unit in ClopHfit: run the quality gates, compare numerical benchmarks (bias, CI coverage, residuals) before/after, and verify the public API is unchanged. Run after each refactoring unit and before stopping on any failure."

# Code Change Verification (ClopHfit)

Run these checks after **every refactoring unit** (each completed, self-contained
change to a fitting method, data structure, parser, or public module). Do not
start the next unit until all applicable gates are green.

## Gate 0 — Scope the change

Before running anything, state which gates apply:

| Touched area                                                                            | Required gates                          |
| --------------------------------------------------------------------------------------- | --------------------------------------- |
| Any `src/clophfit/` code                                                                | Gate 1                                  |
| Any `src/clophfit/fitting/*` logic (models, backends, data structures)                  | Gate 1 (incl. `-m slow`) + Gate 2       |
| Any fitting method behavior (signatures, priors, sigma handling, reweighting, defaults) | Gate 1 + Gate 2 (benchmark diff ritual) |
| Docstrings with `>>>` examples                                                          | Gate 1 + `make xdoc`                    |
| `src/clophfit/__init__.py`, `__main__.py`, or any `__all__`                             | Gate 1 + Gate 3                         |

## Gate 1 — Static + test gates (CI-enforced; run all as the baseline)

```bash
make lint    # ruff check + ruff format --check + pre-commit tooling
make type    # mypy --strict on src/ tests/ docs/conf.py
make test    # pytest (fast tests) + coverage
```

If you edited docstrings with doctests:

```bash
make xdoc
```

If a fitting method's edge-case behavior may have changed (outliers, robust
loss, hierarchical priors), include the slow suite too:

```bash
uv run pytest -v -m slow
```

**Any red gate here stops the refactor loop.** Fix or report; move on to the
next refactoring unit only after green.

## Gate 2 — Numerical regression (fitting-method changes only)

This is the check nothing else in the repo enforces. Linters and unit tests
cannot tell you that a refactor silently changed CI coverage or bias.

### Ritual: capture → refactor → compare

1. **Before** touching the code, produce a baseline and save it:

   ```bash
   # Use the SAME --n-repeats / --noise-levels / --labels / --outliers / --methods
   # both times (see `uv run python benchmarks/run_benchmark.py --help`).
   # Before:
   uv run python benchmarks/run_benchmark.py --output-dir benchmarks/baseline
   ```

   Or, for a focused pairwise check:

   ```bash
   uv run python benchmarks/focused_comparison.py
   uv run python benchmarks/comprehensive_fitter_comparison.py
   ```

1. **After** the change, re-run with the **identical** arguments into a
   different output directory and diff the CSVs:

   ```bash
   uv run python benchmarks/run_benchmark.py --output-dir benchmarks/after
   # then diff the corresponding CSVs under benchmarks/baseline vs benchmarks/after
   ```

1. **Pass criteria** (per refactored method):

   - 95% CI coverage within ~±1–2 percentage points of baseline
     (see `calculate_coverage` in `src/clophfit/testing/evaluation.py`)
   - Bias and RMSE within noise of baseline
     (`calculate_bias`, `calculate_rmse`)
   - Residual diagnostics — Shapiro-Wilk / Anderson-Darling p-values and
     normality shape — statistically indistinguishable
     (`evaluate_residuals`)
   - No new divergences, `NaN`s, or convergence failures in `FitResult`

   A pure mechanical refactor (rename, extract helper, move module) must
   reproduce the baseline **bit-for-bit in practice** (identical floats given
   the same inputs and seed). Any non-trivial drift is a **failure to report**,
   not a result you quietly accept.

1. Record both CSV paths and the diff verdict (or "identical") in your
   report.

## Gate 3 — Public API surface

The package's import surface is the contract. Verify it is unchanged unless
the refactoring unit deliberately changes it:

```bash
# top-level API
python -c "import clophfit; print(clophfit.__all__)"

# fitting-method signatures (lmfit/odr/pymc backends)
python - <<'PY'
import inspect
from clophfit.fitting import core as c, odr as o, bayes as b
for mod in (c, o, b):
    for name, fn in inspect.getmembers(mod, inspect.isfunction):
        if name.startswith("fit_") or name.startswith("outlier"):
            print(mod.__name__, name, inspect.signature(fn))
PY
```

Compare against the pre-change listing (run it **before** the unit as part of
the capture step). Any changed signature, renamed symbol, or removed export
must be called out explicitly in the report. `FitResult[MiniT]` return types
must be preserved on all `fit_*` entry points.

If the change *intentionally* alters the API (e.g. promoting a method to a
keeper, demoting one to `src/clophfit/old/`), list exactly:
what changed, what replaces it, and which call sites (CLI in `__main__.py`,
benchmarks, tests) were updated.

## Reporting

After each unit, report in this order:

1. **Scope** — what changed and which gates applied (Gate 0 table).
1. **Static / test** — pass/fail per `make lint`, `make type`, `make test`,
   `make xdoc` (include the failing output verbatim on failure).
1. **Numerical** — baseline CSV path, post-change CSV path, diff verdict, and
   the key metric deltas (coverage, bias, RMSE, residual p-values).
1. **API** — unchanged, or the explicit change list.
1. **Verdict** — `PASS` or `FAIL`. On `FAIL`, stop; do not proceed to the next
   refactoring unit. Surface the failure and the suspected cause.
