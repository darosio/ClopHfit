# Agents.md — AI Coding Agent Instructions for ClopHfit

## Project Overview

ClopHfit is a scientific Python package for fitting pH titration and
chloride-binding assay data from Tecan and EnSpire plate readers. The primary
model is a single-site binding isotherm (Henderson-Hasselbalch for pH). The
package provides CLI tools, data parsers, SVD analysis, and **multiple fitting
backends** (LMFit, scipy.odr, PyMC).

## Build, Lint, Test, and Type-Check

```bash
# Install all dependencies (locked)
uv sync --locked --group dev --group lint --group tests

# Run all pre-commit linters (ruff, mypy, typos, mdformat, …)
make lint

# Run tests with coverage
make test          # pytest + coverage run
make cov           # coverage combine + report + xml

# Type-check
make type          # mypy --strict on src/ tests/ docs/conf.py

# Docstring tests
make xdoc          # xdoctest on clophfit

# Build docs
make docs          # sphinx build into docs/_build
```

Always run `make lint` and `make test` before proposing changes. Run `make type` when touching type annotations. Run `make xdoc` when editing docstrings
with `>>>` examples.

## Coding Conventions

| Convention                  | Detail                                                                                                   |
| --------------------------- | -------------------------------------------------------------------------------------------------------- |
| **Python**                  | ≥ 3.12 required; tested on 3.12, 3.13, 3.14                                                              |
| **Package manager**         | `uv` with lockfile (`uv.lock`)                                                                           |
| **Build backend**           | `hatchling`                                                                                              |
| **Linter/formatter**        | `ruff` (very broad rule selection, see `pyproject.toml [tool.ruff.lint]`)                                |
| **Type checker**            | `mypy --strict` (enforced in CI on all platforms)                                                        |
| **Docstrings**              | NumPy convention (`tool.ruff.lint.pydocstyle.convention = "numpy"`)                                      |
| **Pre-commit hooks**        | ruff, mypy, typos, mdformat, nbstripout, shellcheck, shfmt, uv-lock, conventional-pre-commit             |
| **Commit messages**         | [Conventional Commits](https://www.conventionalcommits.org/) enforced by pre-commit                      |
| **CI**                      | GitHub Actions: lint + tests × {3.12, 3.13, 3.14} × {Linux, macOS, Windows}                              |
| **Ruff ignores**            | `E501`, `T20`, `DOC201`, `DOC501`, `PLR0914`, `PLR1702` globally; per-file ignores for CLI and notebooks |
| **Excluded from lint/type** | `src/clophfit/old/`, `scripts/`, `benchmarks/`                                                           |

### Style Rules

- Keep comments minimal; prefer self-documenting code and docstrings.
- Use `typing` for all signatures; `mypy --strict` must pass.
- Parameter names `K`, `S0`, `S1` are uppercase by lmfit convention — suppress
  `N803` where needed.
- Scientific code may use `# noqa: PLR0913` for functions with many physical
  parameters.

## Repository Layout

```
src/clophfit/
├── __init__.py          # Package init, configure_logging()
├── __main__.py          # Click CLI: clop, ppr, fit-titration, note_to_csv
├── _config.py           # Root-logger configuration (library-hostile, known debt)
├── clophfit_types.py    # Type aliases (ArrayF, etc.)
├── fitting/
│   ├── models.py        # binding_1site(), kd() — core math models
│   ├── data_structures.py  # DataArray, Dataset, FitResult[MiniT]
│   ├── errors.py        # Custom exceptions
│   ├── core.py          # LMFit backend: fit_binding_glob(), outlier2(), etc.
│   ├── odr.py           # scipy.odr backend: fit_binding_odr(), etc.
│   ├── bayes.py         # PyMC backend: fit_binding_pymc*(), hierarchical, etc.
│   └── plotting.py      # Result visualisation
├── prtecan/
│   └── prtecan.py       # Tecan parser + domain + orchestration (god module, 1840 LOC)
├── prenspire/
│   └── prenspire.py     # EnSpire parser
├── testing/
│   ├── synthetic.py     # make_dataset(), make_simple_dataset() for benchmarks
│   ├── evaluation.py    # calculate_bias(), coverage, residual metrics
│   └── fitter_test_utils.py  # Shared test helpers
└── old/                 # Legacy code (excluded from quality gates)

benchmarks/              # Fitting-method comparison scripts (excluded from lint)
tests/                   # pytest suite (278+ tests), @pytest.mark.slow for heavy ones
```

## The Fitting-Methods Landscape (Critical Context)

**This is the most active area of the codebase.** Multiple fitting approaches
have been implemented during research exploration and are now candidates for
consolidation. A thorough comparison between methods is in progress.

### Current Methods Inventory

#### LMFit Backend (`fitting/core.py`)

| Function                               | Description                    | Status                            |
| -------------------------------------- | ------------------------------ | --------------------------------- |
| `fit_binding_glob()`                   | Standard global least-squares  | **Keeper** — production workhorse |
| `fit_binding_glob(robust=True)`        | Huber-loss variant             | Under evaluation                  |
| `fit_binding_glob_reweighted()`        | IRLS + outlier removal         | Under evaluation                  |
| `fit_binding_glob_recursive()`         | Iterative reweighting          | Under evaluation                  |
| `fit_binding_glob_recursive_outlier()` | Recursive with outlier removal | Under evaluation                  |
| `outlier2()`                           | Two-stage outlier detection    | Under evaluation                  |
| `outlier_glob()`                       | Global outlier detection       | Under evaluation                  |

#### ODR Backend (`fitting/odr.py`)

| Function                              | Description                                | Status           |
| ------------------------------------- | ------------------------------------------ | ---------------- |
| `fit_binding_odr()`                   | Single-pass orthogonal distance regression | Under evaluation |
| `fit_binding_odr_recursive()`         | Iterative reweighting ODR                  | Under evaluation |
| `fit_binding_odr_recursive_outlier()` | Recursive ODR + outlier removal            | Under evaluation |

⚠ **Known bug:** `generalized_combined_model()` at line ~109 hardcodes
`is_ph=True`, breaking non-pH ODR fits.

#### PyMC/Bayesian Backend (`fitting/bayes.py`)

| Function                     | Description                   | Status           |
| ---------------------------- | ----------------------------- | ---------------- |
| `fit_binding_pymc()`         | Single-level Bayesian         | Under evaluation |
| `fit_binding_pymc2()`        | Variant with different priors | Under evaluation |
| `fit_binding_pymc_compare()` | Model comparison              | Under evaluation |
| `fit_binding_pymc_odr()`     | Bayesian ODR hybrid           | Under evaluation |
| `fit_binding_pymc_multi()`   | Hierarchical Bayesian         | Under evaluation |
| `fit_binding_pymc_multi2()`  | Hierarchical variant          | Under evaluation |
| `fit_pymc_hierarchical()`    | Full hierarchical             | Under evaluation |

`bayes.py` has ~15 TODO/FIXME comments marking unfinished design decisions
(lines 114, 176, 187, 337, 398, 406, 606, 623, 654, 690, 795–796).

### Comparison Infrastructure

The following tools exist for systematically comparing fitting methods:

- **`benchmarks/run_benchmark.py`** — Flexible comparison script (synthetic +
  real data; supports bias, coverage, residual normality metrics).
- **`benchmarks/comprehensive_fitter_comparison.py`** — All methods including
  ODR; generates CSV results.
- **`benchmarks/compare_fitting_methods.py`** — Focused pairwise comparison.
- **`benchmarks/compare_error_models.py`** — Error model evaluation.
- **`src/clophfit/testing/synthetic.py`** — Synthetic dataset generation with
  realistic noise profiles (uses real L2/L4 dataset parameters).
- **`src/clophfit/testing/evaluation.py`** — Metrics: `calculate_bias()`,
  coverage analysis, residual distribution tests.

### How to Approach Fitting-Method Cleanup

1. **Do not delete methods prematurely.** Each function was implemented to test
   a specific hypothesis. The comparison benchmarks must run before deciding
   which to keep.
1. **Run benchmarks** (`benchmarks/`) to produce quantitative comparisons.
   These scripts are excluded from lint, so they may not meet strict style —
   that is intentional.
1. **Key evaluation criteria** (in priority order):
   - 95% CI coverage of the true parameter (most important)
   - Residual normality (Shapiro-Wilk / Anderson-Darling)
   - Bias and RMSE vs. noise level
   - Robustness to outliers
1. **After comparison is complete**, the goal is to:
   - Keep 1–2 primary methods per backend (LMFit, ODR, Bayesian).
   - Move deprecated methods to `src/clophfit/old/` or remove entirely.
   - Unify function signatures and return types around `FitResult[MiniT]`.
   - Extract shared logic (reweighting, outlier detection) into composable
     helpers rather than copy-pasted variants.
1. **Watch for subtle differences** between similarly named functions:
   different priors, different sigma handling, different convergence criteria.
   These differences are the whole point of the comparison — document them
   before refactoring.

## Known Bugs and Architectural Debt

### Bugs to Fix

1. **`odr.py:~109`** — `generalized_combined_model()` hardcodes `is_ph=True`.
1. **`__main__.py:432-435`** — `fit_enspire()` inverts `is_ph` assignment
   (pH → False, Cl → True).

### Architectural Debt

- **`prtecan.py` (1840 LOC)** is a god module mixing parsing, domain logic,
  fitting orchestration, and CSV/Excel export. When refactoring, decompose
  into: parser, domain model (`Titration`, `TitrationResults`), and
  orchestration/export layers.
- **`_config.py`** configures the root logger — hostile to library consumers.
  Should configure a `clophfit`-namespaced logger instead.
- **`core.py:89-91`** has mutable module-level globals (`N_BOOT=20`,
  `EMCEE_STEPS=1800`). These should become function parameters or
  configuration objects.
- **`prenspire`** imports `lookup_listoflines` from `prtecan` — cross-domain
  coupling that should be extracted to a shared utility.
- **Hardcoded plot colors** in `data_structures.py:378`.

## Guidance for Common Tasks

### Adding a New Fitting Method

1. Implement in the appropriate backend module (`core.py`, `odr.py`, or
   `bayes.py`).
1. Return `FitResult[MiniT]` (from `data_structures.py`) for consistency.
1. Accept `Dataset` as input (the standard multi-label container).
1. Add a benchmark entry in `benchmarks/run_benchmark.py`.
1. Add unit tests in `tests/` with both synthetic and (if feasible) real data.
1. Do not forget `is_ph` handling — it switches between Henderson-Hasselbalch
   and standard binding isotherm in `models.binding_1site()`.

### Refactoring a Fitting Method

1. Run the benchmarks **before** refactoring to capture baseline metrics.
1. Make changes.
1. Run benchmarks **after** and compare CSV outputs to verify no regression.
1. Run `make test && make type && make lint` — all must pass.
1. Check `@pytest.mark.slow` tests too: `uv run pytest -m slow`.

### Working with Data Structures

- **`DataArray`** — a single titration curve (x, y, x_errors, y_errors, weights).
- **`Dataset`** — `dict[str, DataArray]`, one entry per label/wavelength.
- **`FitResult[MiniT]`** — wraps fit output + parameters + uncertainties.
  Generic over `MiniType` (lmfit `Minimizer`, ODR output, or PyMC trace).

### CLI Entry Points

| Command         | Module                   | Description               |
| --------------- | ------------------------ | ------------------------- |
| `clop`          | `__main__:clop`          | Main entry group          |
| `ppr`           | `__main__:ppr`           | Parse plate-reader data   |
| `fit-titration` | `__main__:fit_titration` | Fit titration from CLI    |
| `note_to_csv`   | `__main__:note2csv`      | Convert note files to CSV |

### Running Slow/Integration Tests

```bash
# Run only fast tests (default)
uv run pytest -v

# Include slow integration tests
uv run pytest -v -m slow

# Run all tests
uv run pytest -v -m ""
```
