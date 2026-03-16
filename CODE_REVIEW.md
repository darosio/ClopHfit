# ClopHfit Architecture + Quality Review (2026-03-16)

## Executive summary

ClopHfit has a solid scientific-computing foundation: a clear package layout, reusable fitting data structures, multiple fitting backends, and meaningful quality gates. The architecture is fundamentally good, but maintainability risk is rising because a few orchestration-heavy modules now carry too many responsibilities.

Current local validation is strong:

- `make lint`: passed
- `make test`: passed (`278 passed, 1 warning`)
- `make type`: passed
- `make xdoc`: passed
- `make cov`: failed because stale coverage data references missing `src/clophfit/fitting/residuals.py`

The highest-priority issues are a pair of likely correctness bugs and a broken coverage-reporting workflow.

## Strengths

### Clear functional layering

The codebase has a sensible top-down structure:

- CLI entry points in `src/clophfit/__main__.py`
- domain/parser modules in `src/clophfit/prtecan/` and `src/clophfit/prenspire/`
- shared fitting abstractions in `src/clophfit/fitting/data_structures.py`
- fitting backends in `src/clophfit/fitting/core.py`, `src/clophfit/fitting/odr.py`, and `src/clophfit/fitting/bayes.py`
- plotting and error types in `src/clophfit/fitting/plotting.py` and `src/clophfit/fitting/errors.py`

That layering is a real architectural asset: the mathematical models and data containers are reusable outside the CLI, and the fitting backends are clearly discoverable.

### Strong shared data model

`DataArray`, `Dataset`, and `FitResult` in `src/clophfit/fitting/data_structures.py` provide a consistent interface across LMFit, ODR, and Bayesian workflows. This is one of the healthiest parts of the design because it reduces duplicated glue code and gives tests a stable surface to exercise.

### Real quality gates already exist

The repository is not relying on ad hoc quality checks:

- `Makefile:41-58` wires linting, tests, typing, coverage, and doctests
- `.github/workflows/ci.yml:26-88` runs linting plus a test/type/xdoc matrix across Python `3.12`-`3.14` and Linux/macOS/Windows

That is a strong signal that the project is already run like production software, not just an exploratory notebook dump.

### Tests map well to the code structure

The test layout mirrors the implementation:

- `tests/test_prtecan.py`
- `tests/test_fitting.py`
- `tests/test_bayes.py`
- `tests/test_odr.py`
- `tests/test_cli.py`

This is good architecture hygiene. It makes it easy to see which subsystem is protected and where additional tests belong.

## Architecture and maintainability risks

### `prtecan.py` is a god module

`src/clophfit/prtecan/prtecan.py` is `1840` lines and mixes:

- file parsing
- metadata extraction
- domain modeling
- titration orchestration
- fitting selection
- plotting/export concerns

This is the main maintainability hotspot in the repository.

The `Titration` class (line 1222+) is the worst offender: it handles file parsing, data preparation, fitting orchestration, and export/plotting. Suggested decomposition:

- `DataPreparer` — background subtraction, dilution correction, normalization
- `FitOrchestrator` — manages LM/ODR/MCMC fitting dispatch
- `ResultExporter` — handles data and plot export

A split into parser/domain/processing/export-focused modules would reduce change risk substantially.

### `TitrationResults` has a complex dunder interface

`src/clophfit/prtecan/prtecan.py:1053` — `TitrationResults` uses `__getitem__`, `__repr__`, and several cached properties to present a dict-like interface. This makes the object harder to understand than a plain container. Consider simplifying or documenting the protocol.

### `bayes.py` is too large and variant-heavy

`src/clophfit/fitting/bayes.py` is `1091` lines. It contains several closely related fitting flows and a visible trail of `TODO`/`FIXME` comments. The Bayesian backend is valuable, but it is becoming hard to reason about because multiple variants and helper paths live in one file.

### CLI orchestration is denser than it should be

`src/clophfit/__main__.py` is `633` lines. The CLI is functional, but it owns a lot of orchestration logic directly instead of delegating to service/helper functions. That is manageable today, but it increases the chance of option-handling bugs like the `fit_enspire()` inversion above.

The `fit_enspire()` function and the `tecan()` command share similar nested-iteration patterns that could be extracted into a shared helper.

### Mutable module-level constants

- `src/clophfit/fitting/core.py:89-91` defines `N_BOOT = 20` and `EMCEE_STEPS = 1800` as module globals

These are effectively configuration but are scattered across source files. Consider centralizing them in `_config.py` or a `Config` dataclass alongside `TitrationConfig`, `TecanConfig`, and `PlotParameters`.

### Cross-domain coupling exists where it should not

- `src/clophfit/prenspire/prenspire.py:18` imports `lookup_listoflines` from `clophfit.prtecan`
- `src/clophfit/fitting/bayes.py:27` imports from `prtecan` under `TYPE_CHECKING`

The first helper is generic and should not force a dependency from EnSpire parsing back into the Tecan domain. A small shared `utils` module would make the dependency graph cleaner.

### Logging setup is library-hostile

- `src/clophfit/_config.py:17-43` configures the root logger
- `src/clophfit/_config.py:73-91` adds handlers at the root level

This is convenient for a CLI, but risky for library consumers because importing or calling package code can affect global logging behavior. A package-scoped logger would be safer.

## Quality and hygiene findings

### TODO/FIXME debt

~23 `TODO`/`FIXME` comments remain across the codebase, concentrated in `bayes.py` and `prtecan.py`. These should be triaged: resolve, convert to issues, or remove if stale.

### README drift

- `README.md:163-171` documents `clophfit.binding`
- no corresponding `src/clophfit/binding` package exists

Impact: public-facing documentation is ahead of or diverged from the actual package surface.

### Rotated log files are not ignored

- `.gitignore:5-7` ignores `*.log`
- it does not ignore rotated files such as `*.log.1`, `*.log.2`, etc.

That matches the working tree clutter visible in this repository (`ppr_tecan_cli.log.1`, `.2`, `.3`).

### Dataset plotting makes label assumptions

- `src/clophfit/fitting/data_structures.py:378-380` hardcodes colors for `y0`, `y1`, and `y2`

This is not a blocker, but it makes a generic container behave like it knows specific label conventions.

### Legacy code remains in-tree

`src/clophfit/old/` is still present. Even if excluded from lint/type/coverage paths, it increases cognitive load because it is not obvious to a new contributor whether it is archival, experimental, or still semi-supported.

## Recommended next steps

1. Fix the two model-selection bugs in `fitting/odr.py` and `__main__.py`.
1. Split `prtecan.py` — extract `Titration` responsibilities into focused classes.
1. Refactor `bayes.py` around shared helpers or explicit fitting strategies.
1. Centralize configuration constants (`N_BOOT`, `EMCEE_STEPS`, etc.).
1. Move generic helpers out of `prtecan` into a neutral shared module.
1. Triage TODO/FIXME comments — resolve, file issues, or remove.
1. Update README package references and extend `.gitignore` for rotated logs.
