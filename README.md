# ClopHfit

[![PyPI](https://img.shields.io/pypi/v/ClopHfit.svg)](https://pypi.org/project/ClopHfit/)
[![CI](https://github.com/darosio/ClopHfit/actions/workflows/ci.yml/badge.svg)](https://github.com/darosio/ClopHfit/actions/workflows/ci.yml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/darosio/ClopHfit/main.svg)](https://results.pre-commit.ci/latest/github/darosio/ClopHfit/main)
[![codecov](https://codecov.io/gh/darosio/ClopHfit/branch/main/graph/badge.svg?token=OU6F9VFUQ6)](https://codecov.io/gh/darosio/ClopHfit)
[![](https://img.shields.io/badge/Pages-blue?logo=github)](https://darosio.github.io/ClopHfit/)
[![RtD](https://readthedocs.org/projects/clophfit/badge/)](https://clophfit.readthedocs.io/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6354111.svg)](https://doi.org/10.5281/zenodo.6354111)

This package provides a command line interface for fitting pH titration or
binding assay data for macromolecules, such as fluorescence spectra. With this
tool, users can easily analyze their data and obtain accurate fitting results.

## Features

- **Plate Reader Data Parser**: Support for Tecan and EnSpire (PerkinElmer) file formats
- **Non-linear Least Square Fitting**: Advanced fitting algorithms with uncertainty estimation
- **pH and Chloride Titration Analysis**: Specialized tools for GFP library analysis
  - Multi-wavelength analysis (e.g., 400, 485 nm) with separate and global fitting options
  - Bootstrap uncertainty estimation
  - Automatic buffer subtraction for each titration point
  - Control variant reporting (S202N, E2, V224Q)
  - Dilution correction for titration additions
  - Robust error handling with automatic plot and data export

### Fitting Methods

ClopHfit provides multiple fitting algorithms validated on 364+ real experimental wells:

- **ODR-Recursive**: Orthogonal Distance Regression (best error bar precision) ✅ **Recommended**. ODR is provided by `odrpack`.
- **outlier2**: Two-stage outlier detection (good precision, fast)
- **IRLS**: Iteratively Reweighted Least Squares
- **Recursive**: Iterative reweighting
- **Bayesian (PyMC)**: MCMC for rigorous uncertainty quantification
- Standard LM and Robust Huber also available

**See**: [`FITTING_METHODS_SUMMARY.md`](FITTING_METHODS_SUMMARY.md) for guidance on method selection.

**Full analysis**: [`FITTING_METHODS_COMPARISON.md`](FITTING_METHODS_COMPARISON.md)

## Installation

### From PyPI

Using pip:

```bash
pip install clophfit
```

### Recommended: Using pipx

For isolated installation (recommended):

```bash
pipx install clophfit
```

### Shell Completion

Enable shell completion for the `clop` command:

#### Bash

```bash
_CLOP_COMPLETE=bash_source clop > ~/.local/bin/clop-complete.bash
source ~/.local/bin/clop-complete.bash
# Add to your ~/.bashrc to make it permanent:
echo 'source ~/.local/bin/clop-complete.bash' >> ~/.bashrc
```

#### Fish

```bash
_CLOP_COMPLETE=fish_source clop | source
# Add to fish config to make it permanent:
_CLOP_COMPLETE=fish_source clop > ~/.config/fish/completions/clop.fish
```

## Usage

Docs: https://clophfit.readthedocs.io/

### CLI

ClopHfit provides several command line interface tools for fitting and
processing data.

#### prtecan - Tecan File Analysis

Extract and fit titrations from Tecan files collected at various pH or chloride concentrations:

```bash
# pH titration analysis
ppr -o output_folder --is-ph tecan list.pH --scheme scheme.txt --norm \
    --dil additions.pH --Klim 6.8,8.4

# Chloride titration analysis
ppr -o output_folder tecan list.Cl --scheme scheme.txt --norm \
    --dil additions.Cl --Klim 1.0,50.0
```

**Note**: Use `--no-weight` option to reproduce older pr.tecan behavior.

#### eq1 - Dissociation Constant Prediction

Predict chloride dissociation constant K_d at a given pH:

```bash
clop eq1 --help
clop eq1 12 8.2 7.3
```

#### prenspire - EnSpire File Parser

Process EnSpire (PerkinElmer) files:

```bash
# Basic parsing
ppr -o output_folder enspire data.csv

# With annotation file for titration analysis
ppr -o output_folder enspire data.csv note.csv
```

Destination folder (default: "./Meas-${version}") will contain for each Measurement:

- a table (csv) in wide tabular form e.g. \<lambda, A01, ..., H12>;
- corresponding graphical (png) representation of spectra from all wells.

By adding a note.csv file:

```
ppr -o folder enspire file.csv note.csv
```

destination folder (default: "./Meas-${version}") will also contain:

- a pdf for each titration and label;
- a pdf for global (multiband) analysis pdfalong with global and all_SVD;
- a pdf for SVD analysis of all concatenated spectra.

#### note_to_csv - Note File Converter

Convert and combine note files for different experimental conditions:

```bash
# Convert notes for different temperatures
note_to_csv -t 37.0 -l "B E F" -o temp_37 NTT-G03-Cl_note
note_to_csv -t 20.0 -l "A C D" -o temp_20 NTT-G03-Cl_note

# Combine temperature conditions
cat temp_20 temp_37 > G03_Cl_note.csv
```

### Python

ClopHfit can be imported and used as a Python package. The following modules are
available:

```
clophfit.prenspire - parser for EnSpire (PerkinElmer) files
clophfit.prtecan - perform fitting of pH titration or binding assay data
clophfit.binding - perform fitting of macromolecule binding assay data
```

To use clophfit in your python:

```
from clophfit import prenspire, prtecan, binding
```

### Model Validation & Diagnostics (`v0.7+`)

New reusable modules for comparing fitting methods and validating Bayesian
models:

```python
from clophfit.fitting.model_validation import (
    residuals_from_multifit,          # MultiFitResult → long DataFrame
    residuals_from_fit_results,       # dict[str, FitResult] → long DataFrame
    model_residual_score_table,       # all metrics in one call
    residual_distribution_summary,    # per-label σ, MAD, tail fractions
    residual_x_trend_summary,         # residual trend vs titration step
    residual_lag1_autocorrelation,    # per-well lag-1 correlation
    residual_cross_label_correlation, # cross-label residual structure
    x_axis_sanity,                    # check x_per_well invariants
    trace_diagnostics,                # ESS, R̂, divergences, optional LOO
)

from clophfit.fitting.ctr_validation import (
    make_ctr_holdout_scheme,          # remove one CTR well from scheme
    iter_ctr_holdouts,               # iterate all possible CTR holdouts
    summarize_bayesian_ctr_holdout,   # ΔK = K_heldout − K_ctr_group
    classical_ctr_holdout_rows,       # post-hoc CTR-LOO for classical fits
)

from clophfit.fitting.diagnostic_plots import (
    plot_residual_overview,           # 4-panel matplotlib (no seaborn)
)
```

**Example — Bayesian fit diagnostics:**

```python
from clophfit.fitting.model_validation import (
    residuals_from_multifit, trace_diagnostics, x_axis_sanity,
)
from clophfit.fitting.models import binding_1site

multi = fit_binding_pymc_multi(results, scheme, ...)
diag = trace_diagnostics(multi.trace, compute_loo=False)
x_check = x_axis_sanity(multi.trace)
assert x_check.get("x_step0_max_abs_spread", 1.0) < 1e-6

residuals = residuals_from_multifit(
    multi, trace_id="my_model", binding_function=binding_1site,
)
# columns: trace_id, well, label, step, x, y, yhat, sigma, raw_res, std_res
```

**Example — CTR holdout validation:**

```python
from clophfit.fitting.ctr_validation import (
    make_ctr_holdout_scheme, summarize_bayesian_ctr_holdout,
)

heldout = make_ctr_holdout_scheme(scheme, group_name="E2GFP", heldout_well="B12")
multi_loo = fit_binding_pymc_multi(results, heldout, ...)
summary = summarize_bayesian_ctr_holdout(
    multi_loo.trace, trace_id="loo_E2GFP_B12",
    ctr_group="E2GFP", heldout_well="B12", rope=0.10,
)
# summary["delta_k_hdi94_contains_zero"] → True if K_heldout ≈ K_group
```

## Development

Requires Python `uv`.

With uv:

```bash
# one-time
pre-commit install
# dev tools and deps
uv sync --group dev
# lint/test
uv run ruff check .  (or: make lint)
uv run pytest -q  (or: make test)
```

## Dependency updates (Renovate)

We use Renovate to keep dependencies current.

Enable Renovate:

1. Install the GitHub App: https://github.com/apps/renovate (Settings → Integrations → GitHub Apps → Configure → select this repo/org).
1. This repo includes a `renovate.json` policy. Renovate will open a “Dependency Dashboard” issue and PRs accordingly.

Notes:

- Commit style: `build(deps): bump <dep> from <old> to <new>`
- Pre-commit hooks are grouped and labeled; Python version bumps in `pyproject.toml` are disabled by policy.

Migrating from Dependabot:

- You may keep “Dependabot alerts” ON for vulnerability visibility, but disable Dependabot security PRs.

## Template updates (Cruft)

This project is linked to its Cookiecutter template with Cruft.

- Check for updates: `cruft check`
- Apply updates: `cruft update -y` (resolve conflicts, then commit)

CI runs a weekly job to open a PR when template updates are available.

First-time setup if you didn’t generate with Cruft:

```bash
pipx install cruft  # or: pip install --user cruft
cruft link --checkout main https://github.com/darosio/cookiecutter-python.git
```

Notes:

- The CI workflow skips if `.cruft.json` is absent.
- If you maintain a stable template branch (e.g., `v1`), link with `--checkout v1`. You can also update within that line using `cruft update -y --checkout v1`.

## License

We use a shared copyright model that enables all contributors to maintain the
copyright on their contributions.

All code is licensed under the terms of the [revised BSD license](LICENSE.txt).

## Contributing

If you are interested in contributing to the project, please read our
[contributing](https://darosio.github.io/ClopHfit/references/contributing.html)
and
[development environment](https://darosio.github.io/ClopHfit/references/development.html)
guides, which outline the guidelines and conventions that we follow for
contributing code, documentation, and other resources.
