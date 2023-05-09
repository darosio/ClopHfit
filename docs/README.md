# ClopHfit

[![PyPI](https://img.shields.io/pypi/v/ClopHfit.svg)](https://pypi.org/project/ClopHfit/)
[![CI](https://github.com/darosio/ClopHfit/actions/workflows/ci.yml/badge.svg)](https://github.com/darosio/ClopHfit/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/darosio/ClopHfit/branch/main/graph/badge.svg?token=OU6F9VFUQ6)](https://codecov.io/gh/darosio/ClopHfit)
[![RtD](https://readthedocs.org/projects/clophfit/badge/)](https://clophfit.readthedocs.io/)
[![zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.6354112.svg)](https://doi.org/10.5281/zenodo.6354112)

Cli for fitting macromolecule pH titration or binding assay data, e.g.
fluorescence spectra.

- Version: "0.5.0"

## Features

- Plate Reader data Parser.
- Perform non-linear least square fitting.
- Extract and fit pH and chloride titrations of GFP libraries.
  - For 2 labelblocks (e.g. 400, 485 nm) fit data separately and globally.
  - Estimate uncertainty using bootstrap.
  - Subtract buffer for each titration point.
  - Report controls e.g. S202N, E2 and V224Q.
  - Correct for dilution of titration additions.
  - Plot data when fitting fails and save txt file anyway.

## Usage

- Extract and fit titrations from a list of tecan files collected at various pH
  or chloride concentrations:

      clop prtecan --help

  For example:

        clop prtecan list.pH -k ph --scheme ../scheme.txt --dil additions.pH \
            --norm --out prova2 --Klim 6.8,8.4 --sel 7.6,20

  To reproduce older pr.tecan add `--no-weight` option:

        clop prtecan list.pH -k ph --scheme ../scheme.txt --no-bg --no-weight \
            --out 4old --Klim 6.8,8.4 --sel 7.6,20

- Predict chloride dissociation constant `K_d` at given pH:

        clop eq1 --help

- Parser for EnSpire (PerkinElmer) file:

        clop prenspire file.csv -out folder

  Destination folder (default: "./Meas") will contain for each _Measurement_:

  - a table (csv) in wide tabular form e.g. <lambda, A01, ..., H12>;
  - corresponding graphical (png) representation of spectra from all wells.

To use clophfit in your python:

    from clophfit import prenspire, prtecan, binding

## Installation

    pipx install clophfit

You can get the library directly from
[PyPI](https://pypi.org/project/ClopHfit/):

    pip install clophfit

## Development

You need the following requirements:

- `hatch` for test automation and package dependency managements. If you don't
  have hatch, you can use `pipx run hatch` to run it without installing, or
  `pipx install hatch`. Dependencies are locked thanks to
  [pip-deepfreeze](https://pypi.org/project/pip-deepfreeze/). You can run
  `hatch env show` to list available environments and scripts.

        hatch run init  # init repo with pre-commit hooks
        hatch run sync  # sync venv with deepfreeze

        hatch run lint:run
        hatch run tests.py3.10:all

  Hatch handles everything for you, including setting up an temporary virtual
  environment for each run.

- `pre-commit` for all style and consistency checking. While you can run it with
  nox, this is such an important tool that it deserves to be installed on its
  own. If pre-commit fails during pushing upstream then stage changes, Commit
  Extend (into previous commit), and repeat pushing.

`pip`, `pip-deepfreeze` and `hatch` are pinned in
.github/workflows/constraints.txt for consistency with CI/CD.

    pipx install pre-commit
    pipx install hatch
    pipx runpip hatch install hatch-pip-deepfreeze

### Setting up a development with direnv

    echo "layout hatch" > .envrc
    hatch run init

### Setting up a development environment manually

You can set up a development environment by running:

    python3 -m venv .venv
    source ./.venv/bin/activate
    pip install -v -e .[dev,tests,docs]

With direnv for using [Jupyter](https://jupyter.org/) during development:

    jupiter notebook

And only in case you need a system wide easy accessible kernel:

    python -m ipykernel install --user --name="clop"

### Testing and coverage

Use pytest to run the unit checks:

    pytest

Use `coverage` to generate coverage reports:

    coverage run --parallel -m pytest

Or use hatch:

    hatch run tests:all
    hatch run coverage:combine
    hatch run coverage:report

### Building docs

You can build the docs using:

    hatch run docs:sync
    hatch run docs:build

You can see a preview with:

    hatch run docs:serve

When needed (e.g. API updates):

    sphinx-apidoc -f -o docs/api/ src/clophfit/

### Bump and releasing

To bump version and upload build to test.pypi using:

    hatch run bump
    hatch run bump "--increment PATCH" "--files-only" \
        ["--no-verify" to bypass pre-commit and commit-msg hooks]

while to update only the CHANGELOG.md file:

    hatch run ch

Release will automatically occur after pushing.

(Otherwise)

    pipx run --spec commitizen cz bump --changelog-to-stdout --files-only \
        (--prerelease alpha) --increment MINOR

To keep clean development history use branches and pr:

    gh pr create --fill
    gh pr merge --squash --delete-branch [-t “fix|ci|feat: msg”]

### Configuration files

Manually updated pinned dependencies for CI/CD:

- .github/workflows/constraints.txt (testing dependabot)

Configuration files:

- pre-commit configured in .pre-commit-config.yaml;
- flake8 (for rst-docstrings and bandit) configured in .flake8 (pinned in
  pre-commit);
- black configured in pyproject.toml (pinned in pre-commit);
- ruff configured in pyproject.toml (pinned in pre-commit);
- darglint configured in .darglint (pinned in pre-commit);
- codespell configured in .codespellrc (pinned in pre-commit);
- coverage configured in pyproject.toml (tests deps);
- mypy configured in pyproject.toml (tests deps);
- commitizen in pyproject.toml (dev deps and pinned in pre-commit).

pip-df generates requirements[-dev,docs,tests].txt.

Other manual actions:

    pylint src/ tests/
    bandit -r src/
