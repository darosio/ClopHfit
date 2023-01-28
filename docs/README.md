[![PyPI](https://img.shields.io/pypi/v/ClopHfit.svg)](https://pypi.org/project/ClopHfit/)
[![CI](https://github.com/darosio/ClopHfit/actions/workflows/ci.yml/badge.svg)](https://github.com/darosio/ClopHfit/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/darosio/ClopHfit/branch/main/graph/badge.svg?token=OU6F9VFUQ6)](https://codecov.io/gh/darosio/ClopHfit)
[![RtD](https://readthedocs.org/projects/clophfit/badge/)](https://clophfit.readthedocs.io/)
[![zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.6354112.svg)](https://doi.org/10.5281/zenodo.6354112)

# ClopHfit

Cli for fitting macromolecule pH titration or binding assay data, e.g.
fluorescence spectra.

- Version: "0.3.20"

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

      clop prtecan list.pH -k ph --scheme ../scheme.txt --dil additions.pH --norm --out prova2 --Klim 6.8,8.4 --sel 7.6,20

  To reproduce older pr.tecan add [\--no-weight]{.title-ref} option:

      clop prtecan list.pH -k ph --scheme ../scheme.txt --no-bg --no-weight --out 4old --Klim 6.8,8.4 --sel 7.6,20

- Predict chloride dissociation constant [K_d]{.title-ref} at given pH:

      clop eq1 --help

To use clophfit in your python:

    from clophfit import prtecan, binding

## Installation

    pipx install clophfit

You can get the library directly from
[PyPI](https://pypi.org/project/ClopHfit/):

    pip install clophfit

## Development

You need the following requirements:

- `nox` for test automation. If you don't have nox, you can use `pipx run nox`
  to run it without installing, or `pipx install nox`. To use, run `nox`. This
  will lint and test using every installed version of Python on your system,
  skipping ones that are not installed. You can also run specific jobs:
  ```bash
  nox -l  # list available tests
  nox -s lint  # Lint only
  nox -s tests-3.9  # Python 3.9 tests only
  nox -s docs -- serve  # Build and serve the docs
  nox -s build  # Make an SDist and wheel
  ```
  Nox handles everything for you, including setting up an temporary virtual
  environment for each run.
- `pdm` for package dependency managements;
- `pre-commit` for all style and consistency checking. While you can run it with
  nox, this is such an important tool that it deserves to be installed on its
  own. Install pre-commit and run:
  ```bash
  pre-commit install --hook-type commit-msg --hook-type pre-push
  (or: nox -s init)
  pre-commit run -a
  ```
  to check all files. If pre-commit fails during pushing upstream then stage
  changes, Commit Extend (into previous commit), and repeat pushing.

`pip`, `nox`, `pdm` and `pre-commit` are pinned in
.github/workflows/constraints.txt for consistency with CI/CD.

### Setting up a development environment manually

You can set up a development environment by running:

```bash
python3 -m venv .venv
source ./.venv/bin/activate
pip install -v -e .[dev,tests,docs]
```

For using [Jupyter](https://jupyter.org/) during development:

    pdm run jupiter notebook

And only in case you need a system wide easy accessible kernel:

    pdm run python -m ipykernel install --user --name="cloph-310"

### Testing and coverage

Use pytest to run the unit checks:

    pytest

Use `coverage` to generate coverage reports:

    coverage run --parallel -m pytest

Or use nox:

    nox -rs tests typeguard xdoctest

### Building docs

You can build the docs using:

    nox -s docs

You can see a preview with:

    nox -s docs -- serve

When needed (e.g. API updates):

    sphinx-apidoc -f -o docs/api/ src/clophfit/

### Bump and releasing

I can bump and upload build to test.pypi using:

    nox -rs bump

I can test new dist in local venv using:

    python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ clophfit

- Revise changelog, stage files, commit and push
- gh pr create (wait for test completion)
- git co main
- pr merge --squash --delete-branch -t "fix| refactor| ci: msg" #number
- gh release create ”v0.3.12”
- git pull --tags

Basically use commitizen and github-cli to release:

    pipx run --spec commitizen cz bump --changelog-to-stdout --files-only (--prerelease alpha) --increment MINOR
    gh release create (--target devel) v0.3.0a0

### Configuration files

Manually updated pinned dependencies for CI/CD:

- docs/requirements.txt
- .github/workflows/constraints.txt

Configuration files:

- pre-commit configured in .pre-commit-config.yaml;
- flake8 configured in .flake8 (pinned in pre-commit);
- black configured in pyproject.toml (pinned in pre-commit);
- isort configured in pyproject.toml (pinned in pre-commit);
- darglint configured in .darglint (pinned in pre-commit);
- coverage configured in pyproject.toml (tests deps);
- mypy configured in pyproject.toml (tests deps);
- commitizen in pyproject.toml (after `cz init`) (both pinned in pre-commit and
  test deps).

pre-commit ensures up-to-date constraints.txt pin/freeze all packages in
“.[dev,docs,tests]”.

## TODO

- mypy into lint?

- clean all commented leftover

- release drafter; maybe useful when merging pr into main.
- readthedocs or ghpages?
  <https://www.docslikecode.com/articles/github-pages-python-sphinx/>
