[![PyPI](https://img.shields.io/pypi/v/ClopHfit.svg)](https://pypi.org/project/ClopHfit/)
[![Tests](https://github.com/darosio/ClopHfit/actions/workflows/tests.yml/badge.svg)](https://github.com/darosio/ClopHfit/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/darosio/ClopHfit/branch/main/graph/badge.svg?token=OU6F9VFUQ6)](https://codecov.io/gh/darosio/ClopHfit)
[![RtD](https://readthedocs.org/projects/clophfit/badge/)](https://clophfit.readthedocs.io/)
[![zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.6354112.svg)](https://doi.org/10.5281/zenodo.6354112)

# ClopHfit

Cli for fitting macromolecule pH titration or binding assay data, e.g.
fluorescence spectra.

-   Version: "0.3.10"

## Features

-   Plate Reader data Parser.
-   Perform non-linear least square fitting.
-   Extract and fit pH and chloride titrations of GFP libraries.
    -   For 2 labelblocks (e.g. 400, 485 nm) fit data separately and
        globally.
    -   Estimate uncertainty using bootstrap.
    -   Subtract buffer for each titration point.
    -   Report controls e.g. S202N, E2 and V224Q.
    -   Correct for dilution of titration additions.
    -   Plot data when fitting fails and save txt file anyway.

## Usage

-   Extract and fit titrations from a list of tecan files collected at
    various pH or chloride concentrations:

        clop prtecan --help

    For example:

        clop prtecan list.pH -k ph --scheme ../scheme.txt --dil additions.pH --norm \
          --out prova2 --Klim 6.8,8.4 --sel 7.6,20

    To reproduce older pr.tecan add [\--no-weight]{.title-ref} option:

        clop prtecan list.pH -k ph --scheme ../scheme.txt --no-bg --no-weight \
          --out 4old --Klim 6.8,8.4 --sel 7.6,20

-   Predict chloride dissociation constant [K_d]{.title-ref} at given
    pH:

        clop eq1 --help

To use clophfit in a project:

    from clophfit import prtecan, binding

## Installation

You can get the library directly from
[![PyPI](https://img.shields.io/pypi/v/ClopHfit.svg)](https://pypi.org/project/ClopHfit/):

    pip install clophfit

## Development

Prepare a virtual development environment and test first installation:

    pyenv install 3.10.2
    poetry env use 3.10
    poetry install
    poetry run pytest -v

Make sure:

    pre-commit install
    pre-commit install --hook-type commit-msg

For [Jupyter](https://jupyter.org/):

    poetry run python -m ipykernel install --user --name="cloph-310"

To generate docs:

    poetry run nox -rs docs

When needed (e.g. API updates):

    sphinx-apidoc -f -o docs/api/ src/clophfit/

Use commitizen and github-cli to release:

    poetry run cz bump --changelog-to-stdout --files-only (--prerelease alpha) --increment MINOR
    gh release create (--target devel) v0.3.0a0

Remember!!! Update::
- ClopHfit/docs/requirements.txt
- ClopHfit/.github/workflows/constraints.txt

### Development environment

-   Test automation requires nox and nox-poetry.

-   Formatting with black\[jupyter\] configured in pyproject.

-   Linters are configured in .flake8 .darglint and .isort.cfg and
    include:

        - flake8-isort
        - flake8-bugbear
        - flake8-docstrings
        - darglint
        - flake8-eradicate
        - flake8-comprehensions
        - flake8-pytest-style
        - flake8-annotations (see mypy)
        - flake8-rst-docstrings

    > -   rst-lint

-   pre-commit configured in .pre-commit-config.yaml activated with:

        - pre-commit install
        - commitizen install --hook-type commit-msg

-   Tests coverage (pytest-cov) configured in .coveragerc.

-   Type annotation configured in mypy.ini.

-   [Commitizen](https://commitizen-tools.github.io/commitizen/) also
    used to bump version:

        cz bump --changelog-to-stdout --files-only --prerelease alpha --increment MINOR

    -   need one-time initialization:

            (cz init)

-   xdoctest

-   sphinx with pydata-sphinx-theme and sphinx-autodoc-typehints.
    (nbsphinx, sphinxcontrib-plantuml):

        mkdir docs; cd docs
        sphinx-quickstart

    Edit conf.py \[\"sphinx.ext.autodoc\"\] and index.rst \[e.g.
    api/modules\]:

        sphinx-apidoc -f -o docs/api/ src/clophfit/

-   CI/CD configured in .github/workflows:

        tests.yml
        release.yml

    Remember to update tools version e.g. nox_poetry==0.9.

### What is missing to [modernize](https://cjolowicz.github.io/posts/hypermodern-python-06-ci-cd/):

-   coveralls/Codecov
-   release drafter; maybe useful when merging pr into main.
-   readthedocs or ghpages?
    <https://www.docslikecode.com/articles/github-pages-python-sphinx/>

## Code of Conduct

Everyone interacting in the readme_renderer project\'s codebases, issue
trackers, chat rooms, and mailing lists is expected to follow the [PSF
Code of
Conduct](https://github.com/pypa/.github/blob/main/CODE_OF_CONDUCT.md).
