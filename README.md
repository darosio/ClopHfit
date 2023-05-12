# ClopHfit

[![PyPI](https://img.shields.io/pypi/v/ClopHfit.svg)](https://pypi.org/project/ClopHfit/)
[![CI](https://github.com/darosio/ClopHfit/actions/workflows/ci.yml/badge.svg)](https://github.com/darosio/ClopHfit/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/darosio/ClopHfit/branch/main/graph/badge.svg?token=OU6F9VFUQ6)](https://codecov.io/gh/darosio/ClopHfit)
[![RtD](https://readthedocs.org/projects/clophfit/badge/)](https://clophfit.readthedocs.io/)
[![zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.6354112.svg)](https://doi.org/10.5281/zenodo.6354112)

This package provides a command line interface for fitting pH titration or
binding assay data for macromolecules, such as fluorescence spectra. With this
tool, users can easily analyze their data and obtain accurate fitting results.

- Version: "0.6.0"

## Installation

You can get the library directly from
[PyPI](https://pypi.org/project/ClopHfit/):

    pip install clophfit

or with [pipx](https://pypa.github.io/pipx/):

    pipx install clophfit

## Usage

You can check out the documentation on <https://darosio.github.io/ClopHfit> for
up to date usage information and examples.

### CLI

ClopHfit provides several command line interface tools for fitting and
processing data.

#### prtecan

Extract and fit titrations from a list of Tecan files collected at various pH or
chloride concentrations:

    clop prtecan list.pH -k ph --scheme ../scheme.txt --dil additions.pH --norm
        --out prova2 --Klim 6.8,8.4 --sel 7.6,20

Use the --no-weight option to reproduce an older pr.tecan version.

#### eq1

Predict chloride dissociation constant K_d at a given pH:

    clop eq1 --help

#### prenspire

Parser for EnSpire (PerkinElmer) file:

    clop prenspire file.csv -out folder

Destination folder (default: "./Meas") will contain for each Measurement:

- a table (csv) in wide tabular form e.g. <lambda, A01, ..., H12>;
- corresponding graphical (png) representation of spectra from all wells.

### Python

ClopHfit can be imported and used as a Python package. The following modules are
available:

    clophfit.prenspire - parser for EnSpire (PerkinElmer) files
    clophfit.prtecan - perform fitting of pH titration or binding assay data
    clophfit.binding - perform fitting of macromolecule binding assay data

To use clophfit in your python:

    from clophfit import prenspire, prtecan, binding

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

## License

We use a shared copyright model that enables all contributors to maintain the
copyright on their contributions.

All code is licensed under the terms of the [revised BSD license](LICENSE.txt).

## Contributing

If you are interested in contributing to the project, please read our
[contributing](https://darosio.github.io/ClopHfit/references/contributing.html)
and
[development environment](https://darosio.github.io/ClopHfit/references/contributing.html#development)
guides, which outline the guidelines and conventions that we follow for
contributing code, documentation, and other resources.
