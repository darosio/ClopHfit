# ClopHfit

[![PyPI](https://img.shields.io/pypi/v/ClopHfit.svg)](https://pypi.org/project/ClopHfit/)
[![CI](https://github.com/darosio/ClopHfit/actions/workflows/ci.yml/badge.svg)](https://github.com/darosio/ClopHfit/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/darosio/ClopHfit/branch/main/graph/badge.svg?token=OU6F9VFUQ6)](https://codecov.io/gh/darosio/ClopHfit)
[![RtD](https://readthedocs.org/projects/clophfit/badge/)](https://clophfit.readthedocs.io/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6354111.svg)](https://doi.org/10.5281/zenodo.6354111)

This package provides a command line interface for fitting pH titration or
binding assay data for macromolecules, such as fluorescence spectra. With this
tool, users can easily analyze their data and obtain accurate fitting results.

- Version: "0.10.8"

## Installation

You can get the library directly from [PyPI](https://pypi.org/project/ClopHfit/)
using `pip`:

    pip install clophfit

Alternatively, you can use [pipx](https://pypa.github.io/pipx/) to install it in
an isolated environment:

    pipx install clophfit

To enable auto completion for the `clop` command, follow these steps:

1.  Generate the completion script by running the following command:

        _CLOP_COMPLETE=bash_source clop > ~/.local/bin/clop-complete.bash

2.  Source the generated completion script to enable auto completion:

        source ~/.local/bin/clop-complete.bash

## Usage

You can check out the documentation on <https://darosio.github.io/ClopHfit> for
up to date usage information and examples.

### CLI

ClopHfit provides several command line interface tools for fitting and
processing data.

#### prtecan

Extract and fit titrations from a list of Tecan files collected at various pH or
chloride concentrations:

    ppr -o prova2 --is-ph tecan list.pH --scheme ../scheme.txt --norm
        --dil additions.pH --Klim 6.8,8.4 --sel 7.6,20

Use the --no-weight option to reproduce an older pr.tecan version.

#### eq1

Predict chloride dissociation constant K_d at a given pH:

    clop eq1 --help

#### prenspire

Parser for EnSpire (PerkinElmer) file:

    ppr -o folder enspire file.csv

Destination folder (default: "./Meas-${version}") will contain for each Measurement:

- a table (csv) in wide tabular form e.g. <lambda, A01, ..., H12>;
- corresponding graphical (png) representation of spectra from all wells.

By adding a note.csv file:

    ppr -o folder enspire file.csv note.csv

destination folder (default: "./Meas-${version}") will also contain:

- a pdf for each titration and label;
- a pdf for global (multiband) analysis pdfalong with global and all_SVD;
- a pdf for SVD analysis of all concatenated spectra.

#### note_to_csv

    note_to_csv -t 37.0 -l "B E F" -o 37 NTT-G03-Cl_note
    note_to_csv -t 20.0 -l "A C D" -o 20 NTT-G03-Cl_note
    cat 20 37 > G03_Cl_note.csv

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
[development environment](https://darosio.github.io/ClopHfit/references/development.html)
guides, which outline the guidelines and conventions that we follow for
contributing code, documentation, and other resources.
