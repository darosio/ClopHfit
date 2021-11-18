[![PyPI](https://img.shields.io/pypi/v/ClopHfit.svg)](https://pypi.org/project/ClopHfit/)

# ClopHfit #

* Cli for fitting macromolecule pH titration or binding assays data e.g. fluorescence spectra.
* Version: "0.1.1"

## Installation

At this stage few scripts are available in src/clophfit/old.

    pyenv install 3.6.15
    poetry install
    poetry run pytest -v

## Use

### fit_titration.py ###

A single script for pK and Cl and various methods w/out bootstraping:
1. svd
2. bands and
3. single lambda.

>   input ← csvtable and note_file

>   output → pK spK (stdout) and pdf of analysis

#### To do

- Bootstrap svd with optimize or lmfit.
- **Average spectra**
- Join spectra ['B', 'E', 'F']
- Compute band integral (or sums)

### fit_titration_global.py ###

A script for fitting tuples (y1, y2) of values for each concentration (x).
It uses lmfit confint and bootstrap.

>   input ← x y1 y2 (file)

>   output → K SA1 SB1 SA2 SB2 , png and correl.png
    
In global fit the best approach was using lmfit without bootstraping.

#### Example
     
	 for i in *.dat; do gfit $i png2 --boot 99 > png2/$i.txt; done
