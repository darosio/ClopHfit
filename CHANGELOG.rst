=========
Changelog
=========

0.2.0 (2021-11-18)
------------------

* pandas.rpy (<=0.19) now lives in rpy2.

* usage.org (exported to .rst) serves as tutorial in docs and includes::

  - liaisan-data
  - new-bootstrap
  - lmfit global
  - emcee (very slow)

* Test automation requires nox and nox-poetry.
* Formatting with black[jupyter] configured in pyproject.
* Linters are configured in .flake8 .darglint and .isort.cfg and include::

  - flake8-isort
  - flake8-bugbear
  - flake8-docstrings
  - darglint
  - flake8-eradicate
  - flake8-comprehensions
  - flake8-pytest-style
  - flake8-annotations (see mypy)

* pre-commit configured in .pre-commit-config.yaml activated with::

  - pre-commit install
  - commitizen install --hook-type commit-msg
* Tests with pytest-cov and .coveragerc configuration for old scripts::

  - fit_titration.py

* Type annotation configured in mypy.ini.

* __version__ in __init__ for py>=38 (MAYBE cz bump --changelog)

- xdoctest
- sphinx (sphinxcontrib-plantuml, pydata-sphinx-theme, sphinx-autodoc-typehints)
  - mkdir docs; cd docs
  - sphinx-quickstart
  - edit conf.py ["sphinx.ext.autodoc"] and index.rst [e.g. references/modules]
  - sphinx-apidoc -f -o docs src/prtecan/
  - nbsphinx


0.1.1 (2021-11-14)
------------------

* Reference for older scripts with reproducibility entrusted to Poetry_ and
  Pyenv_::

	LDFLAGS=-L/usr/lib/openssl-1.0/ CFLAGS=-I/usr/include/openssl-1.0/ pyenv install 3.4.10
	++CONFIGURE_OPTS="--without-ensurepip" pyenv install 3.5.8++
	CC=clang pyenv install 3.5.10
	poetry env use 3.5
	poetry install
	../../src/clophfit/fit_titration.py Meas/A04\ Cl_A.csv NTT-A04-Cl_note -t cl -d output-enspire
	../../src/clophfit/fit_titration_global.py D05.dat output-D05 --boot 99
	../../src/clophfit/fit_titration_global.py -t cl --boot 999 B05-20130628-cor.dat output-B05
* Note that fit_rpy.py did never work (indeed did not use #!/usr/bin/env python).
* Tested deps for fit_titration* without any warning::

	cycler          0.10.0 Composable style cycles
    lmfit           0.8.3  Least-Squares Minimization with Bounds and Constraints
    matplotlib      1.5.3  Python plotting package
    numpy           1.10.4 NumPy: array processing for numbers, strings, records, and objects.
    pandas          0.18.1 Powerful data structures for data analysis, time series,and statistics
    pyparsing       2.4.7  Python parsing module
    python-dateutil 2.8.1  Extensions to the standard Python datetime module
    pytz            2021.3 World timezone definitions, modern and historical
    rpy2            2.3.10 Python interface to the R language (embedded R)
    scipy           0.18.1 SciPy: Scientific Library for Python
    seaborn         0.7.1  Seaborn: statistical data visualization
    six             1.16.0 Python 2 and 3 compatibility utilities
	
0.1.0 (2021-3-4)
----------------

* Initial placeholder.

.. _Poetry: https://python-poetry.org
.. _Pyenv: https://github.com/pyenv/pyenv
