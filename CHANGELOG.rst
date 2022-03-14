=========
Changelog
=========

0.3.0 (2022-03-14)
--------------------

Added
~~~~~

- Tecan file parser.
- usage.org (exported to .rst) serves as tutorial in docs and includes:

  - liaisan-data
  - new-bootstrap
  - lmfit global
  - emcee (very slow)
- command ``clop``.

Changed
~~~~~~~

- Update to python 3.9 and 3.10.
- Update dependencies::

	poetry show --outdated
  with required minor changes in old scipts.
- nox-poetry.
- pandas.rpy (<=0.19) now lives in rpy2.

Fixed
~~~~~

- warning for keys_unk set used as index in pd.


0.2.1 (2021-11-18)
------------------

- Update to python 3.6.
- Py.test for:

  - ``fit_titration.py``
  - ``fit_titration_global.py``
- lmfit==0.8.3 to prevent ``fit-titration_global.py`` to fail.
- `_tmpoutput` is not deleted; watch out for false positive.


0.2.0 (2021-11-14)
------------------

- Reference for running older scripts; reproducibility thanks to Poetry_ and
  Pyenv_::

	LDFLAGS=-L/usr/lib/openssl-1.0/ CFLAGS=-I/usr/include/openssl-1.0/ pyenv install 3.4.10
	++CONFIGURE_OPTS="--without-ensurepip" pyenv install 3.5.8++
	CC=clang pyenv install 3.5.10
	poetry env use 3.5
	poetry install
	../../src/clophfit/fit_titration.py Meas/A04\ Cl_A.csv NTT-A04-Cl_note -t cl -d output-enspire
	../../src/clophfit/fit_titration_global.py D05.dat output-D05 --boot 99
	../../src/clophfit/fit_titration_global.py -t cl --boot 999 B05-20130628-cor.dat output-B05
- Note that ``fit_rpy.py`` did never work (indeed did not use #!/usr/bin/env python).
- Tested dependencies for ``fit_titration`` (without warnings)::

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

- Initial placeholder.

.. _Poetry: https://python-poetry.org
.. _Pyenv: https://github.com/pyenv/pyenv
