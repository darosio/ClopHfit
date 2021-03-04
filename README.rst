|Tests| |PyPI|

ClopHfit
========

-  Cli for fitting macromolecule pH titration or binding assays data
   e.g. fluorescence spectra.
-  Version: “0.3.0a0”


Installation
------------

At this stage few scripts are available in src/clophfit/old.

::

   pyenv install 3.6.15
   poetry install
   poetry run pytest -v


Use
---

fit_titration.py
~~~~~~~~~~~~~~~~

A single script for pK and Cl and various methods w/out bootstraping: 1.
svd 2. bands and 3. single lambda.

   input ← csvtable and note_file

..

   output → pK spK (stdout) and pdf of analysis

To do
^^^^^

-  Bootstrap svd with optimize or lmfit.
-  **Average spectra**
-  Join spectra [‘B’, ‘E’, ‘F’]
-  Compute band integral (or sums)

fit_titration_global.py
~~~~~~~~~~~~~~~~~~~~~~~

A script for fitting tuples (y1, y2) of values for each concentration
(x). It uses lmfit confint and bootstrap.

   input ← x y1 y2 (file)

..

   output → K SA1 SB1 SA2 SB2 , png and correl.png


In global fit the best approach was using lmfit without bootstraping.

Example
^^^^^^^

::

    for i in *.dat; do gfit $i png2 --boot 99 > png2/$i.txt; done


Old tecan todo list
-------------------

I do not know how to unittest

- better fit 400, 485 (also separated) e bootstrap to estimate
  uncertanty

- print sorted output

- buffer correction and report controls e.g. S202N, E2 and V224Q

- dilution correction

- check metadata and report the diff REMEMBER 8.8; dataframe groupby
  per meta_pre, ma anche enspire

- **fit chloride**

- fluorescence is constant? GREAT

- plot data when fit fail and save txt file


Development
-----------

TL;DR
~~~~~

::

   poetry env use 3.9
   poetry install
   pre-commit install
   pre-commit install --hook-type commit-msg

When needed (e.g. API updates)::

   sphinx-apidoc -f -o docs/api/ src/clophfit/

For Jupyter_::

    poetry run python -m ipykernel install --user --name="clophfit"

Development environment
~~~~~~~~~~~~~~~~~~~~~~~

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

* Tests coverage (pytest-cov) configured in .coveragerc.

* Type annotation configured in mypy.ini.

* Commitizen_ also used to bump version::

	cz bump --changelog --prerelease alpha --increment MINOR

  * need one-time initialization::

	  (cz init)

* xdoctest

* sphinx with pydata-sphinx-theme and sphinx-autodoc-typehints. (nbsphinx, sphinxcontrib-plantuml)::

	mkdir docs; cd docs
	sphinx-quickstart
  
  Edit conf.py ["sphinx.ext.autodoc"] and index.rst [e.g. api/modules]::

    sphinx-apidoc -f -o docs/api/ src/clophfit/

* CI/CD to PYPI_ configured in .github/::

	tests.yml
	release.yml

What is missing to modernize_:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- coveralls/Codecov
- automate triggering release to PYPI from github
- readthedocs or ghpages?





.. |Tests| image:: https://github.com/darosio/ClopHfit/workflows/Tests/badge.svg
   :target: https://github.com/darosio/ClopHfit/actions?workflow=Tests
.. |PyPI| image:: https://img.shields.io/pypi/v/ClopHfit.svg
   :target: https://pypi.org/project/ClopHfit/

.. _Commitizen: https://commitizen-tools.github.io/commitizen/

.. _Jupyter: https://jupyter.org/

.. _modernize: https://cjolowicz.github.io/posts/hypermodern-python-06-ci-cd/

.. _PYPI: https://pypi.org/project/clophfit/
