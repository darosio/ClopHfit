# Changelog

## v0.3.18 (2023-01-28)

### CI/CD

- fix CD publish

## v0.3.17 (2023-01-28)

### Fix

- mypy after updates

### Style

- pyproject deps and updates

### Build

- **commitizen**: pre-commit autoupdate
- **ci**: dropped nox and pdm in favor of hatch (#141)
- bump websocket-client from 1.4.2 to 1.5.0 (#140)
- bump nbclassic from 0.4.8 to 0.5.1 (#139)
- bump jupyter-client from 7.4.8 to 8.0.1 (#138)
- bump prometheus-client from 0.15.0 to 0.16.0 (#130)
- bump sphinxcontrib-applehelp from 1.0.2 to 1.0.4 (#129)
- bump jupyter-core from 5.1.0 to 5.1.5 (#134)
- bump nbconvert from 7.2.6 to 7.2.9 (#136)
- bump debugpy from 1.6.4 to 1.6.6 (#131)
- bump sphinx-autodoc-typehints from 1.21.1 to 1.21.8 (#135)
- bump commitizen from 2.39.1 to 2.40.0 (#127)
- bump bleach from 5.0.1 to 6.0.0 (#128)
- bump nbsphinx from 0.8.11 to 0.8.12 (#125)
- bump coverage from 7.0.5 to 7.1.0 (#137)
- bump pandas from 1.5.2 to 1.5.3 (#123)
- bump markupsafe from 2.1.1 to 2.1.2 (#118)

## v0.3.16 (2023-01-16)

### Fix

- clop --version after click updates (#68)

### Style

- correct few typos (#69)

## v0.3.15 (2023-01-13)

### Refactor

- tests and src for pylint
- **py3.11**: include py3.11 in supported versions

## v0.3.14 (2023-01-01)

### CI/CD

- fix CD

## v0.3.13 (2022-12-30)

### Style

- prettify .yml and .md files

### Build

- **deps**: pdm update
- **pre-commit**: autoupdate prettier isort cz pdm shellcheck
- **cz**: to bump and generate changelog

### CI/CD

- automating CD
- test cleaning workflows

## v0.3.12 (2022-12-13)

### CI Changes

- Drop poetry in favor of PDM.
- Taken inspiration from [scikit-hep](https://github.com/scikit-hep/cookie).
- PDM, pre-commit and nox are required on the local system e.g. archlinux.
- Many format checkers are pinned in .pre-commit-config.yaml.

### Fix

- docs: pre-commit install instruction in readme.

### Feat

- docs: pages.yml generates gh-pages.

### Update

- build: manual update poetry deps
- build(deps): bump virtualenv in /.github/workflows (#53)
- build(deps): bump nox from 2022.8.7 to 2022.11.21 in /.github/workflows (#50)
- build(deps): bump numpy from 1.23.4 to 1.23.5 (#52)
- build(deps): bump pandas from 1.5.1 to 1.5.2 (#51)
- build(deps): bump rpy2 from 3.5.5 to 3.5.6 (#49)
- build(deps): bump pip from 22.3 to 22.3.1 in /.github/workflows (#47)

## v0.3.11 (2022-11-04)

### Fix

- docs build and update deps

### Update

- build(deps): bump virtualenv in /.github/workflows (#43)
- build(deps): bump actions/setup-python from 1 to 4 (#42)

## v0.3.10 (2022-11-02)

### Feat

- Add --fit-all flag exporting fit along with data.
- Always export all available data (directory hard coded).

### Refactor

- Avoid code introspection.
- Use fit_routine() in **main**.
- Add Metadata and PlateScheme dataclasses.
- `Map` and `filter` for metadata manipulation.
- TitrationAnalysis inherits from Titration.
- Class method moved out of class into module level function.
- tit and titan .data can be None -> many type ignore required.
- buffer_wells set to components.
- slots and InitVar for lines in Labelblock, but dataclass slots works only in
  py3.10.
- Labelblock is created by a list-of-lines.
- Considered: LabelblockList(OrderedList).

### Update

- build(deps): bump seaborn from 0.12.0 to 0.12.1 (#29)
- build(deps): bump scipy from 1.9.1 to 1.9.3 (#28)
- build(deps): bump pandas from 1.5.0 to 1.5.1 (#27)
- build(deps): bump pip from 22.2.2 to 22.3 in /.github/workflows (#26)
- build(deps): bump pandas-stubs from 1.4.4.220919 to 1.5.0.221012 (#25)
- build(deps): bump numpy from 1.23.3 to 1.23.4 (#24)
- build(deps): bump rpy2 from 3.5.4 to 3.5.5 (#23)
- build(deps): bump poetry from 1.2.0 to 1.2.2 in /.github/workflows (#21)
- build(deps): bump emcee from 3.1.2 to 3.1.3 (#18)
- build(deps): bump codecov/codecov-action from 3.1.0 to 3.1.1 (#16)
- build(deps): bump actions/setup-python from 1 to 4 (#15)

## v0.3.9 (2022-10-02)

### Feat

- LabelblocksGroup metadata now list all values if not identical.
- [ci] Add nox -s bump.
- [ci] Add nox -s coverage and codecov flag in readme.

### Refactor

- Removed temperatures attribute as already present in metadata.
- Slightly improved design for TecanfilesGroup(List[Tecanfile]).
- Prototyped NormalizedLabelblock class.
- Removed of os.path in favor of pathlib.Path

## 0.3.8 (2022-09-24)

### Fix

- missing xlrd issue.

## 0.3.7 (2022-09-23)

### Fix

- `clop prtecan` Install from pipx.

### Refactor

- Path and str are mixing yet it is fixed.
- tests do not use os.chdir() anymore.

## 0.3.6 (2022-09-21)

### Feat

- [docs] Add click-sphinx.

### Fix

- pipx installation?
- Remain an issue in lbG test ph5,cl20 all metadata are list (exception str).

### Refactor

- dataclasses in prtecan.
- string normalization to black default.
- [ci] isort and pyupgrade from pre-commit.
- [ci] safety.
- [docs] README.md and CHANGELOG.md.

## 0.3.5 (2022-09-07)

### Fixed

- Python deps \"\>3.8, \<3.11\" in pyproject.

### Changed

- Update poetry 1.2.0 and notebook.

## 0.3.4 (2022-03-14)

### Fixed

- Read the docs.
- Removed :math: directive from README.rst.

### Added

- Tecan file parser.
- usage.org (exported to .rst) serves as tutorial in docs and includes:
  - liaisan-data
  - new-bootstrap
  - lmfit global
  - emcee (very slow)
- command `clop`.
- <https://pypi.org/project/readme-renderer/> in lint.

### Changed

- Update to python 3.9 and 3.10.
- Update dependencies:

      poetry show --outdated

  with required minor changes in old scripts.

- nox-poetry.
- pandas.rpy (\<=0.19) now lives in rpy2.

### Fixed

- warning for keys_unk set used as index in pd.

## 0.2.1 (2021-11-18)

- Update to python 3.6.
- Py.test for:
  - `fit_titration.py`
  - `fit_titration_global.py`
- lmfit==0.8.3 to prevent `fit-titration_global.py` to fail.
- [\_tmpoutput]{.title-ref} is not deleted; watch out for false positive.

## 0.2.0 (2021-11-14)

- Reference for running older scripts; reproducibility thanks to
  [Poetry](https://python-poetry.org) and
  [Pyenv](https://github.com/pyenv/pyenv):

      LDFLAGS=-L/usr/lib/openssl-1.0/ CFLAGS=-I/usr/include/openssl-1.0/ pyenv install 3.4.10
      ++CONFIGURE_OPTS="--without-ensurepip" pyenv install 3.5.8++
      CC=clang pyenv install 3.5.10
      poetry env use 3.5
      poetry install
      ../../src/clophfit/fit_titration.py Meas/A04\ Cl_A.csv NTT-A04-Cl_note -t cl -d output-enspire
      ../../src/clophfit/fit_titration_global.py D05.dat output-D05 --boot 99
      ../../src/clophfit/fit_titration_global.py -t cl --boot 999 B05-20130628-cor.dat output-B05

- Note that `fit_rpy.py` did never work (indeed did not use #!/usr/bin/env
  python).

- Tested dependencies for `fit_titration` (without warnings):

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

## 0.1.0 (2021-3-4)

- Initial placeholder.
