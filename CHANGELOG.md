<!-- markdownlint-disable MD024 -->
<!-- vale write-good.TooWordy = NO -->

# Changelog

## v0.10.8 (2024-06-25)

### Fix

- hatch bump script
- missing int conversion `svd fit with NaN value`
- svd fit with NaN value
- numpy plugin for mypy

### Build

- **pre-commit**: update hooks (#617)
- **deps**: update mypy requirement from <=1.10.0 to <=1.10.1 (#619)
- **deps**: update scipy requirement from <=1.13.1 to <=1.14.0 (#618)
- **deps**: update sphinx-autodoc-typehints requirement (#616)
- **deps**: update coverage[toml] requirement from <=7.5.3 to <=7.5.4 (#615)
- **deps**: bump pip from 24.0 to 24.1 in /.github/workflows (#614)
- **deps**: update ruff requirement from <=0.4.9 to <=0.4.10 (#613)
- **deps**: update sphinx-autodoc-typehints requirement (#612)
- **pre-commit**: update hooks (#611)
- **deps**: update ruff requirement from <=0.4.8 to <=0.4.9 (#610)
- **deps**: update numpy requirement from <=1.26.4 to <=2.0.0 (#609)
- **deps**: bump codecov/codecov-action from 4.4.1 to 4.5.0 (#608)
- **deps**: update openpyxl requirement from <=3.1.3 to <=3.1.4 (#607)
- **pre-commit**: update hooks (#606)
- **deps**: update xdoctest requirement from <=1.1.4 to <=1.1.5 (#605)
- **deps**: update ruff requirement from <=0.4.7 to <=0.4.8 (#604)
- **deps**: update pytest requirement from <=8.2.1 to <=8.2.2 (#603)
- **deps**: update sphinx-autodoc-typehints requirement (#602)
- **deps**: update pandas-stubs requirement (#601)
- **pre-commit**: update hooks (#600)
- **deps**: update xdoctest requirement from <=1.1.3 to <=1.1.4 (#599)
- **deps**: update ruff requirement from <=0.4.6 to <=0.4.7 (#598)
- **deps**: update openpyxl requirement from <=3.1.2 to <=3.1.3 (#597)
- **deps**: update sympy requirement from <=1.12 to <=1.12.1 (#596)
- **deps**: bump hatch from 1.11.1 to 1.12.0 in /.github/workflows (#595)
- **deps**: update pydata-sphinx-theme requirement (#594)
- **deps**: update ruff requirement from <=0.4.5 to <=0.4.6 (#593)
- **deps**: update coverage[toml] requirement from <=7.5.2 to <=7.5.3 (#592)
- **pre-commit**: update hooks (#591)
- **deps**: update coverage[toml] requirement from <=7.5.1 to <=7.5.2 (#590)
- **deps**: bump hatch from 1.11.0 to 1.11.1 in /.github/workflows (#589)
- **deps**: update sphinxcontrib-plantuml requirement (#588)
- **deps**: update ruff requirement from <=0.4.4 to <=0.4.5 (#587)
- **deps**: update scipy requirement from <=1.13.0 to <=1.13.1 (#586)
- **pre-commit**: update hooks (#583)
- update pytest requirement from <=8.2.0 to <=8.2.1 (#582)
- update commitizen requirement from <=3.25.0 to <=3.26.0 (#581)
- update matplotlib requirement from <=3.8.4 to <=3.9.0 (#580)
- update sphinx-click requirement from <=5.2.2 to <=6.0.0 (#579)
- bump codecov/codecov-action from 4.3.1 to 4.4.0 (#578)
- bump hatch from 1.10.0 to 1.11.0 in /.github/workflows (#577)
- update pandas-stubs requirement (#576)
- update sphinx-click requirement from <=5.1.0 to <=5.2.2 (#575)
- **pre-commit**: update hooks (#574)
- update pre-commit requirement from <=3.7.0 to <=3.7.1 (#573)
- update ruff requirement from <=0.4.3 to <=0.4.4 (#572)
- update nbsphinx requirement from <=0.9.3 to <=0.9.4 (#571)
- **pre-commit**: update hooks (#570)
- update pygments requirement from <=2.17.2 to <=2.18.0 (#569)
- update ruff requirement from <=0.4.2 to <=0.4.3 (#568)
- update coverage[toml] requirement from <=7.5.0 to <=7.5.1 (#567)
- bump hatch from 1.9.7 to 1.10.0 in /.github/workflows (#566)
- update tqdm requirement from <=4.66.2 to <=4.66.4 (#565)
- bump codecov/codecov-action from 4.3.0 to 4.3.1 (#564)
- update commitizen requirement from <=3.24.0 to <=3.25.0 (#563)
- **pre-commit**: update hooks (#562)
- update pytest requirement from <=8.1.1 to <=8.2.0 (#561)
- update ruff requirement from <=0.4.1 to <=0.4.2 (#560)
- bump hatch from 1.9.4 to 1.9.7 in /.github/workflows (#559)
- update mypy requirement from <=1.9.0 to <=1.10.0 (#558)
- update coverage[toml] requirement from <=7.4.4 to <=7.5.0 (#557)
- update emcee requirement from <=3.1.5 to <=3.1.6 (#555)
- update ruff requirement from <=0.4.0 to <=0.4.1 (#554)
- update lmfit requirement from <=1.3.0 to <=1.3.1 (#553)
- update sphinx requirement from <=7.3.6 to <=7.3.7 (#552)
- update commitizen requirement from <=3.22.0 to <=3.24.0 (#551)
- update ruff requirement from <=0.3.7 to <=0.4.0 (#550)
- update sphinx-autodoc-typehints requirement (#549)
- update sphinx requirement from <=7.3.4 to <=7.3.6 (#548)
- update sphinx requirement from <=7.2.6 to <=7.3.4 (#547)
- update emcee requirement from <=3.1.4 to <=3.1.5 (#546)
- bump codecov/codecov-action from 4.2.0 to 4.3.0 (#539)
- **fix**: prettier in pre-commit

### CI/CD

- Merge docs into ci.yml

### chore

- Modify pre-commit update message
- update pre-commit hooks (#556)
- **build**: Adjust pre-commit config

## v0.10.7 (2024-04-16)

### Fix

- **ci**: Use token to upload to TestPyPI
- The spectra glob emcee progress to False

### Docs

- Correct tutorials using pymc with cov_matrix
- Fix prenspire tutorial
- Fix prtecan tutorial
- Fix LMfit tutorial
- Fix LMfit tutorial

### Build

- update ruff requirement from <=0.3.6 to <=0.3.7 (#544)
- update commitizen requirement from <=3.21.3 to <=3.22.0 (#543)
- update ruff requirement from <=0.3.5 to <=0.3.6 (#542)
- update pandas requirement from <=2.2.1 to <=2.2.2 (#541)
- update sphinx-autodoc-typehints requirement from <=2.0.0 to <=2.0.1 (#540)
- **docs**: Update to the official upload-artifact action
- **style**: Fix few ruff linting errors
- Drop python 3.9; add 3.12
- update scipy requirement from <=1.12.0 to <=1.13.0
- update arviz requirement from <=0.17.1 to <=0.18.0 (#537)
- update lmfit requirement from <=1.2.2 to <=1.3.0 (#536)
- update matplotlib requirement from <=3.8.3 to <=3.8.4 (#535)
- bump codecov/codecov-action from 4.1.1 to 4.2.0 (#534)
- update ruff requirement from <=0.3.4 to <=0.3.5 (#532)
- update commitizen requirement from <=3.20.0 to <=3.21.3 (#530)
- bump actions/configure-pages from 4 to 5 (#529)

### Refactor

- **build**: Adopt pydoclint and drop darglint
- **build**: From black to ruff format; tutorials

### chore

- update pre-commit hooks (#545)
- update pre-commit hooks (#538)
- update pre-commit hooks (#531)

## v0.10.6 (2024-03-28)

### Fix

- **ci**: Add codecov token
- commitizen version v3.13.0 from 3.12.0
- avoid warnings from some pytest

### Docs

- Adopt github-pages-deploy-action

### Build

- bump codecov/codecov-action from 4.1.0 to 4.1.1 (#528)
- update pre-commit requirement from <=3.6.2 to <=3.7.0 (#526)
- update ruff requirement from <=0.3.3 to <=0.3.4 (#525)
- update commitizen requirement from <=3.18.4 to <=3.20.0 (#524)
- update ruff requirement from <=0.3.2 to <=0.3.3 (#522)
- update pandas-stubs requirement (#521)
- update coverage[toml] requirement from <=7.4.3 to <=7.4.4 (#520)
- update commitizen requirement from <=3.18.3 to <=3.18.4 (#519)
- update arviz requirement from <=0.17.0 to <=0.17.1 (#518)
- bump hatch from 1.9.3 to 1.9.4 in /.github/workflows (#516)
- update commitizen requirement from <=3.18.0 to <=3.18.3 (#515)
- update mypy requirement from <=1.8.0 to <=1.9.0 (#514)
- update pytest requirement from <=8.1.0 to <=8.1.1 (#513)
- update ruff requirement from <=0.3.1 to <=0.3.2 (#512)
- update sphinxcontrib-plantuml requirement from <=0.28 to <=0.29 (#511)
- update commitizen requirement from <=3.17.0 to <=3.18.0 (#510)
- update ruff requirement from <=0.3.0 to <=0.3.1 (#509)
- update commitizen requirement from <=3.16.0 to <=3.17.0 (#508)
- update pytest requirement from <=8.0.2 to <=8.1.0 (#507)
- update ruff requirement from <=0.2.2 to <=0.3.0 (#506)
- bump codecov/codecov-action from 4.0.2 to 4.1.0 (#505)
- update commitizen requirement from <=3.15.0 to <=3.16.0 (#504)
- bump codecov/codecov-action from 4.0.1 to 4.0.2 (#503)
- update pytest requirement from <=8.0.1 to <=8.0.2 (#502)
- update pandas requirement from <=2.2.0 to <=2.2.1 (#501)
- update coverage[toml] requirement from <=7.4.2 to <=7.4.3 (#500)
- update sphinxcontrib-plantuml requirement from <=0.27 to <=0.28 (#499)
- update coverage[toml] requirement from <=7.4.1 to <=7.4.2 (#498)
- update ruff requirement from <=0.2.1 to <=0.2.2 (#496)
- update pytest requirement from <=8.0.0 to <=8.0.1 (#495)
- update pre-commit requirement from <=3.6.1 to <=3.6.2 (#494)
- update commitizen requirement from <=3.14.1 to <=3.15.0 (#493)
- update pandas-stubs requirement (#492)
- update matplotlib requirement from <=3.8.2 to <=3.8.3 (#491)
- update tqdm requirement from <=4.66.1 to <=4.66.2 (#489)
- update pre-commit requirement from <=3.6.0 to <=3.6.1 (#488)
- update sphinx-autodoc-typehints requirement (#487)
- update ruff requirement from <=0.2.0 to <=0.2.1 (#486)
- update numpy requirement from <=1.26.3 to <=1.26.4 (#485)
- bump pip from 23.3.2 to 24.0 in /.github/workflows (#484)
- update commitizen requirement from <=3.14.0 to <=3.14.1 (#483)
- bump codecov/codecov-action from 3.1.6 to 4.0.1 (#482)
- update commitizen requirement from <=3.13.0 to <=3.14.0 (#480)
- update ruff requirement from <=0.1.15 to <=0.2.0 (#481)
- bump codecov/codecov-action from 3.1.5 to 3.1.6 (#478)
- update xdoctest requirement from <=1.1.2 to <=1.1.3 (#477)
- update ruff requirement from <=0.1.14 to <=0.1.15 (#476)
- update pytest requirement from <=7.4.4 to <=8.0.0 (#475)
- update coverage[toml] requirement from <=7.4.0 to <=7.4.1 (#474)
- update sphinx-autodoc-typehints requirement (#473)
- bump codecov/codecov-action from 3.1.4 to 3.1.5 (#472)
- update seaborn requirement from <=0.13.1 to <=0.13.2 (#471)
- bump hatch from 1.9.2 to 1.9.3 in /.github/workflows (#470)
- bump hatch from 1.9.1 to 1.9.2 in /.github/workflows (#469)
- update pandas requirement from <=2.1.4 to <=2.2.0 (#467)
- update ruff requirement from <=0.1.13 to <=0.1.14 (#468)
- update scipy requirement from <=1.11.4 to <=1.12.0 (#466)
- update pydata-sphinx-theme requirement from <=0.15.1 to <=0.15.2 (#465)
- update ruff requirement from <=0.1.12 to <=0.1.13 (#464)
- update ruff requirement from <=0.1.11 to <=0.1.12 (#463)
- update autodocsumm requirement from <=0.2.11 to <=0.2.12 (#462)
- update pydata-sphinx-theme requirement from <=0.14.4 to <=0.15.1 (#461)
- update numpy requirement from <=1.26.2 to <=1.26.3 (#460)
- update ruff requirement from <=0.1.9 to <=0.1.11 (#459)
- update seaborn requirement from <=0.13.0 to <=0.13.1 (#458)
- update pytest requirement from <=7.4.3 to <=7.4.4 (#457)
- update arviz requirement from <=0.16.1 to <=0.17.0 (#456)
- update coverage[toml] requirement from <=7.3.4 to <=7.4.0 (#455)
- update pandas-stubs requirement (#454)
- bump hatch from 1.9.0 to 1.9.1 in /.github/workflows (#453)
- update ruff requirement from <=0.1.8 to <=0.1.9 (#452)
- update mypy requirement from <=1.7.1 to <=1.8.0 (#451)
- update coverage[toml] requirement from <=7.3.3 to <=7.3.4 (#450)
- bump hatch from 1.8.1 to 1.9.0 in /.github/workflows (#446)
- update pandas-stubs requirement (#445)
- bump pip from 23.3.1 to 23.3.2 in /.github/workflows (#444)
- bump actions/download-artifact from 3 to 4 (#443)
- update coverage[toml] requirement from <=7.3.2 to <=7.3.3 (#442)
- bump hatch from 1.8.0 to 1.8.1 in /.github/workflows (#441)
- update ruff requirement from <=0.1.7 to <=0.1.8 (#440)
- bump hatch from 1.7.0 to 1.8.0 in /.github/workflows (#439)
- update pandas requirement from <=2.1.3 to <=2.1.4 (#438)
- update pre-commit requirement from <=3.5.0 to <=3.6.0 (#437)
- bump actions/setup-python from 4 to 5 (#436)
- bump actions/deploy-pages from 2 to 3 (#435)
- bump actions/configure-pages from 2 to 4 (#434)
- update ruff requirement from <=0.1.6 to <=0.1.7 (#433)
- update mypy requirement from <=1.7 to <=1.7.1 (#432)
- update commitizen requirement from <3.12.1 to <3.13.1
- bump pydata-sphinx-theme from 0.14.3 to 0.14.4 (#429)
- update pygments requirement from <2.17.2 to <2.17.3 (#428)
- bump sphinx-click from 5.0.1 to 5.1.0 (#427)
- update pygments requirement from <2.16.2 to <2.17.2 (#425)
- bump ruff from 0.1.5 to 0.1.6 (#424)
- update matplotlib requirement from <3.8.2 to <3.8.3 (#423)
- update scipy requirement from <1.11.4 to <1.11.5 (#422)
- update numpy requirement from <1.26.2 to <1.26.3 (#420)
- bump sphinx-autodoc-typehints from 1.25.0 to 1.25.2 (#419)
- update mypy requirement from <1.7 to <1.8 (#418)
- update pandas requirement from <2.1.3 to <2.1.4 (#417)
- bump sphinx-autodoc-typehints from 1.24.1 to 1.25.0 (#416)
- bump ruff from 0.1.4 to 0.1.5 (#415)
- bump ruff from 0.1.3 to 0.1.4 (#413)
- bump sphinx-autodoc-typehints from 1.24.0 to 1.24.1 (#412)
- update matplotlib requirement from <3.8.1 to <3.8.2 (#411)
- bump sphinxcontrib-plantuml from 0.26 to 0.27 (#410)
- bump pydata-sphinx-theme from 0.14.2 to 0.14.3 (#409)
- update pandas requirement from <2.1.2 to <2.1.3 (#407)
- bump ruff from 0.1.2 to 0.1.3 (#406)
- update xdoctest requirement from <1.1.2 to <1.1.3 (#405)
- bump pydata-sphinx-theme from 0.14.1 to 0.14.2 (#404)
- bump ruff from 0.1.1 to 0.1.2 (#403)
- update pytest requirement from <7.4.3 to <7.4.4 (#402)
- bump pip from 23.3 to 23.3.1 in /.github/workflows (#400)
- update commitizen requirement from <3.10.2 to <3.12.1 (#398)
- bump ruff from 0.1.0 to 0.1.1 (#399)
- bump ruff from 0.0.292 to 0.1.0 (#397)
- bump pip from 23.2.1 to 23.3 in /.github/workflows (#395)
- bump pre-commit from 3.4.0 to 3.5.0 (#394)
- update numpy requirement from <1.26.1 to <1.26.2 (#393)
- update commitizen requirement from <3.10.1 to <3.10.2 (#392)
- update mypy requirement from <1.6 to <1.7 (#391)
- update coverage[toml] requirement from <7.3.2 to <7.3.3 (#388)
- bump ruff from 0.0.291 to 0.0.292 (#387)
- update seaborn requirement from <0.12.3 to <0.13.1 (#386)
- bump pandas-stubs from 2.0.3.230814 to 2.1.1.230928 (#385)
- update scipy requirement from <1.11.3 to <1.11.4 (#384)
- bump ruff from 0.0.290 to 0.0.291 (#382)
- update commitizen requirement from <3.9.1 to <3.10.1 (#381)
- update pandas requirement from <2.1.1 to <2.1.2 (#380)
- bump pydata-sphinx-theme from 0.14.0 to 0.14.1 (#379)

### CI/CD

- Sync dependabot update time
- Docs from gh actions

### Refactor

- **build**: Run pre-commit autoupdate

### chore

- update pre-commit hooks (#527)
- update pre-commit hooks (#523)
- update pre-commit hooks (#517)
- update pre-commit hooks (#497)
- update pre-commit hooks (#490)
- update pre-commit hooks (#426)
- update pre-commit hooks (#421)
- update pre-commit hooks (#414)
- update pre-commit hooks (#408)
- update pre-commit hooks (#401)
- update pre-commit hooks (#396)
- update pre-commit hooks (#390)
- update pre-commit hooks (#389)
- update pre-commit hooks (#383)
- update pre-commit hooks (#378)

## v0.10.5 (2023-09-18)

### Docs

- Improve README.md

### Build

- update matplotlib requirement from <3.7.4 to <3.8.1
- update numpy requirement from <1.25.3 to <1.26.1 (#377)
- bump pydata-sphinx-theme from 0.13.3 to 0.14.0 (#376)
- update commitizen requirement from <3.8.3 to <3.9.1 (#374)
- bump ruff from 0.0.289 to 0.0.290 (#373)
- bump sphinx from 7.2.5 to 7.2.6 (#372)
- bump ruff from 0.0.288 to 0.0.289 (#371)
- bump ruff from 0.0.287 to 0.0.288 (#369)
- update matplotlib requirement from <3.7.3 to <3.7.4 (#368)
- update commitizen requirement from <3.8.1 to <3.8.3 (#367)

### Refactor

- Fix type annotations of matplotlib-3.8.1

### chore

- update pre-commit hooks (#370)

## v0.10.4 (2023-09-08)

### Fix

- Remove insane `min=0` for S0

### Docs

- Fix readme usage instruction

### Test

- Fix test_fit_titration_glob missing a parametrized condition test

### Build

- update pytest requirement from <7.4.2 to <7.4.3 (#366)

## v0.10.3 (2023-09-07)

### Fix

- **note_to_csv**: Ensure unix EOL

### Docs

- Add usage instruction for `ppr enspire` and `note_to_csv`

### Test

- For note_to_csv

### Build

- update coverage[toml] requirement from <7.3.1 to <7.3.2 (#365)
- update commitizen requirement from <3.7.2 to <3.8.1 (#364)

## v0.10.2 (2023-09-05)

### Fix

- **docs**: click cli

### Build

- bump actions/checkout from 3 to 4 (#363)
- update commitizen requirement from <3.7.1 to <3.7.2 (#361)
- bump pre-commit from 3.3.3 to 3.4.0 (#360)
- bump ruff from 0.0.286 to 0.0.287 (#359)
- update pytest requirement from <7.4.1 to <7.4.2 (#358)
- update pandas requirement from <2.0.4 to <2.1.1 (#357)
- bump sphinx from 7.2.4 to 7.2.5 (#356)
- bump sphinx from 7.2.3 to 7.2.4 (#354)
- update commitizen requirement from <3.6.1 to <3.7.1 (#353)
- bump sphinxcontrib-plantuml from 0.25 to 0.26 (#352)
- bump nbsphinx from 0.9.2 to 0.9.3 (#351)
- bump ruff from 0.0.285 to 0.0.286 (#350)
- bump sphinx from 7.2.2 to 7.2.3 (#349)
- bump sphinx-click from 4.4.0 to 5.0.1 (#346)
- update click requirement from <8.1.7 to <8.1.8 (#347)
- bump ruff from 0.0.284 to 0.0.285 (#345)
- update scipy requirement from <1.11.2 to <1.11.3 (#344)
- bump sphinx from 7.1.2 to 7.2.2 (#343)

### chore

- update pre-commit hooks (#362)
- update pre-commit hooks (#355)
- update pre-commit hooks (#348)

## v0.10.1 (2023-08-16)

### Fix

- (#342)
- pd.Index generic (#341)
- Refactor to accommodate ruff update to 0.0.280

### Docs

- Add pdf to RtD
- Add all formats to RtD

### Build

- bump pandas-stubs from 2.0.2.230605 to 2.0.3.230814
- bump pandas-stubs from 2.0.2.230605 to 2.0.3.230814
- update coverage[toml] requirement from <7.2.8 to <7.3.1 (#340)
- update mypy requirement from <1.5 to <1.6 (#339)
- update tqdm requirement from <4.65.3 to <4.66.2 (#338)
- bump ruff from 0.0.283 to 0.0.284 (#337)
- update tqdm requirement from <4.65.1 to <4.65.3 (#335)
- bump ruff from 0.0.282 to 0.0.283 (#334)
- update pygments requirement from <2.15.2 to <2.16.2 (#332)
- prettier --prose-wrap=preserve
- bump sphinx from 7.1.1 to 7.1.2 (#331)
- bump ruff from 0.0.281 to 0.0.282 (#330)
- update commitizen requirement from <3.5.5 to <3.6.1 (#327)
- update numpy requirement from <1.25.2 to <1.25.3 (#328)
- bump ruff from 0.0.280 to 0.0.281 (#326)
- update commitizen requirement from <3.5.4 to <3.5.5 (#325)
- bump sphinx from 7.0.1 to 7.1.1 (#324)
- bump pip from 23.2 to 23.2.1 in /.github/workflows (#321)
- bump ruff from 0.0.278 to 0.0.280 (#320)
- bump sphinx-autodoc-typehints from 1.23.3 to 1.24.0 (#319)
- update click requirement from <8.1.6 to <8.1.7 (#318)

### chore

- update pre-commit hooks
- update pre-commit hooks
- update pre-commit hooks (#333)
- ruff 0.281 keys() refactor
- update pre-commit hooks
- update pre-commit hooks

## v0.10.0 (2023-07-19)

### Feat

- Add spec to `fit-titration` dropping fit_titration_old
- Add fit-titration glob dropping fit_titration_glob_old

### Docs

- Update ipynb

### Test

- Update coverage run

### Build

- update arviz requirement from <0.16.1 to <0.16.2 (#317)
- bump pre-commit from 3.0.1 to 3.3.3 (#315)
- Eliminate pip-deepfreeze
- Relocate 'docs' from 'envs.docs.scripts' to 'default.scripts'
- Relocate coverage into tests
- Relocate 'lint' from 'envs.lint.scripts' to 'default.scripts'
- bump pip from 23.1.2 to 23.2 in /.github/workflows (#314)
- update commitizen requirement from <3.5.3 to <3.5.4 (#313)
- Eliminate dependency on typeguard

### Refactor

- Simplify plot_emcee fix options in fit-titration glob
- Reorganize `fit-titration glob` test data files
- Add FitKwargs into TitrationAnalysis to streamline fit() method
- Remove type ignore in PlateScheme also add PlateScheme test

### chore

- update pre-commit hooks (#316)

## v0.9.1 (2023-07-15)

### Docs

- Fix zenodo badge to auto-update

### Refactor

- Remove silencer noqa for merge_md and norm metadata

## v0.9.0 (2023-07-15)

### Feat

- Change tecan and enspire as subcommands of `ppr`
- pr.tecan exports png for each well
- **prtecan**: new fit; sort results; add export_png
- New glob-emcee fitting; add note_to_csv

### Fix

- plots of critical cl fit
- tit_cl K>epsilon; refactor is_ph as Titration attribute
- fit_tecan in main cli

### Docs

- Update tutorials and (partially) UML
- Update zotero DOI
- Add click doc for note_to_csv

### Build

- update click requirement from <8.1.5 to <8.1.6 (#311)
- update lmfit requirement from <1.2.2 to <1.2.3 (#312)
- update arviz requirement from <0.15.2 to <0.16.1 (#310)
- bump ruff from 0.0.277 to 0.0.278 (#309)
- bump actions/upload-pages-artifact from 1 to 2 (#308)
- update numpy requirement from <1.25.1 to <1.25.2 (#306)
- update click requirement from <8.1.4 to <8.1.5 (#304)
- update matplotlib requirement from <3.7.2 to <3.7.3 (#303)
- bump ruff from 0.0.275 to 0.0.277 (#301)

### Refactor

- Rename TitrationAnalysis.results and .result_dfs
- Use `is_ph` instead of `kind`
- Removes confint
- **cli**: pr.tecan and pr.enspire
- Drop py3.8

### chore

- Remove redundant NaN filtering
- update pre-commit hooks (#307)
- update pre-commit hooks (#300)

## v0.8.1 (2023-07-03)

### Docs

- fix pointer to changelog in project.toml (#274)

### Build

- update pandas requirement from <2.0.3 to <2.0.4 (#297)
- update scipy requirement from <1.11.1 to <1.11.2 (#296)
- bump sphinx-autodoc-typehints from 1.23.2 to 1.23.3 (#294)
- update pytest requirement from <7.3.3 to <7.4.1 (#292)
- update commitizen requirement from <3.3.1 to <3.5.3 (#291)
- update scipy requirement from <1.10.2 to <1.11.1 (#293)
- bump ruff from 0.0.272 to 0.0.275 (#289)
- update mypy requirement from <1.4 to <1.5 (#286)
- update numpy requirement from <1.24.4 to <1.25.1 (#284)
- bump sphinx-autodoc-typehints from 1.23.0 to 1.23.2 (#283)
- update commitizen requirement from <3.2.3 to <3.3.1 (#281)
- update pytest requirement from <7.3.2 to <7.3.3 (#279)
- bump ruff from 0.0.270 to 0.0.272 (#278)
- bump pandas-stubs from 2.0.1.230501 to 2.0.2.230605 (#276)

### Refactor

- **docs**: Drop rpy; convert into LMfit.ipynb

### chore

- update pre-commit hooks (#295)
- update pre-commit hooks (#285)
- update pre-commit hooks (#280)
- update pre-commit hooks (#275)

## v0.8.0 (2023-06-05)

### Feat

- **prenspire**: build and fit single-and-global SVD-or-band titrations
- **clop**: fit_titration for old note
- **clop**: `fit titration global old` and `fit titration old` as cli command
- add \_\_version\_\_ for output into "Meas-{version}"

### Test

- **clop**: add test for new fit_titration cli

### Docs

- Add instruction to install auto completion

### Build

- remove pygrep-hook: python-check-blanket-noqa and python-no-eval
- update coverage[toml] requirement from <7.2.7 to <7.2.8 (#272)
- update pandas requirement from <2.0.2 to <2.0.3 (#271)
- bump nbsphinx from 0.9.1 to 0.9.2 (#269)
- bump ruff from 0.0.269 to 0.0.270 (#268)
- update coverage[toml] requirement from <7.2.6 to <7.2.7 (#267)
- update rpy2 requirement from <3.5.12 to <3.5.13 (#265)
- bump ruff from 0.0.267 to 0.0.269 (#264)

### Refactor

- **prenspire**: new Note (titration build) and fit_titration cmd; keep the old
  ExpNote for compatibility with old note files.
- **prenspire**: analyze_spectra_glob for SVD or multiple bands
- **prenspire**: SVD spectra expand (with helpers functions) on old
  `fit_titration` script

### chore

- update pre-commit hooks (#270)
- update pre-commit hooks (#266)

## v0.7.1 (2023-05-18)

### Fix

- **docs**: README links to pages

### Docs

- Enhance the alignment between the changelog and tags

### Build

- bump codecov/codecov-action from 3.1.3 to 3.1.4 (#262)

### Refactor

- **enspire**: Adopt dataclass with helper functions

### chore

- update pre-commit hooks (#261)

## v0.7.0 (2023-05-15)

### Feat

- **prtecan**: Typecheck PlateScheme and improve docs

### Test

- Refactor fixtures for prtecan

### Build

- bump ruff from 0.0.265 to 0.0.267 (#259)
- bump sphinx from 7.0.0 to 7.0.1 (#258)
- bump typeguard from 4.0.0rc6 to 4.0.0 (#257)

## v0.6.0 (2023-05-12)

### Feat

- **prenspire**: Read platemap othen than 96-well

### Build

- update commitizen requirement from <3.2.2 to <3.2.3 (#256)

## v0.5.4 (2023-05-11)

### Docs

- Reorganize readme sphinx etc.

### Build

- update mypy requirement from <1.3 to <1.4 (#253)
- update sympy requirement from <1.11.2 to <1.13.0 (#251)

## v0.5.3 (2023-05-09)

### Docs

- updated docs/README.md

## v0.5.2 (2023-05-09)

### CI/CD

- Unify CI/CD into a single workflow

## v0.5.1 (2023-05-09)

### Fix

- **CI/CD**: release to PyPI

## v0.5.0 (2023-05-09)

### Feat

- Add prenspire to parse PerkinElmer files

### Build

- bump ruff from 0.0.264 to 0.0.265 (#246)
- bump typeguard from 4.0.0rc5 to 4.0.0rc6 (#245)
- update commitizen requirement from <3.2.1 to <3.2.2 (#244)

### chore

- update pre-commit hooks (#247)
- default to python 3.11

## v0.4.10 (2023-05-03)

### Refactor

- Labelblock KEYS

## v0.4.9 (2023-05-03)

### Build

- bump sphinx from 6.2.1 to 7.0.0 (#236)
- bump ruff from 0.0.263 to 0.0.264 (#243)
- bump autodocsumm from 0.2.10 to 0.2.11 (#242)
- update lmfit requirement from <1.2.1 to <1.2.2 (#241)

### Refactor

- update typeguard to 4.0.0rc5

## v0.4.8 (2023-05-02)

### Docs

- Lint README.md with markdownlint-cli

### Build

- update lmfit requirement from <1.1.1 to <1.2.1 (#213)
- remove pyupgrade as ruff "UP" fully replaces it
- bump pandas-stubs from 2.0.0.230412 to 2.0.1.230501 (#240)
- update commitizen requirement from <3.1.1 to <3.2.1 (#239)
- update coverage[toml] requirement from <7.2.4 to <7.2.6 (#238)
- bump pip from 23.1.1 to 23.1.2 in /.github/workflows (#235)
- update commitizen requirement from <2.42.2 to <3.1.1 (#234)
- bump sphinx from 6.1.3 to 6.2.1 (#233)
- bump ruff from 0.0.262 to 0.0.263 (#232)
- update pandas requirement from <2.0.1 to <2.0.2 (#231)
- bump pip from 23.1 to 23.1.1 in /.github/workflows (#229)
- update numpy requirement from <1.24.3 to <1.24.4 (#227)
- bump codecov/codecov-action from 3.1.2 to 3.1.3 (#225)
- update corner requirement from <2.2.2 to <2.2.3 (#211)
- update pygments requirement from <2.14.1 to <2.15.2 (#223)
- bump pip from 23.0.1 to 23.1 in /.github/workflows (#222)
- update pytest requirement from <7.2.3 to <7.3.2 (#220)
- bump pandas-stubs from 1.5.3.230321 to 2.0.0.230412 (#218)
- bump ruff from 0.0.260 to 0.0.262 (#224)
- update rpy2 requirement from <3.5.11 to <3.5.12 (#221)
- bump sphinx-autodoc-typehints from 1.22 to 1.23.0 (#219)
- bump codecov/codecov-action from 3.1.1 to 3.1.2 (#217)
- bump pip-deepfreeze from 1.1.0 to 1.2.0 in /.github/workflows (#216)
- update mypy requirement from <1.2 to <1.3 (#210)
- update coverage[toml] requirement from <7.2.3 to <7.2.4 (#209)
- bump hatch from 1.6.3 to 1.7.0 in /.github/workflows (#207)
- update pandas requirement from <1.5.4 to <2.0.1 (#205)
- bump pydata-sphinx-theme from 0.13.1 to 0.13.3 (#204)
- bump ruff from 0.0.259 to 0.0.260 (#203)
- bump ruff from 0.0.258 to 0.0.259 (#201)
- bump ruff from 0.0.257 to 0.0.258 (#199)
- bump pandas-stubs from 1.5.3.230304 to 1.5.3.230321 (#198)

### CI/CD

- skip tests when only docs are changed

### chore

- update pre-commit hooks (#230)
- update pre-commit hooks (#214)
- update pre-commit hooks (#206)

## v0.4.7 (2023-03-21)

### Docs

- Update Readme

### Build

- **docs**: fix RTD

## v0.4.6 (2023-03-21)

### Style

- yaml with prettier through apheleia

### Build

- bump actions/deploy-pages from 1 to 2 (#196)
- update coverage[toml] requirement from <7.2.2 to <7.2.3 (#193)
- bump nbsphinx from 0.8.12 to 0.9.1 (#191)
- update openpyxl requirement from <3.1.2 to <3.1.3 (#189)
- bump myst-parser from 0.19.1 to 1.0.0 (#187)
- update mypy requirement from <1.1 to <1.2 (#186)
- bump pydata-sphinx-theme from 0.13.0 to 0.13.1 (#185)
- bump pandas-stubs from 1.5.3.230227 to 1.5.3.230304 (#184)
- update rpy2 requirement from <3.5.10 to <3.5.11 (#183)
- update tqdm requirement from <4.64.2 to <4.65.1 (#182)
- update pytest requirement from <7.2.2 to <7.2.3 (#181)
- bump sphinxcontrib-plantuml from 0.24.1 to 0.25 (#180)
- update matplotlib requirement from <3.7.1 to <3.7.2 (#179)
- bump myst-parser from 0.18.1 to 0.19.1 (#178)
- bump pydata-sphinx-theme from 0.12.0 to 0.13.0 (#176)
- bump pandas-stubs from 1.5.3.230214 to 1.5.3.230227 (#175)
- update rpy2 requirement from <3.5.9 to <3.5.10 (#174)
- update coverage[toml] requirement from <7.2.1 to <7.2.2 (#173)
- update commitizen requirement from <2.42.1 to <2.42.2 (#172)
- update coverage[toml] requirement from <7.1.1 to <7.2.1 (#171)
- bump pip from 23.0 to 23.0.1 in /.github/workflows (#170)
- update scipy requirement from <1.10.1 to <1.10.2 (#169)
- bump pandas-stubs from 1.5.3.230203 to 1.5.3.230214 (#168)
- update openpyxl requirement from <3.1.1 to <3.1.2 (#167)
- update matplotlib requirement from <3.6.4 to <3.7.1 (#166)
- bump pip-deepfreeze from 1.0.0 to 1.1.0 in /.github/workflows (#165)
- update commitizen requirement from <2.41.1 to <2.42.1 (#164)
- **style**: more pre-commit checks and formatting
- update commitizen requirement from <2.40.1 to <2.41.1
- update mypy requirement from <0.992 to <1.1 (#162)
- pylsp-ruff,mypy as dev deps
- **lint**: updt
- update rpy2 requirement from <3.5.8 to <3.5.9 (#160)
- bump pandas-stubs from 1.5.2.230105 to 1.5.3.230203 (#159)
- update numpy requirement from <1.24.2 to <1.24.3 (#158)

### Refactor

- **build**: pre-commit autoupdate
- remove unneeded symbols in prtecan.\_\_all\_\_
- separate fitting from prtecan and define \_\_all\_\_

### chore

- update pre-commit hooks (#197)

## v0.4.5 (2023-02-03)

### Fix

- docs update

## v0.4.4 (2023-02-03)

### Fix

- **docs**: emcee sections

## v0.4.3 (2023-02-03)

### Build

- update ipykernel requirement from <6.20.3 to <6.21.2 (#156)
- **docs**: use stable-python 3.11; update docs and style cleaning

## v0.4.2 (2023-02-01)

### Fix

- CD testpypi

## v0.4.1 (2023-02-01)

### Fix

- **docs**: CI badge

### Docs

- venv with hatch and pre-commit init instruction

### Build

- bumping lmfit from 1.0.3 to 1.1.0
- update xdoctest requirement from <1.1.1 to <1.1.2 (#155)
- update openpyxl requirement from <3.0.11 to <3.1.1 (#154)
- update ipython requirement from <8.8.1 to <8.9.1 (#152)
- update emcee requirement from <3.1.4 to <3.1.5 (#151)
- bump pip from 22.3.1 to 23.0 in /.github/workflows (#148)
- dropped pip-df
- **deps**: bump numpy 1.24.1

## v0.4.0 (2023-01-30)

### Feat

- **build**: ruff (#144)

## v0.3.20 (2023-01-28)

### Build

- bump jupyter-client from 7.4.9 to 8.0.1 (#143)

### CI/CD

- fix rtd and sudden lint failing isort installation

## v0.3.19 (2023-01-28)

### Fix

- build and rtd

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

## v0.3.8 (2022-09-24)

### Fix

- missing xlrd issue.

## v0.3.7 (2022-09-23)

### Fix

- `clop prtecan` Install from pipx.

### Refactor

- Path and str are mixing.
- tests do not use os.chdir() anymore.

## v0.3.6 (2022-09-21)

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

## v0.3.5 (2022-09-11)

### Changed

- Update poetry 1.2.0 and notebook.

## v0.3.4 (2022-09-07)

### Fixed

- Python deps \"\>3.8, \<3.11\" in pyproject.

### Changed

- Update poetry 1.2.0 and notebook.

## v0.3.3 (2022-05-27)

- Update deps for notebooks.

## v0.3.2 (2022-03-15)

- Add zenodo.

## v0.3.1 (2022-03-14)

### Fixed

- Removed :math: directive from README.rst and plantuml.

## v0.3.0 (2022-03-14)

### Fixed

- warning for keys_unk set used as index in pd.

### Added

- Tecan file parser.
- Import fit_titrations scripts with many test data.
- usage.org (exported to .rst) serves as tutorial in docs and includes:
  - liaisan-data
  - new-bootstrap
  - lmfit global
  - emcee (extremely slow)
- command `clop`.
- (docs) usage.org.
- Adopt pre-commit, poetry and nox.
- <https://pypi.org/project/readme-renderer/> in lint.
- (build) Github actions.

### Changed

- Update to python 3.9 and 3.10.
- Update dependencies:

      poetry show --outdated

  with required minor changes in old scripts.

- nox-poetry.
- pandas.rpy (\<=0.19) now lives in rpy2.

## v0.2.1 (2021-11-18)

- Update to python 3.6.
- Py.test for:
  - `fit_titration.py`
  - `fit_titration_global.py`
- lmfit==0.8.3 to prevent `fit-titration_global.py` to fail.
- [\_tmpoutput]{.title-ref} is not deleted; watch out for false positive.

## v0.2.0 (2021-11-14)

- Reference for running older scripts; reproducibility thanks to
  [Poetry](https://python-poetry.org) and
  [Pyenv](https://github.com/pyenv/pyenv):

      LDFLAGS=-L/usr/lib/openssl-1.0/ CFLAGS=-I/usr/include/openssl-1.0/ \
        pyenv install 3.4.10
      ++CONFIGURE_OPTS="--without-ensurepip" pyenv install 3.5.8++
      CC=clang pyenv install 3.5.10
      poetry env use 3.5
      poetry install
      .. fit_titration.py Meas/A04\ Cl_A.csv NTT-A04-Cl_note -t cl -d output-enspire
      .. fit_titration_global.py D05.dat output-D05 --boot 99
      .. fit_titration_global.py -t cl --boot 999 B05-20130628-cor.dat output-B05

- Note that `fit_rpy.py` did never work (indeed did not use #!/usr/bin/env
  python).

- Tested dependencies for `fit_titration` (without warnings):

      cycler          0.10.0 Composable style cycles
      lmfit           0.8.3  Least-Squares Minimization with Bounds and Constraints
      matplotlib      1.5.3  Python plotting package
      numpy           1.10.4 NumPy: array processing for numbers, strings, …
      pandas          0.18.1 Powerful data structures for data analysis, …
      pyparsing       2.4.7  Python parsing module
      python-dateutil 2.8.1  Extensions to the standard Python datetime module
      pytz            2021.3 World timezone definitions, modern and historical
      rpy2            2.3.10 Python interface to the R language (embedded R)
      scipy           0.18.1 SciPy: Scientific Library for Python
      seaborn         0.7.1  Seaborn: statistical data visualization
      six             1.16.0 Python 2 and 3 compatibility utilities

## 0.1.0 (2021-3-4)

- Initial placeholder.
