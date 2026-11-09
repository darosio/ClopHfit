# Refactor status

Status of the refactor plan (A-E), as of 2026-11-08. The reasoning behind each
step is in the commit messages and PR descriptions; decisions are recorded in
`docs/adr/`.

| Item                        | Outcome                                                                                           | Where                                                            |
| --------------------------- | ------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| A. EnSpire run interface    | Done. Every run lives in `clophfit.prenspire.runs`; the CLI parses options and prints.            | PR #1470 (`refactor/enspire-run-least-squares`)                  |
| B. Global-Bayes runner      | Closed. The two models stay separate (ADR 0001).                                                  | `docs/adr/0001-global-bayes-models-stay-separate.md` in PR #1471 |
| C. Split `fitting/bayes.py` | Done. Now the `fitting/bayes/` package; `__init__` re-exports every former name.                  | PR #1471 (`refactor/bayes-split`)                                |
| D. Figures out of the fit   | Done. `fit_titrations_spectral` returns a `SpectralRun`; drawing lives in `prenspire/figures.py`. | PR #1472 (`refactor/spectral-figures`), stacked on #1470         |
| E. Legacy svd path          | Kept, no change.                                                                                  | This note                                                        |

## Merge order

1. #1470, then #1472 (stacked).
1. #1471, independent of the above.

## Open items

- Two tests in `tests/test_enspire_xlatent.py` fail on `main` as well
  (R-hat 1.0544 against 1.05; ESS 47 against 50). They need a separate fix.
- The successful `--join` path has no test. Only the rejection path was checked.
- PR #1470 is large (one commit per run group). Splitting it is optional.

## Reproducing the checks

```bash
uv sync --locked --group dev --group lint --group tests
make lint
uv run pytest -m "not slow"
uv run pytest -m slow tests/test_enspire_golden.py
```
