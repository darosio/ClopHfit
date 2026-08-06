# TitrationConfig Decoupling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove Bayesian-sampler vocabulary from `TitrationConfig`, replacing it with the existing config objects passed explicitly to the export layer, and delete the dead CLI choices and cache-reset names orphaned by commit `01735f12`.

**Architecture:** Six fields leave `TitrationConfig`; `export.py` stops reading `titration.params` for anything concerning sampling and instead receives an `McmcSpec` threaded through `export_data_fit` → `export_fit` → `fit_single_mcmc`. `__main__.py` keeps its flags and builds the spec. Dead code goes first, in its own commits, so the refactor's diff stays about the refactor.

**Tech Stack:** Python 3.12+, click, PyMC, pytest, ruff, mypy (strict), pre-commit (mdformat, conventional-commit).

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-06-titration-config-decoupling-design.md`.
- Fields that **stay** on `TitrationConfig`: `bg`, `bg_adj`, `dil`, `nrm`, `bg_mth`, `fit_method`, `outlier`, `mask_outliers`, `outlier_threshold`, `noise_alpha`, `noise_gain`. Do not touch `noise_alpha`/`noise_gain` — `_apply_error_model` (`titration.py:1181`) reads them to build the `PlateNoiseModel` that sets `y_err` for every fit.
- Fields that **leave**: `mcmc`, `nuts_sampler`, `n_mcmc_samples`, `ctr_free_k`, `mcmc_noise`, `noise_mode`.
- Live `--mcmc` values after this work: `None`, `single`, `single-refit`. No others.
- Do not add subcommands, do not restructure `__main__.py` — that is sub-project 2.
- mypy runs `strict`; every new function needs full annotations and a numpy-style docstring (`pydoclint` runs in pre-commit).
- Commit messages must satisfy the Conventional Commit hook. `mdformat` rewrites markdown on commit: if a commit aborts with "files were modified by this hook", re-`git add` and re-run the same commit.

______________________________________________________________________

### Task 1: Delete the dead cache-reset tuple

All twelve names popped by `_reset_data_and_results` are gone. `Titration`'s only `cached_property` is `fit_keys` (`titration.py:788`) and it is not in the tuple; `clear_all_data_results` deletes it separately. The method's real work is one line.

**Files:**

- Modify: `src/clophfit/prtecan/titration.py:919-936`
- Modify: `src/clophfit/fitting/plotting.py:1340` (stale docstring naming the deleted `tit.result_global`)
- Test: `tests/test_prtecan.py`

**Interfaces:**

- Consumes: nothing.

- Produces: `Titration._reset_data_and_results(self) -> None` — unchanged signature, reduced body.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_prtecan.py`:

```python
def test_params_change_resets_derived_data() -> None:
    """Setting any surviving params field discards the derived data cache.

    Asserts the effect, not a list of attribute names: a name list is exactly
    what rotted into twelve dead entries after 01735f12.
    """
    tit = prtecan.Titration.fromlistfile(data_tests / "140220/list.pH.csv", is_ph=True)
    tit.load_scheme(data_tests / "140220" / "scheme.txt")
    assert tit.data  # populate the lazily-built cache
    assert tit._data != {}  # noqa: SLF001

    tit.params.nrm = not tit.params.nrm

    # Do not touch tit.data here — reading it would refill the cache.
    assert tit._data == {}  # noqa: SLF001
```

Note: assert on `_data` rather than on `bg`/`bg_err`. Those are properties delegating to `self.buffer` (`titration.py:960`, `:972`), so they may legitimately recompute and never read back as `{}`.

- [ ] **Step 2: Run the test to see the current state**

Run: `python -m pytest tests/test_prtecan.py::test_params_change_resets_derived_data -v`
Expected: PASS against the current code. This is a characterization test, not a red-green cycle — the dead code is invisible from outside, which is exactly why it survived since May. The test's job is to fail if Step 3 removes too much.

- [ ] **Step 3: Reduce the method**

Replace `titration.py:919-936` with:

```python
    def _reset_data_and_results(self) -> None:
        """Discard derived data so the next access recomputes it."""
        self._data = {}
```

- [ ] **Step 4: Fix the stale docstring**

In `src/clophfit/fitting/plotting.py:1340`, replace the phrase

```
        Normally this is `tit.result_global.results`.
```

with

```
        Normally this is the ``results`` mapping of a ``TitrationResults``.
```

- [ ] **Step 5: Run the full prtecan and plotting suites**

Run: `python -m pytest tests/test_prtecan.py -q`
Expected: PASS, no new failures.

- [ ] **Step 6: Commit**

```bash
git add src/clophfit/prtecan/titration.py src/clophfit/fitting/plotting.py tests/test_prtecan.py
git commit -m "refactor(prtecan): drop the dead cache-reset tuple

All twelve names popped by _reset_data_and_results were removed by 01735f12;
Titration's only cached_property is fit_keys, which clear_all_data_results
handles separately. The method's real work was always self._data = {}."
```

______________________________________________________________________

### Task 2: Remove the dead CLI surface

Two inert flags, one root cause. `params.mcmc` is read only at `export.py:269` and `:282`, both testing for `"single"`/`"single-refit"`, so `multi`, `multi-noise` and `multi-noise-xrw` are advertised no-ops. `--ctr-free-k` is worse: `__main__.py:280` writes it to `params.ctr_free_k` and **nothing in `src/` ever reads it**. It is a keyword of the multi-well fitter (`bayes.py:2401`), which the CLI has been unable to reach since `01735f12`. Both return in sub-project 2 as `ppr tecan mcmc --model multi [--ctr-free-k]`, where they will be wired to `fit_binding_pymc_multi` and actually do something.

**Files:**

- Modify: `src/clophfit/__main__.py:157` (`--mcmc` choices), `:160` (`--ctr-free-k` option), `:190` (its function parameter), `:280` (its assignment)
- Test: `tests/test_cli.py`

**Interfaces:**

- Consumes: nothing.

- Produces: `--mcmc` accepts exactly `None`, `single`, `single-refit`; `--ctr-free-k` no longer exists.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cli.py`:

```python
def test_prtecan_rejects_retired_mcmc_modes(tmp_path: Path, runner: CliRunner) -> None:
    """The multi modes were no-ops after 01735f12; they must not be offered."""
    list_f = str(tpath / "Tecan" / "140220" / "list.pH.csv")
    for mode in ("multi", "multi-noise", "multi-noise-xrw"):
        result = runner.invoke(
            ppr,
            ["--out", str(tmp_path / "out"), "tecan", list_f, "--mcmc", mode, "--dry-run"],
        )
        assert result.exit_code != 0, f"{mode} was accepted"
        assert "single-refit" in result.output


def test_prtecan_rejects_retired_ctr_free_k(tmp_path: Path, runner: CliRunner) -> None:
    """--ctr-free-k was written to params and read by nobody; it must not be offered."""
    list_f = str(tpath / "Tecan" / "140220" / "list.pH.csv")
    result = runner.invoke(
        ppr,
        ["--out", str(tmp_path / "out"), "tecan", list_f, "--ctr-free-k", "--dry-run"],
    )
    assert result.exit_code != 0
    assert "no such option" in result.output.lower()
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/test_cli.py -k "retired" -v`
Expected: both FAIL — the modes and the flag are currently accepted, so `exit_code == 0`.

- [ ] **Step 3: Prune the choice list and drop the inert flag**

At `src/clophfit/__main__.py:157`, replace the option with:

```python
@click.option("--mcmc", type=_FlexChoice(["None", "single", "single-refit"], case_sensitive=False), default="None", show_default=True, help="Per-well MCMC sampling: None, single, single-refit (robust screening pass then refit).")  # fmt: skip
```

Delete the whole `--ctr-free-k` option line at `:160`, its `ctr_free_k: bool,` parameter at `:190`, and its assignment `tit.params.ctr_free_k = ctr_free_k` at `:280`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_cli.py -k "retired" -v`
Expected: PASS

- [ ] **Step 5: Run the whole CLI suite**

Run: `python -m pytest tests/test_cli.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/clophfit/__main__.py tests/test_cli.py
git commit -m "refactor(cli): retire the no-op --mcmc multi modes

params.mcmc is read in two places, both testing for single/single-refit, so
multi, multi-noise and multi-noise-xrw have done nothing since 01735f12
deleted the multi-MCMC path. CLI access returns in sub-project 2 over
fit_binding_pymc_multi."
```

______________________________________________________________________

### Task 3: Introduce `McmcSpec`

**Files:**

- Modify: `src/clophfit/prtecan/titration.py` (add after `TecanConfig`, which ends at `titration.py:1488`)
- Modify: `src/clophfit/prtecan/__init__.py` (export it beside `TecanConfig`)
- Test: `tests/test_prtecan.py`

**Interfaces:**

- Consumes: `SamplerConfig`, `NoiseConfig` from `clophfit.fitting.bayes_config`.

- Produces: `McmcSpec(model: Literal["single", "single-refit"], sampler: SamplerConfig, structured_noise: bool = False, noise_mode: Literal["centered", "fixed"] = "centered")` — Tasks 4-6 rely on exactly these field names. No `ctr_free_k`: nothing reads it, and sub-project 2 will add it when `--model multi` can actually consume it.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_prtecan.py`:

```python
def test_mcmc_spec_defaults() -> None:
    """McmcSpec carries the sampling decision without touching TitrationConfig."""
    spec = prtecan.McmcSpec(model="single", sampler=SamplerConfig(n_samples=8))
    assert spec.model == "single"
    assert spec.sampler.n_samples == 8
    assert spec.structured_noise is False
    assert spec.noise_mode == "centered"
    assert not hasattr(spec, "ctr_free_k")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_prtecan.py::test_mcmc_spec_defaults -v`
Expected: FAIL with `AttributeError: module 'clophfit.prtecan' has no attribute 'McmcSpec'`

- [ ] **Step 3: Add the dataclass**

Insert into `src/clophfit/prtecan/titration.py` immediately after the `TecanConfig` dataclass:

```python
@dataclass(frozen=True)
class McmcSpec:
    """A per-well MCMC request, decided by the caller rather than parsed downstream.

    Parameters
    ----------
    model : Literal["single", "single-refit"]
        Which per-well fit to run. ``"single"`` samples each well once;
        ``"single-refit"`` runs the robust screening pass then refits.
    sampler : SamplerConfig
        NUTS controls forwarded to ``pm.sample``.
    structured_noise : bool
        Build the physical ``floor + gain * y + (alpha * y) ** 2`` observation
        noise instead of scaling ``y_err`` by a learned ``ye_mag`` multiplier.
    noise_mode : Literal["centered", "fixed"]
        How a supplied gain/alpha hint is treated when *structured_noise* is
        set: centred on (a hint the posterior may leave) or pinned to it. A
        parameter with no supplied value is always free.
    """

    model: Literal["single", "single-refit"]
    sampler: SamplerConfig
    structured_noise: bool = False
    noise_mode: Literal["centered", "fixed"] = "centered"
```

Add `Literal` to the `typing` import at the top of the module if it is not already imported, and `SamplerConfig` from `clophfit.fitting.bayes_config`.

- [ ] **Step 4: Export it**

In `src/clophfit/prtecan/__init__.py`, add `McmcSpec` to the same import and `__all__` entry that already carries `TecanConfig`.

- [ ] **Step 5: Run the test to verify it passes**

Run: `python -m pytest tests/test_prtecan.py::test_mcmc_spec_defaults -v`
Expected: PASS

- [ ] **Step 6: Typecheck and commit**

```bash
mypy src/clophfit/prtecan/titration.py
git add src/clophfit/prtecan/titration.py src/clophfit/prtecan/__init__.py tests/test_prtecan.py
git commit -m "feat(prtecan): add McmcSpec to carry the sampling decision"
```

______________________________________________________________________

### Task 4: Make `_structured_noise` take its mode explicitly

`_structured_noise` (`export.py:136`) reads `titration.params.noise_mode`. `noise_gain`/`noise_alpha` stay on `params` and keep being read from there.

**Files:**

- Modify: `src/clophfit/prtecan/export.py:136-171`
- Test: `tests/test_prtecan.py`

**Interfaces:**

- Consumes: nothing from Task 3.

- Produces: `_structured_noise(titration: Titration, *, noise_mode: Literal["centered", "fixed"]) -> NoiseConfig`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_prtecan.py`:

```python
def test_structured_noise_takes_mode_as_argument() -> None:
    """The mode is passed in, not read off titration.params."""
    from clophfit.prtecan.export import _structured_noise  # noqa: PLC0415

    tit = prtecan.Titration.fromlistfile(data_tests / "140220/list.pH.csv", is_ph=True)
    tit.load_scheme(data_tests / "140220" / "scheme.txt")
    tit.params.noise_gain = (1.0, 1.0)

    cfg = _structured_noise(tit, noise_mode="fixed")

    assert cfg.gain_mode == "fixed"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_prtecan.py::test_structured_noise_takes_mode_as_argument -v`
Expected: FAIL with `TypeError: _structured_noise() got an unexpected keyword argument 'noise_mode'`

- [ ] **Step 3: Change the signature**

In `src/clophfit/prtecan/export.py`, change the definition to

```python
def _structured_noise(
    titration: Titration, *, noise_mode: Literal["centered", "fixed"]
) -> NoiseConfig:
```

replace the body line `supplied_mode = titration.params.noise_mode` with `supplied_mode = noise_mode`, and update the docstring's "Parameters" section to document `noise_mode` (pydoclint enforces this). Import `Literal` if absent.

- [ ] **Step 4: Update the one existing call site**

At `export.py:294`, `noise = _structured_noise(titration)` becomes `noise = _structured_noise(titration, noise_mode=spec.noise_mode)` — but `spec` does not exist until Task 5. For this task only, pass the value still on params: `noise = _structured_noise(titration, noise_mode=titration.params.noise_mode)`. Task 5 replaces it.

- [ ] **Step 5: Run the tests**

Run: `python -m pytest tests/test_prtecan.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/clophfit/prtecan/export.py tests/test_prtecan.py
git commit -m "refactor(prtecan): pass noise_mode into _structured_noise"
```

______________________________________________________________________

### Task 5: Thread `McmcSpec` through the export chain

**Files:**

- Modify: `src/clophfit/prtecan/export.py:245-321` (`fit_single_mcmc`), `:324-392` (`export_fit`), `:395-426` (`export_data_fit`)
- Modify: `src/clophfit/__main__.py:277-286`, `:358`
- Test: `tests/test_prtecan.py`

**Interfaces:**

- Consumes: `McmcSpec` from Task 3; `_structured_noise(..., noise_mode=...)` from Task 4.

- Produces: `fit_single_mcmc(titration, datasets, outfit, spec: McmcSpec | None) -> TitrationResults | None`; `export_fit(titration, subfolder, config, spec: McmcSpec | None = None) -> None`; `export_data_fit(titration, tecan_config, mcmc: McmcSpec | None = None) -> None`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_prtecan.py`:

```python
def test_fit_single_mcmc_returns_none_without_spec(tmp_path: Path) -> None:
    """No spec means no sampling, regardless of what params says."""
    from clophfit.prtecan.export import fit_single_mcmc  # noqa: PLC0415

    tit = prtecan.Titration.fromlistfile(data_tests / "140220/list.pH.csv", is_ph=True)
    assert fit_single_mcmc(tit, {}, tmp_path, None) is None
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_prtecan.py::test_fit_single_mcmc_returns_none_without_spec -v`
Expected: FAIL with `TypeError: fit_single_mcmc() takes 3 positional arguments but 4 were given`

- [ ] **Step 3: Rewrite `fit_single_mcmc`'s head**

Replace the signature and the first branch (`export.py:245-300`) so it reads from `spec`:

```python
def fit_single_mcmc(
    titration: Titration,
    datasets: dict[str, typing.Any],
    outfit: Path,
    spec: McmcSpec | None,
) -> TitrationResults | None:
```

Update the docstring's Parameters and Returns to describe *spec* (pydoclint), then replace the body's opening logic:

```python
    if spec is None:
        return None

    if spec.model == "single":
        mcmc_fits = {
            key: fit_binding_pymc(ds, sampler=spec.sampler)
            for key, ds in datasets.items()
        }
        return TitrationResults(titration.scheme, titration.fit_keys, mcmc_fits)

    sampler = spec.sampler
    structured = spec.structured_noise
    if structured:
        # One config for both passes: unlike ye_mag, whose refit prior is
        # recentred on the screening pass's learned multiplier, the structured
        # model's floor/gain/alpha hints do not shift between passes.
        noise = _structured_noise(titration, noise_mode=spec.noise_mode)
        screening_noise, refit_noise = noise, noise
    else:
        screening_noise = _ye_mag_screening_noise(titration.bg_noise)
        refit_noise = NoiseConfig.ye_mag(
            shared=False, prior="lognormal", mu=0.0, sigma=0.25
        )
```

Leave the loop from `mcmc_fits = {}` (`export.py:301`) onward unchanged.

- [ ] **Step 4: Thread the parameter through the two callers**

In `export_fit`, add `spec: McmcSpec | None = None` to the signature and change `export.py:368` to `mcmc_res = fit_single_mcmc(titration, datasets, outfit, spec)`.

In `export_data_fit`, add `mcmc: McmcSpec | None = None` to the signature and pass it at both `export_fit(...)` call sites (`export.py:418` and `:426`) as `export_fit(titration, subfolder, tecan_config, mcmc)`.

- [ ] **Step 5: Build the spec in the CLI**

In `src/clophfit/__main__.py`, delete these six assignments from the `tit.params` block at `:277-286`:

```python
    tit.params.mcmc = mcmc
    tit.params.nuts_sampler = nuts_sampler
    tit.params.n_mcmc_samples = mcmc_samples
    tit.params.mcmc_noise = mcmc_noise
    tit.params.noise_mode = noise_mode
```

(`tit.params.ctr_free_k` was already removed in Task 2 along with its flag.) Keep `tit.params.noise_alpha = noise_alpha` and `tit.params.noise_gain = noise_gain`. Then build the spec just before the `export_data_fit` call at `:359`:

```python
    mcmc_spec = (
        None
        if mcmc == "None"
        else McmcSpec(
            model=cast('Literal["single", "single-refit"]', mcmc),
            sampler=SamplerConfig(n_samples=mcmc_samples, nuts_sampler=nuts_sampler),
            structured_noise=mcmc_noise == "structured",
            noise_mode=cast('Literal["centered", "fixed"]', noise_mode),
        )
    )
    export_data_fit(tit, tecan_config, mcmc_spec)
```

Add `McmcSpec` to the existing `from clophfit.prtecan import ...` at `:48`, `SamplerConfig` from `clophfit.fitting.bayes_config`, and `cast`/`Literal` from `typing`.

- [ ] **Step 6: Run the tests**

Run: `python -m pytest tests/test_prtecan.py tests/test_cli.py -q`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/clophfit/prtecan/export.py src/clophfit/__main__.py tests/test_prtecan.py
git commit -m "refactor(prtecan): pass an McmcSpec into the export chain

export.py no longer reads titration.params to decide whether or how to sample."
```

______________________________________________________________________

### Task 6: Delete the six fields from `TitrationConfig`

**Files:**

- Modify: `src/clophfit/prtecan/titration.py:131-134`, `:154-168`
- Test: `tests/test_prtecan.py`

**Interfaces:**

- Consumes: everything from Tasks 3-5. No reader of the six fields may remain.

- Produces: a `TitrationConfig` with eleven fields.

- [ ] **Step 1: Prove there are no readers left**

Run: `grep -rn "params\.mcmc\b\|params\.nuts_sampler\|params\.n_mcmc_samples\|params\.ctr_free_k\|params\.mcmc_noise\|params\.noise_mode" src/ tests/`
Expected: no output. If anything appears, fix that call site before continuing.

- [ ] **Step 2: Write the failing test**

Append to `tests/test_prtecan.py`:

```python
def test_titration_config_carries_no_sampler_fields() -> None:
    """Sampling choices belong to the call that samples, not to the plate."""
    names = {f.name for f in dataclasses.fields(prtecan.TitrationConfig)}
    retired = {
        "mcmc",
        "nuts_sampler",
        "n_mcmc_samples",
        "ctr_free_k",
        "mcmc_noise",
        "noise_mode",
    }
    assert names & retired == set()
    assert {"noise_alpha", "noise_gain"} <= names
```

Add `import dataclasses` to the test module if absent.

- [ ] **Step 3: Run it to verify it fails**

Run: `python -m pytest tests/test_prtecan.py::test_titration_config_carries_no_sampler_fields -v`
Expected: FAIL — the six fields are still present.

- [ ] **Step 4: Delete the fields**

From `src/clophfit/prtecan/titration.py`, delete `mcmc` (`:131`), `nuts_sampler` (`:132`), `n_mcmc_samples` (`:133`), `ctr_free_k` (`:134`), and the `mcmc_noise` (`:154`) and `noise_mode` (`:162`) fields together with their docstrings. Leave `noise_alpha` and `noise_gain` and their docstrings in place.

- [ ] **Step 5: Run the test to verify it passes**

Run: `python -m pytest tests/test_prtecan.py::test_titration_config_carries_no_sampler_fields -v`
Expected: PASS

- [ ] **Step 6: Run the full suite and the type checker**

Run: `python -m pytest tests -q -x`
Expected: PASS.
Run: `mypy src/clophfit`
Expected: `Success: no issues found`.
Run: `ruff check src/clophfit tests`
Expected: `All checks passed!`

- [ ] **Step 7: Commit**

```bash
git add src/clophfit/prtecan/titration.py tests/test_prtecan.py
git commit -m "refactor(prtecan): remove sampler vocabulary from TitrationConfig

mcmc, nuts_sampler, n_mcmc_samples, ctr_free_k, mcmc_noise and noise_mode now
travel as an McmcSpec. noise_alpha and noise_gain stay: _apply_error_model
reads them to build the PlateNoiseModel that sets y_err for every fit."
```

______________________________________________________________________

## Verification

After Task 6, confirm the spec's claims hold:

- [ ] `grep -rn "params\.mcmc\|params\.noise_mode\|params\.mcmc_noise\|params\.ctr_free_k\|params\.nuts_sampler\|params\.n_mcmc_samples" src/ tests/` returns nothing.
- [ ] `ppr tecan --help` lists `--mcmc` with exactly `None|single|single-refit`, and no `--ctr-free-k`.
- [ ] `grep -rn "noise_alpha\|noise_gain" src/clophfit/prtecan/titration.py` still shows both fields and `_apply_error_model` reading them.
- [ ] `python -m pytest tests -q` passes.
- [ ] `mypy src/clophfit` passes.
- [ ] `ruff check src/clophfit tests` passes.

## Deviation from the spec

The spec names six departing fields and three dead `--mcmc` choices. While writing this plan a seventh piece of the same debris turned up: `--ctr-free-k` is written to `params.ctr_free_k` at `__main__.py:280` and read nowhere in `src/`. It is retired in Task 2 and omitted from `McmcSpec`, on the same reasoning the spec applies to the `--mcmc multi` modes. If that is unwanted, the change is confined to Task 2 and the `McmcSpec` field list.
