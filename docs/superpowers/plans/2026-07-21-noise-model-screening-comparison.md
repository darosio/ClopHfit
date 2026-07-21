# Noise-model & screening comparison — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `ye_mag_parameterization` model option (centered / noncentered / hierarchical) to the Bayesian plate fitter, two residual-calibration metrics, and a classically-characterized MAD pre-screen, then run a convergence-gated 13-config comparison on plates L2/L3/L4.

**Architecture:** Three independent pieces. Piece 1 is a PyMC model change in `bayes.py` (a new keyword threaded to the per-well `ye_mag` prior builder, plus switching the structured `learn_ye_mags` multiplier to LogNormal). Pieces 2–3 live entirely in the research harness `scripts/compare_plate_pipelines.py` and consume the fitter. The two calibration metrics are cross-cutting additions to the harness's `score_pipeline`.

**Tech Stack:** Python, PyMC + nutpie, ArviZ (v1.2 API), NumPy/pandas/scipy, pytest, mypy, ruff, pre-commit.

## Global Constraints

- Design doc: `docs/superpowers/specs/2026-07-21-noise-model-screening-comparison-design.md`.
- Type hints on all public functions, mypy-compatible; numpy-style docstrings on public API.
- `make type` (mypy over `src tests docs/conf.py`) and `make lint` (pre-commit) must pass before each commit. **`ruff check` auto-fixes on write** (`pyproject.toml` `fix = true`); use `ruff check --no-fix` when only inspecting, and never let it mass-rewrite `# noqa:` comments in unrelated files — stage only intended changes.
- The new `ye_mag_parameterization` keyword is **opt-in, default `"centered"`**. On the `ye_mag`-noise path the default reproduces existing behaviour exactly. The **one deliberate, spec-approved exception** (see the design doc, Piece 1): the structured `learn_ye_mags` multiplier switches `HalfNormal(5.0)` → `LogNormal(0, 1.5)` for all parameterizations including `"centered"`, so the parameterization applies to structured noise without a prior/geometry confound. This is intended, not a regression.
- Real data lives at `/home/dati/arslanbaeva/data/raw/{L2,L3,L4}` (each has `list.pH.csv`, `additions.pH`, `scheme.txt`).
- Production sampler for the grid: `nutpie`, `target_accept=0.98`, `n_tune=1000`, `n_samples=10000`, `compute_log_likelihood=True`.
- Work on branch `noise-model-screening-comparison` (already created). Commit after each task.
- End commit messages with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

______________________________________________________________________

## File Structure

- `src/clophfit/fitting/bayes.py` — add `parameterization` to `_build_multi_ye_mag_priors`; add `ye_mag_parameterization` kwarg to `fit_binding_pymc_multi`; switch the structured `learn_ye_mags` call site to LogNormal. (Tasks 3–5)
- `tests/test_bayes.py` — prior-predictive equivalence, hierarchical prior correlation, convergence smoke. (Tasks 3–5)
- `scripts/compare_plate_pipelines.py` — `bulk_sd` + `resid_step_corr` in `score_pipeline` (Tasks 1–2); MAD pre-screen characterization (Task 6); the 13-config base + auto-top-7 screening grid (Task 7).
- `tests/test_compare_pipelines_metrics.py` — new, unit tests for the two metric helpers. (Tasks 1–2)

______________________________________________________________________

## Task 1: `bulk_sd` calibration metric

**Files:**

- Create: `tests/test_compare_pipelines_metrics.py`
- Modify: `scripts/compare_plate_pipelines.py` (add `bulk_sd_scores`; call it from `score_pipeline` near line 179, where `normality_scores` is merged in)

**Interfaces:**

- Produces: `bulk_sd_scores(residuals: pd.DataFrame, column: str = "std_res") -> dict[str, float]` returning keys `bulk_sd_pooled`, `bulk_sd_lbl{L}` for each label L. `bulk_sd = IQR/1.349` of the column.

- [ ] **Step 1: Write the failing test**

Create `tests/test_compare_pipelines_metrics.py`:

```python
"""Unit tests for the calibration-metric helpers in compare_plate_pipelines."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

_SPEC = importlib.util.spec_from_file_location(
    "compare_plate_pipelines",
    Path(__file__).resolve().parents[1] / "scripts" / "compare_plate_pipelines.py",
)
cpp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cpp)


def test_bulk_sd_recovers_unit_scale() -> None:
    """A standard-Normal std_res column gives bulk_sd close to 1."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "label": np.repeat(["1", "2"], 2000),
        "std_res": rng.normal(0, 1, 4000),
    })
    out = cpp.bulk_sd_scores(df)
    assert abs(out["bulk_sd_pooled"] - 1.0) < 0.1
    assert abs(out["bulk_sd_lbl1"] - 1.0) < 0.1


def test_bulk_sd_ignores_tails() -> None:
    """Heavy tails do not inflate bulk_sd; it tracks the central spread."""
    rng = np.random.default_rng(1)
    core = rng.normal(0, 1, 1000)
    contaminated = np.concatenate([core, rng.normal(0, 20, 30)])
    clean = pd.DataFrame({"label": ["1"] * len(core), "std_res": core})
    dirty = pd.DataFrame({"label": ["1"] * len(contaminated), "std_res": contaminated})
    assert abs(cpp.bulk_sd_scores(dirty)["bulk_sd_lbl1"]
               - cpp.bulk_sd_scores(clean)["bulk_sd_lbl1"]) < 0.15
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_compare_pipelines_metrics.py::test_bulk_sd_recovers_unit_scale -q`
Expected: FAIL with `AttributeError: module ... has no attribute 'bulk_sd_scores'`.

- [ ] **Step 3: Write minimal implementation**

In `scripts/compare_plate_pipelines.py`, add after `normality_scores` (ends ~line 106):

```python
def bulk_sd_scores(residuals: pd.DataFrame, column: str = "std_res") -> dict[str, float]:
    """Robust SD of the residual bulk (IQR/1.349), pooled and per label.

    Target is ~1. Below 1 means the model's sigma is too large for the bulk
    (scale over-inflation); ~1 with heavy Anderson tails means a likelihood
    shape mismatch, not scale.

    Parameters
    ----------
    residuals : pd.DataFrame
        Canonical residual table.
    column : str
        Column to summarize; ``std_res`` is on a Normal scale by construction.

    Returns
    -------
    dict[str, float]
        ``bulk_sd_pooled`` and ``bulk_sd_lbl{label}`` per label.
    """
    def _bulk_sd(values: pd.Series) -> float:
        x = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        if x.size < 8:
            return float("nan")
        q25, q75 = np.percentile(x, [25, 75])
        return float((q75 - q25) / 1.349)

    out = {"bulk_sd_pooled": _bulk_sd(residuals[column])}
    for lbl, grp in residuals.groupby("label", observed=True):
        out[f"bulk_sd_lbl{lbl}"] = _bulk_sd(grp[column])
    return out
```

Then in `score_pipeline`, immediately after the `row.update(normality_scores(res.residuals))` block (~line 180), add:

```python
    try:
        row.update(bulk_sd_scores(res.residuals))
    except Exception as exc:  # noqa: BLE001
        row["bulk_sd_error"] = str(exc)[:120]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_compare_pipelines_metrics.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Lint, type, commit**

Run: `make lint && make type`
Expected: both pass.

```bash
git add scripts/compare_plate_pipelines.py tests/test_compare_pipelines_metrics.py
git commit -m "feat(harness): add bulk_sd residual-calibration metric

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

______________________________________________________________________

## Task 2: `resid_step_corr` structural-misfit metric

**Files:**

- Modify: `scripts/compare_plate_pipelines.py` (add `resid_step_corr_scores`; call from `score_pipeline`)
- Modify: `tests/test_compare_pipelines_metrics.py` (add test)

**Interfaces:**

- Consumes: canonical residual table with `label`, `step`, `std_res`.

- Produces: `resid_step_corr_scores(residuals: pd.DataFrame) -> dict[str, float]` returning `resid_step_rho_lbl{L}` and `resid_step_p_lbl{L}` (Spearman of `std_res` vs `step`, per label).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_compare_pipelines_metrics.py`:

```python
def test_resid_step_corr_flags_structured_misfit() -> None:
    """A step-dependent residual gives a large |rho|; noise alone gives ~0."""
    rng = np.random.default_rng(2)
    steps = np.tile(np.arange(7), 40)
    structured = pd.DataFrame({
        "label": ["1"] * len(steps), "step": steps,
        "std_res": 0.4 * steps + rng.normal(0, 0.2, len(steps)),
    })
    noise = pd.DataFrame({
        "label": ["1"] * len(steps), "step": steps,
        "std_res": rng.normal(0, 1, len(steps)),
    })
    assert abs(cpp.resid_step_corr_scores(structured)["resid_step_rho_lbl1"]) > 0.8
    assert abs(cpp.resid_step_corr_scores(noise)["resid_step_rho_lbl1"]) < 0.2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_compare_pipelines_metrics.py::test_resid_step_corr_flags_structured_misfit -q`
Expected: FAIL with `AttributeError: ... 'resid_step_corr_scores'`.

- [ ] **Step 3: Write minimal implementation**

In `scripts/compare_plate_pipelines.py`, after `bulk_sd_scores`:

```python
def resid_step_corr_scores(residuals: pd.DataFrame) -> dict[str, float]:
    """Spearman correlation of std_res with titration step, per label.

    A non-trivial correlation flags a step-dependent (structural) model misfit,
    the benign alternative to over-robustness for a flat residual bulk.

    Parameters
    ----------
    residuals : pd.DataFrame
        Canonical residual table with ``label``, ``step``, ``std_res``.

    Returns
    -------
    dict[str, float]
        ``resid_step_rho_lbl{label}`` and ``resid_step_p_lbl{label}`` per label.
    """
    out: dict[str, float] = {}
    for lbl, grp in residuals.groupby("label", observed=True):
        s = pd.to_numeric(grp["step"], errors="coerce").to_numpy(dtype=float)
        r = pd.to_numeric(grp["std_res"], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(s) & np.isfinite(r)
        if ok.sum() < 8 or np.unique(s[ok]).size < 2:
            out[f"resid_step_rho_lbl{lbl}"] = float("nan")
            out[f"resid_step_p_lbl{lbl}"] = float("nan")
            continue
        res = stats.spearmanr(s[ok], r[ok])
        out[f"resid_step_rho_lbl{lbl}"] = float(res.statistic)
        out[f"resid_step_p_lbl{lbl}"] = float(res.pvalue)
    return out
```

`stats` (scipy.stats) is already imported at the top of the file. Then in `score_pipeline`, after the `bulk_sd_scores` block:

```python
    try:
        row.update(resid_step_corr_scores(res.residuals))
    except Exception as exc:  # noqa: BLE001
        row["resid_step_corr_error"] = str(exc)[:120]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_compare_pipelines_metrics.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Lint, type, commit**

Run: `make lint && make type`

```bash
git add scripts/compare_plate_pipelines.py tests/test_compare_pipelines_metrics.py
git commit -m "feat(harness): add residual-vs-step structural-misfit metric

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

______________________________________________________________________

## Task 3: `parameterization="noncentered"` in the ye_mag builder

**Files:**

- Modify: `src/clophfit/fitting/bayes.py` (`_build_multi_ye_mag_priors`, line 870)
- Modify: `tests/test_bayes.py`

**Interfaces:**

- Produces: `_build_multi_ye_mag_priors(..., parameterization: Literal["centered", "noncentered", "hierarchical"] = "centered")`. For `"noncentered"` on the per-well LogNormal path, each label's `ye_mag_{lbl}` is `exp(mu + sigma * z)` with `z ~ Normal(0,1, dims="well")`, giving the same marginal as centered `LogNormal(mu, sigma)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_bayes.py` (imports `numpy as np`, `pymc as pm`, `pytest` already present — add any missing at top):

```python
def test_ye_mag_noncentered_matches_centered_prior() -> None:
    """Non-centered ye_mag has the same prior marginal as centered LogNormal."""
    from clophfit.fitting.bayes import _build_multi_ye_mag_priors

    coords = {"well": [f"w{i}" for i in range(20)]}
    with pm.Model(coords=coords):
        _build_multi_ye_mag_priors(["1"], per_well=True, parameterization="centered")
        centered = pm.sample_prior_predictive(200, var_names=["ye_mag_1"]).prior
    with pm.Model(coords=coords):
        _build_multi_ye_mag_priors(["1"], per_well=True, parameterization="noncentered")
        nc = pm.sample_prior_predictive(200, var_names=["ye_mag_1"]).prior

    c = np.asarray(centered["ye_mag_1"]).ravel()
    n = np.asarray(nc["ye_mag_1"]).ravel()
    # Same log-scale mean/sd (LogNormal(0, 1.5)) within Monte-Carlo tolerance.
    assert abs(np.log(c).mean() - np.log(n).mean()) < 0.2
    assert abs(np.log(c).std() - np.log(n).std()) < 0.2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_bayes.py::test_ye_mag_noncentered_matches_centered_prior -q`
Expected: FAIL with `TypeError: _build_multi_ye_mag_priors() got an unexpected keyword argument 'parameterization'`.

- [ ] **Step 3: Write minimal implementation**

In `bayes.py`, change the signature of `_build_multi_ye_mag_priors` (line 870) to add the parameter, and replace the final per-well LogNormal `return` block (lines 907–915) so it branches on `parameterization`. Add `Literal` import if not present (it is — used elsewhere).

New signature:

```python
def _build_multi_ye_mag_priors(  # noqa: PLR0913
    labels: Sequence[str],
    *,
    per_well: bool = False,
    shared_ye_mags: bool = False,
    prior: Literal["halfnormal", "lognormal"] = "lognormal",
    mu: float | Mapping[str, float] = 0.0,
    sigma: float | Mapping[str, float] = 1.5,
    parameterization: Literal["centered", "noncentered", "hierarchical"] = "centered",
) -> dict[str, typing.Any]:
```

Replace the final `return { lbl: pm.LogNormal(...) }` block (the per-well, non-shared, lognormal case, lines 907–915) with:

```python
    if parameterization != "centered":
        return _build_reparam_ye_mag_priors(
            labels, mu=mu, sigma=sigma, parameterization=parameterization
        )
    return {
        lbl: pm.LogNormal(
            f"ye_mag_{lbl}",
            mu=_ye_mag_value(mu, lbl),
            sigma=_ye_mag_sigma(sigma, lbl),
            dims="well",
        )
        for lbl in labels
    }
```

Note: `parameterization` only affects this per-well, non-shared, LogNormal branch. The HalfNormal and shared branches ignore it (they are reached only when `prior="halfnormal"` or `shared_ye_mags=True`; the grid never combines those with a non-centered parameterization). Add the new helper directly below `_build_multi_ye_mag_priors`:

```python
def _build_reparam_ye_mag_priors(
    labels: Sequence[str],
    *,
    mu: float | Mapping[str, float],
    sigma: float | Mapping[str, float],
    parameterization: Literal["noncentered", "hierarchical"],
) -> dict[str, typing.Any]:
    """Per-well ye_mag priors under a non-centered or hierarchical form.

    ``"noncentered"`` reparameterizes each label's LogNormal independently
    (same marginal, better geometry). ``"hierarchical"`` shares a per-well
    factor across labels with a learned deviation scale (partial pooling).

    Parameters
    ----------
    labels : Sequence[str]
        Emission labels.
    mu, sigma : float | Mapping[str, float]
        Log-scale location and scale of the per-well ye_mag prior.
    parameterization : {"noncentered", "hierarchical"}
        Which reparameterization to build.

    Returns
    -------
    dict[str, typing.Any]
        Label -> per-well ``ye_mag`` tensor (dims ``"well"``).
    """
    if parameterization == "noncentered":
        out: dict[str, typing.Any] = {}
        for lbl in labels:
            z = pm.Normal(f"ye_mag_z_{lbl}", 0.0, 1.0, dims="well")
            out[lbl] = pm.Deterministic(
                f"ye_mag_{lbl}",
                pm.math.exp(_ye_mag_value(mu, lbl) + _ye_mag_sigma(sigma, lbl) * z),
                dims="well",
            )
        return out
    return _build_hierarchical_ye_mag_priors(labels, mu=mu, sigma=sigma)
```

For this task, add a stub for the hierarchical branch so the module imports (Task 4 fills it in):

```python
def _build_hierarchical_ye_mag_priors(
    labels: Sequence[str],
    *,
    mu: float | Mapping[str, float],
    sigma: float | Mapping[str, float],
) -> dict[str, typing.Any]:
    """Placeholder; implemented in Task 4."""
    raise NotImplementedError
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_bayes.py::test_ye_mag_noncentered_matches_centered_prior -q`
Expected: PASS.

- [ ] **Step 5: Lint, type, commit**

Run: `make lint && make type`

```bash
git add src/clophfit/fitting/bayes.py tests/test_bayes.py
git commit -m "feat(bayes): non-centered per-well ye_mag parameterization

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

______________________________________________________________________

## Task 4: `parameterization="hierarchical"` (shared factor + learned deviation)

**Files:**

- Modify: `src/clophfit/fitting/bayes.py` (`_build_hierarchical_ye_mag_priors`)
- Modify: `tests/test_bayes.py`

**Interfaces:**

- Produces: `_build_hierarchical_ye_mag_priors(labels, *, mu, sigma)` building `log ye_mag_lbl[w] = (mu + sigma*z_w[w]) + tau_delta*z_delta_lbl[w]`, with `tau_delta ~ HalfNormal(0.5)`, all non-centered. At the prior the two labels' `ye_mag` are positively correlated across wells.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_bayes.py`:

```python
def test_ye_mag_hierarchical_correlates_labels_at_prior() -> None:
    """Shared per-well factor induces positive cross-label prior correlation."""
    from clophfit.fitting.bayes import _build_multi_ye_mag_priors

    coords = {"well": [f"w{i}" for i in range(30)]}
    with pm.Model(coords=coords):
        _build_multi_ye_mag_priors(
            ["1", "2"], per_well=True, parameterization="hierarchical"
        )
        pr = pm.sample_prior_predictive(
            400, var_names=["ye_mag_1", "ye_mag_2", "ye_mag_tau_delta"]
        ).prior

    a = np.log(np.asarray(pr["ye_mag_1"]).reshape(-1, 30)).mean(axis=0)
    b = np.log(np.asarray(pr["ye_mag_2"]).reshape(-1, 30)).mean(axis=0)
    # Per-well means of the two labels track each other via the shared factor.
    assert np.corrcoef(a, b)[0, 1] > 0.5
    assert float(np.asarray(pr["ye_mag_tau_delta"]).min()) >= 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_bayes.py::test_ye_mag_hierarchical_correlates_labels_at_prior -q`
Expected: FAIL with `NotImplementedError`.

- [ ] **Step 3: Write minimal implementation**

Replace the `_build_hierarchical_ye_mag_priors` stub in `bayes.py` with:

```python
def _build_hierarchical_ye_mag_priors(
    labels: Sequence[str],
    *,
    mu: float | Mapping[str, float],
    sigma: float | Mapping[str, float],
) -> dict[str, typing.Any]:
    """Hierarchical per-well ye_mag: shared factor x learned label deviation.

    ``log ye_mag_lbl[w] = (mu + sigma * z_w[w]) + tau_delta * z_delta_lbl[w]``,
    with ``tau_delta ~ HalfNormal(0.5)`` learned and everything non-centered.
    The shared ``z_w`` factor partially pools the labels' per-well multipliers;
    the mean-zero deviation priors keep the common level in the shared factor.

    Parameters
    ----------
    labels : Sequence[str]
        Emission labels.
    mu, sigma : float | Mapping[str, float]
        Log-scale location and scale of the shared per-well factor.

    Returns
    -------
    dict[str, typing.Any]
        Label -> per-well ``ye_mag`` tensor (dims ``"well"``).
    """
    tau_delta = pm.HalfNormal("ye_mag_tau_delta", sigma=0.5)
    z_w = pm.Normal("ye_mag_z_well", 0.0, 1.0, dims="well")
    log_w = _shared_ye_mag_value(mu) + _shared_ye_mag_sigma(sigma) * z_w
    out: dict[str, typing.Any] = {}
    for lbl in labels:
        z_d = pm.Normal(f"ye_mag_z_delta_{lbl}", 0.0, 1.0, dims="well")
        out[lbl] = pm.Deterministic(
            f"ye_mag_{lbl}", pm.math.exp(log_w + tau_delta * z_d), dims="well"
        )
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_bayes.py::test_ye_mag_hierarchical_correlates_labels_at_prior -q`
Expected: PASS.

- [ ] **Step 5: Lint, type, commit**

Run: `make lint && make type`

```bash
git add src/clophfit/fitting/bayes.py tests/test_bayes.py
git commit -m "feat(bayes): hierarchical partially-pooled ye_mag parameterization

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

______________________________________________________________________

## Task 5: Wire `ye_mag_parameterization` into `fit_binding_pymc_multi`; structured -> LogNormal

**Files:**

- Modify: `src/clophfit/fitting/bayes.py` (`fit_binding_pymc_multi` signature line 2618; both `_build_multi_ye_mag_priors` call sites at 2932 and 2941)
- Modify: `tests/test_bayes.py`

**Interfaces:**

- Consumes: Tasks 3–4 builders.

- Produces: `fit_binding_pymc_multi(..., ye_mag_parameterization: Literal["centered", "noncentered", "hierarchical"] = "centered")`. Threads the value to both call sites. The structured `learn_ye_mags` call site (line 2932) is changed to `prior="lognormal"`, `mu=0.0`, `sigma=1.5` (dropping `prior="halfnormal", sigma=5.0`) so the parameterization applies in structured mode too.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_bayes.py`. Reuse an existing small multi-well fixture if present; otherwise build a 4-well dataset inline. This test asserts the kwarg is accepted and produces the expected latent variables (not convergence — that is the smoke test below).

```python
@pytest.mark.parametrize("param", ["centered", "noncentered", "hierarchical"])
def test_fit_binding_pymc_multi_accepts_parameterization(param: str) -> None:
    """Each parameterization builds and prior-samples with pwym on."""
    import numpy as np
    from clophfit.fitting.bayes import fit_binding_pymc_multi
    from clophfit.fitting.bayes_config import SamplerConfig
    from clophfit.fitting.data_structures import DataArray, Dataset
    from clophfit.fitting.models import binding_1site
    from clophfit.prtecan import PlateScheme

    rng = np.random.default_rng(0)
    x = np.linspace(5.5, 8.5, 7)
    dsd = {}
    for w in ("A01", "A02", "A03", "A04"):
        y1 = binding_1site(x, 7.0, 600.0, 50.0, is_ph=True) + rng.normal(0, 20, 7)
        y2 = binding_1site(x, 7.0, 40.0, 500.0, is_ph=True) + rng.normal(0, 8, 7)
        dsd[w] = Dataset(
            {"1": DataArray(x, y1, y_errc=np.full(7, 20.0)),
             "2": DataArray(x, y2, y_errc=np.full(7, 8.0))},
            is_ph=True,
        )
    scheme = PlateScheme()
    fit = fit_binding_pymc_multi(
        dsd, scheme, per_well_ye_mags=True, ye_mag_parameterization=param,
        sampler=SamplerConfig(nuts_sampler="pymc", n_tune=20, n_samples=20,
                              chains=2, progressbar=False),
    )
    assert "ye_mag_1" in fit.trace.posterior
    if param == "hierarchical":
        assert "ye_mag_tau_delta" in fit.trace.posterior
```

(If `SamplerConfig` field names for chains/progressbar differ, check `src/clophfit/fitting/bayes_config.py` and adjust; keep tune/draws tiny.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_bayes.py::test_fit_binding_pymc_multi_accepts_parameterization -q`
Expected: FAIL with `TypeError: ... unexpected keyword argument 'ye_mag_parameterization'`.

- [ ] **Step 3: Write minimal implementation**

In `fit_binding_pymc_multi` (signature block ending line 2638), add the kwarg after `per_well_ye_mags`:

```python
    ye_mag_parameterization: Literal["centered", "noncentered", "hierarchical"] = "centered",
```

Document it in the docstring (numpy style). Then at the structured call site (line 2932) change to:

```python
            if learn_ye_mags:
                ye_mags = _build_multi_ye_mag_priors(
                    labels,
                    per_well=use_per_well_ye_mags,
                    shared_ye_mags=shared_ye_mags,
                    prior="lognormal",
                    mu=0.0,
                    sigma=1.5,
                    parameterization=ye_mag_parameterization,
                )
```

And at the ye_mag-noise call site (line 2941) add the parameterization argument:

```python
            ye_mags = _build_multi_ye_mag_priors(
                labels,
                per_well=use_per_well_ye_mags,
                shared_ye_mags=shared_ye_mags,
                prior=ye_mag_prior,
                mu=ye_mag_mu,
                sigma=ye_mag_sigma,
                parameterization=ye_mag_parameterization,
            )
```

(Confirm the exact existing kwargs at 2941–2947 and append `parameterization=` — do not drop any.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_bayes.py::test_fit_binding_pymc_multi_accepts_parameterization -q`
Expected: PASS (3 parametrizations).

- [ ] **Step 5: Convergence smoke test (non-centered fixes the funnel)**

Append to `tests/test_bayes.py` a slower, marked test that a per-well fit mixes under `noncentered`. Keep it small but real (nutpie if available, else pymc). This is the piece-1 success criterion.

```python
@pytest.mark.slow
def test_noncentered_pwym_converges() -> None:
    """Non-centered pwym=1 reaches acceptable r_hat on a small plate."""
    import arviz as az
    import numpy as np
    from clophfit.fitting.bayes import fit_binding_pymc_multi
    from clophfit.fitting.bayes_config import RobustConfig, SamplerConfig
    from clophfit.fitting.data_structures import DataArray, Dataset
    from clophfit.fitting.models import binding_1site
    from clophfit.prtecan import PlateScheme

    rng = np.random.default_rng(1)
    x = np.linspace(5.5, 8.5, 7)
    dsd = {}
    for i in range(8):
        y1 = binding_1site(x, 7.0, 600.0, 50.0, is_ph=True) + rng.normal(0, 20, 7)
        y2 = binding_1site(x, 7.0, 40.0, 500.0, is_ph=True) + rng.normal(0, 8, 7)
        dsd[f"A{i:02d}"] = Dataset(
            {"1": DataArray(x, y1, y_errc=np.full(7, 20.0)),
             "2": DataArray(x, y2, y_errc=np.full(7, 8.0))},
            is_ph=True,
        )
    fit = fit_binding_pymc_multi(
        dsd, PlateScheme(), per_well_ye_mags=True,
        ye_mag_parameterization="noncentered",
        robust=RobustConfig(enabled=True, likelihood="student_t", nu=1),
        sampler=SamplerConfig(nuts_sampler="pymc", n_tune=500, n_samples=500,
                              chains=2, progressbar=False),
    )
    rh = az.rhat(fit.trace)
    worst = max(float(np.nanmax(v.values)) for v in rh.data_vars.values())
    assert worst < 1.2
```

Run: `python -m pytest tests/test_bayes.py::test_noncentered_pwym_converges -q`
Expected: PASS (may take a minute).

- [ ] **Step 6: Full targeted test, lint, type, commit**

Run: `python -m pytest tests/test_bayes.py -q -k "ye_mag or parameterization or noncentered" && make lint && make type`

```bash
git add src/clophfit/fitting/bayes.py tests/test_bayes.py
git commit -m "feat(bayes): thread ye_mag_parameterization; structured ye_mag -> LogNormal

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

______________________________________________________________________

## Task 6: Piece 2 — MAD pre-screen characterization (classical)

**Files:**

- Modify: `scripts/compare_plate_pipelines.py` (add `characterize_mad_prescreen`; add a `--tier prescreen` branch in `main`)

**Interfaces:**

- Consumes: `add_robust_scores` (`clophfit.fitting.utils`), `load_plate` (existing in harness).

- Produces: `characterize_mad_prescreen(raw: Path, plates: tuple[str, ...], out: Path) -> pd.DataFrame` writing a flag-count table (per plate, label, grouping, threshold) and example-well curve ONGs to `out`.

- [ ] **Step 1: Write the implementation**

`add_robust_scores` already computes `robust_z_well_label` (per well×label) and `robust_z_label` (pooled per label) plus `ye_mag_est`. Reuse it — do not reimplement MAD. Add to `scripts/compare_plate_pipelines.py`:

```python
def characterize_mad_prescreen(
    raw: Path, plates: tuple[str, ...], out: Path
) -> pd.DataFrame:
    """Tabulate what a MAD pre-screen would flag, per grouping and threshold.

    lm-fits every well, annotates raw residuals with per-well x label and
    per-label robust z (via add_robust_scores), and counts flags at |z|>5 and
    |z|>7 under each grouping. Writes the table and a few example-well curves.

    Parameters
    ----------
    raw : Path
        Raw-data directory holding the plate folders.
    plates : tuple[str, ...]
        Plate folder names.
    out : Path
        Output directory for the CSV and ONGs.

    Returns
    -------
    pd.DataFrame
        One row per (plate, label, grouping, threshold) with the flag count and
        fraction.
    """
    from clophfit.fitting.utils import add_robust_scores

    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for plate in plates:
        tit = load_plate(raw, plate)
        res = tit.fit_plate(method="lm")
        good = {w: fr for w, fr in res.results.items() if fr.result is not None}
        scored = add_robust_scores(res.residuals, levels=("well_label", "label"))
        for grouping, col in (("well_label", "robust_z_well_label"),
                              ("label", "robust_z_label")):
            for thr in (5.0, 7.0):
                flag = scored[col] > thr
                for lbl, grp in scored.groupby("label", observed=True):
                    f = grp[col] > thr
                    rows.append(dict(
                        plate=plate, label=str(lbl), grouping=grouping,
                        threshold=thr, n_flagged=int(f.sum()),
                        n_points=int(len(grp)),
                        frac=float(f.mean()),
                    ))
        _plot_prescreen_examples(plate, scored, good, out)
    df = pd.DataFrame(rows)
    df.to_csv(out / "mad_prescreen_flags.csv", index=False)
    print(df.to_string(index=False), flush=True)
    return df
```

Add a small plotting helper `_plot_prescreen_examples(plate, scored, good, out)` that picks one quiet and one noisy well (by `robust_sigma_well_label` median per well) and plots their curves with the well×label-flagged and label-flagged points marked differently, saved to `out / f"{plate}_prescreen_examples.png"`. Use matplotlib Agg. Keep it under ~30 lines.

Then in `main`, extend the `--tier` choices to include `"prescreen"` and add:

```python
    if args.tier == "prescreen":
        print("=== MAD pre-screen characterization ===", flush=True)
        characterize_mad_prescreen(args.raw, tuple(args.plates), args.out)
        return
```

- [ ] **Step 2: Run it on all three plates**

Run: `python scripts/compare_plate_pipelines.py --tier prescreen --out pipeline_comparison/prescreen`
Expected: prints a flag-count table; writes `mad_prescreen_flags.csv` and three example ONGs. Sanity: pooled grouping flags more in high-signal label 1 and near-zero in quiet wells; well×label flags are more evenly spread.

- [ ] **Step 3: Lint, type, commit**

Run: `make lint && make type`

```bash
git add scripts/compare_plate_pipelines.py
git commit -m "feat(harness): MAD pre-screen characterization tier

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

- [ ] **Step 4: HUMAN CHECKPOINT — decide the pre-screen rule**

Present the flag table and example ONGs to the user. Record their chosen grouping + threshold (spec default expectation: per-well×label, |z|>5, `min_keep=5`). This decision parameterizes Task 7's MAD arm. **Do not proceed to Task 7 until the rule is chosen.**

______________________________________________________________________

## Task 7: Piece 3 — 13-config base grid + auto top-7 screening

**Files:**

- Modify: `scripts/compare_plate_pipelines.py` (add `run_comparison_grid`; add `--tier grid`)

**Interfaces:**

- Consumes: `fit_binding_pymc_multi` with `ye_mag_parameterization`; `score_pipeline` (now emitting `bulk_sd_*`, `resid_step_*`); `loo_and_rhat`; `mark_outliers`/`ResidualTail`/`apply_exclusions`; the Task-6 pre-screen rule; the piece-2 per-well×label MAD via `add_robust_scores`.

- Produces: `run_comparison_grid(raw, plates, *, prescreen_grouping, prescreen_threshold, prescreen_min_keep, sampler_kwargs, out) -> list[dict]`.

- [ ] **Step 1: Write the base-config builder**

Add a helper returning the 13 base configs as dicts `{name, robust, noise_factory, pwym, parameterization}`. `noise_factory` is a callable `tit -> NoiseConfig` (structured needs `tit.bg_noise`). Encode configs 1–13 exactly per the spec table. Config 9 is `parameterization="centered"` with `pwym=1` (the divergent control). Configs 10–13 are `parameterization="hierarchical"`. Configs 2/4/6/8 are `"noncentered"`.

```python
def _base_configs(tit: Titration) -> list[dict[str, Any]]:
    """Return the 13 base MCMC configs for one plate (see the design doc)."""
    st = lambda: RobustConfig(enabled=True, likelihood="student_t", nu=1)
    mx = lambda: RobustConfig(enabled=True, likelihood="mixture",
                              contamination_frac_prior={"1": 0.15, "2": 0.015})
    yemag = lambda _t=None: NoiseConfig.ye_mag()
    struct = lambda t: NoiseConfig.structured(
        floor=t.bg_noise, gain=0.5, alpha=0.02, floor_mode="centered",
        gain_mode="free", alpha_mode="free", learn_ye_mags=True)
    C = []
    def add(name, rob, noise, pwym, param):
        C.append(dict(name=name, robust=rob, noise=noise, pwym=pwym, param=param))
    add("studentt_nu1|yemag|pwym0",        st(), yemag,  False, "centered")
    add("studentt_nu1|yemag|pwym1_nc",     st(), yemag,  True,  "noncentered")
    add("studentt_nu1|structured|pwym0",   st(), struct, False, "centered")
    add("studentt_nu1|structured|pwym1_nc",st(), struct, True,  "noncentered")
    add("mixture|yemag|pwym0",             mx(), yemag,  False, "centered")
    add("mixture|yemag|pwym1_nc",          mx(), yemag,  True,  "noncentered")
    add("mixture|structured|pwym0",        mx(), struct, False, "centered")
    add("mixture|structured|pwym1_nc",     mx(), struct, True,  "noncentered")
    add("studentt_nu1|yemag|pwym1_c",      st(), yemag,  True,  "centered")   # control
    add("studentt_nu1|yemag|hier",         st(), yemag,  True,  "hierarchical")
    add("studentt_nu1|structured|hier",    st(), struct, True,  "hierarchical")
    add("mixture|yemag|hier",              mx(), yemag,  True,  "hierarchical")
    add("mixture|structured|hier",         mx(), struct, True,  "hierarchical")
    return C
```

- [ ] **Step 2: Write the grid runner**

```python
def run_comparison_grid(  # noqa: PLR0913
    raw: Path, plates: tuple[str, ...], *,
    prescreen_grouping: str, prescreen_threshold: float, prescreen_min_keep: int,
    out: Path,
) -> list[dict[str, Any]]:
    """Run 13 base configs + auto-top-7 screening arms, per plate.

    Base configs run first; converged ones (r_hat <= 1.05) are ranked by
    anderson_A2_pooled and the top 7 get two screening arms each: a Normal
    refit after removing |likelihood_res|>4, and a MAD pre-screen using the
    chosen grouping/threshold before re-fitting with the base likelihood.

    Parameters
    ----------
    raw : Path
        Raw-data directory.
    plates : tuple[str, ...]
        Plate folder names.
    prescreen_grouping : str
        ``"well_label"`` or ``"label"`` (from Task 6).
    prescreen_threshold : float
        Robust-z cutoff for the MAD arm.
    prescreen_min_keep : int
        Minimum points retained per label by the MAD arm.
    out : Path
        Output directory (incremental CSV writes).

    Returns
    -------
    list[dict[str, Any]]
        One scored row per (plate, config, screen).
    """
    from clophfit.fitting.utils import add_robust_scores

    sampler = SamplerConfig(nuts_sampler="nutpie", target_accept=0.98,
                            n_tune=1000, n_samples=10000, compute_log_likelihood=True)
    common = dict(n_sd=7, n_xerr=1.0, x_error_model="per_well",
                  ctr_free_k=True, init=INIT, sampler=sampler)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for plate in plates:
        tit = load_plate(raw, plate)
        base_ds = tit.create_dataset_dict()
        base_rows: list[dict[str, Any]] = []
        for cfg in _base_configs(tit):
            meta = {"plate": plate, "fitter": "mcmc_multi", "screen": "none",
                    "config": cfg["name"]}
            t0 = time.time()
            try:
                fit = fit_binding_pymc_multi(
                    base_ds, tit.scheme, per_well_ye_mags=cfg["pwym"],
                    ye_mag_parameterization=cfg["param"],
                    robust=cfg["robust"], noise=cfg["noise"](tit), **common)
                row = score_pipeline(tit, fit, meta)
                row.update(loo_and_rhat(fit.trace)); row["secs"] = time.time() - t0
            except Exception:  # noqa: BLE001
                row = {**meta, "error": traceback.format_exc(limit=1)[:200]}
            rows.append(row); base_rows.append(row)
            print(f"  base {plate} {cfg['name']:34} "
                  f"A2={row.get('anderson_A2_pooled', float('nan')):7.2f} "
                  f"rhat={row.get('rhat_max', float('nan')):.3f}", flush=True)
            pd.DataFrame(rows).to_csv(out / "comparison_grid_partial.csv", index=False)

        # Rank converged base configs, take top 7 by A2, screen them.
        by_cfg = {r["config"]: r for r in base_rows}
        converged = [r for r in base_rows
                     if r.get("rhat_max", 9.9) <= 1.05 and "error" not in r]
        top = sorted(converged, key=lambda r: r.get("anderson_A2_pooled", 9e9))[:7]
        cfg_by_name = {c["name"]: c for c in _base_configs(tit)}
        for r in top:
            cfg = cfg_by_name[r["config"]]
            # Arm A: likres>4 -> Normal refit.
            _run_screen_arm(rows, tit, base_ds, cfg, common, out, arm="likres>4",
                            marker="likres")
            # Arm B: MAD pre-screen -> refit same likelihood.
            _run_screen_arm(rows, tit, base_ds, cfg, common, out, arm="mad_prescreen",
                            marker="mad", grouping=prescreen_grouping,
                            threshold=prescreen_threshold, min_keep=prescreen_min_keep)

    pd.DataFrame(rows).to_csv(out / "comparison_grid.csv", index=False)
    return rows
```

Full `_run_screen_arm` implementation:

```python
def _run_screen_arm(  # noqa: PLR0913
    rows: list[dict[str, Any]], tit: Titration, base_ds: dict[str, Any],
    cfg: dict[str, Any], base_fit: Any, common: dict[str, Any], out: Path, *,
    arm: str, grouping: str = "well_label", threshold: float = 5.0,
    min_keep: int = 5,
) -> None:
    """Run one screening arm of a base config and append its scored row.

    ``arm="likres>4"`` marks |likelihood_res|>4 on the *base* fit, removes those
    points, and refits Normal. ``arm="mad_prescreen"`` drops points whose robust
    z (grouping/threshold) exceeds the cutoff on an lm fit, then refits with the
    base config's own likelihood/noise/pwym/parameterization.

    Parameters
    ----------
    rows : list[dict[str, Any]]
        Accumulator the scored row is appended to.
    tit : Titration
        The plate.
    base_ds : dict[str, Any]
        The unscreened per-well datasets.
    cfg : dict[str, Any]
        The base-config dict from ``_base_configs``.
    base_fit : Any
        The fitted base result (used by the likres arm).
    common : dict[str, Any]
        Shared ``fit_binding_pymc_multi`` kwargs.
    out : Path
        Output dir for the incremental CSV.
    arm : str
        ``"likres>4"`` or ``"mad_prescreen"``.
    grouping, threshold, min_keep : str, float, int
        MAD-arm parameters.
    """
    from clophfit.fitting.utils import add_robust_scores

    meta = {"plate": tit.list_file.parent.name if hasattr(tit, "list_file")
            else cfg["name"], "fitter": "mcmc_multi", "screen": arm,
            "config": cfg["name"]}
    meta["plate"] = rows[-1]["plate"]  # same plate as the surrounding loop
    t0 = time.time()
    try:
        if arm == "likres>4":
            marked = mark_outliers(base_fit.residuals, ResidualTail(
                residual_col="likelihood_res", threshold=4.0,
                allowed_tail_fraction=0.0, min_allowed_tail_count=0))
            masked = apply_exclusions(base_ds, marked, min_keep=5)
            robust = RobustConfig(enabled=False)
        else:
            col = "robust_z_well_label" if grouping == "well_label" else "robust_z_label"
            scored = add_robust_scores(
                tit.fit_plate(method="lm").residuals,
                levels=("well_label", "label"))
            marked = scored.assign(exclude_outlier=scored[col] > threshold)
            masked = apply_exclusions(base_ds, marked, min_keep=min_keep)
            robust = cfg["robust"]
        fit = fit_binding_pymc_multi(
            masked, tit.scheme, per_well_ye_mags=cfg["pwym"],
            ye_mag_parameterization=cfg["param"], robust=robust,
            noise=cfg["noise"](tit), **common)
        row = score_pipeline(tit, fit, meta)
        row.update(loo_and_rhat(fit.trace)); row["secs"] = time.time() - t0
    except Exception:  # noqa: BLE001
        row = {**meta, "error": traceback.format_exc(limit=1)[:200]}
    rows.append(row)
    print(f"  arm  {meta['plate']} {cfg['name']:30} {arm:14} "
          f"A2={row.get('anderson_A2_pooled', float('nan')):7.2f}", flush=True)
    pd.DataFrame(rows).to_csv(out / "comparison_grid_partial.csv", index=False)
```

`apply_exclusions` expects an `exclude_outlier` column (its default `exclude_col`); the MAD arm builds it directly from the robust-z threshold, the likres arm gets it from `mark_outliers`. In the grid loop, retain each top-7 config's `fit` object (store `(row, fit)` in `base_rows`) so the likres arm reuses it; drop non-top-7 fits to bound memory.

- [ ] **Step 3: Add the `--tier grid` branch**

In `main`, add `"grid"` to `--tier` choices and CLI args `--prescreen-grouping` (default `well_label`), `--prescreen-threshold` (default `5.0`), `--prescreen-min-keep` (default `5`), then:

```python
    if args.tier == "grid":
        print("=== comparison grid (production sampler) ===", flush=True)
        rows += run_comparison_grid(
            args.raw, tuple(args.plates),
            prescreen_grouping=args.prescreen_grouping,
            prescreen_threshold=args.prescreen_threshold,
            prescreen_min_keep=args.prescreen_min_keep,
            out=args.out)
```

- [ ] **Step 4: Smoke-test the grid wiring cheaply**

Before the ~13.5 h production run, verify the wiring end-to-end on one plate at tiny sample counts by temporarily overriding the sampler (run a one-off inline python snippet that imports `run_comparison_grid` after monkeypatching a small `SamplerConfig`, on `--plates L2` — or add a hidden `--smoke` flag that swaps to `n_tune=50, n_samples=50, nuts_sampler="pymc"`). Confirm: 13 base rows produced, top-7 selected, screening arms run, `comparison_grid.csv` has `bulk_sd_pooled`, `resid_step_rho_lbl1`, `rhat_max`, `elpd_loo` columns populated.

Expected: completes in minutes, no exceptions, CSV columns present.

- [ ] **Step 5: Lint, type, commit**

Run: `make lint && make type`

```bash
git add scripts/compare_plate_pipelines.py
git commit -m "feat(harness): 13-config comparison grid with auto top-7 screening

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

- [ ] **Step 6: Launch the production run (HUMAN-GATED)**

With the pre-screen rule from Task 6, launch detached:

```bash
setsid nohup python scripts/compare_plate_pipelines.py --tier grid \
  --prescreen-grouping well_label --prescreen-threshold 5 --prescreen-min-keep 5 \
  --out pipeline_comparison/grid > pipeline_comparison/grid_run.log 2>&1 < /dev/null &
```

Monitor `pipeline_comparison/grid/comparison_grid_partial.csv` and `grid_run.log`. ~13.5 h. Report r̂>1.05 configs and the convergence-gated ranking when complete.

**Fixed-`tau_delta` fallback (spec contingency).** If a hierarchical config (10–13) reports `rhat_max > 1.05`, re-run just that config with a fixed deviation scale: add an optional `tau_delta_fixed: float | None = None` to `_build_hierarchical_ye_mag_priors` (when set, use `tau_delta = tau_delta_fixed` instead of the `HalfNormal` prior), thread a `ye_mag_tau_delta_fixed` kwarg through `fit_binding_pymc_multi`, and re-run that config with `0.5`. Record which variant (learned/fixed) produced the row via a `tau_delta_mode` column. This is a targeted re-run, not a grid expansion.

______________________________________________________________________

## Analysis (after Task 7 completes)

Aggregate `comparison_grid.csv` across plates: convergence-gate on `rhat_max <= 1.05`, then rank by `anderson_A2_pooled`, with `bulk_sd_pooled` (target ~1), `elpd_loo`, `p_loo`, `ctr_median_absdK`, `ctr_frac_in_rope`, `resid_step_rho_*` reported alongside. State, per noise model: does non-centered pwym converge and help; does hierarchical pooling beat independent pwym; does either screening arm improve over none; and the overall best pipeline. Confirm config 9 (centered pwym1) diverges where config 2 (non-centered) converges.
