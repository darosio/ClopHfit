# Noise Prior Symmetry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a calibrated gain of `0.0` an estimable prior around zero instead of a hard constant, and unify the two disagreeing gain prior widths.

**Architecture:** `build_pymc_noise_priors` (`src/clophfit/fitting/bayes.py:96`) turns a calibrated `PlateNoiseModel` into PyMC priors. NNLS clamps the collinear `y`/`y**2` basis and emits exact `0.0` for whichever term lost a label's decomposition. A zero therefore means "the partner term won here", not "this term is absent", so the prior must let the posterior re-decide. Gain carries signal units and has no universal around-zero width, so a zeroed label borrows one from the labels that did resolve a gain.

**Tech Stack:** Python 3.14, PyMC, PyTensor, NumPy, pytest, mypy, ruff.

**Spec:** `docs/superpowers/specs/2026-07-18-noise-prior-symmetry-design.md`

## Global Constraints

- Numpy-style docstrings on every public function; the repo runs `pydoclint` in pre-commit.
- Type hints on all new functions; `make type` must pass (`uv run mypy src tests docs/conf.py`).
- Formatting and linting via `ruff` only; pre-commit also runs `mdformat` on markdown.
- Never reformat code unrelated to the task.
- Commits must follow Conventional Commit format — pre-commit enforces it.
- Tests requiring PyMC must start with `pytest.importorskip("pymc")`, matching the existing tests in `tests/test_bayes.py`.
- `_MIN_NOISE_PRIOR_SCALE = 1e-3` (`bayes.py:63`) stays the floor for **alpha** widths only. It must never be applied to gain — against a gain of ~1.6 it is 0.06%, a hard zero in disguise.
- Gain uses **two** width factors, not one reused constant: `0.2 * mu_g` for a *resolved* hint (known to about 20%), and a new module-level constant `_ZERO_HINT_GAIN_WIDTH = 1.0`, applied as `_ZERO_HINT_GAIN_WIDTH * plate_gain_scale`, for an *unresolved* (exactly-zero) hint. Reusing `0.2` for the zero-hint case makes the prior far too tight — `HalfNormal(sigma=0.2 * 1.6 = 0.32)` puts the value the collinear partner could equally have taken (1.6) about five sigma out, so the posterior cannot recover the split. **This supersedes the original constraint below, which forbade a new module-level constant for the gain width; that constraint was based on treating both cases as the same relative width, which is incorrect.**
- ~~The relative width constant for gain is `0.2`, reused from the existing positive-hint branches. Do not introduce a new module-level constant for it.~~ (superseded, see above)

______________________________________________________________________

### Task 1: Pin the gate invariant and document why it is asymmetric

Gain is omitted entirely when no label resolved a positive gain; alpha is not. This is correct and must stay correct, because Task 2 relies on it: it is what guarantees the borrowed width is never zero. There is no behaviour change in this task — it adds a characterization test and a comment.

**Files:**

- Modify: `src/clophfit/fitting/bayes.py:146-148`
- Test: `tests/test_bayes.py`

**Interfaces:**

- Consumes: nothing.

- Produces: the invariant `"gain" in priors implies some label has gain > 0`, relied on by Task 2.

- [ ] **Step 1: Write the characterization test**

This test pins behaviour that already exists, so it passes immediately. That is intentional — it is a regression guard for Task 2's precondition, not a TDD red step.

Append to `tests/test_bayes.py`:

```python
def test_gain_omitted_when_no_label_resolves_a_gain() -> None:
    """No positive gain anywhere -> the Poisson term is omitted, not invented.

    Gain carries the units of the signal, so unlike alpha there is no
    plate-independent around-zero width to fall back on. This invariant is what
    guarantees the zeroed-gain branch never borrows a width of zero.
    """
    pytest.importorskip("pymc")
    nm = PlateNoiseModel({
        "1": NoiseModelParams(sigma_floor=1.0, gain=0.0, alpha=0.03),
        "2": NoiseModelParams(sigma_floor=1.0, gain=0.0, alpha=0.02),
    })
    with pm.Model():
        priors = bayes.build_pymc_noise_priors(
            nm, gain_mode="centered", alpha_mode="centered"
        )
    # Gain has nothing to borrow from, so it is absent.
    assert "gain" not in priors
    # Alpha is dimensionless and always has _MIN_NOISE_PRIOR_SCALE to fall back
    # on, so it stays present even though every label calibrated to a positive
    # value here.
    assert "rel_error" in priors
```

- [ ] **Step 2: Run the test to verify it passes**

Run: `uv run pytest tests/test_bayes.py::test_gain_omitted_when_no_label_resolves_a_gain -v`

Expected: PASS. This is a characterization test — it pins behaviour that already exists. If it fails, stop: the premise of Task 2 is wrong and the spec needs revisiting.

- [ ] **Step 3: Add the explanatory comment**

In `src/clophfit/fitting/bayes.py`, replace lines 146-148:

```python
    # 2. Gain (Poisson term)
    has_gain = any(p.gain > 0 for p in noise_model.values())
    if has_gain or gain_mode == "free":
```

with:

```python
    # 2. Gain (Poisson term). The gate is deliberately narrower than alpha's
    # below. Alpha is dimensionless, so _MIN_NOISE_PRIOR_SCALE is a meaningful
    # universal around-zero width and alpha can always stay soft. Gain carries
    # the units of the signal, so its around-zero width has to be borrowed from
    # labels that did resolve a gain; when no label resolved one there is
    # nothing to borrow, and omitting the term beats inventing a scale. Do not
    # "symmetrise" this gate — it is what guarantees plate_gain_scale > 0.
    has_gain = any(p.gain > 0 for p in noise_model.values())
    if has_gain or gain_mode == "free":
```

- [ ] **Step 4: Verify nothing broke**

Run: `uv run pytest tests/test_bayes.py -q && make type`

Expected: all tests pass, mypy reports `Success: no issues found`.

- [ ] **Step 5: Commit**

```bash
git add src/clophfit/fitting/bayes.py tests/test_bayes.py
git commit -m "test: pin gain-gate invariant and document its asymmetry"
```

______________________________________________________________________

### Task 2: Make a zeroed gain estimable and unify the prior widths

This is the core change. It replaces the hard `pt.as_tensor_variable(0.0)` with a `HalfNormal` whose width is borrowed from the plate, and routes both the shared and per-label branches through one helper so their 10x-divergent floors disappear.

**Files:**

- Modify: `src/clophfit/fitting/bayes.py` — add helper after `_build_floor_prior` (ends line 93); rewrite the gain block at lines 146-181
- Test: `tests/test_bayes.py`

**Interfaces:**

- Consumes: the `has_gain` gate invariant from Task 1.

- Produces: `_gain_prior_sigma(mu_g: float, plate_gain_scale: float) -> float`, a module-private helper in `bayes.py`.

- [ ] **Step 1: Write the tests**

The first two are TDD red steps and must fail before Step 3. The third is a characterization test that passes immediately, guarding `fixed` mode against the change.

Append to `tests/test_bayes.py`:

```python
def test_zeroed_gain_borrows_width_from_resolved_labels() -> None:
    """An exact-zero gain stays estimable, scaled by the labels that resolved one.

    NNLS clamps the collinear y/y**2 basis, so gain=0.0 on one label means alpha
    won that label's decomposition -- not that the Poisson term is absent. The
    prior must let the posterior re-decide.
    """
    pytest.importorskip("pymc")
    nm = PlateNoiseModel({
        "1": NoiseModelParams(sigma_floor=1.0, gain=0.0, alpha=0.02),
        "2": NoiseModelParams(sigma_floor=1.0, gain=1.6, alpha=0.0),
    })
    with pm.Model():
        priors = bayes.build_pymc_noise_priors(nm, gain_mode="centered")
        draws = pm.draw(priors["gain"]["1"], draws=8000, random_seed=0)
    # Non-degenerate and non-negative: a sampled variable, not a hard constant.
    assert float(draws.std()) > 0.0
    assert float(draws.min()) >= 0.0
    # An unresolved (zero) hint uses _ZERO_HINT_GAIN_WIDTH * plate_gain_scale,
    # not the 20% relative width used for resolved hints:
    # HalfNormal(sigma=1.0 * 1.6 = 1.6); its mean is sigma * sqrt(2/pi) ~= 1.2767.
    # abs=0.08 is roughly 7 sampling standard errors at draws=8000.
    assert float(draws.mean()) == pytest.approx(1.6 * np.sqrt(2 / np.pi), abs=0.08)


def test_gain_prior_width_agrees_between_shared_and_per_label() -> None:
    """One hint gives one width, whether the gain is pooled or per-label.

    The shared branch used to floor sigma at 0.1 and the per-label branch at
    0.01 -- a 10x disagreement with no rationale.
    """
    pytest.importorskip("pymc")
    nm = PlateNoiseModel({"1": NoiseModelParams(sigma_floor=1.0, gain=0.3)})
    with pm.Model():
        shared = bayes.build_pymc_noise_priors(
            nm, shared_gain=True, gain_mode="centered"
        )
        shared_std = float(pm.draw(shared["gain"], draws=8000, random_seed=0).std())
    with pm.Model():
        per_label = bayes.build_pymc_noise_priors(
            nm, shared_gain=False, gain_mode="centered"
        )
        per_std = float(pm.draw(per_label["gain"]["1"], draws=8000, random_seed=0).std())
    # Both are 0.2 * 0.3 = 0.06 now; previously 0.1 (shared) vs 0.06 (per-label).
    assert shared_std == pytest.approx(per_std, rel=0.05)
    assert shared_std == pytest.approx(0.06, abs=0.006)


def test_fixed_mode_keeps_hard_constants_for_both_terms() -> None:
    """Mode ``fixed`` is the one mode where a zero genuinely means absent.

    The softening of zeroed gains must not leak into "fixed", which callers use
    to pin or disable a term outright.
    """
    pytest.importorskip("pymc")
    nm = PlateNoiseModel({
        "1": NoiseModelParams(sigma_floor=1.0, gain=0.0, alpha=0.02),
        "2": NoiseModelParams(sigma_floor=1.0, gain=1.6, alpha=0.0),
    })
    with pm.Model():
        priors = bayes.build_pymc_noise_priors(
            nm, gain_mode="fixed", alpha_mode="fixed"
        )
    # Constants, not sampled variables: they evaluate to the hint exactly and
    # carry no randomness.
    assert float(priors["gain"]["1"].eval()) == 0.0
    assert float(priors["gain"]["2"].eval()) == pytest.approx(1.6)
    assert float(priors["rel_error"]["1"].eval()) == pytest.approx(0.02)
    assert float(priors["rel_error"]["2"].eval()) == 0.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_bayes.py -k "zeroed_gain_borrows or width_agrees or fixed_mode_keeps" -v`

Expected: two FAIL, one PASS.

- `test_zeroed_gain_borrows_width_from_resolved_labels` FAILS inside `pm.draw` — `priors["gain"]["1"]` is currently the constant `0.0`, so `draws.std()` is `0.0`.

- `test_gain_prior_width_agrees_between_shared_and_per_label` FAILS on the first assertion, with `shared_std` near `0.1` against `per_std` near `0.06`.

- `test_fixed_mode_keeps_hard_constants_for_both_terms` PASSES already — it is a characterization test guarding against regression in Step 4.

- [ ] **Step 3: Add the helper**

In `src/clophfit/fitting/bayes.py`, insert after `_build_floor_prior` ends (line 93) and before `build_pymc_noise_priors`:

```python
# Width factor for a *zeroed* gain hint, in units of plate_gain_scale. A
# resolved gain hint is known to about 20%, but a hint of exactly 0.0 is not a
# measurement of zero -- it means the collinear alpha term won this label's
# NNLS decomposition, so the width must span the plausible range of the
# plate's gains rather than a tight band around zero.
_ZERO_HINT_GAIN_WIDTH = 1.0


def _gain_prior_sigma(mu_g: float, plate_gain_scale: float) -> float:
    """Prior width for one gain hint.

    Gain carries the units of the signal, so its width is always relative, but
    the two cases mean different things and use different factors. A
    *resolved* hint (``mu_g > 0``) is a real calibrated value known to about
    20% (``0.2 * mu_g``). An *unresolved* hint (``mu_g == 0``) is not a
    measurement of zero: it comes from the NNLS boundary and means the
    collinear alpha term won this label's decomposition, so the width instead
    has to span the plausible range of the plate's gains
    (``_ZERO_HINT_GAIN_WIDTH * plate_gain_scale``) rather than being a tight
    band around zero.

    Parameters
    ----------
    mu_g : float
        This label's calibrated gain hint.
    plate_gain_scale : float
        Mean of the positive gains on the plate, used when *mu_g* is 0. The
        caller's ``has_gain`` gate guarantees this is non-zero whenever a
        zero hint can reach here.

    Returns
    -------
    float
        Standard deviation for the gain prior.
    """
    if mu_g > 0.0:
        return 0.2 * mu_g
    return _ZERO_HINT_GAIN_WIDTH * plate_gain_scale
```

- [ ] **Step 4: Rewrite the gain block**

In `src/clophfit/fitting/bayes.py`, replace the body of the gain block (the lines from `if shared_gain:` through `priors["gain"][lbl] = pt.as_tensor_variable(0.0)`, currently lines 149-181) with:

```python
        positive_gains = [p.gain for p in noise_model.values() if p.gain > 0]
        plate_gain_scale = float(np.mean(positive_gains)) if positive_gains else 0.0
        if shared_gain:
            mu_g = plate_gain_scale
            if gain_mode == "fixed":
                priors["gain"] = pt.as_tensor_variable(mu_g)
            elif gain_mode == "free":
                # Hint sets the Exponential prior mean (= 1/lam), floored so a 0
                # hint is the tightest around-zero prior (Poisson term ~off).
                lam = 1.0 / max(mu_g, _MIN_NOISE_PRIOR_SCALE)
                priors["gain"] = pm.Exponential("gain", lam=lam)
            else:  # centered
                # In the shared branch mu_g IS plate_gain_scale (set just
                # above), so this can only ever take the resolved-hint arm of
                # _gain_prior_sigma; the zero-hint fallback is per-label only.
                priors["gain"] = pm.TruncatedNormal(
                    "gain",
                    mu=mu_g,
                    sigma=_gain_prior_sigma(mu_g, plate_gain_scale),
                    lower=0.0,
                )
        else:
            priors["gain"] = {}
            for lbl in labels:
                mu_g = getattr(noise_model[lbl], "gain", 0.0)
                if gain_mode == "fixed":
                    priors["gain"][lbl] = pt.as_tensor_variable(mu_g)
                elif gain_mode == "free":
                    # Hint sets the Exponential mean; a 0 hint -> tightest (~off).
                    lam = 1.0 / max(mu_g, _MIN_NOISE_PRIOR_SCALE)
                    priors["gain"][lbl] = pm.Exponential(f"gain_{lbl}", lam=lam)
                elif mu_g > 0.0:
                    priors["gain"][lbl] = pm.TruncatedNormal(
                        f"gain_{lbl}",
                        mu=mu_g,
                        sigma=_gain_prior_sigma(mu_g, plate_gain_scale),
                        lower=0.0,
                    )
                else:
                    # Exact 0 from the NNLS boundary: alpha won this label's
                    # decomposition. Keep the term estimable around 0 with a
                    # width spanning the plate's gain scale (not a fraction of
                    # it -- see _ZERO_HINT_GAIN_WIDTH), so the posterior can
                    # undo an arbitrary collinear split.
                    priors["gain"][lbl] = pm.HalfNormal(
                        f"gain_{lbl}",
                        sigma=_gain_prior_sigma(mu_g, plate_gain_scale),
                    )
```

Note the shared branch now derives `mu_g` from `plate_gain_scale` instead of recomputing the same mean into a local `gains` list. The `fixed` branches are untouched — `fixed` is the one mode where a zero genuinely means absent.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_bayes.py -k "zeroed_gain_borrows or width_agrees or fixed_mode_keeps" -v`

Expected: all three PASS.

- [ ] **Step 6: Run the full suite and typecheck**

Run: `uv run pytest tests/test_bayes.py -q && make type`

Expected: all pass. In particular `test_centered_zero_alpha_is_prior_around_zero` and `test_free_noise_priors_scale_from_hints` must still pass — neither plate in those tests has a zeroed gain alongside a positive one, so this task does not touch them.

- [ ] **Step 7: Commit**

```bash
git add src/clophfit/fitting/bayes.py tests/test_bayes.py
git commit -m "fix(bayes): keep a zeroed gain estimable and unify prior widths"
```

______________________________________________________________________

### Task 3: Make the free-mode hint the prior mean for both terms

`Exponential(lam=1/h)` has mean exactly `h`; `HalfNormal(sigma=h)` has mean `0.798*h`. The same hint means two different things. Alpha moves to match gain.

**Files:**

- Modify: `src/clophfit/fitting/bayes.py:197-199` (shared) and `bayes.py:214-216` (per-label) — line numbers shift by Task 2; locate by content
- Modify: `src/clophfit/fitting/bayes_config.py:283-288` (docstring)
- Test: `tests/test_bayes.py:720-726` (update existing assertions)

**Interfaces:**

- Consumes: nothing from Tasks 1-2.

- Produces: nothing consumed later.

- [ ] **Step 1: Update the existing test to the new contract**

In `tests/test_bayes.py`, inside `test_free_noise_priors_scale_from_hints`, replace lines 720-726:

```python
    # Hinted: Exponential mean == gain hint (0.5); HalfNormal mean == sigma*sqrt(2/pi).
    assert gain_mean == pytest.approx(0.5, abs=0.05)
    assert alpha_mean == pytest.approx(0.03 * np.sqrt(2 / np.pi), abs=0.005)
    # gain=0 / alpha=0 -> tightest around-zero priors (floored at 1e-3), so each
    # width is strictly *below* a small positive hint (monotonic in the hint).
    assert gain0_mean == pytest.approx(1e-3, abs=5e-4)
    assert alpha0_mean == pytest.approx(1e-3 * np.sqrt(2 / np.pi), abs=5e-4)
```

with:

```python
    # The hint is the prior *mean* for both terms: Exponential(lam=1/h) and
    # HalfNormal(sigma=h*sqrt(pi/2)) both have mean h.
    assert gain_mean == pytest.approx(0.5, abs=0.05)
    assert alpha_mean == pytest.approx(0.03, abs=0.005)
    # gain=0 / alpha=0 -> tightest around-zero priors (floored at 1e-3), so each
    # width is strictly *below* a small positive hint (monotonic in the hint).
    assert gain0_mean == pytest.approx(1e-3, abs=5e-4)
    assert alpha0_mean == pytest.approx(1e-3, abs=5e-4)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_bayes.py::test_free_noise_priors_scale_from_hints -v`

Expected: FAIL on the `alpha_mean` assertion — currently ~0.0239 against an expected 0.03.

- [ ] **Step 3: Change both free-mode alpha priors**

In `src/clophfit/fitting/bayes.py`, in the shared alpha branch replace:

```python
            elif alpha_mode == "free":
                priors["rel_error"] = pm.HalfNormal(
                    "rel_error", sigma=max(mu_a, _MIN_NOISE_PRIOR_SCALE)
                )
```

with:

```python
            elif alpha_mode == "free":
                # sqrt(pi/2) converts the hint from a scale to a mean, so the
                # hint means the same thing here as it does for gain's
                # Exponential (whose mean is exactly its hint).
                priors["rel_error"] = pm.HalfNormal(
                    "rel_error",
                    sigma=max(mu_a, _MIN_NOISE_PRIOR_SCALE) * np.sqrt(np.pi / 2),
                )
```

and in the per-label alpha branch replace:

```python
                elif alpha_mode == "free":
                    priors["rel_error"][lbl] = pm.HalfNormal(
                        f"rel_error_{lbl}", sigma=max(mu_a, _MIN_NOISE_PRIOR_SCALE)
                    )
```

with:

```python
                elif alpha_mode == "free":
                    # See the shared branch: sqrt(pi/2) makes the hint a mean.
                    priors["rel_error"][lbl] = pm.HalfNormal(
                        f"rel_error_{lbl}",
                        sigma=max(mu_a, _MIN_NOISE_PRIOR_SCALE) * np.sqrt(np.pi / 2),
                    )
```

Leave the `centered` alpha branches alone — they use `TruncatedNormal(mu=mu_a, ...)`, where the hint is already the centre.

- [ ] **Step 4: Update the block comment above the alpha section**

In `src/clophfit/fitting/bayes.py`, in the comment beginning `# 3. Alpha (proportional error).`, replace this sentence:

```
    # is the prior scale (HalfNormal sigma / TruncatedNormal mean), floored at
```

with:

```
    # is the prior mean in every mode (HalfNormal sigma is scaled by sqrt(pi/2)
    # to achieve that; TruncatedNormal takes it directly), floored at
```

- [ ] **Step 5: Update the public docstring**

In `src/clophfit/fitting/bayes_config.py`, in `NoiseConfig.structured`, replace:

```
        The *alpha* hint is the prior scale, not a hard value: in ``"free"``
        mode it is the ``HalfNormal`` sigma and in ``"centered"`` mode the
        ``TruncatedNormal`` mean. It defaults to ``0.02`` (a weak 2% prior);
```

with:

```
        The *alpha* hint is the prior mean, not a hard value: it is the
        ``TruncatedNormal`` centre in ``"centered"`` mode, and in ``"free"``
        mode the ``HalfNormal`` sigma is scaled so the mean matches the hint.
        It defaults to ``0.02`` (a weak 2% prior);
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_bayes.py -q && make type`

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/clophfit/fitting/bayes.py src/clophfit/fitting/bayes_config.py tests/test_bayes.py
git commit -m "fix(bayes): make the free-mode noise hint a prior mean for both terms"
```

______________________________________________________________________

### Task 4: Full-suite regression and real-plate validation

The three code tasks change posteriors. This task confirms nothing else in the package depended on the old behaviour, then measures the effect on real plates.

**Files:**

- No source changes expected. If a failure requires one, stop and report rather than adjusting an assertion to match new output.

**Interfaces:**

- Consumes: Tasks 1-3 complete.

- Produces: a validation note for the PR description.

- [ ] **Step 1: Run the whole test suite**

Run: `uv run pytest -q`

Expected: all pass. Failures outside `tests/test_bayes.py` mean a caller depended on the old prior shapes — report the failure, do not silence it.

- [ ] **Step 2: Lint and typecheck**

Run: `uv run ruff check src tests && uv run ruff format --check src tests && make type`

Expected: clean.

- [ ] **Step 3: Confirm the changed branch actually runs end-to-end**

Run:

```bash
uv run python -c "
import numpy as np, pymc as pm
from clophfit.fitting import bayes
from clophfit.fitting.data_structures import PlateNoiseModel, NoiseModelParams
nm = PlateNoiseModel({
    '1': NoiseModelParams(sigma_floor=1.0, gain=0.0, alpha=0.02),
    '2': NoiseModelParams(sigma_floor=1.0, gain=1.6, alpha=0.0),
})
with pm.Model():
    p = bayes.build_pymc_noise_priors(nm, gain_mode='centered')
    d = pm.draw(p['gain']['1'], draws=4000, random_seed=0)
print('gain_1 mean', d.mean(), 'std', d.std(), 'min', d.min())
"
```

Expected: a mean near `1.6 * sqrt(2/pi)` ~= 1.2767, a non-zero std, and a min of 0.0 or above. A std of exactly 0 means the hard constant is still in place.

- [ ] **Step 4: Real-plate validation**

Run: `./scripts/compare_methods.sh L2 noise`

Compare posterior Kd and its 94% HDI against the pre-change values.

Expected: gain uncertainty widens on labels that calibrated to zero; Kd point estimates stay roughly stable. A large Kd shift is a signal to re-examine, not to accept. Watch the divergence count — making a previously-frozen parameter estimable can expose funnel geometry the hard zero was hiding.

If divergences appear where there were none, record it and stop for review. That is a real finding about the model, not a reason to revert the prior.

- [ ] **Step 5: Commit any validation notes**

If `scripts/` outputs are worth keeping, add them deliberately. Otherwise commit nothing here and report results in the PR description.

______________________________________________________________________

## Notes for the implementer

**Do not "symmetrise" the two gates.** `has_gain or gain_mode == "free"` for gain versus `has_alpha or alpha_mode in {"free", "centered"}` for alpha looks like an oversight and is not. Task 1 adds a comment saying so, and `test_centered_zero_alpha_is_prior_around_zero` (`tests/test_bayes.py:731`) plus `test_gain_omitted_when_no_label_resolves_a_gain` guard both directions. An earlier draft of the spec proposed narrowing alpha's gate and was withdrawn — see commit `27678c9`.

**Do not include zeros when pooling.** The `if p.gain > 0` and `if p.alpha > 0` filters in the shared branches are deliberate. The zeros are anti-correlated artifacts of a single collinear decomposition, so pooling alpha over `{0.02, 0.0}` gives 0.01 and understates it.

**Do not add a tolerance to the `> 0` tests.** With floors held fixed, `scipy.optimize.nnls` returns exact `0.0` at the boundary, so `> 0` is the correct predicate.
