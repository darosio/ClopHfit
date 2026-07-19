# Noise prior symmetry: zeroed terms stay estimable

- **Date:** 2026-07-18
- **Status:** approved, pending implementation plan
- **Scope:** `clophfit.fitting.bayes` (`build_pymc_noise_priors`,
  `get_pymc_variance`), `clophfit.fitting.bayes_config`
  (`NoiseConfig.structured` docstring), `tests/test_bayes.py`

Both touched functions are public API, so the behaviour changes below are
observable by callers, not internal-only.

## Problem

`build_pymc_noise_priors` (`bayes.py:96`) turns a calibrated `PlateNoiseModel`
into PyMC priors for the three variance terms of

```
sigma^2 = sigma_floor^2 + gain * y + alpha^2 * y^2
```

The gain and alpha branches were written independently and disagree in four
ways. One of them silently destroys information on the plates we actually fit.

### Where the zeros come from

`Titration.fit_plate`'s FGLS loop calibrates with floors held fixed
(`titration.py:1339`):

```python
floors, gains, alphas = fit_noise_model_nnls(df_res, sigma_floor_fixed=floors_in)
```

With floors fixed, `fit_noise_model_nnls` regresses squared residuals on
`column_stack([y, y**2])` through `scipy.optimize.nnls`
(`noise_calibration.py:279-281`), reading `gain = coeffs[0]` and
`alpha = sqrt(coeffs[1])`. Because `y` and `y**2` are strongly collinear over a
7-point titration, NNLS routinely puts all excess variance on one basis vector
and clamps the other to the boundary. The active-set solver returns **exact**
`0.0` there, so the surviving zeros are exact and the existing `> 0` predicates
detect them correctly. No tolerance-based test is needed.

**A zero therefore means "the collinear partner won this label's
decomposition", not "this term is physically absent."** That reading drives
every decision below, and it survived validation. Which *term* zeroes did not.

Measured on plates L2, L3 and L4 via `fgls_fit_plate`, floors fixed from
`bg_noise`:

| plate | label | gain   | alpha           |
| ----- | ----- | ------ | --------------- |
| L2    | 1     | 5.5349 | **exactly 0.0** |
| L2    | 2     | 0.5155 | **exactly 0.0** |
| L3    | 1     | 2.1095 | 0.008349        |
| L3    | 2     | 0.4046 | **exactly 0.0** |
| L4    | 1     | 4.9315 | 0.10593         |
| L4    | 2     | 1.3413 | **exactly 0.0** |

**Alpha is the term that lands on the boundary — 4 of 6 label-fits, always
label 2 — and gain never does (0 of 6).** Alpha reaches ~0.106; gain is O(1)
to O(5).

An earlier draft of this spec asserted the opposite, from a remembered example:
an *anti-correlated* pattern with label 1 at `alpha=0.02, gain=0.0` and label 2
at `alpha=0.0, gain=1.6`. No such pattern occurs on any of the three plates.
The consequence is recorded in section 6.

Two further paths produce exact zeros for *both* terms at once: the calibration
failure fallback (`titration.py:1344`) and the per-label short-circuit when
fewer than two residuals exceed the floor (`noise_calibration.py:274`).

### The asymmetries

Four were identified. One (the gates) turned out to be justified rather than
defective; see design section 1.

1. **A zeroed gain is unrecoverable; a zeroed alpha is not.** In centered mode
   the per-label gain branch emits `pt.as_tensor_variable(0.0)`
   (`bayes.py:181`) — a hard constant with no posterior. The per-label alpha
   branch emits `HalfNormal(sigma=1e-3)` (`bayes.py:225`), which stays
   estimable. Alpha's own comment (`bayes.py:183-188`) states the intended
   principle — "a calibrated alpha of 0 becomes a tight prior around 0, not a
   hard 0; only 'fixed' leaves the term truly absent" — and the gain branch
   violates it.

   On the table above this is not an edge case but the **dominant** one: label
   2's `gain=1.6` opens the gate, so label 1's exact zero reaches `bayes.py:181`
   and is frozen. The arbitrary side of a collinear split becomes permanent.

1. **The gates differ** — *and this one is correct as-is.* Gain builds priors
   when `has_gain or gain_mode == "free"` (`bayes.py:148`); alpha builds them
   when `has_alpha or alpha_mode in {"free", "centered"}` (`bayes.py:190`). So
   a plate whose every label has `gain == 0.0` omits the gain term entirely,
   while the same plate in alpha keeps a tight estimable prior. Design section 1
   explains why this is forced rather than accidental, and leaves it unchanged.

1. **Gain disagrees with itself on prior width.** The shared branch floors sigma
   at `0.1` (`bayes.py:161`), the per-label branch at `0.01` (`bayes.py:177`).
   A 10x difference with no stated rationale. Impact is modest — at `gain=1.6`
   the relative term `0.2 * mu_g = 0.32` dominates both floors, so the floors
   only bind below `gain=0.5` (shared) or `gain=0.05` (per-label).

1. **The free-mode hint means different things.** Gain uses
   `Exponential(lam=1/mu_g)`, whose mean is exactly `mu_g`. Alpha uses
   `HalfNormal(sigma=mu_a)`, whose mean is `0.798 * mu_a`. The same number is a
   mean in one branch and a scale in the other; `NoiseConfig.structured`
   documents it as a scale (`bayes_config.py:283-288`).

`Exponential` for gain versus `HalfNormal` for alpha is **not** counted as an
asymmetry. Different positive-scale parameterizations are a legitimate modelling
choice and are deliberately preserved.

## Design

Guiding principle: **the calibrated point estimate is a hint, not a verdict.**
Because the zeros are collinearity artifacts, the prior must let the posterior
re-decide. A term is switched off only when there is no way to give it a
sensible prior width — which, per section 1, means gain with no positive gain
anywhere on the plate.

### 1. The gates stay as they are — documentation only

**No code change.** Both gates are already correct, for a reason that was never
written down. The asymmetry is forced by the two parameters having different
dimensions.

Alpha is dimensionless and empirically below ~0.1 on every plate, so a
*universal* fallback width exists and alpha can always stay soft, even when no
label resolved a positive value.

(As first drafted this section named `_MIN_NOISE_PRIOR_SCALE = 1e-3` as that
width. Section 6 shows 1e-3 is far too tight — 100 sigma below a plausible
alpha — and replaces it with `_ZERO_HINT_ALPHA_SCALE = 0.1`. The argument here
is unaffected: what matters is that *some* defensible universal width exists
for a dimensionless parameter, not which one.)

Gain carries the units of the signal and ranges over orders of magnitude across
instruments and plate types. No universal constant exists. Its around-zero width
must be *borrowed* from labels that did resolve a gain (section 2), so when no
label resolved one there is nothing to borrow, and omitting the term is the only
defensible option. `has_gain` expresses exactly that condition.

This corrects a mistake made when the spec was first drafted. Section 1
originally narrowed alpha's gate to match gain's, on the premise that an
all-zero term means calibration found no structure. That premise is false: a
term is zero across every label whenever its collinear partner won *every*
label's decomposition, which is a routine outcome, not a failure. The existing
test `test_centered_zero_alpha_spans_plate_alpha_scale` (renamed in section 6;
originally `test_centered_zero_alpha_is_prior_around_zero`)
pins precisely that case — `alpha=0.0` on both labels with `gain` at 2.0 and 1.0
— and asserts alpha stays estimable. The test is right; the original section 1
would have broken it, and would have contradicted this spec's own guiding
principle.

Consequence for section 2: because the `has_gain` gate is retained, at least one
label has a positive gain whenever the zeroed-gain branch runs, so the borrowed
`plate_gain_scale` is guaranteed non-zero.

The downstream gates at `bayes.py:286` and `bayes.py:294` are likewise unchanged
and safe: `has_X` is the `any()` over the same values, so `params.X > 0` can
never be true when the key is absent.

Deliverable for this section is a code comment at `bayes.py:148` recording why
gain's gate is narrower than alpha's, so the asymmetry is not "fixed" later.

### 2. A zeroed gain becomes estimable

`bayes.py:181` changes from a hard constant to a prior whose width is borrowed
from the labels that *did* resolve a gain. This requires two distinct width
factors, not one reused constant, because a resolved hint and an unresolved
(zero) hint mean fundamentally different things:

```python
_ZERO_HINT_WIDTH = 1.0  # module-level, alongside _MIN_NOISE_PRIOR_SCALE

plate_gain_scale = mean(p.gain for p in noise_model.values() if p.gain > 0)
sigma = 0.2 * mu_g if mu_g > 0.0 else _ZERO_HINT_WIDTH * plate_gain_scale
priors["gain"][lbl] = pm.HalfNormal(f"gain_{lbl}", sigma=sigma)
```

A *resolved* hint (`mu_g > 0`) is a real calibrated value, known to about 20%,
so `0.2 * mu_g` is the right width — this is unchanged from the positive-hint
branches at `bayes.py:161,177`. A hint of exactly `0.0` is not a measurement of
zero: it means the collinear alpha term won this label's NNLS decomposition,
so the true gain could plausibly be anywhere up to the plate's other gains.
Reusing the same `0.2` factor there would make the prior far too tight — at
`plate_gain_scale = 1.6`, `HalfNormal(sigma=0.2 * 1.6 = 0.32)` puts the value
NNLS could equally have assigned this label (1.6) about five sigma out, with
~6e-7 of the prior mass — the posterior cannot recover the split, defeating
the point of making the term estimable at all. `_ZERO_HINT_WIDTH = 1.0`
instead spans the plate's gain scale directly: `sigma = 1.0 * 1.6 = 1.6`, wide
enough that the posterior can genuinely re-decide the split. This does
introduce one new module-level constant, alongside `_MIN_NOISE_PRIOR_SCALE`;
that is the accepted cost of giving the two semantics their own knobs instead
of overloading `0.2`.

For the real-plate table, label 1's zeroed gain gets `sigma = 1.0 * 1.6 = 1.6`
— the collinear partner supplies the scale.

The structure now mirrors alpha exactly: `HalfNormal` at a zero hint,
`TruncatedNormal` at a positive one.

Mirroring alpha's *constant* would not work. `_MIN_NOISE_PRIOR_SCALE = 1e-3`
(`bayes.py:63`) is a sensible width for a dimensionless alpha of ~0.02, but
against a gain of ~1.6 it is 0.06% — a hard zero in all but name. Gain needs a
gain-scaled width.

The `has_gain` gate retained in section 1 guarantees at least one positive gain
exists whenever this branch runs, so `plate_gain_scale` is never zero and needs
no fallback. That guarantee is the reason section 1 must leave the gate alone —
widening it would admit `plate_gain_scale == 0` and a degenerate zero-sigma
`HalfNormal`.

### 3. The width floors dissolve

Both absolute floors are deleted rather than reconciled. One helper serves the
shared and per-label branches:

```python
def _gain_prior_sigma(mu_g: float, plate_gain_scale: float) -> float:
    """Relative prior width, falling back to the plate scale at a zero hint."""
    return 0.2 * (mu_g if mu_g > 0 else plate_gain_scale)
```

The floors existed only to stop `sigma -> 0` as `mu_g -> 0`; section 2 now
handles zero explicitly, so they have no remaining job. Shared and per-label
agree by construction, resolving asymmetry 3.

Behaviour change: shared gain with `mu_g < 0.5` gets a tighter prior than the
`0.1` floor gave it.

### 4. The hint is the prior mean

Alpha's free-mode `HalfNormal` sigma becomes `hint * sqrt(pi/2)`, so its mean
equals the hint and matches `Exponential`. A calibrated `alpha=0.02` then
centers at 0.02 rather than 0.016.

Applies to both free-mode alpha sites (`bayes.py:197-199` shared,
`bayes.py:214-216` per-label). `NoiseConfig.structured`'s docstring changes from
"the alpha hint is the prior scale" to "the prior mean".

`_MIN_NOISE_PRIOR_SCALE` keeps its role as the floor on alpha's width.

### 5. Zero-pooling is left unchanged

No change to the `if p.gain > 0` / `if p.alpha > 0` filters in the shared
branches (`bayes.py:150,192`), nor to the `shared_floor` filter
(`bayes.py:136`).

This inverts the original review, which argued a clamped zero is real evidence
that should pull the pooled mean down. Under the generating process above the
zeros are **anti-correlated artifacts of a single collinear decomposition**, not
independent measurements. Pooling alpha over `{0.02, 0.0}` yields 0.01 and
understates it, because label 2's zero reports which basis vector won, not the
absence of proportional error.

The `shared_floor` filter is separately fine: floors are fixed from
`tit.bg_noise` and are never zero, so it guards a case that should not arise. A
read-noise floor of exactly zero is not an attainable measurement, so treating
it as missing is correct.

### 6. A zeroed alpha spans the plate's alpha scale

Added after implementation, once the measured data in "Where the zeros come
from" showed that **alpha**, not gain, is the term that actually zeroes.

The defect is the exact analogue of section 2, in the branch that fires on real
plates. A calibrated alpha of `0.0` in centered mode built
`HalfNormal(sigma=_MIN_NOISE_PRIOR_SCALE)` = `HalfNormal(1e-3)`. Against a
plausible alpha of 0.1 that is **100 sigma** — an order of magnitude worse than
the 5-sigma gain case section 2 was written to fix. On L2, where *both* labels
zeroed, every alpha prior was that tight.

The fix mirrors section 2 exactly:

```python
_ZERO_HINT_WIDTH = 1.0        # renamed from _ZERO_HINT_GAIN_WIDTH; now serves both
_ZERO_HINT_ALPHA_SCALE = 0.1  # plate scale when NO label resolved a positive alpha

positive_alphas = [p.alpha for p in noise_model.values() if p.alpha > 0]
plate_alpha_scale = mean(positive_alphas) if positive_alphas else _ZERO_HINT_ALPHA_SCALE
sigma = max(0.25 * mu_a, _MIN_NOISE_PRIOR_SCALE) if mu_a > 0 else _ZERO_HINT_WIDTH * plate_alpha_scale
```

Alpha needs the `_ZERO_HINT_ALPHA_SCALE` fallback where gain needs none, and
section 1 explains why: gain's `has_gain` gate omits the term entirely when no
label resolved one, while alpha's gate keeps building a prior. `0.1` is chosen
so the observed maximum (0.106) sits at about one sigma, mirroring how
`_ZERO_HINT_WIDTH` places `plate_gain_scale` at one sigma for gain.

Resolved-hint behaviour is unchanged in both terms.

This is the change that actually moves posteriors on L2–L4. Section 2's gain
path, though correct, is not reached on those plates.

## Testing

Unit tests in `tests/test_bayes.py`, each pinning one branch:

- **Mixed plate** (`alpha=0.02/gain=0` and `alpha=0/gain=1.6`, centered mode):
  label 1's gain is a sampled `HalfNormal`, not a constant, with
  `sigma == _ZERO_HINT_WIDTH * 1.6 == 1.6` — the unresolved-hint width, not the
  20% resolved-hint one. Guards section 2. Note this plate is **synthetic**: no
  measured plate produces a zeroed gain, so the test pins the branch's
  behaviour rather than a case seen in the data.
- **Zeroed alpha borrows from a resolved label** (modelled on real L4:
  `gain=4.93/alpha=0.106` and `gain=1.34/alpha=0.0`, centered mode): label 2's
  alpha is a `HalfNormal` with `sigma == 0.106`, borrowed from label 1. Guards
  section 6 and the case that actually occurs.
- **Every alpha zeroed** (real L2's shape): each label's alpha spans
  `_ZERO_HINT_ALPHA_SCALE == 0.1` rather than `1e-3`. This is the renamed
  `test_centered_zero_alpha_spans_plate_alpha_scale`.
- **Gain zero on every label, alpha positive**: `"gain"` is absent from `priors`
  and the variance carries only floor and alpha. Guards the retained `has_gain`
  gate and, critically, guards section 2 against ever seeing
  `plate_gain_scale == 0`.
- **Alpha zero on every label, gain positive**: `"rel_error"` is present and
  estimable. This is the existing
  `test_centered_zero_alpha_is_prior_around_zero` (`tests/test_bayes.py:731`),
  which must keep passing unchanged — it is the regression guard against
  re-introducing the withdrawn section 1.
- **Shared/per-label agreement**: for one label with a given positive hint, the
  shared and per-label branches produce the same sigma. Guards section 3 and
  would have caught the 0.1/0.01 divergence.
- **Free-mode means**: `Exponential` and `HalfNormal` priors built from the same
  hint have means equal to that hint. Guards section 4.
- **Fixed mode unchanged**: `fixed` still yields hard constants for both terms —
  the one mode where a zero genuinely means absent.

## Validation

**`scripts/compare_methods.sh` cannot validate any of this.** Its `noise` mode
runs `--fit-method huber` with `--noise-alpha`/`--noise-gain`, the lmfit path,
and even its `mcmc` mode reaches `NoiseConfig.ye_mag` — but
`build_pymc_noise_priors` is only called when `noise.kind == "structured"`
(`bayes.py:2171`). `NoiseConfig.structured` has **zero production callers**; it
appears only in `tests/test_bayes.py` and `tests/test_fitting.py`. Everything
in this spec is therefore library-API surface, not CLI pipeline behaviour. An
earlier draft prescribed `compare_methods.sh <plate> noise` here, which would
have burned hours of MCMC measuring nothing.

What *was* validated is the premise, via `fgls_fit_plate` on L2/L3/L4 — the
live production path — yielding the table in "Where the zeros come from".

Expected effect of each section on those plates:

- Section 1 is documentation only and cannot move results.
- Section 2 (zeroed gain) is **not reached**: gain is never exactly 0.0 on
  L2–L4. Correct, but inert.
- Section 3 should be inert: `0.2 * mu_g` dominates both deleted floors at the
  observed gains of 0.4–5.5.
- Section 4 affects `free` mode only; the FGLS path uses `centered`.
- **Section 6 (zeroed alpha) is the change that actually moves posteriors**,
  widening 4 of 6 label-fits from `HalfNormal(1e-3)` to a prior spanning the
  plate's alpha scale.

### Measured outcome

Run on L4, 8 wells, `fit_binding_pymc` with `NoiseConfig.structured` seeded
from the plate's own FGLS-calibrated model, `alpha_mode="centered"`, 1000
draws. "Before" reproduced by restoring `_alpha_prior_sigma`'s pre-fix
formula, so only the zeroed-alpha width differs.

The alpha posterior on the zeroed label is released from the prior:

| well | `rel_error_2` before     | after                  |
| ---- | ------------------------ | ---------------------- |
| A01  | 0.00078, HDI [0, 0.0018] | 0.0866, HDI [0, 0.164] |
| A02  | 0.00080, HDI [0, 0.0019] | 0.0452, HDI [0, 0.113] |

**Kd credible intervals widen systematically: all 8 of 8 wells, ratio 1.10 to
1.57, mean 1.20 (median 1.15).** Kd point estimates are stable — mean shift
-0.005, largest 0.039 on A01, itself under 8% of that well's HDI width.

So the fix does not move the answer; it widens the honest uncertainty around
it by roughly 15–20%. That is material when reporting a pKa.

**Caveat on identifiability.** A01's post-fix alpha posterior mean (0.0866)
is essentially the prior mean (0.0846), so on that well the likelihood barely
constrains alpha and the posterior largely reports the prior back. A02 (0.045)
sits below the prior mean, so there the data does inform it. This is the
gain/alpha identifiability problem listed under "Out of scope", now visible in
the posterior instead of masked by a prior tight enough to hide it. Treat
`rel_error` on such labels as an uncertainty statement, not a measurement.

**Methodological note.** A first attempt at this comparison passed
`NoiseConfig.structured(noise_model=calib)` without setting modes.
`_resolve_noise_modes` resolves `None` to `"free"` for raw `Dataset` input,
and section 6 only touches `"centered"`, so that run measured nothing and
wrongly appeared to show no effect. The tell was `rel_error_2` returning
exactly `0.00100` in both arms — `_MIN_NOISE_PRIOR_SCALE`, a value the fix
should have eliminated. Any future comparison must set the mode explicitly.

Two existing assertions in `test_free_noise_priors_scale_from_hints`
(`tests/test_bayes.py:722,726`) encode the old scale semantics and must be
updated to the new means as part of section 4. That is an intended contract
change, not a regression.

Watch for divergences: making a previously-frozen parameter estimable can
surface funnel geometry that the hard zero was hiding.

## Out of scope

**Collinearity is left at its source.** Sharing gain across labels, or fitting
only the better-identified term at 7 points per label, would attack the cause
rather than the symptom. That is deliberately deferred: section 2 lets the
posterior recover from an arbitrary split, which is the cheaper fix and does not
change the calibration contract.

**`_plate_noise_model_from_nnls` is misnamed.** It and `calibrate_noise_robust`
use closed-form moment estimators clamped by `max(0.0, ...)`
(`noise_calibration.py:164,219`); only `fit_noise_model_nnls` is true NNLS.
Renaming is unrelated churn.

No changes to the calibration stage, to `PlateNoiseModel`, or to the ye_mag
noise family.
