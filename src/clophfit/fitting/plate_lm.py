"""Fit a whole plate at once, classically, with plate-wide noise scales.

Every classical arm in this campaign fits one well at a time: `fit_binding_glob`
takes a single well's `Dataset`, and the control groups are only pooled
afterwards, by averaging per-well K estimates. The Bayesian multi-well model does
something the classical side cannot answer to - it shares one K across a control
group *during* the fit, and learns a per-label noise scale from the whole plate -
so the two have never been compared like for like on the control holdout.

This closes that gap. Residuals from every well are concatenated into one
least-squares problem: K is shared within each control group and free elsewhere,
S0 and S1 stay per well and label, and the per-label scale is *profiled* rather
than fitted.

Profiling is not an optimisation detail. Multiplying every weight by a constant
leaves the argmin of a least-squares objective untouched - only chi-square and
the covariance move - so a global noise scale is not identifiable that way at
all. For fixed structural parameters the maximum-likelihood scale has a closed
form, the root-mean-square of that label's standardised residuals, so the fit
alternates: solve with current scales, update the scales, repeat. That is what
lmfit's ``scale_covar`` does with one global factor, generalised to one per
label, which is the structure section 1 found to be the right amount.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.optimize import least_squares

from clophfit.fitting.data_structures import NoiseModelParams
from clophfit.fitting.models import binding_1site
from clophfit.fitting.noise_calibration import fit_noise_model_nnls

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "PlateLMResult",
    "apply_excluded_points",
    "fit_plate_lm",
    "fit_plate_lm_screened",
    "fractional_outliers",
    "profile_k_intervals",
    "ratiometric_exempt",
    "screening_sigma_floor",
]

# Below this many unmasked points a well cannot suggest its own midpoint.
_MIN_POINTS_FOR_SEED = 2
_FALLBACK_K = 7.0
# A group needs a member left over to compare the held-out one against.
_MIN_GROUP_FOR_HOLDOUT = 2
# The pH scale, matching the bounds lmfit's per-well path uses.
_K_MIN_PH = 3.0
_K_MAX_PH = 11.0
# Half-width of a 94% interval in standard errors.
_Z94 = 1.881
# Residuals are already divided by y_err, so a robust loss switches from
# quadratic to linear at this many nominal sigma.
_ROBUST_F_SCALE = 2.0
# MAD -> sigma for a normal distribution.
_MAD_TO_SIGMA = 1.4826
_LN10 = float(np.log(10.0))
# A robust loss never met scipy's termination tests here because the
# parameters were unscaled: K near 7 beside plateaus near 1000, so the step and
# gradient tests, built for comparably sized parameters, never tripped and the
# fit crept on to 100 * n_params evaluations. A cap of 200 hid that on L2, but on
# L5b K was still moving after 5000 evaluations (0.23 pH on one well, the bench
# score getting worse). Scaling by the Jacobian (x_scale="jac") lets the robust
# fit converge by scipy's own criteria: over eleven plates the first solve of a
# fit needs 600-1087 evaluations and every later one under 40. The cap is only a
# safety net, set at about five times that, and reaching it is reported. The
# step-size test (xtol) is switched off for these solves: it compares the raw
# step with norm(x), which the plateaus dominate, so once a few rejected huber
# steps shrink the trust region it fired while K was still moving (a planted-
# outlier well 0.09 pH off after 20 evaluations; 0.014, as unscaled, once the
# cost test decides).
_ROBUST_MAX_NFEV = 5000
# The same safety net for linear loss, which terminates in well under 200
# evaluations unless the weights have degenerated: a calibrated floor driven to
# zero leaves dim points with near-infinite weight, and one such solve took 8713
# evaluations (202 s of dense SVDs). Reaching it is reported, not hidden.
_LINEAR_MAX_NFEV = 1000
# Step fraction applied to a calibrated noise term after its update reverses
# direction. With the floor free, floor and gain are near-collinear and the
# undamped iteration cycled (floor 0 -> 126 -> 27 -> 15 -> 0.3 -> 0 on L6b)
# without converging; halving the step at each reversal turns the cycle into a
# converging oscillation, and a term that moves monotonically is untouched.
_CALIBRATION_REVERSAL_DAMPING = 0.5
_NOISE_TERMS = ("sigma_floor", "gain", "alpha")
# Largest number of refits spent walking out one side of one profile.
_PROFILE_MAX_STEPS = 12
# First step out from K-hat, in standard errors.
_PROFILE_STEP_SE = 0.75


@dataclass
class PlateLMResult:
    """Outcome of one plate-wide classical fit.

    Parameters
    ----------
    k : dict[str, float]
        Fitted K per well. Wells sharing a control group hold the identical
        value, not an average of separate fits.
    k_stderr : dict[str, float]
        Standard error on each K, from the Jacobian at the solution, scaled by
        the profiled noise. Conditional on the profiled scales, so mildly
        optimistic.
    ye_mag : dict[str, float]
        Profiled noise multiplier per label, relative to the supplied ``y_err``.
    params : dict[str, dict[str, float]]
        Per well, the whole fitted curve - ``K``, ``sK`` and the ``S0_``/``S1_``
        plateaus per label - not only K. Exporting K alone made the plate fit a
        second-class output: it could be tabulated but not plotted, while every
        other fitter in the same run produced a K plot and per-well figures.
    n_points : int
        Unmasked observations entering the fit.
    n_params : int
        Free structural parameters.
    success : bool
        Whether the final least-squares solve converged.
    n_excluded : int
        Observations dropped by a screening pass, 0 for a single fit.
    noise : dict[str, dict[str, float]]
        Per-label ``sigma_floor``, ``gain`` and ``alpha`` actually used. When
        calibration ran these are what it estimated, and they are the whole
        point of asking for it - a run that fits the noise and reports only
        ``ye_mag`` has thrown the answer away.
    excluded_points : dict[str, dict[str, list[int]]]
        Well to label to the original indices a screening pass dropped. The
        screen builds its own masked copies internally, so without this a
        caller plotting the input datasets would draw the discarded points as
        though they had been fitted.
    residuals : list[dict[str, Any]]
        One record per unmasked observation, on the library's canonical
        columns: ``well``, ``label``, ``step``, ``yhat``, ``raw_res``,
        ``sigma`` and ``std_res``.  ``sigma`` includes the profiled scale, so
        ``std_res`` has per-label SD ~1 by construction and only its shape -
        tails, step dependence - carries information; ``raw_res`` and ``yhat``
        are the signal-scale pair the noise-calibration estimators read.
    budget_reached : bool
        Whether any solve stopped on its evaluation cap rather than on scipy's
        own convergence tests. Kept apart from ``success``: the caps are safety
        nets, and a caller deciding whether to trust a fit needs to know which
        of the two it is looking at.
    """

    k: dict[str, float] = field(default_factory=dict)
    k_stderr: dict[str, float] = field(default_factory=dict)
    ye_mag: dict[str, float] = field(default_factory=dict)
    params: dict[str, dict[str, float]] = field(default_factory=dict)
    n_points: int = 0
    n_params: int = 0
    success: bool = False
    n_excluded: int = 0
    noise: dict[str, dict[str, float]] = field(default_factory=dict)
    excluded_points: dict[str, dict[str, list[int]]] = field(default_factory=dict)
    residuals: list[dict[str, Any]] = field(default_factory=list)
    budget_reached: bool = False


def _solver_kwargs(loss: str) -> dict[str, Any]:
    """Loss-specific ``least_squares`` settings, shared by every solve.

    Parameters
    ----------
    loss : str
        ``"linear"`` or a robust loss name.

    Returns
    -------
    dict[str, Any]
        ``loss``, ``f_scale`` and ``max_nfev``, plus ``x_scale="jac"`` and
        ``xtol=None`` for a robust loss, which needs both to converge on the
        cost test (see ``_ROBUST_MAX_NFEV``). Linear keeps scipy's defaults so
        its fits are unchanged.
    """
    if loss == "linear":
        return {"loss": loss, "f_scale": 1.0, "max_nfev": _LINEAR_MAX_NFEV}
    return {
        "loss": loss,
        "f_scale": _ROBUST_F_SCALE,
        "max_nfev": _ROBUST_MAX_NFEV,
        "x_scale": "jac",
        "xtol": None,
    }


@dataclass
class _DampingState:
    """Per (label, term): the last applied change and the current step fraction."""

    last: dict[tuple[str, str], float] = field(default_factory=dict)
    step: dict[tuple[str, str], float] = field(default_factory=dict)


def _damped_update(
    current: Mapping[str, Any] | None,
    proposed: Mapping[str, Any] | None,
    state: _DampingState,
) -> Mapping[str, Any] | None:
    """Move each calibrated noise term toward its proposal, halving on reversal.

    A term whose update keeps its direction takes the proposal exactly, so a
    calibration that never oscillates is unchanged. When a term's change
    reverses sign, its step fraction halves from then on; that is what stops
    the floor/gain cycle a free floor produced.

    Parameters
    ----------
    current : Mapping[str, Any] | None
        Noise model in force before this pass.
    proposed : Mapping[str, Any] | None
        What ``_calibrate_noise`` estimated from this pass's residuals.
    state : _DampingState
        Carried across the passes of one fit; updated in place.

    Returns
    -------
    Mapping[str, Any] | None
        The noise model to use for the next pass.
    """
    if current is None or proposed is None or proposed is current:
        return proposed
    out: dict[str, Any] = {}
    for lbl, new in proposed.items():
        old = current.get(lbl)
        if old is None:
            out[lbl] = new
            continue
        values = {}
        for term in _NOISE_TERMS:
            o, n = float(getattr(old, term, 0.0)), float(getattr(new, term, 0.0))
            key = (str(lbl), term)
            delta = n - o
            if delta * state.last.get(key, 0.0) < 0:
                state.step[key] = (
                    state.step.get(key, 1.0) * _CALIBRATION_REVERSAL_DAMPING
                )
            # Never reversed: take the proposal itself, not o + 1.0 * delta.
            values[term] = o + state.step[key] * delta if key in state.step else n
            if delta != 0:
                state.last[key] = delta
        out[lbl] = NoiseModelParams(
            sigma_floor=values["sigma_floor"],
            gain=values["gain"],
            alpha=values["alpha"],
            sigma_ph=float(getattr(new, "sigma_ph", 0.0)),
        )
    return out


def _k_index(
    wells: Sequence[str], groups: Mapping[str, Sequence[str]]
) -> tuple[dict[str, int], int]:
    """Map each well to its K parameter, one per control group and one per free well.

    Parameters
    ----------
    wells : Sequence[str]
        Wells to fit.
    groups : Mapping[str, Sequence[str]]
        Control group name to its member wells.

    Returns
    -------
    tuple[dict[str, int], int]
        Well to K-parameter index, and the number of distinct K parameters.
    """
    index: dict[str, int] = {}
    n = 0
    for members in groups.values():
        present = [w for w in members if w in wells]
        if not present:
            continue
        for well in present:
            index[well] = n
        n += 1
    for well in wells:
        if well not in index:
            index[well] = n
            n += 1
    return index, n


def _seed_k(dataset: Any) -> float:  # ruff: ignore[any-type] - a clophfit Dataset, untyped upstream
    """Guess K as the x nearest the halfway point of the first label's sweep."""
    da = next(iter(dataset.values()))
    x = np.asarray(da.xc, dtype=float)[np.asarray(da.mask)]
    y = np.asarray(da.yc, dtype=float)[np.asarray(da.mask)]
    if len(y) < _MIN_POINTS_FOR_SEED:
        return float(np.median(x)) if len(x) else _FALLBACK_K
    return float(x[int(np.argmin(np.abs(y - (y[0] + y[-1]) * 0.5)))])


@dataclass
class _Problem:
    """Everything the residual closure needs, assembled once."""

    kidx: dict[str, int]
    sidx: dict[tuple[str, str], int]
    p0: np.ndarray
    obs: list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray]]
    # Parallel to `obs`: where each kept point sat in the original array, so a
    # residual can be traced back to the observation that produced it. Every
    # other fitter emits this as `raw_i`; without it a screen cannot mask.
    raw_i: list[np.ndarray]
    labels: list[str]
    is_ph: bool
    n_k: int
    n_points: int
    n_curves: dict[str, int]


def _assemble(
    datasets: Mapping[str, Any],
    groups: Mapping[str, Sequence[str]],
) -> _Problem:
    """Lay out the parameter vector and flatten every observation.

    Parameters
    ----------
    datasets : Mapping[str, Any]
        Well identifier to `Dataset`.
    groups : Mapping[str, Sequence[str]]
        Control group name to member wells.

    Returns
    -------
    _Problem
        Parameter indices, seeds, and the flattened observations.
    """
    wells = list(datasets)
    labels = list(dict.fromkeys(lbl for ds in datasets.values() for lbl in ds))
    kidx, n_k = _k_index(wells, groups)

    p0: list[float] = [0.0] * n_k
    seeded: dict[int, list[float]] = {}
    for well, dataset in datasets.items():
        seeded.setdefault(kidx[well], []).append(_seed_k(dataset))
    for i, guesses in seeded.items():
        p0[i] = float(np.median(guesses))

    sidx: dict[tuple[str, str], int] = {}
    obs: list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray]] = []
    raw_i: list[np.ndarray] = []
    for well, dataset in datasets.items():
        for lbl, da in dataset.items():
            m = np.asarray(da.mask)
            x = np.asarray(da.xc, dtype=float)[m]
            y = np.asarray(da.yc, dtype=float)[m]
            yerr = np.asarray(da.y_errc, dtype=float)[m]
            sidx[well, lbl] = len(p0)
            p0.extend([float(y[0]) if len(y) else 0.0, float(y[-1]) if len(y) else 0.0])
            obs.append((well, lbl, x, y, yerr))
            raw_i.append(np.flatnonzero(m))

    return _Problem(
        kidx=kidx,
        sidx=sidx,
        p0=np.asarray(p0, dtype=float),
        obs=obs,
        raw_i=raw_i,
        labels=labels,
        is_ph=bool(next(iter(datasets.values())).is_ph),
        n_k=n_k,
        n_points=int(sum(len(x) for _, _, x, _, _ in obs)),
        n_curves={lbl: sum(1 for _, o, _, _, _ in obs if o == lbl) for lbl in labels},
    )


def _raw_residuals(
    prob: _Problem, p: np.ndarray, label: str, yerrs: Sequence[np.ndarray]
) -> np.ndarray:
    """Return one label's residuals divided by ``y_err`` only, not by any scale.

    Parameters
    ----------
    prob : _Problem
        Assembled problem.
    p : np.ndarray
        Current parameter vector.
    label : str
        Label to collect.
    yerrs : Sequence[np.ndarray]
        Per-observation y_err, parallel to ``prob.obs``.

    Returns
    -------
    np.ndarray
        Standardised residuals for that label.
    """
    parts = []
    for i, (well, lbl, x, y, _ye) in enumerate(prob.obs):
        if lbl != label:
            continue
        s0, s1 = p[prob.sidx[well, lbl]], p[prob.sidx[well, lbl] + 1]
        model = binding_1site(x, p[prob.kidx[well]], s0, s1, is_ph=prob.is_ph)
        parts.append((y - model) / yerrs[i])
    return np.concatenate(parts) if parts else np.zeros(0)


def _calibrate_noise(  # ruff: ignore[too-many-arguments] - one per fit input
    prob: _Problem,
    p: np.ndarray,
    scales: Mapping[str, float],
    yerrs: Sequence[np.ndarray],
    current: Mapping[str, Any] | None,
    *,
    noise_free: Sequence[str] = ("gain", "alpha"),
) -> Mapping[str, Any] | None:
    """Re-estimate the free noise terms per label from this pass's residuals.

    The plate's own residuals say how the noise grows with signal, so the
    weights need not be guessed. ``sigma^2 = floor^2 + gain * yhat +
    (alpha * yhat)^2`` is fitted by non-negative least squares, with the floor
    pinned to the measured read noise: over a titration's narrow dynamic range
    ``yhat`` and ``yhat^2`` are close to collinear, and letting all three float
    trades a real floor against a spurious gain.

    This is the feasible-generalised-least-squares step - estimate the weights
    from a fit, refit under them - reusing the estimator the per-well FGLS path
    already uses, so both paths calibrate the same way.

    Parameters
    ----------
    prob : _Problem
        Assembled problem.
    p : np.ndarray
        Current parameter vector.
    scales : Mapping[str, float]
        Current per-label noise multiplier.
    yerrs : Sequence[np.ndarray]
        Per-observation y_err for this pass.
    current : Mapping[str, Any] | None
        Noise model in force, supplying the floors to pin and the values any
        held term keeps.
    noise_free : Sequence[str]
        Which of ``"floor"``, ``"gain"`` and ``"alpha"`` may move. Anything
        absent keeps the value *current* gave it, so holding a term at zero
        disables it and holding it at a supplied value pins it there. An empty
        sequence returns *current* unchanged.

    Returns
    -------
    Mapping[str, Any] | None
        Recalibrated model, or *current* unchanged when the estimate fails -
        a plate that cannot support the estimate keeps the weights it had
        rather than taking a degenerate one.
    """
    floors = {
        lbl: float(getattr(params, "sigma_floor", 0.0))
        for lbl, params in (current or {}).items()
    }
    if not floors:
        return current
    free = set(noise_free)
    held_gain = {
        lbl: float(getattr(params, "gain", 0.0))
        for lbl, params in (current or {}).items()
    }
    held_alpha = {
        lbl: float(getattr(params, "alpha", 0.0))
        for lbl, params in (current or {}).items()
    }
    if not free:
        return current
    table = _residual_table(prob, p, scales, yerrs)
    if not table:
        return current
    df = pd.DataFrame(table)
    # The estimator keys on label, raw_res and yhat, which the table carries.
    df["label"] = df["label"].astype(str)
    try:
        fitted_floors, gains, alphas = fit_noise_model_nnls(
            df,
            sigma_floor_fixed=None if "floor" in free else floors,
            gain_fixed=None if "gain" in free else held_gain,
            rel_error_fixed=None if "alpha" in free else held_alpha,
        )
    except (ValueError, np.linalg.LinAlgError):
        logger.debug("plate noise calibration failed; keeping current weights")
        return current
    return {
        lbl: NoiseModelParams(
            sigma_floor=fitted_floors.get(lbl, floors.get(lbl, 0.0)),
            gain=gains.get(lbl, 0.0),
            alpha=alphas.get(lbl, 0.0),
        )
        for lbl in floors
    }


def _reweight(
    prob: _Problem,
    p: np.ndarray,
    yerrs: list[np.ndarray],
    noise_model: Mapping[str, Any] | None,
) -> float:
    """Re-evaluate a signal-dependent y_err at the model prediction.

    ``sigma`` grows with signal as ``sqrt(floor^2 + gain*y + (alpha*y)^2)``, and
    evaluating that at the *observed* y makes a point's own noise set its own
    weight: a downward fluctuation is trusted more than an upward one, which
    drags the curve down. Evaluating at the prediction breaks that feedback.
    The weights are held fixed within a solve and refreshed here between
    solves, which is the ordinary iteratively-reweighted arrangement.

    Does nothing without a noise model, and nothing when the model is a flat
    floor, since then the prediction cannot change the answer.

    Parameters
    ----------
    prob : _Problem
        Assembled problem.
    p : np.ndarray
        Current parameter vector.
    yerrs : list[np.ndarray]
        Per-observation y_err, updated in place.
    noise_model : Mapping[str, Any] | None
        Label to an object with ``compute_y_err``.

    Returns
    -------
    float
        Largest relative change in y_err, so the caller can fold it into its
        own convergence test.
    """
    if noise_model is None:
        return 0.0
    moved = 0.0
    for i, (well, lbl, x, _y, _ye) in enumerate(prob.obs):
        params = noise_model.get(lbl)
        if params is None:
            continue
        s0, s1 = p[prob.sidx[well, lbl]], p[prob.sidx[well, lbl] + 1]
        yhat = binding_1site(x, p[prob.kidx[well]], s0, s1, is_ph=prob.is_ph)
        new = np.asarray(params.compute_y_err(yhat), dtype=float)
        new = np.where(np.isfinite(new) & (new > 0), new, yerrs[i])
        moved = max(moved, float(np.max(np.abs(new - yerrs[i]) / new)))
        yerrs[i][:] = new
    return moved


def _residual_table(
    prob: _Problem,
    p: np.ndarray,
    scales: Mapping[str, float],
    yerrs: Sequence[np.ndarray],
) -> list[dict[str, Any]]:
    """Flatten the fit into the canonical residual table the metrics expect.

    Parameters
    ----------
    prob : _Problem
        Assembled problem.
    p : np.ndarray
        Parameter vector at the solution.
    scales : Mapping[str, float]
        Profiled noise multiplier per label.
    yerrs : Sequence[np.ndarray]
        Per-observation y_err, parallel to ``prob.obs``.

    Returns
    -------
    list[dict[str, Any]]
        One row per observation on the canonical residual columns.  ``step`` is
        the rank of the point on the titration axis, so masked points do not
        shift the numbering of the ones that survive.
    """
    rows: list[dict[str, Any]] = []
    for i, ((well, lbl, x, y, _ye), raw_idx) in enumerate(
        zip(prob.obs, prob.raw_i, strict=True)
    ):
        yerr = yerrs[i]
        s0, s1 = p[prob.sidx[well, lbl]], p[prob.sidx[well, lbl] + 1]
        model = binding_1site(x, p[prob.kidx[well]], s0, s1, is_ph=prob.is_ph)
        sigma = yerr * scales[lbl]
        raw = y - model
        order = np.argsort(np.argsort(x))
        rows.extend(
            {
                "well": well,
                "label": lbl,
                "step": int(step),
                "raw_i": int(ri),
                "yhat": float(yh),
                "raw_res": float(r),
                "sigma": float(sd),
                "std_res": float(r / sd),
            }
            for step, ri, yh, r, sd in zip(
                order, raw_idx, model, raw, sigma, strict=True
            )
        )
    return rows


def _profiled_scale(
    prob: _Problem, resid: np.ndarray, label: str, *, robust: bool = False
) -> float:
    """Closed-form noise scale for one label, corrected for degrees of freedom.

    The plain maximum-likelihood scale divides by *n* and is biased low, because
    the fit has already absorbed part of the scatter into its own parameters.
    Dividing by the residual degrees of freedom is the correction lmfit's
    ``scale_covar`` applies through reduced chi-square; each label owns its S0
    and S1, and the shared K values are apportioned by point count.

    Parameters
    ----------
    prob : _Problem
        Assembled problem.
    resid : np.ndarray
        That label's standardised residuals.
    label : str
        Label being scaled.
    robust : bool
        Estimate the scale from the median absolute deviation instead of the
        root-mean-square. A single bad point inflates the RMS, and since this
        scale is the denominator of every standardised residual, that hides the
        very point a screen is looking for.

    Returns
    -------
    float
        Positive noise multiplier.
    """
    n_lbl = len(resid)
    if n_lbl == 0:
        return 1.0
    if robust:
        mad = float(np.median(np.abs(resid - np.median(resid))))
        return max(_MAD_TO_SIGMA * mad, 1e-12)
    p_lbl = 2 * prob.n_curves[label] + prob.n_k * n_lbl / max(prob.n_points, 1)
    dof = max(n_lbl - p_lbl, 1.0)
    return max(float(np.sqrt(np.sum(resid**2) / dof)), 1e-12)


def _standard_errors(jac: np.ndarray, n_params: int) -> np.ndarray:
    """Parameter standard errors from the Jacobian at the solution.

    Parameters
    ----------
    jac : np.ndarray
        Jacobian returned by the solver.
    n_params : int
        Length of the parameter vector.

    Returns
    -------
    np.ndarray
        Standard error per parameter, NaN where the Jacobian is singular.
    """
    dense = np.asarray(jac, dtype=float)
    try:
        _, s, vt = np.linalg.svd(dense, full_matrices=False)
    except np.linalg.LinAlgError:  # pragma: no cover - singular Jacobian
        return np.full(n_params, np.nan)
    keep = s > np.finfo(float).eps * max(dense.shape) * s[0]
    cov = (vt[keep].T / s[keep] ** 2) @ vt[keep]
    out: np.ndarray = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    return out


def _residual_fn(
    prob: _Problem, scales: Mapping[str, float], yerrs: Sequence[np.ndarray]
) -> Callable[[np.ndarray], np.ndarray]:
    """Build the scaled-residual function for one plate problem.

    Parameters
    ----------
    prob : _Problem
        Assembled plate problem.
    scales : Mapping[str, float]
        Current per-label noise scale.
    yerrs : Sequence[np.ndarray]
        Per-observation y_err, parallel to ``prob.obs``. Held separately from
        the problem so a signal-dependent model can reweight between passes.

    Returns
    -------
    Callable[[np.ndarray], np.ndarray]
        Residuals, already divided by ``y_err`` and the label scale.
    """

    def residuals(params: np.ndarray) -> np.ndarray:
        parts = []
        for i, (well, lbl, x, y, _ye) in enumerate(prob.obs):
            s0, s1 = params[prob.sidx[well, lbl]], params[prob.sidx[well, lbl] + 1]
            model = binding_1site(x, params[prob.kidx[well]], s0, s1, is_ph=prob.is_ph)
            parts.append((y - model) / (yerrs[i] * scales[lbl]))
        return np.concatenate(parts) if parts else np.zeros(0)

    return residuals


def _jacobian_fn(
    prob: _Problem, scales: Mapping[str, float], yerrs: Sequence[np.ndarray]
) -> Callable[[np.ndarray], np.ndarray]:
    """Build the exact Jacobian of the scaled residuals.

    Each residual block belongs to one (well, label) curve and depends on three
    parameters only, so the matrix is almost all zeros. Estimating it by finite
    differences costs one pass over the whole plate per parameter - several
    hundred passes for a 96-well plate - which is what made the robust losses
    unusably slow. The closed forms below cost one pass in total and are exact,
    so the solver also steps better.

    Parameters
    ----------
    prob : _Problem
        Assembled plate problem.
    scales : Mapping[str, float]
        Current per-label noise scale.
    yerrs : Sequence[np.ndarray]
        Per-observation y_err, parallel to ``prob.obs``.

    Returns
    -------
    Callable[[np.ndarray], np.ndarray]
        Jacobian, one row per residual.

    Notes
    -----
    When y_err is reweighted on the model prediction it depends on the
    parameters, and these derivatives treat it as fixed. That is the standard
    iteratively-reweighted step: the weights are held constant within a pass
    and updated between passes, so the fixed point is the same and only the
    path to it differs.
    """
    n_res = sum(len(y) for *_, y, _ in prob.obs)

    def jacobian(params: np.ndarray) -> np.ndarray:
        jac = np.zeros((n_res, len(params)))
        row = 0
        for i, (well, lbl, x, y, _ye) in enumerate(prob.obs):
            n, yerr = len(y), yerrs[i]
            ki, si = prob.kidx[well], prob.sidx[well, lbl]
            k, s0, s1 = params[ki], params[si], params[si + 1]
            # Reuse the model itself for the sigmoid, so the overflow-safe
            # algebra lives in exactly one place.
            f = binding_1site(x, k, 0.0, 1.0, is_ph=prob.is_ph)
            dk = (
                (s1 - s0) * _LN10 * f * (1.0 - f)
                if prob.is_ph
                else -(s1 - s0) * f * (1.0 - f) / k
            )
            denom = yerr * scales[lbl]
            block = slice(row, row + n)
            jac[block, ki] = -dk / denom
            jac[block, si] = -(1.0 - f) / denom
            jac[block, si + 1] = -f / denom
            row += n
        return jac

    return jacobian


def fit_plate_lm(  # ruff: ignore[too-many-arguments]
    datasets: Mapping[str, Any],
    groups: Mapping[str, Sequence[str]],
    *,
    max_iter: int = 6,
    tol: float = 1e-3,
    loss: Literal["linear", "huber", "soft_l1", "cauchy"] = "linear",
    noise_model: Mapping[str, Any] | None = None,
    calibrate_noise: bool = False,
    noise_free: Sequence[str] = ("gain", "alpha"),
) -> PlateLMResult:
    """Fit every well of a plate jointly, profiling one noise scale per label.

    Parameters
    ----------
    datasets : Mapping[str, Any]
        Well identifier to `Dataset`.
    groups : Mapping[str, Sequence[str]]
        Control group name to member wells; those wells share one K.
    max_iter : int
        Maximum alternations between solving and rescaling.
    tol : float
        Stop when every scale moves by less than this, relatively.
    loss : Literal["linear", "huber", "soft_l1", "cauchy"]
        Loss handed to the solver. ``"linear"`` is ordinary least squares.
        Anything else down-weights large residuals, which matters when the fit
        is about to be used to *find* outliers: least squares drags the curve
        toward a bad point, shrinking its own residual and inflating its
        neighbours', and the profiled scale then inflates too because it is the
        RMS of those residuals. Screening on such a fit decides with a bent
        curve and a stretched ruler. With a robust loss the scale is taken from
        the median absolute deviation for the same reason.
    noise_model : Mapping[str, Any] | None
        Per-label noise parameters. When given, a signal-dependent ``y_err`` is
        re-evaluated at the model prediction between passes instead of being
        taken at the observation, where a point's own fluctuation would set its
        own weight. A flat floor makes this a no-op.
    calibrate_noise : bool
        Estimate ``gain`` and ``alpha`` per label from each pass's residuals,
        floor pinned, and refit under the result. Off by default: it describes
        the residuals better and fits K worse, because down-weighting
        high-signal points down-weights the plateaus that pin S0 and S1.
    noise_free : Sequence[str]
        Which noise terms ``calibrate_noise`` may move: any of ``"floor"``,
        ``"gain"`` and ``"alpha"``. The default frees both signal terms with
        the floor pinned, which is the only case the estimator could express
        before and the one where gain and alpha are collinear. Holding one of
        them lets the iteration converge instead of wandering along that ridge.

    Returns
    -------
    PlateLMResult
        Fitted K per well, its standard error, and the profiled scales.
    """
    prob = _assemble(datasets, groups)
    scales = dict.fromkeys(prob.labels, 1.0)
    p = prob.p0
    bounds = _k_bounds(prob.n_k, len(p), is_ph=prob.is_ph)

    yerrs = [np.array(ye, dtype=float, copy=True) for *_, ye in prob.obs]
    residuals = _residual_fn(prob, scales, yerrs)

    jacobian = _jacobian_fn(prob, scales, yerrs)

    fit = None
    budget_reached = False
    damping = _DampingState()
    for _ in range(max_iter):
        fit = least_squares(
            residuals,
            p,
            jac=jacobian,
            method="trf",
            bounds=bounds,
            **_solver_kwargs(loss),
        )
        budget_reached = budget_reached or fit.status == 0
        p = fit.x
        if calibrate_noise:
            proposed = _calibrate_noise(
                prob, p, scales, yerrs, noise_model, noise_free=noise_free
            )
            noise_model = _damped_update(noise_model, proposed, damping)
        moved = _reweight(prob, p, yerrs, noise_model)
        for lbl in prob.labels:
            new = _profiled_scale(
                prob,
                _raw_residuals(prob, p, lbl, yerrs),
                lbl,
                robust=loss != "linear",
            )
            moved = max(moved, abs(new - scales[lbl]) / new)
            scales[lbl] = new
        if moved < tol:
            break

    result = PlateLMResult(
        ye_mag=dict(scales),
        noise={
            str(lbl): {
                "sigma_floor": float(getattr(params, "sigma_floor", 0.0)),
                "gain": float(getattr(params, "gain", 0.0)),
                "alpha": float(getattr(params, "alpha", 0.0)),
            }
            for lbl, params in (noise_model or {}).items()
        },
        n_points=prob.n_points,
        n_params=len(p),
        success=bool(fit.success) if fit is not None else False,
        budget_reached=budget_reached,
    )
    if fit is None:
        return result
    result.residuals = _residual_table(prob, p, scales, yerrs)
    err = _standard_errors(np.asarray(fit.jac, dtype=float), len(p))
    for well in datasets:
        result.k[well] = float(p[prob.kidx[well]])
        result.k_stderr[well] = float(err[prob.kidx[well]])
        row = {
            "K": float(p[prob.kidx[well]]),
            "sK": float(err[prob.kidx[well]]),
        }
        for (w, lbl), i in prob.sidx.items():
            if w == well:
                row[f"S0_{lbl}"] = float(p[i])
                row[f"S1_{lbl}"] = float(p[i + 1])
                row[f"sS0_{lbl}"] = float(err[i])
                row[f"sS1_{lbl}"] = float(err[i + 1])
        result.params[well] = row
    return result


def profile_k_intervals(  # ruff: ignore[too-many-arguments]
    datasets: Mapping[str, Any],
    groups: Mapping[str, Sequence[str]],
    wells: Sequence[str],
    *,
    level: float = 0.94,
    loss: Literal["linear", "huber", "soft_l1", "cauchy"] = "linear",
    noise_model: Mapping[str, Any] | None = None,
) -> dict[str, tuple[float, float]]:
    """Profile-likelihood interval on K, for wells the quadratic error fails.

    ``k_stderr`` comes from the curvature at the optimum, which describes the
    likelihood only where it is close to a parabola. That assumption breaks on
    exactly the wells worth checking: a flat basin reports a standard error of
    1e5, and a well whose K is genuinely pinned down can still sit in an
    asymmetric valley. Profiling makes no such assumption - it fixes K, lets
    every other parameter re-optimise, and reads off where the cost has risen by
    the chi-square quantile for one degree of freedom.

    One refit per profile point, so this is deliberately not run for a whole
    plate. Pass the wells that need it.

    Parameters
    ----------
    datasets : Mapping[str, Any]
        Well identifier to `Dataset`, as passed to :func:`fit_plate_lm`.
    groups : Mapping[str, Sequence[str]]
        Control group name to member wells; those wells share one K.
    wells : Sequence[str]
        Wells to profile. Grouped wells share a K, so profiling any member
        profiles the group.
    level : float
        Interval mass, 0.94 to match the HDI this project reports elsewhere.
    loss : Literal["linear", "huber", "soft_l1", "cauchy"]
        Loss for the refits; use the one the reported fit used.
    noise_model : Mapping[str, Any] | None
        Signal-dependent noise model, if the reported fit used one. The
        profile has to weight the data the same way the fit did or its
        criterion refers to a different likelihood.

    Returns
    -------
    dict[str, tuple[float, float]]
        Well to (lower, upper). A bound that runs into the edge of the K range
        without the cost rising enough comes back as ``-inf`` or ``inf``: the
        data does not bound K on that side, which is the honest answer and the
        one a standard error cannot express.
    """
    base = fit_plate_lm(datasets, groups, loss=loss, noise_model=noise_model)
    prob = _assemble(datasets, groups)
    scales = base.ye_mag
    p_base = np.asarray(_params_from(base, prob), dtype=float)
    yerrs = [np.array(ye, dtype=float, copy=True) for *_, ye in prob.obs]
    _reweight(prob, p_base, yerrs, noise_model)
    residuals = _residual_fn(prob, scales, yerrs)
    jacobian = _jacobian_fn(prob, scales, yerrs)
    p_hat = p_base
    lo_b, hi_b = _k_bounds(prob.n_k, len(p_hat), is_ph=prob.is_ph)
    # 2 * cost is the sum of squared residuals, so the usual chi-square
    # threshold on one profiled parameter applies directly.
    crit = float(sp_stats.chi2.ppf(level, 1))
    cost0 = float(np.sum(residuals(p_hat) ** 2))

    ctx = _ProfileCtx(residuals, jacobian, p_hat, lo_b, hi_b, crit, cost0, loss)
    out: dict[str, tuple[float, float]] = {}
    for well in wells:
        ki = prob.kidx[well]
        se = base.k_stderr.get(well, np.nan)
        step = _PROFILE_STEP_SE * se if np.isfinite(se) and se > 0 else 0.25
        out[well] = (
            _profile_side(ctx, ki, -step, float(lo_b[ki])),
            _profile_side(ctx, ki, step, float(hi_b[ki])),
        )
    return out


@dataclass(frozen=True)
class _ProfileCtx:
    """Everything a profile walk needs that does not change between wells.

    Attributes
    ----------
    residuals : Callable[[np.ndarray], np.ndarray]
        Full-vector residual function.
    jacobian : Callable[[np.ndarray], np.ndarray]
        Full-vector Jacobian function.
    p_hat : np.ndarray
        Parameters at the unconstrained optimum, warm-starting each refit.
    lo_b : np.ndarray
        Lower bounds on the full parameter vector.
    hi_b : np.ndarray
        Upper bounds on the full parameter vector.
    crit : float
        Rise in the sum of squares that marks an interval edge.
    cost0 : float
        Sum of squares at the optimum.
    loss : Literal["linear", "huber", "soft_l1", "cauchy"]
        Loss for the refits.
    """

    residuals: Callable[[np.ndarray], np.ndarray]
    jacobian: Callable[[np.ndarray], np.ndarray]
    p_hat: np.ndarray
    lo_b: np.ndarray
    hi_b: np.ndarray
    crit: float
    cost0: float
    loss: Literal["linear", "huber", "soft_l1", "cauchy"]


def _profile_side(ctx: _ProfileCtx, ki: int, step: float, edge: float) -> float:
    """Walk one direction until the cost rises by the criterion, then interpolate.

    Parameters
    ----------
    ctx : _ProfileCtx
        Problem pieces shared by every walk.
    ki : int
        Index of the K being profiled.
    step : float
        Signed first step; its sign picks the direction.
    edge : float
        Limit of the K range in this direction.

    Returns
    -------
    float
        The interval edge, or an infinite bound when the K range runs out
        before the cost rises enough - the data does not bound K on that side.
    """
    p_hat = ctx.p_hat
    k_hat = float(p_hat[ki])
    free = np.delete(np.arange(len(p_hat)), ki)
    bounds = (np.delete(ctx.lo_b, ki), np.delete(ctx.hi_b, ki))

    def refit(kval: float) -> float:
        def res(q: np.ndarray) -> np.ndarray:
            full = np.empty(len(p_hat))
            full[free], full[ki] = q, kval
            return ctx.residuals(full)

        def jac(q: np.ndarray) -> np.ndarray:
            full = np.empty(len(p_hat))
            full[free], full[ki] = q, kval
            return ctx.jacobian(full)[:, free]

        fit = least_squares(
            res,
            p_hat[free],
            jac=jac,
            method="trf",
            bounds=bounds,
            **_solver_kwargs(ctx.loss),
        )
        return float(np.sum(fit.fun**2))

    prev_k, prev_d = k_hat, 0.0
    for i in range(1, _PROFILE_MAX_STEPS + 1):
        k = k_hat + step * i
        if (step < 0 and k <= edge) or (step > 0 and k >= edge):
            return -np.inf if step < 0 else np.inf
        delta = refit(k) - ctx.cost0
        if delta >= ctx.crit:
            # Linear in the rise between the last two points; the curve is
            # smooth here and a finer walk costs another refit per step.
            span = delta - prev_d
            frac = (ctx.crit - prev_d) / span if span > 0 else 0.0
            return float(prev_k + (k - prev_k) * frac)
        prev_k, prev_d = k, delta
    return -np.inf if step < 0 else np.inf


def _params_from(result: PlateLMResult, prob: _Problem) -> np.ndarray:
    """Rebuild the solver's parameter vector from a finished fit.

    Parameters
    ----------
    result : PlateLMResult
        A fit produced from the same datasets and groups.
    prob : _Problem
        The problem those datasets assemble into.

    Returns
    -------
    np.ndarray
        Parameter vector in solver order.
    """
    p = np.zeros(prob.p0.shape)
    for well, i in prob.kidx.items():
        p[i] = result.k[well]
    for (well, lbl), i in prob.sidx.items():
        row = result.params[well]
        p[i], p[i + 1] = row[f"S0_{lbl}"], row[f"S1_{lbl}"]
    return p


def ctr_holdout(
    datasets: Mapping[str, Any],
    groups: Mapping[str, Sequence[str]],
    *,
    rope: float = 0.10,
    plate: str = "",
) -> list[dict[str, Any]]:
    """Leave one control out at a time and score its K against its group.

    The Bayesian ctr-loo promotes the held-out control to a free K while its
    group-mates keep the shared one, then reports the posterior difference. This
    is the same experiment run classically: refit the plate with that well freed,
    and compare its K against the group's, propagating both standard errors.

    Parameters
    ----------
    datasets : Mapping[str, Any]
        Well identifier to `Dataset`.
    groups : Mapping[str, Sequence[str]]
        Control group name to member wells.
    rope : float
        Half-width of the region of practical equivalence, in pH.
    plate : str
        Recorded in each row, for pooling across plates.

    Returns
    -------
    list[dict[str, Any]]
        One row per held-out control, matching the ctr-loo pooled schema closely
        enough to sit beside the Bayesian arms.
    """
    rows: list[dict[str, Any]] = []
    for group, members in groups.items():
        present = [w for w in members if w in datasets]
        if len(present) < _MIN_GROUP_FOR_HOLDOUT:
            continue
        for held in present:
            remaining = [w for w in present if w != held]
            trimmed = {
                g: (remaining if g == group else list(v)) for g, v in groups.items()
            }
            fit = fit_plate_lm(datasets, trimmed)
            k_held = fit.k.get(held, float("nan"))
            k_ref = fit.k.get(remaining[0], float("nan"))
            se_held = fit.k_stderr.get(held, float("nan"))
            se_ref = fit.k_stderr.get(remaining[0], float("nan"))
            rows.append(
                _holdout_row(
                    plate=plate,
                    cell="plate_lm",
                    group=group,
                    held=held,
                    n_remaining=len(remaining),
                    k_held=k_held,
                    k_ref=k_ref,
                    se_held=se_held,
                    se_ref=se_ref,
                    ye_mag=fit.ye_mag,
                    success=fit.success,
                    rope=rope,
                )
            )
    return rows


def _holdout_row(  # ruff: ignore[too-many-arguments] - a flat record; every argument is one field
    *,
    plate: str,
    cell: str,
    group: str,
    held: str,
    n_remaining: int,
    k_held: float,
    k_ref: float,
    se_held: float,
    se_ref: float,
    ye_mag: Mapping[str, float],
    success: bool,
    rope: float,
) -> dict[str, Any]:
    """Assemble one holdout row, in the schema the Bayesian arms are pooled in.

    Shared by the least-squares and errors-in-variables fitters so their rows can
    be concatenated and scored by the same code.

    Parameters
    ----------
    plate : str
        Plate identifier.
    cell : str
        Arm name recorded in the row.
    group : str
        Control group of the held-out well.
    held : str
        Held-out well.
    n_remaining : int
        Group members still sharing a K.
    k_held : float
        K fitted for the held-out well.
    k_ref : float
        K shared by the remaining members.
    se_held : float
        Standard error on *k_held*.
    se_ref : float
        Standard error on *k_ref*.
    ye_mag : Mapping[str, float]
        Profiled noise multiplier per label.
    success : bool
        Whether the solve converged.
    rope : float
        Half-width of the region of practical equivalence.

    Returns
    -------
    dict[str, Any]
        One row.
    """
    delta = k_held - k_ref
    sd = float(np.sqrt(se_held**2 + se_ref**2))
    return {
        "plate": plate,
        "cell": cell,
        "ctr_group": group,
        "heldout_well": held,
        "n_remaining_ctr": n_remaining,
        "delta_k_mean": float(delta),
        "delta_k_sd": sd,
        "z_delta_k": float(delta / sd) if sd > 0 else float("nan"),
        # 94% interval, matching the width of the Bayesian arms' HDI
        "delta_k_hdi94_contains_zero": bool(abs(delta) < _Z94 * sd),
        "p_abs_delta_k_lt_rope": float(abs(delta) < rope),
        "rope": rope,
        "ye_mag_1": ye_mag.get("1", float("nan")),
        "ye_mag_2": ye_mag.get("2", float("nan")),
        "success": success,
    }


def _k_bounds(n_k: int, n_params: int, *, is_ph: bool) -> tuple[np.ndarray, np.ndarray]:
    """Keep K on the pH scale and leave every other parameter free.

    Unbounded, the solver walks K hundreds of units away from the data while the
    plateaus are still poorly seeded, which overflows the model and explores
    nothing useful. lmfit's per-well path has bounded K at [3, 11] all along.

    Parameters
    ----------
    n_k : int
        Number of K parameters, which come first in the vector.
    n_params : int
        Length of the whole parameter vector.
    is_ph : bool
        Whether K is a pH, and so bounded above as well as below.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Lower and upper bounds for `scipy.optimize.least_squares`.
    """
    lo = np.full(n_params, -np.inf)
    hi = np.full(n_params, np.inf)
    lo[:n_k] = _K_MIN_PH if is_ph else np.finfo(float).tiny
    if is_ph:
        hi[:n_k] = _K_MAX_PH
    return lo, hi


def apply_excluded_points(
    datasets: Mapping[str, Any],
    excluded_points: Mapping[str, Mapping[str, Sequence[int]]],
    *,
    min_keep: int = 5,
) -> dict[str, Any]:
    """Mask the points a screen rejected, so the next fit inherits its verdict.

    :func:`fit_plate_lm_screened` builds its masked copies internally and hands
    the caller's datasets back untouched, which is right for it and wrong for a
    pipeline: a Bayesian fit that runs afterwards would otherwise see every
    point the screen threw out. This applies the screen's ``excluded_points`` to
    a fresh copy, under the same ``min_keep`` guard the screen itself uses, so
    the two agree about which drops were actually taken.

    Parameters
    ----------
    datasets : Mapping[str, Any]
        Well key to dataset. Not modified.
    excluded_points : Mapping[str, Mapping[str, Sequence[int]]]
        Well to label to positional indices, as ``PlateLMResult.excluded_points``
        records them. Unknown wells, labels and out-of-range indices are
        ignored: a screen may have run over a superset of what is fitted here.
    min_keep : int
        Never leave a label with fewer unmasked points than this.

    Returns
    -------
    dict[str, Any]
        A copy with the rejected points masked out.
    """
    out: dict[str, Any] = {}
    for well, ds in datasets.items():
        arrays: dict[Any, Any] = {}
        uncleanable: list[Any] = []
        for lbl, da in ds.items():
            bad = set(excluded_points.get(well, {}).get(str(lbl), ()))
            mask = np.asarray(da.mask).copy()
            if bad:
                keep = mask.copy()
                for i in bad:
                    if 0 <= i < len(keep):
                        keep[i] = False
                if int(keep.sum()) >= min_keep:
                    mask = keep
                elif int(keep.sum()) < int(mask.sum()):
                    # Too few would survive to fit a curve. Refusing to remove
                    # them leaves the label in the fit *with* its outliers,
                    # which is the failure min_keep exists to prevent, pointing
                    # the other way: on L4, H10's 400 nm channel has three
                    # points 39-73% of its range off the sigmoid and all three
                    # were kept. Drop the label instead, as per-label bad-well
                    # detection already does, and fit the well on what is left.
                    uncleanable.append(lbl)
            new = copy.deepcopy(da)
            new.mask = mask
            arrays[lbl] = new
        # Never leave a well with nothing: a poor fit reports its own error
        # bars, an empty well says nothing and may break the caller.
        for lbl in uncleanable[: max(len(arrays) - 1, 0)]:
            del arrays[lbl]
        out[well] = type(ds)(arrays, is_ph=ds.is_ph)
    return out


# The screen's ruler must admit that the model is imperfect. sigma is the
# physical measurement noise, and at the dim end of a falling channel it
# collapses onto the read-noise floor -- the Poisson term is clamped at zero for
# a non-positive prediction, so a label with no proportional term has variance
# exactly floor^2, which is 0.42 counts here. A 1.3-count model error is then a
# 3-sigma event, and the screen deletes the transition points that pin the
# plateaus. Campaign-wide, 34% of |z|>3 flags miss by under 10% of their curve's
# range; 3% of the fitted span removes those and leaves the genuine ones.
_SCREEN_AMPLITUDE_FRACTION = 0.03


def screening_sigma_floor(
    residuals: Sequence[Mapping[str, Any]],
    *,
    fraction: float = _SCREEN_AMPLITUDE_FRACTION,
) -> dict[tuple[str, str], float]:
    """Smallest sigma the screen may judge a point by, per well and label.

    The amplitude is the span of the *fitted* curve over the observed points,
    ``max(yhat) - min(yhat)``, not the span of the data. An outlier inflates the
    observed span -- on these plates 31 cells of 1904 have a fitted span below
    half the observed one -- which would raise the floor and make the screen
    inert on exactly the curve that needed it. It is also not ``|S1 - S0|``,
    which is the range extrapolated beyond the titrated window and can far
    exceed anything the curve actually traverses.

    Parameters
    ----------
    residuals : Sequence[Mapping[str, Any]]
        Rows from a first-pass fit, carrying ``well``, ``label`` and ``yhat``.
    fraction : float
        Fraction of the amplitude to floor sigma at. ``0`` disables the floor
        and restores judging purely on the physical sigma.

    Returns
    -------
    dict[tuple[str, str], float]
        ``(well, label)`` to the floor, ``0.0`` where the fitted curve is flat
        and there is no amplitude to take a fraction of.
    """
    spans: dict[tuple[str, str], list[float]] = {}
    for row in residuals:
        yhat = float(row.get("yhat", float("nan")))
        if not np.isfinite(yhat):
            continue
        spans.setdefault((str(row["well"]), str(row["label"])), []).append(yhat)
    return {
        key: float(fraction * (max(v) - min(v))) if len(v) > 1 else 0.0
        for key, v in spans.items()
    }


# A z-score fails at both ends of a titration: sigma tracks the signal while
# model error tracks the curve, so a 5% miss at the dim end reads as 3.6 sigma
# and a 36% miss at the bright end reads as 2.7. Reviewer calls over three
# plates separate instead on the *signed* residual over the prediction. Only
# deficits are screened: both documented label-1 artefacts are shortfalls -- the
# acidic turnover falls below its peak, and the step-0 effect is a uniform ~7%
# multiplicative deficit -- while a positive excursion is scatter. Restricting
# to deficits is what makes the threshold work at all: reviewer keeps run from
# +0.097 to +0.112 and a discard sits at -0.093, so no cut on |frac| can order
# them. Over 50 adjudicated points on four plates, -0.09 separates cleanly.
_FRACTIONAL_THRESHOLD = 0.09
# Both channels moving the same way by a comparable fraction is a well-level
# multiplicative artefact, which the ratiometric measurement cancels; removing
# those points biases the plateau instead of cleaning it.
_RATIOMETRIC_RATIO = 0.7


def fractional_outliers(
    residuals: Sequence[Mapping[str, Any]],
    *,
    frac_threshold: float = _FRACTIONAL_THRESHOLD,
    ratiometric_ratio: float = _RATIOMETRIC_RATIO,
    label: str | None = None,
) -> dict[tuple[str, str], set[int]]:
    """Points whose fractional deviation is too large to be ordinary noise.

    Applied to the first label only. The second channel's plateau falls to zero,
    where a fraction of the prediction stops being meaningful; its outliers are
    left to the z-screen with its amplitude floor.

    Parameters
    ----------
    residuals : Sequence[Mapping[str, Any]]
        Rows from a fit, carrying ``well``, ``label``, ``step``, ``raw_i``,
        ``raw_res`` and ``yhat`` -- the schema ``fit_plate_lm`` produces, which
        has no separate ``y``.
    frac_threshold : float
        Deficits deeper than this fraction of the prediction are screened;
        ``raw_res / yhat`` must be below ``-frac_threshold``. Positive
        excursions are never screened.
    ratiometric_ratio : float
        Spare a point when the other channel moves the same way by at least this
        fraction of the first channel's move.
    label : str | None
        Label to screen. ``None`` takes the lowest label present, which is the
        400 nm channel on these plates.

    Returns
    -------
    dict[tuple[str, str], set[int]]
        ``(well, label)`` to the ``raw_i`` values to drop.
    """
    by_key: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = {}
    for row in residuals:
        by_key.setdefault((str(row["well"]), int(row["step"])), {})[
            str(row["label"])
        ] = row
    labels = {str(r["label"]) for r in residuals}
    if not labels:
        return {}
    target = label if label is not None else min(labels)

    def frac(row: Mapping[str, Any] | None) -> float:
        if row is None:
            return float("nan")
        yhat = float(row["yhat"])
        # raw_res is y - yhat; the rows carry no separate y.
        return float("nan") if yhat == 0 else float(row["raw_res"]) / yhat

    out: dict[tuple[str, str], set[int]] = {}
    for (well, _step), per_label in by_key.items():
        target_row = per_label.get(target)
        if target_row is None:
            continue
        f1 = frac(target_row)
        # Deficits only: a positive excursion is scatter, not a known artefact.
        if not np.isfinite(f1) or f1 >= -frac_threshold:
            continue
        others = [frac(r) for lbl, r in per_label.items() if lbl != target]
        shared = any(
            np.isfinite(f2)
            and np.sign(f2) == np.sign(f1)
            and abs(f2) >= ratiometric_ratio * abs(f1)
            for f2 in others
        )
        if not shared:
            out.setdefault((well, target), set()).add(int(target_row["raw_i"]))
    return out


def ratiometric_exempt(
    residuals: Sequence[Mapping[str, Any]],
    *,
    frac_threshold: float = _FRACTIONAL_THRESHOLD,
    ratiometric_ratio: float = _RATIOMETRIC_RATIO,
) -> dict[tuple[str, str], set[int]]:
    """Points no screen should take, because the ratio already cancels them.

    When both channels of a well shift the same way at the same step by a
    comparable fraction, the cause is a well-level multiplicative artefact -- a
    volume or read error -- and a ratiometric measurement is immune to it.
    Removing such points biases the plateau instead of cleaning it.

    This is a veto over every screening criterion, not a clause of one. On L3
    C06 the z-screen takes label 2 at |z| 6.8 and 6.7 while both channels are
    simply reading 15% high then 15% low; sparing those points only inside the
    fractional criterion leaves the z-screen free to take them anyway.

    Parameters
    ----------
    residuals : Sequence[Mapping[str, Any]]
        Rows from a fit, carrying ``well``, ``label``, ``step``, ``raw_i``,
        ``raw_res`` and ``yhat``.
    frac_threshold : float
        Only shifts larger than this are treated as artefacts; ordinary noise
        moves both channels a little and is not exempted.
    ratiometric_ratio : float
        How closely the two channels must agree, as a fraction of the larger
        move.

    Returns
    -------
    dict[tuple[str, str], set[int]]
        ``(well, label)`` to the ``raw_i`` values to spare, for every label
        involved in the shared shift.
    """
    by_key: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = {}
    for row in residuals:
        by_key.setdefault((str(row["well"]), int(row["step"])), {})[
            str(row["label"])
        ] = row

    def frac(row: Mapping[str, Any]) -> float:
        yhat = float(row["yhat"])
        return float("nan") if yhat == 0 else float(row["raw_res"]) / yhat

    out: dict[tuple[str, str], set[int]] = {}
    for (well, _step), per_label in by_key.items():
        fracs = {lbl: frac(row) for lbl, row in per_label.items()}
        usable = {k: v for k, v in fracs.items() if np.isfinite(v)}
        if len(usable) < 2:  # ruff: ignore[magic-value-comparison] - a ratio needs two channels
            continue
        big = max(usable.values(), key=abs)
        if abs(big) <= frac_threshold:
            continue
        if all(
            np.sign(v) == np.sign(big) and abs(v) >= ratiometric_ratio * abs(big)
            for v in usable.values()
        ):
            for lbl in usable:
                out.setdefault((well, lbl), set()).add(int(per_label[lbl]["raw_i"]))
    return out


def _screen_refit(
    data: Mapping[str, Any],
    groups: Mapping[str, Sequence[str]],
    noise_model: Mapping[str, Any] | None,
) -> PlateLMResult:
    """Fit K after a screen: calibrated on ``noise_model`` if given, else plain.

    Parameters
    ----------
    data : Mapping[str, Any]
        Well identifier to `Dataset`, screened or not.
    groups : Mapping[str, Sequence[str]]
        Control group name to member wells.
    noise_model : Mapping[str, Any] | None
        Starting noise model for a calibrated refit; ``None`` for the plain one.

    Returns
    -------
    PlateLMResult
        The refit.
    """
    if noise_model is None:
        return fit_plate_lm(data, groups)
    return fit_plate_lm(data, groups, noise_model=noise_model, calibrate_noise=True)


def fit_plate_lm_screened(  # ruff: ignore[too-many-arguments] - each knob is an independent screening choice
    datasets: Mapping[str, Any],
    groups: Mapping[str, Sequence[str]],
    *,
    noise_model: Mapping[str, Any] | None = None,
    threshold: float = 3.0,
    min_keep: int = 5,
    amplitude_fraction: float = _SCREEN_AMPLITUDE_FRACTION,
    frac_threshold: float | None = None,
    calibrate_screen: bool = True,
    calibrate_refit: bool = False,
) -> PlateLMResult:
    """Find outliers with a calibrated ruler, then fit K with the plain one.

    A standardised residual is only as good as the sigma it is divided by, and
    the default sigma is one number per label - the read-noise floor - while the
    noise plainly grows with signal. On L6a the binned |std residual| runs from
    0.065 at low signal to 1.113 at high, so a fixed threshold is a far harsher
    test of a bright point than a dim one. Screening on that ruler discards the
    plateaus, which are what pin S0 and S1 and hence where the midpoint sits:
    over eleven plates it threw away 947 points and made K worse.

    So the two jobs are split, each given the model measured to win at it. The
    screening pass calibrates gain and alpha per label from its own residuals,
    which puts every point on a comparable scale; the refit then uses the plain
    weighting, because calibrated weights describe residuals better and fit K
    worse. Same threshold, right ruler: 99 points dropped instead of 947, and
    sum_log -5.92 against -5.49 for no screening at all, improving nine plates
    of eleven.

    The threshold is not a free knob. At 2.5 the screen turns harmful again
    (-4.42), so 3.0 is not a rounding of "about three sigma" but the value that
    separates a useful screen from a destructive one.

    Parameters
    ----------
    datasets : Mapping[str, Any]
        Well identifier to `Dataset`.
    groups : Mapping[str, Sequence[str]]
        Control group name to member wells; those wells share one K.
    noise_model : Mapping[str, Any] | None
        Per-label noise parameters supplying the floors the calibration pins.
        Without it there is nothing to calibrate and the screening pass falls
        back to the plain fit, which is the behaviour this function exists to
        avoid - pass it.
    threshold : float
        Standardised-residual magnitude above which a point is dropped, judged
        on the calibrated scale.
    min_keep : int
        Never leave a label with fewer points than this, whatever their
        residuals. Four cannot locate a midpoint and two plateaus: on L4, E03
        lost three of seven, landed exactly on the old default of four, and its
        fit collapsed to a flat line - taking the well's other label with it,
        since K is shared between them.
    amplitude_fraction : float
        Floor the screening sigma at this fraction of each curve's fitted span,
        so a point is judged against model error and not only against
        measurement noise. ``0`` restores judging on the physical sigma alone.
        See :func:`screening_sigma_floor`.
    frac_threshold : float | None
        When given, also screen the first label on ``|y - yhat| / yhat``
        exceeding this, sparing points the second channel moves with. ``None``
        leaves the z-screen alone. See :func:`fractional_outliers`.
    calibrate_screen : bool
        Judge points on the calibrated ruler (the default, and the reason this
        function exists) or on the weights as built.
    calibrate_refit : bool
        Fit K with calibrated weights instead of the plain ones. The two jobs
        used to be fixed to one choice each, and the CLI then reused
        ``--plate-noise calibrated`` to mean calibrated weights without a
        screen but plain weights after one -- a confound worth ~1.1 in the
        eleven-plate bench score, credited to the screen. Each is now its own
        switch; the defaults keep the split described above.

    Returns
    -------
    PlateLMResult
        The refit, carrying the count of dropped observations in
        ``n_excluded``. When nothing crosses the threshold this is the
        single-pass fit.
    """
    first = fit_plate_lm(
        datasets,
        groups,
        noise_model=noise_model,
        calibrate_noise=calibrate_screen and noise_model is not None,
    )
    floors = screening_sigma_floor(first.residuals, fraction=amplitude_fraction)
    drop: dict[tuple[str, str], set[int]] = {}
    for row in first.residuals:
        key = (str(row["well"]), str(row["label"]))
        # Judge on sigma or the amplitude floor, whichever is larger. The refit
        # below still weights by the physical sigma: only the decision changes.
        sigma = max(float(row["sigma"]), floors.get(key, 0.0))
        if abs(float(row["raw_res"])) / sigma > threshold:
            drop.setdefault(key, set()).add(int(row["raw_i"]))
    if frac_threshold is not None:
        for key, idx in fractional_outliers(
            first.residuals, frac_threshold=frac_threshold
        ).items():
            drop.setdefault(key, set()).update(idx)
        # A veto over every criterion above, including the z-screen.
        for key, idx in ratiometric_exempt(
            first.residuals, frac_threshold=frac_threshold
        ).items():
            if key in drop:
                drop[key] -= idx
                if not drop[key]:
                    del drop[key]
    refit_noise = noise_model if calibrate_refit else None
    if not drop:
        return _screen_refit(datasets, groups, refit_noise)

    # One implementation, so the screen's own refit and any fit downstream of it
    # agree about which points were taken and which labels were dropped whole.
    by_well: dict[str, dict[str, list[int]]] = {}
    for (well, lbl), idx in drop.items():
        by_well.setdefault(well, {})[lbl] = sorted(idx)
    screened = apply_excluded_points(datasets, by_well, min_keep=min_keep)
    n_excluded = sum(
        int(np.asarray(da.mask).sum()) for ds in datasets.values() for da in ds.values()
    ) - sum(
        int(np.asarray(da.mask).sum()) for ds in screened.values() for da in ds.values()
    )

    out = _screen_refit(screened, groups, refit_noise)
    out.n_excluded = n_excluded
    for (well, lbl), idx in drop.items():
        out.excluded_points.setdefault(well, {})[lbl] = sorted(idx)
    return out
