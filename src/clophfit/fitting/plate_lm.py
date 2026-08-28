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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.optimize import least_squares

from clophfit.fitting.models import binding_1site

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = ["PlateLMResult", "fit_plate_lm"]

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
    n_points : int
        Unmasked observations entering the fit.
    n_params : int
        Free structural parameters.
    success : bool
        Whether the final least-squares solve converged.
    residuals : list[dict[str, Any]]
        One record per unmasked observation, with ``well``, ``label``, ``step``
        and ``std_res``.  ``std_res`` divides by ``y_err`` *and* the profiled
        scale, so its per-label SD is ~1 by construction and only its shape -
        tails, step dependence - carries information.
    """

    k: dict[str, float] = field(default_factory=dict)
    k_stderr: dict[str, float] = field(default_factory=dict)
    ye_mag: dict[str, float] = field(default_factory=dict)
    n_points: int = 0
    n_params: int = 0
    success: bool = False
    residuals: list[dict[str, Any]] = field(default_factory=list)


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
    for well, dataset in datasets.items():
        for lbl, da in dataset.items():
            m = np.asarray(da.mask)
            x = np.asarray(da.xc, dtype=float)[m]
            y = np.asarray(da.yc, dtype=float)[m]
            yerr = np.asarray(da.y_errc, dtype=float)[m]
            sidx[well, lbl] = len(p0)
            p0.extend([float(y[0]) if len(y) else 0.0, float(y[-1]) if len(y) else 0.0])
            obs.append((well, lbl, x, y, yerr))

    return _Problem(
        kidx=kidx,
        sidx=sidx,
        p0=np.asarray(p0, dtype=float),
        obs=obs,
        labels=labels,
        is_ph=bool(next(iter(datasets.values())).is_ph),
        n_k=n_k,
        n_points=int(sum(len(x) for _, _, x, _, _ in obs)),
        n_curves={lbl: sum(1 for _, o, _, _, _ in obs if o == lbl) for lbl in labels},
    )


def _raw_residuals(prob: _Problem, p: np.ndarray, label: str) -> np.ndarray:
    """Return one label's residuals divided by ``y_err`` only, not by any scale.

    Parameters
    ----------
    prob : _Problem
        Assembled problem.
    p : np.ndarray
        Current parameter vector.
    label : str
        Label to collect.

    Returns
    -------
    np.ndarray
        Standardised residuals for that label.
    """
    parts = []
    for well, lbl, x, y, yerr in prob.obs:
        if lbl != label:
            continue
        s0, s1 = p[prob.sidx[well, lbl]], p[prob.sidx[well, lbl] + 1]
        model = binding_1site(x, p[prob.kidx[well]], s0, s1, is_ph=prob.is_ph)
        parts.append((y - model) / yerr)
    return np.concatenate(parts) if parts else np.zeros(0)


def _residual_table(
    prob: _Problem, p: np.ndarray, scales: Mapping[str, float]
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

    Returns
    -------
    list[dict[str, Any]]
        ``well``, ``label``, ``step`` and ``std_res`` per observation.  ``step``
        is the rank of the point on the titration axis, so masked points do not
        shift the numbering of the ones that survive.
    """
    rows: list[dict[str, Any]] = []
    for well, lbl, x, y, yerr in prob.obs:
        s0, s1 = p[prob.sidx[well, lbl]], p[prob.sidx[well, lbl] + 1]
        model = binding_1site(x, p[prob.kidx[well]], s0, s1, is_ph=prob.is_ph)
        std = (y - model) / (yerr * scales[lbl])
        order = np.argsort(np.argsort(x))
        rows.extend(
            {"well": well, "label": lbl, "step": int(step), "std_res": float(r)}
            for step, r in zip(order, std, strict=True)
        )
    return rows


def _profiled_scale(prob: _Problem, resid: np.ndarray, label: str) -> float:
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

    Returns
    -------
    float
        Positive noise multiplier.
    """
    n_lbl = len(resid)
    if n_lbl == 0:
        return 1.0
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
    try:
        _, s, vt = np.linalg.svd(np.asarray(jac, dtype=float), full_matrices=False)
    except np.linalg.LinAlgError:  # pragma: no cover - singular Jacobian
        return np.full(n_params, np.nan)
    keep = s > np.finfo(float).eps * max(jac.shape) * s[0]
    cov = (vt[keep].T / s[keep] ** 2) @ vt[keep]
    out: np.ndarray = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    return out


def fit_plate_lm(
    datasets: Mapping[str, Any],
    groups: Mapping[str, Sequence[str]],
    *,
    max_iter: int = 6,
    tol: float = 1e-3,
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

    Returns
    -------
    PlateLMResult
        Fitted K per well, its standard error, and the profiled scales.
    """
    prob = _assemble(datasets, groups)
    scales = dict.fromkeys(prob.labels, 1.0)
    p = prob.p0
    bounds = _k_bounds(prob.n_k, len(p), is_ph=prob.is_ph)

    def residuals(params: np.ndarray) -> np.ndarray:
        parts = []
        for well, lbl, x, y, yerr in prob.obs:
            s0, s1 = params[prob.sidx[well, lbl]], params[prob.sidx[well, lbl] + 1]
            model = binding_1site(x, params[prob.kidx[well]], s0, s1, is_ph=prob.is_ph)
            parts.append((y - model) / (yerr * scales[lbl]))
        return np.concatenate(parts) if parts else np.zeros(0)

    fit = None
    for _ in range(max_iter):
        fit = least_squares(residuals, p, method="trf", bounds=bounds)
        p = fit.x
        moved = 0.0
        for lbl in prob.labels:
            new = _profiled_scale(prob, _raw_residuals(prob, p, lbl), lbl)
            moved = max(moved, abs(new - scales[lbl]) / new)
            scales[lbl] = new
        if moved < tol:
            break

    result = PlateLMResult(
        ye_mag=dict(scales),
        n_points=prob.n_points,
        n_params=len(p),
        success=bool(fit.success) if fit is not None else False,
    )
    if fit is None:
        return result
    result.residuals = _residual_table(prob, p, scales)
    err = _standard_errors(np.asarray(fit.jac, dtype=float), len(p))
    for well in datasets:
        result.k[well] = float(p[prob.kidx[well]])
        result.k_stderr[well] = float(err[prob.kidx[well]])
    return result


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
