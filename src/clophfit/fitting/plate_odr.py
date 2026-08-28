"""Plate-wide fit that lets the pH axis move, not only the signal.

`plate_lm` takes x at face value. That is the wrong assumption here twice over:
the recorded pH of each titration step carries its own measured uncertainty -
`list.pH.csv` has it in the third column, 0.006 to 0.086 on L6a - and section 2
found that modelling x as uncertain is the single largest improvement anywhere in
this campaign, with every latent-x arm beating every fixed-x one.

This is the classical counterpart: an errors-in-variables fit, which is what
orthogonal distance regression is. Each titration step gets one shift, penalised
by its recorded uncertainty, and the shift is *shared across wells* because the
pH of a step is a property of the plate rather than of a well. Seven extra
parameters for a seven-step titration, not one per point.

The scale behaves differently here than in `plate_lm`, and that is the reason to
have both. A least-squares objective is invariant to scaling every weight, so the
noise scale is not identifiable and has to be profiled. Once x errors enter, the
*ratio* between the y and x penalties is part of the objective, so the balance
between them is doing real work rather than cancelling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.optimize import least_squares

from clophfit.fitting.models import binding_1site
from clophfit.fitting.plate_lm import (
    _MIN_GROUP_FOR_HOLDOUT,
    _assemble,
    _holdout_row,
    _k_bounds,
    _profiled_scale,
    _standard_errors,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = ["PlateODRResult", "ctr_holdout_odr", "fit_plate_odr"]


@dataclass
class PlateODRResult:
    """Outcome of one plate-wide errors-in-variables fit.

    Parameters
    ----------
    k : dict[str, float]
        Fitted K per well; control-group members share one value.
    k_stderr : dict[str, float]
        Standard error per well from the Jacobian at the solution.
    ye_mag : dict[str, float]
        Profiled noise multiplier per label.
    dx : np.ndarray
        Fitted shift of each titration step, in pH, shared across wells.
    n_points : int
        Unmasked observations entering the fit.
    n_params : int
        Free parameters, structural shifts included.
    success : bool
        Whether the final solve converged.
    residuals : list[dict[str, Any]]
        One record per unmasked observation on the library's canonical columns
        (``well``, ``label``, ``step``, ``yhat``, ``raw_res``, ``sigma``,
        ``std_res``), taken on the shifted pH grid, so it reflects the model the
        fit actually settled on rather than the recorded axis.
    """

    k: dict[str, float] = field(default_factory=dict)
    k_stderr: dict[str, float] = field(default_factory=dict)
    ye_mag: dict[str, float] = field(default_factory=dict)
    dx: np.ndarray = field(default_factory=lambda: np.zeros(0))
    n_points: int = 0
    n_params: int = 0
    success: bool = False
    residuals: list[dict[str, Any]] = field(default_factory=list)


def fit_plate_odr(
    datasets: Mapping[str, Any],
    groups: Mapping[str, Sequence[str]],
    *,
    x_err: np.ndarray | None = None,
    max_iter: int = 6,
    tol: float = 1e-3,
) -> PlateODRResult:
    """Fit a plate with the pH of each step free to move within its uncertainty.

    Parameters
    ----------
    datasets : Mapping[str, Any]
        Well identifier to `Dataset`.
    groups : Mapping[str, Sequence[str]]
        Control group name to member wells; those wells share one K.
    x_err : np.ndarray | None
        Recorded uncertainty per titration step. ``None`` uses 0.015 pH, the
        pipetting scale section 2 settled on.
    max_iter : int
        Maximum alternations between solving and rescaling.
    tol : float
        Stop when every scale moves by less than this, relatively.

    Returns
    -------
    PlateODRResult
        Fitted K per well, the per-step pH shifts, and the profiled scales.

    Raises
    ------
    ValueError
        If *x_err* does not carry one entry per titration step.
    """
    prob = _assemble(datasets, groups)
    n_steps = max((len(x) for _, _, x, _, _ in prob.obs), default=0)
    sigma_x = (
        np.full(n_steps, 0.015) if x_err is None else np.asarray(x_err, dtype=float)
    )
    if len(sigma_x) != n_steps:  # pragma: no cover - caller mismatch
        msg = f"x_err has {len(sigma_x)} entries for {n_steps} steps."
        raise ValueError(msg)

    scales = dict.fromkeys(prob.labels, 1.0)
    n_struct = len(prob.p0)
    p = np.concatenate([prob.p0, np.zeros(n_steps)])
    bounds = _k_bounds(prob.n_k, len(p), is_ph=prob.is_ph)

    def residuals(params: np.ndarray) -> np.ndarray:
        dx = params[n_struct:]
        parts = []
        for well, lbl, x, y, yerr in prob.obs:
            s0 = params[prob.sidx[well, lbl]]
            s1 = params[prob.sidx[well, lbl] + 1]
            shifted = x + dx[: len(x)]
            model = binding_1site(
                shifted, params[prob.kidx[well]], s0, s1, is_ph=prob.is_ph
            )
            parts.append((y - model) / (yerr * scales[lbl]))
        # One penalty per step: a shift costs what its recorded uncertainty says.
        parts.append(dx / sigma_x)
        return np.concatenate(parts)

    fit = None
    for _ in range(max_iter):
        fit = least_squares(residuals, p, method="trf", bounds=bounds)
        p = fit.x
        moved = 0.0
        for lbl in prob.labels:
            resid = _label_residuals(prob, p, lbl, n_struct)
            new = _profiled_scale(prob, resid, lbl)
            moved = max(moved, abs(new - scales[lbl]) / new)
            scales[lbl] = new
        if moved < tol:
            break

    result = PlateODRResult(
        ye_mag=dict(scales),
        dx=p[n_struct:].copy(),
        n_points=prob.n_points,
        n_params=len(p),
        success=bool(fit.success) if fit is not None else False,
    )
    if fit is None:  # pragma: no cover - least_squares always returns
        return result
    result.residuals = _residual_table(prob, p, scales, n_struct)
    err = _standard_errors(np.asarray(fit.jac, dtype=float), len(p))
    for well in datasets:
        result.k[well] = float(p[prob.kidx[well]])
        result.k_stderr[well] = float(err[prob.kidx[well]])
    return result


def _residual_table(
    prob: Any,  # ruff: ignore[any-type] - the private _Problem record
    p: np.ndarray,
    scales: Mapping[str, float],
    n_struct: int,
) -> list[dict[str, Any]]:
    """Flatten the fit into the canonical residual table the metrics expect.

    Parameters
    ----------
    prob : Any
        Assembled problem.
    p : np.ndarray
        Parameter vector at the solution, shifts appended.
    scales : Mapping[str, float]
        Profiled noise multiplier per label.
    n_struct : int
        Where the shifts start in *p*.

    Returns
    -------
    list[dict[str, Any]]
        One row per observation on the canonical residual columns.
    """
    dx = p[n_struct:]
    rows: list[dict[str, Any]] = []
    for well, lbl, x, y, yerr in prob.obs:
        s0, s1 = p[prob.sidx[well, lbl]], p[prob.sidx[well, lbl] + 1]
        model = binding_1site(
            x + dx[: len(x)], p[prob.kidx[well]], s0, s1, is_ph=prob.is_ph
        )
        sigma = yerr * scales[lbl]
        raw = y - model
        order = np.argsort(np.argsort(x))
        rows.extend(
            {
                "well": well,
                "label": lbl,
                "step": int(step),
                "yhat": float(yh),
                "raw_res": float(r),
                "sigma": float(sd),
                "std_res": float(r / sd),
            }
            for step, yh, r, sd in zip(order, model, raw, sigma, strict=True)
        )
    return rows


def _label_residuals(
    prob: Any,  # ruff: ignore[any-type] - the private _Problem record
    p: np.ndarray,
    label: str,
    n_struct: int,
) -> np.ndarray:
    """Standardised residuals for one label, on the shifted x grid.

    Parameters
    ----------
    prob : Any
        Assembled problem.
    p : np.ndarray
        Current parameter vector, shifts appended.
    label : str
        Label to collect.
    n_struct : int
        Where the shifts start in *p*.

    Returns
    -------
    np.ndarray
        Residuals divided by ``y_err`` only.
    """
    dx = p[n_struct:]
    parts = []
    for well, lbl, x, y, yerr in prob.obs:
        if lbl != label:
            continue
        s0, s1 = p[prob.sidx[well, lbl]], p[prob.sidx[well, lbl] + 1]
        model = binding_1site(
            x + dx[: len(x)], p[prob.kidx[well]], s0, s1, is_ph=prob.is_ph
        )
        parts.append((y - model) / yerr)
    return np.concatenate(parts) if parts else np.zeros(0)


def ctr_holdout_odr(
    datasets: Mapping[str, Any],
    groups: Mapping[str, Sequence[str]],
    *,
    x_err: np.ndarray | None = None,
    rope: float = 0.10,
    plate: str = "",
) -> list[dict[str, Any]]:
    """Leave one control out at a time, with the pH axis free to move.

    The same experiment as :func:`clophfit.fitting.plate_lm.ctr_holdout`, fitted with
    errors in variables so a wrongly recorded step is corrected rather than
    pushed into K. Rows share the schema, so the two can be pooled together.

    Parameters
    ----------
    datasets : Mapping[str, Any]
        Well identifier to `Dataset`.
    groups : Mapping[str, Sequence[str]]
        Control group name to member wells.
    x_err : np.ndarray | None
        Recorded uncertainty per titration step, from the third column of
        ``list.pH.csv``. ``None`` falls back to 0.015 pH.
    rope : float
        Half-width of the region of practical equivalence, in pH.
    plate : str
        Recorded in each row, for pooling across plates.

    Returns
    -------
    list[dict[str, Any]]
        One row per held-out control.
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
            fit = fit_plate_odr(datasets, trimmed, x_err=x_err)
            rows.append(
                _holdout_row(
                    plate=plate,
                    cell="plate_odr",
                    group=group,
                    held=held,
                    n_remaining=len(remaining),
                    k_held=fit.k.get(held, float("nan")),
                    k_ref=fit.k.get(remaining[0], float("nan")),
                    se_held=fit.k_stderr.get(held, float("nan")),
                    se_ref=fit.k_stderr.get(remaining[0], float("nan")),
                    ye_mag=fit.ye_mag,
                    success=fit.success,
                    rope=rope,
                )
            )
    return rows
