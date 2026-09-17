"""Calibrate a gain-only noise model from a fit's own residuals, then refit.

Feasible generalised least squares with ``sigma^2 = floor^2 + gain * yhat`` and
alpha held at 0: fit every well under the current weights, pool the residuals
of all wells, estimate the gain (and optionally the floor), rebuild the
weights, refit, and repeat until the noise terms stop moving. The pool is one
plate or several; with several, one gain per label is shared by every plate and
each plate keeps its own floor.

Two fitters plug into the same loop: the per-well least-squares fit
(:func:`~clophfit.fitting.core.fit_binding_glob`, ``lm`` or ``huber``) and the
plate-wide :func:`~clophfit.fitting.plate_lm.fit_plate_lm`.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from itertools import starmap
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from clophfit.fitting.core import fit_binding_glob
from clophfit.fitting.data_structures import FitResult, NoiseModelParams
from clophfit.fitting.errors import InsufficientDataError
from clophfit.fitting.models import binding_1site
from clophfit.fitting.noise_calibration import dof_scale, fit_gain_nnls
from clophfit.fitting.plate_lm import (
    PlateLMResult,
    _damped_update,
    _DampingState,
    fit_plate_lm,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from clophfit.fitting.data_structures import Dataset

__all__ = [
    "GainCalibration",
    "calibrate_gain",
    "calibrate_plate_lm",
    "calibrate_single_well",
    "plate_lm_residuals",
    "single_well_residuals",
]

logger = logging.getLogger(__name__)

#: Plate to label to noise parameters.
NoiseByPlate = dict[str, dict[str, NoiseModelParams]]


@dataclass
class GainCalibration[T]:
    """What the calibration loop converged to, and the fits made under it.

    Attributes
    ----------
    fits : dict[str, T]
        Plate to its fit under ``noise``, the weights the last pass used.
    noise : NoiseByPlate
        Plate to label to the floor and gain in force for ``fits``; alpha is 0.
    history : pd.DataFrame
        One row per pass, plate and label: ``iteration``, ``plate``, ``label``,
        ``gain`` and ``sigma_floor`` in force for that pass, and ``n``, the
        residuals that pass contributed.
    converged : bool
        Whether the last adopted step moved every term by less than the
        tolerance. Damping halves a term's step whenever it reverses, so a loop
        that oscillates settles between its two answers and still counts as
        converged; ``proposal_moved`` says whether that is a fixed point.
    proposal_moved : float
        Largest relative change the last *undamped* estimate asked for. Below
        the tolerance, the noise is a fixed point of fit -> estimate -> refit;
        above it, the loop settled by damping alone.
    n_iter : int
        Fitting passes made.
    """

    fits: dict[str, T] = field(default_factory=dict)
    noise: NoiseByPlate = field(default_factory=dict)
    history: pd.DataFrame = field(default_factory=pd.DataFrame)
    converged: bool = False
    proposal_moved: float = float("nan")
    n_iter: int = 0


def single_well_residuals(results: Mapping[str, FitResult]) -> pd.DataFrame:
    """Raw residuals of per-well fits, with each well's degrees-of-freedom factor.

    Parameters
    ----------
    results : Mapping[str, FitResult]
        Well to its fit; wells without a result are skipped.

    Returns
    -------
    pd.DataFrame
        ``well``, ``label``, ``yhat``, ``raw_res`` and ``dof_scale`` (one
        value per well, from all its labels' points and varied parameters).
    """
    frames: list[pd.DataFrame] = []
    for well, fr in results.items():
        if fr.dataset is None or fr.result is None:
            continue
        pars = fr.result.params
        n_params = sum(bool(p.vary) for p in pars.values())
        parts = []
        for lbl, da in fr.dataset.items():
            yhat = binding_1site(
                da.x,
                pars["K"].value,
                pars[f"S0_{lbl}"].value,
                pars[f"S1_{lbl}"].value,
                is_ph=fr.dataset.is_ph,
            )
            parts.append(
                pd.DataFrame({
                    "well": well,
                    "label": str(lbl),
                    "yhat": np.asarray(yhat, dtype=float),
                    "raw_res": np.asarray(da.y, dtype=float) - yhat,
                })
            )
        if not parts:
            continue
        frame = pd.concat(parts, ignore_index=True)
        frame["dof_scale"] = dof_scale(len(frame), n_params)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plate_lm_residuals(result: PlateLMResult) -> pd.DataFrame:
    """Raw residuals of a plate-wide fit, with the plate's degrees-of-freedom factor.

    Parameters
    ----------
    result : PlateLMResult
        A plate fit.

    Returns
    -------
    pd.DataFrame
        ``well``, ``label``, ``yhat``, ``raw_res`` and ``dof_scale``.
    """
    if not result.residuals:
        return pd.DataFrame()
    frame = pd.DataFrame(result.residuals)[["well", "label", "yhat", "raw_res"]]
    frame["label"] = frame["label"].astype(str)
    frame["dof_scale"] = dof_scale(result.n_points, result.n_params)
    return frame


def _floor_only(floors: Mapping[str, Mapping[str, float]]) -> NoiseByPlate:
    """Start every plate on its floor alone, the weighting before any calibration."""
    return {
        plate: {lbl: NoiseModelParams(sigma_floor=float(f)) for lbl, f in per.items()}
        for plate, per in floors.items()
    }


def _estimate(
    pooled: pd.DataFrame,
    noise: NoiseByPlate,
    gain_amp: Mapping[str, Mapping[str, float]] | None,
    *,
    fit_floor: bool,
) -> NoiseByPlate:
    """Pooled floor and gain from one pass's residuals, as a new noise model."""

    def floor_of(plate: str, lbl: str) -> float:
        params = noise.get(plate, {}).get(lbl)
        return float("nan") if params is None else float(params.sigma_floor)

    table = pooled.assign(
        sigma_floor=list(
            starmap(floor_of, zip(pooled["plate"], pooled["label"], strict=True))
        ),
        gain_amp=[
            (gain_amp or {}).get(p, {}).get(lbl, 1.0)
            for p, lbl in zip(pooled["plate"], pooled["label"], strict=True)
        ],
    )
    est = fit_gain_nnls(table, fit_floor=fit_floor)
    new = copy.deepcopy(noise)
    for plate, label, floor, gain in zip(
        est["plate"].astype(str),
        est["label"].astype(str),
        est["sigma_floor"].to_numpy(dtype=float),
        est["gain"].to_numpy(dtype=float),
        strict=True,
    ):
        amp = (gain_amp or {}).get(plate, {}).get(label, 1.0)
        new[plate][label] = NoiseModelParams(
            sigma_floor=float(floor), gain=float(gain) * amp
        )
    return new


def _moved(old: NoiseByPlate, new: NoiseByPlate) -> float:
    """Largest relative change of any floor or gain between two noise models."""
    worst = 0.0
    for plate, per in new.items():
        for lbl, params in per.items():
            before = old[plate][lbl]
            for attr in ("sigma_floor", "gain"):
                a, b = getattr(before, attr), getattr(params, attr)
                worst = max(worst, abs(b - a) / max(abs(a), abs(b), 1e-12))
    return worst


def calibrate_gain[T](  # ruff: ignore[too-many-arguments] - one per loop input
    fit: Callable[[str, dict[str, NoiseModelParams]], tuple[T, pd.DataFrame]],
    floors: Mapping[str, Mapping[str, float]],
    *,
    fit_floor: bool = False,
    gain_amp: Mapping[str, Mapping[str, float]] | None = None,
    max_iter: int = 20,
    tol: float = 1e-3,
) -> GainCalibration[T]:
    """Iterate fit -> pooled residuals -> floor and gain -> refit, to a fixed point.

    The first pass weights each plate by its floor alone. Each later pass
    weights by the floor and gain the previous pass's residuals gave, pooled
    over every well of every plate in *floors*. A term whose update reverses
    direction takes half the step from then on, which settles a loop that would
    otherwise cycle. The loop stops when no floor or gain moves by more than
    *tol* (relative), and returns the fits made under the weights that no
    longer move.

    Parameters
    ----------
    fit : Callable[[str, dict[str, NoiseModelParams]], tuple[T, pd.DataFrame]]
        Fits one plate under one noise model, returning the fit and its raw
        residual table (``label``, ``yhat``, ``raw_res``, ``dof_scale``), as
        :func:`single_well_residuals` and :func:`plate_lm_residuals` build.
    floors : Mapping[str, Mapping[str, float]]
        Plate to label to its read-noise floor, held unless *fit_floor*, and
        the only weights of the first pass either way.
    fit_floor : bool
        Estimate each plate's floor along with the shared gain.
    gain_amp : Mapping[str, Mapping[str, float]] | None
        Plate to label to the amplification its gain scales with; the shared
        gain is quoted at 1. ``None`` shares the gain unscaled.
    max_iter : int
        Most fitting passes.
    tol : float
        Relative change below which the noise terms have converged.

    Returns
    -------
    GainCalibration[T]
        The last pass's fits and the noise they were made under.

    Raises
    ------
    ValueError
        If a pass leaves no residuals to calibrate from.
    """
    noise = _floor_only(floors)
    damping = {plate: _DampingState() for plate in floors}
    out: GainCalibration[T] = GainCalibration()
    history: list[dict[str, Any]] = []
    for it in range(max_iter):
        frames = []
        for plate in floors:
            out.fits[plate], table = fit(plate, noise[plate])
            if not table.empty:
                frames.append(table.assign(plate=plate))
            for lbl, params in noise[plate].items():
                n = int((table["label"] == lbl).sum()) if not table.empty else 0
                history.append({
                    "iteration": it,
                    "plate": plate,
                    "label": lbl,
                    "gain": params.gain,
                    "sigma_floor": params.sigma_floor,
                    "n": n,
                })
        if not frames:
            msg = "no plate produced residuals to calibrate the noise from"
            raise ValueError(msg)
        out.n_iter = it + 1
        proposed = _estimate(
            pd.concat(frames, ignore_index=True), noise, gain_amp, fit_floor=fit_floor
        )
        # Damped as plate_lm's own calibration is: a term takes its proposal
        # until its change reverses sign, then half the step from there on.
        # A huber fit needs it - its threshold is in units of sigma, so each
        # new weighting flips which points it down-weights, and on L2 the gain
        # cycled between 4.76 and 5.13 undamped. The first move, off the
        # floor-only start, is taken whole and not remembered as a direction:
        # the small correction after that jump is no sign of a cycle.
        step = {
            plate: proposed[plate]
            if it == 0
            else dict(
                _damped_update(noise[plate], proposed[plate], damping[plate]) or {}
            )
            for plate in noise
        }
        moved = _moved(noise, step)
        out.proposal_moved = _moved(noise, proposed)
        logger.info(
            "gain calibration pass %d: largest change %.2g (proposal %.2g)",
            it + 1,
            moved,
            out.proposal_moved,
        )
        if moved < tol:
            out.converged = True
            break
        if it + 1 < max_iter:
            # Adopted only if a refit follows: the returned noise must be the
            # one the returned fits were made under, converged or not.
            noise = step
    out.noise = noise
    out.history = pd.DataFrame(history)
    return out


def _weighted(
    ds: Dataset,
    noise: Mapping[str, NoiseModelParams],
    prev: Mapping[str, Any] | None,
) -> Dataset:
    """Return a copy of *ds* whose y_err is *noise* at the previous prediction.

    Evaluated at the prediction rather than the observation, so a point's own
    fluctuation does not set its weight (see ``plate_lm._reweight``); at the
    observation on the first pass, where there is no prediction yet.

    Parameters
    ----------
    ds : Dataset
        The well's data.
    noise : Mapping[str, NoiseModelParams]
        Label to the noise parameters to weight by.
    prev : Mapping[str, Any] | None
        The well's fitted parameters from the previous pass (lmfit
        ``Parameters``), or ``None`` on the first.

    Returns
    -------
    Dataset
        The reweighted copy.
    """
    out = copy.deepcopy(ds)
    for lbl, da in out.items():
        params = noise.get(str(lbl))
        if params is None:
            continue
        y = da.yc
        if prev is not None and "K" in prev:
            y = binding_1site(
                da.xc,
                prev["K"].value,
                prev[f"S0_{lbl}"].value,
                prev[f"S1_{lbl}"].value,
                is_ph=ds.is_ph,
            )
        y_err = np.asarray(params.compute_y_err(np.nan_to_num(np.asarray(y))))
        kept = da.y_errc if da.y_errc.size == y_err.size else np.ones_like(y_err)
        da.y_errc = np.where(np.isfinite(y_err) & (y_err > 0), y_err, kept)
    return out


def calibrate_single_well(  # ruff: ignore[too-many-arguments]
    plates: Mapping[str, Mapping[str, Dataset]],
    floors: Mapping[str, Mapping[str, float]],
    *,
    method: Literal["lm", "huber"] = "lm",
    fit_floor: bool = False,
    gain_amp: Mapping[str, Mapping[str, float]] | None = None,
    max_iter: int = 20,
    tol: float = 1e-3,
    **fit_kwargs: Any,  # ruff: ignore[any-type]
) -> GainCalibration[dict[str, FitResult]]:
    """Fit every well on its own, calibrating floor and gain across all of them.

    Parameters
    ----------
    plates : Mapping[str, Mapping[str, Dataset]]
        Plate to well to dataset. One plate calibrates that plate; several
        share one gain per label.
    floors : Mapping[str, Mapping[str, float]]
        Plate to label to read-noise floor.
    method : Literal["lm", "huber"]
        The per-well loss, passed to
        :func:`~clophfit.fitting.core.fit_binding_glob`.
    fit_floor : bool
        Estimate each plate's floor as well as the gain.
    gain_amp : Mapping[str, Mapping[str, float]] | None
        See :func:`calibrate_gain`.
    max_iter : int
        Most fitting passes.
    tol : float
        Relative convergence tolerance on floor and gain.
    **fit_kwargs : Any
        Forwarded to ``fit_binding_glob`` (e.g. ``remove_outliers``).

    Returns
    -------
    GainCalibration[dict[str, FitResult]]
        Per plate, well to its fit under the calibrated weights.
    """
    previous: dict[str, dict[str, Any]] = {plate: {} for plate in plates}

    def fit(
        plate: str, noise: dict[str, NoiseModelParams]
    ) -> tuple[dict[str, FitResult], pd.DataFrame]:
        results: dict[str, FitResult] = {}
        for well, ds in plates[plate].items():
            prev = previous[plate].get(well)
            try:
                results[well] = fit_binding_glob(
                    _weighted(ds, noise, prev), method=method, **fit_kwargs
                )
            except InsufficientDataError:
                results[well] = FitResult()
            fitted = results[well].result
            if fitted is not None:
                previous[plate][well] = fitted.params
        return results, single_well_residuals(results)

    return calibrate_gain(
        fit, floors, fit_floor=fit_floor, gain_amp=gain_amp, max_iter=max_iter, tol=tol
    )


def calibrate_plate_lm(  # ruff: ignore[too-many-arguments]
    plates: Mapping[str, Mapping[str, Dataset]],
    groups: Mapping[str, Mapping[str, Sequence[str]]],
    floors: Mapping[str, Mapping[str, float]],
    *,
    loss: Literal["linear", "huber", "soft_l1", "cauchy"] = "linear",
    fit_floor: bool = False,
    gain_amp: Mapping[str, Mapping[str, float]] | None = None,
    max_iter: int = 20,
    tol: float = 1e-3,
) -> GainCalibration[PlateLMResult]:
    """Fit each plate jointly, calibrating floor and gain across all of them.

    Each pass is an ordinary :func:`~clophfit.fitting.plate_lm.fit_plate_lm`
    with the noise model fixed; the calibration happens between passes, so the
    fitter's own ``calibrate_noise`` (gain and alpha, uncorrected for degrees
    of freedom) is not involved. The profiled ``ye_mag`` still rescales each
    label, and with a well-calibrated gain it should come out near 1.

    Parameters
    ----------
    plates : Mapping[str, Mapping[str, Dataset]]
        Plate to well to dataset.
    groups : Mapping[str, Mapping[str, Sequence[str]]]
        Plate to its control groups (group name to member wells), which pool
        one K; an empty mapping fits every well its own K.
    floors : Mapping[str, Mapping[str, float]]
        Plate to label to read-noise floor.
    loss : Literal["linear", "huber", "soft_l1", "cauchy"]
        The plate fit's loss.
    fit_floor : bool
        Estimate each plate's floor as well as the gain.
    gain_amp : Mapping[str, Mapping[str, float]] | None
        See :func:`calibrate_gain`.
    max_iter : int
        Most fitting passes.
    tol : float
        Relative convergence tolerance on floor and gain.

    Returns
    -------
    GainCalibration[PlateLMResult]
        Per plate, the plate fit under the calibrated weights.
    """

    def fit(
        plate: str, noise: dict[str, NoiseModelParams]
    ) -> tuple[PlateLMResult, pd.DataFrame]:
        weighted = {
            well: _weighted(ds, noise, None) for well, ds in plates[plate].items()
        }
        result = fit_plate_lm(
            weighted, groups.get(plate, {}), loss=loss, noise_model=noise
        )
        return result, plate_lm_residuals(result)

    return calibrate_gain(
        fit, floors, fit_floor=fit_floor, gain_amp=gain_amp, max_iter=max_iter, tol=tol
    )
