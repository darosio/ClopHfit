"""Observation likelihoods, outlier mixtures, and weighted residuals."""

from __future__ import annotations

import copy
import typing
from collections.abc import Mapping, Mapping as MappingABC, Sequence
from typing import Any

import numpy as np
import pymc as pm  # type: ignore[import-untyped]
import pytensor.tensor as pt
from pymc import math as pm_math
from pytensor.tensor import as_tensor_variable

from clophfit.fitting.bayes_config import (
    ContaminationFracPrior,
    NoiseConfig,
    RobustLikelihood,
    _validate_contamination_frac_prior,
)
from clophfit.fitting.data_structures import (
    DataArray,
    Dataset,
    NoiseModelParams,
    PlateNoiseModel,
)
from clophfit.fitting.models import binding_1site

if typing.TYPE_CHECKING:
    import pandas as pd
    from lmfit import Parameters  # type: ignore[import-untyped]


# NOTE: If model recompilation becomes a bottleneck, consider pm.MutableData.


def _hdi_bounds_or_none(row: pd.Series) -> tuple[float | None, float | None]:
    """Extract HDI bounds from a summary row, returning None if absent.

    ArviZ 1.x renamed these columns from ``hdi_3%``/``hdi_97%`` to
    ``hdi<prob>_lb``/``hdi<prob>_ub`` and made an ETI the default interval.
    Looking only for the old names found neither, so every PyMC fit exported
    its credible interval as -inf/+inf - a bound where an interval was meant,
    with nothing raised. Both spellings are accepted; an ETI is deliberately
    not, since it is a different interval and should not be relabelled as one.
    """
    lo_key = next(
        (
            k
            for k in row.index
            if str(k) == "hdi_3%"
            or (str(k).startswith("hdi") and str(k).endswith("_lb"))
        ),
        None,
    )
    hi_key = next(
        (
            k
            for k in row.index
            if str(k) == "hdi_97%"
            or (str(k).startswith("hdi") and str(k).endswith("_ub"))
        ),
        None,
    )
    if lo_key is None or hi_key is None:
        return None, None
    try:
        lo = float(row[lo_key])
        hi = float(row[hi_key])
    except Exception:  # ruff: ignore[blind-except]
        return None, None

    if not (np.isfinite(lo) and np.isfinite(hi)):
        return None, None
    if lo > hi:
        lo, hi = hi, lo
    if np.isclose(lo, hi, atol=1e-13, rtol=1e-13):
        return None, None
    return lo, hi


def _add_param_from_summary(rpars: Parameters, name: str, row: pd.Series) -> None:
    """Add parameter with bounds and uncertainty from trace summary."""
    mean = float(row["mean"])
    lo, hi = _hdi_bounds_or_none(row)
    if lo is None or hi is None:
        rpars.add(name, value=mean)
    else:
        rpars.add(name, value=mean, min=lo, max=hi)
    rpars[name].stderr = float(row["sd"])
    rpars[name].init_value = row.get("r_hat", np.nan)


def _add_y_likelihood(  # ruff: ignore[too-many-arguments]
    name: str,
    y_model: typing.Any,  # ruff: ignore[any-type]  # pt.TensorVariable (no stubs)
    da: DataArray,
    sigma: typing.Any,  # ruff: ignore[any-type]  # pt.TensorVariable | np.ndarray
    *,
    robust: bool = False,
    student_t_nu: typing.Any = 3.0,  # ruff: ignore[any-type]  # float | TensorVariable
    robust_likelihood: RobustLikelihood = "student_t",
    pi_outlier: typing.Any | None = None,  # ruff: ignore[any-type]  # TensorVariable
    outlier_inflate: typing.Any | None = None,  # ruff: ignore[any-type]  # TensorVariable
) -> None:
    """Add a Normal, StudentT, or contamination-mixture likelihood for one label."""
    mu = y_model[da.mask]
    if robust and robust_likelihood == "mixture":
        if pi_outlier is None or outlier_inflate is None:
            msg = "Mixture likelihood requires pi_outlier and outlier_inflate."
            raise ValueError(msg)
        _add_mixture_likelihood(
            name,
            mu,
            sigma,
            da.y,
            pi_outlier=pi_outlier,
            outlier_inflate=outlier_inflate,
        )
    elif robust:
        pm.StudentT(
            name,
            nu=student_t_nu,
            mu=mu,
            sigma=sigma,
            observed=da.y,
        )
    else:
        pm.Normal(name, mu=mu, sigma=sigma, observed=da.y)


def _add_mixture_likelihood(  # ruff: ignore[too-many-arguments]
    name: str,
    mu: typing.Any,  # ruff: ignore[any-type]  # pt.TensorVariable
    sigma_normal: typing.Any,  # ruff: ignore[any-type]  # pt.TensorVariable | np.ndarray
    observed: np.ndarray,
    *,
    pi_outlier: typing.Any,  # ruff: ignore[any-type]  # pt.TensorVariable
    outlier_inflate: typing.Any,  # ruff: ignore[any-type]  # pt.TensorVariable
) -> None:
    """Add a marginalized normal/outlier contamination mixture likelihood."""
    sigma_outlier = sigma_normal * (1.0 + outlier_inflate)
    w = pi_outlier * pt.ones_like(mu)  # type: ignore[no-untyped-call]
    weights = pt.stack([1.0 - w, w], axis=-1)
    comp_dists = [
        pm.Normal.dist(mu=mu, sigma=sigma_normal),
        pm.Normal.dist(mu=mu, sigma=sigma_outlier),
    ]
    pm.Mixture(name, w=weights, comp_dists=comp_dists, observed=observed)

    normal_logp = pm.logp(comp_dists[0], observed)
    outlier_logp = pm.logp(comp_dists[1], observed)
    log_p_normal = pt.log1p(-pi_outlier) + normal_logp
    log_p_outlier = pt.log(pi_outlier) + outlier_logp
    pm.Deterministic(
        name.replace("y_likelihood", "outlier_probability"),
        pm_math.exp(
            log_p_outlier - pt.logaddexp(log_p_normal, log_p_outlier)  # type: ignore[attr-defined,no-untyped-call]
        ),
    )


def _build_outlier_priors(
    labels: Sequence[str], contamination_frac_prior: ContaminationFracPrior
) -> tuple[dict[str, typing.Any], typing.Any]:
    """Build per-label outlier fractions and a shared scale inflation prior."""
    if isinstance(contamination_frac_prior, MappingABC):
        values = {
            str(label): _validate_contamination_frac_prior(value)
            for label, value in contamination_frac_prior.items()
        }
        fallback = float(np.nanmedian(list(values.values()))) if values else 0.15
        pi_outliers = {}
        for lbl in labels:
            contamination_frac = values.get(str(lbl), fallback)
            beta = (1.0 / contamination_frac) - 1.0
            pi_outliers[lbl] = pm.Beta(f"pi_outlier_{lbl}", alpha=1.0, beta=beta)
    else:
        contamination_frac = _validate_contamination_frac_prior(
            contamination_frac_prior
        )
        beta = (1.0 / contamination_frac) - 1.0
        pi_outliers = {
            lbl: pm.Beta(f"pi_outlier_{lbl}", alpha=1.0, beta=beta) for lbl in labels
        }
    outlier_inflate = pm.HalfNormal("outlier_inflate", sigma=5.0)
    return pi_outliers, outlier_inflate


def _student_t_nu_value(student_t_nu: float | None) -> typing.Any:  # ruff: ignore[any-type]
    """Return a fixed or inferred Student-t degrees-of-freedom value.

    In both branches a ``student_t_nu`` variable is registered in the model, so a
    fit's robustness and nu are recoverable from the trace alone (see
    :func:`clophfit.fitting.model_validation.robust_settings_from_trace`). A fixed
    value is recorded as a constant ``Deterministic`` while the plain float is
    still handed to the likelihood; ``None`` infers nu with support above 2.
    """
    if student_t_nu is not None:
        if student_t_nu <= 0:
            msg = "student_t_nu must be positive, or None to infer it."
            raise ValueError(msg)
        nu = float(student_t_nu)
        pm.Deterministic("student_t_nu", as_tensor_variable(nu))
        return nu
    nu_minus_two = pm.Exponential("student_t_nu_minus_two", lam=1 / 30)
    return pm.Deterministic("student_t_nu", nu_minus_two + 2.0)


def _compute_weighted_residuals(ds: Dataset, rpars: Parameters) -> np.ndarray:
    """Compute weighted residuals from posterior mean predictions.

    Weighted residuals = (1 / y_err) * (observed - predicted), using masked
    values (``.x``, ``.y``, ``.y_err``) for consistency.
    """
    residuals_list: list[np.ndarray] = []
    for lbl, da in ds.items():
        model = binding_1site(
            da.x,
            rpars["K"].value,
            rpars[f"S0_{lbl}"].value,
            rpars[f"S1_{lbl}"].value,
            is_ph=ds.is_ph,
        )
        raw = da.y - model
        residuals_list.append((1.0 / da.y_err * raw) if da.y_err.size > 0 else raw)
    return np.concatenate(residuals_list)


def dataset_with_unit_yerr(ds: Dataset) -> Dataset:
    """Return a deep-copied dataset with all observation errors set to one."""
    unit_ds = copy.deepcopy(ds)
    for da in unit_ds.values():
        da.y_err = np.ones_like(da.yc, dtype=float)
    return unit_ds


def _plate_noise_from_bg(
    bg_noise: Mapping[str, float] | float,
    labels: typing.Iterable[str],
    *,
    alpha: float,
) -> PlateNoiseModel:
    """Build a floor-plus-proportional noise model from bg-noise hints."""
    model = PlateNoiseModel()
    for label in labels:
        floor = (
            bg_noise.get(str(label), bg_noise.get(label, 1.0))
            if isinstance(bg_noise, MappingABC)
            else bg_noise
        )
        floor = float(floor)
        if not np.isfinite(floor) or floor <= 0.0:
            floor = 1.0
        model[str(label)] = NoiseModelParams(
            sigma_floor=floor,
            gain=0.0,
            alpha=alpha,
        )
    return model


def _noise_hint_value(
    hint: float | Mapping[str, float] | None, label: str
) -> float | None:
    """Return the per-label value of a scalar/mapping noise hint, or ``None``."""
    if hint is None:
        return None
    if isinstance(hint, MappingABC):
        value = hint.get(label, hint.get(str(label)))
        return None if value is None else float(value)
    return float(hint)


def _default_floor_from_data(da: DataArray) -> float:
    """Return a positive floor scale hint from a label's observed errors."""
    scale = float(np.nanmedian(np.asarray(da.y_err, dtype=float)))
    return scale if np.isfinite(scale) and scale > 0.0 else 1.0


def _resolve_structured_noise_model(
    noise: NoiseConfig, ds: Mapping[str, Any]
) -> PlateNoiseModel:
    """Return the explicit noise model or synthesize one from the dataset.

    Used by the ``"structured"`` noise path so the ``*_mode`` selectors work
    without a hand-built :class:`PlateNoiseModel`. Each label's ``sigma_floor``
    comes from the *floor* hint (falling back to the label's ``y_err`` scale),
    plus the *gain* and *alpha* hints.
    """
    if noise.noise_model is not None:
        return noise.noise_model
    model = PlateNoiseModel()
    for lbl, da in ds.items():
        floor = _noise_hint_value(noise.floor, str(lbl))
        if floor is None or not np.isfinite(floor) or floor <= 0.0:
            floor = _default_floor_from_data(da)
        gain = _noise_hint_value(noise.gain, str(lbl)) or 0.0
        alpha = _noise_hint_value(noise.alpha, str(lbl)) or 0.0
        model[str(lbl)] = NoiseModelParams(
            sigma_floor=float(floor), gain=float(gain), alpha=float(alpha)
        )
    return model
