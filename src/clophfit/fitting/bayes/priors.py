"""Priors on K, the Kd parameterisation, and the data-driven starting points."""

from __future__ import annotations

import copy
import typing

import numpy as np
import pymc as pm  # type: ignore[import-untyped]
import pytensor.tensor as pt
from lmfit import Parameters  # type: ignore[import-untyped]

from clophfit.fitting.bayes.axis import (
    _safe_sigma,
)
from clophfit.fitting.core import (
    fit_binding_glob,
)  # local to avoid circular import
from clophfit.fitting.data_structures import (
    DataArray,
    Dataset,
    FitResult,
    _Result,
)
from clophfit.fitting.models import KD_MIN

if typing.TYPE_CHECKING:
    from collections.abc import Mapping

    from clophfit.fitting.bayes_config import (
        DataKPrior,
        InitStrategy,
    )


def _active_xy_for_prior(da: DataArray) -> tuple[np.ndarray, np.ndarray]:
    """Return active x/y arrays for data-derived prior construction."""
    mask = np.asarray(da.mask, dtype=bool)
    x_source = da.xc if np.asarray(da.xc).size == mask.size else da.x
    y_source = da.yc if np.asarray(da.yc).size == mask.size else da.y
    x = np.asarray(x_source, dtype=float)
    y = np.asarray(y_source, dtype=float)
    if x.size == mask.size:
        x = x[mask]
    if y.size == mask.size:
        y = y[mask]
    finite = np.isfinite(x) & np.isfinite(y)
    return x[finite], y[finite]


def _edge_mean(values: np.ndarray, *, start: bool, n_points: int) -> float:
    """Return a finite mean from the first or last active edge window."""
    if values.size == 0:
        return 0.0
    n = max(1, min(int(n_points), values.size))
    window = values[:n] if start else values[-n:]
    finite_window = window[np.isfinite(window)]
    if finite_window.size:
        return float(finite_window.mean())
    finite_values = values[np.isfinite(values)]
    return float(finite_values.mean()) if finite_values.size else 0.0


def _edge_signal_priors(
    x: np.ndarray,
    y: np.ndarray,
    *,
    is_ph: bool,
    edge_points: int,
) -> tuple[float, float]:
    """Estimate S0/S1 from x-ordered edge windows."""
    if y.size == 0:
        return 0.0, 0.0
    order = np.argsort(x) if x.size == y.size else np.arange(y.size)
    y_sorted = y[order]
    low_x_signal = _edge_mean(y_sorted, start=True, n_points=edge_points)
    high_x_signal = _edge_mean(y_sorted, start=False, n_points=edge_points)
    if is_ph:
        return high_x_signal, low_x_signal
    return low_x_signal, high_x_signal


def _midpoint_x_for_prior(
    x: np.ndarray,
    y: np.ndarray,
    *,
    high_edge: float,
    low_edge: float,
    bounds: tuple[float, float],
) -> float | None:
    """Estimate half-transition x from the point closest to the edge midpoint."""
    if x.size == 0 or y.size == 0:
        return None
    target = 0.5 * (high_edge + low_edge)
    idx = int(np.nanargmin(np.abs(y - target)))
    guess = float(x[idx])
    lo, hi = sorted(map(float, bounds))
    if not np.isfinite(guess):
        return None
    return float(np.clip(guess, lo, hi))


def _resolve_data_prior_k_bounds(
    k_bounds: tuple[float, float] | None, *, is_ph: bool
) -> tuple[float, float]:
    """Return valid K bounds, falling back to an ``is_ph``-appropriate range."""
    default = (4.5, 9.0) if is_ph else (KD_MIN, 1e6)
    if k_bounds is None:
        return default
    lo, hi = sorted(map(float, k_bounds))
    if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
        return default
    return lo, hi


def _fit_result_from_data_priors(  # ruff: ignore[too-many-arguments]
    ds: Dataset,
    *,
    edge_points: int,
    signal_sigma_scale: float,
    k_prior: DataKPrior,
    k_bounds: tuple[float, float] | None,
    k_sigma: float,
) -> FitResult:
    """Create an lmfit-like seed result using only data-derived priors."""
    params = Parameters()
    lo, hi = _resolve_data_prior_k_bounds(k_bounds, is_ph=ds.is_ph)

    k_guesses: list[float] = []
    signal_scale = max(float(signal_sigma_scale), 1e-6)
    for lbl, da in ds.items():
        x, y = _active_xy_for_prior(da)
        if y.size == 0:
            s0 = 0.0
            s1 = 0.0
            y_range = 1.0
        else:
            s0, s1 = _edge_signal_priors(x, y, is_ph=ds.is_ph, edge_points=edge_points)
            y_range = float(np.nanmax(y) - np.nanmin(y))
            if not np.isfinite(y_range) or y_range <= 0.0:
                y_range = max(abs(s0), abs(s1), 1.0)
        sigma = max(signal_scale * y_range, 1e-6)
        params.add(f"S0_{lbl}", value=float(s0))
        params[f"S0_{lbl}"].stderr = sigma
        params.add(f"S1_{lbl}", value=float(s1))
        params[f"S1_{lbl}"].stderr = sigma
        midpoint = _midpoint_x_for_prior(
            x,
            y,
            high_edge=float(s0),
            low_edge=float(s1),
            bounds=(lo, hi),
        )
        if midpoint is not None:
            k_guesses.append(midpoint)

    if k_prior == "uniform":
        k_value = 0.5 * (lo + hi)
    elif k_guesses:
        k_value = float(np.nanmedian(k_guesses))
    else:
        k_value = 0.5 * (lo + hi)
    params.add("K", value=float(np.clip(k_value, lo, hi)), min=lo, max=hi)
    params["K"].stderr = max(float(k_sigma), 1e-6)
    return FitResult(result=_Result(params), dataset=copy.deepcopy(ds))


def _normalize_fit_input(  # ruff: ignore[too-many-arguments]
    ds_or_fr: Dataset | FitResult,
    *,
    init_strategy: InitStrategy = "lmfit",
    data_prior_edge_points: int = 2,
    data_prior_signal_sigma_scale: float = 0.5,
    data_prior_k_prior: DataKPrior = "midpoint_truncnorm",
    data_prior_k_bounds: tuple[float, float] | None = None,
    data_prior_k_sigma: float = 1.5,
) -> tuple[FitResult, bool]:
    """Normalize PyMC input to a copied preliminary fit result.

    Parameters
    ----------
    ds_or_fr : Dataset | FitResult
        Either a raw dataset or a pre-fit result used to seed the Bayesian
        model.
    init_strategy : InitStrategy
        ``"lmfit"`` fits ``ds_or_fr`` with LMFit first; ``"data_priors"``
        derives a seed directly from the data instead.
    data_prior_edge_points : int
        Edge-window size forwarded to :func:`_fit_result_from_data_priors`.
    data_prior_signal_sigma_scale : float
        Signal-prior sigma scale forwarded to
        :func:`_fit_result_from_data_priors`.
    data_prior_k_prior : DataKPrior
        K prior family forwarded to :func:`_fit_result_from_data_priors`.
    data_prior_k_bounds : tuple[float, float] | None
        K bounds forwarded to :func:`_fit_result_from_data_priors`.
    data_prior_k_sigma : float
        K prior sigma forwarded to :func:`_fit_result_from_data_priors`.

    Returns
    -------
    tuple[FitResult, bool]
        The normalized fit result and whether the caller explicitly supplied a
        pre-fit ``FitResult``.
    """
    if init_strategy == "data_priors":
        ds = ds_or_fr.dataset if isinstance(ds_or_fr, FitResult) else ds_or_fr
        if ds is None:
            return FitResult(), False
        return (
            _fit_result_from_data_priors(
                ds,
                edge_points=data_prior_edge_points,
                signal_sigma_scale=data_prior_signal_sigma_scale,
                k_prior=data_prior_k_prior,
                k_bounds=data_prior_k_bounds,
                k_sigma=data_prior_k_sigma,
            ),
            False,
        )
    if isinstance(ds_or_fr, Dataset):
        return fit_binding_glob(ds_or_fr), False
    return copy.deepcopy(ds_or_fr), True


def _normalize_fit_inputs(  # ruff: ignore[too-many-arguments]
    results: Mapping[str, Dataset | FitResult],
    *,
    init_strategy: InitStrategy = "lmfit",
    data_prior_edge_points: int = 2,
    data_prior_signal_sigma_scale: float = 0.5,
    data_prior_k_prior: DataKPrior = "midpoint_truncnorm",
    data_prior_k_bounds: tuple[float, float] | None = None,
    data_prior_k_sigma: float = 1.5,
) -> tuple[dict[str, FitResult], bool]:
    """Normalize multi-well PyMC inputs to copied preliminary fit results.

    Parameters
    ----------
    results : Mapping[str, Dataset | FitResult]
        Per-well raw datasets or pre-fit results.
    init_strategy : InitStrategy
        ``"lmfit"`` fits each raw dataset with LMFit; ``"data_priors"`` seeds
        each well directly from its data via
        :func:`_fit_result_from_data_priors`, skipping LMFit entirely (any
        supplied pre-fit result is ignored in favour of its dataset).
    data_prior_edge_points : int
        Edge-window size forwarded to :func:`_fit_result_from_data_priors`.
    data_prior_signal_sigma_scale : float
        Signal-prior sigma scale forwarded to
        :func:`_fit_result_from_data_priors`.
    data_prior_k_prior : DataKPrior
        K prior family forwarded to :func:`_fit_result_from_data_priors`.
    data_prior_k_bounds : tuple[float, float] | None
        K bounds forwarded to :func:`_fit_result_from_data_priors`.
    data_prior_k_sigma : float
        K prior sigma forwarded to :func:`_fit_result_from_data_priors`.

    Returns
    -------
    tuple[dict[str, FitResult], bool]
        Normalized per-well fit results and whether the centered-noise defaults
        should be preferred (every input already a ``FitResult`` and not using
        the data-prior strategy).
    """
    normalized: dict[str, FitResult] = {}
    all_prefit = True
    for key, value in results.items():
        if init_strategy == "data_priors":
            ds = value.dataset if isinstance(value, FitResult) else value
            if ds is None:
                continue
            normalized[key] = _fit_result_from_data_priors(
                ds,
                edge_points=data_prior_edge_points,
                signal_sigma_scale=data_prior_signal_sigma_scale,
                k_prior=data_prior_k_prior,
                k_bounds=data_prior_k_bounds,
                k_sigma=data_prior_k_sigma,
            )
            all_prefit = False
        elif isinstance(value, Dataset):
            normalized[key] = fit_binding_glob(value)
            all_prefit = False
        else:
            normalized[key] = copy.deepcopy(value)
    return normalized, all_prefit


# Smallest prior SD of log Kd: a preliminary Kd known to 1% is not a reason to
# pin the sampler to it.
_LOG_KD_SIGMA_MIN = 0.05
# Largest logit slope of the log-Kd prior. A logit-normal is single-peaked up
# to a slope of sqrt(2) and piles mass onto both bounds beyond it, which would
# favour "binds at the floor" and "does not bind" over everything between.
_LOGIT_SLOPE_MAX = float(np.sqrt(2.0))
# Trace-name prefix of the standard-normal variable behind a Deterministic Kd.
KD_RAW_PREFIX = "z_"


def log_kd_prior_params(
    k: float | None,
    stderr: float | None,
    n_sd: float,
    bounds: tuple[float, float],
) -> tuple[float, float]:
    """Centre and width, in log Kd, of a prior seeded by a preliminary fit.

    The pH model centres a Normal on the preliminary K with ``n_sd`` of its
    standard errors; this is the same prior moved to the log scale by the
    delta method (``sd(log K) = sd(K) / K``). A preliminary fit with no finite
    positive Kd or standard error -- a well that does not bind, whose curve is
    flat -- gets a prior spanning the whole range instead, centred on its
    geometric middle, so neither binding nor its absence is favoured.

    Parameters
    ----------
    k : float | None
        Preliminary Kd (mM).
    stderr : float | None
        Its standard error.
    n_sd : float
        Prior width in preliminary standard errors.
    bounds : tuple[float, float]
        ``(lo, hi)`` Kd range the prior is truncated to; see ``kd_bounds``.

    Returns
    -------
    tuple[float, float]
        ``(mu, sigma)`` of the log-Kd Normal before truncation.
    """
    lo, hi = bounds
    span = float(np.log(hi / lo))
    wide = float(np.log(lo)) + span / 2, span
    if k is None or stderr is None:
        return wide
    if not (np.isfinite(k) and np.isfinite(stderr) and k > 0 and stderr > 0):
        return wide
    mu = float(np.log(np.clip(k, lo, hi)))
    return mu, float(np.clip(n_sd * stderr / k, _LOG_KD_SIGMA_MIN, span))


def log_kd_prior(
    name: str,
    mu: float | np.ndarray,
    sigma: float | np.ndarray,
    bounds: tuple[float, float],
    dims: str | None = None,
) -> typing.Any:  # ruff: ignore[any-type]  # pt.TensorVariable (no stubs)
    """Build a Kd bounded to *bounds*, sampled through a standard Normal.

    ``log Kd = log lo + span * sigmoid(a + b z)`` with ``z ~ Normal(0, 1)``
    (``z_<name>`` in the trace); ``name`` itself is a Deterministic, so traces
    and results carry the Kd under the name they always had. The floor and
    ceiling hold by construction: an unbounded Normal returned Kd = -2.7 mM,
    and a well that does not bind samples the top of the range instead of
    wandering to infinity.

    Why not a truncated Normal in log Kd: NUTS starts each chain jittered by
    +/-1 in the *transformed* space, which for an interval as wide as
    1 mM - 16 M moves a 10 mM Kd elevenfold. On a test plate two of four
    chains never recovered (r-hat 4-8, every variable). Here ``a`` and ``b``
    are chosen so the slope of ``log Kd`` in ``z`` at the centre is *sigma*,
    so the jitter is a one-prior-SD move, as it is for a pKa.

    Parameters
    ----------
    name : str
        Name of the Kd variable, e.g. ``"K"`` or ``"K_free"``.
    mu : float | np.ndarray
        Prior centre(s) in log Kd (see :func:`log_kd_prior_params`).
    sigma : float | np.ndarray
        Prior SD(s) in log Kd. The logit slope is capped at ``sqrt(2)``: a
        prior as wide as the range becomes the flattest single-peaked shape
        rather than one piled onto both bounds.
    bounds : tuple[float, float]
        ``(lo, hi)`` Kd range.
    dims : str | None
        PyMC dims, for a vector of wells.

    Returns
    -------
    typing.Any
        The Kd tensor.
    """
    lo, hi = bounds
    log_lo = float(np.log(lo))
    span = float(np.log(hi)) - log_lo
    # Centre as a fraction of the range, kept off the bounds so logit is finite.
    p = np.clip((np.asarray(mu, dtype=float) - log_lo) / span, 1e-3, 1 - 1e-3)
    a = np.log(p / (1 - p))
    b = np.minimum(
        np.asarray(sigma, dtype=float) / (span * p * (1 - p)), _LOGIT_SLOPE_MAX
    )
    z = pm.Normal(f"{KD_RAW_PREFIX}{name}", mu=0.0, sigma=1.0, dims=dims)
    log_k = log_lo + span * pm.math.sigmoid(a + b * z)
    return pm.Deterministic(name, pt.exp(log_k), dims=dims)


def create_parameter_priors(  # ruff: ignore[too-many-arguments]
    params: Parameters,
    n_sd: float,
    key: str = "",
    ctr_name: str = "",
    default_sigma: float = 1e-3,
    *,
    kd_range: tuple[float, float] | None = None,
) -> dict[str, pm.Distribution]:
    """Create PyMC parameter prior distributions from lmfit Parameters.

    Parameters
    ----------
    params : Parameters
        lmfit Parameters to convert to PyMC priors.
    n_sd : float
        Scaling factor for parameter standard errors.
    key : str
        Optional suffix to add to parameter names.
    ctr_name : str
        If specified, skip creating K prior (shared from control group).
    default_sigma : float
        Default sigma when stderr is not available (default: 1e-3).
    kd_range : tuple[float, float] | None
        For a dissociation constant: the ``(lo, hi)`` range K is sampled in,
        as log Kd (see :func:`log_kd_prior`). ``None`` (pH) keeps the Normal.

    Returns
    -------
    dict[str, pm.Distribution]
        Dictionary of PyMC distribution objects.
    """
    priors: dict[str, pm.Distribution] = {}

    def param_name(p_name: str) -> str:
        return f"{p_name}_{key}" if key else p_name

    for name, p in params.items():
        # Use specified default sigma if stderr is not available
        sigma = _safe_sigma(p.stderr, n_sd, default=default_sigma)
        # Skip creating a separate K prior if it belongs to a control group
        if ctr_name and name == "K":
            continue
        prior_name = param_name(name)
        if prior_name in priors:
            continue
        if name == "K" and kd_range is not None:
            mu, log_sigma = log_kd_prior_params(p.value, p.stderr, n_sd, kd_range)
            priors[prior_name] = log_kd_prior(prior_name, mu, log_sigma, kd_range)
            continue
        priors[prior_name] = pm.Normal(prior_name, mu=p.value, sigma=sigma)
    return priors


def create_data_parameter_priors(
    params: Parameters,
    *,
    k_prior: DataKPrior = "midpoint_truncnorm",
    k_bounds: tuple[float, float] | None = None,
    is_ph: bool = False,
) -> dict[str, pm.Distribution]:
    """Create PyMC priors from data-derived parameter estimates."""
    priors: dict[str, pm.Distribution] = {}
    lo, hi = _resolve_data_prior_k_bounds(k_bounds, is_ph=is_ph)
    for name, p in params.items():
        if name == "K":
            if k_prior == "uniform":
                priors[name] = pm.Uniform(name, lower=lo, upper=hi)
            else:
                priors[name] = pm.TruncatedNormal(
                    name,
                    mu=float(np.clip(p.value, lo, hi)),
                    sigma=_safe_sigma(p.stderr, 1.0, default=1.5),
                    lower=lo,
                    upper=hi,
                )
            continue
        sigma = _safe_sigma(p.stderr, 1.0, default=1.0)
        priors[name] = pm.Normal(name, mu=p.value, sigma=sigma)
    return priors


def _ctr_param_name(group_name: str) -> str:
    """Return a control-group K variable name that cannot collide with well names."""
    return f"K_ctr_{group_name}"
