"""Bayesian (PyMC) fitting utilities and pipelines."""

from __future__ import annotations

import contextlib
import copy
import os
import re
import typing
from collections.abc import Mapping, Mapping as MappingABC
from typing import Literal

import arviz as az  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
import pymc as pm  # type: ignore[import-untyped]
import pytensor.tensor as pt
import xarray as xr
from lmfit import Parameters  # type: ignore[import-untyped]
from matplotlib import figure
from pymc import math as pm_math
from pytensor.tensor import as_tensor_variable
from scipy.optimize import isotonic_regression

from clophfit.fitting.bayes_config import (
    ChainMethod,
    ContaminationFracPrior,
    DataKPrior,
    InitConfig,
    InitStrategy,
    NoiseConfig,
    NoiseParamMode,
    RobustConfig,
    RobustLikelihood,
    SamplerConfig,
    _validate_contamination_frac_prior,
)
from clophfit.fitting.models import binding_1site
from clophfit.fitting.plotting import PlotParameters, plot_fit

from .core import N_BOOT, fit_binding_glob  # local to avoid circular import
from .data_structures import (
    DataArray,
    Dataset,
    FitResult,
    MultiFitResult,
    NoiseModelParams,
    PlateNoiseModel,
    _Result,
)

if typing.TYPE_CHECKING:
    from collections.abc import Sequence

    from clophfit.clophfit_types import ArrayF
    from clophfit.prtecan import PlateScheme


_X_TRUE_INDEX_RE = re.compile(r"^x_true\[(\d+)\]$")
# Numeric floor for free/centered noise-prior scales, so a 0 hint gives the
# tightest around-zero prior (never a degenerate 0) and the prior width grows
# monotonically with the hint.
_MIN_NOISE_PRIOR_SCALE = 1e-3


def _build_floor_prior(name: str, mu: float, mode: NoiseParamMode) -> typing.Any:  # noqa: ANN401
    """Build one floor prior from a hint, shared by the pooled and per-label paths.

    Parameters
    ----------
    name : str
        Name of the PyMC variable to create.
    mu : float
        Calibrated ``sigma_floor`` hint.
    mode : NoiseParamMode
        How to treat the floor parameter: "fixed", "free", or "centered".

    Returns
    -------
    typing.Any
        A PyMC random variable, or a constant tensor when *mode* is "fixed".
    """
    if mode == "fixed":
        return pt.as_tensor_variable(mu)
    if mode == "free":
        # Uninformative prior, using mu only as a scale hint
        return pm.HalfNormal(name, sigma=max(mu, 1.0))
    # centered
    n_pts = 7
    dof = max(1, n_pts - 1)
    rel_sigma = float(np.clip(1.0 / np.sqrt(2 * dof), 0.05, 0.5))
    sigma = max(rel_sigma * mu, 0.01)
    return pm.TruncatedNormal(name, mu=mu, sigma=sigma, lower=0.0)


def _gain_prior_sigma(mu_g: float, plate_gain_scale: float) -> float:
    """Prior width for one gain hint.

    Gain carries the units of the signal, so its width is always relative --
    20% of this label's hint, or of the plate-wide scale when the hint is
    exactly 0. An exact 0 comes from the NNLS boundary and means the collinear
    alpha term won this label's decomposition, not that the Poisson term is
    absent, so the width is borrowed rather than collapsed.

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
    return 0.2 * (mu_g if mu_g > 0.0 else plate_gain_scale)


def build_pymc_noise_priors(  # noqa: C901, PLR0912, PLR0913, PLR0915
    noise_model: PlateNoiseModel,
    *,
    shared_alpha: bool = False,
    shared_gain: bool = False,
    shared_floor: bool = False,
    floor_mode: NoiseParamMode = "centered",
    gain_mode: NoiseParamMode = "centered",
    alpha_mode: NoiseParamMode = "centered",
) -> dict[str, typing.Any]:
    """Create PyMC priors from a ``PlateNoiseModel`` specification.

    Parameters
    ----------
    noise_model : PlateNoiseModel
        Per-label noise parameter specification.
    shared_alpha : bool
        If ``True``, pool the proportional error into a single global variable.
    shared_gain : bool
        If ``True``, pool the photon gain into a single global variable.
    shared_floor : bool
        If ``True``, pool the noise floor into a single global variable.
    floor_mode : NoiseParamMode
        How to treat the floor parameter: "fixed", "free", or "centered".
    gain_mode : NoiseParamMode
        How to treat the gain parameter: "fixed", "free", or "centered".
    alpha_mode : NoiseParamMode
        How to treat the alpha (rel_error) parameter: "fixed", "free", or "centered".

    Returns
    -------
    dict[str, typing.Any]
        A structured dictionary of PyMC random variables or constants representing
        the noise model components, consumed by :func:`get_pymc_variance`.
    """
    priors: dict[str, typing.Any] = {}
    labels = list(noise_model.keys())

    # 1. Floor Priors
    if shared_floor:
        floors = [p.sigma_floor for p in noise_model.values() if p.sigma_floor > 0]
        mu_f = float(np.mean(floors)) if floors else 0.0
        priors["floor"] = _build_floor_prior("floor", mu_f, floor_mode)
    else:
        priors["floor"] = {}
        for lbl in labels:
            priors["floor"][lbl] = _build_floor_prior(
                f"floor_{lbl}", noise_model[lbl].sigma_floor, floor_mode
            )

    # 2. Gain (Poisson term). The gate is deliberately narrower than alpha's
    # below. Alpha is dimensionless, so _MIN_NOISE_PRIOR_SCALE is a meaningful
    # universal around-zero width and alpha can always stay soft. Gain carries
    # the units of the signal, so its around-zero width has to be borrowed from
    # labels that did resolve a gain; when no label resolved one there is
    # nothing to borrow, and omitting the term beats inventing a scale. Do not
    # "symmetrise" this gate — it is what guarantees plate_gain_scale > 0.
    has_gain = any(p.gain > 0 for p in noise_model.values())
    if has_gain or gain_mode == "free":
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
                    # width borrowed from the labels that did resolve a gain,
                    # so the posterior can undo an arbitrary collinear split.
                    priors["gain"][lbl] = pm.HalfNormal(
                        f"gain_{lbl}",
                        sigma=_gain_prior_sigma(mu_g, plate_gain_scale),
                    )

    # 3. Alpha (proportional error). "centered" and "free" always build a prior
    # so a calibrated alpha of 0 becomes a tight prior *around* 0, not a hard 0;
    # only "fixed" leaves the term truly absent when alpha is 0. The alpha hint
    # is the prior mean in every mode (HalfNormal sigma is scaled by sqrt(pi/2)
    # to achieve that; TruncatedNormal takes it directly), floored at
    # _MIN_NOISE_PRIOR_SCALE so the width grows monotonically with the hint (the
    # weak 0.02 default lives in NoiseConfig.alpha, not here).
    has_alpha = any(p.alpha > 0 for p in noise_model.values())
    if has_alpha or alpha_mode in {"free", "centered"}:
        if shared_alpha:
            alphas = [p.alpha for p in noise_model.values() if p.alpha > 0]
            mu_a = float(np.mean(alphas)) if alphas else 0.0
            if alpha_mode == "fixed":
                priors["rel_error"] = pt.as_tensor_variable(mu_a)
            elif alpha_mode == "free":
                # sqrt(pi/2) converts the hint from a scale to a mean, so the
                # hint means the same thing here as it does for gain's
                # Exponential (whose mean is exactly its hint).
                priors["rel_error"] = pm.HalfNormal(
                    "rel_error",
                    sigma=max(mu_a, _MIN_NOISE_PRIOR_SCALE) * np.sqrt(np.pi / 2),
                )
            else:  # centered
                priors["rel_error"] = pm.TruncatedNormal(
                    "rel_error",
                    mu=mu_a,
                    sigma=max(0.25 * mu_a, _MIN_NOISE_PRIOR_SCALE),
                    lower=0.0,
                )
        else:
            priors["rel_error"] = {}
            for lbl in labels:
                mu_a = getattr(noise_model[lbl], "alpha", 0.0)
                if alpha_mode == "fixed":
                    priors["rel_error"][lbl] = pt.as_tensor_variable(mu_a)
                elif alpha_mode == "free":
                    # See the shared branch: sqrt(pi/2) makes the hint a mean.
                    priors["rel_error"][lbl] = pm.HalfNormal(
                        f"rel_error_{lbl}",
                        sigma=max(mu_a, _MIN_NOISE_PRIOR_SCALE) * np.sqrt(np.pi / 2),
                    )
                elif mu_a > 0.0:  # centered on a positive hint
                    priors["rel_error"][lbl] = pm.TruncatedNormal(
                        f"rel_error_{lbl}",
                        mu=mu_a,
                        sigma=max(0.25 * mu_a, _MIN_NOISE_PRIOR_SCALE),
                        lower=0.0,
                    )
                else:  # centered on 0 -> tight prior around 0
                    priors["rel_error"][lbl] = pm.HalfNormal(
                        f"rel_error_{lbl}", sigma=_MIN_NOISE_PRIOR_SCALE
                    )

    return priors


def _active_noise_model(
    noise_model: PlateNoiseModel, labels: Sequence[str]
) -> PlateNoiseModel:
    """Return noise parameters only for labels present in the fitted dataset."""
    active = PlateNoiseModel()
    missing: list[str] = []
    for lbl in labels:
        if lbl in noise_model:
            active[lbl] = noise_model[lbl]
        elif "default" in noise_model:
            active[lbl] = noise_model["default"]
        else:
            missing.append(lbl)
    if missing:
        available = ", ".join(map(str, noise_model.keys()))
        needed = ", ".join(missing)
        msg = f"Noise model is missing fitted label(s) {needed}. Available: {available}"
        raise KeyError(msg)
    return active


def get_pymc_variance(
    mu: pm.math.TensorVariable,
    label: str,
    noise_model: PlateNoiseModel,
    priors: dict[str, typing.Any],
) -> pm.math.TensorVariable:
    """Construct PyMC symbolic variance from a noise model specification.

    Parameters
    ----------
    mu : pm.math.TensorVariable
        Symbolic predicted signal (must be positive).
    label : str
        Label key into *noise_model*.
    noise_model : PlateNoiseModel
        Per-label noise parameter specification.
    priors : dict[str, typing.Any]
        Priors dict from :func:`build_pymc_noise_priors`.

    Returns
    -------
    pm.math.TensorVariable
        Symbolic variance tensor.
    """
    params = noise_model[label]
    y_pos = pm_math.maximum(1e-6, mu)
    floor = (
        priors["floor"][label] if isinstance(priors["floor"], dict) else priors["floor"]
    )
    # Broadcast floor² to match y_pos shape so the result is always
    # per-point (never 0-d), even when gain and alpha are both absent.
    var = floor**2 * pt.ones_like(y_pos)  # type: ignore[no-untyped-call]

    if params.gain > 0 or "gain" in priors:
        gain = (
            priors["gain"][label]
            if isinstance(priors["gain"], dict)
            else priors["gain"]
        )
        var += gain * y_pos

    if params.alpha > 0 or "rel_error" in priors:
        alpha = (
            priors["rel_error"][label]
            if isinstance(priors["rel_error"], dict)
            else priors["rel_error"]
        )
        var += (alpha * y_pos) ** 2

    return var


__all__ = [
    "fit_binding_pymc",
    "fit_binding_pymc_multi",
    "process_trace",
]


def _pymc_sample_parallel_args(
    nuts_sampler: str = "default",
    *,
    chains: int | None = None,
    cores: int | None = None,
    chain_method: ChainMethod = "auto",
) -> dict[str, object]:
    """Return sampling kwargs for ``pm.sample()``.

    Parameters
    ----------
    nuts_sampler : str
        NUTS backend (``"default"``, ``"nutpie"``, ``"blackjax"``,
        ``"numpyro"``). Non-default JAX backends are import-checked here.
    chains : int | None
        Number of MCMC chains. ``None`` leaves the PyMC default.
    cores : int | None
        Number of CPU cores for parallel chains. ``None`` leaves the PyMC
        default.
    chain_method : ChainMethod
        Chain strategy for JAX backends. ``"auto"`` selects ``"vectorized"``
        (all chains on one GPU); ignored by CPU backends.

    Returns
    -------
    dict[str, object]
        Keyword arguments to splat into ``pm.sample()``.

    Raises
    ------
    ImportError
        If *nuts_sampler* names a JAX/nutpie backend whose package is not
        installed.
    """
    kwargs: dict[str, object] = {}
    if nuts_sampler != "default":
        sampler_import_check = {
            "blackjax": ("blackjax", "pip install clophfit[gpu]"),
            "numpyro": ("numpyro", "pip install clophfit[gpu]"),
            "nutpie": ("nutpie", "pip install clophfit[gpu]"),
        }
        if nuts_sampler in sampler_import_check:
            pkg, install_hint = sampler_import_check[nuts_sampler]
            try:
                __import__(pkg)
            except ImportError as e:
                msg = (
                    f"NUTS sampler '{nuts_sampler}' requires package '{pkg}'. "
                    f"Install it with: {install_hint}  "
                    f"(or: uv sync --extra gpu)"
                )
                raise ImportError(msg) from e
        kwargs["nuts_sampler"] = nuts_sampler
        # JAX-based samplers (blackjax, numpyro) use jax.pmap by default which
        # requires N devices for N chains.  On a single GPU, use chain_method=
        # "vectorized" (jax.vmap) so all chains run on one device.
        if nuts_sampler in {"blackjax", "numpyro"}:
            method = "vectorized" if chain_method == "auto" else chain_method
            kwargs["chain_method"] = method
            # The blackjax inner progress bar uses JAX IO callbacks which are
            # not supported inside jax.vmap ("IO effect not supported in
            # vmap-of-cond"), so disable it under vectorized chains.
            if method == "vectorized":
                kwargs["progressbar"] = False
    if chains is not None:
        kwargs["chains"] = chains
    if cores is not None:
        kwargs["cores"] = cores
    if "PYTEST_CURRENT_TEST" in os.environ:
        kwargs.update({"cores": 1, "chains": 1})
    return kwargs


def _compute_sample_log_likelihood(trace: xr.DataTree) -> xr.DataTree:
    """Populate the log_likelihood group on sampled PyMC inference data."""
    return typing.cast(
        "xr.DataTree",
        pm.compute_log_likelihood(trace, extend_inferencedata=True, progressbar=False),
    )


def _sample_trace(
    n_samples: int,
    *,
    tune: int,
    target_accept: float,
    sampler: SamplerConfig,
) -> xr.DataTree:
    """Run ``pm.sample`` with :class:`SamplerConfig` controls.

    Must be called inside an active ``pm.Model`` context. Assembles the
    backend, chain/core, tree-depth, seed and compile options from *sampler*.

    Parameters
    ----------
    n_samples : int
        Number of posterior draws per chain.
    tune : int
        Number of tuning draws.
    target_accept : float
        Target acceptance probability.
    sampler : SamplerConfig
        Sampling controls (backend, chains, cores, chain method, seed,
        max tree depth, compile options).

    Returns
    -------
    xr.DataTree
        The sampled inference data (without the log_likelihood group).
    """
    sample_kwargs: dict[str, object] = {
        "tune": tune,
        "target_accept": target_accept,
        "return_inferencedata": True,
        **_pymc_sample_parallel_args(
            sampler.nuts_sampler,
            chains=sampler.chains,
            cores=sampler.cores,
            chain_method=sampler.chain_method,
        ),
    }
    if sampler.max_treedepth is not None:
        sample_kwargs["max_treedepth"] = sampler.max_treedepth
    if sampler.random_seed is not None:
        sample_kwargs["random_seed"] = sampler.random_seed
    if sampler.backend is not None:
        sample_kwargs["backend"] = sampler.backend
    if sampler.compile_kwargs is not None:
        sample_kwargs["compile_kwargs"] = dict(sampler.compile_kwargs)
    return typing.cast("xr.DataTree", pm.sample(n_samples, **sample_kwargs))


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
    out = float(np.nanmean(window))
    return out if np.isfinite(out) else float(np.nanmean(values))


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
    default = (4.5, 9.0) if is_ph else (1e-6, 1e6)
    if k_bounds is None:
        return default
    lo, hi = sorted(map(float, k_bounds))
    if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
        return default
    return lo, hi


def _fit_result_from_data_priors(  # noqa: PLR0913
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


def _normalize_fit_input(  # noqa: PLR0913
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


def _normalize_fit_inputs(  # noqa: PLR0913
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


def _resolve_noise_modes(
    *,
    prefer_centered: bool,
    floor_mode: NoiseParamMode | None,
    gain_mode: NoiseParamMode | None,
    alpha_mode: NoiseParamMode | None,
) -> tuple[NoiseParamMode, NoiseParamMode, NoiseParamMode]:
    """Resolve automatic noise-mode defaults from the input kind."""
    default_mode: NoiseParamMode = "centered" if prefer_centered else "free"
    return (
        floor_mode or default_mode,
        gain_mode or default_mode,
        alpha_mode or default_mode,
    )


def _build_ye_mag_priors(
    labels: Sequence[str],
    *,
    shared_ye_mags: bool = False,
    prior: Literal["halfnormal", "lognormal"] = "lognormal",
    mu: float | Mapping[str, float] = 0.0,
    sigma: float | Mapping[str, float] = 1.5,
) -> dict[str, typing.Any]:
    """Build label-indexed observation-error scale priors."""
    if shared_ye_mags:
        shared_sigma = _shared_ye_mag_sigma(sigma)
        if prior == "halfnormal":
            ye_mag = pm.HalfNormal("ye_mag", sigma=shared_sigma)
        else:
            ye_mag = pm.LogNormal(
                "ye_mag", mu=_shared_ye_mag_value(mu), sigma=shared_sigma
            )
        return dict.fromkeys(labels, ye_mag)

    if prior == "halfnormal":
        return {
            lbl: pm.HalfNormal(f"ye_mag_{lbl}", sigma=_ye_mag_sigma(sigma, lbl))
            for lbl in labels
        }
    return {
        lbl: pm.LogNormal(
            f"ye_mag_{lbl}",
            mu=_ye_mag_value(mu, lbl),
            sigma=_ye_mag_sigma(sigma, lbl),
        )
        for lbl in labels
    }


def _build_multi_ye_mag_priors(  # noqa: PLR0913
    labels: Sequence[str],
    *,
    per_well: bool = False,
    shared_ye_mags: bool = False,
    prior: Literal["halfnormal", "lognormal"] = "lognormal",
    mu: float | Mapping[str, float] = 0.0,
    sigma: float | Mapping[str, float] = 1.5,
) -> dict[str, typing.Any]:
    """Build label-indexed ye_mag priors for multi-well models."""
    if not per_well:
        return _build_ye_mag_priors(
            labels,
            shared_ye_mags=shared_ye_mags,
            prior=prior,
            mu=mu,
            sigma=sigma,
        )
    if shared_ye_mags:
        shared_sigma = _shared_ye_mag_sigma(sigma)
        if prior == "halfnormal":
            ye_mag = pm.HalfNormal("ye_mag", sigma=shared_sigma, dims="well")
        else:
            ye_mag = pm.LogNormal(
                "ye_mag",
                mu=_shared_ye_mag_value(mu),
                sigma=shared_sigma,
                dims="well",
            )
        return dict.fromkeys(labels, ye_mag)
    if prior == "halfnormal":
        return {
            lbl: pm.HalfNormal(
                f"ye_mag_{lbl}", sigma=_ye_mag_sigma(sigma, lbl), dims="well"
            )
            for lbl in labels
        }
    return {
        lbl: pm.LogNormal(
            f"ye_mag_{lbl}",
            mu=_ye_mag_value(mu, lbl),
            sigma=_ye_mag_sigma(sigma, lbl),
            dims="well",
        )
        for lbl in labels
    }


def _ye_mag_value(value: float | Mapping[str, float], label: str) -> float:
    """Return a finite ye_mag prior value for *label*."""
    if isinstance(value, MappingABC):
        values = np.asarray(list(value.values()), dtype=float)
        values = values[np.isfinite(values)]
        fallback = float(np.nanmean(values)) if values.size else 0.0
        raw = value.get(label, value.get(str(label), fallback))
    else:
        raw = value
    out = float(raw)
    return out if np.isfinite(out) else 0.0


def _shared_ye_mag_value(value: float | Mapping[str, float]) -> float:
    """Return a finite shared ye_mag prior value from a scalar or mapping."""
    if not isinstance(value, MappingABC):
        out = float(value)
        return out if np.isfinite(out) else 0.0
    values = np.asarray(list(value.values()), dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    return float(np.nanmean(values))


def _ye_mag_sigma(sigma: float | Mapping[str, float], label: str) -> float:
    """Return a finite positive ye_mag prior sigma for *label*."""
    if isinstance(sigma, MappingABC):
        values = np.asarray(list(sigma.values()), dtype=float)
        values = values[np.isfinite(values) & (values > 0.0)]
        fallback = float(np.nanmean(values)) if values.size else 1e-6
        raw = sigma.get(label, sigma.get(str(label), fallback))
    else:
        raw = sigma
    value = float(raw)
    if not np.isfinite(value) or value <= 0.0:
        return 1e-6
    return value


def _shared_ye_mag_sigma(sigma: float | Mapping[str, float]) -> float:
    """Return a finite positive shared sigma from a scalar or label mapping."""
    if not isinstance(sigma, MappingABC):
        value = float(sigma)
        return value if np.isfinite(value) and value > 0.0 else 1e-6
    values = np.asarray(list(sigma.values()), dtype=float)
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        return 1e-6
    return float(np.nanmean(values))


def _scale_ye_mag_sigma(
    sigma: float | Mapping[str, float], scale: float
) -> float | dict[str, float]:
    """Scale scalar or label-indexed ye_mag sigma hints."""
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    if isinstance(sigma, MappingABC):
        return {str(label): float(value) * scale for label, value in sigma.items()}
    return float(sigma) * scale


def _log_scaled_ye_mag_mu(
    median: float | Mapping[str, float], scale: float
) -> float | dict[str, float]:
    """Return log-median values for a scaled LogNormal ye_mag prior."""
    scaled = _scale_ye_mag_sigma(median, scale)
    if isinstance(scaled, dict):
        return {
            label: float(np.log(max(float(value), 1e-6)))
            for label, value in scaled.items()
        }
    return float(np.log(max(float(scaled), 1e-6)))


def _masked_obs_err_matrices(
    fit_results: Mapping[str, FitResult],
    wells_list: Sequence[str],
    lbl: str,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build row-major mask, observation, and error matrices for multi-well likelihood.

    Parameters
    ----------
    fit_results : Mapping[str, FitResult]
        Per-well fit results whose ``.dataset`` supplies the arrays.
    wells_list : Sequence[str]
        Well keys in the order they appear as columns.
    lbl : str
        Label key to extract from each well's dataset.
    n_steps : int
        Expected number of titration steps per well.

    Returns
    -------
    mask : np.ndarray
        Row-major boolean mask, shape ``(n_steps, n_wells)``.
    y_obs : np.ndarray
        Row-major observation matrix, shape ``(n_steps, n_wells)``.
    y_err : np.ndarray
        Row-major error matrix, shape ``(n_steps, n_wells)``.

    Raises
    ------
    ValueError
        If a well's dataset or its arrays have unexpected shapes.
    """
    n_wells = len(wells_list)
    mask = np.zeros((n_steps, n_wells), dtype=bool)
    y_obs = np.full((n_steps, n_wells), np.nan, dtype=float)
    y_err = np.full((n_steps, n_wells), np.nan, dtype=float)

    for w_idx, key in enumerate(wells_list):
        ds_well = fit_results[key].dataset
        if ds_well is None:
            msg = f"Dataset for well {key} is missing."
            raise ValueError(msg)

        da = ds_well[lbl]

        if da.mask.shape != (n_steps,):
            msg = (
                f"Mask shape mismatch for well {key}, label {lbl}: "
                f"{da.mask.shape} != {(n_steps,)}"
            )
            raise ValueError(msg)
        if da.yc.shape != (n_steps,):
            msg = (
                f"yc shape mismatch for well {key}, label {lbl}: "
                f"{da.yc.shape} != {(n_steps,)}"
            )
            raise ValueError(msg)

        mask[:, w_idx] = da.mask
        y_obs[:, w_idx] = np.asarray(da.yc, dtype=float)

        if da.y_errc.size > 0:
            if da.y_errc.shape != (n_steps,):
                msg = (
                    f"y_errc shape mismatch for well {key}, label {lbl}: "
                    f"{da.y_errc.shape} != {(n_steps,)}"
                )
                raise ValueError(msg)
            err = np.asarray(da.y_errc, dtype=float)
            err = np.where(np.isfinite(err) & (err > 0), err, 1.0)
        else:
            err = np.ones(n_steps, dtype=float)

        y_err[:, w_idx] = err

    return mask, y_obs, y_err


def _trace_summary_df(
    trace_or_df: xr.DataTree | MultiFitResult | pd.DataFrame,
) -> pd.DataFrame:
    """Return a numeric ArviZ summary DataFrame from a trace or summary."""
    if isinstance(trace_or_df, MultiFitResult):
        trace_or_df = trace_or_df.trace
    rdf = (
        az.summary(trace_or_df) if isinstance(trace_or_df, xr.DataTree) else trace_or_df
    )
    if not isinstance(rdf, pd.DataFrame):
        msg = "az.summary did not return a DataFrame"
        raise TypeError(msg)
    numeric_rdf = rdf.apply(pd.to_numeric, errors="coerce")
    return typing.cast("pd.DataFrame", numeric_rdf)


def _summary_mean_or_none(trace_df: pd.DataFrame, name: str) -> float | None:
    """Return a summary mean if present and finite."""
    try:
        value = float(
            np.asarray(trace_df.loc[name, "mean"], dtype=float).reshape(-1)[0]
        )
    except Exception:  # noqa: BLE001
        return None
    return value if np.isfinite(value) else None


# Number of pH-monitor wells the step-0 SD is measured over; the true starting
# pH is anchored by the standard error of their mean (strong but not fixed).
_N_PH_REPLICATE_WELLS = 3
# Minimum per-addition pipetting variance, as a fraction of the average
# per-addition variance, so a noisy flat 3-well SD cannot pin a step to ~zero:
# every 2 uL delivery carries a real, nonzero volume error.
_MIN_PIPETTING_STEP_FRAC = 0.25
# Default between-well scale for the per-well starting-x (x_start_well). Small and
# nonzero so multi-well fits give each well its own x0, tightly anchored on the
# shared plate x_start prior. Set to 0 for a single shared x_start.
_DEFAULT_X_START_BETWEEN_SIGMA = 0.05


def _pipetting_step_sigmas(x_errc_scaled: ArrayF) -> tuple[float, ArrayF]:
    """Split a cumulative pH-SD profile into per-addition pipetting sigmas.

    The 3-well SD at step ``N`` is cumulative:
    ``x_errc[N]**2 = read_noise**2 + sum_{k<=N} pipetting_var[k]``. The recurring
    read noise (the step-0 SD) cancels in consecutive differences, so per-addition
    variance is the increment of the cumulative variance.

    Accumulated pipetting variance is physically non-decreasing, but the 3-well
    SDs are noisy and routinely violate that (an atypically large early value, or
    dips). We therefore fit the least-squares monotone (isotonic) cumulative
    variance rather than a running maximum: ``np.maximum.accumulate`` would
    *freeze* a noisy early spike and propagate it forward (inflating the
    ``x_start`` anchor and, via the K/pH degeneracy, K), whereas isotonic pulls
    such a spike down to be consistent with the smaller later values. A positive
    floor keeps every addition nonzero, so a flat stretch cannot pin a step.

    Parameters
    ----------
    x_errc_scaled : ArrayF
        Per-step cumulative pH SD, already scaled by ``n_xerr`` and floored > 0.

    Returns
    -------
    tuple[float, ArrayF]
        ``(x_start_sigma, step_sigmas)``: ``x_start_sigma`` anchors the starting
        pH (standard error of the step-0 mean, so the recurring read noise does
        not enter the walk); ``step_sigmas[k]`` is addition ``k``'s pipetting
        sigma (length ``len(x_errc_scaled) - 1``).
    """
    cum_var = isotonic_regression(x_errc_scaled**2).x
    x_start_sigma = float(np.sqrt(cum_var[0]) / np.sqrt(_N_PH_REPLICATE_WELLS))
    if len(cum_var) <= 1:
        return x_start_sigma, np.empty(0, dtype=float)
    raw_step_var = np.diff(cum_var)
    total_pipetting_var = max(float(cum_var[-1] - cum_var[0]), 0.0)
    floor_var = _MIN_PIPETTING_STEP_FRAC * total_pipetting_var / (len(cum_var) - 1)
    step_var = np.maximum(raw_step_var, max(floor_var, 1e-12))
    return x_start_sigma, np.sqrt(step_var)


def _pipetting_walk_params(
    xc: ArrayF,
    x_errc: ArrayF,
    n_xerr: float,
    *,
    min_x_step: float,
    lower_nsd: float = 2.5,
) -> tuple[float, float, ArrayF, ArrayF, ArrayF]:
    """Derive the shared numpy priors for the pipetting random walk.

    Single source of truth for the de-noised per-addition step derivation used
    by :func:`create_x_true` (single latent ``x_true``) and by the ``per_well``
    x-error model in :func:`fit_binding_pymc_multi`. Each caller builds its own
    RVs from these arrays with the appropriate shape/dims.

    Parameters
    ----------
    xc : ArrayF
        Nominal x (pH) values, monotone in either direction.
    x_errc : ArrayF
        Per-step cumulative x SD (from the pH-monitor wells).
    n_xerr : float
        Scaling factor applied to ``x_errc``.
    min_x_step : float
        Minimum inferred change in x between consecutive steps.
    lower_nsd : float
        Number of step sigmas below the nominal step for the truncation lower
        bound, before flooring at ``min_x_step``.

    Returns
    -------
    tuple[float, float, ArrayF, ArrayF, ArrayF]
        ``(direction, x_start_sigma, step_nominal, step_sigmas, min_steps)``.
        ``direction`` is ``+1`` for increasing x, ``-1`` for decreasing;
        ``x_start_sigma`` anchors the (shared) starting x; ``step_nominal`` are
        the direction-signed nominal steps (positive); ``step_sigmas`` are the
        de-noised per-addition sigmas; ``min_steps`` is the order-preserving
        lower bound for each step.
    """
    x_errc_scaled = np.maximum(x_errc * n_xerr, 1e-6)
    x_start_sigma, step_sigmas = _pipetting_step_sigmas(x_errc_scaled)
    direction = 1.0 if np.all(np.diff(xc) >= 0.0) else -1.0
    step_nominal = direction * np.diff(xc)
    min_steps = np.maximum(step_nominal - lower_nsd * step_sigmas, min_x_step)
    return direction, x_start_sigma, step_nominal, step_sigmas, min_steps


def _build_multi_x_start(
    xc: ArrayF,
    x_start_sigma: float,
    x_start_between_sigma: float,
) -> typing.Any:  # noqa: ANN401
    """Build the starting-x anchor for the multi-well x-error models.

    The plate shares one buffer and one pre-addition reading, so the starting x
    (pH) is common: a single ``x_start`` RV anchored at the de-noised standard
    error of the step-0 mean (``x_start_sigma``). When
    ``x_start_between_sigma > 0`` an opt-in per-well term
    ``x_start_well ~ Normal(x_start, x_start_between_sigma)`` is added, letting
    each well's x0 wobble around the shared plate value — a common-mode global
    anchor plus independent per-well jitter. The between-well scale is fixed,
    not sampled, to avoid a centered funnel.

    Parameters
    ----------
    xc : ArrayF
        Nominal x (pH) values; ``xc[0]`` centres the shared anchor.
    x_start_sigma : float
        De-noised prior sigma for the shared ``x_start`` anchor (standard error
        of the step-0 mean).
    x_start_between_sigma : float
        Fixed between-well scale. ``<= 0`` returns the shared scalar anchor;
        ``> 0`` adds the per-well ``x_start_well`` term.

    Returns
    -------
    typing.Any
        A scalar ``x_start`` tensor (shared anchor) when
        ``x_start_between_sigma <= 0``, otherwise a ``dims="well"`` vector
        centred on the shared anchor. Both broadcast against the
        ``(step, well)`` offset/drop tensors.
    """
    x_start = pm.Normal("x_start", mu=xc[0], sigma=max(x_start_sigma, 1e-6))
    if x_start_between_sigma <= 0:
        return x_start
    return pm.Normal(
        "x_start_well",
        mu=x_start,
        sigma=x_start_between_sigma,
        dims="well",
    )


def create_x_true(
    xc: ArrayF,
    x_errc: ArrayF,
    n_xerr: float,
    lower_nsd: float = 2.5,
    min_x_step: float = 0.2,
) -> ArrayF | pm.Deterministic:
    """Create latent variables for x-values with uncertainty.

    Models the pH axis as a pipetting random walk: the true pH accumulates
    independent per-addition volume errors, so ``x_true = x_start +
    cumsum(x_step)`` with each step's sigma taken from the per-addition pipetting
    variance (see :func:`_pipetting_step_sigmas`). Returns a PyMC Deterministic
    variable when in a Model context with uncertainty, or a numpy array when
    there's no uncertainty or no active Model.
    """
    if n_xerr > 0 and np.any(x_errc > 0):
        direction, x_start_sigma, step_nominal, step_sigmas, min_steps = (
            _pipetting_walk_params(
                xc, x_errc, n_xerr, min_x_step=min_x_step, lower_nsd=lower_nsd
            )
        )
        x_start = pm.Normal("x_start", mu=xc[0], sigma=max(x_start_sigma, 1e-6))
        x_step = pm.TruncatedNormal(
            "x_step",
            mu=step_nominal,
            sigma=np.maximum(step_sigmas, 1e-6),
            lower=min_steps,
            shape=len(xc) - 1,
        )
        x_cumsum = pm.math.cumsum(x_step)
        x_offsets = pm.math.concatenate([as_tensor_variable(np.array([0.0])), x_cumsum])
        return pm.Deterministic("x_true", x_start + direction * x_offsets)
    # No uncertainty - check if we're in a Model context
    try:
        model = pm.Model.get_context(error_if_none=False)
        if model is not None:
            # In a model context, wrap as Deterministic
            return pm.Deterministic("x_true", as_tensor_variable(xc))
    except Exception:  # noqa: S110, BLE001
        pass
    # Outside model context or error, return numpy array
    return xc


def _safe_sigma(
    stderr: float | None,
    n_sd: float,
    *,
    default: float = 0.2,
    floor: float = 1e-3,
    cap: float | None = None,
) -> float:
    """Return a safe prior *sigma* from an lmfit ``stderr`` estimate.

    Handles ``None``, ``NaN``, ``inf``, and non-positive values by falling
    back to *default*.  The result is clamped to [*floor*, *cap*].
    """
    if stderr is None or not np.isfinite(stderr) or stderr <= 0:
        sigma = default
    else:
        sigma = max(stderr * n_sd, floor)
    if cap is not None:
        sigma = min(sigma, cap)
    return float(sigma)


def create_parameter_priors(
    params: Parameters,
    n_sd: float,
    key: str = "",
    ctr_name: str = "",
    default_sigma: float = 1e-3,
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


# NOTE: If model recompilation becomes a bottleneck, consider pm.MutableData.


def _hdi_bounds_or_none(row: pd.Series) -> tuple[float | None, float | None]:
    """Extract HDI bounds from summary row, returning None if invalid."""
    try:
        lo = float(row["hdi_3%"])
        hi = float(row["hdi_97%"])
    except Exception:  # noqa: BLE001
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


def _add_y_likelihood(  # noqa: PLR0913
    name: str,
    y_model: typing.Any,  # noqa: ANN401  # pt.TensorVariable (no stubs)
    da: DataArray,
    sigma: typing.Any,  # noqa: ANN401  # pt.TensorVariable | np.ndarray
    *,
    robust: bool = False,
    student_t_nu: typing.Any = 3.0,  # noqa: ANN401  # float | TensorVariable
    robust_likelihood: RobustLikelihood = "student_t",
    pi_outlier: typing.Any | None = None,  # noqa: ANN401  # TensorVariable
    outlier_inflate: typing.Any | None = None,  # noqa: ANN401  # TensorVariable
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


def _add_mixture_likelihood(  # noqa: PLR0913
    name: str,
    mu: typing.Any,  # noqa: ANN401  # pt.TensorVariable
    sigma_normal: typing.Any,  # noqa: ANN401  # pt.TensorVariable | np.ndarray
    observed: np.ndarray,
    *,
    pi_outlier: typing.Any,  # noqa: ANN401  # pt.TensorVariable
    outlier_inflate: typing.Any,  # noqa: ANN401  # pt.TensorVariable
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
            log_p_outlier - pt.logaddexp(log_p_normal, log_p_outlier)  # type: ignore[no-untyped-call]
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


def _student_t_nu_value(student_t_nu: float | None) -> typing.Any:  # noqa: ANN401
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


def _resolve_structured_noise_model(noise: NoiseConfig, ds: Dataset) -> PlateNoiseModel:
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


def process_trace(
    trace: xr.DataTree, p_names: typing.Iterable[str], ds: Dataset
) -> FitResult:
    """Process the trace to extract parameter estimates and update datasets.

    Parameters
    ----------
    trace : xr.DataTree
        The posterior samples from PyMC sampling.
    p_names: typing.Iterable[str]
        Parameter names.
    ds : Dataset
        The dataset containing titration data.

    Returns
    -------
    FitResult
        The updated fit result with extracted parameter values and datasets.
        Residuals are WEIGHTED (weight * (obs - pred)) where weight = 1/y_err,
        computed using posterior mean parameter estimates.

    """
    # Extract summary statistics for parameters
    rdf = _trace_summary_df(trace)
    p_names_set = set(p_names)
    rpars = Parameters()
    for name, row in rdf.iterrows():
        if str(name) in p_names_set:
            _add_param_from_summary(rpars, str(name), row)
    # x_true and x_errc
    nxc, nx_errc = _extract_x_true_from_trace_df(rdf)
    if nxc.size > 0:
        for da in ds.values():
            da.xc = nxc  # Update x_true values in the dataset
            da.x_errc = nx_errc  # Posterior already incorporates n_xerr prior scaling
    # Update y_errc from sigma_obs or scale by ye_mag — use xarray
    # directly to avoid az.summary rounding and fragile string parsing.
    # Fall back to az.summary for callers with legacy/empty traces.
    updated = _extract_sigma_obs_from_xarray(trace, ds)
    if not updated:
        updated = _scale_yerr_by_ye_mag_from_xarray(trace, ds)
    if not updated:
        _update_dataset_yerr_from_sigma_obs(rdf, ds)
    # Create figure
    fig = figure.Figure()
    ax = fig.add_subplot(111)
    plot_fit(ax, ds, rpars, nboot=N_BOOT, pp=PlotParameters(ds.is_ph))

    residuals = _compute_weighted_residuals(ds, rpars)
    return FitResult(fig, _Result(rpars, residual=residuals), trace=trace, dataset=ds)


def extract_fit(  # noqa: PLR0913, C901, PLR0912
    key: str,
    ctr: str,
    trace_df: xr.DataTree | MultiFitResult | pd.DataFrame,
    ds: Dataset,
    well_key: str = "",
    *,
    raw_trace: xr.DataTree | None = None,
    global_p_names: typing.Iterable[str] = (),
) -> FitResult:
    """Compute individual dataset fit from a multi-well trace summary.

    Parameters
    ----------
    key : str
        Well identifier used to filter per-well parameters in *trace_df*.
    ctr : str
        Control group name used to filter shared K parameters.
    trace_df : xr.DataTree | MultiFitResult | pd.DataFrame
        Either the raw multi-well PyMC trace or an ArviZ summary DataFrame.
    ds : Dataset
        Per-well dataset whose x values are updated in-place from the trace.
    well_key : str, optional
        When provided, per-well x posteriors (``x_true[step, well_key]``)
        are used instead of the global ``x_true``.  Pass the well identifier
        for xrw fits so each well's .dat/.png uses its own inferred pH axis.
    raw_trace : xr.DataTree | None, optional
        Raw multi-well PyMC trace for xarray-based x extraction, bypassing
        potential ``az.summary`` indexing bugs.  When ``None``, falls back
        to DataFrame-based extraction from *trace_df*.
    global_p_names : typing.Iterable[str], optional
        Names of model-wide (non-per-well) parameters, such as
        ``student_t_nu`` or per-label outlier terms, to copy verbatim from
        *trace_df* into the per-well result.

    Returns
    -------
    FitResult
        Fit result with figure, parameters, and dataset using posterior x.
    """
    trace_obj = trace_df.trace if isinstance(trace_df, MultiFitResult) else trace_df
    trace_df = _trace_summary_df(trace_df)
    rpars = Parameters()
    # Handle both old _well and new [well] format
    for name, row in trace_df.iterrows():
        n_str = str(name)
        if n_str.endswith((f"_{key}", f"[{key}]")):
            extracted_name = n_str.replace(f"_{key}", "").replace(f"[{key}]", "")
            # ctr_free_k=True: K for CTR wells is named K_{ctr}_{well}.
            # After stripping _{well}, we get K_{ctr} — normalize to "K".
            if extracted_name.startswith("K_"):
                extracted_name = "K"
            _add_param_from_summary(rpars, extracted_name, row)
    if ctr and "K" not in rpars:
        # Shared-K mode (ctr_free_k=False): CTR K is named with _ctr_ prefix.
        rdf = trace_df[trace_df.index == _ctr_param_name(ctr)]
        for name, row in rdf.iterrows():
            extracted_name = str(name).replace(_ctr_param_name(ctr), "K")
            _add_param_from_summary(rpars, extracted_name, row)
    for gp_name in global_p_names:
        rdf = trace_df[trace_df.index == gp_name]
        for _, row in rdf.iterrows():
            _add_param_from_summary(rpars, str(gp_name), row)
    # Use per-well x (xrw model) when available; fall back to global x_true.
    # Prefer xarray-based extraction (bypasses ArviZ multi-dim indexing bug);
    # fall back to df-based extraction for pre-computed summary DataFrames.
    nxc = np.array([])
    nx_errc = np.array([])
    if well_key and raw_trace is not None:
        nxc, nx_errc = _extract_x_per_well_from_xarray(raw_trace, well_key)
    if nxc.size == 0:
        nxc, nx_errc = _extract_x_per_well_from_trace_df(trace_df, well_key)
    if nxc.size == 0:
        nxc, nx_errc = _extract_x_true_from_trace_df(trace_df)
    if nxc.size > 0:
        for da in ds.values():
            da.xc = nxc
            da.x_errc = nx_errc
    # Update y_errc from sigma_obs or scale by ye_mag.
    # Prefer xarray extraction (no rounding, no fragile string parsing);
    # fall back to az.summary DataFrame for callers without a raw trace.
    if raw_trace is not None:
        updated = _extract_sigma_obs_from_xarray(raw_trace, ds, well_key=key)
        if not updated:
            _scale_yerr_by_ye_mag_from_xarray(raw_trace, ds, well_key=key)
    else:
        _update_dataset_yerr_from_sigma_obs(trace_df, ds, well_key=key)
    # Create figure
    fig = figure.Figure()
    ax = fig.add_subplot(111)
    plot_fit(ax, ds, rpars, nboot=N_BOOT, pp=PlotParameters(ds.is_ph))
    residuals = _compute_weighted_residuals(ds, rpars)
    trace = trace_obj if isinstance(trace_obj, xr.DataTree) else xr.DataTree()
    return FitResult(fig, _Result(rpars, residual=residuals), trace=trace, dataset=ds)


def _extract_x_true_from_trace_df(
    trace_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract x_true values and errors from an ArviZ summary DataFrame.

    Parameters
    ----------
    trace_df : pd.DataFrame
        ArviZ summary DataFrame containing trace results.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Arrays of x_true means and standard deviations.
    """
    rows: dict[int, tuple[float, float]] = {}
    fallback_rows: list[tuple[float, float]] = []
    for name, row in trace_df.iterrows():
        # Skip per-well ``x_true[step, well]`` rows (comma-indexed): this reads
        # only the global 1-D ``x_true[step]`` axis.
        if isinstance(name, str) and name.startswith("x_true") and "," not in name:
            x_mean = float(row["mean"])
            x_sd = float(row["sd"])
            match = _X_TRUE_INDEX_RE.match(name)
            if match:
                rows[int(match.group(1))] = (x_mean, x_sd)
            else:
                fallback_rows.append((x_mean, x_sd))
    ordered_rows = [rows[idx] for idx in sorted(rows)] if rows else fallback_rows
    if not ordered_rows:
        return np.array([]), np.array([])
    nxc = [row[0] for row in ordered_rows]
    nx_errc = [row[1] for row in ordered_rows]
    return np.array(nxc), np.array(nx_errc)


def _extract_x_per_well_from_xarray(
    trace: xr.DataTree, well_key: str
) -> tuple[np.ndarray, np.ndarray]:
    """Extract per-well x posterior from raw xarray trace, bypassing az.summary.

    Parameters
    ----------
    trace : xr.DataTree
        Raw multi-well PyMC trace.
    well_key : str
        Well identifier (e.g. ``"A01"``).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Arrays of per-well x posterior means and standard deviations.
        Empty if a per-well ``x_true`` (with a ``well`` dim) is absent.
    """
    if not hasattr(trace, "posterior") or "x_true" not in trace.posterior:
        return np.array([]), np.array([])
    da = trace.posterior["x_true"]
    if "well" not in da.dims:
        # Global 1-D x_true (deterministic x-error): not a per-well axis.
        return np.array([]), np.array([])
    sample_dims = [d for d in da.dims if d in {"chain", "draw"}]
    # mean/std across chains and draws for the specific well
    well_da = da.sel(well=well_key)
    if sample_dims:
        nxc = well_da.mean(dim=sample_dims).to_numpy()
        nx_errc = well_da.std(dim=sample_dims).to_numpy()
    else:
        nxc = well_da.to_numpy()
        nx_errc = np.zeros_like(nxc)
    return np.asarray(nxc), np.asarray(nx_errc)


def _extract_x_per_well_from_trace_df(
    trace_df: pd.DataFrame,
    well_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract per-well x values for *well_key* from an xrw trace summary.

    ArviZ names the per-well ``x_true`` deterministic (with dims ``step`` x
    ``well``) as ``x_true[{step}, {well}]``.  This function collects those rows
    for a specific well and returns them sorted by step index.

    When found, per-well x values take precedence over the global 1-D
    ``x_true``, allowing each well in an xrw fit to use its own inferred pH axis.

    Parameters
    ----------
    trace_df : pd.DataFrame
        ArviZ summary DataFrame from ``fit_binding_pymc_multi`` (with xrw).
    well_key : str
        Well identifier used as the ``well`` coord (e.g. ``"A01"``).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Arrays of per-well x posterior means and standard deviations ordered
        by step.  Both arrays are empty if per-well ``x_true`` rows are absent.

    Notes
    -----
    Returned arrays are ordered by step index.  We assume the original ``xc``
    order matches the step order, which holds for the current xrw
    implementation.  A more robust future approach would explicitly match
    steps to ``xc`` indices.
    """
    suffix = f", {well_key}]"
    rows: dict[int, tuple[float, float]] = {}
    for name, row in trace_df.iterrows():
        if (
            isinstance(name, str)
            and name.startswith("x_true[")
            and name.endswith(suffix)
        ):
            step_str = name[len("x_true[") : -len(suffix)]
            try:
                step = int(step_str)
                rows[step] = (float(row["mean"]), float(row["sd"]))
            except ValueError:
                pass
    if not rows:
        return np.array([]), np.array([])
    nxc = [rows[s][0] for s in sorted(rows)]
    nx_errc = [rows[s][1] for s in sorted(rows)]
    return np.array(nxc), np.array(nx_errc)


def x_true_from_trace_df(
    trace_df: xr.DataTree | MultiFitResult | pd.DataFrame,
) -> DataArray:
    """Extract x_true from a PyMC trace or ArviZ summary DataFrame."""
    nxc, nx_errc = _extract_x_true_from_trace_df(_trace_summary_df(trace_df))
    return DataArray(xc=nxc, yc=np.ones_like(nxc), x_errc=nx_errc)


def _extract_sigma_obs_from_xarray(
    trace: xr.DataTree, ds: Dataset, *, well_key: str | None = None
) -> bool:
    """Extract ``sigma_obs`` posterior means from xarray, update *ds* in-place.

    Bypasses :func:`az.summary` to avoid rounding errors and fragile
    string-parsing of coordinate labels.  Modeled on
    :func:`_extract_x_per_well_from_xarray`.

    Parameters
    ----------
    trace : xr.DataTree
        Raw PyMC trace with posterior samples.
    ds : Dataset
        Dataset whose ``y_errc`` arrays will be updated in-place.
    well_key : str | None
        Well coordinate value to select.  ``None`` for single-well traces
        that lack a ``well`` dimension.

    Returns
    -------
    bool
        ``True`` if any label's ``y_errc`` was updated.
    """
    if not hasattr(trace, "posterior"):
        return False
    posterior = trace.posterior
    any_updated = False
    for lbl, da in ds.items():
        var_name = f"sigma_obs_{lbl}"
        if var_name not in posterior:
            continue
        var_da = posterior[var_name]
        sample_dims = [d for d in var_da.dims if d in {"chain", "draw"}]
        if "well" in var_da.dims:
            if well_key is None:
                continue
            var_da = var_da.sel(well=well_key)
        if sample_dims:
            sigma_means = np.asarray(var_da.mean(dim=sample_dims).to_numpy())
        else:
            sigma_means = np.asarray(var_da.to_numpy())
        sigma_means = np.maximum(sigma_means.ravel(), 1e-6)
        n_steps = len(da.yc)
        if len(sigma_means) != n_steps:
            continue
        da.y_errc = sigma_means[:n_steps]
        any_updated = True
    return any_updated


def _scale_yerr_by_ye_mag_from_xarray(
    trace: xr.DataTree, ds: Dataset, *, well_key: str | None = None
) -> bool:
    """Scale *ds* ``y_errc`` by posterior ``ye_mag`` from xarray.

    Parameters
    ----------
    trace : xr.DataTree
        Raw PyMC trace with posterior samples.
    ds : Dataset
        Dataset whose ``y_errc`` arrays will be scaled in-place.
    well_key : str | None
        Well coordinate to select when ``ye_mag`` variables have a ``well``
        dimension.

    Returns
    -------
    bool
        ``True`` if any label's ``y_errc`` was scaled.
    """
    if not hasattr(trace, "posterior"):
        return False
    posterior = trace.posterior
    sample_dims = ["chain", "draw"]
    any_scaled = False
    for lbl, da in ds.items():
        per_label = f"ye_mag_{lbl}"
        if per_label in posterior:
            mag_da = posterior[per_label]
        elif "ye_mag" in posterior:
            mag_da = posterior["ye_mag"]
        else:
            continue
        if "well" in mag_da.dims:
            if well_key is None:
                continue
            mag_da = mag_da.sel(well=well_key)
        mag = float(mag_da.mean(dim=sample_dims).values)
        if da.y_errc.size == 0:
            da.y_errc = np.ones_like(da.yc)
        da.y_errc *= mag
        any_scaled = True
    return any_scaled


def _update_dataset_yerr_from_sigma_obs(  # noqa: C901, PLR0912
    trace_df: pd.DataFrame, ds: Dataset, *, well_key: str | None = None
) -> None:
    """Replace per-data-array ``y_errc`` with posterior ``sigma_obs`` from the trace.

    When a heteroscedastic noise model is used, the posterior ``sigma_obs``
    values are narrower and more meaningful than the raw data ``y_err`` for
    computing weighted residuals.  If ``sigma_obs`` is not present, scales the
    original ``y_errc`` by the posterior mean of ``ye_mag_{lbl}`` or ``ye_mag``.

    Only rows matching *well_key* are applied to the per-well dataset.
    """
    for lbl, da in ds.items():
        sigma_rows: dict[int, float] = {}
        prefix_with_bracket = f"sigma_obs_{lbl}["
        for name, row in trace_df.iterrows():
            row_name = str(name)
            if not row_name.startswith(prefix_with_bracket):
                continue
            # Extract indices from sigma_obs_lbl[idx1, idx2, ...]
            inner = row_name[len(prefix_with_bracket) : -1]
            parts = [p.strip() for p in inner.split(",")]
            if len(parts) == 2 and well_key is not None:  # noqa: PLR2004
                # Multi-well case: expected parts [step, well]
                if parts[1] == well_key:
                    with contextlib.suppress(ValueError, KeyError):
                        sigma_rows[int(parts[0])] = float(row["mean"])
            elif len(parts) == 1:
                # Single-well case: expected parts [step]
                with contextlib.suppress(ValueError, KeyError):
                    sigma_rows[int(parts[0])] = float(row["mean"])

        if sigma_rows:
            n_steps = len(da.yc)
            y_errc = np.ones(n_steps, dtype=float)
            for step_idx, sigma_val in sigma_rows.items():
                if 0 <= step_idx < n_steps:
                    y_errc[step_idx] = max(sigma_val, 1e-6)
            da.y_errc = y_errc
        else:
            # Fallback to ye_mag scaling
            mag = _summary_mean_or_none(trace_df, f"ye_mag_{lbl}")
            if mag is None:
                mag = _summary_mean_or_none(trace_df, "ye_mag")
            if mag is not None:
                if da.y_errc.size == 0:
                    da.y_errc = np.ones_like(da.yc)
                da.y_errc *= mag


def _per_well_fit_results_from_trace(
    trace: xr.DataTree,
    fit_results: Mapping[str, FitResult],
    scheme: PlateScheme,
    *,
    x_error_model: Literal["deterministic", "per_well"],
    global_p_names: typing.Iterable[str] = (),
) -> dict[str, FitResult]:
    """Reconstruct per-well fit results from a shared multi-well trace."""
    trace_df = _trace_summary_df(trace)
    per_well_results: dict[str, FitResult] = {}
    for key, fr in fit_results.items():
        if fr.dataset is None:
            continue
        ctr = next((name for name, wells in scheme.names.items() if key in wells), "")
        well_key = key if x_error_model == "per_well" else ""
        ds = copy.deepcopy(fr.dataset)
        per_well_results[key] = extract_fit(
            key,
            ctr,
            trace_df,
            ds,
            well_key=well_key,
            raw_trace=trace,
            global_p_names=global_p_names,
        )
        per_well_results[key].trace = trace
    return per_well_results


_DEFAULT_NOISE = NoiseConfig()
_DEFAULT_ROBUST = RobustConfig()
_DEFAULT_INIT = InitConfig()
_DEFAULT_SAMPLER = SamplerConfig()


def fit_binding_pymc(  # noqa: PLR0913, PLR0915
    ds_or_fr: Dataset | FitResult,
    *,
    n_sd: float = 10.0,
    n_xerr: float = 1.0,
    min_x_step: float = 0.2,
    noise: NoiseConfig = _DEFAULT_NOISE,
    robust: RobustConfig = _DEFAULT_ROBUST,
    init: InitConfig = _DEFAULT_INIT,
    sampler: SamplerConfig = _DEFAULT_SAMPLER,
) -> FitResult:
    """Analyze multi-label titration datasets using PyMC (single model).

    Parameters
    ----------
    ds_or_fr : Dataset | FitResult
        Either a ``Dataset`` (an initial LS fit is run) or a ``FitResult`` whose
        params seed the PyMC priors.
    n_sd : float
        Number of standard deviations for parameter priors.
    n_xerr : float
        Scaling factor for x-error.
    min_x_step : float
        Minimum inferred change in ``x`` between consecutive titration steps
        when latent-x modeling is enabled.
    noise : NoiseConfig
        Observation-noise configuration. Default scales ``y_err`` by a learned
        ``ye_mag`` multiplier; use :meth:`NoiseConfig.structured` for a
        floor/gain/alpha noise model. See :class:`NoiseConfig`.
    robust : RobustConfig
        Robust-likelihood configuration (Student-t or contamination mixture).
        See :class:`RobustConfig`.
    init : InitConfig
        Prior-initialization strategy (``"lmfit"`` prefit or ``"data_priors"``).
        See :class:`InitConfig`.
    sampler : SamplerConfig
        NUTS sampling controls. See :class:`SamplerConfig`.

    Returns
    -------
    FitResult
        Bayesian fitting results.
    """
    robust_on = robust.enabled
    robust_likelihood = robust.likelihood
    student_t_nu = robust.nu
    contamination_frac_prior = robust.contamination_frac_prior
    shared_alpha = noise.shared_alpha
    shared_gain = noise.shared_gain
    shared_floor = noise.shared_floor
    learn_ye_mags = noise.learn_ye_mags
    shared_ye_mags = noise.shared_ye_mags
    ye_mag_prior = noise.ye_mag_prior
    ye_mag_mu = noise.ye_mag_mu
    ye_mag_sigma = noise.ye_mag_sigma
    n_samples = sampler.n_samples

    fr, prefer_centered = _normalize_fit_input(
        ds_or_fr,
        init_strategy=init.strategy,
        data_prior_edge_points=init.edge_points,
        data_prior_signal_sigma_scale=init.signal_sigma_scale,
        data_prior_k_prior=init.k_prior,
        data_prior_k_bounds=init.k_bounds,
        data_prior_k_sigma=init.k_sigma,
    )

    if fr.result is None or fr.dataset is None:
        return FitResult()
    params = fr.result.params
    ds = copy.deepcopy(fr.dataset)
    labels = list(ds.keys())
    xc = next(iter(ds.values())).xc
    x_errc = next(iter(ds.values())).x_errc
    floor_mode, gain_mode, alpha_mode = _resolve_noise_modes(
        prefer_centered=prefer_centered,
        floor_mode=noise.floor_mode,
        gain_mode=noise.gain_mode,
        alpha_mode=noise.alpha_mode,
    )
    with pm.Model():
        pars = (
            create_data_parameter_priors(
                params,
                k_prior=init.k_prior,
                k_bounds=init.k_bounds,
                is_ph=ds.is_ph,
            )
            if init.strategy == "data_priors"
            else create_parameter_priors(params, n_sd)
        )
        x_true = create_x_true(xc, x_errc, n_xerr, min_x_step=min_x_step)
        robust_nu = (
            _student_t_nu_value(student_t_nu)
            if robust_on and robust_likelihood == "student_t"
            else 3.0
        )
        if robust_on and robust_likelihood == "mixture":
            pi_outliers, outlier_inflate = _build_outlier_priors(
                labels, contamination_frac_prior
            )
        else:
            pi_outliers, outlier_inflate = {}, None

        if noise.kind == "ye_mag":
            ye_mags = _build_ye_mag_priors(
                labels,
                shared_ye_mags=shared_ye_mags,
                prior=ye_mag_prior,
                mu=ye_mag_mu,
                sigma=ye_mag_sigma,
            )
            for lbl, da in ds.items():
                y_model = binding_1site(
                    x_true,
                    pars["K"],
                    pars[f"S0_{lbl}"],
                    pars[f"S1_{lbl}"],
                    is_ph=ds.is_ph,
                )
                sigma = ye_mags[lbl] * da.y_err
                _add_y_likelihood(
                    f"y_likelihood_{lbl}",
                    y_model,
                    da,
                    sigma,
                    robust=robust_on,
                    student_t_nu=robust_nu,
                    robust_likelihood=robust_likelihood,
                    pi_outlier=pi_outliers.get(lbl),
                    outlier_inflate=outlier_inflate,
                )
        else:
            noise_model = _resolve_structured_noise_model(noise, ds)
            active_noise_model = _active_noise_model(noise_model, labels)
            noise_priors = build_pymc_noise_priors(
                active_noise_model,
                shared_alpha=shared_alpha,
                shared_gain=shared_gain,
                shared_floor=shared_floor,
                floor_mode=floor_mode,
                gain_mode=gain_mode,
                alpha_mode=alpha_mode,
            )
            if learn_ye_mags:
                ye_mags = _build_ye_mag_priors(
                    labels,
                    shared_ye_mags=shared_ye_mags,
                    prior="halfnormal",
                    sigma=5.0,
                )

            for lbl, da in ds.items():
                y_model = binding_1site(
                    x_true,
                    pars["K"],
                    pars[f"S0_{lbl}"],
                    pars[f"S1_{lbl}"],
                    is_ph=ds.is_ph,
                )
                noise_var = get_pymc_variance(
                    y_model, lbl, active_noise_model, noise_priors
                )
                sigma = pm_math.sqrt(noise_var)
                if learn_ye_mags:
                    sigma = ye_mags[lbl] * sigma

                sigma_det = pm.Deterministic(f"sigma_obs_{lbl}", sigma)
                _add_y_likelihood(
                    f"y_likelihood_{lbl}",
                    y_model,
                    da,
                    sigma_det[da.mask],
                    robust=robust_on,
                    student_t_nu=robust_nu,
                    robust_likelihood=robust_likelihood,
                    pi_outlier=pi_outliers.get(lbl),
                    outlier_inflate=outlier_inflate,
                )

        # Inference
        tune = sampler.n_tune if sampler.n_tune is not None else n_samples // 2
        target_accept_ = (
            sampler.target_accept
            if sampler.target_accept is not None
            else 0.95
            if n_xerr > 0
            else 0.9
        )
        trace = _sample_trace(
            n_samples, tune=tune, target_accept=target_accept_, sampler=sampler
        )
        if sampler.compute_log_likelihood:
            trace = _compute_sample_log_likelihood(trace)
    p_names = list(params.keys())
    if robust_on and robust_likelihood == "student_t" and student_t_nu is None:
        p_names.append("student_t_nu")
    if robust_on and robust_likelihood == "mixture":
        p_names.extend([f"pi_outlier_{lbl}" for lbl in labels])
        p_names.append("outlier_inflate")
    return process_trace(trace, p_names, ds)


# ------------------------------------------------------------------
# Helper: weighted statistics
# ------------------------------------------------------------------


def weighted_stats(
    values: Mapping[str, Sequence[float | None]],
    stderr: Mapping[str, Sequence[float | None]],
    *,
    min_stderr: float = 1e-3,
) -> dict[str, tuple[float, float]]:
    """Weighted mean and stderr for control priors.

    Filters out ``NaN``, ``inf``, and non-positive stderr, and floors
    stderr at *min_stderr* to avoid infinite weights.
    """
    results: dict[str, tuple[float, float]] = {}
    for sample in values:
        pairs = []
        for v, s in zip(values[sample], stderr[sample], strict=True):
            if v is None or s is None:
                continue
            if not (np.isfinite(v) and np.isfinite(s)):
                continue
            s_val = min_stderr if s <= 0 else float(s)
            pairs.append((float(v), s_val))

        if not pairs:
            msg = f"No valid finite (value, stderr) pairs for {sample!r}"
            raise ValueError(msg)

        x, se = map(np.asarray, zip(*pairs, strict=True))
        w = 1.0 / se**2
        results[sample] = (
            float(np.average(x, weights=w)),
            float(np.sqrt(1.0 / w.sum())),
        )
    return results


def _ctr_param_name(group_name: str) -> str:
    """Return a control-group K variable name that cannot collide with well names."""
    return f"K_ctr_{group_name}"


def _build_ctr_k_params(  # noqa: PLR0913
    scheme: PlateScheme,
    ctr_ks: dict[str, tuple[float, float]],
    active_wells: set[str],
    *,
    ctr_free_k: bool,
    well_k_init: dict[str, tuple[float, float]] | None = None,
    fallback_sigma: float = 0.6,
) -> tuple[dict[str, typing.Any], dict[str, typing.Any]]:
    """Build K parameters for control groups.

    Parameters
    ----------
    scheme : PlateScheme
        Plate scheme with named control groups.
    ctr_ks : dict[str, tuple[float, float]]
        Weighted K mean and stderr per control name.
    active_wells : set[str]
        Well keys that have valid results and datasets.
    ctr_free_k : bool
        If True, each CTR replicate gets its own independent K prior
        initialized from its own preliminary fit result (same as unknown
        wells).  If False (default), all replicates of the same CTR share
        a single K.
    well_k_init : dict[str, tuple[float, float]] | None
        Maps well key → ``(K_value, K_sigma)`` from the individual
        preliminary fit or pH@mid-fluorescence.  Used only when
        ``ctr_free_k=True``; wells absent from this dict fall back to the
        group weighted mean with *fallback_sigma*.
    fallback_sigma : float
        Prior sigma used for wells absent from *well_k_init* (free CTR
        mode) or as a floor for the constrained group sigma.  Default 0.6.

    Returns
    -------
    k_params : dict[str, typing.Any]
        Maps CTR name → shared PyMC K variable (populated only when
        ``ctr_free_k=False``).
    k_replicate : dict[str, typing.Any]
        Maps well key → individual PyMC K variable (populated only when
        ``ctr_free_k=True``).
    """
    k_params: dict[str, typing.Any] = {}
    k_replicate: dict[str, typing.Any] = {}
    if ctr_free_k:
        # Per-well K: initialized from each well's own preliminary fit result
        # (same as unknown wells), falling back to group weighted mean if unavailable.
        for name, scheme_wells in scheme.names.items():
            for well in scheme_wells:
                if well in active_wells:
                    if well_k_init and well in well_k_init:
                        mu, sigma = well_k_init[well]
                    else:
                        mu, sigma = ctr_ks[name][0], fallback_sigma
                    k_replicate[well] = pm.Normal(
                        f"K_{name}_{well}", mu=mu, sigma=sigma
                    )
    else:
        k_params = {
            name: pm.Normal(
                _ctr_param_name(name),
                mu=ctr_ks[name][0],
                sigma=max(ctr_ks[name][1], fallback_sigma / 2),
            )
            for name in scheme.names
        }
    return k_params, k_replicate


def _free_k_init(  # noqa: PLR0913
    fit_results: dict[str, FitResult],
    wells_list: list[str],
    scheme: PlateScheme,
    ctr_ks: dict[str, tuple[float, float]],
    n_sd: float,
    *,
    ctr_free_k: bool,
    fallback_sigma: float = 0.6,
) -> tuple[list[str], ArrayF, ArrayF, dict[str, str]]:
    """Classify wells for the multi-fit K prior and return free-well init.

    Wells that receive an *individual* K prior — every unknown well, plus each
    control replicate when *ctr_free_k* is True — are collected into
    ``free_wells`` with aligned prior mean/sigma vectors, so the caller can
    build a single vectorized ``K_free`` random variable (one graph node)
    instead of one scalar ``pm.Normal`` per well.  Wells whose K is shared
    across a control group (``ctr_free_k`` False) are returned in ``shared_of``
    mapping well -> group name; the caller supplies their shared scalar RV.

    The per-well ``(mu, sigma)`` values reproduce the previous scalar-prior
    construction exactly, including the mode-dependent sigma floor.

    Parameters
    ----------
    fit_results : dict[str, FitResult]
        Per-well preliminary fit results.
    wells_list : list[str]
        Ordered active well keys.
    scheme : PlateScheme
        Plate scheme defining control groups.
    ctr_ks : dict[str, tuple[float, float]]
        Weighted K mean and stderr per control group.
    n_sd : float
        Prior width multiplier applied to the preliminary-fit stderr.
    ctr_free_k : bool
        If True, control replicates get individual priors (like unknowns).
    fallback_sigma : float
        Prior sigma for control replicates lacking a reliable preliminary fit.

    Returns
    -------
    free_wells : list[str]
        Wells receiving an individual (vectorized) K prior, in input order.
    mu : ArrayF
        Prior means aligned with *free_wells*.
    sigma : ArrayF
        Prior sigmas aligned with *free_wells*.
    shared_of : dict[str, str]
        Maps each shared (control-group) well to its group name.

    Raises
    ------
    ValueError
        If an active well in *wells_list* has no preliminary fit result.
    """
    well_to_group: dict[str, str] = {}
    for name, wells in scheme.names.items():
        for well in wells:
            well_to_group.setdefault(well, name)
    well_k_init = (
        _well_k_init_from_results(fit_results, scheme, n_sd) if ctr_free_k else {}
    )
    free_wells: list[str] = []
    mus: list[float] = []
    sigmas: list[float] = []
    shared_of: dict[str, str] = {}
    for key in wells_list:
        group = well_to_group.get(key, "")
        if group and not ctr_free_k:
            shared_of[key] = group
            continue
        result = fit_results[key].result
        if result is None:
            msg = f"Fit result for well {key} is missing."
            raise ValueError(msg)
        if group:  # ctr_free_k replicate: own preliminary fit or group fallback
            mu, sigma = well_k_init.get(key, (ctr_ks[group][0], fallback_sigma))
        elif ctr_free_k:  # unknown well (free-CTR mode uses _safe_sigma default)
            p = result.params["K"]
            mu, sigma = p.value, _safe_sigma(p.stderr, n_sd)
        else:  # unknown well (shared-CTR mode uses a 1e-3 sigma floor)
            p = result.params["K"]
            mu = p.value
            sigma = max(p.stderr * n_sd, 1e-3) if p.stderr else 1e-3
        free_wells.append(key)
        mus.append(float(mu))
        sigmas.append(float(sigma))
    return free_wells, np.asarray(mus), np.asarray(sigmas), shared_of


def _resolve_well_k(  # noqa: PLR0913
    key: str,
    ctr_name: str,
    pars: dict[str, typing.Any],
    k_params: dict[str, typing.Any],
    k_replicate: dict[str, typing.Any],
    *,
    ctr_free_k: bool,
) -> typing.Any:  # noqa: ANN401
    """Return the PyMC K variable for *key* given the current CTR-K mode."""
    if ctr_free_k:
        k_rep = k_replicate.get(key)
        return k_rep if k_rep is not None else pars[f"K_{key}"]
    return k_params[ctr_name] if ctr_name else pars[f"K_{key}"]


def _ph_at_mid_fluorescence(da: DataArray) -> float | None:
    """Estimate pH at 50 % of fluorescence range by linear interpolation.

    Parameters
    ----------
    da : DataArray
        Titration data (x = pH, y = fluorescence).

    Returns
    -------
    float | None
        Interpolated pH at ``(y_max + y_min) / 2``, or ``None`` if the
        midpoint crossing cannot be found.
    """
    x, y = da.x, da.y
    if len(x) < 2:  # noqa: PLR2004
        return None
    y_mid = (float(y.max()) + float(y.min())) / 2
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    for i in range(len(ys) - 1):
        if (ys[i] - y_mid) * (ys[i + 1] - y_mid) <= 0:
            denom = ys[i + 1] - ys[i]
            if denom == 0:
                return float(xs[i])
            t = (y_mid - ys[i]) / denom
            return float(xs[i] + t * (xs[i + 1] - xs[i]))
    return None


def _well_k_init_from_results(
    results: dict[str, FitResult],
    scheme: PlateScheme,
    n_sd: float,
    fallback_sigma: float = 0.6,
) -> dict[str, tuple[float, float]]:
    """Extract per-well K initialization from preliminary fit results.

    Returns a dict mapping each CTR well key to ``(K_value, K_sigma)``.
    Wells whose K is at the optimizer bound or whose ``stderr`` exceeds 1.0
    fall back to a pH-at-mid-fluorescence estimate with *fallback_sigma*.
    For wells with a reliable preliminary fit the sigma is
    ``min(max(sK * n_sd, 0.2), fallback_sigma)``.

    Parameters
    ----------
    results : dict[str, FitResult]
        Preliminary fit results keyed by well name.
    scheme : PlateScheme
        Plate scheme; only wells listed in ``scheme.names`` are processed.
    n_sd : float
        Number of standard deviations used to widen the sigma for reliable
        wells (same as for unknown wells in
        :func:`create_parameter_priors`).
    fallback_sigma : float
        Prior sigma used when the preliminary fit is unreliable (K at
        bound or large stderr).  Also acts as an upper cap for good fits.
        Default 0.6 (covers roughly ±1 pH unit, as suggested by user).

    Returns
    -------
    dict[str, tuple[float, float]]
        Mapping each CTR well key to ``(K_value, K_sigma)``.
    """
    ctr_wells = {well for wells in scheme.names.values() for well in wells}
    well_k: dict[str, tuple[float, float]] = {}
    for well in ctr_wells:
        r = results.get(well)
        if not (r and r.result and "K" in r.result.params):
            continue
        p = r.result.params["K"]
        at_bound = (p.min is not None and abs(p.value - p.min) < 0.01) or (  # noqa: PLR2004
            p.max is not None and abs(p.value - p.max) < 0.01  # noqa: PLR2004
        )
        large_stderr = p.stderr is None or p.stderr > 1.0
        if at_bound or large_stderr:
            # Pick pH@mid from the label with the largest signal range (Δy),
            # as that label has the clearest sigmoid and most reliable midpoint.
            ph_mid: float | None = None
            if r.dataset:
                best_delta, best_ph = -1.0, None
                for da in r.dataset.values():
                    if len(da.y) < 2:  # noqa: PLR2004
                        continue
                    delta = float(da.y.max() - da.y.min())
                    ph = _ph_at_mid_fluorescence(da)
                    if ph is not None and delta > best_delta:
                        best_delta, best_ph = delta, ph
                ph_mid = best_ph
            if ph_mid is not None:
                well_k[well] = (ph_mid, fallback_sigma)
            # else: leave absent → _build_ctr_k_params uses group mean fallback
        else:
            sigma = _safe_sigma(
                p.stderr, n_sd, default=0.2, floor=0.2, cap=fallback_sigma
            )
            well_k[well] = (p.value, sigma)
    return well_k


def fit_binding_pymc_multi(  # noqa: C901, PLR0912, PLR0913, PLR0915
    results: Mapping[str, Dataset | FitResult],
    scheme: PlateScheme,
    *,
    n_sd: float = 5.0,
    n_xerr: float = 1.0,
    min_x_step: float = 0.2,
    x_error_model: Literal["deterministic", "per_well"] = "deterministic",
    x_start_between_sigma: float = _DEFAULT_X_START_BETWEEN_SIGMA,
    ctr_free_k: bool = False,
    sample_ppc: bool = False,
    per_well_ye_mags: bool | None = None,
    well_noise_scale: bool = False,
    shared_well_noise_scale: bool = False,
    label_noise_scale_sigma: float = 0.3,
    well_noise_sd_sigma: float = 0.3,
    well_noise_scale_sigma: float | None = None,
    noise: NoiseConfig = _DEFAULT_NOISE,
    robust: RobustConfig = _DEFAULT_ROBUST,
    init: InitConfig = _DEFAULT_INIT,
    sampler: SamplerConfig = _DEFAULT_SAMPLER,
) -> MultiFitResult:
    """Multi-well PyMC with shared K per control group and per-label noise.

    Parameters
    ----------
    results : Mapping[str, Dataset | FitResult]
        Per-well datasets or initial fit results. Raw datasets are first fitted
        with :func:`fit_binding_glob` to seed the Bayesian model.
    scheme : PlateScheme
        Plate scheme defining control groups for shared-K priors.
    n_sd : float
        Prior width multiplier for per-well S0/S1 parameters.
    n_xerr : float
        Scaling factor applied to x-value uncertainties.
    min_x_step : float
        Minimum inferred change in ``x`` between consecutive titration steps.
    x_error_model : Literal["deterministic", "per_well"]
        Model for x-error propagation across wells. ``"deterministic"`` shares
        one latent ``x_true`` walk across all wells; ``"per_well"`` gives each
        well its own cumulative-additions walk.
    x_start_between_sigma : float
        Fixed between-well scale for the starting x (pH). Defaults to a small
        nonzero value, so each well gets its own ``x_start_well ~ Normal(
        x_start, x_start_between_sigma)`` tightly anchored on the shared plate
        ``x_start`` prior (common-mode anchor plus independent per-well jitter).
        Set to ``0.0`` for a single shared ``x_start`` across all wells.
    ctr_free_k : bool
        If ``True``, each control replicate gets an independent flat K prior
        instead of a shared control K.
    sample_ppc : bool
        If ``True``, add posterior predictive samples to the returned trace.
    per_well_ye_mags : bool | None
        Learn per-well (not just per-label) ``ye_mag`` factors. ``None`` follows
        the noise config's ``learn_ye_mags`` when a structured model is supplied.
    well_noise_scale : bool
        Enable a per-well multiplicative noise scale.
    shared_well_noise_scale : bool
        Share the well-noise scale across labels.
    label_noise_scale_sigma : float
        LogNormal prior scale for the per-label noise scale.
    well_noise_sd_sigma : float
        HalfNormal prior scale for the per-well noise spread.
    well_noise_scale_sigma : float | None
        Backwards-compatible alias for *well_noise_sd_sigma*.
    noise : NoiseConfig
        Observation-noise configuration (ye_mag multiplier or structured
        floor/gain/alpha). See :class:`NoiseConfig`.
    robust : RobustConfig
        Robust-likelihood configuration (Student-t or contamination mixture).
        See :class:`RobustConfig`.
    init : InitConfig
        Prior-initialization strategy (``"lmfit"`` prefit or ``"data_priors"``).
        See :class:`InitConfig`.
    sampler : SamplerConfig
        NUTS sampling controls. See :class:`SamplerConfig`.

    Returns
    -------
    MultiFitResult
        Shared PyMC trace together with reconstructed per-well fit results.

    Raises
    ------
    ValueError
        If no valid dataset is found in results.
    """
    robust_on = robust.enabled
    robust_likelihood = robust.likelihood
    student_t_nu = robust.nu
    contamination_frac_prior = robust.contamination_frac_prior
    shared_alpha = noise.shared_alpha
    shared_gain = noise.shared_gain
    shared_floor = noise.shared_floor
    learn_ye_mags = noise.learn_ye_mags
    shared_ye_mags = noise.shared_ye_mags
    ye_mag_prior = noise.ye_mag_prior
    ye_mag_mu = noise.ye_mag_mu
    ye_mag_sigma = noise.ye_mag_sigma
    n_samples = sampler.n_samples
    n_tune = sampler.n_tune
    target_accept = sampler.target_accept
    init_strategy = init.strategy
    data_prior_edge_points = init.edge_points
    data_prior_signal_sigma_scale = init.signal_sigma_scale
    data_prior_k_prior = init.k_prior
    data_prior_k_bounds = init.k_bounds
    data_prior_k_sigma = init.k_sigma
    floor_mode = noise.floor_mode
    gain_mode = noise.gain_mode
    alpha_mode = noise.alpha_mode

    fit_results, prefer_centered = _normalize_fit_inputs(
        results,
        init_strategy=init_strategy,
        data_prior_edge_points=data_prior_edge_points,
        data_prior_signal_sigma_scale=data_prior_signal_sigma_scale,
        data_prior_k_prior=data_prior_k_prior,
        data_prior_k_bounds=data_prior_k_bounds,
        data_prior_k_sigma=data_prior_k_sigma,
    )
    ds = next((r.dataset for r in fit_results.values() if r.dataset), None)
    if ds is None:
        msg = "No valid dataset found in results."
        raise ValueError(msg)
    noise_model = (
        _resolve_structured_noise_model(noise, ds)
        if noise.kind == "structured"
        else None
    )
    xc = next(iter(ds.values())).xc
    x_errc = next(iter(ds.values())).x_errc
    labels = list(ds.keys())
    if well_noise_scale_sigma is not None:
        well_noise_sd_sigma = well_noise_scale_sigma
    use_per_well_ye_mags = (
        learn_ye_mags
        if noise_model is not None and per_well_ye_mags is None
        else bool(per_well_ye_mags)
    )
    floor_mode, gain_mode, alpha_mode = _resolve_noise_modes(
        prefer_centered=prefer_centered,
        floor_mode=floor_mode,
        gain_mode=gain_mode,
        alpha_mode=alpha_mode,
    )
    values: dict[str, list[float | None]] = {}
    stderr: dict[str, list[float | None]] = {}

    for name, wells in scheme.names.items():
        values[name] = [
            v.result.params["K"].value
            for well, v in fit_results.items()
            if v.result and well in wells
        ]
        stderr[name] = [
            v.result.params["K"].stderr
            for well, v in fit_results.items()
            if v.result and well in wells
        ]
    ctr_ks = weighted_stats(values, stderr)
    active_wells = {key for key, r in fit_results.items() if r.result and r.dataset}
    wells_list = [
        key
        for key in fit_results
        if fit_results[key].result and fit_results[key].dataset
    ]
    n_wells = len(wells_list)
    n_steps = len(xc)

    # Split wells into individually-fit (vectorized K_free) and shared-group K.
    free_wells, k_free_mu, k_free_sigma, shared_of = _free_k_init(
        fit_results, wells_list, scheme, ctr_ks, n_sd, ctr_free_k=ctr_free_k
    )

    coords: dict[str, list[int] | list[str]] = {
        "well": wells_list,
        "step": list(range(n_steps)),
        "step_diff": list(range(n_steps - 1)),
    }
    if free_wells:
        coords["free_well"] = free_wells

    with pm.Model(coords=coords):
        robust_nu = (
            _student_t_nu_value(student_t_nu)
            if robust_on and robust_likelihood == "student_t"
            else 3.0
        )
        if robust_on and robust_likelihood == "mixture":
            pi_outliers, outlier_inflate = _build_outlier_priors(
                labels, contamination_frac_prior
            )
        else:
            pi_outliers, outlier_inflate = {}, None

        if x_error_model not in {"deterministic", "per_well"}:
            msg = (
                f"Unknown x_error_model {x_error_model!r}; valid options are "
                "'deterministic' and 'per_well' "
                "('hierarchical_per_well' was removed)."
            )
            raise ValueError(msg)

        if x_error_model == "per_well" and n_xerr > 0:
            direction, x_start_sigma, step_nominal, step_sigmas, min_steps = (
                _pipetting_walk_params(xc, x_errc, n_xerr, min_x_step=min_x_step)
            )
            x_start = _build_multi_x_start(xc, x_start_sigma, x_start_between_sigma)
            x_step = pm.TruncatedNormal(
                "x_step",
                mu=step_nominal[:, None],
                sigma=np.maximum(step_sigmas, 1e-6)[:, None],
                lower=min_steps[:, None],
                shape=(n_steps - 1, n_wells),
                dims=("step_diff", "well"),
            )
            x_cumsum = pm.math.cumsum(x_step, axis=0)
            x_offsets = pm.math.concatenate([pt.zeros((1, n_wells)), x_cumsum], axis=0)
            x_w_all = pm.Deterministic(
                "x_true",
                x_start + direction * x_offsets,
                dims=("step", "well"),
            )

        elif x_error_model == "per_well":
            x_matrix = np.empty((n_steps, n_wells), dtype=float)
            for w_idx, key in enumerate(wells_list):
                well_ds = fit_results[key].dataset
                if well_ds is None:
                    msg = f"Dataset for well {key} is missing."
                    raise ValueError(msg)
                well_x = next(iter(well_ds.values())).xc
                if len(well_x) != n_steps:
                    msg = f"Dataset for well {key} has inconsistent x length."
                    raise ValueError(msg)
                x_matrix[:, w_idx] = np.asarray(well_x, dtype=float)
            x_w_all = pm.Deterministic(
                "x_true",
                as_tensor_variable(x_matrix),
                dims=("step", "well"),
            )

        else:
            x_true: typing.Any = create_x_true(
                xc,
                x_errc,
                n_xerr,
                min_x_step=min_x_step,
            )
            x_w_all = pt.tile(x_true[:, None], (1, n_wells))

        # Vectorized K priors: the individually-fit wells (all unknowns, plus
        # every control replicate when ctr_free_k) share a single K_free RV
        # with dims="free_well" — one graph node instead of a scalar pm.Normal
        # per well, which speeds up compilation and NUTS sampling. Shared
        # control groups keep one scalar K_ctr_* RV each, broadcast across
        # their replicate wells.
        k_params: dict[str, typing.Any] = {}
        if not ctr_free_k:
            k_params, _ = _build_ctr_k_params(
                scheme, ctr_ks, active_wells, ctr_free_k=False
            )
        k_free = (
            pm.Normal("K_free", mu=k_free_mu, sigma=k_free_sigma, dims="free_well")
            if free_wells
            else None
        )
        free_pos = {well: i for i, well in enumerate(free_wells)}
        k_segments: list[typing.Any] = []
        for key in wells_list:
            if key in free_pos:
                assert k_free is not None  # noqa: S101 — free_pos implies k_free built
                k_segments.append(k_free[free_pos[key]])
            else:
                k_segments.append(k_params[shared_of[key]])
        k_all = pt.stack(k_segments)  # (n_wells,)

        # Collect and create vectorized S0 and S1 priors
        s0_vars = {}
        s1_vars = {}
        for lbl in labels:
            mu_s0, sig_s0 = [], []
            mu_s1, sig_s1 = [], []
            for key in wells_list:
                r = fit_results[key]
                if r.result is None:
                    msg = f"Fit result for well {key} is missing."
                    raise ValueError(msg)
                p_s0 = r.result.params[f"S0_{lbl}"]
                p_s1 = r.result.params[f"S1_{lbl}"]
                mu_s0.append(p_s0.value)
                sig_s0.append(_safe_sigma(p_s0.stderr, n_sd))
                mu_s1.append(p_s1.value)
                sig_s1.append(_safe_sigma(p_s1.stderr, n_sd))
            s0_vars[lbl] = pm.Normal(
                f"S0_{lbl}", mu=np.array(mu_s0), sigma=np.array(sig_s0), dims="well"
            )
            s1_vars[lbl] = pm.Normal(
                f"S1_{lbl}", mu=np.array(mu_s1), sigma=np.array(sig_s1), dims="well"
            )

        if noise_model is not None:
            active_noise_model = _active_noise_model(noise_model, labels)
            noise_priors = build_pymc_noise_priors(
                active_noise_model,
                shared_alpha=shared_alpha,
                shared_gain=shared_gain,
                shared_floor=shared_floor,
                floor_mode=floor_mode,
                gain_mode=gain_mode,
                alpha_mode=alpha_mode,
            )
            if learn_ye_mags:
                ye_mags = _build_multi_ye_mag_priors(
                    labels,
                    per_well=use_per_well_ye_mags,
                    shared_ye_mags=shared_ye_mags,
                    prior="halfnormal",
                    sigma=5.0,
                )
        else:
            noise_priors = {}
            ye_mags = _build_multi_ye_mag_priors(
                labels,
                per_well=use_per_well_ye_mags,
                shared_ye_mags=shared_ye_mags,
                prior=ye_mag_prior,
                mu=ye_mag_mu,
                sigma=ye_mag_sigma,
            )
        if well_noise_scale:
            label_scale_sigma = max(float(label_noise_scale_sigma), 1e-6)
            well_sd_sigma = max(float(well_noise_sd_sigma), 1e-6)
            well_noise_scales = {}
            if shared_well_noise_scale:
                shared_label_noise_scale = pm.LogNormal(
                    "label_noise_scale",
                    mu=0.0,
                    sigma=label_scale_sigma,
                )
                shared_well_noise_sd = pm.HalfNormal(
                    "well_noise_sd", sigma=well_sd_sigma
                )
                shared_scale = pm.LogNormal(
                    "well_noise_scale",
                    mu=pm_math.log(shared_label_noise_scale),
                    sigma=shared_well_noise_sd,
                    dims="well",
                )
                well_noise_scales = dict.fromkeys(labels, shared_scale)
            else:
                for lbl in labels:
                    label_noise_scale = pm.LogNormal(
                        f"label_noise_scale_{lbl}",
                        mu=0.0,
                        sigma=label_scale_sigma,
                    )
                    well_noise_sd = pm.HalfNormal(
                        f"well_noise_sd_{lbl}",
                        sigma=well_sd_sigma,
                    )
                    well_noise_scales[lbl] = pm.LogNormal(
                        f"well_noise_scale_{lbl}",
                        mu=pm_math.log(label_noise_scale),
                        sigma=well_noise_sd,
                        dims="well",
                    )
        else:
            well_noise_scales = {}

        # Likelihoods
        first_ds = next(iter(fit_results.values())).dataset
        if first_ds is None:
            msg = "At least one dataset is required."
            raise ValueError(msg)
        is_ph = first_ds.is_ph
        for lbl in labels:
            mask_lbl, y_obs_full, y_err_full = _masked_obs_err_matrices(
                fit_results, wells_list, lbl, n_steps
            )

            y_model_all = binding_1site(
                x_w_all, k_all, s0_vars[lbl], s1_vars[lbl], is_ph=is_ph
            )
            mu_vec = y_model_all[mask_lbl]
            y_obs_vec = y_obs_full[mask_lbl]

            if noise_model is not None:
                # Heteroscedastic noise model
                noise_var = get_pymc_variance(
                    y_model_all, lbl, active_noise_model, noise_priors
                )
                sigma_obs_all = pm_math.sqrt(noise_var)
                if learn_ye_mags:
                    sigma_obs_all = ye_mags[lbl] * sigma_obs_all
                if well_noise_scale:
                    sigma_obs_all *= well_noise_scales[lbl][None, :]

                sigma_obs_all_det = pm.Deterministic(
                    f"sigma_obs_{lbl}", sigma_obs_all, dims=("step", "well")
                )
                sigma_vec = sigma_obs_all_det[mask_lbl]
            else:
                # Homoscedastic (scaled) noise model
                y_err_all = np.where(np.isfinite(y_err_full), y_err_full, 1.0)
                sigma_all = ye_mags[lbl] * y_err_all
                if well_noise_scale:
                    sigma_all *= well_noise_scales[lbl][None, :]
                sigma_obs_all_det = pm.Deterministic(
                    f"sigma_obs_{lbl}", sigma_all, dims=("step", "well")
                )
                sigma_vec = sigma_obs_all_det[mask_lbl]

            if robust_on and robust_likelihood == "mixture":
                _add_mixture_likelihood(
                    f"y_likelihood_{lbl}",
                    mu_vec,
                    sigma_vec,
                    y_obs_vec,
                    pi_outlier=pi_outliers[lbl],
                    outlier_inflate=outlier_inflate,
                )
            elif robust_on:
                pm.StudentT(
                    f"y_likelihood_{lbl}",
                    nu=robust_nu,
                    mu=mu_vec,
                    sigma=sigma_vec,
                    observed=y_obs_vec,
                )
            else:
                pm.Normal(
                    f"y_likelihood_{lbl}",
                    mu=mu_vec,
                    sigma=sigma_vec,
                    observed=y_obs_vec,
                )

        tune_steps = n_tune if n_tune is not None else n_samples // 2

        target_accept_ = (
            target_accept if target_accept is not None else 0.95 if n_xerr > 0 else 0.9
        )
        trace: xr.DataTree = _sample_trace(
            n_samples, tune=tune_steps, target_accept=target_accept_, sampler=sampler
        )

        if sample_ppc:
            pm.sample_posterior_predictive(trace, extend_inferencedata=True)

        if sampler.compute_log_likelihood:
            trace = _compute_sample_log_likelihood(trace)

    global_names: list[str] = []
    if robust_on and robust_likelihood == "student_t" and student_t_nu is None:
        global_names.append("student_t_nu")
    if robust_on and robust_likelihood == "mixture":
        global_names.extend(f"pi_outlier_{lbl}" for lbl in labels)
        global_names.append("outlier_inflate")

    return MultiFitResult(
        trace=trace,
        results=_per_well_fit_results_from_trace(
            trace,
            fit_results,
            scheme,
            x_error_model=x_error_model,
            global_p_names=global_names,
        ),
    )
