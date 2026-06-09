"""Bayesian (PyMC) fitting utilities and pipelines."""

from __future__ import annotations

import copy
import os
import re
import typing
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
from pytensor.configdefaults import config as pytensor_config
from pytensor.tensor import as_tensor_variable

from clophfit.fitting.models import binding_1site
from clophfit.fitting.plotting import PlotParameters, plot_fit

from .core import N_BOOT, fit_binding_glob  # local to avoid circular import
from .data_structures import (
    DataArray,
    Dataset,
    FitResult,
    MiniT,
    MultiFitResult,
    PlateNoiseModel,
    _Result,
)

if typing.TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from clophfit.clophfit_types import ArrayF
    from clophfit.prtecan import PlateScheme


NoiseParamMode = Literal["fixed", "free", "centered"]

_X_TRUE_INDEX_RE = re.compile(r"^x_true\[(\d+)\]$")


def build_pymc_noise_priors(  # noqa: C901, PLR0912, PLR0913, PLR0915
    noise_model: PlateNoiseModel,
    *,
    shared_alpha: bool = False,
    shared_gain: bool = False,
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
    priors["floor"] = {}
    for lbl in labels:
        mu = noise_model[lbl].sigma_floor
        if floor_mode == "fixed":
            priors["floor"][lbl] = pt.as_tensor_variable(mu)
        elif floor_mode == "free":
            # Uninformative prior, using mu only as a scale hint
            priors["floor"][lbl] = pm.HalfNormal(f"floor_{lbl}", sigma=max(mu, 1.0))
        else:  # centered
            n_pts = 7
            dof = max(1, n_pts - 1)
            rel_sigma = float(np.clip(1.0 / np.sqrt(2 * dof), 0.05, 0.5))
            sigma = max(rel_sigma * mu, 0.01)
            priors["floor"][lbl] = pm.Normal(f"floor_{lbl}", mu=mu, sigma=sigma)

    # 2. Gain (Poisson term)
    has_gain = any(p.gain > 0 for p in noise_model.values())
    if has_gain or gain_mode == "free":
        if shared_gain:
            gains = [p.gain for p in noise_model.values() if p.gain > 0]
            mu_g = np.mean(gains) if gains else 1.0
            if gain_mode == "fixed":
                priors["gain"] = pt.as_tensor_variable(mu_g)
            elif gain_mode == "free":
                priors["gain"] = pm.Exponential("gain", lam=1.0)
            else:  # centered
                priors["gain"] = pm.Normal("gain", mu=mu_g, sigma=max(0.2 * mu_g, 0.1))
        else:
            priors["gain"] = {}
            for lbl in labels:
                mu_g = getattr(noise_model[lbl], "gain", 0.0)
                if gain_mode == "fixed":
                    priors["gain"][lbl] = pt.as_tensor_variable(mu_g)
                elif gain_mode == "free":
                    priors["gain"][lbl] = pm.Exponential(f"gain_{lbl}", lam=1.0)
                elif mu_g > 0.0:
                    priors["gain"][lbl] = pm.Normal(
                        f"gain_{lbl}", mu=mu_g, sigma=max(0.20 * mu_g, 0.01)
                    )
                else:
                    priors["gain"][lbl] = pt.as_tensor_variable(0.0)

    # 3. Alpha (proportional error)
    has_alpha = any(p.alpha > 0 for p in noise_model.values())
    if has_alpha or alpha_mode == "free":
        if shared_alpha:
            alphas = [p.alpha for p in noise_model.values() if p.alpha > 0]
            mu_a = np.mean(alphas) if alphas else 0.02
            if alpha_mode == "fixed":
                priors["rel_error"] = pt.as_tensor_variable(mu_a)
            elif alpha_mode == "free":
                priors["rel_error"] = pm.HalfNormal("rel_error", sigma=0.02)
            else:  # centered
                priors["rel_error"] = pm.TruncatedNormal(
                    "rel_error", mu=mu_a, sigma=max(0.25 * mu_a, 0.001), lower=0.0
                )
        else:
            priors["rel_error"] = {}
            for lbl in labels:
                mu_a = getattr(noise_model[lbl], "alpha", 0.0)
                if alpha_mode == "fixed":
                    priors["rel_error"][lbl] = pt.as_tensor_variable(mu_a)
                elif alpha_mode == "free":
                    priors["rel_error"][lbl] = pm.HalfNormal(
                        f"rel_error_{lbl}", sigma=0.02
                    )
                elif mu_a > 0.0:
                    priors["rel_error"][lbl] = pm.TruncatedNormal(
                        f"rel_error_{lbl}",
                        mu=mu_a,
                        sigma=max(0.25 * mu_a, 0.001),
                        lower=0.0,
                    )
                else:
                    priors["rel_error"][lbl] = pm.HalfNormal(
                        f"rel_error_{lbl}", sigma=0.02
                    )

    return priors


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
    # Broadcast floor² to match y_pos shape so the result is always
    # per-point (never 0-d), even when gain and alpha are both absent.
    var = priors["floor"][label] ** 2 * pt.ones_like(y_pos)  # type: ignore[no-untyped-call]

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


pytensor_config.linker = "cvm"  # ← ripristina backend C come PyMC 5


def _pymc_sample_parallel_args(nuts_sampler: str = "default") -> dict[str, object]:
    """Return sampling kwargs for pm.sample(), including optional nuts_sampler."""
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
        # The blackjax inner progress bar uses JAX IO callbacks which are not
        # supported inside jax.vmap ("IO effect not supported in vmap-of-cond"),
        # so disable it via progressbar=False.
        if nuts_sampler in {"blackjax", "numpyro"}:
            kwargs["nuts_sampler_kwargs"] = {"chain_method": "vectorized"}
            kwargs["progressbar"] = False
    if "PYTEST_CURRENT_TEST" in os.environ:
        kwargs.update({"cores": 1, "chains": 1})
    return kwargs


def _compute_sample_log_likelihood(trace: xr.DataTree) -> xr.DataTree:
    """Populate the log_likelihood group on sampled PyMC inference data."""
    return typing.cast(
        "xr.DataTree",
        pm.compute_log_likelihood(trace, extend_inferencedata=True, progressbar=False),
    )


def _normalize_fit_input(
    ds_or_fr: Dataset | FitResult[MiniT],
) -> tuple[FitResult[MiniT], bool]:
    """Normalize PyMC input to a copied preliminary fit result.

    Parameters
    ----------
    ds_or_fr : Dataset | FitResult[MiniT]
        Either a raw dataset or a pre-fit result used to seed the Bayesian
        model.

    Returns
    -------
    tuple[FitResult[MiniT], bool]
        The normalized fit result and whether the caller explicitly supplied a
        pre-fit ``FitResult``.
    """
    if isinstance(ds_or_fr, Dataset):
        return fit_binding_glob(ds_or_fr), False
    return copy.deepcopy(ds_or_fr), True


def _normalize_fit_inputs(
    results: Mapping[str, Dataset | FitResult[MiniT]],
) -> tuple[dict[str, FitResult[MiniT]], bool]:
    """Normalize multi-well PyMC inputs to copied preliminary fit results.

    Parameters
    ----------
    results : Mapping[str, Dataset | FitResult[MiniT]]
        Per-well raw datasets or pre-fit results.

    Returns
    -------
    tuple[dict[str, FitResult[MiniT]], bool]
        Normalized per-well fit results and whether every input item was already
        a ``FitResult``.
    """
    normalized: dict[str, FitResult[MiniT]] = {}
    all_prefit = True
    for key, value in results.items():
        if isinstance(value, Dataset):
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


def _masked_obs_err_matrices(
    fit_results: Mapping[str, FitResult[MiniT]],
    wells_list: Sequence[str],
    lbl: str,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build row-major mask, observation, and error matrices for multi-well likelihood.

    Parameters
    ----------
    fit_results : Mapping[str, FitResult[MiniT]]
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


def create_x_true(
    xc: ArrayF,
    x_errc: ArrayF,
    n_xerr: float,
    lower_nsd: float = 2.5,
    min_x_step: float = 0.2,
) -> ArrayF | pm.Deterministic:
    """Create latent variables for x-values with uncertainty.

    Returns a PyMC Deterministic variable when in a Model context with uncertainty,
    or a numpy array when there's no uncertainty or no active Model.
    """
    if n_xerr > 0 and np.any(x_errc > 0):
        x_errc_scaled = np.maximum(x_errc * n_xerr, 1e-6)
        x_errc_cumulative = np.maximum.accumulate(x_errc_scaled)
        step_sigmas = np.empty_like(x_errc_cumulative)
        step_sigmas[0] = x_errc_cumulative[0]
        if len(xc) > 1:
            step_var = np.maximum(
                x_errc_cumulative[1:] ** 2 - x_errc_cumulative[:-1] ** 2,
                1e-12,
            )
            step_sigmas[1:] = np.sqrt(step_var)

        direction = 1.0 if np.all(np.diff(xc) >= 0.0) else -1.0
        step_nominal = direction * np.diff(xc)
        min_steps = np.maximum(
            step_nominal - lower_nsd * step_sigmas[1:],
            min_x_step,
        )

        x_start = pm.Normal("x_start", mu=xc[0], sigma=step_sigmas[0])
        x_step = pm.TruncatedNormal(
            "x_step",
            mu=step_nominal,
            sigma=np.maximum(step_sigmas[1:], 1e-6),
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
        sigma = max(p.stderr * n_sd, 1e-3) if p.stderr else default_sigma
        # Skip creating a separate K prior if it belongs to a control group
        if ctr_name and name == "K":
            continue
        prior_name = param_name(name)
        if prior_name in priors:
            continue
        priors[prior_name] = pm.Normal(prior_name, mu=p.value, sigma=sigma)
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


def _add_y_likelihood(
    name: str,
    y_model: typing.Any,  # noqa: ANN401  # pt.TensorVariable (no stubs)
    da: DataArray,
    sigma: typing.Any,  # noqa: ANN401  # pt.TensorVariable | np.ndarray
    *,
    robust: bool = False,
) -> None:
    """Add a Normal or StudentT likelihood for one label."""
    if robust:
        pm.StudentT(name, nu=3.0, mu=y_model[da.mask], sigma=sigma, observed=da.y)
    else:
        pm.Normal(name, mu=y_model[da.mask], sigma=sigma, observed=da.y)


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


def process_trace(
    trace: xr.DataTree, p_names: typing.KeysView[str], ds: Dataset
) -> FitResult[xr.DataTree]:
    """Process the trace to extract parameter estimates and update datasets.

    Parameters
    ----------
    trace : xr.DataTree
        The posterior samples from PyMC sampling.
    p_names: typing.KeysView[str]
        Parameter names.
    ds : Dataset
        The dataset containing titration data.

    Returns
    -------
    FitResult[xr.DataTree]
        The updated fit result with extracted parameter values and datasets.
        Residuals are WEIGHTED (weight * (obs - pred)) where weight = 1/y_err,
        computed using posterior mean parameter estimates.

    """
    # Extract summary statistics for parameters
    rdf = _trace_summary_df(trace)
    rpars = Parameters()
    for name, row in rdf.iterrows():
        if name in p_names:
            _add_param_from_summary(rpars, str(name), row)
    # x_true and x_errc
    nxc, nx_errc = _extract_x_true_from_trace_df(rdf)
    if nxc.size > 0:
        for da in ds.values():
            da.xc = nxc  # Update x_true values in the dataset
            da.x_errc = nx_errc  # Posterior already incorporates n_xerr prior scaling
    # Scale y_errc if present. Newer models use per-label ye_mag_{lbl}; older
    # traces may still carry a single global ye_mag.
    global_mag = _summary_mean_or_none(rdf, "ye_mag")
    for lbl, da in ds.items():
        mag = _summary_mean_or_none(rdf, f"ye_mag_{lbl}")
        if mag is None:
            mag = global_mag
        if mag is not None:
            da.y_errc *= mag
    # Create figure
    fig = figure.Figure()
    ax = fig.add_subplot(111)
    plot_fit(ax, ds, rpars, nboot=N_BOOT, pp=PlotParameters(ds.is_ph))

    residuals = _compute_weighted_residuals(ds, rpars)
    return FitResult(fig, _Result(rpars, residual=residuals), trace, ds)


def extract_fit(
    key: str,
    ctr: str,
    trace_df: xr.DataTree | MultiFitResult | pd.DataFrame,
    ds: Dataset,
    well_key: str = "",
) -> FitResult[xr.DataTree]:
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
        When provided, per-well x posteriors (``x_per_well[step, well_key]``)
        are used instead of the global ``x_true``.  Pass the well identifier
        for xrw fits so each well's .dat/.png uses its own inferred pH axis.

    Returns
    -------
    FitResult[xr.DataTree]
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
    # Use per-well x (xrw model) when available; fall back to global x_true.
    nxc, nx_errc = (
        _extract_x_per_well_from_trace_df(trace_df, well_key)
        if well_key
        else (np.array([]), np.array([]))
    )
    if nxc.size == 0:
        nxc, nx_errc = _extract_x_true_from_trace_df(trace_df)
    if nxc.size > 0:
        for da in ds.values():
            da.xc = nxc
            da.x_errc = nx_errc
    # Create figure
    fig = figure.Figure()
    ax = fig.add_subplot(111)
    plot_fit(ax, ds, rpars, nboot=N_BOOT, pp=PlotParameters(ds.is_ph))
    residuals = _compute_weighted_residuals(ds, rpars)
    mini = trace_obj if isinstance(trace_obj, xr.DataTree) else xr.DataTree()
    return FitResult(fig, _Result(rpars, residual=residuals), mini, ds)


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
        if isinstance(name, str) and name.startswith("x_true"):
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


def _extract_x_per_well_from_trace_df(
    trace_df: pd.DataFrame,
    well_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract per-well x values for *well_key* from an xrw trace summary.

    ArviZ names the ``x_per_well`` deterministic (with dims ``step`` x ``well``)
    as ``x_per_well[{step}, {well}]``.  This function collects those rows for a
    specific well and returns them sorted by step index.

    When found, per-well x values take precedence over the global ``x_true``,
    allowing each well in an xrw fit to use its own inferred pH axis.

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
        by step.  Both arrays are empty if ``x_per_well`` rows are absent.

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
            and name.startswith("x_per_well[")
            and name.endswith(suffix)
        ):
            step_str = name[len("x_per_well[") : -len(suffix)]
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


def _per_well_fit_results_from_trace(
    trace: xr.DataTree,
    fit_results: Mapping[str, FitResult[MiniT]],
    scheme: PlateScheme,
    *,
    x_error_model: Literal["deterministic", "per_well", "hierarchical_per_well"],
) -> dict[str, FitResult[xr.DataTree]]:
    """Reconstruct per-well fit results from a shared multi-well trace."""
    trace_df = _trace_summary_df(trace)
    per_well_results: dict[str, FitResult[xr.DataTree]] = {}
    for key, fr in fit_results.items():
        if fr.dataset is None:
            continue
        ctr = next((name for name, wells in scheme.names.items() if key in wells), "")
        well_key = key if x_error_model in {"per_well", "hierarchical_per_well"} else ""
        per_well_results[key] = extract_fit(
            key,
            ctr,
            trace_df,
            copy.deepcopy(fr.dataset),
            well_key=well_key,
        )
        per_well_results[key].mini = trace
    return per_well_results


def fit_binding_pymc(  # noqa: PLR0913
    ds_or_fr: Dataset | FitResult[MiniT],
    n_sd: float = 10.0,
    n_xerr: float = 1.0,
    n_samples: int = 2000,
    nuts_sampler: str = "default",
    *,
    n_tune: int | None = None,
    target_accept: float | None = None,
    max_treedepth: int | None = None,
    noise_model: PlateNoiseModel | None = None,
    robust: bool = False,
    floor_mode: NoiseParamMode | None = None,
    gain_mode: NoiseParamMode | None = None,
    alpha_mode: NoiseParamMode | None = None,
    learn_ye_mags: bool = False,
    min_x_step: float = 0.2,
) -> FitResult[xr.DataTree]:
    """Analyze multi-label titration datasets using PyMC (single model).

    Parameters
    ----------
    ds_or_fr : Dataset | FitResult[MiniT]
        Either a Dataset (will run initial LS fit) or a FitResult with initial params.
    n_sd : float
        Number of standard deviations for parameter priors.
    n_xerr : float
        Scaling factor for x-error.
    n_samples : int
        Number of MCMC samples.
    nuts_sampler : str
        NUTS sampler backend: ``"default"`` (PyMC C/pytensor), ``"blackjax"``,
        ``"numpyro"``, or ``"nutpie"``.
    n_tune : int | None
        Number of tuning steps. If ``None`` (default), use ``n_samples // 2``.
    target_accept : float | None
        NUTS target acceptance probability. If ``None`` (default), 0.95 is
        used when *n_xerr* > 0 and 0.9 otherwise.
    max_treedepth : int | None
        Maximum tree depth for NUTS sampler. If ``None`` (default), PyMC's
        default is used.
    noise_model : PlateNoiseModel | None
        Noise model specification.  ``None`` (default) uses a simple per-label
        ``ye_mag_{lbl}`` HalfNormal to scale the existing ``y_err``.  Pass a
        ``PlateNoiseModel`` to infer per-label floor, gain, and alpha
        from the full heteroscedastic noise model.
    robust : bool
        If ``True``, use StudentT likelihood (nu=3) for robust regression.
    floor_mode : NoiseParamMode | None
        How to treat the floor parameter.  ``None`` (default) resolves to
        ``"centered"`` for pre-fit ``FitResult`` input and ``"free"`` for raw
        ``Dataset`` input.
    gain_mode : NoiseParamMode | None
        How to treat the gain parameter.  ``None`` follows the same input-based
        rule as *floor_mode*.
    alpha_mode : NoiseParamMode | None
        How to treat the alpha (rel_error) parameter.  ``None`` follows the
        same input-based rule as *floor_mode*.
    learn_ye_mags : bool
        If ``True``, learn per-label scaling factors (``ye_mag_{lbl}``) even
        when a full *noise_model* is provided.
    min_x_step : float
        Minimum inferred change in ``x`` between consecutive titration steps
        when latent-x modeling is enabled.

    Returns
    -------
    FitResult[xr.DataTree]
        Bayesian fitting results.
    """
    fr, prefer_centered = _normalize_fit_input(ds_or_fr)

    if fr.result is None or fr.dataset is None:
        return FitResult()
    params = fr.result.params
    ds = copy.deepcopy(fr.dataset)
    labels = list(ds.keys())
    xc = next(iter(ds.values())).xc
    x_errc = next(iter(ds.values())).x_errc
    floor_mode, gain_mode, alpha_mode = _resolve_noise_modes(
        prefer_centered=prefer_centered,
        floor_mode=floor_mode,
        gain_mode=gain_mode,
        alpha_mode=alpha_mode,
    )
    with pm.Model():
        pars = create_parameter_priors(params, n_sd)
        x_true = create_x_true(xc, x_errc, n_xerr, min_x_step=min_x_step)

        if noise_model is None:
            ye_mags = {lbl: pm.HalfNormal(f"ye_mag_{lbl}", sigma=1.0) for lbl in labels}
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
                    f"y_likelihood_{lbl}", y_model, da, sigma, robust=robust
                )
        else:
            noise_priors = build_pymc_noise_priors(
                noise_model,
                floor_mode=floor_mode,
                gain_mode=gain_mode,
                alpha_mode=alpha_mode,
            )
            if learn_ye_mags:
                ye_mags = {
                    lbl: pm.HalfNormal(f"ye_mag_{lbl}", sigma=1.0) for lbl in labels
                }

            for lbl, da in ds.items():
                y_model = binding_1site(
                    x_true,
                    pars["K"],
                    pars[f"S0_{lbl}"],
                    pars[f"S1_{lbl}"],
                    is_ph=ds.is_ph,
                )
                noise_var = get_pymc_variance(y_model, lbl, noise_model, noise_priors)
                sigma = pm_math.sqrt(noise_var)
                if learn_ye_mags:
                    sigma = ye_mags[lbl] * sigma

                sigma_det = pm.Deterministic(f"sigma_obs_{lbl}", sigma)
                _add_y_likelihood(
                    f"y_likelihood_{lbl}",
                    y_model,
                    da,
                    sigma_det[da.mask],
                    robust=robust,
                )

        # Inference
        tune = n_tune if n_tune is not None else n_samples // 2
        target_accept_ = (
            target_accept if target_accept is not None else 0.95 if n_xerr > 0 else 0.9
        )
        sample_kwargs: dict[str, object] = {
            "tune": tune,
            "target_accept": target_accept_,
            "return_inferencedata": True,
            **_pymc_sample_parallel_args(nuts_sampler),
        }
        if max_treedepth is not None:
            sample_kwargs["max_treedepth"] = max_treedepth
        trace = pm.sample(n_samples, **sample_kwargs)
        trace = _compute_sample_log_likelihood(trace)
    return process_trace(trace, params.keys(), ds)


# ------------------------------------------------------------------
# Helper: weighted statistics
# ------------------------------------------------------------------


def weighted_stats(
    values: Mapping[str, Sequence[float | None]],
    stderr: Mapping[str, Sequence[float | None]],
) -> dict[str, tuple[float, float]]:
    """Weighted mean and stderr for control priors."""
    results: dict[str, tuple[float, float]] = {}
    for sample in values:
        pairs = [
            (v, s)
            for v, s in zip(values[sample], stderr[sample], strict=True)
            if v is not None and s is not None
        ]
        if not pairs:
            msg = f"No valid (value, stderr) pairs for sample '{sample}'"
            raise ValueError(msg)
        x, se = zip(*pairs, strict=True)
        x_arr = np.array(x, dtype=float)
        se_arr = np.array(se, dtype=float)
        weighted_mean = np.average(x_arr, weights=1 / se_arr**2)
        weighted_stderr = np.sqrt(1 / np.sum(1 / se_arr**2))
        results[sample] = (weighted_mean, weighted_stderr)
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
    results: dict[str, FitResult[MiniT]],
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
    results : dict[str, FitResult[MiniT]]
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
            sigma = min(max(p.stderr * n_sd, 0.2), fallback_sigma)
            well_k[well] = (p.value, sigma)
    return well_k


def fit_binding_pymc_multi(  # noqa: C901, PLR0912, PLR0913, PLR0915, PLR0917
    results: Mapping[str, Dataset | FitResult[MiniT]],
    scheme: PlateScheme,
    n_sd: float = 5.0,
    n_xerr: float = 1.0,
    n_samples: int = 2000,
    nuts_sampler: str = "default",
    *,
    noise_model: PlateNoiseModel | None = None,
    shared_alpha: bool = True,
    shared_gain: bool = False,
    n_tune: int | None = None,
    target_accept: float | None = None,
    max_treedepth: int | None = None,
    x_error_model: Literal[
        "deterministic", "per_well", "hierarchical_per_well"
    ] = "deterministic",
    acid_drop_between_sigma: float = 0.005,
    ctr_free_k: bool = False,
    sample_ppc: bool = False,
    robust: bool = False,
    floor_mode: NoiseParamMode | None = None,
    gain_mode: NoiseParamMode | None = None,
    alpha_mode: NoiseParamMode | None = None,
    learn_ye_mags: bool = False,
    min_x_step: float = 0.2,
) -> MultiFitResult:
    """Multi-well PyMC with shared K per control group and per-label noise.

    Parameters
    ----------
    results : Mapping[str, Dataset | FitResult[MiniT]]
        Per-well datasets or initial fit results. Raw datasets are first fitted
        with :func:`fit_binding_glob` to seed the Bayesian model.
    scheme : PlateScheme
        Plate scheme defining control groups for shared-K priors.
    n_sd : float
        Prior width multiplier for per-well S0/S1 parameters.
    n_xerr : float
        Scaling factor applied to x-value uncertainties.
    n_samples : int
        Number of MCMC posterior samples per chain.
    nuts_sampler : str
        NUTS sampler backend (``"default"``, ``"blackjax"``, ``"numpyro"``,
        ``"nutpie"``).
    noise_model : PlateNoiseModel | None
        Noise model specification.  ``None`` (default) uses per-label
        ``ye_mag_{lbl}`` HalfNormal to scale existing ``y_err``.  Pass a
        ``PlateNoiseModel`` to infer floor, gain, and alpha from the full
        heteroscedastic noise model.
    shared_alpha : bool
        If ``True`` (default), use a single ``rel_error`` variable for all
        labels (comprehensive model).  If ``False``, use per-label
        ``rel_error_{lbl}`` (proportional model).  Only used when
        *noise_model* is provided.
    shared_gain : bool
        If ``True``, use a single ``gain`` variable for all labels.
        If ``False`` (default), use per-label ``gain_{lbl}``.  Only used
        when *noise_model* is provided and gain terms are present.
    n_tune : int | None
        Number of tuning steps for MCMC. If None, defaults to n_samples // 2.
    target_accept : float | None
        NUTS target acceptance probability. If ``None`` (default), 0.95 is
        used when *n_xerr* > 0 and 0.9 otherwise.
    max_treedepth : int | None
        Maximum tree depth for NUTS sampler. If ``None`` (default), PyMC's
        default is used.
    x_error_model : Literal["deterministic", "per_well", "hierarchical_per_well"]
        Model for x-error propagation. ``"deterministic"`` uses one shared
        ``x_true`` across all wells. ``"per_well"`` gives each well its own
        independent ``x_step`` (shared ``x_start``, per-well cumulative
        additions constrained by ``min_x_step``).
        ``"hierarchical_per_well"`` uses an acid-addition formulation:
        shared ``acid_drop_global`` per step (uncertainty from quadrature
        sum of adjacent ``x_errc``), per-well ``acid_drop_well`` deviating
        at fixed ``acid_drop_between_sigma`` scale (not inferred).
    acid_drop_between_sigma : float
        Fixed between-well scale for the ``acid_drop`` variation used by
        ``"hierarchical_per_well"`` (not inferred — set to the experimental
        tolerance).  0.005 (default) is very tight — suitable when all wells
        receive identical 2 uL additions.  Increase to 0.01--0.02 for larger
        pipetting/buffer differences.
    ctr_free_k : bool
        If True, each CTR replicate well gets its own independent flat K prior
        ``Normal(group_mean, 0.2)`` — identical to UNK well treatment, no
        hierarchical shrinkage.  The spread of K posteriors across replicates
        then quantifies between-replicate accuracy.  If False (default), all
        replicates of the same CTR share a single K.
    sample_ppc : bool
        If True, generates posterior predictive samples and adds them to the
        returned InferenceData object. Needed for `plot_ppc_well`.
    robust : bool
        If True, use StudentT likelihood (nu=3) for robust regression instead
        of Normal.
    floor_mode : NoiseParamMode | None
        How to treat the floor parameter.  ``None`` (default) resolves to
        ``"centered"`` when every input is already a ``FitResult`` and to
        ``"free"`` when any raw ``Dataset`` is supplied.
    gain_mode : NoiseParamMode | None
        How to treat the gain parameter.  ``None`` follows the same input-based
        rule as *floor_mode*.
    alpha_mode : NoiseParamMode | None
        How to treat the alpha (rel_error) parameter.  ``None`` follows the
        same input-based rule as *floor_mode*.
    learn_ye_mags : bool
        If ``True``, learn per-label scaling factors (``ye_mag_{lbl}``) even
        when a full *noise_model* is provided.
    min_x_step : float
        Minimum inferred change in ``x`` between consecutive titration steps
        when latent-x modeling is enabled.

    Returns
    -------
    MultiFitResult
        Shared PyMC trace together with reconstructed per-well fit results.

    Raises
    ------
    ValueError
        If no valid dataset is found in results.
    """
    fit_results, prefer_centered = _normalize_fit_inputs(results)
    ds = next((r.dataset for r in fit_results.values() if r.dataset), None)
    if ds is None:
        msg = "No valid dataset found in results."
        raise ValueError(msg)
    xc = next(iter(ds.values())).xc
    x_errc = next(iter(ds.values())).x_errc
    labels = list(ds.keys())
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
    {key: i for i, key in enumerate(wells_list)}
    n_wells = len(wells_list)
    n_steps = len(xc)

    coords: dict[str, list[int] | list[str]] = {
        "well": wells_list,
        "step": list(range(n_steps)),
        "step_diff": list(range(n_steps - 1)),
    }

    with pm.Model(coords=coords):
        if x_error_model == "hierarchical_per_well" and n_xerr > 0:
            if not np.all(np.diff(xc) < 0):
                msg = (
                    "hierarchical_per_well acid-addition model expects "
                    "decreasing pH values."
                )
                raise ValueError(msg)

            nominal_drop = -np.diff(xc)  # positive pH drops

            x_sigma = np.maximum(x_errc * n_xerr, 1e-6)

            # Uncertainty of a measured pH difference (quadrature sum).
            drop_meas_sigma = np.sqrt(x_sigma[:-1] ** 2 + x_sigma[1:] ** 2)

            lower_nsd = 2.5
            min_acid_drop = 1e-6
            min_drops = np.maximum(
                nominal_drop - lower_nsd * drop_meas_sigma,
                min_acid_drop,
            )

            x_start = pm.Normal(
                "x_start",
                mu=xc[0],
                sigma=max(x_sigma[0], 1e-6),
            )

            acid_drop_global = pm.TruncatedNormal(
                "acid_drop_global",
                mu=nominal_drop,
                sigma=np.maximum(drop_meas_sigma, 1e-6),
                lower=min_drops,
                dims="step_diff",
            )

            # Fixed between-well scale — do not sample a variance parameter;
            # the centered funnel destroys sampler performance.
            acid_drop_well = pm.TruncatedNormal(
                "acid_drop_well",
                mu=acid_drop_global[:, None],
                sigma=max(acid_drop_between_sigma, 1e-6),
                lower=min_drops[:, None],
                shape=(n_steps - 1, n_wells),
                dims=("step_diff", "well"),
            )

            cumulative_drop = pm.math.cumsum(acid_drop_well, axis=0)

            # Use ones_like on a slice to inherit the well-dimension shape.
            start_row = pt.ones_like(acid_drop_well[:1, :]) * x_start  # type: ignore[no-untyped-call]
            x_matrix = pm.math.concatenate(
                [start_row, x_start - cumulative_drop], axis=0
            )

            x_w_all = pm.Deterministic(
                "x_per_well",
                x_matrix,
                dims=("step", "well"),
            )

        elif x_error_model == "per_well" and n_xerr > 0:
            x_errc_scaled = np.maximum(x_errc * n_xerr, 1e-6)
            x_errc_cumulative = np.maximum.accumulate(x_errc_scaled)
            step_sigmas = np.empty_like(x_errc_cumulative)
            step_sigmas[0] = x_errc_cumulative[0]
            if len(xc) > 1:
                step_var = np.maximum(
                    x_errc_cumulative[1:] ** 2 - x_errc_cumulative[:-1] ** 2,
                    1e-12,
                )
                step_sigmas[1:] = np.sqrt(step_var)
            direction = 1.0 if np.all(np.diff(xc) >= 0.0) else -1.0
            step_nominal = direction * np.diff(xc)
            lower_nsd = 2.5
            min_steps = np.maximum(
                step_nominal - lower_nsd * step_sigmas[1:],
                min_x_step,
            )
            x_start = pm.Normal("x_start", mu=xc[0], sigma=step_sigmas[0])
            x_step = pm.TruncatedNormal(
                "x_step",
                mu=step_nominal[:, None],
                sigma=np.maximum(step_sigmas[1:], 1e-6)[:, None],
                lower=min_steps[:, None],
                shape=(n_steps - 1, n_wells),
                dims=("step_diff", "well"),
            )
            x_cumsum = pm.math.cumsum(x_step, axis=0)
            x_offsets = pm.math.concatenate([pt.zeros((1, n_wells)), x_cumsum], axis=0)
            x_w_all = pm.Deterministic(
                "x_per_well",
                x_start + direction * x_offsets,
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

        k_params, k_replicate = _build_ctr_k_params(
            scheme,
            ctr_ks,
            active_wells,
            ctr_free_k=ctr_free_k,
            well_k_init=_well_k_init_from_results(fit_results, scheme, n_sd),
        )

        # Collect vectorized K
        k_list = []
        for key in wells_list:
            ctr_name = next(
                (name for name, wells in scheme.names.items() if key in wells), ""
            )
            r = fit_results[key]
            if r.result is None:
                msg = f"Fit result for well {key} is missing."
                raise ValueError(msg)
            # We need to access the pars dict to get the UNK K's, but we don't have it yet.
            # Let's just resolve K here directly.
            if ctr_free_k:
                k_well = k_replicate.get(key)
                if k_well is None:
                    # It's an UNK well, create its K prior
                    p = r.result.params["K"]
                    sigma = max(p.stderr * n_sd, 1e-3) if p.stderr else 1e-3
                    k_well = pm.Normal(f"K_{key}", mu=p.value, sigma=sigma)
            elif ctr_name:
                k_well = k_params[ctr_name]
            else:
                # It's an UNK well, create its K prior
                p = r.result.params["K"]
                sigma = max(p.stderr * n_sd, 1e-3) if p.stderr else 1e-3
                k_well = pm.Normal(f"K_{key}", mu=p.value, sigma=sigma)
            k_list.append(k_well)
        k_all = pt.stack(k_list)  # (n_wells,)

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
                sig_s0.append(max(p_s0.stderr * n_sd, 1e-3) if p_s0.stderr else 1e-3)
                mu_s1.append(p_s1.value)
                sig_s1.append(max(p_s1.stderr * n_sd, 1e-3) if p_s1.stderr else 1e-3)
            s0_vars[lbl] = pm.Normal(
                f"S0_{lbl}", mu=np.array(mu_s0), sigma=np.array(sig_s0), dims="well"
            )
            s1_vars[lbl] = pm.Normal(
                f"S1_{lbl}", mu=np.array(mu_s1), sigma=np.array(sig_s1), dims="well"
            )

        if noise_model is not None:
            noise_priors = build_pymc_noise_priors(
                noise_model,
                shared_alpha=shared_alpha,
                shared_gain=shared_gain,
                floor_mode=floor_mode,
                gain_mode=gain_mode,
                alpha_mode=alpha_mode,
            )
            if learn_ye_mags:
                ye_mags = {
                    lbl: pm.HalfNormal(f"ye_mag_{lbl}", sigma=1.0) for lbl in labels
                }
        else:
            noise_priors = {}
            ye_mags = {lbl: pm.HalfNormal(f"ye_mag_{lbl}", sigma=1.0) for lbl in labels}

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
            y_err_vec = y_err_full[mask_lbl]

            if noise_model is not None:
                # Heteroscedastic noise model
                noise_var = get_pymc_variance(
                    y_model_all, lbl, noise_model, noise_priors
                )
                sigma_obs_all = pm_math.sqrt(noise_var)
                if learn_ye_mags:
                    sigma_obs_all = ye_mags[lbl] * sigma_obs_all

                sigma_obs_all_det = pm.Deterministic(
                    f"sigma_obs_{lbl}", sigma_obs_all, dims=("step", "well")
                )
                sigma_vec = sigma_obs_all_det[mask_lbl]
            else:
                # Homoscedastic (scaled) noise model
                sigma_vec = ye_mags[lbl] * y_err_vec

            if robust:
                pm.StudentT(
                    f"y_likelihood_{lbl}",
                    nu=3.0,
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
        sample_kwargs: dict[str, object] = {
            "tune": tune_steps,
            "target_accept": target_accept_,
            "return_inferencedata": True,
            **_pymc_sample_parallel_args(nuts_sampler),
        }
        if max_treedepth is not None:
            sample_kwargs["max_treedepth"] = max_treedepth
        trace: xr.DataTree = pm.sample(n_samples, **sample_kwargs)

        if sample_ppc:
            pm.sample_posterior_predictive(trace, extend_inferencedata=True)

        trace = _compute_sample_log_likelihood(trace)

    return MultiFitResult(
        trace=trace,
        results=_per_well_fit_results_from_trace(
            trace,
            fit_results,
            scheme,
            x_error_model=x_error_model,
        ),
    )
