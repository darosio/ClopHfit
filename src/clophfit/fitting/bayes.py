"""Bayesian (PyMC) fitting utilities and pipelines."""

from __future__ import annotations

import copy
import os
import typing
import warnings
from collections.abc import Mapping, Sequence
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
from scipy import optimize

from clophfit.fitting.models import binding_1site
from clophfit.fitting.plotting import PlotParameters, plot_fit

from .core import N_BOOT, fit_binding_glob  # local to avoid circular import
from .data_structures import DataArray, Dataset, FitResult, MiniT, _Result

if typing.TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from clophfit.clophfit_types import ArrayF, FloatFunc
    from clophfit.prtecan import PlateScheme


__all__ = [
    "fit_binding_pymc",
    "fit_binding_pymc_compare",
    "fit_binding_pymc_multi",
    "fit_binding_pymc_odr",
    "process_trace",
]


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


def create_x_true(
    xc: ArrayF, x_errc: ArrayF, n_xerr: float, lower_nsd: float = 2.5
) -> ArrayF | pm.Deterministic:
    """Create latent variables for x-values with uncertainty.

    Returns a PyMC Deterministic variable when in a Model context with uncertainty,
    or a numpy array when there's no uncertainty or no active Model.
    """
    if n_xerr > 0 and np.any(x_errc > 0):
        x_errc_scaled = x_errc * n_xerr
        xd = -np.diff(xc)
        xd_err = np.sqrt(x_errc_scaled[:-1] ** 2 + x_errc_scaled[1:] ** 2)
        lower = xd.min() - lower_nsd * xd_err[np.argmin(xd)]
        # MAYBE: It was logger.info(f"min pH distance: {lower}")
        x_diff = pm.TruncatedNormal(
            "x_diff", mu=xd, sigma=xd_err, lower=lower, shape=len(xc) - 1
        )
        x_start = pm.Normal("x_start", mu=xc[0], sigma=x_errc_scaled[0])
        x_cumsum = pm.math.cumsum(x_diff)
        return pm.Deterministic(
            "x_true", pm.math.concatenate([[x_start], x_start - x_cumsum])
        )
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


# TODO:
# 🧪 Test posterior integrity (e.g., credible intervals contain true Kd)
# 🧱 Replace repetitive for lbl in ds.items() logic using helper functions
# 🔁 Use pm.MutableData (in newer PyMC versions) to avoid model recompilation
def rename_keys(data: dict[str, typing.Any]) -> dict[str, typing.Any]:
    """Rename dictionary keys coming from multi-trace into base names."""
    renamed_dict: dict[str, typing.Any] = {}
    for key, value in data.items():
        if key.startswith("K_"):
            new_key = "K"
        elif key.rfind("_") > 1:
            new_key = key[: key.rfind("_")]
        else:
            new_key = key
        renamed_dict[new_key] = value
    return renamed_dict


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


def process_trace(
    trace: xr.DataTree, p_names: typing.KeysView[str], ds: Dataset, n_xerr: float
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
    n_xerr : float
        Scaling factor for `x_errc`.

    Returns
    -------
    FitResult[xr.DataTree]
        The updated fit result with extracted parameter values and datasets.
        Residuals are WEIGHTED (weight * (obs - pred)) where weight = 1/y_err,
        computed using posterior mean parameter estimates.

    Raises
    ------
    TypeError
        If az.summary does not return a DataFrame.
    """
    # Extract summary statistics for parameters
    rdf = az.summary(trace)
    if not isinstance(rdf, pd.DataFrame):
        msg = "az.summary did not return a DataFrame"
        raise TypeError(msg)
    # Ensure numeric types (ArviZ 1.x might return strings)
    rdf = rdf.apply(pd.to_numeric, errors="coerce")
    rpars = Parameters()
    for name, row in rdf.iterrows():
        if name in p_names:
            _add_param_from_summary(rpars, str(name), row)
    # x_true and x_errc
    nxc, nx_errc = _extract_x_true_from_trace_df(rdf)
    if nxc.size > 0:
        for da in ds.values():
            da.xc = nxc  # Update x_true values in the dataset
            da.x_errc = nx_errc * n_xerr  # Scale the errors FIXME: n_xerr not needed
    # Scale y_errc if present
    try:
        mag = float(rdf.loc["ye_mag", "mean"])
    except Exception:  # noqa: BLE001
        mag = 1.0
    for da in ds.values():
        da.y_errc *= mag  # Scale y errors by the magnitude
    # Create figure
    fig = figure.Figure()
    ax = fig.add_subplot(111)
    # FIXME: multi need this renaming quite surely
    rename_keys(rpars)
    plot_fit(ax, ds, rpars, nboot=N_BOOT, pp=PlotParameters(ds.is_ph))

    # Compute weighted residuals from posterior mean predictions
    # Weighted residuals = (1/y_err) * (observed - predicted)
    # Use masked values (.x, .y, .y_err) for consistency
    residuals_list: list[np.ndarray] = []
    for lbl, da in ds.items():
        model = binding_1site(
            da.x,
            rpars["K"].value,
            rpars[f"S0_{lbl}"].value,
            rpars[f"S1_{lbl}"].value,
            is_ph=ds.is_ph,
        )
        raw_residuals = da.y - model
        # Weight by precision (1/y_err)
        if da.y_err.size > 0:
            weight = 1 / da.y_err
            weighted = weight * raw_residuals
        else:
            weighted = raw_residuals
        residuals_list.append(weighted)
    residuals = np.concatenate(residuals_list)

    # Return the fit result
    return FitResult(fig, _Result(rpars, residual=residuals), trace, ds)


def extract_fit(
    key: str,
    ctr: str,
    trace_df: pd.DataFrame,
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
    trace_df : pd.DataFrame
        ArviZ summary DataFrame (``fmt="wide"``) from the multi-well MCMC run.
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
    rpars = Parameters()
    rdf = trace_df[trace_df.index.str.endswith(key)]
    for name, row in rdf.iterrows():
        extracted_name = str(name).replace(f"_{key}", "")
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
    for da in ds.values():
        da.xc = nxc
        da.x_errc = nx_errc
    # Create figure
    fig = figure.Figure()
    ax = fig.add_subplot(111)
    plot_fit(ax, ds, rpars, nboot=N_BOOT, pp=PlotParameters(ds.is_ph))

    # Compute weighted residuals from posterior mean predictions
    # Weighted residuals = (1/y_err) * (observed - predicted)
    # Use masked values (.x, .y, .y_err) for consistency
    residuals_list: list[np.ndarray] = []
    for lbl, da in ds.items():
        model = binding_1site(
            da.x,
            rpars["K"].value,
            rpars[f"S0_{lbl}"].value,
            rpars[f"S1_{lbl}"].value,
            is_ph=ds.is_ph,
        )
        raw_residuals = da.y - model
        # Weight by precision (1/y_err)
        if da.y_err.size > 0:
            weight = 1 / da.y_err
            weighted = weight * raw_residuals
        else:
            weighted = raw_residuals
        residuals_list.append(weighted)
    residuals = np.concatenate(residuals_list)

    return FitResult(fig, _Result(rpars, residual=residuals), xr.DataTree(), ds)


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
    # Ensure numeric types (ArviZ 1.x might return strings)
    trace_df = trace_df.apply(pd.to_numeric, errors="coerce")
    nxc: list[float] = []
    nx_errc: list[float] = []
    for name, row in trace_df.iterrows():
        if isinstance(name, str) and name.startswith("x_true"):
            nxc.append(row["mean"])
            nx_errc.append(row["sd"])
    return np.array(nxc), np.array(nx_errc)


def _extract_x_per_well_from_trace_df(
    trace_df: pd.DataFrame,
    well_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract per-well x values for *well_key* from an xrw trace summary.

    ArviZ names the ``x_per_well`` deterministic (with dims ``step`` x ``well``)
    as ``x_per_well[{step}, {well}]``.  This function collects those rows for a
    specific well and returns them sorted by step index.

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
    """
    # Ensure numeric types (ArviZ 1.x might return strings)
    trace_df = trace_df.apply(pd.to_numeric, errors="coerce")
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


def x_true_from_trace_df(trace_df: pd.DataFrame) -> DataArray:
    """Extract x_true from an ArviZ summary DataFrame."""
    nxc, nx_errc = _extract_x_true_from_trace_df(trace_df)
    return DataArray(xc=nxc, yc=np.ones_like(nxc), x_errc=nx_errc)


def fit_binding_pymc(  # noqa: PLR0913,PLR0917
    ds_or_fr: Dataset | FitResult[MiniT],
    n_sd: float = 10.0,
    n_xerr: float = 1.0,
    ye_scaling: float = 1.0,
    n_samples: int = 2000,
    nuts_sampler: str = "default",
    *,
    error_model: str = "shared",
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
    ye_scaling : float
        Scaling factor for y-error magnitude prior.
    n_samples : int
        Number of MCMC samples.
    nuts_sampler : str
        NUTS sampler backend: ``"default"`` (PyMC C/pytensor), ``"blackjax"``,
        ``"numpyro"``, or ``"nutpie"``.
    error_model : str
        Error model to use: ``"shared"`` (single ye_mag) or ``"separate"`` (per-label ye_mag).

    Returns
    -------
    FitResult[xr.DataTree]
        Bayesian fitting results.
    """
    # Handle both Dataset and FitResult inputs
    fr = fit_binding_glob(ds_or_fr) if isinstance(ds_or_fr, Dataset) else ds_or_fr

    if fr.result is None or fr.dataset is None:
        return FitResult()
    params = fr.result.params
    ds = copy.deepcopy(fr.dataset)
    xc = next(iter(ds.values())).xc  # # TODO: move up out
    x_errc = next(iter(ds.values())).x_errc
    with pm.Model():
        pars = create_parameter_priors(params, n_sd)
        x_true = create_x_true(xc, x_errc, n_xerr)
        # Add likelihoods for each dataset
        if error_model == "separate":
            ye_mag_dict: dict[str, pm.Distribution] = {
                lbl: pm.HalfNormal(f"ye_mag_{lbl}", sigma=ye_scaling * 10.0)
                for lbl in ds
            }
        else:
            ye_mag_shared = pm.HalfNormal("ye_mag", sigma=ye_scaling)

        for lbl, da in ds.items():
            y_model = binding_1site(
                x_true, pars["K"], pars[f"S0_{lbl}"], pars[f"S1_{lbl}"], is_ph=ds.is_ph
            )
            sigma = (
                ye_mag_dict[lbl] * np.ones_like(da.y_err)
                if error_model == "separate"
                else ye_mag_shared * da.y_err
            )
            pm.Normal(
                f"y_likelihood_{lbl}",
                mu=y_model[da.mask],
                sigma=sigma,
                observed=da.y,
            )
        # Inference
        tune = n_samples // 2
        trace = pm.sample(
            n_samples,
            tune=tune,
            target_accept=0.9,
            return_inferencedata=True,
            **_pymc_sample_parallel_args(nuts_sampler),
        )
        trace = _compute_sample_log_likelihood(trace)
    return process_trace(trace, params.keys(), ds, n_xerr)


def fit_binding_pymc_compare(  # noqa: PLR0913
    fr: FitResult[MiniT],
    buffer_sd: dict[str, float],
    *,
    learn_separate_y_mag: bool = False,
    n_sd: float = 10.0,
    n_xerr: float = 1.0,
    n_samples: int = 2000,
) -> xr.DataTree:
    """
    Fits a Bayesian binding model with two different noise models for comparison.

    Parameters
    ----------
    fr : FitResult[MiniT]
        The fit result from a previous run, providing initial parameters and dataset.
    buffer_sd : dict[str, float]
        bg_err
    learn_separate_y_mag : bool
        If True, learns a unique noise scaling factor for each dataset label.
        If False, learns a single scaling factor for all pre-weighted data.
    n_sd : float
        Prior width for parameters in create_parameter_priors.
    n_xerr : float
        Scaling factor for x_errc in create_x_true.
    n_samples : int
        Number of MCMC samples to draw.

    Returns
    -------
    xr.DataTree
        The posterior samples from PyMC for the specified noise model.
    """
    """
    if fr.result is None or fr.dataset is None:
        msg = "Input FitResult object must contain a result and a dataset."
        raise ValueError(msg)
    """
    if fr.result:
        params = fr.result.params
    if fr.dataset:
        ds = copy.deepcopy(fr.dataset)

    # Use the first dataset's x values. Assumes all datasets have same x points.
    xc = next(iter(ds.values())).xc
    x_errc = next(iter(ds.values())).x_errc

    with pm.Model():
        # Create priors for all parameters (K, S0_y1, S1_y1, etc.)
        pars = create_parameter_priors(params, n_sd)
        # Model the x-values with their uncertainties
        x_true = create_x_true(xc, x_errc, n_xerr)
        # ---------------------------------------------------------------------
        # Core conditional logic for the noise model

        if learn_separate_y_mag:
            # Model 1: Learn a unique noise scaling factor for each label
            # This is robust when you don't trust the initial y_err values
            ye_mag: dict[str | int, float] = {}
            true_buffer = {}
            for lbl, da in ds.items():
                ye_mag[lbl] = pm.HalfNormal(f"ye_mag_{lbl}", sigma=da.y_err.mean())
                y_model = binding_1site(
                    x_true,
                    pars["K"],
                    pars[f"S0_{lbl}"],
                    pars[f"S1_{lbl}"],
                    is_ph=ds.is_ph,
                )
                true_buffer[lbl] = pm.Normal(
                    f"true_buffer_{lbl}", mu=0, sigma=da.y_err.mean()
                )
                sigma = 10 * pm.math.sqrt(
                    (ye_mag[lbl] * np.ones_like(da.y_err)) ** 2 + buffer_sd[lbl] ** 2
                    # Alternatively use: ye_mag[lbl] ** 2 * da.y + buffer_sd[lbl] ** 2
                )

                pm.Normal(
                    f"y_likelihood_{lbl}",
                    mu=y_model[da.mask] + true_buffer[lbl],
                    sigma=sigma,
                    # Noise is learned from scratch and shot noise model
                    # Alternatively use: * np.ones_like(da.y_err),# Noise is learned from scratch
                    observed=da.y,
                )
        else:
            # Model 2: Learn a single noise scaling factor for all data
            # This is appropriate when you trust the relative y_err values
            ye_mag0 = pm.HalfNormal("ye_mag", sigma=10.0)
            for lbl, da in ds.items():
                y_model = binding_1site(
                    x_true,
                    pars["K"],
                    pars[f"S0_{lbl}"],
                    pars[f"S1_{lbl}"],
                    is_ph=ds.is_ph,
                )
                pm.Normal(
                    f"y_likelihood_{lbl}",
                    mu=y_model[da.mask],
                    sigma=ye_mag0 * da.y_err,  # Apply a single scaling factor
                    # Alternatively use:  sigma=da.y_err,  # Apply a single scaling factor
                    observed=da.y,
                )
        # ---------------------------------------------------------------------
        # Run MCMC sampling
        trace: xr.DataTree = pm.sample(
            n_samples,
            return_inferencedata=True,
            target_accept=0.9,
            **_pymc_sample_parallel_args(),
        )
        return _compute_sample_log_likelihood(trace)


def closest_point_on_curve(f: FloatFunc, x_obs: float, y_obs: float) -> float:
    """Find the closest point on the model curve."""

    def objective(x_prime: float) -> float:
        return (x_obs - x_prime) ** 2 + (y_obs - f(x_prime)) ** 2

    result = optimize.minimize_scalar(objective)
    return float(result.x)


def fit_binding_pymc_odr(
    fr: FitResult[MiniT],
    n_sd: float = 10.0,
    xe_scaling: float = 1.0,
    ye_scaling: float = 10.0,
    n_samples: int = 2000,
) -> xr.DataTree | pm.backends.base.MultiTrace:
    """Analyze using deprecated Bayesian ODR-like modeling of x and y errors."""
    warnings.warn(
        "fit_binding_pymc_odr is deprecated and may not work with recent PyMC versions.",
        DeprecationWarning,
        stacklevel=2,
    )
    if fr.result is None or fr.dataset is None:
        return xr.DataTree()  # FitResult()
    params = fr.result.params
    ds = copy.deepcopy(fr.dataset)
    xc = next(iter(ds.values())).xc
    x_errc = next(iter(ds.values())).x_errc
    with pm.Model() as _:
        pars = create_parameter_priors(params, n_sd)
        # Add likelihoods for each dataset
        ye_mag = pm.HalfNormal("ye_mag", sigma=ye_scaling)
        xe_mag = pm.HalfNormal("xe_mag", sigma=xe_scaling)

        for lbl, da in ds.items():

            def _y_model(x: float, lbl: str = lbl) -> float:
                return binding_1site(
                    x, pars["K"], pars[f"S0_{lbl}"], pars[f"S1_{lbl}"], is_ph=ds.is_ph
                )

            # Define symbolic closest points using PyMC-compatible operations
            x_prime = pm.Deterministic(
                f"x_prime_{lbl}",
                pm.math.stack(  # noqa: PD013
                    [
                        closest_point_on_curve(
                            lambda x_val: _y_model(x_val).eval(),  # type: ignore[attr-defined]
                            x_obs,
                            y_obs,
                        )
                        for x_obs, y_obs in zip(xc, da.y, strict=True)
                    ]
                ),
            )

            y_prime = pm.Deterministic(
                f"y_prime_{lbl}",
                pm.math.stack([_y_model(x) for x in x_prime.eval()]),  # noqa: PD013
            )
            y_model = pm.Deterministic(
                f"y_model_{lbl}",
                pm.math.stack([_y_model(x) for x in xc]),  # noqa: PD013
            )
            # TODO:  y_model = as_tensor_variable([_y_model(x) for x in xc])

            mask = as_tensor_variable(da.mask)
            # Orthogonal distance likelihood
            distances = ((x_prime - xc) / (xe_mag * x_errc)) ** 2 + (
                (y_prime - y_model) / (ye_mag * da.y_err)
            ) ** 2
            pm.Normal(
                f"orthogonal_likelihood_{lbl}",
                mu=distances[mask],
                sigma=1,
                observed=np.zeros(len(distances[mask].eval())),
            )
        # Inference
        return pm.sample(
            n_samples, return_inferencedata=True, **_pymc_sample_parallel_args()
        )
    # TODO:  return process_trace(trace, params.keys(), ds, 0)


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


def fit_binding_pymc_multi(  # noqa: C901,PLR0912,PLR0913,PLR0915,PLR0917
    results: dict[str, FitResult[MiniT]],
    scheme: PlateScheme,
    n_sd: float = 5.0,
    n_xerr: float = 1.0,
    ye_scaling: float = 1.0,
    n_samples: int = 2000,
    nuts_sampler: str = "default",
    *,
    x_error_model: Literal["deterministic", "random_walk"] = "deterministic",
    sigma_pip_prior: float = 0.02,
    ctr_free_k: bool = False,
    bg_noise: dict[int, ArrayF] | None = None,
    sample_ppc: bool = False,
    infer_gain: bool = False,
    robust: bool = False,
) -> xr.DataTree:
    """Multi-well PyMC with shared K per control group and per-label noise.

    Parameters
    ----------
    results : dict[str, FitResult[MiniT]]
        Per-well initial fit results.
    scheme : PlateScheme
        Plate scheme defining control groups for shared-K priors.
    n_sd : float
        Prior width multiplier for per-well S0/S1 parameters.
    n_xerr : float
        Scaling factor applied to x-value uncertainties.
    ye_scaling : float
        HalfNormal sigma for the per-label y-error scaling factor.
    n_samples : int
        Number of MCMC posterior samples per chain.
    nuts_sampler : str
        NUTS sampler backend (``"default"``, ``"blackjax"``, ``"numpyro"``,
        ``"nutpie"``).
    x_error_model : Literal["deterministic", "random_walk"]
        Model for x-error propagation (default: "deterministic").
    sigma_pip_prior : float
        Prior for random_walk sigma pipette error parameter.
    ctr_free_k : bool
        If True, each CTR replicate well gets its own independent flat K prior
        ``Normal(group_mean, 0.2)`` — identical to UNK well treatment, no
        hierarchical shrinkage.  The spread of K posteriors across replicates
        then quantifies between-replicate accuracy.  If False (default), all
        replicates of the same CTR share a single K.
    bg_noise : dict[int, ArrayF] | None
        Background noise for each signal band. If provided, uses heteroscedastic
        noise model combining buffer and signal, ignoring `ye_scaling`.
    sample_ppc : bool
        If True, generates posterior predictive samples and adds them to the
        returned InferenceData object. Needed for `plot_ppc_well`.
    infer_gain : bool
        If True (and ``bg_noise`` is provided), jointly infer per-label Poisson
        gain and a single shared relative error. If False (default), infer
        per-label relative error with no gain term.
    robust : bool
        If True, use StudentT likelihood (nu=3) for robust regression instead of Normal.

    Returns
    -------
    xr.DataTree
        The PyMC posterior trace.

    Raises
    ------
    ValueError
        If no valid dataset is found in results.
    """
    # FIXME: pytensor.config.floatX = "float32"  # type: ignore[attr-defined]
    ds = next((r.dataset for r in results.values() if r.dataset), None)
    if ds is None:
        msg = "No valid dataset found in results."
        raise ValueError(msg)
    xc = next(iter(ds.values())).xc
    x_errc = next(iter(ds.values())).x_errc * n_xerr
    labels = list(ds.keys())
    values: dict[str, list[float | None]] = {}
    stderr: dict[str, list[float | None]] = {}

    for name, wells in scheme.names.items():
        values[name] = [
            v.result.params["K"].value
            for well, v in results.items()
            if v.result and well in wells
        ]
        stderr[name] = [
            v.result.params["K"].stderr
            for well, v in results.items()
            if v.result and well in wells
        ]
    ctr_ks = weighted_stats(values, stderr)
    active_wells = {key for key, r in results.items() if r.result and r.dataset}
    wells_list = [
        key for key in results if results[key].result and results[key].dataset
    ]
    well_idx = {key: i for i, key in enumerate(wells_list)}
    n_wells = len(wells_list)
    n_steps = len(xc)

    coords: dict[str, list[int] | list[str]] = {
        "well": wells_list,
        "step": list(range(n_steps)),
        "step_diff": list(range(n_steps - 1)),
    }

    with pm.Model(coords=coords):
        x_true = create_x_true(xc, x_errc, n_xerr)
        if x_error_model == "random_walk":
            sigma_pip = pm.HalfNormal("sigma_pip", sigma=sigma_pip_prior)
            z_pip = pm.Normal(
                "z_pip", 0, 1, shape=(n_steps - 1, n_wells), dims=("step_diff", "well")
            )
            cum_dev = pm.math.cumsum(sigma_pip * z_pip, axis=0)
            x_dev = pm.math.concatenate([pt.zeros((1, n_wells)), cum_dev], axis=0)
            x_per_well = pm.Deterministic(
                "x_per_well",
                x_true[:, None] + x_dev,
                dims=("step", "well"),
            )

        k_params, k_replicate = _build_ctr_k_params(
            scheme,
            ctr_ks,
            active_wells,
            ctr_free_k=ctr_free_k,
            well_k_init=_well_k_init_from_results(results, scheme, n_sd),
        )

        if bg_noise is None:
            ye_mag: dict[str, pm.Distribution] = {
                label: pm.HalfNormal(f"ye_mag_{label}", sigma=ye_scaling)
                for label in labels
            }
        else:
            est_sigma = {
                lbl: float(np.sqrt(np.mean(np.array(bg_noise[i]) ** 2)))
                for i, lbl in enumerate(labels, start=1)
            }

            # Prior width matched to degrees of freedom of the empirical noise estimate
            n_pts = len(bg_noise[1]) if bg_noise else 1
            n_buf = len(scheme.buffer) if scheme.buffer else 1
            dof = max(1, n_pts * n_buf - 2)
            rel_sigma = float(np.clip(1.0 / np.sqrt(2 * dof), 0.05, 0.5))

            floor = {
                lbl: pm.Normal(
                    f"floor_{lbl}", mu=est_sigma[lbl], sigma=rel_sigma * est_sigma[lbl]
                )
                for lbl in labels
            }
            floor_sq = {lbl: floor[lbl] ** 2 for lbl in labels}

            if infer_gain:
                gain_rv: dict[str, pm.Distribution] = {
                    lbl: pm.Exponential(f"gain_{lbl}", 1.0) for lbl in labels
                }
                rel_error_common: pm.Distribution = pm.HalfNormal(
                    "rel_error", sigma=0.04
                )
            else:
                rel_error: dict[str, pm.Distribution] = {
                    lbl: pm.HalfNormal(f"rel_error{lbl}", sigma=0.2) for lbl in labels
                }

        for key, r in results.items():
            if r.result and r.dataset:
                ds_well = r.dataset
                ctr_name = next(
                    (name for name, wells in scheme.names.items() if key in wells), ""
                )
                pars = create_parameter_priors(r.result.params, n_sd, key, ctr_name)
                K = _resolve_well_k(  # noqa: N806
                    key, ctr_name, pars, k_params, k_replicate, ctr_free_k=ctr_free_k
                )

                if x_error_model == "random_walk":
                    w_idx = well_idx[key]
                    x_w = x_per_well[:, w_idx]
                else:
                    x_w = x_true

                for _i, (lbl, da) in enumerate(ds_well.items(), start=1):
                    y_model = binding_1site(
                        x_w,
                        K,
                        pars[f"S0_{lbl}_{key}"],
                        pars[f"S1_{lbl}_{key}"],
                        is_ph=ds_well.is_ph,
                    )

                    if bg_noise is None:
                        sigma_val = ye_mag[lbl] * da.y_err
                    else:
                        y_pos = pm_math.maximum(1e-6, y_model)
                        if infer_gain:
                            noise_var = (
                                floor_sq[lbl]
                                + gain_rv[lbl] * y_pos
                                + (rel_error_common * y_pos) ** 2
                            )
                        else:
                            noise_var = floor_sq[lbl] + (rel_error[lbl] * y_pos) ** 2
                        sigma_obs = pm.Deterministic(
                            f"sigma_obs_{lbl}_{key}",
                            pm_math.sqrt(noise_var),
                        )
                        sigma_val = sigma_obs[da.mask]

                    if robust:
                        pm.StudentT(
                            f"y_likelihood_{lbl}_{key}",
                            nu=3.0,
                            mu=y_model[da.mask],
                            sigma=sigma_val,
                            observed=da.y,
                        )
                    else:
                        pm.Normal(
                            f"y_likelihood_{lbl}_{key}",
                            mu=y_model[da.mask],
                            sigma=sigma_val,
                            observed=da.y,
                        )

        trace: xr.DataTree = pm.sample(
            n_samples,
            target_accept=0.9,
            return_inferencedata=True,
            **_pymc_sample_parallel_args(nuts_sampler),
        )

        if sample_ppc:
            pm.sample_posterior_predictive(trace, extend_inferencedata=True)

    return trace


# ------------------------------------------------------------------
# 2.2b  Noise-model helpers and multi-well fit with learned noise
# ------------------------------------------------------------------
