"""Bayesian (PyMC) fitting utilities and pipelines."""

from __future__ import annotations

import copy
import os
import typing
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm  # type: ignore[import-untyped]
import pytensor.tensor as pt
import seaborn as sns  # type: ignore[import-untyped]
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
    "fit_binding_pymc_multi_noise",
    "fit_binding_pymc_odr",
    "fit_pymc_hierarchical",
    "plot_noise_vs_index",
    "plot_noise_vs_signal",
    "plot_ppc_well",
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
    trace: az.InferenceData, p_names: typing.KeysView[str], ds: Dataset, n_xerr: float
) -> FitResult[az.InferenceData]:
    """Process the trace to extract parameter estimates and update datasets.

    Parameters
    ----------
    trace : az.InferenceData
        The posterior samples from PyMC sampling.
    p_names: typing.KeysView[str]
        Parameter names.
    ds : Dataset
        The dataset containing titration data.
    n_xerr : float
        Scaling factor for `x_errc`.

    Returns
    -------
    FitResult[az.InferenceData]
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
        mag = float(rdf.loc["ye_mag", "mean"])  # type: ignore[arg-type]
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
) -> FitResult[az.InferenceData]:
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
    FitResult[az.InferenceData]
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

    return FitResult(fig, _Result(rpars, residual=residuals), az.InferenceData(), ds)


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
        ArviZ summary DataFrame from ``fit_binding_pymc_multi_noise_xrw``.
    well_key : str
        Well identifier used as the ``well`` coord (e.g. ``"A01"``).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Arrays of per-well x posterior means and standard deviations ordered
        by step.  Both arrays are empty if ``x_per_well`` rows are absent.
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
) -> FitResult[az.InferenceData]:
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
    FitResult[az.InferenceData]
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
            idata_kwargs={"log_likelihood": True},
            **_pymc_sample_parallel_args(nuts_sampler),
        )
    return process_trace(trace, params.keys(), ds, n_xerr)


def fit_binding_pymc2(
    ds_or_fr: Dataset | FitResult[MiniT],
    n_sd: float = 10.0,
    n_xerr: float = 1.0,
    n_samples: int = 2000,
) -> FitResult[az.InferenceData]:
    """Analyze multi-label titration datasets using deprecated PyMC with separate ye_mag per label."""
    warnings.warn(
        "fit_binding_pymc2 is deprecated. Use fit_binding_pymc(..., error_model='separate') instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return fit_binding_pymc(
        ds_or_fr,
        n_sd=n_sd,
        n_xerr=n_xerr,
        ye_scaling=1.0,
        n_samples=n_samples,
        error_model="separate",
    )


def fit_binding_pymc_compare(  # noqa: PLR0913
    fr: FitResult[MiniT],
    buffer_sd: dict[str, float],
    *,
    learn_separate_y_mag: bool = False,
    n_sd: float = 10.0,
    n_xerr: float = 1.0,
    n_samples: int = 2000,
) -> az.InferenceData:
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
    az.InferenceData
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
        trace: az.InferenceData = pm.sample(
            n_samples,
            return_inferencedata=True,
            target_accept=0.9,
            idata_kwargs={"log_likelihood": True},
            **_pymc_sample_parallel_args(),
        )
    return trace


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
) -> az.InferenceData | pm.backends.base.MultiTrace:
    """Analyze using deprecated Bayesian ODR-like modeling of x and y errors."""
    warnings.warn(
        "fit_binding_pymc_odr is deprecated and may not work with recent PyMC versions.",
        DeprecationWarning,
        stacklevel=2,
    )
    if fr.result is None or fr.dataset is None:
        return az.InferenceData()  # FitResult()
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


def fit_binding_pymc_multi(  # noqa: PLR0913,PLR0917
    results: dict[str, FitResult[MiniT]],
    scheme: PlateScheme,
    n_sd: float = 5.0,
    n_xerr: float = 1.0,
    ye_scaling: float = 1.0,
    n_samples: int = 2000,
    nuts_sampler: str = "default",
    *,
    ctr_free_k: bool = False,
    bg_err: dict[int, ArrayF] | None = None,
    sample_ppc: bool = False,
) -> az.InferenceData:
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
    ctr_free_k : bool
        If True, each CTR replicate well gets its own independent flat K prior
        ``Normal(group_mean, 0.2)`` — identical to UNK well treatment, no
        hierarchical shrinkage.  The spread of K posteriors across replicates
        then quantifies between-replicate accuracy.  If False (default), all
        replicates of the same CTR share a single K.
    bg_err : dict[int, ArrayF] | None
        Background error for each signal band. If provided, uses heteroscedastic
        noise model combining buffer and signal, ignoring `ye_scaling`.
    sample_ppc : bool
        If True, generates posterior predictive samples and adds them to the
        returned InferenceData object. Needed for `plot_ppc_well`.

    Returns
    -------
    az.InferenceData
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

    with pm.Model():
        x_true = create_x_true(xc, x_errc, n_xerr)
        k_params, k_replicate = _build_ctr_k_params(
            scheme,
            ctr_ks,
            active_wells,
            ctr_free_k=ctr_free_k,
            well_k_init=_well_k_init_from_results(results, scheme, n_sd),
        )

        if bg_err is None:
            ye_mag: dict[str, pm.Distribution] = {
                label: pm.HalfNormal(f"ye_mag_{label}", sigma=ye_scaling)
                for label in labels
            }
        else:
            gain: dict[str, pm.Distribution] = {
                lbl: pm.Exponential(f"gain_{lbl}", 1.0) for lbl in labels
            }
            alpha_sq: dict[str, pm.Distribution] = {
                lbl: pm.LogNormal(f"S_{lbl}", mu=np.log(1e-5), sigma=2.0)
                for lbl in labels
            }
            floor_sq = {
                lbl: float(np.mean(bg_err[i])) ** 2
                for i, lbl in enumerate(labels, start=1)
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

                for _i, (lbl, da) in enumerate(ds_well.items(), start=1):
                    y_model = binding_1site(
                        x_true,
                        K,
                        pars[f"S0_{lbl}_{key}"],
                        pars[f"S1_{lbl}_{key}"],
                        is_ph=ds_well.is_ph,
                    )

                    if bg_err is None:
                        pm.Normal(
                            f"y_likelihood_{lbl}_{key}",
                            mu=y_model[da.mask],
                            sigma=ye_mag[lbl] * da.y_err,
                            observed=da.y,
                        )
                    else:
                        sigma_obs = pm.Deterministic(
                            f"sigma_obs_{lbl}_{key}",
                            pm_math.sqrt(
                                floor_sq[lbl]
                                + gain[lbl] * pm_math.maximum(1e-6, y_model)
                                + alpha_sq[lbl] * y_model**2
                            ),
                        )
                        pm.Normal(
                            f"y_likelihood_{lbl}_{key}",
                            mu=y_model[da.mask],
                            sigma=sigma_obs[da.mask],
                            observed=da.y,
                        )

        trace: az.InferenceData = pm.sample(
            n_samples,
            target_accept=0.9,
            return_inferencedata=True,
            **_pymc_sample_parallel_args(nuts_sampler),
        )

        if sample_ppc:
            pm.sample_posterior_predictive(trace, extend_inferencedata=True)

    return trace


def fit_binding_pymc_multi2(  # noqa: PLR0913,PLR0917
    results: dict[str, FitResult[MiniT]],
    scheme: PlateScheme,
    bg_err: dict[int, ArrayF],
    n_sd: float = 5.0,
    n_xerr: float = 1.0,
    # Ponder this: ye_scaling: float = 1.0, # This parameter is no longer needed in the same way
    n_samples: int = 2000,
) -> az.InferenceData:
    """Analyze multi-well PyMC with heteroscedastic noise (deprecated)."""
    warnings.warn(
        "fit_binding_pymc_multi2 is deprecated. Use fit_binding_pymc_multi(..., bg_err=bg_err) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return fit_binding_pymc_multi(
        results,
        scheme,
        n_sd=n_sd,
        n_xerr=n_xerr,
        n_samples=n_samples,
        bg_err=bg_err,
    )


# ------------------------------------------------------------------
# 2.2b  Noise-model helpers and multi-well fit with learned noise
# ------------------------------------------------------------------


@dataclass(frozen=True)
class NoisePriors:
    """Prior scale parameters for the 3-component heteroscedastic noise model.

    All values are HalfNormal sigma parameters.  The variance model is::

        Var(y | mu) = sigma_read**2 + gain * max(0, mu) + alpha**2 * mu**2

    Parameters
    ----------
    sigma_read : float
        HalfNormal sigma for the readout-floor noise (RFU).
    gain : float
        HalfNormal sigma for the Poisson-like gain term (RFU/RFU).
    alpha : float
        HalfNormal sigma for the multiplicative CV term (dimensionless).
    """

    sigma_read: float
    gain: float
    alpha: float


def _noise_priors_from_buffer(
    buffer_df: dict[int, pd.DataFrame],
    labels: list[str],
) -> dict[str, NoisePriors]:
    """Derive noise-model prior scales from buffer replicate variance.

    For each label, performs a simple Var-Mean decomposition on the buffer
    well replicates (rows = pH points, columns = wells) to estimate::

        Var(y) approx sigma_read**2 + gain * mu + alpha**2 * mu**2

    The HalfNormal sigma for each parameter is set to 2x the empirical
    estimate to give a moderately informative but permissive prior.  If
    buffer data are absent or insufficient (< 2 replicate wells), a
    weakly-informative fallback is used.

    Parameters
    ----------
    buffer_df : dict[int, pd.DataFrame]
        Buffer DataFrames keyed by integer label index (1-based).  Each
        DataFrame has well IDs as columns and pH points as rows; the columns
        ``Label``, ``fit``, ``fit_err``, ``mean``, and ``sem`` are ignored.
    labels : list[str]
        String label names in order (e.g. ``['y1', 'y2']``).  Label ``labels[i]``
        corresponds to ``buffer_df[i+1]``.

    Returns
    -------
    dict[str, NoisePriors]
        Prior parameters keyed by string label.
    """
    meta_cols = {"Label", "fit", "fit_err", "mean", "sem"}
    min_replicates = 2
    default = NoisePriors(sigma_read=50.0, gain=10.0, alpha=0.05)
    priors: dict[str, NoisePriors] = {}

    for idx, lbl in enumerate(labels, start=1):
        df = buffer_df.get(idx)
        if df is None or df.empty:
            priors[lbl] = default
            continue

        well_cols = [c for c in df.columns if c not in meta_cols]
        if len(well_cols) < min_replicates:
            priors[lbl] = default
            continue

        data = df[well_cols].to_numpy(dtype=float)
        row_means = data.mean(axis=1)
        row_vars = data.var(axis=1, ddof=1)

        valid = (row_means > 0) & np.isfinite(row_vars) & np.isfinite(row_means)
        if not valid.any():
            priors[lbl] = default
            continue

        m, v = row_means[valid], row_vars[valid]
        cv = float(np.sqrt(v.mean()) / m.mean())
        gain_est = float((v / m).mean())
        sigma_read_est = float(np.sqrt(np.maximum(0.0, v - gain_est * m).mean()))

        priors[lbl] = NoisePriors(
            sigma_read=max(sigma_read_est * 2.0, 1.0),
            gain=max(gain_est * 2.0, 0.1),
            alpha=min(cv * 2.0, 0.5),
        )

    return priors


def fit_binding_pymc_multi_noise(  # noqa: PLR0913,PLR0917
    results: dict[str, FitResult[MiniT]],
    scheme: PlateScheme,
    buffer_df: dict[int, pd.DataFrame],
    n_sd: float = 5.0,
    n_xerr: float = 1.0,
    n_samples: int = 2000,
    nuts_sampler: str = "default",
    *,
    ctr_free_k: bool = False,
) -> az.InferenceData:
    """Multi-well PyMC fit with shared learnable heteroscedastic noise model.

    Fits all wells simultaneously.  Per-label noise parameters
    (``sigma_read``, ``gain``, ``alpha``) are shared across all wells and
    inferred from the data.  The variance model is::

        Var(y | mu) = sigma_read**2 + gain * max(0, mu) + alpha**2 * mu**2

    where *mu* is the model-predicted (background-subtracted) signal.
    Priors for the noise parameters are derived empirically from the buffer
    replicate variance via :func:`_noise_priors_from_buffer`.

    Input data must be background-subtracted (i.e. the standard Tecan
    pipeline output where buffer mean has already been removed).

    Parameters
    ----------
    results : dict[str, FitResult[MiniT]]
        Per-well initial fit results, typically from ``fit_binding_glob``.
    scheme : PlateScheme
        Plate scheme defining control groups for shared-K priors.
    buffer_df : dict[int, pd.DataFrame]
        Buffer DataFrames (integer label index -> DataFrame with well
        columns), used to derive noise priors from replicate variance.
    n_sd : float
        Prior width multiplier for per-well S0/S1 parameters.
    n_xerr : float
        Scaling factor applied to x-value uncertainties.
    n_samples : int
        Number of MCMC posterior samples per chain.
    nuts_sampler : str
        NUTS sampler backend: ``"default"`` (pytensor/CPU), ``"blackjax"``
        (JAX/GPU), ``"numpyro"`` (JAX/GPU), or ``"nutpie"`` (Rust/CPU).
    ctr_free_k : bool
        If True, each CTR replicate well gets its own independent flat K prior
        ``Normal(group_mean, 0.2)`` — identical to UNK well treatment, no
        hierarchical shrinkage.  The spread of K posteriors across replicates
        quantifies between-replicate accuracy.  If False (default), all
        replicates share a single K.

    Returns
    -------
    az.InferenceData
        Posterior trace.  Noise parameters are accessible as
        ``trace.posterior["sigma_read_<lbl>"]``,
        ``trace.posterior["gain_<lbl>"]``, and
        ``trace.posterior["alpha_<lbl>"]``.

    Raises
    ------
    ValueError
        If no valid dataset is found in *results*.
    """
    ds_template = next((r.dataset for r in results.values() if r.dataset), None)
    if ds_template is None:
        msg = "No valid dataset found in results."
        raise ValueError(msg)

    xc = next(iter(ds_template.values())).xc
    x_errc = next(iter(ds_template.values())).x_errc * n_xerr
    labels = list(ds_template.keys())

    noise_priors = _noise_priors_from_buffer(buffer_df, labels)

    values: dict[str, list[float | None]] = {}
    stderr_: dict[str, list[float | None]] = {}
    for name, wells in scheme.names.items():
        values[name] = [
            r.result.params["K"].value
            for well, r in results.items()
            if r.result and well in wells
        ]
        stderr_[name] = [
            r.result.params["K"].stderr
            for well, r in results.items()
            if r.result and well in wells
        ]
    ctr_ks = weighted_stats(values, stderr_)
    active_wells = {key for key, r in results.items() if r.result and r.dataset}

    with pm.Model():
        x_true = create_x_true(xc, x_errc, n_xerr)

        sigma_read = {
            lbl: pm.HalfNormal(f"sigma_read_{lbl}", sigma=noise_priors[lbl].sigma_read)
            for lbl in labels
        }
        gain = {
            lbl: pm.HalfNormal(f"gain_{lbl}", sigma=noise_priors[lbl].gain)
            for lbl in labels
        }
        alpha = {
            lbl: pm.HalfNormal(f"alpha_{lbl}", sigma=noise_priors[lbl].alpha)
            for lbl in labels
        }

        k_params, k_replicate = _build_ctr_k_params(
            scheme,
            ctr_ks,
            active_wells,
            ctr_free_k=ctr_free_k,
            well_k_init=_well_k_init_from_results(results, scheme, n_sd),
        )

        for key, r in results.items():
            if not (r.result and r.dataset):
                continue
            ds = r.dataset
            ctr_name = next(
                (name for name, wells in scheme.names.items() if key in wells), ""
            )
            pars = create_parameter_priors(r.result.params, n_sd, key, "")
            K = _resolve_well_k(  # noqa: N806
                key, ctr_name, pars, k_params, k_replicate, ctr_free_k=ctr_free_k
            )

            for lbl, da in ds.items():
                mu_pred = binding_1site(
                    x_true,
                    K,
                    pars[f"S0_{lbl}_{key}"],
                    pars[f"S1_{lbl}_{key}"],
                    is_ph=ds.is_ph,
                )
                mu_nn = pm_math.maximum(0.0, mu_pred)
                sigma_obs = pm_math.sqrt(
                    sigma_read[lbl] ** 2
                    + gain[lbl] * mu_nn
                    + alpha[lbl] ** 2 * mu_pred**2
                )
                pm.Normal(
                    f"y_likelihood_{lbl}_{key}",
                    mu=mu_pred[da.mask],
                    sigma=sigma_obs[da.mask],
                    observed=da.y,
                )

        trace: az.InferenceData = pm.sample(
            n_samples,
            tune=n_samples // 2,
            target_accept=0.9,
            return_inferencedata=True,
            **_pymc_sample_parallel_args(nuts_sampler),
        )
    return trace


def fit_binding_pymc_multi_noise_xrw(  # noqa: PLR0913,PLR0917
    results: dict[str, FitResult[MiniT]],
    scheme: PlateScheme,
    buffer_df: dict[int, pd.DataFrame],
    n_sd: float = 5.0,
    n_xerr: float = 1.0,
    n_samples: int = 2000,
    sigma_pip_prior: float = 0.02,
    nuts_sampler: str = "default",
    *,
    ctr_free_k: bool = False,
) -> az.InferenceData:
    """Multi-well PyMC fit with shared noise model and per-well pH random walk.

    Extends :func:`fit_binding_pymc_multi_noise` with a hierarchical
    random-walk model for per-well pH deviations.  The first titration step
    is common to all wells (same buffer).  Each subsequent acid addition
    introduces independent Normal(0, sigma_pip²) deviations that accumulate,
    so the variance of the pH deviation at step *t* is *t · sigma_pip²*.

    Non-centred parameterisation is used for numerical efficiency::

        z_pip[t, w] ~ Normal(0, 1)  (shape: n_steps-1 x n_wells)
        x_dev[:, w] = concat([0, cumsum(sigma_pip * z_pip[:, w])])
        x_per_well  = x_nominal[:, None] + x_dev   (shape: n_steps x n_wells)

    Parameters
    ----------
    results : dict[str, FitResult[MiniT]]
        Per-well initial fit results, typically from ``fit_binding_glob``.
    scheme : PlateScheme
        Plate scheme defining control groups for shared-K priors.
    buffer_df : dict[int, pd.DataFrame]
        Buffer DataFrames (integer label index -> DataFrame with well
        columns), used to derive noise priors from replicate variance.
    n_sd : float
        Prior width multiplier for per-well S0/S1 parameters.
    n_xerr : float
        Scaling factor applied to x-value uncertainties.
    n_samples : int
        Number of MCMC posterior samples per chain.
    sigma_pip_prior : float
        Prior scale (HalfNormal sigma) for the per-step pipetting SD,
        in the same units as the x-axis (pH units by default).
    nuts_sampler : str
        NUTS sampler backend: ``"default"`` (pytensor/CPU), ``"blackjax"``
        (JAX/GPU), ``"numpyro"`` (JAX/GPU), or ``"nutpie"`` (Rust/CPU).
    ctr_free_k : bool
        If True, each CTR replicate well gets its own independent flat K prior
        ``Normal(group_mean, 0.2)`` — identical to UNK well treatment, no
        hierarchical shrinkage.  The spread of K posteriors across replicates
        quantifies between-replicate accuracy.  If False (default), all
        replicates share a single K.

    Returns
    -------
    az.InferenceData
        Posterior trace.  Per-well x is accessible as
        ``trace.posterior["x_per_well"]`` with dims ``("chain", "draw",
        "step", "well")``.  Noise parameters are accessible as
        ``trace.posterior["sigma_read_<lbl>"]`` etc.

    Raises
    ------
    ValueError
        If no valid dataset is found in *results*.
    """
    ds_template = next((r.dataset for r in results.values() if r.dataset), None)
    if ds_template is None:
        msg = "No valid dataset found in results."
        raise ValueError(msg)

    xc = next(iter(ds_template.values())).xc
    x_errc = next(iter(ds_template.values())).x_errc * n_xerr
    labels = list(ds_template.keys())
    n_steps = len(xc)

    wells = [key for key, r in results.items() if r.result and r.dataset]
    n_wells = len(wells)
    well_idx = {key: i for i, key in enumerate(wells)}
    active_wells = set(wells)

    noise_priors = _noise_priors_from_buffer(buffer_df, labels)

    values: dict[str, list[float | None]] = {}
    stderr_: dict[str, list[float | None]] = {}
    for name, scheme_wells in scheme.names.items():
        values[name] = [
            r.result.params["K"].value
            for well, r in results.items()
            if r.result and well in scheme_wells
        ]
        stderr_[name] = [
            r.result.params["K"].stderr
            for well, r in results.items()
            if r.result and well in scheme_wells
        ]
    ctr_ks = weighted_stats(values, stderr_)

    coords: dict[str, list[int] | list[str]] = {
        "well": wells,
        "step": list(range(n_steps)),
        "step_diff": list(range(n_steps - 1)),
    }

    with pm.Model(coords=coords):
        x_true = create_x_true(xc, x_errc, n_xerr)

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

        sigma_read = {
            lbl: pm.HalfNormal(f"sigma_read_{lbl}", sigma=noise_priors[lbl].sigma_read)
            for lbl in labels
        }
        gain = {
            lbl: pm.HalfNormal(f"gain_{lbl}", sigma=noise_priors[lbl].gain)
            for lbl in labels
        }
        alpha = {
            lbl: pm.HalfNormal(f"alpha_{lbl}", sigma=noise_priors[lbl].alpha)
            for lbl in labels
        }

        k_params, k_replicate = _build_ctr_k_params(
            scheme,
            ctr_ks,
            active_wells,
            ctr_free_k=ctr_free_k,
            well_k_init=_well_k_init_from_results(results, scheme, n_sd),
        )

        for key, r in results.items():
            if not (r.result and r.dataset):
                continue
            ds = r.dataset
            ctr_name = next(
                (name for name, sw in scheme.names.items() if key in sw), ""
            )
            pars = create_parameter_priors(r.result.params, n_sd, key, "")
            K = _resolve_well_k(  # noqa: N806
                key, ctr_name, pars, k_params, k_replicate, ctr_free_k=ctr_free_k
            )
            w_idx = well_idx[key]
            x_w = x_per_well[:, w_idx]

            for lbl, da in ds.items():
                mu_pred = binding_1site(
                    x_w,
                    K,
                    pars[f"S0_{lbl}_{key}"],
                    pars[f"S1_{lbl}_{key}"],
                    is_ph=ds.is_ph,
                )
                mu_nn = pm_math.maximum(0.0, mu_pred)
                sigma_obs = pm_math.sqrt(
                    sigma_read[lbl] ** 2
                    + gain[lbl] * mu_nn
                    + alpha[lbl] ** 2 * mu_pred**2
                )
                pm.Normal(
                    f"y_likelihood_{lbl}_{key}",
                    mu=mu_pred[da.mask],
                    sigma=sigma_obs[da.mask],
                    observed=da.y,
                )

        trace: az.InferenceData = pm.sample(
            n_samples,
            tune=n_samples // 2,
            target_accept=0.9,
            return_inferencedata=True,
            **_pymc_sample_parallel_args(nuts_sampler),
        )
    return trace


# ------------------------------------------------------------------
# 2.3  Posterior-predictive helper - visualise one well at a time
# ------------------------------------------------------------------
def _extract_sigma_df(trace: az.InferenceData) -> pd.DataFrame:
    """Extract heteroscedastic sigma_obs parameters from a PyMC trace into a DataFrame."""
    trace_df = az.summary(trace)
    sigma_df = trace_df[trace_df.index.str.startswith("sigma_obs_")].copy()
    if len(sigma_df) == 0:
        return pd.DataFrame()

    pattern = r"sigma_obs_(?P<label>[A-Za-z0-9]+)_(?P<well>[^\[]+)\[(?P<idx>\d+)\]"
    extracted = sigma_df.index.to_series().str.extract(pattern)
    sigma_df = pd.concat([sigma_df, extracted], axis=1)  # type: ignore[call-overload]
    sigma_df["idx"] = sigma_df["idx"].astype(int)
    return sigma_df  # type: ignore[no-any-return]


def plot_noise_vs_index(
    trace: az.InferenceData,
    wells: Sequence[str] | str | None = None,
    figsize_per_well: tuple[float, float] = (5, 4),
    max_cols: int = 4,
) -> figure.Figure:
    """Plot inferred noise (sigma) across titration steps for specified wells.

    Parameters
    ----------
    trace : az.InferenceData
        The PyMC inference trace containing `sigma_obs` deterministic nodes.
    wells : Sequence[str] | str | None, optional
        A specific well ID (e.g., 'A01'), a list of well IDs, or None to plot all
        wells found in the trace. Default is None.
    figsize_per_well : tuple[float, float], optional
        The width and height allocated per well subplot.
    max_cols : int, optional
        Maximum number of columns in the subplot grid.

    Returns
    -------
    figure.Figure
        The constructed matplotlib figure.
    """
    sigma_df = _extract_sigma_df(trace)
    if len(sigma_df) == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No 'sigma_obs' data found.", ha="center")
        return fig

    if wells is None:
        wells_to_plot = list(sigma_df["well"].unique())
    elif isinstance(wells, str):
        wells_to_plot = [wells]
    else:
        wells_to_plot = list(wells)

    n_wells = len(wells_to_plot)
    n_cols = min(n_wells, max_cols)
    n_rows = max(1, (n_wells - 1) // n_cols + 1)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * figsize_per_well[0], n_rows * figsize_per_well[1]),
        squeeze=False,
    )

    for i, well in enumerate(wells_to_plot):
        ax = axes.flat[i]
        well_data = sigma_df[sigma_df["well"] == well]
        if len(well_data) == 0:
            ax.set_title(f"Well {well} (No Data)")
            continue

        sns.lineplot(data=well_data, x="idx", y="mean", hue="label", marker="o", ax=ax)

        # Add error bands (94% HDI) for the noise estimate
        for lbl in well_data["label"].unique():
            lbl_data = well_data[well_data["label"] == lbl]
            ax.fill_between(
                lbl_data["idx"],
                lbl_data["hdi_3%"],
                lbl_data["hdi_97%"],
                alpha=0.2,
            )

        ax.set_title(rf"Inferred Noise ($\sigma_{{obs}}$) - Well {well}")
        ax.set_xlabel("Titration Step (Index)")
        ax.set_ylabel(r"Inferred $\sigma$ (RFU)")

    # Hide unused axes
    for j in range(len(wells_to_plot), len(axes.flat)):
        axes.flat[j].set_visible(False)

    plt.tight_layout()
    return fig


def plot_noise_vs_signal(
    trace: az.InferenceData,
    results: Mapping[str, FitResult[MiniT]],
    figsize_per_label: tuple[float, float] = (6, 5),
) -> figure.Figure:
    """Plot inferred noise (sigma) versus observed signal across all wells.

    This function extracts the `sigma_obs_...` parameters from a heteroscedastic
    PyMC trace (like `fit_binding_pymc_multi`) and plots them against the observed
    fluorescence values `da.y` to visualize the noise-to-signal relationship.

    Parameters
    ----------
    trace : az.InferenceData
        The PyMC inference trace containing the `sigma_obs` deterministic nodes.
    results : Mapping[str, FitResult[MiniT]]
        The dictionary of well results containing datasets with `.y` arrays.
        Normally this is `tit.result_global.results`.
    figsize_per_label : tuple[float, float], optional
        The width and height to allocate per band/label in the final figure.

    Returns
    -------
    figure.Figure
        The constructed matplotlib figure.
    """
    sigma_df = _extract_sigma_df(trace)
    if len(sigma_df) == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No 'sigma_obs' data found.", ha="center")
        return fig

    # Build a combined DataFrame of (Observed Y, Inferred Sigma) for all wells
    records = []
    for well, fr in results.items():
        if fr.dataset is not None:
            for lbl, da in fr.dataset.items():
                # Get the sigma_obs rows for this well and label, sorted by index
                mask_well_lbl = (sigma_df["well"] == well) & (sigma_df["label"] == lbl)
                well_lbl_sigmas = sigma_df[mask_well_lbl].sort_values("idx")

                # Ensure we have PyMC results for this well
                if len(well_lbl_sigmas) > 0 and hasattr(da, "mask"):
                    # Apply the dataset mask to the full sigma array to match valid da.y points
                    valid_sigmas = well_lbl_sigmas[da.mask]["mean"].to_numpy()
                    valid_y = da.y

                    for sig, y_val in zip(valid_sigmas, valid_y, strict=False):
                        records.append({
                            "well": well,
                            "label": lbl,
                            "sigma_mean": sig,
                            "y_obs": y_val,
                        })
    all_wells_df = pd.DataFrame(records)
    if len(all_wells_df) == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No 'sigma_obs' data found.", ha="center")
        return fig

    # Plotting
    labels_found = all_wells_df["label"].unique()
    fig, axes = plt.subplots(
        1,
        len(labels_found),
        figsize=(figsize_per_label[0] * len(labels_found), figsize_per_label[1]),
    )
    if len(labels_found) == 1:
        axes = [axes]

    for ax, lbl in zip(axes, labels_found, strict=False):
        data = all_wells_df[all_wells_df["label"] == lbl]

        # Plot all points as a semi-transparent scatter
        sns.scatterplot(
            data=data,
            x="y_obs",
            y="sigma_mean",
            alpha=0.3,
            edgecolor=None,
            color="indigo",
            s=20,
            ax=ax,
        )

        # Add a trendline to highlight the heteroscedastic curve shape
        sns.regplot(
            data=data,
            x="y_obs",
            y="sigma_mean",
            scatter=False,
            order=2,
            color="orange",
            line_kws={"linestyle": "--", "linewidth": 2},
            ax=ax,
        )

        ax.set_title(f"Band {lbl} (All Wells)")
        ax.set_xlabel("Observed Fluorescence ($y$)")
        ax.set_ylabel(r"Inferred $\sigma_{obs}$ (mean)")

    plt.suptitle("Noise vs Signal Intensity (All Wells)", y=1.05, fontsize=14)
    plt.tight_layout()
    return fig


def plot_ppc_well(
    trace: az.InferenceData,
    key: str,
    labels: list[str] | None = None,
    figsize: tuple[float, float] = (8, 4),
) -> figure.Figure:
    """Draw posterior predictive samples for a particular well (and all its labels).

    The returned figure can be displayed with matplotlib.

    Parameters
    ----------
    trace   : az.InferenceData
        Trace produced by PyMC fitting with posterior predictive data included.
    key     : str
        Well identifier (e.g. 'A01').
    labels  : list[str] | None
        Names of the bands to show.  If *None* the function will
        automatically look for all variables starting with
        ``'y_likelihood'`` that contain this key.
    figsize: tuple[float, float]
        size?

    Returns
    -------
    figure.Figure
        Plot

    Raises
    ------
    AttributeError
        If the trace does not contain `posterior_predictive` data.
    """
    if not hasattr(trace, "posterior_predictive"):
        msg = (
            "The InferenceData object does not contain 'posterior_predictive'. "
            "You must run pm.sample_posterior_predictive() inside the pm.Model "
            "context after sampling to generate this data."
        )
        raise AttributeError(msg)

    if labels is None:
        labels = [
            str(var).split("_")[2]
            for var in trace.posterior_predictive.data_vars
            if f"_{key}" in str(var) and str(var).startswith("y_likelihood")
        ]

    fig, axes = plt.subplots(
        len(labels), 1, figsize=(figsize[0], figsize[1] * len(labels)), sharex=True
    )
    if len(labels) == 1:
        axes = [axes]

    for ax, lbl in zip(axes, labels, strict=True):
        var_name = f"y_likelihood_{lbl}_{key}"

        # Prepare dictionary for az.from_dict
        trace_dict = {
            "posterior_predictive": {var_name: trace.posterior_predictive[var_name]}
        }
        if hasattr(trace, "observed_data") and var_name in trace.observed_data:
            trace_dict["observed_data"] = {var_name: trace.observed_data[var_name]}

        az.plot_ppc(  # type: ignore[no-untyped-call]
            az.from_dict(
                **trace_dict,  # type: ignore[arg-type]
                coords={"y": [lbl]},
                dims={"y": 0},
            ),
            ax=ax,
        )
        ax.set_title(f"Well {key} - band {lbl}")
        ax.set_xlabel("Observed y")
        ax.set_ylabel("Posterior predictive")

    plt.tight_layout()
    return fig


# ------------------------------------------------------------------
# 2.4  Comparison of posteriors with deterministic fits
# ------------------------------------------------------------------
def compare_posteriors(
    trace: az.InferenceData, results: dict[str, FitResult[MiniT]]
) -> None:
    """Print posterior mean ± 95 % C.I.

    For the K parameter for each well, and juxtapose it with the deterministic K
    (from fit_binding_pymc).

    Parameters
    ----------
    trace   : az.InferenceData
        Output of ``fit_binding_pymc_advanced``.
    results : dict[str, FitResult[MiniT]]
        Deterministic fits produced by the old pipeline.
    """
    # Summarise the trace
    summary = az.summary(trace, var_names=["K_*"], round_to=3)
    print("\nPosterior for K (averaged over all draws)")
    print(summary[["mean", "hdi_2.5%", "hdi_97.5%"]])

    # Add deterministic K to the table for easy comparison
    deterministic = {}
    for k, fr in results.items():
        if fr.result and "K" in fr.result.params:
            deterministic[k] = fr.result.params["K"].value

    print("\nDeterministic K  (fit_binding_pymc)")
    for k, v in deterministic.items():
        print(f"  {k:6s}  {v:0.3f}")

    # Align rows
    table = summary.join(pd.Series(deterministic, name="deterministic_K"))
    print("\nCombined table")
    print(table[["mean", "hdi_2.5%", "hdi_97.5%", "deterministic_K"]])


def fit_pymc_hierarchical(  # noqa: PLR0913,PLR0917
    results: dict[str, FitResult[MiniT]],
    scheme: PlateScheme,
    bg_err: dict[int, ArrayF],
    n_sd: float = 5.0,
    n_xerr: float = 1.0,
    n_samples: int = 2000,
) -> az.InferenceData:
    """
    Analyze multiple titrations with a hierarchical Bayesian model.

    This model shares information about the dissociation constant 'K' among
    wells belonging to the same control group, leading to more robust estimates.

    Parameters
    ----------
    results : dict[str, FitResult[MiniT]]
        A dictionary mapping well IDs to their initial `FitResult` from a
        prior `fit_lm` run.
    scheme : PlateScheme
        The plate scheme defining control groups.
    bg_err : dict[int, ArrayF]
        Background error for each signal band.
    n_sd : float
        The number of standard deviations for the prior width of S0/S1.
    n_xerr : float
        Scaling factor for x-value uncertainties.
    n_samples : int
        Number of MCMC samples.

    Returns
    -------
    az.InferenceData
        The PyMC trace containing the posterior distributions.

    Raises
    ------
    ValueError
        With invalid dataset.
    """
    ds_template = next(r.dataset for r in results.values() if r.dataset)
    if not ds_template:
        msg = "No valid dataset found in results."
        raise ValueError(msg)

    xc = next(iter(ds_template.values())).xc
    x_errc = next(iter(ds_template.values())).x_errc * n_xerr
    labels = list(ds_template.keys())

    # --- Pre-calculate weighted stats for K priors ---
    k_values = {
        name: [
            r.result.params["K"].value
            for well, r in results.items()
            if r.result and well in wells
        ]
        for name, wells in scheme.names.items()
    }
    k_stderr = {
        name: [
            r.result.params["K"].stderr
            for well, r in results.items()
            if r.result and r.result.params["K"].stderr and well in wells
        ]
        for name, wells in scheme.names.items()
    }
    print(k_stderr)

    with pm.Model():
        x_true = create_x_true(xc, x_errc, n_xerr)

        # --- Priors for noise model ---
        sigma_signal_scale = {
            lbl: pm.HalfNormal(f"sigma_signal_scale_{lbl}", sigma=10.0)
            for lbl in labels
        }
        true_buffer_mean = {
            lbl: pm.Normal(f"true_buffer_{lbl}", mu=0, sigma=bg_err[i + 1])
            for i, lbl in enumerate(labels)
        }
        variance_buffer_contrib = {
            lbl: bg_err[i + 1] ** 2 for i, lbl in enumerate(labels)
        }

        # --- Priors for shared K values in control groups ---
        k_params = {}
        for name in scheme.names:
            mean_k = np.mean(k_values[name]) if k_values[name] else 7.0
            k_params[name] = pm.Normal(_ctr_param_name(name), mu=mean_k, sigma=0.5)

        # --- Loop through each well to define its model ---
        for key, r in results.items():
            if r.result and r.dataset:
                ctr_name = next(
                    (name for name, wells in scheme.names.items() if key in wells), ""
                )
                priors = create_parameter_priors(
                    r.result.params, n_sd, key, ctr_name, default_sigma=1.0
                )
                k_param = k_params[ctr_name] if ctr_name else priors[f"K_{key}"]

                for lbl, da in r.dataset.items():
                    y_model_signal = binding_1site(
                        x_true,
                        k_param,
                        priors[f"S0_{lbl}_{key}"],
                        priors[f"S1_{lbl}_{key}"],
                        is_ph=r.dataset.is_ph,
                    )
                    mu_total = pm.Deterministic(
                        f"mu_total_{lbl}_{key}", y_model_signal + true_buffer_mean[lbl]
                    )
                    var_signal = sigma_signal_scale[lbl] * pm_math.maximum(
                        1e-6, y_model_signal
                    )
                    total_var = pm.Deterministic(
                        f"total_var_{lbl}_{key}",
                        variance_buffer_contrib[lbl] + var_signal,
                    )
                    sigma_obs = pm.Deterministic(
                        f"sigma_obs_{lbl}_{key}", pm_math.sqrt(total_var)
                    )

                    pm.Normal(
                        f"y_likelihood_{lbl}_{key}",
                        mu=mu_total[da.mask],
                        sigma=sigma_obs[da.mask],
                        observed=da.y,
                    )

        trace: az.InferenceData = pm.sample(
            n_samples,
            tune=n_samples // 2,
            target_accept=0.9,
            return_inferencedata=True,
            **_pymc_sample_parallel_args(),
        )
    return trace
