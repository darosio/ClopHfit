"""Bayesian (PyMC) fitting utilities and pipelines."""

from __future__ import annotations

import copy
import typing

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm  # type: ignore[import-untyped]
from lmfit import Parameters  # type: ignore[import-untyped]
from matplotlib import figure
from pymc import math as pm_math
from pytensor.tensor import as_tensor_variable
from scipy import optimize

from clophfit.fitting.models import binding_1site
from clophfit.fitting.plotting import PlotParameters, plot_fit

from .core import N_BOOT  # local to avoid circular import
from .data_structures import DataArray, Dataset, FitResult, MiniT, _Result

if typing.TYPE_CHECKING:
    from clophfit.clophfit_types import ArrayF, FloatFunc
    from clophfit.prtecan import PlateScheme


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
        priors[param_name(name)] = pm.Normal(param_name(name), mu=p.value, sigma=sigma)
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
            rpars.add(name, value=row["mean"], min=row["hdi_3%"], max=row["hdi_97%"])
            rpars[name].stderr = row["sd"]
            rpars[name].init_value = row.get("r_hat", np.nan)
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

    # Return the fit result
    return FitResult(fig, _Result(rpars), trace, ds)


def extract_fit(
    key: str, ctr: str, trace_df: pd.DataFrame, ds: Dataset
) -> FitResult[az.InferenceData]:
    """Compute individual dataset fit from a multi-well trace summary."""
    rpars = Parameters()
    rdf = trace_df[trace_df.index.str.endswith(key)]
    for name, row in rdf.iterrows():
        extracted_name = str(name).replace(f"_{key}", "")
        rpars.add(
            extracted_name, value=row["mean"], min=row["hdi_3%"], max=row["hdi_97%"]
        )
        rpars[extracted_name].stderr = row["sd"]
        rpars[extracted_name].init_value = row.get("r_hat", np.nan)
    if ctr:
        rdf = trace_df[trace_df.index.str.endswith(ctr)]
        for name, row in rdf.iterrows():
            extracted_name = str(name).replace(f"_{ctr}", "")
            rpars.add(
                extracted_name, value=row["mean"], min=row["hdi_3%"], max=row["hdi_97%"]
            )
            rpars[extracted_name].stderr = row["sd"]
            rpars[extracted_name].init_value = row.get("r_hat", np.nan)
    nxc, nx_errc = _extract_x_true_from_trace_df(trace_df)
    for da in ds.values():
        da.xc = nxc
        da.x_errc = nx_errc
    # Create figure
    fig = figure.Figure()
    ax = fig.add_subplot(111)
    plot_fit(ax, ds, rpars, nboot=N_BOOT, pp=PlotParameters(ds.is_ph))
    return FitResult(fig, _Result(rpars), az.InferenceData(), ds)


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


def x_true_from_trace_df(trace_df: pd.DataFrame) -> DataArray:
    """Extract x_true from an ArviZ summary DataFrame."""
    nxc, nx_errc = _extract_x_true_from_trace_df(trace_df)
    return DataArray(xc=nxc, yc=np.ones_like(nxc), x_errc=nx_errc)


def fit_binding_pymc(
    fr: FitResult[MiniT],
    n_sd: float = 10.0,
    n_xerr: float = 1.0,
    ye_scaling: float = 1.0,
    n_samples: int = 2000,
) -> FitResult[az.InferenceData]:
    """Analyze multi-label titration datasets using PyMC (single model)."""
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
        ye_mag = pm.HalfNormal("ye_mag", sigma=ye_scaling)
        for lbl, da in ds.items():
            y_model = binding_1site(
                x_true, pars["K"], pars[f"S0_{lbl}"], pars[f"S1_{lbl}"], is_ph=ds.is_ph
            )
            pm.Normal(
                f"y_likelihood_{lbl}",
                mu=y_model[da.mask],
                sigma=ye_mag * da.y_err,
                observed=da.y,
            )
        # Inference
        tune = n_samples // 2
        trace = pm.sample(
            n_samples,
            tune=tune,
            target_accept=0.9,
            cores=4,
            return_inferencedata=True,
            idata_kwargs={"log_likelihood": True},
        )
    return process_trace(trace, params.keys(), ds, n_xerr)


def fit_binding_pymc2(
    fr: FitResult[MiniT],
    n_sd: float = 10.0,
    n_xerr: float = 1.0,
    n_samples: int = 2000,
) -> FitResult[az.InferenceData]:
    """Analyze multi-label titration datasets using PyMC with separate ye_mag per label."""
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
        ye_mag: dict[str, pm.Distribution] = {}
        for lbl in ds:
            ye_mag[lbl] = pm.HalfNormal(f"ye_mag_{lbl}", sigma=10.0)  # TODO: 100
        for lbl, da in ds.items():
            y_model = binding_1site(
                x_true, pars["K"], pars[f"S0_{lbl}"], pars[f"S1_{lbl}"], is_ph=ds.is_ph
            )
            pm.Normal(
                f"y_likelihood_{lbl}",
                mu=y_model[da.mask],
                sigma=ye_mag[lbl] * np.ones_like(da.y_err),
                observed=da.y,
            )
        # Inference
        tune = n_samples // 2
        trace = pm.sample(
            n_samples,
            tune=tune,
            target_accept=0.9,
            cores=4,
            return_inferencedata=True,
            idata_kwargs={"log_likelihood": True},
        )
    return process_trace(trace, params.keys(), ds, n_xerr)


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
            cores=4,
            return_inferencedata=True,
            target_accept=0.9,
            idata_kwargs={"log_likelihood": True},
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
    """Bayesian ODR-like modeling of x and y errors."""
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
        return pm.sample(n_samples, cores=4, return_inferencedata=True)
    # TODO:  return process_trace(trace, params.keys(), ds, 0)


# ------------------------------------------------------------------
# Helper: weighted statistics
# ------------------------------------------------------------------


def weighted_stats(
    values: dict[str, list[float]], stderr: dict[str, list[float]]
) -> dict[str, tuple[float, float]]:
    """Weighted mean and stderr for control priors."""
    results: dict[str, tuple[float, float]] = {}
    for sample in values:  # noqa:PLC0206
        x = np.array(values[sample])
        se = np.array(stderr[sample])
        weighted_mean = np.average(x, weights=1 / se**2)
        weighted_stderr = np.sqrt(1 / np.sum(1 / se**2))
        results[sample] = (weighted_mean, weighted_stderr)
    return results


def fit_binding_pymc_multi(  # noqa: PLR0913,PLR0917
    results: dict[str, FitResult[MiniT]],
    scheme: PlateScheme,
    n_sd: float = 5.0,
    n_xerr: float = 1.0,
    ye_scaling: float = 1.0,
    n_samples: int = 2000,
) -> az.InferenceData:
    """Multi-well PyMC with shared K per control group and per-label noise."""
    # FIXME: pytensor.config.floatX = "float32"  # type: ignore[attr-defined]
    ds = next((r.dataset for r in results.values() if r.dataset), None)
    if ds is None:
        msg = "No valid dataset found in results."
        raise ValueError(msg)
    xc = next(iter(ds.values())).xc
    x_errc = next(iter(ds.values())).x_errc * n_xerr
    labels = list(ds.keys())
    values: dict[str, list[float]] = {}
    stderr: dict[str, list[float]] = {}

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

    with pm.Model():
        ye_mag: dict[str, pm.Distribution] = {
            label: pm.HalfNormal(f"ye_mag_{label}", sigma=ye_scaling)
            for label in labels
        }
        x_true = create_x_true(xc, x_errc, n_xerr)

        # Create shared K parameters for each control group
        k_params = {
            control_name: pm.Normal(
                f"K_{control_name}",
                mu=ctr_ks[control_name][0],
                sigma=0.2,  # FIXME: use var
            )
            for control_name in scheme.names
        }
        for key, r in results.items():
            if r.result and r.dataset:
                ds = r.dataset
                # Determine if the well is associated with a control group
                ctr_name = next(
                    (name for name, wells in scheme.names.items() if key in wells), ""
                )
                pars = create_parameter_priors(r.result.params, n_sd, key, ctr_name)
                # Use shared K for control group wells or create a unique K otherwise
                K = k_params[ctr_name] if ctr_name else pars[f"K_{key}"]  # noqa: N806

                for lbl, da in ds.items():
                    y_model = binding_1site(
                        x_true,
                        K,
                        pars[f"S0_{lbl}_{key}"],
                        pars[f"S1_{lbl}_{key}"],
                        is_ph=ds.is_ph,
                    )
                    pm.Normal(
                        f"y_likelihood_{lbl}_{key}",
                        mu=y_model[da.mask],
                        sigma=ye_mag[lbl] * da.y_err,
                        observed=da.y,
                    )

        trace: az.InferenceData = pm.sample(
            n_samples, target_accept=0.9, return_inferencedata=True
        )

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
    """Multi-well PyMC with heteroscedastic noise combining buffer and signal."""
    ds_example = next((r.dataset for r in results.values() if r.dataset), None)
    ds = next((result.dataset for result in results.values() if result.dataset), None)

    if ds_example is None:
        msg = "No valid dataset found in results."
        raise ValueError(msg)
    # Extract common data once
    xc = next(iter(ds_example.values())).xc
    x_errc = next(iter(ds_example.values())).x_errc * n_xerr
    labels = list(ds_example.keys())  # e.g., ['y1', 'y2']
    # --- Pre-calculate weighted stats for K priors (remains the same) ---
    values: dict[str, list[float]] = {}
    stderr: dict[str, list[float]] = {}
    for name, wells in scheme.names.items():
        values[name] = [
            v.result.params["K"].value
            for well, v in results.items()
            if v.result
            and well in wells  # and "K" in v.result.params._params # Check if K exists
        ]
        stderr[name] = [
            v.result.params["K"].stderr
            for well, v in results.items()
            if v.result and well in wells  # and "K" in v.result.params._params
        ]
    ctr_ks = weighted_stats(values, stderr)
    # MAYBE: Restore logger.info(f"Weighted K stats for control groups: {ctr_ks}")

    with pm.Model():
        # --- Common Priors / Variables for the entire model ---
        x_true = create_x_true(xc, x_errc, n_xerr)
        # Global scaling factors for the signal-dependent noise for each label (band)
        # `sigma` prior for HalfNormal should be set considering the expected scale of your signal values.
        # If your fluorescence values are thousands, a sigma of 10-100 might be reasonable for the scaling factor.
        sigma_signal_scale: dict[str, pm.Distribution] = {
            lbl: pm.HalfNormal(f"sigma_signal_scale_{lbl}", sigma=10.0)
            for lbl in labels
        }
        true_buffer = {
            lbl: pm.Normal(f"true_buffer_{lbl}", mu=0, sigma=bg_err[i])
            for i, lbl in enumerate(labels, start=1)
        }
        variance_buffer_contrib = {i: bg_err[i] ** 2 for i in range(1, len(labels) + 1)}

        # Degrees of freedom for Student's T distribution (for robustness)
        # Can be shared or per-label. Shared is often fine for similar data types.
        # this was for student: nu_common = pm.Gamma("nu_common", alpha=2, beta=0.1)

        # Create shared K parameters for each control group
        k_params = {
            control_name: pm.Normal(
                f"K_{control_name}",
                mu=ctr_ks[control_name][0],
                # if ctr_ks[control_name][0] is not np.nan
                # else 7.0,  # Handle case where no K values found
                sigma=0.2,  # FIXME: consider using ctr_ks[control_name][1] for sigma
                # TODO: sigma=ctr_ks[control_name][1] if ctr_ks[control_name][1] is not np.nan else 0.5, # Default sigma for K if no stderr
            )
            for control_name in scheme.names
        }
        print(k_params)
        # --- Loop through each well (key) and its data ---
        for key, r in results.items():
            if r.result and r.dataset:
                ds = r.dataset
                # Determine if the well is associated with a control group
                ctr_name = next(
                    (name for name, wells in scheme.names.items() if key in wells), ""
                )
                # Parameters for S0, S1 (unique to each well and label)
                # `create_parameter_priors` should return a dict of PyMC distributions for S0, S1 for this key/well
                pars = create_parameter_priors(r.result.params, n_sd, key, ctr_name)
                # Use shared K for control group wells or create a unique K otherwise
                k_param_for_well = k_params[ctr_name] if ctr_name else pars[f"K_{key}"]
                # --- Loop through each fluorescence label (e.g., 'y1', 'y2') within the current well ---
                for i, (lbl, da) in enumerate(ds.items(), start=1):
                    # for lbl, da in ds.items():
                    # --- Predicted signal from the binding model ---
                    y_model_signal = binding_1site(
                        x_true,
                        k_param_for_well,
                        pars[f"S0_{lbl}_{key}"],
                        pars[f"S1_{lbl}_{key}"],
                        is_ph=ds.is_ph,
                    )
                    # Predicted Total Fluorescence (Signal + Buffer) for the likelihood mu
                    mu_total_pred = pm.Deterministic(
                        f"mu_total_pred_{lbl}_{key}", y_model_signal + true_buffer[lbl]
                    )
                    # --- Model the Noise ---
                    # A common model for fluorescence noise is that standard deviation scales with sqrt(mean)
                    # or that variance scales linearly with mean (similar to Poisson, but continuous/scaled).
                    # Let's use a simpler and common power-law for noise (SD = a * mu^b)
                    # Here, we'll assume `b=0.5` (sqrt) and `a` is our `sigma_signal_scale`.
                    # Calculate the variance for the signal component (excluding buffer noise)
                    # Ensuring non-negativity for sqrt
                    # Variance from signal itself (heteroscedastic: proportional to predicted signal mean)
                    # Use pm_math.maximum to avoid issues with negative predicted signals for variance
                    variance_signal_contrib = sigma_signal_scale[lbl] * pm_math.maximum(
                        1e-6, y_model_signal
                    )
                    # Total variance is the sum of independent variances
                    total_variance_obs = pm.Deterministic(
                        f"total_variance_obs_{lbl}_{key}",
                        # variance_buffer_contrib + variance_signal_contrib,
                        variance_buffer_contrib[i] + variance_signal_contrib,
                    )
                    # Total standard deviation for the likelihood
                    sigma_obs = pm.Deterministic(
                        f"sigma_obs_{lbl}_{key}", pm_math.sqrt(total_variance_obs)
                    )
                    # --- Likelihood ---
                    # Use Student's T distribution for robustness against outliers
                    # Apply mask to observed data and corresponding mu/sigma
                    # This is the learned, heteroscedastic SD
                    pm.Normal(
                        f"y_likelihood_{lbl}_{key}",
                        # this was for student: nu=nu_common,  # Use the shared nu_common
                        mu=mu_total_pred[da.mask],
                        sigma=sigma_obs[da.mask],
                        observed=da.y,
                    )

        trace: az.InferenceData = pm.sample(
            n_samples, target_accept=0.9, return_inferencedata=True
        )

    return trace


# ------------------------------------------------------------------
# 2.3  Posterior-predictive helper - visualise one well at a time
# ------------------------------------------------------------------
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
        Trace produced by ``fit_binding_pymc_advanced``.
    key     : str
        Well identifier (e.g. 'A01').
    labels  : list[str] | None
        Names of the bands to show.  If *None* the function will
        automatically look for all variables starting with
        ``'y_'`` that contain this key.
    figsize: tuple[float, float]
        size?

    Returns
    -------
    figure.Figure
        Plot
    """
    if labels is None:
        labels = [
            var.split("_")[1]
            for var in trace.posterior.data_vars  # type: ignore[attr-defined]
            if f"{key}" in var and var.startswith("y_")
        ]

    fig, axes = plt.subplots(
        len(labels), 1, figsize=(figsize[0], figsize[1] * len(labels)), sharex=True
    )
    if len(labels) == 1:
        axes = [axes]

    for ax, lbl in zip(axes, labels, strict=True):
        var_name = f"y_{lbl}_{key}"
        az.plot_ppc(  # type: ignore[no-untyped-call]
            az.from_dict(
                {"posterior_predictive": trace.posterior_predictive[var_name]},  # type: ignore[attr-defined]
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
            k_params[name] = pm.Normal(f"K_{name}", mu=mean_k, sigma=0.5)

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
            n_samples, tune=n_samples // 2, target_accept=0.9, return_inferencedata=True
        )
    return trace
