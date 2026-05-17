"""Bayesian (PyMC) fitting utilities and pipelines."""

from __future__ import annotations

import copy
import os
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
from pytensor.tensor import as_tensor_variable

from clophfit.fitting.models import binding_1site
from clophfit.fitting.plotting import PlotParameters, plot_fit

from .core import N_BOOT, fit_binding_glob  # local to avoid circular import
from .data_structures import DataArray, Dataset, FitResult, MiniT, _Result

if typing.TYPE_CHECKING:
    from clophfit.clophfit_types import ArrayF
    from clophfit.prtecan import PlateScheme


__all__ = [
    "fit_binding_pymc",
    "fit_binding_pymc_multi",
    "process_trace",
]


def _pymc_sample_parallel_args(
    nuts_sampler: str = "default",
    cores: int | None = None,
    chains: int | None = None,
) -> dict[str, object]:
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
        if nuts_sampler == "blackjax":
            kwargs["chain_method"] = "vectorized"
        if nuts_sampler in {"blackjax", "numpyro"}:
            kwargs["progressbar"] = False
    if cores is not None:
        kwargs["cores"] = cores
    else:
        kwargs["cores"] = os.cpu_count() or 1
    if chains is not None:
        kwargs["chains"] = chains
    else:
        kwargs["chains"] = 4
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


def extract_fit(  # noqa: C901,PLR0912
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

    # Try new bracket notation first: S0_y1[A01], K_free[A01]
    suffix_new = f"[{key}]"
    rdf_new = trace_df[trace_df.index.str.endswith(suffix_new)]

    if not rdf_new.empty:
        for name, row in rdf_new.iterrows():
            extracted_name = str(name).replace(suffix_new, "")
            if "K" in extracted_name:
                extracted_name = "K"
            _add_param_from_summary(rpars, extracted_name, row)
    else:
        # Fall back to old underscore notation: S0_y1_A01, K_A01
        suffix_old = f"_{key}"
        rdf_old = trace_df[trace_df.index.str.endswith(suffix_old)]
        for name, row in rdf_old.iterrows():
            extracted_name = str(name).replace(suffix_old, "")
            if extracted_name.startswith("K_"):
                extracted_name = "K"
            _add_param_from_summary(rpars, extracted_name, row)

    # Shared-K mode: try group K parameter (new: K_{ctr}, old: K_ctr_{ctr})
    if ctr and "K" not in rpars:
        for candidate in [f"K_{ctr}", f"K_ctr_{ctr}"]:
            if candidate in trace_df.index:
                row = typing.cast("pd.Series", trace_df.loc[candidate])
                _add_param_from_summary(rpars, "K", row)
                break

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


def fit_binding_pymc_multi(  # noqa: C901,PLR0912,PLR0913,PLR0915,PLR0917
    results: dict[str, FitResult[MiniT]],
    scheme: PlateScheme,
    n_sd: float = 5.0,
    n_xerr: float = 1.0,
    ye_scaling: float = 1.0,
    n_samples: int = 2000,
    nuts_sampler: str = "default",
    cores: int | None = None,
    chains: int | None = None,
    *,
    x_error_model: Literal["deterministic", "random_walk"] = "deterministic",
    sigma_pip_prior: float = 0.02,
    ctr_free_k: bool = False,
    bg_noise: dict[int, ArrayF] | None = None,
    sample_ppc: bool = False,
    infer_gain: bool = False,
    robust: bool = False,
    compile_kwargs: dict[str, typing.Any] | None = None,
) -> xr.DataTree:
    """Multi-well PyMC with vectorized coords/dims.

    Parameters
    ----------
    results : dict[str, FitResult[MiniT]]
        Per-well initial fit results.
    scheme : PlateScheme
        Plate scheme defining control groups.
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
    cores : int | None
        Number of CPU cores for parallel chain execution. Defaults to all
        available cores.
    chains : int | None
        Number of MCMC chains. Defaults to 4.
    x_error_model : Literal["deterministic", "random_walk"]
        Model for x-error propagation (default: "deterministic").
    sigma_pip_prior : float
        Prior for random_walk sigma pipette error parameter.
    ctr_free_k : bool
        If True, each CTR replicate well gets its own independent K prior
        — identical to UNK well treatment, no hierarchical shrinkage.
        If False (default), all replicates of the same CTR share a single K.
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
    compile_kwargs : dict[str, typing.Any] | None
        Compilation options forwarded to the NUTS sampler backend
        (e.g. ``{"backend": "jax"}`` to use JAX JIT instead of Numba).

    Returns
    -------
    xr.DataTree
        The PyMC posterior trace.

    Raises
    ------
    ValueError
        If no valid dataset is found in results.
    """
    ds = next((r.dataset for r in results.values() if r.dataset), None)
    if ds is None:
        msg = "No valid dataset found in results."
        raise ValueError(msg)

    labels = list(ds.keys())
    is_ph = ds.is_ph

    # Build well list and group mapping from scheme
    wells_list = [
        key for key in results if results[key].result and results[key].dataset
    ]
    n_wells = len(wells_list)

    well_to_group: dict[str, str] = {}
    for grp_name, grp_well_set in scheme.names.items():
        for w in grp_well_set:
            well_to_group[w] = grp_name

    # Split into free wells (individual K) and named-group wells (shared K)
    groups: dict[str, list[str]] = {}
    free_wells: list[str] = []
    for w in wells_list:
        grp = well_to_group.get(w)
        if grp and not ctr_free_k:
            groups.setdefault(grp, []).append(w)
        else:
            free_wells.append(w)

    group_names = list(groups.keys())
    n_free = len(free_wells)

    # Collect per-well data (wells may have different titration lengths)
    xc_wells: dict[str, np.ndarray] = {}
    yc_wells: dict[str, dict[str, np.ndarray]] = {}
    ye_wells: dict[str, dict[str, np.ndarray]] = {}
    well_masks: dict[str, dict[str, np.ndarray]] = {}
    lens: list[int] = []
    for w in wells_list:
        first_da = results[w].dataset[labels[0]]  # type: ignore[index]
        xc_wells[w] = first_da.xc
        lens.append(len(xc_wells[w]))
        for lbl in labels:
            da = results[w].dataset[lbl]  # type: ignore[index]
            yc_wells.setdefault(lbl, {})[w] = da.yc
            y_err_full = da.y_errc
            if y_err_full.size == 0:
                y_err_full = np.ones_like(da.xc)
            ye_wells.setdefault(lbl, {})[w] = y_err_full
            well_masks.setdefault(lbl, {})[w] = da.mask

    # Reference x-grid from the longest well (for x-error model)
    max_steps = max(lens) if lens else 1
    longest_idx = int(np.argmax(np.array(lens))) if lens else 0
    longest_well = wells_list[longest_idx]
    xc = xc_wells[longest_well]
    first_ds = results[longest_well].dataset[labels[0]]  # type: ignore[index]
    x_errc_arr = first_ds.x_errc
    if x_errc_arr is None:
        x_errc_arr = np.zeros_like(xc, dtype=float)
    x_errc = x_errc_arr * n_xerr
    n_steps = max_steps

    # K init from preliminary fits
    def _k_log(results: dict[str, FitResult[MiniT]], w: str) -> float:
        r = results[w]
        assert r.result is not None  # noqa: S101
        return float(np.log(r.result.params["K"].value))

    k_group_log_init: dict[str, float] = {}
    for grp_name, grp_well_list in groups.items():
        vals = [_k_log(results, w) for w in grp_well_list]
        k_group_log_init[grp_name] = np.log(np.mean(vals)) if vals else np.log(15.0)

    # S0/S1 init from preliminary fits — per-well mu and sigma vectors

    # Coords
    coords: dict[str, list[str] | list[int]] = {
        "well": wells_list,
        "step": list(range(n_steps)),
        "step_diff": list(range(n_steps - 1)),
    }
    if n_free > 0:
        coords["free_well"] = free_wells

    with pm.Model(coords=coords):
        # -- x-error model --
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

        # -- K priors: per-well from LM fit for free wells, shared per group --
        k_segments: list[typing.Any] = []
        if n_free > 0:
            k_free_mu = np.array([_k_log(results, w) for w in free_wells])

            def _k_sigma(w: str) -> float:
                r = results[w].result
                if (
                    r is not None
                    and r.params["K"].stderr is not None
                    and r.params["K"].value != 0
                ):
                    return max(r.params["K"].stderr / r.params["K"].value, 0.3)  # type: ignore[no-any-return]
                return 0.3

            k_free_sigma = np.array([_k_sigma(w) for w in free_wells])
            k_free = pm.LogNormal(
                "K_free",
                mu=k_free_mu,
                sigma=k_free_sigma,
                dims="free_well",
            )
            k_segments.append(k_free)
        for grp_name in group_names:
            log_k_mu = k_group_log_init[grp_name]
            k_grp = pm.LogNormal(f"K_{grp_name}", mu=log_k_mu, sigma=0.3)
            k_segments.append(k_grp * pt.ones((len(groups[grp_name]),)))

        K_all = pm.math.concatenate(k_segments)  # (n_wells,)  # noqa: N806

        # -- Scale parameters: per-well priors from LM fit --
        s0_prior: dict[str, pm.Distribution] = {}
        s1_prior: dict[str, pm.Distribution] = {}

        def _param_sd(w: str, pname: str) -> float:
            r = results[w].result
            if r is not None and r.params[pname].stderr is not None:
                return max(r.params[pname].stderr, 0.2) * n_sd  # type: ignore[no-any-return]
            return 0.2 * n_sd

        for lbl in labels:
            s0_mu_vec = np.array([
                results[w].result.params[f"S0_{lbl}"].value  # type: ignore[union-attr]
                for w in wells_list
            ])
            s0_sd_vec = np.array([_param_sd(w, f"S0_{lbl}") for w in wells_list])
            s1_mu_vec = np.array([
                results[w].result.params[f"S1_{lbl}"].value  # type: ignore[union-attr]
                for w in wells_list
            ])
            s1_sd_vec = np.array([_param_sd(w, f"S1_{lbl}") for w in wells_list])
            s0_prior[lbl] = pm.Normal(
                f"S0_{lbl}", mu=s0_mu_vec, sigma=s0_sd_vec, dims="well"
            )
            s1_prior[lbl] = pm.Normal(
                f"S1_{lbl}", mu=s1_mu_vec, sigma=s1_sd_vec, dims="well"
            )

        # -- Noise model --
        if bg_noise is None:
            ye_mag: dict[str, pm.Distribution] = {
                lbl: pm.HalfNormal(f"ye_mag_{lbl}", sigma=ye_scaling) for lbl in labels
            }
        else:
            est_sigma = {
                lbl: float(np.sqrt(np.mean(np.array(bg_noise[i]) ** 2)))
                for i, lbl in enumerate(labels, start=1)
            }
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

        # -- Per-well binding model and likelihood --
        for i, w in enumerate(wells_list):
            n_i = lens[i]

            # x values for this well (full grid)
            if x_error_model == "random_walk":
                x_i = x_per_well[:n_i, i]  # (n_i,)
            else:
                x_i = pt.constant(xc_wells[w])  # (n_i,)

            K_i = K_all[i]  # noqa: N806

            for lbl in labels:
                m = well_masks[lbl][w]  # boolean mask

                # Binding model on full grid
                if is_ph:
                    exp_term = 10.0 ** (K_i - x_i)
                    frac = exp_term / (1.0 + exp_term)
                else:
                    frac = (x_i / K_i) / (1.0 + x_i / K_i)

                mu_i = s0_prior[lbl][i] + (s1_prior[lbl][i] - s0_prior[lbl][i]) * frac

                # Noise model per well
                if bg_noise is None:
                    sigma_i = ye_mag[lbl] * pt.constant(ye_wells[lbl][w])
                else:
                    y_pos_i = pm_math.maximum(1e-6, mu_i)
                    if infer_gain:
                        noise_var_i = (
                            floor_sq[lbl]
                            + gain_rv[lbl] * y_pos_i
                            + (rel_error_common * y_pos_i) ** 2
                        )
                    else:
                        noise_var_i = floor_sq[lbl] + (rel_error[lbl] * y_pos_i) ** 2
                    sigma_i = pm.Deterministic(
                        f"sigma_obs_{lbl}_{w}", pm_math.sqrt(noise_var_i)
                    )

                mu_obs = mu_i[m]
                sigma_obs = sigma_i[m]
                y_obs = yc_wells[lbl][w][m]

                # Likelihood
                if robust:
                    pm.StudentT(
                        f"y_likelihood_{lbl}_{w}",
                        nu=3.0,
                        mu=mu_obs,
                        sigma=sigma_obs,
                        observed=y_obs,
                    )
                else:
                    pm.Normal(
                        f"y_likelihood_{lbl}_{w}",
                        mu=mu_obs,
                        sigma=sigma_obs,
                        observed=y_obs,
                    )

        trace: xr.DataTree = pm.sample(
            n_samples,
            target_accept=0.8,
            return_inferencedata=True,
            compile_kwargs=compile_kwargs,
            **_pymc_sample_parallel_args(nuts_sampler, cores, chains),
        )

        if sample_ppc:
            pm.sample_posterior_predictive(trace, extend_inferencedata=True)

    return trace
