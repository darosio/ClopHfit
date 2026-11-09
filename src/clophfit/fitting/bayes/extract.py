"""Reading fit results and posterior summaries out of PyMC traces."""

from __future__ import annotations

import contextlib
import copy
import typing
from typing import Any, Literal

import arviz as az  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
import xarray as xr
from lmfit import Parameters  # type: ignore[import-untyped]
from matplotlib import figure

from clophfit.fitting.core import (
    N_BOOT,
)  # local to avoid circular import
from clophfit.fitting.data_structures import (
    DataArray,
    Dataset,
    FitResult,
    MultiFitResult,
    _Result,
)
from clophfit.fitting.plotting import PlotParameters, plot_fit

if typing.TYPE_CHECKING:
    from collections.abc import Mapping

    from clophfit.prtecan import PlateScheme

from clophfit.fitting.bayes.axis import (
    _X_TRUE_INDEX_RE,
)
from clophfit.fitting.bayes.likelihood import (
    _add_param_from_summary,
    _compute_weighted_residuals,
)
from clophfit.fitting.bayes.priors import (
    KD_RAW_PREFIX,
    _ctr_param_name,
)


def _az_summary(trace: typing.Any) -> typing.Any:  # ruff: ignore[any-type]
    """Summarise a trace with a 94% HDI, across ArviZ versions.

    ArviZ 1.x defaults to an 89% ETI and needs ``ci_kind``/``ci_prob`` to give
    an HDI at all; 0.x took ``hdi_prob``. Ask for the interval this project
    reports rather than accepting whichever default is installed.
    """
    for kwargs in ({"ci_kind": "hdi", "ci_prob": 0.94}, {"hdi_prob": 0.94}, {}):
        try:
            return az.summary(trace, **kwargs)
        except TypeError:
            continue
    return az.summary(trace)


def _trace_summary_df(
    trace_or_df: xr.DataTree | MultiFitResult | pd.DataFrame,
) -> pd.DataFrame:
    """Return a numeric ArviZ summary DataFrame from a trace or summary."""
    if isinstance(trace_or_df, MultiFitResult):
        trace_or_df = trace_or_df.trace
    rdf = (
        _az_summary(trace_or_df)
        if isinstance(trace_or_df, xr.DataTree)
        else trace_or_df
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
    except Exception:  # ruff: ignore[blind-except]
        return None
    return value if np.isfinite(value) else None


def label_representatives(
    datasets: typing.Iterable[Mapping[str, Any] | None],
) -> dict[str, Any]:
    """One representative curve per label, taken across every well.

    The structured noise path needs a curve per label to fall back on when no
    floor hint is supplied. Taking them from a single well is wrong once
    per-label detection is active: that well may be single-label, and the model
    then carries no entry for a label the plate is still fitting. The label list
    is already a union over wells, so the noise model has to be one too.

    Parameters
    ----------
    datasets : typing.Iterable[Mapping[str, Any] | None]
        Per-well datasets, in a deterministic order. ``None`` entries - wells
        with no usable fit - are skipped.

    Returns
    -------
    dict[str, Any]
        Label to the first data array carrying it, keyed by string.
    """
    reps: dict[str, Any] = {}
    for ds in datasets:
        if ds is None:
            continue
        for lbl, da in ds.items():
            reps.setdefault(str(lbl), da)
    return reps


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


def extract_fit(  # ruff: ignore[too-many-arguments, complex-structure, too-many-branches]
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
    well_key : str
        When provided, per-well x posteriors (``x_true[step, well_key]``)
        are used instead of the global ``x_true``.  Pass the well identifier
        for xrw fits so each well's .dat/.png uses its own inferred pH axis.
    raw_trace : xr.DataTree | None
        Raw multi-well PyMC trace for xarray-based x extraction, bypassing
        potential ``az.summary`` indexing bugs.  When ``None``, falls back
        to DataFrame-based extraction from *trace_df*.
    global_p_names : typing.Iterable[str]
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
        # The sampled variable behind a Deterministic Kd: K itself is reported.
        if n_str.startswith(f"{KD_RAW_PREFIX}K"):
            continue
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


def _update_dataset_yerr_from_sigma_obs(  # ruff: ignore[complex-structure, too-many-branches]
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
            if len(parts) == 2 and well_key is not None:  # ruff: ignore[magic-value-comparison]
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
