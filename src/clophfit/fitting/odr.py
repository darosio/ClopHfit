"""Orthogonal Distance Regression (ODR) utilities and fitting pipeline."""

from __future__ import annotations

import copy
import typing

import matplotlib.pyplot as plt
import numpy as np
from lmfit import Parameters  # type: ignore[import-untyped]
from matplotlib import figure
from scipy import odr

from clophfit.fitting.models import binding_1site
from clophfit.fitting.plotting import PlotParameters, plot_fit

from .core import fit_binding_glob
from .data_structures import Dataset, FitResult, MiniT, _Result

if typing.TYPE_CHECKING:
    from clophfit.clophfit_types import ArrayF, ArrayMask


def _compute_odr_residuals(
    ds: Dataset, params: Parameters, original_y_err: np.ndarray
) -> np.ndarray:
    """Compute weighted residuals from ODR output using ORIGINAL y_err.

    Uses the original (pre-ODR) y_err for weighting, not the ODR-estimated y_err.
    ODR estimates y_err from residuals (y_err = 2*|eps|), so using ODR's y_err
    would be circular and always produce ±0.5.

    Parameters
    ----------
    ds : Dataset
        The dataset (with ODR-modified y_err, but we don't use it)
    params : Parameters
        Fitted parameters from ODR
    original_y_err : np.ndarray
        The y_err BEFORE ODR modified it (physics-based or user-provided)

    Returns
    -------
    np.ndarray
        Weighted residuals = raw_residual / original_y_err
    """
    residuals_list: list[np.ndarray] = []
    K = params["K"].value  # noqa: N806

    idx = 0
    for lbl, da in ds.items():
        n_points = len(da.y)
        # Use masked values (.x, .y) for consistency
        model = binding_1site(
            da.x,
            K,
            params[f"S0_{lbl}"].value,
            params[f"S1_{lbl}"].value,
            is_ph=ds.is_ph,
        )
        raw_residuals = da.y - model
        # Weight by ORIGINAL y_err (not ODR-modified)
        label_y_err = original_y_err[idx : idx + n_points]
        weighted = raw_residuals / label_y_err
        residuals_list.append(weighted)
        idx += n_points

    return np.concatenate(residuals_list)


def format_estimate(
    value: float, error: float, significant_digit_limit: int = 5
) -> str:
    """Format the value and its associated error into "{value} ± {error}" string."""
    large_number = np.exp(significant_digit_limit)
    small_number = np.exp(-significant_digit_limit)
    significant_digits = max(0, -int(np.floor(np.log10(abs(error)))))
    use_scientific = (
        significant_digits > significant_digit_limit
        or abs(value) > large_number
        or abs(value) < small_number
    )
    formatted_value = (
        f"{value:.{significant_digits + 1}e}"
        if use_scientific
        else f"{value:.{significant_digits + 1}f}"
    )
    formatted_error = (
        f"{error:.{significant_digits + 1}e}"
        if use_scientific
        else f"{error:.{significant_digits + 1}f}"
    )
    return f"{formatted_value} ± {formatted_error}"


def generalized_combined_model(
    pars: list[float], x: ArrayF, dataset_lengths: list[int]
) -> ArrayF:
    """Handle multiple datasets with different lengths and masks."""
    start_idx = 0
    results = []
    # The shared K parameter is always the first parameter
    K = pars[0]  # noqa: N806
    for i, length in enumerate(dataset_lengths):
        end_idx = start_idx + length
        current_x = x[start_idx:end_idx]
        S0 = pars[1 + 2 * i]  # S0 for dataset i # noqa: N806
        S1 = pars[2 + 2 * i]  # S1 for dataset i # noqa: N806
        model_output = binding_1site(current_x, K, S0, S1, is_ph=True)
        results.append(model_output)
        start_idx = end_idx
    return np.concatenate(results)


def fit_binding_odr(
    ds_or_fr: Dataset | FitResult[MiniT],
) -> FitResult[odr.Output]:
    """Analyze multi-label titration datasets using ODR.

    Parameters
    ----------
    ds_or_fr : Dataset | FitResult[MiniT]
        Either a Dataset (will run initial LS fit) or a FitResult with initial params.

    Returns
    -------
    FitResult[odr.Output]
        ODR fitting results. Residuals are WEIGHTED by the ORIGINAL y_err
        (before ODR modified it), making them comparable to LM residuals.
    """
    # Handle both Dataset and FitResult inputs
    fr = fit_binding_glob(ds_or_fr) if isinstance(ds_or_fr, Dataset) else ds_or_fr

    if fr.result is None or fr.dataset is None:
        return FitResult()

    params = fr.result.params
    ds = copy.deepcopy(fr.dataset)

    # Store ORIGINAL y_err from LM fit (for residual weighting)
    # These are physics-based errors, not inflated by shot_factor
    original_y_err = np.concatenate([da.y_err for da in ds.values()])

    # Apply shot_factor for ODR fitting (not for residual weighting)
    for da in ds.values():
        # even if da.y_err is set to [1,1,..] array y_errc remains []
        shot_factor = 1 + np.sqrt(np.abs(da.yc))
        da.y_err = da.y_errc * shot_factor if da.y_errc.size > 0 else 1.0 * shot_factor

    # Collect dataset lengths
    dataset_lengths = [len(da.y) for da in ds.values()]
    x_data, y_data, x_err, y_err = ds.concatenate_data()
    data = odr.RealData(x_data, y_data, sx=x_err, sy=y_err)
    # Initial parameters setup
    initial_params = [params["K"].value]
    for lbl in ds:
        initial_params.extend([params[f"S0_{lbl}"].value, params[f"S1_{lbl}"].value])

    # Define the combined model
    def combined_model_odr(p: list[float], x: ArrayF) -> ArrayF:
        return generalized_combined_model(p, x, dataset_lengths)

    combined_model = odr.Model(combined_model_odr)  # type: ignore[arg-type]
    odr_obj = odr.ODR(data, combined_model, beta0=initial_params)
    output = odr_obj.run()
    # reassign x_err and y_err to ds (ODR-estimated values)
    start_idx = 0
    for da in ds.values():
        end_idx = start_idx + len(da.y)
        da.x_errc[da.mask] = 2 * np.abs(output.delta[start_idx:end_idx])
        da.y_errc[da.mask] = 2 * np.abs(output.eps[start_idx:end_idx])
        start_idx = end_idx
    # Update the parameters with results from ODR
    p_names = ["K"]
    for lbl in ds:
        p_names.extend((f"S0_{lbl}", f"S1_{lbl}"))
    params = Parameters()
    for name, value, error in zip(p_names, output.beta, output.sd_beta, strict=True):
        params.add(name, value=value)
        params[name].stderr = error

    # Compute weighted residuals using ORIGINAL y_err (not ODR-modified)
    residuals = _compute_odr_residuals(ds, params, original_y_err)

    # Create figure and result
    fig = figure.Figure()
    ax = fig.add_subplot(111)
    plot_fit(ax, ds, params, nboot=20, pp=PlotParameters(ds.is_ph))
    return FitResult(
        fig, _Result(params, residual=residuals, redchi=output.res_var), output, ds
    )


def fit_binding_odr_recursive(
    ds_or_fr: Dataset | FitResult[MiniT],
    max_iterations: int = 15,
    tol: float = 0.1,
) -> FitResult[odr.Output]:
    """Analyze multi-label titration datasets using iterative ODR.

    Parameters
    ----------
    ds_or_fr : Dataset | FitResult[MiniT]
        Either a Dataset (will run initial LS fit) or a FitResult with initial params.
    max_iterations : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance for residual variance.

    Returns
    -------
    FitResult[odr.Output]
        ODR fitting results.
    """
    # Handle both Dataset and FitResult inputs
    if isinstance(ds_or_fr, Dataset):
        fr = fit_binding_glob(ds_or_fr)
    else:
        fr = copy.deepcopy(ds_or_fr)

    if fr.result is None or fr.dataset is None:
        return FitResult()

    # Initial fit
    ro = fit_binding_odr(fr)
    residual_variance = ro.mini.res_var if isinstance(ro.mini, odr.Output) else 0.0
    for _ in range(max_iterations):
        rn = fit_binding_odr(ro)
        if rn.mini and rn.mini.res_var == 0:
            rn = ro
            break
        # Check convergence
        if rn.mini and residual_variance - rn.mini.res_var < tol:
            break
        residual_variance = rn.mini.res_var if rn.mini else 0.0
        ro = rn
    return rn


def outlier(
    output: odr.Output, *, threshold: float = 2.0, plot_z_scores: bool = False
) -> ArrayMask:
    """Identify outliers."""
    residuals_x = output.delta
    residuals_y = output.eps
    residuals = np.sqrt(residuals_x**2 + residuals_y**2)
    z_scores = np.abs((residuals - np.mean(residuals)) / np.std(residuals))
    if plot_z_scores:
        plt.scatter(range(len(z_scores)), z_scores)
        plt.axhline(y=threshold, color="r", linestyle="-")
        plt.title("Z-scores")
    outliers: ArrayMask = z_scores > threshold
    return outliers


def fit_binding_odr_recursive_outlier(
    ds_or_fr: Dataset | FitResult[MiniT],
    tol: float = 0.5,
    threshold: float = 2.0,
) -> FitResult[odr.Output]:
    """Analyze multi-label titration datasets using ODR with outlier removal.

    Parameters
    ----------
    ds_or_fr : Dataset | FitResult[MiniT]
        Either a Dataset (will run initial LS fit) or a FitResult with initial params.
    tol : float
        Convergence tolerance for residual variance.
    threshold : float
        Z-score threshold for outlier detection.

    Returns
    -------
    FitResult[odr.Output]
        ODR fitting results.
    """
    # Handle both Dataset and FitResult inputs
    if isinstance(ds_or_fr, Dataset):
        fr = fit_binding_glob(ds_or_fr)
    else:
        fr = copy.deepcopy(ds_or_fr)

    if fr.result is None or fr.dataset is None:
        return FitResult()

    # Initial fit
    ro = fit_binding_odr_recursive(fr, tol=tol)
    if ro.mini:
        omask = outlier(ro.mini, threshold=threshold)
    while omask.any() and ro.dataset:
        fr.dataset.apply_mask(~omask)
        ro = fit_binding_odr_recursive(fr, tol=tol)
        if ro.mini:
            omask = outlier(ro.mini, threshold=3.0)
    return ro
