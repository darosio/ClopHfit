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

from .core import FitResult  # Local import to avoid cycle

if typing.TYPE_CHECKING:
    from clophfit.clophfit_types import ArrayF, ArrayMask


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


class _Result:
    def __init__(self, params: Parameters) -> None:
        self.params = params


def fit_binding_odr(fr: FitResult) -> FitResult:
    """Analyze multi-label titration datasets using ODR."""
    if fr.result is None or fr.dataset is None:
        return FitResult()
    params = fr.result.params
    ds = copy.deepcopy(fr.dataset)
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
    # reassign x_err and y_err to ds
    start_idx = 0
    for da in ds.values():
        end_idx = start_idx + len(da.y)
        da.x_errc[da.mask] = 2 * np.abs(output.delta[start_idx:end_idx])
        da.y_errc[da.mask] = 2 * np.abs(output.eps[start_idx:end_idx])
        start_idx = end_idx
    # Update the parameters with results from ODR
    p_names = ["K"]
    for lbl in ds:
        p_names.append(f"S0_{lbl}")
        p_names.append(f"S1_{lbl}")
    params = Parameters()
    for name, value, error in zip(p_names, output.beta, output.sd_beta, strict=True):
        params.add(name, value=value)
        params[name].stderr = error
    fig = figure.Figure()
    ax = fig.add_subplot(111)
    plot_fit(ax, ds, params, nboot=20, pp=PlotParameters(ds.is_ph))
    return FitResult(fig, _Result(params), output, ds)


def fit_binding_odr_recursive(
    fr: FitResult, max_iterations: int = 15, tol: float = 0.1
) -> FitResult:
    """Analyze multi-label titration datasets using ODR."""
    result = copy.deepcopy(fr)
    if result.result is None or result.dataset is None:
        return FitResult()
    # Initial fit
    ro = fit_binding_odr(result)
    residual_variance = ro.mini.res_var if ro.mini else 0.0
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
    output: odr.Output, threshold: float = 2.0, plot_z_scores: bool = False
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
    fr: FitResult, tol: float = 0.5, threshold: float = 2.0
) -> FitResult:
    """Analyze multi-label titration datasets using ODR."""
    result = copy.deepcopy(fr)
    if result.result is None or result.dataset is None:
        return FitResult()
    # Initial fit
    ro = fit_binding_odr_recursive(result, tol=tol)
    omask = outlier(ro.mini, threshold)
    while omask.any() and ro.dataset:
        result.dataset.apply_mask(~omask)
        ro = fit_binding_odr_recursive(result, tol=tol)
        omask = outlier(ro.mini, 3.0)
    return ro
