"""Orthogonal Distance Regression (ODR) utilities and fitting pipeline."""

from __future__ import annotations

import copy
import typing

import matplotlib.pyplot as plt
import numpy as np
import odrpack
from lmfit import Parameters  # type: ignore[import-untyped]
from matplotlib import figure

from clophfit.fitting.models import binding_1site
from clophfit.fitting.plotting import PlotParameters, plot_fit
from clophfit.fitting.utils import identify_outliers_zscore, parse_remove_outliers
from clophfit.utils import weights_from_sigma

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
    pars: ArrayF | list[float], x: ArrayF, dataset_lengths: list[int], *, is_ph: bool
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
        model_output = binding_1site(current_x, K, S0, S1, is_ph=is_ph)
        results.append(model_output)
        start_idx = end_idx
    return np.concatenate(results)


def fit_binding_odr(  # noqa: C901, PLR0915
    ds_or_fr: Dataset | FitResult[MiniT],
    *,
    reweight: bool = False,
    remove_outliers: str | None = None,
    max_iter: int = 15,
    tol: float = 0.1,
) -> FitResult[odrpack.OdrResult]:
    """Analyze multi-label titration datasets using ODR.

    Parameters
    ----------
    ds_or_fr : Dataset | FitResult[MiniT]
        Either a Dataset (will run initial LS fit) or a FitResult with initial params.
    reweight : bool
        Whether to perform iterative reweighting (recursive ODR).
    remove_outliers : str | None
        Outlier removal configuration, e.g., "zscore:2.5" or "zscore:2.5:5".
        If set, performs recursive ODR and masks outliers iteratively.
    max_iter : int
        Maximum number of iterations for recursive ODR.
    tol : float
        Convergence tolerance for residual variance.

    Returns
    -------
    FitResult[odrpack.OdrResult]
        ODR fitting results. Residuals are WEIGHTED by the ORIGINAL y_err
        (before ODR modified it), making them comparable to LM residuals.
    """
    # Handle both Dataset and FitResult inputs
    fr = (
        fit_binding_glob(ds_or_fr)
        if isinstance(ds_or_fr, Dataset)
        else copy.deepcopy(ds_or_fr)
    )

    if fr.result is None or fr.dataset is None:
        return FitResult()

    def _single_odr_fit(current_fr: FitResult[MiniT]) -> FitResult[odrpack.OdrResult]:
        if current_fr.result is None or current_fr.dataset is None:
            return FitResult()
        params = current_fr.result.params
        ds = copy.deepcopy(current_fr.dataset)

        original_y_err = np.concatenate([da.y_err for da in ds.values()])

        for da in ds.values():
            shot_factor = 1 + np.sqrt(np.abs(da.yc))
            da.y_err = (
                da.y_errc * shot_factor if da.y_errc.size > 0 else 1.0 * shot_factor
            )

        dataset_lengths = [len(da.y) for da in ds.values()]
        x_data, y_data, x_err, y_err = ds.concatenate_data()
        weight_x = weights_from_sigma(x_err)
        weight_y = weights_from_sigma(y_err)

        initial_params = [params["K"].value]
        for lbl in ds:
            initial_params.extend([
                params[f"S0_{lbl}"].value,
                params[f"S1_{lbl}"].value,
            ])

        def combined_model_odr(x: ArrayF, p: ArrayF) -> ArrayF:
            return generalized_combined_model(p, x, dataset_lengths, is_ph=ds.is_ph)

        output = odrpack.odr_fit(
            combined_model_odr,
            x_data,
            y_data,
            initial_params,
            weight_x=weight_x,
            weight_y=weight_y,
        )

        start_idx = 0
        for da in ds.values():
            end_idx = start_idx + len(da.y)
            da.x_errc[da.mask] = 2 * np.abs(output.delta[start_idx:end_idx])
            da.y_errc[da.mask] = 2 * np.abs(output.eps[start_idx:end_idx])
            start_idx = end_idx

        p_names = ["K"]
        for lbl in ds:
            p_names.extend((f"S0_{lbl}", f"S1_{lbl}"))
        params = Parameters()
        for name, value, error in zip(
            p_names, output.beta, output.sd_beta, strict=True
        ):
            params.add(name, value=value)
            params[name].stderr = error

        residuals = _compute_odr_residuals(ds, params, original_y_err)

        fig = figure.Figure()
        ax = fig.add_subplot(111)
        plot_fit(ax, ds, params, nboot=20, pp=PlotParameters(ds.is_ph))
        return FitResult(
            fig, _Result(params, residual=residuals, redchi=output.res_var), output, ds
        )

    # Initial fit
    ro = _single_odr_fit(fr)

    # If neither reweighting nor outlier removal is requested, return early
    if not reweight and not remove_outliers:
        return ro

    # Parse outlier config if present
    threshold = 2.0
    if remove_outliers:
        _method, threshold, _min_keep = parse_remove_outliers(remove_outliers)

    residual_variance = ro.mini.res_var if ro.mini else 0.0

    for _ in range(max_iter):
        if remove_outliers and ro.mini:
            omask = outlier(ro.mini, threshold=threshold)
            if omask.any() and ro.dataset:
                # Apply mask to the starting FitResult's dataset to exclude points
                fr.dataset.apply_mask(~omask)

        rn = _single_odr_fit(fr)
        if rn.mini and rn.mini.res_var == 0:
            rn = ro
            break

        if rn.mini and residual_variance - rn.mini.res_var < tol:
            if not remove_outliers:
                break
            # If removing outliers, also require no new outliers to converge
            if remove_outliers:
                omask_new = outlier(rn.mini, threshold=threshold)
                if not omask_new.any():
                    ro = rn
                    break

        residual_variance = rn.mini.res_var if rn.mini else 0.0
        ro = rn
        fr = copy.deepcopy(ro)  # update starting point for next iteration

    return ro


def outlier(
    output: odrpack.OdrResult, *, threshold: float = 2.0, plot_z_scores: bool = False
) -> ArrayMask:
    """Identify outliers."""
    residuals_x = output.delta
    residuals_y = output.eps
    residuals = np.sqrt(residuals_x**2 + residuals_y**2)

    if plot_z_scores:
        z_scores = np.abs((residuals - np.mean(residuals)) / np.std(residuals))
        plt.scatter(range(len(z_scores)), z_scores)
        plt.axhline(y=threshold, color="r", linestyle="-")
        plt.title("Z-scores")

    return identify_outliers_zscore(residuals, threshold=threshold)
