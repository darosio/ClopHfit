"""Pipeline orchestrators for fitting multistage workflows (e.g., FGLS)."""

import logging
import typing

import numpy as np

from clophfit.fitting.core import fit_binding_glob
from clophfit.fitting.data_structures import Dataset, FitResult
from clophfit.fitting.error_models import ComprehensiveErrorModel
from clophfit.fitting.residuals import collect_multi_residuals
from clophfit.fitting.utils import fit_gain_and_rel_error_from_residuals

logger = logging.getLogger(__name__)


def fgls_plate_fit(
    datasets: dict[str, Dataset],
    sigma_floor: dict[str, float],
    *,
    first_pass_method: str = "huber",  # noqa: S107
    second_pass_method: str = "lm",  # noqa: S107
) -> tuple[dict[str, FitResult[typing.Any]], dict[str, tuple[float, float, float]]]:
    """Two-stage Feasible Generalized Least Squares (FGLS) plate fit.

    1. First-pass fit (typically robust like 'huber') on each well.
    2. Extract residuals globally and calibrate the comprehensive error model,
       anchoring the constant noise term to the provided ``sigma_floor``.
    3. Second-pass fit using the exact pooled weights derived from the model.

    Parameters
    ----------
    datasets : dict[str, Dataset]
        The dataset dictionary keyed by well name.
    sigma_floor : dict[str, float]
        Known read-noise floor per label (e.g. from buffer wells).
    first_pass_method : str
        Method for the first-pass fit (default 'huber').
    second_pass_method : str
        Method for the second-pass, calibrated fit (default 'lm').

    Returns
    -------
    tuple[dict[str, FitResult[typing.Any]], dict[str, tuple[float, float, float]]]
        Final fit results and the calibrated error model parameters
        (sigma_read, gain, alpha) for each label.
    """
    # 1. First Pass
    logger.info("Starting FGLS Pass 1: %s fit", first_pass_method)
    first_pass_results = {}
    for well, ds in datasets.items():
        first_pass_results[well] = fit_binding_glob(ds, method=first_pass_method)

    # 2. Collect residuals for calibration
    logger.info("Starting FGLS Noise Calibration")
    df_res = collect_multi_residuals(first_pass_results)

    # 3. Calibrate noise parameters using physically anchored OLS
    try:
        gains, alphas = fit_gain_and_rel_error_from_residuals(df_res, sigma_floor)
    except ValueError as e:
        logger.warning(
            "FGLS calibration failed (%s). Falling back to basic proportional error.", e
        )
        gains = dict.fromkeys(sigma_floor, 0.0)
        alphas = dict.fromkeys(sigma_floor, 0.03)

    # Format the noise parameters for the return value and logging
    noise_params = {}
    for lbl, sr in sigma_floor.items():
        gain = gains.get(lbl, 0.0)
        alpha = alphas.get(lbl, 0.0)
        noise_params[lbl] = (sr, gain, alpha)

        logger.info(
            "Calibrated Noise [%s]: sigma=%.2f, gain=%.3f, alpha=%.3f",
            lbl,
            sr,
            gain,
            alpha,
        )

    # Instantiate the global error model
    global_error_model = ComprehensiveErrorModel(
        sigma_read=typing.cast("dict[int | str, float]", sigma_floor),
        gain=typing.cast("dict[int | str, float]", gains),
        rel_error=typing.cast("dict[int | str, float]", alphas),
    )

    # 4. Second Pass
    logger.info(
        "Starting FGLS Pass 2: %s fit with calibrated weights", second_pass_method
    )
    final_results = {}
    for well, ds in datasets.items():
        # Inject the new model-derived weights for the full array (yc)
        for lbl, da in ds.items():
            var = global_error_model.compute_variance(da.yc, lbl)
            var = np.maximum(1.0, var)  # avoid division by zero
            da.y_err = np.sqrt(var)

        final_results[well] = fit_binding_glob(ds, method=second_pass_method)

    return final_results, noise_params
