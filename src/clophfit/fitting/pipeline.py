"""Pipeline orchestrators for fitting multistage workflows (e.g., FGLS)."""

import logging
import typing

from clophfit.fitting.bayes import fit_binding_pymc
from clophfit.fitting.core import fit_binding_glob
from clophfit.fitting.data_structures import Dataset, FitResult
from clophfit.fitting.errors import InsufficientDataError
from clophfit.fitting.odr import fit_binding_odr
from clophfit.fitting.residuals import collect_multi_residuals
from clophfit.fitting.utils import (
    assign_error_model,
    fit_gain_and_rel_error_from_residuals,
)

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
        try:
            first_pass_results[well] = fit_binding_glob(ds, method=first_pass_method)
        except InsufficientDataError:
            logger.warning("Skip FGLS Pass 1 fit for well %s.", well)
            first_pass_results[well] = FitResult()

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

    # 4. Second Pass
    logger.info(
        "Starting FGLS Pass 2: %s fit with calibrated weights", second_pass_method
    )
    final_results = {}
    for well, ds in datasets.items():
        # Update ds with new variance model
        ds_updated = assign_error_model(
            ds, sigma_floor=sigma_floor, gain=gains, rel_error=alphas
        )

        try:
            final_results[well] = fit_binding_glob(
                ds_updated, method=second_pass_method
            )
        except InsufficientDataError:
            logger.warning("Skip FGLS Pass 2 fit for well %s.", well)
            final_results[well] = FitResult()

    return final_results, noise_params


def fit_plate(
    datasets: dict[str, Dataset],
    method: str = "",
    **kwargs: typing.Any,  # noqa: ANN401
) -> dict[str, FitResult[typing.Any]]:
    """Run a single-pass fit on an entire plate of datasets.

    Parameters
    ----------
    datasets : dict[str, Dataset]
        A mapping of well keys (e.g. 'A01') to `Dataset` objects.
    method : str
        The fitting method to use: 'lm' (default), 'huber', 'odr', or 'mcmc'.
        Other methods supported by :func:`clophfit.fitting.core.fit_binding_glob`
        may also be used.
    **kwargs : typing.Any
        Additional keyword arguments passed to the specific fitting function.

    Returns
    -------
    dict[str, FitResult[typing.Any]]
        A dictionary mapping well keys to their corresponding `FitResult`.
    """
    results: dict[str, FitResult[typing.Any]] = {}

    if method == "odr":
        for well, ds in datasets.items():
            try:
                results[well] = fit_binding_odr(ds, **kwargs)
            except InsufficientDataError:
                logger.warning("Skip ODR fit for well %s.", well)
                results[well] = FitResult()
    elif method == "mcmc":
        for well, ds in datasets.items():
            try:
                results[well] = fit_binding_pymc(ds, **kwargs)
            except InsufficientDataError:
                logger.warning("Skip MCMC fit for well %s.", well)
                results[well] = FitResult()
    else:
        if not method:
            method = "lm"
        print(method)
        for well, ds in datasets.items():
            try:
                results[well] = fit_binding_glob(ds, method=method, **kwargs)
            except InsufficientDataError:
                logger.warning("Skip fit for well %s.", well)
                results[well] = FitResult()

    return results
