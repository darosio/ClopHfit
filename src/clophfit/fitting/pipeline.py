"""Pipeline orchestrators for fitting multistage workflows (e.g., FGLS)."""

import logging
import typing

import pandas as pd

from clophfit.fitting.core import fit_binding_glob
from clophfit.fitting.data_structures import (
    Dataset,
    FitResult,
    NoiseModelParams,
    PlateNoiseModel,
)
from clophfit.fitting.errors import InsufficientDataError
from clophfit.fitting.model_validation import residuals_from_fit_results
from clophfit.fitting.models import binding_1site
from clophfit.fitting.utils import (
    compute_plate_slopes,
    fit_gain_from_residuals,
    fit_noise_model_nnls,
    fit_ph_slope_noise,
    fit_rel_error_from_residuals,
)

logger = logging.getLogger(__name__)


def _noise_params_converged(
    old: PlateNoiseModel,
    new: PlateNoiseModel,
    tol: float = 1e-3,
) -> bool:
    """Check whether gain and alpha converged across labels."""
    for lbl in new:
        if lbl not in old:
            return False
        for attr in ("gain", "alpha"):
            old_val = getattr(old[lbl], attr)
            new_val = getattr(new[lbl], attr)
            denom = max(old_val, new_val, 1e-12)
            if abs(new_val - old_val) / denom > tol:
                return False
    return True


def _plate_noise_model_from_nnls(
    sigma_floor: dict[str, float],
    gains: dict[str, float],
    alphas: dict[str, float],
    sigma_ph: float = 0.0,
) -> PlateNoiseModel:
    """Build a PlateNoiseModel from NNLS output dicts."""
    model = PlateNoiseModel()
    for lbl in sigma_floor:
        model[lbl] = NoiseModelParams(
            sigma_floor=sigma_floor.get(lbl, 0.0),
            gain=gains.get(lbl, 0.0),
            alpha=alphas.get(lbl, 0.0),
            sigma_ph=sigma_ph,
        )
    return model


def calibrate_noise_robust(
    residuals: pd.DataFrame,
    sigma_floor: dict[str, float],
    *,
    p_threshold: float = 0.9,
    min_keep: int = 3,
) -> PlateNoiseModel:
    """Calibrate a per-label noise model from outlier-screened residuals.

    Drops points whose posterior outlier probability exceeds *p_threshold*
    (from a PyMC mixture fit) and then estimates ``gain`` and ``alpha`` per
    label with the single-term moment estimators
    (:func:`~clophfit.fitting.utils.fit_gain_from_residuals`,
    :func:`~clophfit.fitting.utils.fit_rel_error_from_residuals`) on the
    retained points. Screening with the mixture's ``p_outlier`` keeps
    outliers from inflating the estimate, while the two single-term estimators
    avoid the gain/alpha collinearity of the joint NNLS over narrow titration
    ranges.

    Parameters
    ----------
    residuals : pd.DataFrame
        Canonical residual table (e.g. ``MultiFitResult.residuals`` from a
        mixture fit). Must have ``label``, ``raw_res``, ``yhat`` columns; a
        ``p_outlier`` column enables screening (otherwise all points
        are used).
    sigma_floor : dict[str, float]
        Known read-noise floor per label, e.g. ``tit.bg_noise``. Used as the
        fixed floor and copied into the returned model.
    p_threshold : float, optional
        Posterior outlier probability above which a point is dropped.
    min_keep : int, optional
        Per label, if screening would retain fewer than this many points the
        full (unscreened) set is used instead.

    Returns
    -------
    PlateNoiseModel
        Per-label model with ``sigma_floor`` from *sigma_floor* and calibrated
        ``gain``/``alpha``.
    """
    if "p_outlier" in residuals.columns:
        kept: list[pd.DataFrame] = []
        for _label, group in residuals.groupby("label", observed=True):
            inliers = group[group["p_outlier"].fillna(0.0) < p_threshold]
            kept.append(group if len(inliers) < min_keep else inliers)
        clean = pd.concat(kept, ignore_index=True) if kept else residuals
    else:
        clean = residuals
    gains = fit_gain_from_residuals(clean, sigma_floor)
    alphas = fit_rel_error_from_residuals(clean, sigma_floor)
    return _plate_noise_model_from_nnls(dict(sigma_floor), gains, alphas)


def fgls_plate_fit(  # noqa: PLR0913
    datasets: dict[str, Dataset],
    sigma_floor: dict[str, float],
    *,
    first_pass_method: str = "huber",  # noqa: S107
    second_pass_method: str = "lm",  # noqa: S107
    max_iter: int = 3,
    tol: float = 1e-3,
) -> tuple[dict[str, FitResult[typing.Any]], PlateNoiseModel]:
    """Run iterative Feasible Generalized Least Squares (FGLS) on a plate.

    1. First-pass fit (robust) on each well with existing ``y_errc``.
    2. Calibrate noise model from residuals, anchoring floor to *sigma_floor*.
    3. Re-apply calibrated weights, re-fit, re-calibrate — iterating until
       gain and alpha converge or *max_iter* is reached.
    4. Return final fits and the noise model from the last calibration.

    Parameters
    ----------
    datasets : dict[str, Dataset]
        Plate datasets keyed by well name.
    sigma_floor : dict[str, float]
        Known read-noise floor per label (e.g. from buffer wells).
    first_pass_method : str
        Method for the first-pass fit (default ``"huber"``).
    second_pass_method : str
        Method for subsequent passes (default ``"lm"``).
    max_iter : int
        Maximum FGLS iterations (default 3).
    tol : float
        Relative tolerance for gain/alpha convergence (default 1e-3).

    Returns
    -------
    tuple[dict[str, FitResult[typing.Any]], PlateNoiseModel]
        Final fit results keyed by well, and the converged (or last)
        calibrated noise model.
    """
    noise_model: PlateNoiseModel | None = None
    results: dict[str, FitResult[typing.Any]] = {}

    for iteration in range(max_iter):
        # Choose method
        method = first_pass_method if iteration == 0 else second_pass_method

        # Apply current noise model (if any) and compute slopes
        if iteration == 0:
            current_ds = datasets
            plate_slopes = None
        else:
            plate_slopes = compute_plate_slopes(results)
            current_ds = noise_model.apply_to_plate(datasets, plate_slopes)  # type: ignore[union-attr]

        logger.info("FGLS iteration %d: %s fit", iteration + 1, method)

        # Fit all wells
        results = {}
        for well, ds in current_ds.items():
            try:
                results[well] = fit_binding_glob(ds, method=method)
            except InsufficientDataError:
                logger.warning(
                    "Skip FGLS fit for well %s (iteration %d).", well, iteration + 1
                )
                results[well] = FitResult()

        # Calibrate per-label noise (floor fixed, gain + alpha via NNLS)
        df_res = residuals_from_fit_results(
            results, trace_id="", binding_function=binding_1site
        )
        try:
            floors, gains, alphas = fit_noise_model_nnls(
                df_res, sigma_floor_fixed=sigma_floor
            )
        except ValueError as e:
            logger.warning("FGLS calibration failed (%s).", e)
            gains = dict.fromkeys(sigma_floor, 0.0)
            alphas = dict.fromkeys(sigma_floor, 0.0)
            floors = dict(sigma_floor)

        # Fit global sigma_ph from excess variance after per-label model
        plate_slopes = compute_plate_slopes(results)
        tmp_noise = _plate_noise_model_from_nnls(floors, gains, alphas)
        sigma_ph = fit_ph_slope_noise(df_res, tmp_noise, plate_slopes)
        new_noise = _plate_noise_model_from_nnls(floors, gains, alphas, sigma_ph)

        for lbl, params in new_noise.items():
            logger.info(
                "Calibrated [%s] iter %d: sigma=%.2f, gain=%.3f, alpha=%.3f, "
                "sigma_ph=%.4f",
                lbl,
                iteration + 1,
                params.sigma_floor,
                params.gain,
                params.alpha,
                params.sigma_ph,
            )

        # Check convergence (skip on first iteration)
        if iteration > 0 and _noise_params_converged(noise_model, new_noise, tol):  # type: ignore[arg-type]
            logger.info("FGLS converged after %d iterations.", iteration + 1)
            noise_model = new_noise
            break

        noise_model = new_noise

    return results, noise_model  # type: ignore[return-value]
