"""Noise-model calibration from fit residuals.

Estimators that turn a canonical residual table into a
:class:`~clophfit.fitting.data_structures.PlateNoiseModel`: per-label floor,
photon gain, and proportional error, plus the plate slope helpers used to
propagate x-axis noise.
"""

import logging
import typing

import numpy as np
import pandas as pd
from scipy import optimize

from clophfit.fitting.data_structures import (
    NoiseModelParams,
    PlateNoiseModel,
    compute_noise_variance,
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


def fit_rel_error_from_residuals(
    df: "pd.DataFrame",
    sigma_floor: dict[str, float],
) -> dict[str, float]:
    r"""Estimate proportional error (alpha) per label via moment estimator.

    Assumes the simplified noise model ``sigma^2 = floor^2 + alpha^2 * yhat^2``
    (no Poisson gain term). With ``floor`` known from buffer measurements
    and using model-predicted values ``yhat`` in the denominator to avoid
    noise-in-variables bias, the closed-form moment estimator is:

    .. math::

        \hat{\alpha}^2 =
        \frac{\overline{r^2} - \sigma_{\text{floor}}^2}{\overline{\hat{y}^2}}

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns ``label`` (str), ``raw_res`` (float), and
        ``yhat`` (float -- the model-predicted signal at each point).
        Typically from :func:`clophfit.fitting.model_validation.residuals_from_fit_results`.
    sigma_floor : dict[str, float]
        Known read-noise floor per label, e.g. from ``tit.bg_noise``.

    Returns
    -------
    dict[str, float]
        Per-label proportional error estimate ``alpha`` (non-negative).

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> y_pred = np.linspace(50, 500, 200)
    >>> floor, true_alpha = 5.0, 0.02
    >>> sigma = np.sqrt(floor**2 + (true_alpha * y_pred) ** 2)
    >>> resid = sigma * rng.standard_normal(200)
    >>> df = pd.DataFrame({"label": "1", "raw_res": resid, "yhat": y_pred})
    >>> alpha = fit_rel_error_from_residuals(df, sigma_floor={"1": floor})
    >>> round(alpha["1"], 2)  # should be close to true_alpha=0.02
    0.02
    """
    result: dict[str, float] = {}
    for lbl, grp in df.groupby("label"):
        lbl_str = str(lbl)
        r2_mean = float((grp["raw_res"] ** 2).mean())
        pred2_mean = float((grp["yhat"] ** 2).mean())
        floor = float(sigma_floor.get(lbl_str, 0.0))
        alpha_sq = max(0.0, r2_mean - floor**2) / max(pred2_mean, 1e-12)
        result[lbl_str] = float(np.sqrt(alpha_sq))
    return result


def fit_gain_from_residuals(
    df: "pd.DataFrame",
    sigma_floor: dict[str, float],
) -> dict[str, float]:
    r"""Estimate Poisson gain per label via moment estimator.

    Symmetric counterpart to :func:`fit_rel_error_from_residuals`. Assumes the
    Poisson-only noise model ``sigma^2 = floor^2 + gain * yhat`` (no
    proportional term), which sidesteps the gain/alpha collinearity of the
    joint fit. With ``floor`` known from buffer measurements and using
    model-predicted values ``yhat``, the closed-form moment estimator is:

    .. math::

        \hat{\text{gain}} =
        \frac{\overline{r^2} - \sigma_{\text{floor}}^2}{\overline{\hat{y}}}

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns ``label`` (str), ``raw_res`` (float), and
        ``yhat`` (float -- the model-predicted signal at each point).
        Typically from :func:`clophfit.fitting.model_validation.residuals_from_fit_results`.
    sigma_floor : dict[str, float]
        Known read-noise floor per label, e.g. from ``tit.bg_noise``.

    Returns
    -------
    dict[str, float]
        Per-label Poisson gain estimate (non-negative).

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> y_pred = np.linspace(50, 500, 400)
    >>> floor, true_gain = 5.0, 0.8
    >>> sigma = np.sqrt(floor**2 + true_gain * y_pred)
    >>> resid = sigma * rng.standard_normal(400)
    >>> df = pd.DataFrame({"label": "1", "raw_res": resid, "yhat": y_pred})
    >>> gain = fit_gain_from_residuals(df, sigma_floor={"1": floor})
    >>> round(gain["1"], 1)  # should be close to true_gain=0.8
    0.8
    """
    result: dict[str, float] = {}
    for lbl, grp in df.groupby("label"):
        lbl_str = str(lbl)
        r2_mean = float((grp["raw_res"] ** 2).mean())
        pred_mean = float(grp["yhat"].mean())
        floor = float(sigma_floor.get(lbl_str, 0.0))
        gain = max(0.0, r2_mean - floor**2) / max(pred_mean, 1e-12)
        result[lbl_str] = float(gain)
    return result


def fit_noise_model_nnls(
    df: pd.DataFrame,
    sigma_floor_fixed: dict[str, float] | None = None,
    rel_error_fixed: dict[str, float] | None = None,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    r"""Fit heteroscedastic noise model via non-negative least squares.

    Model:  :math:`\sigma^2 = \sigma_\text{floor}^2 + \text{gain} \cdot y
    + \alpha^2 \cdot y^2`

    Uses :func:`scipy.optimize.nnls` to enforce non-negativity on all
    parameters, which stabilises estimates when :math:`y` and :math:`y^2`
    are highly collinear (typical for narrow-range titrations).

    Parameters
    ----------
    df : pd.DataFrame
        Residual DataFrame with columns ``label``, ``raw_res``, ``yhat``.
    sigma_floor_fixed : dict[str, float] | None
        If given, fix floor per label and only fit gain and alpha.
    rel_error_fixed : dict[str, float] | None
        If given, fix alpha per label and only fit floor and gain.

    Returns
    -------
    tuple[dict[str, float], dict[str, float], dict[str, float]]
        ``(sigma_floor, gain, alpha)`` per label — all non-negative.

    Raises
    ------
    ValueError
        If both *sigma_floor_fixed* and *rel_error_fixed* are provided.
    """
    if sigma_floor_fixed is not None and rel_error_fixed is not None:
        msg = "Cannot fix both sigma_floor and rel_error simultaneously."
        raise ValueError(msg)

    sigma_floor_out: dict[str, float] = {}
    gain_out: dict[str, float] = {}
    alpha_out: dict[str, float] = {}

    for lbl, grp in df.groupby("label"):
        lbl_str = str(lbl)
        y = grp["yhat"].to_numpy().astype(float)
        r2 = grp["raw_res"].to_numpy().astype(float) ** 2

        if sigma_floor_fixed is not None:
            floor = sigma_floor_fixed.get(lbl_str, 0.0)
            adjusted = r2 - floor**2
            positive = adjusted > 0
            if positive.sum() < 2:  # noqa: PLR2004
                gain_out[lbl_str] = 0.0
                alpha_out[lbl_str] = 0.0
                sigma_floor_out[lbl_str] = floor
                continue
            x_mat = np.column_stack([y[positive], y[positive] ** 2])
            b_vec = adjusted[positive]
            coeffs, _ = optimize.nnls(x_mat, b_vec)
            sigma_floor_out[lbl_str] = floor
            gain_out[lbl_str] = float(coeffs[0])
            alpha_out[lbl_str] = float(np.sqrt(coeffs[1]))

        elif rel_error_fixed is not None:
            alpha_fixed = rel_error_fixed.get(lbl_str, 0.0)
            adjusted = r2 - (alpha_fixed * y) ** 2
            x_mat = np.column_stack([np.ones_like(y), y])
            coeffs, _ = optimize.nnls(x_mat, adjusted)
            sigma_floor_out[lbl_str] = float(np.sqrt(coeffs[0]))
            gain_out[lbl_str] = float(coeffs[1])
            alpha_out[lbl_str] = alpha_fixed

        else:
            x_mat = np.column_stack([np.ones_like(y), y, y**2])
            coeffs, _ = optimize.nnls(x_mat, r2)
            sigma_floor_out[lbl_str] = float(np.sqrt(coeffs[0]))
            gain_out[lbl_str] = float(coeffs[1])
            alpha_out[lbl_str] = float(np.sqrt(coeffs[2]))

    return sigma_floor_out, gain_out, alpha_out


# ------------------------------------------------------------------
# pH-dependent noise (pipetting error amplified by titration slope)
# ------------------------------------------------------------------


def compute_binding_slope(
    ph: np.ndarray,
    pka: float,
    s0: float,
    s1: float,
) -> np.ndarray:
    r"""Compute |dS/dpH| for the Henderson-Hasselbalch equation.

    ``dS/dpH = (s1 - s0) * ln(10) * t / (1 + t)^2`` where ``t = 10^(pka - ph)``.
    Returns the absolute value (sign irrelevant for variance).
    """
    t = 10.0 ** (pka - ph)
    result: np.ndarray = np.abs((s1 - s0) * np.log(10) * t / (1.0 + t) ** 2)
    return result


def compute_plate_slopes(
    results: dict[str, typing.Any],
) -> dict[str, dict[str, np.ndarray]]:
    """Compute per-well per-label ``∂S/∂pH`` from pass-1 fit results.

    Parameters
    ----------
    results : dict[str, typing.Any]
        Fit results keyed by well (must have ``.result`` and ``.dataset``).

    Returns
    -------
    dict[str, dict[str, np.ndarray]]
        ``{well: {label: slope_array}}``.
    """
    slopes: dict[str, dict[str, np.ndarray]] = {}
    for well, fr in results.items():
        if fr.result is None or fr.dataset is None:
            continue
        rpars = fr.result.params
        if "K" not in rpars:
            continue
        pka = rpars["K"].value
        well_slopes: dict[str, np.ndarray] = {}
        for lbl, da in fr.dataset.items():
            s0 = rpars[f"S0_{lbl}"].value
            s1 = rpars[f"S1_{lbl}"].value
            well_slopes[lbl] = compute_binding_slope(da.xc, pka, s0, s1)
        slopes[well] = well_slopes
    return slopes


_MIN_POINTS_FOR_PH_SLOPE_FIT = 2


def fit_ph_slope_noise(
    df: pd.DataFrame,
    noise_model: PlateNoiseModel,
    plate_slopes: dict[str, dict[str, np.ndarray]],
) -> float:
    r"""Fit global ``sigma_ph`` from excess variance after per-label model.

    After subtracting the per-label noise model variance, the leftover
    ``r^2 - var_model`` is regressed against ``(dS/dpH)^2`` via NNLS.

    Parameters
    ----------
    df : pd.DataFrame
        Residual DataFrame with columns ``label``, ``well``, ``raw_res``,
        ``yhat``, and ``raw_i``.
    noise_model : PlateNoiseModel
        Per-label noise model (floor, gain, alpha) fitted in the same pass.
    plate_slopes : dict[str, dict[str, np.ndarray]]
        Per-well per-label derivative ``|dS/dpH|`` arrays.

    Returns
    -------
    float
        Global ``sigma_ph`` estimate (>= 0).
    """
    df = df.copy()
    df["var_model"] = np.nan
    for lbl in df["label"].unique():
        params = noise_model[str(lbl)]
        mask = df["label"] == lbl
        df.loc[mask, "var_model"] = compute_noise_variance(
            df.loc[mask, "yhat"].to_numpy(dtype=float),
            params.sigma_floor,
            params.gain,
            params.alpha,
        )
    df["var_excess"] = df["raw_res"] ** 2 - df["var_model"]
    df["slope_sq"] = np.nan
    for (lbl, well), grp in df.groupby(["label", "well"]):
        w_slopes_dict = plate_slopes.get(str(well))
        w_slopes = w_slopes_dict.get(str(lbl)) if w_slopes_dict is not None else None
        if w_slopes is not None:
            raw_is = grp["raw_i"].to_numpy(dtype=int)
            df.loc[grp.index, "slope_sq"] = w_slopes[raw_is] ** 2

    pos = df["var_excess"] > 0
    valid = pos & df["slope_sq"].notna()
    if valid.sum() < _MIN_POINTS_FOR_PH_SLOPE_FIT:
        return 0.0
    x = np.column_stack([df.loc[valid, "slope_sq"].to_numpy(dtype=float)])
    b = df.loc[valid, "var_excess"].to_numpy(dtype=float)
    coeffs, _ = optimize.nnls(x, b)
    return float(np.sqrt(max(0.0, coeffs[0])))
