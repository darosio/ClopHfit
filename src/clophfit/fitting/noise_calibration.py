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
    (:func:`~clophfit.fitting.noise_calibration.fit_gain_from_residuals`,
    :func:`~clophfit.fitting.noise_calibration.fit_rel_error_from_residuals`) on the
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


def _design_for_free_terms(
    y: np.ndarray,
    target: np.ndarray,
    fixed: tuple[float | None, float | None, float | None],
) -> tuple[list[np.ndarray], list[str], np.ndarray, np.ndarray]:
    """Split the three noise terms into fitted columns and known contributions.

    A fixed term contributes a known amount, which comes off the squared
    residual; the rest become columns of the design. Zero is an ordinary fixed
    value and not a disabled term, so fixing alpha at 0 subtracts nothing and
    still leaves floor and gain to be estimated.

    Parameters
    ----------
    y : np.ndarray
        Predicted signal per observation.
    target : np.ndarray
        Squared residual, before any subtraction.
    fixed : tuple[float | None, float | None, float | None]
        Fixed floor, gain and alpha for this label; ``None`` means fit it.

    Returns
    -------
    tuple[list[np.ndarray], list[str], np.ndarray, np.ndarray]
        Design columns, their names, the reduced target, and the variance the
        fixed terms already account for.
    """
    floor_fx, gain_fx, alpha_fx = fixed
    columns: list[np.ndarray] = []
    names: list[str] = []
    fixed_var = np.zeros_like(y)
    if floor_fx is None:
        columns.append(np.ones_like(y))
        names.append("floor")
    else:
        fixed_var += float(floor_fx) ** 2
        target -= float(floor_fx) ** 2
    if gain_fx is None:
        columns.append(y)
        names.append("gain")
    else:
        fixed_var += float(gain_fx) * y
        target -= float(gain_fx) * y
    if alpha_fx is None:
        columns.append(y**2)
        names.append("alpha")
    else:
        fixed_var += (float(alpha_fx) * y) ** 2
        target -= (float(alpha_fx) * y) ** 2
    return columns, names, target, fixed_var


def fit_noise_model_nnls(
    df: pd.DataFrame,
    sigma_floor_fixed: dict[str, float] | None = None,
    rel_error_fixed: dict[str, float] | None = None,
    gain_fixed: dict[str, float] | None = None,
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
        If given, fix alpha per label and only fit the remaining terms.
    gain_fixed : dict[str, float] | None
        If given, fix gain per label and only fit the remaining terms. Any
        subset may be fixed, including all three, which simply returns them.

    Returns
    -------
    tuple[dict[str, float], dict[str, float], dict[str, float]]
        ``(sigma_floor, gain, alpha)`` per label — all non-negative.

    """
    sigma_floor_out: dict[str, float] = {}
    gain_out: dict[str, float] = {}
    alpha_out: dict[str, float] = {}

    for lbl, grp in df.groupby("label"):
        lbl_str = str(lbl)
        y = grp["yhat"].to_numpy().astype(float)
        target = grp["raw_res"].to_numpy().astype(float) ** 2

        # A fixed term contributes a known amount, which comes off the squared
        # residual; the rest are columns of the design. Zero is an ordinary
        # fixed value, so fixing alpha at 0 subtracts nothing and still leaves
        # floor and gain to be estimated.
        floor_fx = None if sigma_floor_fixed is None else sigma_floor_fixed.get(lbl_str)
        gain_fx = None if gain_fixed is None else gain_fixed.get(lbl_str)
        alpha_fx = None if rel_error_fixed is None else rel_error_fixed.get(lbl_str)

        columns, names, target, _fixed_var = _design_for_free_terms(
            y, target, (floor_fx, gain_fx, alpha_fx)
        )

        if not columns:
            sigma_floor_out[lbl_str] = float(floor_fx or 0.0)
            gain_out[lbl_str] = float(gain_fx or 0.0)
            alpha_out[lbl_str] = float(alpha_fx or 0.0)
            continue

        # Every point is used, including those whose remainder went negative
        # after a fixed contribution was subtracted. NNLS constrains the
        # coefficients to be non-negative, not the data, and dropping the
        # negative side keeps only upward fluctuations of a chi-square: with
        # floor and alpha both held at their true values that bias returned a
        # gain of 8.69 against a true 0.5.
        if len(target) < len(columns) + 1:
            fitted = dict.fromkeys(names, 0.0)
        else:
            x_mat = np.column_stack(columns)
            # Two passes. The target is a squared residual, whose own variance
            # goes as sigma^4, so an unweighted fit lets the brightest points
            # dominate: unbiased but with a scatter of 0.31 on a gain of 0.5.
            # The second pass weights by 1/var^2 from the first, which is the
            # inverse variance of a squared Gaussian residual, and cuts that to
            # 0.016 -- the difference between an iterative refit converging and
            # wandering along the gain/alpha ridge.
            coeffs, _ = optimize.nnls(x_mat, target)
            var_hat = x_mat @ coeffs
            if floor_fx is not None:
                var_hat += float(floor_fx) ** 2
            if gain_fx is not None:
                var_hat += float(gain_fx) * y
            if alpha_fx is not None:
                var_hat += (float(alpha_fx) * y) ** 2
            good = np.isfinite(var_hat) & (var_hat > 0)
            if good.sum() >= len(columns) + 1:
                w = np.sqrt(1.0 / var_hat[good] ** 2)
                coeffs, _ = optimize.nnls(x_mat[good] * w[:, None], target[good] * w)
            fitted = dict(zip(names, map(float, coeffs), strict=True))

        sigma_floor_out[lbl_str] = (
            float(floor_fx) if floor_fx is not None else float(np.sqrt(fitted["floor"]))
        )
        gain_out[lbl_str] = (
            float(gain_fx) if gain_fx is not None else float(fitted["gain"])
        )
        alpha_out[lbl_str] = (
            float(alpha_fx) if alpha_fx is not None else float(np.sqrt(fitted["alpha"]))
        )

    return sigma_floor_out, gain_out, alpha_out


_MERGED_MIN_BIN = 5
_MAD_TO_SIGMA = 1.4826


def _binned_variance(
    yhat: np.ndarray, raw_res: np.ndarray, n_bins: int, dof_scale: float
) -> pd.DataFrame:
    """Robust residual variance per equal-count signal bin.

    Parameters
    ----------
    yhat : np.ndarray
        Fitted signal per observation.
    raw_res : np.ndarray
        Raw residual per observation.
    n_bins : int
        Equal-count bins across the signal range.
    dof_scale : float
        ``sqrt(n / (n - p))``, undoing the variance the fit absorbed.

    Returns
    -------
    pd.DataFrame
        ``y``, ``var`` and ``n`` per bin; bins below five points are dropped.
    """
    order = np.argsort(yhat)
    y_sorted, r_sorted = yhat[order], raw_res[order]
    rows: list[dict[str, float]] = []
    for idx in np.array_split(np.arange(len(y_sorted)), n_bins):
        if len(idx) < _MERGED_MIN_BIN:
            continue
        chunk = r_sorted[idx]
        # MAD**2 rather than the plain variance: these plates carry real
        # outliers, and one of them in a bin moves the variance further than
        # the noise level it is meant to measure.
        mad = _MAD_TO_SIGMA * np.median(np.abs(chunk - np.median(chunk)))
        rows.append({
            "y": float(np.median(y_sorted[idx])),
            "var": float((mad * dof_scale) ** 2),
            "n": float(len(idx)),
        })
    return pd.DataFrame(rows)


def fit_noise_model_merged(df: pd.DataFrame, *, n_bins: int = 10) -> pd.DataFrame:
    r"""Fit one gain and one alpha per label, on every plate's residuals at once.

    The pooled estimator in its literal sense: the residual tables from all
    plates are concatenated and a single ``gain`` and ``alpha`` are fitted to
    the lot, rather than fitting each plate and averaging the answers. Fitting
    per plate and summarising cannot recover a shared value when the per-plate
    fits are themselves at a boundary, which they routinely are - ``y`` and
    ``y**2`` are near-collinear over the range one plate covers.

    The floor is supplied, never fitted, and differs per plate when it is scaled
    to that plate's reader Gain. Each plate's own floor variance is therefore
    removed before pooling, and what is fitted is the remainder:

    .. math:: \sigma^2 - \text{floor}^2 = \text{gain} \cdot a \cdot y
              + (\alpha \cdot y)^2

    where :math:`a` is ``gain_amp``. Binning happens within a plate, so a bin
    never mixes plates of different brightness, and bins are weighted by count.

    Parameters
    ----------
    df : pd.DataFrame
        Residual table with columns ``plate``, ``label``, ``yhat``, ``raw_res``
        and ``sigma_floor``. Two columns are optional: ``gain_amp``, the
        amplification the gain term is quoted against (default 1.0), and
        ``dof_scale`` (default 1.0). Pass ``gain_amp`` only for a label whose
        noise actually tracks reader Gain - scaling a label whose floor is
        supplied unscaled quotes the two terms at different references.
    n_bins : int
        Equal-count signal bins per plate and label.

    Returns
    -------
    pd.DataFrame
        One row per label: ``label``, ``n_bins``, ``n_plates``, ``gain`` and
        ``alpha`` (both at ``gain_amp == 1``), ``log_misfit`` and
        ``negative_bins``.

    Raises
    ------
    ValueError
        If a required column is missing. The floor in particular is an input,
        because a fit that frees all three terms drives one to its boundary.
    """
    required = {"plate", "label", "yhat", "raw_res", "sigma_floor"}
    missing = required - set(df.columns)
    if missing:
        msg = f"residual table is missing required column(s): {sorted(missing)}"
        raise ValueError(msg)

    work = df.copy()
    work["label"] = work["label"].astype(str)
    for col, default in (("gain_amp", 1.0), ("dof_scale", 1.0)):
        if col not in work.columns:
            work[col] = default

    bins: list[pd.DataFrame] = []
    for (plate, label), grp in work.groupby(["plate", "label"], sort=True):
        binned = _binned_variance(
            grp["yhat"].to_numpy(dtype=float),
            grp["raw_res"].to_numpy(dtype=float),
            n_bins,
            float(grp["dof_scale"].iloc[0]),
        )
        if binned.empty:
            continue
        binned["plate"] = str(plate)
        binned["label"] = str(label)
        binned["floor"] = float(grp["sigma_floor"].iloc[0])
        binned["gain_amp"] = float(grp["gain_amp"].iloc[0])
        bins.append(binned)
    if not bins:
        msg = "no signal bin held enough observations to estimate a variance"
        raise ValueError(msg)

    allbins = pd.concat(bins, ignore_index=True)
    rows: list[dict[str, object]] = []
    for label, cells in allbins.groupby("label", sort=True):
        y = cells["y"].to_numpy()
        floor = cells["floor"].to_numpy()
        amp = cells["gain_amp"].to_numpy()
        # Each plate's own supplied floor comes off before pooling, so what
        # remains is only the signal-dependent part the shared terms describe.
        remainder = cells["var"].to_numpy() - floor**2
        weight = np.sqrt(cells["n"].to_numpy())
        design = np.column_stack([amp * y, y**2])
        coeffs, _ = optimize.nnls(design * weight[:, None], remainder * weight)
        gain, alpha = float(coeffs[0]), float(np.sqrt(coeffs[1]))
        pred = floor**2 + gain * amp * y + (alpha * y) ** 2
        misfit = float(np.sqrt(np.mean((np.log(pred) - np.log(cells["var"])) ** 2)))
        rows.append({
            "label": str(label),
            "n_bins": len(cells),
            "n_plates": int(cells["plate"].nunique()),
            "gain": gain,
            "alpha": alpha,
            "log_misfit": misfit,
            "negative_bins": int((remainder < 0).sum()),
        })
    return pd.DataFrame(rows)


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
