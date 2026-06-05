"""Utility functions for fitting modules."""

import copy
import typing

import numpy as np
import pandas as pd
from scipy import optimize, stats

from clophfit.clophfit_types import ArrayMask
from clophfit.fitting.data_structures import (
    Dataset,
    PlateNoiseModel,
    compute_noise_variance,
)


def parse_remove_outliers(spec: str) -> tuple[str, float, int]:
    """Parse outlier specification ``"method:threshold:min_keep"``.

    Parameters
    ----------
    spec : str
        The string to parse.

    Returns
    -------
    tuple[str, float, int]
        A tuple of `method`, `threshold`, `min_keep`.

    Examples
    --------
    - ``"zscore:2.5:5"`` -> ("zscore", 2.5, 5)
    - ``"method"`` -> ("method", 2.0, 1)
    """
    n_threshold_parts = 1
    n_min_keep_parts = 2
    parts = spec.split(":")
    method = parts[0]
    threshold = float(parts[1]) if len(parts) > n_threshold_parts else 2.0
    min_keep = int(parts[2]) if len(parts) > n_min_keep_parts else 1
    return method, threshold, min_keep


def identify_outliers_zscore(
    residuals: np.ndarray, threshold: float = 2.0
) -> ArrayMask:
    """Identify outliers using the Z-score method on a 1D array of residuals.

    Parameters
    ----------
    residuals : np.ndarray
        The residuals to analyze.
    threshold : float
        The Z-score threshold beyond which a point is considered an outlier.

    Returns
    -------
    ArrayMask
        A boolean mask where True indicates an outlier.
    """
    if len(residuals) == 0:
        return np.zeros(0, dtype=bool)

    mean_r = float(np.mean(residuals))
    std_r = float(np.std(residuals))

    if std_r > 0:
        z = np.abs((residuals - mean_r) / std_r)
        return z > threshold
    return np.zeros(len(residuals), dtype=bool)


def reweight_from_residuals(ds: Dataset, residuals: np.ndarray) -> Dataset:
    """Update y_errc in a Dataset from the mean absolute residuals of each label.

    Parameters
    ----------
    ds : Dataset
        The input dataset.
    residuals : np.ndarray
        The combined 1D array of residuals for all labels in the dataset,
        in the order of ds.values().

    Returns
    -------
    Dataset
        A new dataset with updated y_err.
    """
    updated_ds = copy.deepcopy(ds)
    for i, da in enumerate(updated_ds.values()):
        len_x = len(da.y)
        label_residuals = residuals[i * len_x : (i + 1) * len_x]
        sigma_val = float(max(float(np.mean(np.abs(label_residuals))), 1e-3))
        # Important: y_errc must match the length of xc, but the current
        # weights are uniformly applied. We apply it to all points, even masked.
        da.y_errc = np.full(da.xc.shape, sigma_val)
    return updated_ds


def flag_trend_outliers(
    x: pd.Series, y: pd.Series, threshold: float = 3.0
) -> pd.Series:
    """Flag outliers using robust Theil-Sen regression of y on x.

    A point is flagged if its residual is far from the trendline (Z-score < -threshold)
    OR if its x-value is extremely low compared to the population (Z-score < -threshold).

    Parameters
    ----------
    x : pd.Series
        The independent variable (e.g., maximum signal, mean).
    y : pd.Series
        The dependent variable (e.g., signal span, std, or dynamic range).
    threshold : float
        The Z-score threshold for flagging an outlier.

    Returns
    -------
    pd.Series
        A boolean Series of the same length as x, True for outliers.
    """
    if len(x) < 3:  # noqa: PLR2004
        return pd.Series(data=False, index=x.index)

    x_np = x.to_numpy()
    y_np = y.to_numpy()

    # Robust linear regression
    res = stats.theilslopes(y_np, x_np)
    m, c = res[0], res[1]

    predicted = m * x_np + c
    residuals = y_np - predicted

    mad = np.median(np.abs(residuals - np.median(residuals)))
    if mad < 1e-6:  # noqa: PLR2004
        mad = np.std(residuals)

    z_scores = np.zeros_like(x_np, dtype=float)
    if mad > 1e-6:  # noqa: PLR2004
        z_scores = (residuals - np.median(residuals)) / (1.4826 * mad)

    # Marginal Z-score for x (signal amplitude)
    mad_x = np.median(np.abs(x_np - np.median(x_np)))
    z_x = np.zeros_like(x_np, dtype=float)
    if mad_x < 1e-6:  # noqa: PLR2004
        mad_x = np.std(x_np)
    if mad_x > 1e-6:  # noqa: PLR2004
        z_x = (x_np - np.median(x_np)) / (1.4826 * mad_x)

    # We want to flag wells that are far from the trendline (both above or below)
    # OR if its signal is incredibly low (z_x < -threshold)
    return pd.Series((np.abs(z_scores) > threshold) | (z_x < -threshold), index=x.index)


def fit_trendline(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    """Fit a robust Theil-Sen regression line.

    Parameters
    ----------
    x : pd.Series
        The independent variable.
    y : pd.Series
        The dependent variable.

    Returns
    -------
    tuple[float, float]
        Slope and intercept.
    """
    if len(x) < 2:  # noqa: PLR2004
        return 0.0, 0.0
    res = stats.theilslopes(y.to_numpy(), x.to_numpy())
    return float(res[0]), float(res[1])


def smoothness(y: np.ndarray) -> float:
    r"""Calculate the smoothness of a curve.

    Sum of \|consecutive diffs\| / total span.
    = 1 for perfectly monotone, > 1 for noisy/non-monotone.

    Parameters
    ----------
    y : np.ndarray
        The signal array.

    Returns
    -------
    float
        The smoothness value.
    """
    consec = float(np.sum(np.abs(np.diff(y))))
    span = float(np.abs(y[-1] - y[0]))  # or y.max() - y.min() if not sorted by x
    return consec / span if span > 0 else np.nan


def roughness(y: np.ndarray) -> float:
    """Calculate the roughness of a curve.

    Excess path fraction: 0 = perfectly monotone, 1 = all noise, flat-safe.
    roughness = (consec - span) / consec.

    Parameters
    ----------
    y : np.ndarray
        The signal array.

    Returns
    -------
    float
        The roughness value.
    """
    span = float(np.abs(y[-1] - y[0]))
    consec = float(np.sum(np.abs(np.diff(y))))
    if consec < 1e-12:  # noqa: PLR2004
        return 0.0  # flat and smooth → good
    return (consec - span) / consec


def outlier_scores_extended(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute outlier scores for each point using geometric deviation.

    Uses a hybrid approach for edge points:
    - If `edge_step` > 2 * `local_step`: anomalously large jump → use full projection deviation
    - Elif wrong direction (reversal): use projection deviation
    - Else (correct direction / plateau approach): score = 0

    For internal points: triangle inequality score.

    Parameters
    ----------
    x : np.ndarray
        x-values (e.g. pH or concentration).
    y : np.ndarray
        Observed y-values.

    Returns
    -------
    np.ndarray
        Per-point outlier scores (non-negative; higher = more anomalous).

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> y = np.array([10.0, 8.0, 15.0, 4.0, 2.0])
    >>> scores = outlier_scores_extended(x, y)
    >>> bool(scores[2] > 0.4)
    True
    """
    scores = np.zeros(len(y))
    if len(y) < 3:  # noqa: PLR2004
        return scores
    consec = float(np.sum(np.abs(np.diff(y))))

    # Internal points: triangle inequality
    for i in range(1, len(y) - 1):
        through = abs(y[i] - y[i - 1]) + abs(y[i + 1] - y[i])
        direct = abs(y[i + 1] - y[i - 1])
        scores[i] = (through - direct) / (consec + 1e-12)

    # Edge points: hybrid check
    def _edge_score(  # noqa: PLR0913,PLR0917
        y_edge: float,
        y_near: float,
        y_far: float,
        x_edge: float,
        x_near: float,
        x_far: float,
    ) -> float:
        edge_step = abs(y_edge - y_near)
        local_step = abs(y_near - y_far)
        dx_near = x_near - x_far
        dx_edge = x_edge - x_near
        slope = (y_near - y_far) / (dx_near + 1e-12)
        projected = y_near + slope * dx_edge
        deviation = abs(y_edge - projected)

        if edge_step > 2.0 * local_step + 1e-12:
            return deviation / (consec + 1e-12)
        # Direction check: wrong direction is an anomaly
        expected_direction = y_near - y_far
        actual_direction = y_edge - y_near
        if expected_direction * actual_direction < 0:
            return deviation / (consec + 1e-12)
        return 0.0

    scores[0] = _edge_score(y[0], y[1], y[2], x[0], x[1], x[2])
    scores[-1] = _edge_score(y[-1], y[-2], y[-3], x[-1], x[-2], x[-3])
    return scores


def apply_outlier_mask(
    ds: Dataset,
    threshold: float = 0.2,
    min_keep: int = 3,
) -> Dataset:
    """Mask outlier points iteratively in each DataArray of a Dataset.

    Removes the single worst outlier (if above threshold) and recomputes
    scores, repeating until no score exceeds the threshold or fewer than
    min_keep unmasked points remain.

    Parameters
    ----------
    ds : Dataset
        Dataset to process (deep-copied; input is not modified).
    threshold : float, optional
        Outlier score above which a point is masked. Default is 0.2.
    min_keep : int, optional
        Minimum number of unmasked points to retain. Default is 3.

    Returns
    -------
    Dataset
        A new Dataset with outlier points masked.
    """
    result = copy.deepcopy(ds)
    for da in result.values():
        while True:
            unmasked = np.where(da.mask)[0]
            if len(unmasked) <= min_keep:
                break
            x_active = da.xc[unmasked]
            y_active = da.yc[unmasked]
            scores = outlier_scores_extended(x_active, y_active)
            worst_local = int(np.argmax(scores))
            if scores[worst_local] <= threshold:
                break
            worst_global = unmasked[worst_local]
            da.mask[worst_global] = False
    return result


def fit_rel_error_from_residuals(
    df: "pd.DataFrame",
    sigma_floor: dict[str, float],
) -> dict[str, float]:
    r"""Estimate proportional error (alpha) per label via moment estimator.

    Assumes the simplified noise model ``sigma^2 = floor^2 + alpha^2 * that^2``
    (no Poisson gain term). With ``floor`` known from buffer measurements
    and using model-predicted values ``that`` in the denominator to avoid
    noise-in-variables bias, the closed-form moment estimator is:

    .. math::

        \hat{\alpha}^2 =
        \frac{\overline{r^2} - \sigma_{\text{floor}}^2}{\overline{\hat{y}^2}}

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns ``label`` (str), ``resid_raw`` (float), and
        ``predicted`` (float -- the model-predicted signal at each point).
        Typically from :func:`clophfit.fitting.residuals.collect_multi_residuals`.
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
    >>> df = pd.DataFrame({"label": "1", "resid_raw": resid, "predicted": y_pred})
    >>> alpha = fit_rel_error_from_residuals(df, sigma_floor={"1": floor})
    >>> round(alpha["1"], 2)  # should be close to true_alpha=0.02
    0.02
    """
    result: dict[str, float] = {}
    for lbl, grp in df.groupby("label"):
        lbl_str = str(lbl)
        r2_mean = float((grp["resid_raw"] ** 2).mean())
        pred2_mean = float((grp["predicted"] ** 2).mean())
        floor = float(sigma_floor.get(lbl_str, 0.0))
        alpha_sq = max(0.0, r2_mean - floor**2) / max(pred2_mean, 1e-12)
        result[lbl_str] = float(np.sqrt(alpha_sq))
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
        Residual DataFrame with columns ``label``, ``resid_raw``, ``predicted``.
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
        y = grp["predicted"].to_numpy().astype(float)
        r2 = grp["resid_raw"].to_numpy().astype(float) ** 2

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


def fit_noise_model_from_residuals(
    df: "pd.DataFrame",
    rel_error: float | dict[str, float] = 0.003,
) -> tuple[dict[str, float], dict[str, float]]:
    r"""Fit per-label noise model parameters from first-pass residuals.

    With ``rel_error`` fixed, the noise equation becomes linear in two unknowns
    via non-negative least squares.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns ``label``, ``resid_raw``, ``predicted``.
    rel_error : float | dict[str, float], optional
        Fixed proportional error. A single float is broadcast to all labels.
        Default is 0.003.

    Returns
    -------
    tuple[dict[str, float], dict[str, float]]
        ``(sigma_floor_dict, gain_dict)`` per label (non-negative).
    """
    if isinstance(rel_error, float):
        labels = df["label"].unique()
        rel_error = {str(lbl): rel_error for lbl in labels}
    sigma_floor, gain_d, _ = fit_noise_model_nnls(df, rel_error_fixed=rel_error)
    return sigma_floor, gain_d


def fit_gain_and_rel_error_from_residuals(
    df: "pd.DataFrame",
    sigma_floor: dict[str, float],
) -> tuple[dict[str, float], dict[str, float]]:
    r"""Fit gain and rel_error per label from residuals with known floor.

    Uses non-negative least squares on ``r^2 - floor^2 = gain * y + alpha^2 * y^2``
    to handle collinearity between :math:`y` and :math:`y^2`.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns ``label``, ``resid_raw``, ``predicted``.
    sigma_floor : dict[str, float]
        Known noise floor per label.

    Returns
    -------
    tuple[dict[str, float], dict[str, float]]
        ``(gain_dict, rel_error_dict)`` per label (non-negative).
    """
    _, gain_d, alpha_d = fit_noise_model_nnls(df, sigma_floor_fixed=sigma_floor)
    return gain_d, alpha_d


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
        Residual DataFrame with columns ``label``, ``well``, ``resid_raw``,
        ``predicted``, and ``raw_i``.
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
            df.loc[mask, "predicted"].to_numpy(dtype=float),
            params.sigma_floor,
            params.gain,
            params.alpha,
        )
    df["var_excess"] = df["resid_raw"] ** 2 - df["var_model"]
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
