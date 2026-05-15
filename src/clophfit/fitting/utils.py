"""Utility functions for fitting modules."""

import copy

import numpy as np
import pandas as pd
from scipy import stats

from clophfit.clophfit_types import ArrayF, ArrayMask
from clophfit.fitting.data_structures import Dataset


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
    """Calculate the smoothness of a curve.

    Sum of |consecutive diffs| / total span.
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
    - If |edge_step| > 2 * |local_step|: anomalously large jump → use full projection deviation
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


def assign_error_model(
    ds: Dataset,
    sigma_floor: float | ArrayF | dict[str, float | ArrayF] = 1.0,
    gain: float | dict[str, float] | None = None,
    rel_error: float | dict[str, float] = 0.03,
) -> Dataset:
    """Assign heteroscedastic weights based on a physical detector noise model.

    Supports two model variants depending on which parameters are supplied:

    - **Full model** (``gain`` provided):
      ``sigma_i = sqrt(floor² + gain * max(y_i, 0) + (rel_error * y_i)²)``
    - **Simplified model** (``gain=0`` or omitted, proportional-only):
      ``sigma_i = sqrt(floor² + (rel_error * y_i)²)``

    When *gain* is ``None`` a per-label heuristic is used (y1: 1.8, y2: 0.7).
    Pass ``gain=0`` explicitly to use the simplified proportional model.

    Parameters
    ----------
    ds : Dataset
        The dataset to update.
    sigma_floor : float | ArrayF | dict[str, float | ArrayF]
        Baseline noise floor. Can be a single value or a per-label dict
        (e.g. ``{f"y{lbl}": float(np.mean(v)) for lbl, v in tit.bg_noise.items()}``).
    gain : float | dict[str, float] | None, optional
        Poisson shot-noise scaling factor. ``None`` -> per-label defaults
        (y1: 1.8, y2: 0.7). Pass ``0`` to disable the Poisson term entirely.
    rel_error : float | dict[str, float], optional
        Proportional error coefficient. Can be a per-label dict when estimated
        separately per label (e.g. from :func:`fit_rel_error_from_residuals`).
        Default is 0.03 (3 %).

    Returns
    -------
    Dataset
        A deep copy with physically modelled ``y_errc`` weights.

    Examples
    --------
    >>> import numpy as np
    >>> from clophfit.fitting.data_structures import Dataset, DataArray
    >>> y = np.array([100.0, 200.0, 300.0])
    >>> da = DataArray(xc=np.array([1.0, 2.0, 3.0]), yc=y, y_errc=np.ones_like(y))
    >>> ds = Dataset({"y1": da})
    >>> ds_new = assign_error_model(ds, sigma_floor=10.0, gain=0.0, rel_error=0.05)
    >>> np.round(ds_new["y1"].y_errc, 2)
    array([11.18, 14.14, 18.03])
    """
    updated_ds = copy.deepcopy(ds)
    gain_defaults = {"y1": 1.8, "y2": 0.7}
    for lbl, da in updated_ds.items():
        floor = sigma_floor[lbl] if isinstance(sigma_floor, dict) else sigma_floor
        alpha = rel_error[lbl] if isinstance(rel_error, dict) else rel_error

        if gain is None:
            poisson_term = gain_defaults.get(lbl, 0.0) * np.maximum(da.yc, 0.0)
        else:
            g = gain[lbl] if isinstance(gain, dict) else float(gain)
            poisson_term = g * np.maximum(da.yc, 0.0)

        floor_val = np.asarray(floor, dtype=float)
        new_err = np.sqrt(floor_val**2 + poisson_term + (alpha * da.yc) ** 2)
        da.y_errc = new_err

    return updated_ds


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
    >>> df = pd.DataFrame({"label": "y1", "resid_raw": resid, "predicted": y_pred})
    >>> alpha = fit_rel_error_from_residuals(df, sigma_floor={"y1": floor})
    >>> round(alpha["y1"], 2)  # should be close to true_alpha=0.02
    0.02
    """
    result: dict[str, float] = {}
    for lbl, grp in df.groupby("label"):
        lbl_str = str(lbl)
        r2_mean = float((grp["resid_raw"] ** 2).mean())
        pred2_mean = float((grp["predicted"] ** 2).mean())
        floor = sigma_floor.get(lbl_str, 0.0)
        alpha_sq = max(0.0, r2_mean - floor**2) / max(pred2_mean, 1e-12)
        result[lbl_str] = float(np.sqrt(alpha_sq))
    return result


def fit_noise_model_from_residuals(
    df: "pd.DataFrame",
    rel_error: float = 0.003,
) -> tuple[dict[str, float], dict[str, float]]:
    r"""Fit per-label noise model parameters from first-pass residuals.

    With ``rel_error`` fixed, the noise equation becomes linear in two unknowns.
    Rearranging: ``r_i^2 - (rel_error * y_i)^2 = sigma_read^2 + gain * y_i``
    → OLS on the adjusted residuals.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns ``label``, ``resid_raw``, ``y`` from a first-pass fit.
    rel_error : float, optional
        Fixed proportional error (default 0.003).

    Returns
    -------
    tuple[dict[str, float], dict[str, float]]
        ``(sigma_floor_dict, gain_dict)`` per label (non-negative, clamped).
    """
    sigma_floor: dict[str, float] = {}
    gain_d: dict[str, float] = {}
    for lbl, grp in df.groupby("label"):
        lbl_str = str(lbl)
        y = grp["y"].to_numpy()
        adjusted = grp["resid_raw"].to_numpy() ** 2 - (rel_error * y) ** 2
        # OLS: adjusted = sigma_read^2 + gain * y  →  x_mat = [1, y]
        x_mat = np.column_stack([np.ones_like(y), y])
        coeffs, _, _, _ = np.linalg.lstsq(x_mat, adjusted, rcond=None)
        sigma_floor[lbl_str] = float(np.sqrt(max(0.0, coeffs[0])))
        gain_d[lbl_str] = float(max(0.0, coeffs[1]))
    return sigma_floor, gain_d


def fit_gain_and_rel_error_from_residuals(
    df: "pd.DataFrame",
    sigma_floor: dict[str, float],
) -> tuple[dict[str, float], dict[str, float]]:
    r"""Fit gain and rel_error per label from residuals with known floor.

    No-intercept OLS on ``adjusted = gain * y + rel_error^2 * y^2``
    where ``adjusted = r^2 - floor^2``.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns ``label``, ``resid_raw``, ``y``.
    sigma_floor : dict[str, float]
        Known noise floor per label.

    Returns
    -------
    tuple[dict[str, float], dict[str, float]]
        ``(gain_dict, rel_error_dict)`` per label (non-negative, clamped).
    """
    gain_d: dict[str, float] = {}
    rel_error_d: dict[str, float] = {}
    for lbl, grp in df.groupby("label"):
        lbl_str = str(lbl)
        y = grp["y"].to_numpy()
        floor = sigma_floor.get(lbl_str, 0.0)
        adjusted = grp["resid_raw"].to_numpy() ** 2 - floor**2
        # No-intercept OLS: adjusted = gain * y + c * y^2
        x_mat = np.column_stack([y, y**2])
        coeffs, _, _, _ = np.linalg.lstsq(x_mat, adjusted, rcond=None)
        gain_d[lbl_str] = float(max(0.0, coeffs[0]))
        c_raw = coeffs[1]
        rel_error_d[lbl_str] = float(np.sqrt(max(0.0, float(c_raw))))
    return gain_d, rel_error_d
