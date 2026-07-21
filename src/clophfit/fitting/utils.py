"""Utility functions for fitting modules."""

import copy

import numpy as np
import pandas as pd
from scipy import stats

from clophfit.clophfit_types import ArrayMask
from clophfit.fitting.data_structures import Dataset

# Per-method default thresholds. They are not interchangeable: for "mad" the
# number is a robust z cutoff, for "studentized" it is the family-wise alpha
# from which a Bonferroni-corrected Student-t cutoff is derived.
DEFAULT_OUTLIER_THRESHOLD = {"mad": 3.5, "studentized": 0.05}

# Outlier-screening methods accepted in a ``remove_outliers`` spec.
OUTLIER_METHODS = frozenset(DEFAULT_OUTLIER_THRESHOLD)


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
    - ``"mad:4.0:5"`` -> ("mad", 4.0, 5)
    - ``"mad"`` -> ("mad", 3.5, 1)
    - ``"studentized"`` -> ("studentized", 0.05, 1)
    """
    n_threshold_parts = 1
    n_min_keep_parts = 2
    parts = spec.split(":")
    method = parts[0]
    default_threshold = DEFAULT_OUTLIER_THRESHOLD.get(method, 3.5)
    threshold = float(parts[1]) if len(parts) > n_threshold_parts else default_threshold
    min_keep = int(parts[2]) if len(parts) > n_min_keep_parts else 1
    return method, threshold, min_keep


def robust_scale(residuals: np.ndarray) -> float:
    """Estimate a robust standard deviation from residuals via the MAD.

    The MAD is scaled to be a consistent estimator of the standard deviation
    under normality. It collapses to zero when more than half the residuals are
    identical, so the scale falls back to a normal-consistent IQR and then to
    the (non-robust) standard deviation.

    Parameters
    ----------
    residuals : np.ndarray
        The residuals to estimate a scale from. NaNs are ignored.

    Returns
    -------
    float
        A positive scale, or ``0.0`` if no positive scale can be found.
    """
    if len(residuals) == 0:
        return 0.0
    sigma = float(
        stats.median_abs_deviation(residuals, nan_policy="omit", scale="normal")
    )
    if not np.isfinite(sigma) or sigma <= 0:
        q25, q75 = np.nanpercentile(residuals, [25, 75])
        sigma = float((q75 - q25) / 1.349)
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.nanstd(residuals))
    return sigma if np.isfinite(sigma) and sigma > 0 else 0.0


def robust_z_scores(residuals: np.ndarray, sigma: float | None = None) -> np.ndarray:
    """Score residuals by a robust z built on the median and the MAD.

    A mean/standard-deviation z-score is inflated by the very outlier it is
    meant to expose, which caps the attainable score at ``sqrt(n - 1)``: at
    ``n = 7`` nothing can score above 2.45, and the score saturates rather than
    growing with the outlier. The median and MAD do not respond to the outlier,
    so this score has no such ceiling.

    Parameters
    ----------
    residuals : np.ndarray
        The residuals to score. NaNs are ignored when locating the centre and
        scale, and score as NaN.
    sigma : float | None
        Scale to divide by. ``None`` estimates it from *residuals* themselves.
        Pass a pooled scale (e.g. estimated plate-wide) when the per-fit sample
        is too small for a stable MAD.

    Returns
    -------
    np.ndarray
        Absolute robust z-scores, all zeros if no positive scale is available.
    """
    if len(residuals) == 0:
        return np.zeros(0, dtype=float)
    scale = robust_scale(residuals) if sigma is None else sigma
    if not np.isfinite(scale) or scale <= 0:
        return np.zeros(len(residuals), dtype=float)
    med = float(np.nanmedian(residuals))
    return np.abs((np.asarray(residuals, dtype=float) - med) / scale)


def studentized_scores(
    residuals: np.ndarray, jacobian: np.ndarray
) -> tuple[np.ndarray, int]:
    r"""Score residuals by the externally studentized (deleted) residual.

    Ordinary residuals are not comparable across points: a high-leverage point
    pulls the fit towards itself, so its residual is shrunk by
    :math:`\sqrt{1 - h_{ii}}` and it can hide from any test applied to raw
    residuals. This score divides that shrinkage out and rescales by a
    leave-one-out variance, so the point under test contributes nothing to the
    scale used to judge it:

    .. math::
        t_i = \frac{r_i}{s_{(i)}\sqrt{1 - h_{ii}}}

    Under the null it follows a Student-t with :math:`n - p - 1` degrees of
    freedom, which makes a calibrated threshold possible (see
    :func:`bonferroni_threshold`) rather than a hand-picked constant.

    Parameters
    ----------
    residuals : np.ndarray
        Residual vector actually minimized (weighted, if the fit was weighted),
        so that it is homoscedastic and matches *jacobian*.
    jacobian : np.ndarray
        The ``(n, p)`` Jacobian of those residuals at the solution.

    Returns
    -------
    tuple[np.ndarray, int]
        Absolute studentized residuals, and the degrees of freedom
        ``n - p - 1``. Scores are zeros when there is no residual freedom left.
    """
    r = np.asarray(residuals, dtype=float)
    n = r.size
    p = int(np.asarray(jacobian).shape[1])
    dof = n - p - 1
    if dof <= 0 or n == 0:
        return np.zeros(n, dtype=float), max(dof, 0)

    # Leverage from the hat matrix H = J (J'J)^-1 J', via a pivoted QR for
    # stability: h_ii is the row-wise squared norm of Q.
    q, _ = np.linalg.qr(np.asarray(jacobian, dtype=float))
    h = np.clip(np.einsum("ij,ij->i", q, q), 0.0, 1.0 - 1e-12)

    sse = float(r @ r)
    # Leave-one-out variance, in closed form (no refits needed).
    s2_i = (sse - r**2 / (1.0 - h)) / dof
    s2_i = np.where(s2_i > 0, s2_i, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = r / np.sqrt(s2_i * (1.0 - h))
    return np.abs(np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)), dof


def bonferroni_threshold(n: int, dof: int, alpha: float = 0.05) -> float:
    """Two-sided Student-t cutoff, Bonferroni-corrected for testing *n* points.

    Testing every point for outlyingness is *n* simultaneous tests, so the
    per-test level is ``alpha / n``. Without the correction, a plate of 90
    wells would flag points at the nominal rate by chance alone.

    Parameters
    ----------
    n : int
        Number of points being tested simultaneously.
    dof : int
        Degrees of freedom of the studentized residual (``n - p - 1``).
    alpha : float
        Family-wise error rate. Default 0.05.

    Returns
    -------
    float
        The cutoff, or ``inf`` when there is no residual freedom to test with.
    """
    if n <= 0 or dof <= 0:
        return float("inf")
    return float(stats.t.ppf(1.0 - alpha / (2.0 * n), dof))


def identify_outliers_mad(residuals: np.ndarray, threshold: float = 3.5) -> ArrayMask:
    """Identify outliers by robust z-score, using the median and the MAD.

    Parameters
    ----------
    residuals : np.ndarray
        The residuals to analyze. Use raw (unweighted) residuals, so that the
        scale is estimated from the data rather than from ``y_err``.
    threshold : float
        The robust z-score beyond which a point is considered an outlier.

    Returns
    -------
    ArrayMask
        A boolean mask where True indicates an outlier.
    """
    return np.asarray(robust_z_scores(residuals) > threshold, dtype=bool)


def cap_by_min_keep(flagged: ArrayMask, scores: np.ndarray, min_keep: int) -> ArrayMask:
    """Drop the weakest flags until at least `min_keep` points survive.

    Parameters
    ----------
    flagged : ArrayMask
        Boolean mask, True where a point is a candidate for removal.
    scores : np.ndarray
        Per-point outlier scores used to rank candidates worst-first.
    min_keep : int
        Minimum number of points that must remain unflagged.

    Returns
    -------
    ArrayMask
        A mask flagging at most ``len(flagged) - min_keep`` of the worst points.
    """
    allowed = max(len(flagged) - max(min_keep, 0), 0)
    if int(flagged.sum()) <= allowed:
        return flagged
    capped = np.zeros_like(flagged)
    if allowed:
        ranked = np.argsort(np.where(flagged, scores, -np.inf))[::-1][:allowed]
        capped[ranked] = True
    return capped


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


# Pooling levels for add_robust_scores, finest grouping first.
ROBUST_SCORE_LEVELS: dict[str, tuple[str, ...]] = {
    "well_label": ("well", "label"),
    "well": ("well",),
    "label": ("label",),
    "global": (),
}


def _level_is_degenerate(level: str, *, n_wells: int, n_labels: int) -> bool:
    """Say whether a pooling level collapses onto a finer one.

    A pooled level is only meaningful when it actually pools something the
    finer level does not: pooling wells needs more than one well, pooling
    labels more than one label.

    Parameters
    ----------
    level : str
        One of the keys of :data:`ROBUST_SCORE_LEVELS`.
    n_wells : int
        Distinct wells present.
    n_labels : int
        Distinct labels present.

    Returns
    -------
    bool
        True when the level carries no information beyond a finer one.
    """
    min_groups = 2
    if level == "well":
        return n_labels < min_groups
    if level == "label":
        return n_wells < min_groups
    if level == "global":
        return n_wells < min_groups and n_labels < min_groups
    return False


def add_robust_scores(
    residuals: pd.DataFrame,
    *,
    levels: tuple[str, ...] = ("well_label", "well", "label", "global"),
    residual_col: str = "raw_res",
) -> pd.DataFrame:
    """Add robust scale and z-score columns at several pooling levels.

    For each requested level this adds ``robust_sigma_{level}`` (a
    normal-consistent MAD estimate of the noise scale over that grouping) and
    ``robust_z_{level}`` (``|r - median| / sigma``, the median always taken
    per well and label, so only the *scale* is pooled).

    When ``"well_label"`` and ``"label"`` are both present it also adds
    ``ye_mag_est``, their scale ratio. That is a classical, sampler-free
    estimate of the per-well noise inflation the hierarchical model learns as
    ``ye_mag_{lbl}``, useful for cross-checking it cheaply.

    A level that is not identifiable from *residuals* -- a pooled level on a
    single-well table, say -- yields all-NaN columns rather than a silently
    degenerate duplicate of a finer level. Which levels those were is recorded
    in ``df.attrs["robust_score_degenerate_levels"]``.

    Parameters
    ----------
    residuals : pd.DataFrame
        Canonical residual table carrying ``well``, ``label`` and
        *residual_col*.
    levels : tuple[str, ...]
        Any of ``"well_label"``, ``"well"``, ``"label"``, ``"global"``.
        ``"well"`` pools the labels of one well, so it only means something
        when the labels share a noise scale; on plates where they do not (a
        bright band and a dim one, say) its scale is dominated by the noisier
        label and ``"well_label"`` is the one to use.
    residual_col : str
        Residual column to score. Defaults to the raw (unweighted) residual, so
        the scale is estimated from the data rather than from an assumed
        ``y_err``.

    Returns
    -------
    pd.DataFrame
        Copy of *residuals* with the added columns.

    Raises
    ------
    ValueError
        If *levels* names an unknown level, or *residual_col* is missing.
    """
    unknown = set(levels) - set(ROBUST_SCORE_LEVELS)
    if unknown:
        msg = f"Unknown level(s) {sorted(unknown)}; expected {sorted(ROBUST_SCORE_LEVELS)}."
        raise ValueError(msg)
    if residual_col not in residuals.columns:
        msg = f"residuals has no column {residual_col!r}."
        raise ValueError(msg)

    out = residuals.copy()
    n_wells = out["well"].nunique() if "well" in out.columns else 1
    n_labels = out["label"].nunique() if "label" in out.columns else 1
    degenerate: list[str] = []

    # Centre per well and label always: pooling is about scale, not location.
    centre_keys = [k for k in ("well", "label") if k in out.columns]
    centre = (
        out.groupby(centre_keys, observed=True)[residual_col].transform("median")
        if centre_keys
        else out[residual_col].median()
    )

    for level in levels:
        keys = [k for k in ROBUST_SCORE_LEVELS[level] if k in out.columns]
        # A pooled level needs something to pool over that the finer level lacks.
        if _level_is_degenerate(level, n_wells=n_wells, n_labels=n_labels):
            out[f"robust_sigma_{level}"] = np.nan
            out[f"robust_z_{level}"] = np.nan
            degenerate.append(level)
            continue
        if keys:
            sigma = out.groupby(keys, observed=True)[residual_col].transform(
                lambda s: robust_scale(s.to_numpy(dtype=float))
            )
        else:
            sigma = pd.Series(
                robust_scale(out[residual_col].to_numpy(dtype=float)), index=out.index
            )
        sigma = sigma.astype(float)
        out[f"robust_sigma_{level}"] = sigma
        out[f"robust_z_{level}"] = np.where(
            sigma > 0, (out[residual_col] - centre).abs() / sigma, np.nan
        )

    if {"well_label", "label"} <= set(levels) and "label" not in degenerate:
        denom = out["robust_sigma_label"]
        out["ye_mag_est"] = np.where(
            denom > 0, out["robust_sigma_well_label"] / denom, np.nan
        )
    out.attrs["robust_score_degenerate_levels"] = degenerate
    return out
