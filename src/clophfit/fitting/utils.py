"""Utility functions for fitting modules."""

import copy

import numpy as np
import pandas as pd
from scipy import stats

from clophfit.clophfit_types import ArrayMask
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
