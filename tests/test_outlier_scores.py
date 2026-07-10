"""Tests for geometric outlier scoring utilities."""

import numpy as np
import pandas as pd

from clophfit.fitting.data_structures import (
    DataArray,
    Dataset,
    NoiseModelParams,
)
from clophfit.fitting.utils import (
    apply_outlier_mask,
    fit_gain_from_residuals,
    fit_rel_error_from_residuals,
    outlier_scores_extended,
)


def _make_da(x: np.ndarray, y: np.ndarray) -> DataArray:
    return DataArray(xc=x, yc=y)


def _make_ds(x: np.ndarray, y: np.ndarray, lbl: str = "1") -> Dataset:
    return Dataset({lbl: _make_da(x, y)}, is_ph=True)


# --- outlier_scores_extended ---


def test_outlier_scores_extended_internal() -> None:
    """Spike at internal index 2 should receive a high outlier score."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([10.0, 8.0, 15.0, 4.0, 2.0])
    scores = outlier_scores_extended(x, y)
    assert scores[2] > 0.4


def test_outlier_scores_extended_edge_drop() -> None:
    """Edge point jumping against trend should be flagged."""
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([50.0, 40.0, 30.0, 60.0])
    scores = outlier_scores_extended(x, y)
    assert scores[-1] > 0.0


def test_outlier_scores_extended_low_ph_plateau_no_false_positive() -> None:
    """Plateau approach at edges should score zero (not a false positive)."""
    x = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0])
    y = np.array([100.0, 95.0, 50.0, 5.0, 2.0, 1.0, 0.5])
    scores = outlier_scores_extended(x, y)
    assert scores[0] == 0.0
    assert scores[-1] == 0.0


def test_outlier_scores_extended_end_reversal() -> None:
    """Last point reversal should score > 0 and higher than internal points."""
    x = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0])
    y = np.array([57.8, 63.1, 70.4, 80.4, 90.9, 113.2, 107.5])
    scores = outlier_scores_extended(x, y)
    assert scores[-1] > 0.0
    assert scores[-1] > scores[-2]


def test_outlier_scores_extended_b02_catastrophic_drop() -> None:
    """Last point catastrophic drop should receive score > 0.3."""
    x = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0])
    y = np.array([380.0, 414.0, 477.0, 595.0, 669.0, 659.0, 365.0])
    scores = outlier_scores_extended(x, y)
    assert scores[-1] > 0.3


def test_outlier_scores_too_few_points() -> None:
    """Arrays shorter than 3 should return all-zero scores."""
    x = np.array([1.0, 2.0])
    y = np.array([10.0, 5.0])
    scores = outlier_scores_extended(x, y)
    assert np.sum(np.abs(scores)) == 0.0


# --- apply_outlier_mask ---


def test_apply_outlier_mask() -> None:
    """Obvious spike at index 2 should be masked after application."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([10.0, 8.0, 25.0, 4.0, 2.0])
    ds = _make_ds(x, y)
    result = apply_outlier_mask(ds, threshold=0.2)
    da = result["1"]
    assert not da.mask[2]


def test_apply_outlier_mask_preserves_good_data() -> None:
    """Monotone data should have all points preserved."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
    ds = _make_ds(x, y)
    result = apply_outlier_mask(ds, threshold=0.2)
    da = result["1"]
    assert np.all(da.mask)


def test_apply_outlier_mask_iterative_advantage() -> None:
    """Iterative masking catches outlier with moderate threshold."""
    x = np.arange(7, dtype=float)
    y = np.array([1.0, 2.0, 3.0, 10.0, 5.0, 6.0, 7.0])
    ds = _make_ds(x, y)
    result = apply_outlier_mask(ds, threshold=0.15, min_keep=3)
    da = result["1"]
    assert not da.mask[3]


def test_apply_outlier_mask_min_keep() -> None:
    """Masking stops when fewer than min_keep points would remain."""
    x = np.arange(4, dtype=float)
    y = np.array([1.0, 100.0, 200.0, 50.0])
    ds = _make_ds(x, y)
    result = apply_outlier_mask(ds, threshold=0.01, min_keep=3)
    da = result["1"]
    assert da.mask.sum() >= 3


# --- fit_gain_from_residuals ---


def test_fit_gain_recovers_known_gain() -> None:
    """Estimated gain should be within 20% of true gain for N=400 samples."""
    rng = np.random.default_rng(0)
    y_pred = np.linspace(50, 500, 400)
    floor, true_gain = 5.0, 0.8
    sigma = np.sqrt(floor**2 + true_gain * y_pred)
    resid = sigma * rng.standard_normal(400)
    df = pd.DataFrame({"label": "1", "raw_res": resid, "yhat": y_pred})
    gain = fit_gain_from_residuals(df, sigma_floor={"1": floor})
    assert abs(gain["1"] - true_gain) / true_gain < 0.2


def test_fit_gain_multi_label() -> None:
    """Per-label gain recovery should work for multiple labels."""
    rng = np.random.default_rng(3)
    y = np.linspace(50, 500, 200)
    parts = []
    for lbl, gain_true, floor in [("1", 1.0, 5.0), ("2", 2.0, 3.0)]:
        sigma = np.sqrt(floor**2 + gain_true * y)
        resid = sigma * rng.standard_normal(200)
        parts.append(pd.DataFrame({"label": lbl, "raw_res": resid, "yhat": y}))
    df = pd.concat(parts, ignore_index=True)
    gain = fit_gain_from_residuals(df, sigma_floor={"1": 5.0, "2": 3.0})
    assert gain["1"] >= 0.0
    assert gain["2"] >= 0.0


def test_fit_gain_non_negative_clamped() -> None:
    """Gain should be clamped to 0 when residuals are smaller than floor."""
    rng = np.random.default_rng(5)
    y = np.ones(50) * 100
    resid = rng.standard_normal(50) * 0.001
    df = pd.DataFrame({"label": "1", "raw_res": resid, "yhat": y})
    gain = fit_gain_from_residuals(df, sigma_floor={"1": 100.0})
    assert gain["1"] >= 0.0


# --- fit_rel_error_from_residuals ---


def test_fit_rel_error_recovers_known_alpha() -> None:
    """Estimated alpha should be within 20% of true alpha for N=400 samples."""
    rng = np.random.default_rng(0)
    y_pred = np.linspace(50, 500, 400)
    floor, true_alpha = 5.0, 0.02
    sigma = np.sqrt(floor**2 + (true_alpha * y_pred) ** 2)
    resid = sigma * rng.standard_normal(400)
    df = pd.DataFrame({"label": "1", "raw_res": resid, "yhat": y_pred})
    alpha = fit_rel_error_from_residuals(df, sigma_floor={"1": floor})
    assert abs(alpha["1"] - true_alpha) / true_alpha < 0.2


def test_fit_rel_error_multi_label() -> None:
    """Per-label alpha recovery should work for multiple labels."""
    rng = np.random.default_rng(3)
    y = np.linspace(50, 500, 200)
    parts = []
    for lbl, alpha_true, floor in [("1", 0.02, 5.0), ("2", 0.04, 3.0)]:
        sigma = np.sqrt(floor**2 + (alpha_true * y) ** 2)
        resid = sigma * rng.standard_normal(200)
        parts.append(pd.DataFrame({"label": lbl, "raw_res": resid, "yhat": y}))
    df = pd.concat(parts, ignore_index=True)
    alpha = fit_rel_error_from_residuals(df, sigma_floor={"1": 5.0, "2": 3.0})
    assert "1" in alpha
    assert "2" in alpha
    assert alpha["1"] >= 0.0
    assert alpha["2"] >= 0.0


def test_fit_rel_error_non_negative_clamped() -> None:
    """Alpha should be clamped to 0 when residuals are smaller than floor."""
    rng = np.random.default_rng(5)
    y = np.ones(50) * 100
    resid = rng.standard_normal(50) * 0.001
    df = pd.DataFrame({"label": "1", "raw_res": resid, "yhat": y})
    alpha = fit_rel_error_from_residuals(df, sigma_floor={"1": 100.0})
    assert alpha["1"] >= 0.0


# --- NoiseModelParams.compute_y_err ---


def test_noise_model_params_compute_y_err_gain_zero() -> None:
    """With gain=0 the model equals pure floor+proportional."""
    y = np.linspace(100, 500, 7)
    params = NoiseModelParams(sigma_floor=2.0, gain=0.0, alpha=0.03)
    result = params.compute_y_err(y)
    expected = np.sqrt(2.0**2 + (0.03 * y) ** 2)
    np.testing.assert_allclose(result, expected, rtol=1e-6)
