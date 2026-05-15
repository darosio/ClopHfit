"""Tests for geometric outlier scoring utilities."""

import numpy as np
import pandas as pd

from clophfit.fitting.data_structures import DataArray, Dataset
from clophfit.fitting.utils import (
    apply_outlier_mask,
    assign_error_model,
    fit_gain_and_rel_error_from_residuals,
    fit_noise_model_from_residuals,
    fit_rel_error_from_residuals,
    outlier_scores_extended,
)


def _make_da(x: np.ndarray, y: np.ndarray) -> DataArray:
    return DataArray(xc=x, yc=y)


def _make_ds(x: np.ndarray, y: np.ndarray, lbl: str = "y1") -> Dataset:
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
    da = result["y1"]
    assert not da.mask[2]


def test_apply_outlier_mask_preserves_good_data() -> None:
    """Monotone data should have all points preserved."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
    ds = _make_ds(x, y)
    result = apply_outlier_mask(ds, threshold=0.2)
    da = result["y1"]
    assert np.all(da.mask)


def test_apply_outlier_mask_iterative_advantage() -> None:
    """Iterative masking catches outlier with moderate threshold."""
    x = np.arange(7, dtype=float)
    y = np.array([1.0, 2.0, 3.0, 10.0, 5.0, 6.0, 7.0])
    ds = _make_ds(x, y)
    result = apply_outlier_mask(ds, threshold=0.15, min_keep=3)
    da = result["y1"]
    assert not da.mask[3]


def test_apply_outlier_mask_min_keep() -> None:
    """Masking stops when fewer than min_keep points would remain."""
    x = np.arange(4, dtype=float)
    y = np.array([1.0, 100.0, 200.0, 50.0])
    ds = _make_ds(x, y)
    result = apply_outlier_mask(ds, threshold=0.01, min_keep=3)
    da = result["y1"]
    assert da.mask.sum() >= 3


# --- fit_noise_model_from_residuals ---


def test_fit_noise_model_recovers_known_params() -> None:
    """Floor and gain estimates should be within 3x of true values."""
    rng = np.random.default_rng(42)
    true_floor, true_gain, rel = 5.0, 2.0, 0.003
    y = np.linspace(10, 1000, 300)
    sigma = np.sqrt(true_floor**2 + true_gain * y + (rel * y) ** 2)
    resid = sigma * rng.standard_normal(300)
    df = pd.DataFrame({"label": "y1", "resid_raw": resid, "y": y})
    floor_d, gain_d = fit_noise_model_from_residuals(df, rel_error=rel)
    assert abs(floor_d["y1"] - true_floor) <= true_floor
    assert abs(gain_d["y1"] - true_gain) < true_gain * 3


def test_fit_noise_model_fallback_on_negative_params() -> None:
    """Parameters should be clamped to non-negative values."""
    rng = np.random.default_rng(0)
    y = np.ones(50)
    resid = rng.standard_normal(50) * 0.01
    df = pd.DataFrame({"label": "y1", "resid_raw": resid, "y": y})
    floor_d, gain_d = fit_noise_model_from_residuals(df, rel_error=0.003)
    assert floor_d["y1"] >= 0.0
    assert gain_d["y1"] >= 0.0


def test_fit_noise_model_multi_label() -> None:
    """Function handles multiple labels in the DataFrame."""
    rng = np.random.default_rng(7)
    y = np.linspace(10, 500, 100)
    parts = []
    for lbl, floor in [("y1", 3.0), ("y2", 10.0)]:
        sigma = np.sqrt(floor**2 + 1.0 * y + (0.003 * y) ** 2)
        resid = sigma * rng.standard_normal(100)
        parts.append(pd.DataFrame({"label": lbl, "resid_raw": resid, "y": y}))
    df = pd.concat(parts, ignore_index=True)
    floor_d, _ = fit_noise_model_from_residuals(df, rel_error=0.003)
    assert "y1" in floor_d
    assert "y2" in floor_d


# --- fit_gain_and_rel_error_from_residuals ---


def test_fit_gain_and_rel_error_recovers_known_params() -> None:
    """Combined variance from estimates should be within 3x of true combined variance."""
    rng = np.random.default_rng(99)
    true_gain, true_alpha, floor = 2.0, 0.02, 5.0
    y = np.linspace(50, 500, 300)
    sigma = np.sqrt(floor**2 + true_gain * y + (true_alpha * y) ** 2)
    resid = sigma * rng.standard_normal(300)
    df = pd.DataFrame({"label": "y1", "resid_raw": resid, "y": y})
    gain_d, alpha_d = fit_gain_and_rel_error_from_residuals(df, {"y1": floor})
    assert gain_d["y1"] >= 0.0
    assert alpha_d["y1"] >= 0.0
    combined_est = gain_d["y1"] * np.mean(y) + (alpha_d["y1"] * np.mean(y)) ** 2
    combined_true = true_gain * np.mean(y) + (true_alpha * np.mean(y)) ** 2
    assert combined_est < 3 * combined_true


def test_fit_gain_and_rel_error_non_negative() -> None:
    """Estimated gain and rel_error should be non-negative."""
    rng = np.random.default_rng(1)
    y = np.ones(50) * 100
    resid = rng.standard_normal(50) * 0.001
    df = pd.DataFrame({"label": "y1", "resid_raw": resid, "y": y})
    gain_d, alpha_d = fit_gain_and_rel_error_from_residuals(df, {"y1": 10.0})
    assert gain_d["y1"] >= 0.0
    assert alpha_d["y1"] >= 0.0


# --- fit_rel_error_from_residuals ---


def test_fit_rel_error_recovers_known_alpha() -> None:
    """Estimated alpha should be within 20% of true alpha for N=400 samples."""
    rng = np.random.default_rng(0)
    y_pred = np.linspace(50, 500, 400)
    floor, true_alpha = 5.0, 0.02
    sigma = np.sqrt(floor**2 + (true_alpha * y_pred) ** 2)
    resid = sigma * rng.standard_normal(400)
    df = pd.DataFrame({"label": "y1", "resid_raw": resid, "predicted": y_pred})
    alpha = fit_rel_error_from_residuals(df, sigma_floor={"y1": floor})
    assert abs(alpha["y1"] - true_alpha) / true_alpha < 0.2


def test_fit_rel_error_multi_label() -> None:
    """Per-label alpha recovery should work for multiple labels."""
    rng = np.random.default_rng(3)
    y = np.linspace(50, 500, 200)
    parts = []
    for lbl, alpha_true, floor in [("y1", 0.02, 5.0), ("y2", 0.04, 3.0)]:
        sigma = np.sqrt(floor**2 + (alpha_true * y) ** 2)
        resid = sigma * rng.standard_normal(200)
        parts.append(pd.DataFrame({"label": lbl, "resid_raw": resid, "predicted": y}))
    df = pd.concat(parts, ignore_index=True)
    alpha = fit_rel_error_from_residuals(df, sigma_floor={"y1": 5.0, "y2": 3.0})
    assert "y1" in alpha
    assert "y2" in alpha
    assert alpha["y1"] >= 0.0
    assert alpha["y2"] >= 0.0


def test_fit_rel_error_non_negative_clamped() -> None:
    """Alpha should be clamped to 0 when residuals are smaller than floor."""
    rng = np.random.default_rng(5)
    y = np.ones(50) * 100
    resid = rng.standard_normal(50) * 0.001
    df = pd.DataFrame({"label": "y1", "resid_raw": resid, "predicted": y})
    alpha = fit_rel_error_from_residuals(df, sigma_floor={"y1": 100.0})
    assert alpha["y1"] >= 0.0


# --- assign_error_model ---


def test_assign_error_model_gain_zero_is_no_poisson() -> None:
    """With gain=0 the model should equal the pure floor+proportional model."""
    x = np.linspace(5, 10, 7)
    y = np.linspace(100, 500, 7)
    ds = _make_ds(x, y)
    result_gain0 = assign_error_model(ds, sigma_floor=2.0, gain=0, rel_error=0.03)
    result_no_poisson = assign_error_model(
        ds, sigma_floor=2.0, gain=0.0, rel_error=0.03
    )
    np.testing.assert_allclose(
        result_gain0["y1"].y_errc, result_no_poisson["y1"].y_errc
    )
    expected = np.sqrt(2.0**2 + (0.03 * y) ** 2)
    np.testing.assert_allclose(result_gain0["y1"].y_errc, expected, rtol=1e-6)
