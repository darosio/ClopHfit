"""Residual assumption tests: each detects the failure it names, and only that one.

Residual tables are simulated around known binding curves, so every test has a
right answer: a fan for Breusch-Pagan, a U-shaped variance only White sees, a
wave for Durbin-Watson, heavy tails for Shapiro-Wilk, and a planted point for
the studentized t-test and PRESS.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from clophfit.fitting.models import binding_1site
from clophfit.fitting.residual_tests import (
    _well_linearised,  # ruff: ignore[import-private-name] - the least-squares residuals under test
    breusch_pagan,
    durbin_watson,
    durbin_watson_null,
    normality_tests,
    press_statistics,
    residual_tests,
    runs_test,
    runs_test_null,
    studentized_outliers,
    well_leverage,
    white_test,
)

PH = np.array([8.9, 8.3, 7.7, 7.1, 6.5, 5.9, 5.3])


def _plate(
    n_wells: int = 40, *, seed: int = 0, noise: str = "normal", ar: float = 0.0
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Simulate a residual table and its true parameters, both labels per well."""
    rng = np.random.default_rng(seed)
    rows, params = [], {}
    for w in range(n_wells):
        well = f"W{w:02d}"
        p = {
            "K": rng.normal(7.0, 0.4),
            "S0_1": 400.0,
            "S1_1": 1200.0,
            "S0_2": 250.0,
            "S1_2": 40.0,
        }
        params[well] = p
        for lbl, sigma in (("1", 20.0), ("2", 5.0)):
            yhat = binding_1site(PH, p["K"], p[f"S0_{lbl}"], p[f"S1_{lbl}"], is_ph=True)
            if noise == "t2":
                e = rng.standard_t(2, PH.size)
            else:
                e = rng.standard_normal(PH.size)
            if ar:
                for i in range(1, e.size):
                    e[i] = ar * e[i - 1] + np.sqrt(1 - ar**2) * e[i]
            raw = sigma * e
            rows += [
                {
                    "well": well,
                    "label": lbl,
                    "step": i,
                    "x": PH[i],
                    "yhat": yhat[i],
                    "raw_res": raw[i],
                    "sigma": sigma,
                    "std_res": raw[i] / sigma,
                }
                for i in range(PH.size)
            ]
    return pd.DataFrame(rows), params


def test_breusch_pagan_sees_a_fan_and_not_its_absence() -> None:
    """A spread that grows with the prediction is detected; a flat one is not."""
    rng = np.random.default_rng(1)
    y = rng.uniform(100, 1000, 2000)
    x = rng.uniform(5, 9, 2000)
    flat = breusch_pagan(rng.standard_normal(2000), y, x)
    fan = breusch_pagan(rng.standard_normal(2000) * y / 500, y, x)
    assert flat["p"] > 0.01
    assert fan["p"] < 1e-10
    assert fan["r2"] > 10 * flat["r2"]


def test_white_sees_a_variance_breusch_pagan_cannot() -> None:
    """High variance at both ends of the signal: linear BP misses it, White does not."""
    rng = np.random.default_rng(2)
    y = rng.uniform(-1, 1, 3000)
    x = rng.uniform(5, 9, 3000)
    z = rng.standard_normal(3000) * (0.3 + 2 * y**2)
    assert breusch_pagan(z, y, x)["r2"] < 0.01
    assert white_test(z, y, x)["p"] < 1e-10


def test_durbin_watson_separates_a_wave_from_independent_steps() -> None:
    """Near 2 for independent errors; well below it when neighbouring steps share one."""
    iid, _ = _plate(seed=3)
    wave, _ = _plate(seed=3, ar=0.8)
    # Seven-point series: independence gives 2 (n - 1) / n = 1.71, not 2.
    assert durbin_watson(iid)["dw_expected"] == pytest.approx(12 / 7)
    assert durbin_watson(iid)["dw"] == pytest.approx(12 / 7, abs=0.2)
    assert durbin_watson(wave)["dw"] < 1.0
    assert durbin_watson(wave)["lag1_r"] > 0.5


def test_durbin_watson_never_differences_across_wells() -> None:
    """A table whose wells each hold a constant offset has no within-series wave."""
    rows = [
        {"well": f"W{w}", "label": "1", "step": i, "std_res": (-1.0) ** w}
        for w in range(20)
        for i in range(7)
    ]
    assert durbin_watson(pd.DataFrame(rows))["dw"] == pytest.approx(0.0)


def _least_squares_residuals(
    table: pd.DataFrame, params: dict[str, dict[str, float]]
) -> pd.DataFrame:
    """Replace each well's residuals by those of its linearised least-squares fit."""
    out = table.sort_values(["well", "label", "step"]).copy()
    for well, g in out.groupby("well", sort=False):
        lin = _well_linearised(g, params[str(well)], is_ph=True)
        assert lin is not None
        _, r_ls, w, _ = lin
        out.loc[g.index, "std_res"] = r_ls
        out.loc[g.index, "raw_res"] = r_ls / w
    return out


def test_durbin_watson_null_predicts_fitted_residuals() -> None:
    """Fitted iid noise alternates: DW lands on the hat-matrix null, well above 12/7."""
    table, params = _plate(n_wells=300, seed=9)
    fitted = _least_squares_residuals(table, params)
    null = durbin_watson_null(fitted, params).to_dict(orient="index")
    for lbl in ("1", "2"):
        observed = durbin_watson(fitted[fitted.label == lbl])
        assert null[lbl]["dw_expected_fit"] > 2.0
        assert observed["dw"] == pytest.approx(null[lbl]["dw_expected_fit"], abs=0.1)
        assert observed["lag1_r"] == pytest.approx(null[lbl]["lag1_expected"], abs=0.06)
    wave = _least_squares_residuals(*_plate(n_wells=300, seed=9, ar=0.8))
    wave_dw = durbin_watson(wave[wave.label == "1"])["dw"]
    assert wave_dw < null["1"]["dw_expected_fit"] - 0.5


def test_normality_separates_heavy_tails_from_gaussian() -> None:
    """Student-t(2) residuals fail Shapiro-Wilk and Lilliefors; Gaussian ones pass."""
    rng = np.random.default_rng(4)
    gauss = normality_tests(rng.standard_normal(800), n_mc=299)
    heavy = normality_tests(rng.standard_t(2, 800), n_mc=299)
    assert gauss["sw_w"] > 0.99
    assert gauss["lillie_p"] > 0.01
    assert heavy["sw_p"] < 1e-6
    assert heavy["lillie_p"] < 0.01
    assert gauss["ad_p"] > 0.01
    assert heavy["ad_p"] < 0.01


def test_ks_against_n01_sees_scale_that_lilliefors_ignores() -> None:
    """Residuals shrunk to SD 0.8 are Normal in shape but not N(0, 1)."""
    z = 0.8 * np.random.default_rng(5).standard_normal(2000)
    out = normality_tests(z, n_mc=299)
    assert out["ks_n01_p"] < 1e-4
    assert out["lillie_p"] > 0.01


def test_leverage_sums_to_the_parameter_count() -> None:
    """The hat matrix trace equals p: here K plus two plateaus per label, 5 per well."""
    table, params = _plate(n_wells=6, seed=6)
    h = well_leverage(table, params)
    assert h.between(0, 1).all()
    per_well = h.groupby(table["well"]).sum()
    assert np.allclose(per_well, 5.0, atol=1e-6)


def test_studentized_t_flags_a_planted_point_and_press_feels_it() -> None:
    """One point 15 sigma off is an outlier; clean wells are almost never flagged."""
    table, params = _plate(n_wells=40, seed=7)
    clean_press = press_statistics(table, params)
    hit = table.index[(table.well == "W05") & (table.label == "1") & (table.step == 3)][
        0
    ]
    table.loc[hit, "raw_res"] += 15 * table.loc[hit, "sigma"]
    table.loc[hit, "std_res"] = table.loc[hit, "raw_res"] / table.loc[hit, "sigma"]
    out = studentized_outliers(table, params)
    assert bool(out.loc[hit, "outlier"])
    others = out.drop(index=table.index[table.well == "W05"])
    assert others["outlier"].groupby(table["well"]).any().mean() <= 0.1
    dirty_press = press_statistics(table, params)
    assert dirty_press["press"] > clean_press["press"]
    assert dirty_press["press_ratio"] >= 1.0


def test_residual_tests_reports_every_family_per_label() -> None:
    """One row per label with all four families; leverage columns need parameters."""
    table, params = _plate(n_wells=30, seed=8)
    full = residual_tests(table, params, n_mc=199)
    assert list(full.index) == ["1", "2"]
    for col in (
        "bp_r2",
        "white_p",
        "dw",
        "dw_expected_fit",
        "lag1_expected",
        "sw_w",
        "lillie_p",
        "ad_p",
        "runs",
        "runs_p",
        "runs_expected_fit",
        "runs_p_fit",
        "t_max",
        "outlier_rate",
        "press_ratio",
    ):
        assert col in full
        assert full[col].notna().all()
    bare = residual_tests(table, None, n_mc=199)
    assert bare["dw"].notna().all()
    assert "press_ratio" not in bare or bare["press_ratio"].isna().all()
    assert "dw_expected_fit" not in bare
    assert bare["runs_p"].notna().all()
    assert "runs_p_fit" not in bare


def test_runs_test_sees_a_wave_and_passes_independent_errors() -> None:
    """Independent errors match the textbook runs null; a wave has far too few runs."""
    iid, _ = _plate(n_wells=200, seed=10)
    wave, _ = _plate(n_wells=200, seed=10, ar=0.8)
    ok = runs_test(iid)
    bad = runs_test(wave)
    assert ok["runs_p"] > 0.01
    assert ok["n_series"] == 400
    assert bad["runs"] < bad["runs_expected"]
    assert bad["runs_p"] < 1e-10


def test_runs_test_ignores_heavy_tails() -> None:
    """Student-t(2) errors, which inflate DW's sums of squares, leave the sign runs calibrated."""
    heavy, _ = _plate(n_wells=200, seed=11, noise="t2")
    assert runs_test(heavy)["runs_p"] > 0.01


def test_runs_test_skips_series_without_both_signs() -> None:
    """A one-signed series has a single run and no variance; it adds nothing."""
    rows = [
        {"well": "W0", "label": "1", "step": i, "std_res": 1.0 + i} for i in range(7)
    ]
    out = runs_test(pd.DataFrame(rows))
    assert out["runs"] == pytest.approx(1.0)
    assert np.isnan(out["runs_p"])


def test_runs_test_null_predicts_fitted_residuals() -> None:
    """Fitted iid noise alternates: runs exceed the textbook null and match the simulated one."""
    table, params = _plate(n_wells=300, seed=12)
    fitted = _least_squares_residuals(table, params)
    null = runs_test_null(fitted, params, n_mc=499).to_dict(orient="index")
    tests = residual_tests(fitted, params, n_mc=499).to_dict(orient="index")
    for lbl in ("1", "2"):
        observed = runs_test(fitted[fitted.label == lbl])
        expected = null[lbl]["runs_expected_fit"]
        assert expected > observed["runs_expected"]
        assert abs(observed["runs"] - expected) < 3 * null[lbl]["runs_sd_fit"]
        assert tests[lbl]["runs_p_fit"] > 0.01
    wave = _least_squares_residuals(*_plate(n_wells=300, seed=12, ar=0.8))
    wave_row = residual_tests(wave, params, n_mc=499).to_dict(orient="index")["1"]
    assert wave_row["runs"] < wave_row["runs_expected_fit"]
    assert wave_row["runs_p_fit"] < 0.01
