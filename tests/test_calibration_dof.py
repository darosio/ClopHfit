"""Every noise calibration reads the residual variance a fit left, corrected for what it took.

A fit that spends ``p`` parameters on ``n`` points leaves residuals whose mean
square is ``(n - p) / n`` of the noise variance. On a seven-step, two-label pH
well that is ~0.64, and a gain, alpha or scale read from those residuals
without ``sqrt(n / (n - p))`` comes out a third low. These tests simulate the
known answer and check each estimator recovers it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from clophfit.fitting.core import fit_binding_glob
from clophfit.fitting.data_structures import DataArray, Dataset, NoiseModelParams
from clophfit.fitting.models import binding_1site
from clophfit.fitting.noise_calibration import (
    fit_gain_from_residuals,
    fit_noise_model_nnls,
)
from clophfit.fitting.plate_lm import fit_plate_lm
from clophfit.prtecan.titration import (
    _with_well_dof_scale,  # ruff: ignore[import-private-name]
)

PH = np.array([8.9, 8.3, 7.7, 7.1, 6.5, 5.9, 5.3])
FLOOR, GAIN = 4.0, 1.0


def _shrunk_residuals(n_rows: int, shrink: float, seed: int) -> pd.DataFrame:
    """Residuals of known floor/gain noise, shrunk as a fit shrinks them."""
    rng = np.random.default_rng(seed)
    yhat = rng.uniform(100.0, 2000.0, n_rows)
    sigma = np.sqrt(FLOOR**2 + GAIN * yhat)
    return pd.DataFrame({
        "label": "1",
        "yhat": yhat,
        "raw_res": shrink * sigma * rng.standard_normal(n_rows),
    })


def test_nnls_restores_the_gain_through_dof_scale() -> None:
    """Uncorrected, the gain is low by (n - p) / n; with dof_scale it is recovered."""
    shrink2 = 9.0 / 14.0  # five parameters on fourteen points
    df = _shrunk_residuals(20000, np.sqrt(shrink2), seed=1)
    floor = {"1": FLOOR}
    _, raw, _ = fit_noise_model_nnls(
        df, sigma_floor_fixed=floor, rel_error_fixed={"1": 0}
    )
    df["dof_scale"] = 1.0 / np.sqrt(shrink2)
    _, fixed, _ = fit_noise_model_nnls(
        df, sigma_floor_fixed=floor, rel_error_fixed={"1": 0}
    )
    assert raw["1"] == pytest.approx(
        GAIN * shrink2 - FLOOR**2 * (1 - shrink2) / 1050, rel=0.06
    )
    assert fixed["1"] == pytest.approx(GAIN, rel=0.03)
    moment = fit_gain_from_residuals(df, floor)
    assert moment["1"] == pytest.approx(GAIN, rel=0.03)


def _plate(
    n_wells: int, seed: int, *, sigma: float | None = None
) -> dict[str, Dataset]:
    """Seven-step two-label wells with floor + gain noise (or a flat *sigma*)."""
    rng = np.random.default_rng(seed)
    out: dict[str, Dataset] = {}
    for i in range(n_wells):
        k = rng.uniform(6.3, 7.9)
        arrays = {}
        for lbl, (s0, s1) in (("1", (300.0, 1800.0)), ("2", (1500.0, 150.0))):
            clean = binding_1site(PH, k, s0, s1, is_ph=True)
            sd = np.full(PH.size, sigma) if sigma else np.sqrt(FLOOR**2 + GAIN * clean)
            arrays[lbl] = DataArray(
                PH, clean + rng.normal(0.0, sd), y_errc=np.full(PH.size, sigma or FLOOR)
            )
        out[f"W{i:03d}"] = Dataset(arrays, is_ph=True)
    return out


def test_plate_calibration_recovers_the_gain() -> None:
    """fit_plate_lm's calibrated gain is unbiased; it used to land near 0.64."""
    model = {
        lbl: NoiseModelParams(sigma_floor=FLOOR, gain=0.0, alpha=0.0)
        for lbl in ("1", "2")
    }
    r = fit_plate_lm(
        _plate(120, seed=2),
        groups={},
        noise_model=model,
        calibrate_noise=True,
        noise_free=("gain",),
        max_iter=10,
    )
    for lbl in ("1", "2"):
        assert r.noise[lbl]["gain"] == pytest.approx(GAIN, rel=0.12)


def test_robust_scale_takes_the_same_correction_as_least_squares() -> None:
    """With y_err equal to the true sigma, both losses profile a scale near 1."""
    plate = _plate(80, seed=3, sigma=20.0)
    linear = fit_plate_lm(plate, groups={}, loss="linear")
    huber = fit_plate_lm(plate, groups={}, loss="huber")
    for lbl in ("1", "2"):
        assert linear.ye_mag[lbl] == pytest.approx(1.0, abs=0.06)
        assert huber.ye_mag[lbl] == pytest.approx(1.0, abs=0.08)


def test_fgls_scales_each_well_and_drops_wells_without_freedom() -> None:
    """Per-well sqrt(n / (n - p)); a well with no residual freedom says nothing."""
    plate = _plate(2, seed=4)
    results = {well: fit_binding_glob(ds, method="lm") for well, ds in plate.items()}
    rows = [
        {"well": well, "label": lbl, "raw_res": 1.0, "yhat": 100.0}
        for well, n in (("W000", 7), ("W001", 2))
        for lbl in ("1", "2")
        for _ in range(n)
    ]
    out = _with_well_dof_scale(pd.DataFrame(rows), results)
    assert set(out["well"]) == {"W000"}
    assert out["dof_scale"].iloc[0] == pytest.approx(np.sqrt(14 / 9))
