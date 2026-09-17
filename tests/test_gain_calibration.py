"""Gain-only noise calibration: floor fixed or fitted, alpha at 0, one plate or many.

Plates are simulated from ``sigma^2 = floor^2 + gain * y`` on the pH design the
libraries use, so every estimator has a known answer to recover. The loop tests
check the part the estimators cannot: that fitting, pooling and refitting
converge to the true gain rather than to the one-third-low value uncorrected
residuals give.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from clophfit.fitting.core import fit_binding_glob
from clophfit.fitting.data_structures import DataArray, Dataset, NoiseModelParams
from clophfit.fitting.gain_calibration import (
    calibrate_gain,
    calibrate_plate_lm,
    calibrate_single_well,
    single_well_residuals,
)
from clophfit.fitting.models import binding_1site
from clophfit.fitting.noise_calibration import (
    dof_scale,
    fit_gain_and_floor,
    fit_gain_floor_fixed,
    fit_gain_nnls,
)

PH = np.array([8.9, 8.3, 7.7, 7.1, 6.5, 5.9, 5.3, 4.9])
FLOOR = {"1": 3.59, "2": 0.42}
GAIN = {"1": 1.0, "2": 0.41}


def _plate(
    n_wells: int, seed: int, floor: dict[str, float] | None = None
) -> dict[str, Dataset]:
    """Two-label pH wells with noise drawn from the gain-only model."""
    floor = floor or FLOOR
    rng = np.random.default_rng(seed)
    plate = {}
    for w in range(n_wells):
        k = rng.normal(7.0, 0.5)
        plateaus = {
            "1": (rng.uniform(300, 600), rng.uniform(900, 1500)),
            "2": (rng.uniform(150, 300), rng.uniform(20, 60)),
        }
        arrays = {}
        for lbl, (s0, s1) in plateaus.items():
            mu = binding_1site(PH, k, s0, s1, is_ph=True)
            sd = np.sqrt(floor[lbl] ** 2 + GAIN[lbl] * mu)
            y = mu + sd * rng.standard_normal(len(PH))
            arrays[lbl] = DataArray(PH.copy(), y, y_errc=np.full(len(PH), floor[lbl]))
        plate[f"W{w:02d}"] = Dataset(arrays, is_ph=True)
    return plate


def _residuals(
    n: int, floor: float, gain: float, *, seed: int = 0, label: str = "1"
) -> pd.DataFrame:
    """Draw a residual table straight from the noise model, no fit involved."""
    rng = np.random.default_rng(seed)
    yhat = rng.uniform(20.0, 1500.0, n)
    return pd.DataFrame({
        "label": label,
        "yhat": yhat,
        "raw_res": np.sqrt(floor**2 + gain * yhat) * rng.standard_normal(n),
    })


def test_dof_scale() -> None:
    """16 points and 5 parameters: residual variance is 11/16 of the noise."""
    assert dof_scale(16, 5) == pytest.approx(np.sqrt(16 / 11))
    assert np.isnan(dof_scale(5, 5))


def test_gain_floor_fixed_recovers_gain() -> None:
    """With the floor known, the gain alone is fitted."""
    df = _residuals(20000, 3.0, 0.8)
    assert fit_gain_floor_fixed(df, {"1": 3.0})["1"] == pytest.approx(0.8, rel=0.05)


def test_gain_and_floor_recovers_both() -> None:
    """A floor large against the gain term is identifiable, and is recovered."""
    df = _residuals(40000, 20.0, 0.5)
    floors, gains = fit_gain_and_floor(df)
    assert floors["1"] == pytest.approx(20.0, rel=0.05)
    assert gains["1"] == pytest.approx(0.5, rel=0.1)


def test_dof_scale_column_undoes_shrinkage() -> None:
    """Residuals shrunk by a fit's dof come back to the true gain via ``dof_scale``."""
    df = _residuals(20000, 3.0, 0.8)
    shrink = np.sqrt(11 / 16)
    df["raw_res"] *= shrink
    low = fit_gain_floor_fixed(df, {"1": 3.0})["1"]
    df["dof_scale"] = 1 / shrink
    assert low < 0.6
    assert fit_gain_floor_fixed(df, {"1": 3.0})["1"] == pytest.approx(0.8, rel=0.05)


def test_pooled_shares_gain_and_keeps_floors_per_plate() -> None:
    """Two plates, two floors, one gain: each plate's floor, the common gain."""
    df = pd.concat([
        _residuals(40000, 10.0, 0.5, seed=1).assign(plate="a"),
        _residuals(40000, 25.0, 0.5, seed=2).assign(plate="b"),
    ])
    out = fit_gain_nnls(df, fit_floor=True).set_index("plate")
    assert out.loc["a", "sigma_floor"] == pytest.approx(10.0, rel=0.1)
    assert out.loc["b", "sigma_floor"] == pytest.approx(25.0, rel=0.05)
    assert out.loc["a", "gain"] == out.loc["b", "gain"]
    assert out.loc["a", "gain"] == pytest.approx(0.5, rel=0.1)


def test_held_floor_is_required() -> None:
    """Holding the floor without supplying it is an error, not a zero floor."""
    with pytest.raises(ValueError, match="sigma_floor"):
        fit_gain_nnls(_residuals(10, 3.0, 0.8))


def test_single_well_residuals_carry_dof() -> None:
    """Each well's factor counts all its labels' points and its varied parameters."""
    plate = _plate(2, seed=3)
    table = single_well_residuals({w: fit_binding_glob(ds) for w, ds in plate.items()})
    assert set(table.columns) >= {"well", "label", "yhat", "raw_res", "dof_scale"}
    assert table["dof_scale"].iloc[0] == pytest.approx(np.sqrt(16 / 11))


@pytest.mark.parametrize("method", ["lm", "huber"])
def test_single_well_loop_converges_to_true_gain(method: str) -> None:
    """Fit, pool, calibrate, refit: the loop lands on the gain the data were drawn with."""
    cal = calibrate_single_well({"p": _plate(96, seed=4)}, {"p": FLOOR}, method=method)  # type: ignore[arg-type]
    assert cal.converged
    assert cal.n_iter <= 6
    for lbl in ("1", "2"):
        assert cal.noise["p"][lbl].gain == pytest.approx(GAIN[lbl], rel=0.15)
        assert cal.noise["p"][lbl].sigma_floor == FLOOR[lbl]
        assert cal.noise["p"][lbl].alpha == 0.0
    first = cal.history[cal.history["iteration"] == 0]
    assert (first["gain"] == 0).all()


def test_plate_lm_loop_converges_to_true_gain() -> None:
    """The plate-wide fitter reaches the same gain, and its profiled scale sits near 1."""
    cal = calibrate_plate_lm({"p": _plate(96, seed=5)}, {"p": {}}, {"p": FLOOR})
    assert cal.converged
    for lbl in ("1", "2"):
        assert cal.noise["p"][lbl].gain == pytest.approx(GAIN[lbl], rel=0.15)
        assert cal.fits["p"].ye_mag[lbl] == pytest.approx(1.0, abs=0.1)


def test_pooled_loop_over_plates() -> None:
    """Two plates with different floors share one fitted gain."""
    floor_b = {"1": 7.0, "2": 0.84}
    plates = {"a": _plate(96, seed=6), "b": _plate(96, seed=7, floor=floor_b)}
    cal = calibrate_single_well(plates, {"a": FLOOR, "b": floor_b})
    assert cal.converged
    for lbl in ("1", "2"):
        assert cal.noise["a"][lbl].gain == cal.noise["b"][lbl].gain
        assert cal.noise["a"][lbl].gain == pytest.approx(GAIN[lbl], rel=0.15)
        assert cal.noise["b"][lbl].sigma_floor == floor_b[lbl]


def test_loop_damps_a_two_cycle() -> None:
    """A fit whose residuals flip with the weights settles between the two gains.

    Undamped, this alternates 1.2, 0.8, 1.2, ... for ever, as huber did on L2:
    its threshold is in units of sigma, so the weights decide which points it
    down-weights and therefore the residuals the next gain is read from.
    """
    y = np.linspace(100.0, 1000.0, 200)

    def fit(
        _plate: str, noise: dict[str, NoiseModelParams]
    ) -> tuple[None, pd.DataFrame]:
        gain = 1.2 if noise["1"].gain < 1.0 else 0.8
        table = pd.DataFrame({"label": "1", "yhat": y, "raw_res": np.sqrt(gain * y)})
        return None, table

    cal = calibrate_gain(fit, {"p": {"1": 0.0}}, max_iter=40)
    assert cal.converged
    assert cal.noise["p"]["1"].gain == pytest.approx(1.0, abs=0.02)


def test_unconverged_loop_reports_the_weights_its_fits_used() -> None:
    """Stopped early, the noise returned is the one the returned fits were made under."""
    cal = calibrate_single_well({"p": _plate(8, seed=8)}, {"p": FLOOR}, max_iter=1)
    assert not cal.converged
    assert all(p.gain == 0.0 for p in cal.noise["p"].values())
