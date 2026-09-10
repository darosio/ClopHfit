"""Let the iterative calibration hold terms, so it can converge at all.

``fit_plate_lm`` refits its noise model until the terms stop moving, but it
could only ever do that with the floor pinned and both signal terms free --
the one case where gain and alpha are collinear over a plate's signal range,
so the pair wanders along a ridge rather than settling.

``noise_free`` names the terms allowed to move; anything else keeps the value
the supplied model gave it. That makes the four comparisons expressible:

=========================  ===========================================
``noise_free``             what it fits
=========================  ===========================================
``("gain", "alpha")``      the previous behaviour, floor pinned
``("gain",)``              floor pinned, alpha held (at 0 if supplied 0)
``("alpha",)``             floor pinned, gain held
``("floor", "gain")``      floor free as well
=========================  ===========================================
"""

from __future__ import annotations

import numpy as np
import pytest

from clophfit.fitting.data_structures import DataArray, Dataset, NoiseModelParams
from clophfit.fitting.plate_lm import fit_plate_lm


def _plate() -> dict[str, Dataset]:
    """Build a small plate with signal-dependent noise on both labels."""
    rng = np.random.default_rng(5)
    x = np.linspace(5.0, 9.0, 12)
    out: dict[str, Dataset] = {}
    for i in range(8):
        k = 7.0 + 0.1 * i
        arrays = {}
        for lbl, (s0, s1) in (("1", (100.0, 900.0)), ("2", (800.0, 60.0))):
            clean = s1 + (s0 - s1) / (1.0 + 10.0 ** (x - k))
            sigma = np.sqrt(4.0**2 + 0.5 * np.maximum(clean, 0) + (0.03 * clean) ** 2)
            arrays[lbl] = DataArray(
                x, clean + rng.normal(0.0, sigma), y_errc=np.full(x.size, 4.0)
            )
        out[f"A{i:02d}"] = Dataset(arrays, is_ph=True)
    return out


def _model(**kw: float) -> dict[str, NoiseModelParams]:
    base = {"sigma_floor": 4.0, "gain": 0.0, "alpha": 0.0} | kw
    return {lbl: NoiseModelParams(**base) for lbl in ("1", "2")}  # type: ignore[arg-type]


def test_the_default_still_fits_both_signal_terms() -> None:
    """Existing behaviour: floor pinned, gain and alpha both estimated."""
    r = fit_plate_lm(_plate(), groups={}, noise_model=_model(), calibrate_noise=True)
    assert r.noise["1"]["gain"] > 0.0 or r.noise["1"]["alpha"] > 0.0


def test_holding_alpha_leaves_it_where_it_was() -> None:
    """Case (a)/(c): alpha stays at the supplied value, gain is fitted."""
    r = fit_plate_lm(
        _plate(),
        groups={},
        noise_model=_model(),
        calibrate_noise=True,
        noise_free=("gain",),
    )
    assert r.noise["1"]["alpha"] == 0.0
    assert r.noise["1"]["gain"] > 0.0


def test_holding_gain_leaves_it_where_it_was() -> None:
    """Case (b)/(d): gain stays put, alpha is fitted."""
    r = fit_plate_lm(
        _plate(),
        groups={},
        noise_model=_model(),
        calibrate_noise=True,
        noise_free=("alpha",),
    )
    assert r.noise["1"]["gain"] == 0.0
    assert r.noise["1"]["alpha"] > 0.0


def test_a_held_term_keeps_a_non_zero_supplied_value() -> None:
    """Holding is not zeroing: a supplied gain survives the calibration."""
    r = fit_plate_lm(
        _plate(),
        groups={},
        noise_model=_model(gain=0.7),
        calibrate_noise=True,
        noise_free=("alpha",),
    )
    assert r.noise["1"]["gain"] == pytest.approx(0.7)


def test_the_floor_can_be_freed_too() -> None:
    """Case (c)/(d) with the floor centred rather than pinned."""
    held = fit_plate_lm(
        _plate(),
        groups={},
        noise_model=_model(),
        calibrate_noise=True,
        noise_free=("gain", "alpha"),
    )
    freed = fit_plate_lm(
        _plate(),
        groups={},
        noise_model=_model(),
        calibrate_noise=True,
        noise_free=("floor", "gain", "alpha"),
    )
    assert held.noise["1"]["sigma_floor"] == pytest.approx(4.0)
    assert freed.noise["1"]["sigma_floor"] != pytest.approx(4.0)


def test_holding_everything_returns_the_supplied_model() -> None:
    """Nothing free is a valid request, not an error."""
    r = fit_plate_lm(
        _plate(),
        groups={},
        noise_model=_model(gain=0.5, alpha=0.02),
        calibrate_noise=True,
        noise_free=(),
    )
    assert r.noise["1"]["sigma_floor"] == pytest.approx(4.0)
    assert r.noise["1"]["gain"] == pytest.approx(0.5)
    assert r.noise["1"]["alpha"] == pytest.approx(0.02)
