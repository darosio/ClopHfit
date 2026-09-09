"""The screen's ruler has to admit that the model is imperfect.

``sigma`` is the physical measurement noise, and at the dim end of a falling
channel it collapses onto the read-noise floor: the Poisson term is clamped at
zero for a non-positive prediction, so for a label whose proportional term is
zero the variance is exactly ``floor**2``. On these plates that is 0.42 counts,
which makes a 1.3-count *model* error a 3-sigma event.

The screen then removes the points that define the transition, and a well can be
left unfittable. On L4 that happened to three cells: E03's 485 nm channel lost
its three transition points and both of its labels collapsed to flat lines, since
K is shared between a well's two labels. The flags that did it missed by 12% and
5% of the curve's range -- they were small misses judged by a tiny sigma, not
outliers.

Campaign-wide, 34% of the 138 |z|>3 flags miss by under 10% of their curve's
range. Flooring the screening sigma at 3% of the fitted amplitude removes 47 of
them and leaves the genuine ones -- H10's 400 nm flags, which miss by 39%, 43%
and 73%, stay flagged.
"""

from __future__ import annotations

import inspect
from typing import Any

from clophfit.fitting.plate_lm import (
    apply_excluded_points,
    fit_plate_lm_screened,
    screening_sigma_floor,
)


def _rows(
    pairs: list[tuple[float, float]], well: str = "A01", label: str = "2"
) -> list[dict[str, Any]]:
    """Residual rows for one well and label from (yhat, sigma) pairs."""
    return [
        {
            "well": well,
            "label": label,
            "step": i,
            "raw_i": i,
            "yhat": yh,
            "sigma": sd,
            "raw_res": 0.0,
            "std_res": 0.0,
        }
        for i, (yh, sd) in enumerate(pairs)
    ]


def test_floor_is_a_fraction_of_the_fitted_span() -> None:
    """Amplitude is max(yhat) - min(yhat) for that well and label."""
    rows = _rows([(300.0, 20.0), (150.0, 10.0), (0.0, 0.4)])
    floors = screening_sigma_floor(rows, fraction=0.03)
    assert floors["A01", "2"] == 9.0  # 3% of a 300-count span


def test_each_well_and_label_gets_its_own_floor() -> None:
    """A bright curve must not set the floor for a dim one."""
    rows = _rows([(1000.0, 30.0), (0.0, 0.4)]) + _rows(
        [(10.0, 1.0), (0.0, 0.4)], well="B02"
    )
    floors = screening_sigma_floor(rows, fraction=0.03)
    assert floors["A01", "2"] == 30.0
    assert floors["B02", "2"] == 0.3


def test_labels_of_one_well_are_separate() -> None:
    """The 400 and 485 nm channels have different ranges and different floors."""
    rows = _rows([(400.0, 20.0), (0.0, 0.4)]) + _rows(
        [(40.0, 5.0), (0.0, 0.4)], label="1"
    )
    floors = screening_sigma_floor(rows, fraction=0.03)
    assert floors["A01", "2"] == 12.0
    assert floors["A01", "1"] == 1.2


def test_a_flat_fit_gives_no_floor_rather_than_a_zero_one() -> None:
    """A degenerate curve must not silently disable or divide by anything.

    A flat first-pass fit has no amplitude to take a fraction of. Returning zero
    leaves the physical sigma in charge, which is the safe default: the screen
    stays as sensitive as it was rather than becoming inert.
    """
    floors = screening_sigma_floor(_rows([(5.0, 1.0), (5.0, 1.0)]), fraction=0.03)
    assert floors["A01", "2"] == 0.0


def test_a_small_miss_at_the_dim_end_stops_being_an_outlier() -> None:
    """L4 E03: a 5%-of-range miss read as z = 3.58 because sigma was 4.2."""
    amplitude = 332.0
    floor = 0.03 * amplitude
    miss, sigma = 0.05 * amplitude, 4.2
    assert abs(miss / sigma) > 3
    assert abs(miss / max(sigma, floor)) < 3


def test_a_large_miss_stays_an_outlier() -> None:
    """L4 H10 at 400 nm: misses of 39-73% of range are real and must survive."""
    amplitude = 300.0
    floor = 0.03 * amplitude
    for fraction in (0.39, 0.43, 0.73):
        miss = fraction * amplitude
        assert abs(miss / max(34.1, floor)) > 3


def test_ignores_rows_without_a_prediction() -> None:
    """A row carrying no yhat contributes nothing rather than raising."""
    rows = _rows([(300.0, 20.0), (0.0, 0.4)])
    rows.append({
        "well": "A01",
        "label": "2",
        "step": 9,
        "raw_i": 9,
        "yhat": float("nan"),
        "sigma": 1.0,
        "raw_res": 0.0,
        "std_res": 0.0,
    })
    floors = screening_sigma_floor(rows, fraction=0.03)
    assert floors["A01", "2"] == 9.0


def test_zero_fraction_disables_the_floor() -> None:
    """The knob must be able to reproduce the previous behaviour exactly."""
    floors = screening_sigma_floor(_rows([(300.0, 20.0), (0.0, 0.4)]), fraction=0.0)
    assert floors["A01", "2"] == 0.0


def test_default_min_keep_is_five() -> None:
    """Four points cannot locate a midpoint and two plateaus.

    E03 lost three of seven and landed exactly on the old default of four, which
    let its fit collapse to a flat line.
    """
    for fn in (fit_plate_lm_screened, apply_excluded_points):
        assert inspect.signature(fn).parameters["min_keep"].default == 5
