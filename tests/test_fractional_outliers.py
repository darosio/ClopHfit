"""Screen the 400 nm channel on fractional deviation, not on a z-score.

A standardised residual fails at both ends of a titration. Sigma tracks the
signal, while model error tracks the curve, so at the dim end a 5% miss reads as
3.6 sigma and the screen deletes the points that pin the plateau, while at the
bright end a 36% miss reads as 2.7 sigma and the screen sees nothing. Both were
observed: L4 E03 was gutted by the first, and six L10b wells carry an
unscreened acidic turnover from the second.

The reviewer's calls separate cleanly on ``|residual / prediction|`` instead:
keeps at 0.097-0.111, discards from 0.126 up, over three plates. That matches
the physics -- the noise is multiplicative at bright signal and the documented
step-0 artefact is a uniform ~7% multiplicative deficit, so 10% is ordinary
variation and 12% is outside it.

The second clause is the reviewer's own observation on L3 C06, where both
channels read ~15% high at one step and ~15% low at the next. That is a
well-level multiplicative artefact, which a ratiometric measurement cancels
exactly; removing those points would bias the plateau rather than clean it. So a
point is spared when the other channel moves with it, in the same direction and
by a comparable fraction.
"""

from __future__ import annotations

from typing import Any

from clophfit.fitting.plate_lm import fractional_outliers


def _row(well: str, label: str, step: int, y: float, yhat: float) -> dict[str, Any]:
    return {
        "well": well,
        "label": label,
        "step": step,
        "raw_i": step,
        "y": y,
        "yhat": yhat,
        "sigma": 1.0,
        "raw_res": y - yhat,
        "std_res": 0.0,
    }


def test_a_large_fractional_miss_is_screened() -> None:
    """L10b A03 step 6: 20% below prediction, label 2 unmoved."""
    rows = [_row("A03", "1", 6, 2563.6, 3222.7), _row("A03", "2", 6, 100.0, 104.8)]
    assert fractional_outliers(rows) == {("A03", "1"): {6}}


def test_an_ordinary_fractional_miss_is_left_alone() -> None:
    """L4 B02 step 3: 11% off, which is inside normal variation."""
    rows = [_row("B02", "1", 3, 697.53, 627.65), _row("B02", "2", 3, 100.0, 102.4)]
    assert fractional_outliers(rows) == {}


def test_a_well_level_artefact_is_spared() -> None:
    """L3 C06 step 0: both channels read ~15% high, so the ratio cancels it."""
    rows = [_row("C06", "1", 0, 805.63, 694.14), _row("C06", "2", 0, 1169.86, 1018.57)]
    assert fractional_outliers(rows) == {}


def test_a_single_channel_deviation_is_not_spared() -> None:
    """L3 A09 step 0: label 1 off by 13%, label 2 by 0.1%."""
    rows = [_row("A09", "1", 0, 869.27, 996.46), _row("A09", "2", 0, 1323.5, 1322.53)]
    assert fractional_outliers(rows) == {("A09", "1"): {0}}


def test_opposite_signs_are_never_ratiometric() -> None:
    """L4 H10 step 6: label 1 down 114%, label 2 up 452% - not a shared gain."""
    rows = [_row("H10", "1", 6, -20.0, 150.0), _row("H10", "2", 6, 550.0, 100.0)]
    assert fractional_outliers(rows) == {("H10", "1"): {6}}


def test_only_the_first_label_is_screened() -> None:
    """Label 2's plateau goes to zero, where a fractional test misbehaves.

    Its failure mode is different and is handled by the z-screen with the
    amplitude floor, so this criterion deliberately does not touch it.
    """
    rows = [_row("A01", "1", 0, 100.0, 100.0), _row("A01", "2", 0, 1.0, 50.0)]
    assert fractional_outliers(rows) == {}


def test_a_missing_partner_channel_does_not_spare_the_point() -> None:
    """A single-label well cannot cancel anything, so the point stands or falls."""
    rows = [_row("A01", "1", 0, 800.0, 1000.0)]
    assert fractional_outliers(rows) == {("A01", "1"): {0}}


def test_a_zero_prediction_is_skipped_rather_than_dividing() -> None:
    """A fraction of zero is not a measurement."""
    rows = [_row("A01", "1", 0, 5.0, 0.0), _row("A01", "2", 0, 1.0, 1.0)]
    assert fractional_outliers(rows) == {}


def test_thresholds_are_adjustable() -> None:
    """The two knobs are the reviewer's break points, not constants of nature."""
    rows = [_row("A01", "1", 0, 890.0, 1000.0), _row("A01", "2", 0, 100.0, 100.0)]
    assert fractional_outliers(rows) == {}
    assert fractional_outliers(rows, frac_threshold=0.05) == {("A01", "1"): {0}}
