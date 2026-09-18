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

from clophfit.fitting.plate_lm import fractional_outliers, ratiometric_exempt


def _row(well: str, label: str, step: int, y: float, yhat: float) -> dict[str, Any]:
    """Build a residual row in the schema ``fit_plate_lm`` actually emits.

    Note there is no ``y``: the deviation is carried by ``raw_res`` alone. An
    earlier version of this helper invented a ``y`` key, so the tests passed
    against a schema the fitter does not produce and the CLI failed with
    KeyError on the first real plate.
    """
    return {
        "well": well,
        "label": label,
        "step": step,
        "raw_i": step,
        "yhat": yhat,
        "sigma": 1.0,
        "raw_res": y - yhat,
        "std_res": 0.0,
    }


def test_a_large_fractional_miss_is_screened() -> None:
    """L10b A03 step 6: 20% below prediction, label 2 unmoved."""
    rows = [_row("A03", "1", 6, 2563.6, 3222.7), _row("A03", "2", 6, 100.0, 104.8)]
    assert fractional_outliers(rows) == {("A03", "1"): {6}}


def test_a_positive_excursion_is_never_screened() -> None:
    """L4 B02 step 3 is 11% high, and kept.

    Reviewer keeps run to +0.112 while a discard sits at -0.093, so only the
    sign tells them apart. Both documented label-1 artefacts are deficits.
    """
    rows = [_row("B02", "1", 3, 697.53, 627.65), _row("B02", "2", 3, 100.0, 102.4)]
    assert fractional_outliers(rows) == {}


def test_a_shallow_deficit_is_left_alone() -> None:
    """L5b C04 step 6: 5.8% below prediction, inside ordinary variation."""
    rows = [_row("C04", "1", 6, 1212.01, 1286.85), _row("C04", "2", 6, 100.0, 100.0)]
    assert fractional_outliers(rows) == {}


def test_the_shallowest_reviewer_discard_is_caught() -> None:
    """L3 D03 step 5 at -0.093, the shallowest deficit marked for removal."""
    rows = [_row("D03", "1", 5, 626.67, 690.68), _row("D03", "2", 5, 100.0, 100.3)]
    assert fractional_outliers(rows) == {("D03", "1"): {5}}


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
    assert fractional_outliers(rows, frac_threshold=0.2) == {}
    assert fractional_outliers(rows, frac_threshold=0.05) == {("A01", "1"): {0}}


def test_ratiometric_points_are_exempt_from_any_screen() -> None:
    """The exemption is a veto, not a clause of one criterion.

    On L3 C06 the z-screen takes label 2 steps 0 and 1 at |z| 6.8 and 6.7, and
    label 1 step 0 at 3.4. All three are the same well-level multiplicative
    shift -- both channels ~15% high at step 0 and ~15% low at step 1 -- which
    the ratio cancels. The reviewer asked for label 2's to be kept. Sparing them
    only within the fractional criterion leaves the z-screen free to take them
    anyway, which is what the first run did.
    """
    rows = [
        _row("C06", "1", 0, 805.63, 694.14),
        _row("C06", "2", 0, 1169.86, 1018.57),
        _row("C06", "1", 1, 592.83, 707.39),
        _row("C06", "2", 1, 839.51, 985.95),
    ]
    exempt = ratiometric_exempt(rows)
    assert exempt == {("C06", "1"): {0, 1}, ("C06", "2"): {0, 1}}


def test_a_single_channel_deviation_is_not_exempt() -> None:
    """L3 A09 step 0: label 2 does not move, so nothing cancels."""
    rows = [_row("A09", "1", 0, 869.27, 996.46), _row("A09", "2", 0, 1323.5, 1322.53)]
    assert ratiometric_exempt(rows) == {}


def test_small_shared_moves_are_not_exempt() -> None:
    """Ordinary noise moves both channels a little; that is not an artefact."""
    rows = [_row("A01", "1", 0, 101.0, 100.0), _row("A01", "2", 0, 101.0, 100.0)]
    assert ratiometric_exempt(rows) == {}
