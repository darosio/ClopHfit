"""Carry the plate screen's verdict into whatever fits next.

``fit_plate_lm_screened`` finds outliers on a calibrated ruler and refits K on
the plain one, but it builds its masked copies internally: the caller's
datasets come back untouched, so a Bayesian fit that runs afterwards sees every
point the screen rejected. Applying ``excluded_points`` is what makes the
hybrid - classical screen, Bayesian fit - one pipeline rather than two fits of
different data.
"""

from __future__ import annotations

import numpy as np
import pytest

from clophfit.fitting.data_structures import Dataset
from clophfit.fitting.plate_lm import apply_excluded_points

X = np.array([9.0, 8.0, 7.0, 6.5, 6.0, 5.5, 5.0])


def _datasets() -> dict[str, Dataset]:
    """Two wells, two labels, nothing masked."""
    out = {}
    for well, scale in (("A01", 1.0), ("B02", 2.0)):
        out[well] = Dataset(
            {
                "1": DataArrayStub(X, np.arange(7) * scale),
                "2": DataArrayStub(X, np.arange(7) * scale * 10),
            },
            is_ph=True,
        )
    return out


class DataArrayStub:
    """Minimal stand-in carrying the x, y and mask the fitters read."""

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = x
        self.y = y
        self.mask = np.ones(len(x), dtype=bool)
        self.is_ph = True


def test_masks_exactly_the_named_points() -> None:
    """Two indices on one label of one well, and nothing else moves."""
    out = apply_excluded_points(_datasets(), {"A01": {"1": [2, 5]}})
    assert list(np.asarray(out["A01"]["1"].mask)) == [
        True,
        True,
        False,
        True,
        True,
        False,
        True,
    ]
    assert np.all(np.asarray(out["A01"]["2"].mask))
    assert np.all(np.asarray(out["B02"]["1"].mask))


def test_leaves_the_input_untouched() -> None:
    """The caller keeps its unscreened datasets, as the screen itself does."""
    original = _datasets()
    apply_excluded_points(original, {"A01": {"1": [0, 1, 2]}})
    assert np.all(np.asarray(original["A01"]["1"].mask))


def test_refuses_to_mask_below_min_keep() -> None:
    """A curve through three points is no improvement on one with an outlier.

    Same rule the screen applies when it builds its own masked copies, so the
    two agree about which drops were actually taken.
    """
    out = apply_excluded_points(
        _datasets(), {"A01": {"1": [0, 1, 2, 3, 4]}}, min_keep=4
    )
    assert int(np.asarray(out["A01"]["1"].mask).sum()) == 7


def test_ignores_wells_and_labels_that_are_not_there() -> None:
    """A screen run on a superset of wells must not raise here."""
    out = apply_excluded_points(_datasets(), {"Z99": {"1": [0]}, "A01": {"9": [0]}})
    assert np.all(np.asarray(out["A01"]["1"].mask))


def test_ignores_an_index_outside_the_curve() -> None:
    """Indices are positional; one past the end is dropped, not an error."""
    out = apply_excluded_points(_datasets(), {"A01": {"1": [99, 3]}})
    assert int(np.asarray(out["A01"]["1"].mask).sum()) == 6


def test_no_exclusions_is_a_faithful_copy() -> None:
    """An empty verdict still returns something safe to fit."""
    out = apply_excluded_points(_datasets(), {})
    assert sorted(out) == ["A01", "B02"]
    assert np.all(np.asarray(out["A01"]["1"].mask))


@pytest.mark.parametrize("bad", [{"A01": {"1": [1]}}])
def test_masking_is_cumulative_with_what_was_already_masked(bad: dict) -> None:  # type: ignore[type-arg]
    """A point the outlier mask already removed stays removed."""
    ds = _datasets()
    ds["A01"]["1"].mask[4] = False
    out = apply_excluded_points(ds, bad)
    kept = np.asarray(out["A01"]["1"].mask)
    assert not kept[1]
    assert not kept[4]
