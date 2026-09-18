"""A label that cannot be cleaned is dropped, not kept dirty.

``min_keep`` stops the screen deleting so many points that a curve can no longer
be fitted. But refusing to remove them leaves the label in the fit *with* its
outliers, which is the failure it was meant to prevent, in the other direction:
on L4, H10's 400 nm channel has three points off the sigmoid by 39%, 43% and
73% of its range -- the acidic turnover -- and removing three of seven would
leave four, below the guard. The guard therefore kept all three, and the fitted
curve is pulled by exactly the points the screen identified.

Excluding the whole label instead mirrors what per-label bad-well detection
already does: a well keeps the labels that are usable and is fitted on those.
A well is never left with no labels at all -- there the points stay, because
something is better than nothing and the well's own error bars will say so.
"""

from __future__ import annotations

from collections import UserDict
from typing import Any

import numpy as np

from clophfit.fitting.plate_lm import apply_excluded_points

X = np.array([9.0, 8.0, 7.5, 7.0, 6.5, 6.0, 5.0])


class DataArrayStub:
    """Minimal stand-in carrying the x, y and mask the fitters read."""

    def __init__(self, y: np.ndarray) -> None:
        self.x = X
        self.y = y
        self.mask = np.ones(len(X), dtype=bool)
        self.is_ph = True


class DatasetStub(UserDict):  # type: ignore[type-arg]
    """A dataset is a label-keyed mapping that remembers ``is_ph``."""

    def __init__(self, arrays: dict[str, Any], *, is_ph: bool = True) -> None:
        super().__init__(arrays)
        self.is_ph = is_ph


def _two_label_well() -> dict[str, Any]:
    """One well with both channels, seven points each."""
    return {
        "H10": DatasetStub({
            "1": DataArrayStub(np.arange(7.0) * 10),
            "2": DataArrayStub(np.arange(7.0) * 20),
        })
    }


def test_a_label_that_cannot_be_cleaned_is_dropped() -> None:
    """Three of seven would leave four, below min_keep: drop the label."""
    out = apply_excluded_points(
        _two_label_well(), {"H10": {"1": [0, 1, 6]}}, min_keep=5
    )
    assert "1" not in out["H10"]
    assert "2" in out["H10"]


def test_the_other_label_is_untouched() -> None:
    """Dropping one channel must not disturb the one that was fine."""
    out = apply_excluded_points(
        _two_label_well(), {"H10": {"1": [0, 1, 6]}}, min_keep=5
    )
    assert np.all(np.asarray(out["H10"]["2"].mask))


def test_a_label_that_can_be_cleaned_is_only_masked() -> None:
    """Two of seven leaves five, which the guard allows: mask, do not drop."""
    out = apply_excluded_points(_two_label_well(), {"H10": {"1": [0, 6]}}, min_keep=5)
    assert "1" in out["H10"]
    assert int(np.asarray(out["H10"]["1"].mask).sum()) == 5


def test_the_last_label_is_never_dropped() -> None:
    """A well with nothing left cannot be fitted at all.

    Keeping the points is the lesser evil: the fit is poor and its own error
    bars say so, where an empty well says nothing and may break the caller.
    """
    single = {"H10": DatasetStub({"1": DataArrayStub(np.arange(7.0) * 10)})}
    out = apply_excluded_points(single, {"H10": {"1": [0, 1, 6]}}, min_keep=5)
    assert "1" in out["H10"]
    assert int(np.asarray(out["H10"]["1"].mask).sum()) == 7


def test_both_labels_uncleanable_keeps_one() -> None:
    """Dropping every label of a well is the same as deleting the well."""
    out = apply_excluded_points(
        _two_label_well(), {"H10": {"1": [0, 1, 6], "2": [0, 1, 6]}}, min_keep=5
    )
    assert len(out["H10"]) >= 1
