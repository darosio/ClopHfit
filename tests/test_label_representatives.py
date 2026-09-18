"""One representative curve per label, across every well on the plate.

The structured noise path used to synthesize its per-label parameters from
whichever well ``next()`` returned first. Per-label detection can leave that
well single-label, and then the model carried no entry for the other label -
which the fit rejects, because its label list is a union over all wells.

Whether it broke depended on set iteration order, so it surfaced as an
intermittent failure: on L6a the run fails under ``PYTHONHASHSEED=1`` and
succeeds under 0, 2, 3, 4 and 5, with everything else held fixed.
"""

from __future__ import annotations

import numpy as np

from clophfit.fitting.bayes import label_representatives

X = np.array([9.0, 8.0, 7.0, 6.0, 5.0])


class DataArrayStub:
    """Minimal stand-in carrying what the noise synthesizer reads."""

    def __init__(self, y: np.ndarray) -> None:
        self.x = X
        self.y = y
        self.y_err = np.full(len(X), 1.0)
        self.mask = np.ones(len(X), dtype=bool)


def test_finds_a_label_the_first_well_does_not_have() -> None:
    """The bug: well order decided whether label 1 existed at all."""
    datasets = [
        {"2": DataArrayStub(np.arange(5.0))},
        {"1": DataArrayStub(np.arange(5.0) * 2), "2": DataArrayStub(np.arange(5.0))},
    ]
    assert sorted(label_representatives(datasets)) == ["1", "2"]


def test_prefers_the_first_well_carrying_each_label() -> None:
    """Deterministic in well order, so two runs agree about the noise model."""
    first = DataArrayStub(np.arange(5.0))
    later = DataArrayStub(np.arange(5.0) * 99)
    reps = label_representatives([{"1": first}, {"1": later}])
    assert reps["1"] is first


def test_ignores_wells_without_a_dataset() -> None:
    """A well that failed to fit contributes nothing and raises nothing."""
    reps = label_representatives([None, {"1": DataArrayStub(np.arange(5.0))}])
    assert sorted(reps) == ["1"]


def test_no_datasets_is_empty_not_an_error() -> None:
    """An empty plate is the caller's problem to report, not this helper's."""
    assert label_representatives([]) == {}


def test_labels_are_strings() -> None:
    """Label keys are compared against string label lists downstream."""
    reps = label_representatives([{1: DataArrayStub(np.arange(5.0))}])  # type: ignore[dict-item]  # an int key, on purpose
    assert list(reps) == ["1"]
