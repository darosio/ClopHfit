"""Post-fit K-precision cut, in pH units.

A rule written as ``sK / K`` is inert on a pH titration: K is a pKa of 5-8, so
``sK / K = sK / 7`` and a ratio of 0.3 asks for an uncertainty of 2.1 pH. Across
1002 wells of the eleven-plate campaign the largest ratio observed is 0.255, so
no such threshold ever fires. pH is an interval scale with an arbitrary origin;
the uncertainty has to be judged in its own units.

Control wells get the tighter limit because they anchor accuracy against known
pK values. The numbers below are the measured ones from that campaign.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from clophfit.fitting.diagnostics import flag_imprecise_k

# Measured sK, in pH, from the free-K refit.
MEASURED = pd.DataFrame({
    "well": ["G09", "B08", "D09", "G12", "F04", "B02", "H11", "A12"],
    "sK": [1.645, 0.776, 0.624, 0.584, 0.445, 0.202, 0.193, 0.030],
})
CONTROLS = ("G12", "A12")


def test_discards_the_wells_whose_k_is_not_pinned() -> None:
    """The three widest non-controls and the one wide control, and nothing else."""
    assert flag_imprecise_k(MEASURED, CONTROLS) == ["B08", "D09", "G09", "G12"]


def test_controls_are_held_to_the_tighter_limit() -> None:
    """One sK, two verdicts: 0.445 passes as a well and fails as a control."""
    one = pd.DataFrame({"well": ["X"], "sK": [0.445]})
    assert flag_imprecise_k(one, control_wells=()) == []
    assert flag_imprecise_k(one, control_wells=("X",)) == ["X"]


def test_a_large_k_does_not_excuse_a_large_uncertainty() -> None:
    """The cut is on sK, not on sK / K, which no pKa ever trips.

    Both wells carry the same 0.8 pH uncertainty. A ratio rule would keep the
    one whose midpoint sits higher, purely because pH has an arbitrary origin.
    """
    df = pd.DataFrame({"well": ["acid", "base"], "sK": [0.8, 0.8], "K": [5.2, 8.6]})
    assert flag_imprecise_k(df, control_wells=()) == ["acid", "base"]


def test_a_non_finite_uncertainty_is_never_kept() -> None:
    """A fit that could not report an error has not constrained anything."""
    df = pd.DataFrame({"well": ["nan", "inf", "ok"], "sK": [np.nan, np.inf, 0.03]})
    assert flag_imprecise_k(df, control_wells=()) == ["inf", "nan"]


def test_limits_are_inclusive_of_the_threshold() -> None:
    """A well exactly at the limit is kept; the rule discards what exceeds it."""
    df = pd.DataFrame({"well": ["at", "over"], "sK": [0.60, 0.601]})
    assert flag_imprecise_k(df, control_wells=()) == ["over"]


def test_rejects_a_table_without_the_uncertainty() -> None:
    """Nothing can be judged from a fit table that carries no sK."""
    with pytest.raises(ValueError, match="sK"):
        flag_imprecise_k(pd.DataFrame({"well": ["A01"]}), control_wells=())
