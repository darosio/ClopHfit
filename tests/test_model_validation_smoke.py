"""Tiny smoke tests for reusable validation helpers.

These tests are intentionally minimal.  In the real ClopHfit repo, extend them
with the committed example plate and 20-draw PyMC smoke fits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from clophfit.fitting.ctr_validation import iter_ctr_holdouts, make_ctr_holdout_scheme
from clophfit.fitting.model_validation import model_residual_score_table


def test_residual_score_table_smoke() -> None:
    """Smoke test: model_residual_score_table returns expected columns."""
    df = pd.DataFrame({
        "trace_id": ["m1", "m1", "m1", "m1"],
        "well": ["A01", "A01", "A02", "A02"],
        "label": ["1", "1", "1", "1"],
        "step": [0, 1, 0, 1],
        "x": [8.9, 8.2, 8.9, 8.2],
        "std_res": [0.1, -0.2, 0.0, 0.3],
    })
    model, per_label, by_step, _lag, _cross = model_residual_score_table(df)
    assert model.loc[0, "trace_id"] == "m1"
    assert np.isfinite(model.loc[0, "residual_x_median_rms"])
    assert not per_label.empty
    assert not by_step.empty


def test_ctr_holdout_scheme_uses_sets() -> None:
    """CTR holdout must preserve dict[str, set[str]] types on PlateScheme.names."""

    class Scheme:
        def __init__(self) -> None:
            self._names: dict[str, set[str]] = {"ctrl": {"A01", "A02", "A03"}}

        @property
        def names(self) -> dict[str, set[str]]:
            return self._names

        @names.setter
        def names(self, value: dict[str, set[str]]) -> None:
            assert isinstance(value, dict)
            assert all(isinstance(k, str) for k in value)
            assert all(isinstance(v, set) for v in value.values())
            self._names = value

    s = Scheme()
    s2 = make_ctr_holdout_scheme(s, group_name="ctrl", heldout_well="A01")
    assert s2.names == {"ctrl": {"A02", "A03"}}
    tasks = list(iter_ctr_holdouts(s, min_remaining=1))
    assert len(tasks) == 3
