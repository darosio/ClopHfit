"""Unit tests for the calibration-metric helpers in compare_plate_pipelines."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

_SPEC = importlib.util.spec_from_file_location(
    "compare_plate_pipelines",
    Path(__file__).resolve().parents[1] / "scripts" / "compare_plate_pipelines.py",
)
assert _SPEC is not None
assert _SPEC.loader is not None
cpp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cpp)


def test_bulk_sd_recovers_unit_scale() -> None:
    """A standard-Normal std_res column gives bulk_sd close to 1."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "label": np.repeat(["1", "2"], 2000),
        "std_res": rng.normal(0, 1, 4000),
    })
    out = cpp.bulk_sd_scores(df)
    assert abs(out["bulk_sd_pooled"] - 1.0) < 0.1
    assert abs(out["bulk_sd_lbl1"] - 1.0) < 0.1


def test_bulk_sd_ignores_tails() -> None:
    """Heavy tails do not inflate bulk_sd; it tracks the central spread."""
    rng = np.random.default_rng(1)
    core = rng.normal(0, 1, 1000)
    contaminated = np.concatenate([core, rng.normal(0, 20, 30)])
    clean = pd.DataFrame({"label": ["1"] * len(core), "std_res": core})
    dirty = pd.DataFrame({"label": ["1"] * len(contaminated), "std_res": contaminated})
    assert (
        abs(
            cpp.bulk_sd_scores(dirty)["bulk_sd_lbl1"]
            - cpp.bulk_sd_scores(clean)["bulk_sd_lbl1"]
        )
        < 0.15
    )
