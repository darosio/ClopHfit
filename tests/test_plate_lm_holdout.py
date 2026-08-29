"""Tests for leave-one-control-out on the plate-wide classical fit."""

from __future__ import annotations

import numpy as np

from clophfit.fitting.data_structures import DataArray, Dataset
from clophfit.fitting.models import binding_1site
from clophfit.fitting.plate_lm import ctr_holdout


def plate(ks: dict[str, float], sd: float = 3.0, seed: int = 1) -> dict[str, Dataset]:
    """Synthetic plate, one label, known K per well."""
    rng = np.random.default_rng(seed)
    x = np.linspace(5.0, 9.0, 7)
    return {
        w: Dataset(
            {
                "1": DataArray(
                    x,
                    binding_1site(x, k, 1000.0, 200.0, is_ph=True)
                    + rng.normal(0.0, sd, len(x)),
                    y_errc=np.ones_like(x),
                )
            },
            is_ph=True,
        )
        for w, k in ks.items()
    }


def test_holdout_frees_only_the_held_out_well() -> None:
    """The held-out control gets its own K; its group-mates keep sharing one."""
    ks = {"A01": 7.0, "A02": 7.0, "A03": 7.0, "B01": 6.0}
    rows = ctr_holdout(plate(ks), {"ctrl": ["A01", "A02", "A03"]})
    assert {r["heldout_well"] for r in rows} == {"A01", "A02", "A03"}
    for r in rows:
        assert r["ctr_group"] == "ctrl"
        assert r["n_remaining_ctr"] == 2


def test_delta_k_is_near_zero_for_agreeing_replicates() -> None:
    """Replicates that share a true K produce a ΔK consistent with zero."""
    ks = {"A01": 7.0, "A02": 7.0, "A03": 7.0, "B01": 6.0}
    rows = ctr_holdout(plate(ks), {"ctrl": ["A01", "A02", "A03"]})
    for r in rows:
        assert abs(r["delta_k_mean"]) < 0.1
        assert r["delta_k_sd"] > 0.0


def test_a_discordant_control_is_flagged() -> None:
    """A control whose true K differs shows up as a large ΔK, and only it does."""
    ks = {"A01": 7.0, "A02": 7.0, "A03": 7.6, "B01": 6.0}
    rows = {
        r["heldout_well"]: r for r in ctr_holdout(plate(ks), {"ctrl": list(ks)[:3]})
    }
    assert abs(rows["A03"]["delta_k_mean"]) > 0.3
    assert abs(rows["A01"]["delta_k_mean"]) < abs(rows["A03"]["delta_k_mean"])
