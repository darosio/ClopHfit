"""Tests for the plate-wide errors-in-variables fit."""

from __future__ import annotations

import numpy as np
import pytest

from clophfit.fitting.data_structures import DataArray, Dataset
from clophfit.fitting.models import binding_1site
from clophfit.fitting.plate_lm import ctr_holdout, fit_plate_lm
from clophfit.fitting.plate_odr import ctr_holdout_odr, fit_plate_odr


def plate(
    ks: dict[str, float], x: np.ndarray, sd: float = 3.0, seed: int = 2
) -> dict[str, Dataset]:
    """Synthetic plate on a given x grid, one label."""
    rng = np.random.default_rng(seed)
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


def test_negligible_x_error_reproduces_the_least_squares_fit() -> None:
    """With x pinned, the errors-in-variables fit collapses onto plate_lm."""
    x = np.linspace(5.0, 9.0, 7)
    ks = {f"A{i:02d}": 6.8 + 0.05 * i for i in range(1, 9)}
    ds = plate(ks, x)
    lm = fit_plate_lm(ds, groups={})
    odr = fit_plate_odr(ds, groups={}, x_err=np.full(len(x), 1e-6))
    for w in ks:
        assert odr.k[w] == pytest.approx(lm.k[w], abs=0.01)


def test_recovers_a_wrongly_recorded_step() -> None:
    """A step whose recorded pH is wrong is corrected, and K with it.

    One step is recorded 0.25 pH away from where the titration actually sat.
    Least squares must take x at face value and absorbs the error into K; the
    errors-in-variables fit is allowed to move that step instead.
    """
    true_x = np.linspace(5.0, 9.0, 7)
    recorded = true_x.copy()
    recorded[3] += 0.25
    ks = {f"A{i:02d}": 7.0 for i in range(1, 13)}
    ds = plate(ks, true_x, sd=2.0)  # y generated on the true grid
    for d in ds.values():  # but the fitter is told the wrong grid
        d["1"].xc = recorded
    odr = fit_plate_odr(ds, groups={}, x_err=np.full(len(recorded), 0.15))
    lm = fit_plate_lm(ds, groups={})

    assert odr.dx[3] < -0.1  # the step is pulled back toward the true pH
    # It cannot be pulled all the way: 0.25 costs 1.7 sigma, so the fit splits
    # the difference with K. What matters is that it lands nearer the truth than
    # the fit that had to take x at face value.
    bias_odr = abs(np.median([odr.k[w] for w in ks]) - 7.0)
    bias_lm = abs(np.median([lm.k[w] for w in ks]) - 7.0)
    assert bias_odr < bias_lm


def test_one_shift_per_step_shared_across_wells() -> None:
    """The pH of a step is a property of the plate, not of each well."""
    x = np.linspace(5.0, 9.0, 7)
    odr = fit_plate_odr(
        plate({f"A{i:02d}": 7.0 for i in range(1, 5)}, x),
        groups={},
        x_err=np.full(len(x), 0.05),
    )
    assert len(odr.dx) == len(x)


def test_odr_holdout_frees_only_the_held_out_control() -> None:
    """One row per control, group-mates still sharing a single K."""
    x = np.linspace(5.0, 9.0, 7)
    ks = {"A01": 7.0, "A02": 7.0, "A03": 7.0, "B01": 6.0}
    rows = ctr_holdout_odr(
        plate(ks, x), {"ctrl": ["A01", "A02", "A03"]}, x_err=np.full(len(x), 0.05)
    )
    assert {r["heldout_well"] for r in rows} == {"A01", "A02", "A03"}
    assert all(r["n_remaining_ctr"] == 2 for r in rows)
    assert all(r["cell"] == "plate_odr" for r in rows)


def test_a_plate_wide_x_error_cancels_in_the_holdout() -> None:
    """A wrongly recorded step biases K but not ΔK, for either fitter.

    Every well on a plate shares one pH grid, so a step recorded in the wrong
    place shifts every member of a control group by the same amount, and the
    shift cancels in the held-out-minus-group difference. The holdout metric
    measures agreement *within* a group, not absolute accuracy, so it cannot see
    this class of error — and correcting x therefore buys nothing here, however
    much it helps the K estimates themselves.
    """
    true_x = np.linspace(5.0, 9.0, 7)
    recorded = true_x.copy()
    recorded[3] += 0.30
    ks = {"A01": 7.0, "A02": 7.0, "A03": 7.0, "B01": 6.4, "B02": 6.4}
    ds = plate(ks, true_x, sd=2.0)
    for d in ds.values():
        d["1"].xc = recorded
    groups = {"ctrl": ["A01", "A02", "A03"]}

    # The absolute estimate is visibly wrong ...
    assert abs(np.median([fit_plate_lm(ds, groups={}).k[w] for w in ks]) - 7.0) > 0.1

    # ... yet both holdouts report agreement, because the error is common-mode.
    for rows in (
        ctr_holdout(ds, groups),
        ctr_holdout_odr(ds, groups, x_err=np.full(len(recorded), 0.15)),
    ):
        assert np.median([abs(r["delta_k_mean"]) for r in rows]) < 0.05
