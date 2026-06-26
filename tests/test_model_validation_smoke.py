"""Tiny smoke tests for reusable validation helpers.

These tests are intentionally minimal.  In the real ClopHfit repo, extend them
with the committed example plate and 20-draw PyMC smoke fits.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
import xarray as xr
from lmfit import Parameters  # type: ignore[import-untyped]

from clophfit.fitting.ctr_validation import (
    iter_ctr_holdouts,
    make_ctr_holdout_scheme,
    summarize_bayesian_ctr_holdout,
    weighted_mean_reference,
)
from clophfit.fitting.data_structures import (
    DataArray,
    Dataset,
    FitResult,
    MultiFitResult,
)
from clophfit.fitting.model_validation import (
    excess_tail_outlier_mask,
    mark_excess_residual_outliers,
    masked_datasets_from_residual_outliers,
    merge_log_likelihoods,
    model_residual_score_table,
    residual_normal_scores,
    residuals_from_multifit,
    robust_residual_outlier_mask,
)
from clophfit.fitting.models import binding_1site


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
    assert np.isfinite(cast("float", model.loc[0, "residual_x_median_rms"]))
    assert not per_label.empty
    assert not by_step.empty


def test_student_t_residuals_use_normal_scores() -> None:
    """Robust residual diagnostics should use t-CDF calibrated z-scores."""
    z = residual_normal_scores(np.array([3.0]), robust=True, student_t_nu=3.0)

    assert 1.8 < z[0] < 1.9
    assert not robust_residual_outlier_mask(
        np.array([3.0]), robust=True, student_t_nu=3.0, threshold=3.0
    )[0]
    assert robust_residual_outlier_mask(
        np.array([12.0]), robust=True, student_t_nu=3.0, threshold=3.0
    )[0]


def test_excess_tail_outlier_mask_leaves_allowed_tail() -> None:
    """Only excess residuals above the allowed tail fraction should be removed."""
    residuals = np.array([0.0] * 95 + [3.1, 3.2, 3.3, 4.0, 5.0])

    remove = excess_tail_outlier_mask(
        residuals,
        threshold=3.0,
        allowed_tail_fraction=0.01,
    )

    assert remove.sum() == 4
    assert not remove[95]
    assert remove[-1]


def test_excess_tail_outlier_mask_uses_floor_for_allowed_tail() -> None:
    """A one-percent tail in 1056 rows allows 10, not 11, residuals."""
    residuals = np.array([0.0] * 1045 + [3.1] * 10 + [8.0])

    remove = excess_tail_outlier_mask(
        residuals,
        threshold=3.0,
        allowed_tail_fraction=0.01,
    )

    assert remove.sum() == 1
    assert remove[-1]


def test_excess_tail_outlier_mask_adjusts_student_t_residuals() -> None:
    """Student-t likelihood residuals are judged on calibrated Normal scores."""
    residuals = np.array([0.0] * 95 + [3.1, 4.0, 12.0, 20.0, 30.0])

    remove = excess_tail_outlier_mask(
        residuals,
        threshold=3.0,
        allowed_tail_fraction=0.01,
        robust=True,
        student_t_nu=3.0,
    )

    assert remove.sum() == 2
    assert remove[-1]
    assert remove[-2]


def test_mark_excess_residual_outliers_groups_by_trace_label() -> None:
    """Residual table annotation should leave one percent tail per group."""
    df = pd.DataFrame({
        "trace_id": ["m1"] * 200,
        "label": ["1"] * 100 + ["2"] * 100,
        "std_res": [0.0] * 95
        + [3.1, 3.2, 3.3, 4.0, 5.0]
        + [0.0] * 94
        + [3.1, 3.2, 3.3, 4.0, 5.0, 6.0],
    })

    marked = mark_excess_residual_outliers(df, allowed_tail_fraction=0.01)

    assert marked["exclude_residual_outlier"].sum() == 9
    assert not bool(marked.loc[95, "exclude_residual_outlier"])
    assert bool(marked.loc[99, "exclude_residual_outlier"])
    assert bool(marked.loc[199, "exclude_residual_outlier"])


def test_masked_datasets_from_residual_outliers_masks_marked_rows() -> None:
    """Marked residual rows should become masked inputs for a second fit."""
    ds = Dataset(
        {
            "1": DataArray(
                np.array([6.0, 7.0, 8.0, 9.0]),
                np.array([10.0, 11.0, 12.0, 13.0]),
            )
        },
        is_ph=True,
    )
    fr: FitResult[Any] = FitResult(result=object(), dataset=ds)
    residuals = pd.DataFrame({
        "well": ["A01"],
        "label": ["1"],
        "raw_i": [2],
        "exclude_residual_outlier": [True],
    })

    masked = masked_datasets_from_residual_outliers({"A01": fr}, residuals)

    assert np.array_equal(masked["A01"]["1"].mask, np.array([True, True, False, True]))
    assert np.all(ds["1"].mask)


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


def test_merge_log_likelihoods_builds_single_obs_variable() -> None:
    """Multiple pointwise log-likelihood arrays should become one obs variable."""
    posterior = xr.Dataset(
        data_vars={"K": (("chain", "draw"), np.array([[1.0, 1.1]]))},
        coords={"chain": [0], "draw": [0, 1]},
    )
    log_likelihood = xr.Dataset(
        data_vars={
            "obs_a": (("chain", "draw", "obs_a_dim"), np.zeros((1, 2, 3))),
            "obs_b": (("chain", "draw", "obs_b_dim"), np.ones((1, 2, 2))),
        },
        coords={"chain": [0], "draw": [0, 1]},
    )
    trace = xr.DataTree.from_dict({
        "posterior": posterior,
        "log_likelihood": log_likelihood,
    })

    merged = merge_log_likelihoods(trace)

    assert list(merged.log_likelihood.data_vars) == ["obs"]
    assert merged.log_likelihood["obs"].sizes["obs_id"] == 5


def test_weighted_mean_reference_prefers_precise_controls() -> None:
    """Weighted CTR references should favor precise remaining controls."""
    reference, weights = weighted_mean_reference([
        np.array([1.0, 1.0, 1.0, 1.0]),
        np.array([0.0, 2.0, 0.0, 2.0]),
    ])

    assert weights[0] > weights[1]
    assert reference.shape == (4,)


def test_summarize_bayesian_ctr_holdout_weighted_mean_reference() -> None:
    """Free-CTR Bayesian holdouts compare to remaining free-control Ks."""
    posterior = xr.Dataset(
        data_vars={
            "K_B12": (("chain", "draw"), np.array([[1.0, 1.1, 0.9, 1.0]])),
            "K_E2GFP_C01": (("chain", "draw"), np.array([[1.0, 1.0, 1.0, 1.0]])),
            "K_E2GFP_F12": (("chain", "draw"), np.array([[0.9, 1.1, 1.0, 1.0]])),
        },
        coords={"chain": [0], "draw": [0, 1, 2, 3]},
    )
    trace = xr.DataTree.from_dict({"posterior": posterior})

    row = summarize_bayesian_ctr_holdout(
        trace,
        trace_id="free-model",
        ctr_group="E2GFP",
        heldout_well="B12",
        remaining_ctr_wells=["C01", "F12"],
        reference_mode="weighted_mean",
    )

    assert row["ctr_reference_mode"] == "weighted_mean"
    assert row["ctr_reference_n"] == 2
    assert row["delta_k_abs_mean"] < 0.1


def test_residuals_from_multifit_does_not_double_apply_ye_mag() -> None:
    """Reconstructed PyMC datasets already store the sigma used by likelihood."""
    posterior = xr.Dataset(
        data_vars={"ye_mag_1": (("chain", "draw"), np.array([[5.0, 5.0]]))},
        coords={"chain": [0], "draw": [0, 1]},
    )
    trace = xr.DataTree.from_dict({"posterior": posterior})

    params = Parameters()
    params.add("K", value=7.0)
    params.add("S0_1", value=100.0)
    params.add("S1_1", value=200.0)
    ds = Dataset(
        {
            "1": DataArray(
                np.array([6.0, 7.0]),
                np.array([90.0, 150.0]),
                y_errc=np.array([10.0, 20.0]),
            )
        },
        is_ph=True,
    )
    fr = FitResult(
        result=type("Result", (), {"params": params})(), mini=trace, dataset=ds
    )
    multi = MultiFitResult(trace=trace, results={"A01": fr})

    residuals = residuals_from_multifit(multi, "trace", binding_1site)

    np.testing.assert_allclose(residuals["sigma"].to_numpy(), np.array([10.0, 20.0]))
    assert "likelihood_res" in residuals.columns
    assert "is_residual_outlier" in residuals.columns
