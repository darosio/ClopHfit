"""Tiny smoke tests for reusable validation helpers.

These tests are intentionally minimal.  In the real ClopHfit repo, extend them
with the committed example plate and 20-draw PyMC smoke fits.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from lmfit import Parameters  # type: ignore[import-untyped]

from clophfit.fitting import residuals as residuals_module
from clophfit.fitting.ctr_validation import (
    classical_ctr_holdout_rows,
    iter_ctr_holdouts,
    make_ctr_holdout_scheme,
    summarize_bayesian_ctr_holdout,
    summarize_ctr_loo_table,
    weighted_mean_reference,
    widen_heldout_k_prior,
)
from clophfit.fitting.data_structures import (
    DataArray,
    Dataset,
    FitResult,
    MultiFitResult,
)
from clophfit.fitting.model_validation import (
    RESIDUAL_TABLE_COLUMNS,
    OutlierProbability,
    ResidualComparison,
    ResidualDiagnostics,
    ResidualTail,
    apply_exclusions,
    excess_tail_outlier_mask,
    mark_outliers,
    merge_log_likelihoods,
    model_residual_score_table,
    pareto_k_summary,
    pareto_k_table,
    residual_normal_scores,
    residuals_from_fit_results,
    residuals_from_multifit,
    robust_residual_outlier_mask,
    trace_diagnostics,
    trace_parameter_summary,
    x_axis_sanity,
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


def test_residual_diagnostics_wrapper_summarizes_and_annotates() -> None:
    """ResidualDiagnostics should package repeated residual audit steps."""
    df = pd.DataFrame({
        "trace_id": ["m1"] * 4,
        "well": ["A01", "A02", "B01", "B02"],
        "label": ["1", "1", "2", "2"],
        "step": [0, 1, 0, 1],
        "x": [8.9, 8.2, 8.9, 8.2],
        "y": [1.0, 1.1, 2.0, 2.1],
        "yhat": [1.0, 1.2, 1.9, 2.0],
        "sigma": [1.0, 1.0, 1.0, 1.0],
        "std_res": [0.1, -0.2, 0.3, -0.4],
    })
    fit_df = pd.DataFrame({
        "well": ["A01", "A02", "B01", "B02"],
        "K": [6.5, 6.6, 6.7, 6.8],
    })

    diag = ResidualDiagnostics(df).annotate(fit_df=fit_df, ctrl_wells={"A01"})
    centered = diag.step_centered()

    assert {"row", "col", "role", "K"}.issubset(diag.residuals.columns)
    assert diag.residuals.loc[0, "role"] == "ctr"
    assert "std_res_step_centered" in centered.residuals.columns
    assert not diag.normality().empty
    assert not diag.step_summary().empty
    assert set(diag.position_summary()) >= {"row", "col", "edge_col", "role"}
    assert len(diag.tail_rows(2)) == 2


def test_residual_diagnostics_relative_well_scaled_and_trace_summary() -> None:
    """Diagnostics should package the repeated notebook residual comparisons."""
    df = pd.DataFrame({
        "trace_id": ["m1"] * 8,
        "well": ["A01"] * 4 + ["A02"] * 4,
        "label": ["1", "1", "2", "2"] * 2,
        "step": [0, 1, 0, 1] * 2,
        "x": [8.9, 8.2, 8.9, 8.2] * 2,
        "y": [10.0, 12.0, 20.0, 18.0, 11.0, 14.0, 22.0, 17.0],
        "yhat": [9.0, 13.0, 21.0, 19.0, 10.0, 15.0, 20.0, 18.0],
        "sigma": [1.0] * 8,
        "std_res": [1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 2.0, -1.0],
    })
    posterior = xr.Dataset(
        data_vars={
            "ye_mag_1": (("chain", "draw"), np.array([[2.0, 4.0]])),
            "rel_error_2": (("chain", "draw"), np.array([[0.01, 0.03]])),
            "floor": (("chain", "draw"), np.array([[5.0, 7.0]])),
            "label_noise_scale_1": (("chain", "draw"), np.array([[0.9, 1.1]])),
            "well_noise_sd_1": (("chain", "draw"), np.array([[0.2, 0.4]])),
            "well_noise_scale_1": (
                ("chain", "draw", "well"),
                np.array([[[10.0, 20.0], [12.0, 22.0]]]),
            ),
        },
        coords={"chain": [0], "draw": [0, 1], "well": ["A01", "A02"]},
    )
    trace = xr.DataTree.from_dict({"posterior": posterior})
    dataset = Dataset(
        {
            "1": DataArray(np.array([8.9, 8.2]), np.array([10.0, 12.0])),
            "2": DataArray(np.array([8.9, 8.2]), np.array([20.0, 18.0])),
        },
        is_ph=True,
    )
    results = {
        "A01": FitResult(result=object(), trace=trace, dataset=dataset),
        "A02": FitResult(result=object(), trace=trace, dataset=dataset),
    }

    diag = (
        ResidualDiagnostics(df)
        .annotate(ctrl_wells={"A01"})
        .with_relative_residuals()
        .well_scaled()
    )
    well = diag.well_summary()
    params = trace_parameter_summary(results)
    comparison = ResidualComparison(
        diagnostics=diag,
        well=well.merge(params, on=["well", "label"], how="left"),
        parameters=params,
    )

    assert "std_res_well_scaled" in comparison.diagnostics.residuals.columns
    assert "rel_sd" in comparison.well.columns
    assert comparison.parameters.loc[0, "floor_mean"] == 6.0
    assert comparison.parameters["ye_mag_mean"].dropna().iloc[0] == 3.0
    a01_l1 = comparison.parameters[
        (comparison.parameters["well"] == "A01")
        & (comparison.parameters["label"] == "1")
    ].iloc[0]
    assert a01_l1["label_noise_scale_mean"] == 1.0
    assert a01_l1["well_noise_sd_mean"] == pytest.approx(0.3)
    assert a01_l1["well_noise_scale_mean"] == 11.0


def test_residual_comparison_from_fit_results_and_value_switch() -> None:
    """Comparison factory should build diagnostics, summaries, and fit parameters."""
    params = Parameters()
    params.add("K", value=7.0)
    params.add("S0_1", value=100.0)
    params.add("S1_1", value=200.0)
    x = np.array([6.0, 7.0, 8.0])
    y = binding_1site(x, 7.0, 100.0, 200.0, is_ph=True) + np.array([1.0, 0.0, -1.0])
    ds = Dataset({"1": DataArray(x, y, y_errc=np.ones_like(y))}, is_ph=True)
    fr: FitResult = FitResult(
        result=type("Result", (), {"params": params})(), dataset=ds
    )

    comparison = ResidualComparison.from_fit_results(
        {"A01": fr},
        "classical",
        binding_1site,
        fit_df=pd.DataFrame({"well": ["A01"], "K": [7.0], "sK": [0.1]}),
        ctrl_wells={"A01"},
    )
    rel = comparison.with_value("rel_res")

    assert comparison.parameters.empty
    assert comparison.well.loc[0, "role"] == "ctr"
    assert "K" in comparison.diagnostics.residuals.columns
    assert rel.diagnostics.value_col == "rel_res"
    assert "rel_sd" in rel.well.columns


def test_residual_diagnostics_scaling_and_plot_guards() -> None:
    """Scaling helpers should handle zero-variance groups and plot preconditions."""
    df = pd.DataFrame({
        "trace_id": ["m1"] * 8,
        "well": ["A01"] * 4 + ["A02"] * 4,
        "label": ["1"] * 8,
        "step": [0, 1, 2, 3] * 2,
        "x": [6.0, 7.0, 8.0, 9.0] * 2,
        "y": [1.0] * 8,
        "yhat": [1.0] * 8,
        "sigma": [1.0] * 8,
        "std_res": [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 2.0],
    })

    diag = ResidualDiagnostics(df)
    label_scaled = diag.label_scaled()
    well_scaled = diag.well_scaled(min_count=5)

    assert "std_res_label_scaled" in label_scaled.residuals.columns
    assert well_scaled.residuals["std_res_well_scaled"].isna().all()
    with pytest.raises(ValueError, match="annotate"):
        diag.plot_role()
    with pytest.raises(ValueError, match="annotate"):
        diag.plot_col()


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

    marked = mark_outliers(df, ResidualTail(allowed_tail_fraction=0.01))

    assert marked["exclude_outlier"].sum() == 9
    assert not bool(marked.loc[95, "exclude_outlier"])
    assert bool(marked.loc[99, "exclude_outlier"])
    assert bool(marked.loc[199, "exclude_outlier"])


def test_apply_exclusions_masks_marked_rows() -> None:
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
    fr: FitResult = FitResult(result=object(), dataset=ds)
    residuals = pd.DataFrame({
        "well": ["A01"],
        "label": ["1"],
        "raw_i": [2],
        "exclude_outlier": [True],
    })

    masked = apply_exclusions({"A01": fr}, residuals)

    assert np.array_equal(masked["A01"]["1"].mask, np.array([True, True, False, True]))
    assert np.all(ds["1"].mask)


def test_apply_exclusions_accepts_ds_dict() -> None:
    """Posterior outlier probabilities should mask matching raw ds_dict rows."""
    ds = Dataset(
        {
            "1": DataArray(
                np.array([6.0, 7.0, 8.0, 9.0]),
                np.array([10.0, 11.0, 12.0, 13.0]),
            )
        },
        is_ph=True,
    )
    residuals = pd.DataFrame({
        "well": ["A01", "A01"],
        "label": ["1", "1"],
        "step": [1, 2],
        "p_outlier": [0.9, 0.95],
    })

    masked = apply_exclusions(
        {"A01": ds}, mark_outliers(residuals, OutlierProbability())
    )
    marked = mark_outliers(residuals, OutlierProbability())

    assert marked["exclude_outlier"].to_list() == [False, True]
    assert np.array_equal(masked["A01"]["1"].mask, np.array([True, True, False, True]))
    assert np.all(ds["1"].mask)


def test_outlier_probability_helpers_reject_residuals_module() -> None:
    """A shadowed residuals import should fail with a clear DataFrame hint."""
    with pytest.raises(TypeError, match="residuals must be a pandas DataFrame"):
        mark_outliers(cast("pd.DataFrame", residuals_module), OutlierProbability())

    with pytest.raises(TypeError, match="residuals must be a pandas DataFrame"):
        apply_exclusions(
            {},
            cast("pd.DataFrame", residuals_module),
        )


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


def test_pareto_k_table_maps_multiwell_likelihood_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pointwise Pareto-k rows should preserve PyMC's step-major observation order."""
    posterior = xr.Dataset(
        data_vars={"K": (("chain", "draw"), np.array([[1.0, 1.1]]))},
        coords={"chain": [0], "draw": [0, 1]},
    )
    log_likelihood = xr.Dataset(
        data_vars={
            "y_likelihood_1": (("chain", "draw", "obs_1"), np.zeros((1, 2, 5))),
            "y_likelihood_2": (("chain", "draw", "obs_2"), np.zeros((1, 2, 4))),
        },
        coords={"chain": [0], "draw": [0, 1]},
    )
    trace = xr.DataTree.from_dict({
        "posterior": posterior,
        "log_likelihood": log_likelihood,
    })

    def ds(y1: np.ndarray, y2: np.ndarray, mask1: np.ndarray) -> Dataset:
        da1 = DataArray(
            np.array([8.0, 7.0, 6.0]),
            y1,
        )
        da1.mask = mask1
        return Dataset(
            {
                "1": da1,
                "2": DataArray(
                    np.array([8.0, 7.0]),
                    y2,
                ),
            },
            is_ph=True,
        )

    results: dict[str, FitResult] = {
        "A01": FitResult(
            dataset=ds(
                np.array([10.0, 11.0, 12.0]),
                np.array([1.0, 2.0]),
                np.array([True, True, False]),
            )
        ),
        "A02": FitResult(
            dataset=ds(
                np.array([20.0, 21.0, 22.0]),
                np.array([3.0, 4.0]),
                np.array([True, True, True]),
            )
        ),
    }
    multi = MultiFitResult(trace=trace, results=results)

    class FakeLoo:
        def __init__(self, values: list[float]) -> None:
            self.pareto_k = xr.DataArray(np.asarray(values), dims=("obs",))

    def fake_loo(_trace: object, *, var_name: str, pointwise: bool) -> FakeLoo:
        assert pointwise is True
        values = {
            "y_likelihood_1": [0.1, 0.2, 0.3, 0.4, 0.8],
            "y_likelihood_2": [0.5, 0.6, 0.7, 0.9],
        }[var_name]
        return FakeLoo(values)

    monkeypatch.setattr("clophfit.fitting.model_validation.az.loo", fake_loo)

    table = pareto_k_table(multi)

    assert table["pareto_k"].tolist() == [0.1, 0.2, 0.3, 0.4, 0.8, 0.5, 0.6, 0.7, 0.9]
    assert table.loc[:4, ["well", "label", "step"]].to_dict("records") == [
        {"well": "A01", "label": "1", "step": 0},
        {"well": "A02", "label": "1", "step": 0},
        {"well": "A01", "label": "1", "step": 1},
        {"well": "A02", "label": "1", "step": 1},
        {"well": "A02", "label": "1", "step": 2},
    ]
    assert bool(table.loc[4, "pareto_k_warn"])

    summary = pareto_k_summary(table)

    row = summary[(summary["label"] == "1") & (summary["well"] == "A02")].iloc[0]
    assert row["n"] == 3
    assert row["pareto_k_max"] == 0.8
    assert row["pareto_k_frac_gt_0p7"] == pytest.approx(1 / 3)


def test_weighted_mean_reference_prefers_precise_controls() -> None:
    """Weighted CTR references should favor precise remaining controls."""
    reference, weights = weighted_mean_reference([
        np.array([1.0, 1.0, 1.0, 1.0]),
        np.array([0.0, 2.0, 0.0, 2.0]),
    ])

    assert weights[0] > weights[1]
    assert reference.shape == (4,)


def test_weighted_mean_reference_edge_cases() -> None:
    """CTR weighted references handle exact and invalid variances explicitly."""
    exact_reference, exact_weights = weighted_mean_reference([
        np.array([2.0, 2.0, 2.0]),
        np.array([0.0, 1.0, 2.0]),
    ])
    fallback_reference, fallback_weights = weighted_mean_reference([
        np.array([np.nan, np.nan]),
        np.array([np.nan, np.nan]),
    ])

    with pytest.raises(ValueError, match="No arrays"):
        weighted_mean_reference([])
    np.testing.assert_allclose(exact_reference, np.array([2.0, 2.0, 2.0]))
    np.testing.assert_allclose(exact_weights, np.array([1.0, 0.0]))
    np.testing.assert_allclose(fallback_weights, np.array([0.5, 0.5]))
    np.testing.assert_allclose(fallback_reference, np.array([0.0, 0.0]))
    assert fallback_reference.shape == (2,)


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


def test_summarize_bayesian_ctr_holdout_shared_and_errors() -> None:
    """Bayesian CTR holdout summaries should validate required posterior variables."""
    posterior = xr.Dataset(
        data_vars={
            "K_A01": (("chain", "draw"), np.array([[7.0, 7.1, 6.9, 7.0]])),
            "K_ctr_ctrl": (("chain", "draw"), np.array([[7.0, 7.0, 7.0, 7.0]])),
        },
        coords={"chain": [0], "draw": [0, 1, 2, 3]},
    )
    trace = xr.DataTree.from_dict({"posterior": posterior})

    row = summarize_bayesian_ctr_holdout(
        trace,
        trace_id="shared-model",
        ctr_group="ctrl",
        heldout_well="A01",
        reference_mode="shared",
    )

    assert row["ctr_reference_vars"] == "K_ctr_ctrl"
    assert row["delta_k_hdi94_contains_zero"] is True
    with pytest.raises(ValueError, match="remaining_ctr_wells"):
        summarize_bayesian_ctr_holdout(
            trace,
            trace_id="bad",
            ctr_group="ctrl",
            heldout_well="A01",
            reference_mode="weighted_mean",
        )
    with pytest.raises(ValueError, match="Unsupported"):
        summarize_bayesian_ctr_holdout(
            trace,
            trace_id="bad",
            ctr_group="ctrl",
            heldout_well="A01",
            reference_mode="median",
        )
    with pytest.raises(KeyError, match="Missing heldout"):
        summarize_bayesian_ctr_holdout(
            trace,
            trace_id="bad",
            ctr_group="ctrl",
            heldout_well="B01",
        )


def test_ctr_summary_classical_rows_and_prior_widening() -> None:
    """Classical and preliminary-fit CTR helpers should preserve uncertainty logic."""

    class Scheme:
        def __init__(self) -> None:
            self.names = {"ctrl": {"A01", "A02", "A03"}}

    def fit_result(k: float, stderr: float | None) -> FitResult:
        params = Parameters()
        params.add("K", value=k)
        params["K"].stderr = stderr
        return FitResult(result=type("Result", (), {"params": params})())

    results = {
        "A01": fit_result(7.0, 0.05),
        "A02": fit_result(7.2, 0.10),
        "A03": fit_result(6.8, 0.20),
    }

    widened = widen_heldout_k_prior(results, "A01", n_sd=2.0, prior_sigma=0.6)
    rows = classical_ctr_holdout_rows(results, Scheme(), trace_id="lm", rope=0.3)
    summary = summarize_ctr_loo_table(
        pd.DataFrame({
            "trace_id": ["lm", "lm"],
            "delta_k_mean": [0.1, -0.2],
            "delta_k_abs_mean": [0.1, 0.2],
            "delta_k_sd": [0.05, 0.10],
            "z_delta_k": [2.0, -2.0],
            "delta_k_hdi89_contains_zero": [True, False],
            "delta_k_hdi94_contains_zero": [True, True],
            "p_abs_delta_k_lt_rope": [1.0, 0.0],
        })
    )

    assert widened["A01"].result.params["K"].stderr == pytest.approx(0.3)
    assert set(rows["heldout_well"]) == {"A01", "A02", "A03"}
    assert rows.loc[rows["heldout_well"] == "A01", "z_delta_k"].notna().all()
    assert summary.loc[0, "ctr_loo_n"] == 2
    assert summary.loc[0, "ctr_loo_rmse"] == pytest.approx(np.sqrt(0.025))


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
    x = np.array([6.0, 7.0])
    y = binding_1site(x, 7.0, 100.0, 200.0, is_ph=True)
    ds = Dataset(
        {
            "1": DataArray(
                x,
                y,
                y_errc=np.array([10.0, 20.0]),
            )
        },
        is_ph=True,
    )
    fr = FitResult(
        result=type("Result", (), {"params": params})(), trace=trace, dataset=ds
    )
    multi = MultiFitResult(trace=trace, results={"A01": fr})

    residuals = residuals_from_multifit(multi, "trace", binding_1site)

    np.testing.assert_allclose(residuals["sigma"].to_numpy(), np.array([10.0, 20.0]))
    assert "likelihood_res" in residuals.columns
    assert "is_residual_outlier" in residuals.columns


def test_residuals_from_multifit_returns_convenient_schema() -> None:
    """MultiFit residuals should expose notebook-friendly residual metadata."""
    trace = xr.DataTree.from_dict({"posterior": xr.Dataset()})

    params = Parameters()
    params.add("K", value=7.0)
    params.add("S0_1", value=100.0)
    params.add("S1_1", value=200.0)
    x = np.array([6.0, 7.0])
    y = binding_1site(x, 7.0, 100.0, 200.0, is_ph=True)
    ds = Dataset(
        {
            "1": DataArray(
                x,
                y,
                y_errc=np.array([10.0, 20.0]),
            )
        },
        is_ph=True,
    )
    fr = FitResult(
        result=type("Result", (), {"params": params})(), trace=trace, dataset=ds
    )
    multi = MultiFitResult(trace=trace, results={"A01": fr})

    residuals = residuals_from_multifit(multi, "trace", binding_1site)

    assert list(residuals.columns) == RESIDUAL_TABLE_COLUMNS
    assert residuals.loc[0, "trace_id"] == "trace"
    assert residuals.loc[0, "well"] == "A01"
    assert residuals.loc[0, "label"] == "1"
    assert residuals.loc[0, "step"] == 0
    assert residuals.loc[0, "residual_likelihood"] == "normal"
    assert bool(residuals["p_outlier"].isna().iloc[0])
    assert not bool(residuals.loc[0, "is_residual_outlier"])


def test_residuals_from_multifit_maps_outlier_probability_by_point() -> None:
    """Mixture outlier probabilities should align with well/step residual rows."""
    posterior = xr.Dataset(
        data_vars={
            "outlier_probability_1": (
                ("chain", "draw", "obs"),
                np.array([[[0.1, 0.2, 0.3, 0.4]]]),
            )
        },
        coords={"chain": [0], "draw": [0], "obs": [0, 1, 2, 3]},
    )
    trace = xr.DataTree.from_dict({"posterior": posterior})

    params = Parameters()
    params.add("K", value=7.0)
    params.add("S0_1", value=100.0)
    params.add("S1_1", value=200.0)
    x = np.array([6.0, 7.0])
    y = binding_1site(x, 7.0, 100.0, 200.0, is_ph=True)

    def fit_result() -> FitResult:
        ds = Dataset(
            {"1": DataArray(x, y, y_errc=np.array([10.0, 20.0]))},
            is_ph=True,
        )
        return FitResult(
            result=type("Result", (), {"params": params})(),
            trace=trace,
            dataset=ds,
        )

    multi = MultiFitResult(
        trace=trace,
        results={"A01": fit_result(), "A02": fit_result()},
    )

    residuals = residuals_from_multifit(multi, "trace", binding_1site)

    assert residuals[["well", "step"]].to_dict("records") == [
        {"well": "A01", "step": 0},
        {"well": "A01", "step": 1},
        {"well": "A02", "step": 0},
        {"well": "A02", "step": 1},
    ]
    np.testing.assert_allclose(
        residuals["p_outlier"].to_numpy(),
        np.array([0.1, 0.3, 0.2, 0.4]),
    )


def test_residuals_from_multifit_empty_result_keeps_schema() -> None:
    """Empty MultiFit residuals should still be easy to concatenate/display."""
    trace = xr.DataTree.from_dict({"posterior": xr.Dataset()})
    multi = MultiFitResult(trace=trace, results={})

    residuals = residuals_from_multifit(multi, "trace", binding_1site)

    assert residuals.empty
    assert list(residuals.columns) == RESIDUAL_TABLE_COLUMNS


def test_residuals_from_fit_results_robust_params_and_drop_invalid_sigma() -> None:
    """Classical residual tables should expose optional params and drop bad sigma rows."""
    params = Parameters()
    params.add("K", value=7.0)
    params.add("S0_1", value=100.0)
    params.add("S1_1", value=200.0)
    x = np.array([6.0, 7.0, 8.0])
    y = binding_1site(x, 7.0, 100.0, 200.0, is_ph=True) + np.array([1.0, 0.0, 1.0])
    ds = Dataset(
        {"1": DataArray(x, y, y_errc=np.array([1.0, 0.0, 2.0]))},
        is_ph=True,
    )
    fr: FitResult = FitResult(
        result=type("Result", (), {"params": params})(), dataset=ds
    )

    residuals = residuals_from_fit_results(
        {"A01": fr},
        "robust",
        binding_1site,
        include_fit_params=True,
        robust=True,
        student_t_nu=3.0,
        outlier_threshold=0.5,
    )

    assert len(residuals) == 2
    assert residuals["residual_likelihood"].eq("student_t").all()
    assert {"K", "S0_1", "S1_1"}.issubset(residuals.columns)
    assert residuals["is_residual_outlier"].any()


def _single_well_fit_result(mini: object) -> FitResult:
    """Build a minimal single-well FitResult carrying *mini* as its trace."""
    params = Parameters()
    params.add("K", value=7.0)
    params.add("S0_1", value=100.0)
    params.add("S1_1", value=200.0)
    x = np.array([6.0, 7.0, 8.0])
    y = binding_1site(x, 7.0, 100.0, 200.0, is_ph=True) + np.array([1.0, 0.0, 40.0])
    ds = Dataset({"1": DataArray(x, y, y_errc=np.array([1.0, 1.0, 2.0]))}, is_ph=True)
    result = cast("Any", type("Result", (), {"params": params})())
    return FitResult(result=result, dataset=ds, trace=cast("Any", mini))


def test_single_well_residuals_extract_p_outlier_from_mixture_trace() -> None:
    """A single-well mixture trace should surface per-point outlier probabilities."""
    known = np.array([0.02, 0.10, 0.97])
    outlier_prob = xr.DataArray(
        np.broadcast_to(known, (2, 4, 3)), dims=["chain", "draw", "obs"]
    )
    trace = type(
        "TraceMix",
        (),
        {"posterior": xr.Dataset({"outlier_probability_1": outlier_prob})},
    )()
    df = _single_well_fit_result(trace).residual_table(robust=False, well="A01")

    assert "p_outlier" in df.columns
    assert np.allclose(df["p_outlier"].to_numpy(), known)
    # Schema parity with the multi-well builder.
    assert list(df.columns)[: len(RESIDUAL_TABLE_COLUMNS)] == RESIDUAL_TABLE_COLUMNS


@pytest.mark.parametrize(
    "mini",
    [
        None,
        type(
            "TraceNoMix",
            (),
            {"posterior": xr.Dataset({"K": xr.DataArray(np.zeros((2, 4)))})},
        )(),
    ],
)
def test_single_well_residuals_p_outlier_nan_without_mixture(mini: object) -> None:
    """Without a mixture deterministic (or any trace) the column stays NaN."""
    df = _single_well_fit_result(mini).residual_table(robust=False, well="A01")
    assert "p_outlier" in df.columns
    assert df["p_outlier"].isna().all()


def test_single_well_residuals_label_mixture_without_t_transform() -> None:
    """A mixture fit is labeled 'mixture' with Normal-standardized std_res.

    The mixture uses Normal components, so ``std_res`` must be the identity of
    ``likelihood_res`` (no Student-t transform) and ``student_t_nu`` NaN; the
    outlier structure is reported via ``p_outlier``.
    """
    known = np.array([0.02, 0.10, 0.97])
    op = xr.DataArray(np.broadcast_to(known, (2, 4, 3)), dims=["chain", "draw", "obs"])
    pi = xr.DataArray(np.full((2, 4), 0.1), dims=["chain", "draw"])
    infl = xr.DataArray(np.full((2, 4), 0.5), dims=["chain", "draw"])
    trace = type(
        "TraceMix",
        (),
        {
            "posterior": xr.Dataset({
                "outlier_probability_1": op,
                "pi_outlier_1": pi,
                "outlier_inflate": infl,
            })
        },
    )()
    # No robust override: the family is auto-detected from the trace.
    df = _single_well_fit_result(trace).residual_table(well="A01")

    assert (df["residual_likelihood"] == "mixture").all()
    assert df["student_t_nu"].isna().all()
    np.testing.assert_allclose(
        df["std_res"].to_numpy(), df["likelihood_res"].to_numpy()
    )
    assert np.allclose(df["p_outlier"].to_numpy(), known)


def test_trace_diagnostics_collects_stats_loo_and_x_sanity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trace diagnostics should combine sample stats, summary, LOO, and x-axis checks."""
    posterior = xr.Dataset(
        data_vars={
            "K": (("chain", "draw"), np.array([[7.0, 7.1]])),
            "x_true": (
                ("chain", "draw", "step", "well"),
                np.array([[[[8.0, 8.0], [7.0, 7.1]], [[8.1, 8.1], [7.2, 7.3]]]]),
            ),
        },
        coords={"chain": [0], "draw": [0, 1], "step": [0, 1], "well": ["A01", "A02"]},
    )
    sample_stats = xr.Dataset(
        data_vars={
            "diverging": (("chain", "draw"), np.array([[False, True]])),
            "tree_depth": (("chain", "draw"), np.array([[3, 5]])),
            "reached_max_treedepth": (("chain", "draw"), np.array([[False, True]])),
            "energy": (("chain", "draw"), np.array([[1.0, 3.0]])),
        },
        coords={"chain": [0], "draw": [0, 1]},
    )
    log_likelihood = xr.Dataset(
        data_vars={"obs": (("chain", "draw", "obs_id"), np.zeros((1, 2, 2)))},
        coords={"chain": [0], "draw": [0, 1], "obs_id": [0, 1]},
    )
    trace = xr.DataTree.from_dict({
        "posterior": posterior,
        "sample_stats": sample_stats,
        "log_likelihood": log_likelihood,
    })

    class FakeLoo:
        elpd_loo = -1.0
        p_loo = 0.5
        se = 0.25
        pareto_k = np.array([0.2, 0.8])

    monkeypatch.setattr(
        "clophfit.fitting.model_validation.az.summary",
        lambda *_args, **_kwargs: pd.DataFrame({
            "r_hat": [1.01],
            "ess_bulk": [100.0],
            "ess_tail": [80.0],
        }),
    )
    monkeypatch.setattr(
        "clophfit.fitting.model_validation.az.loo",
        lambda *_args, **_kwargs: FakeLoo(),
    )

    row = trace_diagnostics(trace, compute_loo=True)
    sanity = x_axis_sanity(trace)

    assert row["n_divergences"] == 1
    assert row["tree_depth_max"] == 5
    assert row["rhat_max"] == 1.01
    assert row["elpd_loo"] == -1.0
    assert row["pareto_k_frac_gt_0p7"] == 0.5
    assert row["x_first_well"] == "A01"
    assert sanity["x_step0_max_abs_spread"] == pytest.approx(0.0)


@pytest.fixture
def annotated_diagnostics() -> ResidualDiagnostics:
    """Build a small annotated ResidualDiagnostics for plot smoke tests."""
    df = pd.DataFrame({
        "trace_id": ["m1"] * 8,
        "well": ["A01", "A02", "B01", "B02"] * 2,
        "label": ["1", "1", "2", "2"] * 2,
        "step": [0, 0, 0, 0, 1, 1, 1, 1],
        "x": [8.9, 8.9, 8.9, 8.9, 8.2, 8.2, 8.2, 8.2],
        "y": [1.0, 1.1, 2.0, 2.1, 1.2, 1.3, 2.2, 2.3],
        "yhat": [1.0, 1.2, 1.9, 2.0, 1.1, 1.4, 2.1, 2.2],
        "sigma": [1.0] * 8,
        "std_res": [0.1, -0.2, 0.3, -0.4, 0.2, -0.1, 0.4, -0.3],
    })
    return ResidualDiagnostics(df).annotate(ctrl_wells={"A01"})


def test_residual_diagnostics_plots_smoke(
    annotated_diagnostics: ResidualDiagnostics,
) -> None:
    """Plotting helpers should build figures without a display backend."""
    import matplotlib as mpl  # noqa: PLC0415

    mpl.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    diag = annotated_diagnostics.label_scaled()
    for figure in (
        diag.plot_hist_qq(),
        diag.plot_step(),
        diag.plot_role(),
        diag.plot_col(),
        diag.plot_well_summary(),
    ):
        assert figure is not None
    plt.close("all")


def test_residual_diagnostics_plot_role_requires_annotation() -> None:
    """plot_role should refuse to run before role annotation."""
    df = pd.DataFrame({
        "trace_id": ["m1", "m1"],
        "well": ["A01", "A02"],
        "label": ["1", "1"],
        "step": [0, 1],
        "x": [8.9, 8.2],
        "std_res": [0.1, -0.2],
    })
    with pytest.raises(ValueError, match="annotate"):
        ResidualDiagnostics(df).plot_role()


def test_residual_diagnostics_plot_col_requires_annotation() -> None:
    """plot_col should refuse to run before column annotation."""
    df = pd.DataFrame({
        "trace_id": ["m1", "m1"],
        "well": ["A01", "A02"],
        "label": ["1", "1"],
        "step": [0, 1],
        "x": [8.9, 8.2],
        "std_res": [0.1, -0.2],
    })
    with pytest.raises(ValueError, match="annotate"):
        ResidualDiagnostics(df).plot_col()


def test_residual_diagnostics_analysis_methods_smoke() -> None:
    """ResidualDiagnostics exposes the distribution/trend/correlation analyses."""
    df = pd.DataFrame({
        "trace_id": ["m1"] * 8,
        "well": ["A01", "A02"] * 4,
        "label": ["1", "1", "2", "2"] * 2,
        "step": [0, 0, 0, 0, 1, 1, 1, 1],
        "x": [8.9, 8.9, 8.9, 8.9, 8.2, 8.2, 8.2, 8.2],
        "std_res": np.linspace(-1.0, 1.0, 8),
    })
    diag = ResidualDiagnostics(df)
    assert len(diag.analysis.distribution_summary()) == 2  # one row per label
    assert isinstance(diag.analysis.x_correlation(), pd.DataFrame)
    lag, lag_summary = diag.analysis.lag1_autocorrelation()
    assert isinstance(lag, pd.DataFrame)
    assert isinstance(lag_summary, pd.DataFrame)
