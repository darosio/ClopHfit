"""Residuals of a robust likelihood are standardized through that likelihood.

``ppr`` exported every residual table as Normal: ``export_residuals`` called
``residuals_from_fit_results`` without the fit's likelihood, so a Student-t
fit's ``std_res`` was the raw ``(y - mu) / sigma`` - t-tailed by design, 5.8%
beyond |3| at nu = 3 - and its residual diagnostics reported the model's own
tails as outliers. A contamination mixture had no transform at all. Under a
correct model, ``Phi^-1(F(r))`` with ``F`` the likelihood's CDF is standard
Normal whatever the family, so residuals of different models compare on one scale.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest
import xarray as xr
from lmfit import Parameters  # type: ignore[import-untyped]
from scipy import stats as sp_stats

from clophfit.fitting.data_structures import DataArray, Dataset, FitResult
from clophfit.fitting.model_validation import (
    mixture_params_from_trace,
    residual_normal_scores,
)
from clophfit.fitting.models import binding_1site
from clophfit.prtecan import export

if TYPE_CHECKING:
    import pandas as pd

RNG = np.random.default_rng(7)


def _posterior(**vars_: float) -> xr.DataTree:
    post = xr.Dataset(
        {k: (("chain", "draw"), np.full((2, 3), v)) for k, v in vars_.items()},
        coords={"chain": [0, 1], "draw": [0, 1, 2]},
    )
    return xr.DataTree.from_dict({"posterior": post})


# -- the transforms -------------------------------------------------------------


def test_mixture_scores_are_standard_normal_under_the_mixture() -> None:
    """Residuals drawn from the mixture itself map to N(0, 1)."""
    pi, inflate, n = 0.1, 4.0, 20000
    wide = RNG.random(n) < pi
    r = RNG.normal(0.0, np.where(wide, 1.0 + inflate, 1.0))
    z = residual_normal_scores(r, mixture=(pi, inflate))
    assert sp_stats.kstest(z, "norm").pvalue > 0.01
    assert np.mean(np.abs(r) > 3) > 0.05  # the raw residual is heavy-tailed
    assert np.mean(np.abs(z) > 3) == pytest.approx(0.0027, abs=0.0015)


def test_student_t_scores_are_standard_normal_under_the_t() -> None:
    """The existing Student-t transform, for comparison."""
    r = sp_stats.t.rvs(df=3, size=20000, random_state=RNG)
    z = residual_normal_scores(r, robust=True, student_t_nu=3.0)
    assert sp_stats.kstest(z, "norm").pvalue > 0.01


def test_a_mixture_without_outliers_is_the_identity() -> None:
    """Pi = 0 leaves Normal residuals as they are."""
    r = np.array([-2.0, -0.3, 0.0, 1.5, 3.2])
    np.testing.assert_allclose(
        residual_normal_scores(r, mixture=(0.0, 3.0)), r, atol=1e-9
    )


def test_mixture_params_come_from_the_trace() -> None:
    """Posterior means of pi_outlier_<label> and outlier_inflate, per label."""
    trace = _posterior(pi_outlier_1=0.05, pi_outlier_2=0.2, outlier_inflate=3.0)
    assert mixture_params_from_trace(trace, "1") == pytest.approx((0.05, 3.0))
    assert mixture_params_from_trace(trace, "2") == pytest.approx((0.2, 3.0))
    assert mixture_params_from_trace(_posterior(student_t_nu=3.0), "1") is None
    assert mixture_params_from_trace(None, "1") is None


# -- what ppr writes --------------------------------------------------------------


def _fit(trace: object, noise: np.ndarray) -> FitResult:
    params = Parameters()
    params.add("K", value=7.0)
    params.add("S0_1", value=100.0)
    params.add("S1_1", value=200.0)
    x = np.linspace(5.0, 9.0, 7)
    y = binding_1site(x, 7.0, 100.0, 200.0, is_ph=True) + noise
    ds = Dataset({"1": DataArray(x, y, y_errc=np.full(7, 1.0))}, is_ph=True)
    result = cast("Any", type("Result", (), {"params": params})())
    return FitResult(result=result, dataset=ds, trace=cast("Any", trace))


def _col(table: pd.DataFrame, well: str, column: str) -> np.ndarray:
    return table.loc[table.well == well, column].to_numpy(dtype=float)


def test_export_standardizes_each_fit_by_its_own_likelihood() -> None:
    """Student-t, mixture and least-squares fits on one plate, each by its family."""
    noise = np.array([0.5, -1.0, 6.0, 0.2, -0.4, 1.1, -7.0])
    t_trace = _posterior(student_t_nu=3.0)
    mix_trace = _posterior(pi_outlier_1=0.1, outlier_inflate=4.0)
    fits = {
        "A01": _fit(t_trace, noise),
        "A02": _fit(mix_trace, noise),
        "A03": _fit(None, noise),
    }
    table = export.likelihood_residual_table(fits)
    r = _col(table, "A01", "likelihood_res")
    assert set(table[table.well == "A01"].residual_likelihood) == {"student_t"}
    np.testing.assert_allclose(
        _col(table, "A01", "std_res"),
        residual_normal_scores(r, robust=True, student_t_nu=3.0),
    )
    assert set(table[table.well == "A02"].residual_likelihood) == {"mixture"}
    np.testing.assert_allclose(
        _col(table, "A02", "std_res"), residual_normal_scores(r, mixture=(0.1, 4.0))
    )
    assert set(table[table.well == "A03"].residual_likelihood) == {"normal"}
    np.testing.assert_allclose(_col(table, "A03", "std_res"), r)
    # The raw 6-sigma point is an outlier only to the Normal fit.
    normal_max = np.abs(_col(table, "A03", "std_res")).max()
    assert normal_max > 5
    assert np.abs(_col(table, "A01", "std_res")).max() < normal_max


def test_single_well_traces_are_not_pooled() -> None:
    """Each single-well sample keeps its own nu."""
    noise = np.array([0.5, -1.0, 6.0, 0.2, -0.4, 1.1, -7.0])
    fits = {
        "A01": _fit(_posterior(student_t_nu=2.0), noise),
        "A02": _fit(_posterior(student_t_nu=30.0), noise),
    }
    table = export.likelihood_residual_table(fits)
    assert set(_col(table, "A01", "student_t_nu")) == {2.0}
    assert set(_col(table, "A02", "student_t_nu")) == {30.0}


def test_a_multi_well_trace_is_resolved_once_for_all_its_wells() -> None:
    """Wells sharing one trace share its settings."""
    noise = np.zeros(7)
    shared = _posterior(student_t_nu=4.0)
    fits = {w: _fit(shared, noise) for w in ("A01", "A02", "A03")}
    table = export.likelihood_residual_table(fits)
    assert set(table.student_t_nu) == {4.0}
    assert len(table) == 21
