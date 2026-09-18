"""Sandwich standard errors on short titration series: what they fix, and what they only look like."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from clophfit.fitting.core import fit_binding_glob
from clophfit.fitting.data_structures import DataArray, Dataset
from clophfit.fitting.models import binding_1site
from clophfit.fitting.robust_se import hc_covariance, robust_k_se, sandwich_se

PH = np.array([8.9, 8.3, 7.7, 7.1, 6.5, 5.9, 5.3])
K_TRUE, FLOOR = 7.0, 8.0


def _wells(
    n: int, *, gain: float, seed: int
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Fit n wells with flat weights; return their residual table and fitted parameters.

    ``gain`` sets how much the noise grows with signal: 0 is homoscedastic, and
    the weights (a flat floor) are then correct.
    """
    rng = np.random.default_rng(seed)
    rows, params = [], {}
    for i in range(n):
        well = f"W{i:03d}"
        arrays, truth = {}, {"K": K_TRUE}
        for lbl, (s0, s1) in (("1", (300.0, 1800.0)), ("2", (1500.0, 150.0))):
            clean = binding_1site(PH, K_TRUE, s0, s1, is_ph=True)
            sd = np.sqrt(FLOOR**2 + gain * clean)
            arrays[lbl] = DataArray(
                PH, clean + rng.normal(0.0, sd), y_errc=np.full(PH.size, FLOOR)
            )
            truth |= {f"S0_{lbl}": s0, f"S1_{lbl}": s1}
        fit = fit_binding_glob(Dataset(arrays, is_ph=True), method="lm")
        if fit.result is None:
            continue
        pars = {n_: float(fit.result.params[n_].value) for n_ in truth}
        params[well] = pars
        for lbl in ("1", "2"):
            yhat = binding_1site(
                PH, pars["K"], pars[f"S0_{lbl}"], pars[f"S1_{lbl}"], is_ph=True
            )
            rows += [
                {
                    "well": well,
                    "label": lbl,
                    "step": s,
                    "x": PH[s],
                    "yhat": yhat[s],
                    "raw_res": float(arrays[lbl].y[s] - yhat[s]),
                    "sigma": FLOOR,
                    "k_fitted": pars["K"],
                    "k_se_lmfit": float(fit.result.params["K"].stderr or np.nan),
                }
                for s in range(PH.size)
            ]
    return pd.DataFrame(rows), params


def _coverage(
    table: pd.DataFrame, se: pd.DataFrame, col: str, mult: np.ndarray | float
) -> float:
    """Share of wells whose interval covers the true K."""
    k = table.groupby("well")["k_fitted"].first().loc[se.index]
    return float(np.mean(np.abs(k - K_TRUE) < mult * se[col]))


def test_both_errors_track_the_real_spread_of_k() -> None:
    """Flat weights on signal-dependent noise still give an SE near the spread of K."""
    table, params = _wells(150, gain=6.0, seed=1)
    se = robust_k_se(table, params)
    spread = float(table.groupby("well")["k_fitted"].first().std())
    # The classical error is not blind to heteroscedasticity here: it takes its
    # scale from the residuals themselves, and only the leverage weighting differs.
    assert float(se["k_se_model"].median()) == pytest.approx(spread, rel=0.2)
    assert float(se["k_se_robust"].median()) == pytest.approx(spread, rel=0.25)
    assert float(se["ratio"].median()) > 1.0


def test_the_inflation_is_a_small_sample_correction_not_a_heteroscedasticity_signal() -> (
    None
):
    """Nine degrees of freedom: HC3 inflates by the same ~25% with correct weights.

    And a Student-t quantile on the classical error covers as well as HC3 with a
    normal one, so on series this short the sandwich is not what fixes coverage.
    """
    flat, flat_p = _wells(200, gain=0.0, seed=2)
    fan, fan_p = _wells(200, gain=30.0, seed=3)
    se_flat, se_fan = robust_k_se(flat, flat_p), robust_k_se(fan, fan_p)
    assert float(se_flat["ratio"].median()) == pytest.approx(
        float(se_fan["ratio"].median()), abs=0.1
    )
    t_quantile = stats.t.ppf(0.975, se_fan["dof"].to_numpy())
    assert _coverage(fan, se_fan, "k_se_model", 1.96) < 0.93
    assert _coverage(fan, se_fan, "k_se_model", t_quantile) == pytest.approx(
        0.95, abs=0.03
    )
    assert _coverage(fan, se_fan, "k_se_robust", 1.96) == pytest.approx(0.95, abs=0.03)


def test_hc3_is_the_most_conservative_and_hc0_the_least() -> None:
    """The small-sample corrections order as HC0 <= HC2 <= HC3 on the same fit."""
    table, params = _wells(20, gain=6.0, seed=3)
    med = {
        kind: float(robust_k_se(table, params, kind=kind)["k_se_robust"].median())
        for kind in ("HC0", "HC2", "HC3")
    }
    assert med["HC0"] < med["HC2"] < med["HC3"]


def test_a_rank_deficient_design_has_no_covariance() -> None:
    """A duplicated column leaves the parameters unidentified: None, not a wrong number."""
    x = np.linspace(0, 1, 10)
    jac = np.column_stack([x, x])
    assert hc_covariance(jac, np.ones(10)) is None
    assert sandwich_se(jac, np.ones(10)) is None
