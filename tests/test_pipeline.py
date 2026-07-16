"""Tests for plate-level fitting pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pytest

from clophfit.fitting import pipeline
from clophfit.fitting.data_structures import DataArray, Dataset, FitResult
from clophfit.fitting.errors import InsufficientDataError


def _dataset() -> Dataset:
    x = np.array([6.0, 7.0, 8.0])
    y = np.array([1.0, 2.0, 3.0])
    return Dataset({"1": DataArray(x, y, y_errc=np.ones_like(y))}, is_ph=True)


@dataclass
class _Result:
    residual: np.ndarray
    success: bool = True


def test_noise_model_building_and_convergence() -> None:
    """Noise-model helpers should compare gain and alpha per label."""
    old = pipeline._plate_noise_model_from_nnls(  # noqa: SLF001
        {"1": 1.0}, {"1": 0.2}, {"1": 0.03}
    )
    same = pipeline._plate_noise_model_from_nnls(  # noqa: SLF001
        {"1": 1.0}, {"1": 0.2001}, {"1": 0.03001}
    )
    changed = pipeline._plate_noise_model_from_nnls(  # noqa: SLF001
        {"1": 1.0}, {"1": 0.3}, {"1": 0.03}
    )
    missing = pipeline._plate_noise_model_from_nnls(  # noqa: SLF001
        {"1": 1.0, "2": 2.0},
        {"1": 0.2, "2": 0.0},
        {"1": 0.03, "2": 0.0},
    )

    assert old["1"].sigma_floor == 1.0
    assert old["1"].sigma_ph == 0.0
    assert pipeline._noise_params_converged(old, same, tol=1e-2)  # noqa: SLF001
    assert not pipeline._noise_params_converged(  # noqa: SLF001
        old, changed, tol=1e-2
    )
    assert not pipeline._noise_params_converged(  # noqa: SLF001
        old, missing, tol=1e-2
    )


@pytest.mark.parametrize(
    ("method", "target"),
    [
        ("", "lm"),
        ("huber", "huber"),
        ("odr", "odr"),
        ("mcmc", "mcmc"),
    ],
)
def test_fit_plate_routes_methods_and_preserves_well_keys(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    target: str,
) -> None:
    """Plate fitting should dispatch each method to the intended backend."""
    calls: list[tuple[str, str]] = []

    def fake_glob(
        ds: Dataset, *, method: str = "lm", **_kwargs: object
    ) -> FitResult[Any]:
        calls.append(("glob", method))
        return FitResult(result=_Result(np.zeros(len(ds["1"].y))), dataset=ds)

    def fake_odr(ds: Dataset, **_kwargs: object) -> FitResult[Any]:
        calls.append(("odr", "odr"))
        return FitResult(result=_Result(np.zeros(len(ds["1"].y))), dataset=ds)

    def fake_mcmc(ds: Dataset, **_kwargs: object) -> FitResult[Any]:
        calls.append(("mcmc", "mcmc"))
        return FitResult(result=_Result(np.zeros(len(ds["1"].y))), dataset=ds)

    monkeypatch.setattr("clophfit.fitting.pipeline.fit_binding_glob", fake_glob)
    monkeypatch.setattr("clophfit.fitting.pipeline.fit_binding_odr", fake_odr)
    monkeypatch.setattr("clophfit.fitting.pipeline.fit_binding_pymc", fake_mcmc)

    results = pipeline.fit_plate({"A01": _dataset(), "A02": _dataset()}, method=method)

    assert set(results) == {"A01", "A02"}
    assert all(fr.result is not None for fr in results.values())
    assert calls == [(target if target in {"odr", "mcmc"} else "glob", target)] * 2


@pytest.mark.parametrize("method", ["lm", "odr", "mcmc"])
def test_fit_plate_turns_insufficient_data_into_empty_result(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
) -> None:
    """Per-well insufficient-data failures should not abort the whole plate."""

    def fail(*_args: object, **_kwargs: object) -> FitResult[Any]:
        msg = "too few points"
        raise InsufficientDataError(msg)

    monkeypatch.setattr("clophfit.fitting.pipeline.fit_binding_glob", fail)
    monkeypatch.setattr("clophfit.fitting.pipeline.fit_binding_odr", fail)
    monkeypatch.setattr("clophfit.fitting.pipeline.fit_binding_pymc", fail)

    results = pipeline.fit_plate({"A01": _dataset()}, method=method)

    assert results["A01"].result is None


def test_fgls_plate_fit_uses_calibration_fallback_and_second_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FGLS should continue with fixed floors when calibration raises."""
    methods: list[str] = []

    def fake_fit(
        ds: Dataset, *, method: str = "lm", **_kwargs: object
    ) -> FitResult[Any]:
        methods.append(method)
        return FitResult(result=_Result(np.zeros(len(ds["1"].y))), dataset=ds)

    def fake_residuals(
        results: dict[str, FitResult[Any]], *_args: object, **_kwargs: object
    ) -> pd.DataFrame:
        assert "A01" in results
        return pd.DataFrame({
            "well": ["A01"],
            "label": ["1"],
            "x": [7.0],
            "resid_weighted": [0.0],
            "resid_raw": [0.0],
            "y_err": [1.0],
            "predicted": [2.0],
        })

    monkeypatch.setattr("clophfit.fitting.pipeline.fit_binding_glob", fake_fit)
    monkeypatch.setattr(
        "clophfit.fitting.pipeline.residuals_from_fit_results", fake_residuals
    )
    monkeypatch.setattr(
        "clophfit.fitting.pipeline.fit_noise_model_nnls",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("singular")),
    )
    monkeypatch.setattr(
        "clophfit.fitting.pipeline.compute_plate_slopes", lambda *_args: {}
    )
    monkeypatch.setattr(
        "clophfit.fitting.pipeline.fit_ph_slope_noise", lambda *_args: 0.05
    )

    first_method = "huber"
    second_method = "lm"
    results, noise = pipeline.fgls_plate_fit(
        {"A01": _dataset()},
        {"1": 1.0},
        first_pass_method=first_method,
        second_pass_method=second_method,
        max_iter=2,
    )

    assert methods == ["huber", "lm"]
    assert results["A01"].result is not None
    assert noise["1"].sigma_floor == 1.0
    assert noise["1"].gain == 0.0
    assert noise["1"].alpha == 0.0
    assert noise["1"].sigma_ph == 0.05


def test_fgls_plate_fit_converges_and_handles_failed_well(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FGLS should insert empty FitResult for failed wells."""
    calls = 0

    def fake_fit(
        ds: Dataset, *, method: str = "lm", **_kwargs: object
    ) -> FitResult[Any]:
        nonlocal calls
        assert method in {"huber", "lm"}
        calls += 1
        if ds is datasets["bad"]:
            msg = "too few points"
            raise InsufficientDataError(msg)
        return FitResult(result=_Result(np.zeros(len(ds["1"].y))), dataset=ds)

    def fake_residuals(
        _results: dict[str, FitResult[Any]], *_args: object, **_kwargs: object
    ) -> pd.DataFrame:
        return pd.DataFrame({
            "well": ["good"],
            "label": ["1"],
            "x": [7.0],
            "resid_weighted": [0.0],
            "resid_raw": [0.0],
            "y_err": [1.0],
            "predicted": [2.0],
        })

    datasets = {"good": _dataset(), "bad": _dataset()}
    monkeypatch.setattr("clophfit.fitting.pipeline.fit_binding_glob", fake_fit)
    monkeypatch.setattr(
        "clophfit.fitting.pipeline.residuals_from_fit_results", fake_residuals
    )
    monkeypatch.setattr(
        "clophfit.fitting.pipeline.fit_noise_model_nnls",
        lambda *_args, **_kwargs: ({"1": 1.0}, {"1": 0.2}, {"1": 0.03}),
    )
    monkeypatch.setattr(
        "clophfit.fitting.pipeline.compute_plate_slopes", lambda *_args: {}
    )
    monkeypatch.setattr(
        "clophfit.fitting.pipeline.fit_ph_slope_noise", lambda *_args: 0.0
    )

    results, noise = pipeline.fgls_plate_fit(datasets, {"1": 1.0}, max_iter=1, tol=1e-9)

    assert calls == 2
    assert results["good"].result is not None
    assert results["bad"].result is None
    assert noise["1"].gain == 0.2


def test_calibrate_noise_robust_screens_high_p_outlier() -> None:
    """A high-p_outlier point is dropped, so it no longer inflates the gain."""
    y = np.linspace(50.0, 500.0, 120)
    df = pd.DataFrame({
        "label": "1",
        "raw_res": np.full(120, 2.0),
        "yhat": y,
        "p_outlier": np.zeros(120),
    })
    # Inject one gross outlier flagged by the mixture.
    df.loc[0, "raw_res"] = 1.0e4
    df.loc[0, "p_outlier"] = 0.99

    screened = pipeline.calibrate_noise_robust(df, {"1": 1.0}, p_threshold=0.9)
    # p_threshold above 1.0 keeps every point (nothing screened).
    unscreened = pipeline.calibrate_noise_robust(df, {"1": 1.0}, p_threshold=2.0)

    assert screened["1"].sigma_floor == 1.0
    assert screened["1"].gain < unscreened["1"].gain


def test_calibrate_noise_robust_without_probability_column() -> None:
    """Without p_outlier, all points are used (no screening)."""
    y = np.linspace(50.0, 500.0, 80)
    df = pd.DataFrame({"label": "1", "raw_res": np.full(80, 2.0), "yhat": y})
    model = pipeline.calibrate_noise_robust(df, {"1": 1.5})
    assert model["1"].sigma_floor == 1.5
    assert model["1"].gain >= 0.0
    assert model["1"].alpha >= 0.0
