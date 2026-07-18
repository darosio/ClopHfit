"""Tests for plate-level fitting pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from clophfit.fitting import noise_calibration
from clophfit.fitting.data_structures import DataArray, Dataset, FitResult
from clophfit.fitting.errors import InsufficientDataError
from clophfit.prtecan import Titration

data_tests = Path(__file__).parent / "Tecan"


@pytest.fixture(scope="module")
def titration() -> Titration:
    """Build a real Titration used only as the `self` for `fgls_fit_plate`.

    The FGLS tests below pass `datasets` and `sigma_floor` explicitly, so
    only `.scheme` and `.fit_keys` (used to build the returned
    `TitrationResults`) need to be real attributes of a live plate.
    """
    return Titration.fromlistfile(data_tests / "140220/list.pH.csv", is_ph=True)


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
    old = noise_calibration._plate_noise_model_from_nnls(  # noqa: SLF001
        {"1": 1.0}, {"1": 0.2}, {"1": 0.03}
    )
    same = noise_calibration._plate_noise_model_from_nnls(  # noqa: SLF001
        {"1": 1.0}, {"1": 0.2001}, {"1": 0.03001}
    )
    changed = noise_calibration._plate_noise_model_from_nnls(  # noqa: SLF001
        {"1": 1.0}, {"1": 0.3}, {"1": 0.03}
    )
    missing = noise_calibration._plate_noise_model_from_nnls(  # noqa: SLF001
        {"1": 1.0, "2": 2.0},
        {"1": 0.2, "2": 0.0},
        {"1": 0.03, "2": 0.0},
    )

    assert old["1"].sigma_floor == 1.0
    assert old["1"].sigma_ph == 0.0
    assert noise_calibration._noise_params_converged(  # noqa: SLF001
        old, same, tol=1e-2
    )
    assert not noise_calibration._noise_params_converged(  # noqa: SLF001
        old, changed, tol=1e-2
    )
    assert not noise_calibration._noise_params_converged(  # noqa: SLF001
        old, missing, tol=1e-2
    )


def test_fgls_plate_fit_uses_calibration_fallback_and_second_pass(
    titration: Titration,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FGLS should continue with fixed floors when calibration raises."""
    methods: list[str] = []

    def fake_fit(ds: Dataset, *, method: str = "lm", **_kwargs: object) -> FitResult:
        methods.append(method)
        return FitResult(result=_Result(np.zeros(len(ds["1"].y))), dataset=ds)

    def fake_residuals(
        results: dict[str, FitResult], *_args: object, **_kwargs: object
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

    monkeypatch.setattr("clophfit.prtecan.titration.fit_binding_glob", fake_fit)
    monkeypatch.setattr(
        "clophfit.prtecan.titration.residuals_from_fit_results", fake_residuals
    )
    monkeypatch.setattr(
        "clophfit.prtecan.titration.fit_noise_model_nnls",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("singular")),
    )
    monkeypatch.setattr(
        "clophfit.prtecan.titration.compute_plate_slopes", lambda *_args: {}
    )
    monkeypatch.setattr(
        "clophfit.prtecan.titration.fit_ph_slope_noise", lambda *_args: 0.05
    )

    first_method = "huber"
    second_method = "lm"
    res = titration.fgls_fit_plate(
        {"A01": _dataset()},
        sigma_floor={"1": 1.0},
        first_pass_method=first_method,
        second_pass_method=second_method,
        max_iter=2,
    )

    assert methods == ["huber", "lm"]
    assert res.results["A01"].result is not None
    assert res.noise_model is not None
    assert res.noise_model["1"].sigma_floor == 1.0
    assert res.noise_model["1"].gain == 0.0
    assert res.noise_model["1"].alpha == 0.0
    assert res.noise_model["1"].sigma_ph == 0.05


def test_fgls_plate_fit_converges_and_handles_failed_well(
    titration: Titration,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FGLS should insert empty FitResult for failed wells."""
    calls = 0

    def fake_fit(ds: Dataset, *, method: str = "lm", **_kwargs: object) -> FitResult:
        nonlocal calls
        assert method in {"huber", "lm"}
        calls += 1
        if ds is datasets["bad"]:
            msg = "too few points"
            raise InsufficientDataError(msg)
        return FitResult(result=_Result(np.zeros(len(ds["1"].y))), dataset=ds)

    def fake_residuals(
        _results: dict[str, FitResult], *_args: object, **_kwargs: object
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
    monkeypatch.setattr("clophfit.prtecan.titration.fit_binding_glob", fake_fit)
    monkeypatch.setattr(
        "clophfit.prtecan.titration.residuals_from_fit_results", fake_residuals
    )
    monkeypatch.setattr(
        "clophfit.prtecan.titration.fit_noise_model_nnls",
        lambda *_args, **_kwargs: ({"1": 1.0}, {"1": 0.2}, {"1": 0.03}),
    )
    monkeypatch.setattr(
        "clophfit.prtecan.titration.compute_plate_slopes", lambda *_args: {}
    )
    monkeypatch.setattr(
        "clophfit.prtecan.titration.fit_ph_slope_noise", lambda *_args: 0.0
    )

    res = titration.fgls_fit_plate(
        datasets, sigma_floor={"1": 1.0}, max_iter=1, tol=1e-9
    )

    assert calls == 2
    assert res.results["good"].result is not None
    assert res.results["bad"].result is None
    assert res.noise_model is not None
    assert res.noise_model["1"].gain == 0.2


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

    screened = noise_calibration.calibrate_noise_robust(df, {"1": 1.0}, p_threshold=0.9)
    # p_threshold above 1.0 keeps every point (nothing screened).
    unscreened = noise_calibration.calibrate_noise_robust(
        df, {"1": 1.0}, p_threshold=2.0
    )

    assert screened["1"].sigma_floor == 1.0
    assert screened["1"].gain < unscreened["1"].gain


def test_calibrate_noise_robust_without_probability_column() -> None:
    """Without p_outlier, all points are used (no screening)."""
    y = np.linspace(50.0, 500.0, 80)
    df = pd.DataFrame({"label": "1", "raw_res": np.full(80, 2.0), "yhat": y})
    model = noise_calibration.calibrate_noise_robust(df, {"1": 1.5})
    assert model["1"].sigma_floor == 1.5
    assert model["1"].gain >= 0.0
    assert model["1"].alpha >= 0.0
