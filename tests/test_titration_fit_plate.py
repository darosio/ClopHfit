"""Tests for Titration.fit_plate dispatch and dataset construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from clophfit.fitting.data_structures import DataArray, Dataset, FitResult
from clophfit.fitting.errors import InsufficientDataError
from clophfit.prtecan.titration import Titration, TitrationResults

data_tests = Path(__file__).parent / "Tecan"


@dataclass
class _Result:
    residual: np.ndarray
    success: bool = True


@pytest.fixture(scope="module")
def tit() -> Titration:
    """Build a small pH titration; the fits themselves are monkeypatched away."""
    return Titration.fromlistfile(data_tests / "L1" / "list.pH.csv", is_ph=True)


def _dataset() -> Dataset:
    x = np.array([6.0, 7.0, 8.0])
    y = np.array([1.0, 2.0, 3.0])
    return Dataset({"1": DataArray(x, y, y_errc=np.ones_like(y))}, is_ph=True)


@pytest.mark.parametrize(
    ("method", "target"),
    [("", "lm"), ("huber", "huber"), ("odr", "odr"), ("mcmc", "mcmc")],
)
def test_fit_plate_routes_methods_and_preserves_well_keys(
    monkeypatch: pytest.MonkeyPatch, tit: Titration, method: str, target: str
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

    monkeypatch.setattr("clophfit.prtecan.titration.fit_binding_glob", fake_glob)
    monkeypatch.setattr("clophfit.prtecan.titration.fit_binding_odr", fake_odr)
    monkeypatch.setattr("clophfit.prtecan.titration.fit_binding_pymc", fake_mcmc)

    results = tit.fit_plate({"A01": _dataset(), "A02": _dataset()}, method=method)

    assert isinstance(results, TitrationResults)
    assert set(results.results) == {"A01", "A02"}
    assert all(fr.result is not None for fr in results.results.values())
    assert calls == [(target if target in {"odr", "mcmc"} else "glob", target)] * 2


@pytest.mark.parametrize("method", ["lm", "odr", "mcmc"])
def test_fit_plate_turns_insufficient_data_into_empty_result(
    monkeypatch: pytest.MonkeyPatch, tit: Titration, method: str
) -> None:
    """One well's insufficient-data failure must not abort the other well's fit."""

    def backend(ds: Dataset, **_kwargs: object) -> FitResult[Any]:
        if len(ds["1"].y) < len(_dataset()["1"].y):
            msg = "too few points"
            raise InsufficientDataError(msg)
        return FitResult(result=_Result(np.zeros(len(ds["1"].y))), dataset=ds)

    monkeypatch.setattr("clophfit.prtecan.titration.fit_binding_glob", backend)
    monkeypatch.setattr("clophfit.prtecan.titration.fit_binding_odr", backend)
    monkeypatch.setattr("clophfit.prtecan.titration.fit_binding_pymc", backend)

    failing_ds = Dataset(
        {"1": DataArray(np.array([6.0]), np.array([1.0]), y_errc=np.array([1.0]))},
        is_ph=True,
    )
    results = tit.fit_plate({"A01": failing_ds, "A02": _dataset()}, method=method)

    assert results["A01"].result is None
    assert results["A02"].result is not None


def test_fit_plate_carries_plate_metadata(
    monkeypatch: pytest.MonkeyPatch, tit: Titration
) -> None:
    """The returned container carries this titration's scheme and fit_keys."""

    def fake_glob(ds: Dataset, **_kwargs: object) -> FitResult[Any]:
        return FitResult(result=_Result(np.zeros(len(ds["1"].y))), dataset=ds)

    monkeypatch.setattr("clophfit.prtecan.titration.fit_binding_glob", fake_glob)

    results = tit.fit_plate({"A01": _dataset()})

    assert results.scheme is tit.scheme
    assert results.fit_keys == tit.fit_keys


def test_fit_plate_builds_datasets_when_none_given(
    monkeypatch: pytest.MonkeyPatch, tit: Titration
) -> None:
    """Omitting datasets delegates construction to create_dataset_dict."""
    seen: list[str | None] = []
    original = tit.create_dataset_dict

    def spy(label: str | None = None) -> dict[str, Dataset]:
        seen.append(label)
        return original(label)

    monkeypatch.setattr(tit, "create_dataset_dict", spy)
    monkeypatch.setattr(
        "clophfit.prtecan.titration.fit_binding_glob",
        lambda ds, **_kw: FitResult(
            result=_Result(np.zeros(len(next(iter(ds.values())).y))), dataset=ds
        ),
    )

    tit.fit_plate(label="1")

    assert seen == ["1"]


def test_fit_plate_rejects_datasets_and_label_together(tit: Titration) -> None:
    """Datasets and label are alternative spellings of the same input."""
    with pytest.raises(ValueError, match="not both"):
        tit.fit_plate({"A01": _dataset()}, label="1")
