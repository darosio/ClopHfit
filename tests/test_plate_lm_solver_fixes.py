"""Three plate_lm defects found on real plates, pinned.

1. A robust loss never met scipy's termination tests because K (~7) and the
   plateaus (~1000) were unscaled; a 200-evaluation cap hid it, and on L5b K was
   still moving after 5000. Robust solves are now scaled by the Jacobian, the
   cap is a safety net, and reaching it is reported apart from ``success``.
2. ``--plate-noise calibrated`` meant calibrated weights without a screen and
   plain weights after one. The screen's ruler and the fit's weights are now
   separate switches, with defaults that keep the old split.
3. With the floor free, the calibration drove it to zero and cycled
   (0 -> 126 -> 27 -> 15 -> 0.3 -> 0 on L6b) without converging, and a solve with
   near-infinite weights took 8713 evaluations. A term's step now halves when
   its update reverses, and linear solves have a safety cap too.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from clophfit.fitting import plate_lm
from clophfit.fitting.data_structures import NoiseModelParams
from clophfit.fitting.plate_lm import (
    fit_plate_lm,
    fit_plate_lm_screened,
)
from clophfit.prtecan import export
from clophfit.prtecan.titration import TecanConfig
from tests.test_plate_lm import make_plate

if TYPE_CHECKING:
    from pathlib import Path


def test_robust_solves_are_scaled_and_linear_ones_are_not() -> None:
    """Scaling is what lets huber converge; linear keeps scipy's default path."""
    robust = plate_lm._solver_kwargs("huber")  # ruff: ignore[private-member-access] - the seam under test
    linear = plate_lm._solver_kwargs("linear")  # ruff: ignore[private-member-access]
    assert robust["x_scale"] == "jac"
    assert robust["xtol"] is None  # the step test fired while K still moved
    assert "x_scale" not in linear
    assert "xtol" not in linear
    assert robust["max_nfev"] >= 1000
    assert linear["max_nfev"] >= 1000  # a safety net, not a default budget


def test_budget_reached_is_reported_apart_from_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A solve stopped by its cap must say so; a converged one must not."""
    datasets = make_plate({"A01": 6.8, "A02": 7.4}, {"1": 3.0, "2": 2.0})
    free = fit_plate_lm(datasets, groups={}, loss="huber")
    assert not free.budget_reached

    monkeypatch.setattr(plate_lm, "_ROBUST_MAX_NFEV", 1)
    capped = fit_plate_lm(datasets, groups={}, loss="huber")
    assert capped.budget_reached


def _model(floor: float, gain: float = 0.0) -> dict[str, NoiseModelParams]:
    return {"1": NoiseModelParams(sigma_floor=floor, gain=gain, alpha=0.0)}


def test_a_monotone_calibration_is_left_exactly_as_proposed() -> None:
    """No reversal, no damping: the proposal is taken bit for bit."""
    state = plate_lm._DampingState()  # ruff: ignore[private-member-access]
    current: Any = _model(10.0, 1.0)
    for floor in (8.0, 6.5, 6.0):
        proposed = _model(floor, 1.0)
        current = plate_lm._damped_update(current, proposed, state)  # ruff: ignore[private-member-access]
        assert current["1"].sigma_floor == floor


def test_a_reversing_calibration_is_damped() -> None:
    """Each reversal halves the step, so a two-point cycle converges."""
    state = plate_lm._DampingState()  # ruff: ignore[private-member-access]
    current: Any = _model(10.0)
    floors = []
    for proposed in (0.0, 20.0, 0.0, 20.0):
        current = plate_lm._damped_update(current, _model(proposed), state)  # ruff: ignore[private-member-access]
        floors.append(current["1"].sigma_floor)
    # 10 -> 0 (full step), 0 -> +10 (half of 20), 10 -> 7.5 (quarter of -10),
    # 7.5 -> 9.0625 (eighth of +12.5): the swing shrinks instead of repeating.
    assert floors == pytest.approx([0.0, 10.0, 7.5, 9.0625])


def _noise(datasets: dict[str, Any]) -> dict[str, NoiseModelParams]:
    return {
        lbl: NoiseModelParams(sigma_floor=2.0) for lbl in next(iter(datasets.values()))
    }


def test_screened_refit_follows_its_own_switch() -> None:
    """Plain refit by default (the old behaviour); calibrated only when asked."""
    ks = {f"A{i:02d}": 6.5 + 0.1 * i for i in range(1, 9)}
    datasets = make_plate(ks, {"1": 4.0, "2": 3.0})
    noise = _noise(datasets)
    plain = fit_plate_lm_screened(datasets, groups={}, noise_model=noise, threshold=0.5)
    calib = fit_plate_lm_screened(
        datasets, groups={}, noise_model=noise, threshold=0.5, calibrate_refit=True
    )
    assert plain.noise == {}, "the default refit must stay plain"
    assert set(calib.noise) == set(noise), "calibrate_refit never reached the refit"


def test_screen_ruler_can_be_the_weights_as_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """calibrate_screen=False must stop the screening pass from calibrating."""
    calls: list[bool] = []
    real = plate_lm.fit_plate_lm

    def spy(*args: Any, **kwargs: Any) -> Any:  # ruff: ignore[any-type] - a pass-through spy
        calls.append(bool(kwargs.get("calibrate_noise")))
        return real(*args, **kwargs)

    monkeypatch.setattr(plate_lm, "fit_plate_lm", spy)
    datasets = make_plate({"A01": 6.8, "A02": 7.4}, {"1": 3.0, "2": 2.0})
    noise = _noise(datasets)
    fit_plate_lm_screened(
        datasets, groups={}, noise_model=noise, calibrate_screen=False
    )
    assert calls[0] is False, "the screening pass calibrated although told not to"
    calls.clear()
    fit_plate_lm_screened(datasets, groups={}, noise_model=noise)
    assert calls[0] is True, "the default ruler is the calibrated one"


def test_export_passes_the_two_switches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """--plate-noise and --plate-screen-noise must both reach the screened fit."""
    seen: dict[str, Any] = {}
    real = fit_plate_lm_screened

    def spy(*args: Any, **kwargs: Any) -> Any:  # ruff: ignore[any-type] - a pass-through spy
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(export, "fit_plate_lm_screened", spy)
    datasets = make_plate({"A01": 6.8, "A02": 7.4, "A03": 7.0}, {"1": 3.0, "2": 2.0})
    tit = SimpleNamespace(
        scheme=SimpleNamespace(names={}), x_err=None, data={}, params=None
    )
    export.export_plate_fit(
        tit,  # type: ignore[arg-type]  # SimpleNamespace test double
        datasets,
        tmp_path,
        "lm",
        png=False,
        screen_z=3.0,
        calibrate_noise=True,
        calibrate_screen=False,
    )
    assert seen["calibrate_refit"] is True
    assert seen["calibrate_screen"] is False


@pytest.mark.parametrize(
    ("flag", "expected"), [((), "calibrated"), (("fixed",), "fixed")]
)
def test_tecan_config_keeps_positional_order(
    flag: tuple[str, ...], expected: str
) -> None:
    """The CLI builds TecanConfig positionally; the new field must come last."""
    # The positional prefix __main__ passes, up to and including plate_noise.
    cli_args: tuple[Any, ...] = (
        None,
        False,
        None,
        "",
        True,
        False,
        True,
        False,
        None,
        None,
        "calibrated",
    )
    cfg = TecanConfig(*cli_args, *flag)
    assert cfg.plate_noise == "calibrated"
    assert cfg.plate_screen_noise == expected
