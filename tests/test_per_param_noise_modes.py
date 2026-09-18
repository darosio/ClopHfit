"""Each noise term needs its own mode, not one switch for all three.

``--noise-mode`` applied to every supplied hint at once, so pinning alpha at
zero also pinned the floor, and a cell meant to test "gain only" instead tested
"gain only, with sigma unable to rescale at all". Those cells came back with
calibration ratios of 1.9 to 3.0 against 1.07 for the free-floor ones, which is
the floor being unable to move rather than anything about gain.

The four combinations that actually separate the terms are floor fixed or
centred, crossed with alpha pinned at zero or gain pinned at zero.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from clophfit.prtecan.export import (
    _structured_noise,  # ruff: ignore[import-private-name] - the seam under test is private
)


def _titration(gain: tuple[float, ...], alpha: tuple[float, ...]) -> SimpleNamespace:
    """Build a stand-in carrying just what the noise builder reads."""
    return SimpleNamespace(
        data={"1": {}, "2": {}},
        params=SimpleNamespace(noise_gain=gain, noise_alpha=alpha),
        sigma_floor={"1": 3.59, "2": 0.42},
    )


def test_one_mode_still_applies_to_every_term() -> None:
    """The existing behaviour must survive: one mode sets all three."""
    cfg = _structured_noise(_titration((0.0, 0.41), (0.07, 0.0)), noise_mode="fixed")  # type: ignore[arg-type]
    assert (cfg.floor_mode, cfg.gain_mode, cfg.alpha_mode) == (
        "fixed",
        "fixed",
        "fixed",
    )


def test_the_floor_can_be_centred_while_a_term_is_pinned() -> None:
    """Case (c): floor centred, alpha pinned at zero, gain free to move."""
    cfg = _structured_noise(
        _titration((), (0.0, 0.0)),  # type: ignore[arg-type]
        noise_mode="fixed",
        floor_mode="centered",
    )
    assert cfg.floor_mode == "centered"
    assert cfg.alpha_mode == "fixed"
    assert cfg.gain_mode == "free"


def test_the_floor_can_be_pinned_while_a_term_is_centred() -> None:
    """Case (a): floor fixed, alpha pinned, gain centred on its hint."""
    cfg = _structured_noise(
        _titration((0.0, 0.41), (0.0, 0.0)),  # type: ignore[arg-type]
        noise_mode="fixed",
        gain_mode="centered",
    )
    assert cfg.floor_mode == "fixed"
    assert cfg.gain_mode == "centered"
    assert cfg.alpha_mode == "fixed"


@pytest.mark.parametrize(
    ("floor_mode", "gain_mode", "alpha_mode"),
    [
        ("fixed", "centered", "fixed"),  # a: floor fixed, alpha=0, gain centred
        ("fixed", "fixed", "centered"),  # b: floor fixed, gain=0, alpha centred
        ("centered", "centered", "fixed"),  # c: floor centred, alpha=0
        ("centered", "fixed", "centered"),  # d: floor centred, gain=0
    ],
)
def test_the_four_combinations_are_expressible(
    floor_mode: str, gain_mode: str, alpha_mode: str
) -> None:
    """Every cell the comparison needs can be asked for."""
    cfg = _structured_noise(
        _titration((0.0, 0.41), (0.07, 0.0)),  # type: ignore[arg-type]
        noise_mode="centered",
        floor_mode=floor_mode,  # type: ignore[arg-type]
        gain_mode=gain_mode,  # type: ignore[arg-type]
        alpha_mode=alpha_mode,  # type: ignore[arg-type]
    )
    assert (cfg.floor_mode, cfg.gain_mode, cfg.alpha_mode) == (
        floor_mode,
        gain_mode,
        alpha_mode,
    )


def test_an_unsupplied_term_is_still_free_whatever_the_mode() -> None:
    """A term with no hint has nothing to centre on or pin to."""
    cfg = _structured_noise(
        _titration((), ()),  # type: ignore[arg-type]
        noise_mode="fixed",
        gain_mode="centered",
        alpha_mode="fixed",
    )
    assert cfg.gain_mode == "free"
    assert cfg.alpha_mode == "free"
    assert np.isclose(cfg.floor["1"], 3.59)  # type: ignore[index]
