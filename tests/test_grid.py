"""Tests for the grid-expansion machinery."""

from __future__ import annotations

import pytest

from clophfit.fitting.grid import (
    Knob,
    apply_blocks,
    check_unique_signature,
    direct_kwargs,
    inert_knobs,
    model_signature,
)


def test_apply_blocks_uses_defaults_and_source_keys() -> None:
    """A knob reads its own key, falling back to its default."""
    knobs = [
        Knob("noise_model_key", "noise", "model", "none"),
        Knob("learn_ye_mags", "noise", default=False),
    ]
    cfg: dict[str, object] = {}
    apply_blocks(cfg, knobs, {"noise": {"model": "BgRel"}})
    assert cfg == {"noise_model_key": "BgRel", "learn_ye_mags": False}


def test_apply_blocks_rejects_a_missing_block() -> None:
    """A knob naming an unsupplied block fails here, not inside a fit."""
    with pytest.raises(KeyError, match="was not supplied"):
        apply_blocks({}, [Knob("k", "absent")], {"noise": {}})


def test_model_signature_ignores_names_and_tracks_values() -> None:
    """Two specs differing only by block name share a signature."""
    knobs = [Knob("a", "x"), Knob("b", "x")]
    left = {"a": 1, "b": 2, "noise_name": "cheerful"}
    right = {"a": 1, "b": 2, "noise_name": "morose"}
    assert model_signature(left, knobs) == model_signature(right, knobs)
    assert model_signature(left, knobs) != model_signature({"a": 1, "b": 3}, knobs)


def test_model_signature_includes_extra_fields() -> None:
    """Non-knob fields that determine the model still separate specs."""
    knobs = [Knob("a", "x")]
    assert model_signature({"a": 1, "n_sd": 4}, knobs, ("n_sd",)) != model_signature(
        {"a": 1, "n_sd": 3}, knobs, ("n_sd",)
    )


def test_check_unique_signature_rejects_a_duplicate_model() -> None:
    """Two identifiers sharing a signature is an error naming both."""
    seen: dict[str, str] = {}
    check_unique_signature(seen, "abc", "cell_one")
    check_unique_signature(seen, "abc", "cell_one")  # idempotent
    with pytest.raises(ValueError, match="resolve to the same model"):
        check_unique_signature(seen, "abc", "cell_two")


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        ([{"k": "<MISSING>"}, {"k": "<MISSING>"}], "never reaches the model"),
        ([{"k": 1}, {"k": "<MISSING>"}], "reaches the model for only some specs"),
        ([{"k": 1}, {"k": 1}], "identical in all 2 specs (1)"),
    ],
)
def test_inert_knobs_names_the_failure(
    rows: list[dict[str, object]], expected: str
) -> None:
    """Each way a knob can be inert is reported distinctly."""
    assert inert_knobs(rows, [Knob("k", "x")])["k"] == expected


def test_inert_knobs_allows_a_single_spec_grid() -> None:
    """One spec pins its knobs by construction, so constancy is not a fault."""
    assert inert_knobs([{"k": 1}], [Knob("k", "x")]) == {}


def test_inert_knobs_passes_when_a_knob_varies() -> None:
    """A knob taking different values across specs is doing its job."""
    assert inert_knobs([{"k": 1}, {"k": 2}], [Knob("k", "x")]) == {}


def test_direct_kwargs_passes_only_targeted_knobs() -> None:
    """A knob with a target is renamed through; one without is left alone."""
    knobs = [
        Knob("n_sd", "run", target="n_sd"),
        Knob("acid_drop", "x", target="x_start_between_sigma"),
        Knob("learn_ye_mags", "noise"),  # folded into a config object elsewhere
    ]
    cfg = {"n_sd": 4, "acid_drop": 0.015, "learn_ye_mags": True}
    assert direct_kwargs(cfg, knobs) == {"n_sd": 4, "x_start_between_sigma": 0.015}
