"""Machinery for expanding a grid of fit specifications.

A grid search over fitting options is generic: something describes which knobs
exist, a product is taken over configuration blocks, each combination becomes a
flat specification, and specifications that describe the same model must not
appear twice under different names. Only the *vocabulary* - which knobs a given
project exposes and what it calls them - is project-specific.

This module owns the machinery. The caller supplies the knobs.

The separation matters because of how it failed before. When a project kept the
knob descriptions, the expansion, and the checks in three places that had to
agree, six knobs were configured and silently never reached the model: runs
succeeded, diagnostics looked normal, and trace ids claimed distinctions the
fitted models did not have. Keeping the mechanism here means a project describes
each knob once and gets expansion, signatures and duplicate detection from it.
"""

from __future__ import annotations

import hashlib
import json
import typing
from dataclasses import dataclass

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class Knob:
    """One configurable quantity, from configuration block to model kwarg.

    Parameters
    ----------
    name : str
        Key this knob occupies in an expanded specification.
    block : str
        Name of the configuration block it is read from.
    source_key : str
        Key within that block. Differs from *name* only where the configuration
        spelling and the specification spelling diverge.
    default : typing.Any
        Value used when the block omits the key.
    target : str
        Model-kwarg this knob is passed as, when the pass-through is a plain
        rename with no coercion or conditional construction. Empty means the
        caller places it, which is the case for anything that lands inside a
        config object or needs a type conversion.
    reader : Callable[[dict[str, typing.Any]], typing.Any] | None
        How to read the resulting value back out of the built model kwargs.
        ``None`` marks a knob that is consumed on the way through and has no one
        corresponding kwarg, so it cannot be verified directly.
    """

    name: str
    block: str
    source_key: str = ""
    default: typing.Any = None
    target: str = ""
    reader: Callable[[dict[str, typing.Any]], typing.Any] | None = None

    @property
    def key(self) -> str:
        """The key to read from the source block."""
        return self.source_key or self.name


def direct_kwargs(
    cfg: Mapping[str, typing.Any], knobs: Iterable[Knob]
) -> dict[str, typing.Any]:
    """Return the model kwargs that are plain pass-throughs of a knob.

    Parameters
    ----------
    cfg : Mapping[str, typing.Any]
        An expanded specification.
    knobs : Iterable[Knob]
        Knobs to consider; those without a *target* are skipped.

    Returns
    -------
    dict[str, typing.Any]
        Target name to value.

    Notes
    -----
    Only knobs whose journey to the model is a rename belong here. A knob that
    is coerced, made conditional, or folded into a config object stays with the
    caller: expressing those generically would need a coercion language whose
    mistakes are exactly as silent as the ones this registry exists to prevent.
    """
    return {knob.target: cfg[knob.name] for knob in knobs if knob.target}


def apply_blocks(
    cfg: dict[str, typing.Any],
    knobs: Iterable[Knob],
    blocks: Mapping[str, Mapping[str, typing.Any]],
) -> None:
    """Copy every knob from its source block into *cfg*.

    Parameters
    ----------
    cfg : dict[str, typing.Any]
        Specification being built; mutated in place.
    knobs : Iterable[Knob]
        Knobs to copy.
    blocks : Mapping[str, Mapping[str, typing.Any]]
        Source block per block name.

    Raises
    ------
    KeyError
        If a knob names a block that was not supplied, which would otherwise
        surface much later as a missing value in a fitted model.
    """
    for knob in knobs:
        if knob.block not in blocks:
            msg = f"Knob {knob.name!r} reads block {knob.block!r}, which was not supplied."
            raise KeyError(msg)
        cfg[knob.name] = blocks[knob.block].get(knob.key, knob.default)


def model_signature(
    cfg: Mapping[str, typing.Any],
    knobs: Iterable[Knob],
    extra_fields: Sequence[str] = (),
) -> str:
    """Return a digest of everything in *cfg* that changes the fitted model.

    Parameters
    ----------
    cfg : Mapping[str, typing.Any]
        An expanded specification.
    knobs : Iterable[Knob]
        Knobs whose resolved values determine the model.
    extra_fields : Sequence[str]
        Further specification keys that also determine the model but are not
        knobs, such as a dataset variant or a sampling width.

    Returns
    -------
    str
        Short hex digest over the resolved values.

    Notes
    -----
    Identifiers built from the *names* of configuration blocks distinguish two
    blocks that are named differently whether or not they describe different
    models. This digest is taken from resolved values instead, so specifications
    that agree as models agree here.
    """
    material = {knob.name: repr(cfg.get(knob.name)) for knob in knobs}
    material |= {field: repr(cfg.get(field)) for field in extra_fields}
    return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()[
        :12
    ]


def check_unique_signature(
    seen: dict[str, str], signature: str, identifier: str
) -> None:
    """Record a specification's signature, rejecting a duplicate model.

    Parameters
    ----------
    seen : dict[str, str]
        Signature to identifier, accumulated across one grid; mutated in place.
    signature : str
        Signature of the specification being added.
    identifier : str
        Human-readable identifier, used in the error message.

    Raises
    ------
    ValueError
        If a different identifier already claimed this signature, meaning two
        cells of one grid would fit identical models under different names.
    """
    previous = seen.setdefault(signature, identifier)
    if previous != identifier:
        msg = (
            f"{identifier!r} and {previous!r} resolve to the same model "
            f"(signature {signature}). Two cells of this grid would fit "
            "identical models under different names."
        )
        raise ValueError(msg)


def inert_knobs(
    rows: Sequence[Mapping[str, typing.Any]], knobs: Sequence[Knob]
) -> dict[str, str]:
    """Return the knobs that never arrived at the model or never varied.

    Parameters
    ----------
    rows : Sequence[Mapping[str, typing.Any]]
        One mapping of knob name to arrived value per specification, with
        ``"<MISSING>"`` where a knob produced no kwarg.
    knobs : Sequence[Knob]
        Knobs the grid is supposed to vary.

    Returns
    -------
    dict[str, str]
        Knob name to the reason it is inert; empty when every knob varies.

    Notes
    -----
    The constancy check applies only to a grid of more than one specification. A
    single-specification grid pins its knobs by construction, so a constant value
    there is the intended configuration rather than evidence a knob went missing.
    """
    bad: dict[str, str] = {}
    for knob in knobs:
        values = [row[knob.name] for row in rows]
        if all(v == "<MISSING>" for v in values):
            bad[knob.name] = "never reaches the model"
        elif any(v == "<MISSING>" for v in values):
            bad[knob.name] = "reaches the model for only some specs"
        elif len(values) > 1 and len({repr(v) for v in values}) == 1:
            bad[knob.name] = f"identical in all {len(values)} specs ({values[0]!r})"
    return bad
