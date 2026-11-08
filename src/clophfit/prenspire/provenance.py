"""Provenance records: how a result was made, written beside it as JSON.

The record names the library version, the invocation and - for a posterior -
the sampler stack, so a folder of results traces back to the command that
made it. Nothing here is time-dependent.
"""

from __future__ import annotations

import importlib.metadata
import json
from typing import TYPE_CHECKING

from clophfit._config import __version__

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path


def is_bayes_run(options: Mapping[str, object] | None) -> bool:
    """Whether this invocation sampled a posterior, i.e. needs its stack recorded."""
    method = str((options or {}).get("method", "")).lower()
    return "bayes" in method


def sampler_stack() -> dict[str, str]:
    """Versions of the packages that determine a posterior, when they are installed."""
    out: dict[str, str] = {}
    for name in ("pymc", "pytensor", "arviz", "nutpie", "h5netcdf"):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return out


def write_provenance(
    path: Path,
    *,
    inputs: Mapping[str, str],
    options: Mapping[str, object] | None,
    outputs: Sequence[str],
) -> Path:
    """Record how a result was produced, beside the result, as JSON.

    The table's columns already carry the options that move its numbers; this
    carries the rest - the library version and the whole invocation - so a folder
    of results can be traced back to the command that made it. Sorted keys, no
    timestamp: the same run writes the same bytes.
    """
    payload: dict[str, object] = {
        "clophfit_version": __version__,
        "inputs": dict(inputs),
        "options": dict(options or {}),
        "outputs": list(outputs),
    }
    if is_bayes_run(options):
        # a posterior is only reproducible against the sampler stack that made it
        payload["environment"] = sampler_stack()
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path
