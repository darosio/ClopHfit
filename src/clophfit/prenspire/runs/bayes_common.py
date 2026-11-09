"""Helpers shared by the Bayesian runs: x-sigma, isosbestic choice, posterior screens."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from clophfit.prenspire import (
    EnspireFile,
    bands as bands_module,
)
from clophfit.prenspire.spectral import (
    isosbestic_by_group,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pandas as pd


from clophfit.prenspire.runs.common import EnspireInputError, EnspireMessage

#: Screening floors for a posterior, not convergence criteria: they catch a
#: posterior that is visibly unhealthy, and passing them proves nothing.
_RHAT_SCREEN = 1.01
_ESS_BULK_SCREEN = 400.0
_BFMI_SCREEN = 0.3
#: share of trajectories hitting the depth cap before the geometry is suspect
_TREEDEPTH_SCREEN = 0.2


# Prior SD (pH) floor for --x-sigma: strictly positive, and small enough to
# keep the posterior honest instead of smearing every point onto its plateau.
X_SIGMA_MIN_PH = 0.01


def _diag_number(diag: Mapping[str, object], key: str) -> float | None:
    """Read one screening number, or ``None`` when the sampler did not record it."""
    value = diag.get(key)
    if value is None:
        return None
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def posterior_screen(diag: Mapping[str, object]) -> str:
    """Say whether a posterior passed the sampler's screening floors.

    The floors are screening values, not convergence criteria: no divergences,
    rank r-hat below 1.01, bulk ESS above 400, energy BFMI above 0.3. Passing them
    still does not establish that the model is identified or the priors sensible,
    and the line stays silent when all four pass.
    """
    problems: list[str] = []
    divergences = _diag_number(diag, "n_divergences")
    if divergences:
        problems.append(f"{divergences:.0f} divergences")
    rhat = _diag_number(diag, "rhat_max")
    if rhat is not None and rhat > _RHAT_SCREEN:
        problems.append(f"r-hat {rhat:.3f}")
    ess = _diag_number(diag, "ess_bulk_min")
    if ess is not None and ess < _ESS_BULK_SCREEN:
        problems.append(f"bulk ESS {ess:.0f}")
    bfmi = _diag_number(diag, "bfmi")
    if bfmi is not None and bfmi < _BFMI_SCREEN:
        problems.append(f"BFMI {bfmi:.2f}")
    depth = _diag_number(diag, "reached_max_treedepth_frac")
    if depth is not None and depth > _TREEDEPTH_SCREEN:
        problems.append(f"{100 * depth:.0f}% at max tree depth")
    if not problems:
        return ""
    return "  ! screen: " + ", ".join(problems) + " (screening only, not proof)"


def posterior_columns(diag: Mapping[str, object]) -> dict[str, object]:
    """Collect the screening numbers as table columns beside the estimate."""
    return {
        "n_divergences": diag.get("n_divergences", np.nan),
        "rhat_max": diag.get("rhat_max", np.nan),
        "ess_bulk_min": diag.get("ess_bulk_min", np.nan),
        "ess_tail_min": diag.get("ess_tail_min", np.nan),
        "bfmi": diag.get("bfmi", np.nan),
        "max_treedepth": diag.get("max_treedepth", np.nan),
        "n_reached_max_treedepth": diag.get("n_reached_max_treedepth", np.nan),
        "reached_max_treedepth_frac": diag.get("reached_max_treedepth_frac", np.nan),
        "sampler": diag.get("sampler", "unknown"),
    }


def shared_isosbestic(
    ef: EnspireFile,
    note: pd.DataFrame,
    isosbestic: float | str | None,
    *,
    buffer: bool,
) -> dict[tuple[str, float], float | None] | None:
    """Resolve ``auto-scale`` into one crossing per (sample, temperature).

    Parameters
    ----------
    ef : EnspireFile
        The parsed EnSpire export.
    note : pd.DataFrame
        The normalised note.
    isosbestic : float | str | None
        The option as parsed: a wavelength, ``"auto"``, ``"auto-scale"`` or
        ``None``.
    buffer : bool
        Subtract the buffer wells before locating the crossings.

    Returns
    -------
    dict[tuple[str, float], float | None] | None
        The per-sample crossings, or ``None`` when the caller should let each
        group locate its own.
    """
    if isinstance(isosbestic, str) and isosbestic == "auto-scale":
        return isosbestic_by_group(ef, note, buffer=buffer)
    return None


def lam_for(
    isosbestic: float | str | None,
    lam_map: dict[tuple[str, float], float | None] | None,
    name: object,
    temp: object,
) -> float | str | None:
    """Pick the crossing to hand to one (sample, temperature) group."""
    if lam_map is None:
        return isosbestic
    return lam_map.get((str(name), float(str(temp))))


def resolve_x_sigma(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
    x_sigma: str | None,
    is_ph: bool,  # ruff: ignore[boolean-type-hint-positional-argument]
    one: object,
    name: str,
    t: str,
    messages: list[EnspireMessage],
) -> float | None:
    """Resolve ``--x-sigma`` for one (sample, temperature) bayes fit.

    ``None`` leaves the axis fixed at the note values. ``auto`` pools the
    note's replicate-row spread per index column (rows A-C and D-F are the
    same nominal targets at their temperature) into the prior sigma - on the
    pH axis for a pH titration, on the Cl column (mM) for a ligand one; a
    number passes through after bounds checking, in the axis's units.
    """
    if x_sigma is None:
        return None
    one = cast("pd.DataFrame", one)
    xcol = "pH" if is_ph else "Cl"
    if x_sigma.strip().lower() == "auto":
        sigma = bands_module.estimate_x_error_sigma(one["Well"], one[xcol])
        if sigma is None:
            messages.append(
                EnspireMessage(
                    "warning",
                    f"warning: {name} at {t}: fewer than two index columns with two "
                    "or more replicates; using a fixed axis (note values)",
                )
            )
            return None
        return sigma
    try:
        sigma = float(x_sigma)
    except ValueError:
        msg = "--x-sigma must be 'auto' or a number, e.g. 0.15"
        raise EnspireInputError(msg) from None
    if sigma < X_SIGMA_MIN_PH:
        units = "pH" if is_ph else "mM"
        msg = f"--x-sigma must be >= {X_SIGMA_MIN_PH} {units} (or 'auto')"
        raise EnspireInputError(msg)
    return sigma
