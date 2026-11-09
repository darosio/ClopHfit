"""The latent titrant axis (x_true): pooling, pipetting steps, and column ties."""

from __future__ import annotations

import re
import typing
from typing import Literal

import numpy as np
import pymc as pm  # type: ignore[import-untyped]
from pytensor.tensor import as_tensor_variable
from scipy.optimize import isotonic_regression

if typing.TYPE_CHECKING:
    from collections.abc import Sequence

    from clophfit.clophfit_types import ArrayF


_X_TRUE_INDEX_RE = re.compile(r"^x_true\[(\d+)\]$")


# Number of pH-monitor wells the step-0 SD is measured over; the true starting
# pH is anchored by the standard error of their mean (strong but not fixed).
_N_PH_REPLICATE_WELLS = 3
# Minimum per-addition pipetting variance, as a fraction of the average
# per-addition variance, so a noisy flat 3-well SD cannot pin a step to ~zero:
# every 2 uL delivery carries a real, nonzero volume error.
_MIN_PIPETTING_STEP_FRAC = 0.25
# Default between-well scale for the per-well starting-x (x_start_well). Small and
# nonzero so multi-well fits give each well its own x0, tightly anchored on the
# shared plate x_start prior. Set to 0 for a single shared x_start.
_DEFAULT_X_START_BETWEEN_SIGMA = 0.05


def _pipetting_step_sigmas(x_errc_scaled: ArrayF) -> tuple[float, ArrayF]:
    """Split a cumulative pH-SD profile into per-addition pipetting sigmas.

    The 3-well SD at step ``N`` is cumulative:
    ``x_errc[N]**2 = read_noise**2 + sum_{k<=N} pipetting_var[k]``. The recurring
    read noise (the step-0 SD) cancels in consecutive differences, so per-addition
    variance is the increment of the cumulative variance.

    Accumulated pipetting variance is physically non-decreasing, but the 3-well
    SDs are noisy and routinely violate that (an atypically large early value, or
    dips). We therefore fit the least-squares monotone (isotonic) cumulative
    variance rather than a running maximum: ``np.maximum.accumulate`` would
    *freeze* a noisy early spike and propagate it forward (inflating the
    ``x_start`` anchor and, via the K/pH degeneracy, K), whereas isotonic pulls
    such a spike down to be consistent with the smaller later values. A positive
    floor keeps every addition nonzero, so a flat stretch cannot pin a step.

    Parameters
    ----------
    x_errc_scaled : ArrayF
        Per-step cumulative pH SD, already scaled by ``n_xerr`` and floored > 0.

    Returns
    -------
    tuple[float, ArrayF]
        ``(x_start_sigma, step_sigmas)``: ``x_start_sigma`` anchors the starting
        pH (standard error of the step-0 mean, so the recurring read noise does
        not enter the walk); ``step_sigmas[k]`` is addition ``k``'s pipetting
        sigma (length ``len(x_errc_scaled) - 1``).
    """
    cum_var = isotonic_regression(x_errc_scaled**2).x
    x_start_sigma = float(np.sqrt(cum_var[0]) / np.sqrt(_N_PH_REPLICATE_WELLS))
    if len(cum_var) <= 1:
        return x_start_sigma, np.empty(0, dtype=float)
    raw_step_var = np.diff(cum_var)
    total_pipetting_var = max(float(cum_var[-1] - cum_var[0]), 0.0)
    floor_var = _MIN_PIPETTING_STEP_FRAC * total_pipetting_var / (len(cum_var) - 1)
    step_var = np.maximum(raw_step_var, max(floor_var, 1e-12))
    return x_start_sigma, np.sqrt(step_var)


def _pipetting_walk_params(
    xc: ArrayF,
    x_errc: ArrayF,
    n_xerr: float,
    *,
    min_x_step: float,
    lower_nsd: float = 2.5,
) -> tuple[float, float, ArrayF, ArrayF, ArrayF]:
    """Derive the shared numpy priors for the pipetting random walk.

    Single source of truth for the de-noised per-addition step derivation used
    by :func:`create_x_true` (single latent ``x_true``) and by the ``per_well``
    x-error model in :func:`fit_binding_pymc_multi`. Each caller builds its own
    RVs from these arrays with the appropriate shape/dims.

    Parameters
    ----------
    xc : ArrayF
        Nominal x (pH) values, monotone in either direction.
    x_errc : ArrayF
        Per-step cumulative x SD (from the pH-monitor wells).
    n_xerr : float
        Scaling factor applied to ``x_errc``.
    min_x_step : float
        Minimum inferred change in x between consecutive steps.
    lower_nsd : float
        Number of step sigmas below the nominal step for the truncation lower
        bound, before flooring at ``min_x_step``.

    Returns
    -------
    tuple[float, float, ArrayF, ArrayF, ArrayF]
        ``(direction, x_start_sigma, step_nominal, step_sigmas, min_steps)``.
        ``direction`` is ``+1`` for increasing x, ``-1`` for decreasing;
        ``x_start_sigma`` anchors the (shared) starting x; ``step_nominal`` are
        the direction-signed nominal steps (positive); ``step_sigmas`` are the
        de-noised per-addition sigmas; ``min_steps`` is the order-preserving
        lower bound for each step.
    """
    x_errc_scaled = np.maximum(x_errc * n_xerr, 1e-6)
    x_start_sigma, step_sigmas = _pipetting_step_sigmas(x_errc_scaled)
    direction = 1.0 if np.all(np.diff(xc) >= 0.0) else -1.0
    step_nominal = direction * np.diff(xc)
    min_steps = np.maximum(step_nominal - lower_nsd * step_sigmas, min_x_step)
    return direction, x_start_sigma, step_nominal, step_sigmas, min_steps


def _build_multi_x_start(
    xc: ArrayF,
    x_start_sigma: float,
    x_start_between_sigma: float,
    *,
    learn_between: bool = False,
) -> typing.Any:  # ruff: ignore[any-type]
    """Build the starting-x anchor for the multi-well x-error models.

    The plate shares one buffer and one pre-addition reading, so the starting x
    (pH) is common: a single ``x_start`` RV anchored at the de-noised standard
    error of the step-0 mean (``x_start_sigma``). When
    ``x_start_between_sigma > 0`` an opt-in per-well term
    ``x_start_well ~ Normal(x_start, x_start_between_sigma)`` is added, letting
    each well's x0 wobble around the shared plate value — a common-mode global
    anchor plus independent per-well jitter. The between-well scale is fixed by
    default, not sampled, because sampling it centred makes a funnel.

    ``learn_between`` estimates it instead, non-centred
    (``x_start_well = x_start + x_start_between * z``, ``z ~ Normal(0, 1)``),
    which has no funnel: the supplied *x_start_between_sigma* becomes the
    HalfNormal prior scale rather than the value. A pinned scale asserts how far
    apart the wells' pH axes sit, and that distance shifts K well for well, so
    it is worth letting the plate say.

    Parameters
    ----------
    xc : ArrayF
        Nominal x (pH) values; ``xc[0]`` centres the shared anchor.
    x_start_sigma : float
        De-noised prior sigma for the shared ``x_start`` anchor (standard error
        of the step-0 mean).
    x_start_between_sigma : float
        Between-well scale: the fixed value, or with *learn_between* the
        HalfNormal prior scale for it. ``<= 0`` returns the shared scalar
        anchor; ``> 0`` adds the per-well ``x_start_well`` term.
    learn_between : bool
        Estimate the between-well scale (reported as ``x_start_between``)
        instead of pinning it.

    Returns
    -------
    typing.Any
        A scalar ``x_start`` tensor (shared anchor) when
        ``x_start_between_sigma <= 0``, otherwise a ``dims="well"`` vector
        centred on the shared anchor. Both broadcast against the
        ``(step, well)`` offset/drop tensors.
    """
    x_start = pm.Normal("x_start", mu=xc[0], sigma=max(x_start_sigma, 1e-6))
    if x_start_between_sigma <= 0:
        return x_start
    if learn_between:
        between = pm.HalfNormal("x_start_between", sigma=x_start_between_sigma)
        z_well = pm.Normal("x_start_well_z", mu=0.0, sigma=1.0, dims="well")
        return pm.Deterministic("x_start_well", x_start + between * z_well, dims="well")
    return pm.Normal(
        "x_start_well",
        mu=x_start,
        sigma=x_start_between_sigma,
        dims="well",
    )


XPool = Literal["free", "shared", "partial"]


def well_groups(wells: Sequence[str]) -> np.ndarray:
    """Replicate-group index per well, from the trailing column number of its label.

    EnSpire notes lay one temperature out as replicate rows of one nominal target
    series, so ``A04``/``B04``/``C04`` are three wells of the *same* solution: the
    recorded titrant differs between them only by the axis error. Grouping on the
    column is therefore the replicate unit, and it is the same key the ``column``
    QC strata use.

    Parameters
    ----------
    wells : Sequence[str]
        Well labels, e.g. ``["A04", "B04", "A05"]``.

    Returns
    -------
    np.ndarray
        Group index per well, in ``0..n_groups-1``, ordered by first appearance.

    Raises
    ------
    ValueError
        If a label carries no trailing column number, so a silent fallback cannot
        pretend every well is its own replicate when it is not.
    """
    index: dict[str, int] = {}
    out = np.empty(len(wells), dtype=int)
    for i, well in enumerate(wells):
        found = re.match(r"^[A-Za-z]+(\d+)$", str(well).strip())
        if found is None:
            msg = (
                f"well {well!r} has no trailing column number, so the replicate "
                "groups cannot be read from it; group the wells explicitly or use "
                "the free axis"
            )
            raise ValueError(msg)
        column = found.group(1)
        out[i] = index.setdefault(column, len(index))
    return out


def column_axis_tie(
    wells_low: Sequence[str],
    ph_low: ArrayF,
    wells_high: Sequence[str],
    ph_high: ArrayF,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Match two temperatures' wells by column and return the index plus the pH offset.

    One temperature's titration axis is the other's plus one offset per column: a
    column holds one nominal target, and the buffer's pH moves with temperature the
    same way for every well in it. Tying that removes the between-temperature half
    of the K-x ridge - only the reference temperature's axes stay free, and the
    other follows from them plus a measured offset.

    The match is on the trailing column number of the well label, the same replicate
    unit as :func:`well_groups` and the ``column`` QC strata.

    Parameters
    ----------
    wells_low : Sequence[str]
        Well labels of the lower temperature.
    ph_low : ArrayF
        Recorded pH of each lower-temperature well, in the same order.
    wells_high : Sequence[str]
        Well labels of the higher temperature.
    ph_high : ArrayF
        Recorded pH of each higher-temperature well, in the same order.

    Returns
    -------
    tuple[np.ndarray, np.ndarray] | None
        For every high-temperature well, the index of the low-temperature well of
        the same column and the recorded pH difference ``ph_high - ph_low``; or
        ``None`` when the two lists do not match by column, so a caller falls back
        to an independent axis rather than tie the wrong wells together.
    """
    index: dict[str, list[int]] = {}
    for i, well in enumerate(wells_low):
        found = re.match(r"^[A-Za-z]+(\d+)$", str(well).strip())
        if found is None:
            return None
        index.setdefault(found.group(1), []).append(i)
    pair = np.empty(len(wells_high), dtype=int)
    offsets = np.empty(len(wells_high), dtype=np.float64)
    used: dict[str, int] = {}
    for i, well in enumerate(wells_high):
        found = re.match(r"^[A-Za-z]+(\d+)$", str(well).strip())
        if found is None:
            return None
        column = found.group(1)
        candidates = index.get(column)
        if not candidates:
            return None
        k = used.get(column, 0)
        if k >= len(candidates):
            return None
        pair[i] = candidates[k]
        used[column] = k + 1
        offsets[i] = float(ph_high[i]) - float(ph_low[pair[i]])
    return pair, offsets


def create_x_true_pooled(
    xc: ArrayF,
    *,
    pool: XPool = "free",
    x_sigma: float | None = None,
    groups: Sequence[int] | None = None,
    name: str = "x_true",
) -> ArrayF | pm.Deterministic:
    """Latent titrant axis, with the wells of one replicate sharing their truth.

    The recorded axis is a measurement of a value the wells of a replicate group
    share, so three structures are offered and two of them nest the third:

    ``free``
        ``x_true_i ~ N(x_i, x_sigma)``, independent per well - what every fit did
        before this existed. It lets a well move (a wrongly pipetted concentration, a
        drifting electrode) but it also *believes* three different recorded values
        for one solution, so a shared fraction at three different recorded axes can
        only be reconciled by moving K.
    ``shared``
        ``x_g ~ N(mean(x_i in g), tau)`` and every well of ``g`` uses ``x_g``: the
        wells are one solution, and the recorded spread is the axis error rather
        than truth.
    ``partial``
        ``x_g ~ N(mean(x_i in g), tau)`` and ``x_true_i ~ N(x_g, omega)``: the group
        shares a truth but a well may still deviate, and ``omega`` is inferred. It
        nests ``shared`` (``omega -> 0``) and, without the mean-1 style constraint
        ``free`` lacks, approximates ``free`` as ``omega`` grows, so the data decide
        how much pooling is warranted instead of the caller asserting it. On a
        ligand plate the posterior ``omega`` is the pipetting error the note never
        measured; on a pH plate a large ``omega`` says the rows were not one
        solution after all.

    ``tau`` and ``omega`` are inferred (half-normal around the measurement-error
    scale ``x_sigma``), so neither the pooling strength nor the axis noise is a
    plug-in. A missing ``x_sigma`` fixes the axis at the recorded values, which is
    only meaningful for ``free``.

    Parameters
    ----------
    xc : ArrayF
        Recorded titrant value per well.
    pool : XPool
        ``free``, ``shared`` or ``partial`` as above.
    x_sigma : float | None
        Measurement-error scale of the recorded axis, in its own units. It is the
        centre of the half-normal priors on ``tau`` and ``omega``; ``None`` fixes
        the axis.
    groups : Sequence[int] | None
        Group index per well (see :func:`well_groups`); required for ``shared``
        and ``partial``.
    name : str
        Name of the returned deterministic variable.

    Returns
    -------
    ArrayF | pm.Deterministic
        The latent axis, ready to use in a model.

    Raises
    ------
    ValueError
        For a pooled structure without groups or without a measurement-error
        scale, rather than silently fitting something else.
    """
    xa = np.asarray(xc, dtype=np.float64)
    if pool == "free" or x_sigma is None or not np.isfinite(x_sigma):
        if x_sigma is None or not np.isfinite(x_sigma):
            return pm.Deterministic(name, as_tensor_variable(xa))
        return pm.Normal(
            name, mu=as_tensor_variable(xa), sigma=float(x_sigma), shape=xa.size
        )
    if groups is None:
        msg = f"a {pool!r} axis needs the replicate groups; pass --x-groups column"
        raise ValueError(msg)
    idx = np.asarray(groups, dtype=int)
    if idx.size != xa.size:
        msg = f"groups has {idx.size} entries for {xa.size} wells"
        raise ValueError(msg)
    n_groups = int(idx.max()) + 1 if idx.size else 0
    group_mean = np.array([float(np.mean(xa[idx == g])) for g in range(n_groups)])
    scale = float(x_sigma)
    # non-centred: the group value is mean + tau * z with z ~ N(0, 1). Written
    # centred instead, tau and the group values form a funnel - measured at 4
    # chains x 2000 draws it produced 33-98 divergences and r-hat up to 1.53,
    # against 0 divergences for the free axis, because a small tau forces the
    # group values into a narrow mouth that NUTS cannot follow.
    tau = pm.HalfNormal(f"{name}_group_sd", sigma=scale)
    z_group = pm.Normal(f"{name}_group_z", mu=0.0, sigma=1.0, shape=n_groups)
    group_mu = as_tensor_variable(group_mean)
    shared = group_mu + tau * z_group
    if pool == "shared":
        return pm.Deterministic(name, shared[idx])
    omega = pm.HalfNormal(f"{name}_well_sd", sigma=scale)
    z_well = pm.Normal(f"{name}_well_z", mu=0.0, sigma=1.0, shape=xa.size)
    return pm.Deterministic(name, shared[idx] + omega * z_well)


def create_x_true(
    xc: ArrayF,
    x_errc: ArrayF,
    n_xerr: float,
    lower_nsd: float = 2.5,
    min_x_step: float = 0.2,
) -> ArrayF | pm.Deterministic:
    """Create latent variables for x-values with uncertainty.

    Models the pH axis as a pipetting random walk: the true pH accumulates
    independent per-addition volume errors, so ``x_true = x_start +
    cumsum(x_step)`` with each step's sigma taken from the per-addition pipetting
    variance (see :func:`_pipetting_step_sigmas`). Returns a PyMC Deterministic
    variable when in a Model context with uncertainty, or a numpy array when
    there's no uncertainty or no active Model.
    """
    if n_xerr > 0 and np.any(x_errc > 0):
        direction, x_start_sigma, step_nominal, step_sigmas, min_steps = (
            _pipetting_walk_params(
                xc, x_errc, n_xerr, min_x_step=min_x_step, lower_nsd=lower_nsd
            )
        )
        x_start = pm.Normal("x_start", mu=xc[0], sigma=max(x_start_sigma, 1e-6))
        x_step = pm.TruncatedNormal(
            "x_step",
            mu=step_nominal,
            sigma=np.maximum(step_sigmas, 1e-6),
            lower=min_steps,
            shape=len(xc) - 1,
        )
        x_cumsum = pm.math.cumsum(x_step)
        x_offsets = pm.math.concatenate([as_tensor_variable(np.array([0.0])), x_cumsum])
        return pm.Deterministic("x_true", x_start + direction * x_offsets)
    # No uncertainty - check if we're in a Model context
    try:
        model = pm.Model.get_context(error_if_none=False)
        if model is not None:
            # In a model context, wrap as Deterministic
            return pm.Deterministic("x_true", as_tensor_variable(xc))
    except Exception:  # ruff: ignore[try-except-pass, blind-except]
        pass
    # Outside model context or error, return numpy array
    return xc


def _safe_sigma(
    stderr: float | None,
    n_sd: float,
    *,
    default: float = 0.2,
    floor: float = 1e-3,
    cap: float | None = None,
) -> float:
    """Return a safe prior *sigma* from an lmfit ``stderr`` estimate.

    Handles ``None``, ``NaN``, ``inf``, and non-positive values by falling
    back to *default*.  The result is clamped to [*floor*, *cap*].
    """
    if stderr is None or not np.isfinite(stderr) or stderr <= 0:
        sigma = default
    else:
        sigma = max(stderr * n_sd, floor)
    if cap is not None:
        sigma = min(sigma, cap)
    return float(sigma)
