"""Noise model priors: floors, gain, proportional error, and the ye_mag family."""

from __future__ import annotations

import typing
from collections.abc import Mapping, Mapping as MappingABC, Sequence
from typing import Literal

import numpy as np
import pymc as pm  # type: ignore[import-untyped]
import pytensor.tensor as pt
from pymc import math as pm_math

from clophfit.fitting.data_structures import (
    FitResult,
    PlateNoiseModel,
)
from clophfit.fitting.utils import resolve_y_err

if typing.TYPE_CHECKING:
    import pandas as pd

    from clophfit.fitting.bayes_config import (
        NoiseParamMode,
    )


# Numeric floor for free/centered noise-prior scales, so a 0 hint gives the
# tightest around-zero prior (never a degenerate 0) and the prior width grows
# monotonically with the hint.
_MIN_NOISE_PRIOR_SCALE = 1e-3
# Width factor for any *unresolved* (exactly-zero) hint, in units of that
# parameter's plate scale. A resolved hint is known to about 20% (see
# _gain_prior_sigma / _alpha_prior_sigma), but a hint of exactly 0.0 is not a
# measurement of zero -- it means the collinear partner (gain vs. alpha) won
# this label's NNLS decomposition. The width therefore has to span the
# plausible range of the plate's values for that parameter, not a tight band
# around zero, so it gets its own factor rather than reusing the 20% relative
# width.
_ZERO_HINT_WIDTH = 1.0
# Plate-wide alpha scale to assume when NO label resolved a positive alpha
# (all NNLS fits landed on the alpha=0 boundary). Alpha is dimensionless, and
# empirically reaches ~0.106 on real plates (see build_pymc_noise_priors), so
# 0.1 places a plausible alpha at about one sigma of the resulting prior.
# Gain needs no equivalent: build_pymc_noise_priors's has_gain gate omits the
# gain term entirely when no label resolved a positive gain, whereas alpha's
# gate there keeps building a prior even when every hint is 0.
_ZERO_HINT_ALPHA_SCALE = 0.1
# Prior scale for the per-label pH axis on ye_mag (`separable_step`). Set from
# the measured profile rather than taste: label 1's robust per-step SD spans
# roughly 0.70-1.45 of its plate mean, a log range of ~0.73, so a log-scale SD
# near 0.25 covers the observed structure while still letting a label with no
# reproducible profile - label 2, cross-plate correlation +0.04 - shrink to
# flat. Wide enough to express what is there, tight enough not to invent it.
_YE_MAG_STEP_TAU_SIGMA = 0.5
# Bounds on the data-derived version of the same scale. The floor keeps a label
# whose prefit happens to look flat from having its pH axis pinned shut; the
# ceiling stops one pathological well from opening it arbitrarily wide.
_YE_MAG_STEP_TAU_MIN = 0.1
_YE_MAG_STEP_TAU_MAX = 1.5
_MIN_STEPS_FOR_TAU_HINT = 3


def _build_floor_prior(name: str, mu: float, mode: NoiseParamMode) -> typing.Any:  # ruff: ignore[any-type]
    """Build one floor prior from a hint, shared by the pooled and per-label paths.

    Parameters
    ----------
    name : str
        Name of the PyMC variable to create.
    mu : float
        Calibrated ``sigma_floor`` hint.
    mode : NoiseParamMode
        How to treat the floor parameter: "fixed", "free", or "centered".

    Returns
    -------
    typing.Any
        A PyMC random variable, or a constant tensor when *mode* is "fixed".
    """
    if mode == "fixed":
        return pt.as_tensor_variable(mu)
    if mode == "free":
        # Uninformative prior, using mu only as a scale hint
        return pm.HalfNormal(name, sigma=max(mu, 1.0))
    # centered
    n_pts = 7
    dof = max(1, n_pts - 1)
    rel_sigma = float(np.clip(1.0 / np.sqrt(2 * dof), 0.05, 0.5))
    sigma = max(rel_sigma * mu, 0.01)
    return pm.TruncatedNormal(name, mu=mu, sigma=sigma, lower=0.0)


def _gain_prior_sigma(mu_g: float, plate_gain_scale: float) -> float:
    """Prior width for one gain hint.

    Gain carries the units of the signal, so its width is always relative, but
    the two cases mean different things and use different factors. A
    *resolved* hint (``mu_g > 0``) is a real calibrated value known to about
    20% (``0.2 * mu_g``). An *unresolved* hint (``mu_g == 0``) is not a
    measurement of zero: it comes from the NNLS boundary and means the
    collinear alpha term won this label's decomposition, so the width instead
    has to span the plausible range of the plate's gains
    (``_ZERO_HINT_WIDTH * plate_gain_scale``) rather than being a tight
    band around zero.

    Parameters
    ----------
    mu_g : float
        This label's calibrated gain hint.
    plate_gain_scale : float
        Mean of the positive gains on the plate, used when *mu_g* is 0. The
        caller's ``has_gain`` gate guarantees this is non-zero whenever a
        zero hint can reach here.

    Returns
    -------
    float
        Standard deviation for the gain prior.
    """
    if mu_g > 0.0:
        return 0.2 * mu_g
    return _ZERO_HINT_WIDTH * plate_gain_scale


def _alpha_prior_sigma(mu_a: float, plate_alpha_scale: float) -> float:
    """Prior width for one alpha hint.

    Mirrors :func:`_gain_prior_sigma`: a *resolved* hint (``mu_a > 0``) is a
    real calibrated value, so its width is a relative fraction of the hint
    itself (``max(0.25 * mu_a, _MIN_NOISE_PRIOR_SCALE)``, preserving the prior
    resolved-hint behaviour). An *unresolved* hint (``mu_a == 0``) is not a
    measurement of zero: it comes from the NNLS boundary and means the
    collinear gain term won this label's decomposition, so the width instead
    has to span the plausible range of the plate's alphas
    (``_ZERO_HINT_WIDTH * plate_alpha_scale``) rather than being a tight band
    around zero.

    Parameters
    ----------
    mu_a : float
        This label's calibrated alpha hint.
    plate_alpha_scale : float
        Mean of the positive alphas on the plate, or ``_ZERO_HINT_ALPHA_SCALE``
        when no label resolved a positive alpha. Used when *mu_a* is 0.

    Returns
    -------
    float
        Standard deviation for the alpha prior.
    """
    if mu_a > 0.0:
        return max(0.25 * mu_a, _MIN_NOISE_PRIOR_SCALE)
    return _ZERO_HINT_WIDTH * plate_alpha_scale


def build_pymc_noise_priors(  # ruff: ignore[complex-structure, too-many-branches, too-many-arguments, too-many-statements]
    noise_model: PlateNoiseModel,
    *,
    shared_alpha: bool = False,
    shared_gain: bool = False,
    shared_floor: bool = False,
    floor_mode: NoiseParamMode = "centered",
    gain_mode: NoiseParamMode = "centered",
    alpha_mode: NoiseParamMode = "centered",
) -> dict[str, typing.Any]:
    """Create PyMC priors from a ``PlateNoiseModel`` specification.

    Parameters
    ----------
    noise_model : PlateNoiseModel
        Per-label noise parameter specification.
    shared_alpha : bool
        If ``True``, pool the proportional error into a single global variable.
    shared_gain : bool
        If ``True``, pool the photon gain into a single global variable.
    shared_floor : bool
        If ``True``, pool the noise floor into a single global variable.
    floor_mode : NoiseParamMode
        How to treat the floor parameter: "fixed", "free", or "centered".
    gain_mode : NoiseParamMode
        How to treat the gain parameter: "fixed", "free", or "centered".
    alpha_mode : NoiseParamMode
        How to treat the alpha (rel_error) parameter: "fixed", "free", or "centered".

    Returns
    -------
    dict[str, typing.Any]
        A structured dictionary of PyMC random variables or constants representing
        the noise model components, consumed by :func:`get_pymc_variance`.
    """
    priors: dict[str, typing.Any] = {}
    labels = list(noise_model.keys())

    # 1. Floor Priors
    if shared_floor:
        floors = [p.sigma_floor for p in noise_model.values() if p.sigma_floor > 0]
        mu_f = float(np.mean(floors)) if floors else 0.0
        priors["floor"] = _build_floor_prior("floor", mu_f, floor_mode)
    else:
        priors["floor"] = {}
        for lbl in labels:
            priors["floor"][lbl] = _build_floor_prior(
                f"floor_{lbl}", noise_model[lbl].sigma_floor, floor_mode
            )

    # 2. Gain (Poisson term). The gate is deliberately narrower than alpha's
    # below. Alpha is dimensionless and empirically reaches about 0.1 at most,
    # so _ZERO_HINT_ALPHA_SCALE is a meaningful universal around-zero width and
    # alpha can always stay soft even with no positive hint. Gain carries
    # the units of the signal, so its around-zero width has to be borrowed from
    # labels that did resolve a gain; when no label resolved one there is
    # nothing to borrow, and omitting the term beats inventing a scale. Do not
    # "symmetrise" this gate — it is what guarantees plate_gain_scale > 0.
    has_gain = any(p.gain > 0 for p in noise_model.values())
    if has_gain or gain_mode == "free":
        positive_gains = [p.gain for p in noise_model.values() if p.gain > 0]
        plate_gain_scale = float(np.mean(positive_gains)) if positive_gains else 0.0
        if shared_gain:
            mu_g = plate_gain_scale
            if gain_mode == "fixed":
                priors["gain"] = pt.as_tensor_variable(mu_g)
            elif gain_mode == "free":
                # Hint sets the Exponential prior mean (= 1/lam), floored so a 0
                # hint is the tightest around-zero prior (Poisson term ~off).
                lam = 1.0 / max(mu_g, _MIN_NOISE_PRIOR_SCALE)
                priors["gain"] = pm.Exponential("gain", lam=lam)
            else:  # centered
                # In the shared branch mu_g IS plate_gain_scale (set just
                # above), so this can only ever take the resolved-hint arm of
                # _gain_prior_sigma; the zero-hint fallback is per-label only.
                priors["gain"] = pm.TruncatedNormal(
                    "gain",
                    mu=mu_g,
                    sigma=_gain_prior_sigma(mu_g, plate_gain_scale),
                    lower=0.0,
                )
        else:
            priors["gain"] = {}
            for lbl in labels:
                mu_g = getattr(noise_model[lbl], "gain", 0.0)
                if gain_mode == "fixed":
                    priors["gain"][lbl] = pt.as_tensor_variable(mu_g)
                elif gain_mode == "free":
                    # Hint sets the Exponential mean; a 0 hint -> tightest (~off).
                    lam = 1.0 / max(mu_g, _MIN_NOISE_PRIOR_SCALE)
                    priors["gain"][lbl] = pm.Exponential(f"gain_{lbl}", lam=lam)
                elif mu_g > 0.0:
                    priors["gain"][lbl] = pm.TruncatedNormal(
                        f"gain_{lbl}",
                        mu=mu_g,
                        sigma=_gain_prior_sigma(mu_g, plate_gain_scale),
                        lower=0.0,
                    )
                else:
                    # Exact 0 from the NNLS boundary: alpha won this label's
                    # decomposition. Keep the term estimable around 0 with a
                    # width spanning the plate's gain scale (not a fraction of
                    # it -- see _ZERO_HINT_WIDTH), so the posterior can
                    # undo an arbitrary collinear split.
                    priors["gain"][lbl] = pm.HalfNormal(
                        f"gain_{lbl}",
                        sigma=_gain_prior_sigma(mu_g, plate_gain_scale),
                    )

    # 3. Alpha (proportional error). "centered" and "free" always build a prior
    # so a calibrated alpha of 0 becomes a prior spanning the plate's alpha
    # scale, not a hard 0; only "fixed" leaves the term truly absent when
    # alpha is 0. In "free" mode and in "centered" on a positive hint, the
    # alpha hint is the prior mean (HalfNormal sigma is scaled by sqrt(pi/2)
    # to achieve that; TruncatedNormal takes it directly), floored at
    # _MIN_NOISE_PRIOR_SCALE so the width grows monotonically with the hint
    # (the weak 0.02 default lives in NoiseConfig.alpha, not here). The one
    # exception, centered mode with a zero hint, routes through
    # _alpha_prior_sigma, whose zero-hint arm spans plate_alpha_scale rather
    # than being a tight band around zero -- see _gain_prior_sigma for the
    # analogous gain case.
    has_alpha = any(p.alpha > 0 for p in noise_model.values())
    if has_alpha or alpha_mode in {"free", "centered"}:
        positive_alphas = [p.alpha for p in noise_model.values() if p.alpha > 0]
        plate_alpha_scale = (
            float(np.mean(positive_alphas))
            if positive_alphas
            else _ZERO_HINT_ALPHA_SCALE
        )
        if shared_alpha:
            mu_a = float(np.mean(positive_alphas)) if positive_alphas else 0.0
            if alpha_mode == "fixed":
                priors["rel_error"] = pt.as_tensor_variable(mu_a)
            elif alpha_mode == "free":
                # sqrt(pi/2) converts the hint from a scale to a mean, so the
                # hint means the same thing here as it does for gain's
                # Exponential (whose mean is exactly its hint).
                priors["rel_error"] = pm.HalfNormal(
                    "rel_error",
                    sigma=max(mu_a, _MIN_NOISE_PRIOR_SCALE) * np.sqrt(np.pi / 2),
                )
            else:  # centered
                priors["rel_error"] = pm.TruncatedNormal(
                    "rel_error",
                    mu=mu_a,
                    sigma=_alpha_prior_sigma(mu_a, plate_alpha_scale),
                    lower=0.0,
                )
        else:
            priors["rel_error"] = {}
            for lbl in labels:
                mu_a = getattr(noise_model[lbl], "alpha", 0.0)
                if alpha_mode == "fixed":
                    priors["rel_error"][lbl] = pt.as_tensor_variable(mu_a)
                elif alpha_mode == "free":
                    # See the shared branch: sqrt(pi/2) makes the hint a mean.
                    priors["rel_error"][lbl] = pm.HalfNormal(
                        f"rel_error_{lbl}",
                        sigma=max(mu_a, _MIN_NOISE_PRIOR_SCALE) * np.sqrt(np.pi / 2),
                    )
                elif mu_a > 0.0:  # centered on a positive hint
                    priors["rel_error"][lbl] = pm.TruncatedNormal(
                        f"rel_error_{lbl}",
                        mu=mu_a,
                        sigma=_alpha_prior_sigma(mu_a, plate_alpha_scale),
                        lower=0.0,
                    )
                else:
                    # Exact 0 from the NNLS boundary: gain won this label's
                    # decomposition. Keep the term estimable around 0 with a
                    # width spanning the plate's alpha scale (not a fraction
                    # of it -- see _ZERO_HINT_WIDTH), so the posterior can
                    # undo an arbitrary collinear split.
                    priors["rel_error"][lbl] = pm.HalfNormal(
                        f"rel_error_{lbl}",
                        sigma=_alpha_prior_sigma(mu_a, plate_alpha_scale),
                    )

    return priors


def _active_noise_model(
    noise_model: PlateNoiseModel, labels: Sequence[str]
) -> PlateNoiseModel:
    """Return noise parameters only for labels present in the fitted dataset."""
    active = PlateNoiseModel()
    missing: list[str] = []
    for lbl in labels:
        if lbl in noise_model:
            active[lbl] = noise_model[lbl]
        elif "default" in noise_model:
            active[lbl] = noise_model["default"]
        else:
            missing.append(lbl)
    if missing:
        available = ", ".join(map(str, noise_model.keys()))
        needed = ", ".join(missing)
        msg = f"Noise model is missing fitted label(s) {needed}. Available: {available}"
        raise KeyError(msg)
    return active


def get_pymc_variance(
    mu: pm.math.TensorVariable,
    label: str,
    noise_model: PlateNoiseModel,
    priors: dict[str, typing.Any],
) -> pm.math.TensorVariable:
    """Construct PyMC symbolic variance from a noise model specification.

    Parameters
    ----------
    mu : pm.math.TensorVariable
        Symbolic predicted signal (must be positive).
    label : str
        Label key into *noise_model*.
    noise_model : PlateNoiseModel
        Per-label noise parameter specification.
    priors : dict[str, typing.Any]
        Priors dict from :func:`build_pymc_noise_priors`.

    Returns
    -------
    pm.math.TensorVariable
        Symbolic variance tensor.
    """
    params = noise_model[label]
    y_pos = pm_math.maximum(1e-6, mu)
    floor = (
        priors["floor"][label] if isinstance(priors["floor"], dict) else priors["floor"]
    )
    # Broadcast floor² to match y_pos shape so the result is always
    # per-point (never 0-d), even when gain and alpha are both absent.
    var = floor**2 * pt.ones_like(y_pos)  # type: ignore[no-untyped-call]

    if params.gain > 0 or "gain" in priors:
        gain = (
            priors["gain"][label]
            if isinstance(priors["gain"], dict)
            else priors["gain"]
        )
        var += gain * y_pos

    if params.alpha > 0 or "rel_error" in priors:
        alpha = (
            priors["rel_error"][label]
            if isinstance(priors["rel_error"], dict)
            else priors["rel_error"]
        )
        var += (alpha * y_pos) ** 2

    return var


def _resolve_noise_modes(
    *,
    prefer_centered: bool,
    floor_mode: NoiseParamMode | None,
    gain_mode: NoiseParamMode | None,
    alpha_mode: NoiseParamMode | None,
) -> tuple[NoiseParamMode, NoiseParamMode, NoiseParamMode]:
    """Resolve automatic noise-mode defaults from the input kind."""
    default_mode: NoiseParamMode = "centered" if prefer_centered else "free"
    return (
        floor_mode or default_mode,
        gain_mode or default_mode,
        alpha_mode or default_mode,
    )


def _build_ye_mag_priors(
    labels: Sequence[str],
    *,
    shared_ye_mags: bool = False,
    prior: Literal["halfnormal", "lognormal"] = "lognormal",
    mu: float | Mapping[str, float] = 0.0,
    sigma: float | Mapping[str, float] = 1.5,
) -> dict[str, typing.Any]:
    """Build label-indexed observation-error scale priors."""
    if shared_ye_mags:
        shared_sigma = _shared_ye_mag_sigma(sigma)
        if prior == "halfnormal":
            ye_mag = pm.HalfNormal("ye_mag", sigma=shared_sigma)
        else:
            ye_mag = pm.LogNormal(
                "ye_mag", mu=_shared_ye_mag_value(mu), sigma=shared_sigma
            )
        return dict.fromkeys(labels, ye_mag)

    if prior == "halfnormal":
        return {
            lbl: pm.HalfNormal(f"ye_mag_{lbl}", sigma=_ye_mag_sigma(sigma, lbl))
            for lbl in labels
        }
    return {
        lbl: pm.LogNormal(
            f"ye_mag_{lbl}",
            mu=_ye_mag_value(mu, lbl),
            sigma=_ye_mag_sigma(sigma, lbl),
        )
        for lbl in labels
    }


def _build_multi_ye_mag_priors(  # ruff: ignore[too-many-arguments]
    labels: Sequence[str],
    *,
    per_well: bool = False,
    shared_ye_mags: bool = False,
    prior: Literal["halfnormal", "lognormal"] = "lognormal",
    mu: float | Mapping[str, float] = 0.0,
    sigma: float | Mapping[str, float] = 1.5,
    parameterization: Literal[
        "centered", "hierarchical", "separable", "separable_step"
    ] = "centered",
    step_tau_sigma: Mapping[str, float] | None = None,
) -> dict[str, typing.Any]:
    """Build label-indexed ye_mag priors for multi-well models.

    ``parameterization`` only affects the per-well, non-shared, LogNormal branch:
    ``"centered"`` reproduces the historical independent-per-well construction;
    ``"hierarchical"`` shares a per-well factor across labels with a learned
    deviation scale (partial pooling; see
    :func:`_build_hierarchical_ye_mag_priors`).
    """
    if not per_well:
        return _build_ye_mag_priors(
            labels,
            shared_ye_mags=shared_ye_mags,
            prior=prior,
            mu=mu,
            sigma=sigma,
        )
    if shared_ye_mags:
        shared_sigma = _shared_ye_mag_sigma(sigma)
        if prior == "halfnormal":
            ye_mag = pm.HalfNormal("ye_mag", sigma=shared_sigma, dims="well")
        else:
            ye_mag = pm.LogNormal(
                "ye_mag",
                mu=_shared_ye_mag_value(mu),
                sigma=shared_sigma,
                dims="well",
            )
        return dict.fromkeys(labels, ye_mag)
    if prior == "halfnormal":
        return {
            lbl: pm.HalfNormal(
                f"ye_mag_{lbl}", sigma=_ye_mag_sigma(sigma, lbl), dims="well"
            )
            for lbl in labels
        }
    builders = {
        "hierarchical": _build_hierarchical_ye_mag_priors,
        "separable": _build_separable_ye_mag_priors,
        "separable_step": _build_separable_step_ye_mag_priors,
    }
    if parameterization == "separable_step":
        return _build_separable_step_ye_mag_priors(
            labels, mu=mu, sigma=sigma, step_tau_sigma=step_tau_sigma
        )
    builder = builders.get(parameterization)
    if builder is not None:
        return builder(labels, mu=mu, sigma=sigma)
    return {
        lbl: pm.LogNormal(
            f"ye_mag_{lbl}",
            mu=_ye_mag_value(mu, lbl),
            sigma=_ye_mag_sigma(sigma, lbl),
            dims="well",
        )
        for lbl in labels
    }


def _build_separable_ye_mag_priors(
    labels: Sequence[str],
    *,
    mu: float | Mapping[str, float],
    sigma: float | Mapping[str, float],
) -> dict[str, typing.Any]:
    """Separable per-well ye_mag: per-label level plus one shared well factor.

    ``log ye_mag_lbl[w] = mu_lbl + tau_well * z_well[w]``, costing ``#lbl +
    #well + 1`` parameters against ``#well * #lbl`` for the centered and
    hierarchical per-well priors.

    The hierarchical prior shares ``mu`` across labels, so it has no label main
    effect and must spend per-well deviations to represent one. It does exactly
    that: over eleven plates its fitted deviations are systematically negative
    for label 1 and positive for label 2 on every plate, giving geometric-mean
    ``ye_mag_2/ye_mag_1`` ratios of 1.1 to 8.0. That is ``#well * #lbl``
    parameters carrying a ``#lbl`` effect, and it inflates the effective
    parameter count enough to break PSIS-LOO. Here the label level is explicit
    and the well factor is shared, giving the same structure far more cheaply.

    ``z_well`` is mean-zero and the level lives entirely in ``mu_lbl``, so the
    two are identifiable.

    Parameters
    ----------
    labels : Sequence[str]
        Emission labels.
    mu : float | Mapping[str, float]
        Log-scale prior location for each label's level.
    sigma : float | Mapping[str, float]
        Log-scale prior scale for each label's level.

    Returns
    -------
    dict[str, typing.Any]
        Label -> per-well ``ye_mag`` tensor (dims ``"well"``).
    """
    tau_well = pm.HalfNormal("ye_mag_tau_well", sigma=_shared_ye_mag_sigma(sigma))
    z_well = pm.Normal("ye_mag_z_well", 0.0, 1.0, dims="well")
    log_well = tau_well * z_well
    out: dict[str, typing.Any] = {}
    for lbl in labels:
        mu_lbl = pm.Normal(
            f"ye_mag_mu_{lbl}", _ye_mag_value(mu, lbl), _ye_mag_sigma(sigma, lbl)
        )
        out[lbl] = pm.Deterministic(
            f"ye_mag_{lbl}", pm.math.exp(mu_lbl + log_well), dims="well"
        )
    return out


def step_tau_hint_from_residuals(
    residuals: pd.DataFrame, labels: Sequence[str]
) -> dict[str, float]:
    """Estimate the ye_mag pH-axis prior scale from a fast first-round fit.

    A fixed scale is a guess about how much the scatter varies along a titration,
    and it was a poor one: set at 0.25 from a residual-SD profile, the posteriors
    came back at 0.37-0.75, so the data had to fight the prior. The seeding
    least-squares fits already contain the answer, so ask them.

    Per label, take the robust SD of the prefit standardised residuals at each
    step, normalise by its own mean, and return the SD in log space - directly
    the scale of ``tau_step``. A flat profile gives a small number and the pH
    axis stays shut; a varying one opens it as far as the data warrant.

    Robust statistics throughout: a single wild well must not be able to
    manufacture a pH axis for the whole plate.

    Parameters
    ----------
    residuals : pd.DataFrame
        Canonical residual table from the seeding fits, with ``label``,
        ``step`` and ``std_res`` columns.
    labels : Sequence[str]
        Emission labels.

    Returns
    -------
    dict[str, float]
        Label -> prior scale, clipped to a sane range. Labels the table cannot
        speak to fall back to :data:`_YE_MAG_STEP_TAU_SIGMA`.
    """
    out: dict[str, float] = {}
    for lbl in labels:
        out[lbl] = _YE_MAG_STEP_TAU_SIGMA
        sub_df = residuals[residuals["label"].astype(str) == str(lbl)]
        if sub_df.empty:
            continue
        prof = sub_df.groupby("step")["std_res"].apply(
            lambda v: float(1.4826 * np.median(np.abs(v - np.median(v))))
        )
        prof = prof[np.isfinite(prof) & (prof > 0)]
        if len(prof) < _MIN_STEPS_FOR_TAU_HINT:
            continue
        rel = np.asarray(prof, dtype=float) / float(np.mean(prof))
        out[lbl] = float(
            np.clip(np.std(np.log(rel)), _YE_MAG_STEP_TAU_MIN, _YE_MAG_STEP_TAU_MAX)
        )
    return out


def _build_separable_step_ye_mag_priors(
    labels: Sequence[str],
    *,
    mu: float | Mapping[str, float],
    sigma: float | Mapping[str, float],
    step_tau_sigma: Mapping[str, float] | None = None,
) -> dict[str, typing.Any]:
    """Separable ye_mag with a pH axis: label level x well factor x step factor.

    ``log ye_mag_lbl[s, w] = mu_lbl + tau_well * z_well[w]
    + tau_step_lbl * z_step_lbl[s]``.

    One ye_mag per label assumes the scatter is constant along the titration,
    and for label 1 it is not: the robust per-step SD varies about 2.6x within a
    plate, and nine plates agree on *which* steps are noisy - profile
    correlation +0.58, step 6 noisiest on every one. That is the acidic end,
    where wells differ in how far their label-1 signal turns over, so the excess
    scatter there is real. Label 2 shows a similar 2.4x spread with no
    cross-plate agreement (+0.04), which is sampling noise in the SD estimate.

    So the pH axis is per label and can switch itself off: ``tau_step_lbl``
    shrinks to zero for a label with no reproducible profile, leaving a single
    ye_mag, and grows for one that has. The data decides rather than the caller.

    Unlike a mean-shift term, this reweights points instead of moving them, so
    it cannot bias K the way a per-step gain does - though it still reaches K
    through the weighting, so it is worth scoring against known pK.

    ``ye_mag_{lbl}`` keeps its per-well dims so existing diagnostics still read
    the level they expect; the pH factor is published as ``ye_mag_step_{lbl}``.

    Parameters
    ----------
    labels : Sequence[str]
        Emission labels.
    mu : float | Mapping[str, float]
        Log-scale prior location for each label's level.
    sigma : float | Mapping[str, float]
        Log-scale prior scale for each label's level.
    step_tau_sigma : Mapping[str, float] | None
        Per-label prior scale for the pH axis, normally estimated from the
        seeding fits by :func:`step_tau_hint_from_residuals`. Falls back to
        :data:`_YE_MAG_STEP_TAU_SIGMA` for any label not covered.

    Returns
    -------
    dict[str, typing.Any]
        Label -> ``ye_mag`` tensor with dims ``("step", "well")``, matching the
        order the multi-well sigma is assembled in.
    """
    tau_well = pm.HalfNormal("ye_mag_tau_well", sigma=_shared_ye_mag_sigma(sigma))
    z_well = pm.Normal("ye_mag_z_well", 0.0, 1.0, dims="well")
    log_well = tau_well * z_well
    out: dict[str, typing.Any] = {}
    for lbl in labels:
        mu_lbl = pm.Normal(
            f"ye_mag_mu_{lbl}", _ye_mag_value(mu, lbl), _ye_mag_sigma(sigma, lbl)
        )
        tau_sigma = (
            _YE_MAG_STEP_TAU_SIGMA
            if step_tau_sigma is None
            else float(step_tau_sigma.get(lbl, _YE_MAG_STEP_TAU_SIGMA))
        )
        tau_step = pm.HalfNormal(f"ye_mag_tau_step_{lbl}", sigma=tau_sigma)
        z_step = pm.Normal(f"ye_mag_z_step_{lbl}", 0.0, 1.0, dims="step")
        log_step = tau_step * z_step
        pm.Deterministic(f"ye_mag_{lbl}", pm.math.exp(mu_lbl + log_well), dims="well")
        pm.Deterministic(f"ye_mag_step_{lbl}", pm.math.exp(log_step), dims="step")
        out[lbl] = pm.math.exp(mu_lbl + log_well[None, :] + log_step[:, None])
    return out


def _build_hierarchical_ye_mag_priors(
    labels: Sequence[str],
    *,
    mu: float | Mapping[str, float],
    sigma: float | Mapping[str, float],
) -> dict[str, typing.Any]:
    """Hierarchical per-well ye_mag: shared factor x learned label deviation.

    ``log ye_mag_lbl[w] = (mu + sigma * z_w[w]) + tau_delta * z_delta_lbl[w]``,
    with ``tau_delta ~ HalfNormal(0.5)`` learned and everything non-centered. The
    shared ``z_w`` factor partially pools the labels' per-well multipliers; the
    mean-zero deviation priors keep the common level in the shared factor.

    Parameters
    ----------
    labels : Sequence[str]
        Emission labels.
    mu : float | Mapping[str, float]
        Log-scale location of the shared per-well factor.
    sigma : float | Mapping[str, float]
        Log-scale scale of the shared per-well factor.

    Returns
    -------
    dict[str, typing.Any]
        Label -> per-well ``ye_mag`` tensor (dims ``"well"``).
    """
    tau_delta = pm.HalfNormal("ye_mag_tau_delta", sigma=0.5)
    z_w = pm.Normal("ye_mag_z_well", 0.0, 1.0, dims="well")
    log_w = _shared_ye_mag_value(mu) + _shared_ye_mag_sigma(sigma) * z_w
    out: dict[str, typing.Any] = {}
    for lbl in labels:
        z_d = pm.Normal(f"ye_mag_z_delta_{lbl}", 0.0, 1.0, dims="well")
        out[lbl] = pm.Deterministic(
            f"ye_mag_{lbl}", pm.math.exp(log_w + tau_delta * z_d), dims="well"
        )
    return out


def _ye_mag_value(value: float | Mapping[str, float], label: str) -> float:
    """Return a finite ye_mag prior value for *label*."""
    if isinstance(value, MappingABC):
        values = np.asarray(list(value.values()), dtype=float)
        values = values[np.isfinite(values)]
        fallback = float(np.nanmean(values)) if values.size else 0.0
        raw = value.get(label, value.get(str(label), fallback))
    else:
        raw = value
    out = float(raw)
    return out if np.isfinite(out) else 0.0


def _shared_ye_mag_value(value: float | Mapping[str, float]) -> float:
    """Return a finite shared ye_mag prior value from a scalar or mapping."""
    if not isinstance(value, MappingABC):
        out = float(value)
        return out if np.isfinite(out) else 0.0
    values = np.asarray(list(value.values()), dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    return float(np.nanmean(values))


def _ye_mag_sigma(sigma: float | Mapping[str, float], label: str) -> float:
    """Return a finite positive ye_mag prior sigma for *label*."""
    if isinstance(sigma, MappingABC):
        values = np.asarray(list(sigma.values()), dtype=float)
        values = values[np.isfinite(values) & (values > 0.0)]
        fallback = float(np.nanmean(values)) if values.size else 1e-6
        raw = sigma.get(label, sigma.get(str(label), fallback))
    else:
        raw = sigma
    value = float(raw)
    if not np.isfinite(value) or value <= 0.0:
        return 1e-6
    return value


def _shared_ye_mag_sigma(sigma: float | Mapping[str, float]) -> float:
    """Return a finite positive shared sigma from a scalar or label mapping."""
    if not isinstance(sigma, MappingABC):
        value = float(sigma)
        return value if np.isfinite(value) and value > 0.0 else 1e-6
    values = np.asarray(list(sigma.values()), dtype=float)
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        return 1e-6
    return float(np.nanmean(values))


def _scale_ye_mag_sigma(
    sigma: float | Mapping[str, float], scale: float
) -> float | dict[str, float]:
    """Scale scalar or label-indexed ye_mag sigma hints."""
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    if isinstance(sigma, MappingABC):
        return {str(label): float(value) * scale for label, value in sigma.items()}
    return float(sigma) * scale


def _log_scaled_ye_mag_mu(
    median: float | Mapping[str, float], scale: float
) -> float | dict[str, float]:
    """Return log-median values for a scaled LogNormal ye_mag prior."""
    scaled = _scale_ye_mag_sigma(median, scale)
    if isinstance(scaled, dict):
        return {
            label: float(np.log(max(float(value), 1e-6)))
            for label, value in scaled.items()
        }
    return float(np.log(max(float(scaled), 1e-6)))


def _masked_obs_err_matrices(
    fit_results: Mapping[str, FitResult],
    wells_list: Sequence[str],
    lbl: str,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build row-major mask, observation, and error matrices for multi-well likelihood.

    Parameters
    ----------
    fit_results : Mapping[str, FitResult]
        Per-well fit results whose ``.dataset`` supplies the arrays.
    wells_list : Sequence[str]
        Well keys in the order they appear as columns.
    lbl : str
        Label key to extract from each well's dataset.
    n_steps : int
        Expected number of titration steps per well.

    Returns
    -------
    mask : np.ndarray
        Row-major boolean mask, shape ``(n_steps, n_wells)``.
    y_obs : np.ndarray
        Row-major observation matrix, shape ``(n_steps, n_wells)``.
    y_err : np.ndarray
        Row-major error matrix, shape ``(n_steps, n_wells)``.

    Raises
    ------
    ValueError
        If a well's dataset or its arrays have unexpected shapes.
    """
    n_wells = len(wells_list)
    mask = np.zeros((n_steps, n_wells), dtype=bool)
    y_obs = np.full((n_steps, n_wells), np.nan, dtype=float)
    y_err = np.full((n_steps, n_wells), np.nan, dtype=float)

    for w_idx, key in enumerate(wells_list):
        ds_well = fit_results[key].dataset
        if ds_well is None:
            msg = f"Dataset for well {key} is missing."
            raise ValueError(msg)

        if lbl not in ds_well:
            # Not an error: this well is fitted on its other labels, and its
            # column here stays masked off so it contributes nothing.
            continue

        da = ds_well[lbl]

        if da.mask.shape != (n_steps,):
            msg = (
                f"Mask shape mismatch for well {key}, label {lbl}: "
                f"{da.mask.shape} != {(n_steps,)}"
            )
            raise ValueError(msg)
        if da.yc.shape != (n_steps,):
            msg = (
                f"yc shape mismatch for well {key}, label {lbl}: "
                f"{da.yc.shape} != {(n_steps,)}"
            )
            raise ValueError(msg)

        mask[:, w_idx] = da.mask
        y_obs[:, w_idx] = np.asarray(da.yc, dtype=float)

        if da.y_errc.size > 0:
            if da.y_errc.shape != (n_steps,):
                msg = (
                    f"y_errc shape mismatch for well {key}, label {lbl}: "
                    f"{da.y_errc.shape} != {(n_steps,)}"
                )
                raise ValueError(msg)
            err = np.asarray(da.y_errc, dtype=float)
            err = resolve_y_err(err, label=lbl, well=key)
        else:
            err = np.ones(n_steps, dtype=float)

        y_err[:, w_idx] = err

    return mask, y_obs, y_err
