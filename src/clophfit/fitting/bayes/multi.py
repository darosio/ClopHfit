"""Global multi-well PyMC fits (fit_binding_pymc_multi and its screened variant)."""

from __future__ import annotations

import typing
import warnings
from typing import Literal

import numpy as np
import pandas as pd
import pymc as pm  # type: ignore[import-untyped]
import pytensor.tensor as pt
from pymc import math as pm_math
from pytensor.tensor import as_tensor_variable

from clophfit.fitting.bayes_config import (
    InitConfig,
    NoiseConfig,
    RobustConfig,
    SamplerConfig,
)
from clophfit.fitting.data_structures import (
    Dataset,
    FitResult,
    MultiFitResult,
)
from clophfit.fitting.model_validation import (
    EXCLUDE_COL,
    OutlierCriterion,
    RobustZMad,
    apply_exclusions,
    mark_outliers,
)
from clophfit.fitting.models import binding_1site, kd_bounds
from clophfit.fitting.utils import resolve_y_err

if typing.TYPE_CHECKING:
    from collections.abc import Mapping

    import xarray as xr

    from clophfit.prtecan import PlateScheme

from clophfit.fitting.bayes.axis import (
    _DEFAULT_X_START_BETWEEN_SIGMA,
    _build_multi_x_start,
    _pipetting_walk_params,
    _safe_sigma,
    create_x_true,
)
from clophfit.fitting.bayes.extract import (
    _per_well_fit_results_from_trace,
    label_representatives,
)
from clophfit.fitting.bayes.likelihood import (
    _add_mixture_likelihood,
    _build_outlier_priors,
    _resolve_structured_noise_model,
    _student_t_nu_value,
)
from clophfit.fitting.bayes.noise import (
    _active_noise_model,
    _build_multi_ye_mag_priors,
    _masked_obs_err_matrices,
    _resolve_noise_modes,
    build_pymc_noise_priors,
    get_pymc_variance,
    step_tau_hint_from_residuals,
)
from clophfit.fitting.bayes.priors import (
    _normalize_fit_inputs,
    log_kd_prior,
)
from clophfit.fitting.bayes.sampling import (
    _compute_sample_log_likelihood,
    _sample_trace,
)
from clophfit.fitting.bayes.single import (
    _DEFAULT_INIT,
    _DEFAULT_NOISE,
    _DEFAULT_ROBUST,
    _DEFAULT_SAMPLER,
    _build_ctr_k_params,
    _free_k_init,
    weighted_stats,
)


def fit_binding_pymc_multi(  # ruff: ignore[complex-structure, too-many-branches, too-many-arguments, too-many-statements]
    results: Mapping[str, Dataset | FitResult],
    scheme: PlateScheme,
    *,
    n_sd: float = 5.0,
    n_xerr: float = 1.0,
    min_x_step: float = 0.2,
    x_error_model: Literal["deterministic", "per_well"] = "deterministic",
    x_start_between_sigma: float = _DEFAULT_X_START_BETWEEN_SIGMA,
    learn_x_start_between: bool = False,
    ctr_free_k: bool = False,
    sample_ppc: bool = False,
    per_well_ye_mags: bool | None = None,
    ye_mag_parameterization: Literal[
        "centered", "hierarchical", "separable", "separable_step"
    ] = "centered",
    learn_hill: bool = False,
    hill_prior_sigma: float = 0.2,
    acid_scale: bool = False,
    acid_scale_sigma: float = 0.3,
    well_noise_scale: bool = False,
    shared_well_noise_scale: bool = False,
    label_noise_scale_sigma: float = 0.3,
    well_noise_sd_sigma: float = 0.3,
    well_noise_scale_sigma: float | None = None,
    ctr_sigma_w_prior: float | None = None,
    noise: NoiseConfig = _DEFAULT_NOISE,
    robust: RobustConfig = _DEFAULT_ROBUST,
    init: InitConfig = _DEFAULT_INIT,
    sampler: SamplerConfig = _DEFAULT_SAMPLER,
) -> MultiFitResult:
    """Multi-well PyMC with shared K per control group and per-label noise.

    Parameters
    ----------
    results : Mapping[str, Dataset | FitResult]
        Per-well datasets or initial fit results. Raw datasets are first fitted
        with :func:`fit_binding_glob` to seed the Bayesian model.
    scheme : PlateScheme
        Plate scheme defining control groups for shared-K priors.
    n_sd : float
        Prior width multiplier for per-well S0/S1 parameters.
    n_xerr : float
        Scaling factor applied to x-value uncertainties.
    min_x_step : float
        Minimum inferred change in ``x`` between consecutive titration steps.
    x_error_model : Literal["deterministic", "per_well"]
        Model for x-error propagation across wells. ``"deterministic"`` shares
        one latent ``x_true`` walk across all wells; ``"per_well"`` gives each
        well its own cumulative-additions walk.
    x_start_between_sigma : float
        Fixed between-well scale for the starting x (pH). Defaults to a small
        nonzero value, so each well gets its own ``x_start_well ~ Normal(
        x_start, x_start_between_sigma)`` tightly anchored on the shared plate
        ``x_start`` prior (common-mode anchor plus independent per-well jitter).
        Set to ``0.0`` for a single shared ``x_start`` across all wells.
    learn_x_start_between : bool
        Estimate that between-well scale (``x_start_between``) instead of
        pinning it, non-centred so it does not funnel; *x_start_between_sigma*
        is then its HalfNormal prior scale. A well's pH offset shifts its K, so
        a pinned scale asserts how far apart two wells' K may sit.
    ctr_free_k : bool
        If ``True``, each control replicate gets an independent flat K prior
        instead of a shared control K.
    sample_ppc : bool
        If ``True``, add posterior predictive samples to the returned trace.
    per_well_ye_mags : bool | None
        Learn per-well (not just per-label) ``ye_mag`` factors. ``None`` follows
        the noise config's ``learn_ye_mags`` when a structured model is supplied.
    ye_mag_parameterization : Literal["centered", "hierarchical", "separable", "separable_step"]
        Parameterization of the per-well ``ye_mag`` prior. ``"centered"`` (default)
        reproduces the historical independent-per-well construction;
        ``"hierarchical"`` partially pools the labels' per-well factors with a
        learned deviation scale (the only per-well form that samples reliably).
        Applies in both noise modes; a no-op for shared or non-per-well ``ye_mag``.
    learn_hill : bool
        Fit one plate-wide Hill coefficient, shared across wells and labels, so
        the transition may be broader or sharper than Henderson-Hasselbalch.
        ``False`` fixes it at 1 and reproduces the plain model exactly. K remains
        the midpoint at any value, so estimates stay comparable.
    hill_prior_sigma : float
        Width of the ``LogNormal(0, sigma)`` prior on that coefficient. The
        default keeps 95% of its mass between about 0.68 and 1.48, so the data
        must pull it off 1.
    acid_scale : bool
        pH titrations: give every well a factor on its most acidic step, shared by
        all labels (``acid_scale ~ LogNormal(0, acid_scale_sigma)``, dims ``well``),
        for the loss of emission at the last acid addition. The size of that loss
        depends on the mutant, so the factor is per well; the step's label ratio
        still informs K. ``False`` reproduces the plain model exactly.
    acid_scale_sigma : float
        Log-scale prior SD of that factor.
    well_noise_scale : bool
        Enable a per-well multiplicative noise scale.
    shared_well_noise_scale : bool
        Share the well-noise scale across labels.
    label_noise_scale_sigma : float
        LogNormal prior scale for the per-label noise scale.
    well_noise_sd_sigma : float
        HalfNormal prior scale for the per-well noise spread.
    well_noise_scale_sigma : float | None
        Backwards-compatible alias for *well_noise_sd_sigma*.
    ctr_sigma_w_prior : float | None
        Prior scale, in pH, of ``K_sigma_w``: the spread of a control group's
        replicates around the group's own K. Given, each replicate keeps its
        own K (still reported as ``K_free``) drawn a HalfNormal-distributed
        distance from the group's, instead of sharing one K (*ctr_free_k*
        False) or ignoring the group (*ctr_free_k* True). Controls on these
        plates miss their mates by more than their intervals allow, and this
        is the term that says by how much. pH only.
    noise : NoiseConfig
        Observation-noise configuration (ye_mag multiplier or structured
        floor/gain/alpha). See :class:`NoiseConfig`.
    robust : RobustConfig
        Robust-likelihood configuration (Student-t or contamination mixture).
        See :class:`RobustConfig`.
    init : InitConfig
        Prior-initialization strategy (``"lmfit"`` prefit or ``"data_priors"``).
        See :class:`InitConfig`.
    sampler : SamplerConfig
        NUTS sampling controls. See :class:`SamplerConfig`.

    Returns
    -------
    MultiFitResult
        Shared PyMC trace together with reconstructed per-well fit results.

    Raises
    ------
    ValueError
        If no valid dataset is found in results.
    """
    robust_on = robust.enabled
    robust_likelihood = robust.likelihood
    student_t_nu = robust.nu
    contamination_frac_prior = robust.contamination_frac_prior
    shared_alpha = noise.shared_alpha
    shared_gain = noise.shared_gain
    shared_floor = noise.shared_floor
    learn_ye_mags = noise.learn_ye_mags
    shared_ye_mags = noise.shared_ye_mags
    ye_mag_prior = noise.ye_mag_prior
    ye_mag_mu = noise.ye_mag_mu
    ye_mag_sigma = noise.ye_mag_sigma
    n_samples = sampler.n_samples
    n_tune = sampler.n_tune
    target_accept = sampler.target_accept
    init_strategy = init.strategy
    data_prior_edge_points = init.edge_points
    data_prior_signal_sigma_scale = init.signal_sigma_scale
    data_prior_k_prior = init.k_prior
    data_prior_k_bounds = init.k_bounds
    data_prior_k_sigma = init.k_sigma
    floor_mode = noise.floor_mode
    gain_mode = noise.gain_mode
    alpha_mode = noise.alpha_mode

    fit_results, prefer_centered = _normalize_fit_inputs(
        results,
        init_strategy=init_strategy,
        data_prior_edge_points=data_prior_edge_points,
        data_prior_signal_sigma_scale=data_prior_signal_sigma_scale,
        data_prior_k_prior=data_prior_k_prior,
        data_prior_k_bounds=data_prior_k_bounds,
        data_prior_k_sigma=data_prior_k_sigma,
    )
    ds = next((r.dataset for r in fit_results.values() if r.dataset), None)
    if ds is None:
        msg = "No valid dataset found in results."
        raise ValueError(msg)
    # Across every well, not this one: the well ``next()`` returned may be
    # single-label after per-label detection, and a noise model built from it
    # would omit a label the plate is still fitting. The label list below is a
    # union for exactly this reason; the model has to match it. Which well came
    # first depended on set iteration order, so omitting this failed
    # intermittently rather than always.
    noise_model = (
        _resolve_structured_noise_model(
            noise,
            label_representatives(r.dataset for r in fit_results.values()),
        )
        if noise.kind == "structured"
        else None
    )
    xc = next(iter(ds.values())).xc
    head_da = next(iter(ds.values()))
    # Latent-x modeling needs an SD per step; a dataset with no x_errc
    # historically walked the x_err property's +-1 default (plot_fit used to
    # materialize it as a side effect). Keep that behavior here explicitly so
    # figures can stop arming datasets.
    x_errc = head_da.x_errc if head_da.x_errc.size else np.ones_like(head_da.xc)
    # Union over wells, not this one well's labels: per-label detection can
    # leave the first well single-label, and taking the list from it would
    # silently drop every observation of the other label on the plate.
    labels = list(
        dict.fromkeys(
            lbl
            for r in fit_results.values()
            if r.dataset is not None
            for lbl in r.dataset
        )
    ) or list(ds.keys())
    # The seeding least-squares fits already say how much the scatter varies
    # along the titration, so the pH-axis prior is read off them rather than
    # guessed. Only needed for the parameterization that has a pH axis.
    step_tau_hint: dict[str, float] | None = None
    if ye_mag_parameterization == "separable_step":
        frames = [r.residuals for r in fit_results.values() if r.result is not None]
        if frames:
            step_tau_hint = step_tau_hint_from_residuals(
                pd.concat(frames, ignore_index=True), labels
            )
    if well_noise_scale_sigma is not None:
        well_noise_sd_sigma = well_noise_scale_sigma
    # Resolving one knob from another couples two factors that callers expect to
    # set independently: a structured noise model silently switches per-well
    # ye_mags on while a uniform one leaves them off, so "structured vs uniform"
    # cannot be separated from "per-well vs per-label" unless the caller passes
    # this explicitly. That confound went unnoticed through a whole campaign.
    # The behaviour is kept for compatibility, but it now says so.
    if noise_model is not None and per_well_ye_mags is None:
        warnings.warn(
            "per_well_ye_mags was not given, so it is being resolved from "
            f"learn_ye_mags ({learn_ye_mags!r}) because a structured noise model "
            "is in use. This couples the noise base to the ye_mag structure. "
            "Pass per_well_ye_mags explicitly to keep the two independent.",
            DeprecationWarning,
            stacklevel=2,
        )
    use_per_well_ye_mags = (
        learn_ye_mags
        if noise_model is not None and per_well_ye_mags is None
        else bool(per_well_ye_mags)
    )
    floor_mode, gain_mode, alpha_mode = _resolve_noise_modes(
        prefer_centered=prefer_centered,
        floor_mode=floor_mode,
        gain_mode=gain_mode,
        alpha_mode=alpha_mode,
    )
    values: dict[str, list[float | None]] = {}
    stderr: dict[str, list[float | None]] = {}

    for name, wells in scheme.names.items():
        values[name] = [
            v.result.params["K"].value
            for well, v in fit_results.items()
            if v.result and well in wells
        ]
        stderr[name] = [
            v.result.params["K"].stderr
            for well, v in fit_results.items()
            if v.result and well in wells
        ]
    ctr_ks = weighted_stats(values, stderr)
    is_ph = next(
        (fr.dataset.is_ph for fr in fit_results.values() if fr.dataset is not None),
        True,
    )
    # A Kd is sampled in log Kd between the floor and 100x the top
    # concentration (log_kd_prior); a pKa keeps its Normal.
    kd_range = None if is_ph else kd_bounds(float(np.nanmax(xc)))
    # A pH control group without any finite preliminary K -- replicates that
    # all failed -- has nothing to centre a K prior on, and its wells are left
    # out of the fit. A chloride group without one is a construct that does
    # not bind (V224Q: Kd is infinite), which is a result, not a failure: it
    # is fitted under a log-Kd prior spanning the whole range, and the report
    # says "does not bind" when the data put Kd beyond the titration.
    no_k = [name for name in scheme.names if name not in ctr_ks]
    if no_k and is_ph:
        dropped = {w for name in no_k for w in scheme.names[name]}
        warnings.warn(
            f"Control group(s) {no_k} have no finite K estimate; their wells "
            f"{sorted(dropped & set(fit_results))} are left out of the multi-well fit.",
            stacklevel=2,
        )
        fit_results = {k: v for k, v in fit_results.items() if k not in dropped}
    active_wells = {key for key, r in fit_results.items() if r.result and r.dataset}
    wells_list = [
        key
        for key in fit_results
        if fit_results[key].result and fit_results[key].dataset
    ]
    n_wells = len(wells_list)
    n_steps = len(xc)

    # Split wells into individually-fit (vectorized K_free) and shared-group K.
    # A hierarchical control group is neither: its replicates get their own K
    # around the group's, a distance sigma_w apart, so the group's spread is
    # estimated rather than assumed zero (shared) or infinite (free).
    hierarchical_ctr = ctr_sigma_w_prior is not None
    if hierarchical_ctr and kd_range is not None:
        msg = "ctr_sigma_w_prior is a pH-titration option; Kd fits have no group spread"
        raise ValueError(msg)
    free_wells, k_free_mu, k_free_sigma, shared_of = _free_k_init(
        fit_results,
        wells_list,
        scheme,
        ctr_ks,
        n_sd,
        ctr_free_k=ctr_free_k and not hierarchical_ctr,
        kd_range=kd_range,
    )
    # Controls keep their own K, so they join free_well; their prior is built
    # from the group K below instead of from their own preliminary fit.
    unknown_wells = list(free_wells)
    ctr_wells = (
        [key for key in wells_list if key in shared_of] if hierarchical_ctr else []
    )
    if hierarchical_ctr:
        free_wells = unknown_wells + ctr_wells

    coords: dict[str, list[int] | list[str]] = {
        "well": wells_list,
        "step": list(range(n_steps)),
        "step_diff": list(range(n_steps - 1)),
    }
    if free_wells:
        coords["free_well"] = free_wells
    if ctr_wells:
        coords["unknown_well"] = unknown_wells
        coords["ctr_well"] = ctr_wells

    with pm.Model(coords=coords):
        robust_nu = (
            _student_t_nu_value(student_t_nu)
            if robust_on and robust_likelihood == "student_t"
            else 3.0
        )
        if robust_on and robust_likelihood == "mixture":
            pi_outliers, outlier_inflate = _build_outlier_priors(
                labels, contamination_frac_prior
            )
        else:
            pi_outliers, outlier_inflate = {}, None

        if x_error_model not in {"deterministic", "per_well"}:
            msg = (
                f"Unknown x_error_model {x_error_model!r}; valid options are "
                "'deterministic' and 'per_well' "
                "('hierarchical_per_well' was removed)."
            )
            raise ValueError(msg)

        if x_error_model == "per_well" and n_xerr > 0:
            direction, x_start_sigma, step_nominal, step_sigmas, min_steps = (
                _pipetting_walk_params(xc, x_errc, n_xerr, min_x_step=min_x_step)
            )
            x_start = _build_multi_x_start(
                xc,
                x_start_sigma,
                x_start_between_sigma,
                learn_between=learn_x_start_between,
            )
            x_step = pm.TruncatedNormal(
                "x_step",
                mu=step_nominal[:, None],
                sigma=np.maximum(step_sigmas, 1e-6)[:, None],
                lower=min_steps[:, None],
                shape=(n_steps - 1, n_wells),
                dims=("step_diff", "well"),
            )
            x_cumsum = pm.math.cumsum(x_step, axis=0)
            x_offsets = pm.math.concatenate([pt.zeros((1, n_wells)), x_cumsum], axis=0)
            x_w_all = pm.Deterministic(
                "x_true",
                x_start + direction * x_offsets,
                dims=("step", "well"),
            )

        elif x_error_model == "per_well":
            x_matrix = np.empty((n_steps, n_wells), dtype=float)
            for w_idx, key in enumerate(wells_list):
                well_ds = fit_results[key].dataset
                if well_ds is None:
                    msg = f"Dataset for well {key} is missing."
                    raise ValueError(msg)
                well_x = next(iter(well_ds.values())).xc
                if len(well_x) != n_steps:
                    msg = f"Dataset for well {key} has inconsistent x length."
                    raise ValueError(msg)
                x_matrix[:, w_idx] = np.asarray(well_x, dtype=float)
            x_w_all = pm.Deterministic(
                "x_true",
                as_tensor_variable(x_matrix),
                dims=("step", "well"),
            )

        else:
            x_true: typing.Any = create_x_true(
                xc,
                x_errc,
                n_xerr,
                min_x_step=min_x_step,
            )
            x_w_all = pt.tile(x_true[:, None], (1, n_wells))

        # Vectorized K priors: the individually-fit wells (all unknowns, plus
        # every control replicate when ctr_free_k) share a single K_free RV
        # with dims="free_well" — one graph node instead of a scalar pm.Normal
        # per well, which speeds up compilation and NUTS sampling. Shared
        # control groups keep one scalar K_ctr_* RV each, broadcast across
        # their replicate wells.
        k_params: dict[str, typing.Any] = {}
        if not ctr_free_k or hierarchical_ctr:
            k_params, _ = _build_ctr_k_params(
                scheme,
                ctr_ks,
                active_wells,
                ctr_free_k=False,
                kd_range=kd_range,
                n_sd=n_sd,
            )
        k_free: typing.Any = None  # pt.TensorVariable
        if hierarchical_ctr:
            # Non-centred: a replicate is its group's K plus sigma_w * z, so the
            # sampler never has to move K_ctr and its replicates together.
            # K_free stays the name every reader indexes by free_well.
            k_unknown = (
                pm.Normal(
                    "K_unknown", mu=k_free_mu, sigma=k_free_sigma, dims="unknown_well"
                )
                if unknown_wells
                else pt.zeros(0)
            )
            sigma_w = pm.HalfNormal("K_sigma_w", sigma=float(ctr_sigma_w_prior or 0.1))
            z_ctr = pm.Normal("K_ctr_z", mu=0.0, sigma=1.0, dims="ctr_well")
            group_mu = pt.stack([k_params[shared_of[key]] for key in ctr_wells])
            k_stacked = pt.join(  # type: ignore[no-untyped-call]
                0, k_unknown, group_mu + sigma_w * z_ctr
            )
            k_free = pm.Deterministic("K_free", k_stacked, dims="free_well")
        elif free_wells and kd_range is not None:
            k_free = log_kd_prior(
                "K_free", k_free_mu, k_free_sigma, kd_range, dims="free_well"
            )
        elif free_wells:
            k_free = pm.Normal(
                "K_free", mu=k_free_mu, sigma=k_free_sigma, dims="free_well"
            )
        free_pos = {well: i for i, well in enumerate(free_wells)}
        k_segments: list[typing.Any] = []
        for key in wells_list:
            if key in free_pos:
                assert k_free is not None  # ruff: ignore[assert] — free_pos implies k_free built
                k_segments.append(k_free[free_pos[key]])
            else:
                k_segments.append(k_params[shared_of[key]])
        k_all = pt.stack(k_segments)  # (n_wells,)

        # One shape parameter for the whole plate. Off by default, so every
        # existing fit is bit-for-bit unchanged; K stays the midpoint at any
        # value, which keeps estimates comparable across the two settings.
        hill_n = (
            pm.LogNormal("hill_n", mu=0.0, sigma=hill_prior_sigma)
            if learn_hill
            else 1.0
        )

        # Collect and create vectorized S0 and S1 priors
        s0_vars = {}
        s1_vars = {}
        for lbl in labels:
            mu_s0, sig_s0 = [], []
            mu_s1, sig_s1 = [], []
            # A well may carry only some labels: pre-fit detection drops a dim
            # label rather than the whole well. S0/S1 are vectors over every
            # well, so the missing entries still need a value. They are given a
            # placeholder and excluded from this label's likelihood, so they
            # stay at their prior and touch nothing else.
            missing: list[int] = []
            for w_idx, key in enumerate(wells_list):
                r = fit_results[key]
                if r.result is None:
                    msg = f"Fit result for well {key} is missing."
                    raise ValueError(msg)
                if f"S0_{lbl}" not in r.result.params:
                    missing.append(w_idx)
                    mu_s0.append(np.nan)
                    sig_s0.append(np.nan)
                    mu_s1.append(np.nan)
                    sig_s1.append(np.nan)
                    continue
                p_s0 = r.result.params[f"S0_{lbl}"]
                p_s1 = r.result.params[f"S1_{lbl}"]
                mu_s0.append(p_s0.value)
                sig_s0.append(_safe_sigma(p_s0.stderr, n_sd))
                mu_s1.append(p_s1.value)
                sig_s1.append(_safe_sigma(p_s1.stderr, n_sd))
            if missing:
                # Centre the placeholders on the plate so the sampler explores a
                # sane region rather than an arbitrary one.
                for vec in (mu_s0, sig_s0, mu_s1, sig_s1):
                    arr = np.asarray(vec, dtype=float)
                    fill = float(np.nanmedian(arr)) if np.isfinite(arr).any() else 1.0
                    for i in missing:
                        vec[i] = fill
            s0_vars[lbl] = pm.Normal(
                f"S0_{lbl}", mu=np.array(mu_s0), sigma=np.array(sig_s0), dims="well"
            )
            s1_vars[lbl] = pm.Normal(
                f"S1_{lbl}", mu=np.array(mu_s1), sigma=np.array(sig_s1), dims="well"
            )

        if noise_model is not None:
            active_noise_model = _active_noise_model(noise_model, labels)
            noise_priors = build_pymc_noise_priors(
                active_noise_model,
                shared_alpha=shared_alpha,
                shared_gain=shared_gain,
                shared_floor=shared_floor,
                floor_mode=floor_mode,
                gain_mode=gain_mode,
                alpha_mode=alpha_mode,
            )
            if learn_ye_mags:
                # LogNormal (not HalfNormal) so ye_mag_parameterization applies
                # uniformly across noise modes without a prior/geometry confound.
                ye_mags = _build_multi_ye_mag_priors(
                    labels,
                    per_well=use_per_well_ye_mags,
                    shared_ye_mags=shared_ye_mags,
                    prior="lognormal",
                    mu=0.0,
                    sigma=1.5,
                    parameterization=ye_mag_parameterization,
                    step_tau_sigma=step_tau_hint,
                )
        else:
            noise_priors = {}
            ye_mags = _build_multi_ye_mag_priors(
                labels,
                per_well=use_per_well_ye_mags,
                shared_ye_mags=shared_ye_mags,
                prior=ye_mag_prior,
                mu=ye_mag_mu,
                sigma=ye_mag_sigma,
                parameterization=ye_mag_parameterization,
                step_tau_sigma=step_tau_hint,
            )
        if well_noise_scale:
            label_scale_sigma = max(float(label_noise_scale_sigma), 1e-6)
            well_sd_sigma = max(float(well_noise_sd_sigma), 1e-6)
            well_noise_scales = {}
            if shared_well_noise_scale:
                shared_label_noise_scale = pm.LogNormal(
                    "label_noise_scale",
                    mu=0.0,
                    sigma=label_scale_sigma,
                )
                shared_well_noise_sd = pm.HalfNormal(
                    "well_noise_sd", sigma=well_sd_sigma
                )
                shared_scale = pm.LogNormal(
                    "well_noise_scale",
                    mu=pm_math.log(shared_label_noise_scale),
                    sigma=shared_well_noise_sd,
                    dims="well",
                )
                well_noise_scales = dict.fromkeys(labels, shared_scale)
            else:
                for lbl in labels:
                    label_noise_scale = pm.LogNormal(
                        f"label_noise_scale_{lbl}",
                        mu=0.0,
                        sigma=label_scale_sigma,
                    )
                    well_noise_sd = pm.HalfNormal(
                        f"well_noise_sd_{lbl}",
                        sigma=well_sd_sigma,
                    )
                    well_noise_scales[lbl] = pm.LogNormal(
                        f"well_noise_scale_{lbl}",
                        mu=pm_math.log(label_noise_scale),
                        sigma=well_noise_sd,
                        dims="well",
                    )
        else:
            well_noise_scales = {}

        # Likelihoods
        first_ds = next(iter(fit_results.values())).dataset
        if first_ds is None:
            msg = "At least one dataset is required."
            raise ValueError(msg)
        is_ph = first_ds.is_ph
        acid_factor: typing.Any = None
        if acid_scale:
            if not is_ph:
                msg = "acid_scale applies to pH titrations only"
                raise ValueError(msg)
            acid_row = int(np.argmin(np.asarray(xc, dtype=float)))
            acid = pm.LogNormal(
                "acid_scale", mu=0.0, sigma=float(acid_scale_sigma), dims="well"
            )
            row = np.zeros((n_steps, 1))
            row[acid_row] = 1.0
            acid_factor = 1.0 + as_tensor_variable(row) * (acid[None, :] - 1.0)
        for lbl in labels:
            mask_lbl, y_obs_full, y_err_full = _masked_obs_err_matrices(
                fit_results, wells_list, lbl, n_steps
            )

            y_model_all = binding_1site(
                x_w_all, k_all, s0_vars[lbl], s1_vars[lbl], is_ph=is_ph, hill=hill_n
            )
            if acid_factor is not None:
                y_model_all *= acid_factor
            mu_vec = y_model_all[mask_lbl]
            y_obs_vec = y_obs_full[mask_lbl]

            if noise_model is not None:
                # Heteroscedastic noise model
                noise_var = get_pymc_variance(
                    y_model_all, lbl, active_noise_model, noise_priors
                )
                sigma_obs_all = pm_math.sqrt(noise_var)
                if learn_ye_mags:
                    sigma_obs_all = ye_mags[lbl] * sigma_obs_all
                if well_noise_scale:
                    sigma_obs_all *= well_noise_scales[lbl][None, :]

                sigma_obs_all_det = pm.Deterministic(
                    f"sigma_obs_{lbl}", sigma_obs_all, dims=("step", "well")
                )
                sigma_vec = sigma_obs_all_det[mask_lbl]
            else:
                # Homoscedastic (scaled) noise model
                y_err_all = resolve_y_err(
                    np.asarray(y_err_full, dtype=float), label=lbl
                )
                sigma_all = ye_mags[lbl] * y_err_all
                if well_noise_scale:
                    sigma_all *= well_noise_scales[lbl][None, :]
                sigma_obs_all_det = pm.Deterministic(
                    f"sigma_obs_{lbl}", sigma_all, dims=("step", "well")
                )
                sigma_vec = sigma_obs_all_det[mask_lbl]

            if robust_on and robust_likelihood == "mixture":
                _add_mixture_likelihood(
                    f"y_likelihood_{lbl}",
                    mu_vec,
                    sigma_vec,
                    y_obs_vec,
                    pi_outlier=pi_outliers[lbl],
                    outlier_inflate=outlier_inflate,
                )
            elif robust_on:
                pm.StudentT(
                    f"y_likelihood_{lbl}",
                    nu=robust_nu,
                    mu=mu_vec,
                    sigma=sigma_vec,
                    observed=y_obs_vec,
                )
            else:
                pm.Normal(
                    f"y_likelihood_{lbl}",
                    mu=mu_vec,
                    sigma=sigma_vec,
                    observed=y_obs_vec,
                )

        tune_steps = n_tune if n_tune is not None else n_samples // 2

        target_accept_ = (
            target_accept if target_accept is not None else 0.95 if n_xerr > 0 else 0.9
        )
        trace: xr.DataTree = _sample_trace(
            n_samples, tune=tune_steps, target_accept=target_accept_, sampler=sampler
        )

        if sample_ppc:
            pm.sample_posterior_predictive(trace, extend_inferencedata=True)

        if sampler.compute_log_likelihood:
            trace = _compute_sample_log_likelihood(trace)

    global_names: list[str] = []
    if learn_hill:
        global_names.append("hill_n")
    if robust_on and robust_likelihood == "student_t" and student_t_nu is None:
        global_names.append("student_t_nu")
    if robust_on and robust_likelihood == "mixture":
        global_names.extend(f"pi_outlier_{lbl}" for lbl in labels)
        global_names.append("outlier_inflate")

    return MultiFitResult(
        trace=trace,
        results=_per_well_fit_results_from_trace(
            trace,
            fit_results,
            scheme,
            x_error_model=x_error_model,
            global_p_names=global_names,
        ),
    )


def fit_binding_pymc_multi_screened(
    results: Mapping[str, Dataset | FitResult],
    scheme: PlateScheme,
    *,
    criterion: OutlierCriterion | None = None,
    min_keep: int = 3,
    screening_robust: RobustConfig | None = None,
    **kwargs: typing.Any,  # ruff: ignore[any-type]
) -> MultiFitResult:
    """Fit the multi-well model twice, screening residual outliers in between.

    The first pass fits every well jointly and scores its own residuals; the
    points *criterion* rejects are masked, and the second pass refits the
    remaining data. Screening after a joint fit rather than before it means the
    residuals are measured against the shared-K model that will be reported,
    not against a per-well prefit whose own error is larger.

    Deleting points is not free - it lowers the reduced chi-square and so the
    reported uncertainty - so prefer a robust likelihood
    (``robust=RobustConfig(enabled=True)``) unless a holdout comparison shows
    screening earns its place.

    Returns a :class:`MultiFitResult` exactly as
    :func:`fit_binding_pymc_multi` does, so a grid can point at either. What
    the screen excluded is recoverable from the returned per-well dataset
    masks.

    Parameters
    ----------
    results : Mapping[str, Dataset | FitResult]
        Per-well datasets or initial fit results, as for
        :func:`fit_binding_pymc_multi`.
    scheme : PlateScheme
        Plate scheme defining the control groups.
    criterion : OutlierCriterion | None
        What counts as an outlier in the screening pass's residual table.
        ``None`` uses :class:`RobustZMad`, scoring each well and label against
        its own median/MAD scale.
    min_keep : int
        Minimum unmasked points retained per label, however many the criterion
        flags. The mildest flagged points survive when this binds.
    screening_robust : RobustConfig | None
        Likelihood for the screening pass. ``None`` screens with a Student-t,
        whose wide tails keep an outlier from dragging the fit toward itself and
        hiding in the residuals. The refit uses whatever *kwargs* specify.
    **kwargs : typing.Any
        Forwarded unchanged to both passes.

    Returns
    -------
    MultiFitResult
        The refit pass's result.
    """
    screening_kwargs = dict(kwargs)
    screening_kwargs["robust"] = screening_robust or RobustConfig(enabled=True)
    initial = fit_binding_pymc_multi(results, scheme, **screening_kwargs)

    residuals = mark_outliers(
        initial.residual_table(), criterion or RobustZMad(), exclude_col=EXCLUDE_COL
    )
    masked = apply_exclusions(initial.results, residuals, min_keep=min_keep)
    return fit_binding_pymc_multi(masked, scheme, **kwargs)
