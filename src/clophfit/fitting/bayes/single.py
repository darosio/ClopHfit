"""Single-well-group PyMC fit (fit_binding_pymc) and its CTR helpers."""

from __future__ import annotations

import copy
import typing
import warnings

import numpy as np
import pymc as pm  # type: ignore[import-untyped]
from pymc import math as pm_math
from pytensor.tensor import as_tensor_variable

from clophfit.fitting.bayes_config import (
    InitConfig,
    NoiseConfig,
    RobustConfig,
    SamplerConfig,
)
from clophfit.fitting.data_structures import (
    DataArray,
    Dataset,
    FitResult,
)
from clophfit.fitting.models import binding_1site, kd_bounds

if typing.TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from clophfit.clophfit_types import ArrayF
    from clophfit.prtecan import PlateScheme

from clophfit.fitting.bayes.axis import (
    _safe_sigma,
    create_x_true,
)
from clophfit.fitting.bayes.extract import (
    process_trace,
)
from clophfit.fitting.bayes.likelihood import (
    _add_y_likelihood,
    _build_outlier_priors,
    _resolve_structured_noise_model,
    _student_t_nu_value,
)
from clophfit.fitting.bayes.noise import (
    _active_noise_model,
    _build_ye_mag_priors,
    _resolve_noise_modes,
    build_pymc_noise_priors,
    get_pymc_variance,
)
from clophfit.fitting.bayes.priors import (
    _ctr_param_name,
    _normalize_fit_input,
    create_data_parameter_priors,
    create_parameter_priors,
    log_kd_prior,
    log_kd_prior_params,
)
from clophfit.fitting.bayes.sampling import (
    _compute_sample_log_likelihood,
    _sample_trace,
)

_DEFAULT_NOISE = NoiseConfig()
_DEFAULT_ROBUST = RobustConfig()
_DEFAULT_INIT = InitConfig()
_DEFAULT_SAMPLER = SamplerConfig()


def fit_binding_pymc(  # ruff: ignore[too-many-branches, too-many-arguments, too-many-statements]
    ds_or_fr: Dataset | FitResult,
    *,
    n_sd: float = 10.0,
    n_xerr: float = 1.0,
    min_x_step: float = 0.2,
    noise: NoiseConfig = _DEFAULT_NOISE,
    robust: RobustConfig = _DEFAULT_ROBUST,
    init: InitConfig = _DEFAULT_INIT,
    sampler: SamplerConfig = _DEFAULT_SAMPLER,
    x_sigma: float | None = None,
) -> FitResult:
    """Analyze multi-label titration datasets using PyMC (single model).

    Parameters
    ----------
    ds_or_fr : Dataset | FitResult
        Either a ``Dataset`` (an initial LS fit is run) or a ``FitResult`` whose
        params seed the PyMC priors.
    n_sd : float
        Number of standard deviations for parameter priors.
    n_xerr : float
        Scaling factor for x-error.
    min_x_step : float
        Minimum inferred change in ``x`` between consecutive titration steps
        when latent-x modeling is enabled.
    noise : NoiseConfig
        Observation-noise configuration. Default scales ``y_err`` by a learned
        ``ye_mag`` multiplier; use :meth:`NoiseConfig.structured` for a
        floor/gain/alpha noise model. See :class:`NoiseConfig`.
    robust : RobustConfig
        Robust-likelihood configuration (Student-t or contamination mixture).
        See :class:`RobustConfig`.
    init : InitConfig
        Prior-initialization strategy (``"lmfit"`` prefit or ``"data_priors"``).
        See :class:`InitConfig`.
    sampler : SamplerConfig
        NUTS sampling controls.
    x_sigma : float | None
        Independent per-point prior on the true x: ``x_true[i] ~ N(x[i], x_sigma)``
        for every point separately. For axes that are sets of independent
        measurements rather than a cumulative addition series - e.g. EnSpire
        wells, each set to its own nominal pH - the random-walk model below is
        not the right structure, and this is.
        ``None`` (default) leaves the axis fixed unless ``n_xerr > 0`` enables
        the walk. Takes precedence over ``n_xerr`` when set.

    Returns
    -------
    FitResult
        Bayesian fitting results.
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

    fr, prefer_centered = _normalize_fit_input(
        ds_or_fr,
        init_strategy=init.strategy,
        data_prior_edge_points=init.edge_points,
        data_prior_signal_sigma_scale=init.signal_sigma_scale,
        data_prior_k_prior=init.k_prior,
        data_prior_k_bounds=init.k_bounds,
        data_prior_k_sigma=init.k_sigma,
    )

    if fr.result is None or fr.dataset is None:
        return FitResult()
    params = fr.result.params
    ds = copy.deepcopy(fr.dataset)
    labels = list(ds.keys())
    xc = next(iter(ds.values())).xc
    head_da = next(iter(ds.values()))
    # Latent-x modeling needs an SD per step; a dataset with no x_errc
    # historically walked the x_err property's +-1 default (plot_fit used to
    # materialize it as a side effect). Keep that behavior here explicitly so
    # figures can stop arming datasets.
    x_errc = head_da.x_errc if head_da.x_errc.size else np.ones_like(head_da.xc)
    floor_mode, gain_mode, alpha_mode = _resolve_noise_modes(
        prefer_centered=prefer_centered,
        floor_mode=noise.floor_mode,
        gain_mode=noise.gain_mode,
        alpha_mode=noise.alpha_mode,
    )
    with pm.Model():
        pars = (
            create_data_parameter_priors(
                params,
                k_prior=init.k_prior,
                k_bounds=init.k_bounds,
                is_ph=ds.is_ph,
            )
            if init.strategy == "data_priors"
            else create_parameter_priors(
                params,
                n_sd,
                kd_range=None if ds.is_ph else kd_bounds(float(np.nanmax(xc))),
            )
        )
        if x_sigma is not None and x_sigma > 0:
            # Independent per-point prior: each measured x anchors its own
            # true value; there is no shared walk between points. The EnSpire
            # layout (replicate rows of independent nominal targets) is a set,
            # not a cumulative series, so this, not create_x_true, is its
            # latent-x model.
            x_true = pm.Normal(
                "x_true",
                mu=as_tensor_variable(np.asarray(xc, dtype=float)),
                sigma=float(x_sigma),
                shape=len(xc),
            )
        else:
            x_true = create_x_true(xc, x_errc, n_xerr, min_x_step=min_x_step)
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

        if noise.kind == "ye_mag":
            ye_mags = _build_ye_mag_priors(
                labels,
                shared_ye_mags=shared_ye_mags,
                prior=ye_mag_prior,
                mu=ye_mag_mu,
                sigma=ye_mag_sigma,
            )
            for lbl, da in ds.items():
                y_model = binding_1site(
                    x_true,
                    pars["K"],
                    pars[f"S0_{lbl}"],
                    pars[f"S1_{lbl}"],
                    is_ph=ds.is_ph,
                )
                sigma = ye_mags[lbl] * da.y_err
                _add_y_likelihood(
                    f"y_likelihood_{lbl}",
                    y_model,
                    da,
                    sigma,
                    robust=robust_on,
                    student_t_nu=robust_nu,
                    robust_likelihood=robust_likelihood,
                    pi_outlier=pi_outliers.get(lbl),
                    outlier_inflate=outlier_inflate,
                )
        else:
            noise_model = _resolve_structured_noise_model(noise, ds)
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
                ye_mags = _build_ye_mag_priors(
                    labels,
                    shared_ye_mags=shared_ye_mags,
                    prior="halfnormal",
                    sigma=5.0,
                )

            for lbl, da in ds.items():
                y_model = binding_1site(
                    x_true,
                    pars["K"],
                    pars[f"S0_{lbl}"],
                    pars[f"S1_{lbl}"],
                    is_ph=ds.is_ph,
                )
                noise_var = get_pymc_variance(
                    y_model, lbl, active_noise_model, noise_priors
                )
                sigma = pm_math.sqrt(noise_var)
                if learn_ye_mags:
                    sigma = ye_mags[lbl] * sigma

                sigma_det = pm.Deterministic(f"sigma_obs_{lbl}", sigma)
                _add_y_likelihood(
                    f"y_likelihood_{lbl}",
                    y_model,
                    da,
                    sigma_det[da.mask],
                    robust=robust_on,
                    student_t_nu=robust_nu,
                    robust_likelihood=robust_likelihood,
                    pi_outlier=pi_outliers.get(lbl),
                    outlier_inflate=outlier_inflate,
                )

        # Inference
        tune = sampler.n_tune if sampler.n_tune is not None else n_samples // 2
        target_accept_ = (
            sampler.target_accept
            if sampler.target_accept is not None
            else 0.95
            if n_xerr > 0
            else 0.9
        )
        trace = _sample_trace(
            n_samples, tune=tune, target_accept=target_accept_, sampler=sampler
        )
        if sampler.compute_log_likelihood:
            trace = _compute_sample_log_likelihood(trace)
    p_names = list(params.keys())
    if robust_on and robust_likelihood == "student_t" and student_t_nu is None:
        p_names.append("student_t_nu")
    if robust_on and robust_likelihood == "mixture":
        p_names.extend([f"pi_outlier_{lbl}" for lbl in labels])
        p_names.append("outlier_inflate")
    return process_trace(trace, p_names, ds)


# ------------------------------------------------------------------
# Helper: weighted statistics
# ------------------------------------------------------------------


def weighted_stats(
    values: Mapping[str, Sequence[float | None]],
    stderr: Mapping[str, Sequence[float | None]],
    *,
    min_stderr: float = 1e-3,
) -> dict[str, tuple[float, float]]:
    """Weighted mean and stderr for control priors.

    Filters out ``NaN``, ``inf``, and non-positive stderr, and floors
    stderr at *min_stderr* to avoid infinite weights. A sample with no finite
    pair is left out of the result, with a warning, rather than raised on: a
    construct that does not respond (V224Q in a chloride titration has an
    infinite Kd) is a legitimate control, and raising aborted the whole plate.
    Callers must treat a sample missing from the result as unfittable.
    """
    results: dict[str, tuple[float, float]] = {}
    for sample in values:
        pairs = []
        for v, s in zip(values[sample], stderr[sample], strict=True):
            if v is None or s is None:
                continue
            if not (np.isfinite(v) and np.isfinite(s)):
                continue
            s_val = min_stderr if s <= 0 else float(s)
            pairs.append((float(v), s_val))

        if not pairs:
            warnings.warn(
                f"No finite (value, stderr) pair for control {sample!r}: it has "
                "no K estimate (a construct that does not bind, or replicates "
                "that all failed) and is left out of the control priors.",
                stacklevel=2,
            )
            continue

        x, se = map(np.asarray, zip(*pairs, strict=True))
        w = 1.0 / se**2
        results[sample] = (
            float(np.average(x, weights=w)),
            float(np.sqrt(1.0 / w.sum())),
        )
    return results


def _build_ctr_k_params(  # ruff: ignore[too-many-arguments]
    scheme: PlateScheme,
    ctr_ks: dict[str, tuple[float, float]],
    active_wells: set[str],
    *,
    ctr_free_k: bool,
    well_k_init: dict[str, tuple[float, float]] | None = None,
    fallback_sigma: float = 0.6,
    kd_range: tuple[float, float] | None = None,
    n_sd: float = 1.0,
) -> tuple[dict[str, typing.Any], dict[str, typing.Any]]:
    """Build K parameters for control groups.

    Parameters
    ----------
    scheme : PlateScheme
        Plate scheme with named control groups.
    ctr_ks : dict[str, tuple[float, float]]
        Weighted K mean and stderr per control name.
    active_wells : set[str]
        Well keys that have valid results and datasets.
    ctr_free_k : bool
        If True, each CTR replicate gets its own independent K prior
        initialized from its own preliminary fit result (same as unknown
        wells).  If False (default), all replicates of the same CTR share
        a single K.
    well_k_init : dict[str, tuple[float, float]] | None
        Maps well key → ``(K_value, K_sigma)`` from the individual
        preliminary fit or pH@mid-fluorescence.  Used only when
        ``ctr_free_k=True``; wells absent from this dict fall back to the
        group weighted mean with *fallback_sigma*.
    fallback_sigma : float
        Prior sigma used for wells absent from *well_k_init* (free CTR
        mode) or as a floor for the constrained group sigma.  Default 0.6.
    kd_range : tuple[float, float] | None
        For a dissociation constant, the Kd range. A shared group K is then a
        log-Kd prior (:func:`log_kd_prior`) centred on the group estimate, and
        a group with no estimate -- a construct that does not bind -- still
        gets one, spanning the range, as long as it has wells in the fit.
        ``None`` (pH) keeps the Normal and skips groups without an estimate.
    n_sd : float
        Prior width in group standard errors, for *kd_range* only.

    Returns
    -------
    k_params : dict[str, typing.Any]
        Maps CTR name → shared PyMC K variable (populated only when
        ``ctr_free_k=False``).
    k_replicate : dict[str, typing.Any]
        Maps well key → individual PyMC K variable (populated only when
        ``ctr_free_k=True``).
    """
    k_params: dict[str, typing.Any] = {}
    k_replicate: dict[str, typing.Any] = {}
    if ctr_free_k:
        # Per-well K: initialized from each well's own preliminary fit result
        # (same as unknown wells), falling back to group weighted mean if unavailable.
        for name, scheme_wells in scheme.names.items():
            for well in scheme_wells:
                if well in active_wells:
                    if well_k_init and well in well_k_init:
                        mu, sigma = well_k_init[well]
                    else:
                        mu, sigma = ctr_ks[name][0], fallback_sigma
                    k_replicate[well] = pm.Normal(
                        f"K_{name}_{well}", mu=mu, sigma=sigma
                    )
    elif kd_range is not None:
        for name, scheme_wells in scheme.names.items():
            if not set(scheme_wells) & active_wells:
                continue
            k, se = ctr_ks.get(name, (None, None))
            mu, sigma = log_kd_prior_params(k, se, n_sd, kd_range)
            k_params[name] = log_kd_prior(_ctr_param_name(name), mu, sigma, kd_range)
    else:
        # A pH group without a K estimate has no wells in the fit (see
        # fit_binding_pymc_multi), so it gets no shared K either.
        k_params = {
            name: pm.Normal(
                _ctr_param_name(name),
                mu=ctr_ks[name][0],
                sigma=max(ctr_ks[name][1], fallback_sigma / 2),
            )
            for name in scheme.names
            if name in ctr_ks
        }
    return k_params, k_replicate


def _free_k_init(  # ruff: ignore[too-many-arguments]
    fit_results: dict[str, FitResult],
    wells_list: list[str],
    scheme: PlateScheme,
    ctr_ks: dict[str, tuple[float, float]],
    n_sd: float,
    *,
    ctr_free_k: bool,
    fallback_sigma: float = 0.6,
    kd_range: tuple[float, float] | None = None,
) -> tuple[list[str], ArrayF, ArrayF, dict[str, str]]:
    """Classify wells for the multi-fit K prior and return free-well init.

    Wells that receive an *individual* K prior — every unknown well, plus each
    control replicate when *ctr_free_k* is True — are collected into
    ``free_wells`` with aligned prior mean/sigma vectors, so the caller can
    build a single vectorized ``K_free`` random variable (one graph node)
    instead of one scalar ``pm.Normal`` per well.  Wells whose K is shared
    across a control group (``ctr_free_k`` False) are returned in ``shared_of``
    mapping well -> group name; the caller supplies their shared scalar RV.

    The per-well ``(mu, sigma)`` values reproduce the previous scalar-prior
    construction exactly, including the mode-dependent sigma floor.

    Parameters
    ----------
    fit_results : dict[str, FitResult]
        Per-well preliminary fit results.
    wells_list : list[str]
        Ordered active well keys.
    scheme : PlateScheme
        Plate scheme defining control groups.
    ctr_ks : dict[str, tuple[float, float]]
        Weighted K mean and stderr per control group.
    n_sd : float
        Prior width multiplier applied to the preliminary-fit stderr.
    ctr_free_k : bool
        If True, control replicates get individual priors (like unknowns).
    fallback_sigma : float
        Prior sigma for control replicates lacking a reliable preliminary fit.
    kd_range : tuple[float, float] | None
        For a dissociation constant, the Kd range: every free well, control
        or not, then gets a log-Kd prior from its own preliminary fit (see
        :func:`log_kd_prior_params`), and *mu*/*sigma* are in log Kd. The pH
        rules above -- a 0.6 cap, a pH-at-mid-fluorescence fallback -- mean
        nothing for a Kd in mM. ``None`` (pH) keeps them.

    Returns
    -------
    free_wells : list[str]
        Wells receiving an individual (vectorized) K prior, in input order.
    mu : ArrayF
        Prior means aligned with *free_wells* (log Kd under *kd_range*).
    sigma : ArrayF
        Prior sigmas aligned with *free_wells* (log Kd under *kd_range*).
    shared_of : dict[str, str]
        Maps each shared (control-group) well to its group name.

    Raises
    ------
    ValueError
        If an active well in *wells_list* has no preliminary fit result.
    """
    well_to_group: dict[str, str] = {}
    for name, wells in scheme.names.items():
        for well in wells:
            well_to_group.setdefault(well, name)
    well_k_init = (
        _well_k_init_from_results(fit_results, scheme, n_sd)
        if ctr_free_k and kd_range is None
        else {}
    )
    free_wells: list[str] = []
    mus: list[float] = []
    sigmas: list[float] = []
    shared_of: dict[str, str] = {}
    for key in wells_list:
        group = well_to_group.get(key, "")
        if group and not ctr_free_k:
            shared_of[key] = group
            continue
        result = fit_results[key].result
        if result is None:
            msg = f"Fit result for well {key} is missing."
            raise ValueError(msg)
        if kd_range is not None:  # Kd: own preliminary fit, in log Kd
            p = result.params["K"]
            mu, sigma = log_kd_prior_params(p.value, p.stderr, n_sd, kd_range)
        elif group:  # ctr_free_k replicate: own preliminary fit or group fallback
            mu, sigma = well_k_init.get(key, (ctr_ks[group][0], fallback_sigma))
        elif ctr_free_k:  # unknown well (free-CTR mode uses _safe_sigma default)
            p = result.params["K"]
            mu, sigma = p.value, _safe_sigma(p.stderr, n_sd)
        else:  # unknown well (shared-CTR mode uses a 1e-3 sigma floor)
            p = result.params["K"]
            mu = p.value
            sigma = max(p.stderr * n_sd, 1e-3) if p.stderr else 1e-3
        free_wells.append(key)
        mus.append(float(mu))
        sigmas.append(float(sigma))
    return free_wells, np.asarray(mus), np.asarray(sigmas), shared_of


def _resolve_well_k(  # ruff: ignore[too-many-arguments]
    key: str,
    ctr_name: str,
    pars: dict[str, typing.Any],
    k_params: dict[str, typing.Any],
    k_replicate: dict[str, typing.Any],
    *,
    ctr_free_k: bool,
) -> typing.Any:  # ruff: ignore[any-type]
    """Return the PyMC K variable for *key* given the current CTR-K mode."""
    if ctr_free_k:
        k_rep = k_replicate.get(key)
        return k_rep if k_rep is not None else pars[f"K_{key}"]
    return k_params[ctr_name] if ctr_name else pars[f"K_{key}"]


def _ph_at_mid_fluorescence(da: DataArray) -> float | None:
    """Estimate pH at 50 % of fluorescence range by linear interpolation.

    Parameters
    ----------
    da : DataArray
        Titration data (x = pH, y = fluorescence).

    Returns
    -------
    float | None
        Interpolated pH at ``(y_max + y_min) / 2``, or ``None`` if the
        midpoint crossing cannot be found.
    """
    x, y = da.x, da.y
    if len(x) < 2:  # ruff: ignore[magic-value-comparison]
        return None
    y_mid = (float(y.max()) + float(y.min())) / 2
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    for i in range(len(ys) - 1):
        if (ys[i] - y_mid) * (ys[i + 1] - y_mid) <= 0:
            denom = ys[i + 1] - ys[i]
            if denom == 0:
                return float(xs[i])
            t = (y_mid - ys[i]) / denom
            return float(xs[i] + t * (xs[i + 1] - xs[i]))
    return None


def _well_k_init_from_results(
    results: dict[str, FitResult],
    scheme: PlateScheme,
    n_sd: float,
    fallback_sigma: float = 0.6,
) -> dict[str, tuple[float, float]]:
    """Extract per-well K initialization from preliminary fit results.

    Returns a dict mapping each CTR well key to ``(K_value, K_sigma)``.
    Wells whose K is at the optimizer bound or whose ``stderr`` exceeds 1.0
    fall back to a pH-at-mid-fluorescence estimate with *fallback_sigma*.
    For wells with a reliable preliminary fit the sigma is
    ``min(max(sK * n_sd, 0.2), fallback_sigma)``.

    Parameters
    ----------
    results : dict[str, FitResult]
        Preliminary fit results keyed by well name.
    scheme : PlateScheme
        Plate scheme; only wells listed in ``scheme.names`` are processed.
    n_sd : float
        Number of standard deviations used to widen the sigma for reliable
        wells (same as for unknown wells in
        :func:`create_parameter_priors`).
    fallback_sigma : float
        Prior sigma used when the preliminary fit is unreliable (K at
        bound or large stderr).  Also acts as an upper cap for good fits.
        Default 0.6 (covers roughly ±1 pH unit, as suggested by user).

    Returns
    -------
    dict[str, tuple[float, float]]
        Mapping each CTR well key to ``(K_value, K_sigma)``.
    """
    ctr_wells = {well for wells in scheme.names.values() for well in wells}
    well_k: dict[str, tuple[float, float]] = {}
    for well in ctr_wells:
        r = results.get(well)
        if not (r and r.result and "K" in r.result.params):
            continue
        p = r.result.params["K"]
        at_bound = (p.min is not None and abs(p.value - p.min) < 0.01) or (  # ruff: ignore[magic-value-comparison]
            p.max is not None and abs(p.value - p.max) < 0.01  # ruff: ignore[magic-value-comparison]
        )
        large_stderr = p.stderr is None or p.stderr > 1.0
        if at_bound or large_stderr:
            # Pick pH@mid from the label with the largest signal range (Δy),
            # as that label has the clearest sigmoid and most reliable midpoint.
            ph_mid: float | None = None
            if r.dataset:
                best_delta, best_ph = -1.0, None
                for da in r.dataset.values():
                    if len(da.y) < 2:  # ruff: ignore[magic-value-comparison]
                        continue
                    delta = float(da.y.max() - da.y.min())
                    ph = _ph_at_mid_fluorescence(da)
                    if ph is not None and delta > best_delta:
                        best_delta, best_ph = delta, ph
                ph_mid = best_ph
            if ph_mid is not None:
                well_k[well] = (ph_mid, fallback_sigma)
            # else: leave absent → _build_ctr_k_params uses group mean fallback
        else:
            sigma = _safe_sigma(
                p.stderr, n_sd, default=0.2, floor=0.2, cap=fallback_sigma
            )
            well_k[well] = (p.value, sigma)
    return well_k
