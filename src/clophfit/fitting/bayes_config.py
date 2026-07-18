"""Configuration objects for the PyMC binding-fit entry points.

These frozen dataclasses group the many keyword parameters of
:func:`clophfit.fitting.bayes.fit_binding_pymc` into cohesive, self-documenting
bundles.  They live in a dedicated module (importing only from
:mod:`clophfit.fitting.data_structures`) so :mod:`clophfit.fitting.bayes` can
re-export them without a circular import.
"""

from __future__ import annotations

import typing
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

if typing.TYPE_CHECKING:
    from clophfit.fitting.data_structures import PlateNoiseModel

NoiseParamMode = Literal["fixed", "free", "centered"]
RobustLikelihood = Literal["student_t", "mixture"]
InitStrategy = Literal["lmfit", "data_priors"]
DataKPrior = Literal["midpoint_truncnorm", "uniform"]
ChainMethod = Literal["auto", "parallel", "vectorized"]
ContaminationFracPrior = float | Mapping[str, float]

_MIN_CONTAMINATION_FRAC_PRIOR = 1e-3
_MAX_CONTAMINATION_FRAC_PRIOR = 0.5


def _validate_robust_likelihood(robust_likelihood: RobustLikelihood) -> None:
    """Validate the robust likelihood selector."""
    if robust_likelihood not in typing.get_args(RobustLikelihood):
        msg = "robust_likelihood must be 'student_t' or 'mixture'."
        raise ValueError(msg)


def _validate_contamination_frac_prior(contamination_frac_prior: float) -> float:
    """Return a valid prior mean for the outlier fraction."""
    contamination_frac = float(contamination_frac_prior)
    if not (
        _MIN_CONTAMINATION_FRAC_PRIOR
        <= contamination_frac
        <= _MAX_CONTAMINATION_FRAC_PRIOR
    ):
        msg = (
            "contamination_frac_prior must be between "
            f"{_MIN_CONTAMINATION_FRAC_PRIOR:g} and "
            f"{_MAX_CONTAMINATION_FRAC_PRIOR:g}."
        )
        raise ValueError(msg)
    return contamination_frac


def _validate_contamination_frac_prior_spec(
    contamination_frac_prior: ContaminationFracPrior,
) -> None:
    """Validate a scalar or per-label contamination-fraction specification."""
    if isinstance(contamination_frac_prior, Mapping):
        for value in contamination_frac_prior.values():
            _validate_contamination_frac_prior(value)
    else:
        _validate_contamination_frac_prior(contamination_frac_prior)


@dataclass(frozen=True)
class SamplerConfig:
    """NUTS sampling controls forwarded to ``pm.sample``.

    Parameters
    ----------
    n_samples : int
        Number of posterior draws per chain.
    nuts_sampler : str
        NUTS backend. Defaults to ``"nutpie"`` — its warmup/mass-matrix
        adaptation is robust on the multi-well latent-x geometry. ``"pymc"``
        (built-in NUTS, Numba backend) is fine and marginally faster on simple
        deterministic-x fits. ``"default"`` lets
        PyMC auto-select whatever is installed; ``"numpyro"``/``"blackjax"``
        (JAX/GPU) request those explicitly but are unusable here (numpyro lacks
        an ``erfcx`` implementation; blackjax is incompatible with PyMC 6.1).
    n_tune : int | None
        Number of tuning draws. ``None`` uses ``n_samples // 2``.
    target_accept : float | None
        Target acceptance probability. ``None`` selects a latent-x-aware
        default.
    max_treedepth : int | None
        Maximum NUTS tree depth. ``None`` uses the backend default.
    random_seed : int | None
        Seed forwarded to ``pm.sample`` for reproducible draws. ``None`` (the
        default) leaves sampling nondeterministic.
    chains : int | None
        Number of MCMC chains. ``None`` uses the PyMC default (4).
    cores : int | None
        Number of CPU cores for parallel chains. ``None`` uses the PyMC
        default (min of chains and available cores). Set to 1 on a laptop to
        avoid multiprocessing overhead, or higher on a many-core server.
    chain_method : ChainMethod
        Chain execution strategy for the JAX backends (``"blackjax"``,
        ``"numpyro"``). ``"auto"`` selects ``"vectorized"`` so all chains run
        on a single GPU (``jax.vmap``); ``"parallel"`` maps chains across
        devices (``jax.pmap``). Ignored by CPU backends.
    compute_log_likelihood : bool
        Compute the per-observation ``log_likelihood`` group after sampling.
        Disabled by default because it adds a full post-sampling pass and is
        only needed for out-of-sample model comparison (``az.loo`` /
        ``az.compare``). Enable it when a downstream comparison requires it.
    backend : str | None
        Compilation backend forwarded to ``pm.sample(backend=...)``, e.g.
        ``"jax"`` to JIT-compile the model's logp/dlogp through JAX (GPU) or
        ``"numba"``. ``None`` uses the sampler's default. Cannot be combined
        with ``compile_kwargs={"mode": ...}``.
    compile_kwargs : Mapping[str, typing.Any] | None
        Extra keyword arguments forwarded to the functions compiled by the
        step methods (``pm.sample(compile_kwargs=...)``), e.g. a custom
        ``{"mode": ...}``. ``None`` uses the default. Note: use *backend* to
        select JAX/Numba — ``"backend"`` is not a valid ``compile_kwargs`` key.
    """

    n_samples: int = 2000
    nuts_sampler: str = "nutpie"
    n_tune: int | None = None
    target_accept: float | None = None
    max_treedepth: int | None = None
    random_seed: int | None = None
    chains: int | None = None
    cores: int | None = None
    chain_method: ChainMethod = "auto"
    compute_log_likelihood: bool = False
    backend: str | None = None
    compile_kwargs: Mapping[str, typing.Any] | None = None


@dataclass(frozen=True)
class RobustConfig:
    """Robust-likelihood configuration.

    Parameters
    ----------
    enabled : bool
        Use a robust likelihood instead of a plain Normal.
    likelihood : RobustLikelihood
        ``"student_t"`` heavy-tailed likelihood or ``"mixture"`` Normal/outlier
        contamination mixture.
    nu : float | None
        Student-t degrees of freedom. Positive values are fixed; ``None``
        infers ``student_t_nu`` with support above 2.
    contamination_frac_prior : ContaminationFracPrior
        Prior mean for per-label outlier fractions when
        ``likelihood="mixture"``. A mapping supplies label-specific means. Each
        value must be between 0.001 and 0.5.
    """

    enabled: bool = False
    likelihood: RobustLikelihood = "student_t"
    nu: float | None = 3.0
    contamination_frac_prior: ContaminationFracPrior = 0.15

    def __post_init__(self) -> None:
        """Validate the likelihood selector and contamination specification."""
        _validate_robust_likelihood(self.likelihood)
        if self.enabled and self.likelihood == "mixture":
            _validate_contamination_frac_prior_spec(self.contamination_frac_prior)


@dataclass(frozen=True)
class InitConfig:
    """Prior-initialization strategy for the binding parameters.

    Parameters
    ----------
    strategy : InitStrategy
        ``"lmfit"`` fits a raw ``Dataset`` with LMFit first and centers PyMC
        priors on that result. ``"data_priors"`` skips LMFit and derives weak
        priors directly from the observed titration endpoints and midpoint.
    edge_points : int
        Number of active points averaged at each titration edge to initialize
        ``S0``/``S1`` when ``strategy="data_priors"``.
    signal_sigma_scale : float
        Prior sigma for ``S0``/``S1`` as a fraction of each label's observed
        signal range when ``strategy="data_priors"``.
    k_prior : DataKPrior
        K prior family for ``strategy="data_priors"``.
    k_bounds : tuple[float, float] | None
        Lower and upper K bounds for data-derived priors. ``None`` resolves to
        ``(4.5, 9.0)`` for pH datasets or ``(1e-6, 1e6)`` otherwise.
    k_sigma : float
        Truncated-Normal K prior sigma for data-derived priors.
    """

    strategy: InitStrategy = "lmfit"
    edge_points: int = 2
    signal_sigma_scale: float = 0.5
    k_prior: DataKPrior = "midpoint_truncnorm"
    k_bounds: tuple[float, float] | None = None
    k_sigma: float = 1.5


@dataclass(frozen=True)
class NoiseConfig:
    """Observation-noise configuration.

    Prefer the :meth:`ye_mag` and :meth:`structured` factories over the raw
    constructor. ``NoiseConfig()`` is equivalent to :meth:`ye_mag`.

    Two mutually exclusive noise families are supported, selected by *kind*:

    - ``"ye_mag"`` scales each label's supplied ``y_err`` by a learned
      multiplier (``sigma = ye_mag * y_err``).
    - ``"structured"`` builds a floor-plus-Poisson-plus-proportional noise
      model. When *noise_model* is ``None`` the model is synthesized from the
      data using the *floor*/*gain*/*alpha* scale hints, so the
      ``*_mode`` selectors work without hand-building a
      :class:`~clophfit.fitting.data_structures.PlateNoiseModel`.

    Notes
    -----
    A ``*_mode`` of ``None`` resolves at fit time to ``"centered"`` for pre-fit
    ``FitResult`` input and ``"free"`` for raw ``Dataset`` input.
    """

    kind: Literal["ye_mag", "structured"] = "ye_mag"
    # structured-noise fields
    noise_model: PlateNoiseModel | None = None
    floor_mode: NoiseParamMode | None = None
    gain_mode: NoiseParamMode | None = None
    alpha_mode: NoiseParamMode | None = None
    shared_alpha: bool = False
    shared_gain: bool = False
    shared_floor: bool = False
    floor: float | Mapping[str, float] | None = None
    gain: float | Mapping[str, float] = 0.0
    alpha: float | Mapping[str, float] = 0.02
    learn_ye_mags: bool = False
    # ye_mag-multiplier fields
    shared_ye_mags: bool = False
    ye_mag_prior: Literal["halfnormal", "lognormal"] = "lognormal"
    ye_mag_mu: float | Mapping[str, float] = 0.0
    ye_mag_sigma: float | Mapping[str, float] = 1.5

    @classmethod
    def ye_mag(
        cls,
        *,
        shared: bool = False,
        prior: Literal["halfnormal", "lognormal"] = "lognormal",
        mu: float | Mapping[str, float] = 0.0,
        sigma: float | Mapping[str, float] = 1.5,
    ) -> NoiseConfig:
        """Scale supplied ``y_err`` by a learned ``ye_mag`` multiplier."""
        return cls(
            kind="ye_mag",
            shared_ye_mags=shared,
            ye_mag_prior=prior,
            ye_mag_mu=mu,
            ye_mag_sigma=sigma,
        )

    @classmethod
    def structured(  # noqa: PLR0913
        cls,
        *,
        noise_model: PlateNoiseModel | None = None,
        floor_mode: NoiseParamMode | None = None,
        gain_mode: NoiseParamMode | None = None,
        alpha_mode: NoiseParamMode | None = None,
        shared_alpha: bool = False,
        shared_gain: bool = False,
        shared_floor: bool = False,
        floor: float | Mapping[str, float] | None = None,
        gain: float | Mapping[str, float] = 0.0,
        alpha: float | Mapping[str, float] = 0.02,
        learn_ye_mags: bool = False,
        shared_ye_mags: bool = False,
    ) -> NoiseConfig:
        """Floor-plus-Poisson-plus-proportional noise model.

        When *noise_model* is ``None`` the model is synthesized from the data:
        each label gets ``sigma_floor`` from *floor* (fallback: the label's
        ``y_err`` scale), plus *gain* and *alpha*. Setting ``gain_mode="free"``
        or ``alpha_mode="free"`` activates those terms.

        The *alpha* hint is the prior scale, not a hard value: in ``"free"``
        mode it is the ``HalfNormal`` sigma and in ``"centered"`` mode the
        ``TruncatedNormal`` mean. It defaults to ``0.02`` (a weak 2% prior);
        pass ``alpha=0`` for the tightest around-zero prior (floored at
        ``1e-3``), a larger value to widen it, or ``alpha_mode="fixed"`` to
        pin/disable the term.
        """
        return cls(
            kind="structured",
            noise_model=noise_model,
            floor_mode=floor_mode,
            gain_mode=gain_mode,
            alpha_mode=alpha_mode,
            shared_alpha=shared_alpha,
            shared_gain=shared_gain,
            shared_floor=shared_floor,
            floor=floor,
            gain=gain,
            alpha=alpha,
            learn_ye_mags=learn_ye_mags,
            shared_ye_mags=shared_ye_mags,
        )
