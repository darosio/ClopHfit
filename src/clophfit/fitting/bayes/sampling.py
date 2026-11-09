"""PyMC sampler invocation and log-likelihood helpers shared by the fitters."""

from __future__ import annotations

import os
import typing

import pymc as pm  # type: ignore[import-untyped]

if typing.TYPE_CHECKING:
    import xarray as xr

    from clophfit.fitting.bayes_config import (
        ChainMethod,
        SamplerConfig,
    )


def _pymc_sample_parallel_args(
    nuts_sampler: str = "default",
    *,
    chains: int | None = None,
    cores: int | None = None,
    chain_method: ChainMethod = "auto",
) -> dict[str, object]:
    """Return sampling kwargs for ``pm.sample()``.

    Parameters
    ----------
    nuts_sampler : str
        NUTS backend (``"default"``, ``"nutpie"``, ``"blackjax"``,
        ``"numpyro"``). Non-default JAX backends are import-checked here.
    chains : int | None
        Number of MCMC chains. ``None`` leaves the PyMC default.
    cores : int | None
        Number of CPU cores for parallel chains. ``None`` leaves the PyMC
        default.
    chain_method : ChainMethod
        Chain strategy for JAX backends. ``"auto"`` selects ``"vectorized"``
        (all chains on one GPU); ignored by CPU backends.

    Returns
    -------
    dict[str, object]
        Keyword arguments to splat into ``pm.sample()``.

    Raises
    ------
    ImportError
        If *nuts_sampler* names a JAX/nutpie backend whose package is not
        installed.
    """
    kwargs: dict[str, object] = {}
    if nuts_sampler != "default":
        sampler_import_check = {
            "blackjax": ("blackjax", "pip install clophfit[gpu]"),
            "numpyro": ("numpyro", "pip install clophfit[gpu]"),
            "nutpie": ("nutpie", "pip install clophfit[gpu]"),
        }
        if nuts_sampler in sampler_import_check:
            pkg, install_hint = sampler_import_check[nuts_sampler]
            try:
                __import__(pkg)
            except ImportError as e:
                msg = (
                    f"NUTS sampler '{nuts_sampler}' requires package '{pkg}'. "
                    f"Install it with: {install_hint}  "
                    f"(or: uv sync --extra gpu)"
                )
                raise ImportError(msg) from e
        kwargs["nuts_sampler"] = nuts_sampler
        # JAX-based samplers (blackjax, numpyro) use jax.pmap by default which
        # requires N devices for N chains.  On a single GPU, use chain_method=
        # "vectorized" (jax.vmap) so all chains run on one device.
        if nuts_sampler in {"blackjax", "numpyro"}:
            method = "vectorized" if chain_method == "auto" else chain_method
            kwargs["chain_method"] = method
            # The blackjax inner progress bar uses JAX IO callbacks which are
            # not supported inside jax.vmap ("IO effect not supported in
            # vmap-of-cond"), so disable it under vectorized chains.
            if method == "vectorized":
                kwargs["progressbar"] = False
    if chains is not None:
        kwargs["chains"] = chains
    if cores is not None:
        kwargs["cores"] = cores
    if "PYTEST_CURRENT_TEST" in os.environ:
        # Keep the chains in one process so a test run does not fork
        # workers, but do NOT collapse the chain count: R-hat needs two
        # or more chains, and with one chain it is undefined (NaN), not
        # "converged". A single-chain harness therefore reports no
        # convergence information at all, which hides every
        # disagreement between chains - the failure the diagnostic
        # exists to catch. The chain count and the core count are
        # independent knobs; only the core count is folded down here.
        kwargs["cores"] = 1
        if "chains" not in kwargs:
            # No chain count was requested. Pick one that leaves R-hat
            # defined; an explicit request is honoured as asked, so a
            # test that wants a single chain still gets one (and the
            # undefined diagnostic that goes with it).
            kwargs["chains"] = 2
    return kwargs


def _compute_sample_log_likelihood(trace: xr.DataTree) -> xr.DataTree:
    """Populate the log_likelihood group on sampled PyMC inference data."""
    return typing.cast(
        "xr.DataTree",
        pm.compute_log_likelihood(trace, extend_inferencedata=True, progressbar=False),
    )


def _sample_trace(
    n_samples: int,
    *,
    tune: int,
    target_accept: float,
    sampler: SamplerConfig,
) -> xr.DataTree:
    """Run ``pm.sample`` with :class:`SamplerConfig` controls.

    Must be called inside an active ``pm.Model`` context. Assembles the
    backend, chain/core, tree-depth, seed and compile options from *sampler*.

    Parameters
    ----------
    n_samples : int
        Number of posterior draws per chain.
    tune : int
        Number of tuning draws.
    target_accept : float
        Target acceptance probability.
    sampler : SamplerConfig
        Sampling controls (backend, chains, cores, chain method, seed,
        max tree depth, compile options).

    Returns
    -------
    xr.DataTree
        The sampled inference data (without the log_likelihood group).
    """
    sample_kwargs: dict[str, object] = {
        "tune": tune,
        "target_accept": target_accept,
        "return_inferencedata": True,
        **_pymc_sample_parallel_args(
            sampler.nuts_sampler,
            chains=sampler.chains,
            cores=sampler.cores,
            chain_method=sampler.chain_method,
        ),
    }
    if sampler.max_treedepth is not None:
        sample_kwargs["max_treedepth"] = sampler.max_treedepth
    if sampler.random_seed is not None:
        sample_kwargs["random_seed"] = sampler.random_seed
    if sampler.backend is not None:
        sample_kwargs["backend"] = sampler.backend
    if sampler.compile_kwargs is not None:
        sample_kwargs["compile_kwargs"] = dict(sampler.compile_kwargs)
    return typing.cast("xr.DataTree", pm.sample(n_samples, **sample_kwargs))
