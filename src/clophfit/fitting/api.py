"""
Unified fitting API for ClopHfit.

This module consolidates the most useful fit_binding* entry points into a
single, documented interface without breaking existing code. It wraps the
stable implementations and provides a simple dispatcher.

Kept backends
-------------
- Least-squares global fit: fit_binding_glob (core)
- Robust outlier removal for global fit: outlier2 (core)
- Bayesian (single shared ye_mag): fit_binding_pymc (bayes)
- Bayesian (per-label ye_mag): fit_binding_pymc2 (bayes)

Deprecated/legacy variants are still available in their original modules but
should be avoided in new code.
"""

from __future__ import annotations

import warnings
from enum import StrEnum
from typing import TYPE_CHECKING

from .bayes import (
    fit_binding_pymc as _fit_binding_bayes,
    fit_binding_pymc2 as _fit_binding_bayes_perlabel,
)
from .core import (
    fit_binding_glob as _fit_binding_glob,
    fit_binding_glob_recursive as _impl_recursive,
    fit_binding_glob_recursive_outlier as _impl_recursive_outlier,
    fit_binding_glob_reweighted as _impl_reweighted,
    outlier2 as _fit_outlier2,
)
from .data_structures import Dataset, FitResult, MiniT

if TYPE_CHECKING:
    from lmfit.minimizer import Minimizer  # type: ignore[import-untyped]


# Typed helpers to extract optional parameters from kwargs safely


def _get_bool(kwargs: dict[str, object], name: str, *, default: bool) -> bool:
    value = kwargs.get(name, default)
    if isinstance(value, bool):
        return value
    return bool(value)


def _get_str(kwargs: dict[str, object], name: str, default: str) -> str:
    value = kwargs.get(name, default)
    return value if isinstance(value, str) else default


def _get_float(kwargs: dict[str, object], name: str, default: float) -> float:
    value = kwargs.get(name, default)
    if isinstance(value, (int | float)):
        return float(value)
    return default


def _get_int(kwargs: dict[str, object], name: str, default: int) -> int:
    value = kwargs.get(name, default)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return default


class FitMethod(StrEnum):
    """Available unified fit methods."""

    LM = "lm"  # Least-squares global fit (lmfit)
    LM_OUTLIER = "lm_outlier"  # Robust reweighting + outlier removal
    BAYES = "bayes"  # PyMC with a single ye_mag scaling (shared)
    BAYES_PERLABEL = "bayes_perlabel"  # PyMC with per-label ye_mag


def fit_binding(
    dataset: Dataset, method: FitMethod | str = "lm", /, **kwargs: object
) -> FitResult[MiniT]:
    """Unified entry point for binding fits.

    Parameters
    ----------
    dataset : Dataset
        Multi-label titration dataset to fit.
    method : FitMethod | str
        One of: "lm", "lm_outlier", "bayes", "bayes_perlabel".
    **kwargs : object
        Extra keyword arguments forwarded to the backend function.

    Returns
    -------
    FitResult[MiniT]
        Backend-specific FitResult with the appropriate MiniT type.

    Raises
    ------
    ValueError
        When method str is unknown.
    TypeError
        When a well key is not provided.
    """
    m = FitMethod(method) if not isinstance(method, FitMethod) else method
    err = f"Unknown fit method: {method}"

    if m is FitMethod.LM:
        # Optional kwargs: robust: bool = False
        robust: bool = _get_bool(kwargs, "robust", default=False)
        return _fit_binding_glob(dataset, robust=robust)
    if m is FitMethod.LM_OUTLIER:
        # Optional kwargs: key: str = "", threshold: float = 3.0, plot_z_scores: bool = False
        key: str = _get_str(kwargs, "key", "")
        threshold: float = _get_float(kwargs, "threshold", 3.0)
        plot_z_scores: bool = _get_bool(kwargs, "plot_z_scores", default=False)
        return _fit_outlier2(
            dataset, key=key, threshold=threshold, plot_z_scores=plot_z_scores
        )
    if m is FitMethod.BAYES:
        # Expects a previous deterministic FitResult as "fr": FitResult[MiniT]
        fr_obj = kwargs.get("fr")
        if not isinstance(fr_obj, FitResult):
            msg = (
                "fit_binding(method='bayes') requires keyword argument "
                "fr=FitResult from an initial deterministic fit."
            )
            raise TypeError(msg)
        fr_bayes: FitResult[MiniT] = fr_obj
        n_sd: float = _get_float(kwargs, "n_sd", 10.0)
        n_xerr: float = _get_float(kwargs, "n_xerr", 1.0)
        ye_scaling: float = _get_float(kwargs, "ye_scaling", 1.0)
        n_samples: int = _get_int(kwargs, "n_samples", 2000)
        return _fit_binding_bayes(
            fr_bayes,
            n_sd=n_sd,
            n_xerr=n_xerr,
            ye_scaling=ye_scaling,
            n_samples=n_samples,
        )
    if m is FitMethod.BAYES_PERLABEL:
        fr2_obj = kwargs.get("fr")
        if not isinstance(fr2_obj, FitResult):
            msg = (
                "fit_binding(method='bayes_perlabel') requires keyword argument "
                "fr=FitResult from an initial deterministic fit."
            )
            raise TypeError(msg)
        fr_bayes2: FitResult[MiniT] = fr2_obj
        n_sd2: float = _get_float(kwargs, "n_sd", 10.0)
        n_xerr2: float = _get_float(kwargs, "n_xerr", 1.0)
        n_samples2: int = _get_int(kwargs, "n_samples", 2000)
        return _fit_binding_bayes_perlabel(
            fr_bayes2,
            n_sd=n_sd2,
            n_xerr=n_xerr2,
            n_samples=n_samples2,
        )
    raise ValueError(err)


# Convenience named wrappers (stable API)


def fit_binding_lm(dataset: Dataset, **kwargs: object) -> FitResult[Minimizer]:
    """Least-squares global fit (lmfit).

    Parameters
    ----------
    dataset : Dataset
        The dataset to fit.
    **kwargs : object
        Additional keyword arguments passed to the fit method.

    Returns
    -------
    FitResult[Minimizer]
        Fitting results with lmfit minimizer.
    """
    return fit_binding(dataset, method=FitMethod.LM, **kwargs)


def fit_binding_lm_outlier(dataset: Dataset, **kwargs: object) -> FitResult[Minimizer]:
    """Least-squares fit with robust reweighting and outlier removal.

    Parameters
    ----------
    dataset : Dataset
        The dataset to fit.
    **kwargs : object
        Additional keyword arguments passed to the fit method.

    Returns
    -------
    FitResult[Minimizer]
        Fitting results with outlier detection and robust weighting.
    """
    return fit_binding(dataset, method=FitMethod.LM_OUTLIER, **kwargs)


def fit_binding_bayes(fr: FitResult[MiniT], **kwargs: object) -> FitResult[MiniT]:
    """Bayesian single-ye_mag model (shared across labels).

    Parameters
    ----------
    fr : FitResult[MiniT]
        Initial deterministic fit result.
    **kwargs : object
        Additional keyword arguments passed to the fit method.

    Returns
    -------
    FitResult[MiniT]
        Bayesian fitting results with shared ye_mag across labels.
    """
    # Note: requires an initial deterministic FitResult as input
    return fit_binding(
        fr.dataset or Dataset({}), method=FitMethod.BAYES, fr=fr, **kwargs
    )


def fit_binding_bayes_perlabel(
    fr: FitResult[MiniT], **kwargs: object
) -> FitResult[MiniT]:
    """Bayesian model with per-label ye_mag.

    Parameters
    ----------
    fr : FitResult[MiniT]
        Initial deterministic fit result.
    **kwargs : object
        Additional keyword arguments passed to the fit method.

    Returns
    -------
    FitResult[MiniT]
        Bayesian fitting results with separate ye_mag per label.
    """
    return fit_binding(
        fr.dataset or Dataset({}), method=FitMethod.BAYES_PERLABEL, fr=fr, **kwargs
    )


# Deprecation shims for legacy names


def fit_binding_glob_reweighted(
    ds: Dataset, key: str, threshold: float = 2.05
) -> FitResult[Minimizer]:  # pragma: no cover
    """Use fit_binding_lm_outlier instead (deprecated shim).

    Parameters
    ----------
    ds : Dataset
        The dataset to fit.
    key : str
        Key identifier for the dataset.
    threshold : float, optional
        Outlier detection threshold, by default 2.05.

    Returns
    -------
    FitResult[Minimizer]
        Deprecated. Use fit_binding_lm_outlier() instead.
    """
    warnings.warn(
        "fit_binding_glob_reweighted is deprecated. Use fit_binding_lm_outlier() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl_reweighted(ds, key=key, threshold=threshold)


def fit_binding_glob_recursive(
    ds: Dataset, max_iterations: int = 15, tol: float = 0.1
) -> FitResult[Minimizer]:  # pragma: no cover
    """Use fit_binding_lm_outlier instead (deprecated shim).

    Parameters
    ----------
    ds : Dataset
        The dataset to fit.
    max_iterations : int, optional
        Maximum number of iterations, by default 15.
    tol : float, optional
        Tolerance for convergence, by default 0.1.

    Returns
    -------
    FitResult[Minimizer]
        Deprecated. Use fit_binding_lm_outlier() instead.
    """
    warnings.warn(
        "fit_binding_glob_recursive is deprecated. Use fit_binding_lm_outlier() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl_recursive(ds, max_iterations=max_iterations, tol=tol)


def fit_binding_glob_recursive_outlier(
    ds: Dataset, tol: float = 0.01, threshold: float = 3.0
) -> FitResult[Minimizer]:  # pragma: no cover
    """Use fit_binding_lm_outlier instead (deprecated shim).

    Parameters
    ----------
    ds : Dataset
        The dataset to fit.
    tol : float, optional
        Tolerance for convergence, by default 0.01.
    threshold : float, optional
        Outlier detection threshold, by default 3.0.

    Returns
    -------
    FitResult[Minimizer]
        Deprecated. Use fit_binding_lm_outlier() instead.
    """
    warnings.warn(
        "fit_binding_glob_recursive_outlier is deprecated. Use fit_binding_lm_outlier() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl_recursive_outlier(ds, tol=tol, threshold=threshold)
