r"""Heteroskedasticity-robust (sandwich) standard errors for a fitted well.

The usual standard error trusts the weights: it is ``s^2 (J'J)^-1``, which is
only right when the residuals really do have the variance the weights assumed.
On these plates they do not - the scatter grows with signal, and with plate
position - yet weighting sigma to match costs precision in K, because it
down-weights the plateau points that fix S0 and S1 (see
:mod:`clophfit.fitting.gain_calibration`).

The sandwich estimator takes the other route: keep the weights that fit best and
repair only the uncertainty. With ``J`` the weighted Jacobian and ``e`` the
weighted residuals,

.. math:: \widehat{\operatorname{Var}}(\hat\beta)
          = (J'J)^{-1} J' \Omega J (J'J)^{-1}

where ``Omega`` is diagonal in the observed squared residuals rather than in an
assumed sigma: ``e_i^2`` (HC0), ``e_i^2 / (1 - h_i)`` (HC2) or
``e_i^2 / (1 - h_i)^2`` (HC3), ``h`` the leverage. HC3 is the default here
because the series are short - seven points and three parameters per label - and
HC0 is badly biased in small samples.

What this buys: an interval for K that reflects the scatter the well actually
shows, including the part the noise model does not describe. What it does not
buy: a different K.

**Read the size of the correction with care on short series.** On simulated
seven-step, two-label wells (nine residual degrees of freedom) HC3 inflates K's
standard error by 1.25x whether the noise is homoscedastic or strongly
signal-dependent, and it lifts 95% coverage from 0.91 to 0.95 - but so does
using a Student-t quantile with the classical error, which covers 0.94-0.95
while HC3 with the same quantile over-covers at 0.96-0.97. At this series length
the inflation is mostly a small-sample correction and not evidence of
misspecification; the sandwich earns its keep where the design is longer or
the wells are pooled.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from clophfit.fitting.residual_tests import _well_linearised

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["hc_covariance", "robust_k_se", "sandwich_se"]

HCKind = Literal["HC0", "HC2", "HC3"]
_MIN_DOF = 1


def _pinv_xtx(jac: np.ndarray) -> np.ndarray | None:
    """``(J'J)^-1`` by SVD, or None when the Jacobian is rank deficient."""
    try:
        _, s, vt = np.linalg.svd(jac, full_matrices=False)
    except np.linalg.LinAlgError:  # pragma: no cover - singular Jacobian
        return None
    keep = s > np.finfo(float).eps * max(jac.shape) * s[0]
    if not keep.all():
        return None
    out: np.ndarray = (vt.T / s**2) @ vt
    return out


def hc_covariance(
    jac: np.ndarray, resid: np.ndarray, *, kind: HCKind = "HC3"
) -> np.ndarray | None:
    """Sandwich covariance of the parameters, from the residuals as they are.

    Parameters
    ----------
    jac : np.ndarray
        ``(n, p)`` Jacobian of the residuals actually minimised (weighted, if
        the fit was weighted).
    resid : np.ndarray
        Those residuals, on the same scale.
    kind : HCKind
        Small-sample correction: ``HC0`` none, ``HC2`` divides each squared
        residual by ``1 - h``, ``HC3`` by ``(1 - h)^2`` (the default).

    Returns
    -------
    np.ndarray | None
        ``(p, p)`` covariance, or None when the Jacobian is rank deficient.

    Examples
    --------
    A straight line whose noise grows with x: the robust slope error exceeds the
    classical one, which believes the constant variance it was given.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> x = np.linspace(0, 1, 200)
    >>> j = np.column_stack([np.ones_like(x), x])
    >>> e = rng.normal(0, 0.1 + 2 * x)
    >>> cov = hc_covariance(j, e - j @ np.linalg.lstsq(j, e, rcond=None)[0])
    >>> classical = np.var(e, ddof=2) * np.linalg.inv(j.T @ j)
    >>> bool(cov[1, 1] > classical[1, 1])
    True
    """
    j = np.asarray(jac, dtype=float)
    e = np.asarray(resid, dtype=float)
    xtx_inv = _pinv_xtx(j)
    if xtx_inv is None or j.shape[0] <= j.shape[1]:
        return None
    q, _ = np.linalg.qr(j)
    h = np.clip(np.einsum("ij,ij->i", q, q), 0.0, 1.0 - 1e-12)
    weight = {
        "HC0": np.ones_like(h),
        "HC2": 1.0 / (1.0 - h),
        "HC3": 1.0 / (1.0 - h) ** 2,
    }[kind]
    meat = j.T @ (j * (e**2 * weight)[:, None])
    out: np.ndarray = xtx_inv @ meat @ xtx_inv
    return out


def sandwich_se(
    jac: np.ndarray, resid: np.ndarray, *, kind: HCKind = "HC3"
) -> np.ndarray | None:
    """Robust standard error per parameter (the square root of :func:`hc_covariance`).

    Parameters
    ----------
    jac : np.ndarray
        Weighted Jacobian.
    resid : np.ndarray
        Weighted residuals.
    kind : HCKind
        Small-sample correction.

    Returns
    -------
    np.ndarray | None
        Standard errors, or None when the Jacobian is rank deficient.
    """
    cov = hc_covariance(jac, resid, kind=kind)
    return None if cov is None else np.sqrt(np.clip(np.diag(cov), 0.0, None))


def robust_k_se(
    table: pd.DataFrame,
    params: Mapping[str, Mapping[str, float]],
    *,
    is_ph: bool = True,
    kind: HCKind = "HC3",
) -> pd.DataFrame:
    """Model-based and robust standard errors of K, per well.

    Each well is linearised at its fitted parameters (the same weighted
    Jacobian the residual tests use, residuals projected through ``I - H`` so a
    posterior mean is treated as its least-squares equivalent). K is the first
    column of that design, so its standard error is read from the two
    covariances: the classical one, ``s^2 (J'J)^-1``, which trusts the weights,
    and the sandwich one, which does not.

    Parameters
    ----------
    table : pd.DataFrame
        Residual table with ``well``, ``label``, ``x``, ``raw_res``, ``sigma``.
    params : Mapping[str, Mapping[str, float]]
        Well to its fitted ``K``, ``S0_<label>``, ``S1_<label>``.
    is_ph : bool
        pH titration (else concentration).
    kind : HCKind
        Small-sample correction for the sandwich.

    Returns
    -------
    pd.DataFrame
        Indexed by well: ``k_se_model``, ``k_se_robust``, ``ratio``
        (robust / model), ``n_points`` and ``dof``.
    """
    rows: dict[str, dict[str, float]] = {}
    for well, g in table.groupby("well", sort=False):
        p = params.get(str(well))
        lin = None if p is None else _well_linearised(g, p, is_ph=is_ph)
        if lin is None:
            continue
        _, r_ls, _, jac = lin
        dof = jac.shape[0] - jac.shape[1]
        xtx_inv = _pinv_xtx(jac)
        if xtx_inv is None or dof < _MIN_DOF:
            continue
        model = float(np.sqrt(np.sum(r_ls**2) / dof * xtx_inv[0, 0]))
        robust = sandwich_se(jac, r_ls, kind=kind)
        if robust is None:
            continue
        rows[str(well)] = {
            "k_se_model": model,
            "k_se_robust": float(robust[0]),
            "ratio": float(robust[0]) / model if model > 0 else np.nan,
            "n_points": float(jac.shape[0]),
            "dof": float(dof),
        }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("well")
