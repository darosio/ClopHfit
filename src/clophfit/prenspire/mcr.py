"""MCR-ALS: multivariate curve resolution by alternating least squares.

Factor the corrected spectrum matrix of one sample/temperature,
``D[n_wells, n_lambda]``, into non-negative concentrations and spectra::

    D ~= C @ S.T      C >= 0, S >= 0, rows of C closed (sum to 1)

Alternating non-negative least squares (scipy ``nnls``) solves one factor
with the other fixed; closure (``C`` rows normalised to 1) pins the
rotational freedom that ordinary PCA leaves open, which is what makes the
recovered species spectra interpretable without a mechanistic model.

The pKa is read off afterwards: the component whose closed profile tracks
the titrant (largest |Spearman rho|) is fitted against a
Henderson-Hasselbalch fraction, ``profile ~= F0 + (F1 - F0)/(1 + 10**(K - x))``,
leaving K (and a residual affine mixing) free.

Contrast with :func:`clophfit.fitting.spectral.fit_spectra_global`: there
the two-state model *is* the factorisation (one K, species spectra solved
linearly). MCR-ALS relaxes the profile shape - it can expose a third
species or a non-Hill transition - at the price of a less tight K.
``fit_spectra_global``'s ``sv_ratio`` says when that trade is worth it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, lsq_linear, nnls
from scipy.stats import spearmanr

if TYPE_CHECKING:
    from collections.abc import Sequence

    from clophfit.clophfit_types import ArrayF

_MAX_ITER = 500
_TWO_STATES = 2  # the acid/base pair an endpoint start can seed
_TOL = 1e-9


class McrFit(NamedTuple):
    """A converged MCR-ALS factorisation and its reconstruction quality."""

    c: np.ndarray
    """[n_wells, n_species] closed, non-negative concentration profiles."""
    s: np.ndarray
    """[n_species, n_lambda] non-negative species spectra (arbitrary scale)."""
    rel_err: float
    """||D - C S.T|| / ||D||."""


def _nnls_rows(target: np.ndarray, design: np.ndarray) -> np.ndarray:
    """Column-wise NNLS: solve target[:, j] ~= design @ x_j, x_j >= 0."""
    out = np.zeros((design.shape[1], target.shape[1]), dtype=np.float64)
    for j in range(target.shape[1]):
        x, _ = nnls(design, target[:, j])
        out[:, j] = x
    return out


def _als_run(
    d: np.ndarray,
    s0: np.ndarray,
    *,
    closure: bool,
    max_iter: int,
    tol: float,
) -> McrFit:
    """One ALS sweep sequence from a fixed spectra start."""
    n_wells, _n_lam = d.shape
    k = s0.shape[1]
    c = np.full((n_wells, k), 1.0 / k)
    s = s0.copy()
    prev = np.inf
    err = np.inf
    for _ in range(max_iter):
        if k == 1 or not closure:
            c = _nnls_rows(d.T, s).T  # [n, k]: d_row ~= s @ c_row
            if closure:
                tot = c.sum(axis=1, keepdims=True)
                c = np.divide(c, tot, out=np.full_like(c, 1.0 / k), where=tot > 0)
        else:
            # closed profiles: well j's spectrum is s0 + sum_i f_i (s_i - s0),
            # with the fractions boxed to [0, 1] - a simplex-constrained LS
            # solved exactly per well (ALS-SSS).
            v = s[:, 1:] - s[:, :1]  # [n_lam, k-1] difference spectra
            for j in range(n_wells):
                f = lsq_linear(v, d[j] - s[:, 0], bounds=(0.0, 1.0)).x
                frac_sum = float(f.sum())
                if frac_sum > 1.0:
                    f /= frac_sum
                    c[j] = np.concatenate(([0.0], f))
                else:
                    c[j] = np.concatenate(([1.0 - frac_sum], f))
        s = _nnls_rows(d, c).T  # [lam, k]: d[:, j] ~= c @ s[:, j] per wavelength
        err = float(np.linalg.norm(d - c @ s.T) / np.linalg.norm(d))
        if np.isfinite(prev) and prev - err <= tol * max(prev, 1e-12):
            break
        prev = err
    return McrFit(c=c, s=s, rel_err=err)


def mcr_als(  # ruff: ignore[too-many-arguments]
    d: np.ndarray,
    n_species: int = 2,
    *,
    closure: bool = True,
    max_iter: int = _MAX_ITER,
    tol: float = _TOL,
    s_init: np.ndarray | None = None,
) -> McrFit:
    """Resolve D [n_wells, n_lambda] into C [n_wells, k] and S [n_lambda, k].

    Parameters
    ----------
    d : np.ndarray
        Data matrix, one row per well, one column per wavelength (already
        corrected: buffer-subtracted).
    n_species : int
        Number of components to resolve.
    closure : bool
        Normalise each row of C to sum to 1 (fractions), pinning rotation.
    max_iter : int
        ALS sweeps per start.
    tol : float
        Relative error change that counts as converged.
    s_init : np.ndarray | None
        Optional extra spectra start [n_lambda, k] - e.g. the two extreme
        wells of a titration. ALS from a single start is a non-convex
        problem; all starts (this one, clipped and absolute PCA loadings)
        are run and the lowest-error factorisation returned.

    Returns
    -------
    McrFit
        Non-negative factors and the relative reconstruction error.

    Raises
    ------
    ValueError
        If the requested number of species exceeds the matrix rank bound.
    """
    d = np.asarray(d, dtype=np.float64)
    n_wells, n_lam = d.shape
    if n_species < 1 or n_species > min(n_wells, n_lam):
        msg = f"n_species={n_species} impossible for a {n_wells}x{n_lam} matrix"
        raise ValueError(msg)
    starts: list[np.ndarray] = []
    if s_init is not None:
        s0 = np.asarray(s_init, dtype=np.float64)
        if s0.shape == (n_lam, n_species):
            starts.append(np.clip(s0, 0.0, None))
    _u, _sv, vt = np.linalg.svd(d, full_matrices=False)
    loadings = vt[:n_species, :]
    signed = np.where(loadings.sum(axis=1, keepdims=True) < 0, -loadings, loadings)
    clipped = np.clip(signed.T, 0.0, None)
    dead = clipped.sum(axis=0) <= 0.0
    if dead.any():
        clipped[:, dead] = np.abs(signed.T[:, dead])
    starts.extend((clipped, np.abs(loadings.T)))
    best = _als_run(d, starts[0], closure=closure, max_iter=max_iter, tol=tol)
    for s0 in starts[1:]:
        fit = _als_run(d, s0, closure=closure, max_iter=max_iter, tol=tol)
        if fit.rel_err < best.rel_err:
            best = fit
    return best


def _sigmoid(x: np.ndarray, k: float, f0: float, f1: float) -> np.ndarray:
    return f0 + (f1 - f0) / (1.0 + np.power(10.0, k - x))


def _langmuir(x: np.ndarray, k: float, f0: float, f1: float) -> np.ndarray:
    return f0 + (f1 - f0) * x / (k + x)


def component_pka(
    x: np.ndarray, c: np.ndarray, *, is_ph: bool = True
) -> tuple[float, float, float, float, int, float]:
    """Equilibrium constant of the component that tracks the titrant.

    Returns K on the natural scale of the model fitted to the component:
    pKa for a pH titration (Henderson-Hasselbalch fraction), Kd in the
    units of x for a ligand titration (Langmuir fraction - linear, so no
    log-space propagation happens here). The fitted curve parameters
    (F0, F1, on the component's own scale) are returned so callers can
    draw the fit over the scatter.

    Parameters
    ----------
    x : np.ndarray
        Titrant per well.
    c : np.ndarray
        Closed concentration profiles [n_wells, k].
    is_ph : bool
        Henderson-Hasselbalch fraction (pH) if true, Langmuir (ligand) else.

    Returns
    -------
    tuple[float, float, float, float, int, float]
        (K, standard error, F0, F1, component index, signed rho of that
        component). With k = 2 a positively tracking component is preferred:
        under closure the columns are complements, and the signed choice
        keeps the reported component index stable instead of arbitrarily
        picking whichever ties first on |rho|.
    """
    rho = [float(spearmanr(x, c[:, i]).statistic) for i in range(c.shape[1])]
    pos = any(r > 0 for r in rho)
    comp = int(np.argmax(rho)) if pos else int(np.argmin(rho))
    y = c[:, comp]
    model = _sigmoid if is_ph else _langmuir
    try:
        popt, pcov = curve_fit(
            model,
            x,
            y,
            p0=[float(np.median(x)), float(np.min(y)), float(np.max(y))],
            maxfev=20000,
        )
    except RuntimeError:
        return float("nan"), float("nan"), float("nan"), float("nan"), comp, rho[comp]
    se0 = pcov[0, 0]
    se_k = float(np.sqrt(se0)) if np.isfinite(se0) else float("nan")
    k, f0, f1 = float(popt[0]), float(popt[1]), float(popt[2])
    return k, se_k, f0, f1, comp, rho[comp]


def _stack_scans(
    scans: Sequence[tuple[ArrayF, ArrayF]],
) -> tuple[np.ndarray, np.ndarray]:
    """Join per-scan (lambda, Y[n_lambda, n_wells]) blocks rowwise.

    Parameters
    ----------
    scans : Sequence[tuple[ArrayF, ArrayF]]
        One (wavelengths, Y[n_lambda, n_wells]) block per scan.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The concatenated wavelength axis and D[n_wells, n_lambda_total],
        each scan scaled by the common maximum so blocks stay commensurable.
    """
    lams = np.concatenate([np.asarray(lam, dtype=np.float64) for lam, _ in scans])
    mats = [np.asarray(y, dtype=np.float64) for _, y in scans]
    top = max(float(np.nanmax(np.abs(m))) for m in mats)
    d = np.vstack([m / top for m in mats]).T  # rows: wells; cols: all wavelengths
    return lams, d


def mcr_titration(
    spectra_by_temp: dict[str, Sequence[tuple[ArrayF, ArrayF]]],
    x_by_temp: dict[str, ArrayF],
    *,
    n_species: int = 2,
    closure: bool = True,
    is_ph: bool = True,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, tuple[ArrayF, np.ndarray]]]:
    """MCR-ALS every temperature's scans; return (K table, spectra, profiles).

    Parameters
    ----------
    spectra_by_temp : dict[str, Sequence[tuple[ArrayF, ArrayF]]]
        Temperature label to its scans, each ``(lambda, Y[n_lambda, n_wells])``.
    x_by_temp : dict[str, ArrayF]
        Titrant value per well, same temperature keys.
    n_species : int
        Components resolved per temperature.
    closure : bool
        Constrain profiles to fractions.
    is_ph : bool
        Titrant axis kind: pH (Henderson-Hasselbalch) or ligand (Langmuir).

    Returns
    -------
    tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, tuple[ArrayF, np.ndarray]]]
        One table row per (temp): K, se, component, rho, rel_err, n; per
        temperature a table of the resolved species spectra; and per
        temperature the titrant with the closed profiles, for plotting.
    """
    rows: list[dict[str, object]] = []
    spectra: dict[str, pd.DataFrame] = {}
    profiles: dict[str, tuple[ArrayF, np.ndarray]] = {}
    for t in sorted(spectra_by_temp):
        scans = spectra_by_temp[t]
        x = np.asarray(x_by_temp[t], dtype=np.float64)
        if not scans:
            continue
        lams, d = _stack_scans(scans)
        d = np.where(np.isfinite(d), d, 0.0)
        s_init: np.ndarray | None = None
        if n_species == _TWO_STATES and x.size > 1 and float(np.ptp(x)) > 0:
            # the extreme-titrant wells bound the two states of the equilibrium
            i0, i1 = int(np.argmin(x)), int(np.argmax(x))
            s_init = np.column_stack([d[i0], d[i1]])
        fit = mcr_als(d, n_species, closure=closure, s_init=s_init)
        k, se, f0, f1, comp, rho = component_pka(x, fit.c, is_ph=is_ph)
        rows.append({
            "temp": t,
            "K": k,
            "se": se,
            "F0": f0,
            "F1": f1,
            "component": comp,
            "rho": rho,
            "rel_err": fit.rel_err,
            "n_species": n_species,
            "n": x.size,
        })
        spec = pd.DataFrame(lams, columns=["lambda"])
        for i in range(fit.s.shape[1]):
            spec[f"species_{i}"] = fit.s[:, i]
        spectra[t] = spec
        profiles[t] = (x, fit.c)
    return pd.DataFrame(rows), spectra, profiles
