"""Whole-spectrum global fit of a two-state titration (variable projection).

A titration of a two-state probe measured as spectra obeys a bilinear model: every spectrum is a
mixture of two species spectra weighted by the fraction of each state,

    Y_j(lambda, w) = E_j0(lambda) (1 - f(x_w)) + E_j1(lambda) f(x_w),

for each label (scan) j, well w and titrant value x_w, with f the single-site binding fraction
(Henderson-Hasselbalch for pH). Given K (and optionally the Hill slope), the species spectra are
the solution of a linear least-squares problem, so they are eliminated (variable projection,
Golub & Pereyra 1973) and only K is fitted nonlinearly. This joins the two older readouts:

- band averaging (:mod:`clophfit.prenspire.bands`) uses a few windows with a shared K;
- SVD (:func:`clophfit.fitting.core.analyze_spectra`) projects on one principal component.

Here every wavelength of every label enters, each label weighted by its own residual variance, and
the fitted species spectra are returned so that they can be checked against the expected neutral
and anionic chromophore spectra. When pure-state reference spectra are known, fixing E turns the
fit into the classical unmixing of the 1995 SPETTRI programs.

Neighbouring wavelengths of one spectrum share their errors (pipetting, lamp and gain drift act on
the whole spectrum), so the covariance from the Jacobian treats far too many points as independent.
The standard error reported is therefore the delete-one-well jackknife, with the naive value kept
for comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import least_squares

from clophfit.fitting.models import binding_1site

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from clophfit.clophfit_types import ArrayF

_GRID_POINTS = 61
_WEIGHT_PASSES = 3
_ALSO_ITERATIONS = 50
_ALSO_TOL = 1e-7


@dataclass(frozen=True)
class SpectralFit:
    """Result of :func:`fit_spectra_global`.

    Attributes
    ----------
    K : float
        pKa (pH titration) or Kd (ligand titration).
    se : float
        Delete-one-well jackknife standard error of K.
    se_naive : float
        Standard error from the Jacobian, treating every wavelength as independent (too small).
    hill : float
        Hill slope (1 unless fitted).
    species : dict[str, ArrayF]
        Per label, the two species spectra, shape (2, n_lambda), in the order of
        :func:`~clophfit.fitting.models.binding_1site`'s S0 and S1: for pH the deprotonated
        (basic) then the protonated (acidic) state; for a ligand the free then the bound state.
    wavelengths : dict[str, ArrayF]
        Per label, the wavelengths used.
    rms : dict[str, float]
        Per label, the residual root-mean-square (the label's weight is its inverse square).
    residual_sv : ArrayF
        Singular values of the weighted residual matrix (wells x all wavelengths). A first value
        well above the others points to a third species or a drift the two-state model misses.
    n_wells : int
        Wells used.
    well_scale : ArrayF
        Fitted per-well amplitudes (all 1 unless ``well_scale`` was requested).
    """

    K: float
    se: float
    se_naive: float
    hill: float
    species: dict[str, ArrayF]
    wavelengths: dict[str, ArrayF]
    rms: dict[str, float]
    residual_sv: ArrayF
    n_wells: int
    well_scale: ArrayF


def _fractions(x: ArrayF, k: float, hill: float, *, is_ph: bool) -> ArrayF:
    f = np.asarray(
        binding_1site(x, k, 0.0, 1.0, is_ph=is_ph, hill=hill), dtype=np.float64
    )
    return np.column_stack([1.0 - f, f])


def _project(  # ruff: ignore[too-many-arguments]
    x: ArrayF,
    spectra: Sequence[ArrayF],
    k: float,
    hill: float,
    *,
    is_ph: bool,
    well_scale: bool = False,
    sigma: Sequence[float] | None = None,
) -> tuple[list[ArrayF], list[ArrayF], ArrayF]:
    """Species spectra and residuals per label, and the well scales, at fixed K and Hill slope.

    Without ``well_scale`` the species spectra are one linear least-squares solve. With it, each
    well also gets an amplitude s_w (the protein amount actually in the well, mean 1) shared by all
    labels; the model ``s_w C_w E`` is then bilinear and is solved by alternating least squares.
    """
    c = _fractions(x, k, hill, is_ph=is_ph)
    w = np.ones(len(spectra)) if sigma is None else 1 / np.asarray(sigma) ** 2
    scale = np.ones(x.size)
    for _ in range(_ALSO_ITERATIONS if well_scale else 1):
        cs = c * scale[:, None]
        species = [np.linalg.lstsq(cs, y.T, rcond=None)[0] for y in spectra]
        if not well_scale:
            break
        model = [c @ e for e in species]  # (n_wells, n_lambda) before scaling
        num = sum(
            wj * np.sum(m * y.T, axis=1)
            for wj, m, y in zip(w, model, spectra, strict=True)
        )
        den = sum(wj * np.sum(m * m, axis=1) for wj, m in zip(w, model, strict=True))
        new = np.asarray(num / den, dtype=np.float64)
        new /= new.mean()
        converged = np.max(np.abs(new - scale)) < _ALSO_TOL
        scale = new
        if converged:
            break
    cs = c * scale[:, None]
    species = [np.linalg.lstsq(cs, y.T, rcond=None)[0] for y in spectra]
    residuals = [y.T - cs @ e for y, e in zip(spectra, species, strict=True)]
    return species, residuals, scale


def _k_bounds(x: ArrayF, *, is_ph: bool) -> tuple[float, float]:
    if is_ph:
        return float(x.min() - 1.0), float(x.max() + 1.0)
    positive = x[x > 0]
    return float(positive.min() / 10), float(x.max() * 10)


def _solve(  # ruff: ignore[too-many-arguments]
    x: ArrayF,
    spectra: Sequence[ArrayF],
    *,
    is_ph: bool,
    fit_hill: bool,
    well_scale: bool,
    k0: float | None = None,
) -> tuple[float, float, float, list[float]]:
    """Best K (and Hill slope), its naive standard error, and the per-label weights used."""
    lo, hi = _k_bounds(x, is_ph=is_ph)
    grid = (
        np.linspace(lo, hi, _GRID_POINTS)
        if is_ph
        else np.geomspace(lo, hi, _GRID_POINTS)
    )
    sigma = [1.0] * len(spectra)
    k, hill, se = k0 if k0 is not None else float(np.median(grid)), 1.0, np.nan
    for _ in range(_WEIGHT_PASSES):

        def resid(theta: ArrayF, s: list[float] = sigma) -> ArrayF:
            h = float(theta[1]) if fit_hill else 1.0
            _, res, _ = _project(
                x,
                spectra,
                float(theta[0]),
                h,
                is_ph=is_ph,
                well_scale=well_scale,
                sigma=s,
            )
            return np.concatenate([
                (r / sj).ravel() for r, sj in zip(res, s, strict=True)
            ])

        if k0 is None:
            ssr = [float(np.sum(resid(np.array([g, 1.0])) ** 2)) for g in grid]
            k = float(grid[int(np.argmin(ssr))])
        theta0 = np.array([k, hill]) if fit_hill else np.array([k])
        bounds = ([lo, 0.2], [hi, 5.0]) if fit_hill else ([lo], [hi])
        fit = least_squares(resid, theta0, bounds=bounds, x_scale="jac")
        k = float(fit.x[0])
        hill = float(fit.x[1]) if fit_hill else 1.0
        _, res, _ = _project(
            x, spectra, k, hill, is_ph=is_ph, well_scale=well_scale, sigma=sigma
        )
        n_par = (
            2 * len(spectra)
            + (2 if fit_hill else 1)
            + (x.size - 1 if well_scale else 0)
        )
        sigma = [float(np.sqrt(np.sum(r**2) / max(r.size - n_par, 1))) for r in res]
        jac = np.asarray(fit.jac, dtype=np.float64)
        jtj = jac.T @ jac
        dof = max(fit.fun.size - n_par, 1)
        cov = np.linalg.pinv(jtj) * float(np.sum(fit.fun**2)) / dof
        se = float(np.sqrt(cov[0, 0]))
    return k, hill, se, sigma


def fit_spectra_global(  # ruff: ignore[too-many-arguments]
    x: ArrayF,
    spectra: Mapping[str, tuple[ArrayF, ArrayF]],
    *,
    is_ph: bool = True,
    fit_hill: bool = False,
    well_scale: bool = False,
    jackknife: bool = True,
) -> SpectralFit:
    """Fit K to whole spectra of one or more labels, species spectra eliminated linearly.

    Parameters
    ----------
    x : ArrayF
        Titrant value (pH or concentration) per well.
    spectra : Mapping[str, tuple[ArrayF, ArrayF]]
        Label to ``(wavelengths, Y)`` with ``Y`` of shape (n_lambda, n_wells), already corrected
        per well (buffer subtracted, normalised to the protein amount). Wavelengths holding a
        non-finite value in any well are dropped.
    is_ph : bool
        pH titration (Henderson-Hasselbalch) or ligand binding.
    fit_hill : bool
        Also fit the Hill slope (bounded to 0.2-5).
    well_scale : bool
        Give each well a free amplitude shared by its labels (protein amount, mean 1). Use it when
        the spectra are not normalised to a protein reference, or to absorb what is left of the
        well-to-well amount error after normalisation.
    jackknife : bool
        Compute the delete-one-well jackknife standard error (one refit per well).

    Returns
    -------
    SpectralFit
        K, its errors, the species spectra and residual diagnostics.

    Raises
    ------
    ValueError
        With fewer than four wells or no usable wavelength.
    """
    xa = np.asarray(x, dtype=np.float64)
    if xa.size < 4:  # ruff: ignore[magic-value-comparison]
        msg = "a global spectral fit needs at least four wells"
        raise ValueError(msg)
    labels, lams, ys = [], [], []
    for label, (lam, y) in spectra.items():
        ya = np.asarray(y, dtype=np.float64)
        ok = np.all(np.isfinite(ya), axis=1)
        if ok.any():
            labels.append(label)
            lams.append(np.asarray(lam, dtype=np.float64)[ok])
            ys.append(ya[ok])
    if not ys:
        msg = "no wavelength is finite in every well"
        raise ValueError(msg)
    k, hill, se_naive, sigma = _solve(
        xa, ys, is_ph=is_ph, fit_hill=fit_hill, well_scale=well_scale
    )
    species, res, scale = _project(
        xa, ys, k, hill, is_ph=is_ph, well_scale=well_scale, sigma=sigma
    )
    weighted = np.hstack([r / s for r, s in zip(res, sigma, strict=True)])
    se = np.nan
    if jackknife:
        ks = []
        for i in range(xa.size):
            keep = np.arange(xa.size) != i
            ki, *_ = _solve(
                xa[keep],
                [y[:, keep] for y in ys],
                is_ph=is_ph,
                fit_hill=fit_hill,
                well_scale=well_scale,
                k0=k,
            )
            ks.append(ki)
        kj = np.asarray(ks)
        se = float(np.sqrt((kj.size - 1) / kj.size * np.sum((kj - kj.mean()) ** 2)))
    return SpectralFit(
        K=k,
        se=se,
        se_naive=se_naive,
        hill=hill,
        species=dict(zip(labels, species, strict=True)),
        wavelengths=dict(zip(labels, lams, strict=True)),
        rms=dict(zip(labels, sigma, strict=True)),
        residual_sv=np.linalg.svd(weighted, compute_uv=False),
        n_wells=int(xa.size),
        well_scale=scale,
    )
