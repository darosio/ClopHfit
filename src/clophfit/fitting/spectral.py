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

The species spectra are not principal components: for each trial K the fractions C(K) are fixed and
E is the ordinary least-squares solution of Y = C E (with the per-well amplitude, alternating with
it). SVD enters only as a diagnostic, on the residual matrix.

``acid_state`` adds a third state with its own pK below K. On the IBF plates it is not supported:
the well amplitudes are low (median 0.90) in each plate's most acidic column whatever its pH (5.1-5.8),
while on ten-column plates pH 5.4 and 4.7 are not dim; only below pH ~4.5 do they collapse
(denaturation). That points to the last titration step rather than a protonation equilibrium, so the
default stays two states with a per-well amplitude, which keeps K within 0.02 pH (median) of the fit
without that column and limits the shift on a denatured plate to 0.13 pH (1.34 without it).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import least_squares
from scipy.special import logsumexp

from clophfit.fitting.models import binding_1site

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from clophfit.clophfit_types import ArrayF

_GRID_POINTS = 61
_DELTA_MIN, _DELTA_MAX = 0.3, 6.0
_WEIGHT_PASSES = 3
_AMPLITUDE_ITERATIONS = 50
_AMPLITUDE_TOL = 1e-7


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
    k_acid, se_acid : float
        pK2 of the acid state and its jackknife error (nan without ``acid_state``).
    species : dict[str, ArrayF]
        Per label, the species spectra, shape (n_states, n_lambda), in the order of
        :func:`~clophfit.fitting.models.binding_1site`'s S0 and S1: for pH the deprotonated
        (basic) then the protonated (acidic) state, then the acid state if fitted; for a ligand the
        free then the bound state.
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
    k_acid: float
    se_acid: float
    species: dict[str, ArrayF]
    wavelengths: dict[str, ArrayF]
    rms: dict[str, float]
    residual_sv: ArrayF
    n_wells: int
    well_scale: ArrayF


def _fractions(x: ArrayF, theta: ArrayF, model: _Model, *, is_ph: bool) -> ArrayF:
    """Fraction of each state per well: columns S0, S1 (and the acid state).

    ``theta`` is ``[K, (delta), (hill)]``. With an acid state the titration is sequential and
    diprotic: deprotonated <-> protonated (pK1 = K, Hill slope) <-> acid state (pK2 = K - delta).
    """
    k = float(theta[0])
    hill = float(theta[-1]) if model.fit_hill else 1.0
    if not model.acid_state:
        f = np.asarray(
            binding_1site(x, k, 0.0, 1.0, is_ph=is_ph, hill=hill), dtype=np.float64
        )
        return np.column_stack([1.0 - f, f])
    k2 = k - float(theta[1])
    ln10 = np.log(10.0)
    logs = np.column_stack([
        np.zeros_like(x),
        hill * (k - x) * ln10,
        (hill * (k - x) + (k2 - x)) * ln10,
    ])
    return np.asarray(
        np.exp(logs - logsumexp(logs, axis=1, keepdims=True)), dtype=np.float64
    )


@dataclass(frozen=True)
class _Model:
    acid_state: bool
    fit_hill: bool
    well_scale: bool


def _project(
    c: ArrayF,
    spectra: Sequence[ArrayF],
    *,
    well_scale: bool = False,
    sigma: Sequence[float] | None = None,
) -> tuple[list[ArrayF], list[ArrayF], ArrayF]:
    """Species spectra and residuals per label, and the well scales, at fixed state fractions ``c``.

    Without ``well_scale`` the species spectra are one linear least-squares solve. With it, each
    well also gets an amplitude s_w (the protein amount actually in the well, mean 1) shared by all
    labels; the model ``s_w C_w E`` is then bilinear and is solved by alternating least squares.
    """
    w = np.ones(len(spectra)) if sigma is None else 1 / np.asarray(sigma) ** 2
    scale = np.ones(c.shape[0])
    for _ in range(_AMPLITUDE_ITERATIONS if well_scale else 1):
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
        converged = np.max(np.abs(new - scale)) < _AMPLITUDE_TOL
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


def _solve(
    x: ArrayF,
    spectra: Sequence[ArrayF],
    model: _Model,
    *,
    is_ph: bool,
    theta0: ArrayF | None = None,
) -> tuple[ArrayF, float, list[float]]:
    """Best ``theta``, the naive standard error of K, and the per-label weights used."""
    lo, hi = _k_bounds(x, is_ph=is_ph)
    grid = (
        np.linspace(lo, hi, _GRID_POINTS)
        if is_ph
        else np.geomspace(lo, hi, _GRID_POINTS)
    )
    lower, upper, start = [lo], [hi], [float(np.median(grid))]
    if model.acid_state:
        lower, upper, start = [*lower, _DELTA_MIN], [*upper, _DELTA_MAX], [*start, 1.5]
    if model.fit_hill:
        lower, upper, start = [*lower, 0.2], [*upper, 5.0], [*start, 1.0]
    theta = np.asarray(theta0 if theta0 is not None else start, dtype=np.float64)
    sigma = [1.0] * len(spectra)
    se = np.nan
    n_par = (
        len(spectra) * (3 if model.acid_state else 2)
        + theta.size
        + (x.size - 1 if model.well_scale else 0)
    )
    for _ in range(_WEIGHT_PASSES):

        def resid(th: ArrayF, s: list[float] = sigma) -> ArrayF:
            c = _fractions(x, th, model, is_ph=is_ph)
            _, res, _ = _project(c, spectra, well_scale=model.well_scale, sigma=s)
            return np.concatenate([
                (r / sj).ravel() for r, sj in zip(res, s, strict=True)
            ])

        if theta0 is None:
            trial = theta.copy()
            ssr = []
            for g in grid:
                trial[0] = g
                ssr.append(float(np.sum(resid(trial) ** 2)))
            theta[0] = grid[int(np.argmin(ssr))]
        fit = least_squares(resid, theta, bounds=(lower, upper), x_scale="jac")
        theta = np.asarray(fit.x, dtype=np.float64)
        c = _fractions(x, theta, model, is_ph=is_ph)
        _, res, _ = _project(c, spectra, well_scale=model.well_scale, sigma=sigma)
        sigma = [float(np.sqrt(np.sum(r**2) / max(r.size - n_par, 1))) for r in res]
        jac = np.asarray(fit.jac, dtype=np.float64)
        cov = (
            np.linalg.pinv(jac.T @ jac)
            * float(np.sum(fit.fun**2))
            / max(fit.fun.size - n_par, 1)
        )
        se = float(np.sqrt(cov[0, 0]))
    return theta, se, sigma


def fit_spectra_global(  # ruff: ignore[too-many-arguments]
    x: ArrayF,
    spectra: Mapping[str, tuple[ArrayF, ArrayF]],
    *,
    is_ph: bool = True,
    fit_hill: bool = False,
    well_scale: bool = False,
    acid_state: bool = False,
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
    acid_state : bool
        Add a third state below the main transition (pH titrations only): sequential
        deprotonated <-> protonated <-> acid state with pK2 < K, its spectrum solved like the
        others. It models a change of brightness or colour at the acid end that a two-state
        model can only push into the well amplitudes (see the module notes on the IBF plates).
    jackknife : bool
        Compute the delete-one-well jackknife standard error (one refit per well).

    Returns
    -------
    SpectralFit
        K, its errors, the species spectra and residual diagnostics.

    Raises
    ------
    ValueError
        With fewer than four wells, no usable wavelength, or an acid state on a ligand titration.
    """
    if acid_state and not is_ph:
        msg = "the acid state is defined for pH titrations only"
        raise ValueError(msg)
    model = _Model(acid_state=acid_state, fit_hill=fit_hill, well_scale=well_scale)
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
    theta, se_naive, sigma = _solve(xa, ys, model, is_ph=is_ph)
    c = _fractions(xa, theta, model, is_ph=is_ph)
    species, res, scale = _project(c, ys, well_scale=well_scale, sigma=sigma)
    weighted = np.hstack([r / s for r, s in zip(res, sigma, strict=True)])
    se = se_acid = np.nan
    if jackknife:
        thetas = []
        for i in range(xa.size):
            keep = np.arange(xa.size) != i
            ti, *_ = _solve(
                xa[keep], [y[:, keep] for y in ys], model, is_ph=is_ph, theta0=theta
            )
            thetas.append(ti)
        tj = np.asarray(thetas)
        k1 = tj[:, 0]
        se = float(np.sqrt((k1.size - 1) / k1.size * np.sum((k1 - k1.mean()) ** 2)))
        if acid_state:
            k2 = tj[:, 0] - tj[:, 1]
            se_acid = float(
                np.sqrt((k2.size - 1) / k2.size * np.sum((k2 - k2.mean()) ** 2))
            )
    return SpectralFit(
        K=float(theta[0]),
        se=se,
        se_naive=se_naive,
        hill=float(theta[-1]) if fit_hill else 1.0,
        k_acid=float(theta[0] - theta[1]) if acid_state else np.nan,
        se_acid=se_acid,
        species=dict(zip(labels, species, strict=True)),
        wavelengths=dict(zip(labels, lams, strict=True)),
        rms=dict(zip(labels, sigma, strict=True)),
        residual_sv=np.linalg.svd(weighted, compute_uv=False),
        n_wells=int(xa.size),
        well_scale=scale,
    )
