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
import pymc as pm  # type: ignore[import-untyped]
import pytensor.tensor as pt
import xarray as xr
from scipy.interpolate import BSpline
from scipy.optimize import least_squares
from scipy.special import logsumexp
from scipy.stats import chi2

from clophfit.fitting.models import binding_1site

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import arviz as az  # type: ignore[import-untyped]

    from clophfit.clophfit_types import ArrayF

_GRID_POINTS = 61
_MIN_BASIS = 8  # a cubic spline basis needs a handful of coefficients to bend
_DELTA_MIN, _DELTA_MAX = 0.3, 6.0
_WEIGHT_PASSES = 3
_AMPLITUDE_ITERATIONS = 50
_AMPLITUDE_TOL = 1e-7
_MIN_GLOBAL_WELLS = 4


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
    kept : np.ndarray
        Wells kept by the well screen, all True when it was off.
    residuals : dict[str, ArrayF]
        Per label, the wavelength-by-well residual matrix
        ``E_s(lambda, w) = F_s_obs(lambda, w) - F_s_model(lambda, w)``, shape
        ``(n_lambda, n_wells)``.
    well_rms : dict[str, ArrayF]
        Per label, the per-well root-mean-square residual trace across
        wavelengths, ``R_s(w) = sqrt(mean_lambda E_s(lambda, w)**2)``, shape
        ``(n_wells,)``.
    se_boot : float
        Well-cluster bootstrap standard error of K (resampling entire wells
        with all their scans and wavelengths kept together; ``nan`` when
        ``n_boot == 0``).
    ci_boot : tuple[float, float]
        95 % percentile interval ``(q025, q975)`` from the well-cluster
        bootstrap (``(nan, nan)`` when ``n_boot == 0``).
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
    kept: np.ndarray
    """Wells kept by the well screen, all True when it was off."""
    residuals: dict[str, ArrayF]
    well_rms: dict[str, ArrayF]
    se_boot: float = np.nan
    ci_boot: tuple[float, float] = (np.nan, np.nan)


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


def paired_delta(
    value_a: float,
    value_b: float,
    replicates_a: Sequence[float],
    replicates_b: Sequence[float],
    *,
    z: float = 1.96,
) -> dict[str, float]:
    """Difference of two estimates that came from the same replicates.

    Two scans of one plate are not two experiments: they see the same wells, the
    same preparation and the same pipetting, so their K values are correlated and
    the variance of their difference is *not* the sum of the variances. With
    paired replicates (each delete-one-well refit gives one value per scan, or
    each bootstrap draw gives one per scan) the correlation is preserved by
    construction, and

        Var(a - b) = Var(a) + Var(b) - 2 Cov(a, b)

    is estimated directly from the paired differences. Quoting the individual
    errors in quadrature would overstate the difference by about
    ``1 / sqrt(1 - rho)`` - a factor 2-3 at the rho these plates show.

    Parameters
    ----------
    value_a : float
        The first reported estimate (the fit's own K, not a replicate mean).
    value_b : float
        The second reported estimate, from the other scan of the same wells.
    replicates_a : Sequence[float]
        Replicate estimates of the first scan.
    replicates_b : Sequence[float]
        Replicate estimates of the second scan, in the same order: replicate ``i``
        of ``a`` and of ``b`` must come from the same resampled or deleted unit.
    z : float
        Width of the returned interval, in standard errors (1.96 = 95 %).

    Returns
    -------
    dict[str, float]
        ``delta`` (``value_a - value_b``), ``se_paired`` and ``ci_lo``/``ci_hi``
        from the paired differences, ``se_independent`` (the quadrature sum that
        would be wrong), ``rho`` (the replicate-level correlation) and ``n``.
        ``se_paired`` is ``nan`` when fewer than three replicates are given.

    Raises
    ------
    ValueError
        If the two replicate sequences differ in length.
    """
    a = np.asarray(replicates_a, dtype=np.float64)
    b = np.asarray(replicates_b, dtype=np.float64)
    if a.size != b.size:
        msg = f"paired replicates must have the same length, got {a.size} and {b.size}"
        raise ValueError(msg)
    delta = float(value_a - value_b)
    n = int(a.size)
    out = {
        "delta": delta,
        "se_paired": float("nan"),
        "ci_lo": float("nan"),
        "ci_hi": float("nan"),
        "se_independent": float("nan"),
        "rho": float("nan"),
        "n": float(n),
    }
    if n < 3:  # ruff: ignore[magic-value-comparison] - two replicates cannot give a covariance
        return out
    d = a - b
    se_paired = float(np.std(d, ddof=1))
    se_ind = float(np.sqrt(np.var(a, ddof=1) + np.var(b, ddof=1)))
    rho = float(np.corrcoef(a, b)[0, 1])
    out.update({
        "se_paired": se_paired,
        "ci_lo": delta - z * se_paired,
        "ci_hi": delta + z * se_paired,
        "se_independent": se_ind,
        "rho": rho,
    })
    return out


def _bootstrap_wells(  # ruff: ignore[too-many-arguments]
    x: ArrayF,
    spectra: Sequence[ArrayF],
    model: _Model,
    *,
    theta0: ArrayF,
    is_ph: bool,
    n_boot: int,
    seed: int,
    strata: Sequence[object] | None = None,
) -> tuple[float, tuple[float, float]]:
    """Bootstrap K by resampling whole wells, never single wavelengths.

    The observations of a titration are the wells, not the pixels. Every well
    contributes one spectrum per scan, all measured on one preparation at one
    target value, and the wavelengths of a spectrum share their error (lamp
    drift, pipetting, focus). Resampling single wavelengths multiplies the
    apparent sample size by a few hundred and shrinks the interval with it - the
    same mistake the Jacobian makes, and the reason the naive standard error of
    these fits comes out orders of magnitude too small.

    A draw takes wells with replacement; all scans and all wavelengths of a drawn
    well move together. ``strata`` keeps the design intact: resampling inside a
    plate column keeps every replicate spanning the whole titration axis, which
    is what makes the resampled fit identifiable.

    Replicates start from the best fit, so each is one short least-squares solve
    rather than a fresh grid search.

    Parameters
    ----------
    x : ArrayF
        Titrant value per well, already screened.
    spectra : Sequence[ArrayF]
        One ``(n_lambda, n_wells)`` matrix per label.
    model : _Model
        The model the fit used.
    theta0 : ArrayF
        Best-fit parameters, the starting point of every replicate.
    is_ph : bool
        pH titration (else ligand binding).
    n_boot : int
        Number of replicates.
    seed : int
        Seed, so the interval reproduces.
    strata : Sequence[object] | None
        Optional stratum label per well; ``None`` resamples all wells as one pool.

    Returns
    -------
    tuple[float, tuple[float, float]]
        Standard deviation of the bootstrap K values and their 2.5/97.5
        percentiles; ``(nan, (nan, nan))`` when fewer than two replicates were
        usable.
    """
    rng = np.random.default_rng(seed)
    n_wells = x.size
    if strata is None:
        groups = [np.arange(n_wells)]
    else:
        labels = np.asarray(list(strata), dtype=object)
        groups = [np.flatnonzero(labels == key) for key in dict.fromkeys(labels)]
        groups = [g for g in groups if g.size]
    draws: list[float] = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(g, size=g.size, replace=True) for g in groups])
        if np.unique(idx).size < _MIN_GLOBAL_WELLS:
            continue
        theta, *_ = _solve(
            x[idx], [y[:, idx] for y in spectra], model, is_ph=is_ph, theta0=theta0
        )
        draws.append(float(theta[0]))
    if len(draws) < 2:  # ruff: ignore[magic-value-comparison] - two draws are the minimum for a standard deviation
        return float("nan"), (float("nan"), float("nan"))
    values = np.asarray(draws, dtype=np.float64)
    lo, hi = np.quantile(values, [0.025, 0.975])
    return float(np.std(values, ddof=1)), (float(lo), float(hi))


def fit_spectra_global(  # ruff: ignore[too-many-arguments]
    x: ArrayF,
    spectra: Mapping[str, tuple[ArrayF, ArrayF]],
    *,
    is_ph: bool = True,
    fit_hill: bool = False,
    well_scale: bool = False,
    acid_state: bool = False,
    screen: float | None = None,
    jackknife: bool = True,
    n_boot: int = 0,
    seed: int = 0,
    strata: Sequence[object] | None = None,
) -> SpectralFit:
    """Fit K to whole spectra of one or more labels, species spectra eliminated linearly.

    Parameters
    ----------
    x : ArrayF
        Titrant value (pH or concentration) per well.
    spectra : Mapping[str, tuple[ArrayF, ArrayF]]
        Label to ``(wavelengths, Y)`` with ``Y`` of shape (n_lambda, n_wells), already corrected
        per well (buffer subtracted). Wavelengths holding a
        non-finite value in any well are dropped.
    is_ph : bool
        pH titration (Henderson-Hasselbalch) or ligand binding.
    fit_hill : bool
        Also fit the Hill slope (bounded to 0.2-5).
    well_scale : bool
        Give each well a free amplitude shared by its labels (protein amount, mean 1). Use it to
        absorb what is left of the
        well-to-well amount error.
    acid_state : bool
        Add a third state below the main transition (pH titrations only): sequential
        deprotonated <-> protonated <-> acid state with pK2 < K, its spectrum solved like the
        others. It models a change of brightness or colour at the acid end that a two-state
        model can only push into the well amplitudes (see the module notes on the IBF plates).
    screen : float | None
        Family-wise level of the well screen, e.g. ``0.05``: fit once, flag the wells whose
        standardized whole-spectrum residual is beyond the chi-square cutoff (Bonferroni over
        wells, see :func:`_well_screen`), refit on the rest, and report which wells were kept
        in :attr:`SpectralFit.kept`. ``None`` keeps every well - the band path screens single
        (well, band) points by default, this fit does not screen at all, and a comparison
        across the two must say which is which.
    jackknife : bool
        Compute the delete-one-well jackknife standard error (one refit per well).
    n_boot : int
        Number of hierarchical well-cluster bootstrap replicates (0 disables).
        Each bootstrap draw resamples entire wells with replacement so all
        scans and all wavelengths of a resampled well stay coupled.
    seed : int
        Random seed for the well-cluster bootstrap.
    strata : Sequence[object] | None
        Optional stratum label per well (e.g. plate column ``01..08``). When
        given, wells are resampled with replacement within each stratum so
        every bootstrap replicate spans the full titration design.

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
    if xa.size < _MIN_GLOBAL_WELLS:
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
    kept = np.ones(xa.size, dtype=bool)
    if screen is not None:
        c = _fractions(xa, theta, model, is_ph=is_ph)
        _, res, _ = _project(c, ys, well_scale=well_scale, sigma=sigma)
        kept = _well_screen(res, sigma, alpha=screen)
        if not kept.all():
            xa = xa[kept]
            # ys are (wavelength, well): the kept wells are columns
            ys = [y[:, kept] for y in ys]
            theta, se_naive, sigma = _solve(xa, ys, model, is_ph=is_ph)
    c = _fractions(xa, theta, model, is_ph=is_ph)
    species, res, scale = _project(c, ys, well_scale=well_scale, sigma=sigma)
    weighted = np.hstack([r / s for r, s in zip(res, sigma, strict=True)])
    # res[j] has shape (n_wells, n_lambda); store (n_lambda, n_wells) and R_s(w)
    resid_by_label = {
        label: np.asarray(r.T, dtype=np.float64)
        for label, r in zip(labels, res, strict=True)
    }
    well_rms_by_label = {
        label: np.asarray(np.sqrt(np.mean(r**2, axis=1)), dtype=np.float64)
        for label, r in zip(labels, res, strict=True)
    }
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
    se_boot = np.nan
    ci_boot = (np.nan, np.nan)
    if n_boot > 0:
        se_boot, ci_boot = _bootstrap_wells(
            xa,
            ys,
            model,
            theta0=theta,
            is_ph=is_ph,
            n_boot=n_boot,
            seed=seed,
            strata=(
                None
                if strata is None
                else [str(v) for v in np.asarray(list(strata), dtype=object)[kept]]
            ),
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
        kept=kept,
        residuals=resid_by_label,
        well_rms=well_rms_by_label,
        se_boot=se_boot,
        ci_boot=ci_boot,
    )


def _clean_spectra(
    x: ArrayF,
    spectra: Mapping[str, tuple[ArrayF, ArrayF]],
    *,
    x_sigma: float,
) -> tuple[np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Validate the inputs and drop wavelengths non-finite in any well.

    Parameters
    ----------
    x : ArrayF
        Titrant value (pH or concentration) per well.
    spectra : Mapping[str, tuple[ArrayF, ArrayF]]
        Label to ``(wavelengths, Y)`` with ``Y`` of shape (n_lambda, n_wells),
        already corrected per well.
    x_sigma : float
        Prior SD of each well's latent axis value.

    Returns
    -------
    tuple[np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]
        The validated titrant axis and, per label, the cleaned
        ``(wavelengths, Y)`` pair with non-finite wavelengths dropped.

    Raises
    ------
    ValueError
        With fewer than four wells, a non-finite or non-positive ``x_sigma``,
        non-finite ``x``, or no scan with finite wavelengths.
    """
    xa = np.asarray(x, dtype=np.float64)
    if xa.size < _MIN_GLOBAL_WELLS or not np.isfinite(x_sigma) or x_sigma <= 0:
        msg = "global Bayes needs >=4 wells and a positive finite x_sigma"
        raise ValueError(msg)
    if not np.isfinite(xa).all():
        msg = "global Bayes needs a finite titrant axis"
        raise ValueError(msg)
    clean: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, (lam, values) in spectra.items():
        y = np.asarray(values, dtype=np.float64)
        keep = np.all(np.isfinite(y), axis=1)
        if keep.any():
            clean[name] = (np.asarray(lam, dtype=np.float64)[keep], y[keep])
    if not clean:
        msg = "global Bayes needs at least one scan with finite wavelengths"
        raise ValueError(msg)
    return xa, clean


def fit_spectra_global_bayes(  # ruff: ignore[too-many-arguments]
    x: ArrayF,
    spectra: Mapping[str, tuple[ArrayF, ArrayF]],
    *,
    x_sigma: float,
    is_ph: bool = True,
    acid_state: bool = False,
    species_basis: int | None = 24,
    seed: int = 0,
    draws: int = 2000,
    chains: int = 2,
    cores: int | None = None,
    target_accept: float = 0.9,
    tune: int | None = None,
) -> az.InferenceData:
    """Bayesian whole-spectrum global fit with per-well latent titrant values.

    This experimental counterpart to :func:`fit_spectra_global` retains its
    mixture model and per-well amplitude: two species spectra, or three with
    ``acid_state``. Species spectra are nuisance parameters; scan-specific
    Gaussian residual scales are inferred jointly. Wavelength residuals are
    conditionally independent within each scan, an approximation that must be
    considered when interpreting posterior widths.

    The priors are seeded from a :func:`fit_spectra_global` run: ``E0``/``E1``
    (and ``E2``) are centered on the projected species spectra with a width of
    five residual RMS, a mild empirical-Bayes reuse of the same data (constant
    across ``K``, so the shape of the ``K`` posterior is not affected; the
    species draws are systematically a little narrow). Species state ordering
    follows :func:`~clophfit.fitting.models.binding_1site`'s convention: for
    ``is_ph=True`` state 1 carries the *low-pH* weight ``1 / (1 + 10**(pH -
    K))``, hence ``fraction = sigmoid(ln10 * (K - x_true))`` and ``E0`` tracks
    the high-pH spectrum.

    ``acid_state`` adds a third state below the main transition with the
    conventions of the least-squares twin (:func:`fit_spectra_global`): the
    sequential diprotic softmax of :func:`_fractions_rows`, a sampled
    ``delta`` so that ``pK2 = K - delta``, and its own ``E2_label`` species
    spectra, seeded from the acid-state least-squares solution. It is the
    same model the marginalized version samples
    (:func:`fit_spectra_global_bayes_marginal`), here with the species
    spectra kept as free parameters; the traces stay drop-in comparable.

    Input validation is the helper's (:func:`_clean_spectra`): fewer than
    four wells, a non-finite or non-positive ``x_sigma``, a non-finite
    ``x``, or no scan with finite wavelengths raise ``ValueError``.

    Parameters
    ----------
    x : ArrayF
        Titrant value (pH or concentration) per well.
    spectra : Mapping[str, tuple[ArrayF, ArrayF]]
        Label to ``(wavelengths, Y)`` with ``Y`` of shape (n_lambda, n_wells),
        already corrected per well; wavelengths holding a non-finite value in
        any well are dropped.
    x_sigma : float
        Prior SD of each well's latent axis value: ``x_true ~ N(x, x_sigma)``.
    is_ph : bool
        Henderson-Hasselbalch (pH) or Langmuir (ligand) titration axis.
    acid_state : bool
        Add the third (acid) state, as in the least-squares twin. pH
        titrations only; a ligand titration raises ``ValueError``.
    species_basis : int | None
        Coefficients of the clamped cubic B-spline basis the species spectra
        are sampled in, or ``None`` for one free value per wavelength (the
        historical parameterization). The basis is not a convenience: with one
        parameter per wavelength the species block is ~450-dimensional on a
        226-wavelength scan, the K posterior moves by fewer than 15 effective
        draws at 2000 draws, and no budget tested recovers it, because a
        coordinated move of the spectra can mimic a shift of K. At 24
        coefficients the same model reaches bulk ESS of a few hundred and
        reproduces the least-squares species to within the residual scale.
        ``E0_``/``E1_``/``E2_`` are still written at full wavelength
        resolution, so the outputs are unchanged.
    seed : int
        Random seed; a fixed seed reproduces the posterior.
    draws : int
        Posterior draws per chain.
    chains : int
        Number of NUTS chains.
    cores : int | None
        Chain parallelism; None chains in parallel.
    target_accept : float
        NUTS target acceptance, for better mixing at the price of step size.
    tune : int | None
        Tuning draws; None keeps ``draws // 2``.

    Returns
    -------
    az.InferenceData
        Posterior draws for ``K``, the latent axis per well, the per-well
        amplitude and the reconstructed species spectra and noise scales
        for each scan, plus ``delta`` with an acid state, with the
        log-likelihood attached so the comparison scripts can compute elpd
        across methods.

    Raises
    ------
    ValueError
        With fewer than four wells, no usable wavelength, a non-finite or
        non-positive ``x_sigma``, or an acid state on a ligand titration.
    """
    xa, clean = _clean_spectra(x, spectra, x_sigma=x_sigma)
    if acid_state and not is_ph:
        msg = "the acid state is defined for pH titrations only"
        raise ValueError(msg)
    init = fit_spectra_global(
        xa, clean, is_ph=is_ph, acid_state=acid_state, well_scale=True, jackknife=False
    )
    lo, hi = _k_bounds(xa, is_ph=is_ph)
    states = 3 if acid_state else 2
    # Seed the chains from the least-squares solution, as the marginalized
    # twin does: the pK of the acid state is only identified relative to K.
    theta_start = {"K": float(np.clip(init.K, lo + 1e-9, hi - 1e-9))}
    if acid_state:
        theta_start["delta"] = float(
            np.clip(init.K - init.k_acid, _DELTA_MIN + 1e-9, _DELTA_MAX - 1e-9)
        )
    with pm.Model():
        k = pm.Uniform("K", lower=lo, upper=hi)
        delta = (
            pm.Uniform("delta", lower=_DELTA_MIN, upper=_DELTA_MAX)
            if acid_state
            else None
        )
        x_true = pm.Normal("x_true", mu=xa, sigma=x_sigma, shape=xa.size)
        fraction = _fractions_rows(
            k, delta, None, x_true, is_ph=is_ph, acid_state=acid_state
        )
        amp_raw = pm.LogNormal("amplitude_raw", mu=0.0, sigma=0.25, shape=xa.size)
        amplitude = pm.Deterministic(
            "amplitude",
            amp_raw / pt.mean(amp_raw),  # type: ignore[no-untyped-call]
        )
        for name, (_lam, y) in clean.items():
            species = init.species[name]
            resid_scale = max(init.rms[name], 1e-6)
            n_lam = y.shape[0]
            n_coef = int(species_basis) if species_basis is not None else 0
            use_basis = _MIN_BASIS <= n_coef < n_lam
            if use_basis:
                basis = _species_basis(n_lam, n_coef)
                e_states = [
                    pm.Deterministic(
                        f"E{s}_{name}",
                        pt.dot(  # type: ignore[no-untyped-call]
                            pt.as_tensor(basis),
                            pm.Normal(
                                f"c{s}_{name}",
                                mu=np.asarray(basis.T @ species[s], dtype=np.float64),
                                sigma=5 * resid_scale,
                                shape=basis.shape[1],
                            ),
                        ),
                    )
                    for s in range(states)
                ]
            else:
                e_states = [
                    pm.Normal(
                        f"E{s}_{name}",
                        mu=species[s],
                        sigma=5 * resid_scale,
                        shape=n_lam,
                    )
                    for s in range(states)
                ]
            # fraction's columns are the states, so the mixture is one dot
            m_stack = pt.stack(e_states)
            sigma = pm.HalfNormal(f"sigma_{name}", sigma=2 * resid_scale)
            mu = amplitude[:, None] * pt.dot(fraction, m_stack)  # type: ignore[no-untyped-call]
            pm.Normal(f"obs_{name}", mu=mu, sigma=sigma, observed=y.T)
        # The least-squares K starts the chains (clipped into the Uniform's
        # interior, a bound-hugging least-squares K would start NUTS on an edge).
        # It passes as pm.sample(initvals=...), sampler-level, not as a model
        # initval: posterior machinery (log-likelihood computation) rejects a
        # graph with non-default rvs_to_initial_values under the nutpie backend.
        idata = pm.sample(
            draws,
            tune=draws // 2 if tune is None else tune,
            chains=chains,
            cores=cores,
            target_accept=target_accept,
            random_seed=seed,
            initvals=theta_start,
            return_inferencedata=True,
            progressbar=False,
        )
        pm.compute_log_likelihood(idata)  # for elpd comparisons across methods
        return idata


_RIDGE = 4.0  # precision of the level/linear drift: drift SD ~0.5 in normalized units


def _well_screen(
    residuals: Sequence[np.ndarray], sigma: Sequence[float], *, alpha: float = 0.05
) -> np.ndarray:
    """Flag wells whose whole-spectrum residual is larger than the scan's noise.

    The band path screens individual (well, band) points with a studentized
    residual. A whole-spectrum fit has no such points: every well contributes a
    spectrum, so the analogue is a *well* whose standardized residual norm is
    beyond what the scan's own residual scale predicts. The statistic is the
    squared norm of a well's residuals divided by a robust scale, which is
    chi-square with as many degrees of freedom as that well has wavelengths; a
    well is dropped when it is in that tail at ``alpha`` **and** above three
    times the plate's median statistic, so a clean plate does not lose a well to
    the nominal level alone (the shared species spectra eat a few degrees of
    freedom).

    Parameters
    ----------
    residuals : Sequence[np.ndarray]
        Per scan, the ``(n_wells, n_lambda)`` residual matrix of a first fit.
    sigma : Sequence[float]
        Per scan, the residual scale of that fit, used as a floor for the robust
        scale (never as the scale itself: a broken well inflates it and
        standardizes itself away).
    alpha : float
        Family-wise level of the chi-square tail, e.g. 0.05.

    Returns
    -------
    np.ndarray
        Boolean mask of the wells to keep.
    """
    # The scale must be robust, not the fitted one: the residual scale is
    # estimated from all the wells, so a broken well inflates it and standardizes
    # itself away (masking). The median absolute residual does not move for one
    # bad well in thirty, and 1.4826 x MAD is the Gaussian-consistent scale.
    d2 = np.zeros(len(residuals[0]))
    dof = 0
    for res, sd in zip(residuals, sigma, strict=True):
        robust = 1.4826 * float(np.median(np.abs(res)))
        scale = max(min(robust, sd), 1e-12)
        d2 += np.sum((res / scale) ** 2, axis=1)
        dof += res.shape[1]
    d2 /= len(residuals)
    cutoff = float(chi2.ppf(1.0 - alpha, max(dof, 1)))
    # Two conditions, both conservative: the well must be in the chi-square tail
    # *and* stand out from the plate's own wells (three times the median d2), so a
    # clean plate with the shared species eating a few degrees of freedom does not
    # lose a well to the nominal alpha alone.
    keep = (d2 <= cutoff) | (d2 <= 3.0 * float(np.median(d2)))
    minimum = max(6, len(d2) // 2)
    if int(keep.sum()) < minimum:  # a screen that eats the plate is not a screen
        order = np.argsort(d2)
        keep = np.zeros_like(keep)
        keep[order[:minimum]] = True
    return keep


def _species_basis(n_lam: int, n_coef: int) -> np.ndarray:
    """Return an orthonormal clamped cubic B-spline design matrix (n_lam, n_coef).

    The species spectra of a fluorescence titration are smooth, so sampling one
    value per wavelength gives the sampler hundreds of nearly redundant
    parameters: on the 226-wavelength excitation scan the free-parameter
    posterior does not move (bulk ESS of K below 15 at any usable budget), while
    a 24-coefficient basis reaches an ESS in the hundreds. Clamped cubic knots
    keep the first and last wavelength exact, which a Fourier basis does not: the
    scan ends inside the anionic band and a truncated cosine fit rings there.

    Columns are orthonormalized so a Normal prior of width ``w`` on the
    coefficients induces the same ``w`` per wavelength that the per-wavelength
    parameterization used.
    """
    degree = 3
    grid = np.linspace(0.0, 1.0, n_lam)
    n_inner = max(n_coef - degree - 1, 1)
    inner = np.linspace(0.0, 1.0, n_inner + 2)[1:-1]
    knots = np.concatenate([np.zeros(degree + 1), inner, np.ones(degree + 1)])
    design = BSpline.design_matrix(grid, knots, degree).toarray()
    basis, _ = np.linalg.qr(design)
    return np.asarray(basis, dtype=np.float64)


def _as_tensor(value: object) -> pm.math.TensorVariable:
    """Wrap ``value`` as a typed pytensor variable (type checker pacifier)."""
    return pt.as_tensor_variable(
        value  # type: ignore[arg-type, unused-ignore]
    )


_SMOOTH_RHO = 10.0  # for the per-well smooth-noise correlation (see below)


def _smooth_corr(n: int) -> np.ndarray:
    """Unit-normalized smooth error correlation along wavelength.

    ``Sigma = inv(SMOOTH_RHO D'D + I)``: level and linear drifts have unit
    SD, higher-order shapes are shrunk, so ``smooth`` amplitude draws scale
    the whole block and cover slowly varying per-well residuals (an AR(1) or
    offset-sized error) without touching the shared species spectra.
    """
    d2 = _second_difference(n)
    return np.linalg.inv(_SMOOTH_RHO * (d2.T @ d2) + np.eye(n))


def _second_difference(n: int) -> np.ndarray:
    """Build the (n-2, n) second-difference matrix; a smooth spectrum has tiny rows."""
    d2 = np.zeros((n - 2, n))
    for i in range(n - 2):
        d2[i, i : i + 3] = (1.0, -2.0, 1.0)
    return d2


def _fractions_rows(  # ruff: ignore[too-many-arguments]
    k: pm.math.TensorVariable,
    delta: pm.math.TensorVariable | float | None,
    hill: pm.math.TensorVariable | float | None,
    axis: pm.math.TensorVariable,
    *,
    is_ph: bool,
    acid_state: bool,
) -> pm.math.TensorVariable:
    """Symbolic state-fraction rows (columns S0..): the pytensor twin of ``_fractions``.

    As there, ``hill`` stays 1 and ``delta`` unused outside the branches that
    need them; the acid-state columns are the sequential diprotic softmax.
    The parameters are pytensor variables inside the model, floats for the
    plain-function path.
    """
    ln10 = np.log(10.0)
    if not acid_state:
        h = 1.0 if hill is None else hill
        if is_ph:
            f = pm.math.sigmoid(h * ln10 * (k - axis))
        else:
            ratio = (axis / k) ** h
            f = ratio / (1 + ratio)
        return pt.stack([1 - f, f], axis=1)
    if delta is None:
        msg = "the acid state needs a delta"
        raise ValueError(msg)
    h = 1.0 if hill is None else hill
    k2 = k - delta
    logs = pt.stack(
        [
            pt.zeros_like(axis),  # type: ignore[no-untyped-call]
            h * (k - axis) * ln10,
            (h * (k - axis) + (k2 - axis)) * ln10,
        ],
        axis=1,
    )
    mx = pt.max(logs, axis=1, keepdims=True)  # type: ignore[no-untyped-call]
    ex = pt.exp(logs - mx)
    return ex / pt.sum(ex, axis=1, keepdims=True)  # type: ignore[no-untyped-call]


def _conditional_species_draws(  # ruff: ignore[too-many-arguments, too-many-statements]
    idata: az.InferenceData,
    clean: dict[str, tuple[np.ndarray, np.ndarray]],
    means: dict[str, np.ndarray],
    *,
    is_ph: bool,
    fit_hill: bool,
    acid_state: bool,
    seed: int,
) -> None:
    """Reconstruct conditional species-spectrum draws and attach them in place.

    With the species spectra marginalized out of the likelihood the sampler
    never draws them; for each posterior draw this draws once from the exact
    Gaussian conditional of the stacked ``(E_0..E_{S-1}, b_w)`` vector given
    all wells (the species prior is centered on ``means``), with the
    per-well ``smooth`` residuals integrated out of ``(E, b)``'s likelihood
    block. The species spectra stay available to the plots and downstream
    checks exactly as in the free-parameter version.
    """
    post = idata.posterior
    chains, ndraws = post.dims["chain"], post.dims["draw"]

    def flat(v: str) -> np.ndarray:
        arr = np.asarray(post[v].values)
        return arr.reshape(chains * ndraws, *arr.shape[2:])

    k_draws = flat("K")
    delta_draws = flat("delta") if acid_state else np.zeros_like(k_draws)
    hill_draws = flat("hill") if fit_hill else np.ones_like(k_draws)
    x_draws = flat("x_true")
    amp = flat("amplitude")
    n_wells = x_draws.shape[1]
    total = x_draws.shape[0]
    model = _Model(acid_state=acid_state, fit_hill=fit_hill, well_scale=False)
    rng = np.random.default_rng(seed + 1)
    for name, (_lam, y) in clean.items():
        n_lam = y.shape[0]
        curv = _second_difference(n_lam).T @ _second_difference(n_lam)
        bmat = _smooth_corr(n_lam)
        ones_l = np.ones(n_lam)
        s_e = flat(f"species_scale_{name}")
        sig = flat(f"sigma_{name}")
        s_b = flat(f"offset_{name}")
        smooth = flat(f"smooth_{name}")
        states = 3 if acid_state else 2
        out = np.empty((total, states, n_lam))
        for i in range(total):
            theta = [float(k_draws[i])]
            if acid_state:
                theta.append(float(delta_draws[i]))
            if fit_hill:
                theta.append(float(hill_draws[i]))
            c = _fractions(x_draws[i], np.asarray(theta), model, is_ph=is_ph)
            # exact Gaussian conditional of (E stacked, per-well offsets b_w):
            # y_w | E, b_w ~ N(..., sigma^2 I + smooth^2 B) per well, so the
            # likelihood terms use that block's inverse instead of 1/sigma^2
            prec_e = curv / s_e[i] ** 2 + _RIDGE * np.eye(n_lam)
            uinv = np.linalg.inv(smooth[i] ** 2 * bmat + sig[i] ** 2 * np.eye(n_lam))
            one_u = uinv @ ones_l
            prec = np.zeros((states * n_lam + n_wells, states * n_lam + n_wells))
            gmat = np.einsum("w,ws,wt->st", amp[i] ** 2, c, c)
            prec[: states * n_lam, : states * n_lam] = np.kron(np.eye(states), prec_e)
            prec[: states * n_lam, : states * n_lam] += np.kron(gmat, uinv)
            for w in range(n_wells):
                a_c = amp[i][w] * c[w]  # (S,)
                for s in range(states):
                    prec[s * n_lam : (s + 1) * n_lam, states * n_lam + w] = (
                        a_c[s] * one_u
                    )
                    prec[states * n_lam + w, s * n_lam : (s + 1) * n_lam] = (
                        a_c[s] * one_u
                    )
                prec[states * n_lam + w, states * n_lam + w] = 1.0 / s_b[
                    i
                ] ** 2 + float(one_u.sum())
            prec[np.diag_indices(states * n_lam + n_wells)] += 1e-9
            rhs = np.zeros(states * n_lam + n_wells)
            auy = (uinv @ y) * amp[i]  # (n_lam, n_wells)
            for s in range(states):
                rhs[s * n_lam : (s + 1) * n_lam] = (
                    prec_e @ means[name][s] + auy @ c[:, s]
                )
            rhs[states * n_lam :] = (uinv @ y).sum(axis=0) / sig[i] ** 2
            chol = np.linalg.cholesky(prec)
            step = np.linalg.solve(chol, rhs) + rng.standard_normal(chol.shape[0])
            draw = np.linalg.solve(chol.T, step)
            out[i] = draw[: states * n_lam].reshape(states, n_lam)
        for s in range(states):
            post[f"E{s}_{name}"] = xr.DataArray(
                out.reshape(chains, ndraws, states, n_lam)[:, :, s, :],
                dims=("chain", "draw", f"lambda_{name}"),
            )


def _joint_marginal_logp(  # ruff: ignore[too-many-arguments]
    y_flat: pm.math.TensorVariable,
    mu_flat: pm.math.TensorVariable,
    q_wells: pm.math.TensorVariable,
    inwell: pm.math.TensorVariable,
    prec_e: pm.math.TensorVariable,
    *,
    states: int,
    n_wells: int,
    n_lam: int,
) -> pm.math.TensorVariable:
    """Exact log density of the stacked-wells marginal Gaussian.

    The marginal covariance over all wells' wavelengths is
    ``kron(U U', Sigma_E) + kron(I_wells, inwell)`` with ``U = q_wells`` of
    rank <= ``states``, i.e. ``A + W M W'`` with ``A = kron(I, inwell)`` and
    ``M = kron(I_states, Sigma_E)``; by the matrix determinant lemma the
    quadratic form and log determinant reduce to per-well and
    ``(states n_lam)``-sized algebra instead of one dense
    ``(n_wells n_lam)^3`` factorization. ``prec_e`` is ``Sigma_E^{-1}``.
    """
    n_eff = n_wells * n_lam
    jit = _as_tensor(np.eye(n_lam)) * 1e-9  # keeps the cholesky finite
    chol_in = pt.linalg.cholesky(inwell + jit)
    eye_l = _as_tensor(np.eye(n_lam))
    inwell_inv = pt.linalg.solve_triangular(
        pt.transpose(chol_in),  # type: ignore[no-untyped-call,unused-ignore]
        pt.linalg.solve_triangular(chol_in, eye_l, lower=True),  # type: ignore[no-untyped-call,unused-ignore]
        lower=False,
    )  # type: ignore[no-untyped-call,unused-ignore]
    chol_pe = pt.linalg.cholesky(prec_e + jit)
    logdet_pe = 2 * pt.sum(
        pt.log(pt.diagonal(chol_pe, axis1=-2, axis2=-1))  # type: ignore[no-untyped-call,unused-ignore]
    )  # type: ignore[no-untyped-call,unused-ignore]
    resid = (y_flat - mu_flat).reshape((n_wells, n_lam))
    al = pt.linalg.solve_triangular(
        chol_in,
        pt.transpose(resid),  # type: ignore[no-untyped-call,unused-ignore]
        lower=True,
    )  # type: ignore[no-untyped-call,unused-ignore]
    quad = pt.sum(al**2)  # type: ignore[no-untyped-call,unused-ignore]
    ra = pt.linalg.solve_triangular(
        pt.transpose(chol_in),  # type: ignore[no-untyped-call,unused-ignore]
        al,  # type: ignore[no-untyped-call,unused-ignore]
        lower=False,
    )  # type: ignore[no-untyped-call,unused-ignore]  # inwell^{-1} r
    z = pt.dot(
        pt.transpose(q_wells),  # type: ignore[no-untyped-call,unused-ignore]
        pt.transpose(ra),  # type: ignore[no-untyped-call,unused-ignore]
    )  # type: ignore[no-untyped-call,unused-ignore]
    gg = pt.dot(
        pt.transpose(q_wells),  # type: ignore[no-untyped-call,unused-ignore]
        q_wells,  # type: ignore[no-untyped-call,unused-ignore]
    )  # type: ignore[no-untyped-call,unused-ignore]
    qmat = (
        pt.eye(states)[:, None, :, None] * prec_e[None, :, None, :]  # type: ignore[no-untyped-call,unused-ignore]
        + gg[:, None, :, None] * inwell_inv[None, :, None, :]
    ).reshape((states * n_lam, states * n_lam))
    chol_q = pt.linalg.cholesky(qmat + _as_tensor(np.eye(states * n_lam)) * 1e-9)
    alpha_q = pt.linalg.solve_triangular(
        chol_q,
        z.reshape((states * n_lam, 1)),  # type: ignore[no-untyped-call,unused-ignore]
        lower=True,
    )  # type: ignore[no-untyped-call,unused-ignore]
    quad -= pt.sum(alpha_q**2)  # type: ignore[no-untyped-call,unused-ignore]
    logdet_in = 2 * pt.sum(
        pt.log(pt.diagonal(chol_in, axis1=-2, axis2=-1))  # type: ignore[no-untyped-call,unused-ignore]
    )  # type: ignore[no-untyped-call,unused-ignore]
    logdet_q = 2 * pt.sum(
        pt.log(pt.diagonal(chol_q, axis1=-2, axis2=-1))  # type: ignore[no-untyped-call,unused-ignore]
    )  # type: ignore[no-untyped-call,unused-ignore]
    # log|cov| = n_wells log|inwell| - S log|P_E| + log|Q|
    logdet = n_wells * logdet_in - states * logdet_pe + logdet_q
    return -0.5 * (quad + logdet + n_eff * np.log(2 * np.pi))  # type: ignore[no-untyped-call,unused-ignore]


def fit_spectra_global_bayes_marginal(  # ruff: ignore[too-many-arguments]
    x: ArrayF,
    spectra: Mapping[str, tuple[ArrayF, ArrayF]],
    *,
    x_sigma: float,
    is_ph: bool = True,
    fit_hill: bool = False,
    acid_state: bool = False,
    seed: int = 0,
    draws: int = 2000,
    chains: int = 2,
    cores: int | None = None,
    target_accept: float = 0.9,
    tune: int | None = None,
) -> az.InferenceData:
    """Bayesian whole-spectrum global fit with the species spectra marginalized.

    The same generative model as :func:`fit_spectra_global_bayes` - two-state
    mixture, per-well amplitude, latent titrant axis - but the species spectra
    are eliminated analytically instead of sampled: they carry a Gaussian
    prior centered on the least-squares species spectra with a smooth
    covariance (second-difference precision scaled by ``species_scale``, the
    level/linear drift variance fixed), and, being linear in the likelihood,
    integrate out exactly. Because the species spectra are shared by every
    well, the exact marginal couples all wells: each scan contributes one
    multivariate Gaussian over all wells' wavelengths with cross-well blocks
    ``a_w a_w' (C_w . C_w') Sigma_E``, so the sampler holds a handful of
    physical parameters (``K``, ``x_true``, ``amplitude``, per-scan ``sigma``
    and ``offset`` scales) instead of two 30-60 dimensional species vectors
    per scan, removing the strongly correlated posterior geometry of the
    free-parameter version.

    Two refinements against :func:`fit_spectra_global_bayes`:

    - the species covariance is a smoothness prior with a data-free shape;
      the centering on the least-squares species spectra is the same
      documented empirical-Bayes reuse the free-parameter version makes, and
      it is what keeps the mean structure - and with it ``K`` - identified
      despite the per-well amplitude;
    - per-well noise is a two-component covariance: white ``sigma`` plus a
      wavelength-correlated ``smooth`` block (``inv(RHO D'D + I)``), plus an
      ``offset`` scale per scan contributing an ``s_b^2 * J`` block. Those
      give the sampler somewhere to put systematically shaped residuals - the
      correlated-along-wavelength errors noted in the module docstring -
      instead of letting the shared species spectra absorb them. Coverage
      testing with AR(1) wavelength noise (the evaluation is in the method-sweep
      summary, not in the package)
      shows the free-parameter version undercovers there (0.30-0.85 observed
      95% coverage) and the noise model here directly targets that failure;
      the marginalized fitter itself still undercovers in the same sweep
      (0.25-0.70), with
      the constrained MCR posterior best-calibrated from noise 0.06 up (0.80-0.95)
      - so verify coverage for a new data regime with the benchmark before
      trusting the intervals.

    ``fit_hill`` and ``acid_state`` are supported with the same conventions as
    the least-squares twin (:func:`fit_spectra_global`; the acid state is
    pH-only). The conditional species draws are reconstructed post hoc and
    attached as ``E0_label``/``E1_label`` (``E2_label`` with an acid state) so
    the output is drop-in compatible.

    No ``log_likelihood`` group is attached (the likelihood is a potential);
    for elpd comparisons use the free-parameter version.
    """
    xa, clean = _clean_spectra(x, spectra, x_sigma=x_sigma)
    init = fit_spectra_global(
        xa,
        clean,
        is_ph=is_ph,
        fit_hill=fit_hill,
        acid_state=acid_state,
        well_scale=True,
        jackknife=False,
    )
    lo, hi = _k_bounds(xa, is_ph=is_ph)
    top = max(float(np.max(np.abs(values))) for _lam, values in clean.values())
    species_mean = {
        name: np.asarray(init.species[name], dtype=np.float64) / top for name in clean
    }
    theta_start: dict[str, float] = {"K": float(np.clip(init.K, lo + 1e-9, hi - 1e-9))}
    if acid_state:
        if not is_ph:
            msg = "the acid state is defined for pH titrations only"
            raise ValueError(msg)
        theta_start["delta"] = float(
            np.clip(init.K - init.k_acid, _DELTA_MIN + 1e-9, _DELTA_MAX - 1e-9)
        )
    if fit_hill:
        theta_start["hill"] = float(np.clip(init.hill, 0.2, 5.0))
    with pm.Model():
        k = pm.Uniform("K", lower=lo, upper=hi)
        delta = (
            pm.Uniform("delta", lower=_DELTA_MIN, upper=_DELTA_MAX)
            if acid_state
            else None
        )
        hill = pm.Uniform("hill", lower=0.2, upper=5.0) if fit_hill else None
        x_true = pm.Normal("x_true", mu=xa, sigma=x_sigma, shape=xa.size)
        amp_raw = pm.LogNormal("amplitude_raw", mu=0.0, sigma=0.25, shape=xa.size)
        amplitude = pm.Deterministic(
            "amplitude",
            amp_raw / pt.mean(amp_raw),  # type: ignore[no-untyped-call]
        )
        fraction = _fractions_rows(
            k, delta, hill, x_true, is_ph=is_ph, acid_state=acid_state
        )
        states = 3 if acid_state else 2
        for name, (_lam, values) in clean.items():
            n_lam = values.shape[0]
            n_eff = xa.size * n_lam
            rms = max(float(init.rms[name]) / top, 1e-6)
            sb = pm.HalfNormal(f"offset_{name}", sigma=2 * rms)
            sigma = pm.HalfNormal(f"sigma_{name}", sigma=2 * rms)
            s_e = pm.HalfNormal(f"species_scale_{name}", sigma=5 * rms)
            smooth = pm.HalfNormal(f"smooth_{name}", sigma=5 * rms)
            curv = pt.as_tensor_variable(
                _second_difference(n_lam).T @ _second_difference(n_lam)
            )
            bmat = _as_tensor(_smooth_corr(n_lam))
            prec_e = curv / s_e**2 + _RIDGE * pt.eye(n_lam)  # type: ignore[no-untyped-call,unused-ignore]
            m_stack = pt.stack([
                _as_tensor(species_mean[name][s]) for s in range(states)
            ])  # type: ignore[no-untyped-call,unused-ignore]
            mu_flat = (
                amplitude[:, None] * pt.dot(fraction, m_stack)  # type: ignore[no-untyped-call]
            ).reshape((n_eff,))
            y_flat = _as_tensor((values / top).T).reshape((n_eff,))
            q_wells = amplitude[:, None] * fraction  # a_w C_w, rank <= S
            inwell = (
                sigma**2 * _as_tensor(np.eye(n_lam))
                + sb**2 * _as_tensor(np.ones((n_lam, n_lam)))
                + smooth**2 * bmat
            )
            logp = _joint_marginal_logp(
                y_flat,
                mu_flat,
                q_wells,
                inwell,
                prec_e,
                states=states,
                n_wells=xa.size,
                n_lam=n_lam,
            )
            pm.Potential(f"y_{name}", logp)
        idata = pm.sample(
            draws,
            tune=draws // 2 if tune is None else tune,
            chains=chains,
            cores=cores,
            target_accept=target_accept,
            random_seed=seed,
            initvals=theta_start,
            return_inferencedata=True,
            progressbar=False,
        )
    _conditional_species_draws(
        idata,
        clean,
        species_mean,
        is_ph=is_ph,
        fit_hill=fit_hill,
        acid_state=acid_state,
        seed=seed,
    )
    return idata
