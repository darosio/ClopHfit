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
# A closed profile column that varies by less than this across the wells
# carries no per-well information: the component is either wholly present
# in every well or wholly absent, so the decomposition is really a
# lower-order one written with more species. Closure puts a profile in
# [0, 1], so this is a scale-free statement about the fraction range.
_DEAD_PTP_ABS = 1e-6
# A column that is this small RELATIVE to the most mobile one is dead too.
#
# Both terms are needed. On its own the absolute test is unreliable: where
# the scale degeneracy parks the redundant column is a matter of the ALS
# start, so its absolute variation moves over many orders of magnitude
# between runs - on a synthetic three-component decomposition it lands
# anywhere between 1e-14 and 1.6e-6, which an absolute threshold either
# misses or catches only by luck. On its own the relative test fails too:
# when EVERY column is dead - which is what the real plate gives at
# three components - the most mobile column is itself tiny, so the
# threshold collapses onto it and nothing is flagged.
_DEAD_PTP_RATIO = 1e-3
_N_DIMS = 2  # a profile matrix has a well axis and a component axis


class McrFit(NamedTuple):
    """A converged MCR-ALS factorisation and its reconstruction quality."""

    c: np.ndarray
    """[n_wells, n_species] closed, non-negative concentration profiles."""
    s: np.ndarray
    """[n_species, n_lambda] non-negative species spectra (arbitrary scale)."""
    rel_err: float
    """||D - C S.T|| / ||D||."""
    dead_components: tuple[int, ...] = ()
    """Components whose profile never varies across the wells.

    A dead component is the signature of the MCR scale degeneracy: the
    species was parked at an arbitrary scale while its concentration
    column sat at a constant, so it holds no titration information.
    """

    @property
    def degenerate(self) -> bool:
        """Whether any component is dead, i.e. the decomposition is unusable.

        ``rel_err`` cannot detect this: a spurious component is a free
        lamp that lowers the residual while contributing nothing, so a
        degenerate fit can report a *better* error than a sound one.
        """
        return bool(self.dead_components)


def dead_components(
    c: np.ndarray,
    *,
    absolute: float = _DEAD_PTP_ABS,
    ratio: float = _DEAD_PTP_RATIO,
) -> tuple[int, ...]:
    """Find the concentration columns that carry no well-to-well change.

    A component whose profile is the same number in every well cannot be
    fitted for a K: the curve would be fitted to round-off, and the result
    is a confident, meaningless number.

    A column counts as dead when it varies by less than the absolute
    floor, or by less than a small fraction of the most mobile column.
    The absolute floor catches the case where every column is dead; the
    relative term catches a redundant column parked on a scale that is
    not itself tiny.

    Parameters
    ----------
    c : np.ndarray
        Closed concentration profiles ``[n_wells, n_species]``.
    absolute : float
        Peak-to-peak below which a column counts as dead outright.
    ratio : float
        A column whose peak-to-peak is below this fraction of the most
        mobile column's also counts as dead.

    Returns
    -------
    tuple[int, ...]
        The indices of the dead columns, in order.
    """
    if c.ndim != _N_DIMS or c.size == 0:
        return ()
    ptp = np.ptp(c, axis=0)
    threshold = max(absolute, ratio * float(np.max(ptp)))
    return tuple(int(i) for i in np.flatnonzero(ptp < threshold))


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
    empty = clipped.sum(axis=0) <= 0.0
    if empty.any():
        clipped[:, empty] = np.abs(signed.T[:, empty])
    starts.extend((clipped, np.abs(loadings.T)))
    best = _als_run(d, starts[0], closure=closure, max_iter=max_iter, tol=tol)
    for s0 in starts[1:]:
        fit = _als_run(d, s0, closure=closure, max_iter=max_iter, tol=tol)
        if fit.rel_err < best.rel_err:
            best = fit
    dead = dead_components(best.c)
    if dead:
        return best._replace(dead_components=dead)
    return best


def model_fraction(x: np.ndarray, k: float, *, is_ph: bool = True) -> np.ndarray:
    """Return the model's bound fraction on the 0-1 scale.

    This is the quantity the figure draws over the closed profiles, and
    the one the component is compared against. Keeping it a fraction on
    one scale means the curve and the points it overlays share an axis,
    so the distance between them is a real measure of fit quality.

    Parameters
    ----------
    x : np.ndarray
        Titrant values.
    k : float
        pKa for a pH titration, Kd in the units of ``x`` for a ligand one.
    is_ph : bool
        Henderson-Hasselbalch fraction (pH) if true, Langmuir (ligand) else.

    Returns
    -------
    np.ndarray
        The fraction in [0, 1] for every value of ``x``.
    """
    if is_ph:
        return 1.0 / (1.0 + np.power(10.0, k - x))
    return x / (x + k)


def component_pka(
    x: np.ndarray, c: np.ndarray, *, is_ph: bool = True
) -> tuple[float, float, float, float, int, float]:
    """Equilibrium constant of the component that tracks the titrant.

    Returns K on the natural scale of the model fitted to the component:
    pKa for a pH titration (Henderson-Hasselbalch fraction), Kd in the
    units of x for a ligand titration (Langmuir fraction - linear, so no
    log-space propagation happens here). The fitted curve parameters
    (F0, F1) are the tracking component's own plateau pair, i.e. the
    affine mixing that maps the model fraction onto that component's
    scale; they are properties of the component, not of the model, and
    they are free because the component's baseline need not sit at 0 and
    its saturation need not sit at 1.

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

    def model(xx: np.ndarray, k: float, f0: float, f1: float) -> np.ndarray:
        # the model's fraction, mapped onto this component's own scale by
        # its plateau pair. The figure draws the fraction itself, so the
        # affine part is a property of the component, not of the model.
        return f0 + (f1 - f0) * model_fraction(xx, k, is_ph=is_ph)

    try:
        popt, pcov = curve_fit(
            model,
            x,
            y,
            p0=[float(np.median(x)), float(np.min(y)), float(np.max(y))],
            maxfev=20000,
        )
    except (RuntimeError, TypeError):
        # RuntimeError: the fit did not converge. TypeError: fewer wells than
        # the curve's three parameters, so there is nothing to fit.
        return float("nan"), float("nan"), float("nan"), float("nan"), comp, rho[comp]
    se0 = pcov[0, 0]
    se_k = float(np.sqrt(se0)) if np.isfinite(se0) else float("nan")
    k, f0, f1 = float(popt[0]), float(popt[1]), float(popt[2])
    return k, se_k, f0, f1, comp, rho[comp]


class StackedScans(NamedTuple):
    """The scans of one plate, joined into one data matrix.

    The columns keep one block per scan: the scans are different physical
    measurements, so they do not share a wavelength axis.
    """

    tags: np.ndarray
    """Scan name of every column."""
    wavelengths: np.ndarray
    """Wavelength of every column, in its own scan's units."""
    data: np.ndarray
    """D[n_wells, n_columns], each scan scaled by the common maximum."""


def stack_scans(
    scans: Sequence[tuple[str, tuple[ArrayF, ArrayF]]],
) -> StackedScans:
    """Join per-scan (wavelengths, Y[n_lambda, n_wells]) blocks into one matrix.

    The scans of one plate are separate physical measurements - an
    excitation scan and an emission scan are different observables - and
    they do not share one wavelength axis: their ranges overlap but the
    values at an overlapping wavelength are not the same measurement.
    Stacking them therefore keeps one column block per scan, tagged with
    the scan it came from, rather than gluing them into a single ramp
    that runs backwards across the seam.

    Parameters
    ----------
    scans : Sequence[tuple[str, tuple[ArrayF, ArrayF]]]
        One (scan name, (wavelengths, Y[n_lambda, n_wells])) block per scan.

    Returns
    -------
    StackedScans
        The scan tag and the wavelength of every column, and the data
        matrix itself, each scan scaled by the common maximum so blocks
        stay commensurable.
    """
    tags = np.concatenate([
        np.full(np.asarray(lam).size, name) for name, (lam, _y) in scans
    ])
    waves = np.concatenate([
        np.asarray(lam, dtype=np.float64) for _name, (lam, _y) in scans
    ])
    mats = [np.asarray(y, dtype=np.float64) for _name, (_lam, y) in scans]
    top = max(float(np.nanmax(np.abs(m))) for m in mats)
    d = np.vstack([m / top for m in mats]).T  # rows: wells; cols: all scans
    return StackedScans(tags=tags, wavelengths=waves, data=d)


def scan_blocks(stacked: StackedScans) -> dict[str, slice]:
    """Map each scan name to the column block it occupies.

    Parameters
    ----------
    stacked : StackedScans
        The stacked scans.

    Returns
    -------
    dict[str, slice]
        Scan name to the slice of columns it holds, in order of appearance.
    """
    blocks: dict[str, slice] = {}
    for name in dict.fromkeys(stacked.tags.tolist()):
        idx = np.flatnonzero(stacked.tags == name)
        blocks[str(name)] = slice(int(idx[0]), int(idx[-1]) + 1)
    return blocks


def species_by_scan(
    spec: pd.DataFrame, n_species: int
) -> dict[str, tuple[np.ndarray, list[np.ndarray]]]:
    """Split the resolved species spectra into one curve set per scan.

    The species spectra of a scan are that scan's own measurement: an
    excitation scan's species spectrum is an excitation spectrum, and the
    emission scan's is an emission spectrum. Drawing them on one axis
    would put two different observables on the same ramp.

    Parameters
    ----------
    spec : pd.DataFrame
        The resolved spectra, with ``scan``, ``lambda`` and one column
        per species.
    n_species : int
        How many species columns to read.

    Returns
    -------
    dict[str, tuple[np.ndarray, list[np.ndarray]]]
        Scan name to (wavelengths, one array per species).
    """
    out: dict[str, tuple[np.ndarray, list[np.ndarray]]] = {}
    for name, group in spec.groupby("scan", sort=False):
        waves = group["lambda"].to_numpy(dtype=np.float64)
        curves = [
            group[f"species_{j}"].to_numpy(dtype=np.float64) for j in range(n_species)
        ]
        out[str(name)] = (waves, curves)
    return out


def mcr_titration(
    spectra_by_temp: dict[str, Sequence[tuple[str, tuple[ArrayF, ArrayF]]]],
    x_by_temp: dict[str, ArrayF],
    *,
    n_species: int = 2,
    closure: bool = True,
    is_ph: bool = True,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, tuple[ArrayF, np.ndarray]]]:
    """MCR-ALS every temperature's scans; return (K table, spectra, profiles).

    Parameters
    ----------
    spectra_by_temp : dict[str, Sequence[tuple[str, tuple[ArrayF, ArrayF]]]]
        Temperature label to its scans, each ``(name, (lambda, Y[n_lambda, n_wells]))``.
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
        temperature a table of the resolved species spectra, tagged with the
        scan each row came from; and per temperature the titrant with the
        closed profiles, for plotting.
    """
    rows: list[dict[str, object]] = []
    spectra: dict[str, pd.DataFrame] = {}
    profiles: dict[str, tuple[ArrayF, np.ndarray]] = {}
    for t in sorted(spectra_by_temp):
        scans = spectra_by_temp[t]
        x = np.asarray(x_by_temp[t], dtype=np.float64)
        if not scans:
            continue
        stacked = stack_scans(scans)
        d = np.where(np.isfinite(stacked.data), stacked.data, 0.0)
        s_init: np.ndarray | None = None
        if n_species == _TWO_STATES and x.size > 1 and float(np.ptp(x)) > 0:
            # the extreme-titrant wells bound the two states of the equilibrium
            i0, i1 = int(np.argmin(x)), int(np.argmax(x))
            s_init = np.column_stack([d[i0], d[i1]])
        fit = mcr_als(d, n_species, closure=closure, s_init=s_init)
        spec = pd.DataFrame({
            "scan": stacked.tags,
            "lambda": stacked.wavelengths,
        })
        for i in range(fit.s.shape[1]):
            spec[f"species_{i}"] = fit.s[:, i]
        spectra[t] = spec
        profiles[t] = (x, fit.c)
        if fit.degenerate:
            # a component with a constant profile holds no titration
            # information; fitting it would report a confident K describing
            # nothing. Report the sample as unresolvable instead.
            dead = ", ".join(str(i) for i in fit.dead_components)
            rows.append({
                "temp": t,
                "K": float("nan"),
                "se": float("nan"),
                "F0": float("nan"),
                "F1": float("nan"),
                "component": -1,
                "rho": float("nan"),
                "rel_err": fit.rel_err,
                "n_species": n_species,
                "n": x.size,
                "degenerate": True,
                "note": f"degenerate: components {dead} have a dead profile",
            })
            continue
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
            "degenerate": False,
            "note": "",
        })
    return pd.DataFrame(rows), spectra, profiles
