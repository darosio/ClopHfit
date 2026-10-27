"""Band readouts of EnSpire spectra, and the K fitted from them.

A spectral titration can be reduced to one number per well in two ways. The
SVD path (:func:`clophfit.fitting.core.analyze_spectra`) projects each spectrum
onto its first principal component; this module instead averages the intensity
over a named window - the anionic and neutral excitation bands, the emission
band - and lets several such bands share one K while keeping their own
plateaus.

Neither readout is uniformly more precise: on replicate rows of the same plate
they trade places. The band path is here for what it makes visible. Its
plateaus are in the measured units, so ``S1/S0 = 0.01`` reads as "the bound
state is dark"; several bands give a consistency check that a single component
cannot; and a band that has stopped reporting the transition (a flat neutral
band, a readout that reverses sign between runs) is named and dropped rather
than folded into a component that still returns a number.

Two corrections belong to the well, not to the readout, and are applied first:
the buffer well of the same plate column is subtracted, and blank plate rows
are removed by level. There is deliberately no tryptophan normalisation: on
these plates that band is dominated by the medium's scatter and Raman
background (a buffer well keeps ~85 % of it), so dividing by it only adds
well-to-well noise while shifting K by a third of a standard error.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, NamedTuple

import arviz as az  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from clophfit.fitting.bayes import fit_binding_pymc
from clophfit.fitting.bayes_config import RobustConfig, SamplerConfig
from clophfit.fitting.core import fit_binding_glob, weight_multi_ds_titration
from clophfit.fitting.data_structures import DataArray, Dataset
from clophfit.fitting.errors import InsufficientDataError
from clophfit.fitting.models import binding_1site

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from matplotlib.axes import Axes

    from clophfit.prenspire.prenspire import EnspireFile

_COLORS = ("#4c72b0", "#2a9d8f", "#9c6b12", "#7d5ba6")
_FLAG = "#d1495b"

__all__ = [
    "DEFAULT_SCREEN",
    "DIRECT_BANDS",
    "BandFit",
    "Readout",
    "band_rho",
    "band_values",
    "blank_rows",
    "classify",
    "fit_bands",
    "fit_titrations",
    "plot_sample",
    "screen_bands",
    "subtract_buffer",
]

EMISSION = (500.0, 520.0)
ANIONIC = (480.0, 495.0)
NEUTRAL = (395.0, 410.0)
# The anionic window above was chosen from pH reproducibility alone and is
# nearly blind on a chloride titration: the excitation scan's actual
# chloride-responsive region peaks at 426 nm (98% of the titration's
# fractional change is there), which (480, 495) barely reaches. A window
# centred on that peak is not the fix either - it is the best choice at one
# temperature and the worst at the other (row spread 0.78 at 20 C, 1.40 at
# 37 C), same failure mode as whole-spectrum SVD there (0.15 and 1.17). This
# wide window is not the best at either temperature but is the only one that
# is never bad (0.87 and 0.38): see arslanbaeva's rescreen_bands_full.py.
ANIONIC_CL = (395.0, 495.0)
# The two readouts that agree best between sessions. Scored over 39
# sample-temperatures and the constructs measured twice or more: this pair
# lands 0.14 pH apart between sessions against 0.25 for the pair plus the
# neutral excitation band, 0.27 for the anionic band alone, 0.27 for the SVD of
# the excitation label and 0.31 for the SVD of all labels concatenated, and it
# has the smallest worst case of any arm (0.22). Within one plate the arms are
# indistinguishable (0.23-0.27 pH between replicate rows), so it is
# reproducibility across sessions that separates them. One session is left out
# of that scoring as a first attempt whose own fits disagree internally - its
# bands differ by 0.49 pH where the later sessions of the same construct differ
# by 0.00 - and the ranking is the same with it in.
#
# Re-checked once the chloride window above needed fixing, in case the same
# fix (or SVD) also helped pH: it does not. On the same session-spread metric,
# recentring exc_neutral on its own 427 nm peak (rather than dropping it)
# scores 0.27, adding em_exc278 scores 0.30, and whole-spectrum SVD scores
# 0.34 - all worse than this pair's 0.084, not better. See
# arslanbaeva's rescreen_bands_full.py for the full table.
#
# Also checked: swapping the anionic band for that recentred neutral one
# (dropping anionic, keeping em_exc420) rather than adding it - the same
# comparison restricted to the two A11 sessions only, after excluding the one
# G02 group whose row-to-row spread is 2-4x every other group's under every
# arm tested (so it is that group's own data, not the band choice). Current
# pair: 0.18 pH between sessions, worst case 0.22. Neutral+em1 without
# anionic: 0.46, worst case 0.55 - worse by the same margin, not helped by
# removing the noisy group.
DIRECT_BANDS = ("exc_anionic", "em_exc420")
# point screen handed to fit_binding_glob unless asked otherwise
DEFAULT_SCREEN = "studentized:0.05:5"
_PROTEIN_EXCITATION = 350.0
_BLANK_FRACTION = 0.05
_MIN_ROW_POINTS = 5
_MIN_RANK_POINTS = 4
_MIN_DISTINCT = 3
# where a zero-concentration point is drawn on a log axis
ZERO_AT = 0.7


class Readout(NamedTuple):
    """One band of one label."""

    name: str
    label: str
    lo: float
    hi: float


class BandFit(NamedTuple):
    """K with the interval and the residual scale it came with."""

    k: float
    lo: float
    hi: float
    se: float
    rms: float
    n_kept: int
    n_dropped: int
    plateaus: dict[str, tuple[float, float]]
    masks: dict[str, np.ndarray]
    """Per band, which points the screen kept."""


class Sample(NamedTuple):
    """One sample's fitted bands, enough to draw it."""

    data: dict[str, tuple[np.ndarray, np.ndarray]]
    fit: BandFit
    wells: list[str]
    is_ph: bool


class TitrationFits(NamedTuple):
    """What :func:`fit_titrations` found."""

    table: pd.DataFrame
    """One row per (sample, temperature, band, subset)."""
    screened: pd.DataFrame
    """The points the screen removed."""
    samples: dict[str, Sample]
    """Per sample, its bands and shared fit."""


def classify(
    ef: EnspireFile, temp: str | float | None = None, *, is_ph: bool = True
) -> list[Readout]:
    """Split a temperature's labels into the titrating bands their scans offer.

    Each label declares its monochromator and the wavelength it holds fixed, so
    which windows it can offer is read from the file rather than configured: an
    excitation scan carries the anionic and neutral bands, an emission scan
    excited below 350 nm carries the emission band as seen from that excitation
    (``em_exc278``), and any other emission scan carries the emission band
    alone (``em_exc420``).

    Parameters
    ----------
    ef : EnspireFile
        The parsed EnSpire export.
    temp : str | float | None
        Keep only labels measured at this temperature; None keeps every label.
    is_ph : bool
        Which anionic-band window to use: ``ANIONIC`` (480-495 nm, where a pH
        titration's own reproducibility is best) for a pH titration, or
        ``ANIONIC_CL`` (395-495 nm) for a chloride one - the two titrations
        respond over different parts of the excitation scan and one window
        does not serve both; see ``ANIONIC_CL``'s definition for the numbers.

    Returns
    -------
    list[Readout]
        The titrating bands the labels of this temperature offer, in scan
        order.
    """
    wanted = None if temp is None or str(temp) in {"nan", "None", ""} else str(temp)
    anionic = ANIONIC if is_ph else ANIONIC_CL
    readouts: list[Readout] = []
    for label, meas in ef.measurements.items():
        md = meas["metadata"]
        if (
            wanted is not None
            and str(md.get("temp")).split(".")[0] != wanted.split(".")[0]
        ):
            continue
        if md["Monochromator"] == "Excitation":
            readouts += [
                Readout("exc_anionic", label, *anionic),
                Readout("exc_neutral", label, *NEUTRAL),
            ]
        elif float(md["Wavelength"]) < _PROTEIN_EXCITATION:
            readouts.append(Readout("em_exc278", label, *EMISSION))
        else:
            readouts.append(Readout("em_exc420", label, *EMISSION))
    return readouts


def band_values(ef: EnspireFile, readout: Readout, wells: Sequence[str]) -> pd.Series:
    """Mean intensity of one label over one window, per well.

    Parameters
    ----------
    ef : EnspireFile
        The parsed EnSpire export.
    readout : Readout
        Label and window to average.
    wells : Sequence[str]
        Wells to read, in order.

    Returns
    -------
    pd.Series
        One value per well, indexed by well.
    """
    meas = ef.measurements[readout.label]
    lam = np.asarray(meas["lambda"], dtype=float)
    sel = (lam >= readout.lo) & (lam <= readout.hi)
    return pd.Series({
        w: float(np.mean(np.asarray(meas[w], dtype=float)[sel])) for w in wells
    })


def subtract_buffer(values: pd.Series, buffer: pd.Series) -> pd.Series:
    """Subtract the buffer well of the same plate column from each sample well.

    Parameters
    ----------
    values : pd.Series
        Sample values indexed by well.
    buffer : pd.Series
        Buffer values indexed by well; the plate column is what pairs them,
        because a column is one titration step.

    Returns
    -------
    pd.Series
        *values* with the matching buffer removed.
    """
    by_column = {str(w)[1:]: v for w, v in buffer.items()}
    return pd.Series({w: v - by_column.get(str(w)[1:], 0.0) for w, v in values.items()})


def blank_rows(values: pd.Series, *, frac: float = _BLANK_FRACTION) -> list[str]:
    """Plate rows holding no protein, whatever the note calls them.

    Some notes name every well after the mutant while one row of the plate is
    buffer. Such a row sits at the instrument's floor, so it is found by level:
    a row whose median is below *frac* of the brightest row's median is blank.

    Parameters
    ----------
    values : pd.Series
        Raw band intensity per well, indexed by well.
    frac : float
        Fraction of the brightest row's median below which a row is blank.

    Returns
    -------
    list[str]
        The blank plate rows, as single letters.
    """
    by_row = values.groupby([w[0] for w in values.index]).median()
    if by_row.empty or by_row.max() <= 0:
        return []
    return sorted(by_row[by_row < frac * by_row.max()].index)


def band_rho(x: np.ndarray, y: np.ndarray, wells: Sequence[str]) -> float:
    """Rank correlation of a band with the titrant, taken within replicate rows.

    Replicate rows sit at slightly different levels, so ranking the pooled
    points can hide a perfectly monotone titration. Each row is ranked alone
    and the rows are pooled by median.

    Parameters
    ----------
    x : np.ndarray
        Titrant value per point.
    y : np.ndarray
        Band value per point.
    wells : Sequence[str]
        Well of each point; its plate row groups the replicates.

    Returns
    -------
    float
        Median within-row Spearman rho, or the pooled one if no row qualifies.
    """
    per_row = []
    for row in {w[0] for w in wells}:
        sel = np.array([w[0] == row for w in wells])
        if sel.sum() < _MIN_RANK_POINTS or len(np.unique(y[sel])) < _MIN_DISTINCT:
            continue
        rho = float(spearmanr(x[sel], y[sel]).statistic)
        if np.isfinite(rho):
            per_row.append(rho)
    if not per_row:
        rho = float(spearmanr(x, y).statistic)
        return rho if np.isfinite(rho) else 0.0
    return float(np.median(per_row))


def screen_bands(
    data: dict[str, tuple[np.ndarray, np.ndarray]],
    wells: Sequence[str],
    *,
    wanted: Sequence[str] = DIRECT_BANDS,
    min_rho: float = 0.8,
) -> tuple[list[str], dict[str, str]]:
    """Keep the requested bands that titrate monotonically with the titrant.

    Parameters
    ----------
    data : dict[str, tuple[np.ndarray, np.ndarray]]
        Band name to its (x, y).
    wells : Sequence[str]
        Well of each point, in the same order.
    wanted : Sequence[str]
        Bands allowed into the shared fit; the rest are reported only.
    min_rho : float
        Smallest acceptable |rho| between a band and the titrant.

    Returns
    -------
    tuple[list[str], dict[str, str]]
        The bands to fit, and why each of the others is out.
    """
    rejected: dict[str, str] = {}
    kept: list[str] = []
    for name, (x, y) in data.items():
        rho = band_rho(x, y, wells)
        if not np.isfinite(rho) or abs(rho) < min_rho:
            rejected[name] = f"does not titrate (rho {rho:+.2f})"
        elif name not in wanted:
            rejected[name] = f"indirect readout (rho {rho:+.2f}), reported only"
        else:
            kept.append(name)
    return kept, rejected


def fit_bands(
    data: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    is_ph: bool,
    screen: str | None = DEFAULT_SCREEN,
) -> BandFit | None:
    """Fit the given bands with one shared K and a plateau pair each.

    Parameters
    ----------
    data : dict[str, tuple[np.ndarray, np.ndarray]]
        Band name to its (x, y).
    is_ph : bool
        pH titration (else ligand concentration).
    screen : str | None
        Outlier specification handed to :func:`fit_binding_glob`, or None.

    Returns
    -------
    BandFit | None
        The fit, or None when it does not converge or has too few points to fit.
    """
    if not data:
        return None
    ds = Dataset({name: DataArray(x, y) for name, (x, y) in data.items()}, is_ph=is_ph)
    try:
        result = fit_binding_glob(ds, remove_outliers=screen)
    except InsufficientDataError:
        # a band with fewer points than parameters cannot carry K; report
        # it as unfitted (the caller drops the row) instead of aborting
        # the whole folder's fits.
        return None
    if not result.is_valid() or result.result is None:
        return None
    params = result.result.params
    k = float(params["K"].value)
    se = float(params["K"].stderr or np.nan)
    if is_ph:
        lo, hi = k - 1.96 * se, k + 1.96 * se
    else:  # a Kd interval is multiplicative
        rel = se / k if k else np.nan
        lo, hi = k * np.exp(-1.96 * rel), k * np.exp(1.96 * rel)
    kept = int(sum(int(da.mask.sum()) for da in (result.dataset or ds).values()))
    total = sum(len(x) for x, _ in data.values())
    plateaus = {
        name: (float(params[f"S0_{name}"].value), float(params[f"S1_{name}"].value))
        for name in data
    }
    rms = float(np.sqrt(np.mean(np.asarray(result.result.residual) ** 2)))
    fitted = result.dataset or ds
    masks = {name: np.asarray(fitted[name].mask, dtype=bool) for name in data}
    return BandFit(k, lo, hi, se, rms, kept, total - kept, plateaus, masks)


def isosbestic_readout_scale(  # ruff: ignore[too-many-arguments]
    ef: EnspireFile,
    wells: Sequence[str],
    x: np.ndarray,
    readout: Readout,
    buffer_wells: Sequence[str] = (),
    *,
    isosbestic: float | str,
) -> tuple[float | None, np.ndarray | None]:
    """Per-well brightness reference of one readout, taken at the crossing.

    A band readout is one number per well, so normalising it to the isosbestic
    point means reading the whole excitation scan the window came from and
    dividing the well by its signal where the two species cross: the ratio
    removes how much protein the well holds and how brightly it is measured,
    and keeps the fraction. This is the band-level counterpart of
    :func:`~clophfit.prenspire.spectral.sample_spectra`'s ``isosbestic``.

    Parameters
    ----------
    ef : EnspireFile
        The parsed EnSpire export.
    wells : Sequence[str]
        Wells of this sample, in the order of ``x``.
    x : np.ndarray
        Titrant value per well.
    readout : Readout
        The band whose scan supplies the spectrum; its label names the scan.
    buffer_wells : Sequence[str]
        Buffer wells of the same plate column, subtracted before the crossing
        is located, exactly as the band values are.
    isosbestic : float | str
        A wavelength [nm] or ``"auto"``.

    Returns
    -------
    tuple[float | None, np.ndarray | None]
        The crossing used and the scale per well, or ``(None, None)`` when
        ``"auto"`` finds none - which is what a chloride titration returns.
    """
    from clophfit.prenspire.spectral import (  # ruff: ignore[import-outside-top-level]
        EXCITATION_SCAN_NAMES,
        isosbestic_scale,
        plate_isosbestic,
        scan_spectrum,
    )

    lam, y = scan_spectrum(ef, readout.label, wells, buffer_wells)
    spectra = {EXCITATION_SCAN_NAMES[0]: (lam, y)}
    lam_iso = (
        plate_isosbestic(spectra, x)
        if isinstance(isosbestic, str)
        else float(isosbestic)
    )
    if lam_iso is None:
        return None, None
    scale, _usable = isosbestic_scale(spectra, lam_iso)
    return lam_iso, scale


def _sample_bands(  # ruff: ignore[too-many-arguments]
    ef: EnspireFile,
    wells: list[str],
    x: np.ndarray,
    temp: str | float,
    *,
    is_ph: bool,
    buffer_wells: Sequence[str],
    isosbestic: float | str | None = None,
) -> tuple[
    dict[str, tuple[np.ndarray, np.ndarray]], list[str], np.ndarray, dict[str, float]
]:
    """Band values of one sample, buffer-subtracted and scaled to its own max.

    Blank plate rows are detected and removed from the sample here, and serve
    as the buffer when the note declared none.

    Parameters
    ----------
    ef : EnspireFile
        The parsed EnSpire export.
    wells : list[str]
        Wells of this sample.
    x : np.ndarray
        Titrant value per well.
    temp : str | float
        Temperature selecting the labels.
    is_ph : bool
        pH titration (else chloride) - selects the anionic band's window;
        see :func:`classify`.
    buffer_wells : Sequence[str]
        Buffer wells declared by the note, if any.
    isosbestic : float | str | None
        Normalise each well by its signal at the isosbestic point, before the
        band is scaled to its own maximum (see
        :func:`isosbestic_readout_scale`); ``None`` keeps the raw readout.

    Returns
    -------
    tuple[dict[str, tuple[np.ndarray, np.ndarray]], list[str], np.ndarray, dict[str, float]]
        Band name to (x, y), the wells actually used, their x values, and the
        per-band maximum each band was divided by - the last one so that a caller
        can put the temperatures of one sample back on a common scale.
    """
    readouts = classify(ef, temp, is_ph=is_ph)
    if not readouts:
        return {}, wells, x, {}
    probe = band_values(ef, readouts[0], wells)
    blanks = blank_rows(probe)
    if blanks:
        found = [w for w in wells if w[0] in blanks]
        keep = [i for i, w in enumerate(wells) if w[0] not in blanks]
        wells, x = [wells[i] for i in keep], x[keep]
        buffer_wells = list(buffer_wells) or found
    scale = None
    if isosbestic is not None:
        excitation = next((r for r in readouts if r.name == "exc_anionic"), None)
        if excitation is not None:
            _lam_iso, scale = isosbestic_readout_scale(
                ef,
                wells,
                x,
                excitation,
                buffer_wells,
                isosbestic=isosbestic,
            )
    data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    tops: dict[str, float] = {}
    for readout in readouts:
        y = band_values(ef, readout, wells)
        if buffer_wells:
            y = subtract_buffer(y, band_values(ef, readout, buffer_wells))
        arr = np.array(y, dtype=float)
        if scale is not None:
            arr /= scale
        top = float(np.abs(arr).max())
        tops[readout.name] = top
        data[readout.name] = (x, arr / top if top else arr)
    return data, wells, x, tops


def _row_fits(
    kept_data: dict[str, tuple[np.ndarray, np.ndarray]],
    wells: Sequence[str],
    *,
    is_ph: bool,
    screen: str | None,
) -> dict[str, BandFit]:
    """One fit per replicate plate row, so their spread can be seen."""
    out: dict[str, BandFit] = {}
    for row in sorted({w[0] for w in wells}):
        sel = np.array([w[0] == row for w in wells])
        if sel.sum() < _MIN_ROW_POINTS:
            continue
        per_row = {b: (v[0][sel], v[1][sel]) for b, v in kept_data.items()}
        fit = fit_bands(per_row, is_ph=is_ph, screen=screen)
        if fit:
            out[row] = fit
    return out


def _resolve_isosbestic(
    ef: EnspireFile,
    note: pd.DataFrame,
    isosbestic: float | str | Mapping[tuple[str, float], float | None] | None,
    *,
    buffer: bool,
) -> dict[tuple[str, float], float | None] | None:
    """Turn a per-sample request into one crossing per (sample, temperature).

    Parameters
    ----------
    ef : EnspireFile
        The parsed EnSpire export.
    note : pd.DataFrame
        Normalised note.
    isosbestic : float | str | Mapping[tuple[str, float], float | None] | None
        The caller's request; the ``auto-shared``/``auto-scale``/``auto-wavelength``
        forms need a pass over the plate.
    buffer : bool
        Subtract the buffer wells before locating the crossings.

    Returns
    -------
    dict[tuple[str, float], float | None] | None
        The resolved crossings, or ``None`` when the request is per group (the
        data builders then locate their own). A non-``None`` result also means the
        caller should put a sample's temperatures on one amplitude scale.
    """
    if isinstance(isosbestic, str) and isosbestic in {
        "auto-shared",
        "auto-scale",
        "auto-wavelength",
    }:
        from clophfit.prenspire.spectral import (  # ruff: ignore[import-outside-top-level]
            isosbestic_by_group,
        )

        # auto-scale keeps each temperature's own crossing (the unbiased choice)
        # and shares only the amplitude; auto-wavelength shares the crossing too
        shared = isosbestic in {"auto-shared", "auto-wavelength"}
        return isosbestic_by_group(ef, note, buffer=buffer, shared=shared)
    if isinstance(isosbestic, dict):
        return isosbestic
    return None


def _lam_for(
    isosbestic: float | str | Mapping[tuple[str, float], float | None] | None,
    lam_map: dict[tuple[str, float], float | None] | None,
    name: str,
    temp: object,
) -> float | str | None:
    """Pick the crossing for one (sample, temperature) group."""
    if lam_map is None:
        return isosbestic if not isinstance(isosbestic, Mapping) else None
    return lam_map.get((name, float(str(temp))))


def _share_band_scale(
    gathered: list[
        tuple[
            str,
            object,
            bool,
            list[str],
            np.ndarray,
            dict[str, tuple[np.ndarray, np.ndarray]],
            dict[str, float],
        ]
    ],
) -> None:
    """Rescale a sample's temperatures onto one amplitude scale, in place.

    Each group was divided by its own peak, which puts both temperatures at unit
    maximum and throws away how much brighter one of them was. When the crossing
    is shared per sample the amplitudes have to be on one scale too, otherwise the
    two temperatures of one sample cannot be drawn, or fitted jointly, on the same
    axis: the group's own peak is divided out against the sample's, band by band.
    """
    by_sample: dict[str, dict[str, float]] = {}
    for name, _temp, _is_ph, _wells, _x, data, tops in gathered:
        shared = by_sample.setdefault(name, {})
        for band, (bx, by) in data.items():
            _ = (bx, by)
            shared[band] = max(shared.get(band, 0.0), tops.get(band, 0.0))
    for name, _temp, _is_ph, _wells, _x, data, tops in gathered:
        shared = by_sample[name]
        for band, (bx, by) in list(data.items()):
            top, common = tops.get(band, 0.0), shared.get(band, 0.0)
            if top > 0 and common > 0:
                data[band] = (bx, by * (top / common))


def fit_titrations(  # ruff: ignore[too-many-arguments]
    ef: EnspireFile,
    note: pd.DataFrame,
    *,
    buffer: bool = True,
    wanted: Sequence[str] = DIRECT_BANDS,
    min_rho: float = 0.8,
    screen: str | None = DEFAULT_SCREEN,
    isosbestic: float | str | Mapping[tuple[str, float], float | None] | None = None,
) -> TitrationFits:
    """Fit every sample of a note from its bands.

    Parameters
    ----------
    ef : EnspireFile
        The parsed EnSpire export.
    note : pd.DataFrame
        Normalised note, with ``Well``, ``pH``, ``Cl``, ``Name`` and ``Temp``.
    buffer : bool
        Subtract the buffer well of the same plate column.
    wanted : Sequence[str]
        Bands allowed into the shared K; the others are reported only.
    min_rho : float
        Monotonicity gate for a band.
    screen : str | None
        Point screen handed to :func:`fit_binding_glob`.
    isosbestic : float | str | Mapping[tuple[str, float], float | None] | None
        Normalise every well by its signal at the isosbestic point first (see
        :func:`isosbestic_readout_scale`): a wavelength [nm], ``"auto"`` per
        (sample, temperature) group, ``"auto-shared"`` for one crossing per
        sample shared by its temperatures, a mapping of ``(sample, temperature)``
        to that wavelength, or ``None``. With a per-sample crossing the band
        maxima are also shared across the sample's temperatures, so the two
        temperatures keep their measured amplitude ratio instead of each being
        rescaled to its own peak.

    Returns
    -------
    TitrationFits
        The table of fits, the points the screen removed, and per sample the
        band data with its shared fit, for plotting.
    """
    rows: list[dict[str, object]] = []
    screened: list[dict[str, object]] = []
    drawable: dict[str, Sample] = {}
    samples = note[note["Name"] != "buffer"]
    buffers = note[note["Name"] == "buffer"]
    lam_map = _resolve_isosbestic(ef, note, isosbestic, buffer=buffer)
    # gather first, fit second: a per-sample crossing has to be shared by the
    # temperatures before any of them is scaled
    gathered: list[
        tuple[
            str,
            object,
            bool,
            list[str],
            np.ndarray,
            dict[str, tuple[np.ndarray, np.ndarray]],
            dict[str, float],
        ]
    ] = []
    for (name, temp), group in samples.groupby(
        ["Name", "Temp"], sort=True, dropna=False
    ):
        one = group.drop_duplicates("Well")
        is_ph = bool(one["pH"].nunique() > one["Cl"].nunique())
        wells = one["Well"].tolist()
        x = one["pH" if is_ph else "Cl"].astype(float).to_numpy()
        buf = buffers[buffers["Temp"] == temp]["Well"].tolist() if buffer else []
        lam = _lam_for(isosbestic, lam_map, str(name), temp)
        data, wells, x, tops = _sample_bands(
            ef,
            wells,
            x,
            str(temp),
            is_ph=is_ph,
            buffer_wells=buf,
            isosbestic=lam,
        )
        if not data:
            continue
        gathered.append((str(name), temp, is_ph, wells, x, data, tops))
    if lam_map is not None:
        _share_band_scale(gathered)
    for name, temp, is_ph, wells, _x, data, _tops in gathered:
        keep, rejected = screen_bands(data, wells, wanted=wanted, min_rho=min_rho)
        base: dict[str, object] = {"sample": name, "temp": temp}
        for band, why in rejected.items():
            alone = fit_bands({band: data[band]}, is_ph=is_ph, screen=screen)
            rows.append(_row(base, band, f"rejected: {why}", alone))
        kept_data = {b: data[b] for b in keep}
        shared = fit_bands(kept_data, is_ph=is_ph, screen=screen)
        if shared is None:
            continue
        drawable[f"{name} {temp}"] = Sample(kept_data, shared, wells, is_ph=is_ph)
        rows.extend((
            _row(base, "all (shared K)", "screened", shared),
            _row(
                base,
                "all (shared K)",
                "every point",
                fit_bands(kept_data, is_ph=is_ph, screen=None),
            ),
        ))
        for band, (bx, by) in kept_data.items():
            rows.append(
                _row(
                    base,
                    band,
                    "screened",
                    fit_bands({band: (bx, by)}, is_ph=is_ph, screen=screen),
                )
            )
        for row, fit in _row_fits(kept_data, wells, is_ph=is_ph, screen=screen).items():
            rows.append(_row(base, "all (shared K)", f"row {row}", fit))
        for band, mask in shared.masks.items():
            bx, by = kept_data[band]
            for well, xi, yi, keep_it in zip(wells, bx, by, mask, strict=True):
                if not keep_it:
                    screened.append({
                        **base,
                        "band": band,
                        "well": well,
                        "x": float(xi),
                        "y": float(yi),
                    })
    table = pd.DataFrame([r for r in rows if r])
    return TitrationFits(table, pd.DataFrame(screened), drawable)


def _row(
    base: dict[str, object], band: str, subset: str, fit: BandFit | None
) -> dict[str, object]:
    """One table row, or an empty dict when the fit failed."""
    if fit is None:
        return {}
    return {
        **base,
        "band": band,
        "subset": subset,
        "K": fit.k,
        "lo": fit.lo,
        "hi": fit.hi,
        "rms": fit.rms,
        "n": fit.n_kept,
        "dropped": fit.n_dropped,
    }


class BandBayes(NamedTuple):
    """Seeded posterior of the K shared by a set of band readouts."""

    k: float
    """Posterior median of K."""
    hdi03: float
    hdi97: float
    """94 % highest-density interval of K."""
    rhat: float
    ess: float
    trace: object
    """ArviZ trace, for posterior plots and plateau medians."""


def _pooled_hdi(sampled: np.ndarray, *, prob: float = 0.94) -> tuple[float, float]:
    """Highest-density interval of the pooled draws, across every chain.

    ``az.hdi`` reduces over the last axis only, so a ``(chain, draw)`` array
    yields one interval per chain; taking the first one describes a fraction
    of the posterior while the r_hat reported beside it describes all of it.
    Pooling the draws first gives the one interval the whole posterior
    supports.

    Parameters
    ----------
    sampled : np.ndarray
        Posterior draws, shaped ``(chain, draw)`` or ``(draw,)``.
    prob : float
        Mass the interval covers, in (0, 1).

    Returns
    -------
    tuple[float, float]
        The lower and upper bound of the pooled interval.
    """
    flat = np.asarray(sampled, dtype=float).ravel()
    lo, hi = np.asarray(az.hdi(flat, prob=prob), dtype=float).ravel()[:2]
    return float(lo), float(hi)


# Fewest replicate wells in an index column for its spread to count.
_MIN_WELLS_PER_COLUMN = 2
# Fewest pooled degrees of freedom before a sigma can be estimated.
_MIN_DOF_TO_ESTIMATE = 2


def estimate_x_error_sigma(wells: Iterable[str], ph: Iterable[float]) -> float | None:
    """Pooled within-column spread of the measured pH axis of one plate group.

    EnSpire notes lay each temperature out as replicate rows of one nominal
    target series, so the same well index (``A04``/``B04``/``C04``) holds the
    same nominal target and the note's pH values spread around it by the
    per-well axis error only: dispensing, electrode, reading. The pooled
    within-column SD is a calibration-free estimate of that error, usable as
    the prior scale in :func:`fit_bands_bayes` (``x_sigma``).

    Parameters
    ----------
    wells : Iterable[str]
        Well labels of one (sample, temperature) group.
    ph : Iterable[float]
        The note's pH per well, in the same order. Non-finite values are
        skipped.

    Returns
    -------
    float | None
        The pooled SD in pH, over the index columns holding at least two
        wells each. ``None`` when fewer than two index columns hold two or
        more replicates (a single-row plate cannot separate axis error from
        target spread - fall back to a fixed sigma or the fixed axis).

    Notes
    -----
    Relies on the EnSpire replicate-row convention: rows of one temperature
    are parallel replicates of the same nominal series (rows A-C at 20 C,
    D-F at 37 C). The index is taken as the trailing digits of the well label.
    A floor of 0.02 keeps the prior strictly positive.
    """
    idx: dict[str, list[float]] = {}
    for w, p in zip(wells, ph, strict=False):
        m = re.search(r"(\d+)$", str(w))
        key = m.group(1) if m else str(w)
        pv = float(p)
        if np.isfinite(pv):
            idx.setdefault(key, []).append(pv)
    ss = 0.0
    dof = 0
    for vals in idx.values():
        n = len(vals)
        if n >= _MIN_WELLS_PER_COLUMN:
            mean = sum(vals) / n
            ss += sum((v - mean) ** 2 for v in vals)
            dof += n - 1
    if dof < _MIN_DOF_TO_ESTIMATE:
        return None
    return max(float(np.sqrt(ss / dof)), 0.02)


def fit_bands_bayes(  # ruff: ignore[too-many-arguments]
    ds: Dataset,
    *,
    seed: int = 0,
    draws: int = 2000,
    chains: int = 2,
    cores: int | None = None,
    target_accept: float | None = None,
    robust: bool = True,
    x_sigma: float | None = None,
) -> BandBayes:
    """Posterior of the K shared across the bands of ``ds`` (PyMC NUTS).

    The Bayesian counterpart of :func:`fit_bands`: identical corrected band
    readouts and shared-K model, but the interval is a 94 % HDI read off the
    posterior instead of an LM covariance - the estimate the legacy svd+emcee
    path used to provide, now on a readout that is stable across plates and
    with a seed that makes the draws reproducible.

    Parameters
    ----------
    ds : Dataset
        Corrected band readouts (same construction as :func:`fit_bands`).
    seed : int
        NUTS seed; the HDI is reproducible for fixed seed and draws.
    draws : int
        Posterior draws per chain (tuning is half of that).
    chains : int
        MCMC chains. PyMC derives each chain's seed from the master seed
        and the chain index - not from the core count - so a fixed seed
        stays reproducible whether the chains run sequentially or in
        parallel.
    cores : int | None
        CPU cores for chain parallelism. ``None`` lets PyMC run the chains
        in parallel (capped at the core count); pass 1 for sequential.
    target_accept : float | None
        NUTS target acceptance, in (0, 1). ``None`` keeps the sampler
        default; higher values (e.g. 0.9) trade step size for better mixing.
    robust : bool
        Student-t (nu = 3) likelihood, the best-calibrated arm on the
        campaign plates; False gives a plain Normal.
    x_sigma : float | None
        Prior SD (pH) on each well's true axis value:
        ``x_true[i] ~ N(x[i], x_sigma)`` per point, independently. The
        EnSpire note pH is an instrument reading, not an exact level, and
        the replicate-row spread at one nominal target is exactly this
        error - pool it with :func:`estimate_x_error_sigma` or pass a fixed
        0.1-0.2. ``None`` fixes the axis at the note values (previous
        behavior).

    Returns
    -------
    BandBayes
        Median, 94 % HDI, convergence diagnostics and the trace; all NaN if
        the sampler produced nothing.
    """
    weight_multi_ds_titration(ds)
    sampler = SamplerConfig(
        n_samples=draws,
        n_tune=draws // 2,
        chains=chains,
        cores=cores,
        target_accept=target_accept,
        random_seed=seed,
        nuts_sampler="pymc",
    )
    try:
        fr = fit_binding_pymc(
            ds,
            n_xerr=0.0,  # nominal titrant axis: enspire notes carry no additions scheme
            robust=RobustConfig(enabled=robust, likelihood="student_t", nu=3.0),
            sampler=sampler,
            x_sigma=x_sigma,
        )
    except InsufficientDataError:
        # a band with fewer points than parameters cannot carry K; report it
        # as unfitted (the caller drops the row) instead of aborting the
        # whole note's fits, as fit_bands already does. The pre-fit that
        # raises this runs inside the call, before any sampling, so the
        # guard costs nothing: there is no second fit to discard.
        nan = float("nan")
        return BandBayes(nan, nan, nan, nan, nan, None)
    if fr.trace is None or fr.result is None:
        nan = float("nan")
        return BandBayes(nan, nan, nan, nan, nan, None)

    # per-chain draws, so the diagnostics do not depend on how the
    # ArviZ tree stores sample dimensions
    sampled = az.extract(fr.trace, var_names=["K"], combined=False).to_numpy()
    if sampled.ndim == 1:
        sampled = np.atleast_2d(sampled)
    # a genuine highest-density interval over every chain's draws, not the
    # equal-tailed one some arviz builds report by default
    lo, hi = _pooled_hdi(sampled, prob=0.94)
    k = float(np.median(sampled.ravel()))
    return BandBayes(
        k,
        lo,
        hi,
        float(az.rhat(sampled)),
        float(az.ess(sampled, relative=False)),
        fr.trace,
    )


def plot_sample(ax: Axes, sample: Sample, title: str) -> None:
    """Draw one sample's bands, their shared curve, and the screened points.

    Parameters
    ----------
    ax : Axes
        Where to draw.
    sample : Sample
        Band data with the fit that shares K across them.
    title : str
        Panel title; K and its interval are appended.
    """
    data, fit, is_ph = sample.data, sample.fit, sample.is_ph
    xs = np.concatenate([x for x, _ in data.values()]) if data else np.array([0.0])
    if is_ph:
        grid = np.linspace(xs.min() - 0.3, xs.max() + 0.3, 200)
    else:
        grid = np.logspace(np.log10(ZERO_AT), np.log10(max(xs.max(), 1.0) * 1.5), 200)
    for (name, (x, y)), color in zip(data.items(), _COLORS, strict=False):
        px = x if is_ph else np.where(x == 0, ZERO_AT, x)
        ax.scatter(px, y, s=18, color=color, lw=0, zorder=3, label=name)
        s0, s1 = fit.plateaus[name]
        gx = grid if is_ph else np.where(grid <= ZERO_AT, 0.0, grid)
        ax.plot(
            grid,
            binding_1site(gx, fit.k, s0, s1, is_ph=is_ph),
            color=color,
            lw=1.1,
            alpha=0.8,
            zorder=2,
        )
        mask = fit.masks[name]
        if not mask.all():
            ax.scatter(
                px[~mask],
                y[~mask],
                s=52,
                facecolor="none",
                edgecolor=_FLAG,
                lw=1.3,
                zorder=4,
            )
    if not is_ph:
        ax.set_xscale("log")
    ax.set_xlabel("pH" if is_ph else f"[Cl$^-$] (mM), 0 drawn at {ZERO_AT}")
    ax.set_ylabel("relative band value")
    label = "pK$_\\mathrm{a}$" if is_ph else "$K_\\mathrm{d}$"
    ax.set_title(
        f"{title} — {label} {fit.k:.2f} [{fit.lo:.2f}, {fit.hi:.2f}]",
        loc="left",
        fontsize=10.5,
    )
    ax.legend(frameon=False, fontsize=8.5, loc="best")
