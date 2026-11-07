"""Whole-spectrum global K for EnSpire titrations (see :mod:`clophfit.fitting.spectral`).

The wells get the same corrections as the band readouts of :mod:`clophfit.prenspire.bands` - blank
plate rows removed, the buffer spectrum of the same plate column subtracted - so that the two methods
differ only in the readout: named windows sharing K there, every wavelength of the chosen scans here.

Scored with arslanbaeva's ``scripts/score_readout_arms.py`` metrics on the 25 pH plates it ranks
(same exclusions; arslanbaeva's ``scripts/score_spectral_arms.py``): the whole excitation scan with a per-well
amplitude has the smallest between-session spread of every readout (median 0.10 pH, worst 0.16,
against 0.14 and 0.22 for the default band pair) at the same replicate-row spread (0.23). Adding the
420 nm emission scan whole does not help (0.14, worst 0.37), and without the well amplitude it is
worse (0.21, worst 0.80). Only five constructs are repeated across sessions, so the ranking of the
excitation scan against the band pair is suggestive, not settled.

The tryptophan band (330-342 nm of the 278 nm-excited scan) does not depend on pH (Spearman rho
0.02 and -0.06 on G10) and neither causes nor removes the acid-side amplitude drop; it scatters by
15-30 % between wells, more than the fitted amplitudes do (about 5 %), and on these plates it is
dominated by the medium's scatter and Raman background - a buffer well keeps ~85 % of it. Dividing
wells by it therefore only adds noise without changing K (7.972 against 7.970 unnormalised on G10),
so it has been retired: there is no normalisation step in this path.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from clophfit.fitting.models import growth_fraction
from clophfit.fitting.spectral import (
    SpectralFit,
    _fractions,
    _Model,
    fit_spectra_global,
    observed_fraction,
    observed_fraction_from_species,
)
from clophfit.prenspire.bands import PH_ACID_END, band_values, blank_rows, classify

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from clophfit.clophfit_types import ArrayF
    from clophfit.prenspire.prenspire import EnspireFile

_MIN_ROW_WELLS = 5
#: Above this titrant value the axis is a concentration, not a pH; used only to
#: label a panel, never to fit.
_MAX_PH = 20.0
#: Shortest wavelength the crossing search in a panel accepts [nm]; the protein
#: band below it does not titrate and would supply a crossing of its own.
_CHROMOPHORE_MIN = 400.0
#: Refits the screen is allowed after the first drop, to confirm its own verdict.
_QC_MAX_PASSES = 2
#: States of a two-state fit; a bigger count means an acid state was sampled too,
#: which the fraction panels do not draw (they use the two-state curve).
_TWO_STATES = 2
#: Fraction deviation from the common curve that makes a well worth labelling in
#: the per-well panel. One tenth of the titration is visible by eye in a band.
_OFF_CURVE = 0.10
#: Open markers, one per scan, for the fraction a single scan gives on its own.
#: Filled markers are the fraction all scans share; these are what disagree.
_SCAN_MARKERS = ("s", "^", "D", "v")
#: Posterior draws used for the per-well observed fraction. The unmixing is a
#: grid search per well per draw, and the species spectra are smooth in the
#: posterior, so a thinned hundred and fifty draws pins the interval without
#: paying the whole posterior for it.
_OBS_DRAWS = 150
# The excitation scan alone: on the readout-arm sweep (module docstring) adding the 420 nm emission
# scan whole widened the between-session spread (median 0.10 -> 0.14 pH, worst 0.16 -> 0.37).
EXCITATION_SCAN = ("exc_anionic",)
#: The scan name (see :func:`scan_name`) of the excitation scan, which is where
#: the chromophore's two-state crossover can be located.
EXCITATION_SCAN_NAMES = ("excitation",)


def _scan_kind(md: dict[str, Any]) -> str:
    """Return the kind of measurement a scan's metadata describes.

    Parameters
    ----------
    md : dict[str, Any]
        The metadata of one scan.

    Returns
    -------
    str
        The kind: ``"excitation"`` for a scan that holds the emission
        wavelength fixed, ``"emission <wavelength>"`` for one that holds
        the excitation wavelength fixed.
    """
    if md["Monochromator"] == "Excitation":
        return "excitation"
    return f"emission {float(md['Wavelength']):.0f}"


def scan_name(ef: EnspireFile, label: str) -> str:
    """Name a scan by the measurement it makes, not by its plate label.

    The plate's labels are row names: the excitation scan is "C" at
    20 C and "E" at 37 C, so the label cannot say which measurement a
    spectrum is, and the two temperatures of one sample would look like
    different scans. The monochromator and the wavelength it holds
    fixed do say: an excitation scan and a 420 nm-excited emission scan
    are different observables and must not be drawn on one wavelength
    axis.

    The name is unique within a temperature: a plate may hold two
    scans of the same kind (an excitation scan with the filter in the
    top position and another with it in the bottom one), and those are
    different measurements. When a name would repeat at the same
    temperature, the plate label is appended to tell them apart.

    Parameters
    ----------
    ef : EnspireFile
        The parsed EnSpire export.
    label : str
        The plate label of the scan.

    Returns
    -------
    str
        The scan's name, e.g. ``"excitation"`` or ``"emission 420"``.
    """
    md = ef.measurements[label]["metadata"]
    name = _scan_kind(md)
    same = [
        other
        for other, omeas in ef.measurements.items()
        if str(omeas["metadata"].get("temp")) == str(md.get("temp"))
        and _scan_kind(omeas["metadata"]) == name
    ]
    if len(same) > 1:
        name = f"{name} {label}"
    return name


def _spectra(
    ef: EnspireFile, label: str, wells: Sequence[str]
) -> tuple[ArrayF, ArrayF]:
    meas = ef.measurements[label]
    lam = np.asarray(meas["lambda"], dtype=np.float64)
    y = np.column_stack([np.asarray(meas[w], dtype=np.float64) for w in wells])
    return lam, y


def scan_spectrum(
    ef: EnspireFile, label: str, wells: Sequence[str], buffer_wells: Sequence[str] = ()
) -> tuple[ArrayF, ArrayF]:
    """One label's whole spectrum, buffer-subtracted, one column per well.

    Parameters
    ----------
    ef : EnspireFile
        The parsed EnSpire export.
    label : str
        The plate label naming the scan.
    wells : Sequence[str]
        Wells to read, in order.
    buffer_wells : Sequence[str]
        Buffer wells of the same plate column, subtracted column by column;
        empty keeps the raw spectrum.

    Returns
    -------
    tuple[ArrayF, ArrayF]
        Wavelengths and ``Y[n_lambda, n_wells]``.
    """
    lam, y = _spectra(ef, label, wells)
    if buffer_wells:
        _, yb = _spectra(ef, label, list(buffer_wells))
        by_column = {str(w)[1:]: yb[:, i] for i, w in enumerate(buffer_wells)}
        zero = np.zeros(lam.size)
        y -= np.column_stack([by_column.get(str(w)[1:], zero) for w in wells])
    return lam, y


def sample_spectra(  # ruff: ignore[too-many-arguments]
    ef: EnspireFile,
    wells: list[str],
    x: ArrayF,
    temp: str | float,
    *,
    is_ph: bool,
    buffer_wells: Sequence[str],
    bands: Sequence[str] = EXCITATION_SCAN,
    isosbestic: float | str | None = None,
    iso_band: float = 3.0,
) -> tuple[dict[str, tuple[ArrayF, ArrayF]], list[str], ArrayF, dict[str, float]]:
    """Corrected whole spectra of one sample, one entry per scan.

    Parameters
    ----------
    ef : EnspireFile
        The parsed EnSpire export.
    wells : list[str]
        Wells of this sample.
    x : ArrayF
        Titrant value per well.
    temp : str | float
        Temperature selecting the labels.
    is_ph : bool
        pH titration (else chloride).
    buffer_wells : Sequence[str]
        Buffer wells declared by the note; blank rows serve when there are none.
    bands : Sequence[str]
        Band names (as in :func:`~clophfit.prenspire.bands.classify`) whose scans are used whole;
        the default takes the excitation scan only.
    isosbestic : float | str | None
        Normalise every well by its signal at the isosbestic point, which
        removes per-well amount, path length and brightness: a wavelength [nm],
        ``"auto"`` to locate it with :func:`plate_isosbestic`, or ``None``.
        ``"auto"`` on a titration with no crossing (a chloride one) normalises
        nothing and changes no number.
    iso_band : float
        Full width [nm] of the window averaged at the isosbestic crossing to form
        the per-well reference (a mean, never a sum; see :func:`isosbestic_scale`).
        Every measured point inside the band is averaged; the default 3.0 nm
        averages the 3 points of a 1 nm grid and the 2 of a 2 nm grid, and ``0``
        uses the interpolated value at the crossing itself.

    Returns
    -------
    dict[str, tuple[ArrayF, ArrayF]]
        Scan name to (wavelengths, Y[n_lambda, n_wells]). The key is the scan's
        name (see :func:`scan_name`), not its plate label, so the same
        measurement keeps one name at every temperature.
    list[str]
        The wells used, after any buffer row was set aside.
    ArrayF
        The titrant value of each well, aligned with the columns of Y.
    dict[str, float]
        Scan name to the largest absolute value the scan held before it was
        divided by it. With it any consumer recovers the pre-division units
        exactly - dividing one scan by the other at one wavelength is the ratio
        of two measured quantities - by multiplying Y back by the scan's peak,
        so the same ratio can be drawn at a second wavelength without
        re-scaling the species there.
    """
    readouts = classify(ef, temp, is_ph=is_ph)
    labels = list(dict.fromkeys(r.label for r in readouts if r.name in bands))
    if not labels:
        return {}, wells, x, {}
    blanks = blank_rows(band_values(ef, readouts[0], wells))
    if blanks:
        found = [w for w in wells if w[0] in blanks]
        keep = [i for i, w in enumerate(wells) if w[0] not in blanks]
        wells, x = [wells[i] for i in keep], x[keep]
        buffer_wells = list(buffer_wells) or found
    out: dict[str, tuple[ArrayF, ArrayF]] = {}
    peaks: dict[str, float] = {}
    for label in labels:
        lam, y = scan_spectrum(ef, label, wells, buffer_wells)
        name = scan_name(ef, label)
        peaks[name] = float(np.nanmax(np.abs(y)))
        out[name] = (lam, y / peaks[name])
    if isosbestic is not None:
        lam_iso = (
            plate_isosbestic(out, x, x_min=PH_ACID_END if is_ph else None, is_ph=is_ph)
            if isinstance(isosbestic, str)
            else float(isosbestic)
        )
        if lam_iso is not None:
            key = next(
                (k for k in out if k == "excitation" or k.startswith("excitation ")),
                next(iter(out)),
            )
            out = normalise_to_isosbestic(out, lam_iso, scan=key, band=iso_band)
    return out, wells, x, peaks


def _row(base: dict[str, object], subset: str, fit: SpectralFit) -> dict[str, object]:
    return {
        **base,
        "subset": subset,
        "K": fit.K,
        "se": fit.se,
        "se_naive": fit.se_naive,
        "se_boot": fit.se_boot,
        "ci_lo": fit.ci_boot[0],
        "ci_hi": fit.ci_boot[1],
        "hill": fit.hill,
        "n": fit.n_wells,
        "sv_ratio": float(fit.residual_sv[0] / fit.residual_sv[1]),
    }


def slope_profile(
    spectra: Mapping[str, tuple[Any, Any]], x: ArrayF, *, x_min: float | None = None
) -> dict[str, tuple[ArrayF, ArrayF]]:
    """Per label, how strongly each wavelength responds to the titration.

    The isosbestic point of a two-state titration is the wavelength whose signal
    does not change with the titrant: there the two species spectra cross, so the
    slope of the measured signal against the titrant passes through zero. This
    regresses every wavelength's signal on the titrant value and returns the
    normalized slope profile, which is what the isosbestic search reads.

    ``x_min`` drops the spectra below a titrant value. The most acidic wells are
    the ones that carry the extra acid-end drop that a two-state model cannot
    describe, and they are also the noisiest; including them bends the profile
    and moves the apparent crossing.

    Parameters
    ----------
    spectra : Mapping[str, tuple[Any, Any]]
        Label to ``(wavelengths, Y[n_lambda, n_wells])``.
    x : ArrayF
        Titrant value per well.
    x_min : float | None
        Keep only wells with ``x >= x_min``; ``None`` keeps every well.

    Returns
    -------
    dict[str, tuple[ArrayF, ArrayF]]
        Label to ``(wavelengths, slope)``, the slope normalized to unit maximum
        absolute value so labels with different brightness compare.
    """
    xa = np.asarray(x, dtype=np.float64)
    keep = np.ones(xa.size, dtype=bool) if x_min is None else xa >= x_min
    xk = xa[keep] - xa[keep].mean()
    denom = float(np.sum(xk**2)) or 1.0
    out: dict[str, tuple[ArrayF, ArrayF]] = {}
    for label, (lam, y) in spectra.items():
        ya = np.asarray(y, dtype=np.float64)[:, keep]
        slope = (ya - ya.mean(axis=1, keepdims=True)) @ xk / denom
        top = float(np.max(np.abs(slope))) or 1.0
        out[label] = (np.asarray(lam, dtype=np.float64), slope / top)
    return out


def isosbestic_point(
    spectra: Mapping[str, tuple[Any, Any]], x: ArrayF, *, x_min: float | None = None
) -> dict[str, float]:
    """Give the wavelength at which each label's signal stops tracking the titrant.

    The crossing returned is the **longest-wavelength** zero of
    :func:`slope_profile`. A scan has more than one: the far-UV end carries one
    from the protein's own absorbance (342 nm on the NTT plate) and the
    chromophore's bands carry the one that describes the two-state titration
    (461 nm at 20 C). Taking the longest keeps the chromophore's; a window can be
    imposed afterwards by :func:`consensus_isosbestic`.

    Parameters
    ----------
    spectra : Mapping[str, tuple[Any, Any]]
        Label to ``(wavelengths, Y[n_lambda, n_wells])``.
    x : ArrayF
        Titrant value per well.
    x_min : float | None
        Drop the wells below this titrant value before locating the crossing.

    Returns
    -------
    dict[str, float]
        Label to the interpolated crossing wavelength [nm]. Empty when a label's
        slope profile has no zero crossing.
    """
    found: dict[str, float] = {}
    for label, (lam, slope) in slope_profile(spectra, x, x_min=x_min).items():
        crossings = [
            float(lam[i] + (lam[i + 1] - lam[i]) * slope[i] / (slope[i] - slope[i + 1]))
            for i in range(lam.size - 1)
            if slope[i] * slope[i + 1] < 0
        ]
        if not crossings:
            continue
        found[label] = max(crossings)
    return found


def consensus_isosbestic(
    datasets: Sequence[tuple[Mapping[str, tuple[Any, Any]], ArrayF]],
    *,
    x_min: float | None = None,
    window: tuple[float, float] | None = None,
) -> float:
    """One isosbestic wavelength for several titrations at once.

    A single plate's crossing is noisy (the signal barely changes there, which is
    the point), so several measurements of the same construct are pooled: the
    consensus is the wavelength at which the summed absolute slope profile is
    smallest, restricted to a window around the individual crossings. That
    window is what keeps a protein-absorbance crossing in the far UV from
    competing with the chromophore's own.

    Parameters
    ----------
    datasets : Sequence[tuple[Mapping[str, tuple[Any, Any]], ArrayF]]
        One ``(spectra, x)`` pair per measurement.
    x_min : float | None
        Passed to :func:`slope_profile`; drop the acid end of every dataset.
    window : tuple[float, float] | None
        Wavelength range to search. ``None`` uses the span of the individual
        crossings, widened by 20 nm on each side.

    Returns
    -------
    float
        The consensus isosbestic wavelength [nm].

    Raises
    ------
    ValueError
        If no dataset yields a crossing.
    """
    profiles = [slope_profile(spectra, x, x_min=x_min) for spectra, x in datasets]
    individual = [
        w
        for spectra, x in datasets
        for w in isosbestic_point(spectra, x, x_min=x_min).values()
    ]
    if not individual:
        msg = "no dataset shows an isosbestic crossing; check the wavelength range"
        raise ValueError(msg)
    lo, hi = (
        (min(individual) - 20.0, max(individual) + 20.0) if window is None else window
    )
    grid = np.union1d(
        np.concatenate([lam for prof in profiles for lam, _s in prof.values()]),
        np.arange(lo, hi + 0.001, 0.5),
    )
    grid = grid[(grid >= lo) & (grid <= hi)]
    total = np.zeros(grid.size)
    for prof in profiles:
        for lam, slope in prof.values():
            total += np.abs(np.interp(grid, lam, slope))
    return float(grid[int(np.argmin(total))])


def plate_isosbestic(  # ruff: ignore[too-many-arguments]
    spectra: Mapping[str, tuple[Any, Any]],
    x: ArrayF,
    *,
    scans: Sequence[str] = EXCITATION_SCAN_NAMES,
    x_min: float | None = None,
    max_drift: float | None = 0.20,
    is_ph: bool = True,
    prefer_model: bool = True,
) -> float | None:
    """Return the crossing the spectra of one titration agree on, or None.

    :func:`isosbestic_point` answers per scan, and not every scan has an answer:
    an emission scan taken at 420 nm excitation carries the anionic species
    alone, so it crosses nowhere and reports several near-zeros in its noise,
    while the excitation scan carries the two-state crossover the chromophore
    actually shows. This picks the scans that describe the titration (by default
    the excitation scan), takes each one's longest-wavelength zero, and returns
    their median - the number a per-well normalisation needs.

    A chloride titration is the case that returns ``None``: bound and free
    species differ in brightness without crossing in the measured window, so no
    wavelength is titrant-independent and no such normalisation exists.

    Parameters
    ----------
    spectra : Mapping[str, tuple[Any, Any]]
        Scan name to ``(wavelengths, Y[n_lambda, n_wells])``.
    x : ArrayF
        Titrant value per well.
    scans : Sequence[str]
        Scan names (as :func:`scan_name` builds them) that may carry the
        crossing.
    x_min : float | None
        Drop the wells below this titrant value before locating the crossing.
    max_drift : float | None
        Reject the crossing when the signal there still moves with the titrant:
        a true isosbestic point has the same value in every state, so the wells'
        relative signal at that wavelength must be flat. Measured as the max-min
        of the per-tercile median, relative to the plate median. On the good
        plates it is 0.05-0.08; on a synthetic two-state titration whose true
        crossing is 458 nm the slope profile reported 485 nm, with a drift large
        enough that normalising there drove K to its bound. ``None`` accepts any
        crossing, which is what this did before.

    is_ph : bool
        pH titration; a ligand plate has no crossing of its own and the model rule
        is skipped for it.
    prefer_model : bool
        Try :func:`model_isosbestic` first and fall back to the slope zero. The
        model rule asks what an isosbestic point is (where the two species spectra
        cross) and cannot be satisfied by the protein band; the slope rule is the
        model-free witness kept for the cases where the fit cannot speak.

    Returns
    -------
    float | None
        The crossing [nm], or ``None`` when no chosen scan has one, or when the
        one it found still drifts with the titrant.
    """
    chosen = {
        label: value
        for label, value in spectra.items()
        if any(label == name or label.startswith(f"{name} ") for name in scans)
    }
    if not chosen:
        return None
    if prefer_model and is_ph:
        # ask the model first: the crossing of the two fitted species is what an
        # isosbestic point *is*, and unlike the slope profile it cannot be
        # satisfied by the protein band, which does not titrate either
        from_model = model_isosbestic(chosen, x, is_ph=is_ph, x_min=x_min)
        if from_model is not None:
            return from_model
    found = isosbestic_point(chosen, x, x_min=x_min)
    if not found:
        return None
    lam_iso = float(np.median(list(found.values())))
    if max_drift is not None:
        worst = 0.0
        for lam, y in chosen.values():
            lam_a = np.asarray(lam, dtype=np.float64)
            ya = np.asarray(y, dtype=np.float64)
            if not lam_a.min() <= lam_iso <= lam_a.max():
                return None
            ref = np.array([
                np.interp(lam_iso, lam_a, ya[:, i]) for i in range(ya.shape[1])
            ])
            rel = ref / np.median(ref)
            chunks = np.array_split(np.argsort(np.asarray(x, dtype=float)), 3)
            med = [float(np.median(rel[c])) for c in chunks if c.size]
            worst = max(worst, max(med) - min(med))
        if worst > max_drift:
            return None
    return lam_iso


def isosbestic_by_group(
    ef: EnspireFile,
    note: pd.DataFrame,
    *,
    buffer: bool = True,
) -> dict[tuple[str, float], float | None]:
    """Locate the crossing of every (sample, temperature) group on one plate.

    Each temperature keeps its own crossing, which is the unbiased choice: the
    crossing moves with temperature (4.42 +/- 0.41 nm lower at 37 C than at 20 C on
    every plate of the verification set), so one wavelength for both would put one
    of them off its own crossing and bias its K. What a joint fit and a two-panel
    figure need is not one wavelength but one amplitude scale, and that is what
    :func:`~clophfit.prenspire.bands.fit_titrations` and
    :func:`~clophfit.prenspire.thermo.vanthoff_curves` do with the table this
    returns.

    Parameters
    ----------
    ef : EnspireFile
        The parsed EnSpire export.
    note : pd.DataFrame
        Normalised note, with ``Well``, ``pH``, ``Cl``, ``Name`` and ``Temp``.
    buffer : bool
        Subtract the buffer wells of the same plate column before locating the
        crossing.

    Returns
    -------
    dict[tuple[str, float], float | None]
        ``(sample, temperature)`` to the crossing [nm], ``None`` where the group
        has no crossing (a chloride titration).
    """
    samples = note[note["Name"] != "buffer"]
    buffers = note[note["Name"] == "buffer"]
    out: dict[tuple[str, float], float | None] = {}
    for (name, temp), group in samples.groupby(
        ["Name", "Temp"], sort=True, dropna=False
    ):
        one = group.drop_duplicates("Well")
        is_ph = bool(one["pH"].nunique() > one["Cl"].nunique())
        wells = one["Well"].tolist()
        x = np.asarray(one["pH" if is_ph else "Cl"], dtype=np.float64)
        bufw = (
            buffers[np.isclose(buffers["Temp"].astype(float), float(str(temp)))][
                "Well"
            ].tolist()
            if buffer
            else []
        )
        spectra, _used, used_x, _peaks = sample_spectra(
            ef, wells, x, str(temp), is_ph=is_ph, buffer_wells=bufw
        )
        out[str(name), float(str(temp))] = (
            plate_isosbestic(spectra, used_x) if spectra else None
        )
    return out


def _isosbestic_reference(
    lam: ArrayF, y: ArrayF, lam_iso: float, band: float
) -> ArrayF:
    """Per-well reference at the crossing: the mean over ``band``, or the point itself.

    Every measured point that falls inside ``band`` goes into the mean - two points
    are as legitimate as five, and requiring a minimum silently turned the window
    off on a plate measured on a coarse grid (a 3 nm window holds only two points
    of a 2 nm grid). A non-positive band, or one that contains no grid point at
    all, degrades to the interpolated crossing. One implementation for both
    normalisation paths, so they can never drift apart.
    """
    if band > 0.0:
        window = np.abs(lam - lam_iso) <= band / 2.0
        if window.any():
            return np.asarray(np.mean(y[window], axis=0), dtype=np.float64)
    return np.array([np.interp(lam_iso, lam, y[:, i]) for i in range(y.shape[1])])


def isosbestic_scale(
    spectra: Mapping[str, tuple[Any, Any]],
    lam_iso: float,
    *,
    scan: str | None = None,
    floor: float = 0.02,
    band: float = 3.0,
) -> tuple[ArrayF, ArrayF]:
    """Per-well brightness reference at the crossing, and the wells it can serve.

    The value at the crossing is proportional to how much protein the well
    holds times how bright it is measured, and to nothing else - that is what an
    isosbestic point means. Dividing a well by it therefore removes amount,
    path length and pipetting while keeping the titration's shape, and it can be
    applied to *every* readout of that well, not only to the scan the crossing
    was found in.

    Parameters
    ----------
    spectra : Mapping[str, tuple[Any, Any]]
        Scan name to ``(wavelengths, Y[n_lambda, n_wells])``.
    lam_iso : float
        The crossing [nm]; interpolated between grid points.
    scan : str | None
        Scan to read the reference from; ``None`` takes the first one.
    floor : float
        A reference below this fraction of the well's own peak is not used: at
        the instrument's floor the ratio divides by noise.

    band : float
        Full width [nm] of the window averaged at the crossing to form the
        reference. It is always a *mean* over that band, never a sum: a mean makes
        every well read 1.0 at the crossing, which is what the panel's a.u. axis
        display multiplies back by the scan's own peak, and it does not jump with
        the width. Every measured point inside the band is averaged - the count is
        whatever the plate's grid allows, one point at 1 nm spacing and two on the
        2 nm grid of the L1-L4 plates. A non-positive band, or one that holds no
        grid point at all, uses the interpolated value at the crossing
        (``--iso-window 0``).

    Returns
    -------
    tuple[ArrayF, ArrayF]
        The reference per well (1.0 where it cannot be used) and the boolean
        mask of the wells it could be used for, so a caller can report how many
        wells were left untouched.

    Raises
    ------
    ValueError
        If ``scan`` is missing, or ``lam_iso`` lies outside its range.
    """
    key = scan if scan is not None else next(iter(spectra))
    if key not in spectra:
        msg = f"no scan named {key!r}; have {sorted(spectra)}"
        raise ValueError(msg)
    lam_a = np.asarray(spectra[key][0], dtype=np.float64)
    ya = np.asarray(spectra[key][1], dtype=np.float64)
    if not lam_a.min() <= lam_iso <= lam_a.max():
        msg = f"{key}: isosbestic {lam_iso:.1f} nm is outside {lam_a.min():.0f}-{lam_a.max():.0f} nm"
        raise ValueError(msg)
    # the reference is the MEAN over a small window at the crossing, not a sum: a
    # mean puts every well at 1.0 there and is invariant to how many grid points
    # the band happens to hold. The value at the crossing is a *ratio*, so readout
    # noise on it is carried into every wavelength of the well, and averaging a few
    # points cuts that noise. See :func:`_isosbestic_reference` for the fallback.
    ref = _isosbestic_reference(lam_a, ya, lam_iso, band)
    usable = np.asarray(
        np.isfinite(ref) & (np.abs(ref) > floor * np.max(np.abs(ya), axis=0)),
        dtype=bool,
    )
    scale = np.ones(ref.size, dtype=np.float64)
    scale[usable] = ref[usable]
    return scale, usable


def normalise_to_isosbestic(
    spectra: Mapping[str, tuple[Any, Any]],
    lam_iso: float,
    *,
    scan: str | None = None,
    band: float = 3.0,
) -> dict[str, tuple[Any, Any]]:
    """Divide every spectrum by its value at the isosbestic wavelength.

    At an isosbestic point the signal is the same for every titration state, so
    the ratio to it removes per-well brightness, path length and pipetting while
    leaving the titration's shape untouched - and it puts every spectrum through
    1.0 there, which is what makes the crossing visible.

    Parameters
    ----------
    spectra : Mapping[str, tuple[Any, Any]]
        Label to ``(wavelengths, Y[n_lambda, n_wells])``.
    lam_iso : float
        The isosbestic wavelength [nm]; interpolated between grid points.
    scan : str | None
        Take the per-well reference from this scan and apply it to **every**
        scan, which is the physical normalisation when several readouts of one
        well are fitted together: the crossing is a property of the sample, not
        of the detector window, and an emission scan has no crossing of its own
        to divide by. ``None`` divides each scan by its own value there, which is
        what a single-scan normalisation means.
    band : float
        Full width [nm] of the window averaged at the crossing to form the per-well
        reference. It is a *mean*, never a sum; every measured point inside the
        band is averaged, and ``<= 0`` uses the interpolated value at the crossing.
        See :func:`isosbestic_scale`.

    Returns
    -------
    dict[str, tuple[Any, Any]]
        The same mapping, normalized. A well whose value at the crossing is
        non-finite or below 2 % of that well's peak is left untouched, and its
        index is reported by the caller's statistics rather than silently
        dropped.

    Raises
    ------
    ValueError
        If ``lam_iso`` lies outside a label's wavelength range.
    """
    if scan is not None:
        scale, _usable = isosbestic_scale(spectra, lam_iso, scan=scan, band=band)
        return {
            label: (
                np.asarray(lam, dtype=np.float64),
                np.asarray(y, dtype=np.float64) / scale,
            )
            for label, (lam, y) in spectra.items()
        }
    out: dict[str, tuple[Any, Any]] = {}
    for label, (lam, y) in spectra.items():
        lam_a = np.asarray(lam, dtype=np.float64)
        ya = np.asarray(y, dtype=np.float64)
        if not lam_a.min() <= lam_iso <= lam_a.max():
            msg = f"{label}: isosbestic {lam_iso:.1f} nm is outside {lam_a.min():.0f}-{lam_a.max():.0f} nm"
            raise ValueError(msg)
        ref = _isosbestic_reference(lam_a, ya, lam_iso, band)
        floor = 0.02 * np.max(np.abs(ya), axis=0)
        usable = np.isfinite(ref) & (np.abs(ref) > floor)
        scaled = ya.copy()
        scaled[:, usable] /= ref[usable]
        out[label] = (lam_a, scaled)
    return out


def fit_titrations_spectral(  # ruff: ignore[too-many-arguments, too-many-branches, too-many-statements, complex-structure]
    ef: EnspireFile,
    note: pd.DataFrame,
    *,
    buffer: bool = True,
    well_scale: bool = True,
    fit_hill: bool = False,
    screen: float | None = None,
    bands: Sequence[str] = EXCITATION_SCAN,
    isosbestic: float | str | None = None,
    iso_band: float = 3.0,
    n_boot: int = 0,
    seed: int = 0,
    strata: str = "none",
    qc: float | None = None,
    species_min: float | None = None,
    inspect: bool = False,
    figure_stem: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, SpectralFit]]:
    """Fit every sample of a note with the whole-spectrum global model.

    Parameters
    ----------
    ef : EnspireFile
        The parsed EnSpire export.
    note : pd.DataFrame
        Normalised note, with ``Well``, ``pH``, ``Cl``, ``Name`` and ``Temp``.
    buffer : bool
        Subtract the buffer well of the same plate column.
    well_scale : bool
        Fit a per-well amplitude as well (see :func:`fit_spectra_global`).
    fit_hill : bool
        Also fit the Hill slope.
    screen : float | None
        Family-wise level of the well screen passed to
        :func:`~clophfit.fitting.spectral.fit_spectra_global`; ``None`` keeps
        every well.
    bands : Sequence[str]
        Band names whose scans are used whole (default: the excitation scan only).
    isosbestic : float | str | None
        Normalise every well by its signal at the isosbestic point before
        fitting (see :func:`sample_spectra`); ``"auto-scale"`` resolves the
        crossings for the whole plate at once.
    iso_band : float
        Full width [nm] of the window averaged at the isosbestic crossing to form
        the per-well reference (a mean, never a sum - see :func:`isosbestic_scale`).
        Every measured point inside the band is averaged; the default 3.0 nm
        averages the 3 points of a 1 nm grid and the 2 of a 2 nm grid, and ``0``
        uses the interpolated value at the crossing itself.
    n_boot : int
        Well-cluster bootstrap replicates for the reported interval (0 leaves it
        out). The interval resamples whole wells - every scan and every
        wavelength of a well together - because wavelengths inside a spectrum are
        not independent observations, and the Jacobian error of a whole-spectrum
        fit is orders of magnitude too small. See
        :func:`~clophfit.fitting.spectral.fit_spectra_global`.
    seed : int
        Seed of the bootstrap.
    strata : str
        ``"none"`` resamples wells from one pool; ``"column"`` resamples within
        each plate column, which keeps every replicate spanning the titration
        axis and is the design-preserving choice for these plates.
    qc : float | None
        Robust cut for the per-scan screen (:func:`scan_outliers`), applied to the
        corrected scans *before* the fit: a well whose leave-one-out rank-2
        residual exceeds ``median + qc * MAD`` in any scan is dropped from every
        scan of that group, and the dropped wells are recorded in the table. The
        screen never tests a well whose peak is below 5 % of the plate median
        (``min_signal``), so a dark but valid well - the saturating end of a
        ligand ladder - is not mistaken for a broken one. ``None`` fits every
        well, which is what this method did before.
    species_min : float | None
        Lower bound on every species spectrum, per wavelength, passed to
        :func:`~clophfit.fitting.spectral.fit_spectra_global`. ``0`` is the
        physical choice for fluorescence; ``None`` (the default) keeps the
        unconstrained linear solve that every earlier number came from.
    inspect : bool
        With a ``figure_stem``, also write one ``<stem>_global_inspect.txt``
        listing, per (sample, temperature): K and its errors, the wells used and
        the wells the screen dropped, each scan's basic/acidic species peaks and
        residual RMS, the two-state residual singular values, and every well's
        basic fraction and amplitude. The panel figure and the ``--global-qc``
        output are unchanged - this is a text dump only.

    figure_stem : Path | None
        When given, write ``<stem>_<sample>_<temp>_panels.pdf`` per group: the data,
        the model and the species fraction, one row per scan
        (:func:`plot_global_panels`).

    Returns
    -------
    tuple[pd.DataFrame, dict[str, SpectralFit]]
        One row per (sample, temperature, subset) - ``all`` then each replicate row - and, per
        ``"<sample> <temp>"``, the fit on all wells with its species spectra.
    """
    rows: list[dict[str, object]] = []
    fits: dict[str, SpectralFit] = {}
    inspect_lines: list[str] = []
    samples = note[note["Name"] != "buffer"]
    buffers = note[note["Name"] == "buffer"]
    lam_map = (
        isosbestic_by_group(ef, note, buffer=buffer)
        if isosbestic == "auto-scale"
        else None
    )
    for (name, temp), group in samples.groupby(
        ["Name", "Temp"], sort=True, dropna=False
    ):
        one = group.drop_duplicates("Well")
        is_ph = bool(one["pH"].nunique() > one["Cl"].nunique())
        wells = one["Well"].tolist()
        x = np.asarray(one["pH" if is_ph else "Cl"], dtype=np.float64)
        buf = buffers[buffers["Temp"] == temp]["Well"].tolist() if buffer else []
        lam = (
            lam_map.get((str(name), float(str(temp))))
            if lam_map is not None
            else isosbestic
        )
        iso_nm: float | None = None
        if isinstance(lam, str):
            # resolve "auto"/"auto-scale" here rather than inside sample_spectra, so
            # the figure can be divided by the same wavelength the fit used
            raw, _w, xr, _peaks = sample_spectra(
                ef, wells, x, str(temp), is_ph=is_ph, buffer_wells=buf, bands=bands
            )
            iso_nm = (
                plate_isosbestic(
                    raw,
                    xr,
                    x_min=PH_ACID_END if is_ph else None,
                    is_ph=is_ph,
                )
                if raw
                else None
            )
            lam = iso_nm
        elif isinstance(lam, float):
            iso_nm = lam
        spectra, wells, x, peaks = sample_spectra(
            ef,
            wells,
            x,
            str(temp),
            is_ph=is_ph,
            buffer_wells=buf,
            bands=bands,
            isosbestic=lam,
            iso_band=iso_band,
        )
        if not spectra:
            continue
        dropped: list[str] = []
        if qc is not None:
            # screen against the model, not against the wells: the two species of
            # a first fit are a fixed basis, which is scale-free (a dark well is
            # judged on shape) and breaks the symmetry between the two wells of a
            # replicate column. Always unconstrained: the z was calibrated on that
            # residual, so bounding the species must not change which wells are
            # called broken (see _screen_model).
            screen_fit = fit_spectra_global(
                x,
                spectra,
                is_ph=is_ph,
                fit_hill=fit_hill,
                well_scale=well_scale,
                jackknife=False,
            )
            qc_table = model_outliers(screen_fit, spectra, wells, z=qc)
            dropped = list(qc_table.attrs.get("flagged_wells", []))
            if dropped:
                gone = set(dropped)
                mask = np.array([w not in gone for w in wells], dtype=bool)
                spectra = {
                    key: (lam_arr, y[:, mask]) for key, (lam_arr, y) in spectra.items()
                }
                wells = [w for w, k in zip(wells, mask, strict=True) if k]
                x = x[mask]
        base: dict[str, object] = {
            "sample": name,
            "temp": temp,
            "qc_dropped": ";".join(dropped) or "-",
            # the options that move the numbers, recorded per row so two runs that
            # differ only in them cannot leave identical tables
            "scans": ";".join(spectra),
            "isosbestic_nm": np.nan if iso_nm is None else float(iso_nm),
            "iso_window_nm": np.nan if iso_nm is None else float(iso_band),
            "species_min": np.nan if species_min is None else float(species_min),
        }
        qc_passes = 0
        qc_strata = [str(w)[1:] for w in wells] if strata == "column" else None
        fit = fit_spectra_global(
            x,
            spectra,
            is_ph=is_ph,
            fit_hill=fit_hill,
            well_scale=well_scale,
            screen=screen,
            species_min=species_min,
            n_boot=n_boot,
            seed=seed,
            strata=qc_strata,
        )
        # verification pass: the screen was computed from a fit that still held the
        # outliers, so recompute it on the clean model and drop anything it names
        # now. Bounded, because each pass is a refit and the set is stable after
        # one or two; the pass count goes into the table so the iteration is
        # visible rather than implicit.
        for _ in range(_QC_MAX_PASSES):
            if qc is None or not spectra:
                break
            # re-screen on an UNCONSTRAINED refit when the species are bounded: the
            # bound leaves larger residuals by construction, and the z was
            # calibrated on the free model, so screening the bounded fit would
            # change which wells are called broken
            screen_model = (
                fit
                if species_min is None
                else fit_spectra_global(
                    x,
                    spectra,
                    is_ph=is_ph,
                    fit_hill=fit_hill,
                    well_scale=well_scale,
                    screen=screen,
                    jackknife=False,
                )
            )
            check = model_outliers(screen_model, spectra, wells, z=qc)
            extra = [
                w for w in check.attrs.get("flagged_wells", []) if w not in dropped
            ]
            if not extra:
                break
            dropped = [*dropped, *extra]
            qc_passes += 1
            gone = set(dropped)
            mask = np.array([w not in gone for w in wells], dtype=bool)
            spectra = {
                key: (lam_arr, y[:, mask]) for key, (lam_arr, y) in spectra.items()
            }
            wells = [w for w, keep in zip(wells, mask, strict=True) if keep]
            x = x[mask]
            base["qc_dropped"] = ";".join(dropped) or "-"
            fit = fit_spectra_global(
                x,
                spectra,
                is_ph=is_ph,
                fit_hill=fit_hill,
                well_scale=well_scale,
                screen=screen,
                species_min=species_min,
                n_boot=n_boot,
                seed=seed,
                strata=qc_strata,
            )
        base["qc_passes"] = qc_passes
        if figure_stem is not None:
            key = f"{name}_{str(temp).replace('.', 'p')}"
            plot_global_panels(
                figure_stem.parent / f"{figure_stem.name}_{key}_panels.pdf",
                spectra,
                x,
                fit,
                title=f"{name} at {temp}",
                iso_nm=iso_nm,
                peaks=peaks,
            )
        fits[f"{name} {temp}"] = fit
        if inspect:
            inspect_lines.append(
                _group_inspect(
                    name,
                    temp,
                    x,
                    wells,
                    fit,
                    base,
                    is_ph=is_ph,
                    iso_nm=iso_nm,
                    peaks=peaks,
                )
            )
        rows.append(_row(base, "all", fit))
        for plate_row in sorted({w[0] for w in wells}):
            sel = np.array([w[0] == plate_row for w in wells])
            if sel.sum() < _MIN_ROW_WELLS:
                continue
            sub = {lab: (lam, y[:, sel]) for lab, (lam, y) in spectra.items()}
            # the row breakdown is a diagnostic: it keeps every well of the row
            row_fit = fit_spectra_global(
                x[sel],
                sub,
                is_ph=is_ph,
                fit_hill=fit_hill,
                well_scale=well_scale,
                species_min=species_min,
                jackknife=False,
            )
            rows.append(_row(base, f"row {plate_row}", row_fit))

    if inspect and figure_stem is not None:
        insp_fp = figure_stem.parent / f"{figure_stem.name}_global_inspect.txt"
        insp_fp.write_text("\n".join(inspect_lines).rstrip("\n") + "\n")
        print(f"inspect: wrote {insp_fp}")
    return pd.DataFrame(rows), fits


def scan_outliers(  # ruff: ignore[too-many-arguments]
    spectra: Mapping[str, tuple[Any, Any]],
    wells: Sequence[str],
    x: ArrayF,
    *,
    z: float = 6.0,
    n_states: int = 2,
    min_signal: float = 0.0,
) -> pd.DataFrame:
    """Flag wells whose spectrum no two-state mixture can explain, scan by scan.

    The point of this check is to look at each scan on its own, before any model
    is fitted and before several scans are combined: a well that is off in one
    scan only is a measurement fault (bubble, focus, an air gap at 350-390 nm),
    while a well off in all of them is a bad well or a bad pH.

    Each well is projected onto the subspace spanned by the first ``n_states``
    singular vectors of the *other* wells and the residual is kept. Projecting on
    a basis computed without the well is what makes the test honest: fitted on
    all the wells, a spectral outlier helps define the basis it is then compared
    with, and hides.

    Parameters
    ----------
    spectra : Mapping[str, tuple[Any, Any]]
        Scan name to ``(wavelengths, Y[n_lambda, n_wells])``, already corrected
        per well (buffer subtracted).
    wells : Sequence[str]
        Well label per column of every ``Y``.
    x : ArrayF
        Titrant value per well, for reporting.
    z : float
        Robust cut on the residual: a well whose leave-one-out residual root mean
        square exceeds ``median + z * MAD`` in any scan is flagged.
    n_states : int
        Rank of the expected mixture. A two-state titration spans at most rank 2.
    min_signal : float
        Wells whose own peak is below this fraction of the plate's median peak are
        not tested: a well near the instrument floor carries no shape to judge,
        and testing it flags the saturating end of a binding ladder - where the
        bound state is legitimately dark - as broken. They are reported with a
        ``nan`` statistic and are not flagged. ``0`` tests every well.

    Returns
    -------
    pd.DataFrame
        One row per (scan, well): ``scan``, ``well``, ``x``, ``rms`` (the
        leave-one-out residual RMS over wavelengths), ``robust_z`` and
        ``peak_nm`` (where that residual is largest). Sorted by ``robust_z``
        descending, so the worst spectrum of the plate is the first row.
    """
    rows: list[dict[str, object]] = []
    for name, (lam, y) in spectra.items():
        lam_a = np.asarray(lam, dtype=np.float64)
        ya = np.asarray(y, dtype=np.float64)
        n_wells = ya.shape[1]
        if n_wells < n_states + 2:
            continue
        rms = np.zeros(n_wells)
        peak = np.zeros(n_wells)
        for j in range(n_wells):
            others = np.delete(ya, j, axis=1)
            u, _s, _vt = np.linalg.svd(others, full_matrices=False)
            basis = u[:, :n_states]
            resid = ya[:, j] - basis @ (basis.T @ ya[:, j])
            rms[j] = float(np.sqrt(np.mean(resid**2)))
            peak[j] = float(lam_a[int(np.argmax(np.abs(resid)))])
        med = float(np.median(rms))
        mad = float(1.4826 * np.median(np.abs(rms - med)))
        scale = mad if mad > 0 else float(np.std(rms)) or 1.0
        robust_z = (rms - med) / scale
        if min_signal > 0.0:
            peaks = np.nanmax(np.abs(ya), axis=0)
            dark = peaks < min_signal * float(np.median(peaks))
            robust_z = np.where(dark, np.nan, robust_z)
        rows.extend(
            {
                "scan": name,
                "well": wells[j],
                "x": float(np.asarray(x, dtype=np.float64)[j]),
                "rms": rms[j],
                "robust_z": float(robust_z[j])
                if np.isfinite(robust_z[j])
                else float("nan"),
                "peak_nm": peak[j],
                "flagged": bool(np.isfinite(robust_z[j]) and robust_z[j] > z),
            }
            for j in range(n_wells)
        )
    table = pd.DataFrame(
        rows,
        columns=["scan", "well", "x", "rms", "robust_z", "peak_nm", "flagged"],
    )
    if table.empty:
        return table
    return table.sort_values("robust_z", ascending=False).reset_index(drop=True)


def residual_table(fit: SpectralFit, wells: Sequence[str], x: ArrayF) -> pd.DataFrame:
    """Turn a whole-spectrum fit's residuals into the per-scan R_s(pH) table.

    ``R_s(w)`` is what the residual of a scan looks like as a function of the
    titration: the root mean square over that scan's wavelengths. Three such
    traces (one per scan) say *where* a model fails - at the transition, at the
    acid end, on one side of the crossing - which a single pooled RMS cannot.

    Parameters
    ----------
    fit : SpectralFit
        A fit from :func:`~clophfit.fitting.spectral.fit_spectra_global`.
    wells : Sequence[str]
        Well label per column of the residual matrices.
    x : ArrayF
        Titrant value per well.

    Returns
    -------
    pd.DataFrame
        One row per (scan, well): ``scan``, ``well``, ``x`` and ``rms`` (the
        per-well residual RMS over that scan's wavelengths).
    """
    rows: list[dict[str, object]] = []
    for scan, values in fit.well_rms.items():
        for j, value in enumerate(np.asarray(values, dtype=np.float64)):
            rows.append({
                "scan": scan,
                "well": wells[j],
                "x": float(np.asarray(x, dtype=np.float64)[j]),
                "rms": float(value),
            })
    return pd.DataFrame(rows, columns=["scan", "well", "x", "rms"])


def model_outliers(  # ruff: ignore[too-many-arguments]
    fit: SpectralFit,
    spectra: Mapping[str, tuple[Any, Any]],
    wells: Sequence[str],
    *,
    z: float = 6.0,
    min_scans: int = 2,
    z_single: float | None = None,
) -> pd.DataFrame:
    """Flag wells whose spectrum the fitted two species cannot account for.

    The screen works from the fit, not from the wells: the two species spectra the
    whole-spectrum model already solved are a *fixed* basis, so a well is judged
    against what the model says it should look like rather than against its
    neighbours. Each spectrum is projected onto the span of the two species and the
    leftover, relative to that well's own norm, is the statistic - a dark well is
    therefore judged on shape and not on brightness.

    The fixed basis also breaks the symmetry between the two replicate wells of a
    column: a basis rebuilt from a single sibling cannot say which of the pair is
    the odd one, while this one names it. On the G02 chloride plate it flags B06
    (80 mM) at both temperatures with a robust z near 18.

    A well is flagged when it is beyond the cut in at least ``min_scans`` scans, so
    one noisy readout cannot remove a well on its own. A fit that consumes a single
    scan therefore flags nothing: one readout cannot corroborate itself, and
    degrading the rule to it removes the saturating end of a ligand ladder on the
    strength of one noisy scan.

    Parameters
    ----------
    fit : SpectralFit
        A fit of the same wells, from
        :func:`~clophfit.fitting.spectral.fit_spectra_global`.
    spectra : Mapping[str, tuple[Any, Any]]
        Scan name to ``(wavelengths, Y[n_lambda, n_wells])`` - the arrays the fit
        consumed.
    wells : Sequence[str]
        Well label per column of every ``Y``.
    z : float
        Robust cut on the relative residual, applied per scan.
    min_scans : int
        How many scans must exceed ``z`` before the well is flagged.
    z_single : float | None
        A single scan is enough when it puts the well above this much larger cut;
        ``None`` sets it to ``5/3 * z``, so raising ``z`` raises both cuts and the
        option stays monotone. Pinning it at 10 made ``--global-qc 20`` and
        ``--global-qc 200`` behave exactly like ``--global-qc 6``.
        The two-scan rule protects against one noisy readout, but applied alone it
        also protects a well that only one scan can see is broken - on the G02
        chloride plate B06 sits at z = 16-18 in the 278 nm-excited emission and at
        z ~ 3 in the excitation scan, and requiring both scans left it in the fit.

    Returns
    -------
    pd.DataFrame
        One row per (scan, well) with ``scan``, ``well``, ``rms`` (the relative
        residual), ``robust_z`` and ``flagged``. The wells that pass the
        cross-scan rule are listed in ``frame.attrs["flagged_wells"]``.
    """
    rows: list[dict[str, object]] = []
    hits: dict[str, int] = {}
    worst: dict[str, float] = {}
    for name, (_lam, y) in spectra.items():
        if name not in fit.species:
            continue
        species = np.asarray(fit.species[name], dtype=np.float64)
        basis, _ = np.linalg.qr(species.T)
        ya = np.asarray(y, dtype=np.float64)
        rel = np.zeros(ya.shape[1])
        for j in range(ya.shape[1]):
            coef, *_ = np.linalg.lstsq(basis, ya[:, j], rcond=None)
            r = ya[:, j] - basis @ coef
            norm = float(np.linalg.norm(ya[:, j]))
            rel[j] = float(np.linalg.norm(r)) / norm if norm > 0 else 0.0
        med = float(np.median(rel))
        mad = float(1.4826 * np.median(np.abs(rel - med)))
        scale = mad if mad > 0 else float(np.std(rel)) or 1.0
        rz = (rel - med) / scale
        for j, well in enumerate(wells):
            over = bool(rz[j] > z)
            hits[well] = hits.get(well, 0) + int(over)
            worst[well] = max(worst.get(well, -np.inf), float(rz[j]))
            rows.append({
                "scan": name,
                "well": well,
                "rms": float(rel[j]),
                "robust_z": float(rz[j]),
                "flagged": over,
            })
    table = pd.DataFrame(rows, columns=["scan", "well", "rms", "robust_z", "flagged"])
    # never degrade to one scan: a single readout cannot corroborate itself, and
    # degrading drops the saturating end of a ligand ladder on the strength of
    # one noisy scan (seen on the G02 chloride plate: A09/B09 at 1000 mM)
    need = min_scans
    single = 5.0 * z / 3.0 if z_single is None else z_single
    table.attrs["flagged_wells"] = sorted(
        w for w in hits if hits[w] >= need or worst[w] > single
    )
    return table


def model_isosbestic(
    spectra: Mapping[str, tuple[Any, Any]],
    x: ArrayF,
    *,
    is_ph: bool = True,
    lam_min: float = 420.0,
    x_min: float | None = None,
) -> float | None:
    """Crossing of the two fitted species spectra, not of the slope profile.

    The slope profile answers "where does the signal stop tracking the titrant",
    and more than one wavelength satisfies that: the protein's own absorbance in
    the far UV does not titrate either, so on a thin or noisy titration the
    longest-wavelength zero can be the protein's crossing rather than the
    chromophore's, and ranking the zeros by how titrant-independent they are picks
    the protein every time. This asks the question directly instead: fit the two
    states once, and take the wavelength where the two *species spectra* cross,
    restricted to the chromophore's region. A non-titrating band cannot cross its
    own partner there because it does not move between states.

    Parameters
    ----------
    spectra : Mapping[str, tuple[Any, Any]]
        Scan name to ``(wavelengths, Y[n_lambda, n_wells])``.
    x : ArrayF
        Titrant value per well.
    is_ph : bool
        pH titration (else ligand).
    lam_min : float
        Shortest wavelength considered [nm]; the default keeps the protein
        absorbance out of the search.
    x_min : float | None
        Drop the wells below this titrant value before fitting.

    Returns
    -------
    float | None
        The crossing [nm], or ``None`` when the species do not cross above
        ``lam_min`` or the fit fails.
    """
    from clophfit.fitting.spectral import fit_spectra_global  # ruff: ignore[import-outside-top-level]

    xa = np.asarray(x, dtype=np.float64)
    keep = np.ones(xa.size, dtype=bool) if x_min is None else xa >= x_min
    if keep.sum() < 5:  # ruff: ignore[magic-value-comparison] - four wells is the fit's own floor
        return None
    smooth: dict[str, tuple[Any, Any]] = {}
    for label, (lam, y) in spectra.items():
        lam_a = np.asarray(lam, dtype=np.float64)
        ya = np.asarray(y, dtype=np.float64)
        smooth[label] = (lam_a, ya[:, keep])
    try:
        fit = fit_spectra_global(
            xa[keep], smooth, is_ph=is_ph, well_scale=True, jackknife=False
        )
    except (ValueError, np.linalg.LinAlgError):
        return None
    crossings: list[float] = []
    for label, species in fit.species.items():
        lam = np.asarray(fit.wavelengths[label], dtype=np.float64)
        diff = np.asarray(species[0], dtype=np.float64) - np.asarray(
            species[1], dtype=np.float64
        )
        for i in range(lam.size - 1):
            if lam[i] < lam_min:
                continue
            if diff[i] * diff[i + 1] < 0:
                crossings.append(
                    float(
                        lam[i]
                        + (lam[i + 1] - lam[i]) * diff[i] / (diff[i] - diff[i + 1])
                    )
                )
    if not crossings:
        return None
    return max(crossings)


def _group_inspect(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
    name: object,
    temp: object,
    x: ArrayF,
    wells: Sequence[str],
    fit: SpectralFit,
    base: Mapping[str, object],
    *,
    is_ph: bool,
    iso_nm: float | None,
    peaks: Mapping[str, float],
) -> str:
    """Assemble one (sample, temperature) block for the ``--inspect`` sidecar.

    Text only: a human-readable dump of what the table holds for the group, plus
    each scan's basic/acidic species peaks, how far below zero either species
    dips, and every well's basic fraction and amplitude. Nothing here is written
    to the panel figure.
    """
    xa = np.asarray(x, dtype=np.float64)
    k = float(fit.K)
    # ``acidic`` weights species[1] - the protonated state on a pH titration, the
    # bound state on a ligand one. The reported share is the state the titrant
    # grows: the deprotonated anionic (basic) one for pH, the bound one for a
    # ligand. Naming the protonated share "basic" is the f_basic/f_neutral swap.
    acidic = 1.0 / (1.0 + 10.0 ** (xa - k)) if is_ph else xa / (xa + k)
    reported = (1.0 - acidic) if is_ph else acidic
    frac_key, frac_name = (
        ("basic", "basic (anionic) fraction") if is_ph else ("bound", "bound fraction")
    )
    scale = np.asarray(fit.well_scale, dtype=np.float64)
    boot = (
        f" boot {fit.se_boot:.4f} [{fit.ci_boot[0]:.4f}, {fit.ci_boot[1]:.4f}]"
        if np.isfinite(fit.se_boot)
        else ""
    )
    dropped = str(base.get("qc_dropped", "-"))
    hill = f"   hill {fit.hill:.3f}" if abs(fit.hill - 1.0) > 1e-9 else ""  # ruff: ignore[magic-value-comparison]
    lines: list[str] = [
        f"== {name} at {temp} ==",
        f"K = {k:.4f} +/- {fit.se:.4f} (jackknife){boot}{hill}",
        f"wells used: {fit.n_wells}   dropped: {dropped}",
    ]
    if iso_nm is not None:
        lines.append(f"isosbestic crossing: {iso_nm:.2f} nm")
    lines.append(
        "per-scan species peaks and residual RMS (scan divided by its own peak):"
    )
    for label in sorted(fit.species):
        species = np.asarray(fit.species[label], dtype=np.float64)
        lam = np.asarray(fit.wavelengths.get(label, []), dtype=np.float64)
        rms = float(fit.rms[label])
        pf = float(peaks.get(label, 1.0))  # back to the scan's measured units
        n_w = float(np.max(np.abs(species)))
        if species.shape[0] >= 2:  # ruff: ignore[magic-value-comparison]
            b0 = int(np.argmax(np.abs(species[0])))
            a0 = int(np.argmax(np.abs(species[1])))
            b_at = f"{lam[b0]:.0f} nm" if lam.size else "?"
            a_at = f"{lam[a0]:.0f} nm" if lam.size else "?"
            lines.append(
                f"  {label}: basic {n_w:.3e} @ {b_at}   "
                f"acidic {float(np.max(np.abs(species[1]))):.3e} @ {a_at}   "
                f"rms {rms:.3e}"
            )
            # a species that should be dark in a scan has nothing holding it at
            # zero, so the unconstrained solve can give it a negative lobe; that
            # is only visible here, never on the figure
            dips = [
                f"{name} min {float(np.min(s)):.3e} ({int(np.sum(s < 0.0))} pts < 0)"
                for name, s in (
                    ("basic", species[0] * pf),
                    ("acidic", species[1] * pf),
                )
            ]
            lines.append("      " + ", ".join(dips))
        else:
            lines.append(f"  {label}: (single species) peak {n_w:.3e}   rms {rms:.3e}")
    sv = np.asarray(fit.residual_sv, dtype=np.float64)
    if sv.size >= 2 and sv[1] > 0.0:  # ruff: ignore[magic-value-comparison]
        lines.append(
            f"2-state residual sv {sv[0]:.3e}, {sv[1]:.3e}   ratio {sv[0] / sv[1]:.2f}"
        )
    lines.append(f"per-well {frac_name} f / amplitude a:")
    for w, fb, sc in zip(wells, reported, scale, strict=True):
        lines.append(f"  {w}: f_{frac_key} = {fb:.4f}   a = {sc:.4e}")
    lines.append("")
    return "\n".join(lines)


def plot_global_panels(  # ruff: ignore[too-many-arguments, too-many-statements]
    path: Path,
    spectra: Mapping[str, tuple[Any, Any]],
    x: ArrayF,
    fit: SpectralFit,
    *,
    title: str,
    iso_nm: float | None = None,
    peaks: Mapping[str, float] | None = None,
) -> Path:
    """Draw the data, the model and the species fraction of one whole-spectrum fit.

    The species-only figure that accompanied this fit says what the fit found but
    not whether it describes the data. These three do: the measured spectra and
    the model drawn on the same scale, and the fraction of the basic state per
    well against the fitted curve. Every quantity a panel draws carries one
    shared constant - the division of its scan by its peak (``peaks``,
    returned by ``sample_spectra``), on top of the well reference that already
    divided both the data and the species (the excitation value at the shared
    crossing, applied by ``sample_spectra``). So each panel shows the scan in
    its own measured units - the excitation family passes through 1.0 there when
    that reference was applied - and
    the species sit on the data scale by construction, because they were fit
    on those data and are rescaled by the same constant.

    The observed fraction is one per well, shared across scans
    (:func:`observed_fraction`): the per-scan amplitude of a well is a free
    scale, so unmixing the scans separately would trade it into the fraction.

    Parameters
    ----------
    path : Path
        Where to write the figure (``.pdf`` or ``.png``, by suffix).
    spectra : Mapping[str, tuple[Any, Any]]
        The scans the fit consumed, as ``sample_spectra`` returned them.
    x : ArrayF
        Titrant value per well.
    fit : SpectralFit
        The fit on exactly those spectra.
    title : str
        Panel-group title, usually "<sample> at <temperature>".

    iso_nm : float | None
        The shared crossing [nm]; only the first panel marks it, as the
        wavelength the fit divided every scan by. ``None`` falls back to each
        scan's own species crossing.

    peaks : Mapping[str, float] | None
        The peak that ``sample_spectra`` divided each scan by (its 4th return
        value). Each panel multiplies data, model and species by it to restore
        the scan's measured units; ``None`` draws the panels as given.

    Returns
    -------
    Path
        The path of the figure that was written.
    """
    import matplotlib.pyplot as plt  # ruff: ignore[import-outside-top-level]
    from matplotlib import colormaps  # ruff: ignore[import-outside-top-level]
    from matplotlib.colors import Normalize  # ruff: ignore[import-outside-top-level]

    # a scan with no signal above the buffer (the 420 nm emission of some 2013
    # exports) cannot carry a crossing, a species pair or a fraction: drawing it
    # divides by a species value that is zero and reports a crossing at the edge
    # of its window. Those scans are dropped from the figure, not from the fit.
    tops = {
        name: float(np.nanmax(np.abs(np.asarray(y, dtype=np.float64))))
        for name, (_lam, y) in spectra.items()
        if name in fit.species
    }
    topmost = max(tops.values()) if tops else 0.0
    scans = [
        name
        for name in spectra
        if name in fit.species and topmost > 0 and tops[name] > 0.05 * topmost
    ]
    if not scans:
        return path
    is_ph = float(np.nanmax(x)) < _MAX_PH
    # the fit's species[1] is the protonated (or bound) state, so this weight and
    # (1 - f_acid) rebuild the model exactly as _fractions does
    f_acid = 1.0 / (1.0 + 10.0 ** (x - fit.K)) if is_ph else x / (x + fit.K)
    # one convention for the drawn fraction: the state the titrant grows - the
    # deprotonated anionic (basic) state on a pH titration, the bound state on a
    # ligand one. Calling the protonated share "basic" swaps f_basic for f_neutral.
    basic_label = "basic (anionic) state" if is_ph else "unbound state"
    acid_label = "acidic (protonated) state" if is_ph else "bound state"
    frac_ylabel = (
        "fraction of the basic (anionic) state"
        if is_ph
        else "fraction of the bound state"
    )
    frac_title = (
        "basic (anionic) fraction, one per well"
        if is_ph
        else "bound fraction, one per well"
    )
    grid = (
        np.linspace(float(np.nanmin(x)) - 0.2, float(np.nanmax(x)) + 0.2, 200)
        if is_ph
        else np.logspace(np.log10(0.5), np.log10(float(np.nanmax(x)) * 1.3), 200)
    )

    def curve(k: float) -> ArrayF:
        """Share of the state the titrant grows at one K, on the panel's grid."""
        if is_ph:
            # anionic: 0 at low pH, 1 at high pH, the mirror of the neutral share
            ratio = 10.0 ** (grid - k)
            return ratio / (1.0 + ratio)
        return grid / (grid + k)

    fig, axes = plt.subplots(
        1,
        len(scans) + 1,
        figsize=(4.3 * (len(scans) + 1), 3.9),
        squeeze=False,
        constrained_layout=True,
    )
    cmap, norm = (
        colormaps["coolwarm_r"],
        Normalize(float(np.nanmin(x)), float(np.nanmax(x))),
    )
    for col, label in enumerate(scans):
        lam, y = spectra[label]
        lam = np.asarray(lam, dtype=np.float64)
        ya = np.asarray(y, dtype=np.float64)
        species = np.asarray(fit.species[label], dtype=np.float64)
        model = np.asarray(fit.well_scale)[None, :] * (
            (1 - f_acid)[None, :] * species[0][:, None]
            + f_acid[None, :] * species[1][:, None]
        )
        diff = species[0] - species[1]
        if iso_nm is not None:
            iso = float(iso_nm)
        else:
            zero = [
                float(
                    lam[i] + (lam[i + 1] - lam[i]) * diff[i] / (diff[i] - diff[i + 1])
                )
                for i in range(lam.size - 1)
                if diff[i] * diff[i + 1] < 0 and lam[i] > _CHROMOPHORE_MIN
            ]
            iso = max(zero) if zero else float(lam[int(np.argmin(np.abs(diff)))])
        # one shared constant per panel: the peak this scan was divided by
        # (``peaks``, straight from ``sample_spectra``). When the fit also divided
        # every scan by its excitation value at the crossing
        # (--normalise-isosbestic), multiplying by that peak restores the well's own
        # a.u. scale; when it did not, the peak division is all there was. Data,
        # model and species all carry the same constant, so the species stay on the
        # data scale by construction instead of being rescaled onto the scale of
        # some other scan.
        m = float(peaks[label]) if peaks is not None and label in peaks else 1.0
        ax = axes[0][col]
        for i in range(ya.shape[1]):
            ax.plot(lam, m * ya[:, i], color=cmap(norm(x[i])), lw=0.9, alpha=0.45)
            ax.plot(lam, m * model[:, i], color="k", lw=0.6, ls="--", alpha=0.35)
        ax.plot(lam, m * species[0], color="#2a6f97", lw=2.0, label=basic_label)
        ax.plot(
            lam,
            m * species[1],
            color="#d1495b",
            lw=2.0,
            label=acid_label,
        )
        if col == 0:
            # the crossing belongs to the excitation scan: it is the wavelength the
            # fit divided every scan by, not a feature of an emission window. Only
            # claim a divisor when one was actually applied (--normalise-isosbestic);
            # otherwise the line just marks where the two fitted species cross.
            ax.axvline(iso, color="k", lw=0.8, ls=":")
            what = "shared divisor" if iso_nm is not None else "species cross at"
            ax.set_title(
                f"{label}: {fit.n_wells} wells, {what} {iso:.1f} nm",
                fontsize=9.5,
                loc="left",
            )
        else:
            ax.set_title(f"{label}: {fit.n_wells} wells", fontsize=9.5, loc="left")
        ax.legend(fontsize=6.5, frameon=False, loc="upper left")
        ax.set_xlabel("wavelength (nm)")
        # Without --normalise-isosbestic the panel is the well's own measurement:
        # sample_spectra only divided it by the scan peak, and multiplying back
        # restores the a.u. it was recorded in. With the normalisation the fit has
        # divided every well by its crossing value, so what the peak multiplies back
        # is a ratio, not the measured a.u. - and the axis says so.
        ax.set_ylabel(
            "Fluorescence (a.u.)" if iso_nm is None else "signal / well reference"
        )
    axf = axes[0][-1]
    # one observed fraction per well, across all of its scans: the per-scan
    # amplitude of a well is a free scale, taken out before the fraction is
    # read, so a well that is merely bright or dim in one scan is not mistaken
    # for a well that is protonated
    obs = observed_fraction(
        fit, {label: spectra[label] for label in scans}, is_ph=is_ph
    )
    if obs is not None:
        axf.plot(
            x,
            obs,
            "o",
            ms=3.2,
            mec="none",
            color="#2a6f97",
            alpha=0.8,
            label="observed, one per well (all scans)",
        )
    band = 1.96 * fit.se
    axf.fill_between(
        grid,
        curve(fit.K + band),
        curve(fit.K - band),
        color="#d1495b",
        alpha=0.18,
        lw=0,
        label=f"K {fit.K:.3f} +/- {band:.3f} (1.96 sigma, jackknife)",
    )
    axf.plot(grid, curve(fit.K), color="#d1495b", lw=1.6, label="fitted curve")
    if is_ph:
        axf.set_xlabel("pH")
    else:
        axf.set_xscale("log")
        axf.set_xlabel("titrant")
    axf.set_ylabel(frac_ylabel)
    axf.set_title(frac_title, fontsize=9.5, loc="left")
    axf.legend(fontsize=7, frameon=False)
    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=[axes[0][c] for c in range(len(scans))],
        fraction=0.02,
        pad=0.01,
        label="pH",
    )
    fig.suptitle(title, fontsize=10, x=0.01, ha="left")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


@dataclass(frozen=True)
class BayesPanelGroup:
    """One (sample, temperature) group of a whole-spectrum Bayesian fit.

    What :func:`plot_global_bayes_panels` needs in order to draw a group: the data
    the fit consumed, the titrant recorded in the note, and the posterior.
    Nothing is re-read from the note or the export here, so the figure cannot
    disagree with the fit about which wells or which scans were used.

    Attributes
    ----------
    label : str
        Group title, e.g. ``"NTT-B11 at 20.0"``.
    is_ph : bool
        pH titration (else a ligand one).
    wells : Sequence[str]
        Well labels, in the order the fit saw them.
    x_note : ArrayF
        Titrant value recorded in the note, one per well.
    spectra : Mapping[str, tuple[ArrayF, ArrayF]]
        Per scan label, ``(wavelengths, Y[n_lambda, n_wells])`` exactly as the fit
        consumed them (buffer-subtracted, and divided by the scan peak and by the
        isosbestic reference when those were asked for).
    trace : Any
        The ``arviz.InferenceData`` the fit returned.
    peaks : Mapping[str, float] | None
        The constant :func:`sample_spectra` divided each scan by, used to restore
        the scan's own units; ``None`` draws the panels as fitted.
    """

    label: str
    is_ph: bool
    wells: Sequence[str]
    x_note: ArrayF
    spectra: Mapping[str, tuple[ArrayF, ArrayF]]
    trace: Any
    peaks: Mapping[str, float] | None = None


def _draws(trace: Any, name: str) -> ArrayF:  # ruff: ignore[any-type]
    """Posterior draws of one variable, chains and draws flattened together.

    The two leading dimensions are dropped, so a scalar variable comes back as
    ``(n_draws,)`` and ``x_true`` as ``(n_draws, n_wells)``.
    """
    values = np.asarray(trace.posterior[name].values, dtype=np.float64)
    return values.reshape((-1, *values.shape[2:]))


def _band(
    draws: ArrayF, axis: int = 0, prob: float = 0.94
) -> tuple[ArrayF, ArrayF, ArrayF]:
    """Median and central ``prob`` interval of the draws along ``axis``.

    Returns ``(median, lo, hi)``, each with the reduced shape. The interval is
    the quantile pair, not an HDI: it is the one the console line and the K table
    already quote, so a panel and its table cannot disagree.
    """
    tail = (1.0 - prob) / 2.0

    def q(p: float) -> ArrayF:
        return np.asarray(np.quantile(draws, p, axis=axis), dtype=np.float64)

    return q(0.5), q(tail), q(1.0 - tail)


def _row_palette(rows: Sequence[str]) -> dict[str, str]:
    """Colour per plate row, Okabe-Ito, assigned in sorted label order.

    Row letters are what a plate makes comparable (three replicate ladders read
    together), so the panels colour by row rather than by well: a row that sits
    off the curve shows up as one colour leaving the band.
    """
    palette = (
        "#0072B2",
        "#E69F00",
        "#009E73",
        "#CC79A7",
        "#56B4E9",
        "#D55E00",
        "#F0E442",
        "#000000",
    )
    return {row: palette[i % len(palette)] for i, row in enumerate(sorted(set(rows)))}


def _titrant_grid(x: ArrayF, *, is_ph: bool, n: int = 200) -> ArrayF:
    """Axis to draw a titration curve on, padded past the wells on both ends.

    A pH axis is linear and padded by 0.2; a ligand axis spans two and a half
    decades around the data on a log scale, wide enough that a Kd below the
    lowest measured concentration still shows its curvature.
    """
    if is_ph:
        return np.linspace(float(np.nanmin(x)) - 0.2, float(np.nanmax(x)) + 0.2, n)
    positive = x[x > 0]
    low = max(float(np.nanmin(positive)) / 50.0, 1e-3)
    # a chloride ladder starts at a genuine zero, which a log axis cannot draw;
    # the panel uses a symlog axis (see _ligand_scale) and the curve starts at 0
    return np.concatenate([[0.0], np.geomspace(low, float(np.nanmax(x)) * 3.0, n)])


def _ligand_scale(ax: Any) -> None:  # ruff: ignore[any-type]
    """Log axis for a ligand titration that still shows the zero-concentration well.

    These ladders start at a real 0 mM, which a log axis silently drops - and that
    well is the one that pins the free plateau, so it must stay visible. symlog
    with a 1 mM linear region draws 0 at 0 and keeps the decades above it.
    """
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_xlim(left=-0.5)


def _fraction_legend(ax: Any, scans: Sequence[str]) -> None:  # ruff: ignore[any-type]
    """Legend of the fraction panel: rows as colours, marker shape as the readout.

    Filled markers are the fraction every scan shares; with more than one scan the
    open shapes name which scan gave that fraction on its own, so a separation
    between a filled and an open marker of the same colour says the scans disagree
    about that well.
    """
    from matplotlib.lines import Line2D  # ruff: ignore[import-outside-top-level]

    handles, labels = ax.get_legend_handles_labels()
    if len(scans) > 1:
        handles.append(Line2D([], [], marker="o", ls="none", color="0.3", ms=4))
        labels.append("all scans (shared)")
        for scan, marker in zip(scans, _SCAN_MARKERS, strict=False):
            handles.append(
                Line2D([], [], marker=marker, ls="none", mfc="none", mec="0.3", ms=4)
            )
            labels.append(f"{scan} only")
    ax.legend(handles, labels, fontsize=6.2, frameon=False, loc="upper left")


def _state_label(state: int, *, is_ph: bool) -> str:
    """Name of a species state as the model indexes it (0 = without the titrant)."""
    if is_ph:
        return "anionic (basic) state" if state == 0 else "protonated (acidic) state"
    return "free state" if state == 0 else "bound state"


def _fraction_ylabel(*, is_ph: bool) -> str:
    """Axis label for the fraction panels."""
    return (
        "fraction of the basic (anionic) state"
        if is_ph
        else "fraction of the bound state"
    )


def _fraction_title(stem: str, *, is_ph: bool, n_states: int) -> str:
    """Panel title, flagging the two-state curve when a third state was fitted."""
    kind = "basic (anionic)" if is_ph else "bound"
    suffix = (
        ""
        if n_states <= _TWO_STATES
        else " (two-state curve; a third state was fitted)"
    )
    return f"{stem}: {kind}{suffix}"


def _species_crossing(lam: ArrayF, basic: ArrayF, acid: ArrayF) -> float | None:
    """Wavelength [nm] where the two species spectra cross, or ``None``.

    The highest crossing above the chromophore region, i.e. the isosbestic
    point, as :func:`plot_global_panels` locates it. Interpolated linearly
    between the two bracketing points of the measured grid.
    """
    diff = np.asarray(basic, dtype=np.float64) - np.asarray(acid, dtype=np.float64)
    crossings = [
        float(lam[i] + (lam[i + 1] - lam[i]) * diff[i] / (diff[i] - diff[i + 1]))
        for i in range(lam.size - 1)
        if diff[i] * diff[i + 1] < 0 and lam[i] > _CHROMOPHORE_MIN
    ]
    return max(crossings) if crossings else None


def _trace_stat(trace: Any, what: str) -> float | None:  # ruff: ignore[any-type]
    """R-hat or bulk ESS of K from the trace, or ``None`` when arviz cannot say."""
    try:
        import arviz as az  # type: ignore[import-untyped]  # ruff: ignore[import-outside-top-level]

        # a degenerate (zero-variance) trace makes arviz divide by zero; that says
        # nothing a reader of the figure needs, and the seed keeps it reproducible
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            summary = (
                az.rhat(trace, var_names=["K"])
                if what == "rhat"
                else az.ess(trace, var_names=["K"])
            )
        return float(summary["K"].item())
    except (ImportError, KeyError, AttributeError, TypeError):
        return None


def plot_global_bayes_panels(
    path: Path,
    groups: Sequence[BayesPanelGroup],
) -> Path:
    """Draw the data, the posterior model and the per-well species fraction.

    One row per (sample, temperature) group, left to right:

    * one panel per scan - every well's measured spectrum under the
      posterior-median model, the two species spectra with their 94 % bands, and
      the wavelength at which the species cross;
    * **fraction per well** - the titration curve at the posterior-median K and
      one point per well at *that well's own* latent titrant value: the fraction
      with its 94 % interval vertically, the axis with its own 94 % interval
      horizontally, coloured by plate row. Which wells sit off the curve, and is
      it because their recorded axis moved? No correction is applied - the point
      is where the posterior puts the well - and the panel needs no separate
      "latent axis" view, because a well whose recorded axis was off shows as a
      bar that does not cover the curve, and the value recorded in the note is a
      tick at the same fraction, with a hairline joining it to the marker: the
      length of that hairline is the axis correction the posterior asked for.
      When the fit used more than one scan,
      the filled markers are the fraction all scans share (the per-scan amplitude
      is free, so a bright well is not a protonated one) and the open ones - one
      shape per scan - are the fraction each scan gives on its own: where they
      separate, the shared model is being asked for a fraction the scans do not
      agree on;
    * **K posterior** - histogram with median, 94 % interval, R-hat and bulk ESS.

    Parameters
    ----------
    path : Path
        Where to write the figure (``.pdf`` or ``.png``, by suffix).
    groups : Sequence[BayesPanelGroup]
        One entry per (sample, temperature) group, in the order to draw them.

    Returns
    -------
    Path
        The path of the figure that was written; ``path`` unchanged when there
        was nothing to draw.
    """
    import matplotlib.pyplot as plt  # ruff: ignore[import-outside-top-level]

    n_scans = max((len(group.spectra) for group in groups), default=0)
    if not groups or n_scans == 0:
        return path
    ncols = n_scans + 2
    fig, axes = plt.subplots(
        len(groups),
        ncols,
        figsize=(3.9 * ncols, 4.3 * len(groups)),
        squeeze=False,
        constrained_layout=True,
        # the K posterior is a histogram, not a titration: it gets a narrower
        # column and is drawn square (set_box_aspect below), so it reads as an
        # inset next to the panels that carry the titration
        gridspec_kw={"width_ratios": [*([1.0] * n_scans), 1.0, 0.7]},
    )
    for row, group in enumerate(groups):
        _draw_group(axes[row], group, n_scans=n_scans)
    fig.suptitle("whole-spectrum Bayesian fit", fontsize=10, x=0.01, ha="left")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _observed_with_interval(
    group: BayesPanelGroup,
    e_draws: Mapping[str, Mapping[int, ArrayF]],
    *,
    is_ph: bool,
    scans: Sequence[str] | None = None,
) -> tuple[ArrayF, ArrayF, ArrayF]:
    """Observed fraction of every well, with the 94 % interval of its own unmixing.

    Each well's spectra are unmixed against a thinned set of posterior species
    draws (:data:`_OBS_DRAWS`), so the marker carries the uncertainty the species
    spectra bring, not just the posterior of the axis. ``scans`` restricts the
    unmixing to a subset: left out, every scan of the group votes for one shared
    fraction, and the per-scan amplitude absorbs each scan's own scale; given one
    scan, that scan alone fixes the fraction, which is the independent estimate
    the shared one can be compared with.

    Returns ``(median, lo, hi)`` per well; all ``nan`` when no scan matches.
    """
    use = list(e_draws) if scans is None else [s for s in scans if s in e_draws]
    if not use:
        empty = np.full(len(group.wells), np.nan)
        return empty, empty, empty
    first = e_draws[use[0]]
    total = int(np.asarray(first[0]).shape[0])
    pick = np.linspace(0, total - 1, min(_OBS_DRAWS, total)).astype(int)
    spectra = {scan: group.spectra[scan] for scan in use}
    per_draw: list[ArrayF] = []
    for i in pick.tolist():
        species = {
            scan: np.stack([e_draws[scan][0][i], e_draws[scan][1][i]]) for scan in use
        }
        observed = observed_fraction_from_species(species, spectra, is_ph=is_ph)
        if observed is None:
            break
        per_draw.append(np.asarray(observed, dtype=np.float64))
    if not per_draw:
        empty = np.full(len(group.wells), np.nan)
        return empty, empty, empty
    return _band(np.asarray(per_draw), axis=0)


def _draw_group(  # ruff: ignore[complex-structure, too-many-branches, too-many-statements]
    ax_row: Sequence[Any],
    group: BayesPanelGroup,
    *,
    n_scans: int,
) -> None:
    """Fill one group's row of panels: scans, two fractions, latent axis, K."""
    import matplotlib.pyplot as plt  # ruff: ignore[import-outside-top-level]
    from matplotlib import colormaps  # ruff: ignore[import-outside-top-level]
    from matplotlib.colors import Normalize  # ruff: ignore[import-outside-top-level]

    is_ph = bool(group.is_ph)
    xname = "pH" if is_ph else "titrant"
    posterior = group.trace.posterior
    n_wells = len(group.wells)
    k_draws = _draws(group.trace, "K")
    x_draws = _draws(group.trace, "x_true")
    x_med, x_lo, x_hi = _band(x_draws, axis=0)
    amp_med = (
        np.median(_draws(group.trace, "amplitude"), axis=0)
        if "amplitude" in posterior
        else np.ones(n_wells)
    )
    scans = list(group.spectra)
    first_scan = scans[0]
    n_states = _TWO_STATES + (1 if f"E2_{first_scan}" in posterior else 0)
    e_med: dict[str, dict[int, ArrayF]] = {}
    e_lo: dict[str, dict[int, ArrayF]] = {}
    e_hi: dict[str, dict[int, ArrayF]] = {}
    e_draws: dict[str, dict[int, ArrayF]] = {}
    for scan in group.spectra:
        for state in range(n_states):
            draws = _draws(group.trace, f"E{state}_{scan}")
            e_draws.setdefault(scan, {})[state] = draws
            med, lo, hi = _band(draws)
            e_med.setdefault(scan, {})[state] = med
            e_lo.setdefault(scan, {})[state] = lo
            e_hi.setdefault(scan, {})[state] = hi
    rows = [str(well)[0] for well in group.wells]
    palette = _row_palette(rows)
    cmap = colormaps["coolwarm_r"]
    norm = Normalize(float(np.nanmin(x_med)), float(np.nanmax(x_med)))
    k_med = float(np.median(k_draws))
    # observed fraction of every well: its own spectra unmixed against a thinned
    # set of posterior species draws, so the marker carries the unmixing's own
    # uncertainty instead of pretending the species are known exactly
    f_obs_med, f_obs_lo, f_obs_hi = _observed_with_interval(group, e_draws, is_ph=is_ph)
    f_grow_med = growth_fraction(x_med, k_med, is_ph=is_ph)
    if n_states > _TWO_STATES:
        delta_med = float(np.median(_draws(group.trace, "delta")))
        fractions = _fractions(
            x_med,
            np.array([k_med, delta_med]),
            _Model(acid_state=True, fit_hill=False, well_scale=True),
            is_ph=is_ph,
        )
    else:
        # state 1 is occupied (protonated on pH, bound on ligand)
        occupied = 1.0 - f_grow_med if is_ph else f_grow_med
        fractions = np.column_stack([1.0 - occupied, occupied])
    grid = _titrant_grid(x_med, is_ph=is_ph)
    c_med, c_lo, c_hi = _band(
        growth_fraction(grid[None, :], k_draws[:, None], is_ph=is_ph), axis=0
    )

    # --- one panel per scan: data, posterior-median model, species ------------
    for col, (scan, (lam, values)) in enumerate(group.spectra.items()):
        ax = ax_row[col]
        wavelengths = np.asarray(lam, dtype=np.float64)
        meas = np.asarray(values, dtype=np.float64)
        model = amp_med[None, :] * sum(
            fractions[:, state][None, :] * e_med[scan][state][:, None]
            for state in range(n_states)
        )
        m = (
            float(group.peaks[scan])
            if group.peaks is not None and scan in group.peaks
            else 1.0
        )
        for i in range(meas.shape[1]):
            ax.plot(
                wavelengths,
                m * meas[:, i],
                color=cmap(norm(x_med[i])),
                lw=0.8,
                alpha=0.4,
            )
            ax.plot(wavelengths, m * model[:, i], color="k", lw=0.5, ls="--", alpha=0.3)
        for state, colour in ((0, "#2a6f97"), (1, "#d1495b")):
            ax.plot(
                wavelengths,
                m * e_med[scan][state],
                color=colour,
                lw=1.8,
                label=_state_label(state, is_ph=is_ph),
            )
            ax.fill_between(
                wavelengths,
                m * e_lo[scan][state],
                m * e_hi[scan][state],
                color=colour,
                alpha=0.15,
                lw=0,
            )
        crossing = _species_crossing(wavelengths, e_med[scan][0], e_med[scan][1])
        if crossing is not None:
            ax.axvline(crossing, color="k", lw=0.7, ls=":")
            ax.annotate(
                f"{crossing:.1f} nm",
                (crossing, 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(3, -10),
                textcoords="offset points",
                fontsize=6.5,
                rotation=90,
                va="top",
            )
        ax.set_title(f"{scan}: {n_wells} wells", fontsize=9, loc="left")
        ax.set_xlabel("wavelength (nm)")
        ax.set_ylabel("signal" if group.peaks is None else "Fluorescence (a.u.)")
        ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=list(ax_row[:n_scans]),
        fraction=0.02,
        pad=0.01,
        label=f"latent {xname}",
    )

    # --- fraction per well: shared across the scans, and each scan on its own --
    ax = ax_row[n_scans]
    ax.fill_between(grid, c_lo, c_hi, color="#d1495b", alpha=0.12, lw=0)
    ax.plot(grid, c_med, color="#d1495b", lw=1.4, label="common curve, median K")
    # one fraction per well from all scans together: the per-scan amplitude is a
    # free scale, so a well that is bright in one scan is not mistaken for a
    # protonated one
    for row_label in sorted(set(rows)):
        pick = np.array([r == row_label for r in rows])
        ax.errorbar(
            x_med[pick],
            f_obs_med[pick],
            xerr=np.vstack([(x_med - x_lo)[pick], (x_hi - x_med)[pick]]),
            yerr=np.vstack([
                (f_obs_med - f_obs_lo)[pick],
                (f_obs_hi - f_obs_med)[pick],
            ]),
            fmt="o",
            ms=3.6,
            capsize=0,
            lw=0.7,
            color=palette[row_label],
            alpha=0.9,
            label=f"row {row_label}",
        )
    # the value recorded in the note, as a tick at the same fraction: the gap
    # between the tick and the marker is the axis correction the posterior asked
    # for, drawn rather than applied
    note_x = np.asarray(group.x_note, dtype=np.float64)
    ax.plot(
        note_x,
        f_obs_med,
        "|",
        ms=4.6,
        mew=0.8,
        ls="none",
        color="0.45",
        label=f"note {xname}",
    )
    for i in range(n_wells):
        ax.plot(
            [note_x[i], x_med[i]],
            [f_obs_med[i], f_obs_med[i]],
            color="0.75",
            lw=0.5,
            zorder=0,
        )
    # with more than one scan, every scan can also be unmixed alone: where two
    # scans disagree about a well the model is being asked to share a fraction
    # that the data do not, and these markers are where that shows
    for scan, marker in zip(scans, _SCAN_MARKERS, strict=False):
        med, _lo, _hi = _observed_with_interval(
            group, e_draws, is_ph=is_ph, scans=[scan]
        )
        for row_label in sorted(set(rows)):
            pick = np.array([r == row_label for r in rows])
            ax.plot(
                x_med[pick],
                med[pick],
                marker,
                ms=3.2,
                mfc="none",
                mew=0.7,
                ls="none",
                color=palette[row_label],
                alpha=0.9,
            )
    off = np.abs(f_obs_med - f_grow_med) > _OFF_CURVE
    for i in np.flatnonzero(off).tolist():
        ax.annotate(
            str(group.wells[i]),
            (float(x_med[i]), float(f_obs_med[i])),
            textcoords="offset points",
            xytext=(3, 3),
            fontsize=5.5,
            color=palette[rows[i]],
        )
    ax.set_title(
        _fraction_title("fraction per well", is_ph=is_ph, n_states=n_states)
        + f"; {int(off.sum())} off the curve",
        fontsize=9,
        loc="left",
    )
    ax.set_xlabel(f"latent {xname} (94 % interval)")
    ax.set_ylabel(_fraction_ylabel(is_ph=is_ph))
    ax.set_ylim(-0.03, 1.03)
    if not is_ph:
        _ligand_scale(ax)
    _fraction_legend(ax, scans)

    # --- K posterior

    # --- K posterior ----------------------------------------------------------
    ax = ax_row[n_scans + 1]
    k_lo, k_hi = (float(v) for v in _band(k_draws)[1:])
    rhat = _trace_stat(group.trace, "rhat")
    ess = _trace_stat(group.trace, "ess")
    ax.hist(k_draws, bins=35, color="0.72")
    ax.axvline(k_med, color="black", lw=1)
    ax.axvspan(k_lo, k_hi, color="orange", alpha=0.25)
    title = f"K {k_med:.3f} [{k_lo:.3f}, {k_hi:.3f}]"
    if rhat is not None and ess is not None:
        title += f"\nR-hat {rhat:.3f}, ESS {ess:.0f}"
    ax.set_title(title, fontsize=9, loc="left")
    ax.set_xlabel("pKa" if is_ph else "Kd")
    ax.set_ylabel("draw count")
    ax.set_box_aspect(1)
    ax.set_anchor("C")  # square and centred in its (narrower) column
