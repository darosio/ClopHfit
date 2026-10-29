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

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from clophfit.fitting.spectral import SpectralFit, fit_spectra_global
from clophfit.prenspire.bands import PH_ACID_END, band_values, blank_rows, classify

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from clophfit.clophfit_types import ArrayF
    from clophfit.prenspire.prenspire import EnspireFile

_MIN_ROW_WELLS = 5
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
) -> tuple[dict[str, tuple[ArrayF, ArrayF]], list[str], ArrayF]:
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

    Returns
    -------
    tuple[dict[str, tuple[ArrayF, ArrayF]], list[str], ArrayF]
        Scan name to (wavelengths, Y[n_lambda, n_wells]), the wells used and
        their x values. The key is the scan's name (see :func:`scan_name`),
        not its plate label, so the same measurement keeps one name at
        every temperature.
    """
    readouts = classify(ef, temp, is_ph=is_ph)
    labels = list(dict.fromkeys(r.label for r in readouts if r.name in bands))
    if not labels:
        return {}, wells, x
    blanks = blank_rows(band_values(ef, readouts[0], wells))
    if blanks:
        found = [w for w in wells if w[0] in blanks]
        keep = [i for i, w in enumerate(wells) if w[0] not in blanks]
        wells, x = [wells[i] for i in keep], x[keep]
        buffer_wells = list(buffer_wells) or found
    out: dict[str, tuple[ArrayF, ArrayF]] = {}
    for label in labels:
        lam, y = scan_spectrum(ef, label, wells, buffer_wells)
        out[scan_name(ef, label)] = (lam, y / float(np.nanmax(np.abs(y))))
    if isosbestic is not None:
        lam_iso = (
            plate_isosbestic(out, x, x_min=PH_ACID_END if is_ph else None)
            if isinstance(isosbestic, str)
            else float(isosbestic)
        )
        if lam_iso is not None:
            key = next(
                (k for k in out if k == "excitation" or k.startswith("excitation ")),
                next(iter(out)),
            )
            out = normalise_to_isosbestic(out, lam_iso, scan=key)
    return out, wells, x


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


def plate_isosbestic(
    spectra: Mapping[str, tuple[Any, Any]],
    x: ArrayF,
    *,
    scans: Sequence[str] = EXCITATION_SCAN_NAMES,
    x_min: float | None = None,
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

    Returns
    -------
    float | None
        The crossing [nm], or ``None`` when no chosen scan has one.
    """
    chosen = {
        label: value
        for label, value in spectra.items()
        if any(label == name or label.startswith(f"{name} ") for name in scans)
    }
    if not chosen:
        return None
    found = isosbestic_point(chosen, x, x_min=x_min)
    if not found:
        return None
    return float(np.median(list(found.values())))


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
        spectra, _used, used_x = sample_spectra(
            ef, wells, x, str(temp), is_ph=is_ph, buffer_wells=bufw
        )
        out[str(name), float(str(temp))] = (
            plate_isosbestic(spectra, used_x) if spectra else None
        )
    return out


def isosbestic_scale(
    spectra: Mapping[str, tuple[Any, Any]],
    lam_iso: float,
    *,
    scan: str | None = None,
    floor: float = 0.02,
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
    ref = np.array([np.interp(lam_iso, lam_a, ya[:, i]) for i in range(ya.shape[1])])
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
        scale, _usable = isosbestic_scale(spectra, lam_iso, scan=scan)
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
        ref = np.array([
            np.interp(lam_iso, lam_a, ya[:, i]) for i in range(ya.shape[1])
        ])
        floor = 0.02 * np.max(np.abs(ya), axis=0)
        usable = np.isfinite(ref) & (np.abs(ref) > floor)
        scaled = ya.copy()
        scaled[:, usable] /= ref[usable]
        out[label] = (lam_a, scaled)
    return out


def fit_titrations_spectral(  # ruff: ignore[too-many-arguments]
    ef: EnspireFile,
    note: pd.DataFrame,
    *,
    buffer: bool = True,
    well_scale: bool = True,
    fit_hill: bool = False,
    screen: float | None = None,
    bands: Sequence[str] = EXCITATION_SCAN,
    isosbestic: float | str | None = None,
    n_boot: int = 0,
    seed: int = 0,
    strata: str = "none",
    qc: float | None = None,
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

    Returns
    -------
    tuple[pd.DataFrame, dict[str, SpectralFit]]
        One row per (sample, temperature, subset) - ``all`` then each replicate row - and, per
        ``"<sample> <temp>"``, the fit on all wells with its species spectra.
    """
    rows: list[dict[str, object]] = []
    fits: dict[str, SpectralFit] = {}
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
        spectra, wells, x = sample_spectra(
            ef,
            wells,
            x,
            str(temp),
            is_ph=is_ph,
            buffer_wells=buf,
            bands=bands,
            isosbestic=lam,
        )
        if not spectra:
            continue
        dropped: list[str] = []
        if qc is not None:
            qc_table = scan_outliers(spectra, wells, x, z=qc, min_signal=0.05)
            if not qc_table.empty:
                dropped = sorted(set(qc_table.loc[qc_table["flagged"], "well"]))
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
        }
        design = [str(w)[1:] for w in wells] if strata == "column" else None
        fit = fit_spectra_global(
            x,
            spectra,
            is_ph=is_ph,
            fit_hill=fit_hill,
            well_scale=well_scale,
            screen=screen,
            n_boot=n_boot,
            seed=seed,
            strata=design,
        )
        fits[f"{name} {temp}"] = fit
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
                jackknife=False,
            )
            rows.append(_row(base, f"row {plate_row}", row_fit))

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
