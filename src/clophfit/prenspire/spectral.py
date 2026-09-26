"""Whole-spectrum global K for EnSpire titrations (see :mod:`clophfit.fitting.spectral`).

The wells get the same corrections as the band readouts of :mod:`clophfit.prenspire.bands` - blank
plate rows removed, the buffer spectrum of the same plate column subtracted, each well divided by its
tryptophan band - so that the two methods differ only in the readout: named windows sharing K there,
every wavelength of the chosen scans here.

Scored with arslanbaeva's ``scripts/score_readout_arms.py`` metrics on the 25 pH plates it ranks
(same exclusions; arslanbaeva's ``scripts/score_spectral_arms.py``): the whole excitation scan with a per-well
amplitude has the smallest between-session spread of every readout (median 0.10 pH, worst 0.16,
against 0.14 and 0.22 for the default band pair) at the same replicate-row spread (0.23). Adding the
420 nm emission scan whole does not help (0.14, worst 0.37), and without the well amplitude it is
worse (0.21, worst 0.80). Only five constructs are repeated across sessions, so the ranking of the
excitation scan against the band pair is suggestive, not settled.

The tryptophan reference (330-342 nm of the 278 nm-excited scan) does not depend on pH (Spearman
rho 0.02 and -0.06 on G10) and neither causes nor removes the acid-side amplitude drop; it scatters
by 15-30 % between wells, more than the fitted amplitudes do (about 5 %), so with ``well_scale`` it
adds noise without changing K (7.972 against 7.970 unnormalised on G10).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from clophfit.fitting.spectral import SpectralFit, fit_spectra_global
from clophfit.prenspire.bands import band_values, blank_rows, classify

if TYPE_CHECKING:
    from collections.abc import Sequence

    from clophfit.clophfit_types import ArrayF
    from clophfit.prenspire.prenspire import EnspireFile

_MIN_ROW_WELLS = 5
# The excitation scan alone: on the readout-arm sweep (module docstring) adding the 420 nm emission
# scan whole widened the between-session spread (median 0.10 -> 0.14 pH, worst 0.16 -> 0.37).
EXCITATION_SCAN = ("exc_anionic",)


def _spectra(
    ef: EnspireFile, label: str, wells: Sequence[str]
) -> tuple[ArrayF, ArrayF]:
    meas = ef.measurements[label]
    lam = np.asarray(meas["lambda"], dtype=np.float64)
    y = np.column_stack([np.asarray(meas[w], dtype=np.float64) for w in wells])
    return lam, y


def sample_spectra(  # ruff: ignore[too-many-arguments]
    ef: EnspireFile,
    wells: list[str],
    x: ArrayF,
    temp: str | float,
    *,
    is_ph: bool,
    normalise: bool,
    buffer_wells: Sequence[str],
    bands: Sequence[str] = EXCITATION_SCAN,
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
    normalise : bool
        Divide each well by its tryptophan band.
    buffer_wells : Sequence[str]
        Buffer wells declared by the note; blank rows serve when there are none.
    bands : Sequence[str]
        Band names (as in :func:`~clophfit.prenspire.bands.classify`) whose scans are used whole;
        the default takes the excitation scan only.

    Returns
    -------
    tuple[dict[str, tuple[ArrayF, ArrayF]], list[str], ArrayF]
        Label to (wavelengths, Y[n_lambda, n_wells]), the wells used and their x values.
    """
    readouts, reference = classify(ef, temp, is_ph=is_ph)
    labels = list(dict.fromkeys(r.label for r in readouts if r.name in bands))
    if not labels:
        return {}, wells, x
    blanks = blank_rows(band_values(ef, readouts[0], wells))
    if blanks:
        found = [w for w in wells if w[0] in blanks]
        keep = [i for i, w in enumerate(wells) if w[0] not in blanks]
        wells, x = [wells[i] for i in keep], x[keep]
        buffer_wells = list(buffer_wells) or found
    ref = (
        band_values(ef, reference, wells).to_numpy()
        if (normalise and reference)
        else None
    )
    out: dict[str, tuple[ArrayF, ArrayF]] = {}
    for label in labels:
        lam, y = _spectra(ef, label, wells)
        if buffer_wells:
            _, yb = _spectra(ef, label, list(buffer_wells))
            by_column = {str(w)[1:]: yb[:, i] for i, w in enumerate(buffer_wells)}
            zero = np.zeros(lam.size)
            y -= np.column_stack([by_column.get(str(w)[1:], zero) for w in wells])
        if ref is not None:
            y /= ref
        out[label] = (lam, y / float(np.nanmax(np.abs(y))))
    return out, wells, x


def _row(base: dict[str, object], subset: str, fit: SpectralFit) -> dict[str, object]:
    return {
        **base,
        "subset": subset,
        "K": fit.K,
        "se": fit.se,
        "se_naive": fit.se_naive,
        "hill": fit.hill,
        "n": fit.n_wells,
        "sv_ratio": float(fit.residual_sv[0] / fit.residual_sv[1]),
    }


def fit_titrations_spectral(  # ruff: ignore[too-many-arguments]
    ef: EnspireFile,
    note: pd.DataFrame,
    *,
    normalise: bool = True,
    buffer: bool = True,
    well_scale: bool = True,
    fit_hill: bool = False,
    bands: Sequence[str] = EXCITATION_SCAN,
) -> tuple[pd.DataFrame, dict[str, SpectralFit]]:
    """Fit every sample of a note with the whole-spectrum global model.

    Parameters
    ----------
    ef : EnspireFile
        The parsed EnSpire export.
    note : pd.DataFrame
        Normalised note, with ``Well``, ``pH``, ``Cl``, ``Name`` and ``Temp``.
    normalise : bool
        Divide each well by its tryptophan band.
    buffer : bool
        Subtract the buffer well of the same plate column.
    well_scale : bool
        Fit a per-well amplitude as well (see :func:`fit_spectra_global`).
    fit_hill : bool
        Also fit the Hill slope.
    bands : Sequence[str]
        Band names whose scans are used whole (default: the excitation scan only).

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
    for (name, temp), group in samples.groupby(
        ["Name", "Temp"], sort=True, dropna=False
    ):
        one = group.drop_duplicates("Well")
        is_ph = bool(one["pH"].nunique() > one["Cl"].nunique())
        wells = one["Well"].tolist()
        x = np.asarray(one["pH" if is_ph else "Cl"], dtype=np.float64)
        buf = buffers[buffers["Temp"] == temp]["Well"].tolist() if buffer else []
        spectra, wells, x = sample_spectra(
            ef,
            wells,
            x,
            str(temp),
            is_ph=is_ph,
            normalise=normalise,
            buffer_wells=buf,
            bands=bands,
        )
        if not spectra:
            continue
        base: dict[str, object] = {"sample": name, "temp": temp}
        opts = {"is_ph": is_ph, "fit_hill": fit_hill, "well_scale": well_scale}
        fit = fit_spectra_global(x, spectra, **opts)
        fits[f"{name} {temp}"] = fit
        rows.append(_row(base, "all", fit))
        for plate_row in sorted({w[0] for w in wells}):
            sel = np.array([w[0] == plate_row for w in wells])
            if sel.sum() < _MIN_ROW_WELLS:
                continue
            sub = {lab: (lam, y[:, sel]) for lab, (lam, y) in spectra.items()}
            row_fit = fit_spectra_global(x[sel], sub, jackknife=False, **opts)
            rows.append(_row(base, f"row {plate_row}", row_fit))
    return pd.DataFrame(rows), fits
