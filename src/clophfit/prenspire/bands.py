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
the buffer well of the same plate column is subtracted, and each band is
divided by the tryptophan band (330-342 nm of the 278 nm-excited scan), which
measures the protein actually in the well rather than any titration state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from clophfit.fitting.core import fit_binding_glob
from clophfit.fitting.data_structures import DataArray, Dataset
from clophfit.fitting.models import binding_1site

if TYPE_CHECKING:
    from collections.abc import Sequence

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

# the tryptophan window reports protein, not titration state
TRP = (330.0, 342.0)
EMISSION = (500.0, 520.0)
ANIONIC = (480.0, 495.0)
NEUTRAL = (395.0, 410.0)
# The two readouts that agree best between sessions. Scored over 39
# sample-temperatures and five constructs measured twice or more: this pair
# lands 0.14 pH apart between sessions against 0.25 for the pair plus the
# neutral excitation band, 0.29 for the anionic band alone, 0.31 for the SVD of
# the excitation label and 0.38 for the SVD of all labels concatenated. Within
# one plate the arms are indistinguishable (0.21-0.27 pH between replicate
# rows), so it is reproducibility across sessions that separates them.
DIRECT_BANDS = ("exc_anionic", "em_exc420")
# point screen handed to fit_binding_glob unless asked otherwise
DEFAULT_SCREEN = "studentized:0.05:5"
_PROTEIN_EXCITATION = 350.0
_BLANK_FRACTION = 0.05
_MIN_ROW_POINTS = 5
_MIN_RANK_POINTS = 4
_MIN_DISTINCT = 3
# where a zero-concentration point is drawn on a log axis
_ZERO_AT = 0.7


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
    ef: EnspireFile, temp: str | float | None = None
) -> tuple[list[Readout], Readout | None]:
    """Split a temperature's labels into titrating bands and the protein reference.

    Each label declares its monochromator and the wavelength it holds fixed, so
    which windows it can offer is read from the file rather than configured: an
    excitation scan carries the anionic and neutral bands, an emission scan
    excited below 350 nm carries both the tryptophan reference and the emission
    band, and any other emission scan carries the emission band alone.

    Parameters
    ----------
    ef : EnspireFile
        The parsed EnSpire export.
    temp : str | float | None
        Keep only labels measured at this temperature; None keeps every label.

    Returns
    -------
    tuple[list[Readout], Readout | None]
        The bands available, and the tryptophan reference if one was measured.
    """
    wanted = None if temp is None or str(temp) in {"nan", "None", ""} else str(temp)
    readouts: list[Readout] = []
    reference: Readout | None = None
    for label, meas in ef.measurements.items():
        md = meas["metadata"]
        if (
            wanted is not None
            and str(md.get("temp")).split(".")[0] != wanted.split(".")[0]
        ):
            continue
        if md["Monochromator"] == "Excitation":
            readouts += [
                Readout("exc_anionic", label, *ANIONIC),
                Readout("exc_neutral", label, *NEUTRAL),
            ]
        elif float(md["Wavelength"]) < _PROTEIN_EXCITATION:
            reference = Readout("trp", label, *TRP)
            readouts.append(Readout("em_exc278", label, *EMISSION))
        else:
            readouts.append(Readout("em_exc420", label, *EMISSION))
    return readouts, reference


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
        The fit, or None when it does not converge.
    """
    if not data:
        return None
    ds = Dataset({name: DataArray(x, y) for name, (x, y) in data.items()}, is_ph=is_ph)
    result = fit_binding_glob(ds, remove_outliers=screen)
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


def _sample_bands(  # ruff: ignore[too-many-arguments]
    ef: EnspireFile,
    wells: list[str],
    x: np.ndarray,
    temp: str | float,
    *,
    normalise: bool,
    buffer_wells: Sequence[str],
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], list[str], np.ndarray]:
    """Band values of one sample, buffer-subtracted and reference-normalised.

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
    normalise : bool
        Divide each band by the tryptophan band of the same well.
    buffer_wells : Sequence[str]
        Buffer wells declared by the note, if any.

    Returns
    -------
    tuple[dict[str, tuple[np.ndarray, np.ndarray]], list[str], np.ndarray]
        Band name to (x, y), the wells actually used, and their x values.
    """
    readouts, reference = classify(ef, temp)
    if not readouts:
        return {}, wells, x
    probe = band_values(ef, readouts[0], wells)
    blanks = blank_rows(probe)
    if blanks:
        found = [w for w in wells if w[0] in blanks]
        keep = [i for i, w in enumerate(wells) if w[0] not in blanks]
        wells, x = [wells[i] for i in keep], x[keep]
        buffer_wells = list(buffer_wells) or found
    ref = band_values(ef, reference, wells) if (normalise and reference) else None
    data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for readout in readouts:
        y = band_values(ef, readout, wells)
        if buffer_wells:
            y = subtract_buffer(y, band_values(ef, readout, buffer_wells))
        if ref is not None:
            y /= ref
        top = float(np.abs(y).max())
        data[readout.name] = (x, (y / top if top else y).to_numpy())
    return data, wells, x


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


def fit_titrations(  # ruff: ignore[too-many-arguments]
    ef: EnspireFile,
    note: pd.DataFrame,
    *,
    normalise: bool = True,
    buffer: bool = True,
    wanted: Sequence[str] = DIRECT_BANDS,
    min_rho: float = 0.8,
    screen: str | None = DEFAULT_SCREEN,
) -> TitrationFits:
    """Fit every sample of a note from its bands.

    Parameters
    ----------
    ef : EnspireFile
        The parsed EnSpire export.
    note : pd.DataFrame
        Normalised note, with ``Well``, ``pH``, ``Cl``, ``Name`` and ``Temp``.
    normalise : bool
        Divide each band by the tryptophan band of the same well.
    buffer : bool
        Subtract the buffer well of the same plate column.
    wanted : Sequence[str]
        Bands allowed into the shared K; the others are reported only.
    min_rho : float
        Monotonicity gate for a band.
    screen : str | None
        Point screen handed to :func:`fit_binding_glob`.

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
    for (name, temp), group in samples.groupby(
        ["Name", "Temp"], sort=True, dropna=False
    ):
        one = group.drop_duplicates("Well")
        is_ph = bool(one["pH"].nunique() > one["Cl"].nunique())
        wells = one["Well"].tolist()
        x = one["pH" if is_ph else "Cl"].astype(float).to_numpy()
        buf = buffers[buffers["Temp"] == temp]["Well"].tolist() if buffer else []
        data, wells, x = _sample_bands(
            ef, wells, x, str(temp), normalise=normalise, buffer_wells=buf
        )
        if not data:
            continue
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
        grid = np.logspace(np.log10(_ZERO_AT), np.log10(max(xs.max(), 1.0) * 1.5), 200)
    for (name, (x, y)), color in zip(data.items(), _COLORS, strict=False):
        px = x if is_ph else np.where(x == 0, _ZERO_AT, x)
        ax.scatter(px, y, s=18, color=color, lw=0, zorder=3, label=name)
        s0, s1 = fit.plateaus[name]
        gx = grid if is_ph else np.where(grid <= _ZERO_AT, 0.0, grid)
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
    ax.set_xlabel("pH" if is_ph else f"[Cl$^-$] (mM), 0 drawn at {_ZERO_AT}")
    ax.set_ylabel("band / reference")
    label = "pK$_\\mathrm{a}$" if is_ph else "$K_\\mathrm{d}$"
    ax.set_title(
        f"{title} — {label} {fit.k:.2f} [{fit.lo:.2f}, {fit.hi:.2f}]",
        loc="left",
        fontsize=10.5,
    )
    ax.legend(frameon=False, fontsize=8.5, loc="best")
