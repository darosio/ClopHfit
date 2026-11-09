"""Panel figures for the whole-spectrum fits: the data, the model and the species fraction."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from clophfit.fitting.models import growth_fraction
from clophfit.fitting.spectral import (
    SpectralFit,
    _fractions,
    _Model,
    observed_fraction,
    observed_fraction_from_species,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from clophfit.clophfit_types import ArrayF


#: Above this titrant value the axis is a concentration, not a pH; used only to
#: label a panel, never to fit.
_MAX_PH = 20.0
#: Shortest wavelength the crossing search in a panel accepts [nm]; the protein
#: band below it does not titrate and would supply a crossing of its own.
_CHROMOPHORE_MIN = 400.0
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
