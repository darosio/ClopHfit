"""Export data and fit results from Titration objects."""

import copy
import itertools
import logging
import typing
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import arviz as az  # type: ignore[import-untyped]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lmfit import Parameters  # type: ignore[import-untyped]
from scipy import stats as sp_stats

from clophfit.clophfit_types import ArrayF
from clophfit.fitting.bayes import (
    _DEFAULT_NOISE,
    dataset_with_unit_yerr,
    fit_binding_pymc,
    fit_binding_pymc_multi,
)
from clophfit.fitting.bayes_config import NoiseConfig, RobustConfig, SamplerConfig
from clophfit.fitting.data_structures import (
    Dataset,
    FitResult,
    NoiseModelParams,
    PlateNoiseModel,
)
from clophfit.fitting.diagnostics import screen_wells
from clophfit.fitting.gain_calibration import (
    GainCalibration,
    calibrate_plate_lm,
    calibrate_single_well,
)
from clophfit.fitting.model_validation import (
    OutlierCriterion,
    ResidualTail,
    apply_exclusions,
    mark_outliers,
    residuals_from_fit_results,
    robust_likelihood_from_trace,
    robust_settings_from_trace,
)
from clophfit.fitting.models import KD_MIN, binding_1site, kd_bounds
from clophfit.fitting.plate_lm import (
    PlateLMResult,
    apply_excluded_points,
    ctr_holdout,
    fit_plate_lm,
    fit_plate_lm_screened,
)
from clophfit.fitting.plate_odr import (
    PlateODRResult,
    ctr_holdout_odr,
    fit_plate_odr,
)
from clophfit.fitting.plotting import PlotParameters, plot_fit
from clophfit.fitting.residual_tests import residual_tests
from clophfit.fitting.residuals import (
    plot_residual_distribution,
    plot_residual_vs_predicted,
    plot_residual_vs_yerr,
    residual_statistics,
)
from clophfit.prtecan.titration import (
    McmcSpec,
    TecanConfig,
    Titration,
    TitrationResults,
)

logger = logging.getLogger(__name__)


def generate_combinations() -> list[tuple[tuple[bool, ...], str]]:
    """Generate parameter combinations for export and fitting."""
    bool_iter = itertools.product([False, True], repeat=4)
    return [
        (tuple(bool_combo), method)
        for bool_combo in bool_iter
        for method in ["mean", "meansd", "fit"]
    ]


def apply_combination(
    titration: Titration, combination: tuple[tuple[bool, ...], str]
) -> None:
    """Apply a combination of parameters to the Titration."""
    (bg, adj, dil, nrm), method = combination
    logger.info("Params are: ........... %s", ((bg, adj, dil, nrm), method))
    titration.params.bg = bg
    titration.params.bg_adj = adj
    titration.params.dil = dil
    titration.params.nrm = nrm
    titration.params.bg_mth = method


# Output-folder suffix per buffer method. `mean`, `fit` and `meansd` keep the
# names they have always had; anything new gets its own.
_BG_MTH_SUFFIX = {
    "mean": "",
    "fit": "_fit",
    "meansd": "_1sd",
    "median": "_med",
    "mediansd": "_med1sd",
}


def prepare_output_folder(titration: Titration, base_path: Path) -> Path:
    """Prepare the output folder for a given combination of parameters."""
    p = titration.params
    sbg = "_bg" if p.bg else ""
    sadj = "_adj" if p.bg_adj else ""
    sdil = "_dil" if p.dil else ""
    snrm = "_nrm" if p.nrm else ""
    # Every method needs a distinct suffix or its results land on top of
    # another's. The first three are historical and must not move: existing
    # result trees are addressed by those names.
    smth = _BG_MTH_SUFFIX.get(p.bg_mth, f"_{p.bg_mth}")
    subfolder_name = "dat" + sbg + sadj + sdil + snrm + smth
    subfolder_path = base_path / subfolder_name
    subfolder_path.mkdir(parents=True, exist_ok=True)
    return subfolder_path


def likelihood_residual_table(fit_results: Mapping[str, FitResult]) -> pd.DataFrame:
    """Build the residual table of *fit_results*, standardized by each fit's likelihood.

    Called without its likelihood, ``residuals_from_fit_results`` standardizes
    every residual as Normal: a Student-t fit's ``(y - mu) / sigma`` then shows
    t tails (5.8% beyond |3| at nu = 3) as if they were outliers, and a
    mixture's wide component likewise. Here the family, nu and the mixture
    weights are read from each fit's own trace. One call per distinct trace: a
    multi-well fit shares one, while single-well samples each carry their own,
    whose nu, mixture weights and outlier probabilities must not be pooled.

    Parameters
    ----------
    fit_results : Mapping[str, FitResult]
        Well to fit result; least-squares results carry no trace and stay
        Normal.

    Returns
    -------
    pd.DataFrame
        The concatenated canonical residual tables.
    """
    groups: dict[int, dict[str, FitResult]] = {}
    for well, fr in fit_results.items():
        groups.setdefault(id(fr.trace), {})[well] = fr
    tables = []
    for group in groups.values():
        trace = next(iter(group.values())).trace
        robust, nu = robust_settings_from_trace(trace)
        tables.append(
            residuals_from_fit_results(
                dict(group),
                trace_id="",
                binding_function=binding_1site,
                robust=robust,
                student_t_nu=nu,
                trace=trace,
                residual_likelihood=robust_likelihood_from_trace(trace),
            )
        )
    tables = [t for t in tables if not t.empty]
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def _export_residual_tests(
    outfit: Path, fit_results: dict[str, FitResult], all_res: pd.DataFrame, index: int
) -> None:
    """Write the four residual-assumption tests beside the statistics table.

    Diagnostic only: a failure here is logged and skipped, never allowed to cost
    the run whose residuals it describes.

    Parameters
    ----------
    outfit : Path
        Fit output directory.
    fit_results : dict[str, FitResult]
        The fits the residuals came from, supplying each well's parameters.
    all_res : pd.DataFrame
        Their residual table.
    index : int
        The fit's index, for ``residual_tests_<index>.csv``.
    """
    params: dict[str, dict[str, float]] = {}
    is_ph = True
    for well, fr in fit_results.items():
        result = getattr(fr, "result", None)
        if result is None or not hasattr(result, "params"):
            continue
        params[str(well)] = {
            name: float(par.value) for name, par in result.params.items()
        }
        if fr.dataset is not None:
            is_ph = bool(fr.dataset.is_ph)
    try:
        tests = residual_tests(all_res, params or None, is_ph=is_ph, n_mc=199)
    except (ValueError, KeyError, np.linalg.LinAlgError) as exc:
        logger.warning("residual tests for fit %d skipped: %s", index, exc)
        return
    tests.to_csv(outfit / f"residual_tests_{index}.csv")


def export_residuals(
    outfit: Path, fit_results: dict[str, FitResult], index: int
) -> None:
    """Export fit residuals and their statistics to files.

    Writes nothing when no fit produced residuals: the plots are sized by the
    labels present, and an empty table used to raise inside matplotlib
    ("Number of columns must be a positive integer, not 0"), aborting a
    chloride run after its fits had succeeded. A diagnostic must not be able
    to destroy the run it describes.
    """
    try:
        all_res = likelihood_residual_table(fit_results)
    except (ValueError, KeyError):
        return
    if all_res.empty or "label" not in all_res or all_res["label"].nunique() == 0:
        return
    all_res.to_csv(outfit / f"residuals_{index}.csv", index=False)
    stats = residual_statistics(all_res)
    stats.to_csv(outfit / f"residual_stats_{index}.csv")
    _export_residual_tests(outfit, fit_results, all_res, index)
    label = str(index)
    fig_pred = plot_residual_vs_predicted(all_res, title=label)
    fig_pred.savefig(outfit / f"residual_vs_predicted_{index}.png", dpi=150)
    fig_yerr = plot_residual_vs_yerr(all_res, title=label)
    fig_yerr.savefig(outfit / f"residual_vs_yerr_{index}.png", dpi=150)
    fig_dist = plot_residual_distribution(all_res, title=label)
    fig_dist.savefig(outfit / f"residual_distribution_{index}.png", dpi=150)


def export_trace_summary(trace: object, outfit: Path, tag: str) -> None:
    """Write posterior statistics and the trace of a sampled model.

    Sampling and then keeping nothing from it is not a usable result: the
    multi-well path inferred ``x_true``, a per-well ``K`` and the whole ye_mag
    family, and dropped the trace on return, so a run could be performed but not
    inspected. This writes ``trace_summary_<tag>.csv`` - mean, sd, HDI, ``r_hat``
    and ESS per variable, the numbers that say whether to trust the rest - and
    the full trace as NetCDF for anything the summary does not cover.

    Failures here are logged and swallowed. The summary describes a fit that has
    already succeeded, and a diagnostic must not be able to destroy the run it
    reports on.

    Parameters
    ----------
    trace : object
        PyMC/ArviZ inference data from the fit. ``None`` writes nothing.
    outfit : Path
        Directory to write into.
    tag : str
        Suffix identifying the model, e.g. ``"multi"``.
    """
    if trace is None:
        return
    try:
        summary = az.summary(trace)
        summary.to_csv(outfit / f"trace_summary_{tag}.csv")
    except Exception:
        logger.warning("Could not summarise the %s trace", tag, exc_info=True)
        return
    try:
        trace.to_netcdf(outfit / f"trace_{tag}.nc")  # type: ignore[attr-defined]
    except Exception:
        logger.warning("Could not write the %s trace to NetCDF", tag, exc_info=True)


def screen_atypical_wells(titration: Titration) -> pd.DataFrame:
    """Screen every well on the plate, before anything is discarded or fitted.

    Replaces ``bad_wells.csv``, which was unusable: it ORed a single expected
    polarity across two channels that move oppositely by design, so on a real
    plate it reported all 88 of 88 wells as bad, and its "low signal" and "flat
    curve" columns were the same Series under two names.

    "Atypical" rather than "bad" because most of what lands here is not a
    fault. A flat 400 nm channel and a concordant pair are properties of the
    construct; only the dim ones are quality problems, and even those are
    highlighted rather than discarded.

    Parameters
    ----------
    titration : Titration
        Titration whose data and background levels are screened.

    Returns
    -------
    pd.DataFrame
        The atypical wells only, empty when the plate is unremarkable.
    """
    labels = sorted(titration.data)
    if not labels:
        return pd.DataFrame()
    skip = set(titration.scheme.buffer) | set(titration.scheme.nofit_keys)
    wells = {
        well: {
            str(lbl): np.asarray(titration.data[lbl].get(well, []), dtype=float)
            for lbl in labels
        }
        for well in titration.data[labels[0]]
        if well not in skip
    }
    if not wells:
        return pd.DataFrame()
    bg_level = {
        str(lbl): float(np.nanmean(np.asarray(titration.bg.get(lbl, []), dtype=float)))
        if len(titration.bg.get(lbl, []))
        else 0.0
        for lbl in labels
    }
    screened = screen_wells(
        wells, np.asarray(titration.x, dtype=float), bg_level=bg_level
    )
    flags = ["flag_low_signal", "flag_flat_curve", "flag_concordant"]
    present = [c for c in flags if c in screened.columns]
    if not present:
        return pd.DataFrame()
    return screened[screened[present].fillna(value=False).astype(bool).any(axis=1)]


def run_pre_fit_detection(titration: Titration, outfit: Path) -> None:
    """Discard unusable wells and record everything atypical beside them.

    Both files land in the fit folder, and the discard list carries the
    highlighted wells under their own heading: one file then answers "what
    happened to my wells" without joining it to another.

    Parameters
    ----------
    titration : Titration
        Titration to screen and then discard from.
    outfit : Path
        Fit output folder.
    """
    # Screened *before* discarding: a discarded well is atypical by
    # construction and belongs in both lists, and discarding first would drop
    # it from the titration and hide the evidence.
    atypical = screen_atypical_wells(titration)
    discards = titration.detect_and_discard_bad_wells()
    if not discards and atypical.empty:
        return
    outfit.mkdir(parents=True, exist_ok=True)
    if not atypical.empty:
        atypical.to_csv(outfit / "atypical_wells.csv", index=False)
        logger.info(
            "Pre-fit screening: %d atypical well(s) -> atypical_wells.csv",
            len(atypical),
        )
    lines = sorted(discards)
    # A heading per reason, so the file says why each well was singled out and
    # not merely that it was. A well answering to two reasons appears under
    # both; an empty reason is omitted, since a bare heading tells nobody
    # anything.
    if not atypical.empty:
        kept = atypical[~atypical["well"].isin(discards)]
        for flag in ("flag_low_signal", "flag_concordant", "flag_flat_curve"):
            if flag not in kept.columns:
                continue
            wells = sorted(
                kept.loc[kept[flag].fillna(value=False).astype(bool), "well"]
            )
            if wells:
                lines += ["", f"# {flag.removeprefix('flag_')}", *wells]
    (outfit / "discarded_wells.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def undetermined_k(fit: pd.DataFrame, *, is_ph: bool, max_k_se: float) -> pd.Series:
    """Flag the wells whose fitted K the data do not pin down.

    The pre-fit rule (``Titration._k_is_unconstrained``) judges a cheap fit
    against the whole x span, about 4 pH; a well can pass it and still come
    out of the final fit at 14 +/- 9, or widen only once sampled. This judges
    the fit actually reported.

    Parameters
    ----------
    fit : pd.DataFrame
        Per-well fit table indexed by well, with ``K``, ``sK`` and, for a
        sampled fit, ``Khdi03``.
    is_ph : bool
        A pH titration (K is a pKa, an interval scale) rather than chloride
        (K is a Kd, a ratio scale).
    max_k_se : float
        Largest acceptable standard error of a pKa, in pH.

    Returns
    -------
    pd.Series
        Boolean, indexed like *fit*. A missing or non-finite K or sK is
        undetermined. pH: ``sK > max_k_se``. Chloride: ``Kd <= 0``,
        ``sKd > Kd``, or a 94% HDI whose lower edge is the ``KD_MIN`` floor
        (within 10%): the data then bound Kd only from above, and the lower
        edge is the prior's - as a well saturated by the first addition is.
    """
    if not {"K", "sK"} <= set(fit.columns):
        return pd.Series(data=True, index=fit.index, dtype=bool)
    k = pd.Series(pd.to_numeric(fit["K"], errors="coerce"), index=fit.index)
    sk = pd.Series(pd.to_numeric(fit["sK"], errors="coerce"), index=fit.index)
    # NaN compares False, so this is "finite" without np.isfinite's Any.
    bad = ~(k.abs().lt(np.inf) & sk.abs().lt(np.inf))
    if is_ph:
        return bad | sk.gt(max_k_se)
    bad |= k.le(0) | sk.gt(k)
    if "Khdi03" in fit.columns:
        hdi_lo = pd.Series(
            pd.to_numeric(fit["Khdi03"], errors="coerce"), index=fit.index
        )
        bad |= hdi_lo.le(1.1 * KD_MIN)
    return bad


# One-sided 94% bound of a Normal, for fits that report an SE but no HDI.
_Z94 = 1.8808


def kd_lower_bound(fit: pd.DataFrame) -> pd.Series:
    """Lower edge of each well's 94% Kd interval.

    The sampled HDI (``Khdi03``) where there is one, otherwise the Normal
    bound ``K - 1.88 sK``; NaN where neither can be formed.

    Parameters
    ----------
    fit : pd.DataFrame
        Per-well fit table indexed by well, with ``K`` and ``sK``.

    Returns
    -------
    pd.Series
        Float, indexed like *fit*.
    """
    k = pd.Series(pd.to_numeric(fit["K"], errors="coerce"), index=fit.index)
    sk = pd.Series(pd.to_numeric(fit["sK"], errors="coerce"), index=fit.index)
    lower = k - _Z94 * sk
    if "Khdi03" in fit.columns:
        hdi = pd.Series(pd.to_numeric(fit["Khdi03"], errors="coerce"), index=fit.index)
        lower = hdi.where(hdi.notna(), lower)
    return lower


def no_binding_k(fit: pd.DataFrame, *, x_max: float) -> pd.Series:
    """Flag chloride wells whose Kd lies beyond the titration: they do not bind.

    A construct that does not bind has a flat curve, and a Kd the data place
    only above the highest concentration titrated. That is a result - "does
    not bind", Kd > the top concentration - not a failed fit, and it is kept
    apart from :func:`undetermined_k`, which would call it undetermined (its
    sampled Kd spans the range up to the ``kd_bounds`` ceiling, so SE > Kd).

    Parameters
    ----------
    fit : pd.DataFrame
        Per-well Kd table indexed by well, with ``K``, ``sK`` and, for a
        sampled fit, ``Khdi03``.
    x_max : float
        Highest ligand concentration titrated.

    Returns
    -------
    pd.Series
        Boolean, indexed like *fit*: the 94% lower bound (see
        :func:`kd_lower_bound`) is above *x_max*; or a least-squares Kd is
        pinned at the ``kd_bounds`` ceiling, where a flat curve drives it (its
        SE there is meaningless, 1e6 mM on a test plate), or sits above
        *x_max* with no finite SE.
    """
    if not {"K", "sK"} <= set(fit.columns):
        return pd.Series(data=False, index=fit.index, dtype=bool)
    k = pd.Series(pd.to_numeric(fit["K"], errors="coerce"), index=fit.index)
    sk = pd.Series(pd.to_numeric(fit["sK"], errors="coerce"), index=fit.index)
    no_se = ~sk.abs().lt(np.inf)
    ceiling = kd_bounds(x_max)[1]
    pinned = k.ge(0.99 * ceiling) if np.isfinite(ceiling) else k.gt(np.inf)
    return kd_lower_bound(fit).gt(x_max) | pinned | (no_se & k.gt(x_max))


def record_post_fit(outfit: Path, sections: Mapping[str, Sequence[str]]) -> None:
    """Append the reported fit's verdicts to ``discarded_wells.txt``.

    One heading per verdict (``# undetermined_k``, ``# no_binding``), after the
    pre-fit sections: the wells were fitted and are reported, so they are
    highlighted rather than discarded.

    Parameters
    ----------
    outfit : Path
        Fit output folder.
    sections : Mapping[str, Sequence[str]]
        Heading to wells; an empty one is omitted, and nothing is written
        when all are.
    """
    lines = []
    for heading, wells in sections.items():
        if wells:
            lines += ["", f"# {heading}", *sorted(wells)]
            logger.info("Post-fit: %d well(s) %s", len(wells), heading)
    if not lines:
        return
    with (outfit / "discarded_wells.txt").open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def _mark_figure(fr: FitResult, note: str) -> None:
    """Write a post-fit verdict on a well's own figure."""
    if fr.figure is not None:
        fr.figure.text(
            0.01, 0.99, note, ha="left", va="top", fontsize=9, color="firebrick"
        )


def _verdict_note(fit: pd.DataFrame, well: str, verdict: str) -> str:
    """One line saying what was concluded about a well's K, and its value."""
    k = float(pd.to_numeric(fit.loc[well, "K"], errors="coerce"))
    sk = float(pd.to_numeric(fit.loc[well, "sK"], errors="coerce"))
    se = "n/a" if np.isnan(sk) else f"{sk:.2g}"
    return f"{verdict}: K = {k:.3g} +/- {se}"


def k_verdicts(
    fit: pd.DataFrame, *, is_ph: bool, max_k_se: float, x_max: float
) -> pd.DataFrame:
    """Post-fit verdicts on each well's K, as boolean columns.

    Parameters
    ----------
    fit : pd.DataFrame
        Per-well fit table indexed by well.
    is_ph : bool
        A pH titration rather than chloride.
    max_k_se : float
        Largest acceptable SE of a pKa (pH); see :func:`undetermined_k`.
    x_max : float
        Highest concentration titrated; see :func:`no_binding_k`.

    Returns
    -------
    pd.DataFrame
        ``undetermined`` for pH; ``undetermined`` and ``no_binding`` for
        chloride, exclusive: a well that does not bind is a result, not an
        undetermined one.
    """
    undetermined = undetermined_k(fit, is_ph=is_ph, max_k_se=max_k_se)
    if is_ph:
        return pd.DataFrame({"undetermined": undetermined})
    no_binding = no_binding_k(fit, x_max=x_max)
    return pd.DataFrame({
        "undetermined": undetermined & ~no_binding,
        "no_binding": no_binding,
    })


def _ye_mag_screening_noise(
    bg_noise: Mapping[str, float] | float,
) -> NoiseConfig:
    """Build the legacy homoscedastic ``ye_mag`` screening noise config.

    Parameters
    ----------
    bg_noise : Mapping[str, float] | float
        Per-label background-noise hints seeding the LogNormal ``ye_mag`` prior.

    Returns
    -------
    NoiseConfig
        ``ye_mag`` config centred on ``log(bg_noise * 3.6)`` per label.
    """
    if isinstance(bg_noise, Mapping):
        first_mu: float | dict[str, float] = {
            str(label): float(np.log(max(float(value) * 3.6, 1e-6)))
            for label, value in bg_noise.items()
        }
    else:
        first_mu = float(np.log(max(float(bg_noise) * 3.6, 1e-6)))
    return NoiseConfig.ye_mag(shared=False, prior="lognormal", mu=first_mu, sigma=0.5)


def _structured_noise(  # ruff: ignore[too-many-arguments] - one per noise term
    titration: Titration,
    *,
    noise_mode: Literal["centered", "fixed"],
    floor_mode: Literal["centered", "fixed"] | None = None,
    gain_mode: Literal["centered", "fixed"] | None = None,
    alpha_mode: Literal["centered", "fixed"] | None = None,
    learn_ye_mags: bool = False,
) -> NoiseConfig:
    """Build the physical floor/gain/alpha noise config from titration.

    Floors always come from the measured ``bg_noise``; gain and alpha come
    from ``titration.params``. Gain and alpha are
    ``"free"`` when no value was supplied -- there is no hint to centre on, so
    the sampler learns them -- and otherwise take ``noise_mode`` (``"centered"``
    or ``"fixed"``).

    Parameters
    ----------
    titration : Titration
        Titration whose ``bg_noise`` and ``params.noise_gain``/
        ``params.noise_alpha`` supply the hints.
    noise_mode : Literal["centered", "fixed"]
        How supplied floor/gain/alpha values are treated by the sampler. The
        floor hint always exists, since it is measured, so ``"fixed"`` pins it.
    floor_mode : Literal["centered", "fixed"] | None
        Override for the floor alone. One mode for all three terms cannot ask
        the question the terms need: pinning alpha at zero also pinned the
        floor, so a cell meant to isolate gain instead measured a sigma that
        could not rescale at all, and came back understating the noise
        threefold.
    gain_mode : Literal["centered", "fixed"] | None
        Override for gain alone.
    alpha_mode : Literal["centered", "fixed"] | None
        Override for alpha alone.
    learn_ye_mags : bool
        Also learn a ye_mag multiplier on sigma, so the supplied terms set the
        shape and the data the level. Per label, or per well when the caller
        asks for per-well ye_mags.

    Returns
    -------
    NoiseConfig
        A ``structured`` config with per-label floor, gain and alpha hints.
    """
    labels = sorted(titration.data.keys())

    def _per_label(values: tuple[float, ...]) -> dict[str, float]:
        return {
            lbl: float(values[i]) for i, lbl in enumerate(labels) if i < len(values)
        }

    gains = _per_label(titration.params.noise_gain)
    alphas = _per_label(titration.params.noise_alpha)
    floors = {str(lbl): float(v) for lbl, v in dict(titration.sigma_floor).items()}
    return NoiseConfig.structured(
        floor=floors or None,
        gain=gains or 0.0,
        alpha=alphas or 0.0,
        learn_ye_mags=learn_ye_mags,
        # The floor was the one term never pinned, and it is the term that
        # decides whether the model is structured at all. Left free it drifts
        # to several times the measured bg_noise and swallows the variance the
        # signal-dependent terms exist to describe: on L2 label 1 the sampler
        # put the floor at 52.6 against a measured 11.8, leaving gain and alpha
        # under 1% of the variance at every signal level. A "structured" model
        # that reports a flat sigma is not one.
        # A term with no hint stays free whatever the mode says: there is
        # nothing to centre on or pin to.
        floor_mode=(floor_mode or noise_mode) if floors else "free",
        gain_mode=(gain_mode or noise_mode) if gains else "free",
        alpha_mode=(alpha_mode or noise_mode) if alphas else "free",
    )


def _single_refit_two_pass(  # ruff: ignore[too-many-arguments]
    ds: Dataset,
    *,
    screening_noise: NoiseConfig,
    refit_noise: NoiseConfig,
    sampler: SamplerConfig,
    unit_yerr: bool = True,
    criterion: OutlierCriterion | None = None,
) -> tuple[FitResult, pd.DataFrame]:
    """Screen residual outliers with a robust PyMC pass, then refit unrobustly.

    The noise strategy is supplied by the caller so the two-pass sequence itself
    is strategy-agnostic: swapping the homoscedastic ``ye_mag`` pair for a
    heteroscedastic ``NoiseConfig.structured`` pair changes only the call site.

    Parameters
    ----------
    ds : Dataset
        Single-well multi-label titration dataset.
    screening_noise : NoiseConfig
        Noise model for the robust screening pass.
    refit_noise : NoiseConfig
        Noise model for the unrobust refit.
    sampler : SamplerConfig
        Sampling controls used for both passes.
    unit_yerr : bool
        Reset observation errors to one before the screening pass. Required by
        the ``ye_mag`` strategy, whose multiplier learns the scale; a structured
        floor/gain/alpha model builds its own variance and must pass ``False``.
    criterion : OutlierCriterion | None
        What the screening pass counts as an outlier. ``None`` keeps the
        standardized-residual tail rule, tolerating no expected tail; pass
        :class:`RobustZMad` to score against the residuals' own MAD scale
        instead.

    Returns
    -------
    tuple[FitResult, pd.DataFrame]
        The refit result and the screening pass's residual table.
    """
    screening_input = dataset_with_unit_yerr(ds) if unit_yerr else ds
    initial = fit_binding_pymc(
        screening_input,
        robust=RobustConfig(enabled=True),
        noise=screening_noise,
        sampler=sampler,
    )
    residuals = residuals_from_fit_results(
        {"single": initial},
        "pymc_robust_unweighted",
        binding_1site,
        robust=True,
        outlier_threshold=3.0,
    )
    residuals = mark_outliers(
        residuals,
        criterion
        or ResidualTail(
            threshold=3.0, allowed_tail_fraction=0.0, min_allowed_tail_count=0
        ),
    )
    mask_source = initial.dataset if initial.dataset is not None else ds
    holder = FitResult(dataset=copy.deepcopy(mask_source))
    masked = apply_exclusions({"single": holder}, residuals, min_keep=3).get(
        "single", copy.deepcopy(mask_source)
    )
    seeded = copy.deepcopy(initial)
    seeded.dataset = masked

    final = fit_binding_pymc(
        seeded,
        robust=RobustConfig(enabled=False),
        noise=refit_noise,
        sampler=sampler,
    )
    return final, residuals


def _global_fit_method(fit_method: str) -> tuple[str, str | None]:
    """Map a configured fit method onto the plate fit's method and reweighting.

    Parameters
    ----------
    fit_method : str
        Value of ``params.fit_method``.

    Returns
    -------
    tuple[str, str | None]
        Method to fit with, and the reweighting scheme if any.

    Notes
    -----
    Written out per method rather than nested, because a method missing here
    does not fail: it falls through to ``"lm"`` and the run reports success
    having fitted something else. That is how ``--mcmc multi`` spent months as
    a no-op, and adding ``odr`` to the CLI without touching this would have
    repeated it.
    """
    if fit_method == "odr":
        return "odr", None
    if fit_method == "irls":
        return "lm", "irls"
    if fit_method == "huber":
        return "huber", None
    return "lm", None


def fit_single_mcmc(
    titration: Titration,
    datasets: dict[str, typing.Any],
    outfit: Path,
    spec: McmcSpec | None,
) -> TitrationResults | None:
    """Run optional PyMC fits for export, per well or jointly.

    Parameters
    ----------
    titration : Titration
        Titration object containing the plate scheme, fit keys, and
        background noise.
    datasets : dict[str, typing.Any]
        Mapping from well identifiers to datasets to fit.
    outfit : Path
        Output directory used for residual-refit diagnostic CSV files.
    spec : McmcSpec | None
        Sampling request deciding whether and how to run MCMC. ``None``
        disables MCMC export.

    Returns
    -------
    TitrationResults | None
        Per-well PyMC fit results when *spec* is provided. Returns ``None``
        when *spec* is ``None``.
    """
    if spec is None:
        return None

    if spec.model == "multi":
        # Every well fitted jointly, with control K shared across each control
        # group. Only the per-well results are returned, but the shared trace -
        # which carries the pooled control K, x_true and the ye_mag family - is
        # summarised to disk rather than dropped.
        # The noise family has to be passed explicitly: omitting it left
        # --mcmc-noise structured, --noise-gain and --noise-alpha accepted,
        # echoed back in the run configuration, and silently ignored. The pH
        # axis likewise: never passing x_error_model left every run on the
        # shared axis whatever the model could express.
        x_axis: dict[str, typing.Any] = {"x_error_model": spec.x_error_model}
        if spec.x_start_between_sigma is not None:
            x_axis["x_start_between_sigma"] = spec.x_start_between_sigma
        if spec.learn_x_start_between:
            x_axis["learn_x_start_between"] = True
        multi = fit_binding_pymc_multi(
            datasets,
            titration.scheme,
            sampler=spec.sampler,
            per_well_ye_mags=spec.per_well_ye_mags,
            ye_mag_parameterization=spec.ye_mag_parameterization,
            robust=spec.robust,
            ctr_free_k=spec.ctr_free_k,
            ctr_sigma_w_prior=spec.ctr_sigma_w_prior,
            noise=(
                _structured_noise(
                    titration,
                    noise_mode=spec.noise_mode,
                    floor_mode=spec.floor_mode,
                    gain_mode=spec.gain_mode,
                    alpha_mode=spec.alpha_mode,
                    learn_ye_mags=spec.noise_ye_mag,
                )
                if spec.structured_noise
                else _DEFAULT_NOISE
            ),
            **x_axis,
        )
        export_trace_summary(getattr(multi, "trace", None), outfit, "multi")
        return TitrationResults(titration.scheme, titration.fit_keys, multi.results)

    if spec.model == "single":
        mcmc_fits = {
            key: fit_binding_pymc(ds, sampler=spec.sampler, robust=spec.robust)
            for key, ds in datasets.items()
        }
        return TitrationResults(titration.scheme, titration.fit_keys, mcmc_fits)

    sampler = spec.sampler
    structured = spec.structured_noise
    if structured:
        # One config for both passes: unlike ye_mag, whose refit prior is
        # recentred on the screening pass's learned multiplier, the structured
        # model's floor/gain/alpha hints do not shift between passes.
        noise = _structured_noise(
            titration,
            noise_mode=spec.noise_mode,
            floor_mode=spec.floor_mode,
            gain_mode=spec.gain_mode,
            alpha_mode=spec.alpha_mode,
            learn_ye_mags=spec.noise_ye_mag,
        )
        screening_noise, refit_noise = noise, noise
    else:
        screening_noise = _ye_mag_screening_noise(titration.bg_noise)
        refit_noise = NoiseConfig.ye_mag(
            shared=False, prior="lognormal", mu=0.0, sigma=0.25
        )
    mcmc_fits = {}
    residual_rows = []
    for key, ds in datasets.items():
        final, residuals = _single_refit_two_pass(
            ds,
            screening_noise=screening_noise,
            refit_noise=refit_noise,
            sampler=sampler,
            # ye_mag's multiplier learns the scale from unit errors; the
            # structured model builds its own variance and must keep the real
            # y_err it was given.
            unit_yerr=not structured,
        )
        mcmc_fits[key] = final
        if not residuals.empty:
            residual_rows.append(residuals.assign(well=key))
    if residual_rows:
        pd.concat(residual_rows, ignore_index=True).to_csv(
            outfit / "single_refit_initial_residual_outliers.csv", index=False
        )
    return TitrationResults(titration.scheme, titration.fit_keys, mcmc_fits)


def _export_ctr_holdout(
    datasets: Mapping[str, typing.Any],
    groups: Mapping[str, Sequence[str]],
    outfit: Path,
    method: str,
) -> None:
    """Leave each control out in turn and write how far it lands from its group.

    Agreement with a bench pK measures accuracy; this measures repeatability,
    which is the thing a plate can report about itself. Each control is refitted
    with its own free K while its group-mates keep the shared one, so the spread
    of those differences is the precision the plate actually delivers - and it
    needs no external reference to be interpretable.

    Failures are logged, not raised: this is a diagnostic running after the fit
    it describes has already succeeded.

    Parameters
    ----------
    datasets : Mapping[str, typing.Any]
        Well to dataset, as fitted.
    groups : Mapping[str, Sequence[str]]
        Control group name to member wells.
    outfit : Path
        Directory to write into.
    method : str
        ``"lm"`` or ``"odr"``, naming the output file.
    """
    if not groups:
        return
    try:
        rows = (
            ctr_holdout_odr(datasets, groups)
            if method == "odr"
            else ctr_holdout(datasets, groups)
        )
    except Exception:
        logger.warning("Control leave-one-out failed for %s", method, exc_info=True)
        return
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(outfit / f"ctr_loo_{method}.csv", index=False)
    if "delta_k_mean" not in df.columns:
        logger.warning(
            "Control leave-one-out for %s has no delta_k_mean column; "
            "wrote the rows but no summary.",
            method,
        )
        return
    col = df["delta_k_mean"].astype(float)
    row: dict[str, object] = {
        "method": method,
        "n_controls": int(col.notna().sum()),
        "median_abs_delta_k": float(np.nanmedian(np.abs(col))),
        "rms_delta_k": float(np.sqrt(np.nanmean(col**2))),
        "max_abs_delta_k": float(np.nanmax(np.abs(col))),
    }
    if "p_abs_delta_k_lt_rope" in df.columns:
        # How often a held-out control lands close enough to its group to be
        # called the same - the plate's own statement about repeatability.
        row["frac_within_rope"] = float(
            np.nanmean(df["p_abs_delta_k_lt_rope"].astype(float))
        )
    pd.DataFrame([row]).to_csv(outfit / f"ctr_loo_{method}_summary.csv", index=False)


# Shapiro-Wilk says nothing useful below a handful of points; printing it there
# would invite over-reading a number that is mostly noise.
_MIN_N_FOR_SHAPIRO = 5


def _plate_fit_caption(well: str, pars: Parameters, ds: Dataset) -> str:
    """Title carrying the fitted midpoint, its error, and whether it is inside.

    A midpoint fitted outside the range actually titrated is an extrapolation,
    and the figure should say so where it cannot be missed - it is the single
    most useful warning about a well.

    Parameters
    ----------
    well : str
        Well identifier.
    pars : Parameters
        Fitted parameters, carrying ``K`` and its standard error.
    ds : Dataset
        The fitted dataset, whose x span defines "inside".

    Returns
    -------
    str
        One-line title.
    """
    k = pars["K"]
    err = f" ± {k.stderr:.3f}" if k.stderr and np.isfinite(k.stderr) else ""
    xs = np.concatenate([np.asarray(da.xc, dtype=float) for da in ds.values()])
    outside = (
        " — outside titrated range" if not (xs.min() <= k.value <= xs.max()) else ""
    )
    return f"{well}   pK = {k.value:.3f}{err}{outside}"


def _plate_fit_stats(rows: list[dict[str, typing.Any]]) -> str:
    """Per-label residual summary: its scale, and how far from normal.

    ``RMS z`` sits near 1 when the error model is right. A Shapiro p far below
    0.05 says the residuals are not plausibly normal, which on seven points
    usually means one bad point rather than a wrong curve.

    Parameters
    ----------
    rows : list[dict[str, typing.Any]]
        One well's canonical residual records.

    Returns
    -------
    str
        A short monospace block, or "" when there is nothing to report.
    """
    if not rows:
        return ""
    df = pd.DataFrame(rows)
    if "std_res" not in df or "label" not in df:
        return ""
    lines = []
    for lbl, g in df.groupby("label"):
        z = np.asarray(g["std_res"], dtype=float)
        z = z[np.isfinite(z)]
        if z.size == 0:
            continue
        piece = f"lbl{lbl}: n={z.size}  RMS z={float(np.sqrt(np.mean(z**2))):.2f}"
        if z.size >= _MIN_N_FOR_SHAPIRO:
            piece += f"  shapiro p={float(sp_stats.shapiro(z).pvalue):.2f}"
        lines.append(piece)
    return "\n".join(lines)


def _with_screened_mask(ds: Dataset, dropped: dict[str, list[int]]) -> Dataset:
    """Copy a dataset with the screen's discards marked as excluded.

    The screening fit masks its own copies internally, so the datasets handed
    back to the plotter still show every point as fitted. Re-applying the mask
    here is what makes a discarded point render as discarded rather than as a
    point the curve simply misses.

    Parameters
    ----------
    ds : Dataset
        The well's dataset as it was passed to the fitter.
    dropped : dict[str, list[int]]
        Label to the original indices the screen removed.

    Returns
    -------
    Dataset
        A copy carrying the screen's mask, or *ds* itself when nothing was
        dropped for this well.
    """
    if not dropped:
        return ds
    arrays = {}
    for lbl, da in ds.items():
        bad = dropped.get(str(lbl), [])
        new = copy.deepcopy(da)
        if bad:
            mask = np.asarray(new.mask).copy()
            for i in bad:
                if 0 <= i < len(mask):
                    mask[i] = False
            new.mask = mask
        arrays[lbl] = new
    return type(ds)(arrays, is_ph=ds.is_ph)


def _plate_fit_results(
    datasets: Mapping[str, typing.Any],
    result: PlateLMResult | PlateODRResult,
    titration: Titration,
    *,
    with_figures: bool = False,
) -> TitrationResults:
    """Wrap a plate-wide fit as per-well results, so it can be plotted.

    The plate fitters solve one least-squares problem over every well and return
    parameter vectors, not the per-well objects the export path draws from.
    Rebuilding those objects here is what lets a plate fit produce the same K
    plot and per-well figures as the fits beside it.

    Parameters
    ----------
    datasets : Mapping[str, typing.Any]
        Well to the `Dataset` that was fitted.
    result : PlateLMResult | PlateODRResult
        The plate-wide fit.
    titration : Titration
        Supplies the scheme and fit keys carried on the results.

    with_figures : bool
        Also draw each well's fitted curve. The plate solver never plots - it
        returns parameter vectors - so the figures are rendered here from those
        parameters, which is what gives ``--plate-fit`` the same per-well images
        as the fitters beside it.

    Returns
    -------
    TitrationResults
        Per-well results carrying K, its error, the plateaus, and a figure when
        one was asked for.
    """
    resid_by_well: dict[str, list[dict[str, typing.Any]]] = {}
    for r in getattr(result, "residuals", []) or []:
        resid_by_well.setdefault(str(r.get("well")), []).append(r)
    out: dict[str, FitResult] = {}
    excluded = getattr(result, "excluded_points", {}) or {}
    for well, ds_in in datasets.items():
        row = result.params.get(well)
        if not row:
            continue
        ds = _with_screened_mask(ds_in, excluded.get(well, {}))
        pars = Parameters()
        for name, value in row.items():
            if name.startswith("s"):
                continue
            pars.add(name, value=value)
            stderr = row.get(f"s{name}")
            if stderr is not None and np.isfinite(stderr):
                pars[name].stderr = float(stderr)
        fig = None
        if with_figures:
            fig, ax = plt.subplots()
            # nboot=0: the bootstrap band resamples each parameter from its own
            # standard error independently, which for a plate fit would ignore
            # the covariance the joint solve produced and overstate the spread.
            plot_fit(ax, ds, pars, nboot=0, pp=PlotParameters(ds.is_ph))
            ax.set_title(_plate_fit_caption(well, pars, ds), fontsize=10)
            note = _plate_fit_stats(resid_by_well.get(well, []))
            if note:
                ax.text(
                    0.02,
                    0.02,
                    note,
                    transform=ax.transAxes,
                    fontsize=7.5,
                    va="bottom",
                    ha="left",
                    family="monospace",
                    bbox={"facecolor": "white", "alpha": 0.75, "lw": 0},
                )
        out[well] = FitResult(
            figure=fig, result=SimpleNamespace(params=pars), dataset=ds
        )
    return TitrationResults(scheme=titration.scheme, fit_keys=set(out), results=out)


def _write_plate_fit_figures(
    results: TitrationResults, outfit: Path, method: str, *, png: bool
) -> None:
    """Save the K plot and, when asked, one figure per well.

    Parameters
    ----------
    results : TitrationResults
        Per-well results wrapped from the plate fit.
    outfit : Path
        Fit output directory.
    method : str
        ``"lm"`` or ``"odr"``, naming the outputs.
    png : bool
        Write per-well figures as well as the K plot.
    """
    fig = results.plot_k(title=f"plate {method}")
    fig.savefig(outfit / f"K_plate_{method}.png")
    plt.close(fig)
    if not png:
        return
    png_dir = outfit / f"plate_{method}"
    png_dir.mkdir(parents=True, exist_ok=True)
    for well in results.fit_keys:
        well_fig = results[well].figure
        if well_fig is not None:
            well_fig.savefig(png_dir / f"{well}.png")
            plt.close(well_fig)


def _plate_noise_model(titration: Titration) -> dict[str, NoiseModelParams] | None:
    """Per-label floor/gain/alpha, as the classical fitters want it.

    ``Titration`` already assembles this to stamp ``y_errc`` onto a dataset;
    the plate fitters need the parameters themselves so they can re-evaluate
    a signal-dependent sigma at the model prediction.

    Parameters
    ----------
    titration : Titration
        Supplies ``sigma_floor`` and the configured gain/alpha.

    Returns
    -------
    dict[str, NoiseModelParams] | None
        Label to its noise parameters, or ``None`` when the titration does not
        carry enough to build one - the fitter then keeps the ``y_err`` already
        on the datasets, which is the pre-existing behaviour.
    """
    data = getattr(titration, "data", None)
    if not data:
        return None
    params = getattr(titration, "params", None)
    gain = getattr(params, "noise_gain", ()) or ()
    alpha = getattr(params, "noise_alpha", ()) or ()
    # sigma_floor, not bg_noise: an explicit --noise-floor has to reach the
    # plate fitters too. They are the ones that re-evaluate a signal-dependent
    # sigma at their own prediction, so the floor bites hardest here.
    floors = getattr(titration, "sigma_floor", None) or {}
    return {
        lbl: NoiseModelParams(
            sigma_floor=float(floors.get(lbl, 0.0)),
            gain=gain[i] if i < len(gain) else 0.0,
            alpha=alpha[i] if i < len(alpha) else 0.0,
        )
        for i, lbl in enumerate(sorted(data.keys()))
    }


def _write_plate_residual_diagnostics(
    result: PlateLMResult | PlateODRResult, outfit: Path, method: str
) -> None:
    """Write the three error-model diagnostics for a plate-wide fit.

    These are the plots that say whether the *shape* of the error model is
    right, which no single summary number does: squared residual against
    squared sigma for the calibration slope, standardised residual against
    predicted signal for a trend the model does not carry, and the histogram
    with its Q-Q plot for the tails.

    Reading them here differs in one way from the per-label path. This fitter
    profiles a noise scale per label, so a uniform rescaling of ``y_err`` is
    absorbed into ``ye_mag`` and the calibration slope sits near one by
    construction. What survives the rescaling is structure - a trend with
    predicted signal, or tails - and that is exactly what these plots are for.
    The absolute factor is not lost, it is reported as ``ye_mag``.

    Parameters
    ----------
    result : PlateLMResult | PlateODRResult
        The plate-wide fit, carrying its residual table.
    outfit : Path
        Fit output directory.
    method : str
        ``"lm"`` or ``"odr"``, naming the outputs.
    """
    rows = getattr(result, "residuals", []) or []
    if not rows:
        return
    all_res = pd.DataFrame(rows)
    needed = {"label", "yhat", "raw_res", "sigma", "std_res"}
    if not needed.issubset(all_res.columns):
        logger.warning(
            "plate %s residuals lack %s; skipping the error-model plots",
            method,
            sorted(needed - set(all_res.columns)),
        )
        return
    all_res.to_csv(outfit / f"plate_{method}_residuals.csv", index=False)
    title = f"plate {method}"
    figures = [
        (plot_residual_vs_predicted(all_res, title=title), "residual_vs_predicted"),
        (plot_residual_distribution(all_res, title=title), "residual_distribution"),
    ]
    # res^2 against y_err^2 needs y_err to vary. Under the default error model
    # it does not: one sigma per label, so every point lands on the same
    # vertical line and the plot cannot show a slope. It becomes meaningful
    # once a signal-dependent error model is asked for, which is exactly when
    # a calibration slope is worth reading.
    if all_res.groupby("label")["sigma"].nunique().max() > 1:
        figures.insert(
            0, (plot_residual_vs_yerr(all_res, title=title), "residual_vs_yerr")
        )
    else:
        logger.info(
            "plate %s: y_err is constant within each label, so the res^2 vs "
            "y_err^2 calibration plot carries no slope and is skipped; read "
            "residual_vs_predicted instead",
            method,
        )
    for fig, name in figures:
        fig.savefig(outfit / f"plate_{method}_{name}.png", dpi=150)
        plt.close(fig)


def _export_plate_fit_plots(  # ruff: ignore[too-many-arguments]
    titration: Titration,
    datasets: Mapping[str, typing.Any],
    result: PlateLMResult | PlateODRResult,
    outfit: Path,
    method: str,
    *,
    png: bool = True,
) -> None:
    """Write the K plot, and per-well figures, for a plate-wide fit.

    Failures are logged rather than raised: the fit and its CSV have already
    succeeded, and a plot must not be able to fail the run that produced it.

    Parameters
    ----------
    titration : Titration
        Supplies the plate scheme.
    datasets : Mapping[str, typing.Any]
        Well to the `Dataset` that was fitted.
    result : PlateLMResult | PlateODRResult
        The plate-wide fit.
    outfit : Path
        Fit output directory.
    method : str
        ``"lm"`` or ``"odr"``, naming the output.
    png : bool
        Also write one figure per well, under ``plate_<method>/``. Off when the
        run asked for no images, since rendering a curve per well is the
        expensive part.
    """
    try:
        results = _plate_fit_results(datasets, result, titration, with_figures=png)
        _write_plate_fit_figures(results, outfit, method, png=png)
    except Exception:
        logger.warning("Could not plot the plate %s fit", method, exc_info=True)
    # Separate try: the error-model diagnostics read only the residual table,
    # so a failure to render per-well curves must not take them down with it.
    try:
        _write_plate_residual_diagnostics(result, outfit, method)
    except Exception:
        logger.warning(
            "Could not plot the plate %s residual diagnostics", method, exc_info=True
        )


def _floors(noise: Mapping[str, NoiseModelParams] | None) -> dict[str, float]:
    """Return the read-noise floor per label, which a gain calibration holds or starts from."""
    return {lbl: float(p.sigma_floor) for lbl, p in (noise or {}).items()}


def _write_gain_calibration(cal: GainCalibration[typing.Any], path: Path) -> None:
    """Write each calibration pass's floor and gain per label, and whether it settled."""
    cal.history.drop(columns="plate").assign(
        converged=cal.converged, n_iter=cal.n_iter
    ).to_csv(path, index=False)


def _gain_calibrated_plate_fit(  # ruff: ignore[too-many-arguments]
    datasets: dict[str, Dataset],
    groups: dict[str, list[str]],
    noise: Mapping[str, NoiseModelParams] | None,
    gain_terms: str,
    *,
    outfit: Path,
    screen_z: float | None,
    frac_threshold: float | None,
    calibrate_screen: bool,
) -> PlateLMResult:
    """Plate fit weighted by a calibrated floor and gain, after any screen.

    The screen, when asked for, runs exactly as for the other weightings and
    only decides which points go; the gain is then calibrated on what is left.

    Parameters
    ----------
    datasets : dict[str, Dataset]
        Well identifier to global `Dataset`.
    groups : dict[str, list[str]]
        Control groups pooling one K; empty for free K.
    noise : Mapping[str, NoiseModelParams] | None
        The titration's noise model, supplying the floors.
    gain_terms : str
        ``"gain"`` or ``"floor-gain"``.
    outfit : Path
        Fit output directory, for ``plate_lm_noise_history.csv``.
    screen_z : float | None
        Screening threshold, or ``None`` for no screen.
    frac_threshold : float | None
        Fractional screen threshold.
    calibrate_screen : bool
        Judge the screen on the calibrated ruler.

    Returns
    -------
    PlateLMResult
        The fit under the converged weights, carrying the screen's exclusions.
    """
    data: Mapping[str, Dataset] = datasets
    screened: PlateLMResult | None = None
    if screen_z is not None:
        screened = fit_plate_lm_screened(
            datasets,
            groups,
            noise_model=noise,
            threshold=screen_z,
            frac_threshold=frac_threshold,
            calibrate_screen=calibrate_screen,
        )
        data = apply_excluded_points(datasets, screened.excluded_points)
    cal = calibrate_plate_lm(
        {"": data},
        {"": groups},
        {"": _floors(noise)},
        fit_floor=gain_terms == "floor-gain",
    )
    _write_gain_calibration(cal, outfit / "plate_lm_noise_history.csv")
    if not cal.converged:
        logger.warning("plate gain calibration did not settle in %d passes", cal.n_iter)
    result = cal.fits[""]
    if screened is not None:
        result.excluded_points = screened.excluded_points
        result.n_excluded = screened.n_excluded
    return result


def export_plate_fit(  # ruff: ignore[too-many-arguments]
    titration: Titration,
    datasets: dict[str, typing.Any],
    outfit: Path,
    method: str,
    *,
    png: bool = True,
    calibrate_noise: bool = False,
    screen_z: float | None = None,
    frac_threshold: float | None = None,
    ctr_free_k: bool = False,
    calibrate_screen: bool = True,
    gain_terms: str | None = None,
) -> PlateLMResult | PlateODRResult | None:
    """Fit the whole plate at once, classically, and write K per well.

    The per-well fits above give each well its own noise scale; this fits every
    well in one least-squares problem with the observation-noise scale profiled
    per label across the plate, and each control group pooled onto one K. It is
    Its results are wrapped as ordinary per-well fit results, so it produces the
    same K plot and per-well figures as every other fitter in the run rather
    than a bare CSV.

    Parameters
    ----------
    titration : Titration
        Supplies the plate scheme, whose control groups set which wells pool.
    datasets : dict[str, typing.Any]
        Well identifier to global `Dataset`, as built for the other fits.
    outfit : Path
        Fit output directory.
    method : str
        ``"lm"`` for the plate-wide least squares, ``"odr"`` to also let each
        titration step's x move within its recorded uncertainty.
    png : bool
        Write per-well figures as well as the K plot.
    calibrate_noise : bool
        Estimate gain and alpha per label from the fit's own residuals and
        refit under them, instead of taking ``y_err`` as built.
    screen_z : float | None
        Drop points whose calibrated standardised residual exceeds this and
        refit. ``None`` fits once.
    frac_threshold : float | None
        Also screen the first label on ``|y - yhat| / yhat`` exceeding this,
        sparing points the other channel moves with. See
        :func:`~clophfit.fitting.plate_lm.fractional_outliers`.
    ctr_free_k : bool
        Give every well its own K instead of pooling each control group onto a
        shared one. The plate fitters pool by default, exactly as ``--mcmc
        multi`` does, so the flag has to reach here or the two fitters answer
        different questions from the same command line.
    calibrate_screen : bool
        With *screen_z*, judge points on the calibrated ruler (default) rather
        than on the weights as built. The weights K is then fitted with follow
        *calibrate_noise*, screened or not.
    gain_terms : str | None
        ``"gain"`` or ``"floor-gain"``: weight by ``floor^2 + gain * yhat`` with
        alpha 0, the gain (and with ``"floor-gain"`` the floor) calibrated from
        the plate's residuals between refits, corrected for the degrees of
        freedom the fit used (:func:`~clophfit.fitting.gain_calibration.calibrate_plate_lm`).
        Takes the place of *calibrate_noise*; after a screen it calibrates on
        the screened points. ``None`` leaves the weights to *calibrate_noise*.

    Returns
    -------
    PlateLMResult | PlateODRResult | None
        The fit, or ``None`` when no wells were fitted. The result rather than
        the path it wrote, because the caller needs ``excluded_points`` to hand
        the screen's verdict to whatever fits next.
    """
    if not datasets:
        return None
    groups = (
        {}
        if ctr_free_k
        else {
            name: sorted(wells)
            for name, wells in getattr(titration.scheme, "names", {}).items()
        }
    )
    result: PlateLMResult | PlateODRResult
    if method == "odr":
        x_err = getattr(titration, "x_err", None)
        result = fit_plate_odr(
            datasets,
            groups,
            x_err=np.asarray(x_err, dtype=float) if x_err is not None else None,
        )
    else:
        # Hand the fitter the physical noise model so a signal-dependent y_err
        # is re-evaluated at the prediction rather than at the observation.
        # With a flat floor this changes nothing, which is why it is safe to
        # pass unconditionally.
        noise = _plate_noise_model(titration)
        if gain_terms is not None:
            result = _gain_calibrated_plate_fit(
                datasets,
                groups,
                noise,
                gain_terms,
                outfit=outfit,
                screen_z=screen_z,
                frac_threshold=frac_threshold,
                calibrate_screen=calibrate_screen,
            )
        elif screen_z is not None:
            # The screen's ruler and the fit's weights are separate choices
            # (--plate-screen-noise, --plate-noise); --plate-noise used to be
            # overridden here, so "calibrated" meant plain weights after a screen.
            result = fit_plate_lm_screened(
                datasets,
                groups,
                noise_model=noise,
                threshold=screen_z,
                frac_threshold=frac_threshold,
                calibrate_screen=calibrate_screen,
                calibrate_refit=calibrate_noise,
            )
        else:
            result = fit_plate_lm(
                datasets, groups, noise_model=noise, calibrate_noise=calibrate_noise
            )

    pooled = {well: name for name, wells in groups.items() for well in wells}
    rows = [
        {
            "well": well,
            "K": result.k[well],
            "sK": result.k_stderr.get(well, float("nan")),
            "ctr_group": pooled.get(well, ""),
            "k_shared": well in pooled,
        }
        for well in sorted(result.k)
    ]
    pd.DataFrame(rows).to_csv(outfit / f"plate_{method}_K.csv", index=False)
    # What the screen removed, in a form one can join, sort and count. The
    # per-well figures mark these points, but nothing listed them: the L4 wells
    # the screen left unfittable were found by diffing residual tables between
    # arms. Written even when empty, so an absent file means the fit did not
    # run rather than that it dropped nothing.
    dropped = [
        {"well": well, "label": lbl, "raw_i": idx}
        for well, per_label in sorted(
            (getattr(result, "excluded_points", None) or {}).items()
        )
        for lbl, indices in sorted(per_label.items())
        for idx in sorted(indices)
    ]
    pd.DataFrame(dropped, columns=["well", "label", "raw_i"]).to_csv(
        outfit / f"plate_{method}_excluded.csv", index=False
    )
    _export_plate_fit_plots(titration, datasets, result, outfit, method, png=png)
    pd.DataFrame([
        {"label": lbl, "ye_mag": mag} for lbl, mag in sorted(result.ye_mag.items())
    ]).to_csv(outfit / f"plate_{method}_ye_mag.csv", index=False)
    # The noise parameters are the answer when calibration was asked for, so
    # write them next to the multiplier rather than leaving them in the fit.
    noise = getattr(result, "noise", {}) or {}
    if noise:
        pd.DataFrame([
            {
                "label": lbl,
                "sigma_floor": v.get("sigma_floor", float("nan")),
                "gain": v.get("gain", float("nan")),
                "alpha": v.get("alpha", float("nan")),
                "ye_mag": result.ye_mag.get(lbl, float("nan")),
            }
            for lbl, v in sorted(noise.items())
        ]).to_csv(outfit / f"plate_{method}_noise.csv", index=False)
    _export_ctr_holdout(datasets, groups, outfit, method)
    logger.info(
        "plate %s fit: %d wells, %d points, converged=%s, ye_mag=%s",
        method,
        len(result.k),
        result.n_points,
        result.success,
        {k: round(v, 3) for k, v in result.ye_mag.items()},
    )
    return result


def _gain_terms(noise_option: str) -> str | None:
    """Return the gain calibration a ``--plate-noise``/``--fit-noise`` value asks for."""
    return noise_option if noise_option in {"gain", "floor-gain"} else None


def _gain_calibrated_single_fit(
    titration: Titration,
    datasets: dict[str, Dataset],
    method: str,
    fit_noise: str,
    outfit: Path,
) -> TitrationResults:
    """Per-well global fits weighted by a floor and gain calibrated across the plate.

    Parameters
    ----------
    titration : Titration
        Supplies the floors, the scheme and the outlier setting.
    datasets : dict[str, Dataset]
        Well identifier to global `Dataset`.
    method : str
        ``"lm"`` or ``"huber"``.
    fit_noise : str
        ``"gain"`` or ``"floor-gain"``.
    outfit : Path
        Fit output directory, for ``noise_single_history.csv``.

    Returns
    -------
    TitrationResults
        The fits under the converged weights, with that noise model attached.

    Raises
    ------
    ValueError
        If *method* is neither ``"lm"`` nor ``"huber"``.
    """
    if method not in {"lm", "huber"}:
        msg = f"--fit-noise {fit_noise} needs --fit-method lm or huber, not {method}"
        raise ValueError(msg)
    cal = calibrate_single_well(
        {"": datasets},
        {"": _floors(_plate_noise_model(titration))},
        method="huber" if method == "huber" else "lm",
        fit_floor=fit_noise == "floor-gain",
        remove_outliers=titration.params.outlier,
    )
    _write_gain_calibration(cal, outfit / "noise_single_history.csv")
    if not cal.converged:
        logger.warning(
            "single-well gain calibration did not settle in %d passes", cal.n_iter
        )
    return TitrationResults(
        titration.scheme,
        titration.fit_keys,
        cal.fits[""],
        noise_model=PlateNoiseModel(cal.noise[""]),
    )


def export_fit(
    titration: Titration,
    subfolder: Path,
    config: TecanConfig,
    spec: McmcSpec | None = None,
    plate_fit: str | None = None,
) -> None:
    """Export all fitted parameters, plots, and data files."""
    outfit = subfolder / "fit"
    outfit.mkdir(parents=True, exist_ok=True)

    datasets = {k: titration.create_global_ds(k) for k in titration.fit_keys}

    export_list = []
    for label, dat in titration.data.items():
        if dat:
            ds_single = {
                k: titration.create_ds(k, label=label) for k in titration.fit_keys
            }
            export_list.append(
                titration.fit_plate(
                    ds_single,
                    method=titration.params.fit_method,
                    remove_outliers=titration.params.outlier,
                )
            )

    method, reweight = _global_fit_method(titration.params.fit_method)

    fit_noise = getattr(config, "fit_noise", "fixed")
    if fit_noise == "fixed":
        global_res = titration.fit_plate(
            datasets,
            method=method,
            reweight=reweight,
            remove_outliers=titration.params.outlier,
        )
    else:
        global_res = _gain_calibrated_single_fit(
            titration, datasets, method, fit_noise, outfit
        )
    export_list.append(global_res)

    odr_res = titration.fit_plate(
        datasets,
        method="odr",
        remove_outliers=titration.params.outlier,
        reweight=reweight,
    )
    export_list.append(odr_res)

    mcmc_datasets = datasets
    if plate_fit is not None:
        screen_z = getattr(config, "plate_screen_z", None)
        plate_res = export_plate_fit(
            titration,
            datasets,
            outfit,
            plate_fit,
            png=config.png,
            calibrate_noise=getattr(config, "plate_noise", "fixed") == "calibrated",
            screen_z=screen_z,
            frac_threshold=getattr(config, "plate_screen_frac", None),
            ctr_free_k=getattr(config, "ctr_free_k", False),
            calibrate_screen=getattr(config, "plate_screen_noise", "calibrated")
            == "calibrated",
            gain_terms=_gain_terms(getattr(config, "plate_noise", "fixed")),
        )
        # The hybrid: the classical screen decides what is an outlier, on a
        # calibrated ruler, and the Bayesian fit inherits that verdict. Without
        # this the two fit different data and neither result explains the other.
        excluded = getattr(plate_res, "excluded_points", None) if screen_z else None
        if excluded:
            mcmc_datasets = apply_excluded_points(datasets, excluded)
            logger.info(
                "plate screen |z|>%s excluded points in %d well(s); the MCMC "
                "inherits them",
                screen_z,
                len(excluded),
            )

    mcmc_res = fit_single_mcmc(titration, mcmc_datasets, outfit, spec)
    if mcmc_res is not None:
        export_list.append(mcmc_res)

    x_max = float(np.nanmax(titration.x))
    text = _verdict_text(is_ph=titration.is_ph, max_k_se=config.max_k_se, x_max=x_max)
    flagged: dict[str, list[str]] = {}
    for i, results in enumerate(export_list):
        flagged = _export_one_fit(
            i, results, outfit, config, text, is_ph=titration.is_ph, x_max=x_max
        )
    # The last fit is the one reported (the MCMC when run), so its verdicts
    # are the ones recorded beside the pre-fit discards.
    record_post_fit(outfit, {text[col][0]: wells for col, wells in flagged.items()})


def _verdict_text(
    *, is_ph: bool, max_k_se: float, x_max: float
) -> dict[str, tuple[str, str, str]]:
    """Verdict column -> (discarded_wells.txt heading, K-plot reason, figure note)."""
    undetermined = (
        f"K undetermined (SE > {max_k_se:g} pH)"
        if is_ph
        else "K undetermined (SE > Kd, or HDI down to the floor)"
    )
    return {
        "undetermined": ("undetermined_k", "undetermined", undetermined),
        "no_binding": (
            "no_binding",
            "non-binding",
            f"does not bind (Kd > {x_max:g} at 94%)",
        ),
    }


def _export_one_fit(  # ruff: ignore[too-many-arguments]
    i: int,
    results: TitrationResults,
    outfit: Path,
    config: TecanConfig,
    text: Mapping[str, tuple[str, str, str]],
    *,
    is_ph: bool,
    x_max: float,
) -> dict[str, list[str]]:
    """Write one fit's tables, figures and K plot; return its flagged wells.

    Parameters
    ----------
    i : int
        Index of the fit, naming ``ffit{i}.csv``, ``K{i}.png`` and ``lb{i}/``.
    results : TitrationResults
        The fit.
    outfit : Path
        Fit output folder.
    config : TecanConfig
        Supplies ``png``, ``detect_bad``, ``max_k_se``, ``lim`` and ``title``.
    text : Mapping[str, tuple[str, str, str]]
        See :func:`_verdict_text`.
    is_ph : bool
        A pH titration rather than chloride.
    x_max : float
        Highest concentration titrated.

    Returns
    -------
    dict[str, list[str]]
        Verdict column to the wells it flags; empty lists without
        ``detect_bad``, whose columns are written to ``ffit{i}.csv`` regardless.
    """
    fit = results.dataframe
    verdicts = k_verdicts(fit, is_ph=is_ph, max_k_se=config.max_k_se, x_max=x_max)
    flagged = {
        col: sorted(verdicts.index[verdicts[col]]) if config.detect_bad else []
        for col in verdicts.columns
    }
    png_dir = outfit / f"lb{i}"
    data_dir = png_dir / "ds"
    for key in results.fit_keys:
        fr = results[key]
        if config.png:
            if fr.figure:
                for col, wells in flagged.items():
                    if key in wells:
                        _mark_figure(fr, _verdict_note(fit, key, text[col][2]))
                png_dir.mkdir(parents=True, exist_ok=True)
                fr.figure.savefig(png_dir / f"{key}.png")
            if fr.dataset:
                data_dir.mkdir(parents=True, exist_ok=True)
                fr.dataset.export(data_dir / f"{key}.csv")
    fit.join(verdicts).sort_index().to_csv(outfit / f"ffit{i}.csv")
    exclude = {text[col][1]: wells for col, wells in flagged.items()}
    f = results.plot_k(xlim=config.lim, title=config.title + f"lb:{i}", exclude=exclude)
    f.savefig(outfit / f"K{i}.png")
    export_residuals(outfit, results.results, i)
    return flagged


def export_data_fit(
    titration: Titration,
    tecan_config: TecanConfig,
    mcmc: McmcSpec | None = None,
    plate_fit: str | None = None,
) -> None:
    """Export dat files [x,y1,..,yN] from copy of titration.data."""

    def write(x: ArrayF, data: dict[str, dict[str, ArrayF]], out_folder: Path) -> None:
        if any(data):
            out_folder.mkdir(parents=True, exist_ok=True)
            columns = ["x"] + [str(i) for i in data]
            first_label = next(iter(titration.labelblocksgroups.keys()))
            for key in data[first_label]:
                dat = np.vstack((x, [data[i][key] for i in data]))
                datxy = pd.DataFrame(dat.T, columns=columns)
                datxy.to_csv(out_folder / f"{key}.dat", index=False)

    if tecan_config.comb:
        saved_p = copy.copy(titration.params)
        combinations = generate_combinations()
        for combination in combinations:
            apply_combination(titration, combination)
            subfolder = prepare_output_folder(titration, tecan_config.out_fp)
            write(titration.x, titration.data, subfolder)
            if tecan_config.fit:
                if tecan_config.detect_bad:
                    run_pre_fit_detection(titration, subfolder / "fit")
                export_fit(titration, subfolder, tecan_config, mcmc, plate_fit)
        titration.params = saved_p
    else:
        subfolder = prepare_output_folder(titration, tecan_config.out_fp)
        write(titration.x, titration.data, subfolder)
        if tecan_config.fit:
            if tecan_config.detect_bad:
                run_pre_fit_detection(titration, subfolder / "fit")
            export_fit(titration, subfolder, tecan_config, mcmc, plate_fit)
