r"""Well-quality diagnostics for plate-reader titration data.

Two complementary entry points:


- :func:`detect_bad_wells` — reads ``ffit*.csv`` fit results (one label per
  file).  Adds fit-quality criteria (K at bound, K outlier, poor fit) on top
  of the signal-quality checks.

Detection criteria
------------------
- **K at bound** : K equals the optimizer bound (default 3 or 11 for pH).
  Fit converged to a limit, not a true optimum.
- **K outlier** : \|K - median_K\| > ``k_mad_factor * MAD(K)`` across all

  wells on the plate.  Identifies wells with biologically implausible K.
- **Poor fit** : sK / K > ``max_sk_ratio``.  Relative uncertainty so large
  that K is undetermined.
- **Low signal / Flat curve** : Outlier detection based on robust Theil-Sen regression
  between max signal and dynamic range. A well is flagged if its signal span is too low
  compared to the trend, or if its max signal is significantly below the plate median.
- **Inverted curve** : S0 > S1 for pH or S0 < S1 for Cl -- wrong polarity.
  Only checked in :func:`detect_bad_wells` (requires fitted plateaus).
- **High residuals** : per-well residual MAD > ``residual_mad_factor`` times
  the plate median MAD.  Requires the optional ``residual_stats`` DataFrame.
"""

import logging
import typing
from collections.abc import Mapping

import numpy as np
import pandas as pd

from clophfit.fitting.utils import flag_trend_outliers

logger = logging.getLogger(__name__)

__all__ = ["curve_turnover", "detect_bad_wells", "screen_wells"]

_NEAR_ZERO = 1e-9


# The 485 nm upper quartile must clear its own background by this factor.
# On five adjudicated plates every well judged to need attention falls below
# 2.0 and every well judged sound clears it; the ratio is dimensionless, so the
# same number holds across plates whose brightness spans fifteenfold.
_LOW_SIGNAL_RATIO = 2.0
# A channel spanning less than this fraction of its own peak is not titrating.
_FLAT_SPAN_FRACTION = 0.15
# Healthy wells are strongly anti-correlated between channels (median -0.955),
# because the 400 nm neutral form falls with pH as the 485 nm anion rises.
_CONCORDANT_CORR = 0.0
# The anion channel carries the titration; the neutral one varies for reasons
# that belong to the construct rather than to well quality.
_QUALITY_LABEL = "2"


def screen_wells(  # ruff: ignore[too-many-arguments] - four independent thresholds, each named
    wells: Mapping[str, Mapping[str, np.ndarray]],
    x: np.ndarray,
    *,
    bg_level: Mapping[str, float],
    quality_label: str = _QUALITY_LABEL,
    low_signal_ratio: float = _LOW_SIGNAL_RATIO,
    flat_span_fraction: float = _FLAT_SPAN_FRACTION,
    concordant_corr: float = _CONCORDANT_CORR,
) -> pd.DataFrame:
    """Judge wells from the data alone, before any curve is fitted.

    Fitting a well to discover whether it was worth fitting is avoidable. Across
    eleven plates every well with ``sK/K > 0.3`` -- all 55 of them -- has a dim
    485 nm channel, so a poor fit is a *consequence* of weak signal, and the
    signal can be read straight off the data.

    Three independent questions are asked, where the previous implementation
    collapsed two of them into a single trend-outlier test reported under two
    names:

    * **low signal** -- does the channel clear the background it sits on? The
      upper quartile is compared with the background *level*, so the ratio is
      dimensionless and one threshold serves plates of very different
      brightness. The upper quartile rather than the maximum because a single
      spike otherwise rescues a dead well.
    * **flat curve** -- does the channel move at all? A flat but *bright*
      400 nm channel is a property of the construct, so this is reported and
      never treated as a failure.
    * **concordant** -- do the channels move together? They should be opposed:
      the neutral form falls with pH as the anion rises, giving a median
      inter-label correlation of -0.955. Concordance is directly observable and,
      unlike a per-channel direction test, survives the 400 nm acid turnover --
      which is present in 60% of wells and defeats every endpoint-based
      polarity check.

    The well-level verdict is taken on *quality_label* alone. A dim neutral
    channel beside a healthy anion channel is not a bad well: the per-label
    columns record it so the caller can exclude that channel, which is what the
    pre-fit detector does and what reviewers endorse.

    This screens; it does not discard. Discarding is better decided either by
    :meth:`Titration.detect_and_discard_bad_wells` or, post-fit, by ``sK/K``,
    which was the only criterion found to separate unusable wells from merely
    weak ones without taking the weak ones with it.

    Parameters
    ----------
    wells : Mapping[str, Mapping[str, np.ndarray]]
        Well key to label key to that label's signal at each ``x``.
    x : np.ndarray
        Independent variable, in any order.
    bg_level : Mapping[str, float]
        Per-label background level the signal is measured against.
    quality_label : str
        Label whose signal decides the well-level verdict.
    low_signal_ratio : float
        Upper-quartile-to-background ratio below which a channel is too dim.
    flat_span_fraction : float
        Span, as a fraction of the channel's peak, below which it is flat.
    concordant_corr : float
        Inter-label correlation above which the channels count as concordant.

    Returns
    -------
    pd.DataFrame
        One row per well: ``well``; per label ``signal_ratio_{lbl}``,
        ``flag_low_signal_{lbl}``, ``flag_flat_curve_{lbl}`` and
        ``turnover_{lbl}``; and well-level ``flag_low_signal`` (on
        *quality_label*), ``flag_flat_curve``, ``label_corr`` and
        ``flag_concordant``.
    """
    xa = np.asarray(x, dtype=float)
    order = np.argsort(xa)
    labels = sorted({lbl for w in wells.values() for lbl in w})
    rows: list[dict[str, typing.Any]] = []
    for well, per_label in wells.items():
        row: dict[str, typing.Any] = {"well": well}
        series: dict[str, np.ndarray] = {}
        for lbl in labels:
            y = np.asarray(per_label.get(lbl, []), dtype=float)
            if y.size != xa.size or not np.isfinite(y).any():
                row[f"signal_ratio_{lbl}"] = float("nan")
                row[f"flag_low_signal_{lbl}"] = False
                row[f"flag_flat_curve_{lbl}"] = False
                row[f"turnover_{lbl}"] = float("nan")
                continue
            y = y[order]
            series[lbl] = y
            background = float(bg_level.get(lbl, 0.0))
            upper = float(np.nanpercentile(y, 75))
            ratio = upper / background if background > 0 else np.inf
            peak = float(np.nanmax(np.abs(y)))
            span = float(np.nanmax(y) - np.nanmin(y))
            row[f"signal_ratio_{lbl}"] = ratio
            row[f"flag_low_signal_{lbl}"] = bool(ratio < low_signal_ratio)
            row[f"flag_flat_curve_{lbl}"] = bool(
                peak > 0 and span / peak < flat_span_fraction
            )
            row[f"turnover_{lbl}"] = curve_turnover(xa[order], y)
        corr = float("nan")
        if len(series) == 2:  # ruff: ignore[magic-value-comparison] - two labels
            a, b = (series[lbl] for lbl in sorted(series))
            ok = np.isfinite(a) & np.isfinite(b)
            if int(ok.sum()) > 2 and np.std(a[ok]) > 0 and np.std(b[ok]) > 0:  # ruff: ignore[magic-value-comparison]
                corr = float(np.corrcoef(a[ok], b[ok])[0, 1])
        row["label_corr"] = corr
        row["flag_concordant"] = bool(np.isfinite(corr) and corr > concordant_corr)
        # The well-level verdict rests on the anion channel alone.
        row["flag_low_signal"] = bool(row.get(f"flag_low_signal_{quality_label}"))
        row["flag_flat_curve"] = any(
            bool(row.get(f"flag_flat_curve_{lbl}")) for lbl in labels
        )
        rows.append(row)
    return pd.DataFrame(rows)


def curve_turnover(x: np.ndarray, y: np.ndarray) -> float:
    """How far a titration curve comes back down from its own peak.

    The 400 nm neutral-form channel does not fall monotonically with pH: it
    turns over at the acid end. So a channel's direction cannot be read off its
    endpoints, and a rule that tries produces false inversions on every well
    with a pronounced dip -- on this campaign, 44 of them, all with interior
    maxima.

    This measures the effect directly instead. Take the smaller of the two
    drops from the peak to the ends, over the observed range. A monotone curve
    has its peak at an end, so one drop is zero and so is the result; a curve
    that genuinely turns over scores by how far it returns on the side it need
    not. The range, not the peak, is the denominator, so the measure stays
    meaningful where the signal is negative -- which it is on the dimmest
    wells, and where dividing by a peak near zero would not be.

    Parameters
    ----------
    x : np.ndarray
        Independent variable, in any order. Tecan files store pH descending.
    y : np.ndarray
        Signal at each ``x``. NaNs are ignored.

    Returns
    -------
    float
        0.0 for a monotone or flat curve, up to 1.0 for one that returns fully
        to both ends. NaN when fewer than three points survive.
    """
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    keep = np.isfinite(xa) & np.isfinite(ya)
    if int(keep.sum()) < 3:  # ruff: ignore[magic-value-comparison] - two points cannot peak
        return float("nan")
    xa, ya = xa[keep], ya[keep]
    ya = ya[np.argsort(xa)]
    span = float(np.max(ya) - np.min(ya))
    if span <= _NEAR_ZERO:
        # A dead channel has no range, so there is nothing to come back from.
        return 0.0
    peak = float(np.max(ya))
    shallower_return = min(peak - float(ya[0]), peak - float(ya[-1]))
    return shallower_return / span


def detect_bad_wells(  # ruff: ignore[too-many-arguments, too-many-statements]
    ffit: pd.DataFrame,
    *,
    k_min: float = 3.0,
    k_max: float = 11.0,
    k_mad_factor: float = 5.0,
    max_sk_ratio: float = 0.3,
    z_threshold: float = 3.0,
    check_polarity: bool = True,
    is_ph: bool = True,
    ctr_cols: list[int] | None = None,
    residual_stats: pd.DataFrame | None = None,
    residual_mad_factor: float = 5.0,
) -> pd.DataFrame:
    """Flag unreliable wells from a ffit result DataFrame.

    Parameters
    ----------
    ffit : pd.DataFrame
        Per-well fit results with at minimum columns ``well``, ``K``, ``sK``
        and at least one pair of ``S0_{lbl}`` / ``S1_{lbl}`` columns.
    k_min : float
        Lower optimizer bound for K (default 3.0 for pH).
    k_max : float
        Upper optimizer bound for K (default 11.0 for pH).
    k_mad_factor : float
        Outlier threshold: flag if ``|K - median| > k_mad_factor * MAD``.
    max_sk_ratio : float
        Maximum tolerated relative uncertainty sK/K (default 0.30).
    z_threshold : float
        Z-score threshold for outlier detection on the max-vs-span trendline.
    check_polarity : bool
        If True, flag wells where the signal direction is inverted relative
        to the expected biological response.
    is_ph : bool
        If True (default), pH assay: expect S1 > S0 (signal rises with pH).
        If False, Cl assay: expect S0 > S1.
    ctr_cols : list[int] | None
        Column numbers (1-based, e.g. ``[1, 12]``) reserved for control wells.
    residual_stats : pd.DataFrame | None
        Optional DataFrame from ``residual_stats_*.csv``.
    residual_mad_factor : float
        Flag if per-well residual MAD > ``residual_mad_factor`` times the
        plate median MAD (default 5.0).

    Returns
    -------
    pd.DataFrame
        One row per well with boolean flag columns.
    """
    df = ffit.copy()
    n = len(df)
    result = pd.DataFrame({"well": df["well"]})

    if ctr_cols:
        col_nums = df["well"].str.extract(r"(\d+)$", expand=False).astype(int)
        is_ctr = col_nums.isin(ctr_cols)
    else:
        is_ctr = pd.Series(data=False, index=df.index)

    sample_mask = ~is_ctr

    tol = 1e-6
    at_bound = (np.abs(df["K"] - k_min) < tol * (k_max - k_min)) | (
        np.abs(df["K"] - k_max) < tol * (k_max - k_min)
    )
    result["flag_k_at_bound"] = at_bound & sample_mask

    sample_k = df.loc[sample_mask, "K"]
    k_median = float(np.nanmedian(sample_k))
    k_mad = float(np.nanmedian(np.abs(sample_k - k_median)))
    if k_mad < _NEAR_ZERO:
        k_mad = float(np.nanstd(sample_k))
    result["flag_k_outlier"] = (
        np.abs(df["K"] - k_median) > k_mad_factor * k_mad
    ) & sample_mask

    result["flag_poor_fit"] = (df["sK"] / np.abs(df["K"])) > max_sk_ratio
    result["flag_poor_fit"] &= ~at_bound

    s0_cols = [
        c
        for c in df.columns
        if c.startswith("S0_") and not c.endswith(("hdi03", "hdi97"))
    ]
    s1_cols = [
        c
        for c in df.columns
        if c.startswith("S1_") and not c.endswith(("hdi03", "hdi97"))
    ]
    s0_labels = {c[3:] for c in s0_cols}
    s1_labels = {c[3:] for c in s1_cols}
    labels = list(s0_labels & s1_labels)

    low_signal_or_flat = pd.Series(data=False, index=df.index)
    inverted_any = pd.Series(data=False, index=df.index)

    for lbl in labels:
        s0 = df[f"S0_{lbl}"].astype(float)
        s1 = df[f"S1_{lbl}"].astype(float)
        max_sig = pd.Series(np.maximum(np.abs(s0), np.abs(s1)), index=df.index)
        span_val = pd.Series(np.abs(s1 - s0), index=df.index)

        # apply trendline outlier detection
        outliers = flag_trend_outliers(max_sig, span_val, threshold=z_threshold)
        low_signal_or_flat |= outliers

        if check_polarity:
            if is_ph:
                inverted_any |= s1 < s0
            else:
                inverted_any |= s0 < s1

    result["flag_low_signal"] = low_signal_or_flat
    result["flag_flat_curve"] = low_signal_or_flat
    if check_polarity:
        result["flag_inverted"] = inverted_any & sample_mask

    if residual_stats is not None and "mad" in residual_stats.columns:
        if "well" in residual_stats.columns:
            well_mad = residual_stats.groupby("well")["mad"].mean()
            plate_median_mad = float(np.nanmedian(well_mad.to_numpy()))
            if plate_median_mad < _NEAR_ZERO:
                plate_median_mad = float(well_mad.max())
            high_resid = well_mad[
                well_mad > residual_mad_factor * plate_median_mad
            ].index
            result["flag_high_residuals"] = result["well"].isin(high_resid)
        else:
            plate_mad = float(residual_stats["mad"].mean())
            logger.debug(
                "residual_stats lacks 'well' column; plate MAD=%.3f, skipping per-well flag",
                plate_mad,
            )

    flag_cols = [c for c in result.columns if c.startswith("flag_")]
    result["flag_count"] = result[flag_cols].sum(axis=1)
    result["flag_any"] = result["flag_count"] > 0

    n_ctr = int(is_ctr.sum())
    n_flagged = int(result["flag_any"].sum())
    logger.info(
        "detect_bad_wells: %d/%d sample wells flagged (%d CTR excluded); %s",
        n_flagged,
        n - n_ctr,
        n_ctr,
        ", ".join(f"{c}={result[c].sum()}" for c in flag_cols if c != "flag_any"),
    )

    return result.sort_values("flag_count", ascending=False).reset_index(drop=True)
