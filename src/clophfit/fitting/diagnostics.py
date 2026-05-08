"""Well-quality diagnostics for plate-reader titration data.

Two complementary entry points:

- :func:`detect_bad_wells_from_dat` — reads raw ``.dat`` files (one per well,
  all labels together).  No fitting required; works before the fitting pipeline.
  Detects outliers based on robust trendline between signal span and maximum signal.

- :func:`detect_bad_wells` — reads ``ffit*.csv`` fit results (one label per
  file).  Adds fit-quality criteria (K at bound, K outlier, poor fit) on top
  of the signal-quality checks.

Detection criteria
------------------
- **K at bound** : K equals the optimizer bound (default 3 or 11 for pH).
  Fit converged to a limit, not a true optimum.
- **K outlier** : |K - median_K| > ``k_mad_factor * MAD(K)`` across all
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
from pathlib import Path

import numpy as np
import pandas as pd

from clophfit.fitting.utils import flag_trend_outliers

logger = logging.getLogger(__name__)

__all__ = ["detect_bad_wells", "detect_bad_wells_from_dat"]

_NEAR_ZERO = 1e-9


def detect_bad_wells_from_dat(
    data_dir: str | Path,
    *,
    z_threshold: float = 3.0,
    ctr_cols: list[int] | None = None,
) -> pd.DataFrame:
    r"""Flag unreliable wells by reading raw ``.dat`` titration files.

    Reads every ``*.dat`` file in *data_dir* (one per well).  Each file must
    have an ``x`` column and one or more signal columns (e.g. ``y1``, ``y2``).
    All labels are checked together — no fitting is required.

    Parameters
    ----------
    data_dir : str | Path
        Directory containing ``*.dat`` files (one per well, CSV format with
        columns ``x, y1[, y2, ...]``).
    z_threshold : float
        Z-score threshold for outlier detection on the max-vs-span trendline.
    ctr_cols : list[int] | None
        1-based column numbers for control wells (e.g. ``[1, 12]``).
        Currently used only for logging; all flags apply equally to CTR wells.

    Returns
    -------
    pd.DataFrame
        One row per well with columns:

        - ``well``
        - ``flag_low_signal``
        - ``flag_flat_curve``
        - ``flag_any``
        - ``flag_count``

        Sorted by descending ``flag_count``.

    Raises
    ------
    FileNotFoundError
        If no ``*.dat`` files are found in *data_dir*.
    """
    data_dir = Path(data_dir)
    dat_files = sorted(data_dir.glob("*.dat"))
    if not dat_files:
        msg = f"No .dat files found in {data_dir}"
        raise FileNotFoundError(msg)

    # Load all wells
    well_data: dict[str, pd.DataFrame] = {}
    for f in dat_files:
        well_data[f.stem] = pd.read_csv(f)

    wells = list(well_data.keys())
    sig_cols = [c for c in next(iter(well_data.values())).columns if c != "x"]

    # Per label, per well: max |signal| and dynamic range
    max_sig: dict[str, list[float]] = {col: [] for col in sig_cols}
    span_val: dict[str, list[float]] = {col: [] for col in sig_cols}

    for well in wells:
        df = well_data[well]
        for col in sig_cols:
            y = df[col].astype(float).dropna().to_numpy()
            if y.size == 0:
                logger.warning(
                    "detect_bad_wells_from_dat: %s has no valid %s values; flagging as low-signal/flat",
                    well,
                    col,
                )
                max_sig[col].append(0.0)
                span_val[col].append(0.0)
                continue
            abs_max = float(np.max(np.abs(y)))
            span = float(y.max() - y.min())
            max_sig[col].append(abs_max)
            span_val[col].append(span)

    result = pd.DataFrame({"well": wells})

    # We keep the column names flag_low_signal and flag_flat_curve for backwards compatibility
    low_signal_or_flat = pd.Series(data=False, index=result.index)

    for col in sig_cols:
        x_series = pd.Series(max_sig[col], index=result.index)
        y_series = pd.Series(span_val[col], index=result.index)
        outliers = flag_trend_outliers(x_series, y_series, threshold=z_threshold)
        low_signal_or_flat |= outliers

    result["flag_low_signal"] = low_signal_or_flat
    result["flag_flat_curve"] = (
        low_signal_or_flat  # They are the same under the trendline approach
    )
    flag_cols = ["flag_low_signal", "flag_flat_curve"]
    result["flag_count"] = result[flag_cols].sum(axis=1)
    result["flag_any"] = result["flag_count"] > 0

    if ctr_cols:
        col_nums = result["well"].str.extract(r"(\d+)$", expand=False).astype(int)
        is_ctr_mask = col_nums.isin(ctr_cols)
        result["is_ctr"] = is_ctr_mask
        n_ctr = int(is_ctr_mask.sum())
    else:
        n_ctr = 0
    n_flagged = int(result["flag_any"].sum())
    logger.info(
        "detect_bad_wells_from_dat: %d/%d wells flagged (%d CTR wells present); %s",
        n_flagged,
        len(wells),
        n_ctr,
        ", ".join(f"{c}={result[c].sum()}" for c in flag_cols),
    )

    return result.sort_values("flag_count", ascending=False).reset_index(drop=True)


def detect_bad_wells(  # noqa: PLR0913, PLR0915
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
