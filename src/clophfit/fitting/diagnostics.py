"""Well-quality diagnostics for plate-reader titration data.

Provides :func:`detect_bad_wells` to flag unreliable wells from ffit CSV
outputs, based on criteria observed in L1-L4 experimental data.

Detection criteria
------------------
- **K at bound** : K equals the optimizer bound (default 3 or 11 for pH).
  Fit converged to a limit, not a true optimum.
- **K outlier** : |K - median_K| > ``k_mad_factor * MAD(K)`` across all
  wells on the plate.  Identifies wells with biologically implausible K.
- **Poor fit** : sK / K > ``max_sk_ratio``.  Relative uncertainty so large
  that K is undetermined.
- **Flat curve** : dynamic range |S1 - S0| / max(|S0|, |S1|) <
  ``min_dynamic_range``.  Signal barely changes over the pH/Cl range, so K
  is unidentifiable.
- **Inverted curve** : S0 > S1 for pH (fluorescence increases with acid) or
  S0 < S1 for Cl -- unexpected polarity, suggests a broken well.
- **High residuals** : per-well residual MAD > ``residual_mad_factor`` times
  the plate median MAD.  Requires the optional ``residual_stats`` DataFrame.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["detect_bad_wells"]

_NEAR_ZERO = 1e-9


def detect_bad_wells(  # noqa: PLR0913
    ffit: pd.DataFrame,
    *,
    k_min: float = 3.0,
    k_max: float = 11.0,
    k_mad_factor: float = 5.0,
    max_sk_ratio: float = 0.3,
    min_dynamic_range: float = 0.05,
    check_polarity: bool = True,
    is_ph: bool = True,
    residual_stats: pd.DataFrame | None = None,
    residual_mad_factor: float = 5.0,
) -> pd.DataFrame:
    """Flag unreliable wells from a ffit result DataFrame.

    Parameters
    ----------
    ffit : pd.DataFrame
        Per-well fit results with at minimum columns ``well``, ``K``, ``sK``
        and at least one pair of ``S0_{lbl}`` / ``S1_{lbl}`` columns.
        Typically read from ``ffit*.csv`` produced by ``ppr``.
    k_min : float
        Lower optimizer bound for K (default 3.0 for pH).
    k_max : float
        Upper optimizer bound for K (default 11.0 for pH).
    k_mad_factor : float
        Outlier threshold: flag if ``|K - median| > k_mad_factor * MAD``.
    max_sk_ratio : float
        Maximum tolerated relative uncertainty sK/K (default 0.30).
    min_dynamic_range : float
        Minimum required |S1-S0|/max(|S0|,|S1|) per label (default 0.05).
    check_polarity : bool
        If True, flag wells where the signal direction is inverted relative
        to the expected biological response.
    is_ph : bool
        If True (default), pH assay: expect S1 > S0 (signal rises with pH).
        If False, Cl assay: expect S0 > S1.
    residual_stats : pd.DataFrame | None
        Optional DataFrame from ``residual_stats_*.csv`` with columns
        ``label``, ``mad``, and ``well``.  When provided, enables per-well
        residual-MAD outlier detection.
    residual_mad_factor : float
        Flag if per-well residual MAD > ``residual_mad_factor`` times the
        plate median MAD (default 5.0).

    Returns
    -------
    pd.DataFrame
        One row per well with boolean flag columns:

        - ``flag_k_at_bound``
        - ``flag_k_outlier``
        - ``flag_poor_fit``
        - ``flag_flat_curve``
        - ``flag_inverted``    (only when ``check_polarity=True``)
        - ``flag_high_residuals``  (only when ``residual_stats`` provided)
        - ``flag_any``         -- True if any flag is set

        Ordered by descending ``flag_count``.

    Examples
    --------
    >>> import pandas as pd
    >>> ffit = pd.DataFrame({
    ...     "well": ["A01", "B06", "E10"],
    ...     "K": [7.1, 3.0, 11.0],
    ...     "sK": [0.06, 400.0, 35.0],
    ...     "S0_1": [600.0, 45.0, 5890.0],
    ...     "S1_1": [1100.0, -7800.0, 475.0],
    ... })
    >>> flags = detect_bad_wells(ffit, k_min=3.0, k_max=11.0)
    >>> flags[["well", "flag_any"]].values.tolist()
    [['B06', True], ['E10', True], ['A01', False]]
    """
    df = ffit.copy()
    n = len(df)
    result = pd.DataFrame({"well": df["well"]})

    # ---- K at optimizer bound ----
    tol = 1e-6
    result["flag_k_at_bound"] = (np.abs(df["K"] - k_min) < tol * (k_max - k_min)) | (
        np.abs(df["K"] - k_max) < tol * (k_max - k_min)
    )

    # ---- K outlier via MAD ----
    k_median = float(np.nanmedian(df["K"]))
    k_mad = float(np.nanmedian(np.abs(df["K"] - k_median)))
    if k_mad < _NEAR_ZERO:
        k_mad = float(np.nanstd(df["K"]))
    result["flag_k_outlier"] = np.abs(df["K"] - k_median) > k_mad_factor * k_mad

    # ---- Poor fit: relative uncertainty ----
    result["flag_poor_fit"] = (df["sK"] / np.abs(df["K"])) > max_sk_ratio

    # ---- Flat / inverted curves (per label) ----
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
    # Extract label suffix: "S0_y1" -> "y1", "S0_1" -> "1"
    s0_labels = {c[3:] for c in s0_cols}
    s1_labels = {c[3:] for c in s1_cols}
    labels = list(s0_labels & s1_labels)

    flat_any = pd.Series(data=False, index=df.index)
    inverted_any = pd.Series(data=False, index=df.index)

    for lbl in labels:
        s0 = df[f"S0_{lbl}"].astype(float)
        s1 = df[f"S1_{lbl}"].astype(float)
        denom = np.maximum(np.abs(s0), np.abs(s1))
        with np.errstate(invalid="ignore", divide="ignore"):
            dyn = np.where(denom > 0, np.abs(s1 - s0) / denom, 0.0)
        flat_any |= dyn < min_dynamic_range
        if check_polarity:
            if is_ph:
                inverted_any |= s1 < s0  # expect signal to rise with pH
            else:
                inverted_any |= s0 < s1  # expect signal to drop with Cl

    result["flag_flat_curve"] = flat_any
    if check_polarity:
        result["flag_inverted"] = inverted_any

    # ---- High per-well residuals ----
    if residual_stats is not None and "mad" in residual_stats.columns:
        if "well" in residual_stats.columns:
            # Per-well residual stats: aggregate across labels
            well_mad = residual_stats.groupby("well")["mad"].mean()
            plate_median_mad = float(np.nanmedian(well_mad.to_numpy()))
            if plate_median_mad < _NEAR_ZERO:
                plate_median_mad = float(well_mad.max())
            high_resid = well_mad[
                well_mad > residual_mad_factor * plate_median_mad
            ].index
            result["flag_high_residuals"] = result["well"].isin(high_resid)
        else:
            # Plate-level stats only -- flag if any label MAD is extreme
            plate_mad = float(residual_stats["mad"].mean())
            # Cannot flag per-well without well column; skip
            logger.debug(
                "residual_stats lacks 'well' column; plate MAD=%.3f, skipping per-well flag",
                plate_mad,
            )

    # ---- Summary ----
    flag_cols = [c for c in result.columns if c.startswith("flag_")]
    result["flag_count"] = result[flag_cols].sum(axis=1)
    result["flag_any"] = result["flag_count"] > 0

    n_flagged = int(result["flag_any"].sum())
    logger.info(
        "detect_bad_wells: %d/%d wells flagged (%s)",
        n_flagged,
        n,
        ", ".join(f"{c}={result[c].sum()}" for c in flag_cols if c != "flag_any"),
    )

    return result.sort_values("flag_count", ascending=False).reset_index(drop=True)
