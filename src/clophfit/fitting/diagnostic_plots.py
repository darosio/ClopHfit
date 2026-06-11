"""Reusable matplotlib-only diagnostic plots for standardized residuals.

These plots are independent of seaborn and are suitable for both package
tests and manuscript workflows.
"""

from __future__ import annotations

from statistics import NormalDist
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.figure import Figure


def _lag1_residual_correlation(
    residuals: pd.DataFrame,
    *,
    residual_col: str = "std_res",
    label_col: str = "label",
    well_col: str = "well",
    step_col: str = "step",
    x_col: str = "x",
) -> pd.DataFrame:
    """Compute per-well lag-1 residual autocorrelation."""
    rows: list[dict[str, object]] = []

    sort_col = step_col if step_col in residuals.columns else x_col

    for (label, well), g in residuals.groupby([label_col, well_col], observed=True):
        g = g.sort_values(sort_col)
        r = g[residual_col].to_numpy(dtype=float)
        r = r[np.isfinite(r)]

        if len(r) < 3:
            continue

        if np.nanstd(r[:-1]) == 0 or np.nanstd(r[1:]) == 0:
            lag1: float = float("nan")
        else:
            lag1 = float(np.corrcoef(r[:-1], r[1:])[0, 1])

        rows.append(
            {
                label_col: label,
                well_col: well,
                "lag1_corr": lag1,
                "n_points": len(r),
            }
        )

    return pd.DataFrame(rows)


def _normal_qq_points(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (theoretical, ordered) quantiles for a normal Q-Q plot."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    values = np.sort(values)

    n = values.size
    if n == 0:
        return np.array([]), np.array([])

    probs = (np.arange(1, n + 1) - 0.5) / n
    nd = NormalDist()
    theoretical = np.array([nd.inv_cdf(float(p)) for p in probs])
    return theoretical, values


def plot_residual_overview(
    residuals: pd.DataFrame,
    *,
    residual_col: str = "std_res",
    x_col: str = "x",
    label_col: str = "label",
    well_col: str = "well",
    step_col: str = "step",
    output_path: str | Path | None = None,
    title: str | None = None,
    bins: int = 40,
    alpha: float = 0.45,
) -> Figure:
    """Create a generic standardized-residual diagnostic overview.

    Panels: (A) distribution, (B) residuals vs ``x``, (C) lag-1 correlation
    histograms, (D) normal Q-Q plot.  Uses ±2 visual guides.

    The residual column should contain model-standardized residuals::

        std_res = (observed - predicted) / sigma_used_in_fit

    **not** a global z-score of raw residuals.
    """
    required = {residual_col, x_col, label_col, well_col}
    missing = required.difference(residuals.columns)
    if missing:
        msg = f"Missing required residual columns: {sorted(missing)}"
        raise ValueError(msg)

    df = residuals.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[residual_col, x_col, label_col, well_col])

    labels = list(pd.unique(df[label_col]))

    fig = plt.figure(figsize=(16, 12))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    # A. Distribution
    for label in labels:
        vals = df.loc[df[label_col] == label, residual_col].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        ax1.hist(vals, bins=bins, density=True, alpha=alpha, label=f"Label {label}")

    ax1.axvline(0, linestyle="--", color="black", alpha=0.6)
    ax1.axvline(-2, linestyle=":", color="black", alpha=0.4)
    ax1.axvline(2, linestyle=":", color="black", alpha=0.4)
    ax1.set_xlabel("Standardized residual")
    ax1.set_ylabel("Density")
    ax1.set_title("A. Residual distribution")
    ax1.legend()

    # B. Residuals vs x
    for label in labels:
        sub = df[df[label_col] == label]
        ax2.scatter(
            sub[x_col],
            sub[residual_col],
            s=10,
            alpha=0.35,
            label=f"L{label}",
        )

    ax2.axhline(0, linestyle="--", color="black", alpha=0.6)
    ax2.axhline(-2, linestyle=":", color="black", alpha=0.4)
    ax2.axhline(2, linestyle=":", color="black", alpha=0.4)
    ax2.set_xlabel("x / pH")
    ax2.set_ylabel("Standardized residual")
    ax2.set_title("B. Residuals vs x")
    ax2.legend()

    # C. Lag-1 adjacent-point correlation
    lag = _lag1_residual_correlation(
        df,
        residual_col=residual_col,
        label_col=label_col,
        well_col=well_col,
        step_col=step_col,
        x_col=x_col,
    )

    if not lag.empty:
        for label in labels:
            vals = lag.loc[lag[label_col] == label, "lag1_corr"].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            ax3.hist(vals, bins=25, alpha=alpha, label=f"Label {label}")

    ax3.axvline(0, linestyle="--", color="black", alpha=0.6)
    ax3.set_xlabel("Lag-1 residual correlation")
    ax3.set_ylabel("Count")
    ax3.set_title("C. Adjacent-point correlation")
    ax3.legend()

    # D. Q-Q plot
    for label in labels:
        vals = df.loc[df[label_col] == label, residual_col].to_numpy(dtype=float)
        theo, ordered = _normal_qq_points(vals)
        if theo.size == 0:
            continue
        ax4.scatter(theo, ordered, s=10, alpha=0.45, label=f"Label {label}")

    lim = np.nanmax(np.abs(ax4.get_ylim() + ax4.get_xlim()))
    lim = max(float(lim), 3.0)
    ax4.plot([-lim, lim], [-lim, lim], color="black", lw=1, alpha=0.6)
    ax4.axhline(0, linestyle="--", color="black", alpha=0.25)
    ax4.axvline(0, linestyle="--", color="black", alpha=0.25)
    ax4.set_xlabel("Theoretical normal quantiles")
    ax4.set_ylabel("Ordered standardized residuals")
    ax4.set_title("D. Q-Q plot")
    ax4.legend()

    if title:
        fig.suptitle(title, y=1.02)

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig
