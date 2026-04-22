#!/usr/bin/env python3
"""Compare fit-method and MCMC-mode results across L2 and L4 plates.

Reads ffit{0..4}.csv produced by ppr and assembles side-by-side K comparisons.
Run AFTER scripts/compare_methods.sh has completed.

Usage
-----
    python scripts/compare_methods.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

BASE_DIRS = {
    "L2": Path("/home/dati/arslanbaeva/data/raw/L2"),
    "L4": Path("/home/dati/arslanbaeva/data/raw/L4"),
}

FIT_METHODS = ["lm", "huber", "irls", "wls", "iterative", "outlier"]
MCMC_MODES = ["single", "multi", "multi_noise", "multi_noise_xrw"]

LB_LABELS = {
    0: "lb0 (y1 only)",
    1: "lb1 (y2 only)",
    2: "lb2 (global)",
    3: "lb3 (ODR)",
    4: "lb4 (MCMC)",
}


def _ffit_path(base: Path, compare_dir: str, lb: int) -> Path | None:
    """Return path to ffit{lb}.csv or None if not found."""
    for subfolder in base.rglob(f"compare/{compare_dir}/**/fit/ffit{lb}.csv"):
        return subfolder
    return None


def load_k(base: Path, compare_dir: str, lb: int) -> pd.Series | None:
    """Load K column from ffit{lb}.csv; return Series indexed by well."""
    path = _ffit_path(base, compare_dir, lb)
    if path is None:
        return None
    df = pd.read_csv(path, usecols=["well", "K", "sK"]).set_index("well")
    return df["K"]


def load_k_with_err(base: Path, compare_dir: str, lb: int) -> pd.DataFrame | None:
    """Load K ± sK from ffit{lb}.csv; return DataFrame indexed by well."""
    path = _ffit_path(base, compare_dir, lb)
    if path is None:
        return None
    df = pd.read_csv(path, usecols=["well", "K", "sK"]).set_index("well")
    return df


def compare_fit_methods(plate: str, base: Path, lb: int = 2) -> pd.DataFrame:
    """Build DataFrame of K values per well for each fit method."""
    rows = {}
    for method in FIT_METHODS:
        series = load_k(base, method, lb)
        if series is not None:
            rows[method] = series
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def compare_mcmc_modes(plate: str, base: Path) -> pd.DataFrame:
    """Build DataFrame of K values per well for each MCMC mode (lb4)."""
    rows = {}
    for mode in MCMC_MODES:
        series = load_k(base, f"mcmc_{mode}", lb=4)
        if series is not None:
            rows[mode.replace("_", "-")] = series
    # also include the huber global fit for reference
    ref = load_k(base, "huber", lb=2)
    if ref is not None:
        rows["huber-lb2"] = ref
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def plot_comparison(df: pd.DataFrame, title: str, outpath: Path) -> None:
    """Dot-plot: one row per well, columns are methods."""
    if df.empty:
        print(f"  [skip] no data for {title}")
        return
    wells = df.index.tolist()
    n_wells = len(wells)
    n_methods = len(df.columns)
    fig, ax = plt.subplots(figsize=(max(8, n_methods * 1.5), max(6, n_wells * 0.35)))
    y = np.arange(n_wells)
    colors = plt.cm.tab10(np.linspace(0, 1, n_methods))  # type: ignore[attr-defined]
    for i, col in enumerate(df.columns):
        ax.scatter(df[col], y + i * 0.06 - (n_methods * 0.03), s=30,
                   label=col, color=colors[i], zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(wells, fontsize=7)
    ax.set_xlabel("K (pH units)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(axis="x", alpha=0.3)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)
    print(f"  saved → {outpath}")


def print_summary(df: pd.DataFrame, title: str) -> None:
    """Print mean ± std for each method."""
    if df.empty:
        print(f"[{title}] no data")
        return
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    stats = pd.DataFrame({
        "mean_K": df.mean(),
        "std_K": df.std(),
        "min_K": df.min(),
        "max_K": df.max(),
    })
    print(stats.to_string(float_format="{:.4f}".format))


def run() -> None:
    """Entry point."""
    out_root = Path("/tmp/clophfit_compare")
    out_root.mkdir(exist_ok=True)
    any_data = False

    for plate, base in BASE_DIRS.items():
        print(f"\n{'#'*60}")
        print(f"# Plate: {plate}  ({base})")
        print(f"{'#'*60}")

        # ---- Fit method comparison (lb2 = global) ----
        for lb, lb_label in LB_LABELS.items():
            if lb == 4:
                continue  # MCMC handled separately
            df_fit = compare_fit_methods(plate, base, lb=lb)
            if not df_fit.empty:
                any_data = True
                title = f"{plate} | fit-methods | {lb_label}"
                print_summary(df_fit, title)
                plot_comparison(
                    df_fit,
                    title,
                    out_root / f"{plate}_fit_methods_{lb_label.split()[0]}.png",
                )

        # ---- MCMC mode comparison ----
        df_mcmc = compare_mcmc_modes(plate, base)
        if not df_mcmc.empty:
            any_data = True
            title = f"{plate} | MCMC modes | lb4 vs huber-lb2"
            print_summary(df_mcmc, title)
            plot_comparison(df_mcmc, title, out_root / f"{plate}_mcmc_modes.png")

    if not any_data:
        print("\n⚠  No output CSV files found. Run scripts/compare_methods.sh first.")
        sys.exit(1)

    print(f"\n✅ Comparison plots saved to {out_root}/")


if __name__ == "__main__":
    run()
