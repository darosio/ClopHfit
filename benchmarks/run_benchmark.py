#!/usr/bin/env python
"""Flexible comparison of fitting methods.

-This script compares fitting methods on synthetic and real data using 3 key metrics:
-1. Residual distribution normality (Gaussian) - especially for real data
-2. Bias (synthetic only - requires known true pKa)
-3. 95% CI coverage (pKa uncertainty should ensure 95% coverage)

Supports:
- 1 or 2 labels.
- With or without outliers.
- Noise titration.
- Comprehensive plotting (Distributions, Q-Q with stats, Bias/RMSE vs Noise).
"""

import logging
import tempfile
import warnings
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Callable, Dict, List, Sequence, TYPE_CHECKING

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  # type: ignore[import-untyped]
from scipy import stats

from clophfit.testing.evaluation import (
    calculate_rmse,
    compare_methods_statistical,
    evaluate_residuals,
    extract_params,
)
from clophfit.testing.fitter_test_utils import build_fitters
from clophfit.testing.synthetic import TruthParams, make_dataset
from clophfit.fitting.data_structures import Dataset, FitResult, MiniT

# Configure logging
def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# Suppress warnings for cleaner output
# warnings.filterwarnings("ignore", category=RuntimeWarning)


@dataclass
class FitterStats:
    """Aggregated statistics for a fitter."""

    name: str
    k_errors: List[float] = field(default_factory=list)
    covered: List[bool] = field(default_factory=list)
    n_success: int = 0
    n_total: int = 0

    @property
    def success_rate(self) -> float:
        return 100.0 * self.n_success / self.n_total if self.n_total else 0.0

    @property
    def k_mae(self) -> float:
        if not self.k_errors:
            return float("nan")
        return float(np.mean(np.abs(self.k_errors)))

    @property
    def k_rmse(self) -> float:
        if not self.k_errors:
            return float("nan")
        return float(np.sqrt(np.mean(np.square(self.k_errors))))

    @property
    def k_median(self) -> float:
        if not self.k_errors:
            return float("nan")
        return float(np.median(self.k_errors))

    @property
    def coverage_rate(self) -> float:
        if not self.covered:
            return float("nan")
        return 100.0 * (sum(self.covered) / len(self.covered))


def summarize_fitters(df: pd.DataFrame) -> Dict[str, FitterStats]:
    """Summarize errors and success rates per fitter."""
    stats: Dict[str, FitterStats] = {}
    if df.empty:
        return stats

    for method, group in df.groupby("method"):
        st = FitterStats(name=str(method))
        st.n_total = len(group)
        st.n_success = int(group["success"].sum())
        if "estimated_k" in group and "true_k" in group:
            errs = group["estimated_k"] - group["true_k"]
            errs = errs[np.isfinite(errs)]
            st.k_errors.extend(float(e) for e in errs)
        if "coverage" in group:
            covered = group["coverage"].fillna(False).astype(bool)
            st.covered.extend(bool(c) for c in covered)
        elif {"estimated_k", "k_error", "true_k"}.issubset(group.columns):
            est = group["estimated_k"].to_numpy()
            err = group["k_error"].to_numpy()
            true = group["true_k"].to_numpy()
            valid = np.isfinite(est) & np.isfinite(err) & np.isfinite(true)
            covered = np.abs(est[valid] - true[valid]) <= err[valid]
            st.covered.extend(bool(c) for c in covered)
        stats[str(method)] = st

    return stats


def log_fitter_summary(stats: Dict[str, FitterStats]) -> None:
    """Log per-fitter summary to the console."""
    if not stats:
        logger.info("No fitter statistics available.")
        return

    logger.info("Fitter summary:")
    for name, st in sorted(stats.items(), key=lambda x: x[0]):
        logger.info(
            "  %-20s success=%5.1f%%  MAE=%.4f  RMSE=%.4f  median=%.4f  coverage=%5.1f%%",
            name,
            st.success_rate,
            st.k_mae,
            st.k_rmse,
            st.k_median,
            st.coverage_rate,
        )


def compare_fitters_statistically(stats: Dict[str, FitterStats]) -> None:
    """Run pairwise statistical comparisons on fitter errors."""
    methods = sorted(stats.keys())
    if len(methods) < 2:
        logger.info("Not enough fitters for statistical comparison.")
        return

    logger.info("Pairwise fitter comparisons (Mann-Whitney U on |error|):")
    for a, b in combinations(methods, 2):
        sa, sb = stats[a], stats[b]
        if not sa.k_errors or not sb.k_errors:
            logger.info("  %s vs %s: insufficient data", a, b)
            continue
        result = compare_methods_statistical(
            sa.k_errors, sb.k_errors, method1_name=a, method2_name=b
        )
        logger.info(
            "  %s vs %s: p=%.4f, significant=%s, better=%s",
            a,
            b,
            result["p_value"],
            result["significant"],
            result["better_method"],
        )


def run_benchmark(
    n_repeats: int,
    noise_levels: List[float],
    n_labels: int,
    outliers: bool,
    output_dir: Path,
    fitters: dict[str, Callable[[Dataset], FitResult[MiniT]]],
    plots_dir: Path,
    seed: int|None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run benchmark for a given configuration."""
    scenario_name = f"{n_labels}label_{'outliers' if outliers else 'clean'}"
    logger.info(f"Running scenario: {scenario_name}")
    logger.info(f"Noise levels: {noise_levels}")

    true_k = 7.0
    # Define signal parameters based on n_labels
    if n_labels == 1:
        s0 = {"y1": 1000.0}
        s1 = {"y1": 200.0}
    else:
        s0 = {"y1": 1000.0, "y2": 800.0}
        s1 = {"y1": 200.0, "y2": 300.0}

    results = []
    all_residuals = []

    for noise in noise_levels:
        logger.info(f"Processing noise level: {noise}")

        for i in range(n_repeats):
            if seed:
                seed +=1
            # Configure dataset generation
            ds_kwargs = {
                "k": true_k,
                "s0": s0,
                "s1": s1,
                "n_labels": n_labels,
                "error_model": "simple",
                "noise": noise,
                "seed": seed,
            }

            if outliers:
                ds_kwargs.update({
                    "low_ph_drop": True,
                    "low_ph_drop_magnitude": 0.4,
                    "low_ph_drop_label": "y1"
                })
            else:
                ds_kwargs["low_ph_drop"] = False

            ds, truth = make_dataset(**ds_kwargs)

            for name, fitter in fitters.items():
                try:
                    fr = fitter(ds)
                    k_est, k_err = extract_params(fr, "K")

                    # Save individual fit plot
                    if fr.figure:
                        plot_filename = f"{scenario_name}_noise_{noise}_rep_{i}_{name.replace(' ', '_')}.png"
                        fr.figure.savefig(plots_dir / plot_filename)
                        plt.close(fr.figure)

                    residuals_stats = {}
                    if fr.result and hasattr(fr.result, "residual"):
                        residuals_stats = evaluate_residuals(fr.result.residual)
                        # Collect raw residuals
                        for r in fr.result.residual:
                            all_residuals.append({
                                "method": name,
                                "residual": r,
                                "noise": noise,
                                "repeat": i
                            })

                    k_lower = np.nan
                    k_upper = np.nan
                    coverage = False
                    if np.isfinite(k_est):
                        if np.isfinite(k_err):
                            k_lower = k_est - 2*k_err
                            k_upper = k_est + 2*k_err
                            coverage = k_lower <= true_k <= k_upper

                    results.append({
                        "repeat": i,
                        "method": name,
                        "noise": noise,
                        "true_k": true_k,
                        "estimated_k": k_est,
                        "k_error": k_err,
                        "k_lower": k_lower,
                        "k_upper": k_upper,
                        "coverage": coverage,
                        "bias": k_est - true_k if np.isfinite(k_est) else np.nan,
                        "success": fr.result is not None and fr.result.success,
                        "shapiro_p": residuals_stats.get("shapiro_p", np.nan),
                    })
                except Exception as e:
                    logger.debug(f"Error in {name} noise {noise} repeat {i}: {e}")

    df = pd.DataFrame(results)
    df_residuals = pd.DataFrame(all_residuals)

    # Save residuals
    if not df_residuals.empty:
        df_residuals.to_csv(output_dir / f"residuals_{scenario_name}.csv", index=False)

    return df, df_residuals


def generate_plots(
    df: pd.DataFrame,
    df_residuals: pd.DataFrame,
    output_dir: Path,
    scenario_name: str,
    noise_levels: List[float]
) -> None:
    """Generate all requested plots."""
    df_clean = df.dropna(subset=["estimated_k"])
    if df_clean.empty:
        return

    # 1. Trends vs Noise
    plot_trends(df_clean, output_dir, scenario_name)

    # 2. Distributions per noise level
    for noise in noise_levels:
        suffix = f"{scenario_name}_noise_{noise}"
        df_noise = df_clean[df_clean["noise"] == noise]
        df_res_noise = df_residuals[df_residuals["noise"] == noise] if not df_residuals.empty else pd.DataFrame()

        if df_noise.empty:
            continue

        # K Distribution
        plot_k_distribution(df_noise, output_dir, suffix)

        # Bias Distribution
        plot_bias_distribution(df_noise, output_dir, suffix)

        # Residual Distribution
        if not df_res_noise.empty:
            plot_residual_distribution(df_res_noise, output_dir, suffix)
            plot_qq_plots(df_res_noise, output_dir, suffix)

    # 3. Coverage
    plot_coverage(df_clean, output_dir, scenario_name)


def plot_trends(df: pd.DataFrame, output_dir: Path, suffix: str) -> None:
    """Plot Bias and RMSE vs Noise."""
    # Bias
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="noise", y="bias", hue="method", marker="o")
    plt.axhline(0, color="k", linestyle="--", alpha=0.5)
    plt.title("Bias vs Noise Level")
    plt.savefig(output_dir / f"bias_vs_noise_{suffix}.png")
    plt.close()

    # RMSE
    rmse_data = []
    for (method, noise), group in df.groupby(["method", "noise"]):
        rmse = calculate_rmse(group["estimated_k"].values, group["true_k"].iloc[0])
        rmse_data.append({"method": method, "noise": noise, "rmse": rmse})

    df_rmse = pd.DataFrame(rmse_data)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_rmse, x="noise", y="rmse", hue="method", marker="o")
    plt.title("RMSE vs Noise Level")
    plt.savefig(output_dir / f"rmse_vs_noise_{suffix}.png")
    plt.close()


def plot_k_distribution(df: pd.DataFrame, output_dir: Path, suffix: str) -> None:
    """Violin + Strip + Box plot of K estimates."""
    plt.figure(figsize=(12, 8))
    sns.violinplot(data=df, y="method", x="estimated_k", inner=None, color="0.8")
    sns.stripplot(data=df, y="method", x="estimated_k", size=4, color="black", alpha=0.5)
    sns.boxplot(data=df, y="method", x="estimated_k", width=0.2,
                boxprops={'facecolor':'none', 'edgecolor':'blue'},
                whiskerprops={'color':'blue'}, capprops={'color':'blue'})

    if "true_k" in df.columns:
        plt.axvline(x=df["true_k"].iloc[0], color="r", linestyle="--", label="True K")

    plt.title(f"K Distribution ({suffix})")
    plt.tight_layout()
    plt.savefig(output_dir / f"k_distribution_{suffix}.png")
    plt.close()


def plot_bias_distribution(df: pd.DataFrame, output_dir: Path, suffix: str) -> None:
    """Faceted Histogram + KDE + Gaussian fit of Bias."""
    g = sns.FacetGrid(df, col="method", col_wrap=3, sharex=False, sharey=False, height=4)
    g.map(plot_hist_kde_gaussian, "bias")
    g.set_titles("{col_name}")
    g.savefig(output_dir / f"bias_distribution_{suffix}.png")
    plt.close()


def plot_residual_distribution(df: pd.DataFrame, output_dir: Path, suffix: str) -> None:
    """Grouped KDE of residuals."""
    plt.figure(figsize=(12, 8))
    sns.kdeplot(data=df, x="residual", hue="method", common_norm=False, fill=True, alpha=0.3)
    plt.title(f"Residual Distribution ({suffix})")
    plt.tight_layout()
    plt.savefig(output_dir / f"residual_distribution_{suffix}.png")
    plt.close()


def plot_qq_plots(df: pd.DataFrame, output_dir: Path, suffix: str) -> None:
    """Q-Q plots with R^2 statistic."""
    methods = df["method"].unique()
    n_methods = len(methods)
    cols = 3
    rows = (n_methods + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    axes = axes.flatten()

    for i, method in enumerate(methods):
        ax = axes[i]
        method_res = df[df["method"] == method]["residual"]
        # stats.probplot returns (osm, osr), (slope, intercept, r)
        (osm, osr), (slope, intercept, r) = stats.probplot(method_res, dist="norm", plot=ax)
        ax.set_title(f"Q-Q Plot: {method}")
        # Add R^2 stat
        ax.text(0.05, 0.95, f"$R^2 = {r**2:.4f}$", transform=ax.transAxes, va='top',
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(output_dir / f"residual_qq_{suffix}.png")
    plt.close()


def plot_coverage(df: pd.DataFrame, output_dir: Path, suffix: str) -> None:
    """Plot K estimates per repeat with coverage error bars."""
    df_cov = df.dropna(subset=["estimated_k", "true_k"])
    if df_cov.empty:
        return

    methods = sorted(df_cov["method"].unique())
    n_methods = len(methods)
    fig, axes = plt.subplots(
        n_methods,
        1,
        figsize=(12, max(4, 3 * n_methods)),
        sharex=False,
    )
    if n_methods == 1:
        axes = [axes]
    else:
        axes = list(np.atleast_1d(axes).flatten())

    for ax, method in zip(axes, methods):
        method_df = df_cov[df_cov["method"] == method].sort_values("repeat")
        if method_df.empty:
            continue

        x = np.arange(len(method_df))
        est = method_df["estimated_k"].to_numpy()
        yerr = method_df.get("k_error", pd.Series(np.zeros(len(method_df))))
        if isinstance(yerr, pd.Series):
            yerr = yerr.fillna(0).to_numpy()
        else:
            yerr = np.nan_to_num(yerr, nan=0.0)

        if yerr.any():
            ax.errorbar(
                x,
                est,
                yerr=yerr,
                fmt="none",
                ecolor="gray",
                alpha=0.4,
                capsize=3,
            )

        coverage = method_df.get("coverage")
        if coverage is not None:
            covered = coverage.fillna(False).astype(bool).to_numpy()
        else:
            covered = np.full(len(est), True, dtype=bool)

        ax.scatter(
            x[covered],
            est[covered],
            color="tab:green",
            edgecolors="k",
            label="covered",
            zorder=3,
        )
        if (~covered).any():
            ax.scatter(
                x[~covered],
                est[~covered],
                facecolors="none",
                edgecolors="red",
                s=80,
                label="missed coverage",
                zorder=4,
            )

        true_k = method_df["true_k"].iloc[0]
        ax.axhline(true_k, color="black", linestyle="--", label="True K")
        ax.set_title(f"{method} - Coverage")
        ax.set_ylabel("Estimated K")
        ax.set_xlim(-0.5, len(method_df) - 0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(method_df["repeat"].astype(int), rotation=45)
        ax.legend(loc="best")

    axes[-1].set_xlabel("Repeat")
    plt.tight_layout()
    plt.savefig(output_dir / f"coverage_{suffix}.png")
    plt.close()


def plot_hist_kde_gaussian(x: Sequence[float], **kwargs)->None:
    """Helper to plot histogram, KDE and Gaussian fit."""
    vals = np.asarray(x)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return
    ax = plt.gca()
    sns.histplot(vals, kde=True, stat="density", ax=ax, alpha=0.4, label="Data")
    if len(vals) > 1:
        mu, std = stats.norm.fit(vals)
        xmin, xmax = ax.get_xlim()
        x_plot = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x_plot, mu, std)
        # Fixed escape sequences for mu and sigma
        ax.plot(x_plot, p, 'r--', linewidth=2, label=f'Normal\n$\\mu={mu:.2e}$\n$\\sigma={std:.2e}$')
        ax.legend()


@click.command()
@click.option("--n-repeats", default=20, help="Number of repeats per noise level.")
@click.option("--noise-levels", default="0.0,0.01,0.05,0.1", help="Comma-separated noise levels.")
@click.option("--labels", default=1, help="Number of labels (1 or 2).")
@click.option("--outliers/--no-outliers", default=False, help="Include outliers.")
@click.option("--output-dir", default="benchmarks", help="Output directory.")
@click.option("--seed", type=click.INT, default=None, show_default=True, help="Random seed for reproducibility.")
def cli(n_repeats: int, noise_levels: str, labels: int, outliers: bool, output_dir: str, seed: int) -> None:
    """Run flexible benchmark."""
    out_path = Path(output_dir)
    out_path.mkdir(exist_ok=True)
    # Create temporary directory for plots
    plots_dir = Path(tempfile.mkdtemp(prefix="fit_plots_", dir=out_path))
    logger.info(f"Saving individual fit plots to: {plots_dir}")
    # Parse noise levels
    noises = [float(x) for x in noise_levels.split(",")]
    # Define fitters using shared builder to avoid duplication
    fitters_dict = build_fitters(include_odr=True)
    fitters = list(fitters_dict.items())

    # Run benchmark
    df, df_residuals = run_benchmark(
        n_repeats, noises, labels, outliers, out_path, fitters_dict, plots_dir, seed
    )

    scenario_name = f"{labels}label_{'outliers' if outliers else 'clean'}"

    # Save results
    df.to_csv(out_path / f"results_{scenario_name}.csv", index=False)

    # Generate plots
    generate_plots(df, df_residuals, out_path, scenario_name, noises)

    # Log aggregated stats
    summary = summarize_fitters(df)
    log_fitter_summary(summary)
    compare_fitters_statistically(summary)

    logger.info(f"Benchmark complete. Results saved to {out_path}")
    logger.info(f"Individual plots are in {plots_dir}")


if __name__ == "__main__":
    cli()
