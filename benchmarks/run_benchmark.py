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
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Callable, List, Sequence

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
from clophfit.testing.synthetic import make_dataset
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
    residuals: List[float] = field(default_factory=list)
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

    @property
    def residual_stats(self) -> dict[str, float]:
        if not self.residuals:
            return {"shapiro_p": np.nan, "mean": np.nan, "std": np.nan}
        return evaluate_residuals(np.asarray(self.residuals))


def summarize_fitters(
    df: pd.DataFrame,
    residuals: pd.DataFrame | None = None,
) -> dict[str, FitterStats]:
    """Summarize errors and success rates per fitter."""
    stats: dict[str, FitterStats] = {}
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
        if residuals is not None and not residuals.empty:
            method_res = residuals[residuals["method"] == method]
            if not method_res.empty:
                st.residuals.extend(method_res["residual"].dropna().tolist())
        stats[str(method)] = st

    return stats


def log_fitter_summary(stats: dict[str, FitterStats]) -> None:
    """Log per-fitter summary to the console."""
    if not stats:
        logger.info("No fitter statistics available.")
        return

    logger.info("Fitter summary:")
    for name, st in sorted(stats.items(), key=lambda x: x[0]):
        res_stats = st.residual_stats
        logger.info(
            "  %-20s success=%5.1f%%  MAE=%.4f  RMSE=%.4f  median=%.4f  coverage=%5.1f%%  resid_mean=%+.3f std=%.3f shapiro_p=%.3f",
            name,
            st.success_rate,
            st.k_mae,
            st.k_rmse,
            st.k_median,
            st.coverage_rate,
            res_stats.get("mean", np.nan),
            res_stats.get("std", np.nan),
            res_stats.get("shapiro_p", np.nan),
        )


def compare_fitters_statistically(stats: dict[str, FitterStats]) -> pd.DataFrame:
    """Run pairwise statistical comparisons on fitter errors."""
    methods = sorted(stats.keys())
    if len(methods) < 2:
        logger.info("Not enough fitters for statistical comparison.")
        return pd.DataFrame()

    table = pd.DataFrame(0, index=methods, columns=methods, dtype=int)
    logger.info("Pairwise fitter comparisons (Mann-Whitney U on |error|):")
    for a, b in combinations(methods, 2):
        sa, sb = stats[a], stats[b]
        if not sa.k_errors or not sb.k_errors:
            logger.info("  %s vs %s: insufficient data", a, b)
            continue
        result = compare_methods_statistical(
            sa.k_errors, sb.k_errors, method1_name=a, method2_name=b, verbose=False
        )

        score = 0
        if result["significant"]:
            winner = result.get("better_method")
            if winner == a:
                score = 1
            elif winner == b:
                score = -1

        table.loc[a, b] = score
        table.loc[b, a] = -score

        if result["significant"]:
            logger.info("  %s beats %s (p=%.4f)", winner, b if winner == a else a, result["p_value"])
        else:
            logger.info("  %s vs %s: no significant difference (p=%.4f)", a, b, result["p_value"])

    sum_scores = table[methods].sum(axis=1)
    table["sum"] = sum_scores
    table["rank"] = table["sum"].rank(method="dense", ascending=False).astype(int)

    logger.info(
        "Pairwise comparison matrix (+1 better, -1 worse, 0 tie):\n%s",
        table.to_string(),
    )
    return table


def build_stats_df(df_subset: pd.DataFrame, ranking: pd.Series | None = None) -> pd.DataFrame:
    rows = []
    for (method, noise), group in df_subset.groupby(["method", "noise"]):
        rmse = calculate_rmse(group["estimated_k"].values, group["true_k"].iloc[0])
        coverage = group["coverage"].dropna()
        shapiro = group["shapiro_p"].dropna()
        rank = ranking.get(method, np.nan) if isinstance(ranking, pd.Series) else np.nan
        rows.append(
            {
                "method": method,
                "noise": round(float(noise), 4),
                "bias": float(np.mean(group["bias"])),
                "rmse": rmse,
                "coverage": float(coverage.mean()) if not coverage.empty else np.nan,
                "shapiro_p": float(shapiro.mean()) if not shapiro.empty else np.nan,
                "rank": int(rank) if np.isfinite(rank) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize_and_compare(
    label: str,
    df_subset: pd.DataFrame,
    df_residuals: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Log fitter summary and pairwise comparison for one subset."""
    if df_subset.empty:
        logger.info("No results for %s", label)
        return

    residuals_filtered = df_residuals
    if label.startswith("noise="):
        try:
            noise_value = float(label.split("=", 1)[1])
        except ValueError:
            noise_value = None
        if noise_value is not None:
            residuals_filtered = residuals_filtered[residuals_filtered["noise"] == noise_value]

    logger.info("=== Summary for %s ===", label)
    stats = summarize_fitters(df_subset, residuals=residuals_filtered)
    log_fitter_summary(stats)
    table = compare_fitters_statistically(stats)
    suffix = label.replace("=", "_").replace(" ", "_")
    if not table.empty:
        plot_pairwise_matrix(table, output_dir, suffix)
    stats_df = build_stats_df(df_subset, table["rank"] if not table.empty else None)
    plot_dot_grid(stats_df, output_dir, suffix)


def _signal_params(n_labels: int) -> tuple[dict[str, float], dict[str, float]]:
    if n_labels == 1:
        return {"y1": 1000.0}, {"y1": 200.0}
    return {"y1": 1000.0, "y2": 800.0}, {"y1": 200.0, "y2": 300.0}


def _coverage_interval(
    est: float, err: float, *, true: float, z: float = 2.0
) -> tuple[float, float, bool]:
    if not (np.isfinite(est) and np.isfinite(err)):
        return np.nan, np.nan, False
    lower = est - z * err
    upper = est + z * err
    return lower, upper, bool(lower <= true <= upper)


def run_benchmark(
    n_repeats: int,
    noise_levels: List[float],
    n_labels: int,
    outliers: bool,
    output_dir: Path,
    fitters: dict[str, Callable[[Dataset], FitResult[MiniT]]],
    plots_dir: Path,
    seed: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run benchmark for a given configuration."""
    scenario_name = f"{n_labels}label_{'outliers' if outliers else 'clean'}"
    logger.info("Running scenario: %s", scenario_name)
    logger.info("Noise levels: %s", noise_levels)

    true_k = 7.0
    s0, s1 = _signal_params(n_labels)

    results: list[dict[str, object]] = []
    all_residuals: list[dict[str, object]] = []

    seed_counter = 0

    for noise in noise_levels:
        logger.info("Processing noise level: %s", noise)

        for i in range(n_repeats):
            seed_i = None
            if seed is not None:
                seed_counter += 1
                seed_i = seed + seed_counter

            ds_kwargs = {
                "k": true_k,
                "s0": s0,
                "s1": s1,
                "n_labels": n_labels,
                "randomize_signals": True,
                "error_model": "simple",
                "noise": noise,
                "seed": seed_i,
                "low_ph_drop": outliers,
            }
            if outliers:
                ds_kwargs.update(
                    {
                        "low_ph_drop_magnitude": 0.4,
                        "low_ph_drop_label": "y1",
                    }
                )

            ds, _ = make_dataset(**ds_kwargs)

            for name, fitter in fitters.items():
                try:
                    fr = fitter(ds)
                    k_est, k_err = extract_params(fr, "K")

                    if fr.figure:
                        plot_filename = (
                            f"{scenario_name}_noise_{noise}_rep_{i}_{name.replace(' ', '_')}.png"
                        )
                        fr.figure.savefig(plots_dir / plot_filename)
                        plt.close(fr.figure)

                    residuals = None
                    if fr.result is not None:
                        residuals = getattr(fr.result, "residual", None)

                    residuals_stats: dict[str, float] = {}
                    if residuals is not None:
                        residuals_stats = evaluate_residuals(residuals)
                        all_residuals += [
                            {
                                "method": name,
                                "residual": float(r),
                                "noise": noise,
                                "repeat": i,
                            }
                            for r in residuals
                        ]

                    k_lower, k_upper, coverage = _coverage_interval(
                        k_est, k_err, true=true_k
                    )

                    results.append(
                        {
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
                            "success": bool(fr.result is not None and fr.result.success),
                            "shapiro_p": residuals_stats.get("shapiro_p", np.nan),
                        }
                    )
                except Exception:
                    logger.debug(
                        "Error in %s noise %s repeat %s",
                        name,
                        noise,
                        i,
                        exc_info=True,
                    )

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
    noise_levels: List[float],
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
            residual_stats = {
                method: evaluate_residuals(group["residual"].to_numpy())
                for method, group in df_res_noise.groupby("method")
            }
            plot_qq_plots(
                df_res_noise,
                output_dir,
                suffix,
                residual_stats=residual_stats,
            )

        # Coverage per noise
        plot_coverage(df_noise, output_dir, suffix)

    # 3. Coverage
    plot_coverage(df_clean, output_dir, scenario_name)


def plot_trends(df: pd.DataFrame, output_dir: Path, suffix: str) -> None:
    """Plot Bias and RMSE vs Noise in a single figure."""
    rmse_data = []
    for (method, noise), group in df.groupby(["method", "noise"]):
        rmse = calculate_rmse(group["estimated_k"].values, group["true_k"].iloc[0])
        rmse_data.append({"method": method, "noise": noise, "rmse": rmse})

    df_rmse = pd.DataFrame(rmse_data)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True)

    sns.lineplot(data=df, x="noise", y="bias", hue="method", marker="o", ax=axes[0])
    axes[0].axhline(0, color="k", linestyle="--", alpha=0.01)
    axes[0].set_title("Bias vs Noise Level")
    axes[0].set_xlabel("Noise")
    axes[0].set_ylabel("Bias")

    sns.lineplot(data=df_rmse, x="noise", y="rmse", hue="method", marker="o", ax=axes[1])
    axes[1].set_title("RMSE vs Noise Level")
    axes[1].set_xlabel("Noise")
    axes[1].set_ylabel("RMSE")

    plt.tight_layout()
    plt.savefig(output_dir / f"trends_vs_noise_{suffix}.png")
    plt.close()


def plot_dot_grid(stats_df: pd.DataFrame, output_dir: Path, suffix: str) -> None:
    """Build dot plots inspired by seaborn PairGrid dotplot example."""
    if stats_df.empty:
        return

    metrics = ["bias", "rmse", "coverage", "shapiro_p", "rank"]
    melt = stats_df.melt(
        id_vars=["method", "noise"],
        value_vars=[m for m in metrics if m in stats_df.columns],
        var_name="metric",
        value_name="value",
    )

    g = sns.FacetGrid(
        melt,
        row="noise",
        col="metric",
        sharey="row",
        sharex=False,
        height=2.3,
        aspect=1.2,
        margin_titles=True,
    )
    g.map_dataframe(
        sns.stripplot,
        x="value",
        y="method",
        order=sorted(stats_df["method"].unique()),
        orient="h",
        size=7,
        color="tab:blue",
        alpha=0.7,
        jitter=0.15,
    )
    g.fig.subplots_adjust(hspace=0.4)
    plt.tight_layout()
    plt.savefig(output_dir / f"dotgrid_{suffix}.png")
    plt.close()


def plot_pairwise_matrix(table: pd.DataFrame, output_dir: Path, suffix: str) -> None:
    """Visualize pairwise comparison matrix."""
    methods = [idx for idx in table.index if idx in table.columns]
    if not methods:
        return

    matrix = table.loc[methods, methods]
    fig, ax = plt.subplots(
        figsize=(len(methods) * 1.2 + 1, len(methods) * 1.2 + 1)
    )
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="vlag",
        center=0,
        cbar=False,
        linewidths=0.5,
        vmin=-1,
        vmax=1,
        ax=ax,
    )
    ax.set_title("Pairwise Comparison Matrix")
    ax.set_xlabel("Method")
    ax.set_ylabel("Method")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / f"pairwise_matrix_{suffix}.png")
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


def plot_qq_plots(
    df: pd.DataFrame,
    output_dir: Path,
    suffix: str,
    residual_stats: dict[str, dict[str, float]] | None = None,
) -> None:
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
        stats_info = residual_stats.get(method) if residual_stats else {}
        shapiro = stats_info.get("shapiro_p", np.nan)
        ax.text(
            0.05,
            0.95,
            f"$R^2 = {r**2:.4f}$\nShapiro p={shapiro:.3f}",
            transform=ax.transAxes,
            va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

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


def plot_hist_kde_gaussian(x: Sequence[float], **kwargs) -> None:
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
@click.option(
    "--seed",
    type=click.INT,
    default=None,
    show_default=True,
    help="Random seed for reproducibility.",
)
def cli(
    n_repeats: int,
    noise_levels: str,
    labels: int,
    outliers: bool,
    output_dir: str,
    seed: int | None,
) -> None:
    """Run flexible benchmark."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    plots_dir = Path(tempfile.mkdtemp(prefix="fit_plots_", dir=out_path))
    logger.info("Saving individual fit plots to: %s", plots_dir)

    noises = [float(x.strip()) for x in noise_levels.split(",") if x.strip()]

    fitters_dict = build_fitters(include_odr=True)

    # Run benchmark
    df, df_residuals = run_benchmark(
        n_repeats, noises, labels, outliers, out_path, fitters_dict, plots_dir, seed
    )

    scenario_name = f"{labels}label_{'outliers' if outliers else 'clean'}"

    # Save results
    df.to_csv(out_path / f"results_{scenario_name}.csv", index=False)

    # Generate plots
    generate_plots(df, df_residuals, out_path, scenario_name, noises)

    df_clean = df.dropna(subset=["estimated_k"])
    summarize_and_compare("overall", df_clean, df_residuals, out_path)
    for noise in noises:
        summarize_and_compare(
            f"noise={noise}",
            df_clean[df_clean["noise"] == noise],
            df_residuals,
            out_path,
        )

    logger.info("Benchmark complete. Results saved to %s", out_path)
    logger.info("Individual plots are in %s", plots_dir)


if __name__ == "__main__":
    cli()
