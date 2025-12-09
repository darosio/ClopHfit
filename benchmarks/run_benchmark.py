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
import shutil
import tempfile
import warnings
from pathlib import Path
from typing import List

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from clophfit.fitting.core import (
    fit_binding_glob,
    fit_binding_glob_recursive,
    outlier2,
)
from clophfit.fitting.odr import fit_binding_odr
from clophfit.testing.evaluation import (
    calculate_bias,
    calculate_coverage,
    calculate_rmse,
    evaluate_residuals,
    extract_params,
)
from clophfit.testing.synthetic import make_dataset

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


def run_benchmark(
    n_repeats: int,
    noise_levels: List[float],
    n_labels: int,
    outliers: bool,
    output_dir: Path,
    fitters: List[tuple],
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

            for name, fitter in fitters:
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

                    results.append({
                        "repeat": i,
                        "method": name,
                        "noise": noise,
                        "true_k": true_k,
                        "estimated_k": k_est,
                        "k_error": k_err,
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
    """Bar plot of coverage probability."""
    # Calculate coverage per method
    coverage_data = []
    for method, group in df.groupby("method"):
        cov = calculate_coverage(group["estimated_k"].values, group["k_error"].values, group["true_k"].iloc[0])
        coverage_data.append({"method": method, "coverage": cov})

    df_cov = pd.DataFrame(coverage_data)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_cov, x="method", y="coverage")
    plt.axhline(y=0.95, color="r", linestyle="--", label="Target (0.95)")
    plt.title(f"Coverage Probability ({suffix})")
    plt.ylim(0, 1.05)
    plt.savefig(output_dir / f"coverage_{suffix}.png")
    plt.close()


def plot_hist_kde_gaussian(x, **kwargs):
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

    # Define fitters
    fitters = [
        ("LM Standard", lambda d: fit_binding_glob(d, robust=False)),
        ("LM Robust", lambda d: fit_binding_glob(d, robust=True)),
        ("ODR", fit_binding_odr),
        # ("IRLS", fit_binding_glob_recursive),
        ("Outlier2", lambda d: outlier2(d, error_model="uniform")),
    ]

    # Run benchmark
    df, df_residuals = run_benchmark(
        n_repeats, noises, labels, outliers, out_path, fitters, plots_dir, seed
    )

    scenario_name = f"{labels}label_{'outliers' if outliers else 'clean'}"

    # Save results
    df.to_csv(out_path / f"results_{scenario_name}.csv", index=False)

    # Generate plots
    generate_plots(df, df_residuals, out_path, scenario_name, noises)

    logger.info(f"Benchmark complete. Results saved to {out_path}")
    logger.info(f"Individual plots are in {plots_dir}")


if __name__ == "__main__":
    cli()
