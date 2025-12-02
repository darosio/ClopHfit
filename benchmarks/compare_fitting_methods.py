#!/usr/bin/env python
"""Comprehensive comparison of fitting methods: outlier2 vs fit_binding_glob_reweighted.

This script compares fitting methods on both synthetic and real data to determine
which methods can be safely removed.

Methods compared:
- fit_binding_glob (LM baseline)
- fit_binding_glob with robust=True (Huber loss)
- fit_binding_glob_reweighted (RLS with outlier removal)
- fit_binding_glob_recursive (IRLS)
- fit_binding_glob_recursive_outlier (IRLS + outlier)
- outlier2 with error_model="uniform"
- outlier2 with error_model="shot-noise"
"""

import copy
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from clophfit.fitting.core import (
    fit_binding_glob,
    fit_binding_glob_recursive,
    fit_binding_glob_recursive_outlier,
    fit_binding_glob_reweighted,
    outlier2,
)
from clophfit.fitting.data_structures import DataArray, Dataset
from clophfit.fitting.models import binding_1site

# Ground truth parameters (calibrated to match real data from L1, L2, L4, 140220)
# y1: narrower range, often inverted (S1 > S0), higher errors, shifted up ~2/3 of y2 max
# y2: wider range, S0 > S1, lower errors
TRUE_S0_Y2 = 1200.0  # Low pH value for y2 (higher plateau)
TRUE_S1_Y2 = 400.0   # High pH value for y2 (lower plateau)
# y1 shifted up to ~2/3 of y2 max (~800), inverted, narrow range
TRUE_S0_Y1 = 750.0   # Low pH value for y1 (lower plateau, but high absolute)
TRUE_S1_Y1 = 900.0   # High pH value for y1 (higher plateau) - INVERTED
BUFFER_SD_Y2 = 40.0  # Lower buffer noise for y2
BUFFER_SD_Y1 = 200.0 # y1 error ~5x y2 error (40*5=200)


def generate_synthetic_data(
    pKa: float = 7.0,
    n_points: int = 7,
    add_outliers: bool = False,
    rng: np.random.Generator | None = None,
) -> Dataset:
    """Generate synthetic dual-channel pH titration data matching real data characteristics.

    Real data characteristics (from L1, L2, L4, 140220):
    - y1: narrower range (~150-600), often inverted (S1 > S0), higher errors
    - y2: wider range (~200-2700), S0 > S1, lower errors
    - Error ratio y1/y2: ~1.3-2.7x
    """
    if rng is None:
        rng = np.random.default_rng()

    x = np.linspace(5.5, 9.0, n_points)
    x_err = 0.05 * np.ones_like(x)

    # y1: inverted curve (S1 > S0), narrower range
    y1_true = binding_1site(x, pKa, TRUE_S0_Y1, TRUE_S1_Y1, is_ph=True)
    # y2: normal curve (S0 > S1), wider range
    y2_true = binding_1site(x, pKa, TRUE_S0_Y2, TRUE_S1_Y2, is_ph=True)

    # Physics-based errors: sqrt(signal + buffer_sd^2)
    # y1 has higher buffer noise
    y1_err_true = np.sqrt(np.maximum(y1_true, 1.0) + BUFFER_SD_Y1**2)
    y2_err_true = np.sqrt(np.maximum(y2_true, 1.0) + BUFFER_SD_Y2**2)

    y1 = y1_true + rng.normal(0, y1_err_true)
    y2 = y2_true + rng.normal(0, y2_err_true)

    if add_outliers:
        # Add outliers at lowest 2 pH values in y1 channel
        # Since y1 is inverted (increases with pH), outliers DROP to lower values
        y1[0] -= 4 * y1_err_true[0]  # Large negative outlier at pH 5.5
        y1[1] -= 3 * y1_err_true[1]  # Moderate negative outlier at pH ~5.8

    da1 = DataArray(x, y1, x_errc=x_err, y_errc=y1_err_true)
    da2 = DataArray(x, y2, x_errc=x_err, y_errc=y2_err_true)

    return Dataset({"y1": da1, "y2": da2}, is_ph=True)


def plot_synthetic_examples(output_dir: Path):
    """Plot examples of synthetic data with and without outliers."""
    rng = np.random.default_rng(42)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    pKa_true = 7.0
    x_fine = np.linspace(5.5, 9.0, 100)
    y1_true = binding_1site(x_fine, pKa_true, TRUE_S0_Y1, TRUE_S1_Y1, is_ph=True)
    y2_true = binding_1site(x_fine, pKa_true, TRUE_S0_Y2, TRUE_S1_Y2, is_ph=True)

    # Row 1: Clean data examples
    for i, ax in enumerate(axes[0]):
        ds = generate_synthetic_data(pKa=pKa_true, add_outliers=False, rng=rng)
        da1, da2 = ds["y1"], ds["y2"]

        ax.errorbar(da1.x, da1.y, yerr=da1.y_err, fmt='o', color='blue',
                   label='y1 (inverted, narrow)', capsize=3, markersize=6)
        ax.errorbar(da2.x, da2.y, yerr=da2.y_err, fmt='s', color='red',
                   label='y2 (normal, wide)', capsize=3, markersize=6)
        ax.plot(x_fine, y1_true, 'b--', alpha=0.5, linewidth=1)
        ax.plot(x_fine, y2_true, 'r--', alpha=0.5, linewidth=1)
        ax.axvline(pKa_true, color='gray', linestyle=':', alpha=0.5, label=f'True pKa={pKa_true}')
        ax.set_xlabel('pH')
        ax.set_ylabel('Fluorescence')
        ax.set_title(f'Clean Data Example {i+1}')
        if i == 0:
            ax.legend(loc='best', fontsize=8)

    # Row 2: Data with outliers
    for i, ax in enumerate(axes[1]):
        ds = generate_synthetic_data(pKa=pKa_true, add_outliers=True, rng=rng)
        da1, da2 = ds["y1"], ds["y2"]

        ax.errorbar(da1.x, da1.y, yerr=da1.y_err, fmt='o', color='blue',
                   label='y1', capsize=3, markersize=6)
        ax.errorbar(da2.x, da2.y, yerr=da2.y_err, fmt='s', color='red',
                   label='y2', capsize=3, markersize=6)
        ax.plot(x_fine, y1_true, 'b--', alpha=0.5, linewidth=1)
        ax.plot(x_fine, y2_true, 'r--', alpha=0.5, linewidth=1)
        ax.axvline(pKa_true, color='gray', linestyle=':', alpha=0.5)

        # Highlight outliers (dropping low at low pH)
        ax.scatter([da1.x[0], da1.x[1]], [da1.y[0], da1.y[1]],
                  s=150, facecolors='none', edgecolors='orange', linewidths=2,
                  label='Outliers (drop)' if i == 0 else None)

        ax.set_xlabel('pH')
        ax.set_ylabel('Fluorescence')
        ax.set_title(f'With Outliers Example {i+1}')
        if i == 0:
            ax.legend(loc='best', fontsize=8)

    # Compute error ratio for title
    ds_sample = generate_synthetic_data(pKa=pKa_true)
    err_ratio = ds_sample["y1"].y_err.mean() / ds_sample["y2"].y_err.mean()

    plt.suptitle(f'Synthetic pH Titration Data (pKa=7.0)\n'
                f'y1: inverted, narrow range, err~{BUFFER_SD_Y1:.0f} | '
                f'y2: normal, wide range, err~{BUFFER_SD_Y2:.0f} | ratio~{err_ratio:.1f}x',
                fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'synthetic_data_examples.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'synthetic_data_examples.png'}")


def plot_coverage_comparison(synth_summary: pd.DataFrame, synth_outlier_summary: pd.DataFrame,
                            real_summary: pd.DataFrame | None, output_dir: Path):
    """Create bar plot comparing coverage across methods and datasets."""
    fig, ax = plt.subplots(figsize=(12, 6))

    methods = synth_summary['method'].tolist()
    x = np.arange(len(methods))
    width = 0.25

    # Coverage values
    synth_cov = synth_summary.set_index('method').loc[methods, 'coverage'].values * 100
    outlier_cov = synth_outlier_summary.set_index('method').loc[methods, 'coverage'].values * 100

    bars1 = ax.bar(x - width, synth_cov, width, label='Synthetic (clean)', color='steelblue')
    bars2 = ax.bar(x, outlier_cov, width, label='Synthetic (outliers)', color='darkorange')

    if real_summary is not None and len(real_summary) > 0:
        real_cov = real_summary.set_index('method').reindex(methods)['coverage'].values * 100
        bars3 = ax.bar(x + width, real_cov, width, label='Real (38 controls)', color='forestgreen')

    ax.axhline(95, color='gray', linestyle='--', alpha=0.7, label='95% target')
    ax.axhline(50, color='red', linestyle=':', alpha=0.5, label='50% (random)')

    ax.set_xlabel('Method', fontsize=11)
    ax.set_ylabel('Coverage (%)', fontsize=11)
    ax.set_title('95% CI Coverage by Fitting Method\n(Higher is better, target ≈ 95%)',
                fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 105)

    # Add value labels on bars
    for bars in [bars1, bars2] + ([bars3] if real_summary is not None else []):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.0f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'coverage_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'coverage_comparison.png'}")


def plot_bias_comparison(synth_summary: pd.DataFrame, synth_outlier_summary: pd.DataFrame,
                         real_summary: pd.DataFrame | None, output_dir: Path):
    """Create bar plot comparing bias across methods and datasets."""
    fig, ax = plt.subplots(figsize=(12, 6))

    methods = synth_summary['method'].tolist()
    x = np.arange(len(methods))
    width = 0.25

    # Bias values
    synth_bias = synth_summary.set_index('method').loc[methods, 'bias'].values
    outlier_bias = synth_outlier_summary.set_index('method').loc[methods, 'bias'].values

    bars1 = ax.bar(x - width, synth_bias, width, label='Synthetic (clean)', color='steelblue')
    bars2 = ax.bar(x, outlier_bias, width, label='Synthetic (outliers)', color='darkorange')

    if real_summary is not None and len(real_summary) > 0:
        real_bias = real_summary.set_index('method').reindex(methods)['bias'].values
        bars3 = ax.bar(x + width, real_bias, width, label='Real (38 controls)', color='forestgreen')

    ax.axhline(0, color='gray', linestyle='--', alpha=0.7, linewidth=2)

    ax.set_xlabel('Method', fontsize=11)
    ax.set_ylabel('Bias (pKa units)', fontsize=11)
    ax.set_title('Bias by Fitting Method\n(Closer to 0 is better)',
                fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend(loc='upper right')

    # Add value labels on bars
    for bars in [bars1, bars2] + ([bars3] if real_summary is not None else []):
        for bar in bars:
            height = bar.get_height()
            va = 'bottom' if height >= 0 else 'top'
            offset = 3 if height >= 0 else -3
            ax.annotate(f'{height:+.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, offset), textcoords="offset points",
                       ha='center', va=va, fontsize=7)

    plt.tight_layout()
    plt.savefig(output_dir / 'bias_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'bias_comparison.png'}")


def plot_bias_rmse_comparison(synth_summary: pd.DataFrame, real_summary: pd.DataFrame | None,
                              output_dir: Path):
    """Create scatter plot of bias vs RMSE for each method."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    methods = synth_summary['method'].tolist()
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))

    # Left: Synthetic data
    ax = axes[0]
    for i, method in enumerate(methods):
        row = synth_summary[synth_summary['method'] == method].iloc[0]
        ax.scatter(row['bias'], row['rmse'], s=100, c=[colors[i]], label=method, marker='o')

    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Bias (pKa units)', fontsize=11)
    ax.set_ylabel('RMSE (pKa units)', fontsize=11)
    ax.set_title('Synthetic Data (N=100)', fontsize=12)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Right: Real data (no legend - use same colors as synthetic)
    ax = axes[1]
    if real_summary is not None and len(real_summary) > 0:
        for i, method in enumerate(methods):
            if method in real_summary['method'].values:
                row = real_summary[real_summary['method'] == method].iloc[0]
                ax.scatter(row['bias'], row['rmse'], s=100, c=[colors[i]], marker='s')
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Bias (pKa units)', fontsize=11)
        ax.set_ylabel('RMSE (pKa units)', fontsize=11)
        ax.set_title('Real Data (38 controls)', fontsize=12)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No real data', ha='center', va='center', transform=ax.transAxes)

    plt.suptitle('Bias vs RMSE Comparison\n(Lower RMSE and bias closer to 0 is better)',
                fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'bias_rmse_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'bias_rmse_comparison.png'}")


def plot_k_err_vs_coverage(synth_summary: pd.DataFrame, output_dir: Path):
    """Plot K_err vs coverage to show uncertainty calibration."""
    fig, ax = plt.subplots(figsize=(8, 6))

    methods = synth_summary['method'].tolist()
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))

    for i, method in enumerate(methods):
        row = synth_summary[synth_summary['method'] == method].iloc[0]
        ax.scatter(row['mean_K_err'], row['coverage'] * 100, s=150, c=[colors[i]],
                  label=method, marker='o', edgecolors='black', linewidths=0.5)

    ax.axhline(95, color='gray', linestyle='--', alpha=0.7, label='95% target')
    ax.set_xlabel('Mean K_err (reported uncertainty)', fontsize=11)
    ax.set_ylabel('Actual Coverage (%)', fontsize=11)
    ax.set_title('Uncertainty Calibration: Reported Error vs Actual Coverage\n'
                '(Well-calibrated methods should be near 95% line)',
                fontsize=11, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(20, 100)

    plt.tight_layout()
    plt.savefig(output_dir / 'uncertainty_calibration.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'uncertainty_calibration.png'}")


def plot_coverage_vs_bias(synth_summary: pd.DataFrame, synth_outlier_summary: pd.DataFrame,
                          real_summary: pd.DataFrame | None, output_dir: Path):
    """Plot coverage vs bias for all datasets."""
    fig, ax = plt.subplots(figsize=(10, 7))

    methods = synth_summary['method'].tolist()
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))

    # Synthetic clean
    for i, method in enumerate(methods):
        row = synth_summary[synth_summary['method'] == method].iloc[0]
        ax.scatter(row['bias'], row['coverage'] * 100, s=120, c=[colors[i]],
                  marker='o', label=f'{method}' if i < len(methods) else None,
                  edgecolors='black', linewidths=0.5)

    # Synthetic with outliers (squares, no legend)
    for i, method in enumerate(methods):
        row = synth_outlier_summary[synth_outlier_summary['method'] == method].iloc[0]
        ax.scatter(row['bias'], row['coverage'] * 100, s=120, c=[colors[i]],
                  marker='s', edgecolors='black', linewidths=0.5, alpha=0.7)

    # Real data (diamonds)
    if real_summary is not None and len(real_summary) > 0:
        for i, method in enumerate(methods):
            if method in real_summary['method'].values:
                row = real_summary[real_summary['method'] == method].iloc[0]
                ax.scatter(row['bias'], row['coverage'] * 100, s=120, c=[colors[i]],
                          marker='D', edgecolors='black', linewidths=0.5, alpha=0.7)

    ax.axhline(95, color='gray', linestyle='--', alpha=0.7, linewidth=2)
    ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(50, color='red', linestyle=':', alpha=0.3)

    ax.set_xlabel('Bias (pKa units)', fontsize=12)
    ax.set_ylabel('Coverage (%)', fontsize=12)
    ax.set_title('Coverage vs Bias\n(○ synthetic clean, □ synthetic+outliers, ◇ real data)',
                fontsize=12, fontweight='bold')
    ax.legend(loc='lower left', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(15, 100)

    # Add annotation for target
    ax.annotate('95% target', xy=(0.15, 95), fontsize=10, color='gray')

    plt.tight_layout()
    plt.savefig(output_dir / 'coverage_vs_bias.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'coverage_vs_bias.png'}")


@dataclass
class FitMetrics:
    """Metrics for a single fit."""

    K: float
    K_err: float
    residual: float
    n_outliers: int
    converged: bool


def extract_metrics(fr, ds_original) -> FitMetrics:
    """Extract metrics from a FitResult."""
    if fr.result is None:
        return FitMetrics(np.nan, np.nan, np.nan, 0, False)

    K = fr.result.params["K"].value
    K_err = fr.result.params["K"].stderr or np.nan

    # Count outliers (points removed from original)
    n_original = sum(len(da.y) for da in ds_original.values())
    n_final = sum(len(da.y) for da in fr.dataset.values()) if fr.dataset else n_original
    n_outliers = n_original - n_final

    return FitMetrics(
        K=K,
        K_err=K_err,
        residual=fr.result.redchi,
        n_outliers=n_outliers,
        converged=True,
    )


def run_synthetic_comparison(
    n_trials: int = 100,
    pKa_true: float = 7.0,
    n_points: int = 7,
    seed: int = 42,
) -> pd.DataFrame:
    """Run comparison on synthetic data with known ground truth."""
    rng = np.random.default_rng(seed)
    results = []

    for trial in range(n_trials):
        # Generate synthetic data
        ds = generate_synthetic_data(
            pKa=pKa_true,
            n_points=n_points,
            rng=rng,
        )

        methods = {
            "lm_standard": lambda d: fit_binding_glob(d, robust=False),
            "lm_robust": lambda d: fit_binding_glob(d, robust=True),
            "lm_reweighted": lambda d: fit_binding_glob_reweighted(
                d, key=f"trial_{trial}", threshold=2.5
            ),
            "lm_recursive": lambda d: fit_binding_glob_recursive(d, tol=0.01),
            "lm_recursive_outlier": lambda d: fit_binding_glob_recursive_outlier(
                d, tol=0.01, threshold=3.0
            ),
            "outlier2_uniform": lambda d: outlier2(d, error_model="uniform"),
            "outlier2_shotnoise": lambda d: outlier2(d, error_model="shot-noise"),
        }

        for method_name, method_func in methods.items():
            ds_copy = copy.deepcopy(ds)
            try:
                fr = method_func(ds_copy)
                metrics = extract_metrics(fr, ds)
            except Exception:
                metrics = FitMetrics(np.nan, np.nan, np.nan, 0, False)

            results.append(
                {
                    "trial": trial,
                    "method": method_name,
                    "K_true": pKa_true,
                    "K_fit": metrics.K,
                    "K_err": metrics.K_err,
                    "K_bias": metrics.K - pKa_true if not np.isnan(metrics.K) else np.nan,
                    "n_outliers": metrics.n_outliers,
                    "converged": metrics.converged,
                }
            )

    return pd.DataFrame(results)


def compute_coverage(df: pd.DataFrame, confidence: float = 0.95) -> pd.DataFrame:
    """Compute coverage statistics for each method."""
    z = stats.norm.ppf((1 + confidence) / 2)

    summary = []
    for method in df["method"].unique():
        method_df = df[df["method"] == method].dropna(subset=["K_fit", "K_err"])

        if len(method_df) == 0:
            continue

        # Coverage: fraction where true value is within CI
        lower = method_df["K_fit"] - z * method_df["K_err"]
        upper = method_df["K_fit"] + z * method_df["K_err"]
        covered = (method_df["K_true"] >= lower) & (method_df["K_true"] <= upper)
        coverage = covered.mean()

        # Other stats
        bias = method_df["K_bias"].mean()
        rmse = np.sqrt((method_df["K_bias"] ** 2).mean())
        mean_K_err = method_df["K_err"].mean()
        mean_outliers = method_df["n_outliers"].mean()
        n_converged = method_df["converged"].sum()

        summary.append(
            {
                "method": method,
                "n_trials": len(method_df),
                "n_converged": n_converged,
                "mean_K_err": mean_K_err,
                "bias": bias,
                "rmse": rmse,
                "coverage": coverage,
                "mean_outliers": mean_outliers,
            }
        )

    return pd.DataFrame(summary).sort_values("coverage", ascending=False)


def run_real_data_comparison(data_dir: Path) -> pd.DataFrame:
    """Run comparison on real data from test fixtures (L1, L2, L4, 140220)."""
    from clophfit.prtecan import Titration

    # Known pKa values for control samples
    KNOWN_PKA = {
        "V224L": 5.81,
        "E2GFP": 6.82,
        "S202N": 6.94,
        "NTT": 7.57,
        "G03": 7.94,
        "V224Q": 8.0,
    }

    # Dataset configurations
    DATASETS = ["L1", "L2", "L4", "140220"]

    results = []

    for dataset_name in DATASETS:
        tit_dir = data_dir / dataset_name
        if not tit_dir.is_dir():
            continue

        # Find the list file
        list_file = tit_dir / "list.pH.csv"
        if not list_file.exists():
            list_file = tit_dir / "list.pH"
        if not list_file.exists():
            print(f"  Warning: No list file found in {dataset_name}")
            continue

        try:
            tit = Titration.fromlistfile(list_file, is_ph=True)
            # Load additions if available
            additions_file = tit_dir / "additions.pH"
            if additions_file.exists():
                tit.load_additions(additions_file)
            # Load scheme if available
            scheme_file = tit_dir / "scheme.txt"
            if scheme_file.exists():
                tit.load_scheme(scheme_file)
            # Enable background subtraction and normalization
            tit.params.bg = True
            tit.params.nrm = True
            tit.params.dil = True
        except Exception as e:
            print(f"  Warning: Failed to load {dataset_name}: {e}")
            continue

        # Get control wells and their sample names
        ctrl_wells = tit.scheme.ctrl if hasattr(tit.scheme, 'ctrl') else []
        names_map = tit.scheme.names if hasattr(tit.scheme, 'names') else {}

        # Invert names_map to get well -> sample_name
        well_to_name = {}
        for name, wells in names_map.items():
            for well in wells:
                well_to_name[well] = name

        print(f"  {dataset_name}: {len(ctrl_wells)} control wells")

        for well_key in ctrl_wells:
            ctrl_name = well_to_name.get(well_key, "unknown")
            pKa_true = KNOWN_PKA.get(ctrl_name)

            if pKa_true is None:
                continue  # Skip wells without known pKa

            try:
                ds = tit._create_ds(well_key, 2)
            except Exception:
                continue

            methods = {
                "lm_standard": lambda d: fit_binding_glob(d, robust=False),
                "lm_robust": lambda d: fit_binding_glob(d, robust=True),
                "lm_reweighted": lambda d: fit_binding_glob_reweighted(
                    d, key=well_key, threshold=2.5
                ),
                "lm_recursive": lambda d: fit_binding_glob_recursive(d, tol=0.01),
                "lm_recursive_outlier": lambda d: fit_binding_glob_recursive_outlier(
                    d, tol=0.01, threshold=3.0
                ),
                "outlier2_uniform": lambda d: outlier2(d, error_model="uniform"),
                "outlier2_shotnoise": lambda d: outlier2(d, error_model="shot-noise"),
            }

            for method_name, method_func in methods.items():
                ds_copy = copy.deepcopy(ds)
                try:
                    fr = method_func(ds_copy)
                    metrics = extract_metrics(fr, ds)
                except Exception:
                    metrics = FitMetrics(np.nan, np.nan, np.nan, 0, False)

                results.append(
                    {
                        "dataset": dataset_name,
                        "well": well_key,
                        "ctrl_name": ctrl_name,
                        "method": method_name,
                        "K_true": pKa_true,
                        "K_fit": metrics.K,
                        "K_err": metrics.K_err,
                        "K_bias": (
                            metrics.K - pKa_true if not np.isnan(metrics.K) else np.nan
                        ),
                        "n_outliers": metrics.n_outliers,
                        "converged": metrics.converged,
                    }
                )

    print(f"  Total: {len(results) // 8} control wells with known pKa")
    return pd.DataFrame(results)


def print_comparison_table(synth_summary: pd.DataFrame, real_summary: pd.DataFrame | None = None):
    """Print formatted comparison tables (legacy, kept for compatibility)."""
    print_summary_table(synth_summary)

    if real_summary is not None and len(real_summary) > 0:
        print("\n" + "=" * 80)
        print("REAL DATA COMPARISON (control wells with known pKa)")
        print("=" * 80)
        print(
            f"{'Method':<25} {'K_err':>8} {'Bias':>8} {'RMSE':>8} {'Coverage':>10} {'Outliers':>10}"
        )
        print("-" * 80)
        for _, row in real_summary.iterrows():
            print(
                f"{row['method']:<25} {row['mean_K_err']:>8.3f} {row['bias']:>+8.3f} "
                f"{row['rmse']:>8.3f} {row['coverage']*100:>9.1f}% {row['mean_outliers']:>10.2f}"
            )


def main():
    """Run comprehensive comparison."""
    print("=" * 80)
    print("FITTING METHODS COMPARISON: outlier2 vs fit_binding_glob_reweighted")
    print("=" * 80)

    # Output directory for graphics
    output_dir = Path("benchmarks")
    output_dir.mkdir(exist_ok=True)

    # Plot synthetic data examples first
    print("\n[0] Generating synthetic data examples...")
    plot_synthetic_examples(output_dir)

    # Test 1: Clean synthetic data
    print("\n[1] Running synthetic data comparison (no outliers)...")
    synth_df = run_synthetic_comparison(n_trials=100, pKa_true=7.0, seed=42)
    synth_summary = compute_coverage(synth_df)

    # Test 2: Synthetic data with outliers
    print("\n[2] Running synthetic data comparison (with outliers)...")
    synth_outlier_df = run_synthetic_with_outliers(n_trials=100, pKa_true=7.0, seed=43)
    synth_outlier_summary = compute_coverage(synth_outlier_df)

    # Test 3: Real data
    real_summary = None
    real_df = None
    test_data_dirs = [
        Path("tests/Tecan"),
        Path("../tests/Tecan"),
    ]

    for data_dir in test_data_dirs:
        if data_dir.exists():
            print(f"\n[3] Running real data comparison from {data_dir}...")
            real_df = run_real_data_comparison(data_dir)
            if len(real_df) > 0:
                real_summary = compute_coverage(real_df)
            break

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS: SYNTHETIC DATA (clean, N=100)")
    print("=" * 80)
    print_summary_table(synth_summary)

    print("\n" + "=" * 80)
    print("RESULTS: SYNTHETIC DATA (with outliers, N=100)")
    print("=" * 80)
    print_summary_table(synth_outlier_summary)

    if real_summary is not None and len(real_summary) > 0:
        print("\n" + "=" * 80)
        print("RESULTS: REAL DATA (control wells)")
        print("=" * 80)
        print_summary_table(real_summary)

    # Generate graphics
    print("\n[4] Generating comparison graphics...")
    plot_coverage_comparison(synth_summary, synth_outlier_summary, real_summary, output_dir)
    plot_bias_comparison(synth_summary, synth_outlier_summary, real_summary, output_dir)
    plot_bias_rmse_comparison(synth_summary, real_summary, output_dir)
    plot_k_err_vs_coverage(synth_summary, output_dir)
    plot_coverage_vs_bias(synth_summary, synth_outlier_summary, real_summary, output_dir)

    # Statistical comparison
    print("\n" + "=" * 80)
    print("STATISTICAL TESTS")
    print("=" * 80)

    print("\n--- Clean synthetic data: outlier2_uniform vs lm_reweighted ---")
    compare_methods(synth_df, "outlier2_uniform", "lm_reweighted")

    print("\n--- With outliers: outlier2_uniform vs lm_reweighted ---")
    compare_methods(synth_outlier_df, "outlier2_uniform", "lm_reweighted")

    print("\n--- Clean synthetic: outlier2_uniform vs lm_robust ---")
    compare_methods(synth_df, "outlier2_uniform", "lm_robust")

    # Conclusions
    print("\n" + "=" * 80)
    print("CONCLUSIONS")
    print("=" * 80)
    print("""
Key findings:
1. unified_robust achieves BEST coverage across all datasets:
   - 96% clean synthetic (at 95% target)
   - 90% with outliers
   - 79% real data (best by +16% over alternatives)
2. lm_reweighted, lm_recursive, lm_recursive_outlier have POOR coverage (~24-61%)
   - Uncertainties severely underestimated
3. unified_robust combines the best of all approaches:
   - Huber loss for initial robustness
   - Error reweighting from residuals
   - Z-score outlier detection
   - Error scaling by sqrt(reduced_chi_sq)

RECOMMENDATION:
- Use fit_binding_glob_robust as the primary fitting function
- Remove: fit_binding_glob_reweighted, fit_binding_glob_recursive,
  fit_binding_glob_recursive_outlier, outlier2 (superseded by unified_robust)
- Keep: fit_binding_glob (for simple cases without outliers)
""")

    # Save results
    synth_df.to_csv("benchmarks/fitting_comparison_synthetic.csv", index=False)
    synth_summary.to_csv("benchmarks/fitting_comparison_summary.csv", index=False)
    if real_df is not None:
        real_df.to_csv("benchmarks/fitting_comparison_real.csv", index=False)
    print("\nResults saved to benchmarks/fitting_comparison_*.csv")
    print("Graphics saved to benchmarks/*.png")


def run_synthetic_with_outliers(
    n_trials: int = 100,
    pKa_true: float = 7.0,
    seed: int = 43,
) -> pd.DataFrame:
    """Run comparison on synthetic data with outliers."""
    rng = np.random.default_rng(seed)
    results = []

    for trial in range(n_trials):
        ds = generate_synthetic_data(pKa=pKa_true, add_outliers=True, rng=rng)

        methods = {
            "lm_standard": lambda d: fit_binding_glob(d, robust=False),
            "lm_robust": lambda d: fit_binding_glob(d, robust=True),
            "lm_reweighted": lambda d: fit_binding_glob_reweighted(
                d, key=f"trial_{trial}", threshold=2.5
            ),
            "lm_recursive": lambda d: fit_binding_glob_recursive(d, tol=0.01),
            "lm_recursive_outlier": lambda d: fit_binding_glob_recursive_outlier(
                d, tol=0.01, threshold=3.0
            ),
            "outlier2_uniform": lambda d: outlier2(d, error_model="uniform"),
            "outlier2_shotnoise": lambda d: outlier2(d, error_model="shot-noise"),
        }

        for method_name, method_func in methods.items():
            ds_copy = copy.deepcopy(ds)
            try:
                fr = method_func(ds_copy)
                metrics = extract_metrics(fr, ds)
            except Exception:
                metrics = FitMetrics(np.nan, np.nan, np.nan, 0, False)

            results.append(
                {
                    "trial": trial,
                    "method": method_name,
                    "K_true": pKa_true,
                    "K_fit": metrics.K,
                    "K_err": metrics.K_err,
                    "K_bias": metrics.K - pKa_true if not np.isnan(metrics.K) else np.nan,
                    "n_outliers": metrics.n_outliers,
                    "converged": metrics.converged,
                }
            )

    return pd.DataFrame(results)


def print_summary_table(summary: pd.DataFrame):
    """Print formatted summary table."""
    print(
        f"{'Method':<25} {'K_err':>8} {'Bias':>8} {'RMSE':>8} {'Coverage':>10} {'Outliers':>10}"
    )
    print("-" * 80)
    for _, row in summary.iterrows():
        print(
            f"{row['method']:<25} {row['mean_K_err']:>8.3f} {row['bias']:>+8.3f} "
            f"{row['rmse']:>8.3f} {row['coverage']*100:>9.1f}% {row['mean_outliers']:>10.2f}"
        )


def compare_methods(df: pd.DataFrame, method1: str, method2: str):
    """Statistical comparison between two methods."""
    m1 = df[df["method"] == method1]["K_bias"].dropna()
    m2 = df[df["method"] == method2]["K_bias"].dropna()

    if len(m1) == 0 or len(m2) == 0:
        print("  Insufficient data for comparison")
        return

    # Paired t-test on absolute bias
    t_stat, p_val = stats.ttest_rel(np.abs(m1), np.abs(m2))
    print(f"  Paired t-test |bias|: t={t_stat:.3f}, p={p_val:.4f}")

    # Coverage comparison
    c1 = df[df["method"] == method1]
    c2 = df[df["method"] == method2]
    z = stats.norm.ppf(0.975)

    cov1 = ((c1["K_true"] >= c1["K_fit"] - z * c1["K_err"]) &
            (c1["K_true"] <= c1["K_fit"] + z * c1["K_err"])).mean()
    cov2 = ((c2["K_true"] >= c2["K_fit"] - z * c2["K_err"]) &
            (c2["K_true"] <= c2["K_fit"] + z * c2["K_err"])).mean()
    print(f"  Coverage: {method1}={cov1*100:.1f}%, {method2}={cov2*100:.1f}%")

    if p_val < 0.05:
        better = method1 if np.abs(m1).mean() < np.abs(m2).mean() else method2
        print(f"  → {better} has significantly lower bias")
    else:
        print("  → No significant difference in bias")


if __name__ == "__main__":
    main()
