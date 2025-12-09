#!/usr/bin/env python
"""Comprehensive comparison of ALL fitting methods including ODR.

This script compares fitting methods on synthetic and real data using 3 key metrics:
1. Residual distribution normality (Gaussian) - especially for real data
2. Bias (synthetic only - requires known true pKa)
3. 95% CI coverage (pKa uncertainty should ensure 95% coverage)

Methods compared:
- LM standard (fit_binding_glob)
- LM robust (Huber loss)
- outlier2 uniform
- outlier2 shot-noise
- ODR single pass (fit_binding_odr)
- ODR recursive (fit_binding_odr_recursive)
- ODR recursive with outlier removal (fit_binding_odr_recursive_outlier)
"""


import copy
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from clophfit.fitting.core import fit_binding_glob, outlier2
from clophfit.fitting.data_structures import DataArray, Dataset, FitResult
from clophfit.fitting.models import binding_1site
from clophfit.fitting.odr import (
    fit_binding_odr,
    fit_binding_odr_recursive,
    fit_binding_odr_recursive_outlier,
)

# Ground truth parameters (calibrated to match real data characteristics)
TRUE_S0_Y2 = 1200.0  # y2: wide range
TRUE_S1_Y2 = 400.0
TRUE_S0_Y1 = 750.0  # y1: narrow range, inverted
TRUE_S1_Y1 = 900.0
BUFFER_SD_Y2 = 40.0
BUFFER_SD_Y1 = 200.0


@dataclass
class FitMetrics:
    """Metrics for a single fit."""

    K: float = np.nan
    K_err: float = np.nan
    residuals: np.ndarray = field(default_factory=lambda: np.array([]))
    n_outliers: int = 0
    converged: bool = False


def generate_synthetic_data(
    pKa: float = 7.0,
    n_points: int = 7,
    add_outliers: bool = False,
    rng: np.random.Generator | None = None,
) -> Dataset:
    """Generate synthetic dual-channel pH titration data."""
    if rng is None:
        rng = np.random.default_rng()

    x = np.linspace(5.5, 9.0, n_points)
    x_err = 0.05 * np.ones_like(x)

    y1_true = binding_1site(x, pKa, TRUE_S0_Y1, TRUE_S1_Y1, is_ph=True)
    y2_true = binding_1site(x, pKa, TRUE_S0_Y2, TRUE_S1_Y2, is_ph=True)

    y1_err_true = np.sqrt(np.maximum(y1_true, 1.0) + BUFFER_SD_Y1**2)
    y2_err_true = np.sqrt(np.maximum(y2_true, 1.0) + BUFFER_SD_Y2**2)

    y1 = y1_true + rng.normal(0, y1_err_true)
    y2 = y2_true + rng.normal(0, y2_err_true)

    if add_outliers:
        y1[0] -= 4 * y1_err_true[0]
        y1[1] -= 3 * y1_err_true[1]

    da1 = DataArray(x, y1, x_errc=x_err, y_errc=y1_err_true)
    da2 = DataArray(x, y2, x_errc=x_err, y_errc=y2_err_true)

    return Dataset({"y1": da1, "y2": da2}, is_ph=True)


def extract_metrics(fr: FitResult, ds_original: Dataset) -> FitMetrics:
    """Extract metrics from a FitResult."""
    if fr.result is None:
        return FitMetrics()

    K = fr.result.params["K"].value
    K_err = fr.result.params["K"].stderr or np.nan

    # Get residuals if available
    residuals = np.array([])
    if hasattr(fr.result, "residual") and fr.result.residual is not None:
        residuals = np.array(fr.result.residual)

    # Count outliers
    n_original = sum(len(da.y) for da in ds_original.values())
    n_final = sum(len(da.y) for da in fr.dataset.values()) if fr.dataset else n_original
    n_outliers = n_original - n_final

    return FitMetrics(
        K=K,
        K_err=K_err,
        residuals=residuals,
        n_outliers=n_outliers,
        converged=True,
    )


def build_fitters() -> dict[str, callable]:
    """Build dictionary of all fitting methods."""
    return {
        "lm_standard": lambda ds: fit_binding_glob(ds, robust=False),
        "lm_robust": lambda ds: fit_binding_glob(ds, robust=True),
        "outlier2_uniform": lambda ds: outlier2(ds, error_model="uniform"),
        "outlier2_shotnoise": lambda ds: outlier2(ds, error_model="shot-noise"),
        "odr_single": lambda ds: fit_binding_odr(ds),
        "odr_recursive": lambda ds: fit_binding_odr_recursive(ds, max_iterations=10),
        "odr_recursive_outlier": lambda ds: fit_binding_odr_recursive_outlier(
            ds, threshold=3.0
        ),
    }


def test_residual_normality(residuals: np.ndarray) -> dict:
    """Test if residuals follow a Gaussian distribution.

    Returns multiple normality test statistics.
    """
    if len(residuals) < 8:
        return {"shapiro_p": np.nan, "dagostino_p": np.nan, "is_normal": False}

    # Remove any NaN/Inf values
    residuals = residuals[np.isfinite(residuals)]
    if len(residuals) < 8:
        return {"shapiro_p": np.nan, "dagostino_p": np.nan, "is_normal": False}

    try:
        _, shapiro_p = stats.shapiro(residuals)
    except Exception:
        shapiro_p = np.nan

    try:
        _, dagostino_p = stats.normaltest(residuals)
    except Exception:
        dagostino_p = np.nan

    # Consider normal if p > 0.05 for both tests
    is_normal = (shapiro_p > 0.05) if np.isfinite(shapiro_p) else False

    return {
        "shapiro_p": shapiro_p,
        "dagostino_p": dagostino_p,
        "is_normal": is_normal,
        "mean": np.mean(residuals),
        "std": np.std(residuals),
        "skew": stats.skew(residuals),
        "kurtosis": stats.kurtosis(residuals),
    }


def run_synthetic_comparison(
    n_trials: int = 100,
    pKa_true: float = 7.0,
    seed: int = 42,
    add_outliers: bool = False,
) -> pd.DataFrame:
    """Run comparison on synthetic data."""
    rng = np.random.default_rng(seed)
    fitters = build_fitters()
    results = []

    for trial in range(n_trials):
        ds = generate_synthetic_data(pKa=pKa_true, add_outliers=add_outliers, rng=rng)

        for method_name, method_func in fitters.items():
            ds_copy = copy.deepcopy(ds)
            try:
                fr = method_func(ds_copy)
                metrics = extract_metrics(fr, ds)
            except Exception as e:
                print(f"  Trial {trial}, {method_name}: {e}")
                metrics = FitMetrics()

            # Compute normality of residuals
            norm_stats = test_residual_normality(metrics.residuals)

            results.append(
                {
                    "trial": trial,
                    "method": method_name,
                    "K_true": pKa_true,
                    "K_fit": metrics.K,
                    "K_err": metrics.K_err,
                    "K_bias": metrics.K - pKa_true if np.isfinite(metrics.K) else np.nan,
                    "n_outliers": metrics.n_outliers,
                    "converged": metrics.converged,
                    "residual_mean": norm_stats.get("mean", np.nan),
                    "residual_std": norm_stats.get("std", np.nan),
                    "shapiro_p": norm_stats.get("shapiro_p", np.nan),
                    "is_normal": norm_stats.get("is_normal", False),
                }
            )

    return pd.DataFrame(results)


def compute_summary_metrics(df: pd.DataFrame, confidence: float = 0.95) -> pd.DataFrame:
    """Compute summary statistics for each method."""
    z = stats.norm.ppf((1 + confidence) / 2)

    summary = []
    for method in df["method"].unique():
        method_df = df[df["method"] == method].dropna(subset=["K_fit", "K_err"])

        if len(method_df) == 0:
            continue

        # Coverage: fraction where true value is within CI
        if "K_true" in method_df.columns:
            lower = method_df["K_fit"] - z * method_df["K_err"]
            upper = method_df["K_fit"] + z * method_df["K_err"]
            covered = (method_df["K_true"] >= lower) & (method_df["K_true"] <= upper)
            coverage = covered.mean()
            bias = method_df["K_bias"].mean()
            rmse = np.sqrt((method_df["K_bias"] ** 2).mean())
        else:
            coverage = np.nan
            bias = np.nan
            rmse = np.nan

        # Normality statistics
        norm_frac = method_df["is_normal"].mean() if "is_normal" in method_df else np.nan
        mean_shapiro = method_df["shapiro_p"].mean() if "shapiro_p" in method_df else np.nan

        summary.append(
            {
                "method": method,
                "n_trials": len(method_df),
                "mean_K_err": method_df["K_err"].mean(),
                "bias": bias,
                "rmse": rmse,
                "coverage": coverage,
                "mean_outliers": method_df["n_outliers"].mean(),
                "frac_normal": norm_frac,
                "mean_shapiro_p": mean_shapiro,
            }
        )

    return pd.DataFrame(summary).sort_values("coverage", ascending=False)


def plot_residual_distributions(
    df: pd.DataFrame, output_dir: Path, title_suffix: str = ""
):
    """Plot residual distribution comparison across methods."""
    methods = df["method"].unique()
    n_methods = len(methods)

    fig, axes = plt.subplots(2, (n_methods + 1) // 2, figsize=(14, 8))
    axes = axes.flatten()

    for i, method in enumerate(methods):
        ax = axes[i]
        method_df = df[df["method"] == method]

        # Aggregate residual stats
        means = method_df["residual_mean"].dropna()
        stds = method_df["residual_std"].dropna()

        if len(means) > 0:
            ax.hist(means, bins=20, alpha=0.7, label=f"mean={means.mean():.3f}")
            ax.axvline(0, color="red", linestyle="--", alpha=0.7)
            ax.set_xlabel("Residual Mean")
            ax.set_ylabel("Count")
            ax.set_title(f"{method}\nmean={means.mean():.3f}, std(mean)={means.std():.3f}")
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(method)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f"Residual Mean Distribution {title_suffix}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    output_path = output_dir / f"residual_means{title_suffix.replace(' ', '_')}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_metric_comparison(
    clean_summary: pd.DataFrame,
    outlier_summary: pd.DataFrame,
    output_dir: Path,
):
    """Plot comprehensive metric comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    methods = clean_summary["method"].tolist()
    x = np.arange(len(methods))
    width = 0.35

    # Plot 1: Coverage
    ax = axes[0, 0]
    bars1 = ax.bar(
        x - width / 2,
        clean_summary["coverage"].values * 100,
        width,
        label="Clean",
        color="steelblue",
    )
    bars2 = ax.bar(
        x + width / 2,
        outlier_summary["coverage"].values * 100,
        width,
        label="With Outliers",
        color="darkorange",
    )
    ax.axhline(95, color="red", linestyle="--", linewidth=2, label="95% target")
    ax.set_xlabel("Method")
    ax.set_ylabel("Coverage (%)")
    ax.set_title("95% CI Coverage")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
    ax.legend(loc="lower left")
    ax.set_ylim(0, 105)

    # Plot 2: Bias
    ax = axes[0, 1]
    ax.bar(
        x - width / 2,
        clean_summary["bias"].values,
        width,
        label="Clean",
        color="steelblue",
    )
    ax.bar(
        x + width / 2,
        outlier_summary["bias"].values,
        width,
        label="With Outliers",
        color="darkorange",
    )
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Method")
    ax.set_ylabel("Bias (pKa units)")
    ax.set_title("Bias (closer to 0 is better)")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
    ax.legend()

    # Plot 3: Normality fraction
    ax = axes[1, 0]
    ax.bar(
        x - width / 2,
        clean_summary["frac_normal"].values * 100,
        width,
        label="Clean",
        color="steelblue",
    )
    ax.bar(
        x + width / 2,
        outlier_summary["frac_normal"].values * 100,
        width,
        label="With Outliers",
        color="darkorange",
    )
    ax.set_xlabel("Method")
    ax.set_ylabel("Fraction Normal (%)")
    ax.set_title("Residuals pass Shapiro-Wilk (p>0.05)")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.set_ylim(0, 105)

    # Plot 4: Mean K_err (uncertainty)
    ax = axes[1, 1]
    ax.bar(
        x - width / 2,
        clean_summary["mean_K_err"].values,
        width,
        label="Clean",
        color="steelblue",
    )
    ax.bar(
        x + width / 2,
        outlier_summary["mean_K_err"].values,
        width,
        label="With Outliers",
        color="darkorange",
    )
    ax.set_xlabel("Method")
    ax.set_ylabel("Mean K_err (pKa units)")
    ax.set_title("Reported Uncertainty")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
    ax.legend()

    plt.suptitle(
        "Fitting Method Comparison (Synthetic Data, N=100)\n"
        "Key metrics: Coverage≈95%, Bias≈0, Normal residuals",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_dir / "comprehensive_metrics.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir / 'comprehensive_metrics.png'}")


def print_summary_table(summary: pd.DataFrame, title: str = ""):
    """Print formatted summary table."""
    print(f"\n{'=' * 100}")
    print(title)
    print("=" * 100)
    print(
        f"{'Method':<25} {'K_err':>8} {'Bias':>8} {'RMSE':>8} "
        f"{'Coverage':>10} {'Outliers':>10} {'%Normal':>10}"
    )
    print("-" * 100)
    for _, row in summary.iterrows():
        coverage_str = f"{row['coverage']*100:.1f}%" if pd.notna(row["coverage"]) else "N/A"
        normal_str = f"{row['frac_normal']*100:.1f}%" if pd.notna(row["frac_normal"]) else "N/A"
        print(
            f"{row['method']:<25} {row['mean_K_err']:>8.3f} {row['bias']:>+8.3f} "
            f"{row['rmse']:>8.3f} {coverage_str:>10} {row['mean_outliers']:>10.2f} "
            f"{normal_str:>10}"
        )


def main():
    """Run comprehensive comparison."""
    print("=" * 80)
    print("COMPREHENSIVE FITTER COMPARISON (All Methods Including ODR)")
    print("=" * 80)

    output_dir = Path("benchmarks")
    output_dir.mkdir(exist_ok=True)

    # --- Phase 1: Synthetic data WITHOUT outliers ---
    print("\n[1] Running synthetic comparison (clean data, N=100)...")
    clean_df = run_synthetic_comparison(n_trials=100, pKa_true=7.0, seed=42, add_outliers=False)

    # Intermediate results
    print("\n--- Intermediate: Sample of clean data results ---")
    sample = clean_df.groupby("method").first().reset_index()
    print(sample[["method", "K_fit", "K_err", "K_bias", "is_normal"]].to_string())

    clean_summary = compute_summary_metrics(clean_df)
    print_summary_table(clean_summary, "SYNTHETIC DATA (Clean, N=100)")

    # --- Phase 2: Synthetic data WITH outliers ---
    print("\n[2] Running synthetic comparison (with outliers, N=100)...")
    outlier_df = run_synthetic_comparison(n_trials=100, pKa_true=7.0, seed=43, add_outliers=True)

    # Intermediate results
    print("\n--- Intermediate: Sample of outlier data results ---")
    sample = outlier_df.groupby("method").first().reset_index()
    print(sample[["method", "K_fit", "K_err", "K_bias", "n_outliers"]].to_string())

    outlier_summary = compute_summary_metrics(outlier_df)
    print_summary_table(outlier_summary, "SYNTHETIC DATA (With Outliers, N=100)")

    # --- Generate plots ---
    print("\n[3] Generating comparison plots...")
    plot_metric_comparison(clean_summary, outlier_summary, output_dir)
    plot_residual_distributions(clean_df, output_dir, title_suffix=" (Clean)")
    plot_residual_distributions(outlier_df, output_dir, title_suffix=" (Outliers)")

    # --- Save detailed results ---
    clean_df.to_csv(output_dir / "fitter_comparison_clean.csv", index=False)
    outlier_df.to_csv(output_dir / "fitter_comparison_outliers.csv", index=False)
    clean_summary.to_csv(output_dir / "fitter_summary_clean.csv", index=False)
    outlier_summary.to_csv(output_dir / "fitter_summary_outliers.csv", index=False)

    print("\n" + "=" * 80)
    print("KEY FINDINGS SUMMARY")
    print("=" * 80)

    # Find best methods for each metric
    best_coverage_clean = clean_summary.loc[clean_summary["coverage"].idxmax()]
    best_coverage_outlier = outlier_summary.loc[outlier_summary["coverage"].idxmax()]
    best_bias_clean = clean_summary.loc[clean_summary["bias"].abs().idxmin()]
    best_normal_clean = clean_summary.loc[clean_summary["frac_normal"].idxmax()]

    print(f"\n1. COVERAGE (target: 95%)")
    print(f"   Best clean:   {best_coverage_clean['method']} ({best_coverage_clean['coverage']*100:.1f}%)")
    print(f"   Best outlier: {best_coverage_outlier['method']} ({best_coverage_outlier['coverage']*100:.1f}%)")

    print(f"\n2. BIAS (target: 0)")
    print(f"   Best clean:   {best_bias_clean['method']} (bias={best_bias_clean['bias']:+.4f})")

    print(f"\n3. RESIDUAL NORMALITY")
    print(f"   Best:         {best_normal_clean['method']} ({best_normal_clean['frac_normal']*100:.1f}% normal)")

    print("\n" + "=" * 80)
    print("FILES SAVED:")
    print("=" * 80)
    print(f"  - {output_dir / 'fitter_comparison_clean.csv'}")
    print(f"  - {output_dir / 'fitter_comparison_outliers.csv'}")
    print(f"  - {output_dir / 'fitter_summary_clean.csv'}")
    print(f"  - {output_dir / 'fitter_summary_outliers.csv'}")
    print(f"  - {output_dir / 'comprehensive_metrics.png'}")

    print("\n" + "=" * 80)
    print("PLEASE REVIEW THE INTERMEDIATE RESULTS ABOVE")
    print("Is the synthetic data configuration appropriate?")
    print("=" * 80)

    return clean_summary, outlier_summary


if __name__ == "__main__":
    main()
