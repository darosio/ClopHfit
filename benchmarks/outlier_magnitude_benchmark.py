#!/usr/bin/env python
"""Benchmark fitting methods vs outlier severity.

Tests how standard WLS and robust (Huber) fitting handle outliers of varying magnitude.
Results saved to benchmarks/outlier_magnitude_comparison.png

Usage:
    python benchmarks/outlier_magnitude_benchmark.py
"""

import copy

import matplotlib.pyplot as plt
import numpy as np

from clophfit.fitting.core import fit_binding_glob
from clophfit.fitting.data_structures import DataArray, Dataset

TRUE_K = 7.0


def generate_data(outlier_sigma: float = 3.0, seed: int = 42) -> Dataset:
    """Generate synthetic pH titration data with outliers.

    Parameters
    ----------
    outlier_sigma
        Magnitude of outliers in standard deviations (negative = DROP).
    seed
        Random seed for reproducibility.

    Returns
    -------
    Dataset
        Dataset with y1 (inverted, large errors) and y2 (normal, small errors).
    """
    np.random.seed(seed)
    x = np.linspace(5.5, 9.0, 8)

    # y1: inverted curve, narrow range, large errors (like real FRET donor)
    y1_s0, y1_s1 = 600, 800  # inverted: S1 > S0
    y1_true = y1_s0 + (y1_s1 - y1_s0) / (1 + 10 ** (TRUE_K - x))
    y1_err = np.full_like(x, 100.0)
    y1 = y1_true + np.random.normal(0, y1_err)

    # y2: normal curve, wide range, small errors (like real FRET acceptor)
    y2_s0, y2_s1 = 1000, 200
    y2_true = y2_s0 + (y2_s1 - y2_s0) / (1 + 10 ** (TRUE_K - x))
    y2_err = np.full_like(x, 40.0)
    y2 = y2_true + np.random.normal(0, y2_err)

    # Add outliers at low pH (last 2 points) in y1 channel - DROP below expected
    y1[-2] -= outlier_sigma * y1_err[-2]
    y1[-1] -= outlier_sigma * y1_err[-1]

    return Dataset(
        {
            "y1": DataArray(xc=x, yc=y1, y_errc=y1_err),
            "y2": DataArray(xc=x, yc=y2, y_errc=y2_err),
        },
        is_ph=True,
    )


def run_fits(ds: Dataset) -> dict[str, tuple[float, float]]:
    """Run standard and robust fits on dataset.

    Returns
    -------
    dict
        Method name -> (K_value, K_stderr) tuple.
    """
    results = {}

    # Standard WLS
    try:
        r = fit_binding_glob(copy.deepcopy(ds), robust=False)
        if r.result:
            results["lm_standard"] = (
                r.result.params["K"].value,
                r.result.params["K"].stderr,
            )
    except Exception:
        pass

    # Robust (Huber)
    try:
        r = fit_binding_glob(copy.deepcopy(ds), robust=True)
        if r.result:
            results["lm_robust"] = (
                r.result.params["K"].value,
                r.result.params["K"].stderr,
            )
    except Exception:
        pass

    return results


def main() -> None:
    """Run benchmark and generate plots."""
    outlier_sigmas = [0, 1, 2, 3, 4, 5, 6, 8, 10]
    n_trials = 50

    methods = ["lm_standard", "lm_robust"]
    metrics: dict[str, dict[str, list[float]]] = {
        method: {"bias": [], "rmse": [], "coverage": []} for method in methods
    }

    print("Testing outlier magnitudes...")
    for sigma in outlier_sigmas:
        print(f"  Sigma = {sigma}", end="", flush=True)
        method_results: dict[str, list[tuple[float, float]]] = {m: [] for m in methods}

        for trial in range(n_trials):
            ds = generate_data(outlier_sigma=sigma, seed=42 + trial)
            results = run_fits(ds)

            for method, (k, k_err) in results.items():
                if k_err is not None:
                    method_results[method].append((k, k_err))

        for method in methods:
            if method_results[method]:
                ks = np.array([r[0] for r in method_results[method]])
                k_errs = np.array([r[1] for r in method_results[method]])

                bias = np.mean(ks - TRUE_K)
                rmse = float(np.sqrt(np.mean((ks - TRUE_K) ** 2)))
                coverage = float(np.mean(np.abs(ks - TRUE_K) < 1.96 * k_errs) * 100)

                metrics[method]["bias"].append(bias)
                metrics[method]["rmse"].append(rmse)
                metrics[method]["coverage"].append(coverage)
            else:
                metrics[method]["bias"].append(np.nan)
                metrics[method]["rmse"].append(np.nan)
                metrics[method]["coverage"].append(np.nan)
        print(" done")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    colors = {"lm_standard": "blue", "lm_robust": "orange"}
    labels = {"lm_standard": "Standard WLS", "lm_robust": "Robust (Huber)"}

    ax = axes[0]
    for method in methods:
        ax.plot(
            outlier_sigmas,
            metrics[method]["bias"],
            "o-",
            color=colors[method],
            label=labels[method],
            lw=2,
            ms=8,
        )
    ax.axhline(0, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("Outlier Magnitude (σ)")
    ax.set_ylabel("Bias (pKa units)")
    ax.set_title("Bias vs Outlier Magnitude")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for method in methods:
        ax.plot(
            outlier_sigmas,
            metrics[method]["rmse"],
            "o-",
            color=colors[method],
            label=labels[method],
            lw=2,
            ms=8,
        )
    ax.set_xlabel("Outlier Magnitude (σ)")
    ax.set_ylabel("RMSE (pKa units)")
    ax.set_title("RMSE vs Outlier Magnitude")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    for method in methods:
        ax.plot(
            outlier_sigmas,
            metrics[method]["coverage"],
            "o-",
            color=colors[method],
            label=labels[method],
            lw=2,
            ms=8,
        )
    ax.axhline(95, color="gray", ls="--", alpha=0.5, label="95% target")
    ax.set_xlabel("Outlier Magnitude (σ)")
    ax.set_ylabel("Coverage (%)")
    ax.set_title("Coverage vs Outlier Magnitude")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    plt.suptitle(
        "Fitting Method Performance vs Outlier Severity\n"
        "(2 outliers at low pH in y1 channel)",
        fontsize=14,
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(
        "benchmarks/outlier_magnitude_comparison.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print("Saved: benchmarks/outlier_magnitude_comparison.png")

    # Print summary table
    print("\n=== Summary Table ===")
    print(
        f"{'Sigma':>6} | {'Std Bias':>10} | {'Rob Bias':>10} | "
        f"{'Std RMSE':>10} | {'Rob RMSE':>10} | {'Std Cov':>8} | {'Rob Cov':>8}"
    )
    print("-" * 80)
    for i, sigma in enumerate(outlier_sigmas):
        print(
            f"{sigma:>6} | {metrics['lm_standard']['bias'][i]:>10.4f} | "
            f"{metrics['lm_robust']['bias'][i]:>10.4f} | "
            f"{metrics['lm_standard']['rmse'][i]:>10.4f} | "
            f"{metrics['lm_robust']['rmse'][i]:>10.4f} | "
            f"{metrics['lm_standard']['coverage'][i]:>7.1f}% | "
            f"{metrics['lm_robust']['coverage'][i]:>7.1f}%"
        )


if __name__ == "__main__":
    main()
