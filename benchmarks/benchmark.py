#!/usr/bin/env python3
"""Benchmark comparing fitting methods on realistic synthetic and real data.

Run with: python benchmarks/benchmark.py

Results saved to: benchmarks/benchmark_results.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from clophfit import prtecan
from clophfit.fitting.core import (
    fit_binding_glob,
    outlier2,
)
from clophfit.testing.synthetic import make_realistic_dataset

# Constants
N_SAMPLES = 100
K_TRUE = 7.0


def load_real_data():
    """Load L2 titration data."""

    def load_tit(folder, bg_mth="meansd"):
        tit = prtecan.Titration.fromlistfile(folder / "list.pH.csv", is_ph=1)
        tit.load_additions(folder / "additions.pH")
        tit.load_scheme(folder / "scheme.txt")
        tit.params.bg_mth = bg_mth
        tit.params.bg_adj = True
        return tit

    l2_dir = Path("tests/Tecan/L2")
    return load_tit(l2_dir)


def run_benchmark():
    """Run complete benchmark and generate plots."""
    # Methods to compare
    methods = {
        "outlier2": lambda ds: outlier2(ds, error_model="uniform"),
        "lm_standard": fit_binding_glob,
        "lm_robust": lambda ds: fit_binding_glob(ds, robust=True),
    }

    # Run synthetic benchmark
    results_synthetic = {name: {"clean": [], "outliers": []} for name in methods}

    print("Running synthetic benchmark...")
    for i in range(N_SAMPLES):
        seed = 42 + i

        ds_clean, _ = make_realistic_dataset(pka=K_TRUE, seed=seed)
        ds_outliers, _ = make_realistic_dataset(
            pka=K_TRUE, outlier_prob=0.4, outlier_sigma=4.0, seed=seed
        )

        for name, method in methods.items():
            try:
                result_clean = method(ds_clean)
                K_clean = result_clean.result.params["K"].value
                results_synthetic[name]["clean"].append(K_clean - K_TRUE)

                result_outliers = method(ds_outliers)
                K_outliers = result_outliers.result.params["K"].value
                results_synthetic[name]["outliers"].append(K_outliers - K_TRUE)
            except Exception:
                pass

    # Run on real data
    print("Running real data benchmark...")
    tit = load_real_data()

    results_real = {name: [] for name in methods}

    for k in list(tit.fit_keys)[:38]:
        try:
            ds = tit._create_global_ds(k)
            for name, method in methods.items():
                try:
                    result = method(ds)
                    K = result.result.params['K'].value
                    results_real[name].append(K)
                except Exception:
                    pass
        except Exception:
            pass

    # Compute statistics
    stats = {}
    for name in methods:
        clean = np.array(results_synthetic[name]['clean'])
        outliers = np.array(results_synthetic[name]['outliers'])

        stats[name] = {
            'clean_bias': np.mean(clean),
            'outlier_bias': np.mean(outliers),
            'clean_rmse': np.sqrt(np.mean(clean**2)),
            'outlier_rmse': np.sqrt(np.mean(outliers**2)),
            'real_mean': np.mean(results_real[name]) if results_real[name] else np.nan,
            'real_std': np.std(results_real[name]) if results_real[name] else np.nan,
        }

    # Print results
    print("\n=== Synthetic Results ===")
    print(f"{'Method':<15} {'Clean Bias':>12} {'Outlier Bias':>12} "
          f"{'Clean RMSE':>12} {'Outlier RMSE':>12}")
    print("-" * 65)
    for name in methods:
        s = stats[name]
        print(f"{name:<15} {s['clean_bias']:>12.4f} {s['outlier_bias']:>12.4f} "
              f"{s['clean_rmse']:>12.4f} {s['outlier_rmse']:>12.4f}")

    print("\n=== Real Data Results ===")
    print(f"{'Method':<15} {'Mean K':>12} {'Std K':>12} {'N':>8}")
    print("-" * 50)
    for name in methods:
        if results_real[name]:
            s = stats[name]
            print(f"{name:<15} {s['real_mean']:>12.3f} {s['real_std']:>12.3f} "
                  f"{len(results_real[name]):>8}")

    # Create plot
    create_comparison_plot(methods, stats, results_real)

    return stats, results_real


def create_comparison_plot(methods, stats, results_real):
    """Create final comparison plot."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    method_names = list(methods.keys())
    colors = {'outlier2': 'tab:blue', 'lm_standard': 'tab:orange', 'lm_robust': 'tab:green'}

    # 1. Bias comparison
    ax = axes[0]
    x_pos = np.arange(len(method_names))
    width = 0.35
    clean_biases = [stats[n]['clean_bias'] for n in method_names]
    outlier_biases = [stats[n]['outlier_bias'] for n in method_names]

    bars1 = ax.bar(x_pos - width/2, clean_biases, width, label='Clean', color='steelblue')
    bars2 = ax.bar(x_pos + width/2, outlier_biases, width, label='With Outliers', color='coral')
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_ylabel('Bias (pKa units)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(method_names, rotation=30, ha='right')
    ax.legend(loc='upper left')
    ax.set_title('Bias Comparison\n(Lower absolute value is better)')

    for bar, val in zip(bars1, clean_biases):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, f'{val:.3f}',
                ha='center', va='bottom', fontsize=8)
    for bar, val in zip(bars2, outlier_biases):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, f'{val:.3f}',
                ha='center', va='bottom', fontsize=8)

    # 2. RMSE comparison
    ax = axes[1]
    clean_rmses = [stats[n]['clean_rmse'] for n in method_names]
    outlier_rmses = [stats[n]['outlier_rmse'] for n in method_names]

    bars1 = ax.bar(x_pos - width/2, clean_rmses, width, label='Clean', color='steelblue')
    bars2 = ax.bar(x_pos + width/2, outlier_rmses, width, label='With Outliers', color='coral')
    ax.set_ylabel('RMSE (pKa units)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(method_names, rotation=30, ha='right')
    ax.legend(loc='upper left')
    ax.set_title('RMSE Comparison\n(Lower is better)')

    for bar, val in zip(bars1, clean_rmses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, f'{val:.3f}',
                ha='center', va='bottom', fontsize=8)
    for bar, val in zip(bars2, outlier_rmses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, f'{val:.3f}',
                ha='center', va='bottom', fontsize=8)

    # 3. Real data distribution
    ax = axes[2]
    for i, name in enumerate(method_names):
        if results_real[name]:
            data = results_real[name]
            parts = ax.violinplot([data], positions=[i], showmeans=True, showmedians=True)
            for pc in parts['bodies']:
                pc.set_facecolor(list(colors.values())[i])
                pc.set_alpha(0.7)

    ax.set_ylabel('pKa')
    ax.set_xticks(range(len(method_names)))
    ax.set_xticklabels(method_names, rotation=30, ha='right')
    ax.set_title(f'Real Data Distribution\n({len(results_real["outlier2"])} wells)')

    plt.tight_layout()
    plt.savefig('benchmarks/benchmark_results.png', dpi=150, bbox_inches='tight')
    print("\nSaved to benchmarks/benchmark_results.png")


if __name__ == '__main__':
    run_benchmark()
