#!/usr/bin/env python3
"""Variance-Mean analysis and post-fit residual diagnostics.

This script:
1. Loads real Tecan titration data (140220, L1, L2, L4 datasets).
2. Performs Variance-Mean analysis on buffer replicates to calibrate
   the detector gain `g` and multiplicative noise coefficient `alpha`.
3. Runs post-fit residual analysis to validate the error model.

The general detector variance model is:
    Var(y) = sigma_read^2 + g * signal + (alpha * signal)^2

The current pipeline uses:
    y_err = sqrt(signal + bg_err^2)   [i.e., g=1, alpha=0]

This script determines whether g=1 and alpha=0 are justified.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# ---------------------------------------------------------------------------
# Setup paths
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from clophfit.fitting.core import fit_binding_glob  # noqa: E402
from clophfit.fitting.residuals import (  # noqa: E402
    analyze_label_bias,
    collect_multi_residuals,
    residual_statistics,
    validate_residuals,
)
from clophfit.prtecan.prtecan import Titration  # noqa: E402

TECAN_DIR = REPO / "tests" / "Tecan"
OUTPUT_DIR = REPO / "scripts" / "error_model_output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Dataset configurations: name -> loading kwargs
DATASETS: dict[str, dict[str, Any]] = {
    "140220": {
        "list_file": TECAN_DIR / "140220" / "list.pH.csv",
        "additions": TECAN_DIR / "140220" / "additions.pH",
        "scheme": TECAN_DIR / "140220" / "scheme.txt",
        "bg": True,
        "dil": True,
        "nrm": True,
    },
    "L1": {
        "list_file": TECAN_DIR / "L1" / "list.pH.csv",
        "additions": TECAN_DIR / "L1" / "additions.pH",
        "scheme": TECAN_DIR / "L1" / "scheme.txt",
        "bg": True,
        "dil": False,
        "nrm": True,
    },
    "L2": {
        "list_file": TECAN_DIR / "L2" / "list.pH.csv",
        "additions": TECAN_DIR / "L2" / "additions.pH",
        "scheme": TECAN_DIR / "L2" / "scheme.txt",
        "bg": True,
        "dil": False,
        "nrm": True,
    },
    "L4": {
        "list_file": TECAN_DIR / "L4" / "list.pH.csv",
        "additions": TECAN_DIR / "L4" / "additions.pH",
        "scheme": TECAN_DIR / "L4" / "scheme.txt",
        "bg": True,
        "dil": False,
        "nrm": True,
    },
}


def load_titration(cfg: dict[str, Any]) -> Titration:
    """Load and configure a Titration from a dataset config dict."""
    tit = Titration.fromlistfile(cfg["list_file"], is_ph=True)
    tit.load_additions(cfg["additions"])
    tit.load_scheme(cfg["scheme"])
    tit.params.bg = cfg["bg"]
    tit.params.dil = cfg["dil"]
    tit.params.nrm = cfg["nrm"]
    return tit


# ======================================================================
# Part 1: Variance-Mean analysis from buffer replicates
# ======================================================================
def variance_mean_analysis(tit: Titration) -> pd.DataFrame:
    """Compute Var vs Mean across buffer replicates at each pH point.

    For each label and each pH point, we have N_buffer_wells replicate
    measurements.  We compute Mean(y) and Var(y) across those replicates.
    """
    rows = []
    for label, buf_df in tit.buffer.dataframes.items():
        if buf_df.empty:
            continue
        # buf_df columns = buffer well names; rows = pH points
        # Drop non-numeric columns (Label, fit, fit_err, mean, sem)
        well_cols = [c for c in buf_df.columns if c not in
                     ("Label", "fit", "fit_err", "mean", "sem")]
        numeric = buf_df[well_cols].to_numpy(dtype=float)

        for i in range(numeric.shape[0]):
            vals = numeric[i, :]
            vals = vals[np.isfinite(vals)]
            if len(vals) < 2:
                continue
            rows.append({
                "label": label,
                "ph_index": i,
                "x": tit.x[i] if i < len(tit.x) else np.nan,
                "mean": np.mean(vals),
                "var": np.var(vals, ddof=1),
                "std": np.std(vals, ddof=1),
                "n_wells": len(vals),
            })
    return pd.DataFrame(rows)


def fit_variance_model(vm: pd.DataFrame) -> pd.DataFrame:
    """Fit Var = a + b*Mean + c*Mean^2 per label.

    Returns DataFrame with columns: label, sigma_read, gain, alpha, r2.
    """
    def var_model(mean: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        return a + b * mean + c * mean**2

    results = []
    for label, grp in vm.groupby("label"):
        m = grp["mean"].to_numpy()
        v = grp["var"].to_numpy()
        if len(m) < 3:
            continue
        try:
            # Fit the full quadratic model
            popt, pcov = curve_fit(
                var_model, m, v,
                p0=[100.0, 1.0, 1e-4],
                bounds=([0, 0, 0], [np.inf, np.inf, np.inf]),
                maxfev=10000,
            )
            a, b, c = popt
            perr = np.sqrt(np.diag(pcov))
            v_pred = var_model(m, *popt)
            ss_res = np.sum((v - v_pred)**2)
            ss_tot = np.sum((v - np.mean(v))**2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

            results.append({
                "label": label,
                "sigma_read": np.sqrt(a),
                "sigma_read_err": 0.5 * perr[0] / np.sqrt(a) if a > 0 else np.nan,
                "gain": b,
                "gain_err": perr[1],
                "alpha": np.sqrt(c),
                "alpha_err": 0.5 * perr[2] / np.sqrt(c) if c > 0 else np.nan,
                "r2": r2,
                "n_points": len(m),
            })

            # Also try pure-Poisson (Var = a + b*Mean) for comparison
            def var_poisson(mean: np.ndarray, a: float, b: float) -> np.ndarray:
                return a + b * mean

            popt_p, _ = curve_fit(
                var_poisson, m, v,
                p0=[100.0, 1.0],
                bounds=([0, 0], [np.inf, np.inf]),
            )
            v_pred_p = var_poisson(m, *popt_p)
            ss_res_p = np.sum((v - v_pred_p)**2)
            r2_p = 1 - ss_res_p / ss_tot if ss_tot > 0 else np.nan

            results[-1]["r2_poisson_only"] = r2_p
            results[-1]["gain_poisson_only"] = popt_p[1]
            results[-1]["sigma_read_poisson_only"] = np.sqrt(popt_p[0])

        except Exception as e:
            print(f"  Label {label}: fit failed: {e}")

    return pd.DataFrame(results)


def plot_variance_mean(vm: pd.DataFrame, fits: pd.DataFrame) -> plt.Figure:
    """Plot Var vs Mean with fitted models."""
    labels = sorted(vm["label"].unique())
    fig, axes = plt.subplots(1, len(labels), figsize=(6 * len(labels), 5))
    if len(labels) == 1:
        axes = [axes]

    for ax, label in zip(axes, labels):
        grp = vm[vm["label"] == label]
        m = grp["mean"].to_numpy()
        v = grp["var"].to_numpy()
        ax.scatter(m, v, s=40, zorder=3, label="Data")

        fit_row = fits[fits["label"] == label]
        if not fit_row.empty:
            row = fit_row.iloc[0]
            m_plot = np.linspace(m.min() * 0.9, m.max() * 1.1, 200)

            # Full model
            a = row["sigma_read"]**2
            b = row["gain"]
            c = row["alpha"]**2
            v_full = a + b * m_plot + c * m_plot**2
            ax.plot(m_plot, v_full, "r-", lw=2,
                    label=f"Full: g={b:.2f}, a={row['alpha']:.4f}")

            # Pure Poisson
            a_p = row["sigma_read_poisson_only"]**2
            b_p = row["gain_poisson_only"]
            v_pois = a_p + b_p * m_plot
            ax.plot(m_plot, v_pois, "g--", lw=1.5,
                    label=f"Poisson: g={b_p:.2f}")

            # Current model (g=1)
            v_curr = m_plot  # Var = signal (no bg_err here)
            ax.plot(m_plot, v_curr, "b:", lw=1.5,
                    label="Current: g=1, a=0")

        ax.set_xlabel("Mean signal (RFU)")
        ax.set_ylabel("Variance (RFU^2)")
        ax.set_title(f"Label {label}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("Variance-Mean Analysis of Buffer Replicates", fontsize=14)
    fig.tight_layout()
    return fig


# ======================================================================
# Part 2: Post-fit residual diagnostics
# ======================================================================
def run_residual_analysis(
    tit: Titration,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit all wells and collect residual diagnostics.

    Parameters
    ----------
    tit : Titration
        Titration object with preloaded data, additions, and scheme.

    Returns
    -------
    all_res : pd.DataFrame
        Residual table from collect_multi_residuals.
    stats : pd.DataFrame
        Per-label residual statistics.
    bias : pd.DataFrame
        Per-label bias summary.
    """
    # Compute global fits for all wells
    fit_results = {}
    for key in tit.fit_keys:
        fr = tit.result_global[key]
        if fr.result is not None:
            fit_results[key] = fr

    print(f"  Collected {len(fit_results)} successful fits")

    all_res = collect_multi_residuals(fit_results)
    stats = residual_statistics(all_res)

    # Add predicted signal column for the |residual| vs predicted plot
    # predicted = y - raw_residual
    all_res["predicted"] = np.nan  # placeholder
    for key, fr in fit_results.items():
        if fr.dataset is None or fr.result is None:
            continue
        r = np.asarray(fr.result.residual, dtype=float)
        start = 0
        for lbl, da in fr.dataset.items():
            n = len(da.y)
            rw = r[start:start + n]
            raw_res = rw * da.y_err  # raw = weighted * y_err
            predicted = da.y - raw_res
            y_err_vals = da.y_err

            mask = (all_res["well"] == key) & (all_res["label"] == lbl)
            idx = all_res.index[mask]
            if len(idx) == n:
                all_res.loc[idx, "predicted"] = predicted
                all_res.loc[idx, "y_observed"] = da.y
                all_res.loc[idx, "y_err"] = y_err_vals
            start += n

    bias_summary, label_bias = analyze_label_bias(all_res)

    return all_res, stats, label_bias


def plot_residual_vs_predicted(all_res: pd.DataFrame) -> plt.Figure:
    """Plot |standardized residual| vs predicted signal.

    If the error model is correct, standardized residuals should be
    flat (~0.8 for |N(0,1)|) regardless of signal level.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_i, (label, grp) in enumerate(all_res.groupby("label")):
        if ax_i >= len(axes):
            break
        ax = axes[ax_i]
        pred = grp["predicted"].to_numpy()
        std_res = grp["resid_weighted"].to_numpy()

        ax.scatter(pred, np.abs(std_res), s=8, alpha=0.3, color="C0")

        # Binned trend
        valid = np.isfinite(pred)
        if valid.sum() > 10:
            n_bins = min(15, valid.sum() // 5)
            bins = np.linspace(np.nanmin(pred), np.nanmax(pred), n_bins + 1)
            bin_centers = []
            bin_means = []
            for j in range(len(bins) - 1):
                in_bin = (pred >= bins[j]) & (pred < bins[j+1])
                if in_bin.sum() > 2:
                    bin_centers.append((bins[j] + bins[j+1]) / 2)
                    bin_means.append(np.mean(np.abs(std_res[in_bin])))
            ax.plot(bin_centers, bin_means, "r-o", lw=2, ms=5, label="Binned mean")
            ax.axhline(0.798, color="green", ls="--", lw=1.5,
                       label="Expected |N(0,1)| = 0.80")

        ax.set_xlabel("Predicted signal")
        ax.set_ylabel("|Standardized residual|")
        ax.set_title(f"Label {label}")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle(
        "Post-fit residual vs predicted: flat = correct error model", fontsize=13
    )
    fig.tight_layout()
    return fig


def plot_residual_vs_yerr(all_res: pd.DataFrame) -> plt.Figure:
    """Plot raw residual^2 vs y_err^2 to check error calibration."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_i, (label, grp) in enumerate(all_res.groupby("label")):
        if ax_i >= len(axes):
            break
        ax = axes[ax_i]

        y_err = grp["y_err"].to_numpy()
        raw_res = grp["resid_raw"].to_numpy()

        valid = np.isfinite(y_err) & np.isfinite(raw_res) & (y_err > 0)
        y_err_v = y_err[valid]
        raw_res_v = raw_res[valid]

        ax.scatter(y_err_v**2, raw_res_v**2, s=8, alpha=0.3, color="C0")

        # Trend: binned mean of residual^2 vs y_err^2
        n_bins = min(15, valid.sum() // 5)
        if n_bins > 2:
            bins = np.linspace(np.min(y_err_v**2), np.max(y_err_v**2), n_bins + 1)
            bc, bm = [], []
            for j in range(len(bins) - 1):
                in_bin = (y_err_v**2 >= bins[j]) & (y_err_v**2 < bins[j+1])
                if in_bin.sum() > 2:
                    bc.append((bins[j] + bins[j+1]) / 2)
                    bm.append(np.mean(raw_res_v[in_bin]**2))
            ax.plot(bc, bm, "r-o", lw=2, ms=5, label="Binned mean(res^2)")

        # Perfect calibration line: E[res^2] = y_err^2
        lim = max(np.max(y_err_v**2), np.max(raw_res_v**2))
        ax.plot([0, lim], [0, lim], "g--", lw=1.5, label="Perfect: res^2 = y_err^2")

        ax.set_xlabel("y_err^2 (model variance)")
        ax.set_ylabel("residual^2 (observed variance)")
        ax.set_title(f"Label {label}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("Error calibration: res^2 vs y_err^2", fontsize=13)
    fig.tight_layout()
    return fig


# ======================================================================
# Main
# ======================================================================
def _analyse_dataset(name: str, tit: Titration, out_dir: Path) -> pd.DataFrame:
    """Run full variance-mean + residual analysis for one dataset.

    Returns the variance-model fit results DataFrame (one row per label).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'─' * 60}")
    print(f"Dataset: {name}")
    print(f"  {len(tit.x)} pH points, {len(tit.fit_keys)} fit wells")
    print(f"  Buffer wells: {tit.buffer.wells}")

    # Variance-Mean on buffers
    vm = variance_mean_analysis(tit)
    vm.to_csv(out_dir / "variance_mean_data.csv", index=False)

    fits = fit_variance_model(vm)
    fits.to_csv(out_dir / "variance_model_fits.csv", index=False)

    print("  Noise model per label:")
    for _, row in fits.iterrows():
        lbl = row["label"]
        print(
            f"    L{lbl}: gain={row['gain']:.3g}±{row['gain_err']:.2g}  "
            f"alpha={row['alpha']:.4g}±{row['alpha_err']:.2g}  "
            f"sigma_read={row['sigma_read']:.1f}  R2={row['r2']:.3f}"
        )

    fig_vm = plot_variance_mean(vm, fits)
    fig_vm.savefig(out_dir / "variance_mean_plot.png", dpi=150)
    plt.close(fig_vm)

    # Post-fit residual analysis
    try:
        all_res, stats, label_bias = run_residual_analysis(tit)
        all_res.to_csv(out_dir / "residuals.csv", index=False)
        stats.to_csv(out_dir / "residual_stats.csv")
        label_bias.to_csv(out_dir / "label_bias.csv")

        print("  Residual std by label (should be ~1.0 if errors are correct):")
        for lbl, grp in all_res.groupby("label"):
            std = grp["resid_weighted"].std()
            print(f"    {lbl}: std={std:.3f}")

        fig_rp = plot_residual_vs_predicted(all_res)
        fig_rp.savefig(out_dir / "residual_vs_predicted.png", dpi=150)
        plt.close(fig_rp)

        fig_re = plot_residual_vs_yerr(all_res)
        fig_re.savefig(out_dir / "residual_vs_yerr.png", dpi=150)
        plt.close(fig_re)
    except Exception as exc:
        print(f"  [WARN] Residual analysis failed: {exc}")

    return fits


def main() -> None:
    """Run variance-mean and residual analysis for all datasets."""
    print("=" * 70)
    print("Error Model Analysis for ClopHfit — all datasets")
    print("=" * 70)

    all_fits: list[pd.DataFrame] = []

    for ds_name, cfg in DATASETS.items():
        print(f"\nLoading {ds_name}...")
        try:
            tit = load_titration(cfg)
        except Exception as exc:
            print(f"  [SKIP] Could not load {ds_name}: {exc}")
            continue

        out_dir = OUTPUT_DIR / ds_name
        fits = _analyse_dataset(ds_name, tit, out_dir)
        fits.insert(0, "dataset", ds_name)
        all_fits.append(fits)

    # Combined summary CSV
    if all_fits:
        combined = pd.concat(all_fits, ignore_index=True)
        combined.to_csv(OUTPUT_DIR / "all_datasets_noise_params.csv", index=False)
        print(f"\n{'=' * 70}")
        print("Combined noise-model parameters across all datasets:")
        print(combined[["dataset", "label", "gain", "alpha", "sigma_read", "r2"]].to_string(index=False))
        print(f"\nFull results in: {OUTPUT_DIR}")

    print("""
Interpretation:
  gain >> 1 : shot noise underestimated (high-PMT-gain regime)
  gain << 1 : shot noise overestimated (high-count / low-gain)
  alpha > 0 : multiplicative noise present (missing from current model)
  std_resid >> 1 : errors are systematically too small for that label
""")


if __name__ == "__main__":
    main()
