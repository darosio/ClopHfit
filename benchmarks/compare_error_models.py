#!/usr/bin/env python
"""Compare error modeling approaches for pH titration fitting.

This script validates different error estimation strategies across all fitting methods:

LMfit-based methods:
- fit_binding_glob: Standard least-squares with physics/uniform/shot-noise errors
- fit_binding_glob_reweighted: RLS with outlier removal
- fit_binding_glob_recursive: Iterative reweighting
- fit_binding_glob_recursive_outlier: Recursive with outlier detection
- outlier2: Robust reweighting with outlier detection (uniform/shot-noise)

ODR-based methods:
- fit_binding_odr: Orthogonal distance regression
- fit_binding_odr_recursive: Iterative ODR
- fit_binding_odr_recursive_outlier: ODR with outlier detection

Bayesian methods:
- fit_binding_pymc: PyMC with shared ye_mag scaling
- fit_binding_pymc2: PyMC with separate ye_mag per channel

Tests with:
- Synthetic data (known ground truth)
- Real data from control wells with known pKa (L1, L2, L4, 140220 datasets)
"""

from __future__ import annotations

import copy
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from clophfit.fitting.bayes import fit_binding_pymc, fit_binding_pymc2
from clophfit.fitting.core import (
    fit_binding_glob,
    fit_binding_glob_recursive,
    fit_binding_glob_recursive_outlier,
    fit_binding_glob_reweighted,
    outlier2,
    weight_multi_ds_titration,
)
from clophfit.fitting.data_structures import DataArray, Dataset, FitResult
from clophfit.fitting.models import binding_1site
from clophfit.fitting.odr import (
    fit_binding_odr,
    fit_binding_odr_recursive,
    fit_binding_odr_recursive_outlier,
)
from clophfit.prtecan import Titration

if TYPE_CHECKING:
    from clophfit.clophfit_types import ArrayF

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress PyMC sampling output for cleaner logs
logging.getLogger("pymc").setLevel(logging.WARNING)

# Ground truth parameters for synthetic data (calibrated to match real data)
# Based on analysis of L1, L2, L4, 140220 datasets:
# - Signal ranges: y1 ~100-1500, y2 ~50-1700
# - Buffer SD: ~40-45 (estimated from y_err = sqrt(y + buffer_sd^2))
# - SNR: ~17-19
# - n_points: typically 7-14
TRUE_K = 7.0
TRUE_S0_Y1 = 600.0  # High signal channel (mean ~580 in real data)
TRUE_S1_Y1 = 50.0
TRUE_S0_Y2 = 500.0  # Lower signal channel (mean ~520 in real data)
TRUE_S1_Y2 = 40.0
BUFFER_SD = 40.0  # Estimated from real data (~45)

# Known pKa values for control samples (from literature/validation)
# E2GFP is the reference with well-characterized pKa ~7.0
KNOWN_PKA = {
    "V224L": 5.81,  # Approximate known value
    "E2GFP": 6.82,  # Reference GFP variant
    "S202N": 6.94,  # Approximate known value
    "NTT": 7.57,  # Approximate known value
    "G03": 7.94,  # Approximate known value
    "V224Q": 8.,  # Approximate known value
}

# Dataset configurations
DATASETS = {
    "L1": {
        "path": "tests/Tecan/L1",
        "is_ph": True,
        "description": "L1 pH titration dataset",
    },
    "L2": {
        "path": "tests/Tecan/L2",
        "is_ph": True,
        "description": "L2 pH titration dataset",
    },
    "L4": {
        "path": "tests/Tecan/L4",
        "is_ph": True,
        "description": "L4 pH titration dataset",
    },
    "140220": {
        "path": "tests/Tecan/140220",
        "is_ph": True,
        "description": "140220 pH titration dataset",
    },
}


@dataclass
class FitComparison:
    """Results from comparing fitting methods on a single well."""

    well: str
    ctrl_name: str
    dataset: str
    true_K: float | None  # None if unknown
    methods: dict[str, tuple[float, float]] = field(default_factory=dict)

    def add_result(self, method: str, K: float, K_err: float) -> None:
        """Add a fitting result."""
        self.methods[method] = (K, K_err)

    def to_dict(self) -> dict:
        """Convert to flat dictionary for DataFrame."""
        result = {
            "dataset": self.dataset,
            "well": self.well,
            "ctrl_name": self.ctrl_name,
            "true_K": self.true_K,
        }
        for method, (K, K_err) in self.methods.items():
            result[f"{method}_K"] = K
            result[f"{method}_err"] = K_err
            if self.true_K is not None:
                result[f"{method}_bias"] = K - self.true_K
                result[f"{method}_coverage"] = abs(K - self.true_K) < 1.96 * K_err
        return result


def generate_synthetic_data(
    n_points: int = 7,
    buffer_sd: float = BUFFER_SD,
    add_outliers: bool = False,
    seed: int | None = None,
    true_K: float = 7.0,
) -> tuple[Dataset, dict[str, float]]:
    """Generate synthetic dual-channel pH titration data matching real data characteristics.

    Based on empirical analysis of L1, L2, L4, 140220 datasets:
    - Typical n_points: 7 pH values
    - Buffer SD: ~40 fluorescence units
    - Signal ranges: y1 ~50-600, y2 ~40-500
    - True measurement error follows sqrt(y + buffer_sd^2)
    - SNR: ~17-19
    """
    if seed is not None:
        np.random.seed(seed)

    # pH range matching real experiments
    x = np.linspace(5.5, 9.0, n_points)
    x_err = 0.05 * np.ones_like(x)  # pH meter precision

    # Generate true signal using binding model
    y1_true = binding_1site(x, true_K, TRUE_S0_Y1, TRUE_S1_Y1, is_ph=True)
    y2_true = binding_1site(x, true_K, TRUE_S0_Y2, TRUE_S1_Y2, is_ph=True)

    # Physics-informed error model: shot noise + buffer noise
    y1_err_true = np.sqrt(np.maximum(y1_true, 1.0) + buffer_sd**2)
    y2_err_true = np.sqrt(np.maximum(y2_true, 1.0) + buffer_sd**2)

    # Add noise
    y1 = y1_true + np.random.normal(0, y1_err_true)
    y2 = y2_true + np.random.normal(0, y2_err_true)

    # Add outliers if requested (mimics real data issues)
    if add_outliers:
        outlier_idx = [1, n_points - 2]
        y1[outlier_idx[0]] += 4 * y1_err_true[outlier_idx[0]]
        y2[outlier_idx[1]] -= 3 * y2_err_true[outlier_idx[1]]

    da1 = DataArray(x, y1, x_errc=x_err, y_errc=y1_err_true)
    da2 = DataArray(x, y2, x_errc=x_err, y_errc=y2_err_true)

    return Dataset({"y1": da1, "y2": da2}, is_ph=True), {"K": true_K}


def extract_K(fr: FitResult) -> tuple[float, float]:
    """Extract K estimate and uncertainty from FitResult."""
    if fr.result is None:
        return np.nan, np.nan

    if hasattr(fr.result, "params"):
        params = fr.result.params
        K = params["K"].value
        K_err = params["K"].stderr if params["K"].stderr else 0.1
        return K, K_err

    return np.nan, np.nan


def extract_K_from_trace(fr: FitResult) -> tuple[float, float]:
    """Extract K from Bayesian FitResult."""
    if fr.result is None:
        return np.nan, np.nan

    if hasattr(fr.result, "params"):
        params = fr.result.params
        if "K" in params:
            K = params["K"].value
            K_err = params["K"].stderr if params["K"].stderr else 0.1
            return K, K_err

    return np.nan, np.nan


def fit_all_methods(
    ds: Dataset, key: str = "test", run_bayesian: bool = True, n_sd: float = 5.0
) -> dict[str, tuple[float, float]]:
    """Fit dataset with all methods and return K estimates."""
    results = {}

    # === LMfit-based methods ===

    # Method 1: Physics-informed errors (keep original y_err)
    ds_physics = copy.deepcopy(ds)
    fr_physics = fit_binding_glob(ds_physics, robust=False)
    results["lm_physics"] = extract_K(fr_physics)

    # Method 2: Physics with robust fitting
    ds_robust = copy.deepcopy(ds)
    fr_robust = fit_binding_glob(ds_robust, robust=True)
    results["lm_robust"] = extract_K(fr_robust)

    # Method 3: outlier2 with uniform error model
    ds_out2_uni = copy.deepcopy(ds)
    fr_out2_uni = outlier2(ds_out2_uni, key=key, error_model="uniform")
    results["outlier2_uniform"] = extract_K(fr_out2_uni)

    # Method 4: outlier2 with shot-noise error model
    ds_out2_shot = copy.deepcopy(ds)
    fr_out2_shot = outlier2(ds_out2_shot, key=key, error_model="shot-noise")
    results["outlier2_shotnoise"] = extract_K(fr_out2_shot)

    # Method 5: Reweighted fitting
    ds_reweight = copy.deepcopy(ds)
    fr_reweight = fit_binding_glob_reweighted(ds_reweight, key=key, threshold=2.5)
    results["lm_reweighted"] = extract_K(fr_reweight)

    # Method 6: Recursive fitting
    ds_recursive = copy.deepcopy(ds)
    fr_recursive = fit_binding_glob_recursive(ds_recursive, tol=0.01)
    results["lm_recursive"] = extract_K(fr_recursive)

    # Method 7: Recursive with outlier detection
    ds_rec_out = copy.deepcopy(ds)
    fr_rec_out = fit_binding_glob_recursive_outlier(ds_rec_out, tol=0.01, threshold=3.0)
    results["lm_recursive_outlier"] = extract_K(fr_rec_out)

    # Method 8: weight_da (SEM-based)
    ds_weight = copy.deepcopy(ds)
    for da in ds_weight.values():
        da.y_errc = np.ones_like(da.xc)
    weight_multi_ds_titration(ds_weight)
    fr_weight = fit_binding_glob(ds_weight, robust=False)
    results["lm_weight_da"] = extract_K(fr_weight)

    # === ODR-based methods ===

    # Method 9: ODR with physics errors
    if fr_physics.result is not None:
        try:
            fr_odr = fit_binding_odr(fr_physics)
            results["odr_physics"] = extract_K(fr_odr)
        except Exception:
            results["odr_physics"] = (np.nan, np.nan)

    # Method 10: ODR with outlier2 uniform errors
    if fr_out2_uni.result is not None:
        try:
            fr_odr_uni = fit_binding_odr(fr_out2_uni)
            results["odr_uniform"] = extract_K(fr_odr_uni)
        except Exception:
            results["odr_uniform"] = (np.nan, np.nan)

    # Method 11: ODR recursive
    if fr_physics.result is not None:
        try:
            fr_odr_rec = fit_binding_odr_recursive(fr_physics, tol=0.01)
            results["odr_recursive"] = extract_K(fr_odr_rec)
        except Exception:
            results["odr_recursive"] = (np.nan, np.nan)

    # Method 12: ODR recursive with outlier
    if fr_physics.result is not None:
        try:
            fr_odr_rec_out = fit_binding_odr_recursive_outlier(
                fr_physics, tol=0.01, threshold=3.0
            )
            results["odr_recursive_outlier"] = extract_K(fr_odr_rec_out)
        except Exception:
            results["odr_recursive_outlier"] = (np.nan, np.nan)

    # === Bayesian methods (optional, slower) ===
    if run_bayesian:
        # Method 13: PyMC with physics-informed errors + shared ye_mag
        if fr_physics.result is not None:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fr_pymc_physics = fit_binding_pymc(
                        fr_physics, n_sd=n_sd, n_xerr=0, n_samples=1000
                    )
                results["pymc_physics"] = extract_K_from_trace(fr_pymc_physics)
            except Exception as e:
                logger.warning(f"PyMC physics failed for {key}: {e}")
                results["pymc_physics"] = (np.nan, np.nan)

        # Method 14: PyMC with outlier2 uniform errors
        if fr_out2_uni.result is not None:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fr_pymc_uni = fit_binding_pymc(
                        fr_out2_uni, n_sd=n_sd, n_xerr=0, n_samples=1000
                    )
                results["pymc_uniform"] = extract_K_from_trace(fr_pymc_uni)
            except Exception as e:
                logger.warning(f"PyMC uniform failed for {key}: {e}")
                results["pymc_uniform"] = (np.nan, np.nan)

        # Method 15: PyMC2 with physics + separate ye_mag per channel
        if fr_physics.result is not None:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fr_pymc2_phys = fit_binding_pymc2(
                        fr_physics, n_sd=n_sd, n_xerr=0, n_samples=1000
                    )
                results["pymc2_physics"] = extract_K_from_trace(fr_pymc2_phys)
            except Exception as e:
                logger.warning(f"PyMC2 physics failed for {key}: {e}")
                results["pymc2_physics"] = (np.nan, np.nan)

        # Method 16: PyMC2 with outlier2 uniform + separate ye_mag
        if fr_out2_uni.result is not None:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fr_pymc2_uni = fit_binding_pymc2(
                        fr_out2_uni, n_sd=n_sd, n_xerr=0, n_samples=1000
                    )
                results["pymc2_uniform"] = extract_K_from_trace(fr_pymc2_uni)
            except Exception as e:
                logger.warning(f"PyMC2 uniform failed for {key}: {e}")
                results["pymc2_uniform"] = (np.nan, np.nan)

    return results


def analyze_residuals(
    ds: Dataset, key: str = ""
) -> dict[str, dict[str, float]]:
    """Analyze residual distributions for heteroscedasticity testing."""
    methods = {
        "lm_physics": fit_binding_glob(copy.deepcopy(ds)),
        "outlier2_uniform": outlier2(copy.deepcopy(ds), key, error_model="uniform"),
        "outlier2_shotnoise": outlier2(copy.deepcopy(ds), key, error_model="shot-noise"),
    }

    results = {}
    for name, fr in methods.items():
        if fr.result is None or fr.dataset is None:
            continue

        # Get weighted residuals from lmfit
        weighted_res = fr.result.residual
        y_all = np.concatenate([da.y for da in fr.dataset.values()])
        y_err = np.concatenate([da.y_err for da in fr.dataset.values()])

        # Different normalizations
        raw_res = weighted_res * y_err
        shot_res = raw_res / np.sqrt(np.abs(y_all) + 1)

        # Normality tests
        _, p_raw = stats.shapiro(raw_res)
        _, p_weighted = stats.shapiro(weighted_res)
        _, p_shot = stats.shapiro(shot_res)

        results[name] = {
            "raw_std": float(np.std(raw_res)),
            "raw_shapiro_p": p_raw,
            "weighted_std": float(np.std(weighted_res)),
            "weighted_shapiro_p": p_weighted,
            "shot_std": float(np.std(shot_res)),
            "shot_shapiro_p": p_shot,
        }

    return results


def run_synthetic_study(
    n_simulations: int = 30, add_outliers: bool = False, run_bayesian: bool = False
) -> pd.DataFrame:
    """Run simulation study with synthetic data."""
    all_results = []

    for i in range(n_simulations):
        if i % 10 == 0:
            logger.info(f"Synthetic simulation {i + 1}/{n_simulations}")
        try:
            ds, _ = generate_synthetic_data(add_outliers=add_outliers, seed=i)
            methods = fit_all_methods(ds, key=f"sim_{i}", run_bayesian=run_bayesian)

            comparison = FitComparison(
                well=f"sim_{i}",
                ctrl_name="synthetic",
                dataset="synthetic",
                true_K=TRUE_K,
            )
            for method, (K, K_err) in methods.items():
                comparison.add_result(method, K, K_err)
            all_results.append(comparison.to_dict())
        except Exception as e:
            logger.warning(f"Simulation {i} failed: {e}")

    return pd.DataFrame(all_results)


def load_titration(data_path: Path, is_ph: bool = True) -> Titration:
    """Load titration data from a dataset directory."""
    listfile = data_path / "list.pH.csv"
    titan = Titration.fromlistfile(listfile, is_ph=is_ph)

    additions_file = data_path / "additions.pH"
    if additions_file.exists():
        titan.load_additions(additions_file)

    scheme_file = data_path / "scheme.txt"
    if scheme_file.exists():
        titan.load_scheme(scheme_file)

    # Enable background subtraction and normalization (as per CLI --nrm)
    titan.params.bg = True
    titan.params.nrm = True  # Use normalized data for "almost equal" label groups
    titan.params.dil = True

    return titan


def run_single_dataset(
    dataset_name: str, data_path: Path, is_ph: bool = True, run_bayesian: bool = True
) -> pd.DataFrame:
    """Run comparison on a single dataset's control wells."""
    titan = load_titration(data_path, is_ph=is_ph)
    all_results = []

    # Get control wells and their sample names
    ctrl_wells = titan.scheme.ctrl
    names_map = titan.scheme.names

    # Invert names_map to get well -> sample_name
    well_to_name = {}
    for name, wells in names_map.items():
        for well in wells:
            well_to_name[well] = name

    logger.info(
        f"[{dataset_name}] Found {len(ctrl_wells)} control wells: {ctrl_wells}"
    )
    logger.info(f"[{dataset_name}] Sample names: {list(names_map.keys())}")

    for well in ctrl_wells:
        ctrl_name = well_to_name.get(well, "unknown")
        true_K = KNOWN_PKA.get(ctrl_name)

        logger.info(
            f"[{dataset_name}] Fitting well {well} ({ctrl_name}, true pKa={true_K})"
        )

        try:
            ds = titan._create_global_ds(well)
            methods = fit_all_methods(
                ds, key=well, run_bayesian=run_bayesian, n_sd=10.0
            )

            comparison = FitComparison(
                well=well,
                ctrl_name=ctrl_name,
                dataset=dataset_name,
                true_K=true_K,
            )
            for method, (K, K_err) in methods.items():
                comparison.add_result(method, K, K_err)
            all_results.append(comparison.to_dict())

        except Exception as e:
            logger.warning(f"[{dataset_name}] Failed to fit well {well}: {e}")

    return pd.DataFrame(all_results)


def run_all_datasets(
    base_path: Path, run_bayesian: bool = True
) -> pd.DataFrame:
    """Run comparison on all configured datasets."""
    all_results = []

    for name, config in DATASETS.items():
        data_path = base_path / config["path"]
        if not data_path.exists():
            logger.warning(f"Dataset {name} not found at {data_path}")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Processing dataset: {name} ({config['description']})")
        logger.info(f"{'='*60}")

        df = run_single_dataset(
            name, data_path, is_ph=config["is_ph"], run_bayesian=run_bayesian
        )
        all_results.append(df)

    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()


def summarize_by_method(df: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    """Summarize results by method."""
    summary_data = []

    for method in methods:
        K_col = f"{method}_K"
        err_col = f"{method}_err"
        bias_col = f"{method}_bias"
        cov_col = f"{method}_coverage"

        if K_col not in df.columns:
            continue

        row = {"method": method}
        row["K_mean"] = df[K_col].mean()
        row["K_std"] = df[K_col].std()
        row["K_err_mean"] = df[err_col].mean()

        if bias_col in df.columns:
            valid_bias = df[bias_col].dropna()
            if len(valid_bias) > 0:
                row["bias_mean"] = valid_bias.mean()
                row["bias_std"] = valid_bias.std()
                row["RMSE"] = np.sqrt((valid_bias**2).mean())
            else:
                row["bias_mean"] = np.nan
                row["bias_std"] = np.nan
                row["RMSE"] = np.nan

        if cov_col in df.columns:
            valid_cov = df[cov_col].dropna()
            row["coverage_95CI"] = valid_cov.mean() if len(valid_cov) > 0 else np.nan
            row["n_with_known_pKa"] = len(valid_cov)

        summary_data.append(row)

    return pd.DataFrame(summary_data).round(4)


def summarize_by_sample(df: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    """Summarize results by sample type AND dataset (not merged across datasets)."""
    summary_data = []

    for dataset in df["dataset"].unique():
        ds_subset = df[df["dataset"] == dataset]
        for ctrl_name in ds_subset["ctrl_name"].unique():
            subset = ds_subset[ds_subset["ctrl_name"] == ctrl_name]
            true_K = subset["true_K"].iloc[0] if pd.notna(subset["true_K"].iloc[0]) else None

            row = {
                "dataset": dataset,
                "sample": ctrl_name,
                "true_pKa": true_K,
                "n_wells": len(subset),
            }

            for method in methods:
                K_col = f"{method}_K"
                err_col = f"{method}_err"
                cov_col = f"{method}_coverage"

                if K_col in subset.columns:
                    row[f"{method}_K"] = subset[K_col].mean()
                    row[f"{method}_err"] = subset[err_col].mean()

                if cov_col in subset.columns:
                    valid_cov = subset[cov_col].dropna()
                    row[f"{method}_cov"] = valid_cov.mean() if len(valid_cov) > 0 else np.nan

            summary_data.append(row)

    return pd.DataFrame(summary_data).round(3)


def plot_comprehensive_comparison(
    df: pd.DataFrame, output_path: Path, methods: list[str]
) -> None:
    """Create comprehensive comparison plots."""
    available_methods = [m for m in methods if f"{m}_K" in df.columns]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: K estimates by sample type
    sample_summary = summarize_by_sample(df, available_methods)
    x_pos = np.arange(len(sample_summary))
    width = 0.15
    colors = plt.cm.tab10(np.linspace(0, 1, len(available_methods)))

    for i, method in enumerate(available_methods):
        K_col = f"{method}_K"
        err_col = f"{method}_err"
        if K_col in sample_summary.columns:
            offset = (i - len(available_methods) / 2) * width
            axes[0, 0].bar(
                x_pos + offset,
                sample_summary[K_col],
                width,
                yerr=sample_summary[err_col],
                label=method,
                color=colors[i],
                alpha=0.8,
            )

    # Add true pKa markers
    for j, (_, row) in enumerate(sample_summary.iterrows()):
        if pd.notna(row.get("true_pKa")):
            axes[0, 0].scatter(j, row["true_pKa"], marker="*", s=200, c="red", zorder=5)

    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(sample_summary["sample"], rotation=45, ha="right")
    axes[0, 0].set_ylabel("pKa estimate")
    axes[0, 0].legend(loc="upper right", fontsize=8)
    axes[0, 0].set_title("Mean pKa by Sample Type (★ = true pKa)")

    # Plot 2: Bias distribution by method (only for known pKa)
    bias_data = []
    for method in available_methods:
        bias_col = f"{method}_bias"
        if bias_col in df.columns:
            for bias in df[bias_col].dropna():
                bias_data.append({"method": method, "bias": bias})
    bias_df = pd.DataFrame(bias_data)

    if not bias_df.empty:
        bias_df.boxplot(column="bias", by="method", ax=axes[0, 1])
        axes[0, 1].axhline(0, color="red", linestyle="--", linewidth=1)
        axes[0, 1].set_title("Bias Distribution (known pKa only)")
        axes[0, 1].set_ylabel("Bias (estimated - true)")
        plt.suptitle("")

    # Plot 3: Standard error distribution
    err_data = []
    for method in available_methods:
        err_col = f"{method}_err"
        if err_col in df.columns:
            for err in df[err_col].dropna():
                err_data.append({"method": method, "stderr": err})
    err_df = pd.DataFrame(err_data)

    if not err_df.empty:
        err_df.boxplot(column="stderr", by="method", ax=axes[1, 0])
        axes[1, 0].set_title("Standard Error Distribution")
        axes[1, 0].set_ylabel("K stderr")
        plt.suptitle("")

    # Plot 4: Coverage by method
    method_summary = summarize_by_method(df, available_methods)
    if "coverage_95CI" in method_summary.columns:
        valid_summary = method_summary.dropna(subset=["coverage_95CI"])
        if not valid_summary.empty:
            axes[1, 1].bar(
                valid_summary["method"],
                valid_summary["coverage_95CI"],
                color=colors[: len(valid_summary)],
                alpha=0.8,
            )
            axes[1, 1].axhline(0.95, color="red", linestyle="--", label="Nominal 95%")
            axes[1, 1].set_ylabel("Coverage Probability")
            axes[1, 1].set_title("95% CI Coverage (known pKa)")
            axes[1, 1].legend()
            axes[1, 1].set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Saved plot to {output_path}")


def main() -> None:
    """Run the comprehensive comparison study."""
    logger.info("=" * 60)
    logger.info("Comprehensive Error Model Comparison Study")
    logger.info("=" * 60)

    # All methods to compare
    methods = [
        # LMfit methods
        "lm_physics",
        "lm_robust",
        "outlier2_uniform",
        "outlier2_shotnoise",
        "lm_reweighted",
        "lm_recursive",
        "lm_recursive_outlier",
        "lm_weight_da",
        # ODR methods
        "odr_physics",
        "odr_uniform",
        "odr_recursive",
        "odr_recursive_outlier",
        # Bayesian methods
        "pymc_physics",
        "pymc_uniform",
        "pymc2_physics",
        "pymc2_uniform",
    ]

    # Find base path (script location or CWD)
    script_path = Path(__file__).resolve()
    base_path = script_path.parent.parent  # Go up from scripts/ to repo root

    # Study 1: Synthetic data (fast, no Bayesian)
    logger.info("\n--- Study 1: Synthetic Data (Clean) ---")
    df_synthetic = run_synthetic_study(
        n_simulations=30, add_outliers=False, run_bayesian=False
    )
    summary_synthetic = summarize_by_method(df_synthetic, methods)
    print("\nSynthetic Data (Clean) Results:")
    print(summary_synthetic.to_string())

    # Study 2: Synthetic data with outliers
    logger.info("\n--- Study 2: Synthetic Data (With Outliers) ---")
    df_outliers = run_synthetic_study(
        n_simulations=30, add_outliers=True, run_bayesian=False
    )
    summary_outliers = summarize_by_method(df_outliers, methods)
    print("\nSynthetic Data (Outliers) Results:")
    print(summary_outliers.to_string())

    # Study 3: All real datasets
    logger.info("\n--- Study 3: All Real Datasets (with Bayesian) ---")
    df_real = run_all_datasets(base_path, run_bayesian=True)

    if not df_real.empty:
        # Overall summary
        print("\n" + "=" * 60)
        print("OVERALL RESULTS (All Datasets Combined)")
        print("=" * 60)

        summary_real = summarize_by_method(df_real, methods)
        print("\nSummary by Method:")
        print(summary_real.to_string())

        sample_summary = summarize_by_sample(df_real, methods)
        print("\nSummary by Sample Type:")
        print(sample_summary.to_string())

        # Per-dataset summary
        for dataset in df_real["dataset"].unique():
            subset = df_real[df_real["dataset"] == dataset]
            print(f"\n--- {dataset} ---")
            ds_summary = summarize_by_method(subset, methods)
            print(ds_summary.to_string())

        # Save results
        output_dir = Path(".")
        df_real.to_csv(output_dir / "comprehensive_fitting_comparison.csv", index=False)
        plot_comprehensive_comparison(
            df_real, output_dir / "error_model_comparison.png", methods
        )

        # Heteroscedasticity analysis on all datasets (all control wells)
        logger.info("\n--- Heteroscedasticity Analysis (All Datasets) ---")
        all_residuals: dict[str, dict[str, list[float]]] = {}
        for ds_name, config in DATASETS.items():
            data_path = base_path / config["path"]
            if not data_path.exists():
                continue
            titan = load_titration(data_path)
            wells = titan.scheme.ctrl if titan.scheme.ctrl else []
            for well in wells:
                try:
                    ds_well = titan._create_global_ds(well)
                    res_analysis = analyze_residuals(ds_well, well)
                    for method, stats_dict in res_analysis.items():
                        if method not in all_residuals:
                            all_residuals[method] = {k: [] for k in stats_dict}
                        for stat_name, value in stats_dict.items():
                            all_residuals[method][stat_name].append(value)
                except Exception:
                    pass

        # Print aggregated results
        print("\nResidual Distribution Analysis (all control wells):")
        print(f"{'Method':<20} {'weighted_std':>12} {'weighted_p':>12}")
        print("-" * 46)
        for method, stats_dict in all_residuals.items():
            if "weighted_std" in stats_dict and stats_dict["weighted_std"]:
                avg_std = np.mean(stats_dict["weighted_std"])
                avg_p = np.mean(stats_dict["weighted_shapiro_p"])
                print(f"{method:<20} {avg_std:>12.3f} {avg_p:>12.4f}")

        logger.info(f"\nResults saved to {output_dir}")
    else:
        logger.warning("No real data results to save")

    logger.info("\nComparison study complete!")


if __name__ == "__main__":
    main()
