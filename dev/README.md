# Noise Model Development

This directory contains development notebooks and utilities for characterizing and modeling the noise structure in prtecan fluorescence data.

## Quick Start

Start with the dashboard:

```bash
jupyter notebook 00_noise_model_dashboard.ipynb
```

## Workflow

1. **[00_noise_model_dashboard.ipynb](../00_noise_model_dashboard.ipynb)** - Overview and status tracking
1. **[01_noise_characterization.ipynb](../01_noise_characterization.ipynb)** - Analyze real data
1. **[02_synthetic_data_generator.ipynb](../02_synthetic_data_generator.ipynb)** - Build realistic synthetic data
1. **[03_fitting_method_comparison.ipynb](../03_fitting_method_comparison.ipynb)** - Compare GLS vs PyMC

## Key Issues Being Investigated

### Issue 1: Systematic Bias

**Observation:** y1 label at lowest pH is consistently negative (sometimes >3σ)
**Analysis:** `analyze_label_bias()` in `noise_models.py`

### Issue 2: Adjacent Point Correlation

**Observation:** Residuals alternate positive/negative at adjacent points
**Hypothesis:** Either x-value errors or model misspecification
**Analysis:** `detect_adjacent_correlation()` in `noise_models.py`

### Issue 3: X-value Uncertainty

**Observation:** Systematic patterns suggest pH values may be wrong
**Hypothesis:** Per-well or plate-wide pH shifts
**Analysis:** `estimate_x_shift_statistics()` in `noise_models.py`

## Utilities Module

**[noise_models.py](./noise_models.py)** - Reusable functions:

- `residual_points()` - Extract residuals from fit results
- `compute_residual_covariance()` - Covariance matrices by label
- `compute_correlation_matrices()` - Convert covariance to correlation
- `analyze_label_bias()` - Detect systematic bias (Issue 1)
- `detect_adjacent_correlation()` - Test adjacent correlation (Issue 2)
- `estimate_x_shift_statistics()` - Detect pH shifts (Issue 3)
- `simulate_correlated_noise()` - Generate correlated noise samples
- `export_noise_parameters()` - Save parameters for synthetic data

## Output Files

After running `01_noise_characterization.ipynb`, the following files will be saved:

- `dev/cov_matrix_y1.csv` - Covariance matrix for y1 label
- `dev/cov_matrix_y2.csv` - Covariance matrix for y2 label
- `dev/label_bias.csv` - Bias statistics by label
- `dev/correlation_summary.csv` - Lag-1 correlation statistics

These are used by `02_synthetic_data_generator.ipynb` to create realistic synthetic data.

## Method Comparison Goal

Compare **Generalized Least Squares (GLS)** vs **PyMC** approaches:

- **GLS:** Use characterized covariance structure explicitly
- **PyMC:** Hierarchical model learns error structure
- **Metrics:** Parameter recovery, computational cost, robustness

## Original Notebook

The original development notebook `prtecan_devel.ipynb` (90 cells) was split into three focused notebooks for better organization and clarity.
