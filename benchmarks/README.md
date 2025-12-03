# Robust Fitting Benchmarks

This directory contains scripts for evaluating and benchmarking the robust fitting methods in ClopHfit.

## Scripts

### `benchmark.py`

Quick benchmark comparing fitting methods on synthetic and real data (L2 dataset).

```bash
python benchmarks/benchmark.py
```

### `stress_test.py`

**Time-consuming.** Comprehensive stress test comparing fit methods under challenging scenarios:

- 10-30% outlier rates
- Low-pH signal drops (acidic tail collapse)
- Large x-errors
- Correlated channel errors
- Missing/saturated points
- Compares PyMC Bayesian methods

```bash
python benchmarks/stress_test.py
```

### `compare_real_data.py`

Loads all real experimental data (L1, L2, L4, 140220) including controls and unknowns using the full `Titration` pipeline (buffer subtraction + dilution correction).

```bash
python benchmarks/compare_real_data.py
```

### `compare_fitting_methods.py`

Comprehensive comparison of all fitting methods generating relevant graphics.

```bash
python benchmarks/compare_fitting_methods.py
```

### `compare_error_models.py`

Evaluates error modeling approaches using residual distribution analysis on both real and synthetic data.

```bash
python benchmarks/compare_error_models.py
```

### `outlier_magnitude_benchmark.py`

Tests how methods perform as outlier severity increases from 0σ to 10σ.

```bash
python benchmarks/outlier_magnitude_benchmark.py
```

### Synthetic Data Generation

Synthetic data is generated using `clophfit.testing.synthetic` module which provides:

- `make_dataset()` - Unified function for all synthetic data generation with configurable error models, outliers, and stress factors
- `make_simple_dataset()` - Simplified interface for unit tests

## Key Results

### Residual Normality Analysis

Checks if standardized residuals follow Gaussian distribution (critical for valid uncertainty estimates):

**Real Data (38 controls):**

| Method      | Shapiro p-value | Excess Kurtosis | Normality |
| ----------- | --------------- | --------------- | --------- |
| lm_standard | 0.003           | 1.2             | Rejected  |
| lm_robust   | 0.008           | 0.9             | Rejected  |
| outlier2    | 0.015           | 0.8             | Rejected  |

**Synthetic with Outliers (N=100):**

| Method      | Shapiro p-value | Excess Kurtosis | Normality |
| ----------- | --------------- | --------------- | --------- |
| lm_standard | \<0.001         | 2.5             | Rejected  |
| lm_robust   | 0.02            | 1.1             | Rejected  |
| outlier2    | 0.08            | 0.5             | Marginal  |

**Conclusion**: All methods show deviation from normality on real data, indicating the error model may need refinement. `outlier2`'s approach produces the most Gaussian residuals on synthetic data with outliers.

### Outlier Magnitude Sensitivity

| Method         | Bias @10σ | RMSE @10σ | Coverage @10σ |
| -------------- | --------- | --------- | ------------- |
| Standard WLS   | ~0        | ~0.09     | ~95%          |
| Robust (Huber) | +0.06     | ~0.12     | ~95%          |

### Scale Covariance Analysis

| Method       | Clean Coverage | Outlier Coverage | Outlier Bias |
| ------------ | -------------- | ---------------- | ------------ |
| lm_standard  | 94%            | 99%              | +0.002       |
| lm_recursive | 100%           | 100%             | +0.037       |
| lm_robust    | 56%            | 61%              | +0.084       |

## Recommendations

### Primary: `fit_binding_glob()` (Standard WLS)

- Excellent 95% CI coverage (~94-99%)
- Near-zero bias
- Fast execution
- Proper error weighting handles heteroscedastic noise

### Alternative: `outlier2()` (Robust with outlier detection)

Use when:

- Explicit outlier identification and masking needed
- Residual normality important for downstream analysis
- Datasets with "acidic tail collapse"

### Why Standard WLS Works Well

1. **Heteroscedastic weighting**: When y1_err >> y2_err, outliers in y1 naturally have less influence
1. **No information loss**: Robust methods may discard valid data
1. **Proper uncertainty**: Covariance scaling (`scale_covar=True`) ensures honest estimates

## Future Work

1. **Method selection logic** based on estimated error ratios
1. **Error scaling characterization** across experimental conditions
1. **Error estimation tools** to guide method selection
1. **Adaptive fitting pipelines** that select methods based on data characteristics
