# Robust Fitting Benchmarks

This directory contains scripts for evaluating and benchmarking the robust fitting methods in ClopHfit.

## Setup

All benchmarks use the `clophfit.testing.synthetic.make_dataset()` function for synthetic data generation with configurable error models, outliers, and stress factors.

Real data benchmarks require test fixtures in `tests/Tecan/` (L1, L2, L4, 140220 datasets).

## Shared Utilities

`benchmark_utils.py` provides common infrastructure:

- NaN-safe metric computation (RMSE, coverage, summary stats)
- Provenance tracking (timestamps, git commits)
- Logging configuration
- Statistical comparison utilities

## Scripts

### Core Benchmarks

#### `benchmark.py`

Quick benchmark comparing fitting methods on synthetic and real data.

```bash
python benchmarks/benchmark.py [--n-samples N] [--seed SEED] [--verbose]
```

Results: `benchmarks/benchmark_results.png`, `benchmarks/benchmark_summary.csv`

#### `benchmark_fitters.py`

Benchmark multiple fitters on synthetic datasets with randomized signal parameters matching real L4 data.

```bash
python benchmarks/benchmark_fitters.py
```

Features:

- Realistic signal magnitude randomization
- Per-label error scaling (y1 vs y2 differential noise)
- Statistical significance testing

### Comprehensive Comparisons

#### `focused_comparison.py`

Focused comparison of `fit_binding_glob` variants evaluating 95% CI coverage.

```bash
python benchmarks/focused_comparison.py
```

Results: `benchmarks/focused_comparison_*.csv`

#### `comprehensive_fitter_comparison.py`

Comprehensive comparison including ODR methods with residual normality analysis.

```bash
python benchmarks/comprehensive_fitter_comparison.py
```

**Note**: Large script (524 lines), includes deprecated methods.

#### `compare_fitting_methods.py`

Compare `outlier2` vs `fit_binding_glob` variants on synthetic and real data.

```bash
python benchmarks/compare_fitting_methods.py
```

**Note**: Very large script (849 lines), may reference removed methods.

#### `compare_error_models.py`

Evaluates error modeling approaches (uniform, shot-noise, physics-based) using residual distribution analysis.

```bash
python benchmarks/compare_error_models.py
```

**Note**: Massive script (825 lines) testing 16+ methods including Bayesian.

### Stress Testing

#### `stress_test.py`

**Time-consuming.** Tests fitting methods under challenging scenarios:

- High noise (3-4x normal)
- Outliers (10-30%)
- Low-pH signal drops (acidic tail collapse)
- Saturation
- Combined stress factors

```bash
python benchmarks/stress_test.py
```

Results: `stress_test_results.csv`

#### `outlier_magnitude_benchmark.py`

Tests performance vs outlier severity (0σ to 10σ).

```bash
python benchmarks/outlier_magnitude_benchmark.py
```

Results: `benchmarks/outlier_magnitude_comparison.png`

### Production Testing

#### `compare_real_data.py`

Tests all fitting methods using the full `Titration` pipeline (buffer subtraction + dilution correction) on real experimental data.

```bash
python benchmarks/compare_real_data.py
```

Results: `comprehensive_fitting_comparison.csv`

### Visualization

#### `visualize_analysis.py`

Regenerate publication-ready plots from saved CSV results.

```bash
python benchmarks/visualize_analysis.py
```

Requires: `fitting_comparison_*.csv` files from `compare_fitting_methods.py`

## Synthetic Data Generation

All synthetic data uses `clophfit.testing.synthetic.make_dataset()` with configurable:

- Error models: `"simple"`, `"realistic"`, `"physics"`
- Outlier probability and magnitude
- Signal parameter randomization
- Stress factors (low-pH drops, saturation, large x-errors)

## Key Results

**Note**: Results below are from historical benchmarks. Re-run scripts to generate current results with provenance tracking.

### Coverage Analysis (95% CI)

Best methods maintain ~94-96% coverage on clean synthetic data and >90% with outliers.

**Typical performance**:

- Standard WLS: ~94% clean, ~99% with outliers (good heteroscedastic weighting)
- Robust (Huber): ~95% clean, ~90% with outliers
- `outlier2`: ~95% clean, ~90% with outliers (explicit outlier masking)

### Outlier Magnitude Sensitivity

Methods tested with outliers from 0σ to 10σ magnitude:

- Standard WLS: maintains low bias (~0) and ~95% coverage even at 10σ
- Robust (Huber): slight positive bias (+0.06) at 10σ, maintains ~95% coverage

### Stress Test Performance

Methods maintaining >95% success rate under severe stress (30% outliers + 4x noise + pH drop):

- Check `stress_test_results.csv` for current results

## Recommendations

### Primary: `fit_binding_glob()` (Standard WLS)

Best for most use cases:

- Excellent 95% CI coverage (~94-99%)
- Near-zero bias on synthetic data
- Fast execution (~0.01s per fit)
- Proper error weighting handles heteroscedastic noise
- No data discarded

**Why it works**: When y1_err >> y2_err (typical in FRET), heteroscedastic weighting naturally downweights outliers in noisy channels.

### Robust variant: `fit_binding_glob(robust=True)`

Use when outliers suspected:

- Huber loss for outlier resistance
- ~95% coverage maintained
- Slight performance cost (~10% slower)

### Explicit outlier detection: `outlier2()`

Use when:

- Explicit outlier identification needed for reporting
- Residual normality critical for downstream analysis
- Datasets with systematic "acidic tail collapse"
- Visual inspection of masked points desired

## Result Provenance

All benchmark results saved with:

- ISO 8601 timestamp
- Git commit hash
- Script name and parameters
- Reproducible random seeds

Check CSV file headers for metadata.

## Known Issues

1. Some scripts reference deprecated methods (`fit_binding_glob_reweighted`, `fit_binding_glob_recursive_outlier`)
1. `compare_fitting_methods.py` and `compare_error_models.py` are very large (>800 lines) and should be modularized
1. Bayesian methods (PyMC) are slow and often commented out in stress tests

## Future Work

1. Modularize large comparison scripts
1. Unified CLI interface for all benchmarks
1. Automated regression testing for fitting method changes
1. Method selection logic based on estimated error ratios
1. Real-time benchmark dashboard
