# Residuals Module - Quick Reference

## Import Guide

### For General Use (Library)

```python
from clophfit.fitting.residuals import (
    extract_residual_points,    # Extract with metadata
    residual_dataframe,          # Convert to DataFrame
    collect_multi_residuals,     # Aggregate multi-well
    residual_statistics,         # Compute statistics
    validate_residuals,          # Quality checks
)
```

### For Noise Analysis (Research)

```python
from dev.noise_models import (
    compute_residual_covariance,    # Covariance matrices
    analyze_label_bias,             # Systematic bias (Issue 1)
    detect_adjacent_correlation,    # Adjacent correlation (Issue 2)
    estimate_x_shift_statistics,    # pH shifts (Issue 3)
    simulate_correlated_noise,      # Generate synthetic
    export_noise_parameters,        # Save parameters
)
```

## Quick Examples

### Single Fit Validation

```python
fr = fit_binding_glob(dataset)
checks = validate_residuals(fr)
if not all(checks.values()):
    print("Warning: potential issues!")
    df = residual_dataframe(fr)
    print(df.describe())
```

### Multi-Well Comparison

```python
# Fit multiple methods
lm_results = {k: fit_binding_glob(ds) for k, ds in datasets.items()}
pymc_results = fit_binding_pymc_multi(lm_results, scheme)

# Collect residuals
lm_res = collect_multi_residuals(lm_results).assign(method='LM')
pymc_res = collect_multi_residuals(pymc_results).assign(method='PyMC')

# Compare
all_res = pd.concat([lm_res, pymc_res])
for method, g in all_res.groupby('method'):
    print(f"{method}:")
    print(residual_statistics(g))
```

### Noise Characterization

```python
# Collect from PyMC multi-well fit
fit_results = fit_binding_pymc_multi(results, scheme)
all_res = collect_multi_residuals(fit_results)

# Analyze the 3 key issues
bias_summary, label_bias = analyze_label_bias(all_res)
corr_stats, corr_by_label = detect_adjacent_correlation(all_res)
shift_stats = estimate_x_shift_statistics(all_res, fit_results)

# Compute covariance for synthetic data
cov_by_label = compute_residual_covariance(all_res)
export_noise_parameters(cov_by_label, label_bias, corr_stats)
```

## Function Summary

| Function                        | Module       | Purpose                |
| ------------------------------- | ------------ | ---------------------- |
| `extract_residual_points()`     | residuals    | Extract with metadata  |
| `residual_dataframe()`          | residuals    | Single fit → DataFrame |
| `collect_multi_residuals()`     | residuals    | Multi-well → DataFrame |
| `residual_statistics()`         | residuals    | Stats by label         |
| `validate_residuals()`          | residuals    | Quality checks         |
| `compute_residual_covariance()` | noise_models | Cov matrices           |
| `analyze_label_bias()`          | noise_models | Issue 1 analysis       |
| `detect_adjacent_correlation()` | noise_models | Issue 2 analysis       |
| `estimate_x_shift_statistics()` | noise_models | Issue 3 analysis       |
| `simulate_correlated_noise()`   | noise_models | Synthetic noise        |
| `export_noise_parameters()`     | noise_models | Save to dev/           |

## Data Flow

```
FitResult
    ↓
extract_residual_points()  →  list[ResidualPoint]
    ↓
residual_dataframe()  →  DataFrame (single well)
    ↓
collect_multi_residuals()  →  DataFrame (all wells)
    ↓
residual_statistics()  →  Summary by label
validate_residuals()  →  Quality checks

# Noise-specific
collect_multi_residuals()
    ↓
analyze_label_bias()  →  Issue 1 results
detect_adjacent_correlation()  →  Issue 2 results
estimate_x_shift_statistics()  →  Issue 3 results
compute_residual_covariance()  →  Cov matrices
    ↓
export_noise_parameters()  →  dev/*.csv
```

## Migration from Notebook

**Old** (notebook-specific):

```python
# Cell in notebook
@dataclass
class ResidualPoint:
    ...

def residual_points(fr):
    ...
```

**New** (use library):

```python
# Import from library
from clophfit.fitting.residuals import extract_residual_points
```

## Testing

```bash
# Test general residuals
pytest tests/test_fitting.py -k residual

# Test noise models (when implemented)
pytest tests/test_noise_models.py
```
