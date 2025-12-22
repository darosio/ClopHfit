# Residual Analysis Utilities - Organization

## 📦 Module Structure

### General Purpose: `clophfit.fitting.residuals`

**Location:** `src/clophfit/fitting/residuals.py`
**For:** All users analyzing fit quality

Functions for **general residual analysis**:

- `extract_residual_points()` - Extract residuals with metadata
- `residual_dataframe()` - Convert to DataFrame
- `collect_multi_residuals()` - Collect from multiple fits
- `residual_statistics()` - Compute statistics by label
- `validate_residuals()` - Quality checks (bias, outliers, correlation)

### Noise-Specific: `dev.noise_models`

**Location:** `dev/noise_models.py`
**For:** Your noise characterization research

Functions for **noise structure analysis**:

- `compute_residual_covariance()` - Covariance matrices
- `analyze_label_bias()` - Systematic bias detection (Issue 1)
- `detect_adjacent_correlation()` - Adjacent correlation (Issue 2)
- `estimate_x_shift_statistics()` - pH shift detection (Issue 3)
- `simulate_correlated_noise()` - Generate synthetic noise
- `export_noise_parameters()` - Save parameters

## 🎯 Usage Guide

### For General Analysis (All Users)

```python
from clophfit.fitting.residuals import (
    extract_residual_points,
    residual_dataframe,
    collect_multi_residuals,
    residual_statistics,
    validate_residuals,
)

# Single fit analysis
fr = fit_binding_glob(dataset)
df = residual_dataframe(fr)
print(df.describe())

# Validate fit quality
checks = validate_residuals(fr)
if not all(checks.values()):
    print("Warning: potential issues detected")

# Multi-well analysis
results = {well: fit_binding_glob(ds) for well, ds in datasets.items()}
all_res = collect_multi_residuals(results)

# Statistics by label
stats = residual_statistics(all_res)
print(stats)
```

### For Noise Characterization (Your Research)

```python
from clophfit.fitting.residuals import collect_multi_residuals
from dev.noise_models import (
    compute_residual_covariance,
    analyze_label_bias,
    detect_adjacent_correlation,
    estimate_x_shift_statistics,
)

# Collect residuals
all_res = collect_multi_residuals(tit.result_global.results)

# Issue 1: Systematic bias
bias_summary, label_bias = analyze_label_bias(all_res)

# Issue 2: Adjacent correlation
corr_stats, corr_by_label = detect_adjacent_correlation(all_res)

# Issue 3: X-value uncertainty
shift_stats = estimate_x_shift_statistics(all_res, tit.result_global.results)

# Compute covariance for synthetic data
cov_by_label = compute_residual_covariance(all_res)
```

## 📊 Integration with Notebooks

### `01_noise_characterization.ipynb`

```python
# Import general utilities
from clophfit.fitting.residuals import collect_multi_residuals

# Import noise-specific analysis
from dev.noise_models import (
    analyze_label_bias,
    detect_adjacent_correlation,
    estimate_x_shift_statistics,
    compute_residual_covariance,
    export_noise_parameters,
)

# Collect residuals from all methods
pymc_results = fit_binding_pymc_multi(tit.result_global.results, tit.scheme)
all_res = collect_multi_residuals(pymc_results)

# Analyze the 3 key issues
bias_summary, label_bias = analyze_label_bias(all_res)
corr_stats, corr_by_label = detect_adjacent_correlation(all_res)
shift_stats = estimate_x_shift_statistics(all_res, pymc_results)

# Export for synthetic data generation
cov_by_label = compute_residual_covariance(all_res)
export_noise_parameters(cov_by_label, label_bias, corr_stats)
```

### `03_fitting_method_comparison.ipynb`

```python
from clophfit.fitting.residuals import (
    collect_multi_residuals,
    residual_statistics,
)

# Compare methods
lm_results = {k: fit_binding_glob(ds) for k, ds in datasets.items()}
pymc_results = fit_binding_pymc_multi(lm_results, scheme)
gls_results = {k: fit_binding_gls(ds, cov) for k, ds in datasets.items()}

# Collect residuals
lm_res = collect_multi_residuals(lm_results).assign(method='LM')
pymc_res = collect_multi_residuals(pymc_results).assign(method='PyMC')
gls_res = collect_multi_residuals(gls_results).assign(method='GLS')

all_methods = pd.concat([lm_res, pymc_res, gls_res])

# Compare statistics
for method in ['LM', 'PyMC', 'GLS']:
    print(f"\n{method}:")
    print(residual_statistics(all_methods[all_methods.method == method]))
```

## 🔄 Migration Path

### From Notebook Code → Library

**Current** (in notebook):

```python
# In 01_noise_characterization.ipynb
@dataclass(frozen=True)
class ResidualPoint:
    ...

def residual_points(fr):
    ...
```

**New** (use library):

```python
# In notebook
from clophfit.fitting.residuals import (
    extract_residual_points,
    residual_dataframe,
)

# Same functionality, maintained and tested
```

### Backward Compatibility

For existing notebook code that imports from `dev.noise_models`:

```python
# Still works (re-exported for compatibility)
from dev.noise_models import ResidualPoint, extract_residual_points

# But prefer the canonical import
from clophfit.fitting.residuals import extract_residual_points
```

## 🧪 Testing

The general residuals module is tested and maintained as part of the library:

```python
# Run tests
pytest tests/test_fitting.py -k residual
```

## 📝 Decision Log

| What                            | Where                        | Why                          |
| ------------------------------- | ---------------------------- | ---------------------------- |
| `extract_residual_points()`     | `clophfit.fitting.residuals` | General utility, all users   |
| `residual_dataframe()`          | `clophfit.fitting.residuals` | Convenience, all users       |
| `collect_multi_residuals()`     | `clophfit.fitting.residuals` | Common pattern, all users    |
| `residual_statistics()`         | `clophfit.fitting.residuals` | Sanity checks, all users     |
| `validate_residuals()`          | `clophfit.fitting.residuals` | Quality assurance, all users |
| `compute_residual_covariance()` | `dev.noise_models`           | Noise-specific research      |
| `analyze_label_bias()`          | `dev.noise_models`           | Issue 1 - research           |
| `detect_adjacent_correlation()` | `dev.noise_models`           | Issue 2 - research           |
| `estimate_x_shift_statistics()` | `dev.noise_models`           | Issue 3 - research           |
| `simulate_correlated_noise()`   | `dev.noise_models`           | Synthetic data - research    |

## ✅ Benefits

1. **Separation of Concerns**

   - General utilities → library (maintained, tested, documented)
   - Noise-specific → dev (research, experimental)

1. **Reusability**

   - All users can use residual extraction
   - Noise analysis stays in research code

1. **Maintainability**

   - Library code has tests
   - Research code can evolve

1. **Clear Intent**

   - Import location indicates purpose
   - Easy to move noise functions to library later if useful

## 🚀 Next Steps

1. Update your notebooks to import from `clophfit.fitting.residuals`
1. Use noise-specific functions from `dev.noise_models` as needed
1. After validating noise methods, consider promoting useful functions to library
