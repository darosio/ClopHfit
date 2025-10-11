# ClopHfit Fitting Functions Cleanup Recommendations

## Executive Summary

A comprehensive evaluation of 14 fitting functions using corrected realistic synthetic data (with y1 errors ~10x larger than y2 errors) revealed significant insights for codebase cleanup.

### Key Findings:

- **12 out of 14 methods achieved 100% success rate**
- **2 methods failed completely** due to a bug in the weighting function
- **Significant performance differences exist** between methods (ANOVA p < 0.001)
- **Many methods are functional duplicates** with nearly identical results

______________________________________________________________________

## Performance Results

### Top Performing Methods (100% Success Rate)

| Rank | Method                        | K Error (%) | Speed (ms) | Notes            |
| ---- | ----------------------------- | ----------- | ---------- | ---------------- |
| 1    | `fit_lm_standard`             | 0.46        | 37.4       | **Best overall** |
| 2    | `fit_binding_glob_standard`   | 0.46        | 36.9       | **Fastest**      |
| 3    | `api_fit_lm`                  | 0.46        | 37.3       | API wrapper      |
| 4    | `fit_lm_outlier_removal`      | 0.46        | 40.4       | Outlier handling |
| 5    | `fit_lm_robust`               | 0.48        | 49.6       | Huber loss       |
| 6    | `fit_binding_glob_robust`     | 0.48        | 48.1       | Huber loss       |
| 7    | `api_fit_lm_robust`           | 0.48        | 48.6       | API wrapper      |
| 8    | `fit_binding_glob_reweighted` | 0.48        | 76.0       | IRLS method      |

### Failed Methods (0% Success Rate)

- `outlier2`: Failed due to weighting function bug
- `api_fit_lm_outlier`: Failed due to weighting function bug

______________________________________________________________________

## Statistical Analysis

### Clear Performance Tiers:

1. **Tier 1: Excellent (0.46-0.48% K error)**

   - Standard LM methods
   - Basic robust methods
   - Simple IRLS

1. **Tier 2: Good (0.84-0.87% K error)**

   - Advanced iterative methods
   - Complex recursive methods

### Statistically Significant Differences:

- Iterative methods perform significantly worse than standard methods (p < 0.001)
- Recursive methods show similar degradation
- No significant difference within performance tiers

______________________________________________________________________

## Duplicate Methods Analysis

### True Duplicates (Identical Results):

1. **`fit_lm_standard` ≡ `api_fit_lm`**: Perfect functional duplicates
1. **`fit_lm_standard` ≡ `fit_binding_glob_standard`**: Identical implementation
1. **`fit_lm_outlier_removal` ≡ `fit_lm_standard`**: Same results on this dataset
1. **`fit_lm_iterative` ≡ `fit_lm_robust_iterative`**: Identical outputs
1. **`fit_binding_glob_recursive` ≡ `fit_binding_glob_recursive_outlier`**: Same results

### Near Duplicates (Negligible Differences):

- Robust methods cluster together (0.48% error)
- API wrappers mirror their core implementations

______________________________________________________________________

## Cleanup Recommendations

### ✅ **Keep These Methods:**

1. **`fit_lm_standard`**: Best overall performer, primary recommendation
1. **`fit_binding_glob_standard`**: Fastest method, identical to #1
1. **`fit_lm_robust`**: Best robust method for outlier-prone data
1. **`fit_binding_glob_reweighted`**: Best IRLS implementation (current favorite)

### 🔄 **Deprecate These Methods:**

1. **`fit_lm_iterative`**: Complex with worse accuracy
1. **`fit_lm_robust_iterative`**: Duplicate of above
1. **`fit_binding_glob_recursive`**: Slower, less accurate
1. **`fit_binding_glob_recursive_outlier`**: Duplicate of above
1. **API wrappers**: Keep the API but redirect to core implementations

### ❌ **Remove These Methods:**

1. **`outlier2`**: Broken due to weighting bug
1. **`api_fit_lm_outlier`**: Broken due to weighting bug
1. **`fit_lm_outlier_removal`**: Redundant with standard method

______________________________________________________________________

## Bug Fixes Required

### Critical Bug: Dataset Weighting Function

- **Location**: `src/clophfit/fitting/core.py`, `weight_multi_ds_titration()`
- **Error**: `UnboundLocalError: cannot access local variable 'data'`
- **Impact**: Breaks `outlier2` and `api_fit_lm_outlier` functions
- **Priority**: High - Fix before removing methods

______________________________________________________________________

## Implementation Plan

### Phase 1: Bug Fixes

1. Fix weighting function bug in `core.py`
1. Verify affected methods work properly
1. Re-evaluate if needed

### Phase 2: Deprecation Warnings

Add deprecation warnings to methods marked for removal:

```python
warnings.warn(
    "fit_lm_iterative is deprecated. Use fit_lm_standard for better accuracy.",
    DeprecationWarning,
    stacklevel=2,
)
```

### Phase 3: API Consolidation

1. Update API functions to redirect to optimal implementations
1. Maintain backward compatibility
1. Update documentation

### Phase 4: Method Removal

1. Remove deprecated methods after sufficient warning period
1. Clean up unused imports and dependencies
1. Update tests

______________________________________________________________________

## Updated Method Recommendations

### For New Code:

- **Standard fitting**: Use `fit_lm_standard` or `fit_binding_glob_standard`
- **Outlier-robust fitting**: Use `fit_lm_robust`
- **IRLS fitting**: Use `fit_binding_glob_reweighted` (your current preference)
- **High-throughput**: Use `fit_binding_glob_standard` (fastest)

### Migration Guide:

- `fit_lm_iterative` → `fit_lm_standard` (better accuracy)
- `fit_binding_glob_recursive` → `fit_binding_glob_reweighted` (better performance)
- `outlier2` → `fit_lm_robust` (working alternative)

______________________________________________________________________

## Quality Assurance

### Validation Results:

- **100 simulations per method**
- **Realistic experimental conditions** (7 pH points, heteroscedastic errors)
- **Statistical significance testing** (ANOVA + pairwise t-tests)
- **Multiple performance metrics** (accuracy, speed, robustness)

### Confidence Level:

- Results are highly reliable with n=100 simulations
- Corrected error scaling matches real experimental data
- Performance differences are statistically significant

______________________________________________________________________

## Next Steps

1. **Immediate**: Fix the weighting function bug
1. **Short-term**: Add deprecation warnings to poor performers
1. **Medium-term**: Consolidate API and update documentation
1. **Long-term**: Remove deprecated methods after transition period

This cleanup will simplify the codebase while maintaining the best-performing methods for different use cases.
