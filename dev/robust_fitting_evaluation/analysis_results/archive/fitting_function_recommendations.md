# FITTING FUNCTION ANALYSIS RECOMMENDATIONS

\============================================================

## 🏆 TOP PERFORMING FUNCTIONS

Based on stability score (success rate + accuracy + outlier robustness):

1. **fit_binding_lm**

   - Stability Score: 60.0/100
   - Success Rate: 100.0%
   - Parameter Accuracy: 5.618
   - Outlier Robustness: 100.0%
   - Avg Execution Time: 0.066s

1. **fit_binding_lm_outlier**

   - Stability Score: 60.0/100
   - Success Rate: 100.0%
   - Parameter Accuracy: 5.618
   - Outlier Robustness: 100.0%
   - Avg Execution Time: 0.040s

1. **fit_binding (lm)**

   - Stability Score: 60.0/100
   - Success Rate: 100.0%
   - Parameter Accuracy: 5.618
   - Outlier Robustness: 100.0%
   - Avg Execution Time: 0.039s

1. **fit_binding (lm_outlier)**

   - Stability Score: 60.0/100
   - Success Rate: 100.0%
   - Parameter Accuracy: 5.618
   - Outlier Robustness: 100.0%
   - Avg Execution Time: 0.040s

1. **fit_binding_glob**

   - Stability Score: 60.0/100
   - Success Rate: 100.0%
   - Parameter Accuracy: 5.618
   - Outlier Robustness: 100.0%
   - Avg Execution Time: 0.053s

## 🔍 DUPLICATE/EQUIVALENT FUNCTIONS

These functions produce nearly identical results:

- **fit_binding_lm ≈ fit_binding_lm_outlier**

  - Mean difference: 0.00e+00
  - Based on 10 comparisons

- **fit_binding_lm ≈ fit_lm**

  - Mean difference: 0.00e+00
  - Based on 10 comparisons

- **fit_binding_lm ≈ fit_lm (outlier)**

  - Mean difference: 0.00e+00
  - Based on 10 comparisons

- **fit_binding_lm_outlier ≈ fit_lm**

  - Mean difference: 0.00e+00
  - Based on 10 comparisons

- **fit_binding_lm_outlier ≈ fit_lm (outlier)**

  - Mean difference: 0.00e+00
  - Based on 10 comparisons

- **fit_binding (lm) ≈ fit_binding_lm**

  - Mean difference: 0.00e+00
  - Based on 10 comparisons

- **fit_binding (lm) ≈ fit_binding_lm_outlier**

  - Mean difference: 0.00e+00
  - Based on 10 comparisons

- **fit_binding (lm) ≈ fit_binding (lm_outlier)**

  - Mean difference: 0.00e+00
  - Based on 10 comparisons

- **fit_binding (lm) ≈ fit_binding_glob**

  - Mean difference: 0.00e+00
  - Based on 10 comparisons

- **fit_binding (lm) ≈ fit_lm**

  - Mean difference: 0.00e+00
  - Based on 10 comparisons

## ⚡ FAST & RELIABLE FUNCTIONS

Functions with below-median execution time and high stability:

No functions found that are both fast and highly stable.

## ⚠️ PROBLEMATIC FUNCTIONS

Functions with low stability scores (\<30):

- **api_reweighted** (Score: 0.0)
  - Low success rate
  - Poor parameter recovery
  - Not robust to outliers

## 🎯 USE CASE RECOMMENDATIONS

### For General Use (high success rate + outlier robustness):

- `fit_binding_lm`
- `fit_binding_lm_outlier`
- `fit_binding (lm)`

### For Speed-Critical Applications (fast + reliable):

- `fit_binding_lm` (0.066s)
- `fit_binding_lm_outlier` (0.040s)
- `fit_binding (lm)` (0.039s)

### For Maximum Parameter Accuracy:

- Use top-ranked functions from stability score

## 🧹 CLEANUP RECOMMENDATIONS

### Functions to Consider for Deprecation:

- `api_reweighted` - Poor performance (score: 0.0)

### API Consolidation:

- Keep the unified `fit_binding()` dispatcher as the primary interface
- Maintain the best-performing direct functions for advanced users
- Remove deprecated shim functions after migration

## 📋 MIGRATION PLAN

1. **Phase 1**: Update documentation to recommend top performers
1. **Phase 2**: Add deprecation warnings to poor performers
1. **Phase 3**: Consolidate duplicate functions
1. **Phase 4**: Remove deprecated functions in next major version

## 📊 STATISTICAL NOTES

- Analysis based on 16 functions across 100 simulations
- Results are specific to the tested conditions (2 DataArrays, outliers in last 2 pH points)
- Consider running additional tests with different data characteristics
- Parameter recovery error is normalized relative error across all parameters
