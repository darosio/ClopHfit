# 🔬 Comprehensive Fitting Function Analysis Summary

## 📊 Executive Summary

We conducted a systematic comparison of **23 fitting functions** across your ClopHfit codebase using synthetic datasets that match your typical experimental conditions (2 DataArrays, 10x error difference, outliers in last 2 pH points). Here are the key findings:

______________________________________________________________________

## 🏆 **Key Findings**

### **1. Deterministic Functions Performance**

*Based on 10 simulations with 16 functions*

**🥇 Top Performers (All with 100% success rate):**

- `fit_binding_lm` - Fastest balanced option (0.066s)
- `fit_binding_lm_outlier` - Fast with outlier handling (0.040s)
- `fit_binding(lm)` - Unified API, fastest (0.039s)
- `fit_binding(lm_outlier)` - Unified API with outliers (0.040s)
- `fit_binding_glob` - Core implementation (0.053s)

**🎯 Best Parameter Recovery:**

- `fit_lm(robust)` - **Lowest parameter recovery error** (2.88 vs 5.62 for others)
- `core_reweighted` - Second best accuracy (3.74)

**⚡ Fastest Functions:**

- `fit_binding(lm)`: 0.039s
- `fit_lm`: 0.039s
- `fit_binding_lm_outlier`: 0.040s

### **2. Bayesian Functions Performance**

*Based on 5 simulations with 7 advanced functions*

**🥇 Best Bayesian Methods:**

- `fit_binding_bayes` & `fit_binding_bayes_perlabel` - **Best parameter recovery** (7.69), Fast (0.045s)
- `fit_binding_odr` - Good balance of speed and accuracy (0.044s)

**⚠️ Problematic Functions:**

- `fit_binding_pymc` - Very slow (75.6s), poor parameter recovery (296.7)
- `fit_binding_pymc2` - Extremely slow (43.6s), worst parameter recovery (692.6)
- `fit_binding_odr_recursive_outlier` - Catastrophic parameter recovery (60,646!)

______________________________________________________________________

## 🔍 **Duplicate Functions Identified**

**Perfect Duplicates (0.00e+00 difference):**

1. `fit_binding_lm` ≈ `fit_binding_lm_outlier` ≈ `fit_lm` ≈ `fit_lm(outlier)`
1. `fit_binding(lm)` ≈ `fit_binding_lm` ≈ `fit_binding_glob`
1. Multiple API shim functions (`api_recursive`, `api_recursive_outlier`) ≈ core functions

**Consolidation Opportunity:** ~10 function pairs are essentially identical

______________________________________________________________________

## 🎯 **Recommendations by Use Case**

### **For General Users (Recommended)**

```python
# Use the unified API - these are your best options
fit_binding(dataset, method="lm")                    # Fastest, reliable
fit_binding(dataset, method="lm_outlier", key="sample")  # With outlier handling
```

### **For Maximum Accuracy**

```python
# When parameter precision is critical
fit_lm(dataset, robust=True)                        # Best parameter recovery
core_reweighted(dataset, key="sample")               # Second best accuracy
```

### **For Speed-Critical Applications**

```python
# When speed matters most
fit_binding(dataset, method="lm")                    # 0.039s
fit_lm(dataset)                                      # 0.039s
```

### **For Bayesian Analysis**

```python
# After getting deterministic fit first
fr_det = fit_binding(dataset, method="lm")          # Fast initial fit
fr_bayes = fit_binding_bayes(fr_det)                 # Best Bayesian method
```

______________________________________________________________________

## 🧹 **Cleanup Recommendations**

### **🗑️ Functions to Remove**

- `api_reweighted` - **0% success rate**
- `fit_binding_pymc` & `fit_binding_pymc2` - Extremely slow, poor performance
- `fit_binding_odr_recursive_outlier` - Catastrophic parameter recovery

### **🔗 Functions to Consolidate**

- Keep `fit_binding()` unified dispatcher
- Remove duplicate API shims (`api_recursive`, `api_recursive_outlier`)
- Consolidate identical functions (keep one representative from each group)

### **📚 Functions to Deprecate**

Add deprecation warnings to:

- All `api_*` shim functions → direct users to `fit_binding()`
- Direct PyMC functions → direct users to `fit_binding_bayes()`

______________________________________________________________________

## 📋 **Implementation Plan**

### **Phase 1: Documentation Updates**

- Update docs to recommend `fit_binding()` as primary interface
- Document use cases for `fit_lm(robust=True)` for accuracy-critical applications
- Add performance guidance tables

### **Phase 2: Add Deprecation Warnings**

```python
# Example deprecation pattern
@deprecated("Use fit_binding(method='lm_outlier') instead")
def fit_binding_glob_reweighted(...):
    warnings.warn("fit_binding_glob_reweighted is deprecated...", DeprecationWarning)
    return fit_binding(dataset, method="lm_outlier", ...)
```

### **Phase 3: Code Consolidation**

- Remove duplicate function implementations
- Keep only the best performer from each duplicate group
- Maintain backward compatibility through wrappers

### **Phase 4: Remove Deprecated Functions** (Next major version)

- Remove failed functions (`api_reweighted`)
- Remove poorly performing functions (PyMC direct calls)
- Remove deprecated shims

______________________________________________________________________

## 📈 **Performance Summary Table**

| Function Category   | Best Choice                        | Speed  | Accuracy | Outlier Robust | Use Case                   |
| ------------------- | ---------------------------------- | ------ | -------- | -------------- | -------------------------- |
| **General Purpose** | `fit_binding(method="lm")`         | ⚡⚡⚡ | ⭐⭐     | ⭐⭐⭐         | Most users                 |
| **High Accuracy**   | `fit_lm(robust=True)`              | ⚡     | ⭐⭐⭐   | ⭐⭐⭐         | Research                   |
| **With Outliers**   | `fit_binding(method="lm_outlier")` | ⚡⚡   | ⭐⭐     | ⭐⭐⭐         | Noisy data                 |
| **Bayesian**        | `fit_binding_bayes()`              | ⚡⚡   | ⭐⭐⭐   | ⭐⭐           | Uncertainty quantification |
| **Speed Critical**  | `fit_binding(method="lm")`         | ⚡⚡⚡ | ⭐⭐     | ⭐⭐⭐         | Automation                 |

**Legend:** ⚡ = Speed, ⭐ = Quality (more symbols = better)

______________________________________________________________________

## 🔬 **Technical Notes**

- **Dataset**: 2 DataArrays, 12 pH points, 10x error difference, 80% outlier probability
- **Simulations**: 10 deterministic + 5 Bayesian (limited by PyMC computational cost)
- **Metrics**: Success rate, parameter recovery error, execution time, outlier robustness
- **Stability Score**: Weighted combination of success rate (40%) + accuracy (40%) + outlier robustness (20%)

The unified `fit_binding()` API emerged as the clear winner for general use, providing excellent performance with a clean, consistent interface. For specialized needs, `fit_lm(robust=True)` offers superior parameter recovery at the cost of speed.

This analysis provides a solid foundation for consolidating your fitting function landscape and guiding users toward the most appropriate methods for their specific needs.
