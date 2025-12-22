# Prtecan Multi-Well MCMC Properties - Design

## Property Hierarchy

### Primary Properties (Do the Work)

#### `result_multi_trace`

- **Purpose:** Runs `fit_binding_pymc_multi()` (homoscedastic noise model)
- **Returns:** `(fit_results_dict, trace_df)`
  - `fit_results_dict`: `dict[str, FitResult[az.InferenceData]]` - Per-well results with residuals
  - `trace_df`: `pd.DataFrame` - Summary statistics from arviz
- **Side effects:** Saves `list_x_true.csv` with pH corrections
- **When to use:** Direct access to multi-well MCMC results

#### `result_multi_trace2`

- **Purpose:** Runs `fit_binding_pymc_multi2()` (heteroscedastic noise model)
- **Returns:** `(fit_results_dict, trace_df)` - same structure as above
- **When to use:** When you need the advanced noise model

### Wrapper Property (For Export Compatibility)

#### `result_multi_mcmc`

- **Purpose:** Wraps `result_multi_trace` results in a `TitrationResults` container
- **Returns:** `TitrationResults` with pre-populated results
- **Why it exists:** Export functions expect `TitrationResults` interface
- **Implementation:**
  ```python
  fit_results_dict, _trace_df = self.result_multi_trace
  return TitrationResults(..., results=fit_results_dict)
  ```

## Relationship Diagram

```
┌─────────────────────────┐
│  result_multi_trace     │  ← Does MCMC fitting
│  (fit_results, trace_df)│
└────────┬────────────────┘
         │ depends on
         ↓
┌─────────────────────────┐
│  result_multi_mcmc      │  ← Wraps for export
│  (TitrationResults)     │
└─────────────────────────┘
```

## Why This Design?

### Historical Context

Originally, PyMC multi functions returned just `az.InferenceData` (the trace). This made it:

- ❌ Hard to get residuals
- ❌ Inconsistent with single-well PyMC
- ❌ Required manual `extract_fit()` calls

### Current Design

Now PyMC multi functions return `dict[str, FitResult[az.InferenceData]]`:

- ✅ Easy residual access: `fit_results[well].result.residual`
- ✅ Consistent API with single-well
- ✅ Perfect for method comparison

### Why Keep `result_multi_mcmc`?

The export system expects `TitrationResults`:

```python
# In _export_fit()
if self.params.mcmc == "multi":
    export_list.append(self.result_multi_mcmc)  # Needs TitrationResults
```

So `result_multi_mcmc` provides backward compatibility while `result_multi_trace` gives direct access to the new improved structure.

## Usage Guide

### For Analysis (Recommended)

Use `result_multi_trace` directly:

```python
fit_results, trace_df = tit.result_multi_trace

# Easy residual extraction
from clophfit.fitting.residuals import collect_multi_residuals
all_res = collect_multi_residuals(fit_results)

# Access individual wells
well_result = fit_results['A01']
residuals = well_result.result.residual
K_value = well_result.result.params['K'].value
```

### For Export (Automatic)

The export system uses `result_multi_mcmc` automatically:

```python
# This happens internally
tit.result_multi_mcmc.export_pngs(output_dir)
tit.result_multi_mcmc.dataframe.to_csv(...)
```

### For X-true Values

Either property works:

```python
# From trace
fit_results, trace_df = tit.result_multi_trace
x_true = x_true_from_trace_df(trace_df)

# Or CSV file (side effect of calling either property)
x_true_df = pd.read_csv("list_x_true.csv", header=None)
```

## When to Use Each Property

| Property              | When to Use                   | What You Get                     |
| --------------------- | ----------------------------- | -------------------------------- |
| `result_multi_trace`  | Analysis, residual extraction | Direct dict access               |
| `result_multi_trace2` | Advanced noise model          | Dict with heteroscedastic errors |
| `result_multi_mcmc`   | Export (automatic)            | TitrationResults wrapper         |

## Simplification Possibility

In the future, could simplify to:

```python
@cached_property
def result_multi(self) -> dict[str, FitResult[az.InferenceData]]:
    """Multi-well MCMC results."""
    ...
    return fit_results

@property
def result_multi_df(self) -> pd.DataFrame:
    """DataFrame view of multi-well results."""
    return self._results_to_dataframe(self.result_multi)
```

But current design maintains backward compatibility while providing the improved interface.

## Summary

- **Not truly redundant** - `result_multi_mcmc` depends on `result_multi_trace`
- **Clear separation** - `trace` does work, `mcmc` wraps for export
- **Both useful** - Direct access vs. compatibility
- **Well documented** - Now clear what each does

The design is actually clever: expose the dict directly for analysis, but keep the wrapped version for backward compatibility with export code.
