# Noise Analysis Functions - Quick Reference

## Setup

```python
from clophfit.fitting.residuals import collect_multi_residuals
from dev.noise_models import (
    analyze_label_bias,
    detect_adjacent_correlation,
    estimate_x_shift_statistics,
)

# Get residuals
fit_results = fit_binding_pymc_multi(tit.result_global.results, tit.scheme)
all_res = collect_multi_residuals(fit_results)
```

______________________________________________________________________

## Function 1: `analyze_label_bias()` - Issue 1

**Question:** Is y1 at low pH systematically negative?

```python
bias_summary, label_bias = analyze_label_bias(all_res, n_bins=3)
```

### Quick Check

```python
print(label_bias)  # Overall by label
print(bias_summary.loc['y1'])  # By pH bin for y1
```

### Red Flags

- `mean_resid` ≠ 0 → Systematic bias
- `outlier_rate` > 0.05 → High outlier rate
- `negative_bias_frac` > 0.5 → Mostly negative

### Quick Plot

```python
for label in ['y1', 'y2']:
    df = bias_summary.loc[label]
    plt.plot(range(len(df)), df['mean_resid'], 'o-', label=label)
plt.axhline(0, ls='--', c='k')
plt.ylabel('Mean residual')
plt.xlabel('pH bin (low → high)')
plt.legend()
```

______________________________________________________________________

## Function 2: `detect_adjacent_correlation()` - Issue 2

**Question:** Do residuals alternate positive/negative?

```python
corr_stats, corr_by_label = detect_adjacent_correlation(all_res)
```

### Quick Check

```python
for label, corrs in corr_by_label.items():
    print(f"{label}: mean = {corrs.mean():.3f}, "
          f"fraction < -0.5 = {(corrs < -0.5).mean():.1%}")
```

### Red Flags

- Mean correlation < -0.3 → Alternating pattern
- Many wells < -0.5 → Strong alternation (pH error likely!)

### Quick Plot

```python
for label, corrs in corr_by_label.items():
    plt.hist(corrs, bins=30, alpha=0.5, label=label)
plt.axvline(0, ls='--', c='k', label='No correlation')
plt.axvline(-0.5, ls='--', c='r', label='Strong')
plt.xlabel('Lag-1 correlation')
plt.legend()
```

______________________________________________________________________

## Function 3: `estimate_x_shift_statistics()` - Issue 3

**Question:** Are pH values systematically wrong?

```python
shift_stats = estimate_x_shift_statistics(all_res, fit_results)
```

### Quick Check

```python
# Plate-wide shift?
for label in ['y1', 'y2']:
    df = shift_stats[shift_stats['label'] == label]
    print(f"{label}: mean slope = {df['residual_slope'].mean():.3f}, "
          f"strong trends = {(df['trend_strength'] > 1.0).sum()}/{len(df)}")
```

### Red Flags

- Mean slope > 0.3 → Plate-wide pH shift
- Many `trend_strength` > 1.0 → Individual well pH errors

### Quick Plot

```python
for label in ['y1', 'y2']:
    df = shift_stats[shift_stats['label'] == label]
    plt.hist(df['trend_strength'], bins=30, alpha=0.5, label=label)
plt.axvline(1.0, ls='--', c='r', label='Threshold')
plt.xlabel('Trend strength')
plt.legend()
```

______________________________________________________________________

## Complete Example

```python
# Analyze all 3 issues
bias_summary, label_bias = analyze_label_bias(all_res, n_bins=3)
corr_stats, corr_by_label = detect_adjacent_correlation(all_res)
shift_stats = estimate_x_shift_statistics(all_res, fit_results)

# Quick report
print("ISSUE 1: Systematic Bias")
print(label_bias[['mean_resid', 'outlier_rate', 'negative_bias_frac']])

print("\nISSUE 2: Adjacent Correlation")
for label, corrs in corr_by_label.items():
    print(f"{label}: {corrs.mean():.3f} (frac<-0.5: {(corrs<-0.5).mean():.1%})")

print("\nISSUE 3: pH Shifts")
for label in ['y1', 'y2']:
    df = shift_stats[shift_stats['label'] == label]
    print(f"{label}: slope={df['residual_slope'].mean():.3f}, "
          f"strong={100*(df['trend_strength']>1).mean():.0f}%")
```

______________________________________________________________________

## Interpretation Guide

| Finding                              | Likely Cause                   | Action                       |
| ------------------------------------ | ------------------------------ | ---------------------------- |
| y1 negative at low pH                | Systematic bias                | Add to synthetic data        |
| High adjacent correlation (negative) | pH values wrong OR model issue | Try GLS or check calibration |
| Plate-wide slope                     | pH calibration error           | Check all pH values          |
| Some wells have strong trend         | Individual titration errors    | Flag those wells             |
| High outlier rate                    | Wrong error model              | Use heteroscedastic errors   |

______________________________________________________________________

## Save Results

```python
bias_summary.to_csv("dev/bias_summary.csv")
corr_stats.to_csv("dev/correlation_stats.csv")
shift_stats.to_csv("dev/shift_stats.csv")

# Use in notebook 02
from dev.noise_models import export_noise_parameters, compute_residual_covariance
cov_by_label = compute_residual_covariance(all_res)
export_noise_parameters(cov_by_label, label_bias, corr_stats)
```
