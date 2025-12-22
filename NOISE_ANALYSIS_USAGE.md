# Noise Model Functions - Usage Guide

## Quick Start

```python
# In your notebook (e.g., 01_noise_characterization.ipynb)
from clophfit.fitting.residuals import collect_multi_residuals
from dev.noise_models import (
    analyze_label_bias,
    detect_adjacent_correlation,
    estimate_x_shift_statistics,
    compute_residual_covariance,
)

# Step 1: Get fit results and collect residuals
fit_results = fit_binding_pymc_multi(tit.result_global.results, tit.scheme)
all_res = collect_multi_residuals(fit_results)

# Step 2: Analyze your 3 key issues
bias_summary, label_bias = analyze_label_bias(all_res, n_bins=3)
corr_stats, corr_by_label = detect_adjacent_correlation(all_res)
shift_stats = estimate_x_shift_statistics(all_res, fit_results)
```

______________________________________________________________________

## Function 1: `analyze_label_bias()`

**Purpose:** Detect if certain labels (e.g., y1) have systematic bias at specific pH ranges.

### Your Issue 1

> "y1 label at lowest pH is always negative, sometimes >3σ"

### Usage

```python
bias_summary, label_bias = analyze_label_bias(all_res, n_bins=3)
```

### Outputs

#### 1. `label_bias` - Overall statistics by label

```python
print(label_bias)
```

| label | mean_resid | std_resid | median_resid | outlier_rate | negative_bias_frac |
| ----- | ---------- | --------- | ------------ | ------------ | ------------------ |
| y1    | -0.025     | 1.234     | -0.018       | 0.05         | 0.45               |
| y2    | 0.012      | 0.987     | 0.003        | 0.02         | 0.25               |

**What to look for:**

- `mean_resid` ≠ 0 → Systematic bias (should be ~0 if model is correct)
- `outlier_rate` > 0.05 → More than 5% outliers (beyond ±2σ)
- `negative_bias_frac` > 0.5 → More than half are negative (Issue 1!)

#### 2. `bias_summary` - Statistics by label AND pH bin

```python
print(bias_summary)
```

| label | x_bin       | mean_resid | std_resid | count | outlier_rate | mean_std_res |
| ----- | ----------- | ---------- | --------- | ----- | ------------ | ------------ |
| y1    | (5.0, 6.0\] | **-1.234** | 0.456     | 120   | **0.15**     | **-2.1**     |
| y1    | (6.0, 7.0\] | -0.234     | 0.389     | 135   | 0.04         | -0.5         |
| y1    | (7.0, 8.0\] | 0.123      | 0.412     | 128   | 0.02         | 0.3          |
| y2    | (5.0, 6.0\] | 0.045      | 0.298     | 118   | 0.01         | 0.1          |
| ...   | ...         | ...        | ...       | ...   | ...          | ...          |

**What to look for:**

- **pH-dependent bias:** Does `mean_resid` change across pH bins?
- **Issue 1 confirmation:** Is y1 at low pH (5.0-6.0) systematically negative?
- **High outlier rate** in specific pH ranges → Model issue in that region

### Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Plot 1: Mean residual by pH bin
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for label in ['y1', 'y2']:
    df = bias_summary.loc[label]
    axes[0].plot(range(len(df)), df['mean_resid'], 'o-', label=label)

axes[0].axhline(0, ls='--', c='k', alpha=0.3)
axes[0].set_xlabel('pH bin')
axes[0].set_ylabel('Mean residual')
axes[0].legend()
axes[0].set_title('Systematic Bias by pH Range')

# Plot 2: Outlier rate
for label in ['y1', 'y2']:
    df = bias_summary.loc[label]
    axes[1].plot(range(len(df)), df['outlier_rate'], 'o-', label=label)

axes[1].axhline(0.05, ls='--', c='r', alpha=0.3, label='5% threshold')
axes[1].set_xlabel('pH bin')
axes[1].set_ylabel('Outlier rate')
axes[1].legend()
axes[1].set_title('Outlier Rate by pH Range')
plt.tight_layout()
```

### Interpretation

```python
# Check Issue 1: y1 at low pH always negative
low_ph_y1 = bias_summary.loc[('y1', slice(None)), :].iloc[0]  # First (lowest) pH bin

if low_ph_y1['mean_resid'] < -0.5 and low_ph_y1['negative_bias_frac'] > 0.6:
    print("⚠️  Issue 1 CONFIRMED:")
    print(f"   y1 at low pH has mean residual = {low_ph_y1['mean_resid']:.2f}")
    print(f"   {low_ph_y1['negative_bias_frac']*100:.1f}% are negative")
    print(f"   Outlier rate: {low_ph_y1['outlier_rate']*100:.1f}%")
else:
    print("✓ No systematic bias detected at low pH")
```

______________________________________________________________________

## Function 2: `detect_adjacent_correlation()`

**Purpose:** Test if adjacent residuals alternate positive/negative (suggests pH error or model issue).

### Your Issue 2

> "Some residue are often positive and the next closeby negative"

### Usage

```python
corr_stats, corr_by_label = detect_adjacent_correlation(all_res)
```

### Outputs

#### 1. `corr_stats` - Per-well correlation statistics

```python
print(corr_stats.head(10))
```

| label | well | lag1_corr | n_points |
| ----- | ---- | --------- | -------- |
| y1    | A01  | **-0.78** | 7        |
| y1    | A02  | **-0.65** | 7        |
| y1    | A03  | -0.23     | 7        |
| y2    | A01  | **-0.82** | 7        |
| y2    | A02  | -0.15     | 7        |

**What to look for:**

- `lag1_corr` < 0 → Negative correlation (alternating pattern!)
- `lag1_corr` < -0.5 → Strong alternating pattern (Issue 2!)
- Pattern across many wells → Systematic problem

#### 2. `corr_by_label` - Distribution by label

```python
for label, corrs in corr_by_label.items():
    print(f"{label}:")
    print(f"  Mean correlation: {corrs.mean():.3f}")
    print(f"  Median correlation: {np.median(corrs):.3f}")
    print(f"  Fraction < -0.5: {(corrs < -0.5).mean():.1%}")
```

### Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Plot 1: Distribution of correlations
for label, corrs in corr_by_label.items():
    axes[0].hist(corrs, bins=20, alpha=0.5, label=label)
    axes[0].axvline(corrs.mean(), ls='--', label=f'{label} mean')

axes[0].axvline(0, ls='-', c='k', alpha=0.3)
axes[0].set_xlabel('Lag-1 correlation')
axes[0].set_ylabel('Count')
axes[0].legend()
axes[0].set_title('Adjacent Residual Correlation Distribution')

# Plot 2: Per-well heatmap (top 20 wells)
top_wells = corr_stats.nsmallest(20, 'lag1_corr')
sns.barplot(data=top_wells, y='well', x='lag1_corr', hue='label', ax=axes[1])
axes[1].axvline(0, ls='--', c='k', alpha=0.3)
axes[1].axvline(-0.5, ls='--', c='r', alpha=0.3, label='Strong correlation')
axes[1].set_xlabel('Lag-1 correlation')
axes[1].set_title('Wells with Strongest Alternating Pattern')
plt.tight_layout()
```

### Interpretation

```python
# Check Issue 2: Adjacent correlation
for label, corrs in corr_by_label.items():
    frac_negative = (corrs < 0).mean()
    frac_strong = (corrs < -0.5).mean()

    print(f"\n{label}:")
    print(f"  {frac_negative*100:.1f}% of wells have negative correlation")
    print(f"  {frac_strong*100:.1f}% have STRONG alternating pattern (< -0.5)")

    if frac_strong > 0.3:
        print(f"  ⚠️  Issue 2 CONFIRMED for {label}")
        print(f"     Suggests: pH values may be systematically wrong")
        print(f"              OR model doesn't capture curve shape")
    else:
        print(f"  ✓ No strong alternating pattern")
```

### Example: Plot residuals showing alternation

```python
# Find wells with strongest alternating pattern
worst_wells = corr_stats.nsmallest(5, 'lag1_corr')

fig, axes = plt.subplots(len(worst_wells), 1, figsize=(10, 10))

for i, (_, row) in enumerate(worst_wells.iterrows()):
    well = row['well']
    label = row['label']

    well_res = all_res[(all_res['well'] == well) & (all_res['label'] == label)].sort_values('x')

    axes[i].plot(well_res['x'], well_res['resid_weighted'], 'o-')
    axes[i].axhline(0, ls='--', c='k', alpha=0.3)
    axes[i].set_title(f"{well} - {label} (corr={row['lag1_corr']:.2f})")
    axes[i].set_ylabel('Residual')

axes[-1].set_xlabel('pH')
plt.tight_layout()
```

______________________________________________________________________

## Function 3: `estimate_x_shift_statistics()`

**Purpose:** Detect if pH values might be systematically wrong (per-well or plate-wide).

### Your Issue 3

> "Maybe the x-pH-value is wrong either in that titration or in the whole plate"

### Usage

```python
shift_stats = estimate_x_shift_statistics(all_res, fit_results)
```

### Outputs

```python
print(shift_stats.head(10))
```

| label | well | residual_slope | residual_intercept | trend_strength | asymmetry | n_points |
| ----- | ---- | -------------- | ------------------ | -------------- | --------- | -------- |
| y1    | A01  | **0.45**       | -2.1               | **1.8**        | 0.23      | 7        |
| y1    | A02  | **0.38**       | -1.5               | **1.5**        | 0.18      | 7        |
| y1    | A03  | 0.05           | -0.2               | 0.2            | -0.05     | 7        |
| y2    | A01  | **0.52**       | -2.5               | **2.1**        | 0.31      | 7        |

**What each column means:**

- `residual_slope`: Linear trend in residuals vs pH
  - Positive → residuals increase with pH
  - Suggests pH values shifted left/right
- `trend_strength`: Magnitude of trend over pH range
  - `> 1.0` → Strong systematic trend (potential pH error!)
- `asymmetry`: Balance of positive vs negative residuals
  - `> 0.3` → More positive than negative
  - `< -0.3` → More negative than positive

### Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Distribution of trend strengths
for label in ['y1', 'y2']:
    df = shift_stats[shift_stats['label'] == label]
    axes[0, 0].hist(df['trend_strength'], bins=20, alpha=0.5, label=label)

axes[0, 0].axvline(1.0, ls='--', c='r', label='Threshold')
axes[0, 0].set_xlabel('Trend strength')
axes[0, 0].set_ylabel('Count')
axes[0, 0].legend()
axes[0, 0].set_title('Distribution of Residual Trends')

# Plot 2: Scatter - slope vs asymmetry
for label in ['y1', 'y2']:
    df = shift_stats[shift_stats['label'] == label]
    axes[0, 1].scatter(df['residual_slope'], df['asymmetry'], alpha=0.5, label=label)

axes[0, 1].axhline(0, ls='--', c='k', alpha=0.3)
axes[0, 1].axvline(0, ls='--', c='k', alpha=0.3)
axes[0, 1].set_xlabel('Residual slope')
axes[0, 1].set_ylabel('Asymmetry')
axes[0, 1].legend()
axes[0, 1].set_title('Trend vs Asymmetry')

# Plot 3: Top wells with strongest trends
top_wells = shift_stats.nlargest(15, 'trend_strength')
sns.barplot(data=top_wells, y='well', x='trend_strength', hue='label', ax=axes[1, 0])
axes[1, 0].axvline(1.0, ls='--', c='r')
axes[1, 0].set_title('Wells with Strongest pH Shift Evidence')

# Plot 4: Example residual vs pH for top suspect wells
suspect_wells = shift_stats[shift_stats['trend_strength'] > 1.5].head(3)
for _, row in suspect_wells.iterrows():
    well = row['well']
    label = row['label']
    df = all_res[(all_res['well'] == well) & (all_res['label'] == label)]
    axes[1, 1].plot(df['x'], df['resid_weighted'], 'o-', label=f"{well}-{label}")

    # Fit line
    z = np.polyfit(df['x'], df['resid_weighted'], 1)
    p = np.poly1d(z)
    axes[1, 1].plot(df['x'], p(df['x']), '--', alpha=0.5)

axes[1, 1].axhline(0, ls='--', c='k', alpha=0.3)
axes[1, 1].set_xlabel('pH')
axes[1, 1].set_ylabel('Residual')
axes[1, 1].legend()
axes[1, 1].set_title('Residuals vs pH (Suspect Wells)')
plt.tight_layout()
```

### Interpretation

```python
# Check Issue 3: Systematic pH shifts

# Test 1: How many wells show strong trends?
strong_trend = shift_stats[shift_stats['trend_strength'] > 1.0]
print(f"\nWells with strong residual trends (trend_strength > 1.0):")
print(f"  y1: {(strong_trend['label'] == 'y1').sum()}/{len(shift_stats[shift_stats['label'] == 'y1'])}")
print(f"  y2: {(strong_trend['label'] == 'y2').sum()}/{len(shift_stats[shift_stats['label'] == 'y2'])}")

# Test 2: Is there a plate-wide pattern?
mean_slope_y1 = shift_stats[shift_stats['label'] == 'y1']['residual_slope'].mean()
mean_slope_y2 = shift_stats[shift_stats['label'] == 'y2']['residual_slope'].mean()

print(f"\nMean residual slope:")
print(f"  y1: {mean_slope_y1:.3f}")
print(f"  y2: {mean_slope_y2:.3f}")

if abs(mean_slope_y1) > 0.3 or abs(mean_slope_y2) > 0.3:
    print("\n⚠️  Issue 3 CONFIRMED: Plate-wide pH shift suspected")
    print("   All wells show consistent residual trend")
    print("   → Check pH calibration for this plate")
else:
    # Test 3: Per-well shifts?
    if len(strong_trend) > len(shift_stats) * 0.2:  # >20% of wells
        print("\n⚠️  Issue 3 PARTIAL: Some wells have pH shift")
        print(f"   {len(strong_trend)} wells affected")
        print("   → Check individual titrations")
    else:
        print("\n✓ No systematic pH shift detected")
```

______________________________________________________________________

## Complete Analysis Workflow

```python
# In your notebook
from clophfit.fitting.residuals import collect_multi_residuals
from dev.noise_models import (
    analyze_label_bias,
    detect_adjacent_correlation,
    estimate_x_shift_statistics,
)
import matplotlib.pyplot as plt

# Step 1: Collect residuals
fit_results = fit_binding_pymc_multi(tit.result_global.results, tit.scheme)
all_res = collect_multi_residuals(fit_results)

print(f"Collected {len(all_res)} residual points from {all_res['well'].nunique()} wells")

# Step 2: Analyze Issue 1 - Systematic bias
print("\n" + "="*60)
print("ISSUE 1: Systematic Bias Analysis")
print("="*60)

bias_summary, label_bias = analyze_label_bias(all_res, n_bins=3)
print("\nOverall bias by label:")
print(label_bias)

# Plot bias by pH range
fig, ax = plt.subplots(figsize=(10, 5))
for label in ['y1', 'y2']:
    df = bias_summary.loc[label]
    ax.plot(range(len(df)), df['mean_resid'], 'o-', label=label, markersize=10)
ax.axhline(0, ls='--', c='k', alpha=0.3)
ax.set_xlabel('pH bin (low → high)')
ax.set_ylabel('Mean residual')
ax.set_title('Issue 1: Systematic Bias by pH Range')
ax.legend()
plt.show()

# Step 3: Analyze Issue 2 - Adjacent correlation
print("\n" + "="*60)
print("ISSUE 2: Adjacent Correlation Analysis")
print("="*60)

corr_stats, corr_by_label = detect_adjacent_correlation(all_res)

for label, corrs in corr_by_label.items():
    print(f"\n{label}:")
    print(f"  Mean lag-1 correlation: {corrs.mean():.3f}")
    print(f"  Fraction negative: {(corrs < 0).mean():.1%}")
    print(f"  Fraction strongly negative (<-0.5): {(corrs < -0.5).mean():.1%}")

# Plot correlation distribution
fig, ax = plt.subplots(figsize=(10, 5))
for label, corrs in corr_by_label.items():
    ax.hist(corrs, bins=30, alpha=0.5, label=label)
ax.axvline(0, ls='--', c='k', alpha=0.3, label='No correlation')
ax.axvline(-0.5, ls='--', c='r', alpha=0.3, label='Strong alternation')
ax.set_xlabel('Lag-1 correlation')
ax.set_ylabel('Number of wells')
ax.set_title('Issue 2: Distribution of Adjacent Residual Correlations')
ax.legend()
plt.show()

# Step 4: Analyze Issue 3 - pH shift
print("\n" + "="*60)
print("ISSUE 3: pH Shift Analysis")
print("="*60)

shift_stats = estimate_x_shift_statistics(all_res, fit_results)

for label in ['y1', 'y2']:
    df = shift_stats[shift_stats['label'] == label]
    strong = (df['trend_strength'] > 1.0).sum()
    print(f"\n{label}:")
    print(f"  Mean residual slope: {df['residual_slope'].mean():.3f}")
    print(f"  Wells with strong trend: {strong}/{len(df)} ({strong/len(df):.1%})")

# Plot trend distribution
fig, ax = plt.subplots(figsize=(10, 5))
for label in ['y1', 'y2']:
    df = shift_stats[shift_stats['label'] == label]
    ax.hist(df['trend_strength'], bins=30, alpha=0.5, label=label)
ax.axvline(1.0, ls='--', c='r', alpha=0.5, label='Threshold')
ax.set_xlabel('Trend strength')
ax.set_ylabel('Number of wells')
ax.set_title('Issue 3: Distribution of Residual Trends (pH shift evidence)')
ax.legend()
plt.show()

# Step 5: Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print("\nNext steps based on findings:")
print("1. If Issue 1 confirmed → Add label-dependent bias to synthetic data")
print("2. If Issue 2 confirmed → Consider GLS with correlated errors")
print("3. If Issue 3 confirmed → Investigate pH calibration")
```

______________________________________________________________________

## Exporting Results

```python
# Save results for reference
bias_summary.to_csv("dev/bias_summary.csv")
label_bias.to_csv("dev/label_bias.csv")
corr_stats.to_csv("dev/correlation_stats.csv")
shift_stats.to_csv("dev/shift_stats.csv")

# Use in synthetic data generation (notebook 02)
from dev.noise_models import export_noise_parameters, compute_residual_covariance

cov_by_label = compute_residual_covariance(all_res)
export_noise_parameters(cov_by_label, label_bias, corr_stats)
```
