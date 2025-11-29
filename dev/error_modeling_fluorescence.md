# Error Modeling in Multi-Channel Fluorescence pH Titration Analysis

## Overview

This document describes the statistical framework for error propagation and uncertainty quantification in dual-channel fluorescence pH titration experiments, as implemented in the ClopHfit software package. The methodology addresses a fundamental challenge in high-throughput biosensor screening: accurate estimation of measurement uncertainties when fitting sigmoidal binding curves to fluorescence intensity data.

## The Measurement Problem

### Experimental Setup

pH titration experiments measure fluorescence intensity across two spectral channels (y₁, y₂) at multiple pH values. Each measurement comprises:

1. **Dependent variables**: Fluorescence intensities in channels 1 and 2
1. **Independent variable**: pH (measured with finite precision)
1. **Buffer baseline**: Reference measurements from buffer-only wells at each pH

The experimental design typically includes 6 buffer wells per pH point, providing replicate measurements for baseline estimation and error characterization.

### Sources of Uncertainty

Fluorescence measurements exhibit several distinct error sources:

1. **Shot noise (Poisson statistics)**: Photon counting uncertainty proportional to √I, where I is the measured intensity
1. **Instrument noise**: pH meter precision, pipetting variability, temperature fluctuations
1. **Systematic drift**: Photobleaching, evaporation, plate reader thermal effects
1. **Buffer subtraction uncertainty**: Propagated error from baseline correction

## The Error Model

### Initial Hypothesis: Buffer Standard Deviation

The initial approach assumed buffer measurements are pH-independent, treating the standard deviation of 6 replicate buffer wells as a proxy for measurement error:

```
σ_buffer = SD(buffer_wells)
```

This approach systematically underestimates true measurement uncertainty because:

- Buffer wells may exhibit lower intrinsic variability than protein-containing samples
- Shot noise scales with signal intensity, which varies across the titration curve
- Buffer measurements capture only one component of the total error budget

### Improved Error Model: Combined Shot and Buffer Noise

The implemented error model combines Poisson (shot) noise with buffer-derived uncertainty:

```python
signal = max(1.0, α × y^β)  # Parameterized intensity-dependent term
y_err = √(signal + σ_buffer²)
```

This formulation:

- Ensures error scales with signal intensity via shot noise
- Incorporates buffer uncertainty as an additive variance component
- Prevents numerical instability for low/negative signals via flooring

### Bayesian Error Scaling

A key observation during model development: when using pre-specified errors from the buffer/shot model, the PyMC sampler converged to nearly identical scaling factors (ye_mag) for both fluorescence channels. This suggests:

1. The underlying error structure is consistent across channels when properly parameterized
1. A single shared scaling factor may suffice for well-calibrated error estimates
1. Channel-specific scaling factors can detect systematic miscalibration

The `fit_binding_pymc` function implements a shared error scaling model:

```python
ye_mag = pm.HalfNormal("ye_mag", sigma=ye_scaling)
for lbl, da in ds.items():
    pm.Normal(f"y_likelihood_{lbl}", mu=y_model, sigma=ye_mag * da.y_err, observed=da.y)
```

Whereas `fit_binding_pymc2` allows per-channel scaling:

```python
ye_mag[lbl] = pm.HalfNormal(f"ye_mag_{lbl}", sigma=10.0)
```

## Buffer Processing and Background Subtraction

### Buffer Fitting Strategies

The `Buffer` class implements three methods for background estimation:

1. **`fit`**: Linear ODR (Orthogonal Distance Regression) fit to buffer values vs. pH, accounting for pH measurement uncertainty
1. **`mean`**: Simple arithmetic mean of buffer wells at each pH
1. **`meansd`**: Mean with median-normalized standard error (robust to outliers)

The linear fit approach is preferred when buffer exhibits pH dependence (e.g., autofluorescence from pH-sensitive components), while mean-based methods suffice for truly pH-independent backgrounds.

### Error Propagation from Buffer Subtraction

Buffer-subtracted intensity y' and its uncertainty propagate as:

```
y' = y_raw - y_buffer
σ_y' = √(σ_raw² + σ_buffer²)
```

When buffer errors are provided, the DataArray construction includes this propagation:

```python
if self.bg_err:
    y_errc = np.sqrt(signal + self.bg_err[label] ** 2)
else:
    y_errc = np.sqrt(signal)
```

## Weighting and Outlier Detection

### The Current Pipeline: A Hybrid Approach

The actual implementation uses a **two-stage hybrid** approach:

**Stage 1: Initial DataArray creation** (`_create_data_array`)

```python
signal = np.maximum(1.0, alpha * y**beta)  # alpha=1, beta=1 currently
if self.bg_err:
    y_errc = np.sqrt(signal + self.bg_err[label] ** 2)  # Physics-informed
else:
    y_errc = np.sqrt(signal)  # Shot noise only
```

**Stage 2: Global fitting** (`_compute_global_fit`)

```python
return outlier2(ds, key)  # Overwrites y_err with residual-based estimates
```

**Stage 3: Bayesian refinement** (`fit_binding_pymc`)

```python
# Uses y_err from outlier2 (residual-based), NOT the original √y + σ_buffer
sigma=ye_mag * da.y_err  # da.y_err comes from outlier2's reweighting
```

This means the physics-informed prior (`√y + σ_buffer²`) is computed but then **overwritten** by `outlier2` before reaching PyMC. The Bayesian stage inherits residual-based errors, not the mechanistic model.

### Philosophical Comparison of Approaches

| Aspect         | A Priori (`√y + σ_buffer`)                   | Residual-Based (`mean\|residual\|`)           |
| -------------- | -------------------------------------------- | --------------------------------------------- |
| **Philosophy** | Physics-informed, mechanistic                | Empirical, data-driven                        |
| **Bayesian?**  | More Bayesian (incorporates prior knowledge) | More frequentist (learns from data)           |
| **Robustness** | Robust to model misspecification             | Sensitive to model correctness                |
| **Risk**       | Wrong if physics model incorrect             | Circular: conflates measurement + model error |

#### A Priori Model: More Bayesian

The `√y + σ_buffer²` approach is more Bayesian because it incorporates **prior knowledge** about measurement physics:

- Shot noise follows Poisson statistics → variance ∝ intensity
- Buffer SD represents genuine prior information about instrument noise
- Errors are specified *before* seeing how well the model fits

#### Residual-Based: More Robust to Model Misspecification

If the binding model is slightly wrong (e.g., non-ideal cooperativity, multiple binding sites), residuals capture the **actual scatter** including model inadequacy. The physics-informed prior would underestimate uncertainty in this case.

However, residual-based errors have a **circular reasoning risk**: you can't distinguish measurement error from model error. If the model is systematically wrong, you're learning the wrong thing.

#### The Ideal Bayesian Approach

The theoretically correct approach would:

1. Specify physics-informed errors as **priors**
1. Learn a scaling factor to calibrate them
1. Propagate all uncertainty through the posterior

```python
# Ideal (not current) implementation:
y_err_prior = √(y + σ_buffer²)  # physics-informed structure
ye_mag = pm.HalfNormal("ye_mag", sigma=1.0)  # learn calibration
likelihood: σ = ye_mag × y_err_prior  # best of both worlds
```

Currently, the pipeline uses `outlier2`'s residual-based errors as the "prior" for PyMC, which is philosophically inconsistent but pragmatically effective for outlier removal.

### Error Estimation Strategies

Three distinct approaches for estimating y-errors have been explored:

#### 1. A Priori Error Model (Buffer + Shot Noise)

As described above, combining buffer SD with shot noise statistics:

```python
y_err = √(signal + σ_buffer²)
```

#### 2. Residual-Based Reweighting (Current Implementation in `outlier2`)

The current implementation estimates errors empirically from fit residuals. This is a form of **Iteratively Reweighted Least Squares (IRLS)**, though simplified:

```python
# Step 1: Initial robust fit using Huber loss
fr = fit_binding_glob(ds, robust=True)  # loss="huber"

# Step 2: Compute weighted residuals and convert back to raw residuals
weighted_residuals = fr.result.residual  # = (y - model) / y_err
weights = 1.0 / da.y_err
residuals = weighted_residuals / weights  # = (y - model)

# Step 3: Estimate per-channel error as mean absolute residual
for da in reweighted_ds.values():
    residual = |weighted_residual| * da.y_err  # back to raw scale
    sigma = mean(|residual|)  # single value per channel
    da.y_errc = sigma * np.ones_like(da.xc)  # uniform error for channel
```

The key insight: instead of using pre-specified errors, we **learn** the error magnitude from how well the model fits the data. Each channel receives a single uniform error estimate (the mean absolute residual), providing a robust measure of typical scatter.

#### 3. Point-wise IRLS (Not Currently Implemented)

A more sophisticated approach would estimate individual point errors:

```python
# Theoretical IRLS approach
for iteration in range(max_iter):
    residuals = y - model(params)
    weights = 1 / (|residuals| + ε)  # or Huber weights
    params = weighted_least_squares(weights)
```

This was considered but not implemented because:

- Single-point error estimates are noisy with limited replicates
- The channel-mean approach provides sufficient robustness for typical datasets
- Huber loss already down-weights outliers during optimization

### Robust Fitting with `outlier2`

The recommended fitting function `outlier2` implements a three-stage procedure:

**Stage 1: Robust fit with Huber loss**

```python
fr = fit_binding_glob(ds, robust=True)
# Uses scipy's least_squares with loss="huber"
# Huber loss: quadratic for small residuals, linear for large ones
```

**Stage 2: Residual-based reweighting**

- Compute raw residuals from weighted fit
- Estimate per-channel σ as mean absolute residual
- Assign uniform errors to each channel

**Stage 3: Outlier detection and final fit**

```python
z_scores = stats.zscore(residuals)
mask = |z_scores| < threshold  # default threshold = 3.0
ds.apply_mask(mask)  # remove outliers
return fit_binding_glob(ds, robust=False)  # final standard LS fit
```

This approach addresses:

- Heteroscedasticity: per-channel error estimation
- Outliers: z-score detection after error normalization
- Model robustness: Huber loss in initial fit
- Final precision: standard least-squares on clean data

### Multi-Dataset Weighting (`weight_da` and `weight_multi_ds_titration`)

For initial weighting when no buffer errors are available, the code performs preliminary single-channel fits:

```python
def weight_da(da: DataArray, is_ph: bool) -> bool:
    """Estimate y_err from fit residual SEM."""
    mr = lmfit.minimize(residuals, params, args=(ds,))
    sem = np.std(mr.residual, ddof=1) / np.sqrt(len(mr.residual))
    da.y_err = sem  # uniform error from residual scatter
```

For multi-channel global fits:

```python
def weight_multi_ds_titration(ds: Dataset):
    for lbl, da in ds.items():
        weight_da(da, ds.is_ph)  # fit each channel independently
    # Failed fits get 10× the max error (effectively down-weighted)
```

This ensures channels with different absolute intensities contribute appropriately to the global objective function based on their actual fit quality.

## Empirical Validation

### Simulation Study Design

A Monte Carlo simulation study was conducted to compare error estimation approaches using synthetic pH titration data with known ground truth (K=7.0). Each simulation generated dual-channel fluorescence data with:

- 12 pH points spanning 5.0–9.5
- Physics-informed noise: σ = √(y + buffer_sd²) with buffer_sd=20
- Two channels with different signal magnitudes (S0=1000/500, S1=100/50)

### Results: Synthetic Data (n=30 simulations)

**Clean Data (No Outliers):**

| Method              | K̄     | σ_K   | Mean Bias | 95% CI Coverage |
| ------------------- | ----- | ----- | --------- | --------------- |
| Physics-informed    | 7.011 | 0.046 | +0.011    | **96.7%**       |
| Residual (outlier2) | 7.009 | 0.048 | +0.009    | 93.3%           |
| weight_da (SEM)     | 7.011 | 0.048 | +0.011    | 86.7%           |

**With Outliers:**

| Method              | K̄     | σ_K   | Mean Bias | 95% CI Coverage |
| ------------------- | ----- | ----- | --------- | --------------- |
| Physics-informed    | 7.037 | 0.051 | +0.037    | 100%            |
| Residual (outlier2) | 7.033 | 0.056 | +0.033    | **96.7%**       |
| weight_da (SEM)     | 7.045 | 0.055 | +0.045    | 96.7%           |

### Results: Real Control Wells (31 wells with known pKa across 3 datasets)

Comprehensive comparison on real data from Tecan plate reader experiments (L2, L4, 140220 datasets) with all fitting methods:

| Method              | n   | K̄     | K_err Mean | Mean Bias | RMSE  | 95% CI Coverage |
| ------------------- | --- | ----- | ---------- | --------- | ----- | --------------- |
| **lm_robust**       | 31  | 7.250 | 0.171      | +0.056    | 0.139 | **87.1%**       |
| **odr_physics**     | 31  | 7.216 | 0.090      | +0.022    | 0.120 | **83.9%**       |
| lm_physics          | 31  | 7.278 | 0.119      | +0.085    | 0.171 | 80.6%           |
| pymc_physics_shared | 31  | 7.282 | 0.097      | +0.089    | 0.178 | 77.4%           |
| pymc_physics_sep    | 31  | 7.239 | 0.044      | +0.046    | 0.145 | 41.9%           |
| outlier2_uniform    | 31  | 7.214 | 0.041      | +0.021    | 0.135 | 38.7%           |
| pymc_uniform_shared | 31  | 7.211 | 0.041      | +0.018    | 0.142 | 38.7%           |
| pymc_uniform_sep    | 31  | 7.227 | 0.043      | +0.034    | 0.162 | 38.7%           |
| outlier2_shotnoise  | 31  | 7.217 | 0.039      | +0.024    | 0.123 | 35.5%           |
| lm_reweighted       | 31  | 7.255 | 0.030      | +0.062    | 0.146 | 16.1%           |

**Critical Finding**: Methods with best coverage use larger uncertainty estimates (K_err ~0.09-0.17), while residual-based methods (outlier2) produce smaller uncertainties (K_err ~0.04) that are **overconfident**.

**Heteroscedasticity Paradox**: `outlier2_uniform` produces weighted residuals closest to N(0,1) (Shapiro p=0.778), but has poor coverage (38.7%). This means the residuals are well-normalized but the **absolute scale of uncertainties is underestimated**.

**Critical Comparison: Physics vs Residual Errors in PyMC**

| Starting Errors          | ye_mag type | K_err | Bias   | Coverage |
| ------------------------ | ----------- | ----- | ------ | -------- |
| Physics (√y + σ_buffer²) | shared      | 0.097 | +0.089 | **77%**  |
| Physics (√y + σ_buffer²) | separate    | 0.044 | +0.046 | 42%      |
| Residual (outlier2)      | shared      | 0.041 | +0.018 | 39%      |
| Residual (outlier2)      | separate    | 0.043 | +0.034 | 39%      |

Physics-informed errors with shared ye_mag achieve best Bayesian coverage.

**Per-Sample Analysis (by dataset):**

*L1 Dataset:*

| Sample | True pKa | n   | Physics K±err (cov) | PyMC+phys K±err (cov) |
| ------ | -------- | --- | ------------------- | --------------------- |
| V224L  | 5.95     | 2   | 5.93±0.08 (100%)    | 5.91±0.10 (100%)      |
| E2GFP  | 6.80     | 4   | 6.76±0.15 (100%)    | 6.70±0.36 (100%)      |
| V224Q  | 7.95     | 1   | 7.80±0.08 (100%)    | 7.80±0.11 (100%)      |

*L2 Dataset:*

| Sample | True pKa | n   | Physics K±err (cov) | PyMC+phys K±err (cov) |
| ------ | -------- | --- | ------------------- | --------------------- |
| V224L  | 5.95     | 3   | 5.77±0.05 (33%)     | 5.76±0.06 (33%)       |
| E2GFP  | 6.80     | 4   | 6.83±0.12 (75%)     | 6.83±0.10 (75%)       |
| V224Q  | 7.95     | 2   | 8.04±0.17 (100%)    | 8.05±0.10 (100%)      |

*L4 Dataset:*

| Sample | True pKa | n   | Physics K±err (cov) | PyMC+phys K±err (cov) |
| ------ | -------- | --- | ------------------- | --------------------- |
| S202N  | 6.80     | 4   | 6.94±0.15 (100%)    | 6.95±0.13 (100%)      |
| E2GFP  | 6.80     | 3   | 6.94±0.10 (67%)     | 6.94±0.10 (67%)       |
| V224Q  | 7.95     | 3   | 8.01±0.12 (100%)    | 8.02±0.08 (100%)      |

*140220 Dataset:*

| Sample | True pKa | n   | Physics K±err (cov) | PyMC+phys K±err (cov) |
| ------ | -------- | --- | ------------------- | --------------------- |
| S202N  | 6.80     | 3   | 7.12±0.22 (100%)    | 7.14±0.19 (67%)       |
| G03    | 7.90     | 3   | 7.95±0.05 (67%)     | 7.95±0.05 (67%)       |
| V224Q  | 7.95     | 3   | 8.06±0.11 (67%)     | 8.07±0.08 (67%)       |
| NTT    | 7.55     | 3   | 7.65±0.10 (100%)    | 7.65±0.10 (100%)      |

(cov = proportion of wells where 95% CI contains true pKa)

**Per-Dataset Summary:**

| Dataset | n_wells | Physics Coverage | PyMC+physics Coverage |
| ------- | ------- | ---------------- | --------------------- |
| L1      | 7       | **100%**         | **100%**              |
| L2      | 9       | 67%              | 67%                   |
| L4      | 10      | **90%**          | **90%**               |
| 140220  | 12      | 83%              | 75%                   |

### Key Findings

1. **Robust fitting (lm_robust) achieves best coverage (87.1%)** - Huber loss provides natural outlier resistance while preserving reasonable uncertainty estimates.

1. **ODR with physics errors achieves 83.9% coverage** with lowest bias (+0.022) and good RMSE (0.120).

1. **Physics-informed errors outperform residual-based errors** for coverage, even though residual-based methods produce better-normalized residuals.

1. **Heteroscedasticity paradox**: `outlier2_uniform` produces weighted residuals closest to N(0,1) (Shapiro p=0.778) but has poor coverage (38.7%) because uncertainties are underestimated in absolute scale.

1. **Separate ye_mag per channel reduces coverage** to ~40% regardless of starting error model - overconfident.

1. **RMSE ~0.12-0.14 pH units** is achievable with proper error modeling.

### Recommendation

For production use:

1. **Use robust fitting (lm_robust)** - achieves 87.1% coverage with Huber loss

1. **For highest accuracy with good coverage, use ODR** - 83.9% coverage, lowest bias

1. **For Bayesian inference, use PyMC with physics errors and shared ye_mag** - achieves 77% coverage

1. **Avoid residual-based error estimation** (outlier2) for final uncertainty quantification - produces overconfident intervals

1. **Use shared ye_mag** (not separate per channel) to avoid overconfidence

1. **Always use --nrm flag** for datasets with "almost equal" label blocks

A more principled approach would preserve physics-informed structure through to PyMC:

```python
# Proposed improvement
y_err_physics = √(y + σ_buffer²)  # preserve structure
# In PyMC:
ye_mag = pm.HalfNormal("ye_mag", sigma=1.0)
sigma = ye_mag * y_err_physics  # learn calibration factor
```

## Implementation Architecture

### Data Flow

```
TecanFile → LabelBlockGroup → Buffer → Titration → FitResult
                ↓                          ↓
           Buffer.bg_err            DataArray.y_errc
                                           ↓
                                    fit_binding_glob
                                           ↓
                                     outlier2/ODR
                                           ↓
                                    fit_binding_pymc
```

### Key Classes

- **`Buffer`**: Manages buffer well data, fitting, and error estimation
- **`DataArray`**: Stores x, y, and their uncertainties with masking support
- **`Dataset`**: Container for multi-label DataArrays with global fit support
- **`FitResult`**: Generic result container with parameters, figures, and metadata

## Critical Review: Current Limitations

### Areas for Improvement

1. **Shot noise parameterization**: The α, β parameters in the shot noise model are currently hardcoded (α=1, β=1). Instrument-specific calibration could improve error estimates.

1. **Buffer-sample correlation**: The model assumes buffer and sample errors are independent. Systematic effects (e.g., plate edge effects) may introduce correlations.

1. **pH measurement uncertainty**: While ODR fitting accounts for x-errors, the pH uncertainty estimate (typically ±0.05 units) may be optimistic for automated titration systems.

1. **Outlier detection threshold**: The default z-score threshold of 3.0 is somewhat arbitrary. Adaptive thresholds based on dataset size could improve performance.

1. **Model comparison**: The codebase supports multiple error models but lacks systematic model selection criteria for choosing optimal error structures.

### Model Selection: WAIC and LOO-CV

Two Bayesian model comparison metrics would be valuable additions:

#### WAIC (Widely Applicable Information Criterion)

WAIC is a fully Bayesian approach to estimating out-of-sample prediction error:

```
WAIC = -2 × (lppd - p_WAIC)
```

Where:

- **lppd** (log pointwise predictive density): measures model fit
- **p_WAIC**: effective number of parameters (complexity penalty)

WAIC is computed entirely from the posterior samples without cross-validation, making it computationally efficient. ArviZ provides `az.waic(trace)`.

#### LOO-CV (Leave-One-Out Cross-Validation)

LOO-CV estimates predictive accuracy by iteratively holding out each data point:

```
LOO = Σᵢ log p(yᵢ | y₋ᵢ)
```

Direct computation requires N model fits, but **Pareto-smoothed importance sampling (PSIS-LOO)** approximates this efficiently from a single posterior sample. ArviZ provides `az.loo(trace)`.

**Comparison:**

- WAIC: faster, but can be unstable with influential observations
- LOO-CV (PSIS): more robust, provides diagnostics for problematic points
- Both prefer models with better predictive accuracy (lower is better)

These could be used to compare:

- Single vs. per-channel error scaling (`fit_binding_pymc` vs. `fit_binding_pymc2`)
- Different shot noise parameterizations (α, β values)
- Buffer subtraction methods (fit vs. mean vs. meansd)

### Recommended Future Development

1. Implement instrument calibration procedures to estimate shot noise parameters
1. Add hierarchical buffer modeling to account for plate-level correlations
1. Develop automated model selection for error structure using WAIC/LOO-CV
1. Integrate posterior predictive checks for error model validation
1. Consider full IRLS with point-wise error estimation for challenging datasets

## Glossary

- **IRLS**: Iteratively Reweighted Least Squares - an optimization method that updates point weights based on residuals
- **Huber loss**: A loss function that is quadratic for small errors and linear for large ones, providing outlier robustness
- **ODR**: Orthogonal Distance Regression - regression accounting for errors in both x and y
- **WAIC**: Widely Applicable Information Criterion - Bayesian model comparison metric
- **LOO-CV**: Leave-One-Out Cross-Validation - predictive accuracy estimation
- **PSIS**: Pareto-Smoothed Importance Sampling - efficient approximation for LOO-CV

## References

- Gelman, A., et al. (2013). Bayesian Data Analysis, 3rd Edition.
- Salvatier, J., Wiecki, T. V., & Fonnesbeck, C. (2016). Probabilistic programming in Python using PyMC3. PeerJ Computer Science.
