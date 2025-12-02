# Bayesian Hierarchical Modeling for Fluorescence Titration

## Overview

We employ a Bayesian Errors-in-Variables (EIV) model to estimate dissociation constants ($K_d$) and spectral properties from fluorescence titration data. This approach accounts for uncertainties in both the dependent variable (fluorescence intensity) and the independent variable (pH), which is critical when pH measurements are sparse or subject to experimental error.

## Mathematical Model

### 1. Binding Model (Henderson-Hasselbalch)

The fluorescence intensity $y\_{ij}$ for sample $i$ at titration step $j$ is modeled as a function of pH ($x\_{ij}$):

$$ y\_{ij} = S\_{0,i} + (S\_{1,i} - S\_{0,i}) \\frac{10^{x\_{ij} - pK\_{a,i}}}{1 + 10^{x\_{ij} - pK\_{a,i}}} $$

Where:

- $S\_{0,i}$: Fluorescence at acidic plateau (protonated state).
- $S\_{1,i}$: Fluorescence at basic plateau (deprotonated state).
- $pK\_{a,i}$: Negative log of the dissociation constant.

### 2. Errors-in-Variables (EIV) Formulation

Standard regression assumes $x$ (pH) is known perfectly. However, pH measurements have uncertainty ($\\sigma\_{x}$). We treat the "true" pH as a latent variable $x\_{true}$.

**Priors on Latent pH:**
$$ x\_{true, j} \\sim \\mathcal{N}(x\_{obs, j}, \\sigma\_{x, j}) $$

To enforce physical constraints (monotonic titration), we model the differences between adjacent pH steps:
$$ \\Delta x_j = x\_{true, j} - x\_{true, j+1} $$
$$ \\Delta x_j \\sim \\text{TruncatedNormal}(\\mu = x\_{obs, j} - x\_{obs, j+1}, \\sigma = \\sqrt{\\sigma\_{x,j}^2 + \\sigma\_{x,j+1}^2}, \\text{lower}=0) $$
$$ x\_{true} = \\text{cumsum}(\\dots) $$

### 3. Likelihood Function

The observed fluorescence $y\_{obs}$ is modeled with a robust likelihood to handle outliers:

$$ y\_{obs, ij} \\sim \\text{StudentT}(\\nu, \\mu = f(x\_{true, j}, \\theta_i), \\sigma = \\sigma\_{y, ij}) $$

**Noise Model:**
We employ a heteroscedastic noise model where the variance scales with the signal intensity (shot noise approximation) plus a background buffer noise:

$$ \\sigma\_{y, ij} = \\sqrt{\\sigma\_{buffer}^2 + (\\alpha \\cdot y\_{pred, ij})} $$
Or simply learning a scaling factor $\\gamma$ for the provided experimental errors:
$$ \\sigma\_{y, ij} = \\gamma \\cdot \\sigma\_{exp, ij} $$

### 4. Hierarchical Structure (Optional)

For replicate measurements or control groups, parameters can be partially pooled:
$$ pK\_{a, i} \\sim \\mathcal{N}(\\mu\_{pK_a}, \\sigma\_{pK_a}) $$
This allows information sharing across samples, improving estimates for noisy wells.

## Implementation

The model is implemented using **PyMC**, a probabilistic programming library in Python. Posterior distributions are approximated using No-U-Turn Sampler (NUTS).

### Advantages

1. **Robustness:** Accounts for x-errors which can bias $K_d$ estimates if ignored.
1. **Uncertainty Quantification:** Provides full posterior distributions for all parameters.
1. **Flexibility:** Easily extensible to global fitting (shared $pK_a$) or multi-label data (e.g., two emission wavelengths).

## Assessment of the Approach

- Cumulative Error Modeling (create_x_true):
  - Strength: The implementation correctly models the physical process of
    titration. By modeling x_diff (intervals) and x_cumsum rather than
    independent x values, it captures how pipetting errors accumulate across the
    titration series. This is significantly more accurate than assuming
    independent Gaussian errors for each pH point.
  - Benefit: This constrains the "shape" of the pH series (the spacing between
    points) while allowing the absolute position to shift, which reflects
    reality (e.g., initial buffer pH error vs. addition volume error).
- Global Adjustment (MCMC on full plate):
  - Rationale: "pH measured only in 3-4 wells... better adjusted by MCMC".
  - Verdict: Highly Valid. Since the buffer solution is typically shared across
    the plate, pooling data from 96 wells provides massive statistical power to
    estimate the "true" underlying pH profile. This effectively calibrates the
    pH meter/preparation using the protein sensors themselves.
  - Result: list_x_true.csv likely represents this "consensus" pH profile, which
    removes systematic bias present in the sparse manual measurements.
- Individual Adjustment (Per-well EIV):
  - Rationale: "pH values are not exactly the same in all wells".
  - Verdict: Valid but Risky. Allowing pH to vary per well accounts for
    well-specific pipetting variations (heterogeneity).
  - Critical Risk (Identifiability): In a single-site binding model, a shift in
    pH is mathematically indistinguishable from a shift in $pK_a$ (since the
    term is $10^{pH - pK_a}$).
  - If applied individually, the model relies entirely on the prior (x_start,
    x_diff) to anchor the pH.
  - If the prior is too loose (large x_errc), the model may "wiggle" the pH to
    absorb random noise in the fluorescence signal, leading to overfitting and
    potentially biased $pK_a$ estimates.
  - If the prior is too tight, it ignores the well-to-well variation you aim to
    model.

### Conclusion

The strategy is robust. The Global fit is the most powerful step, correcting
systematic errors. The Individual fit is a refinement that trades some parameter
certainty for better residuals, but it must be constrained by realistic priors
(derived from the Global fit or strict pipetting error estimates) to avoid
confounding $pK_a$ with pH shifts.
