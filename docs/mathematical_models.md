# Mathematical Models for Titration Fitting

This document describes the mathematical models used in ClopHfit for fitting pH titration and ligand binding data.

## Overview

ClopHfit provides models for analyzing two types of experiments:

1. **Ligand Binding**: Measuring how a signal (e.g., fluorescence) changes with ligand concentration
1. **pH Titration**: Measuring how a signal changes with pH

Both use variations of the same fundamental binding equation.

## 1-Site Binding Model

### Ligand Concentration Dependence

For a simple 1:1 binding equilibrium between a protein (P) and ligand (L):

```
P + L ⇌ PL
```

The dissociation constant is:

```
Kd = [P][L] / [PL]
```

The fraction of protein bound (θ) is:

```
θ = [L] / (Kd + [L])
```

If the signal changes linearly with binding, the observed signal S is:

```
S(x) = S₀ + (S₁ - S₀) × θ
     = S₀ + (S₁ - S₀) × x / (K + x)
```

where:

- **x** = ligand concentration (mM)
- **K** = Kd (dissociation constant, mM)
- **S₀** = signal when unbound ([L] = 0)
- **S₁** = signal when fully bound ([L] → ∞)

### Implementation

```python
from clophfit.fitting.models import binding_1site

# Chloride binding with Kd = 10 mM
signal = binding_1site(x=5.0, K=10.0, S0=100, S1=200, is_ph=False)
# Returns: 133.33 (one-third bound at x = K/2)
```

### Key Properties

- **At x = 0**: S = S₀ (no ligand, unbound)
- **At x = K**: S = (S₀ + S₁)/2 (half-saturated)
- **At x → ∞**: S → S₁ (saturating ligand, fully bound)
- **Slope at x = K**: (S₁ - S₀) / (4K)

## pH Titration Model (Henderson-Hasselbalch)

### Acid-Base Equilibrium

For a titratable group with acid dissociation:

```
HA ⇌ H⁺ + A⁻
```

The Henderson-Hasselbalch equation describes the fraction in each form:

```
pH = pKa + log([A⁻]/[HA])
```

Rearranging for the fraction protonated:

```
θ_HA = [HA] / ([HA] + [A⁻])
     = 1 / (1 + 10^(pH - pKa))
     = 10^(pKa - pH) / (1 + 10^(pKa - pH))
```

If the signal depends on protonation state:

```
S(pH) = S₀ + (S₁ - S₀) × θ_HA
      = S₀ + (S₁ - S₀) × 10^(K - pH) / (1 + 10^(K - pH))
```

where:

- **pH** = solution pH
- **K** = pKa (acid dissociation constant)
- **S₀** = signal in protonated state (low pH)
- **S₁** = signal in deprotonated state (high pH)

### Implementation

```python
from clophfit.fitting.models import binding_1site

# GFP chromophore with pKa = 7.0
signal = binding_1site(x=8.0, K=7.0, S0=0.5, S1=2.0, is_ph=True)
# Returns: 1.85 (mostly deprotonated at pH > pKa)
```

### Key Properties

- **At pH \<< pKa**: S → S₀ (fully protonated)
- **At pH = pKa**: S = (S₀ + S₁)/2 (half-titrated)
- **At pH >> pKa**: S → S₁ (fully deprotonated)
- **Slope at pH = pKa**: -(S₁ - S₀) × ln(10) / 4 ≈ -0.576 × (S₁ - S₀)

## pH-Dependent Dissociation Constant

### Infinite Cooperativity Model

For systems where ligand binding requires a specific protonation state (e.g., ClopHensor chloride binding):

```
P-H + Cl⁻ ⇌ P-H·Cl⁻     (binds at low pH)
P-H ⇌ P⁻ + H⁺          (titrates at pKa)
P⁻ + Cl⁻ ⇌ P⁻·Cl⁻      (no binding at high pH)
```

Assuming infinite cooperativity (deprotonated form cannot bind):

```
Kd(pH) = Kd₁ × (1 + 10^(pKa - pH)) / 10^(pKa - pH)
```

where:

- **Kd₁** = dissociation constant when fully protonated (pH \<< pKa)
- **pKa** = acid dissociation constant of the binding-critical residue
- **pH** = solution pH

### Implementation

```python
from clophfit.fitting.models import kd

# ClopHensor E2 variant
kd_at_ph7 = kd(kd1=9.0, pka=7.4, ph=7.0)
# Returns: ~10.6 mM (slightly weaker than at low pH)

kd_at_ph8 = kd(kd1=9.0, pka=7.4, ph=8.0)
# Returns: ~23.8 mM (much weaker as protein deprotonates)
```

### Key Properties

- **At pH \<< pKa**: Kd ≈ Kd₁ (fully protonated, intrinsic affinity)
- **At pH = pKa**: Kd = 2 × Kd₁ (half-protonated, 2-fold weaker)
- **At pH >> pKa**: Kd → ∞ (deprotonated, no binding)

### Physical Interpretation

This model describes complete coupling between protonation and binding:

1. At low pH, the residue is protonated and binding is strong
1. As pH increases, the residue deprotonates
1. The deprotonated form completely loses the ability to bind ligand

This is typical of metal coordination sites or hydrogen bond donors that are critical for ligand binding.

## Global Fitting

ClopHfit supports global fitting where a single equilibrium constant (K or pKa) is shared across multiple datasets with different signal amplitudes (S₀, S₁).

### Mathematical Formulation

For *n* datasets measured at the same x values:

```
Minimize: Σᵢ Σⱼ wᵢⱼ × (Sᵢⱼ - S_model(xⱼ; K, S₀ᵢ, S₁ᵢ))²
```

where:

- **i** = dataset index (e.g., different wavelengths)
- **j** = data point index (e.g., different pH values)
- **wᵢⱼ** = weight for point (i,j)
- **K** = shared equilibrium constant
- **S₀ᵢ, S₁ᵢ** = dataset-specific plateau values

### Advantages

1. **More robust K estimation**: Combines information from multiple signals
1. **Better handling of noise**: Individual datasets may be noisy, but global fit averages errors
1. **Enforces consistency**: All datasets must share the same K value

### Example

```python
from clophfit.fitting.core import fit_binding_glob
from clophfit.fitting.data_structures import DataArray, Dataset
import numpy as np

# Two fluorescence channels measuring same pH titration
ph = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
channel1 = np.array([100, 150, 200, 250, 280])  # Increases with pH
channel2 = np.array([500, 450, 400, 350, 320])  # Decreases with pH

ds = Dataset({
    "ch1": DataArray(ph, channel1),
    "ch2": DataArray(ph, channel2)
}, is_ph=True)

result = fit_binding_glob(ds)
print(f"pKa = {result.result.params['K'].value:.2f}")
```

## Parameter Estimation

### Initial Guesses

Good initial parameter guesses improve fitting convergence:

- **K** (dissociation constant or pKa):

  - For Kd: Use the x value at half-maximal signal
  - For pKa: Use the pH at half-titration (typically 6-9 for proteins)

- **S₀** (first plateau):

  - Use the signal at the lowest x value
  - For pH: signal at low pH (typically protonated)

- **S₁** (second plateau):

  - Use the signal at the highest x value
  - For pH: signal at high pH (typically deprotonated)

### Uncertainty Estimation

ClopHfit provides multiple methods for estimating parameter uncertainty:

1. **Covariance matrix**: From least-squares fitting (Jacobian-based)
1. **Bootstrap**: Resampling residuals to estimate confidence intervals
1. **MCMC (Bayesian)**: Full posterior distributions using PyMC

### Typical Uncertainties

For well-designed experiments:

- **K**: ±5-10% (1 standard error)
- **S₀, S₁**: ±2-5% (generally more precise than K)

Poor signal-to-noise or insufficient data range can increase these substantially.

## Model Selection

### When to Use Each Model

**Ligand Binding (is_ph=False)**:

- Titrating with known ligand concentrations
- Examples: Cl⁻, Ca²⁺, metabolites, drugs
- X-axis: mM or μM concentration
- Typical Kd range: 0.1-100 mM (ions), 0.1-1000 nM (high-affinity)

**pH Titration (is_ph=True)**:

- Varying pH of the solution
- Examples: fluorescent protein chromophores, titratable residues
- X-axis: pH units (typically 4-11)
- Typical pKa range: 4-10 for amino acid side chains

**pH-Dependent Kd**:

- When ligand binding affinity changes with pH
- Use `kd()` to predict Kd at different pH values
- Then use `binding_1site()` with the pH-adjusted Kd

### Limitations

The 1-site model assumes:

1. **1:1 stoichiometry**: One ligand per protein
1. **Two-state**: Only bound and unbound (no intermediates)
1. **No cooperativity**: Independent binding sites (for multiple sites)
1. **Equilibrium**: Measurements made after binding equilibrates

For systems that violate these assumptions, more complex models may be needed.

## References

1. **Henderson-Hasselbalch Equation**

   - Henderson, L.J. (1908) *Am. J. Physiol.* 21: 173-179
   - Hasselbalch, K.A. (1917) *Biochem. Z.* 78: 112-144

1. **ClopHensor Development**

   - Arosio et al. (2010) "Simultaneous intracellular chloride and pH measurements using a GFP-based sensor" *Nature Methods* 7: 516-518

1. **Global Fitting Methods**

   - Beechem, J.M. (1992) "Global analysis of biochemical and biophysical data" *Methods in Enzymology* 210: 37-54

1. **Bayesian Parameter Estimation**

   - Gelman et al. (2013) *Bayesian Data Analysis*, 3rd ed. Chapman and Hall/CRC

## See Also

- [API Reference](api/binding.rst) - Detailed function documentation
- [Tutorial](tutorials/tutorials.rst) - Worked examples
- [Data Structures](api/api.rst) - DataArray and Dataset classes
