r"""Model functions for fitting macromolecule titration data.

This module provides mathematical models for fitting pH titration and ligand
binding experiments. The models are based on standard biochemical equilibria
and are used throughout the clophfit package for curve fitting.

Mathematical Background
-----------------------

**1-Site Binding Model**

For ligand binding (e.g., Cl⁻ binding to a protein):

.. math::

    S(x) = S_0 + (S_1 - S_0) \\frac{x/K}{1 + x/K}

where:
- S(x) is the observed signal at ligand concentration x
- S₀ is the signal when fully unbound
- S₁ is the signal when fully bound
- K is the dissociation constant

For pH titrations (Henderson-Hasselbalch):

.. math::

    S(pH) = S_0 + (S_1 - S_0) \\frac{10^{(K-pH)}}{1 + 10^{(K-pH)}}

where K is the pKa value.

**pH-Dependent Kd Model**

Models infinite cooperativity between protonation and ligand binding:

.. math::

    K_d(pH) = K_{d1} \\frac{1 + 10^{(pKa - pH)}}{10^{(pKa - pH)}}

This describes how the dissociation constant changes with pH due to
protonation effects.
"""

import typing

import numpy as np

from clophfit.clophfit_types import ArrayF


# fmt: off
@typing.overload
def binding_1site(
    x: float, K: float, S0: float, S1: float, is_ph: bool = False  # noqa: N803
) -> float: ...

@typing.overload
def binding_1site(
    x: ArrayF, K: float, S0: float, S1: float, is_ph: bool = False  # noqa: N803
) -> ArrayF: ...
# fmt: on


def binding_1site(
    x: float | ArrayF, K: float, S0: float, S1: float, is_ph: bool = False  # noqa: N803
) -> float | ArrayF:  # fmt: skip
    r"""Single site binding model function.

    This is the core model used throughout ClopHfit for fitting titration curves.
    It implements either a standard binding isotherm (for ligand concentration)
    or the Henderson-Hasselbalch equation (for pH titrations).

    Parameters
    ----------
    x : float | ArrayF
        Independent variable values.
        - For ligand binding: concentration in mM (e.g., [Cl⁻])
        - For pH titrations: pH values
    K : float
        Equilibrium constant.
        - For ligand binding: Kd in mM (dissociation constant)
        - For pH titrations: pKa (acid dissociation constant)
        Higher K means weaker binding (higher concentration needed for half-maximal signal).
    S0 : float
        Signal plateau value for the first state.
        - For ligand binding: signal when fully unbound (no ligand)
        - For pH titrations: signal in protonated state (low pH)
        Units depend on the measurement (e.g., fluorescence intensity, absorbance).
    S1 : float
        Signal plateau value for the second state.
        - For ligand binding: signal when fully bound (saturating ligand)
        - For pH titrations: signal in deprotonated state (high pH)
        Same units as S0.
    is_ph : bool, optional
        Selects the equation form:
        - True: Henderson-Hasselbalch for pH titration (default)
        - False: Standard binding isotherm for ligand concentration

    Returns
    -------
    float | ArrayF
        Predicted signal value(s) at the given x value(s).
        Returns same type as input x (scalar if x is scalar, array if x is array).

    Examples
    --------
    Standard chloride binding at half-saturation:

    >>> binding_1site(x=10.0, K=10.0, S0=100, S1=200)
    150.0

    pH titration at the pKa:

    >>> binding_1site(x=7.0, K=7.0, S0=100, S1=200, is_ph=True)
    150.0

    Array input for generating a titration curve:

    >>> import numpy as np
    >>> x_vals = np.array([0.5, 1.0, 2.0])
    >>> binding_1site(x_vals, K=1.0, S0=0, S1=1)
    array([0.33333333, 0.5       , 0.66666667])

    Typical fluorescence titration:

    >>> # GFP with pKa=7.0, fluorescence ratio changes from 0.1 to 2.5
    >>> ph_range = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
    >>> signals = binding_1site(ph_range, K=7.0, S0=0.1, S1=2.5, is_ph=True)
    >>> np.round(signals, 2)
    array([2.48, 2.28, 1.3 , 0.32, 0.12])

    Notes
    -----
    **Parameter Naming Convention**

    K, S0, and S1 use uppercase by convention from the lmfit library, where
    parameter names are case-sensitive and typically uppercase for clarity
    in fitting output.

    **Mathematical Details**

    For ligand binding (is_ph=False):

    .. math::

        S(x) = S_0 + (S_1 - S_0) \\frac{x/K}{1 + x/K}

    This is a rectangular hyperbola with:
    - S(0) = S₀
    - S(∞) = S₁
    - S(K) = (S₀ + S₁)/2 (half-saturation)

    For pH titrations (is_ph=True):

    .. math::

        S(pH) = S_0 + (S_1 - S_0) \\frac{10^{(K-pH)}}{1 + 10^{(K-pH)}}

    This is the Henderson-Hasselbalch equation with:
    - S(pH << K) ≈ S₀ (protonated)
    - S(pH >> K) ≈ S₁ (deprotonated)
    - S(pH = K) = (S₀ + S₁)/2 (half-titration)

    **Typical Use Cases**

    - Chloride binding to ClopHensor proteins (K ~ 5-50 mM)
    - pH titrations of fluorescent proteins (pKa ~ 5-9)
    - General 1:1 protein-ligand binding studies
    - Spectroscopic signal changes during titration

    See Also
    --------
    kd : pH-dependent dissociation constant model
    clophfit.fitting.core.fit_binding_glob : Global fitting across multiple datasets
    """
    if is_ph:
        return S0 + (S1 - S0) * 10 ** (K - x) / (1 + 10 ** (K - x))
    return S0 + (S1 - S0) * x / K / (1 + x / K)


def kd(kd1: float, pka: float, ph: ArrayF | float) -> ArrayF | float:
    r"""Calculate pH-dependent dissociation constant with infinite cooperativity.

    This model describes how the binding affinity of a ligand (e.g., Cl⁻) to a
    protein changes with pH when there is infinite cooperativity between
    protonation and ligand binding. It's particularly useful for biosensors
    like ClopHensor where chloride binding is pH-dependent.

    Parameters
    ----------
    kd1 : float
        Dissociation constant at low pH (mM).
        This is the Kd when the protein is fully protonated, representing the
        intrinsic binding affinity in the protonated state. Typically measured
        at pH << pKa.
    pka : float
        Acid dissociation constant (pKa).
        The pH at which the protein is 50% protonated. This determines how
        sensitive the Kd is to pH changes. For ClopHensor variants, this is
        typically around 7-8.
    ph : ArrayF | float
        pH value(s) at which to calculate Kd.
        Can be a single value or an array for generating pH-dependent curves.

    Returns
    -------
    ArrayF | float
        Predicted Kd value(s) at the given pH (mM).
        Returns same type as input ph (scalar if ph is scalar, array if ph is array).
        Kd increases with pH, meaning weaker binding at higher pH.

    Examples
    --------
    Calculate Kd at a single pH:

    >>> kd(kd1=10.0, pka=8.4, ph=7.4)
    11.0

    The Kd at the pKa is exactly 2*kd1:

    >>> kd(kd1=10.0, pka=8.4, ph=8.4)
    20.0

    Generate a pH-dependent Kd curve:

    >>> import numpy as np
    >>> ph_values = np.array([6.4, 7.4, 8.4, 9.4])
    >>> kd_values = kd(kd1=10.0, pka=8.4, ph=ph_values)
    >>> np.round(kd_values, 1)
    array([ 10.1,  11. ,  20. , 110. ])

    Typical ClopHensor parameters:

    >>> # E2 variant: kd1=9mM, pKa=7.4
    >>> ph_range = np.linspace(6, 9, 7)
    >>> kds = kd(kd1=9.0, pka=7.4, ph=ph_range)
    >>> np.round(kds, 1)
    array([  9.4,  10.1,  12.6,  20.3,  44.8, 122.3, 367.3])

    Notes
    -----
    **Mathematical Model**

    The model assumes infinite cooperativity between protonation and ligand
    binding:

    .. math::

        K_d(pH) = K_{d1} \\frac{1 + 10^{(pKa - pH)}}{10^{(pKa - pH)}}

    This can be rewritten as:

    .. math::

        K_d(pH) = K_{d1} \\left(10^{-(pKa - pH)} + 1\\right)

    **Physical Interpretation**

    - At pH << pKa: Kd ≈ kd1 (fully protonated, strong binding)
    - At pH = pKa: Kd = 2 * kd1 (half-protonated, intermediate binding)
    - At pH >> pKa: Kd → ∞ (deprotonated, no binding)

    The model predicts that deprotonation completely abolishes binding,
    which is consistent with many pH-sensitive biosensors where a critical
    ionizable residue must be protonated for ligand coordination.

    **Applications**

    - Analyzing ClopHensor chloride binding across pH ranges
    - Predicting sensor performance at physiological pH
    - Designing pH-optimized biosensor variants
    - Understanding structure-function relationships in pH-sensitive proteins

    **Limitations**

    - Assumes infinite cooperativity (protonation either completely enables
      or disables binding)
    - Does not account for multiple ionizable groups with different pKa values
    - Simplified model may not capture all aspects of real protein behavior

    See Also
    --------
    binding_1site : The binding model that uses Kd as a parameter
    clophfit.prtecan.calculate_conc : Calculate chloride concentrations for experiments

    References
    ----------
    .. [1] Arosio et al. (2010) "Simultaneous intracellular chloride and pH
           measurements using a GFP-based sensor" Nature Methods 7, 516-518.
    """
    # Support Python scalars and NumPy scalars
    if np.isscalar(ph):
        ph_val = float(ph)  # type: ignore[arg-type]
        return kd1 * (1 + 10 ** (pka - ph_val)) / 10 ** (pka - ph_val)
    ph_array = np.asarray(ph, dtype=np.float64)
    return kd1 * (1 + 10 ** (pka - ph_array)) / 10 ** (pka - ph_array)
