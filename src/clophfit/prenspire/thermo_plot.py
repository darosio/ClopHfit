"""Drawing helpers for the van't Hoff figure.

The joint fit in :mod:`clophfit.prenspire.thermo` already handles both
titration kinds correctly: a pH plate fits ``pKa(T)`` and a chloride
plate fits ``pKd(T)`` and reports ``Kd`` on its native mM scale. These
helpers put the *figure* on the same footing, so the curve drawn is the
curve that was fitted.

The two kinds differ in three places:

- the model: Henderson-Hasselbalch for pH, Langmuir for a ligand;
- the axis: pH is a linear, dimensionless scale, a ligand concentration
  spans decades and is drawn on a log axis;
- the enthalpy sign: for a pH plate ``dH = +ln10 R b``, for a ligand
  ``d ln Kd / d(1/T) = -dH/R`` gives the opposite sign.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from clophfit.prenspire.thermo import LN10, R_GAS, VanthoffModel


class VanthoffAxes(NamedTuple):
    """How one sample's van't Hoff row must be drawn."""

    left_x: str
    """Label of the left panel's titrant axis."""
    left_y: str
    """Label of the left panel's signal axis."""
    left_log: bool
    """Whether the left panel's titrant axis is logarithmic."""
    right_y: str
    """Label of the right panel's constant axis."""
    right_log: bool
    """Whether the right panel's constant axis is logarithmic."""


def vanthoff_curve(
    model: VanthoffModel,
    temp: str,
    grid: np.ndarray,
    plateaus: tuple[float, float],
) -> np.ndarray:
    """Draw the fitted model curve of one temperature, on its own titration's axis.

    Parameters
    ----------
    model : VanthoffModel
        The joint fit of the sample.
    temp : str
        Temperature label; the label the model carries.
    grid : np.ndarray
        Titrant values to evaluate the curve at.
    plateaus : tuple[float, float]
        Fitted (S0, S1) of the band the curve is drawn for.

    Returns
    -------
    np.ndarray
        The modelled signal at every grid point.
    """
    s0, s1 = plateaus
    k = model.k_by_temp[temp][0]
    # a pH plate's k is the pKa and the curve is Henderson-Hasselbalch in
    # pH space; a ligand plate's k is the Kd in mM and the curve is
    # Langmuir in concentration space
    frac = 1.0 / (1.0 + np.power(10.0, k - grid)) if model.is_ph else grid / (grid + k)
    return s0 + (s1 - s0) * frac


def vanthoff_enthalpy(model: VanthoffModel) -> float:
    """Give the enthalpy of the equilibrium in J/mol, with the right sign.

    A pH plate fits ``pKa(T) = A + B (1/T - m)`` and ``pKa = -log10 Ka``,
    so ``dH = +ln10 R B``. A ligand plate fits ``pKd(T)``, and since
    ``d ln Kd / d(1/T) = -dH/R`` the same coefficient carries the opposite
    sign of enthalpy.
    """
    sign = 1.0 if model.is_ph else -1.0
    return sign * LN10 * R_GAS * model.b


def vanthoff_right_point(model: VanthoffModel, temp: str) -> tuple[float, float]:
    """Give the constant of one temperature, as the right panel plots it.

    A pH plate's pKa is already the plotted quantity. A ligand plate's Kd
    is exponential in 1/T, so it is plotted on a log axis: the panel then
    shows the same Kd in mM that the table reports, read next to it.

    Parameters
    ----------
    model : VanthoffModel
        The joint fit of the sample.
    temp : str
        Temperature label.

    Returns
    -------
    tuple[float, float]
        The plotted value and its standard error, on the plotted scale.
    """
    return model.k_by_temp[temp]


def vanthoff_right_line(model: VanthoffModel, inv: np.ndarray) -> np.ndarray:
    """Give the fitted constant along a range of inverse temperatures.

    The joint fit carries the constant as ``A + B (1/T - m)``: a pKa for a
    pH plate, a pKd for a ligand plate. A ligand's plotted constant is the
    Kd, so the line is the exponential of that.

    Parameters
    ----------
    model : VanthoffModel
        The joint fit of the sample.
    inv : np.ndarray
        Inverse temperatures [1/K].

    Returns
    -------
    np.ndarray
        The constant at every inverse temperature, on the plotted scale.
    """
    fitted = model.a + model.b * (inv - model.m)
    if model.is_ph:
        return fitted
    return np.power(10.0, fitted)


def vanthoff_axes(model: VanthoffModel) -> VanthoffAxes:
    """How one sample's van't Hoff row must be drawn.

    Parameters
    ----------
    model : VanthoffModel
        The joint fit of the sample.

    Returns
    -------
    VanthoffAxes
        The labels and the log-scale flags of the two panels.
    """
    if model.is_ph:
        return VanthoffAxes(
            left_x="pH",
            left_y="normalized signal",
            left_log=False,
            right_y="pKa",
            right_log=False,
        )
    return VanthoffAxes(
        left_x="[Cl$^-$] (mM)",
        left_y="normalized signal",
        left_log=True,
        right_y="Kd (mM)",
        right_log=True,
    )
