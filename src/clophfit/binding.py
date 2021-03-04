"""Binding equations."""

from typing import Union

import numpy as np
import numpy.typing as npt

Xtype = Union[npt.ArrayLike, float]


def kd(Kd1: float, pKa: float, pH: Xtype) -> Union[npt.ArrayLike, float]:
    """Infinite cooperativity model.

    It can describe pH-dependence for chloride dissociation constant.

    Parameters
    ----------
    Kd1 : float
        Dissociation constant at pH <= 5.0 (fully protonated).
    pKa : float
        Acid dissociation constant.
    pH : Xtype
        pH value(s).

    Returns
    -------
    float or np.ndarray
        Predicted Kd value(s).

    Examples
    --------
    >>> kd(10, 8.4, 7.4)
    11.0
    >>> kd(10, 8.4, [7.4, 8.4])
    array([11., 20.])

    """
    if type(pH) is list:
        pH = np.array(pH).astype(float)
    return Kd1 * (1 + 10 ** (pKa - pH)) / 10 ** (pKa - pH)


# TODO other from datan
