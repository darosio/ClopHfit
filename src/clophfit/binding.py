"""Binding equations."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt


Xtype = npt.NDArray[np.float_] | float


# TODO: use this like fz in prtecan
def kd(kd1: float, pka: float, ph: Xtype) -> Xtype:
    """Infinite cooperativity model.

    It can describe pH-dependence for chloride dissociation constant.

    Parameters
    ----------
    kd1 : float
        Dissociation constant at pH <= 5.0 (fully protonated).
    pka : float
        Acid dissociation constant.
    ph : Xtype
        pH value(s).

    Returns
    -------
    Xtype
        Predicted Kd value(s).

    Examples
    --------
    >>> kd(10, 8.4, 7.4)
    11.0
    >>> kd(10, 8.4, [7.4, 8.4])
    array([11., 20.])

    """
    return kd1 * (1 + 10 ** (pka - ph)) / 10 ** (pka - ph)


# TODO other from datan
