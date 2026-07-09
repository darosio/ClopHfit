"""Core type definitions in `clophfit`."""

from collections.abc import Callable, Mapping

import numpy as np
import odrpack
import xarray as xr
from lmfit.minimizer import Minimizer  # type: ignore[import-untyped]
from numpy.typing import NDArray

# Array types
ArrayF = NDArray[np.float64]  # Generic float64 array
ArrayMask = NDArray[np.bool_]

# Minimizer/backend result union shared across fitting modules.
# Defined here (a dependency-light leaf module) rather than in
# ``fitting.data_structures`` so it is never pulled into the fitting-package
# import cycle, which would otherwise demote this implicit alias to a plain
# variable in mypy's eyes (``MiniT is not valid as a type``).
MiniT = Minimizer | odrpack.OdrResult | xr.DataTree

# Dictionary types
ArrayDict = dict[
    str, ArrayF
]  # Dictionary with string keys and float64 arrays as values

# Keyword arguments type
Kwargs = Mapping[
    str, str | int | float | bool | None
]  # Immutable dictionary for kwargs

# Callable types
FloatFunc = Callable[[float], float]  # Function taking a float and returning a float
