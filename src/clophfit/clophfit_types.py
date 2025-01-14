"""Core type definitions in `clophfit`."""

from collections.abc import Callable, Mapping

import numpy as np
from numpy.typing import NDArray

# Array types
ArrayF = NDArray[np.float64]  # Generic float64 array
ArrayMask = NDArray[np.bool_]

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
