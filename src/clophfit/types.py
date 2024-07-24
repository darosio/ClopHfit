"""Type definitions used throughout the clophfit package."""

import numpy as np
from numpy.typing import NDArray

ArrayF = NDArray[np.float64]
ArrayDict = dict[str, ArrayF]
Kwargs = dict[str, str | int | float | bool | None]
