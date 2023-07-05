"""Type definitions used throughout the clophfit package.

It defines the following types:

- ArrayF: a NumPy array with dtype float64.

- ArrayDict: a dictionary where keys are strings and values are NumPy arrays of dtype float64.
"""


import numpy as np
from numpy import typing as npt

ArrayF = npt.NDArray[np.float_]
ArrayDict = dict[str, ArrayF]
