"""General utility helpers for clophfit."""

from __future__ import annotations

import numpy as np


def weights_from_sigma(sigma: np.ndarray) -> np.ndarray | None:
    """Convert standard deviations to ODR weights (1 / sigma**2).

    Returns None when sigma is empty to use odrpack defaults.
    """
    if sigma.size == 0:
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = np.where(sigma > 0, 1.0 / (sigma**2), 0.0)
    weights[~np.isfinite(weights)] = 0.0
    return weights
