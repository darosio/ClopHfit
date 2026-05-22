"""Error models for fitting."""

from collections.abc import Mapping
from typing import Protocol

import numpy as np

from clophfit.clophfit_types import ArrayF


class ErrorModel(Protocol):
    """Protocol for error models."""

    def compute_variance(self, signal: ArrayF, label: int | str = "") -> ArrayF:
        """Compute variance given a signal estimate."""
        ...


class ComprehensiveErrorModel:
    """Physical error: var = sigma_read^2 + gain * signal + (rel_error * signal)^2.

    Shot Noise + Proportional: Read noise + Poisson shot noise + Scintillation/Proportional noise.
    Can be used as a Constant Error Model by setting `gain=0` and `rel_error=0`, or
    as a Proportional Error Model by setting `gain=0`.

    Parameters
    ----------
    sigma_read : float | ArrayF | Mapping[int | str, float | ArrayF]
        Read noise floor.
    gain : float | Mapping[int | str, float]
        Instrument gain linking signal to variance (Poisson term).
    rel_error : float | Mapping[int | str, float]
        Proportional error coefficient.
    """

    def __init__(
        self,
        sigma_read: float | ArrayF | Mapping[int | str, float | ArrayF],
        gain: float | Mapping[int | str, float],
        rel_error: float | Mapping[int | str, float],
    ) -> None:
        self.sigma_read = sigma_read
        self.gain = gain
        self.rel_error = rel_error

    def compute_variance(self, signal: ArrayF, label: int | str = "") -> ArrayF:
        """Compute variance."""
        sigma = (
            self.sigma_read[label]
            if isinstance(self.sigma_read, Mapping)
            else self.sigma_read
        )
        g = self.gain[label] if isinstance(self.gain, Mapping) else self.gain
        rel_err = (
            self.rel_error[label]
            if isinstance(self.rel_error, Mapping)
            else self.rel_error
        )

        # Poisson shot noise term applies to raw signal counts, which should be positive
        sig = np.maximum(0.0, signal)

        return sigma**2 + g * sig + (rel_err * sig) ** 2
