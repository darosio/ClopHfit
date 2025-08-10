"""Fit Error classes."""


class FitError(Exception):
    """Base class for fitting errors."""


class InsufficientDataError(FitError):
    """Raised to prevent fitting failure for too few data points."""


class ConvergenceError(FitError):
    """Raised when fitting fails to converge."""


class InvalidDataError(FitError):
    """Raised when input data is invalid."""
