"""Error classes for ClopHfit.

This module defines custom exceptions used throughout the ClopHfit package
for better error handling and user-friendly error messages.
"""


class ClopHfitError(Exception):
    """Base class for all ClopHfit errors."""


class FitError(ClopHfitError):
    """Base class for fitting errors."""


class InsufficientDataError(FitError):
    """Raised to prevent fitting failure for too few data points."""


class ConvergenceError(FitError):
    """Raised when fitting fails to converge."""


class InvalidDataError(FitError):
    """Raised when input data is invalid."""


# CLI-specific errors
class CLIError(ClopHfitError):
    """Base class for CLI-related errors."""


class FileFormatError(CLIError):
    """Raised when an input file has invalid format.

    Parameters
    ----------
    filepath : str
        Path to the problematic file.
    expected_format : str
        Description of expected file format.
    details : str, optional
        Additional details about the error.
    """

    def __init__(self, filepath: str, expected_format: str, details: str = "") -> None:
        self.filepath = filepath
        self.expected_format = expected_format
        message = f"Invalid file format: {filepath}\nExpected: {expected_format}"
        if details:
            message += f"\nDetails: {details}"
        super().__init__(message)


class DataValidationError(CLIError):
    """Raised when data fails validation checks.

    Parameters
    ----------
    message : str
        Description of the validation error.
    suggestions : list[str] | None
        List of suggestions to fix the error.
    """

    def __init__(self, message: str, suggestions: list[str] | None = None) -> None:
        self.suggestions = suggestions or []
        full_message = f"Data validation error: {message}"
        if self.suggestions:
            full_message += "\n\nSuggestions:"
            for suggestion in self.suggestions:
                full_message += f"\n  - {suggestion}"
        super().__init__(full_message)


class MissingDependencyError(CLIError):
    """Raised when required file dependencies are missing.

    Parameters
    ----------
    missing_file : str
        The file that is missing.
    required_by : str
        What requires this file.
    reason : str
        Why this file is needed.
    """

    def __init__(self, missing_file: str, required_by: str, reason: str) -> None:
        message = (
            f"Missing required file: {missing_file}\n"
            f"Required by: {required_by}\n"
            f"Reason: {reason}"
        )
        super().__init__(message)
