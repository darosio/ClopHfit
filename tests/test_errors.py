"""Test cases for custom error classes."""

import pytest

from clophfit.fitting.errors import (
    CLIError,
    ClopHfitError,
    DataValidationError,
    FileFormatError,
    MissingDependencyError,
)


def test_clophfit_error_base_class() -> None:
    """Test that ClopHfitError is the base class."""
    error = ClopHfitError("Base error")
    assert isinstance(error, Exception)
    assert str(error) == "Base error"


def test_cli_error_inheritance() -> None:
    """Test that CLI errors inherit from ClopHfitError."""
    error = CLIError("CLI error")
    assert isinstance(error, ClopHfitError)
    assert isinstance(error, Exception)


def test_file_format_error() -> None:
    """Test FileFormatError with various parameters."""
    # Basic usage
    error = FileFormatError(
        filepath="/path/to/file.csv",
        expected_format="CSV with columns: well, pH, value",
    )
    assert "Invalid file format" in str(error)
    assert "/path/to/file.csv" in str(error)
    assert "CSV with columns" in str(error)

    # With details
    error_with_details = FileFormatError(
        filepath="/path/to/file.csv",
        expected_format="CSV format",
        details="Missing 'pH' column",
    )
    assert "Missing 'pH' column" in str(error_with_details)
    assert error_with_details.filepath == "/path/to/file.csv"
    assert error_with_details.expected_format == "CSV format"


def test_data_validation_error() -> None:
    """Test DataValidationError with suggestions."""
    # Without suggestions
    error = DataValidationError("pH values must be between 0 and 14")
    assert "Data validation error" in str(error)
    assert "pH values must be between 0 and 14" in str(error)
    assert error.suggestions == []

    # With suggestions
    error_with_suggestions = DataValidationError(
        "Invalid concentration values",
        suggestions=["Check that concentrations are positive", "Use mM units"],
    )
    assert "Suggestions:" in str(error_with_suggestions)
    assert "Check that concentrations are positive" in str(error_with_suggestions)
    assert "Use mM units" in str(error_with_suggestions)
    assert len(error_with_suggestions.suggestions) == 2


def test_missing_dependency_error() -> None:
    """Test MissingDependencyError."""
    error = MissingDependencyError(
        missing_file="additions.txt",
        required_by="--cl option",
        reason="Chloride concentrations need addition volumes",
    )

    assert "Missing required file: additions.txt" in str(error)
    assert "Required by: --cl option" in str(error)
    assert "Reason: Chloride concentrations" in str(error)


def test_error_can_be_caught_base_class() -> None:
    """Test that CLIError can be caught as ClopHfitError."""
    msg = "Test error"
    with pytest.raises(ClopHfitError):
        raise CLIError(msg)


def test_error_can_be_caught_specific() -> None:
    """Test that specific errors can be caught."""
    msg = "file.txt"
    with pytest.raises(FileFormatError):
        raise FileFormatError(msg, "Expected format")


def test_error_can_be_caught_by_inheritance() -> None:
    """Test that DataValidationError can be caught as CLIError."""
    msg = "Validation failed"
    with pytest.raises(CLIError):
        raise DataValidationError(msg)


def test_error_messages_are_helpful() -> None:
    """Test that error messages contain actionable information."""
    # FileFormatError should include filepath and expected format
    error1 = FileFormatError("/data/input.csv", "CSV with 'well' and 'value' columns")
    message1 = str(error1)
    assert "/data/input.csv" in message1
    assert "CSV" in message1
    assert "well" in message1

    # DataValidationError should include suggestions when provided
    error2 = DataValidationError(
        "pH out of range",
        suggestions=["pH should be 4-11", "Check measurement equipment"],
    )
    message2 = str(error2)
    assert "pH out of range" in message2
    assert "pH should be 4-11" in message2
    assert "Check measurement equipment" in message2

    # MissingDependencyError should explain what's missing and why
    error3 = MissingDependencyError(
        "scheme.txt", "--bg flag", "Buffer positions must be defined"
    )
    message3 = str(error3)
    assert "scheme.txt" in message3
    assert "--bg flag" in message3
    assert "Buffer positions" in message3
