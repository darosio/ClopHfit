"""Configuration and utilities for clophfit."""

import logging
import sys
from importlib.metadata import PackageNotFoundError, version
from logging.handlers import RotatingFileHandler

try:
    __version__ = version("clophfit")
except PackageNotFoundError:
    __version__ = "unknown"

__enspire_out_dir__ = f"Meas-{__version__}"
__tecan_out_dir__ = f"out-{__version__}"


def configure_logging(
    verbose: int = 0, *, quiet: bool = False, log_file: str = "clophfit.log"
) -> None:
    """Centralized logging configuration for both library and CLI.

    Parameters
    ----------
    verbose : int
        Verbosity level (0=ERROR, 1=WARNING, 2=INFO, 3=DEBUG). Default 0.
    quiet : bool
        Silence terminal output; show only ERROR messages.
    log_file : str
        Path to log file. Default "clophfit.log".
    """
    # Map verbosity levels to logging levels
    level_mapping = {
        0: logging.ERROR,
        1: logging.WARNING,
        2: logging.INFO,
        3: logging.DEBUG,
    }
    # Get appropriate log level
    verbosity = min(max(verbose + 1, 0), 3) if not quiet else 0
    log_level = level_mapping.get(verbosity, logging.ERROR)
    # Configure namespace logger
    logger = logging.getLogger("clophfit")
    logger.setLevel(logging.DEBUG)  # Capture all messages, filter via handlers

    # --- Idempotent configuration: check existing handlers
    def has_handler_of_type(
        logger: logging.Logger, handler_type: type, match_file: str | None = None
    ) -> bool:
        """Check if the logger has a handler of a given type (optionally matching a filename).

        Parameters
        ----------
        logger : logging.Logger
            The logger to check for handlers.
        handler_type : type
            The handler type to look for.
        match_file : str | None, optional
            Optional filename to match against handler's baseFilename.

        Returns
        -------
        bool
            True if handler of the specified type is found, False otherwise.
        """
        for h in logger.handlers:
            if isinstance(h, handler_type) and (
                match_file is None or getattr(h, "baseFilename", None) == match_file
            ):
                return True
        return False

    # File handler (optional, if not already added) - always at DEBUG level to capture everything
    if log_file and not has_handler_of_type(
        logger, RotatingFileHandler, match_file=log_file
    ):
        file_handler = RotatingFileHandler(log_file, maxBytes=10**6, backupCount=3)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)-8s] %(name)-20s : %(message)s",
            datefmt="%Y-%m-%d %H:%M",
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Console handler (if verbosity > 0 or interactive)
    if verbosity > 0 or (sys.stderr.isatty() and verbose == 0):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter(fmt="[%(levelname)-8s]  %(message)s")
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # We configured 'clophfit' logger. We don't need to iterate over all loggers anymore.
    # But if someone wants warnings, we can capture them.
    logging.captureWarnings(capture=True)
    warnings_logger = logging.getLogger("py.warnings")
    if not has_handler_of_type(warnings_logger, logging.StreamHandler):
        # Only add to warnings if not already handled
        # Typically warnings shouldn't be spammed if quiet, but let's let python handle it
        pass
