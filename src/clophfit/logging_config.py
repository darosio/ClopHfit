"""Configure logger."""

import logging
from logging.handlers import RotatingFileHandler


def setup_logger(
    name: str, log_file: str = "clophfit.log", level: int = logging.INFO
) -> logging.Logger:
    """Set up a logger with the given name, log file, and log level.

    Parameters
    ----------
    name : str
        The name of the logger (usually `__name__`).
    log_file : str
        The file to write log messages to.
    level : int
        The logging level (e.g., logging.INFO, logging.DEBUG).

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    # Create a logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Avoid duplicate handlers
    if not logger.handlers:
        # Console handler for stdout
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(console_formatter)
        # File handler with log rotation
        file_handler = RotatingFileHandler(log_file, maxBytes=10**6, backupCount=3)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        # Add handlers to the logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    return logger
