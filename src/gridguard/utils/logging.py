"""Structured logging utilities for research reproducibility."""

import logging
import os
import sys
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    logger_name: str = "gridguard",
) -> logging.Logger:
    """Set up and configure structured console and file logging.

    Args:
        level: Logging level (default INFO).
        log_file: Optional path to output log file.
        logger_name: Name of logger.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.handlers.clear()

    # Formatter with timestamp, level, module name and message
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def get_logger(name: str = "gridguard") -> logging.Logger:
    """Retrieve logger instance for a given module name."""
    return logging.getLogger(name)
