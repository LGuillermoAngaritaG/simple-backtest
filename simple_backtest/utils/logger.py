"""Logging configuration for simple-backtest framework."""

import logging
import sys
from typing import Optional


def get_logger(name: str = "simple_backtest") -> logging.Logger:
    """Get a logger instance for the framework.

    :param name: Logger name (default: 'simple_backtest')
    :return: Configured logger instance
    """
    return logging.getLogger(name)


def setup_logging(
    level: int = logging.INFO,
    format_string: Optional[str] = None,
    add_handler: bool = True,
) -> None:
    """Setup logging configuration for the framework.

    :param level: Logging level (default: INFO)
    :param format_string: Custom format string (default: simple format with timestamp)
    :param add_handler: Whether to add console handler (default: True)

    Example:
        # Setup with default INFO level
        setup_logging()

        # Setup with DEBUG level for detailed logs
        setup_logging(level=logging.DEBUG)

        # Setup with custom format
        setup_logging(
            level=logging.WARNING,
            format_string='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        )
    """
    logger = logging.getLogger("simple_backtest")
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    if add_handler:
        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)

        # Create formatter
        if format_string is None:
            format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

        formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(handler)

    # Prevent propagation to root logger
    logger.propagate = False


def disable_logging() -> None:
    """Disable all logging output from simple-backtest.

    Useful for production environments or when you want to suppress all output.

    Example:
        from simple_backtest.utils.logger import disable_logging
        disable_logging()
    """
    logger = logging.getLogger("simple_backtest")
    logger.setLevel(logging.CRITICAL + 1)  # Disable all logs
    logger.handlers.clear()


def enable_debug_logging() -> None:
    """Enable debug-level logging with detailed format.

    Useful for development and troubleshooting.

    Example:
        from simple_backtest.utils.logger import enable_debug_logging
        enable_debug_logging()
    """
    setup_logging(
        level=logging.DEBUG,
        format_string="%(asctime)s [%(levelname)s] %(name)s.%(funcName)s:%(lineno)d - %(message)s",
    )


# Initialize default logging on import
# Users can override by calling setup_logging() or disable_logging()
setup_logging(level=logging.WARNING)  # Default to WARNING to avoid spam
