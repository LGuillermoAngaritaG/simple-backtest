"""Utility functions module."""

from simple_backtest.utils.commission import (
    create_custom_commission,
    get_commission_calculator,
)
from simple_backtest.utils.execution import (
    create_execution_price_extractor,
    get_execution_price,
)
from simple_backtest.utils.logger import (
    disable_logging,
    enable_debug_logging,
    get_logger,
    setup_logging,
)
from simple_backtest.utils.validation import (
    BacktestError,
    DataValidationError,
    DateRangeError,
    StrategyError,
    validate_dataframe,
    validate_date_range,
    validate_strategies,
)

__all__ = [
    # Commission
    "get_commission_calculator",
    "create_custom_commission",
    # Execution
    "get_execution_price",
    "create_execution_price_extractor",
    # Validation
    "validate_dataframe",
    "validate_date_range",
    "validate_strategies",
    "BacktestError",
    "DataValidationError",
    "DateRangeError",
    "StrategyError",
    # Logging
    "get_logger",
    "setup_logging",
    "disable_logging",
    "enable_debug_logging",
]
