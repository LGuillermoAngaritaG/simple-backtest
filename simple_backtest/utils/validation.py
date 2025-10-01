"""Comprehensive input validation with custom exception hierarchy."""

import warnings
from datetime import datetime
from typing import List

import pandas as pd

from simple_backtest.core.strategy import Strategy


# Custom Exception Hierarchy
class BacktestError(Exception):
    """Base exception for backtesting framework."""

    pass


class DataValidationError(BacktestError):
    """Exception raised for data validation failures."""

    pass


class DateRangeError(BacktestError):
    """Exception raised for date range validation failures."""

    pass


class StrategyError(BacktestError):
    """Exception raised for strategy validation failures."""

    pass


def validate_dataframe(data: pd.DataFrame, strict: bool = True) -> None:
    """Validate DataFrame has required structure for backtesting.

    Args:
        data: DataFrame to validate
        strict: If True, raise errors. If False, only warn for non-critical issues.

    Raises:
        DataValidationError: If data structure is invalid
    """
    if data.empty:
        raise DataValidationError("DataFrame is empty")

    # Check required columns
    required_columns = ["Open", "High", "Low", "Close", "Volume"]
    missing_columns = [col for col in required_columns if col not in data.columns]

    if missing_columns:
        raise DataValidationError(
            f"DataFrame missing required columns: {missing_columns}. "
            f"Required: {required_columns}"
        )

    # Validate data types
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if not pd.api.types.is_numeric_dtype(data[col]):
            raise DataValidationError(
                f"Column '{col}' must be numeric, got {data[col].dtype}"
            )

    # Check for missing values
    missing_counts = data[required_columns].isnull().sum()
    if missing_counts.any():
        if strict:
            raise DataValidationError(
                f"DataFrame contains missing values:\n{missing_counts[missing_counts > 0]}"
            )
        else:
            warnings.warn(
                f"DataFrame contains missing values:\n{missing_counts[missing_counts > 0]}",
                UserWarning,
            )

    # Check for infinite values
    inf_counts = data[required_columns].isin([float("inf"), float("-inf")]).sum()
    if inf_counts.any():
        raise DataValidationError(
            f"DataFrame contains infinite values:\n{inf_counts[inf_counts > 0]}"
        )

    # Verify date index
    if not isinstance(data.index, pd.DatetimeIndex):
        if strict:
            raise DataValidationError(
                f"DataFrame index must be DatetimeIndex, got {type(data.index)}"
            )
        else:
            warnings.warn(
                f"DataFrame index should be DatetimeIndex, got {type(data.index)}. "
                "Date-based operations may not work correctly.",
                UserWarning,
            )

    # Check if index is sorted
    if isinstance(data.index, pd.DatetimeIndex):
        if not data.index.is_monotonic_increasing:
            if strict:
                raise DataValidationError("DataFrame index (dates) must be sorted in ascending order")
            else:
                warnings.warn(
                    "DataFrame index (dates) is not sorted in ascending order. "
                    "This may cause unexpected behavior.",
                    UserWarning,
                )

    # Check for duplicate indices
    if data.index.duplicated().any():
        duplicate_count = data.index.duplicated().sum()
        if strict:
            raise DataValidationError(f"DataFrame has {duplicate_count} duplicate index values")
        else:
            warnings.warn(
                f"DataFrame has {duplicate_count} duplicate index values. "
                "This may cause unexpected behavior.",
                UserWarning,
            )

    # Validate OHLC relationships
    invalid_ohlc = (
        (data["High"] < data["Low"])
        | (data["Open"] < data["Low"])
        | (data["Open"] > data["High"])
        | (data["Close"] < data["Low"])
        | (data["Close"] > data["High"])
    )

    if invalid_ohlc.any():
        invalid_count = invalid_ohlc.sum()
        if strict:
            raise DataValidationError(
                f"DataFrame has {invalid_count} rows with invalid OHLC relationships. "
                f"First invalid row:\n{data[invalid_ohlc].head(1)}"
            )
        else:
            warnings.warn(
                f"DataFrame has {invalid_count} rows with invalid OHLC relationships.",
                UserWarning,
            )

    # Check for negative values
    negative_values = (data[required_columns] < 0).any()
    if negative_values.any():
        raise DataValidationError(
            f"DataFrame contains negative values in columns: "
            f"{negative_values[negative_values].index.tolist()}"
        )

    # Check for suspicious gaps in dates (if DatetimeIndex)
    if isinstance(data.index, pd.DatetimeIndex) and len(data) > 1:
        date_diffs = data.index.to_series().diff()
        median_diff = date_diffs.median()
        large_gaps = date_diffs > median_diff * 5  # Gaps 5x larger than median

        if large_gaps.any():
            gap_count = large_gaps.sum()
            warnings.warn(
                f"Found {gap_count} large gaps in date index (>5x median gap). "
                f"This may indicate missing data.",
                UserWarning,
            )


def validate_date_range(
    data: pd.DataFrame,
    trading_start_date: datetime | None,
    trading_end_date: datetime | None,
    lookback_period: int,
) -> None:
    """Validate trading date range is compatible with data.

    Args:
        data: DataFrame with DatetimeIndex
        trading_start_date: Desired trading start date (None = use data start + lookback)
        trading_end_date: Desired trading end date (None = use data end)
        lookback_period: Number of rows needed for strategy context

    Raises:
        DateRangeError: If date range is invalid
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise DateRangeError("DataFrame must have DatetimeIndex for date range validation")

    if data.empty:
        raise DateRangeError("Cannot validate date range on empty DataFrame")

    data_start = data.index[0]
    data_end = data.index[-1]
    total_rows = len(data)

    # Validate lookback period
    if lookback_period >= total_rows:
        raise DateRangeError(
            f"lookback_period ({lookback_period}) must be less than "
            f"total data rows ({total_rows})"
        )

    # Determine effective trading dates
    if trading_start_date is None:
        # Need at least lookback_period rows before trading can start
        if total_rows <= lookback_period:
            raise DateRangeError(
                f"Need at least {lookback_period + 1} rows for lookback_period={lookback_period}"
            )
        effective_start = data.index[lookback_period]
    else:
        effective_start = trading_start_date

    if trading_end_date is None:
        effective_end = data_end
    else:
        effective_end = trading_end_date

    # Validate start date
    if effective_start < data_start:
        raise DateRangeError(
            f"Trading start date ({effective_start}) is before data start ({data_start})"
        )

    if effective_start > data_end:
        raise DateRangeError(
            f"Trading start date ({effective_start}) is after data end ({data_end})"
        )

    # Validate end date
    if effective_end > data_end:
        raise DateRangeError(
            f"Trading end date ({effective_end}) is after data end ({data_end})"
        )

    if effective_end < data_start:
        raise DateRangeError(
            f"Trading end date ({effective_end}) is before data start ({data_start})"
        )

    # Validate start < end
    if effective_start >= effective_end:
        raise DateRangeError(
            f"Trading start date ({effective_start}) must be before "
            f"trading end date ({effective_end})"
        )

    # Ensure enough data for lookback before trading starts
    start_idx = data.index.get_indexer([effective_start], method="nearest")[0]
    if start_idx < lookback_period:
        raise DateRangeError(
            f"Not enough data before trading start date for lookback_period={lookback_period}. "
            f"Need at least {lookback_period} rows before {effective_start}, "
            f"but only have {start_idx} rows."
        )


def validate_strategies(strategies: List[Strategy]) -> None:
    """Validate list of strategies.

    Args:
        strategies: List of Strategy instances

    Raises:
        StrategyError: If strategy validation fails
    """
    if not strategies:
        raise StrategyError("Must provide at least one strategy")

    for i, strategy in enumerate(strategies):
        # Check inherits from base Strategy
        if not isinstance(strategy, Strategy):
            raise StrategyError(
                f"Strategy at index {i} does not inherit from base Strategy class. "
                f"Got type: {type(strategy)}"
            )

        # Check has predict method
        if not hasattr(strategy, "predict") or not callable(getattr(strategy, "predict")):
            raise StrategyError(
                f"Strategy '{strategy.get_name()}' does not have callable predict() method"
            )

        # Check name is valid
        name = strategy.get_name()
        if not name or not isinstance(name, str):
            raise StrategyError(
                f"Strategy at index {i} has invalid name: {name}. Must be non-empty string."
            )

    # Check for duplicate names
    strategy_names = [s.get_name() for s in strategies]
    duplicate_names = [name for name in strategy_names if strategy_names.count(name) > 1]

    if duplicate_names:
        raise StrategyError(
            f"Duplicate strategy names found: {set(duplicate_names)}. "
            f"All strategies must have unique names."
        )
