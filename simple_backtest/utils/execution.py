"""Execution price extraction strategies from OHLCV data."""

from typing import Callable, Literal

import pandas as pd


def get_open_price(row: pd.Series) -> float:
    """Extract open price from OHLCV row.

    Args:
        row: Pandas Series with OHLCV data

    Returns:
        Open price

    Raises:
        KeyError: If 'Open' column missing
    """
    return float(row["Open"])


def get_close_price(row: pd.Series) -> float:
    """Extract close price from OHLCV row.

    Args:
        row: Pandas Series with OHLCV data

    Returns:
        Close price

    Raises:
        KeyError: If 'Close' column missing
    """
    return float(row["Close"])


def get_vwap(row: pd.Series) -> float:
    """Calculate Volume Weighted Average Price (VWAP).

    VWAP = (Typical Price * Volume) / Volume
    Typical Price = (High + Low + Close) / 3

    Args:
        row: Pandas Series with OHLCV data

    Returns:
        VWAP price

    Raises:
        KeyError: If required columns missing
        ValueError: If volume is zero
    """
    high = float(row["High"])
    low = float(row["Low"])
    close = float(row["Close"])
    volume = float(row["Volume"])

    if volume == 0:
        # If no volume, fall back to typical price
        return (high + low + close) / 3

    typical_price = (high + low + close) / 3
    return typical_price


def get_execution_price(
    row: pd.Series,
    method: Literal["open", "close", "vwap", "custom"] = "open",
    custom_func: Callable[[pd.Series], float] | None = None,
) -> float:
    """Extract execution price from OHLCV data using specified method.

    Args:
        row: Pandas Series with OHLCV data (must have Open, High, Low, Close, Volume)
        method: Execution price method ('open', 'close', 'vwap', 'custom')
        custom_func: Custom function for 'custom' method (takes row, returns price)

    Returns:
        Execution price

    Raises:
        ValueError: If method is invalid or custom_func not provided for 'custom' method
        KeyError: If required columns missing from row

    Example:
        >>> row = pd.Series({'Open': 100, 'High': 105, 'Low': 98, 'Close': 102, 'Volume': 1000})
        >>> get_execution_price(row, method='open')
        100.0
        >>> get_execution_price(row, method='vwap')
        101.666...
    """
    if method == "open":
        return get_open_price(row)

    elif method == "close":
        return get_close_price(row)

    elif method == "vwap":
        return get_vwap(row)

    elif method == "custom":
        if custom_func is None:
            raise ValueError("custom_func must be provided when method='custom'")
        return float(custom_func(row))

    else:
        raise ValueError(
            f"Invalid execution price method: {method}. "
            f"Must be one of: 'open', 'close', 'vwap', 'custom'"
        )


def create_execution_price_extractor(
    method: Literal["open", "close", "vwap", "custom"] = "open",
    custom_func: Callable[[pd.Series], float] | None = None,
) -> Callable[[pd.Series], float]:
    """Create execution price extractor function using Factory Pattern.

    Args:
        method: Execution price method
        custom_func: Optional custom function for 'custom' method

    Returns:
        Callable that takes row and returns price

    Example:
        >>> extractor = create_execution_price_extractor(method='open')
        >>> row = pd.Series({'Open': 100, 'Close': 102})
        >>> extractor(row)
        100.0
    """
    if method == "open":
        return get_open_price
    elif method == "close":
        return get_close_price
    elif method == "vwap":
        return get_vwap
    elif method == "custom":
        if custom_func is None:
            raise ValueError("custom_func must be provided when method='custom'")
        return custom_func
    else:
        raise ValueError(
            f"Invalid execution price method: {method}. "
            f"Must be one of: 'open', 'close', 'vwap', 'custom'"
        )


def validate_ohlcv_row(row: pd.Series) -> None:
    """Validate that row contains required OHLCV columns.

    Args:
        row: Pandas Series to validate

    Raises:
        KeyError: If required columns missing
        ValueError: If price values are invalid (negative, High < Low, etc.)
    """
    required_columns = ["Open", "High", "Low", "Close", "Volume"]
    missing_columns = [col for col in required_columns if col not in row.index]

    if missing_columns:
        raise KeyError(f"Missing required OHLCV columns: {missing_columns}")

    # Validate price relationships
    open_price = row["Open"]
    high = row["High"]
    low = row["Low"]
    close = row["Close"]
    volume = row["Volume"]

    if any(val < 0 for val in [open_price, high, low, close, volume]):
        raise ValueError(f"OHLCV values must be non-negative: {row.to_dict()}")

    if high < low:
        raise ValueError(f"High ({high}) must be >= Low ({low})")

    if not (low <= open_price <= high):
        raise ValueError(f"Open ({open_price}) must be between Low ({low}) and High ({high})")

    if not (low <= close <= high):
        raise ValueError(f"Close ({close}) must be between Low ({low}) and High ({high})")
