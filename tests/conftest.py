"""Pytest fixtures for backtesting framework tests."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from simple_backtest.config.settings import BacktestConfig
from simple_backtest.strategy.strategy_base import Strategy


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=100, freq="D")

    close_prices = 100 + np.random.randn(100).cumsum()
    high_prices = close_prices + np.random.uniform(0, 2, 100)
    low_prices = close_prices - np.random.uniform(0, 2, 100)
    open_prices = close_prices + np.random.uniform(-1, 1, 100)
    volumes = np.random.randint(100000, 1000000, 100)

    return pd.DataFrame(
        {
            "Open": open_prices,
            "High": high_prices,
            "Low": low_prices,
            "Close": close_prices,
            "Volume": volumes,
        },
        index=dates,
    )


@pytest.fixture
def default_config():
    """Default backtest configuration."""
    return BacktestConfig(
        initial_capital=10000.0,
        lookback_period=10,
        commission_type="percentage",
        commission_value=0.001,
        execution_price="open",
        trading_start_date=datetime(2020, 1, 15),
        trading_end_date=datetime(2020, 4, 1),
    )


class DummyStrategy(Strategy):
    """Simple test strategy that always buys."""

    def __init__(self):
        super().__init__(name="Dummy")
        self.call_count = 0

    def predict(self, data, trade_history):
        """Always buy 10 shares."""
        self.call_count += 1
        if not trade_history:
            return {"signal": "buy", "size": 10, "order_ids": None}
        return {"signal": "hold", "size": 0, "order_ids": None}


@pytest.fixture
def dummy_strategy():
    """Dummy strategy for testing."""
    return DummyStrategy()
