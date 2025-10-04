"""Backtesting Framework - High-performance backtesting for trading strategies."""

__version__ = "0.1.0"

from simple_backtest.config.settings import BacktestConfig
from simple_backtest.core.backtest import Backtest
from simple_backtest.core.portfolio import Portfolio
from simple_backtest.strategy.strategy_base import Strategy

__all__ = [
    "BacktestConfig",
    "Backtest",
    "Portfolio",
    "Strategy",
]
