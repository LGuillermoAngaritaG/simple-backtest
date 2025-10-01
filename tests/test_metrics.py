"""Tests for metrics calculations."""

import numpy as np
import pandas as pd
import pytest

from simple_backtest.metrics import definitions as defs


def test_calculate_total_return():
    """Test total return calculation."""
    result = defs.calculate_total_return(1000, 1200)
    assert result == pytest.approx(20.0)

    result = defs.calculate_total_return(1000, 800)
    assert result == pytest.approx(-20.0)


def test_calculate_cagr():
    """Test CAGR calculation."""
    # 10% per year for 2 years
    result = defs.calculate_cagr(1000, 1210, 2)
    assert result == pytest.approx(10.0, rel=0.1)

    # Zero years
    result = defs.calculate_cagr(1000, 1200, 0)
    assert result == 0.0


def test_calculate_volatility():
    """Test volatility calculation."""
    returns = pd.Series([0.01, -0.01, 0.02, -0.02, 0.01])
    vol = defs.calculate_volatility(returns, periods_per_year=252)
    assert vol > 0


def test_calculate_sharpe_ratio():
    """Test Sharpe ratio calculation."""
    # Positive returns
    returns = pd.Series([0.01, 0.02, 0.01, 0.03, 0.02])
    sharpe = defs.calculate_sharpe_ratio(returns, risk_free_rate=0.0)
    assert sharpe > 0

    # Zero volatility
    returns = pd.Series([0.01, 0.01, 0.01])
    sharpe = defs.calculate_sharpe_ratio(returns)
    assert sharpe == 0.0


def test_calculate_sortino_ratio():
    """Test Sortino ratio calculation."""
    returns = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01])
    sortino = defs.calculate_sortino_ratio(returns)
    assert sortino != 0.0

    # No downside returns
    returns = pd.Series([0.01, 0.02, 0.03])
    sortino = defs.calculate_sortino_ratio(returns)
    assert sortino == 0.0


def test_calculate_max_drawdown():
    """Test max drawdown calculation."""
    values = pd.Series([100, 110, 105, 95, 100, 120])
    result = defs.calculate_max_drawdown(values)

    assert result["max_drawdown"] > 0
    assert result["max_drawdown_duration"] >= 0


def test_calculate_calmar_ratio():
    """Test Calmar ratio calculation."""
    ratio = defs.calculate_calmar_ratio(cagr=10.0, max_drawdown=5.0)
    assert ratio == 2.0

    # Zero drawdown
    ratio = defs.calculate_calmar_ratio(cagr=10.0, max_drawdown=0.0)
    assert ratio == 0.0


def test_calculate_win_rate():
    """Test win rate calculation."""
    trades = [
        {"signal": "sell", "pnl": 100},
        {"signal": "sell", "pnl": -50},
        {"signal": "sell", "pnl": 75},
        {"signal": "buy", "pnl": None},
    ]

    win_rate = defs.calculate_win_rate(trades)
    assert win_rate == pytest.approx(66.67, rel=0.01)

    # No trades
    win_rate = defs.calculate_win_rate([])
    assert win_rate == 0.0


def test_calculate_profit_factor():
    """Test profit factor calculation."""
    trades = [
        {"signal": "sell", "pnl": 100},
        {"signal": "sell", "pnl": -50},
        {"signal": "sell", "pnl": 50},
    ]

    pf = defs.calculate_profit_factor(trades)
    assert pf == pytest.approx(3.0)

    # No losses
    trades = [{"signal": "sell", "pnl": 100}]
    pf = defs.calculate_profit_factor(trades)
    assert pf == float("inf")


def test_calculate_expectancy():
    """Test expectancy calculation."""
    trades = [
        {"signal": "sell", "pnl": 100},
        {"signal": "sell", "pnl": -50},
        {"signal": "sell", "pnl": 50},
    ]

    expectancy = defs.calculate_expectancy(trades)
    assert expectancy == pytest.approx(33.33, rel=0.01)


def test_calculate_alpha_beta():
    """Test alpha/beta calculation."""
    strategy_returns = pd.Series([0.01, 0.02, -0.01, 0.03])
    benchmark_returns = pd.Series([0.01, 0.015, -0.005, 0.02])

    result = defs.calculate_alpha_beta(strategy_returns, benchmark_returns)

    assert "alpha" in result
    assert "beta" in result
    assert result["beta"] > 0


def test_calculate_information_ratio():
    """Test information ratio calculation."""
    strategy_returns = pd.Series([0.02, 0.01, 0.03, 0.02])
    benchmark_returns = pd.Series([0.01, 0.01, 0.01, 0.01])

    ir = defs.calculate_information_ratio(strategy_returns, benchmark_returns)
    assert ir != 0.0

    # Same returns
    ir = defs.calculate_information_ratio(benchmark_returns, benchmark_returns)
    assert ir == 0.0
