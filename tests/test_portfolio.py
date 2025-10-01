"""Tests for Portfolio class."""

from datetime import datetime

import pytest

from simple_backtest.core.portfolio import Portfolio


def test_portfolio_initialization():
    """Test portfolio initialization."""
    portfolio = Portfolio(initial_capital=1000.0)
    assert portfolio.cash == 1000.0
    assert portfolio.initial_capital == 1000.0
    assert len(portfolio.positions) == 0
    assert len(portfolio.trade_history) == 0


def test_portfolio_invalid_initial_capital():
    """Test portfolio rejects negative capital."""
    with pytest.raises(ValueError, match="must be positive"):
        Portfolio(initial_capital=-100.0)

    with pytest.raises(ValueError, match="must be positive"):
        Portfolio(initial_capital=0.0)


def test_execute_buy():
    """Test buy execution."""
    portfolio = Portfolio(initial_capital=1000.0)
    trade_info = portfolio.execute_buy(
        shares=10,
        price=50.0,
        commission=5.0,
        timestamp=datetime(2020, 1, 1),
    )

    assert portfolio.cash == 1000.0 - (10 * 50.0) - 5.0  # 495.0
    assert len(portfolio.positions) == 1
    assert trade_info["signal"] == "buy"
    assert trade_info["shares"] == 10
    assert trade_info["price"] == 50.0
    assert trade_info["commission"] == 5.0


def test_execute_buy_insufficient_funds():
    """Test buy fails with insufficient funds."""
    portfolio = Portfolio(initial_capital=100.0)

    with pytest.raises(ValueError, match="Insufficient cash"):
        portfolio.execute_buy(
            shares=10,
            price=50.0,
            commission=5.0,
            timestamp=datetime(2020, 1, 1),
        )


def test_execute_sell():
    """Test sell execution."""
    portfolio = Portfolio(initial_capital=1000.0)

    # First buy
    portfolio.execute_buy(
        shares=10,
        price=50.0,
        commission=5.0,
        timestamp=datetime(2020, 1, 1),
    )

    # Then sell
    trade_info = portfolio.execute_sell(
        shares=10,
        price=60.0,
        commission=5.0,
        timestamp=datetime(2020, 1, 2),
    )

    # Check P&L: (60-50)*10 - 5 = 95
    assert trade_info["pnl"] == pytest.approx(95.0)
    assert len(portfolio.positions) == 0
    assert portfolio.cash == pytest.approx(1000.0 - 505.0 + 600.0 - 5.0)  # 1090.0


def test_execute_sell_insufficient_shares():
    """Test sell fails with insufficient shares."""
    portfolio = Portfolio(initial_capital=1000.0)

    with pytest.raises(ValueError, match="Insufficient shares"):
        portfolio.execute_sell(
            shares=10,
            price=50.0,
            commission=5.0,
            timestamp=datetime(2020, 1, 1),
        )


def test_partial_sell():
    """Test partial position close."""
    portfolio = Portfolio(initial_capital=1000.0)

    portfolio.execute_buy(
        shares=10,
        price=50.0,
        commission=5.0,
        timestamp=datetime(2020, 1, 1),
    )

    # Sell half
    portfolio.execute_sell(
        shares=5,
        price=60.0,
        commission=2.0,
        timestamp=datetime(2020, 1, 2),
    )

    assert portfolio.get_total_shares() == 5
    assert len(portfolio.positions) == 1


def test_get_portfolio_value():
    """Test portfolio valuation."""
    portfolio = Portfolio(initial_capital=1000.0)

    portfolio.execute_buy(
        shares=10,
        price=50.0,
        commission=5.0,
        timestamp=datetime(2020, 1, 1),
    )

    # Portfolio value = cash + position value
    # cash = 1000 - 500 - 5 = 495
    # positions = 10 shares * 60 price = 600
    # total = 1095
    value = portfolio.get_portfolio_value(current_price=60.0)
    assert value == pytest.approx(1095.0)


def test_can_afford():
    """Test affordability check."""
    portfolio = Portfolio(initial_capital=1000.0)

    assert portfolio.can_afford(shares=10, price=50.0, commission=5.0)
    assert not portfolio.can_afford(shares=100, price=50.0, commission=5.0)


def test_reset():
    """Test portfolio reset."""
    portfolio = Portfolio(initial_capital=1000.0)

    portfolio.execute_buy(
        shares=10,
        price=50.0,
        commission=5.0,
        timestamp=datetime(2020, 1, 1),
    )

    portfolio.reset()

    assert portfolio.cash == 1000.0
    assert len(portfolio.positions) == 0
    assert len(portfolio.trade_history) == 0


def test_fifo_selling():
    """Test FIFO order of selling."""
    portfolio = Portfolio(initial_capital=10000.0)

    # Buy at different times
    portfolio.execute_buy(
        shares=10,
        price=50.0,
        commission=5.0,
        timestamp=datetime(2020, 1, 1),
        order_id="ORDER_1",
    )

    portfolio.execute_buy(
        shares=10,
        price=60.0,
        commission=5.0,
        timestamp=datetime(2020, 1, 2),
        order_id="ORDER_2",
    )

    # Sell 10 shares - should sell from ORDER_1 first
    trade_info = portfolio.execute_sell(
        shares=10,
        price=70.0,
        commission=5.0,
        timestamp=datetime(2020, 1, 3),
    )

    # P&L should be based on first order price (50)
    # (70-50)*10 - 5 = 195
    assert trade_info["pnl"] == pytest.approx(195.0)
    assert "ORDER_1" not in portfolio.positions
    assert "ORDER_2" in portfolio.positions
