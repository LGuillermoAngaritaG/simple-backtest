"""Tests for Strategy base class helper methods."""

import pytest

from simple_backtest.strategy.base import Strategy


class ConcreteStrategy(Strategy):
    """Concrete test strategy for testing base class."""

    def predict(self, data, trade_history):
        return {"signal": "hold", "size": 0, "order_ids": None}


class TestStrategyHelperMethods:
    """Tests for Strategy base class helper methods."""

    def test_has_position_without_state_raises_error(self):
        """Test that has_position() raises error when called outside predict()."""
        strategy = ConcreteStrategy()

        with pytest.raises(RuntimeError, match="Portfolio state not available"):
            strategy.has_position()

    def test_has_position_with_state(self):
        """Test has_position() with mock portfolio state."""
        strategy = ConcreteStrategy()

        # Mock portfolio state
        strategy._portfolio_state = {
            "total_shares": 0,
            "cash": 10000,
            "portfolio_value": 10000,
            "current_price": 100,
        }

        assert strategy.has_position() is False

        # Update state to have shares
        strategy._portfolio_state["total_shares"] = 50
        assert strategy.has_position() is True

    def test_get_position_without_state_raises_error(self):
        """Test that get_position() raises error when called outside predict()."""
        strategy = ConcreteStrategy()

        with pytest.raises(RuntimeError, match="Portfolio state not available"):
            strategy.get_position()

    def test_get_position_with_state(self):
        """Test get_position() returns correct shares."""
        strategy = ConcreteStrategy()
        strategy._portfolio_state = {"total_shares": 42}

        assert strategy.get_position() == 42

    def test_get_cash_without_state_raises_error(self):
        """Test that get_cash() raises error when called outside predict()."""
        strategy = ConcreteStrategy()

        with pytest.raises(RuntimeError, match="Portfolio state not available"):
            strategy.get_cash()

    def test_get_cash_with_state(self):
        """Test get_cash() returns correct cash balance."""
        strategy = ConcreteStrategy()
        strategy._portfolio_state = {"cash": 5432.10}

        assert strategy.get_cash() == 5432.10

    def test_get_portfolio_value_without_state_raises_error(self):
        """Test that get_portfolio_value() raises error when called outside predict()."""
        strategy = ConcreteStrategy()

        with pytest.raises(RuntimeError, match="Portfolio state not available"):
            strategy.get_portfolio_value()

    def test_get_portfolio_value_with_state(self):
        """Test get_portfolio_value() returns correct value."""
        strategy = ConcreteStrategy()
        strategy._portfolio_state = {"portfolio_value": 12345.67}

        assert strategy.get_portfolio_value() == 12345.67

    def test_buy_helper(self):
        """Test buy() helper method."""
        strategy = ConcreteStrategy()
        signal = strategy.buy(50)

        assert signal["signal"] == "buy"
        assert signal["size"] == 50
        assert signal["order_ids"] is None

    def test_sell_helper(self):
        """Test sell() helper method."""
        strategy = ConcreteStrategy()
        signal = strategy.sell(25)

        assert signal["signal"] == "sell"
        assert signal["size"] == 25
        assert signal["order_ids"] is None

    def test_sell_helper_with_order_ids(self):
        """Test sell() helper with specific order IDs."""
        strategy = ConcreteStrategy()
        order_ids = ["BUY_1", "BUY_2"]
        signal = strategy.sell(25, order_ids=order_ids)

        assert signal["signal"] == "sell"
        assert signal["size"] == 25
        assert signal["order_ids"] == order_ids

    def test_sell_all_without_state_raises_error(self):
        """Test that sell_all() raises error when called outside predict()."""
        strategy = ConcreteStrategy()

        with pytest.raises(RuntimeError, match="Portfolio state not available"):
            strategy.sell_all()

    def test_sell_all_with_state(self):
        """Test sell_all() returns signal to sell all shares."""
        strategy = ConcreteStrategy()
        strategy._portfolio_state = {"total_shares": 100}

        signal = strategy.sell_all()

        assert signal["signal"] == "sell"
        assert signal["size"] == 100
        assert signal["order_ids"] is None

    def test_hold_helper(self):
        """Test hold() helper method."""
        strategy = ConcreteStrategy()
        signal = strategy.hold()

        assert signal["signal"] == "hold"
        assert signal["size"] == 0
        assert signal["order_ids"] is None

    def test_buy_percent_without_state_raises_error(self):
        """Test that buy_percent() raises error when called outside predict()."""
        strategy = ConcreteStrategy()

        with pytest.raises(RuntimeError, match="Portfolio state not available"):
            strategy.buy_percent(0.1)

    def test_buy_percent_with_state(self):
        """Test buy_percent() calculates shares correctly."""
        strategy = ConcreteStrategy()
        strategy._portfolio_state = {
            "portfolio_value": 10000,
            "current_price": 100,
        }

        # Buy 10% of portfolio
        signal = strategy.buy_percent(0.1)

        assert signal["signal"] == "buy"
        assert signal["size"] == 10.0  # 10% of $10000 / $100 = 10 shares
        assert signal["order_ids"] is None

    def test_buy_percent_with_zero_price(self):
        """Test buy_percent() with zero price returns hold."""
        strategy = ConcreteStrategy()
        strategy._portfolio_state = {
            "portfolio_value": 10000,
            "current_price": 0.0,
        }

        signal = strategy.buy_percent(0.1)

        assert signal["signal"] == "hold"
        assert signal["size"] == 0

    def test_buy_percent_with_negative_price(self):
        """Test buy_percent() with negative price returns hold."""
        strategy = ConcreteStrategy()
        strategy._portfolio_state = {
            "portfolio_value": 10000,
            "current_price": -50,
        }

        signal = strategy.buy_percent(0.1)

        assert signal["signal"] == "hold"
        assert signal["size"] == 0

    def test_buy_cash_without_state_raises_error(self):
        """Test that buy_cash() raises error when called outside predict()."""
        strategy = ConcreteStrategy()

        with pytest.raises(RuntimeError, match="Portfolio state not available"):
            strategy.buy_cash(5000)

    def test_buy_cash_with_state(self):
        """Test buy_cash() calculates shares correctly."""
        strategy = ConcreteStrategy()
        strategy._portfolio_state = {"current_price": 50}

        # Buy $5000 worth
        signal = strategy.buy_cash(5000)

        assert signal["signal"] == "buy"
        assert signal["size"] == 100.0  # $5000 / $50 = 100 shares
        assert signal["order_ids"] is None

    def test_buy_cash_with_zero_price(self):
        """Test buy_cash() with zero price returns hold."""
        strategy = ConcreteStrategy()
        strategy._portfolio_state = {"current_price": 0.0}

        signal = strategy.buy_cash(5000)

        assert signal["signal"] == "hold"
        assert signal["size"] == 0

    def test_buy_cash_with_negative_price(self):
        """Test buy_cash() with negative price returns hold."""
        strategy = ConcreteStrategy()
        strategy._portfolio_state = {"current_price": -25}

        signal = strategy.buy_cash(5000)

        assert signal["signal"] == "hold"
        assert signal["size"] == 0

    def test_predict_not_implemented(self):
        """Test that Strategy cannot be instantiated without implementing predict()."""
        # Strategy is an ABC, so trying to instantiate without predict raises TypeError
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):

            class IncompleteStrategy(Strategy):
                pass

            IncompleteStrategy()

    def test_on_trade_executed_default_implementation(self):
        """Test default on_trade_executed() does nothing."""
        strategy = ConcreteStrategy()
        trade_info = {"type": "buy", "shares": 10, "price": 100}

        # Should not raise
        strategy.on_trade_executed(trade_info)

    def test_reset_state(self):
        """Test reset_state() clears internal state."""
        strategy = ConcreteStrategy()
        strategy._state_initialized = True
        strategy._portfolio_state = {"some": "state"}

        strategy.reset_state()

        assert strategy._state_initialized is False
        # portfolio_state is managed by backtest engine, not cleared by reset
