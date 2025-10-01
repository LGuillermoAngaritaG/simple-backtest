"""Simple example strategies demonstrating framework usage."""

from typing import Any, Dict, List

import pandas as pd

from simple_backtest.core.strategy import Strategy


class SimpleMovingAverageStrategy(Strategy):
    """Simple Moving Average Crossover Strategy.

    Buys when short-term MA crosses above long-term MA.
    Sells when short-term MA crosses below long-term MA.
    """

    def __init__(self, short_window: int = 10, long_window: int = 30, shares: float = 100):
        """Initialize strategy.

        Args:
            short_window: Short moving average window
            long_window: Long moving average window
            shares: Number of shares to trade
        """
        super().__init__(name=f"SMA_{short_window}_{long_window}")
        self.short_window = short_window
        self.long_window = long_window
        self.shares = shares
        self.prev_signal = "hold"

    def predict(self, data: pd.DataFrame, trade_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate trading signal based on moving average crossover.

        Args:
            data: Historical OHLCV data
            trade_history: Past trades

        Returns:
            Dictionary with signal, size, and order_ids
        """
        # Calculate moving averages
        short_ma = data["Close"].tail(self.short_window).mean()
        long_ma = data["Close"].tail(self.long_window).mean()

        # Determine current position
        has_position = any(t["signal"] == "buy" for t in trade_history)
        if trade_history:
            # Check if we have open positions
            last_buys = sum(
                t["shares"] for t in trade_history if t["signal"] == "buy"
            )
            last_sells = sum(
                t["shares"] for t in trade_history if t["signal"] == "sell"
            )
            has_position = last_buys > last_sells

        # Generate signal
        if short_ma > long_ma and not has_position:
            # Buy signal
            return {"signal": "buy", "size": self.shares, "order_ids": None}

        elif short_ma < long_ma and has_position:
            # Sell signal - close all positions
            return {"signal": "sell", "size": self.shares, "order_ids": None}

        else:
            # Hold
            return {"signal": "hold", "size": 0, "order_ids": None}


class BuyAndHoldStrategy(Strategy):
    """Simple buy and hold strategy.

    Buys on first tick and holds until end.
    """

    def __init__(self, shares: float = 100):
        """Initialize strategy.

        Args:
            shares: Number of shares to buy
        """
        super().__init__(name="BuyAndHold")
        self.shares = shares
        self.bought = False

    def predict(self, data: pd.DataFrame, trade_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Buy once and hold.

        Args:
            data: Historical OHLCV data
            trade_history: Past trades

        Returns:
            Dictionary with signal, size, and order_ids
        """
        # Buy on first opportunity
        if not self.bought and len(trade_history) == 0:
            self.bought = True
            return {"signal": "buy", "size": self.shares, "order_ids": None}

        # Hold otherwise
        return {"signal": "hold", "size": 0, "order_ids": None}

    def reset_state(self) -> None:
        """Reset strategy state."""
        super().reset_state()
        self.bought = False


class RSIStrategy(Strategy):
    """Relative Strength Index (RSI) strategy.

    Buys when RSI < oversold threshold.
    Sells when RSI > overbought threshold.
    """

    def __init__(
        self,
        period: int = 14,
        oversold: float = 30,
        overbought: float = 70,
        shares: float = 100,
    ):
        """Initialize strategy.

        Args:
            period: RSI calculation period
            oversold: Oversold threshold (buy signal)
            overbought: Overbought threshold (sell signal)
            shares: Number of shares to trade
        """
        super().__init__(name=f"RSI_{period}")
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.shares = shares

    def calculate_rsi(self, prices: pd.Series) -> float:
        """Calculate RSI for given prices.

        Args:
            prices: Price series

        Returns:
            RSI value
        """
        if len(prices) < self.period + 1:
            return 50.0  # Neutral RSI if not enough data

        # Calculate price changes
        deltas = prices.diff()

        # Separate gains and losses
        gains = deltas.where(deltas > 0, 0)
        losses = -deltas.where(deltas < 0, 0)

        # Calculate average gains and losses
        avg_gain = gains.tail(self.period).mean()
        avg_loss = losses.tail(self.period).mean()

        if avg_loss == 0:
            return 100.0  # All gains, RSI = 100

        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def predict(self, data: pd.DataFrame, trade_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate trading signal based on RSI.

        Args:
            data: Historical OHLCV data
            trade_history: Past trades

        Returns:
            Dictionary with signal, size, and order_ids
        """
        # Calculate RSI
        rsi = self.calculate_rsi(data["Close"])

        # Determine current position
        has_position = False
        if trade_history:
            last_buys = sum(t["shares"] for t in trade_history if t["signal"] == "buy")
            last_sells = sum(t["shares"] for t in trade_history if t["signal"] == "sell")
            has_position = last_buys > last_sells

        # Generate signal
        if rsi < self.oversold and not has_position:
            # Oversold - buy signal
            return {"signal": "buy", "size": self.shares, "order_ids": None}

        elif rsi > self.overbought and has_position:
            # Overbought - sell signal
            return {"signal": "sell", "size": self.shares, "order_ids": None}

        else:
            # Hold
            return {"signal": "hold", "size": 0, "order_ids": None}
