"""Moving average crossover strategy."""

from typing import Any, Dict, List

import pandas as pd

from simple_backtest.strategy.strategy_base import Strategy


class MovingAverageStrategy(Strategy):
    """Buy when short MA crosses above long MA, sell when it crosses below."""

    def __init__(
        self,
        short_window: int = 10,
        long_window: int = 30,
        shares: float = 100,
        name: str | None = None,
    ):
        """Initialize strategy.

        :param short_window: Short MA period
        :param long_window: Long MA period
        :param shares: Shares to trade per signal
        :param name: Strategy name (defaults to "MA_{short}_{long}")
        """
        if short_window <= 0 or long_window <= 0:
            raise ValueError("Moving average windows must be positive")
        if short_window >= long_window:
            raise ValueError("Short window must be less than long window")

        super().__init__(name=name or f"MA_{short_window}_{long_window}")
        self.short_window = short_window
        self.long_window = long_window
        self.shares = shares

    def predict(self, data: pd.DataFrame, trade_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate signal based on MA crossover.

        :param data: OHLCV DataFrame
        :param trade_history: Past trades
        :return: Trading signal dict
        """
        if len(data) < self.long_window:
            return {"signal": "hold", "size": 0, "order_ids": None}

        short_ma = data["Close"].tail(self.short_window).mean()
        long_ma = data["Close"].tail(self.long_window).mean()
        has_position = self._has_open_position(trade_history)

        if short_ma > long_ma and not has_position:
            return {"signal": "buy", "size": self.shares, "order_ids": None}
        elif short_ma < long_ma and has_position:
            return {"signal": "sell", "size": self.shares, "order_ids": None}
        else:
            return {"signal": "hold", "size": 0, "order_ids": None}

    def _has_open_position(self, trade_history: List[Dict[str, Any]]) -> bool:
        """Check if strategy has open positions.

        :param trade_history: Past trades
        :return: True if has open positions
        """
        if not trade_history:
            return False
        total_bought = sum(t["shares"] for t in trade_history if t["signal"] == "buy")
        total_sold = sum(t["shares"] for t in trade_history if t["signal"] == "sell")
        return total_bought > total_sold
