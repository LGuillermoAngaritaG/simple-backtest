"""Buy and hold strategy."""

from typing import Any, Dict, List

import pandas as pd

from simple_backtest.strategy.strategy_base import Strategy


class BuyAndHoldStrategy(Strategy):
    """Buy once and hold until end."""

    def __init__(self, shares: float = 100, name: str | None = None):
        """Initialize strategy.

        :param shares: Number of shares to buy
        :param name: Strategy name (defaults to "BuyAndHold")
        """
        super().__init__(name=name or "BuyAndHold")
        self.shares = shares
        self.bought = False

    def predict(self, data: pd.DataFrame, trade_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Buy once, then hold.

        :param data: OHLCV DataFrame (unused)
        :param trade_history: Past trades
        :return: Trading signal dict
        """
        if not self.bought and len(trade_history) == 0:
            self.bought = True
            return {"signal": "buy", "size": self.shares, "order_ids": None}
        return {"signal": "hold", "size": 0, "order_ids": None}

    def reset_state(self) -> None:
        """Reset state for new backtest."""
        super().reset_state()
        self.bought = False
