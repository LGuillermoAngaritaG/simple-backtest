"""Buy and hold strategy."""

from typing import Any, Dict, List

import pandas as pd

from simple_backtest.strategy.base import Strategy


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
        """Buy once, then hold, sell on last day.

        :param data: OHLCV DataFrame (unused)
        :param trade_history: Past trades (unused - for backward compatibility)
        :return: Trading signal dict
        """
        # Sell all positions on the last day to realize gains
        if (
            self._portfolio_state
            and self._portfolio_state.get("is_last_day", False)
            and self.has_position()
        ):
            return self.sell_all()

        # Buy once at the start
        if not self.bought and not self.has_position():
            self.bought = True
            return self.buy(self.shares)

        return self.hold()

    def reset_state(self) -> None:
        """Reset state for new backtest."""
        super().reset_state()
        self.bought = False
