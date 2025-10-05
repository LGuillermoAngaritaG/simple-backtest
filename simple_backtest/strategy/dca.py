"""Dollar Cost Averaging (DCA) strategy."""

from typing import Any, Dict, List

import pandas as pd

from simple_backtest.strategy.base import Strategy


class DCAStrategy(Strategy):
    """Dollar Cost Averaging - buy fixed amount at regular intervals."""

    def __init__(
        self,
        investment_amount: float = 1000,
        interval_days: int = 7,
        name: str | None = None,
    ):
        """Initialize DCA strategy.

        :param investment_amount: Dollar amount to invest at each interval
        :param interval_days: Number of days between purchases
        :param name: Strategy name (defaults to "DCA_{interval}")
        """
        super().__init__(name=name or f"DCA_{interval_days}d")
        self.investment_amount = investment_amount
        self.interval_days = interval_days
        self.days_since_last_buy = 0
        self.last_trade_date = None

    def predict(self, data: pd.DataFrame, trade_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Buy fixed amount at regular intervals, sell all on last day.

        :param data: OHLCV DataFrame with lookback window
        :param trade_history: Past trades (for backward compatibility)
        :return: Trading signal dict
        """
        # Sell all positions on the last day to realize gains
        if (
            self._portfolio_state
            and self._portfolio_state.get("is_last_day", False)
            and self.has_position()
        ):
            return self.sell_all()

        # Get current date from data
        current_date = data.index[-1] if len(data) > 0 else None

        # Check if it's time to buy
        if self.last_trade_date is None:
            # First trade - buy immediately
            self.last_trade_date = current_date
            return self.buy_cash(self.investment_amount)

        # Calculate days since last buy
        if current_date is not None:
            days_elapsed = (current_date - self.last_trade_date).days

            if days_elapsed >= self.interval_days:
                # Time to buy again
                self.last_trade_date = current_date
                return self.buy_cash(self.investment_amount)

        return self.hold()

    def reset_state(self) -> None:
        """Reset state for new backtest."""
        super().reset_state()
        self.days_since_last_buy = 0
        self.last_trade_date = None
