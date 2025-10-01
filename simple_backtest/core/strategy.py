"""Base Strategy class using Strategy and Template Method design patterns."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import pandas as pd


class Strategy(ABC):
    """Abstract base class for all trading strategies.

    Implements the Strategy Pattern, allowing runtime strategy selection and comparison.
    Uses Template Method Pattern for consistent workflow with customizable hooks.

    Strategies maintain internal state across predict() calls within a single backtest,
    but state is reset between different backtest runs.

    Example:
        ```python
        class SimpleMovingAverageStrategy(Strategy):
            def __init__(self, short_window: int = 10, long_window: int = 30):
                super().__init__(name="SMA_Strategy")
                self.short_window = short_window
                self.long_window = long_window

            def predict(
                self, data: pd.DataFrame, trade_history: List[Dict]
            ) -> Dict[str, Any]:
                # Calculate moving averages
                short_ma = data['Close'].tail(self.short_window).mean()
                long_ma = data['Close'].tail(self.long_window).mean()

                # Generate signal
                if short_ma > long_ma:
                    return {"signal": "buy", "size": 100}
                elif short_ma < long_ma:
                    return {"signal": "sell", "size": 100, "order_ids": None}
                else:
                    return {"signal": "hold", "size": 0}
        ```
    """

    def __init__(self, name: Optional[str] = None):
        """Initialize strategy.

        :param name: Strategy name, defaults to class name
        """
        self._name = name or self.__class__.__name__
        self._state_initialized = False

    def get_name(self) -> str:
        """Get strategy name.

        Returns:
            Strategy name string
        """
        return self._name

    @abstractmethod
    def predict(
        self, data: pd.DataFrame, trade_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate trading signal.

        :param data: OHLCV DataFrame with last N ticks
        :param trade_history: List of past trade dictionaries
        :return: Dict with keys: signal ("buy"/"hold"/"sell"), size (shares), order_ids
        :raises NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("Strategy must implement predict() method")

    def on_trade_executed(self, trade_info: Dict[str, Any]) -> None:
        """Hook called after a trade is executed.

        Override this method to update internal state, log trades, or perform
        post-trade actions. Default implementation does nothing.

        Args:
            trade_info: Dictionary containing trade execution details:
                       - order_id: Unique identifier
                       - timestamp: Execution time
                       - signal: "buy" or "sell"
                       - shares: Number of shares
                       - price: Execution price
                       - commission: Commission paid
                       - portfolio_value: Portfolio value after trade
                       - cash: Cash remaining
                       - positions: Current positions snapshot
                       - pnl: Profit/loss for sells
        """
        pass

    def reset_state(self) -> None:
        """Reset strategy's internal state.

        Called before starting a new backtest run. Override to clear any
        internal caches, models, or state that shouldn't persist between runs.

        Default implementation marks state as not initialized.
        """
        self._state_initialized = False

    def validate_prediction(self, prediction: Dict[str, Any]) -> None:
        """Validate prediction output format.

        Args:
            prediction: Dictionary returned by predict()

        Raises:
            ValueError: If prediction format is invalid
        """
        required_keys = {"signal", "size"}
        missing_keys = required_keys - set(prediction.keys())
        if missing_keys:
            raise ValueError(
                f"Strategy {self._name} prediction missing required keys: {missing_keys}"
            )

        signal = prediction.get("signal")
        valid_signals = {"buy", "hold", "sell"}
        if signal not in valid_signals:
            raise ValueError(
                f"Strategy {self._name} returned invalid signal '{signal}'. "
                f"Must be one of: {valid_signals}"
            )

        size = prediction.get("size")
        if not isinstance(size, (int, float)) or size < 0:
            raise ValueError(
                f"Strategy {self._name} returned invalid size '{size}'. "
                f"Must be a non-negative number."
            )

        if signal == "sell":
            if "order_ids" not in prediction:
                raise ValueError(
                    f"Strategy {self._name} returned 'sell' signal but did not specify 'order_ids'. "
                    f"Must provide list of order_ids to close or None to close oldest positions."
                )

    def __repr__(self) -> str:
        """String representation of strategy."""
        return f"{self.__class__.__name__}(name='{self._name}')"
