"""Base Strategy class using Strategy and Template Method design patterns."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import pandas as pd


class Strategy(ABC):
    """Abstract base class for trading strategies.

    Implement predict() to define strategy logic. State persists within a backtest
    but is reset between runs.
    """

    def __init__(self, name: Optional[str] = None):
        """Initialize strategy.

        :param name: Strategy name (defaults to class name)
        """
        self._name = name or self.__class__.__name__
        self._state_initialized = False

    def get_name(self) -> str:
        """Return strategy name."""
        return self._name

    @abstractmethod
    def predict(self, data: pd.DataFrame, trade_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate trading signal.

        :param data: OHLCV DataFrame with lookback window
        :param trade_history: List of past trades
        :return: Dict with "signal" ("buy"/"hold"/"sell"), "size", "order_ids"
        """
        raise NotImplementedError("Strategy must implement predict() method")

    def on_trade_executed(self, trade_info: Dict[str, Any]) -> None:
        """Hook called after trade execution.

        :param trade_info: Trade details (order_id, timestamp, signal, shares, price, etc.)
        """
        pass

    def reset_state(self) -> None:
        """Reset internal state before new backtest run."""
        self._state_initialized = False

    def validate_prediction(self, prediction: Dict[str, Any]) -> None:
        """Validate prediction format.

        :param prediction: Dict returned by predict()
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
