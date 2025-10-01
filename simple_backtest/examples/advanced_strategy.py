"""Advanced example strategies with more complex logic."""

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from simple_backtest.core.strategy import Strategy


class BollingerBandsStrategy(Strategy):
    """Bollinger Bands mean reversion strategy.

    Buys when price touches lower band.
    Sells when price touches upper band.
    """

    def __init__(
        self,
        period: int = 20,
        num_std: float = 2.0,
        shares: float = 100,
    ):
        """Initialize strategy.

        Args:
            period: Moving average period
            num_std: Number of standard deviations for bands
            shares: Number of shares to trade
        """
        super().__init__(name=f"BB_{period}_{num_std}")
        self.period = period
        self.num_std = num_std
        self.shares = shares

    def predict(self, data: pd.DataFrame, trade_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate signal based on Bollinger Bands.

        Args:
            data: Historical OHLCV data
            trade_history: Past trades

        Returns:
            Dictionary with signal, size, and order_ids
        """
        # Calculate Bollinger Bands
        prices = data["Close"]
        ma = prices.tail(self.period).mean()
        std = prices.tail(self.period).std()

        upper_band = ma + (self.num_std * std)
        lower_band = ma - (self.num_std * std)

        current_price = prices.iloc[-1]

        # Determine position
        has_position = False
        if trade_history:
            last_buys = sum(t["shares"] for t in trade_history if t["signal"] == "buy")
            last_sells = sum(t["shares"] for t in trade_history if t["signal"] == "sell")
            has_position = last_buys > last_sells

        # Generate signal
        if current_price <= lower_band and not has_position:
            # Price at lower band - buy
            return {"signal": "buy", "size": self.shares, "order_ids": None}

        elif current_price >= upper_band and has_position:
            # Price at upper band - sell
            return {"signal": "sell", "size": self.shares, "order_ids": None}

        else:
            return {"signal": "hold", "size": 0, "order_ids": None}


class MomentumStrategy(Strategy):
    """Momentum strategy based on rate of change.

    Buys when momentum is positive and increasing.
    Sells when momentum turns negative.
    """

    def __init__(
        self,
        lookback: int = 10,
        threshold: float = 0.02,
        shares: float = 100,
    ):
        """Initialize strategy.

        Args:
            lookback: Period for momentum calculation
            threshold: Minimum momentum threshold for signals
            shares: Number of shares to trade
        """
        super().__init__(name=f"Momentum_{lookback}")
        self.lookback = lookback
        self.threshold = threshold
        self.shares = shares

    def predict(self, data: pd.DataFrame, trade_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate signal based on momentum.

        Args:
            data: Historical OHLCV data
            trade_history: Past trades

        Returns:
            Dictionary with signal, size, and order_ids
        """
        prices = data["Close"]

        if len(prices) < self.lookback:
            return {"signal": "hold", "size": 0, "order_ids": None}

        # Calculate momentum (rate of change)
        current_price = prices.iloc[-1]
        past_price = prices.iloc[-self.lookback]
        momentum = (current_price - past_price) / past_price

        # Determine position
        has_position = False
        if trade_history:
            last_buys = sum(t["shares"] for t in trade_history if t["signal"] == "buy")
            last_sells = sum(t["shares"] for t in trade_history if t["signal"] == "sell")
            has_position = last_buys > last_sells

        # Generate signal
        if momentum > self.threshold and not has_position:
            # Positive momentum - buy
            return {"signal": "buy", "size": self.shares, "order_ids": None}

        elif momentum < -self.threshold and has_position:
            # Negative momentum - sell
            return {"signal": "sell", "size": self.shares, "order_ids": None}

        else:
            return {"signal": "hold", "size": 0, "order_ids": None}


class MACDStrategy(Strategy):
    """Moving Average Convergence Divergence (MACD) strategy.

    Buys when MACD crosses above signal line.
    Sells when MACD crosses below signal line.
    """

    def __init__(
        self,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        shares: float = 100,
    ):
        """Initialize strategy.

        Args:
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line EMA period
            shares: Number of shares to trade
        """
        super().__init__(name=f"MACD_{fast}_{slow}_{signal}")
        self.fast = fast
        self.slow = slow
        self.signal_period = signal
        self.shares = shares
        self.prev_macd_signal = 0.0

    def calculate_ema(self, prices: pd.Series, period: int) -> float:
        """Calculate Exponential Moving Average.

        Args:
            prices: Price series
            period: EMA period

        Returns:
            EMA value
        """
        if len(prices) < period:
            return prices.mean()

        # Simple EMA calculation
        return prices.ewm(span=period, adjust=False).mean().iloc[-1]

    def predict(self, data: pd.DataFrame, trade_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate signal based on MACD.

        Args:
            data: Historical OHLCV data
            trade_history: Past trades

        Returns:
            Dictionary with signal, size, and order_ids
        """
        prices = data["Close"]

        # Calculate MACD
        fast_ema = self.calculate_ema(prices, self.fast)
        slow_ema = self.calculate_ema(prices, self.slow)
        macd = fast_ema - slow_ema

        # Calculate MACD values for signal line
        if len(prices) < self.slow + self.signal_period:
            return {"signal": "hold", "size": 0, "order_ids": None}

        # Simplified signal line (EMA of MACD)
        # In real implementation, would track MACD history
        macd_signal = macd * 0.9  # Simplified

        # Determine position
        has_position = False
        if trade_history:
            last_buys = sum(t["shares"] for t in trade_history if t["signal"] == "buy")
            last_sells = sum(t["shares"] for t in trade_history if t["signal"] == "sell")
            has_position = last_buys > last_sells

        # Detect crossover
        if macd > macd_signal and self.prev_macd_signal <= 0 and not has_position:
            # Bullish crossover - buy
            self.prev_macd_signal = macd - macd_signal
            return {"signal": "buy", "size": self.shares, "order_ids": None}

        elif macd < macd_signal and self.prev_macd_signal >= 0 and has_position:
            # Bearish crossover - sell
            self.prev_macd_signal = macd - macd_signal
            return {"signal": "sell", "size": self.shares, "order_ids": None}

        else:
            self.prev_macd_signal = macd - macd_signal
            return {"signal": "hold", "size": 0, "order_ids": None}

    def reset_state(self) -> None:
        """Reset strategy state."""
        super().reset_state()
        self.prev_macd_signal = 0.0


class VolatilityBreakoutStrategy(Strategy):
    """Volatility breakout strategy.

    Buys when price breaks above recent high + volatility threshold.
    Sells when price breaks below recent low - volatility threshold.
    """

    def __init__(
        self,
        period: int = 20,
        volatility_factor: float = 1.5,
        shares: float = 100,
    ):
        """Initialize strategy.

        Args:
            period: Lookback period for high/low
            volatility_factor: Multiplier for volatility threshold
            shares: Number of shares to trade
        """
        super().__init__(name=f"VolBreak_{period}")
        self.period = period
        self.volatility_factor = volatility_factor
        self.shares = shares

    def predict(self, data: pd.DataFrame, trade_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate signal based on volatility breakout.

        Args:
            data: Historical OHLCV data
            trade_history: Past trades

        Returns:
            Dictionary with signal, size, and order_ids
        """
        if len(data) < self.period:
            return {"signal": "hold", "size": 0, "order_ids": None}

        recent_data = data.tail(self.period)
        high = recent_data["High"].max()
        low = recent_data["Low"].min()
        volatility = recent_data["Close"].std()

        current_price = data["Close"].iloc[-1]

        # Calculate breakout thresholds
        upper_threshold = high + (self.volatility_factor * volatility)
        lower_threshold = low - (self.volatility_factor * volatility)

        # Determine position
        has_position = False
        if trade_history:
            last_buys = sum(t["shares"] for t in trade_history if t["signal"] == "buy")
            last_sells = sum(t["shares"] for t in trade_history if t["signal"] == "sell")
            has_position = last_buys > last_sells

        # Generate signal
        if current_price > upper_threshold and not has_position:
            # Upside breakout - buy
            return {"signal": "buy", "size": self.shares, "order_ids": None}

        elif current_price < lower_threshold and has_position:
            # Downside breakout - sell
            return {"signal": "sell", "size": self.shares, "order_ids": None}

        else:
            return {"signal": "hold", "size": 0, "order_ids": None}
