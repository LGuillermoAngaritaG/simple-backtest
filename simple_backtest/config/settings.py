"""Backtest configuration with Pydantic validation."""

from datetime import datetime
from typing import List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class BacktestConfig(BaseModel):
    """Backtest configuration with validation.

    This framework is asset-agnostic and supports:
    - Stocks (fractional or whole shares)
    - Forex/Currencies (fractional units)
    - Futures contracts
    - Crypto (fractional coins/tokens)
    - Any other tradable asset

    Note: Throughout the codebase, "shares" refers to generic "units" or "quantity"
    of any asset type (e.g., 1.5 BTC, 100.25 EUR/USD lots, 0.5 shares of AAPL).
    """

    initial_capital: float = Field(
        default=1000.0, gt=0, description="Starting capital (in base currency)"
    )
    lookback_period: int = Field(default=30, ge=1, description="Historical ticks for context")
    commission_type: Literal["percentage", "flat", "tiered", "custom"] = Field(
        default="percentage", description="Commission calculation method"
    )
    commission_value: Union[float, List[Tuple[float, float]]] = Field(
        default=0.001,
        description="Commission value: float for percentage/flat, list of (threshold, rate) tuples for tiered",
    )
    execution_price: Literal["open", "close", "vwap", "custom"] = Field(
        default="open", description="Price for trade execution"
    )
    trading_start_date: Optional[datetime] = Field(default=None, description="Trading period start")
    trading_end_date: Optional[datetime] = Field(default=None, description="Trading period end")
    enable_caching: bool = Field(default=True, description="Enable caching")
    parallel_execution: bool = Field(default=True, description="Parallel strategy execution")
    n_jobs: int = Field(default=-1, description="Number of parallel jobs (-1 = all cores)")
    risk_free_rate: float = Field(default=0.0, ge=0, le=1, description="Annual risk-free rate")
    asset_type: Optional[str] = Field(
        default=None,
        description="Asset type for documentation (e.g., 'stock', 'forex', 'crypto', 'futures'). Optional, for clarity only.",
    )

    @field_validator("n_jobs")
    @classmethod
    def validate_n_jobs(cls, v: int) -> int:
        """Validate n_jobs is -1 or positive."""
        if v != -1 and v < 1:
            raise ValueError("n_jobs must be -1 (all cores) or a positive integer")
        return v

    @model_validator(mode="after")
    def validate_date_range(self) -> "BacktestConfig":
        """Validate date range logic."""
        if self.trading_start_date and self.trading_end_date:
            if self.trading_start_date >= self.trading_end_date:
                raise ValueError(
                    f"trading_start_date ({self.trading_start_date}) must be before "
                    f"trading_end_date ({self.trading_end_date})"
                )
        return self

    @model_validator(mode="after")
    def validate_commission(self) -> "BacktestConfig":
        """Validate commission_value matches commission_type."""
        if self.commission_type == "tiered":
            # For tiered, must be a list of tuples
            if not isinstance(self.commission_value, list):
                raise ValueError(
                    "commission_value must be a list of (threshold, rate) tuples when commission_type='tiered'"
                )

            if len(self.commission_value) == 0:
                raise ValueError("commission_value list cannot be empty for tiered commission")

            # Validate each tier
            prev_threshold = 0.0
            for i, tier in enumerate(self.commission_value):
                if not isinstance(tier, tuple) or len(tier) != 2:
                    raise ValueError(f"Tier {i} must be a (threshold, rate) tuple, got {tier}")

                threshold, rate = tier

                if not isinstance(threshold, (int, float)) or not isinstance(rate, (int, float)):
                    raise ValueError(
                        f"Tier {i}: threshold and rate must be numeric, got ({threshold}, {rate})"
                    )

                if threshold <= prev_threshold and threshold != float("inf"):
                    raise ValueError(
                        f"Tier {i}: thresholds must be in ascending order. "
                        f"Got {threshold} after {prev_threshold}"
                    )

                if rate < 0:
                    raise ValueError(f"Tier {i}: rate must be non-negative, got {rate}")

                prev_threshold = threshold

        else:
            # For non-tiered, must be a float >= 0
            if not isinstance(self.commission_value, (int, float)):
                raise ValueError(
                    f"commission_value must be a number when commission_type='{self.commission_type}', "
                    f"got {type(self.commission_value).__name__}"
                )

            if self.commission_value < 0:
                raise ValueError(
                    f"commission_value must be non-negative, got {self.commission_value}"
                )

        return self

    def validate_against_data(
        self, data_start: datetime, data_end: datetime, total_rows: int
    ) -> None:
        """Validate config against data constraints.

        :param data_start: First date in dataset
        :param data_end: Last date in dataset
        :param total_rows: Total rows in dataset
        """
        # Check lookback period fits in data
        if self.lookback_period >= total_rows:
            raise ValueError(
                f"lookback_period ({self.lookback_period}) must be less than "
                f"total data rows ({total_rows})"
            )

        # Validate trading start date
        if self.trading_start_date:
            if self.trading_start_date < data_start:
                raise ValueError(
                    f"trading_start_date ({self.trading_start_date}) is before "
                    f"data start ({data_start})"
                )

        # Validate trading end date
        if self.trading_end_date:
            if self.trading_end_date > data_end:
                raise ValueError(
                    f"trading_end_date ({self.trading_end_date}) is after data end ({data_end})"
                )

        # Ensure enough data for lookback before trading starts
        if self.trading_start_date is not None:
            # Only validate if explicitly set by user
            if self.trading_start_date == data_start:
                raise ValueError(
                    f"trading_start_date cannot equal data start. "
                    f"Need at least {self.lookback_period} rows before trading_start_date. "
                    f"Either set trading_start_date later or reduce lookback_period."
                )
        # If trading_start_date is None, it will be automatically set to index[lookback_period]
        # in _setup_trading_range(), which is always valid

    @classmethod
    def default(cls, **overrides) -> "BacktestConfig":
        """Create config with sensible defaults.

        :param overrides: Any config parameters to override
        :return: BacktestConfig with defaults applied

        Example:
            config = BacktestConfig.default(initial_capital=10000)
        """
        defaults = {
            "initial_capital": 10000.0,
            "lookback_period": 30,
            "commission_type": "percentage",
            "commission_value": 0.001,  # 0.1%
            "execution_price": "open",
            "parallel_execution": True,
            "n_jobs": -1,
            "risk_free_rate": 0.02,  # 2% annual
        }
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def zero_commission(cls, **overrides) -> "BacktestConfig":
        """Create config for testing without commission costs.

        Useful for debugging strategy logic without commission complexity.

        :param overrides: Any config parameters to override
        :return: BacktestConfig with zero commission

        Example:
            config = BacktestConfig.zero_commission(initial_capital=10000)
        """
        return cls.default(commission_type="flat", commission_value=0.0, **overrides)

    @classmethod
    def high_frequency(cls, **overrides) -> "BacktestConfig":
        """Create config for high-frequency trading strategies.

        Short lookback period, flat commission, VWAP execution.

        :param overrides: Any config parameters to override
        :return: BacktestConfig optimized for HFT

        Example:
            config = BacktestConfig.high_frequency(initial_capital=100000)
        """
        return cls.default(
            lookback_period=5,
            commission_type="flat",
            commission_value=1.0,  # $1 per trade
            execution_price="vwap",
            **overrides,
        )

    @classmethod
    def low_commission(cls, **overrides) -> "BacktestConfig":
        """Create config for discount brokers with low commission.

        Very low percentage commission (0.01%).

        :param overrides: Any config parameters to override
        :return: BacktestConfig with low commission

        Example:
            config = BacktestConfig.low_commission(initial_capital=10000)
        """
        return cls.default(
            commission_type="percentage",
            commission_value=0.0001,  # 0.01%
            **overrides,
        )

    @classmethod
    def swing_trading(cls, **overrides) -> "BacktestConfig":
        """Create config for swing trading strategies.

        Longer lookback period, typical retail commission.

        :param overrides: Any config parameters to override
        :return: BacktestConfig optimized for swing trading

        Example:
            config = BacktestConfig.swing_trading(initial_capital=10000)
        """
        return cls.default(
            lookback_period=100,
            commission_type="percentage",
            commission_value=0.001,  # 0.1%
            execution_price="close",
            **overrides,
        )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "initial_capital": 10000.0,
                "lookback_period": 50,
                "commission_type": "percentage",
                "commission_value": 0.001,
                "execution_price": "open",
                "trading_start_date": "2020-01-01T00:00:00",
                "trading_end_date": "2023-12-31T23:59:59",
                "enable_caching": True,
                "parallel_execution": True,
                "n_jobs": -1,
                "risk_free_rate": 0.02,
            }
        }
    )
