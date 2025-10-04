"""Backtest configuration with Pydantic validation."""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class BacktestConfig(BaseModel):
    """Backtest configuration with validation."""

    initial_capital: float = Field(default=1000.0, gt=0, description="Starting capital")
    lookback_period: int = Field(default=30, ge=1, description="Historical ticks for context")
    commission_type: Literal["percentage", "flat", "tiered", "custom"] = Field(
        default="percentage", description="Commission calculation method"
    )
    commission_value: float = Field(
        default=0.001, ge=0, description="Commission value (depends on type)"
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
        effective_start = self.trading_start_date or data_start
        if effective_start == data_start:
            raise ValueError(
                f"Need at least {self.lookback_period} rows before trading_start_date. "
                f"Either set trading_start_date later or reduce lookback_period."
            )

    class Config:
        """Pydantic config."""

        json_schema_extra = {
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
