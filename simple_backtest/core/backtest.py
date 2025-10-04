"""Backtesting engine with parallelization support."""

from datetime import datetime
from typing import Any, Callable, Dict, List

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from simple_backtest.config.settings import BacktestConfig
from simple_backtest.core.portfolio import Portfolio
from simple_backtest.strategy.strategy_base import Strategy
from simple_backtest.metrics.calculator import calculate_metrics
from simple_backtest.utils.commission import get_commission_calculator
from simple_backtest.utils.execution import create_execution_price_extractor
from simple_backtest.utils.validation import (
    validate_dataframe,
    validate_date_range,
    validate_strategies,
)


class Backtest:
    """Backtesting engine with parallel strategy execution support."""

    def __init__(
        self,
        data: pd.DataFrame,
        config: BacktestConfig,
        commission_calculator: Callable[[float, float], float] | None = None,
    ):
        """Initialize backtest engine.

        :param data: OHLCV DataFrame with DatetimeIndex
        :param config: Backtest configuration
        :param commission_calculator: Custom commission function (uses config if None)
        """
        # Validate data
        validate_dataframe(data, strict=True)

        # Store data and config
        self.data = data.copy()
        self.config = config

        # Validate config against data
        if isinstance(self.data.index, pd.DatetimeIndex):
            config.validate_against_data(
                data_start=self.data.index[0],
                data_end=self.data.index[-1],
                total_rows=len(self.data),
            )

        # Determine trading range
        self._setup_trading_range()

        # Validate date range
        validate_date_range(
            self.data,
            self.trading_start_date,
            self.trading_end_date,
            self.config.lookback_period,
        )

        # Setup commission calculator
        if commission_calculator is None:
            self.commission_calculator = get_commission_calculator(config)
        else:
            self.commission_calculator = commission_calculator

        # Setup execution price extractor
        self.price_extractor = create_execution_price_extractor(method=config.execution_price)

    def _setup_trading_range(self) -> None:
        """Set trading date range from config and data."""
        if self.config.trading_start_date is None:
            # Start after lookback period
            start_idx = self.config.lookback_period
            self.trading_start_date = self.data.index[start_idx]
        else:
            self.trading_start_date = self.config.trading_start_date

        if self.config.trading_end_date is None:
            self.trading_end_date = self.data.index[-1]
        else:
            self.trading_end_date = self.config.trading_end_date

        # Get trading data slice
        self.trading_data = self.data.loc[self.trading_start_date : self.trading_end_date]

    def run(self, strategies: List[Strategy]) -> Dict[str, Dict[str, Any]]:
        """Run backtest for all strategies.

        :param strategies: List of strategies to backtest
        :return: Dict mapping strategy names to results (metrics, portfolio_values, trade_history, returns)
        """
        # Validate strategies
        validate_strategies(strategies)

        # Create benchmark
        benchmark_results = self._run_benchmark()

        # Reset strategies
        for strategy in strategies:
            strategy.reset_state()

        # Run strategies
        if self.config.parallel_execution and len(strategies) > 1:
            # Parallel execution
            n_jobs = self.config.n_jobs if self.config.n_jobs != -1 else -1
            strategy_results = Parallel(n_jobs=n_jobs)(
                delayed(self._run_single_strategy)(strategy) for strategy in strategies
            )
        else:
            # Sequential execution
            strategy_results = []
            for strategy in tqdm(strategies, desc="Running strategies"):
                strategy_results.append(self._run_single_strategy(strategy))

        # Combine results
        results = {"benchmark": benchmark_results}
        for strategy, result in zip(strategies, strategy_results):
            results[strategy.get_name()] = result

        return results

    def _run_single_strategy(self, strategy: Strategy) -> Dict[str, Any]:
        """Run backtest for single strategy.

        :param strategy: Strategy to backtest
        :return: Results dict with metrics, portfolio_values, trade_history, returns
        """
        # Create portfolio
        portfolio = Portfolio(self.config.initial_capital)

        # Track portfolio values over time
        portfolio_values = []
        timestamps = []

        # Get trading date range
        start_idx = self.data.index.get_indexer([self.trading_start_date], method="nearest")[0]
        end_idx = self.data.index.get_indexer([self.trading_end_date], method="nearest")[0]

        # Progress bar (only for non-parallel execution)
        iterator = range(start_idx, end_idx + 1)
        if not self.config.parallel_execution:
            iterator = tqdm(
                iterator,
                desc=f"Backtesting {strategy.get_name()}",
                leave=False,
            )

        # Main backtest loop
        for i in iterator:
            current_date = self.data.index[i]
            current_row = self.data.iloc[i]

            # Extract lookback window
            lookback_start = max(0, i - self.config.lookback_period)
            lookback_data = self.data.iloc[lookback_start:i]

            # Get current price for portfolio valuation
            current_price = self.price_extractor(current_row)

            # Record portfolio value before trading
            portfolio_value = portfolio.get_portfolio_value(current_price)
            portfolio_values.append(portfolio_value)
            timestamps.append(current_date)

            # Skip if not enough lookback data
            if len(lookback_data) < self.config.lookback_period:
                continue

            # Get strategy prediction
            try:
                prediction = strategy.predict(lookback_data, portfolio.get_trade_history())
                strategy.validate_prediction(prediction)
            except Exception as e:
                # Log error and continue
                print(f"Strategy {strategy.get_name()} error at {current_date}: {e}")
                continue

            signal = prediction["signal"]
            size = prediction["size"]

            # Execute trade based on signal
            if signal == "buy" and size > 0:
                commission = self.commission_calculator(size, current_price)

                # Check if can afford
                if portfolio.can_afford(size, current_price, commission):
                    try:
                        trade_info = portfolio.execute_buy(
                            shares=size,
                            price=current_price,
                            commission=commission,
                            timestamp=current_date,
                        )
                        strategy.on_trade_executed(trade_info)
                    except Exception as e:
                        print(f"Buy error for {strategy.get_name()} at {current_date}: {e}")

            elif signal == "sell" and size > 0:
                total_shares = portfolio.get_total_shares()

                # Check if have shares to sell
                if total_shares >= size:
                    commission = self.commission_calculator(size, current_price)
                    order_ids = prediction.get("order_ids")

                    try:
                        trade_info = portfolio.execute_sell(
                            shares=size,
                            price=current_price,
                            commission=commission,
                            timestamp=current_date,
                            order_ids=order_ids,
                        )
                        strategy.on_trade_executed(trade_info)
                    except Exception as e:
                        print(f"Sell error for {strategy.get_name()} at {current_date}: {e}")

        # Create portfolio values series
        portfolio_series = pd.Series(portfolio_values, index=timestamps)

        # Calculate returns
        returns = portfolio_series.pct_change().dropna()

        # Get benchmark values for same period
        benchmark_values = self._get_benchmark_values_for_period(timestamps)

        # Calculate metrics
        metrics = calculate_metrics(
            trade_history=portfolio.get_trade_history(),
            portfolio_values=portfolio_series,
            benchmark_values=benchmark_values,
            initial_capital=self.config.initial_capital,
            risk_free_rate=self.config.risk_free_rate,
        )

        return {
            "metrics": metrics,
            "portfolio_values": portfolio_series,
            "trade_history": portfolio.get_trade_history(),
            "returns": returns,
        }

    def _run_benchmark(self) -> Dict[str, Any]:
        """Run buy-and-hold benchmark."""
        # Create portfolio
        portfolio = Portfolio(self.config.initial_capital)

        # Get first trading date
        start_idx = self.data.index.get_indexer([self.trading_start_date], method="nearest")[0]
        end_idx = self.data.index.get_indexer([self.trading_end_date], method="nearest")[0]

        first_date = self.data.index[start_idx]
        first_row = self.data.iloc[start_idx]
        first_price = self.price_extractor(first_row)

        # Buy as many shares as possible
        commission = self.commission_calculator(1, first_price)
        max_shares = (self.config.initial_capital - commission) / first_price

        if max_shares > 0:
            portfolio.execute_buy(
                shares=max_shares,
                price=first_price,
                commission=commission,
                timestamp=first_date,
            )

        # Track portfolio values
        portfolio_values = []
        timestamps = []

        for i in range(start_idx, end_idx + 1):
            current_date = self.data.index[i]
            current_row = self.data.iloc[i]
            current_price = self.price_extractor(current_row)

            portfolio_value = portfolio.get_portfolio_value(current_price)
            portfolio_values.append(portfolio_value)
            timestamps.append(current_date)

        # Create series
        portfolio_series = pd.Series(portfolio_values, index=timestamps)
        returns = portfolio_series.pct_change().dropna()

        # Benchmark metrics (compare to itself, so alpha/beta will be 1/1)
        metrics = calculate_metrics(
            trade_history=portfolio.get_trade_history(),
            portfolio_values=portfolio_series,
            benchmark_values=portfolio_series,  # Compare to itself
            initial_capital=self.config.initial_capital,
            risk_free_rate=self.config.risk_free_rate,
        )

        return {
            "metrics": metrics,
            "portfolio_values": portfolio_series,
            "trade_history": portfolio.get_trade_history(),
            "returns": returns,
        }

    def _get_benchmark_values_for_period(self, timestamps: List[datetime]) -> pd.Series:
        """Get benchmark values for given timestamps.

        :param timestamps: Timestamps to get values for
        :return: Series of benchmark values
        """
        # This assumes benchmark has already been run
        # In practice, we'd cache the benchmark results
        # For now, return a simple buy-and-hold approximation
        start_price = self.price_extractor(self.data.loc[timestamps[0]])
        benchmark_values = []

        for ts in timestamps:
            current_price = self.price_extractor(self.data.loc[ts])
            # Simple buy-and-hold value
            value = self.config.initial_capital * (current_price / start_price)
            benchmark_values.append(value)

        return pd.Series(benchmark_values, index=timestamps)
