"""Demo script showing framework usage with example data."""

from datetime import datetime

import numpy as np
import pandas as pd

from simple_backtest.config.settings import BacktestConfig
from simple_backtest.core.backtest import Backtest
from simple_backtest.examples.advanced_strategy import (
    BollingerBandsStrategy,
    MomentumStrategy,
)
from simple_backtest.examples.simple_strategy import (
    RSIStrategy,
    SimpleMovingAverageStrategy,
)
from simple_backtest.metrics.calculator import format_metrics
from simple_backtest.visualization.plotter import (
    create_comparison_table,
    plot_all,
    plot_equity_curve,
)


def generate_sample_data(n_days: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate sample OHLCV data for testing.

    Args:
        n_days: Number of days of data
        seed: Random seed for reproducibility

    Returns:
        DataFrame with OHLCV data and DatetimeIndex
    """
    np.random.seed(seed)

    # Generate dates
    dates = pd.date_range(start="2020-01-01", periods=n_days, freq="D")

    # Generate price data with trend and noise
    base_price = 100.0
    trend = np.linspace(0, 20, n_days)
    noise = np.random.randn(n_days).cumsum() * 2

    close_prices = base_price + trend + noise

    # Generate OHLC from close
    high_prices = close_prices + np.random.uniform(0, 2, n_days)
    low_prices = close_prices - np.random.uniform(0, 2, n_days)
    open_prices = (
        close_prices + np.random.uniform(-1, 1, n_days)
    )

    # Generate volume
    volumes = np.random.randint(100000, 1000000, n_days)

    # Create DataFrame
    data = pd.DataFrame(
        {
            "Open": open_prices,
            "High": high_prices,
            "Low": low_prices,
            "Close": close_prices,
            "Volume": volumes,
        },
        index=dates,
    )

    return data


def main() -> None:
    """Run demo backtest."""
    print("=" * 60)
    print("Backtesting Framework Demo")
    print("=" * 60)
    print()

    # Generate sample data
    print("Generating sample data...")
    data = generate_sample_data(n_days=500)
    print(f"Generated {len(data)} days of OHLCV data")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    print()

    # Create configuration
    print("Setting up backtest configuration...")
    config = BacktestConfig(
        initial_capital=10000.0,
        lookback_period=50,
        commission_type="percentage",
        commission_value=0.001,  # 0.1% commission
        execution_price="open",
        trading_start_date=datetime(2020, 3, 1),
        trading_end_date=datetime(2021, 8, 31),
        enable_caching=False,
        parallel_execution=True,
        n_jobs=-1,
        risk_free_rate=0.02,
    )
    print("Configuration:")
    print(f"  Initial Capital: ${config.initial_capital:,.2f}")
    print(f"  Commission: {config.commission_value * 100}%")
    print(f"  Trading Period: {config.trading_start_date} to {config.trading_end_date}")
    print()

    # Create strategies
    print("Creating strategies...")
    strategies = [
        SimpleMovingAverageStrategy(short_window=10, long_window=30, shares=50),
        SimpleMovingAverageStrategy(short_window=20, long_window=50, shares=50),
        RSIStrategy(period=14, oversold=30, overbought=70, shares=50),
        BollingerBandsStrategy(period=20, num_std=2.0, shares=50),
        MomentumStrategy(lookback=10, threshold=0.02, shares=50),
    ]
    print(f"Created {len(strategies)} strategies:")
    for strategy in strategies:
        print(f"  - {strategy.get_name()}")
    print()

    # Run backtest
    print("Running backtest...")
    print("(This may take a moment with parallel execution)")
    print()

    backtest = Backtest(data=data, config=config)
    results = backtest.run(strategies)

    print("Backtest complete!")
    print()

    # Display results
    print("=" * 60)
    print("Results Summary")
    print("=" * 60)
    print()

    for name, result in results.items():
        print(f"\n{name}")
        print("-" * 60)
        metrics = result["metrics"]
        print(f"Total Return: {metrics['total_return']:.2f}%")
        print(f"CAGR: {metrics['cagr']:.2f}%")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
        print(f"Win Rate: {metrics['win_rate']:.2f}%")
        print(f"Total Trades: {metrics['total_trades']}")
        print()

    # Show detailed metrics for best strategy
    best_strategy = max(
        results.items(),
        key=lambda x: x[1]["metrics"]["sharpe_ratio"] if x[0] != "benchmark" else -999,
    )
    print("\n" + "=" * 60)
    print(f"Detailed Metrics for Best Strategy: {best_strategy[0]}")
    print("=" * 60)
    print(format_metrics(best_strategy[1]["metrics"]))
    print()

    # Visualizations
    print("=" * 60)
    print("Generating Visualizations")
    print("=" * 60)
    print()
    print("Opening interactive plots...")
    print("(Close plot windows to continue)")
    print()

    # Show equity curve
    fig = plot_equity_curve(results)
    fig.show()

    # Show comparison table
    fig = create_comparison_table(results)
    fig.show()

    # Optionally show all plots
    # Uncomment to see all visualizations:
    # plot_all(results)

    print("\nDemo complete!")
    print("\nTo see all visualizations, uncomment the plot_all() call in demo.py")


if __name__ == "__main__":
    main()
