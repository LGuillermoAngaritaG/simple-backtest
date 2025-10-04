# Simple Backtest

Simple, high-performance backtesting framework for trading strategies. I created this library so anyone can implement their own trading strategy with simple code, just create a new class that inherits from the Strategy class and implement the predict method and run the backtest on your data.

## Features

- 🚀 Parallel strategy execution
- 📊 20+ performance metrics (Sharpe, Sortino, Calmar, etc.)
- 📈 Interactive Plotly visualizations
- 🎯 Clean Strategy Pattern architecture
- 💰 Flexible commission models

## Installation

```bash
# Using uv (recommended)
uv pip install simple-backtest

# Or with pip
pip install simple-backtest
```

## Quick Start

```python
import pandas as pd
from simple_backtest import BacktestConfig, Backtest
from simple_backtest.strategy.moving_average import MovingAverageStrategy
from simple_backtest.visualization.plotter import plot_equity_curve

# Load OHLCV data
data = pd.read_csv("data.csv", index_col=0, parse_dates=True)

# Configure backtest
config = BacktestConfig(
    initial_capital=10000.0,
    lookback_period=50,
    commission_type="percentage",
    commission_value=0.001,
)

# Run backtest
strategy = MovingAverageStrategy(short_window=10, long_window=30, shares=100)
backtest = Backtest(data=data, config=config)
results = backtest.run([strategy])

# Visualize
plot_equity_curve(results).show()

# Print metrics
print(results[strategy.get_name()]['metrics'])
```

## Create Custom Strategy

```python
from simple_backtest import Strategy

class MyStrategy(Strategy):
    def predict(self, data, trade_history):
        """
        :param data: OHLCV DataFrame
        :param trade_history: List of past trades
        :return: Dict with signal, size, order_ids
        """
        if data['Close'].iloc[-1] < 100:
            return {"signal": "buy", "size": 10, "order_ids": None}
        elif data['Close'].iloc[-1] > 120:
            return {"signal": "sell", "size": 10, "order_ids": None}
        return {"signal": "hold", "size": 0, "order_ids": None}
```

## Notebooks

Check out the `notebooks/` folder for interactive examples:
- **01_basic_usage.ipynb**: Introduction to the framework
- **02_rsi_strategy.ipynb**: RSI momentum strategy
- **03_bollinger_bands_strategy.ipynb**: Mean reversion with Bollinger Bands
- **04_strategy_comparison.ipynb**: Compare multiple strategies

To run the notebooks:
```bash
# Install with dev dependencies
uv sync --all-extras

# Start Jupyter
jupyter notebook
```

## Development

```bash
# Clone repo
git clone <repo-url>
cd simple-backtest

# Install with uv
uv sync --all-extras

# Run tests
uv run pytest

# Lint
uv run ruff check simple_backtest
```

## Metrics

- **Returns**: Total Return, CAGR
- **Risk**: Volatility, Sharpe, Sortino, Calmar, Max Drawdown
- **Trades**: Win Rate, Profit Factor, Expectancy
- **Benchmark**: Alpha, Beta, Information Ratio

## Built-in Strategies

- **Buy and Hold**: Simple baseline strategy
- **Moving Average Crossover**: Trade on MA crossovers

See `notebooks/` for more strategy examples (RSI, Bollinger Bands, etc.).

## License

MIT
