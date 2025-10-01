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
from datetime import datetime
import pandas as pd
from simple_backtest import BacktestConfig, Backtest
from simple_backtest.examples.simple_strategy import SimpleMovingAverageStrategy
from simple_backtest.visualization import plot_equity_curve

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
strategy = SimpleMovingAverageStrategy(short_window=10, long_window=30, shares=100)
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

## Development

```bash
# Clone repo
git clone <repo-url>
cd back-test

# Install with uv
uv sync --all-extras

# Run tests
uv run pytest

# Run demo
uv run python -m simple_backtest.examples.demo

# Lint
uv run ruff check simple_backtest
```

## Metrics

- **Returns**: Total Return, CAGR
- **Risk**: Volatility, Sharpe, Sortino, Calmar, Max Drawdown
- **Trades**: Win Rate, Profit Factor, Expectancy
- **Benchmark**: Alpha, Beta, Information Ratio

## Example Strategies

- Simple Moving Average
- RSI
- Bollinger Bands
- Momentum
- MACD
- Volatility Breakout

See `simple_backtest/examples/` for implementations.

## License

MIT
