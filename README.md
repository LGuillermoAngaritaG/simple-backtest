<div align="center">

# Simple Backtest

**A high-performance backtesting framework for trading strategies**

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Test Coverage](https://img.shields.io/badge/coverage-31%25-orange.svg)](https://github.com/yourusername/simple-backtest)
[![Tests](https://img.shields.io/badge/tests-39%20passed-success.svg)](https://github.com/yourusername/simple-backtest)

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [Examples](#-examples)

</div>

---

## 📖 About

Simple Backtest is a Python framework designed to make backtesting trading strategies straightforward and accessible. Whether you're testing a simple moving average crossover or a complex machine learning model, Simple Backtest provides the tools you need.

**Key Philosophy**: Inherit from `Strategy`, implement `predict()`, and let the framework handle the rest.

## ✨ Features

<table>
<tr>
<td width="50%">

### 🚀 Performance
- **Parallel Execution**: Test multiple strategies simultaneously
- **Optimized Core**: Fast backtesting engine with efficient portfolio tracking
- **Caching Support**: Speed up repeated backtests

</td>
<td width="50%">

### 📊 Analytics
- **20+ Metrics**: Sharpe, Sortino, Calmar, Win Rate, etc.
- **Benchmark Comparison**: Alpha, Beta, Information Ratio
- **Interactive Visualizations**: Plotly-powered charts

</td>
</tr>
<tr>
<td width="50%">

### 🎯 Design
- **Clean Architecture**: Strategy Pattern for extensibility
- **Type Safety**: Pydantic validation for configurations
- **Asset Agnostic**: Stocks, forex, crypto, futures, commodities

</td>
<td width="50%">

### 🔧 Flexibility
- **Custom Strategies**: Easy inheritance model
- **Commission Models**: Percentage, flat, tiered, custom
- **Parameter Optimization**: Grid search, random search, walk-forward

</td>
</tr>
</table>

### Supported Assets

Works with any asset providing OHLC(V) price data:

| Asset Type | Support | Notes |
|------------|---------|-------|
| 📈 **Stocks** | ✅ Full | Fractional or whole shares |
| 💱 **Forex** | ✅ Full | Volume optional |
| ₿ **Crypto** | ✅ Full | Fractional units supported |
| 📊 **ETFs** | ✅ Full | Same as stocks |
| 🛢️ **Commodities** | ✅ Full | Gold, oil, etc. |
| 📉 **Futures** | ⚠️ Partial | No margin/leverage modeling |
| 📊 **Options** | ❌ No | Requires Greeks, strikes, expiration |

## 📦 Installation

```bash
# Using pip
pip install simple-backtest

# Using uv (recommended)
uv add simple-backtest

# From source
git clone https://github.com/yourusername/simple-backtest.git
cd simple-backtest
uv sync --all-extras
```

**Requirements**: Python 3.10+

## 🚀 Quick Start

Get up and running in 3 simple steps:

```python
# 1. Get data (using yfinance for demo, but you can use any other data source)
import yfinance as yf
data = yf.download("AAPL", start="2020-01-01", end="2023-12-31")

# 2. Create strategy (you can use a basic one or create your own)
from simple_backtest import Backtest, BacktestConfig, MovingAverageStrategy

strategy = MovingAverageStrategy(short_window=10, long_window=30, shares=10)

# 3. Run backtest
config = BacktestConfig.default(initial_capital=10000)
backtest = Backtest(data, config)
results = backtest.run([strategy])

# View results
print(results.get_strategy(strategy.get_name()).summary())
```

**Output:**
```
Total Return: 227.91%
CAGR: 36.84%
Sharpe Ratio: 1.09
Max Drawdown: -30.60%
Win Rate: 100.00%
```

## 📚 Documentation

### Creating a Custom Strategy

Implement your own strategy by inheriting from `Strategy` and defining the `predict()` method:

```python
from simple_backtest import Strategy

class MyStrategy(Strategy):
    """Custom trading strategy."""

    def __init__(self, threshold=100, name=None):
        super().__init__(name=name or "MyStrategy")
        self.threshold = threshold

    def predict(self, data, trade_history):
        """Generate trading signal.

        Args:
            data: OHLCV DataFrame with lookback window
            trade_history: List of past trades

        Returns:
            Dict with keys: signal ("buy"/"hold"/"sell"), size, order_ids
        """
        current_price = data['Close'].iloc[-1]

        # Simple logic: buy below threshold, sell above
        if current_price < self.threshold and not self.has_position():
            return self.buy(10)  # Buy 10 shares
        elif current_price > self.threshold * 1.2 and self.has_position():
            return self.sell_all()  # Sell all positions
        else:
            return self.hold()  # Do nothing
```

**Strategy Helper Methods:**
- `self.has_position()` - Check if holding any shares
- `self.get_position()` - Get current share count
- `self.get_cash()` - Get available cash
- `self.get_portfolio_value()` - Get total portfolio value
- `self.buy(shares)` - Return buy signal
- `self.sell(shares)` - Return sell signal
- `self.sell_all()` - Sell all positions
- `self.buy_percent(percent)` - Buy shares worth % of portfolio
- `self.buy_cash(amount)` - Buy shares worth specific amount

### Configuration Presets

Quick configurations for common scenarios:

```python
from simple_backtest import BacktestConfig

# Zero commission (for testing)
config = BacktestConfig.zero_commission(initial_capital=10000)

# High-frequency trading (short lookback, flat commission, VWAP execution)
config = BacktestConfig.high_frequency(initial_capital=100000)

# Swing trading (longer lookback, typical retail commission)
config = BacktestConfig.swing_trading(initial_capital=10000)

# Low commission brokers (0.01% commission)
config = BacktestConfig.low_commission(initial_capital=10000)
```

### Comparing Multiple Strategies

```python
from simple_backtest import (
    Backtest,
    BacktestConfig,
    MovingAverageStrategy,
    BuyAndHoldStrategy,
    DCAStrategy
)

# Create strategies
strategies = [
    MovingAverageStrategy(short_window=10, long_window=30, shares=10),
    BuyAndHoldStrategy(shares=50),
    DCAStrategy(investment_amount=500, interval_days=30)
]

# Run backtest
config = BacktestConfig.default(initial_capital=10000)
backtest = Backtest(data, config)
results = backtest.run(strategies)

# Compare strategies
comparison = results.compare()
print(comparison)

# Get best strategy
best = results.best_strategy('sharpe_ratio')
print(f"Best: {best.name} (Sharpe: {best.metrics['sharpe_ratio']:.2f})")

# Visualize
results.plot_comparison().show()
```

### Parameter Optimization

Find optimal strategy parameters using built-in optimizers:

```python
from simple_backtest import GridSearchOptimizer, BacktestConfig

# Define parameter space
param_space = {
    'short_window': [5, 10, 15, 20],
    'long_window': [30, 40, 50, 60],
    'shares': [10]
}

# Run optimization
optimizer = GridSearchOptimizer(verbose=True)
results = optimizer.optimize(
    data=data,
    config=BacktestConfig.default(),
    strategy_class=MovingAverageStrategy,
    param_space=param_space,
    metric='sharpe_ratio'
)

# View top results
print(results.head(5))
```

**Available Optimizers:**
- `GridSearchOptimizer` - Exhaustive search (best for small spaces)
- `RandomSearchOptimizer` - Random sampling (faster for large spaces)
- `WalkForwardOptimizer` - Train/test split (prevents overfitting)

### Custom Commission Models

Create custom commission structures:

```python
from simple_backtest import Commission

class TieredWithMinimum(Commission):
    """Tiered commission with minimum fee."""

    def __init__(self):
        super().__init__(name="TieredMin")

    def calculate(self, shares, price):
        trade_value = shares * price

        if trade_value < 1000:
            commission = max(trade_value * 0.002, 1.0)  # 0.2%, min $1
        elif trade_value < 10000:
            commission = trade_value * 0.001  # 0.1%
        else:
            commission = trade_value * 0.0005  # 0.05%

        return commission

# Use in config
from simple_backtest import Portfolio
portfolio = Portfolio(10000)
portfolio.commission_calculator = TieredWithMinimum()
```

### Logging Control

Control framework verbosity:

```python
from simple_backtest.utils import setup_logging, disable_logging, enable_debug_logging
import logging

# Default: WARNING level (minimal output)

# For verbose output during optimization
setup_logging(level=logging.INFO)

# For debugging issues
enable_debug_logging()

# To suppress all output
disable_logging()
```

## 📊 Performance Metrics

The framework calculates 20+ metrics automatically:

### Returns
- Total Return (%)
- CAGR (Compound Annual Growth Rate)
- Annualized Return

### Risk Metrics
- Volatility (annualized standard deviation)
- Sharpe Ratio (risk-adjusted return)
- Sortino Ratio (downside risk-adjusted return)
- Calmar Ratio (return vs max drawdown)
- Max Drawdown (%)
- Max Drawdown Duration

### Trade Statistics
- Total Trades
- Win Rate (%)
- Profit Factor
- Average Trade P&L
- Trade Expectancy
- Average Win / Average Loss

### Benchmark Comparison
- Alpha (excess return vs benchmark)
- Beta (correlation with benchmark)
- Information Ratio
- Correlation with benchmark

## 📓 Examples

### Interactive Notebooks

Explore the `notebooks/` directory for comprehensive examples:

1. **01_basic_usage.ipynb** - Introduction, data loading, commission setup, strategy comparison
2. **02_candle_strategies.ipynb** - Candlestick patterns (Engulfing, Hammer, Doji, etc.)
3. **03_ta_strategies.ipynb** - Technical indicators (RSI, MACD, Bollinger Bands, etc.)
4. **04_ml_strategies.ipynb** - Machine learning strategies (Logistic Regression, Random Forest, XGBoost)
5. **05_commission_usage.ipynb** - Commission models comparison and custom implementations
6. **06_advanced_optimization.ipynb** - Grid search, random search, walk-forward optimization

Run notebooks:
```bash
# Install with notebook dependencies
uv sync --all-extras

# Start Jupyter
jupyter notebook
```

### Example Strategies

**RSI Mean Reversion:**
```python
class RSIStrategy(Strategy):
    def __init__(self, period=14, oversold=30, overbought=70, shares=10):
        super().__init__(name=f"RSI_{period}")
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.shares = shares

    def predict(self, data, trade_history):
        if len(data) < self.period:
            return self.hold()

        # Calculate RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        current_rsi = rsi.iloc[-1]

        if current_rsi < self.oversold and not self.has_position():
            return self.buy(self.shares)
        elif current_rsi > self.overbought and self.has_position():
            return self.sell_all()
        else:
            return self.hold()
```

**Dollar Cost Averaging (Built-in):**
```python
from simple_backtest import DCAStrategy

strategy = DCAStrategy(
    investment_amount=500,  # Invest $500 each time
    interval_days=30,       # Every 30 days
    name="DCA_Monthly"
)
```

## 🏗️ Architecture

```
simple_backtest/
├── core/              # Main backtesting logic
│   ├── backtest.py       # Backtest engine
│   ├── portfolio.py      # Portfolio management
│   └── results.py        # Results containers
├── strategy/          # Strategy implementations
│   ├── base.py           # Abstract Strategy class
│   ├── moving_average.py # MA crossover strategy
│   ├── buy_and_hold.py   # Buy & hold strategy
│   └── dca.py            # Dollar cost averaging
├── commission/        # Commission models
│   ├── base.py           # Abstract Commission class
│   ├── percentage.py     # Percentage commission
│   ├── flat.py           # Flat-rate commission
│   └── tiered.py         # Tiered commission
├── optimization/      # Parameter optimization
│   ├── base.py           # Abstract Optimizer class
│   ├── grid_search.py    # Grid search
│   ├── random_search.py  # Random search
│   └── walk_forward.py   # Walk-forward optimization
├── metrics/           # Performance metrics
│   ├── calculator.py     # Metrics calculation
│   └── definitions.py    # Individual metrics
├── visualization/     # Plotting
│   └── plotter.py        # Plotly charts
└── utils/             # Utilities
    ├── logger.py         # Logging configuration
    ├── validation.py     # Input validation
    └── execution.py      # Execution price extraction
```

## 🛠️ Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/yourusername/simple-backtest.git
cd simple-backtest

# Install with uv (recommended)
uv sync --all-extras

# Or with pip
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=simple_backtest

# Run specific test file
uv run pytest tests/test_strategy.py

# Run specific test
uv run pytest tests/test_strategy.py::test_strategy_initialization
```

### Code Quality

```bash
# Lint code
uv run ruff check simple_backtest

# Auto-fix linting issues
uv run ruff check simple_backtest --fix

# Format code
uv run ruff format simple_backtest

# Run pre-commit hooks
pre-commit run --all-files
```

### Pre-commit Hooks

Pre-commit hooks automatically run linting, formatting, and tests on commit:

```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## 🤝 Contributing

Contributions are welcome! Whether you're fixing bugs, adding features, or improving documentation, your help is appreciated.

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**
4. **Run tests**: `uv run pytest`
5. **Run linting**: `uv run ruff check simple_backtest`
6. **Commit your changes**: `git commit -m "Add amazing feature"`
7. **Push to branch**: `git push origin feature/amazing-feature`
8. **Open a Pull Request**

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Pydantic](https://docs.pydantic.dev/) for configuration validation
- Uses [Plotly](https://plotly.com/) for interactive visualizations
- Parallel processing with [Joblib](https://joblib.readthedocs.io/)
- Testing with [Pytest](https://docs.pytest.org/)
- Code quality with [Ruff](https://github.com/astral-sh/ruff)

## 📬 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/simple-backtest/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/simple-backtest/discussions)
- **Email**: guille2005_13@hotmail.com