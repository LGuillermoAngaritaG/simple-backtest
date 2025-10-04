# Notebooks

This folder contains Jupyter notebooks demonstrating various trading strategies using the simple-backtest framework.

## Setup

Install the required dependencies:

```bash
# Install with dev dependencies (includes yfinance and jupyter)
uv sync --all-extras

# Or with pip
pip install -e ".[dev]"
```

## Notebooks

### 01_basic_usage.ipynb
Introduction to the framework using built-in strategies on Apple (AAPL) stock:
- Buy and Hold strategy
- Moving Average crossover strategy
- Running multiple strategies in parallel
- Analyzing trade history

### 02_rsi_strategy.ipynb
RSI (Relative Strength Index) momentum strategy on Tesla (TSLA):
- RSI indicator calculation
- Oversold/overbought signals
- Parameter optimization
- Trade distribution analysis

### 03_bollinger_bands_strategy.ipynb
Bollinger Bands mean reversion strategy on Microsoft (MSFT):
- Bollinger Bands calculation and visualization
- Mean reversion signals
- Parameter tuning
- Risk analysis with drawdown and returns distribution

### 04_strategy_comparison.ipynb
Comprehensive comparison of all strategies on S&P 500 (SPY):
- Side-by-side performance metrics
- Risk-return profiles
- Drawdown comparison
- Monthly returns heatmap
- Summary statistics

## Running the Notebooks

```bash
# Start Jupyter
jupyter notebook

# Or use JupyterLab
jupyter lab
```

Then navigate to the `notebooks/` folder and open any notebook.

## Note

The RSI and Bollinger Bands strategies demonstrated in these notebooks are for educational purposes only. They are not included in the main library but show how easy it is to implement custom strategies by inheriting from the `Strategy` base class.
