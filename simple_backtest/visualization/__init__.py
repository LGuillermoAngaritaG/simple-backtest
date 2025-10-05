"""Visualization module."""

from simple_backtest.visualization.plotter import (
    create_comparison_table,
    plot_all,
    plot_drawdowns,
    plot_equity_curve,
    plot_monthly_returns,
    plot_returns_distribution,
    plot_rolling_metrics,
    plot_strategy_trades,
    plot_trades,
)

__all__ = [
    "plot_equity_curve",
    "plot_drawdowns",
    "plot_returns_distribution",
    "plot_monthly_returns",
    "plot_trades",
    "plot_strategy_trades",
    "plot_rolling_metrics",
    "create_comparison_table",
    "plot_all",
]
