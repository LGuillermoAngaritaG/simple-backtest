"""Interactive visualization suite using Plotly."""

from typing import Any, Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Color palette for consistent styling
COLORS = [
    "#1f77b4",  # Blue
    "#ff7f0e",  # Orange
    "#2ca02c",  # Green
    "#d62728",  # Red
    "#9467bd",  # Purple
    "#8c564b",  # Brown
    "#e377c2",  # Pink
    "#7f7f7f",  # Gray
    "#bcbd22",  # Olive
    "#17becf",  # Cyan
]


def plot_equity_curve(results: Dict[str, Dict[str, Any]]) -> go.Figure:
    """Plot equity curves for all strategies and benchmark.

    Args:
        results: Dictionary mapping strategy names to result dictionaries

    Returns:
        Plotly Figure with equity curves
    """
    fig = go.Figure()

    for i, (name, result) in enumerate(results.items()):
        portfolio_values = result["portfolio_values"]
        color = COLORS[i % len(COLORS)]

        # Make benchmark dashed and gray
        if name == "benchmark":
            line_style = dict(color="gray", dash="dash", width=2)
        else:
            line_style = dict(color=color, width=2)

        fig.add_trace(
            go.Scatter(
                x=portfolio_values.index,
                y=portfolio_values.values,
                mode="lines",
                name=name,
                line=line_style,
                hovertemplate="<b>%{fullData.name}</b><br>"
                + "Date: %{x}<br>"
                + "Value: $%{y:,.2f}<br>"
                + "<extra></extra>",
            )
        )

    fig.update_layout(
        title="Portfolio Equity Curve",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode="x unified",
        template="plotly_white",
        height=500,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    return fig


def plot_drawdowns(results: Dict[str, Dict[str, Any]]) -> go.Figure:
    """Plot drawdown chart for all strategies.

    Args:
        results: Dictionary mapping strategy names to result dictionaries

    Returns:
        Plotly Figure with drawdown curves
    """
    fig = go.Figure()

    for i, (name, result) in enumerate(results.items()):
        portfolio_values = result["portfolio_values"]
        color = COLORS[i % len(COLORS)]

        # Calculate drawdown
        running_max = portfolio_values.cummax()
        drawdown = (portfolio_values - running_max) / running_max * 100

        # Make benchmark dashed
        if name == "benchmark":
            line_style = dict(color="gray", dash="dash", width=2)
            fill = None
        else:
            line_style = dict(color=color, width=2)
            fill = "tozeroy"

        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                mode="lines",
                name=name,
                line=line_style,
                fill=fill,
                fillcolor=color if name != "benchmark" else None,
                hovertemplate="<b>%{fullData.name}</b><br>"
                + "Date: %{x}<br>"
                + "Drawdown: %{y:.2f}%<br>"
                + "<extra></extra>",
            )
        )

    fig.update_layout(
        title="Drawdown Over Time",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        hovermode="x unified",
        template="plotly_white",
        height=400,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    return fig


def plot_returns_distribution(results: Dict[str, Dict[str, Any]]) -> go.Figure:
    """Plot returns distribution histograms.

    Args:
        results: Dictionary mapping strategy names to result dictionaries

    Returns:
        Plotly Figure with return distributions
    """
    fig = go.Figure()

    for i, (name, result) in enumerate(results.items()):
        returns = result["returns"] * 100  # Convert to percentage
        color = COLORS[i % len(COLORS)]

        fig.add_trace(
            go.Histogram(
                x=returns,
                name=name,
                marker_color=color,
                opacity=0.7,
                nbinsx=50,
                hovertemplate="<b>%{fullData.name}</b><br>"
                + "Return: %{x:.2f}%<br>"
                + "Count: %{y}<br>"
                + "<extra></extra>",
            )
        )

    fig.update_layout(
        title="Returns Distribution",
        xaxis_title="Daily Return (%)",
        yaxis_title="Frequency",
        template="plotly_white",
        height=400,
        barmode="overlay",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )

    return fig


def plot_monthly_returns(results: Dict[str, Dict[str, Any]]) -> Dict[str, go.Figure]:
    """Plot monthly returns heatmap for each strategy.

    Args:
        results: Dictionary mapping strategy names to result dictionaries

    Returns:
        Dictionary mapping strategy names to their heatmap figures
    """
    figures = {}

    for name, result in results.items():
        if name == "benchmark":
            continue  # Skip benchmark

        returns = result["returns"]

        # Resample to monthly returns
        monthly_returns = (1 + returns).resample("M").prod() - 1
        monthly_returns = monthly_returns * 100  # Convert to percentage

        # Create year-month matrix
        df = pd.DataFrame({"return": monthly_returns})
        df["year"] = df.index.year
        df["month"] = df.index.month

        # Pivot to create heatmap data
        heatmap_data = df.pivot(index="year", columns="month", values="return")

        # Month names
        month_names = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]

        fig = go.Figure(
            data=go.Heatmap(
                z=heatmap_data.values,
                x=[month_names[i - 1] for i in heatmap_data.columns],
                y=heatmap_data.index,
                colorscale="RdYlGn",
                zmid=0,
                text=heatmap_data.values,
                texttemplate="%{text:.1f}%",
                textfont={"size": 10},
                hovertemplate="Year: %{y}<br>"
                + "Month: %{x}<br>"
                + "Return: %{z:.2f}%<br>"
                + "<extra></extra>",
            )
        )

        fig.update_layout(
            title=f"Monthly Returns Heatmap - {name}",
            xaxis_title="Month",
            yaxis_title="Year",
            template="plotly_white",
            height=400,
        )

        figures[name] = fig

    return figures


def plot_trades(results: Dict[str, Dict[str, Any]]) -> Dict[str, go.Figure]:
    """Plot trade scatter for each strategy.

    Args:
        results: Dictionary mapping strategy names to result dictionaries

    Returns:
        Dictionary mapping strategy names to their trade scatter figures
    """
    figures = {}

    for name, result in results.items():
        if name == "benchmark":
            continue

        trade_history = result["trade_history"]
        sell_trades = [t for t in trade_history if t["signal"] == "sell"]

        if not sell_trades:
            continue

        dates = [t["timestamp"] for t in sell_trades]
        pnls = [t["pnl"] for t in sell_trades]
        colors_list = ["green" if pnl > 0 else "red" for pnl in pnls]

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=dates,
                y=pnls,
                mode="markers",
                marker=dict(
                    size=8,
                    color=colors_list,
                    line=dict(width=1, color="white"),
                ),
                name="Trades",
                hovertemplate="Date: %{x}<br>" + "P&L: $%{y:.2f}<br>" + "<extra></extra>",
            )
        )

        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

        fig.update_layout(
            title=f"Trade P&L Analysis - {name}",
            xaxis_title="Date",
            yaxis_title="P&L ($)",
            template="plotly_white",
            height=400,
        )

        figures[name] = fig

    return figures


def plot_rolling_metrics(results: Dict[str, Dict[str, Any]], window: int = 30) -> go.Figure:
    """Plot rolling Sharpe and Sortino ratios.

    Args:
        results: Dictionary mapping strategy names to result dictionaries
        window: Rolling window size

    Returns:
        Plotly Figure with rolling metrics
    """
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Rolling Sharpe Ratio", "Rolling Volatility"),
        vertical_spacing=0.15,
    )

    for i, (name, result) in enumerate(results.items()):
        returns = result["returns"]
        color = COLORS[i % len(COLORS)]

        # Rolling Sharpe
        rolling_sharpe = (returns.rolling(window).mean() / returns.rolling(window).std()) * np.sqrt(
            252
        )

        # Rolling Volatility
        rolling_vol = returns.rolling(window).std() * np.sqrt(252) * 100

        # Sharpe subplot
        fig.add_trace(
            go.Scatter(
                x=rolling_sharpe.index,
                y=rolling_sharpe.values,
                mode="lines",
                name=f"{name} (Sharpe)",
                line=dict(color=color, width=2),
                legendgroup=name,
            ),
            row=1,
            col=1,
        )

        # Volatility subplot
        fig.add_trace(
            go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol.values,
                mode="lines",
                name=f"{name} (Vol)",
                line=dict(color=color, width=2),
                legendgroup=name,
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=1)
    fig.update_yaxes(title_text="Volatility (%)", row=2, col=1)

    fig.update_layout(
        title=f"Rolling Metrics ({window}-day window)",
        template="plotly_white",
        height=700,
        hovermode="x unified",
    )

    return fig


def create_comparison_table(results: Dict[str, Dict[str, Any]]) -> go.Figure:
    """Create comparison table for all strategies.

    Args:
        results: Dictionary mapping strategy names to result dictionaries

    Returns:
        Plotly Figure with comparison table
    """
    # Extract metrics
    metric_names = [
        "total_return",
        "cagr",
        "volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "win_rate",
        "profit_factor",
        "total_trades",
    ]

    metric_labels = [
        "Total Return (%)",
        "CAGR (%)",
        "Volatility (%)",
        "Sharpe Ratio",
        "Sortino Ratio",
        "Max Drawdown (%)",
        "Win Rate (%)",
        "Profit Factor",
        "Total Trades",
    ]

    # Build table data
    header = ["Metric"] + list(results.keys())
    cells = []

    for label, metric_key in zip(metric_labels, metric_names):
        row = [label]
        for name in results.keys():
            value = results[name]["metrics"].get(metric_key, 0)
            # Format based on metric type
            if metric_key == "total_trades":
                row.append(f"{int(value)}")
            else:
                row.append(f"{value:.2f}")
        cells.append(row)

    # Transpose for table format
    table_data = list(zip(*cells))

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=header,
                    fill_color="lightblue",
                    align="left",
                    font=dict(size=12, color="black"),
                ),
                cells=dict(
                    values=table_data,
                    fill_color="white",
                    align="left",
                    font=dict(size=11),
                ),
            )
        ]
    )

    fig.update_layout(title="Strategy Comparison", template="plotly_white", height=400)

    return fig


def plot_all(results: Dict[str, Dict[str, Any]]) -> None:
    """Display all plots in sequence.

    Args:
        results: Dictionary mapping strategy names to result dictionaries
    """
    # Equity curve
    fig = plot_equity_curve(results)
    fig.show()

    # Drawdowns
    fig = plot_drawdowns(results)
    fig.show()

    # Returns distribution
    fig = plot_returns_distribution(results)
    fig.show()

    # Monthly returns heatmaps
    monthly_figs = plot_monthly_returns(results)
    for fig in monthly_figs.values():
        fig.show()

    # Trade analysis
    trade_figs = plot_trades(results)
    for fig in trade_figs.values():
        fig.show()

    # Rolling metrics
    fig = plot_rolling_metrics(results)
    fig.show()

    # Comparison table
    fig = create_comparison_table(results)
    fig.show()
