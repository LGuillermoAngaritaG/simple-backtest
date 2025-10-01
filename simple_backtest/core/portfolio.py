"""Portfolio management for tracking positions, cash, and trade history."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4


class Portfolio:
    """Tracks portfolio state including cash, positions, and trade history.

    Positions are tracked with unique order IDs to support FIFO selling and
    accurate P&L calculation per trade.

    Attributes:
        initial_capital: Starting capital for the portfolio
        cash: Current available cash
        positions: Dict mapping order_id to position details
        trade_history: List of all executed trades
    """

    def __init__(self, initial_capital: float):
        """Initialize portfolio with starting capital.

        Args:
            initial_capital: Starting cash amount (must be positive)

        Raises:
            ValueError: If initial_capital is not positive
        """
        if initial_capital <= 0:
            raise ValueError(f"initial_capital must be positive, got {initial_capital}")

        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.trade_history: List[Dict[str, Any]] = []

    def get_total_shares(self) -> float:
        """Calculate total number of shares held across all positions.

        Returns:
            Total shares held
        """
        return sum(pos["shares"] for pos in self.positions.values())

    def get_portfolio_value(self, current_price: float) -> float:
        """Calculate total portfolio value (cash + position values).

        Args:
            current_price: Current market price per share

        Returns:
            Total portfolio value
        """
        position_value = self.get_total_shares() * current_price
        return self.cash + position_value

    def can_afford(self, shares: float, price: float, commission: float) -> bool:
        """Check if portfolio has sufficient cash for a purchase.

        Args:
            shares: Number of shares to buy
            price: Price per share
            commission: Commission cost

        Returns:
            True if portfolio can afford the trade
        """
        total_cost = (shares * price) + commission
        return self.cash >= total_cost

    def execute_buy(
        self,
        shares: float,
        price: float,
        commission: float,
        timestamp: datetime,
        order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute a buy order and update portfolio state.

        Args:
            shares: Number of shares to buy (must be positive)
            price: Price per share (must be positive)
            commission: Commission cost (must be non-negative)
            timestamp: Execution timestamp
            order_id: Optional order ID (generated if not provided)

        Returns:
            Trade information dictionary

        Raises:
            ValueError: If parameters are invalid or insufficient cash
        """
        if shares <= 0:
            raise ValueError(f"shares must be positive, got {shares}")
        if price <= 0:
            raise ValueError(f"price must be positive, got {price}")
        if commission < 0:
            raise ValueError(f"commission must be non-negative, got {commission}")

        total_cost = (shares * price) + commission

        if not self.can_afford(shares, price, commission):
            raise ValueError(
                f"Insufficient cash. Need {total_cost:.2f}, have {self.cash:.2f}"
            )

        # Generate order ID if not provided
        if order_id is None:
            order_id = f"BUY_{uuid4().hex[:8]}"

        # Deduct cash
        self.cash -= total_cost

        # Add position
        self.positions[order_id] = {
            "shares": shares,
            "entry_price": price,
            "entry_time": timestamp,
        }

        # Create trade record
        trade_info = {
            "order_id": order_id,
            "timestamp": timestamp,
            "signal": "buy",
            "shares": shares,
            "price": price,
            "commission": commission,
            "portfolio_value": self.get_portfolio_value(price),
            "cash": self.cash,
            "positions": dict(self.positions),
            "pnl": None,
        }

        self.trade_history.append(trade_info)
        return trade_info

    def execute_sell(
        self,
        shares: float,
        price: float,
        commission: float,
        timestamp: datetime,
        order_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Execute a sell order using FIFO or specified order IDs.

        Args:
            shares: Number of shares to sell (must be positive)
            price: Price per share (must be positive)
            commission: Commission cost (must be non-negative)
            timestamp: Execution timestamp
            order_ids: Optional list of specific order IDs to sell from (FIFO if None)

        Returns:
            Trade information dictionary including P&L

        Raises:
            ValueError: If parameters invalid, insufficient shares, or invalid order_ids
        """
        if shares <= 0:
            raise ValueError(f"shares must be positive, got {shares}")
        if price <= 0:
            raise ValueError(f"price must be positive, got {price}")
        if commission < 0:
            raise ValueError(f"commission must be non-negative, got {commission}")

        total_shares_held = self.get_total_shares()
        if shares > total_shares_held:
            raise ValueError(
                f"Insufficient shares. Trying to sell {shares}, have {total_shares_held}"
            )

        # Determine which positions to sell from
        if order_ids is not None:
            # Validate specified order_ids exist
            invalid_ids = set(order_ids) - set(self.positions.keys())
            if invalid_ids:
                raise ValueError(f"Invalid order_ids specified: {invalid_ids}")
            sell_order_ids = order_ids
        else:
            # Use FIFO: sort by entry_time
            sell_order_ids = sorted(
                self.positions.keys(), key=lambda oid: self.positions[oid]["entry_time"]
            )

        # Sell shares from positions
        shares_remaining = shares
        total_pnl = 0.0
        closed_order_ids = []

        for order_id in sell_order_ids:
            if shares_remaining <= 0:
                break

            position = self.positions[order_id]
            position_shares = position["shares"]
            entry_price = position["entry_price"]

            # Calculate how much to sell from this position
            shares_to_sell = min(shares_remaining, position_shares)

            # Calculate P&L for this portion
            pnl = shares_to_sell * (price - entry_price)
            total_pnl += pnl

            # Update position
            position["shares"] -= shares_to_sell
            shares_remaining -= shares_to_sell

            # Remove position if fully closed
            if position["shares"] == 0:
                closed_order_ids.append(order_id)

        # Remove closed positions
        for order_id in closed_order_ids:
            del self.positions[order_id]

        # Add proceeds to cash (minus commission)
        proceeds = (shares * price) - commission
        self.cash += proceeds

        # Adjust total P&L for commission
        total_pnl -= commission

        # Create trade record
        trade_info = {
            "order_id": f"SELL_{uuid4().hex[:8]}",
            "timestamp": timestamp,
            "signal": "sell",
            "shares": shares,
            "price": price,
            "commission": commission,
            "portfolio_value": self.get_portfolio_value(price),
            "cash": self.cash,
            "positions": dict(self.positions),
            "pnl": total_pnl,
        }

        self.trade_history.append(trade_info)
        return trade_info

    def get_trade_history(self) -> List[Dict[str, Any]]:
        """Get complete trade history.

        Returns:
            List of trade information dictionaries
        """
        return self.trade_history.copy()

    def reset(self) -> None:
        """Reset portfolio to initial state."""
        self.cash = self.initial_capital
        self.positions.clear()
        self.trade_history.clear()

    def get_state_snapshot(self) -> Dict[str, Any]:
        """Get current portfolio state snapshot.

        Returns:
            Dictionary with current cash, positions, and total shares
        """
        return {
            "cash": self.cash,
            "positions": dict(self.positions),
            "total_shares": self.get_total_shares(),
        }

    def __repr__(self) -> str:
        """String representation of portfolio."""
        return (
            f"Portfolio(cash={self.cash:.2f}, "
            f"positions={len(self.positions)}, "
            f"total_shares={self.get_total_shares():.2f})"
        )
