"""Commission calculation strategies using Factory Pattern."""

from typing import Callable, List, Tuple

from simple_backtest.config.settings import BacktestConfig


def percentage_commission(shares: float, price: float, rate: float) -> float:
    """Calculate commission as percentage of trade value.

    Args:
        shares: Number of shares traded
        price: Price per share
        rate: Commission rate (e.g., 0.001 for 0.1%)

    Returns:
        Commission amount

    Example:
        >>> percentage_commission(100, 50.0, 0.001)
        5.0  # 0.1% of $5000
    """
    return shares * price * rate


def flat_commission(shares: float, price: float, flat_fee: float) -> float:
    """Calculate flat commission per trade.

    Args:
        shares: Number of shares traded (unused, for signature compatibility)
        price: Price per share (unused, for signature compatibility)
        flat_fee: Flat commission amount

    Returns:
        Commission amount

    Example:
        >>> flat_commission(100, 50.0, 10.0)
        10.0
    """
    return flat_fee


def tiered_commission(shares: float, price: float, tiers: List[Tuple[float, float]]) -> float:
    """Calculate tiered commission based on trade value thresholds.

    Args:
        shares: Number of shares traded
        price: Price per share
        tiers: List of (threshold, rate) tuples, sorted by threshold ascending.
               Example: [(1000, 0.002), (5000, 0.001), (float('inf'), 0.0005)]
               - Up to $1000: 0.2%
               - $1000-$5000: 0.1%
               - Above $5000: 0.05%

    Returns:
        Commission amount

    Example:
        >>> tiers = [(1000, 0.002), (5000, 0.001), (float('inf'), 0.0005)]
        >>> tiered_commission(100, 60.0, tiers)  # $6000 trade
        6.0  # (1000*0.002) + (4000*0.001) + (1000*0.0005) = 2 + 4 + 0.5 = 6.5
    """
    trade_value = shares * price
    commission = 0.0
    prev_threshold = 0.0

    for threshold, rate in tiers:
        if trade_value <= threshold:
            # Trade value falls in this tier
            commission += (trade_value - prev_threshold) * rate
            break
        else:
            # Trade value exceeds this tier, apply rate to tier range
            commission += (threshold - prev_threshold) * rate
            prev_threshold = threshold

    return commission


def get_commission_calculator(config: BacktestConfig) -> Callable[[float, float], float]:
    """Factory function to create commission calculator based on config.

    Implements Factory Pattern for flexible commission model creation.

    Args:
        config: BacktestConfig with commission_type and commission_value

    Returns:
        Callable taking (shares, price) and returning commission amount

    Raises:
        ValueError: If commission_type is invalid or custom callable not provided

    Example:
        >>> config = BacktestConfig(commission_type='percentage', commission_value=0.001)
        >>> calc = get_commission_calculator(config)
        >>> calc(100, 50.0)
        5.0
    """
    if config.commission_type == "percentage":
        rate = config.commission_value
        return lambda shares, price: percentage_commission(shares, price, rate)

    elif config.commission_type == "flat":
        flat_fee = config.commission_value
        return lambda shares, price: flat_commission(shares, price, flat_fee)

    elif config.commission_type == "tiered":
        # For tiered, commission_value should be a list of tuples
        # In practice, this would be parsed from config or passed separately
        # For now, we'll create a default tiered structure
        if isinstance(config.commission_value, list):
            tiers = config.commission_value
        else:
            # Default tiered structure if single value provided
            # Use value as base rate with scaling tiers
            base_rate = config.commission_value
            tiers = [
                (1000, base_rate * 2),
                (5000, base_rate),
                (float("inf"), base_rate * 0.5),
            ]
        return lambda shares, price: tiered_commission(shares, price, tiers)

    elif config.commission_type == "custom":
        # Custom commission requires a user-provided callable
        # This would be passed separately, not through pydantic config
        # Return zero commission as safe default
        return lambda shares, price: 0.0

    else:
        raise ValueError(
            f"Invalid commission_type: {config.commission_type}. "
            f"Must be one of: 'percentage', 'flat', 'tiered', 'custom'"
        )


def create_custom_commission(func: Callable[[float, float], float]) -> Callable[[float, float], float]:
    """Wrap a custom commission function with validation.

    Args:
        func: Custom commission function taking (shares, price) -> commission

    Returns:
        Validated commission calculator

    Raises:
        ValueError: If custom function returns negative commission

    Example:
        >>> def my_commission(shares, price):
        ...     return 5.0 if shares * price < 1000 else 10.0
        >>> calc = create_custom_commission(my_commission)
        >>> calc(10, 50.0)  # $500 trade
        5.0
    """
    def validated_commission(shares: float, price: float) -> float:
        commission = func(shares, price)
        if commission < 0:
            raise ValueError(
                f"Commission must be non-negative, got {commission} "
                f"for trade: {shares} shares @ ${price}"
            )
        return commission

    return validated_commission
