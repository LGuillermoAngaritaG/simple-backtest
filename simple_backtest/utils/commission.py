"""Commission calculation functions."""

from typing import Callable, List, Tuple

from simple_backtest.config.settings import BacktestConfig


def percentage_commission(shares: float, price: float, rate: float) -> float:
    """Calculate commission as percentage of trade value.

    :param shares: Shares traded
    :param price: Price per share
    :param rate: Commission rate (e.g., 0.001 for 0.1%)
    :return: Commission amount
    """
    return shares * price * rate


def flat_commission(shares: float, price: float, flat_fee: float) -> float:
    """Calculate flat commission per trade.

    :param shares: Unused (for signature compatibility)
    :param price: Unused (for signature compatibility)
    :param flat_fee: Flat commission amount
    :return: Commission amount
    """
    return flat_fee


def tiered_commission(shares: float, price: float, tiers: List[Tuple[float, float]]) -> float:
    """Calculate tiered commission based on trade value.

    :param shares: Shares traded
    :param price: Price per share
    :param tiers: List of (threshold, rate) tuples sorted by threshold
    :return: Commission amount
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
    """Create commission calculator from config.

    :param config: BacktestConfig with commission settings
    :return: Commission function (shares, price) -> commission
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


def create_custom_commission(
    func: Callable[[float, float], float],
) -> Callable[[float, float], float]:
    """Wrap custom commission function with validation.

    :param func: Commission function (shares, price) -> commission
    :return: Validated commission calculator
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
