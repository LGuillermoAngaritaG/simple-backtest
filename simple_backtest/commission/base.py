"""Base commission class for trading costs."""

from abc import ABC, abstractmethod


class Commission(ABC):
    """Abstract base class for commission calculation.

    Users can create custom commission structures by inheriting from this class
    and implementing the calculate() method.

    Example:
        class MyCommission(Commission):
            def calculate(self, shares, price):
                # Your commission logic here
                return commission_amount
    """

    def __init__(self, name: str = None):
        """Initialize commission calculator.

        :param name: Commission name (auto-generated if None)
        """
        self._name = name or self.__class__.__name__

    @abstractmethod
    def calculate(self, shares: float, price: float) -> float:
        """Calculate commission for a trade.

        :param shares: Number of shares traded
        :param price: Price per share
        :return: Commission amount (must be non-negative)
        """
        pass

    def get_name(self) -> str:
        """Get commission name."""
        return self._name

    def __call__(self, shares: float, price: float) -> float:
        """Allow commission to be called as a function.

        :param shares: Number of shares traded
        :param price: Price per share
        :return: Commission amount
        """
        commission = self.calculate(shares, price)

        # Validate result
        if commission < 0:
            raise ValueError(
                f"Commission must be non-negative, got {commission} "
                f"for trade: {shares} shares @ ${price}"
            )

        return commission

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name='{self._name}')"
