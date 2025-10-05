"""Random search optimizer - samples random parameter combinations."""

import random
from typing import Any, Dict, List, Type

import pandas as pd
from tqdm import tqdm

from simple_backtest.config.settings import BacktestConfig
from simple_backtest.optimization.base import Optimizer
from simple_backtest.strategy.base import Strategy
from simple_backtest.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)


class RandomSearchOptimizer(Optimizer):
    """Random search optimizer - randomly samples parameter space.

    Faster than grid search for large parameter spaces.
    Samples n_iter random combinations instead of testing all.

    Example:
        optimizer = RandomSearchOptimizer(n_iter=50, random_state=42)
        results = optimizer.optimize(
            data=data,
            config=config,
            strategy_class=MovingAverageStrategy,
            param_space={
                'short_window': list(range(5, 21)),
                'long_window': list(range(20, 61)),
                'shares': [10]
            },
            metric='sharpe_ratio'
        )
    """

    def __init__(
        self,
        n_iter: int = 100,
        random_state: int = None,
        verbose: bool = True,
        name: str = None,
    ):
        """Initialize random search optimizer.

        :param n_iter: Number of random combinations to test
        :param random_state: Random seed for reproducibility
        :param verbose: Show progress bar
        :param name: Optimizer name
        """
        super().__init__(name=name or "RandomSearch")
        self.n_iter = n_iter
        self.random_state = random_state
        self.verbose = verbose

        if random_state is not None:
            random.seed(random_state)

    def optimize(
        self,
        data: pd.DataFrame,
        config: BacktestConfig,
        strategy_class: Type[Strategy],
        param_space: Dict[str, List[Any]],
        metric: str = "sharpe_ratio",
    ) -> pd.DataFrame:
        """Run random search optimization.

        :param data: OHLCV DataFrame with DatetimeIndex
        :param config: Backtest configuration
        :param strategy_class: Strategy class to optimize
        :param param_space: Dict of param_name: [values] to sample from
        :param metric: Metric to optimize
        :return: DataFrame of results sorted by metric
        """
        results = []
        param_names = list(param_space.keys())

        if self.verbose:
            logger.info(f"Testing {self.n_iter} random parameter combinations...")

        # Generate random combinations
        iterator = range(self.n_iter)
        if self.verbose:
            iterator = tqdm(iterator, desc="Random Search")

        for _ in iterator:
            # Sample random values for each parameter
            param_dict = {name: random.choice(param_space[name]) for name in param_names}

            try:
                # Create strategy with these parameters
                strategy = strategy_class(**param_dict)

                # Run backtest
                metrics = self._run_backtest(data, config, strategy)

                # Store results
                results.append({**param_dict, **metrics})

            except Exception as e:
                if self.verbose:
                    logger.warning(f"Error with params {param_dict}: {e}")
                continue

        # Create DataFrame
        df = pd.DataFrame(results)

        if df.empty:
            logger.warning("All parameter combinations failed!")
            return df

        if metric not in df.columns:
            logger.warning(f"Metric '{metric}' not found. Available metrics: {list(df.columns)}")
            return df

        # Sort by metric (descending)
        return df.sort_values(metric, ascending=False).reset_index(drop=True)
