"""Grid search optimizer - exhaustive parameter search."""

import itertools
from typing import Any, Dict, List, Type

import pandas as pd
from tqdm import tqdm

from simple_backtest.config.settings import BacktestConfig
from simple_backtest.optimization.base import Optimizer
from simple_backtest.strategy.base import Strategy
from simple_backtest.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)


class GridSearchOptimizer(Optimizer):
    """Grid search optimizer - tests all parameter combinations.

    Exhaustively searches through all possible parameter combinations.
    Best for small parameter spaces.

    Example:
        optimizer = GridSearchOptimizer()
        results = optimizer.optimize(
            data=data,
            config=config,
            strategy_class=MovingAverageStrategy,
            param_space={
                'short_window': [5, 10, 15],
                'long_window': [20, 30, 40],
                'shares': [10]
            },
            metric='sharpe_ratio'
        )
    """

    def __init__(self, verbose: bool = True, name: str = None):
        """Initialize grid search optimizer.

        :param verbose: Show progress bar
        :param name: Optimizer name
        """
        super().__init__(name=name or "GridSearch")
        self.verbose = verbose

    def optimize(
        self,
        data: pd.DataFrame,
        config: BacktestConfig,
        strategy_class: Type[Strategy],
        param_space: Dict[str, List[Any]],
        metric: str = "sharpe_ratio",
    ) -> pd.DataFrame:
        """Run grid search optimization.

        :param data: OHLCV DataFrame with DatetimeIndex
        :param config: Backtest configuration
        :param strategy_class: Strategy class to optimize
        :param param_space: Dict of param_name: [values] to search
        :param metric: Metric to optimize
        :return: DataFrame of results sorted by metric
        """
        results = []
        param_names = list(param_space.keys())
        param_combinations = list(itertools.product(*param_space.values()))

        if self.verbose:
            logger.info(f"Testing {len(param_combinations)} parameter combinations...")

        # Iterate through all combinations
        iterator = (
            tqdm(param_combinations, desc="Grid Search") if self.verbose else param_combinations
        )

        for params in iterator:
            param_dict = dict(zip(param_names, params))

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
