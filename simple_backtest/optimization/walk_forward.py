"""Walk-forward optimizer - train/test split optimization."""

from typing import Any, Dict, List, Type

import pandas as pd

from simple_backtest.config.settings import BacktestConfig
from simple_backtest.optimization.base import Optimizer
from simple_backtest.optimization.grid_search import GridSearchOptimizer
from simple_backtest.strategy.base import Strategy
from simple_backtest.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)


class WalkForwardOptimizer(Optimizer):
    """Walk-forward optimizer - optimizes on training data, validates on test data.

    More realistic than simple optimization - prevents overfitting by testing
    on out-of-sample data.

    Example:
        optimizer = WalkForwardOptimizer(train_size=0.7)
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

        # Results include both train and test metrics
        print(results[['short_window', 'long_window', 'train_sharpe', 'test_sharpe']])
    """

    def __init__(
        self,
        train_size: float = 0.7,
        base_optimizer: Optimizer = None,
        verbose: bool = True,
        name: str = None,
    ):
        """Initialize walk-forward optimizer.

        :param train_size: Fraction of data to use for training (0.0 to 1.0)
        :param base_optimizer: Optimizer to use for training (default: GridSearch)
        :param verbose: Show progress information
        :param name: Optimizer name
        """
        super().__init__(name=name or "WalkForward")

        if not 0.0 < train_size < 1.0:
            raise ValueError(f"train_size must be between 0 and 1, got {train_size}")

        self.train_size = train_size
        self.base_optimizer = base_optimizer or GridSearchOptimizer(verbose=verbose)
        self.verbose = verbose

    def optimize(
        self,
        data: pd.DataFrame,
        config: BacktestConfig,
        strategy_class: Type[Strategy],
        param_space: Dict[str, List[Any]],
        metric: str = "sharpe_ratio",
    ) -> pd.DataFrame:
        """Run walk-forward optimization.

        :param data: OHLCV DataFrame with DatetimeIndex
        :param config: Backtest configuration
        :param strategy_class: Strategy class to optimize
        :param param_space: Dict of param_name: [values] to search
        :param metric: Metric to optimize
        :return: DataFrame with train and test results for each parameter combo
        """
        # Split data into train and test
        split_idx = int(len(data) * self.train_size)
        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]

        if self.verbose:
            logger.info(f"\n{'='*60}")
            logger.info("Walk-Forward Optimization")
            logger.info(f"{'='*60}")
            logger.info(f"Train period: {train_data.index[0]} to {train_data.index[-1]}")
            logger.info(f"  - Rows: {len(train_data)}")
            logger.info(f"Test period: {test_data.index[0]} to {test_data.index[-1]}")
            logger.info(f"  - Rows: {len(test_data)}")
            logger.info(f"{'='*60}\n")

        # Optimize on training data
        if self.verbose:
            logger.info("Phase 1: Optimizing on training data...")

        train_results = self.base_optimizer.optimize(
            data=train_data,
            config=config,
            strategy_class=strategy_class,
            param_space=param_space,
            metric=metric,
        )

        if train_results.empty:
            logger.warning("Training phase produced no results")
            return train_results

        # Test all parameter combinations on test data
        if self.verbose:
            logger.info(f"\nPhase 2: Testing {len(train_results)} combinations on test data...")

        param_names = list(param_space.keys())
        test_metrics_list = []

        for idx, row in train_results.iterrows():
            param_dict = {name: row[name] for name in param_names}

            try:
                # Create strategy with these parameters
                strategy = strategy_class(**param_dict)

                # Run on test data
                test_metrics = self._run_backtest(test_data, config, strategy)

                test_metrics_list.append(test_metrics)

            except Exception as e:
                if self.verbose:
                    logger.warning(f"Error testing params {param_dict}: {e}")
                # Add empty metrics
                test_metrics_list.append({})

        # Combine train and test results
        results = train_results.copy()

        # Rename train columns
        metric_cols = [
            col
            for col in results.columns
            if col not in param_names and col not in param_space.keys()
        ]
        rename_dict = {col: f"train_{col}" for col in metric_cols}
        results = results.rename(columns=rename_dict)

        # Add test metrics
        test_df = pd.DataFrame(test_metrics_list)
        for col in test_df.columns:
            results[f"test_{col}"] = test_df[col]

        # Calculate train/test difference
        if f"train_{metric}" in results.columns and f"test_{metric}" in results.columns:
            results[f"{metric}_diff"] = results[f"test_{metric}"] - results[f"train_{metric}"]

        # Sort by test metric (more realistic)
        if f"test_{metric}" in results.columns:
            results = results.sort_values(f"test_{metric}", ascending=False).reset_index(drop=True)

        if self.verbose:
            logger.info(f"\n{'='*60}")
            logger.info("Walk-Forward Optimization Complete")
            logger.info(f"{'='*60}")
            if not results.empty and f"test_{metric}" in results.columns:
                best = results.iloc[0]
                logger.info(f"\nBest parameters (by test {metric}):")
                for name in param_names:
                    logger.info(f"  {name}: {best[name]}")
                logger.info("\nPerformance:")
                logger.info(f"  Train {metric}: {best[f'train_{metric}']:.4f}")
                logger.info(f"  Test {metric}: {best[f'test_{metric}']:.4f}")
                if f"{metric}_diff" in results.columns:
                    logger.info(f"  Difference: {best[f'{metric}_diff']:.4f}")
            logger.info(f"{'='*60}\n")

        return results
