"""Caching utilities for data and metric calculations."""

import hashlib
import pickle
from functools import wraps
from pathlib import Path
from typing import Any, Callable

import pandas as pd


class DataCache:
    """Simple file-based cache for DataFrames and computation results.

    Uses pickle for serialization and MD5 hashing for cache keys.
    """

    def __init__(self, cache_dir: str | Path = ".cache", enabled: bool = True):
        """Initialize cache.

        Args:
            cache_dir: Directory to store cache files
            enabled: Whether caching is enabled
        """
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled

        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, *args: Any, **kwargs: Any) -> str:
        """Generate cache key from arguments.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            MD5 hash of serialized arguments
        """
        # Serialize arguments
        serialized = pickle.dumps((args, kwargs))
        # Generate hash
        return hashlib.md5(serialized).hexdigest()

    def _get_cache_path(self, key: str) -> Path:
        """Get path to cache file for given key.

        Args:
            key: Cache key

        Returns:
            Path to cache file
        """
        return self.cache_dir / f"{key}.pkl"

    def get(self, key: str) -> Any | None:
        """Retrieve value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        if not self.enabled:
            return None

        cache_path = self._get_cache_path(key)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except Exception:
            # If cache is corrupted, remove it
            cache_path.unlink(missing_ok=True)
            return None

    def set(self, key: str, value: Any) -> None:
        """Store value in cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        if not self.enabled:
            return

        cache_path = self._get_cache_path(key)

        try:
            with open(cache_path, "wb") as f:
                pickle.dump(value, f)
        except Exception:
            # Silently fail on cache write errors
            pass

    def clear(self) -> None:
        """Clear all cache files."""
        if not self.enabled:
            return

        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink(missing_ok=True)

    def __repr__(self) -> str:
        """String representation."""
        status = "enabled" if self.enabled else "disabled"
        return f"DataCache(dir='{self.cache_dir}', status={status})"


# Global cache instance
_global_cache = DataCache(enabled=False)


def get_global_cache() -> DataCache:
    """Get global cache instance.

    Returns:
        Global DataCache instance
    """
    return _global_cache


def set_global_cache(cache: DataCache) -> None:
    """Set global cache instance.

    Args:
        cache: DataCache instance to use globally
    """
    global _global_cache
    _global_cache = cache


def cached(cache: DataCache | None = None) -> Callable:
    """Decorator to cache function results.

    Args:
        cache: Cache instance to use (None = use global cache)

    Returns:
        Decorator function

    Example:
        ```python
        cache = DataCache(cache_dir=".cache", enabled=True)

        @cached(cache)
        def expensive_computation(data: pd.DataFrame) -> float:
            return data.sum().sum()
        ```
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Determine which cache to use
            cache_instance = cache if cache is not None else get_global_cache()

            if not cache_instance.enabled:
                # Cache disabled, just call function
                return func(*args, **kwargs)

            # Generate cache key including function name
            cache_key = cache_instance._get_cache_key(func.__name__, *args, **kwargs)

            # Try to get from cache
            cached_result = cache_instance.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Compute result
            result = func(*args, **kwargs)

            # Store in cache
            cache_instance.set(cache_key, result)

            return result

        return wrapper

    return decorator


def cache_dataframe_slice(
    data: pd.DataFrame, start_idx: int, end_idx: int, cache: DataCache | None = None
) -> pd.DataFrame:
    """Cache-aware DataFrame slicing.

    Args:
        data: DataFrame to slice
        start_idx: Start index
        end_idx: End index
        cache: Cache instance (None = use global cache)

    Returns:
        DataFrame slice
    """
    cache_instance = cache if cache is not None else get_global_cache()

    if not cache_instance.enabled:
        return data.iloc[start_idx:end_idx]

    # Generate cache key based on data hash and indices
    data_hash = hashlib.md5(pd.util.hash_pandas_object(data.index).values).hexdigest()[:8]
    cache_key = f"slice_{data_hash}_{start_idx}_{end_idx}"

    # Try cache
    cached_slice = cache_instance.get(cache_key)
    if cached_slice is not None:
        return cached_slice

    # Compute slice
    slice_result = data.iloc[start_idx:end_idx]

    # Cache result
    cache_instance.set(cache_key, slice_result)

    return slice_result
