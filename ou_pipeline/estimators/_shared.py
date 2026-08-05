"""Shared utility functions for OU estimators.

Any computation that is common across two or more estimators lives here.
Nothing in this module contains estimation logic — it is purely helpers
(half-life formula, log-transform utilities, input validation).

Import from this module rather than copy-pasting into individual estimators.
"""

from __future__ import annotations

import numpy as np


def half_life_from_theta(theta: float) -> float:
    """Compute the half-life of mean reversion given speed parameter theta.

    The half-life is defined as the expected time for the process to
    travel half the distance from its current value to the long-run mean.

    Formula::

        half_life = ln(2) / theta

    Args:
        theta: Mean-reversion speed, in units of 1/trading-day.
            Must be strictly positive.

    Returns:
        Half-life in trading days.

    Raises:
        ValueError: If *theta* is not strictly positive.
    """
    if theta <= 0.0:
        raise ValueError(f"theta must be strictly positive, got {theta}")
    return float(np.log(2.0) / theta)


def log_transform(prices: np.ndarray) -> np.ndarray:
    """Return the natural log of a positive price array.

    Args:
        prices: 1-D array of positive prices (or any positive numeric
            values).  Must be strictly positive.

    Returns:
        ``np.log(prices)`` as a float64 array of the same shape.

    Raises:
        ValueError: If *prices* contains non-positive or non-finite values.
    """
    prices = np.asarray(prices, dtype=np.float64)
    if not np.isfinite(prices).all():
        raise ValueError("prices contains NaN or Inf values")
    if (prices <= 0.0).any():
        raise ValueError("prices must be strictly positive for a log transform")
    return np.log(prices)


def validate_series(series: np.ndarray, min_length: int = 10) -> np.ndarray:
    """Cast and validate an input series for use by an estimator.

    Args:
        series: Input array to validate.
        min_length: Minimum acceptable length.  Default 10.

    Returns:
        A 1-D float64 array.

    Raises:
        ValueError: If *series* is not 1-D, too short, or contains
            non-finite values.
    """
    series = np.asarray(series, dtype=np.float64)
    if series.ndim != 1:
        raise ValueError(f"series must be 1-D, got shape {series.shape}")
    if len(series) < min_length:
        raise ValueError(f"series must have at least {min_length} observations, got {len(series)}")
    if not np.isfinite(series).all():
        raise ValueError("series contains NaN or Inf values")
    return series
