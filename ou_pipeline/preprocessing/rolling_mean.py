"""Rolling mean detrender for the OU mean-reversion pipeline.

Subtracts a rolling moving average from a price or log-price series to isolate
short-term oscillations around a local baseline.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ou_pipeline.models.results import DetrendResult
from ou_pipeline.preprocessing.base import SeriesTransformer

logger = logging.getLogger(__name__)


class RollingMeanDetrender(SeriesTransformer):
    """Detrends a 1-D time series by subtracting a rolling window mean.

    Args:
        window: Rolling window size in periods (e.g. 20 trading days).
            Must be an integer >= 2.

    Example::

        detrender = RollingMeanDetrender(window=20)
        res = detrender.transform(log_price_series)
        residual = res.residual
        baseline = res.baseline
    """

    def __init__(self, window: int = 20) -> None:
        if window < 2:
            raise ValueError(f"window must be at least 2, got {window}")
        self._window = window

    @property
    def window(self) -> int:
        """Window size used for rolling mean calculation."""
        return self._window

    def transform(self, series: np.ndarray) -> DetrendResult:
        """Subtract rolling mean from *series* and return DetrendResult.

        Args:
            series: 1-D array of observations (e.g. log prices).

        Returns:
            DetrendResult with:
                - residual: 1-D array of (series - baseline)
                - baseline: 1-D array of rolling mean values

        Raises:
            ValueError: If series is not 1-D, contains non-finite values,
                or has length less than window.
        """
        series_arr = np.asarray(series, dtype=np.float64)

        if series_arr.ndim != 1:
            raise ValueError(f"series must be 1-D, got shape {series_arr.shape}")
        if len(series_arr) < self._window:
            raise ValueError(
                f"series length ({len(series_arr)}) must be >= window ({self._window})"
            )
        if not np.isfinite(series_arr).all():
            raise ValueError("series contains NaN or Inf values")

        s = pd.Series(series_arr)
        baseline = s.rolling(window=self._window, min_periods=1).mean().to_numpy(dtype=np.float64)
        residual = series_arr - baseline

        logger.debug(
            "RollingMeanDetrender(window=%d) transformed %d observations",
            self._window,
            len(series_arr),
        )

        return DetrendResult(residual=residual, baseline=baseline)
