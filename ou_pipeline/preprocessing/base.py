"""Abstract base class for series transformers in the OU pipeline.

Every series transformation (e.g. rolling mean detrender) implements
SeriesTransformer so preprocessing steps can be used interchangeably.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ou_pipeline.models.results import DetrendResult


class SeriesTransformer(ABC):
    """Common interface for time series preprocessing and detrending."""

    @abstractmethod
    def transform(self, series: np.ndarray) -> DetrendResult:
        """Transform a 1-D series into a DetrendResult (residual and baseline).

        Args:
            series: 1-D float array of observations (e.g. log prices).

        Returns:
            DetrendResult containing:
                - residual: The detrended series (series - baseline)
                - baseline: The estimated trend/moving average
        """
        ...
