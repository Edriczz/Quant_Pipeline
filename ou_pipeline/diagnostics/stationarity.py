"""Stationarity testing for the OU pipeline.

:class:`StationarityTester` wraps the Augmented Dickey-Fuller test from
``statsmodels`` and returns a typed :class:`~ou_pipeline.models.results.StationarityResult`.

Estimators must NOT call this class internally — the pipeline or app
decides when to run the test and how to gate on the result.
"""

from __future__ import annotations

import logging

import numpy as np
from statsmodels.tsa.stattools import adfuller

from ou_pipeline.config import PipelineConfig
from ou_pipeline.models.results import StationarityResult

logger = logging.getLogger(__name__)


class StationarityTester:
    """Runs an Augmented Dickey-Fuller test on a numeric series.

    Args:
        config: Pipeline configuration.  Only ``config.adf_alpha`` is
            used (default 0.05).  If omitted, a default
            :class:`~ou_pipeline.config.PipelineConfig` is used.

    Example::

        tester = StationarityTester()
        result = tester.test(log_series)
        if result.is_stationary:
            print("Series is mean-reverting at the 5 % level.")
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self._config = config or PipelineConfig()

    def test(self, series: np.ndarray) -> StationarityResult:
        """Run the ADF test on *series* and return a typed result.

        The null hypothesis of the ADF test is that a unit root is
        present (i.e. the series is *non*-stationary).  We reject H₀ —
        and declare the series stationary — when ``p_value < alpha``.

        Args:
            series: A 1-D array of observations (e.g. log-prices).
                Must have at least 20 observations for a meaningful test.

        Returns:
            A :class:`~ou_pipeline.models.results.StationarityResult`
            with ``adf_statistic``, ``p_value``, and ``is_stationary``.

        Raises:
            ValueError: If *series* has fewer than 2 observations or
                contains non-finite values.
        """
        series = np.asarray(series, dtype=np.float64)

        if series.ndim != 1:
            raise ValueError(f"series must be 1-D, got shape {series.shape}")
        if len(series) < 2:
            raise ValueError(f"series must have at least 2 observations, got {len(series)}")
        if not np.isfinite(series).all():
            raise ValueError("series contains NaN or Inf values")

        logger.debug("Running ADF test on series of length %d", len(series))

        # adfuller returns (adf_stat, p_value, usedlag, nobs, crit_values, icbest)
        adf_stat: float
        p_value: float
        adf_stat, p_value, *_ = adfuller(series, autolag="AIC")

        is_stationary = bool(p_value < self._config.adf_alpha)

        logger.debug("ADF stat=%.4f  p=%.4f  stationary=%s", adf_stat, p_value, is_stationary)

        return StationarityResult(
            adf_statistic=float(adf_stat),
            p_value=float(p_value),
            is_stationary=is_stationary,
        )
