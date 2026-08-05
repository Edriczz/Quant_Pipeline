"""Tests for ou_pipeline.preprocessing.rolling_mean.RollingMeanDetrender.

The Phase 11 gate test:
Construct a synthetic series = trend(t) + mean_reverting_wiggle(t).
Verify that:
1. Raw combined series ADF test FAILS stationarity (unit root / trend present).
2. Detrended residual ADF test PASSES stationarity (mean-reverting around moving baseline).
Both on the SAME constructed data.
"""

from __future__ import annotations

import numpy as np
import pytest

from ou_pipeline.config import PipelineConfig
from ou_pipeline.diagnostics.stationarity import StationarityTester
from ou_pipeline.models.results import DetrendResult
from ou_pipeline.preprocessing.rolling_mean import RollingMeanDetrender


def _simulate_ou(
    n: int = 500,
    theta: float = 2.0,
    mu: float = 0.0,
    sigma: float = 0.5,
    seed: int = 42,
) -> np.ndarray:
    """Simulate a discrete OU path."""
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[0] = mu
    dt = 1.0
    e_decay = np.exp(-theta * dt)
    std = sigma * np.sqrt((1 - e_decay**2) / (2 * theta))
    noise = rng.normal(0.0, std, size=n - 1)
    for t in range(1, n):
        x[t] = mu + e_decay * (x[t - 1] - mu) + noise[t - 1]
    return x


class TestSyntheticDetrendingGate:
    """The critical Phase 11 Gate Test."""

    def test_raw_fails_and_detrended_passes_adf(self) -> None:
        """On same trend+wiggle data, raw ADF fails and detrended residual ADF passes."""
        n = 500
        t_arr = np.arange(n, dtype=np.float64)

        # 1. Linear upward trend
        trend = 0.05 * t_arr

        # 2. Mean-reverting OU oscillation
        wiggle = _simulate_ou(n=n, theta=2.0, mu=0.0, sigma=0.5, seed=42)

        # 3. Raw series = trend + wiggle
        raw_series = trend + wiggle

        tester = StationarityTester(config=PipelineConfig(adf_alpha=0.05))
        detrender = RollingMeanDetrender(window=20)

        # Raw series ADF check
        raw_stat = tester.test(raw_series)
        assert not raw_stat.is_stationary, (
            f"Raw series should fail ADF due to strong trend, but passed! "
            f"ADF={raw_stat.adf_statistic:.4f}, p={raw_stat.p_value:.4f}"
        )

        # Detrended residual ADF check
        detrend_res = detrender.transform(raw_series)
        residual_stat = tester.test(detrend_res.residual)
        assert residual_stat.is_stationary, (
            f"Detrended residual should pass ADF as stationary, but failed! "
            f"ADF={residual_stat.adf_statistic:.4f}, p={residual_stat.p_value:.4f}"
        )


class TestRollingMeanDetrenderProperties:
    """Unit tests for RollingMeanDetrender class properties and edge cases."""

    def test_returns_detrend_result(self) -> None:
        detrender = RollingMeanDetrender(window=10)
        series = np.linspace(10.0, 20.0, 50)
        res = detrender.transform(series)
        assert isinstance(res, DetrendResult)
        assert res.residual.shape == series.shape
        assert res.baseline.shape == series.shape

    def test_residual_plus_baseline_equals_series(self) -> None:
        detrender = RollingMeanDetrender(window=15)
        series = np.sin(np.linspace(0, 10, 100)) + np.linspace(0, 5, 100)
        res = detrender.transform(series)
        reconstructed = res.baseline + res.residual
        np.testing.assert_allclose(reconstructed, series, atol=1e-12)

    def test_invalid_window_raises(self) -> None:
        with pytest.raises(ValueError, match="window must be at least 2"):
            RollingMeanDetrender(window=1)

    def test_series_shorter_than_window_raises(self) -> None:
        detrender = RollingMeanDetrender(window=20)
        short_series = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="must be >= window"):
            detrender.transform(short_series)

    def test_nan_or_inf_raises(self) -> None:
        detrender = RollingMeanDetrender(window=5)
        bad_series = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        with pytest.raises(ValueError, match="NaN or Inf"):
            detrender.transform(bad_series)
