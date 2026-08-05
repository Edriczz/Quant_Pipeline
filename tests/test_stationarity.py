"""Tests for ou_pipeline.diagnostics.stationarity.StationarityTester.

Two critical synthetic cases:
1. Simulated OU path (strongly mean-reverting) → must classify as stationary.
2. Simulated random walk (unit root) → must classify as non-stationary.

A high-strength OU process (theta=5.0) and a clean random walk are used
so the ADF test has decisive statistical power and results are stable
across random seeds.
"""

from __future__ import annotations

import numpy as np
import pytest

from ou_pipeline.config import PipelineConfig
from ou_pipeline.diagnostics.stationarity import StationarityTester
from ou_pipeline.models.results import StationarityResult


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------

def _simulate_ou(
    n: int = 1_000,
    theta: float = 5.0,
    mu: float = 0.0,
    sigma: float = 0.5,
    dt: float = 1.0,
    seed: int = 42,
) -> np.ndarray:
    """Simulate a discrete-time OU path using the exact transition formula.

    Args:
        n: Number of time steps.
        theta: Mean-reversion speed (larger → faster reversion).
        mu: Long-run mean.
        sigma: Diffusion coefficient.
        dt: Time step size.
        seed: RNG seed for reproducibility.

    Returns:
        1-D array of length *n*.
    """
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[0] = mu
    e_decay = np.exp(-theta * dt)
    std = sigma * np.sqrt((1 - e_decay**2) / (2 * theta))
    noise = rng.normal(0.0, std, size=n - 1)
    for t in range(1, n):
        x[t] = mu + e_decay * (x[t - 1] - mu) + noise[t - 1]
    return x


def _simulate_random_walk(
    n: int = 1_000,
    sigma: float = 1.0,
    seed: int = 42,
) -> np.ndarray:
    """Simulate a pure random walk (unit root, non-stationary).

    Args:
        n: Number of time steps.
        sigma: Standard deviation of each increment.
        seed: RNG seed for reproducibility.

    Returns:
        1-D array of length *n* (cumulative sum of iid noise).
    """
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(0.0, sigma, size=n))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tester() -> StationarityTester:
    """Default StationarityTester with alpha=0.05."""
    return StationarityTester(config=PipelineConfig(adf_alpha=0.05))


# ---------------------------------------------------------------------------
# Core gate tests (required by Phase 3 gate)
# ---------------------------------------------------------------------------

class TestSyntheticClassification:
    """The two synthetic cases that constitute the Phase 3 gate."""

    def test_ou_path_is_stationary(self, tester: StationarityTester) -> None:
        """A strongly mean-reverting OU path must be classified as stationary."""
        series = _simulate_ou(n=1_000, theta=5.0, mu=0.0, sigma=0.5, seed=42)
        result = tester.test(series)
        assert isinstance(result, StationarityResult)
        assert result.is_stationary, (
            f"OU path (theta=5.0) was NOT classified as stationary.  "
            f"ADF={result.adf_statistic:.4f}  p={result.p_value:.4f}"
        )

    def test_random_walk_is_not_stationary(self, tester: StationarityTester) -> None:
        """A pure random walk (unit root) must NOT be classified as stationary."""
        series = _simulate_random_walk(n=1_000, sigma=1.0, seed=42)
        result = tester.test(series)
        assert isinstance(result, StationarityResult)
        assert not result.is_stationary, (
            f"Random walk WAS incorrectly classified as stationary.  "
            f"ADF={result.adf_statistic:.4f}  p={result.p_value:.4f}"
        )


# ---------------------------------------------------------------------------
# Result structure tests
# ---------------------------------------------------------------------------

class TestResultStructure:
    """Verify the returned dataclass is well-formed."""

    def test_returns_stationarity_result(self, tester: StationarityTester) -> None:
        """test() must return a StationarityResult instance."""
        series = _simulate_ou(n=200, theta=2.0)
        result = tester.test(series)
        assert isinstance(result, StationarityResult)

    def test_p_value_in_unit_interval(self, tester: StationarityTester) -> None:
        """p_value must be in [0, 1]."""
        series = _simulate_ou(n=200, theta=2.0)
        result = tester.test(series)
        assert 0.0 <= result.p_value <= 1.0

    def test_is_stationary_reflects_alpha(self) -> None:
        """is_stationary must agree with p_value < alpha at the chosen alpha."""
        series = _simulate_ou(n=500, theta=3.0, seed=7)
        # Use a very strict alpha so we can force a known outcome.
        strict_tester = StationarityTester(config=PipelineConfig(adf_alpha=1e-10))
        result = strict_tester.test(series)
        # With alpha=1e-10 even a stationary series may not pass.
        assert result.is_stationary == (result.p_value < 1e-10)


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------

class TestInputValidation:
    """StationarityTester must reject bad inputs with ValueError."""

    def test_too_short_raises(self, tester: StationarityTester) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            tester.test(np.array([1.0]))

    def test_nan_raises(self, tester: StationarityTester) -> None:
        series = np.array([1.0, 2.0, float("nan"), 4.0])
        with pytest.raises(ValueError, match="NaN or Inf"):
            tester.test(series)

    def test_inf_raises(self, tester: StationarityTester) -> None:
        series = np.array([1.0, float("inf"), 3.0])
        with pytest.raises(ValueError, match="NaN or Inf"):
            tester.test(series)

    def test_2d_array_raises(self, tester: StationarityTester) -> None:
        series = np.ones((10, 2))
        with pytest.raises(ValueError, match="1-D"):
            tester.test(series)
