"""Tests for ou_pipeline.estimators.ols.OLSEstimator.

The critical gate test is the synthetic recovery test: simulate an OU
path with known (theta, mu, sigma), fit with OLSEstimator, and assert the
recovered parameters are within agreed tolerances.

Tolerances (noiseless case, n=5000):
  - theta:  ±20 %   (OLS is biased toward zero; wider band acceptable)
  - mu:     ±1 %    (OLS is unbiased for mu in the noiseless case)
  - sigma:  ±10 %
"""

from __future__ import annotations

import numpy as np
import pytest

from ou_pipeline.config import PipelineConfig
from ou_pipeline.estimators.ols import OLSEstimator, OUFitError
from ou_pipeline.models.results import OUResult


# ---------------------------------------------------------------------------
# Shared synthetic generator (same as stationarity tests for consistency)
# ---------------------------------------------------------------------------

def _simulate_ou(
    n: int = 5_000,
    theta: float = 1.5,
    mu: float = 5.0,
    sigma: float = 0.3,
    dt: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """Simulate a discrete-time OU path using the exact transition formula."""
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[0] = mu
    e_decay = np.exp(-theta * dt)
    std = sigma * np.sqrt((1.0 - e_decay**2) / (2.0 * theta))
    noise = rng.normal(0.0, std, size=n - 1)
    for t in range(1, n):
        x[t] = mu + e_decay * (x[t - 1] - mu) + noise[t - 1]
    return x


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def estimator() -> OLSEstimator:
    return OLSEstimator()


TRUE_THETA = 1.5
TRUE_MU = 5.0
TRUE_SIGMA = 0.3


# ---------------------------------------------------------------------------
# Phase 5 gate: synthetic recovery test (noiseless)
# ---------------------------------------------------------------------------

class TestSyntheticRecovery:
    """Fit against a noiseless OU path and verify parameter recovery."""

    def test_theta_within_tolerance(self, estimator: OLSEstimator) -> None:
        """Recovered theta must be within ±20% of true value."""
        series = _simulate_ou(theta=TRUE_THETA, mu=TRUE_MU, sigma=TRUE_SIGMA)
        result = estimator.fit(series)
        rel_err = abs(result.theta - TRUE_THETA) / TRUE_THETA
        assert rel_err < 0.20, (
            f"theta recovery failed: got {result.theta:.4f}, "
            f"true={TRUE_THETA}, rel_err={rel_err:.2%}"
        )

    def test_mu_within_tolerance(self, estimator: OLSEstimator) -> None:
        """Recovered mu must be within ±1% of true value."""
        series = _simulate_ou(theta=TRUE_THETA, mu=TRUE_MU, sigma=TRUE_SIGMA)
        result = estimator.fit(series)
        rel_err = abs(result.mu - TRUE_MU) / abs(TRUE_MU)
        assert rel_err < 0.01, (
            f"mu recovery failed: got {result.mu:.4f}, "
            f"true={TRUE_MU}, rel_err={rel_err:.2%}"
        )

    def test_sigma_within_tolerance(self, estimator: OLSEstimator) -> None:
        """Recovered sigma must be within ±10% of true value."""
        series = _simulate_ou(theta=TRUE_THETA, mu=TRUE_MU, sigma=TRUE_SIGMA)
        result = estimator.fit(series)
        rel_err = abs(result.sigma - TRUE_SIGMA) / TRUE_SIGMA
        assert rel_err < 0.10, (
            f"sigma recovery failed: got {result.sigma:.4f}, "
            f"true={TRUE_SIGMA}, rel_err={rel_err:.2%}"
        )

    def test_half_life_positive(self, estimator: OLSEstimator) -> None:
        """half_life_days must be positive."""
        series = _simulate_ou()
        result = estimator.fit(series)
        assert result.half_life_days > 0.0

    def test_half_life_consistent_with_theta(self, estimator: OLSEstimator) -> None:
        """half_life_days must equal ln(2) / theta."""
        series = _simulate_ou()
        result = estimator.fit(series)
        expected_hl = float(np.log(2.0) / result.theta)
        assert abs(result.half_life_days - expected_hl) < 1e-9


# ---------------------------------------------------------------------------
# Result structure tests
# ---------------------------------------------------------------------------

class TestResultStructure:
    """Verify the returned OUResult is well-formed."""

    def test_returns_ou_result(self, estimator: OLSEstimator) -> None:
        series = _simulate_ou()
        result = estimator.fit(series)
        assert isinstance(result, OUResult)

    def test_method_name_is_ols(self, estimator: OLSEstimator) -> None:
        series = _simulate_ou()
        result = estimator.fit(series)
        assert result.method == "OLS"

    def test_converged_always_true(self, estimator: OLSEstimator) -> None:
        """OLS is closed-form; converged must always be True."""
        series = _simulate_ou()
        result = estimator.fit(series)
        assert result.converged is True

    def test_extra_contains_required_keys(self, estimator: OLSEstimator) -> None:
        series = _simulate_ou()
        result = estimator.fit(series)
        for key in ("ar1_coef", "ar1_pvalue", "r_squared", "n_obs"):
            assert key in result.extra, f"Missing key '{key}' in extra"

    def test_extra_n_obs_correct(self, estimator: OLSEstimator) -> None:
        n = 200
        series = _simulate_ou(n=n)
        result = estimator.fit(series)
        assert result.extra["n_obs"] == n


# ---------------------------------------------------------------------------
# OUEstimator interface compliance
# ---------------------------------------------------------------------------

class TestInterfaceCompliance:
    """OLSEstimator must satisfy the OUEstimator contract."""

    def test_method_name_property(self, estimator: OLSEstimator) -> None:
        assert isinstance(estimator.method_name, str)
        assert len(estimator.method_name) > 0

    def test_fit_returns_ou_result(self, estimator: OLSEstimator) -> None:
        series = _simulate_ou(n=50)
        assert isinstance(estimator.fit(series), OUResult)


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------

class TestErrorHandling:
    """OLSEstimator must raise appropriate errors on bad input."""

    def test_series_too_short_raises(self, estimator: OLSEstimator) -> None:
        with pytest.raises(ValueError, match="at least"):
            estimator.fit(np.array([1.0, 2.0, 3.0]))

    def test_series_with_nan_raises(self, estimator: OLSEstimator) -> None:
        series = _simulate_ou(n=100)
        series[10] = float("nan")
        with pytest.raises(ValueError, match="NaN or Inf"):
            estimator.fit(series)

    def test_negatively_autocorrelated_raises_fit_error(self, estimator: OLSEstimator) -> None:
        """A negatively autocorrelated series produces b < 0, outside (0,1).

        This is a deterministic edge-case: alternating +1/-1 always gives
        slope ≈ -1 from OLS, which must raise OUFitError (b must be in (0,1)).
        """
        # Deterministic alternating series: b will be exactly -1 from OLS.
        alternating = np.array([(-1.0) ** i for i in range(500)], dtype=np.float64)
        with pytest.raises(OUFitError):
            estimator.fit(alternating)
