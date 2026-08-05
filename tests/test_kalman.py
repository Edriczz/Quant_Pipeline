"""Tests for ou_pipeline.estimators.kalman.KalmanEstimator.

Two critical tests required by the Phase 6 gate:

1. Noiseless case: Kalman recovers params at least as well as OLS
   (sanity check — it shouldn't be worse in the easy case).

2. Noisy case: Kalman recovers theta CLOSER to the true value than OLS
   on the SAME noisy series. This is the proof-of-value test that
   justifies the Kalman estimator's existence.

Both tests use fixed seeds for full reproducibility.
"""

from __future__ import annotations

import numpy as np
import pytest

from ou_pipeline.config import PipelineConfig
from ou_pipeline.estimators.kalman import KalmanEstimator
from ou_pipeline.estimators.ols import OLSEstimator
from ou_pipeline.models.results import OUResult


# ---------------------------------------------------------------------------
# Shared synthetic data generators
# ---------------------------------------------------------------------------

def _simulate_ou(
    n: int = 2_000,
    theta: float = 1.5,
    mu: float = 5.0,
    sigma: float = 0.3,
    dt: float = 1.0,
    seed: int = 1,
) -> np.ndarray:
    """Simulate a noiseless OU path using the exact transition formula."""
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[0] = mu
    e_decay = np.exp(-theta * dt)
    std = sigma * np.sqrt((1.0 - e_decay**2) / (2.0 * theta))
    noise = rng.normal(0.0, std, size=n - 1)
    for t in range(1, n):
        x[t] = mu + e_decay * (x[t - 1] - mu) + noise[t - 1]
    return x


def _add_observation_noise(
    clean: np.ndarray,
    noise_std: float = 0.4,
    seed: int = 2,
) -> np.ndarray:
    """Add iid Gaussian measurement noise to a clean series."""
    rng = np.random.default_rng(seed)
    return clean + rng.normal(0.0, noise_std, size=len(clean))


# ---------------------------------------------------------------------------
# Ground-truth parameters
# ---------------------------------------------------------------------------

TRUE_THETA = 1.5
TRUE_MU = 5.0
TRUE_SIGMA = 0.3
OBS_NOISE_STD = 0.4   # observation noise std injected in the noisy test


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def kalman() -> KalmanEstimator:
    return KalmanEstimator(config=PipelineConfig())


@pytest.fixture
def ols() -> OLSEstimator:
    return OLSEstimator()


# ---------------------------------------------------------------------------
# Phase 6 gate: Test 1 — Noiseless sanity check
# ---------------------------------------------------------------------------

class TestNoiselessRecovery:
    """Kalman should recover params as well as OLS on a clean OU path."""

    def test_theta_reasonable(self, kalman: KalmanEstimator) -> None:
        """Recovered theta must be within ±30% of true on clean data."""
        series = _simulate_ou(theta=TRUE_THETA, mu=TRUE_MU, sigma=TRUE_SIGMA)
        result = kalman.fit(series)
        rel_err = abs(result.theta - TRUE_THETA) / TRUE_THETA
        assert rel_err < 0.30, (
            f"Kalman theta (noiseless) too far off: got {result.theta:.4f}, "
            f"true={TRUE_THETA}, rel_err={rel_err:.2%}"
        )

    def test_mu_reasonable(self, kalman: KalmanEstimator) -> None:
        """Recovered mu must be within ±2% of true on clean data."""
        series = _simulate_ou(theta=TRUE_THETA, mu=TRUE_MU, sigma=TRUE_SIGMA)
        result = kalman.fit(series)
        rel_err = abs(result.mu - TRUE_MU) / abs(TRUE_MU)
        assert rel_err < 0.02, (
            f"Kalman mu (noiseless) too far off: got {result.mu:.4f}, "
            f"true={TRUE_MU}, rel_err={rel_err:.2%}"
        )

    def test_obs_noise_near_zero_on_clean_data(self, kalman: KalmanEstimator) -> None:
        """On clean (noiseless) data, estimated obs_noise_R must be small."""
        series = _simulate_ou(theta=TRUE_THETA, mu=TRUE_MU, sigma=TRUE_SIGMA)
        result = kalman.fit(series)
        # R should be very small relative to process noise sigma^2/(2*theta)
        process_noise_scale = TRUE_SIGMA**2 / (2.0 * TRUE_THETA)
        assert result.extra["obs_noise_R"] < process_noise_scale, (
            f"obs_noise_R={result.extra['obs_noise_R']:.6f} is suspiciously large "
            f"on noiseless data (process noise scale={process_noise_scale:.4f})"
        )


# ---------------------------------------------------------------------------
# Phase 6 gate: Test 2 — The key comparative test on noisy data
# ---------------------------------------------------------------------------

class TestNoisyComparison:
    """On noisy data, Kalman must recover theta closer to truth than OLS.

    This is the whole point of the Kalman estimator — the test suite must
    prove it earns its complexity.
    """

    def test_kalman_beats_ols_on_noisy_theta(
        self, kalman: KalmanEstimator, ols: OLSEstimator
    ) -> None:
        """abs(kalman_theta - true) < abs(ols_theta - true) on noisy series."""
        clean = _simulate_ou(n=2_000, theta=TRUE_THETA, mu=TRUE_MU, sigma=TRUE_SIGMA, seed=1)
        noisy = _add_observation_noise(clean, noise_std=OBS_NOISE_STD, seed=2)

        ols_result = ols.fit(noisy)
        kalman_result = kalman.fit(noisy)

        ols_err = abs(ols_result.theta - TRUE_THETA)
        kalman_err = abs(kalman_result.theta - TRUE_THETA)

        assert kalman_err < ols_err, (
            f"Kalman did NOT outperform OLS on noisy data.\n"
            f"  True theta:    {TRUE_THETA:.4f}\n"
            f"  OLS theta:     {ols_result.theta:.4f}  (err={ols_err:.4f})\n"
            f"  Kalman theta:  {kalman_result.theta:.4f}  (err={kalman_err:.4f})\n"
            f"  Kalman obs_R:  {kalman_result.extra['obs_noise_R']:.6f}\n"
            "Check the Kalman implementation — it should separate obs noise."
        )

    def test_kalman_obs_noise_approximately_correct(
        self, kalman: KalmanEstimator
    ) -> None:
        """Kalman should estimate obs_noise_R near the true noise variance."""
        clean = _simulate_ou(n=2_000, theta=TRUE_THETA, mu=TRUE_MU, sigma=TRUE_SIGMA, seed=1)
        noisy = _add_observation_noise(clean, noise_std=OBS_NOISE_STD, seed=2)

        result = kalman.fit(noisy)
        true_R = OBS_NOISE_STD**2  # true obs noise variance = 0.16
        # Accept within a factor of 3 (MLE can be noisy on finite samples)
        assert result.extra["obs_noise_R"] < true_R * 3.0, (
            f"obs_noise_R={result.extra['obs_noise_R']:.4f} far exceeds "
            f"true_R={true_R:.4f} × 3"
        )
        assert result.extra["obs_noise_R"] > true_R / 100.0, (
            f"obs_noise_R={result.extra['obs_noise_R']:.4f} is suspiciously tiny "
            f"(true_R={true_R:.4f} / 100 = {true_R/100:.4f})"
        )


# ---------------------------------------------------------------------------
# Result structure tests
# ---------------------------------------------------------------------------

class TestResultStructure:
    """Verify the returned OUResult is well-formed."""

    def test_returns_ou_result(self, kalman: KalmanEstimator) -> None:
        series = _simulate_ou(n=200)
        assert isinstance(kalman.fit(series), OUResult)

    def test_method_name_is_kalman_mle(self, kalman: KalmanEstimator) -> None:
        series = _simulate_ou(n=200)
        assert kalman.fit(series).method == "Kalman-MLE"

    def test_half_life_positive(self, kalman: KalmanEstimator) -> None:
        series = _simulate_ou(n=200)
        assert kalman.fit(series).half_life_days > 0.0

    def test_extra_contains_required_keys(self, kalman: KalmanEstimator) -> None:
        series = _simulate_ou(n=200)
        result = kalman.fit(series)
        for key in ("obs_noise_R", "log_likelihood", "n_obs", "optimizer_message"):
            assert key in result.extra, f"Missing '{key}' in extra"

    def test_half_life_consistent_with_theta(self, kalman: KalmanEstimator) -> None:
        series = _simulate_ou(n=200)
        result = kalman.fit(series)
        expected = float(np.log(2.0) / result.theta)
        assert abs(result.half_life_days - expected) < 1e-9

    def test_method_name_property(self, kalman: KalmanEstimator) -> None:
        assert isinstance(kalman.method_name, str)
        assert kalman.method_name == "Kalman-MLE"


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_too_short_raises(self, kalman: KalmanEstimator) -> None:
        with pytest.raises(ValueError, match="at least"):
            kalman.fit(np.array([1.0, 2.0]))

    def test_nan_raises(self, kalman: KalmanEstimator) -> None:
        series = _simulate_ou(n=50)
        series[5] = float("nan")
        with pytest.raises(ValueError, match="NaN or Inf"):
            kalman.fit(series)
