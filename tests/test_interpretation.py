"""Tests for ou_pipeline.interpretation module.

Covers:
1. calculate_z_score correctness and input validation.
2. build_verdict branches for both "raw" and "detrended" modes.
3. Raw fails / detrended passes combination behavior.
"""

from __future__ import annotations

import pytest

from ou_pipeline.interpretation import build_verdict, calculate_z_score
from ou_pipeline.models.results import OUResult, StationarityResult


@pytest.fixture
def stationary_result() -> StationarityResult:
    return StationarityResult(adf_statistic=-3.5, p_value=0.01, is_stationary=True)


@pytest.fixture
def non_stationary_result() -> StationarityResult:
    return StationarityResult(adf_statistic=-1.2, p_value=0.65, is_stationary=False)


@pytest.fixture
def ou_result() -> OUResult:
    return OUResult(
        method="OLS",
        theta=0.5,
        mu=5.0,
        sigma=0.2,
        half_life_days=1.386,
        converged=True,
        extra={},
    )


@pytest.fixture
def ou_result_residual() -> OUResult:
    return OUResult(
        method="Kalman-MLE",
        theta=1.2,
        mu=0.0,
        sigma=0.1,
        half_life_days=0.578,
        converged=True,
        extra={},
    )


class TestCalculateZScore:
    def test_z_score_at_mean(self, ou_result: OUResult) -> None:
        """When current_val == mu, z-score must be 0.0."""
        z = calculate_z_score(5.0, ou_result)
        assert abs(z) < 1e-9

    def test_z_score_above_mean(self, ou_result: OUResult) -> None:
        """sigma_eq = 0.2 / sqrt(1.0) = 0.2. For current_val = 5.2, z should be +1.0."""
        z = calculate_z_score(5.2, ou_result)
        assert abs(z - 1.0) < 1e-6

    def test_z_score_below_mean(self, ou_result: OUResult) -> None:
        """For current_val = 4.8, z should be -1.0."""
        z = calculate_z_score(4.8, ou_result)
        assert abs(z - (-1.0)) < 1e-6

    def test_invalid_theta_raises(self) -> None:
        bad_ou = OUResult(
            method="OLS", theta=0.0, mu=5.0, sigma=0.2, half_life_days=0.0, converged=True, extra={}
        )
        with pytest.raises(ValueError, match="theta must be strictly positive"):
            calculate_z_score(5.0, bad_ou)

    def test_invalid_sigma_raises(self) -> None:
        bad_ou = OUResult(
            method="OLS",
            theta=0.5,
            mu=5.0,
            sigma=0.0,
            half_life_days=1.386,
            converged=True,
            extra={},
        )
        with pytest.raises(ValueError, match="sigma must be strictly positive"):
            calculate_z_score(5.0, bad_ou)


class TestBuildVerdictRaw:
    def test_not_stationary_branch(
        self, ou_result: OUResult, non_stationary_result: StationarityResult
    ) -> None:
        """Non-stationary ADF result must trigger the warning branch."""
        verdict = build_verdict(ou_result, non_stationary_result, current_val=5.5, mode="raw")
        assert "NOT statistically supported" in verdict
        assert "0.6500" in verdict

    def test_stationary_above_mean_branch(
        self, ou_result: OUResult, stationary_result: StationarityResult
    ) -> None:
        """Stationary + z > 0.5 must indicate ABOVE mean and expected downward adjustment."""
        verdict = build_verdict(
            ou_result, stationary_result, current_val=5.3, mode="raw"
        )  # z = +1.5
        assert "ABOVE" in verdict
        assert "downward adjustment" in verdict

    def test_stationary_below_mean_branch(
        self, ou_result: OUResult, stationary_result: StationarityResult
    ) -> None:
        """Stationary + z < -0.5 must indicate BELOW mean and expected upward adjustment."""
        verdict = build_verdict(
            ou_result, stationary_result, current_val=4.7, mode="raw"
        )  # z = -1.5
        assert "BELOW" in verdict
        assert "upward adjustment" in verdict

    def test_stationary_near_mean_branch(
        self, ou_result: OUResult, stationary_result: StationarityResult
    ) -> None:
        """Stationary + |z| <= 0.5 must indicate NEAR mean."""
        verdict = build_verdict(
            ou_result, stationary_result, current_val=5.02, mode="raw"
        )  # z = +0.1
        assert "NEAR" in verdict
        assert "No strong directional deviation" in verdict


class TestBuildVerdictDetrended:
    def test_detrended_not_stationary_branch(
        self, ou_result_residual: OUResult, non_stationary_result: StationarityResult
    ) -> None:
        verdict = build_verdict(
            ou_result_residual,
            non_stationary_result,
            current_val=0.2,
            mode="detrended",
            detrend_window=20,
        )
        assert "20-day moving average is NOT statistically supported" in verdict

    def test_detrended_stationary_above_baseline(
        self, ou_result_residual: OUResult, stationary_result: StationarityResult
    ) -> None:
        # sigma_eq = 0.1 / sqrt(2.4) = 0.0645. For current_val = 0.1, z = +1.54
        verdict = build_verdict(
            ou_result_residual,
            stationary_result,
            current_val=0.1,
            mode="detrended",
            detrend_window=20,
        )
        assert "ABOVE its 20-day moving average" in verdict
        assert "local baseline" in verdict

    def test_detrended_stationary_below_baseline(
        self, ou_result_residual: OUResult, stationary_result: StationarityResult
    ) -> None:
        verdict = build_verdict(
            ou_result_residual,
            stationary_result,
            current_val=-0.1,
            mode="detrended",
            detrend_window=20,
        )
        assert "BELOW its 20-day moving average" in verdict
        assert "local baseline" in verdict

    def test_raw_fails_but_detrended_passes_combination(
        self,
        ou_result: OUResult,
        ou_result_residual: OUResult,
        non_stationary_result: StationarityResult,
        stationary_result: StationarityResult,
    ) -> None:
        """Verify the exact situation this extension exists to handle."""
        raw_verdict = build_verdict(ou_result, non_stationary_result, current_val=5.5, mode="raw")
        detrended_verdict = build_verdict(
            ou_result_residual,
            stationary_result,
            current_val=0.1,
            mode="detrended",
            detrend_window=20,
        )

        assert "NOT statistically supported" in raw_verdict
        assert "stationary around its moving baseline" in detrended_verdict
