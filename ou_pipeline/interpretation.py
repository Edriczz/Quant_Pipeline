"""Interpretation layer for the OU mean-reversion pipeline.

Pure functions that analyze an OUResult and StationarityResult to produce
human-readable interpretation and trading signals/verdicts.

No Streamlit or rendering dependencies here — pure logic suitable for
testing and reuse.
"""

from __future__ import annotations

import numpy as np

from ou_pipeline.models.results import OUResult, StationarityResult


def calculate_z_score(current_val: float, ou_result: OUResult) -> float:
    """Calculate the z-score of the current value relative to the OU equilibrium distribution.

    The asymptotic stationary standard deviation of an OU process is:
        sigma_eq = sigma / sqrt(2 * theta)

    The z-score is:
        z = (current_val - mu) / sigma_eq

    Args:
        current_val: Current price or log-price observation.
        ou_result: Fitted OU parameters.

    Returns:
        float: Z-score (number of asymptotic standard deviations from mean).

    Raises:
        ValueError: If theta <= 0 or sigma <= 0.
    """
    if ou_result.theta <= 0.0:
        raise ValueError(f"theta must be strictly positive, got {ou_result.theta}")
    if ou_result.sigma <= 0.0:
        raise ValueError(f"sigma must be strictly positive, got {ou_result.sigma}")

    sigma_eq = ou_result.sigma / np.sqrt(2.0 * ou_result.theta)
    return float((current_val - ou_result.mu) / sigma_eq)


def build_verdict(
    ou_result: OUResult,
    stationarity_result: StationarityResult,
    current_val: float,
) -> str:
    """Build a plain-English interpretation verdict for the estimated series.

    Args:
        ou_result: Fitted OU parameters.
        stationarity_result: Result of the ADF test.
        current_val: Current value (log-price or price depending on series used).

    Returns:
        str: Concise interpretation string detailing stationarity status,
             distance from mean (z-score), and mean-reversion expectation.
    """
    if not stationarity_result.is_stationary:
        return (
            f"Mean-reversion is NOT statistically supported (ADF p-value = "
            f"{stationarity_result.p_value:.4f} >= threshold). The series "
            f"exhibits unit-root / trending behavior; OU estimates using {ou_result.method} "
            f"should be interpreted with extreme caution."
        )

    z = calculate_z_score(current_val, ou_result)
    hl = ou_result.half_life_days

    if z > 0.5:
        return (
            f"Series is stationary (ADF p = {stationarity_result.p_value:.4f}). "
            f"Current value is ABOVE the fitted long-run mean by {z:+.2f} standard deviations. "
            f"Mean-reversion implies expected downward adjustment toward mean "
            f"(half-life: {hl:.1f} days)."
        )
    elif z < -0.5:
        return (
            f"Series is stationary (ADF p = {stationarity_result.p_value:.4f}). "
            f"Current value is BELOW the fitted long-run mean by {z:+.2f} standard deviations. "
            f"Mean-reversion implies expected upward adjustment toward mean "
            f"(half-life: {hl:.1f} days)."
        )
    else:
        return (
            f"Series is stationary (ADF p = {stationarity_result.p_value:.4f}). "
            f"Current value is NEAR the fitted long-run mean (z-score = {z:+.2f}). "
            f"No strong directional deviation detected (half-life: {hl:.1f} days)."
        )
