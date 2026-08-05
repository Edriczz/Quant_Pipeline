"""Interpretation layer for the OU mean-reversion pipeline.

Pure functions that analyze an OUResult and StationarityResult to produce
human-readable interpretation and trading signals/verdicts.

No Streamlit or rendering dependencies here — pure logic suitable for
testing and reuse.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from ou_pipeline.models.results import OUResult, StationarityResult


def calculate_z_score(current_val: float, ou_result: OUResult) -> float:
    """Calculate the z-score of the current value relative to the OU equilibrium distribution.

    The asymptotic stationary standard deviation of an OU process is:
        sigma_eq = sigma / sqrt(2 * theta)

    The z-score is:
        z = (current_val - mu) / sigma_eq

    Args:
        current_val: Current price, log-price, or residual observation.
        ou_result: Fitted OU parameters.

    Returns:
        float: Z-score (number of asymptotic standard deviations from mean/baseline).

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
    mode: Literal["raw", "detrended"] = "raw",
    detrend_window: int = 20,
) -> str:
    """Build a plain-English interpretation verdict for the estimated series.

    Args:
        ou_result: Fitted OU parameters.
        stationarity_result: Result of the ADF test.
        current_val: Current value (log-price or residual depending on mode).
        mode: "raw" (fixed long-run mean) or "detrended" (moving baseline).
        detrend_window: Window size used when mode is "detrended".

    Returns:
        str: Concise interpretation string detailing stationarity status,
             distance from mean/baseline (z-score), and mean-reversion expectation.
    """
    p_val = stationarity_result.p_value
    hl = ou_result.half_life_days

    if mode == "detrended":
        if not stationarity_result.is_stationary:
            return (
                f"Mean-reversion around the {detrend_window}-day moving average is "
                f"NOT statistically supported (ADF p-value = {p_val:.4f} >= threshold). "
                f"The detrended residual exhibits unit-root behavior; OU estimates using "
                f"{ou_result.method} should be interpreted with caution."
            )

        z = calculate_z_score(current_val, ou_result)

        if z > 0.5:
            return (
                f"Series is stationary around its moving baseline (ADF p = {p_val:.4f}). "
                f"Current value is ABOVE its {detrend_window}-day moving average by "
                f"{z:+.2f} standard deviations. Mean-reversion implies expected downward "
                f"adjustment toward the local baseline (half-life: {hl:.1f} days)."
            )
        elif z < -0.5:
            return (
                f"Series is stationary around its moving baseline (ADF p = {p_val:.4f}). "
                f"Current value is BELOW its {detrend_window}-day moving average by "
                f"{z:+.2f} standard deviations. Mean-reversion implies expected upward "
                f"adjustment toward the local baseline (half-life: {hl:.1f} days)."
            )
        else:
            return (
                f"Series is stationary around its moving baseline (ADF p = {p_val:.4f}). "
                f"Current value is NEAR its {detrend_window}-day moving average "
                f"(z-score = {z:+.2f}). No strong directional deviation detected "
                f"(half-life: {hl:.1f} days)."
            )

    # Default "raw" mode
    if not stationarity_result.is_stationary:
        return (
            f"Mean-reversion is NOT statistically supported (ADF p-value = "
            f"{p_val:.4f} >= threshold). The series exhibits unit-root / trending "
            f"behavior; OU estimates using {ou_result.method} should be interpreted "
            f"with extreme caution."
        )

    z = calculate_z_score(current_val, ou_result)

    if z > 0.5:
        return (
            f"Series is stationary (ADF p = {p_val:.4f}). "
            f"Current value is ABOVE the fitted long-run mean by {z:+.2f} standard "
            f"deviations. Mean-reversion implies expected downward adjustment toward "
            f"mean (half-life: {hl:.1f} days)."
        )
    elif z < -0.5:
        return (
            f"Series is stationary (ADF p = {p_val:.4f}). "
            f"Current value is BELOW the fitted long-run mean by {z:+.2f} standard "
            f"deviations. Mean-reversion implies expected upward adjustment toward "
            f"mean (half-life: {hl:.1f} days)."
        )
    else:
        return (
            f"Series is stationary (ADF p = {p_val:.4f}). "
            f"Current value is NEAR the fitted long-run mean (z-score = {z:+.2f}). "
            f"No strong directional deviation detected (half-life: {hl:.1f} days)."
        )
