"""Typed result dataclasses for the OU pipeline.

No function in ``ou_pipeline/`` returns a raw dict or tuple for a
"result".  Every output is wrapped in one of these frozen dataclasses
so the Streamlit layer uses attribute access (``result.theta``) rather
than dict-key guessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class OUResult:
    """Estimated Ornstein-Uhlenbeck parameters from one estimation method.

    Attributes:
        method: Short identifier for the estimation method (e.g. ``"OLS"``,
            ``"Kalman-MLE"``).
        theta: Mean-reversion speed, in units of 1/trading-day.
            A larger value means faster reversion.
        mu: Long-run mean (equilibrium level) of the process, in the
            same units as the input series (typically log-price).
        sigma: Instantaneous volatility of the process (diffusion
            coefficient), in units of 1/sqrt(trading-day).
        half_life_days: Half-life of mean reversion in trading days,
            derived as ``ln(2) / theta``.
        converged: Whether the fitting procedure converged successfully.
            Always ``True`` for OLS (closed-form); may be ``False`` for
            numerical methods if the optimiser did not reach tolerance.
        extra: Method-specific supplementary fields.
            OLS stores ``{"ar1_coef": float, "ar1_pvalue": float}``.
            Kalman stores ``{"obs_noise_R": float, "log_likelihood": float}``.
    """

    method: str
    theta: float
    mu: float
    sigma: float
    half_life_days: float
    converged: bool
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StationarityResult:
    """Result of an Augmented Dickey-Fuller stationarity test.

    Attributes:
        adf_statistic: The ADF test statistic (more negative → more
            evidence against a unit root).
        p_value: MacKinnon's approximate p-value for the test statistic.
        is_stationary: ``True`` when ``p_value < alpha`` used at test time.
    """

    adf_statistic: float
    p_value: float
    is_stationary: bool  # p_value < alpha


@dataclass(frozen=True)
class DetrendResult:
    """Result of a series detrending transformation.

    Attributes:
        residual: Detrended 1-D array (original series minus baseline).
        baseline: Trend / baseline 1-D array of the same shape as residual.
    """

    residual: np.ndarray
    baseline: np.ndarray
