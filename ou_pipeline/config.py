"""Pipeline-wide configuration.

All defaults and constants live here as a single frozen dataclass.
Pass a ``PipelineConfig`` instance explicitly into every loader,
estimator, and tester — never read module-level globals elsewhere.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PipelineConfig:
    """Central configuration for the OU mean-reversion pipeline.

    Attributes:
        ticker: Default ticker symbol to analyse.
        period: yfinance lookback period string (e.g. ``"2y"``).
        dt: Time step between observations, in trading days.
        adf_alpha: Significance level for the ADF stationarity test.
        use_detrending: Whether to apply rolling-mean detrending prior to fitting.
        detrend_window: Window size (in trading days) for rolling-mean detrending.
        optimizer_max_iter: Maximum iterations for numerical optimisers
            (used by KalmanEstimator's MLE step).
        optimizer_tol: Convergence tolerance for numerical optimisers.
        optimizer_method: scipy.optimize method name (e.g. ``"L-BFGS-B"``).
    """

    ticker: str = "ASML"
    period: str = "2y"
    dt: float = 1.0  # 1 trading day
    adf_alpha: float = 0.05
    use_detrending: bool = False
    detrend_window: int = 20
    optimizer_max_iter: int = 1_000
    optimizer_tol: float = 1e-8
    optimizer_method: str = "L-BFGS-B"
    optimizer_options: dict[str, Any] = field(
        default_factory=lambda: {"maxiter": 1_000, "ftol": 1e-8}
    )
