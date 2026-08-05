"""OLS-based Ornstein-Uhlenbeck parameter estimator.

The AR(1) regression approach:

    X_{t+1} = a + b * X_t + ε_t

where the OU continuous-time parameters are recovered as:

    theta  = -ln(b) / dt
    mu     = a / (1 - b)
    sigma  = std(residuals) * sqrt(-2 * ln(b) / (dt * (1 - b²)))

This is a closed-form estimator — no numerical optimisation is required,
so ``converged`` is always ``True``.

Reference: Hamilton (1994), "Time Series Analysis", ch. 17.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import stats

from ou_pipeline.estimators._shared import half_life_from_theta, validate_series
from ou_pipeline.estimators.base import OUEstimator
from ou_pipeline.models.results import OUResult

logger = logging.getLogger(__name__)


class OUFitError(Exception):
    """Raised when the OLS estimator cannot produce valid parameters."""


class OLSEstimator(OUEstimator):
    """Estimates OU parameters via AR(1) OLS regression.

    This is the baseline estimator.  It makes no distinction between
    true process noise and observation noise, so it will be biased
    (underestimate theta) when the observed series contains additive
    measurement error.  Use :class:`~ou_pipeline.estimators.kalman.KalmanEstimator`
    when measurement noise is expected to be significant.

    Example::

        est = OLSEstimator()
        result = est.fit(log_series, dt=1.0)
        print(result.theta, result.mu, result.half_life_days)
    """

    @property
    def method_name(self) -> str:
        """Short identifier for this method."""
        return "OLS"

    def fit(self, series: np.ndarray, dt: float = 1.0) -> OUResult:
        """Fit OU parameters to *series* using AR(1) OLS regression.

        Args:
            series: 1-D array of observations (log-prices recommended).
                Must have at least 10 observations and be free of NaN/Inf.
            dt: Time step between observations, in trading days.
                Default is 1.0 (daily data).

        Returns:
            An :class:`~ou_pipeline.models.results.OUResult` with
            ``method="OLS"``.  The ``extra`` dict contains:

            - ``"ar1_coef"`` (*float*): The AR(1) slope coefficient *b*.
            - ``"ar1_pvalue"`` (*float*): Two-sided p-value for *b* ≠ 0.
            - ``"r_squared"`` (*float*): R² of the AR(1) regression.
            - ``"n_obs"`` (*int*): Number of observations used.

        Raises:
            OUFitError: If the AR(1) slope *b* is not in (0, 1), which
                would make ln(b) undefined or produce negative theta.
        """
        series = validate_series(series, min_length=10)

        x_t = series[:-1]   # X_t
        x_t1 = series[1:]   # X_{t+1}

        # OLS: X_{t+1} = a + b * X_t + ε
        slope, intercept, r_value, p_value, _ = stats.linregress(x_t, x_t1)
        b: float = float(slope)
        a: float = float(intercept)

        logger.debug("OLS AR(1): b=%.6f  a=%.6f  p=%.4e", b, a, p_value)

        # Validate that b is in the mean-reverting range (0, 1)
        if b <= 0.0 or b >= 1.0:
            raise OUFitError(
                f"AR(1) coefficient b={b:.6f} is outside (0, 1).  "
                "The series does not appear to be mean-reverting under this model."
            )

        # Recover continuous-time OU parameters
        ln_b = np.log(b)
        theta = float(-ln_b / dt)
        mu = float(a / (1.0 - b))

        # Residual std → sigma of continuous process
        residuals = x_t1 - (a + b * x_t)
        sigma_ar = float(np.std(residuals, ddof=2))
        sigma = float(sigma_ar * np.sqrt(-2.0 * ln_b / (dt * (1.0 - b**2))))

        half_life = half_life_from_theta(theta)

        logger.debug(
            "OLS result: theta=%.4f  mu=%.4f  sigma=%.4f  half_life=%.2f days",
            theta, mu, sigma, half_life,
        )

        return OUResult(
            method=self.method_name,
            theta=theta,
            mu=mu,
            sigma=sigma,
            half_life_days=half_life,
            converged=True,
            extra={
                "ar1_coef": b,
                "ar1_pvalue": float(p_value),
                "r_squared": float(r_value**2),
                "n_obs": len(series),
            },
        )
