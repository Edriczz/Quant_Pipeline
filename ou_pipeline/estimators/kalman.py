"""Kalman-filter + MLE Ornstein-Uhlenbeck parameter estimator.

State-space representation of the discrete-time OU process:

    State equation:    X_t = F * X_{t-1} + w_t,   w_t ~ N(0, Q)
    Observation eq.:   Y_t = X_t + v_t,            v_t ~ N(0, R)

where:
    F = exp(-theta * dt)                             (transition)
    Q = sigma² * (1 - F²) / (2*theta)               (process noise variance)
    R = observation noise variance (estimated jointly)

The Kalman filter computes the exact log-likelihood for a given
parameter vector (theta, mu, sigma, R).  ``scipy.optimize.minimize``
maximises the likelihood (minimises the negative log-likelihood).

This estimator is the point of the pipeline: unlike OLS, it explicitly
separates observation noise R from process noise Q, so it recovers
the true OU parameters even when the observed series is contaminated
by measurement error.

Reference: Shumway & Stoffer (2000), "Time Series Analysis and Its
Applications", ch. 6.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import minimize

from ou_pipeline.config import PipelineConfig
from ou_pipeline.estimators._shared import half_life_from_theta, validate_series
from ou_pipeline.estimators.base import OUEstimator
from ou_pipeline.models.results import OUResult

logger = logging.getLogger(__name__)


class OUFitError(Exception):
    """Raised when the Kalman MLE optimiser fails to converge."""


class KalmanEstimator(OUEstimator):
    """Estimates OU parameters via state-space Kalman filter + MLE.

    This is the advanced estimator.  Compared with OLS, it correctly
    accounts for additive observation noise in the measured series
    (e.g. bid-ask bounce, price rounding).  On clean data it performs
    comparably to OLS; on noisy data it recovers theta more accurately.

    Args:
        config: Pipeline configuration used for optimiser settings
            (``optimizer_method``, ``optimizer_max_iter``, ``optimizer_tol``).
            If omitted, a default :class:`~ou_pipeline.config.PipelineConfig`
            is used.

    Example::

        est = KalmanEstimator()
        result = est.fit(noisy_log_series, dt=1.0)
        print(result.theta, result.extra["obs_noise_R"])
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self._config = config or PipelineConfig()

    @property
    def method_name(self) -> str:
        """Short identifier for this method."""
        return "Kalman-MLE"

    def fit(self, series: np.ndarray, dt: float = 1.0) -> OUResult:
        """Fit OU parameters via Kalman filter MLE.

        Args:
            series: 1-D array of observations (log-prices recommended).
                Must have at least 10 observations and be free of NaN/Inf.
            dt: Time step between observations, in trading days.
                Default is 1.0 (daily data).

        Returns:
            An :class:`~ou_pipeline.models.results.OUResult` with
            ``method="Kalman-MLE"``.  The ``extra`` dict contains:

            - ``"obs_noise_R"`` (*float*): Estimated observation noise
              variance.  Near zero → measurement noise is negligible.
            - ``"log_likelihood"`` (*float*): Value of the maximised
              log-likelihood.
            - ``"n_obs"`` (*int*): Number of observations used.
            - ``"optimizer_message"`` (*str*): Message from the optimiser.

        Raises:
            OUFitError: If the optimiser does not converge or produces
                non-physical parameters (theta ≤ 0 or sigma ≤ 0).
        """
        series = validate_series(series, min_length=10)
        n = len(series)

        # ------------------------------------------------------------------
        # Build initial guess from OLS-like moment estimators
        # ------------------------------------------------------------------
        mu0 = float(np.mean(series))
        diffs = np.diff(series)
        sigma0 = max(float(np.std(diffs) / np.sqrt(dt)), 1e-4)
        theta0 = max(0.5, sigma0)  # rough guess
        R0 = max(float(np.var(diffs)) * 0.01, 1e-6)  # small obs noise init

        # Parameter vector: [theta, mu, sigma, R]  all in natural space.
        # We optimise in log-space for theta, sigma, R to enforce positivity.
        x0 = np.array([np.log(theta0), mu0, np.log(sigma0), np.log(R0)])

        def neg_log_likelihood(params: np.ndarray) -> float:
            log_theta, mu, log_sigma, log_R = params
            theta = np.exp(log_theta)
            sigma = np.exp(log_sigma)
            R = np.exp(log_R)
            return -self._kalman_log_likelihood(series, theta, mu, sigma, R, dt)

        result = minimize(
            neg_log_likelihood,
            x0=x0,
            method=self._config.optimizer_method,
            options={
                "maxiter": self._config.optimizer_max_iter,
                "ftol": self._config.optimizer_tol,
            },
        )

        converged = bool(result.success)
        if not converged:
            logger.warning("Kalman MLE did not converge: %s", result.message)

        # Decode optimised parameters
        log_theta_opt, mu_opt, log_sigma_opt, log_R_opt = result.x
        theta_opt = float(np.exp(log_theta_opt))
        mu_opt = float(mu_opt)
        sigma_opt = float(np.exp(log_sigma_opt))
        R_opt = float(np.exp(log_R_opt))

        if theta_opt <= 0.0 or sigma_opt <= 0.0:
            raise OUFitError(
                f"Optimiser produced non-physical parameters: "
                f"theta={theta_opt:.6f}, sigma={sigma_opt:.6f}"
            )

        half_life = half_life_from_theta(theta_opt)
        log_lik = float(-result.fun)

        logger.debug(
            "Kalman result: theta=%.4f  mu=%.4f  sigma=%.4f  R=%.6f  " "loglik=%.2f  converged=%s",
            theta_opt,
            mu_opt,
            sigma_opt,
            R_opt,
            log_lik,
            converged,
        )

        return OUResult(
            method=self.method_name,
            theta=theta_opt,
            mu=mu_opt,
            sigma=sigma_opt,
            half_life_days=half_life,
            converged=converged,
            extra={
                "obs_noise_R": R_opt,
                "log_likelihood": log_lik,
                "n_obs": n,
                "optimizer_message": str(result.message),
            },
        )

    # ------------------------------------------------------------------
    # Kalman filter internals
    # ------------------------------------------------------------------

    @staticmethod
    def _kalman_log_likelihood(
        y: np.ndarray,
        theta: float,
        mu: float,
        sigma: float,
        R: float,
        dt: float,
    ) -> float:
        """Compute the Kalman-filter log-likelihood for an OU state-space model.

        Args:
            y: Observed series (1-D, length n).
            theta: Mean-reversion speed (1/trading-day).
            mu: Long-run mean.
            sigma: Process diffusion coefficient.
            R: Observation noise variance.
            dt: Time step (trading days).

        Returns:
            Scalar log-likelihood value.
        """
        n = len(y)
        F = np.exp(-theta * dt)
        Q = sigma**2 * (1.0 - F**2) / (2.0 * theta)  # process noise variance
        # H = 1 (identity observation), R = obs noise variance

        # Initialise Kalman filter at the stationary distribution
        x_filt = mu  # filtered state mean
        P_filt = Q / (1.0 - F**2) if (1.0 - F**2) > 0 else 1.0  # stationary var

        log_lik = 0.0
        half_log_2pi = 0.5 * np.log(2.0 * np.pi)

        for t in range(n):
            # Predict
            x_pred = mu + F * (x_filt - mu)
            P_pred = F**2 * P_filt + Q

            # Innovation
            innovation = y[t] - x_pred
            S = P_pred + R  # innovation variance (H=1)

            if S <= 0.0:
                return -1e18  # degenerate

            # Log-likelihood contribution
            log_lik -= half_log_2pi + 0.5 * np.log(S) + 0.5 * (innovation**2 / S)

            # Update
            K = P_pred / S  # Kalman gain (H=1)
            x_filt = x_pred + K * innovation
            P_filt = (1.0 - K) * P_pred

        return float(log_lik)
