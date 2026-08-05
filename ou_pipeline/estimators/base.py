"""Abstract base class for all OU parameter estimation methods.

Every concrete estimator (OLS, Kalman, future methods) must subclass
:class:`OUEstimator` and implement :meth:`fit` and the :attr:`method_name`
property.  This strategy pattern lets the app and tests treat all
estimators interchangeably.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ou_pipeline.models.results import OUResult


class OUEstimator(ABC):
    """Common interface for all OU parameter estimation methods.

    Subclasses must implement:
    - :meth:`fit` — fit OU parameters and return an :class:`~ou_pipeline.models.results.OUResult`.
    - :attr:`method_name` — a short string identifier for the method.

    Each subclass is independently instantiable and has no dependency on
    Streamlit, yfinance, or any other estimator subclass.
    """

    @abstractmethod
    def fit(self, series: np.ndarray, dt: float = 1.0) -> OUResult:
        """Fit OU parameters to a price or log-price series.

        Args:
            series: 1-D array of observations (typically log-prices).
                Must be free of NaN/Inf values.
            dt: Time step between observations, in trading days.
                Defaults to 1.0 (daily data).

        Returns:
            An :class:`~ou_pipeline.models.results.OUResult` dataclass
            containing ``theta``, ``mu``, ``sigma``, ``half_life_days``,
            ``converged``, and method-specific ``extra`` fields.
        """
        ...

    @property
    @abstractmethod
    def method_name(self) -> str:
        """Short identifier for this estimation method.

        Returns:
            A string such as ``"OLS"`` or ``"Kalman-MLE"``.
        """
        ...
