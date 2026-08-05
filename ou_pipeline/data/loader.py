"""Price data loading for the OU pipeline.

The single entry point for market data is :class:`PriceDataLoader`.
Never call ``yf.download`` directly from estimators, the app, or tests —
always go through this class.  This makes it trivial to substitute
synthetic data in tests without hitting the network.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from ou_pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)


class PriceDataLoader:
    """Fetches and pre-processes a single-asset price series.

    Args:
        ticker: The ticker symbol to download (e.g. ``"ASML"``).
        config: Pipeline configuration.  If omitted, a default
            :class:`~ou_pipeline.config.PipelineConfig` is used.

    Example::

        loader = PriceDataLoader("ASML")
        df = loader.load()          # DataFrame with 'price' column
        log_s = loader.log_series() # np.ndarray of log prices
    """

    def __init__(
        self,
        ticker: str,
        config: Optional[PipelineConfig] = None,
    ) -> None:
        self._ticker = ticker
        self._config = config or PipelineConfig()
        self._df: Optional[pd.DataFrame] = None  # cached after first load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> pd.DataFrame:
        """Download (or return cached) the adjusted close price series.

        Returns:
            A :class:`pandas.DataFrame` with:
            - A :class:`pandas.DatetimeIndex` sorted ascending.
            - A single column named ``"price"`` (float64).
            - No ``NaN`` values (rows with missing data are dropped).

        Raises:
            ValueError: If the downloaded data is empty or contains no
                valid price observations after cleaning.
        """
        if self._df is not None:
            return self._df

        logger.debug("Downloading %s (period=%s)", self._ticker, self._config.period)
        raw: pd.DataFrame = yf.download(
            self._ticker,
            period=self._config.period,
            auto_adjust=True,
            progress=False,
        )

        self._df = self._clean(raw)
        return self._df

    def log_series(self) -> np.ndarray:
        """Return the natural log of the price series as a 1-D array.

        Calls :meth:`load` internally if the data has not been fetched yet.

        Returns:
            A ``numpy.ndarray`` of shape ``(n,)`` containing
            ``ln(price)``, where *n* is the number of trading days.
        """
        df = self.load()
        return np.log(df["price"].to_numpy(dtype=np.float64))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean(raw: pd.DataFrame) -> pd.DataFrame:
        """Normalise the raw yfinance output to a clean price DataFrame.

        Args:
            raw: The raw multi-column DataFrame returned by
                ``yf.download``.

        Returns:
            Cleaned single-column DataFrame with column ``"price"``.

        Raises:
            ValueError: If *raw* is empty or has no usable price column.
        """
        if raw.empty:
            raise ValueError("yfinance returned an empty DataFrame — check the ticker.")

        # yfinance returns either 'Close' or 'Adj Close'; after auto_adjust=True
        # the adjusted close is in 'Close'.
        if "Close" in raw.columns:
            price_col = raw["Close"]
        elif "Adj Close" in raw.columns:
            price_col = raw["Adj Close"]
        else:
            raise ValueError(
                f"Cannot find a price column in yfinance output.  "
                f"Available columns: {list(raw.columns)}"
            )

        # If yfinance returns a MultiIndex (multiple tickers), flatten.
        if isinstance(price_col, pd.DataFrame):
            price_col = price_col.iloc[:, 0]

        df = pd.DataFrame({"price": price_col})
        df = df.dropna()
        df = df.sort_index()

        if df.empty:
            raise ValueError("Price series is empty after dropping NaN values.")

        return df
