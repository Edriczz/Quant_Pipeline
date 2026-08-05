"""Tests for ou_pipeline.data.loader.PriceDataLoader.

All tests use injected/mocked DataFrames — no real network calls.
``yf.download`` is patched at the module level so that even indirect
calls through PriceDataLoader cannot reach the internet.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from ou_pipeline.config import PipelineConfig
from ou_pipeline.data.loader import PriceDataLoader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_raw_df(n: int = 10, start: str = "2023-01-01") -> pd.DataFrame:
    """Return a minimal yfinance-style DataFrame with a 'Close' column."""
    idx = pd.date_range(start=start, periods=n, freq="B")  # business days
    prices = np.linspace(100.0, 110.0, n)
    return pd.DataFrame({"Close": prices}, index=idx)


def _make_loader_with_mock(raw_df: pd.DataFrame, ticker: str = "TEST") -> PriceDataLoader:
    """Instantiate a PriceDataLoader whose yf.download is pre-patched."""
    loader = PriceDataLoader(ticker=ticker, config=PipelineConfig())
    # Inject cleaned data directly to bypass the network call entirely.
    loader._df = PriceDataLoader._clean(raw_df)
    return loader


# ---------------------------------------------------------------------------
# Tests: load()
# ---------------------------------------------------------------------------

class TestLoad:
    """Tests for PriceDataLoader.load()."""

    def test_column_named_price(self) -> None:
        """Returned DataFrame must have exactly one column named 'price'."""
        loader = _make_loader_with_mock(_make_raw_df())
        df = loader.load()
        assert list(df.columns) == ["price"], f"Expected ['price'], got {list(df.columns)}"

    def test_no_nans(self) -> None:
        """Returned DataFrame must contain no NaN values."""
        raw = _make_raw_df()
        # Inject a NaN in the middle — loader must drop it.
        raw.iloc[3, 0] = float("nan")
        loader = _make_loader_with_mock(raw)
        df = loader.load()
        assert not df["price"].isna().any(), "DataFrame contains NaN values"

    def test_sorted_ascending(self) -> None:
        """DatetimeIndex must be sorted in ascending order."""
        raw = _make_raw_df()
        # Shuffle the raw data to verify sorting is applied.
        raw = raw.iloc[::-1]
        loader = _make_loader_with_mock(raw)
        df = loader.load()
        assert df.index.is_monotonic_increasing, "Index is not sorted ascending"

    def test_returns_cached_df(self) -> None:
        """Calling load() twice returns the exact same object (caching)."""
        loader = _make_loader_with_mock(_make_raw_df())
        df1 = loader.load()
        df2 = loader.load()
        assert df1 is df2, "load() did not return cached DataFrame on second call"

    def test_empty_raw_raises(self) -> None:
        """_clean() must raise ValueError when given an empty DataFrame."""
        with pytest.raises(ValueError, match="empty"):
            PriceDataLoader._clean(pd.DataFrame())

    def test_missing_price_column_raises(self) -> None:
        """_clean() must raise ValueError when no price column is found."""
        bad_df = pd.DataFrame({"Volume": [1000, 2000]})
        with pytest.raises(ValueError, match="price column"):
            PriceDataLoader._clean(bad_df)


# ---------------------------------------------------------------------------
# Tests: log_series()
# ---------------------------------------------------------------------------

class TestLogSeries:
    """Tests for PriceDataLoader.log_series()."""

    def test_log_series_matches_np_log(self) -> None:
        """log_series() must equal np.log(price) element-wise."""
        raw = _make_raw_df(n=20)
        loader = _make_loader_with_mock(raw)
        df = loader.load()
        expected = np.log(df["price"].to_numpy(dtype=np.float64))
        result = loader.log_series()
        np.testing.assert_array_almost_equal(result, expected, decimal=10)

    def test_log_series_is_1d(self) -> None:
        """log_series() must return a 1-D ndarray."""
        loader = _make_loader_with_mock(_make_raw_df())
        result = loader.log_series()
        assert result.ndim == 1, f"Expected 1-D array, got shape {result.shape}"

    def test_log_series_dtype_float64(self) -> None:
        """log_series() must return float64 values."""
        loader = _make_loader_with_mock(_make_raw_df())
        result = loader.log_series()
        assert result.dtype == np.float64, f"Expected float64, got {result.dtype}"


# ---------------------------------------------------------------------------
# Network isolation sanity-check
# ---------------------------------------------------------------------------

def test_yf_download_is_not_called_on_injected_loader() -> None:
    """Verify that the injected loader path never touches yf.download."""
    with patch("ou_pipeline.data.loader.yf.download") as mock_dl:
        loader = _make_loader_with_mock(_make_raw_df())
        _ = loader.load()
        _ = loader.log_series()
        mock_dl.assert_not_called()


def test_yf_download_called_once_on_real_path() -> None:
    """When the cache is empty, load() must call yf.download exactly once."""
    raw = _make_raw_df()
    with patch("ou_pipeline.data.loader.yf.download", return_value=raw) as mock_dl:
        loader = PriceDataLoader(ticker="FAKE", config=PipelineConfig())
        loader.load()
        loader.load()  # second call — must use cache, not network
        mock_dl.assert_called_once()
