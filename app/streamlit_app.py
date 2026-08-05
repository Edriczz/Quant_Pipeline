"""Streamlit Dashboard for ASML / Equity OU Mean-Reversion Pipeline.

UI ONLY — imports ou_pipeline components. No fitting or statistical logic
lives in this file (per PROJECT_RULES.md §6).
"""

from __future__ import annotations

from typing import Literal

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from ou_pipeline.config import PipelineConfig
from ou_pipeline.data.loader import PriceDataLoader
from ou_pipeline.diagnostics.stationarity import StationarityTester
from ou_pipeline.estimators.base import OUEstimator
from ou_pipeline.estimators.kalman import KalmanEstimator
from ou_pipeline.estimators.ols import OLSEstimator
from ou_pipeline.interpretation import build_verdict, calculate_z_score
from ou_pipeline.models.results import OUResult, StationarityResult
from ou_pipeline.preprocessing.rolling_mean import RollingMeanDetrender

# -----------------------------------------------------------------------------
# Data Loading & Fitting (Cached)
# -----------------------------------------------------------------------------


@st.cache_data(ttl=3600)
def _load_data(ticker: str, period: str) -> pd.DataFrame:
    """Fetch price data via PriceDataLoader. Cached for 1 hour."""
    cfg = PipelineConfig(ticker=ticker, period=period)
    loader = PriceDataLoader(ticker=ticker, config=cfg)
    return loader.load()


def _fit_pipeline(
    df: pd.DataFrame,
    method: str,
    dt: float,
    alpha: float,
    use_detrending: bool = False,
    detrend_window: int = 20,
) -> tuple[OUResult, StationarityResult, np.ndarray | None]:
    """Fit selected estimator and run stationarity test on log-price or detrended residual."""
    cfg = PipelineConfig(
        dt=dt,
        adf_alpha=alpha,
        use_detrending=use_detrending,
        detrend_window=detrend_window,
    )
    log_s = np.log(df["price"].to_numpy(dtype=np.float64))

    baseline: np.ndarray | None = None
    series_to_fit: np.ndarray = log_s

    if use_detrending:
        detrender = RollingMeanDetrender(window=detrend_window)
        detrend_res = detrender.transform(log_s)
        series_to_fit = detrend_res.residual
        baseline = detrend_res.baseline

    # Stationarity test
    tester = StationarityTester(config=cfg)
    stationarity_res = tester.test(series_to_fit)

    # Estimator selection
    estimator: OUEstimator
    if method == "Kalman-MLE":
        estimator = KalmanEstimator(config=cfg)
    elif method == "OLS":
        estimator = OLSEstimator()
    else:
        raise ValueError(f"Unknown estimation method: {method}")

    ou_res = estimator.fit(series_to_fit, dt=dt)
    return ou_res, stationarity_res, baseline


# -----------------------------------------------------------------------------
# Render Helpers (UI Composition)
# -----------------------------------------------------------------------------


def _render_header() -> None:
    st.set_page_config(
        page_title="OU Mean-Reversion Pipeline",
        page_icon="📈",
        layout="wide",
    )
    st.title("📈 Ornstein-Uhlenbeck Mean-Reversion Pipeline")
    st.caption(
        "Quantitative estimation of mean-reversion parameters (OLS & State-Space Kalman Filter) "
        "with Augmented Dickey-Fuller stationarity diagnostics & rolling-mean detrending mode."
    )
    st.markdown("---")


def _render_sidebar() -> tuple[str, str, str, float, bool, str, int]:
    st.sidebar.header("Pipeline Configuration")
    ticker = st.sidebar.text_input("Ticker Symbol", value="ASML").strip().upper()
    period = st.sidebar.selectbox(
        "Lookback Period",
        options=["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"],
        index=4,
    )
    method = st.sidebar.radio(
        "Estimation Method",
        options=["Kalman-MLE", "OLS"],
        index=0,
        help="Kalman-MLE separates observation noise from process noise; OLS is standard AR(1).",
    )
    mode_label = st.sidebar.radio(
        "Pipeline Mode",
        options=["Raw price vs. fixed mean", "Detrended vs. moving average"],
        index=0,
        help=(
            "Raw mode tests for reversion to one fixed mean over the whole window. "
            "Detrended mode tests for reversion to a moving average baseline."
        ),
    )
    mode = "detrended" if "Detrended" in mode_label else "raw"

    detrend_window = 20
    if mode == "detrended":
        detrend_window = st.sidebar.slider(
            "Detrending Window (days)",
            min_value=5,
            max_value=60,
            value=20,
            step=1,
            help="Window size for rolling-mean baseline.",
        )

    alpha = st.sidebar.slider(
        "ADF Alpha (Significance)",
        min_value=0.01,
        max_value=0.10,
        value=0.05,
        step=0.01,
    )
    use_log_scale = st.sidebar.checkbox("Plot in Log Scale", value=False)
    return ticker, period, method, alpha, use_log_scale, mode, detrend_window


def _render_verdict_banner(
    ou_res: OUResult,
    stationarity_res: StationarityResult,
    current_val: float,
    mode: Literal["raw", "detrended"] = "raw",
    detrend_window: int = 20,
) -> None:
    verdict = build_verdict(
        ou_res,
        stationarity_res,
        current_val,
        mode=mode,
        detrend_window=detrend_window,
    )
    if not stationarity_res.is_stationary:
        st.warning(f"⚠️ **Stationarity Verdict:** {verdict}")
    else:
        st.success(f"🎯 **Stationarity Verdict:** {verdict}")


def _render_metrics(
    ou_res: OUResult,
    stationarity_res: StationarityResult,
    current_price: float,
    current_val: float,
    mode: Literal["raw", "detrended"] = "raw",
    baseline_val: float | None = None,
) -> None:
    z_score = calculate_z_score(current_val, ou_res)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric(
            label="Current Price",
            value=f"${current_price:,.2f}",
        )
    with col2:
        if mode == "detrended" and baseline_val is not None:
            base_price = float(np.exp(baseline_val))
            st.metric(
                label="Local Baseline (MA)",
                value=f"${base_price:,.2f}",
                delta=f"{(current_price - base_price) / base_price:.2%} vs MA",
                help="Current value of the rolling-mean baseline",
            )
        else:
            eq_price = float(np.exp(ou_res.mu))
            st.metric(
                label="Equilibrium Price (e^μ)",
                value=f"${eq_price:,.2f}",
                delta=f"{(current_price - eq_price) / eq_price:.2%} vs Mean",
                help=f"Fitted log-mean μ = {ou_res.mu:.4f}",
            )
    with col3:
        st.metric(
            label="Half-Life",
            value=f"{ou_res.half_life_days:.1f} days",
            help=f"Mean-reversion speed θ = {ou_res.theta:.4f} / day",
        )
    with col4:
        st.metric(
            label="Current Z-Score",
            value=f"{z_score:+.2f} σ",
            help="Asymptotic standard deviations from baseline / mean",
        )
    with col5:
        stat_status = "Stationary ✅" if stationarity_res.is_stationary else "Non-Stationary ❌"
        st.metric(
            label="ADF Test (p-value)",
            value=f"{stationarity_res.p_value:.4f}",
            delta=stat_status,
            delta_color="normal" if stationarity_res.is_stationary else "inverse",
        )


def _render_plot(
    df: pd.DataFrame,
    ou_res: OUResult,
    use_log: bool = False,
    mode: Literal["raw", "detrended"] = "raw",
    baseline: np.ndarray | None = None,
) -> None:
    plot_df = df.copy()

    if mode == "detrended" and baseline is not None:
        sigma_eq_log = ou_res.sigma / np.sqrt(2.0 * ou_res.theta)
        if use_log:
            plot_df["value"] = np.log(plot_df["price"])
            plot_df["Mean"] = baseline
            plot_df["+1 σ"] = baseline + sigma_eq_log
            plot_df["-1 σ"] = baseline - sigma_eq_log
            plot_df["+2 σ"] = baseline + 2 * sigma_eq_log
            plot_df["-2 σ"] = baseline - 2 * sigma_eq_log
            y_label = "Log Price"
        else:
            plot_df["value"] = plot_df["price"]
            base_price = np.exp(baseline)
            sigma_eq_price = (np.exp(sigma_eq_log) - 1.0) * base_price
            plot_df["Mean"] = base_price
            plot_df["+1 σ"] = base_price + sigma_eq_price
            plot_df["-1 σ"] = base_price - sigma_eq_price
            plot_df["+2 σ"] = base_price + 2 * sigma_eq_price
            plot_df["-2 σ"] = base_price - 2 * sigma_eq_price
            y_label = "Price ($)"
        chart_title = (
            f"Price Trajectory vs Rolling Moving Average Baseline & OU Bands ({ou_res.method})"
        )
    else:
        if use_log:
            plot_df["value"] = np.log(plot_df["price"])
            mu_val = ou_res.mu
            sigma_eq = ou_res.sigma / np.sqrt(2.0 * ou_res.theta)
            y_label = "Log Price"
        else:
            plot_df["value"] = plot_df["price"]
            mu_val = float(np.exp(ou_res.mu))
            sigma_eq_log = ou_res.sigma / np.sqrt(2.0 * ou_res.theta)
            sigma_eq = (np.exp(sigma_eq_log) - 1.0) * mu_val
            y_label = "Price ($)"

        plot_df["Mean"] = mu_val
        plot_df["+1 σ"] = mu_val + sigma_eq
        plot_df["-1 σ"] = mu_val - sigma_eq
        plot_df["+2 σ"] = mu_val + 2 * sigma_eq
        plot_df["-2 σ"] = mu_val - 2 * sigma_eq
        chart_title = f"Price Trajectory vs Fitted Fixed OU Equilibrium Bands ({ou_res.method})"

    reset_df = plot_df.reset_index()

    # Base line chart for actual price
    price_line = (
        alt.Chart(reset_df)
        .mark_line(color="#1f77b4", strokeWidth=2)
        .encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("value:Q", title=y_label, scale=alt.Scale(zero=False)),
            tooltip=["Date:T", alt.Tooltip("value:Q", format=".2f", title="Price")],
        )
    )

    # Equilibrium / Moving Baseline Line
    mean_line = (
        alt.Chart(reset_df)
        .mark_line(color="#2ca02c", strokeDash=[4, 4], strokeWidth=2)
        .encode(x="Date:T", y="Mean:Q")
    )

    # 1 SD Band
    band_1sd = (
        alt.Chart(reset_df)
        .mark_area(opacity=0.15, color="#2ca02c")
        .encode(x="Date:T", y="-1 σ:Q", y2="+1 σ:Q")
    )

    # 2 SD Band
    band_2sd = (
        alt.Chart(reset_df)
        .mark_area(opacity=0.08, color="#ff7f0e")
        .encode(x="Date:T", y="-2 σ:Q", y2="+2 σ:Q")
    )

    chart = (
        (band_2sd + band_1sd + mean_line + price_line)
        .properties(
            width="container",
            height=450,
            title=chart_title,
        )
        .interactive()
    )

    st.altair_chart(chart, use_container_width=True)


def _render_details(
    ou_res: OUResult,
    stationarity_res: StationarityResult,
    mode: Literal["raw", "detrended"] = "raw",
    detrend_window: int = 20,
) -> None:
    with st.expander("🔬 Model Estimation Details & Diagnostic Statistics"):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Fitted Parameters")
            details = {
                "Method": ou_res.method,
                "Pipeline Mode": mode,
                "Detrending Window (days)": (detrend_window if mode == "detrended" else "N/A"),
                "Speed (theta)": ou_res.theta,
                "Mean / Offset (mu)": ou_res.mu,
                "Volatility (sigma)": ou_res.sigma,
                "Half-Life (days)": ou_res.half_life_days,
                "Optimiser Converged": ou_res.converged,
            }
            if mode == "raw":
                details["Equilibrium Price (exp(mu))"] = float(np.exp(ou_res.mu))
            st.json(details)
        with col2:
            st.subheader("Method-Specific Diagnostics & ADF")
            st.json(
                {
                    "ADF Statistic": stationarity_res.adf_statistic,
                    "ADF p-value": stationarity_res.p_value,
                    "Is Stationary": stationarity_res.is_stationary,
                    "Extra Fields": ou_res.extra,
                }
            )


# -----------------------------------------------------------------------------
# Main Application Flow
# -----------------------------------------------------------------------------


def main() -> None:
    _render_header()
    (
        ticker,
        period,
        method,
        alpha,
        use_log_scale,
        mode,
        detrend_window,
    ) = _render_sidebar()

    try:
        with st.spinner(f"Fetching data for {ticker} ({period})…"):
            df = _load_data(ticker, period)

        current_price = float(df["price"].iloc[-1])
        current_log_price = float(np.log(current_price))

        use_detrending = mode == "detrended"

        with st.spinner(f"Fitting {method} model ({mode} mode)…"):
            ou_res, stationarity_res, baseline = _fit_pipeline(
                df,
                method,
                dt=1.0,
                alpha=alpha,
                use_detrending=use_detrending,
                detrend_window=detrend_window,
            )

        if use_detrending and baseline is not None:
            # Current value for verdict is the latest residual (log_price - baseline)
            latest_residual = current_log_price - float(baseline[-1])
            current_val_for_verdict = latest_residual
            baseline_val_for_metric = float(baseline[-1])
        else:
            current_val_for_verdict = current_log_price
            baseline_val_for_metric = None

        _render_verdict_banner(
            ou_res,
            stationarity_res,
            current_val_for_verdict,
            mode=mode,
            detrend_window=detrend_window,
        )
        _render_metrics(
            ou_res,
            stationarity_res,
            current_price,
            current_val_for_verdict,
            mode=mode,
            baseline_val=baseline_val_for_metric,
        )
        _render_plot(df, ou_res, use_log=use_log_scale, mode=mode, baseline=baseline)
        _render_details(ou_res, stationarity_res, mode=mode, detrend_window=detrend_window)

    except Exception as exc:
        st.error(f"❌ Error executing pipeline for {ticker}: {exc}")


if __name__ == "__main__":
    main()
