"""Manual integration run on real ASML data.

This script is NOT a test and is NOT imported anywhere.
It is a sanity check: run the full pipeline on real data and print
the results so a human can confirm they are directionally sensible.

Usage (from repo root, with .venv activated):
    python notebooks/manual_run.py

Output is also saved to notebooks/first_run_output.txt.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout

from ou_pipeline.config import PipelineConfig
from ou_pipeline.data.loader import PriceDataLoader
from ou_pipeline.diagnostics.stationarity import StationarityTester
from ou_pipeline.estimators.kalman import KalmanEstimator
from ou_pipeline.estimators.ols import OLSEstimator, OUFitError


def run(cfg: PipelineConfig | None = None) -> None:
    """Run the full pipeline on real ASML data and print results."""
    if cfg is None:
        cfg = PipelineConfig()
    sep = "=" * 60

    print(sep)
    print("  OU Pipeline — Integration Run")
    print(f"  Ticker: {cfg.ticker}  |  Period: {cfg.period}")
    print(sep)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("\n[1] Loading price data via PriceDataLoader …")
    loader = PriceDataLoader(ticker=cfg.ticker, config=cfg)
    df = loader.load()
    log_s = loader.log_series()
    print(f"    Loaded {len(df)} trading days")
    print(f"    Price range: {df['price'].min():.2f} – {df['price'].max():.2f}")
    print(f"    Log-price range: {log_s.min():.4f} – {log_s.max():.4f}")

    # ------------------------------------------------------------------
    # 2. Stationarity test
    # ------------------------------------------------------------------
    print("\n[2] ADF stationarity test …")
    tester = StationarityTester(config=cfg)
    stat_result = tester.test(log_s)
    print(f"    ADF statistic : {stat_result.adf_statistic:.4f}")
    print(f"    p-value       : {stat_result.p_value:.4f}")
    print(f"    Stationary    : {stat_result.is_stationary}  " f"(alpha={cfg.adf_alpha})")

    if not stat_result.is_stationary:
        print("    ⚠  Series did NOT pass the ADF test at the chosen alpha.")
        print("       OU parameters will still be estimated, but interpret with caution.")

    # ------------------------------------------------------------------
    # 3. OLS estimation
    # ------------------------------------------------------------------
    print("\n[3] OLS estimation …")
    ols = OLSEstimator()
    try:
        ols_result = ols.fit(log_s, dt=cfg.dt)
        print(f"    theta         : {ols_result.theta:.6f}  (1/trading-day)")
        print(f"    mu            : {ols_result.mu:.6f}  (log-price)")
        print(f"    sigma         : {ols_result.sigma:.6f}")
        print(f"    half-life     : {ols_result.half_life_days:.2f} trading days")
        print(f"    converged     : {ols_result.converged}")
        print(f"    AR(1) coef b  : {ols_result.extra['ar1_coef']:.6f}")
        print(f"    AR(1) p-value : {ols_result.extra['ar1_pvalue']:.4e}")
        print(f"    R²            : {ols_result.extra['r_squared']:.6f}")
    except OUFitError as exc:
        print(f"    OUFitError: {exc}")
        ols_result = None

    # ------------------------------------------------------------------
    # 4. Kalman estimation
    # ------------------------------------------------------------------
    print("\n[4] Kalman-MLE estimation …")
    kalman = KalmanEstimator(config=cfg)
    kalman_result = kalman.fit(log_s, dt=cfg.dt)
    print(f"    theta         : {kalman_result.theta:.6f}  (1/trading-day)")
    print(f"    mu            : {kalman_result.mu:.6f}  (log-price)")
    print(f"    sigma         : {kalman_result.sigma:.6f}")
    print(f"    half-life     : {kalman_result.half_life_days:.2f} trading days")
    print(f"    converged     : {kalman_result.converged}")
    print(f"    obs_noise_R   : {kalman_result.extra['obs_noise_R']:.6f}")
    print(f"    log-likelihood: {kalman_result.extra['log_likelihood']:.2f}")

    # ------------------------------------------------------------------
    # 5. Sanity checks
    # ------------------------------------------------------------------
    print("\n[5] Sanity checks …")
    checks_passed = 0
    checks_total = 0

    def check(cond: bool, msg: str) -> None:
        nonlocal checks_passed, checks_total
        checks_total += 1
        status = "✅" if cond else "❌"
        print(f"    {status}  {msg}")
        if cond:
            checks_passed += 1

    check(kalman_result.theta > 0, f"Kalman theta > 0  ({kalman_result.theta:.4f})")
    check(kalman_result.sigma > 0, f"Kalman sigma > 0  ({kalman_result.sigma:.4f})")
    check(
        kalman_result.half_life_days > 0,
        f"Kalman half-life > 0  ({kalman_result.half_life_days:.2f} days)",
    )
    # mu should be within the observed log-price range
    check(
        log_s.min() <= kalman_result.mu <= log_s.max(),
        f"Kalman mu in observed range  ({kalman_result.mu:.4f})",
    )
    if ols_result is not None:
        check(ols_result.theta > 0, f"OLS theta > 0  ({ols_result.theta:.4f})")
        check(ols_result.sigma > 0, f"OLS sigma > 0  ({ols_result.sigma:.4f})")

    print(f"\n    {checks_passed}/{checks_total} sanity checks passed.")
    print(f"\n{sep}")
    print("  Integration run complete.")
    print(sep)


def main() -> None:
    """Entry point: run pipeline, print to stdout and save to file."""
    output_path = "notebooks/first_run_output.txt"

    buf = io.StringIO()
    with redirect_stdout(buf):
        run()

    output = buf.getvalue()
    # Print to terminal as well
    print(output, end="")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output)

    print(f"\n[saved to {output_path}]")


if __name__ == "__main__":
    main()
