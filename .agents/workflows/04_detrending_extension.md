---
description: 
---

<!--
NAVIGATION
Position: 4 of 4 (extension) in the build workflow.
Prerequisite: 03_app_and_ship.md complete (Phases 8-10 gates all passed,
              i.e. the base pipeline is fully built and deployed).
On completion of Phase 14's gate: project extension complete. Optional
stretch (drifting-mu Kalman, see note at end) is a candidate Part 5,
not required.
Index: 00_INDEX.md
-->

# Workflow Part 4/4 — Detrending Extension (Phases 11–14)

> Prerequisite check before starting: the base pipeline (Parts 1–3,
> Phases 0–10) is fully built, tested, and deployed. This extension
> adds a new capability on top of a working system — it does not modify
> the OLS/Kalman estimator internals built in Part 2.

## Why this extension exists

The base pipeline tests whether raw price reverts to ONE fixed mean
over the whole lookback window. That is a much stronger claim than
"price oscillates around wherever its recent trajectory has been" —
and raw-price ADF tests will usually fail for a trending stock even
when short-term mean-reverting *behavior around a moving baseline* is
present. This extension adds a detrending step so the pipeline can
test the second, more realistic claim, without discarding or replacing
the original raw-price results — both should remain available and
clearly labeled as answering different questions.

---

## Phase 11 — Detrending Component

**Tasks**
- New package: `ou_pipeline/preprocessing/`.
- `ou_pipeline/preprocessing/base.py`: `SeriesTransformer` ABC with a
  single method `transform(series: np.ndarray) -> np.ndarray`, mirroring
  the estimator strategy pattern from `PROJECT_RULES.md` §3.1 — so
  future transforms (e.g. log-return detrending, STL decomposition) can
  be added the same way OLS/Kalman were.
- `ou_pipeline/preprocessing/rolling_mean.py`: `RollingMeanDetrender(SeriesTransformer)`
  — subtracts a rolling mean (configurable window, default 20 trading
  days) from the (log) price series, returns the residual.
- Extend `ou_pipeline/models/results.py`: add a `baseline: np.ndarray`
  field (or a small `DetrendResult` dataclass holding `residual` and
  `baseline`) so the moving baseline used is always retrievable
  alongside the residual — needed later for reconstructing price-level
  equilibrium bands in the dashboard.
- `tests/test_rolling_mean.py`: synthetic test — construct
  `trend(t) + wiggle(t)` explicitly (e.g. linear trend + simulated OU
  noise), detrend it, and assert the residual's ADF test rejects the
  unit root even though the raw combined series' ADF test does not.
  This is the single test that proves the extension does what it's for.

**Gate:** The synthetic test passes — detrended residual is
stationary, raw combined series is not, on the *same* constructed data.

---

## Phase 12 — Pipeline Wiring (config-driven, not a fork)

**Tasks**
- Extend `PipelineConfig` (Phase 1) with: `use_detrending: bool`
  (default `False`), `detrend_window: int` (default 20).
- The existing `OLSEstimator` / `KalmanEstimator` classes are NOT
  modified — they still just take a `series: np.ndarray` and know
  nothing about detrending. Wiring happens one level up: whatever
  orchestrates a full run (the manual integration script from Phase 7,
  and later the Streamlit app) optionally passes the detrended residual
  instead of the raw log-price series into the same estimator classes.
- This keeps `PROJECT_RULES.md` §3.2's rule intact — estimators stay
  method-agnostic about what series they're fed.

**Gate:** Running an estimator on a detrended residual and on raw
log-price requires no code changes inside the estimator classes — only
a different array passed in at the call site. Confirm this by running
both through the same `OLSEstimator` instance in a quick script.

---

## Phase 13 — Stationarity + Interpretation on the Residual

**Tasks**
- `StationarityTester` (Phase 3) is reused as-is on the residual series
  — no changes needed, it already just takes an array.
- Extend the interpretation layer (Phase 8) so `build_verdict(...)`
  accepts a `mode: Literal["raw", "detrended"]` and produces distinct
  wording, e.g.:
  - raw mode: "Price does/does not revert to a single long-run mean of $X."
  - detrended mode: "Price does/does not revert to its own {window}-day
    moving average; currently {z}σ away from that local baseline."
- `tests/test_interpretation.py`: add cases covering both modes,
  including the case where raw says non-stationary but detrended says
  stationary — the exact situation this extension exists to handle.

**Gate:** Interpretation test suite passes for both modes, including
the "raw fails, detrended passes" combination.

---

## Phase 14 — Dashboard: Toggle, Not Replacement

**Tasks**
- Add a toggle in `app/streamlit_app.py`: "Raw price vs. fixed mean"
  vs. "Detrended vs. moving average" — user-selectable, default to raw
  (keep existing behavior as default so nothing breaks for existing
  users of the dashboard).
- In detrended mode, the equilibrium line on the chart is the
  **rolling baseline itself** (a moving line, not a flat dashed line at
  one price) with the OU/Kalman equilibrium bands drawn *around that
  moving baseline*, not around one static price.
- Metrics row updates its labels contextually (e.g. "Equilibrium
  Price" becomes "Local Baseline" in detrended mode) so it's visually
  obvious which question is being answered.
- README: add a short section explaining the two modes and when each
  answers the kind of question a user is actually asking (long-run
  fixed value vs. short-run local reversion) — this is also good
  portfolio narrative, since it shows methodological awareness rather
  than just "here's a chart."

**Gate:** Toggling the mode in the running app changes both the chart
and the verdict text correctly, without errors, on real ASML data.
Confirm visually that detrended mode's equilibrium band tracks the
moving average rather than sitting flat.

---

## End of Part 4

All four gates above (Phases 11–14) must be green.

**Optional future extension (not required, not gated here):** letting
μ itself be a hidden state that drifts via its own noise term inside
the Kalman filter (rather than pre-computing a rolling mean and
handing it to an otherwise-unchanged Kalman/OLS estimator) is a more
statistically integrated version of the same idea. That would be a
new estimator class, e.g. `DriftingMeanKalmanEstimator(OUEstimator)`,
following the exact same strategy-pattern rule from `PROJECT_RULES.md`
§3.1 — worth doing later as its own Part 5 workflow document if you
want the more rigorous version, but the rolling-mean detrending above
already answers the question you actually asked.