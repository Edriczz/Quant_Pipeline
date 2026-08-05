---
description: 
---

<!--
NAVIGATION
Position: 2 of 3 in the build workflow.
Prerequisite: 01_foundation.md complete (Phases 0-4 gates all passed).
On completion of Phase 7's gate: proceed to 03_app_and_ship.md (Phase 8).
Index: 00_INDEX.md
-->

# Workflow Part 2/3 — Estimators (Phases 5–7)

> Prerequisite check before starting: `01_foundation.md` Phases 0–4 are
> all complete and their gates passed. If unsure, verify rather than
> assume — run the Phase 0–4 gates again before proceeding.

Covers: the OLS estimator, the Kalman estimator, and a manual
integration check on real data. This is the mathematical core of the
project — take the validation gates in this document seriously, since
everything downstream (interpretation, dashboard) trusts these numbers
without re-checking them.

---

## Phase 5 — OLS Estimator (build first — it's the baseline)

**Tasks**
- `ou_pipeline/estimators/ols.py`: `OLSEstimator(OUEstimator)`,
  implementing the AR(1) regression fit.
- `tests/test_ols.py`: **synthetic validation test** — simulate an OU
  path with known (θ, μ, σ), no observation noise, fit, assert recovered
  params are within an agreed tolerance (e.g. ±15% on theta, tighter on
  mu/sigma since OLS is unbiased in the noiseless case).

**Gate:** Synthetic recovery test passes. This is the reference
implementation the Kalman estimator will be compared against next.

---

## Phase 6 — Kalman Estimator (builds on Phase 5's validation pattern)

**Tasks**
- `ou_pipeline/estimators/kalman.py`: `KalmanEstimator(OUEstimator)` —
  state-space OU + Kalman filter + MLE via `scipy.optimize`.
- `tests/test_kalman.py`: **two synthetic tests**:
  1. No observation noise: Kalman should recover params about as well
     as OLS (sanity check — it shouldn't be worse in the easy case).
  2. **With injected observation noise**: Kalman should recover theta
     noticeably closer to the true value than OLS does on the *same*
     noisy series. This is the test that proves the Kalman approach
     earns its complexity — assert `abs(kalman_theta - true_theta) <
     abs(ols_theta - true_theta)` on the noisy case.

**Gate:** Both tests pass, and specifically the comparative assertion
in test 2 holds. If Kalman doesn't outperform OLS on noisy synthetic
data, something in the implementation is wrong — do not proceed to
Phase 7 with a broken estimator.

---

## Phase 7 — Pipeline Integration (still no UI)

**Tasks**
- Write a small integration script (`notebooks/manual_run.py`, not
  imported anywhere) that: loads real ASML data via `PriceDataLoader`
  → runs `StationarityTester` → runs both estimators → prints results.
  This is a manual sanity check on real data, not a formal test.
- Confirm results are directionally sensible (half-life not negative,
  theta positive, mu in a plausible price range).

**Gate:** Manual run on real ASML data completes without errors and
produces sane, inspectable numbers. Save the printed output somewhere
(e.g. `notebooks/first_run_output.txt`) so there is a record of the
first real run's results.

---

## End of Part 2

All three gates above (Phases 5–7) must be green before continuing.

**Next:** open `03_app_and_ship.md` and start at Phase 8.