---
description: 
---

<!--
NAVIGATION
Position: 3 of 3 in the build workflow.
Prerequisite: 02_estimators.md complete (Phases 5-7 gates all passed).
On completion of Phase 10's gate: base pipeline is done and deployed.
Optional extension: 04_detrending_extension.md (Phases 11-14).
Index: 00_INDEX.md
-->

# Workflow Part 3/3 — App & Ship (Phases 8–10)

> Prerequisite check before starting: `02_estimators.md` Phases 5–7 are
> all complete and their gates passed, including the Phase 6 comparative
> assertion (Kalman outperforms OLS on noisy synthetic data).

Covers: the interpretation layer, the Streamlit app, and final polish
for the portfolio. Everything here sits on top of already-validated
math from Part 2 — this document is about rendering and packaging
proven results, not discovering new bugs in the OU logic. If a bug in
the estimators surfaces while working through this document, stop and
go back to `02_estimators.md` rather than patching around it here.

---

## Phase 8 — Interpretation Layer

**Tasks**
- Pure function(s) — e.g. `ou_pipeline/interpretation.py` —
  `build_verdict(ou_result, stationarity_result) -> str`, producing the
  plain-English one-liner (z-score direction, gated by whether ADF
  supports stationarity).
- `tests/test_interpretation.py`: test the function directly with
  constructed `OUResult`/`StationarityResult` objects (not live data) —
  cover the "stationary + above mean," "stationary + below mean," and
  "not stationary" branches explicitly.

**Gate:** All three interpretation branches produce the correct message
in tests.

---

## Phase 9 — Streamlit App

**Tasks**
- `app/streamlit_app.py`: inputs (ticker, period, method toggle) →
  `PriceDataLoader` → estimator(s) → `StationarityTester` →
  interpretation function → render (metrics row, plot, verdict banner).
- Cache the data loading step (`@st.cache_data`).
- No business logic inline — every computation is a call into
  `ou_pipeline`, per `PROJECT_RULES.md` §6.

**Gate:** `streamlit run app/streamlit_app.py` loads with ASML as
default ticker, no crash, plot renders, switching OLS/Kalman updates
the numbers.

---

## Phase 10 — Polish for Portfolio

**Tasks**
- README: what this is, screenshot/GIF of the dashboard, architecture
  diagram (folder tree is fine), how to run locally, link to the live
  Streamlit Cloud deployment, link back to the binomial pricing project.
- Optional stretch: multi-ticker comparison mode (ASML, NVDA, and one
  genuinely mean-reverting instrument) — not required for Definition of
  Done, but strengthens the portfolio narrative.
- Run `black .` and `ruff check .` clean across the whole repo.
- Deploy to Streamlit Community Cloud, confirm the live link works from
  a fresh browser session (not just localhost).

**Gate:** Live deployed link works standalone, README is readable by
someone who has never seen the design conversation behind this project.

---

## End of Part 3 — Base Pipeline Complete

All three gates above (Phases 8–10) must be green. There is no required
further phase document — the base pipeline satisfies the Definition of
Done in `PROJECT_RULES.md` §8 for every component at this point.

**Optional next step:** if you want to test mean-reversion relative to
a moving baseline rather than one fixed long-run mean (see the
detrending discussion), continue to `04_detrending_extension.md` and
start at Phase 11.

---

## End of Part 3 — Project Complete

All three gates above (Phases 8–10) must be green. There is no further
phase document — once Phase 10's gate passes, the project satisfies
the Definition of Done in `PROJECT_RULES.md` §8 for every component.