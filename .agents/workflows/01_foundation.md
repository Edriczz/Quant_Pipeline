---
description: plumbing, no math yet — an agent can knock this out fast with low risk.
---

<!--
NAVIGATION
Position: 1 of 3 in the build workflow.
Prerequisite: none — this is the starting document.
On completion of Phase 4's gate: proceed to 02_estimators.md (Phase 5).
Index: 00_INDEX.md
-->

# Workflow Part 1/3 — Foundation (Phases 0–4)

> Read `00_INDEX.md` first if you have not already. This document
> assumes `PROJECT_RULES.md` has been read for component-level
> conventions (OOP structure, typing, docstrings, testing style).

Covers: project scaffold, config, result models, data loader,
stationarity diagnostics, and the base estimator interface. No
estimation math and no UI in this document — that starts in Part 2.

---

## Phase 0 — Scaffolding

**Tasks**
- Create the folder structure exactly as specified in `PROJECT_RULES.md` §2.
- Initialize `pyproject.toml`, `requirements.txt` (yfinance, numpy, pandas,
  scipy, statsmodels, streamlit, pytest, black, ruff).
- Empty `__init__.py` in every package folder.
- Git init, first commit: "scaffold project structure".

**Gate:** `pip install -r requirements.txt` succeeds; folder tree matches spec.

---

## Phase 1 — Config + Models (no logic yet)

**Tasks**
- `ou_pipeline/config.py`: `PipelineConfig` dataclass (ticker default,
  period default, dt, ADF alpha, optimizer settings).
- `ou_pipeline/models/results.py`: `OUResult`, `StationarityResult`
  dataclasses per rules §3.3.

**Gate:** Both files import cleanly with no errors; no other module
depends on them yet, so this is just laying pipes.

---

## Phase 2 — Data Loader

**Tasks**
- `ou_pipeline/data/loader.py`: `PriceDataLoader` class — `load()` and
  `log_series()` per rules §3.4.
- `tests/test_loader.py`: test with a **mocked/injected DataFrame**, not
  a live network call. Verify: correct column name, no NaNs, sorted
  index, `log_series()` matches `np.log(price)`.

**Gate:** `pytest tests/test_loader.py` passes. Confirm no test in this
file makes a real network request (mock `yf.download`).

---

## Phase 3 — Stationarity Diagnostics

**Tasks**
- `ou_pipeline/diagnostics/stationarity.py`: `StationarityTester` class
  wrapping `statsmodels.tsa.stattools.adfuller`, returning
  `StationarityResult`.
- `tests/test_stationarity.py`: two synthetic cases —
  1. Simulated OU path (mean-reverting) → assert `is_stationary == True`.
  2. Simulated random walk (cumsum of iid noise) → assert
     `is_stationary == False`.

**Gate:** Both synthetic cases classify correctly. If either fails,
do not proceed — the ADF wrapper is the gate for the whole pipeline's
credibility, it must be correct before estimators are built on top of it.

---

## Phase 4 — Base Estimator Interface

**Tasks**
- `ou_pipeline/estimators/base.py`: `OUEstimator` ABC per rules §3.1.
- `ou_pipeline/estimators/_shared.py`: shared helpers only (half-life
  calc from theta, log-transform utilities). No estimation logic here.

**Gate:** ABC cannot be instantiated directly (raises `TypeError` if
attempted in a quick smoke test) — confirms `@abstractmethod` is wired
correctly.

---

## End of Part 1

All five gates above (Phases 0–4) must be green before continuing.

**Next:** open `02_estimators.md` and start at Phase 5.