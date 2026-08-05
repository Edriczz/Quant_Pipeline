---
trigger: always_on
---

# Project Rules: ASML OU Mean-Reversion Pipeline

## 1. Purpose

Build a Python pipeline that estimates Ornstein-Uhlenbeck (OU) parameters
for a given stock's price series using multiple estimation methods, tests
whether mean-reversion is statistically supported, and exposes results
through a Streamlit dashboard. Logic and UI must be fully separated.

---

## 2. Architecture (non-negotiable)

```
project/
├── ou_pipeline/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py          # PriceDataLoader class
│   ├── estimators/
│   │   ├── __init__.py
│   │   ├── base.py            # OUEstimator abstract base class
│   │   ├── ols.py             # OLSEstimator(OUEstimator)
│   │   └── kalman.py          # KalmanEstimator(OUEstimator)
│   ├── diagnostics/
│   │   ├── __init__.py
│   │   └── stationarity.py    # StationarityTester class (ADF)
│   ├── models/
│   │   ├── __init__.py
│   │   └── results.py         # OUResult, StationarityResult dataclasses
│   └── config.py              # PipelineConfig dataclass, defaults/constants
├── app/
│   └── streamlit_app.py       # UI ONLY — imports ou_pipeline, no logic here
├── tests/
│   ├── test_ols.py
│   ├── test_kalman.py
│   ├── test_stationarity.py
│   └── test_loader.py
├── notebooks/                 # exploratory only, never imported by app or pipeline
├── requirements.txt
├── README.md
└── pyproject.toml
```

**Rule:** `app/streamlit_app.py` must contain zero fitting/statistical logic.
It only: collects inputs → calls `ou_pipeline` functions/classes → renders
outputs. If you find yourself writing `np.` or `scipy.` inside the app file,
that code belongs in `ou_pipeline/` instead.

---

## 3. OOP Design Rules

### 3.1 Abstract base for all estimation methods

Every estimation method (OLS, Kalman, and any future method — e.g. MLE
via direct likelihood, Bayesian/MCMC) must implement a common interface
so the app and tests can treat them interchangeably (strategy pattern).

```python
# ou_pipeline/estimators/base.py
from abc import ABC, abstractmethod
import numpy as np
from ou_pipeline.models.results import OUResult

class OUEstimator(ABC):
    """Common interface for all OU parameter estimation methods."""

    @abstractmethod
    def fit(self, series: np.ndarray, dt: float = 1.0) -> OUResult:
        """Fit OU parameters to a price/log-price series and return OUResult."""
        ...

    @property
    @abstractmethod
    def method_name(self) -> str:
        """Short identifier, e.g. 'OLS', 'Kalman-MLE'."""
        ...
```

### 3.2 One class per method, one file per class

- `OLSEstimator` in `ols.py` — implements the AR(1) regression approach.
- `KalmanEstimator` in `kalman.py` — implements the state-space MLE approach.
- No shared logic duplicated between them; anything common (e.g. half-life
  calculation, log-transform helpers) goes in a shared utility module
  (`ou_pipeline/estimators/_shared.py`), imported by both — never copy-pasted.
- Each estimator class must be independently instantiable and testable
  with no dependency on Streamlit, yfinance, or the other estimator.

### 3.3 Results as typed dataclasses, not dicts

```python
# ou_pipeline/models/results.py
from dataclasses import dataclass

@dataclass(frozen=True)
class OUResult:
    method: str
    theta: float
    mu: float
    sigma: float
    half_life_days: float
    converged: bool
    extra: dict  # method-specific fields (e.g. Kalman's obs_noise_R, OLS's p-value)

@dataclass(frozen=True)
class StationarityResult:
    adf_statistic: float
    p_value: float
    is_stationary: bool  # p_value < alpha
```
No function returns a raw dict or tuple of numbers for a "result" — always
wrap in the appropriate dataclass. This makes the Streamlit layer just
attribute access (`result.theta`), not dict-key guessing.

### 3.4 Data loading is its own class

```python
# ou_pipeline/data/loader.py
class PriceDataLoader:
    def __init__(self, ticker: str, period: str = "2y"):
        ...
    def load(self) -> pd.DataFrame:
        """Returns a DataFrame with a single 'price' column, no gaps, sorted by date."""
        ...
    def log_series(self) -> np.ndarray:
        ...
```
Never call `yf.download` directly from an estimator, the app, or a test —
always through `PriceDataLoader`. This is what lets tests substitute
synthetic data without hitting the network.

### 3.5 Stationarity testing is separate from estimation

`StationarityTester` (ADF) is its own class in `diagnostics/`. Estimators
must NOT run the ADF test internally — the pipeline/app decides when to
run it and how to interpret it (e.g. gate the displayed verdict). This
keeps "does the model apply" logically separate from "what are the
parameters if I run it anyway."

### 3.6 Configuration, not hardcoded constants

All defaults (lookback period, dt, ADF alpha, optimizer settings) live in
`ou_pipeline/config.py` as a single `PipelineConfig` dataclass, passed
explicitly into loaders/estimators. No magic numbers inline in method
bodies.

---

## 4. Code Quality Rules

- **Type hints on every function signature**, no exceptions — inputs and
  return types. Run `mypy` in CI if possible.
- **Docstrings**: Google-style, on every public class and method. State
  units where relevant (e.g. "theta in 1/trading-day", "dt in trading days").
- **No bare `except:`** — catch specific exceptions (e.g. `ValueError` from
  a non-converging optimizer), and raise a custom exception if needed
  (`class OUFitError(Exception): ...`).
- **No print() inside `ou_pipeline/`** — return results and let the caller
  (app or CLI) decide how to display them. Use the `logging` module for
  internal diagnostics if truly needed, not print.
- **Formatting/linting**: `black` for formatting, `ruff` for linting.
  Line length 100. Run both before every commit.
- **No global mutable state** — estimators and loaders are instantiated
  per-use, not module-level singletons.

---

## 5. Testing Rules

- Every estimator must have a test that fits against **synthetic OU data
  with known parameters** (simulate a path with a chosen θ, μ, σ, assert
  the recovered estimate is within a reasonable tolerance). This is the
  single most important test in the repo — it validates correctness
  independent of any real market data.
- Test that `KalmanEstimator` correctly separates observation noise from
  process noise on synthetic data with injected measurement noise (this
  is the whole point of using Kalman over OLS — the test suite should
  prove it).
- `PriceDataLoader` tests should not hit the network — mock `yf.download`
  or inject a pre-built DataFrame.
- `StationarityTester` tests should cover both a clearly stationary
  synthetic series (e.g. simulated OU) and a clearly non-stationary one
  (e.g. simulated random walk) to confirm the ADF wrapper classifies
  each correctly.
- Target: every class in `ou_pipeline/` has a corresponding test file.

---

## 6. Streamlit App Rules

- `app/streamlit_app.py` is UI composition only: input widgets → call
  `PriceDataLoader`, `OLSEstimator`/`KalmanEstimator`, `StationarityTester`
  → pass results to render functions.
- Prefer small private render helper functions
  (`_render_metrics(result: OUResult)`, `_render_plot(...)`) over one
  giant script body, even though it's Streamlit.
- Cache expensive calls (`@st.cache_data` on the data loader's fetch) so
  re-running the UI on a widget change doesn't re-hit yfinance every time.
- Any interpretation text ("price is above/below the fitted mean") is a
  pure function that takes an `OUResult` + `StationarityResult` and
  returns a string — testable independent of Streamlit rendering.

---

## 7. Extensibility Rule

Adding a third estimation method (e.g. a Bayesian/MCMC estimator later)
must require:
1. A new file in `estimators/`, subclassing `OUEstimator`.
2. A new test file with the synthetic-data validation.
3. One line added to the method dropdown in the Streamlit app.

No other file should need to change. If it does, the abstraction is leaking.

---

## 8. Definition of Done (per component)

A component (estimator, loader, tester) is "done" only when:
- [ ] Fully typed, docstringed.
- [ ] Has a passing synthetic-data validation test.
- [ ] Has zero direct dependency on Streamlit.
- [ ] Returns a dataclass, not a dict/tuple.
- [ ] Passes `black --check` and `ruff check`.