# 📈 Ornstein-Uhlenbeck (OU) Mean-Reversion Pipeline

A robust, object-oriented Python quantitative pipeline and interactive Streamlit dashboard for estimating Ornstein-Uhlenbeck (OU) stochastic process parameters on financial price series.

It validates whether mean-reversion is statistically supported using Augmented Dickey-Fuller (ADF) stationarity diagnostics, fits parameters using multiple independent estimators (**AR(1) OLS** and **State-Space Kalman Filter MLE**), and presents live results with equilibrium confidence bands.

---

## 🎯 Purpose & Overview

In financial modeling, mean-reversion strategies rely on identifying assets whose price or log-price processes revert to a long-run equilibrium mean ($\mu$) over a characteristic half-life ($t_{1/2}$).

Standard regression techniques (OLS) often confuse **measurement/observation noise** (e.g., bid-ask bounce or microstructure noise) with **process diffusion**, biasing the mean-reversion speed ($\theta$) downward. This pipeline implements a state-space model solved via a **Kalman Filter with Maximum Likelihood Estimation (MLE)** to separate observation noise $R$ from process variance $Q$, producing vastly superior parameter recovery on noisy empirical data.

---

## 📐 Mathematical Foundations

### 1. The Continuous-Time OU Process
The stochastic differential equation (SDE) for an Ornstein-Uhlenbeck process $X_t = \ln(P_t)$ is:

$$dX_t = \theta (\mu - X_t) dt + \sigma dW_t$$

- $\theta > 0$: Speed of mean-reversion (units: $1 / \text{trading-day}$).
- $\mu$: Long-run equilibrium mean (log-price level).
- $\sigma > 0$: Instantaneous volatility coefficient.
- $W_t$: Standard Brownian motion.

The half-life of mean-reversion in trading days is derived as:
$$t_{1/2} = \frac{\ln(2)}{\theta}$$

---

### 2. AR(1) OLS Estimation (`OLSEstimator`)
Discretizing the SDE over timestep $\Delta t = 1$:
$$X_{t+1} = a + b X_t + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, \sigma_{AR}^2)$$

Continuous-time parameters are recovered in closed form:
$$\theta = -\frac{\ln(b)}{\Delta t}, \quad \mu = \frac{a}{1 - b}, \quad \sigma = \sigma_{AR} \sqrt{\frac{-2 \ln(b)}{\Delta t (1 - b^2)}}$$

---

### 3. State-Space Kalman Filter MLE (`KalmanEstimator`)
Formulated as a linear Gaussian state-space model:

$$\begin{aligned}
\text{State Equation:} \quad X_t &= \mu + e^{-\theta \Delta t}(X_{t-1} - \mu) + w_t, \quad w_t \sim \mathcal{N}(0, Q) \\
\text{Observation Eq.:} \quad Y_t &= X_t + v_t, \quad v_t \sim \mathcal{N}(0, R)
\end{aligned}$$

where process noise variance $Q = \frac{\sigma^2}{2\theta}(1 - e^{-2\theta \Delta t})$ and observation noise variance $R$ is jointly optimized via `scipy.optimize.minimize`.

---

### 4. Stationarity Diagnostic (`StationarityTester`)
Runs an **Augmented Dickey-Fuller (ADF)** unit-root test on log-prices prior to model interpretation. If the p-value exceeds $\alpha = 0.05$, the pipeline explicitly warns that mean-reversion is not statistically supported.

---

## 🏗️ Architecture & Decoupled Design

Strict separation of concerns: `app/streamlit_app.py` contains **zero** mathematical or fitting logic. It only handles UI layout, widget inputs, and rendering.

```
project/
├── ou_pipeline/               # Core business logic & math (UI-agnostic)
│   ├── __init__.py
│   ├── config.py              # PipelineConfig dataclass
│   ├── interpretation.py      # Z-score & plain-English verdict engine
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py          # PriceDataLoader class (yfinance wrapper + cache)
│   ├── estimators/
│   │   ├── __init__.py
│   │   ├── base.py            # OUEstimator ABC
│   │   ├── _shared.py         # Shared math helpers (half-life, transforms)
│   │   ├── ols.py             # OLSEstimator (AR(1) regression)
│   │   └── kalman.py          # KalmanEstimator (State-space MLE)
│   ├── diagnostics/
│   │   ├── __init__.py
│   │   └── stationarity.py    # StationarityTester class (ADF test)
│   └── models/
│       ├── __init__.py
│       └── results.py         # OUResult, StationarityResult dataclasses
├── app/
│   └── streamlit_app.py       # UI ONLY — Streamlit dashboard layout
├── tests/                     # Full Pytest test suite (48 tests)
│   ├── test_loader.py
│   ├── test_stationarity.py
│   ├── test_ols.py
│   ├── test_kalman.py
│   └── test_interpretation.py
├── notebooks/                 # Exploratory runs & integration output
│   └── manual_run.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## ⚡ Quickstart & Running Locally

### 1. Prerequisites & Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 2. Launch Streamlit Dashboard
```bash
streamlit run app/streamlit_app.py
```

### 3. Run Manual Integration Check on ASML
```bash
python notebooks/manual_run.py
```

---

## 🧪 Verification & Test Suite

The test suite contains **48 unit and integration tests** validating math correctness against synthetic data:
- **Synthetic Recovery Test:** Verifies that estimators accurately recover known $(\theta, \mu, \sigma)$ on generated OU paths.
- **Noise Filtering Proof:** Asserts that `KalmanEstimator` recovers mean-reversion speed $\theta$ significantly closer to the true value than `OLSEstimator` on noisy series.
- **Stationarity Classification:** Validates ADF classification on synthetic mean-reverting vs. random-walk paths.

Run the test suite:
```bash
pytest tests/ -v
```

Code formatting and linting:
```bash
black --check .
ruff check .
```

---

## 🔗 Related Projects
- **Binomial Option Pricing Engine:** Quantitative option pricing models in C++/Python.
