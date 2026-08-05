# OU Mean-Reversion Pipeline

A Python pipeline that estimates Ornstein-Uhlenbeck (OU) parameters for a stock's
price series using multiple estimation methods, tests whether mean-reversion is
statistically supported, and exposes results through a Streamlit dashboard.

## Project Structure

```
ou_pipeline/   — Core library (data, estimators, diagnostics, models, config)
app/           — Streamlit UI (no fitting/statistical logic)
tests/         — Pytest test suite
notebooks/     — Exploratory notebooks (never imported by app or pipeline)
```

## Quickstart

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## Development

```bash
pip install -e ".[dev]"
pytest
black --check .
ruff check .
```
