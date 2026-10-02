# Quantitative Momentum Trading Engine (Indian Equities)

An end-to-end quantitative trading and machine learning framework designed for Indian equities (NSE).

The engine implements an idempotent SQLite data pipeline, constructs stationary feature matrices, evaluates an XGBoost classifier via non-leaking Walk-Forward Cross-Validation (`TimeSeriesSplit`), and executes a vectorized backtest with conviction thresholding, trend filtering, and realistic transaction slippage.

Includes an interactive **Streamlit & Plotly** dashboard for out-of-sample performance attribution, dynamic drawdown tracking, and feature importance diagnostics.

---

## System Architecture

```text
                  +--------------------------------+
                  | Yahoo Finance API (NSE Equities)|
                  +----------------+---------------+
                                   |
                                   v  (Cleanse, Flatten Multi-Index, Deduplicate)
                  +----------------+---------------+
                  | Idempotent ETL Pipeline        |
                  | (src/data_pipeline.py)          |
                  +----------------+---------------+
                                   |
                                   v  (Atomic Upsert / Staging Pattern)
                  +----------------+---------------+
                  | SQLite Database                 |
                  | (data/quant_engine.db)          |
                  +----------------+---------------+
                                   |
                                   v  (Stationary Indicators & 5-Day Forward Targets)
                  +----------------+---------------+
                  | Feature Engineering Engine      |
                  | (src/feature_engine.py)         |
                  +----------------+---------------+
                                   |
                                   v  (Temporal Expanding Window Validation)
                  +----------------+---------------+
                  | Walk-Forward XGBoost Classifier |
                  | (src/ml_model.py)               |
                  +----------------+---------------+
                                   |
                                   v  (Conviction Hurdle >= 0.55 + 20-SMA Filter)
                  +----------------+---------------+
                  | Vectorized Backtesting Engine   |
                  | (src/backtester.py)             |
                  +----------------+---------------+
                                   |
                                   v  (Dynamic Sliders, Equity Curves, Diagnostics)
                  +----------------+---------------+
                  | Interactive Streamlit UI        |
                  | (app.py)                        |
                  +--------------------------------+
```
---
## Key Features

- **Relational Data Layer (SQLite):** Composite primary keys (`ticker`, `date`) enforce unique temporal bars; atomic `INSERT OR REPLACE` staging patterns ensure idempotent pipeline re-runs.
- **Feature Engineering & Stationarity:** Eliminates non-stationary raw price levels; computes multi-period log returns, SMA distance ratios, normalized ATR, 14-day Wilder RSI, and 20-day volume Z-scores.
- **Walk-Forward Validation:** Employs an expanding window `TimeSeriesSplit` to eliminate temporal lookahead bias during out-of-sample XGBoost evaluation.
- **Conviction-Based Backtester:** Enforces a probability hurdle ($\ge 0.55$), a 20-day SMA trend filter, and realistic transaction friction (5 bps slippage).

--- 
## Cross-Sector Validation Basket

---

| **Sector**            | **Ticker**     | **Description**                     |
|-----------------------|----------------|-------------------------------------|
| **Metals & Mining**   | `TATASTEEL.NS` | Cyclical, commodity-driven volatility |
| **FMCG**              | `ITC.NS`       | Defensive, low-beta consumer goods    |
| **Automobile / EV**   | `OLAELEC.NS`   | Growth / high-volatility recent listing |
| **Pharmaceuticals**   | `BIOCON.NS`    | Healthcare / biotech momentum         |
| **Power & Utilities** | `NTPC.NS`      | Capital-intensive infrastructure      |
| **Information Tech**  | `WIPRO.NS`     | Global IT export services              |
---
## 🚀 Quickstart Installation & Execution

### 1. Environment Setup

```bash
git clone https://github.com/<your-username>/quant-momentum-xgboost.git
cd quant-momentum-xgboost

python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```
### 2. Ingest Historical Data

```bash
python -m src.data_pipeline
```
### 3. Run Pipeline Diagnostics & Batch Backtesting

```bash
python -m src.backtester
```
### 4. Launch the Interactive Dashboard

```bash
streamlit run app.py
```