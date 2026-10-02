# src/feature_engine.py
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path

DB_PATH = Path("data/quant_engine.db")


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index (RSI) using exponential moving average."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range (ATR)."""
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def generate_features(df: pd.DataFrame, forward_horizon: int = 1) -> pd.DataFrame:
    """
    Generate stationary features and forward classification target from OHLCV data.
    Ensures zero lookahead leakage for feature calculations.
    """
    data = df.copy().sort_index()

    # --- 1. Multi-horizon Log Returns ---
    for k in [1, 3, 5, 10, 21]:
        data[f"log_ret_{k}"] = np.log(data["close"] / data["close"].shift(k))

    # --- 2. Moving Average Distances ---
    for w in [10, 20, 50, 200]:
        sma = data["close"].rolling(window=w).mean()
        data[f"dist_sma_{w}"] = (data["close"] - sma) / sma

    # --- 3. Momentum Oscillators ---
    data["rsi_14"] = compute_rsi(data["close"], period=14)

    ema_12 = data["close"].ewm(span=12, adjust=False).mean()
    ema_26 = data["close"].ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    data["macd_diff_norm"] = (macd_line - signal_line) / data["close"]

    # --- 4. Volatility Indicators ---
    atr_14 = compute_atr(data, period=14)
    data["atr_norm"] = atr_14 / data["close"]

    rolling_std_20 = data["log_ret_1"].rolling(window=20).std()
    data["volatility_20"] = rolling_std_20 * np.sqrt(252)

    sma_20 = data["close"].rolling(window=20).mean()
    rolling_price_std = data["close"].rolling(window=20).std()
    data["bb_pct"] = (data["close"] - (sma_20 - 2 * rolling_price_std)) / (4 * rolling_price_std).replace(0, np.nan)

    # --- 5. Volume Signals ---
    vol_mean_20 = data["volume"].rolling(window=20).mean()
    vol_std_20 = data["volume"].rolling(window=20).std().replace(0, np.nan)
    data["volume_zscore"] = (data["volume"] - vol_mean_20) / vol_std_20

    # --- 6. Forward-Looking Classification Target ---
    forward_ret = (data["close"].shift(-forward_horizon) - data["close"]) / data["close"]
    data["target"] = (forward_ret > 0).astype(int)

    # For the last N rows, the forward target is unknown (preserve as NaN for live inference)
    data.loc[data.index[-forward_horizon:], "target"] = np.nan

    return data


def build_feature_dataset(ticker: str, db_path: Path = DB_PATH) -> pd.DataFrame:
    """Load raw ticker bars from SQLite, engineer features, and clean NaNs."""
    with sqlite3.connect(db_path) as conn:
        query = "SELECT * FROM ohlcv_data WHERE ticker = ? ORDER BY date ASC;"
        df = pd.read_sql_query(query, conn, params=(ticker,))

    if df.empty:
        raise ValueError(f"No records found for {ticker} in {db_path}.")

    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)

    featured_df = generate_features(df)

    # Drop indicator warmup periods (e.g. 200 SMA), except the very last row which is for prediction
    feature_cols = [c for c in featured_df.columns if
                    c not in ["ticker", "open", "high", "low", "close", "adj_close", "volume", "target"]]

    # Keep only rows where all features are computed
    valid_mask = featured_df[feature_cols].notna().all(axis=1)
    cleaned_df = featured_df[valid_mask].copy()

    return cleaned_df


if __name__ == "__main__":
    test_ticker = "TATASTEEL.NS"
    print(f"[Feature Engine] Computing indicators for {test_ticker}...")
    dataset = build_feature_dataset(test_ticker)

    feature_columns = [col for col in dataset.columns if
                       col not in ["ticker", "open", "high", "low", "close", "adj_close", "volume", "target"]]

    print(f"\n[Feature Engine] Completed successfully!")
    print(f"Total Rows: {len(dataset)}")
    print(f"Features Engineered ({len(feature_columns)}):")
    for f in feature_columns:
        print(f"  - {f}")

    print("\nSample Data (Latest 3 rows):")
    print(dataset[feature_columns + ["target"]].tail(3))