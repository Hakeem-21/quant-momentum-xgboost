# src/data_pipeline.py
import sqlite3
import pandas as pd
import yfinance as yf
from pathlib import Path
import traceback

DB_PATH = Path("data/quant_engine.db")


def init_db(db_path: Path = DB_PATH) -> None:
    """Initialize the SQLite database using schema.sql."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        with open("sql/schema.sql", "r") as f:
            conn.executescript(f.read())


def fetch_and_store(ticker: str, start_date: str = "2018-01-01", end_date: str = None, db_path: Path = DB_PATH) -> int:
    """Fetch daily OHLCV from yfinance, sanitize missing values, and upsert into SQLite."""
    init_db(db_path)

    print(f"[ETL] Fetching data for {ticker} from {start_date}...")
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)

    if df.empty:
        raise ValueError(f"No price data returned for ticker '{ticker}'.")

    # Handle multi-level columns if returned (Price, Ticker)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    # Standardize column headers: lowercase, trimmed, underscores
    df.columns = [str(col).strip().lower().replace(" ", "_") for col in df.columns]

    # Handle missing adjusted close
    if "adj_close" not in df.columns:
        if "close" in df.columns:
            df["adj_close"] = df["close"]
        else:
            raise KeyError("Neither 'adj_close' nor 'close' found in yfinance response.")

    df["ticker"] = ticker
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    # Filter down to the target schema fields
    records = df[["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]].copy()

    # Critical clean-up: Remove rows with null prices (holidays, pre-IPO padding)
    records.dropna(subset=["open", "high", "low", "close"], inplace=True)
    records["volume"] = records["volume"].fillna(0).astype(int)

    if records.empty:
        raise ValueError(f"All returned records for '{ticker}' contained null values.")

    # Safe upsert using a temporary staging table
    with sqlite3.connect(db_path) as conn:
        records.to_sql("temp_ohlcv", conn, if_exists="replace", index=False)

        conn.execute("""
            INSERT OR REPLACE INTO ohlcv_data (ticker, date, open, high, low, close, adj_close, volume)
            SELECT ticker, date, open, high, low, close, adj_close, volume
            FROM temp_ohlcv;
        """)

        conn.execute("DROP TABLE IF EXISTS temp_ohlcv;")

    print(f"[ETL] Successfully saved {len(records)} valid records for {ticker}.")
    return len(records)


def load_from_db(ticker: str, db_path: Path = DB_PATH) -> pd.DataFrame:
    """Read historical data for a ticker from SQLite into a DataFrame."""
    with sqlite3.connect(db_path) as conn:
        query = "SELECT * FROM ohlcv_data WHERE ticker = ? ORDER BY date ASC;"
        df = pd.read_sql_query(query, conn, params=(ticker,))

    if df.empty:
        raise ValueError(f"No records found for {ticker} in {db_path}. Run fetch_and_store first.")

    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    return df


if __name__ == "__main__":
    indian_sector_basket = [
        "OLAELEC.NS",  # EV / Automobile (< 1000)
        "TATASTEEL.NS",  # Metals & Mining (< 1000)
        "ITC.NS",  # FMCG (< 1000)
        "BIOCON.NS",  # Pharmaceuticals (< 1000)
        "NTPC.NS",  # Power & Infrastructure (< 1000)
        "WIPRO.NS"  # Information Technology (< 1000)
    ]

    print("[ETL Pipeline] Ingesting final 6-sector Indian basket...")
    for symbol in indian_sector_basket:
        try:
            fetch_and_store(symbol, start_date="2018-01-01")
        except Exception as e:
            print(f"[Error] Failed to ingest {symbol}: {e}")
            traceback.print_exc()

    print("\n[ETL Pipeline] Multi-sector ingestion complete. Data saved to SQLite.")