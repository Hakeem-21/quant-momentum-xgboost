# src/backtester.py
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from src.feature_engine import build_feature_dataset
from src.ml_model import train_walk_forward


def run_backtest(
    oof_df: pd.DataFrame,
    conviction_threshold: float = 0.55,  # <-- Change from 0.52 to 0.55 here
    transaction_cost_bps: float = 5.0
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Vectorized backtest of walk-forward out-of-fold XGBoost predictions.

    Parameters:
        oof_df: DataFrame containing dates, true close prices, and 'pred_prob'.
        conviction_threshold: Minimum probability required to take a long position.
        transaction_cost_bps: Transaction cost / slippage per trade in basis points (1 bps = 0.0001).
    """
    df = oof_df.copy().sort_index()

    # 1. Base Asset Returns
    df["asset_return"] = df["close"].pct_change().fillna(0)

    # 2. Binary Trading Signal based on threshold (1 = Long, 0 = Cash)
    df["raw_signal"] = (df["pred_prob"] >= conviction_threshold).astype(int)

    # 3. Execution Lag: Today's closing prediction executes for tomorrow's market return
    df["position"] = df["raw_signal"].shift(1).fillna(0)

    # 4. Transaction Cost on Trade Execution (when position changes)
    cost_factor = transaction_cost_bps / 10000.0
    df["trade_action"] = (df["position"] != df["position"].shift(1)).astype(int)
    df["trade_cost"] = df["trade_action"] * cost_factor

    # 5. Net Strategy Returns
    df["strategy_return"] = (df["position"] * df["asset_return"]) - df["trade_cost"]

    # 6. Cumulative Equity Curves (Starting Capital normalized to 1.0)
    df["benchmark_equity"] = (1.0 + df["asset_return"]).cumprod()
    df["strategy_equity"] = (1.0 + df["strategy_return"]).cumprod()

    # 7. Drawdown Calculations
    df["strategy_peak"] = df["strategy_equity"].cummax()
    df["drawdown"] = (df["strategy_equity"] - df["strategy_peak"]) / df["strategy_peak"]

    # Compute Performance Metrics
    metrics = calculate_performance_metrics(df)

    return df, metrics


def calculate_performance_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate standard portfolio risk and return performance metrics."""
    strat_rets = df["strategy_return"]
    n_days = len(df)
    trading_days_per_year = 252

    if n_days == 0:
        return {}

    # Total and Annualized Returns
    total_strat_return = df["strategy_equity"].iloc[-1] - 1.0
    total_bench_return = df["benchmark_equity"].iloc[-1] - 1.0

    years = n_days / trading_days_per_year
    cagr = (df["strategy_equity"].iloc[-1]) ** (1.0 / years) - 1.0 if years > 0 else 0.0
    bench_cagr = (df["benchmark_equity"].iloc[-1]) ** (1.0 / years) - 1.0 if years > 0 else 0.0

    # Volatility & Sharpe Ratio (Risk-free assumed 0 for simplicity)
    daily_vol = strat_rets.std()
    annualized_vol = daily_vol * np.sqrt(trading_days_per_year)
    sharpe_ratio = (strat_rets.mean() / daily_vol * np.sqrt(trading_days_per_year)) if daily_vol > 0 else 0.0

    # Downside Volatility & Sortino Ratio
    downside_rets = strat_rets[strat_rets < 0]
    downside_vol = downside_rets.std() * np.sqrt(trading_days_per_year)
    sortino_ratio = (strat_rets.mean() * np.sqrt(trading_days_per_year) / downside_vol) if downside_vol > 0 else 0.0

    # Maximum Drawdown
    max_drawdown = df["drawdown"].min()

    # Trade Statistics
    trades = df[df["position"] == 1]
    active_days = len(trades)
    win_trades = len(trades[trades["strategy_return"] > 0])
    win_rate = (win_trades / active_days) if active_days > 0 else 0.0

    gross_gains = strat_rets[strat_rets > 0].sum()
    gross_losses = abs(strat_rets[strat_rets < 0].sum())
    profit_factor = (gross_gains / gross_losses) if gross_losses > 0 else 0.0

    return {
        "Total Strategy Return (%)": round(total_strat_return * 100, 2),
        "Total Benchmark Return (%)": round(total_bench_return * 100, 2),
        "Strategy CAGR (%)": round(cagr * 100, 2),
        "Benchmark CAGR (%)": round(bench_cagr * 100, 2),
        "Sharpe Ratio": round(sharpe_ratio, 2),
        "Sortino Ratio": round(sortino_ratio, 2),
        "Max Drawdown (%)": round(max_drawdown * 100, 2),
        "Win Rate (%)": round(win_rate * 100, 2),
        "Profit Factor": round(profit_factor, 2),
        "Total Trades Executed": int(df["trade_action"].sum()),
        "Exposure (%)": round((active_days / n_days) * 100, 2)
    }


if __name__ == "__main__":
    test_ticker = "TATASTEEL.NS"
    print(f"[Backtester] Fetching features and training walk-forward for {test_ticker}...")
    dataset = build_feature_dataset(test_ticker)
    oof_results, model_metrics, _ = train_walk_forward(dataset, n_splits=5)

    print(f"[Backtester] Running simulation with 5 bps transaction cost...")
    results_df, perf = run_backtest(oof_results, conviction_threshold=0.52, transaction_cost_bps=5.0)

    print("\n" + "=" * 45)
    print(f"   BACKTEST PERFORMANCE: {test_ticker}")
    print("=" * 45)
    for k, v in perf.items():
        print(f"  {k:<30}: {v}")
    print("=" * 45)

if __name__ == "__main__":
    basket = [
        "OLAELEC.NS",  # Auto / EV
        "TATASTEEL.NS",  # Metals
        "ITC.NS",  # FMCG
        "BIOCON.NS",  # Pharma
        "NTPC.NS",  # Power
        "WIPRO.NS"  # IT
    ]

    summary = []

    print("\n" + "=" * 70)
    print("      RUNNING BATCH BACKTEST ACROSS ALL 6 SECTORS")
    print("=" * 70)

    for ticker in basket:
        try:
            print(f"\n[Evaluating {ticker}]...")
            dataset = build_feature_dataset(ticker)

            # Skip if not enough historical bars for 5 folds
            if len(dataset) < 100:
                print(f"Skipping {ticker}: insufficient history ({len(dataset)} bars).")
                continue
            oof_results, model_metrics, _ = train_walk_forward(dataset, n_splits=5)

            # Pass conviction_threshold=0.55 here:
            results_df, perf = run_backtest(
                oof_results,
                conviction_threshold=0.55,
                transaction_cost_bps=5.0
            )

            perf["Ticker"] = ticker
            perf["AUC"] = round(model_metrics["roc_auc"], 3)
            summary.append(perf)
        except Exception as e:
            print(f"Error evaluating {ticker}: {e}")

    # Display comparison table
    if summary:
        summary_df = pd.DataFrame(summary).set_index("Ticker")
        cols_to_show = [
            "AUC", "Total Strategy Return (%)", "Total Benchmark Return (%)",
            "Sharpe Ratio", "Max Drawdown (%)", "Win Rate (%)"
        ]
        print("\n" + "=" * 70)
        print("                 CROSS-SECTOR PERFORMANCE SUMMARY")
        print("=" * 70)
        print(summary_df[cols_to_show].to_string())
        print("=" * 70)