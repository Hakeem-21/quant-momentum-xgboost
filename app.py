# app.py
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data_pipeline import load_from_db
from src.feature_engine import build_feature_dataset
from src.ml_model import train_walk_forward, get_feature_importances
from src.back_tester import run_backtest

# Set page configuration
st.set_page_config(
    page_title="Quant Momentum Trading Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- Custom Styling -----------------
st.markdown("""
<style>
    .metric-card {
        background-color: #1e2130;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- Sidebar Configuration -----------------
st.sidebar.title("⚙️ Engine Parameters")

BASKET = {
    "OLAELEC.NS": "Ola Electric (EV / Automobile)",
    "TATASTEEL.NS": "Tata Steel (Metals & Mining)",
    "ITC.NS": "ITC Ltd (FMCG)",
    "BIOCON.NS": "Biocon (Pharmaceuticals)",
    "NTPC.NS": "NTPC (Power & Utilities)",
    "WIPRO.NS": "Wipro (Information Technology)"
}

selected_ticker = st.sidebar.selectbox(
    "Select Equity Ticker",
    options=list(BASKET.keys()),
    format_func=lambda x: f"{x} - {BASKET[x]}"
)

st.sidebar.markdown("---")
st.sidebar.subheader("Model & Backtest Controls")
conviction = st.sidebar.slider("Conviction Threshold (Probability)", 0.50, 0.65, 0.55, 0.01)
slippage_bps = st.sidebar.slider("Transaction Cost / Slippage (bps)", 0.0, 20.0, 5.0, 1.0)
n_splits = st.sidebar.slider("TimeSeriesSplit Folds", 3, 7, 5)

run_button = st.sidebar.button("🚀 Run Backtest Engine", use_container_width=True)

# ----------------- Main Layout -----------------
st.title("📈 Quantitative Momentum Engine (XGBoost Walk-Forward)")
st.markdown(
    f"**Asset:** `{selected_ticker}` | **Relational Store:** `SQLite` | **Validation:** Expanding Walk-Forward CV")

if run_button or "backtest_data" not in st.session_state or st.session_state.get("last_ticker") != selected_ticker:
    with st.spinner(f"Ingesting SQLite features & running walk-forward validation for {selected_ticker}..."):
        try:
            # 1. Feature pipeline
            featured_df = build_feature_dataset(selected_ticker)

            # 2. XGBoost walk-forward
            oof_preds, model_metrics, trained_model = train_walk_forward(featured_df, n_splits=n_splits)

            # 3. Vectorized Backtesting
            results_df, perf_metrics = run_backtest(
                oof_preds,
                conviction_threshold=conviction,
                transaction_cost_bps=slippage_bps
            )

            # 4. Feature Importance
            fi_df = get_feature_importances(trained_model)

            # Store in session state
            st.session_state["backtest_data"] = {
                "results": results_df,
                "perf": perf_metrics,
                "model_metrics": model_metrics,
                "fi": fi_df
            }
            st.session_state["last_ticker"] = selected_ticker
        except Exception as e:
            st.error(f"Error executing engine for {selected_ticker}: {e}")
            st.stop()

# Load cached run
data = st.session_state["backtest_data"]
results_df = data["results"]
perf = data["perf"]
model_metrics = data["model_metrics"]
fi_df = data["fi"]

# ----------------- Top Metric KPIs -----------------
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

strat_ret = perf.get("Total Strategy Return (%)", 0.0)
bench_ret = perf.get("Total Benchmark Return (%)", 0.0)
ret_delta = round(strat_ret - bench_ret, 2)

kpi1.metric("Strategy Return", f"{strat_ret:.2f}%", f"{ret_delta:.2f}% vs Bench")
kpi2.metric("Benchmark Return", f"{bench_ret:.2f}%")
kpi3.metric("Sharpe Ratio", f"{perf.get('Sharpe Ratio', 0.0):.2f}")
kpi4.metric("Win Rate", f"{perf.get('Win Rate (%)', 0.0):.2f}%")
kpi5.metric("Max Drawdown", f"{perf.get('Max Drawdown (%)', 0.0):.2f}%")

st.markdown("---")

# ----------------- Interactive Charts (Plotly) -----------------
tab1, tab2, tab3 = st.tabs(["📊 Equity Curves & Drawdown", "🔬 Model Intelligence & Features", "📋 Trade Logs"])

with tab1:
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Cumulative Equity Performance (Normalized to 1.0)", "Portfolio Drawdown (%)"),
        row_heights=[0.7, 0.3]
    )

    # Strategy Equity
    fig.add_trace(
        go.Scatter(x=results_df.index, y=results_df["strategy_equity"], name="XGBoost Momentum Strategy",
                   line=dict(color="#00FFAA", width=2)),
        row=1, col=1
    )
    # Benchmark Equity
    fig.add_trace(
        go.Scatter(x=results_df.index, y=results_df["benchmark_equity"], name="Buy & Hold Benchmark",
                   line=dict(color="#888888", width=1.5, dash="dash")),
        row=1, col=1
    )

    # Drawdown Chart
    fig.add_trace(
        go.Scatter(x=results_df.index, y=results_df["drawdown"] * 100, name="Strategy Drawdown",
                   fill="tozeroy", line=dict(color="#FF4B4B", width=1)),
        row=2, col=1
    )

    fig.update_layout(
        template="plotly_dark",
        height=600,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text="Equity Multiplier", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown %", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

with tab2:
    col_feat1, col_feat2 = st.columns([1, 1])

    with col_feat1:
        st.subheader("Model Diagnostic Metrics")
        diag_df = pd.DataFrame({
            "Metric": ["ROC-AUC Score", "Precision", "Recall", "F1-Score", "Accuracy"],
            "Score": [
                f"{model_metrics['roc_auc']:.4f}",
                f"{model_metrics['precision']:.4f}",
                f"{model_metrics['recall']:.4f}",
                f"{model_metrics['f1']:.4f}",
                f"{model_metrics['accuracy']:.4f}"
            ]
        })
        st.table(diag_df)

    with col_feat2:
        st.subheader("Top Predictive Features (Information Gain)")
        bar_fig = go.Figure(go.Bar(
            x=fi_df["importance"].head(8),
            y=fi_df["feature"].head(8),
            orientation="h",
            marker=dict(color="#00FFAA")
        ))
        bar_fig.update_layout(
            template="plotly_dark",
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
            yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(bar_fig, use_container_width=True)

with tab3:
    st.subheader("Recent Out-of-Sample Predictions & Execution Logs")
    display_cols = ["close", "pred_prob", "raw_signal", "position", "strategy_return", "strategy_equity"]
    st.dataframe(
        results_df[display_cols].tail(50).sort_index(ascending=False),
        use_container_width=True
    )