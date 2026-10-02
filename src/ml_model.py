# src/ml_model.py
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Any

from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.feature_engine import build_feature_dataset

# Explicit list of stationary features engineered in Milestone 2
FEATURE_COLS = [
    "log_ret_1", "log_ret_3", "log_ret_5", "log_ret_10", "log_ret_21",
    "dist_sma_10", "dist_sma_20", "dist_sma_50", "dist_sma_200",
    "rsi_14", "macd_diff_norm", "atr_norm", "volatility_20", "bb_pct", "volume_zscore"
]


def train_walk_forward(
        df: pd.DataFrame,
        n_splits: int = 5,
        hyperparams: Dict[str, Any] = None
) -> Tuple[pd.DataFrame, Dict[str, float], XGBClassifier]:
    """
    Perform TimeSeriesSplit (Walk-Forward) validation on the engineered dataset.
    Returns:
        - out-of-sample predictions DataFrame
        - aggregated performance metrics dictionary
        - final trained XGBoost model instance (fitted on all historical data)
    """
    # Exclude the latest row if target is NaN (live prediction row)
    trainable_df = df.dropna(subset=["target"]).copy()
    trainable_df["target"] = trainable_df["target"].astype(int)

    X = trainable_df[FEATURE_COLS]
    y = trainable_df["target"]

    if hyperparams is None:
        hyperparams = {
            "n_estimators": 100,
            "max_depth": 3,  # Shallow trees prevent severe financial overfitting
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "eval_metric": "logloss"
        }

    tscv = TimeSeriesSplit(n_splits=n_splits)

    oof_predictions = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        model = XGBClassifier(**hyperparams)
        model.fit(X_train, y_train)

        probs = model.predict_proba(X_val)[:, 1]
        preds = (probs > 0.50).astype(int)

        val_records = trainable_df.iloc[val_idx].copy()
        val_records["pred_prob"] = probs
        val_records["pred_label"] = preds
        val_records["fold"] = fold + 1
        oof_predictions.append(val_records)

    oof_df = pd.concat(oof_predictions).sort_index()

    # Aggregate performance metrics over all out-of-sample folds
    metrics = {
        "accuracy": float(accuracy_score(oof_df["target"], oof_df["pred_label"])),
        "precision": float(precision_score(oof_df["target"], oof_df["pred_label"], zero_division=0)),
        "recall": float(recall_score(oof_df["target"], oof_df["pred_label"], zero_division=0)),
        "f1": float(f1_score(oof_df["target"], oof_df["pred_label"], zero_division=0)),
        "roc_auc": float(roc_auc_score(oof_df["target"], oof_df["pred_prob"]))
    }

    # Final fit on all available data for future inference
    final_model = XGBClassifier(**hyperparams)
    final_model.fit(X, y)

    return oof_df, metrics, final_model


def get_feature_importances(model: XGBClassifier) -> pd.DataFrame:
    """Extract and sort feature importance scores from the fitted XGBoost model."""
    importances = model.feature_importances_
    fi_df = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": importances
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)
    return fi_df


if __name__ == "__main__":
    ticker = "TATASTEEL.NS"
    print(f"[Model Engine] Loading features for {ticker}...")
    dataset = build_feature_dataset(ticker)

    print(f"[Model Engine] Running 5-Fold TimeSeriesSplit validation...")
    oof_results, metrics, final_model = train_walk_forward(dataset, n_splits=5)

    print("\n--- Out-of-Sample Performance Metrics ---")
    for metric_name, value in metrics.items():
        print(f"  {metric_name.upper():<10}: {value:.4f}")

    print("\n--- Top 5 Predictive Features (Gain) ---")
    fi = get_feature_importances(final_model)
    print(fi.head(5).to_string(index=False))