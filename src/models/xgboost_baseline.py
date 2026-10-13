"""
ResistNet-ST XGBoost Baseline
Flatten features, no graph, no temporal sequence.
"""

import sys
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.data_loader import load_all, FEATURE_COLS


def main():
    df, _ = load_all()

    # Train: 2021-2022 (time_index 4..11 after lag drops), Test: 2023
    train = df[df["year"] <= 2022].dropna(subset=["resistance_rate"])
    test = df[df["year"] == 2023].dropna(subset=["resistance_rate"])

    X_train = train[FEATURE_COLS].fillna(0).values
    y_train = train["resistance_rate"].values
    X_test = test[FEATURE_COLS].fillna(0).values
    y_test = test["resistance_rate"].values

    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # Recall on high-risk
    y_true_hr = (y_test >= 0.70).astype(int)
    y_pred_hr = (y_pred >= 0.70).astype(int)
    rec = recall_score(y_true_hr, y_pred_hr, zero_division=0)

    print(f"XGBoost  R2={r2:.4f}  MAE={mae:.4f}  Recall={rec:.4f}")

    Path("results/tables").mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"model": "XGBoost", "r2": r2, "mae": mae, "recall": rec}]).to_csv(
        "results/tables/xgboost_results.csv", index=False)


if __name__ == "__main__":
    main()