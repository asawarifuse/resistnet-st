"""
Consolidate results from all models into one table.
"""

import pandas as pd
from pathlib import Path

TABLES = Path("results/tables")

rows = []

for name, f in [
    ("XGBoost", "xgboost_results.csv"),
    ("LSTM", "lstm_results.csv"),
    ("ST-GNN", "stgnn_results.csv"),
    ("Spatial-only", "spatial_results.csv"),
]:
    p = TABLES / f
    if p.exists():
        d = pd.read_csv(p).iloc[0].to_dict()
        rows.append({"model": name, **d})

df = pd.DataFrame(rows)
out = TABLES / "main_results.csv"
df.to_csv(out, index=False)
print(df.to_string(index=False))
print(f"\nSaved: {out}")