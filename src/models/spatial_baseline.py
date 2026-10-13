"""
ResistNet-ST Spatial-only baseline (no temporal, no GNN).
Uses neighbor mean resistance at each timestep.
"""

import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.data_loader import load_all, build_tensor, FEATURE_COLS


def main():
    df, graph = load_all()
    X, y, mask = build_tensor(df, FEATURE_COLS, "resistance_rate",
                              graph["n_nodes"], 12, start_year=2021)
    edge_index = graph["edge_index"]

    n = graph["n_nodes"]
    adj = torch.zeros(n, n)
    for s, d in edge_index.t().tolist():
        adj[s, d] = 1.0
    deg = adj.sum(dim=1, keepdim=True).clamp(min=1.0)
    adj = adj / deg

    lag1 = X[:, :, 0]                 # res_lag1
    neigh_lag1 = adj @ lag1           # (n, T)

    # Linear calibration on train timesteps 4..7
    train_mask = torch.zeros_like(mask)
    train_mask[:, 4:8] = mask[:, 4:8]

    x_tr = neigh_lag1[train_mask.bool()].numpy()
    y_tr = y[train_mask.bool()].numpy()
    A = np.vstack([x_tr, np.ones_like(x_tr)]).T
    coef, *_ = np.linalg.lstsq(A, y_tr, rcond=None)
    a, b = coef

    preds = a * neigh_lag1 + b

    y_true, y_pred = [], []
    for t in [8, 9, 10, 11]:
        m = mask[:, t].bool()
        y_true.append(y[m, t].numpy())
        y_pred.append(preds[m, t].numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    print(f"Spatial-only  R2={r2:.4f}  MAE={mae:.4f}  (a={a:.3f}, b={b:.3f})")

    Path("results/tables").mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"model": "Spatial-only", "r2": r2, "mae": mae}]).to_csv(
        "results/tables/spatial_results.csv", index=False)


if __name__ == "__main__":
    main()