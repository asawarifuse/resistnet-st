"""
ResistNet-ST Evaluation v2 (district x pathogen nodes)
"""

import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.st_gnn import STGNN
from models.data_loader_v2 import load_and_build_tensors


def evaluate_stgnn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y, mask, meta, edge_index, edge_weight, cols = load_and_build_tensors()
    X, y, mask = X.to(device), y.to(device), mask.to(device)
    edge_index, edge_weight = edge_index.to(device), edge_weight.to(device)

    X_mean = X.mean(dim=(0, 1), keepdim=True)
    X_std = X.std(dim=(0, 1), keepdim=True) + 1e-6
    X = (X - X_mean) / X_std

    model = STGNN(n_features=X.shape[2], gnn_hidden=128, lstm_hidden=256, dropout=0.25).to(device)
    model.load_state_dict(torch.load("results/models/st_gnn.pt", weights_only=True))
    model.eval()

    with torch.no_grad():
        preds = torch.sigmoid(model(X, edge_index, edge_weight))

    y_true, y_pred = [], []
    for t in [8, 9, 10, 11]:
        m = mask[:, t].bool()
        y_true.append(y[m, t].cpu().numpy())
        y_pred.append(preds[m, t].cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    print(f"ST-GNN v2  R2={r2:.4f}  MAE={mae:.4f}  RMSE={rmse:.4f}")
    print(f"Pred range: {y_pred.min():.3f} - {y_pred.max():.3f}  mean: {y_pred.mean():.3f}")

    Path("results/tables").mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"model": "ST-GNN", "r2": r2, "mae": mae, "rmse": rmse}]).to_csv(
        "results/tables/stgnn_results.csv", index=False)


if __name__ == "__main__":
    evaluate_stgnn()
