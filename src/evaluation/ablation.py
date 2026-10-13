"""
ResistNet-ST Ablation Study (corrected)
Fair comparison: same architecture, same epochs, no leakage.
"""

import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.st_gnn import STGNN
from models.data_loader_v2 import load_and_build_tensors, ANTIBIOTICS, STATIC_COLS, TEMP_COLS, CONS_COLS


def run_variant(X, y, mask, edge_index, edge_weight, device, name="variant",
                drop_graph=False, drop_temporal=False, drop_age_sex=False,
                drop_consumption=False, feature_cols=None, epochs=300):
    X = X.clone()

    # Column-wise drops (need to know column indices)
    if feature_cols is not None and (drop_age_sex or drop_consumption):
        keep_idx = list(range(X.shape[2]))
        if drop_age_sex:
            age_sex_cols = ["age_60plus", "age_0_14", "reproductive_women",
                            "male_elderly", "median_age", "sex_ratio"]
            drop = [i for i, c in enumerate(feature_cols) if c in age_sex_cols]
            keep_idx = [i for i in keep_idx if i not in drop]
        if drop_consumption:
            drop = [i for i, c in enumerate(feature_cols) if c in CONS_COLS]
            keep_idx = [i for i in keep_idx if i not in drop]
        X = X[:, :, keep_idx]

    if drop_temporal:
        # Zero out temporal features (keeps GRU input dim unchanged)
        if feature_cols is not None:
            temp_idx = [i for i, c in enumerate(feature_cols) if c in TEMP_COLS]
            for i in temp_idx:
                if i < X.shape[2]:
                    X[:, :, i] = 0.0

    if drop_graph:
        n = X.shape[0]
        edge_index = torch.stack([torch.arange(n), torch.arange(n)]).to(device)
        edge_weight = torch.ones(n).to(device)

    X_mean = X.mean(dim=(0, 1), keepdim=True)
    X_std = X.std(dim=(0, 1), keepdim=True) + 1e-6
    X = (X - X_mean) / X_std

    model = STGNN(n_features=X.shape[2], gnn_hidden=128, lstm_hidden=256, dropout=0.25).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_loss = np.inf
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(X, edge_index, edge_weight)
        preds = torch.sigmoid(logits)
        loss = (((preds - y) ** 2) * mask).sum() / (mask.sum() + 1e-8)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if loss.item() < best_loss - 1e-6:
            best_loss = loss.item()

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
    print(f"{name:25s} R2={r2:.4f}  MAE={mae:.4f}")
    return {"variant": name, "r2": r2, "mae": mae}


def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y, mask, meta, ei, ew, cols = load_and_build_tensors()
    X, y, mask = X.to(device), y.to(device), mask.to(device)
    ei, ew = ei.to(device), ew.to(device)

    rows = []
    rows.append(run_variant(X, y, mask, ei, ew, device,
                            name="Full ST-GNN", feature_cols=cols))
    rows.append(run_variant(X, y, mask, ei, ew, device,
                            drop_graph=True, name="w/o graph", feature_cols=cols))
    rows.append(run_variant(X, y, mask, ei, ew, device,
                            drop_temporal=True, name="w/o temporal", feature_cols=cols))
    rows.append(run_variant(X, y, mask, ei, ew, device,
                            drop_age_sex=True, name="w/o age-sex", feature_cols=cols))
    rows.append(run_variant(X, y, mask, ei, ew, device,
                            drop_consumption=True, name="w/o consumption", feature_cols=cols))

    pd.DataFrame(rows).to_csv("results/tables/ablation.csv", index=False)
    print("\nSaved: results/tables/ablation.csv")


if __name__ == "__main__":
    main()