"""
ResistNet-ST LSTM Baseline (no graph, no spatial).
"""

import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.data_loader import load_all, build_tensor, FEATURE_COLS


class SimpleLSTM(nn.Module):
    def __init__(self, n_features, hidden=128, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        # x: (n_nodes, T, F)
        out, _ = self.lstm(x)
        out = self.drop(out)
        return self.fc(out).squeeze(-1)


def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df, graph = load_all()
    X, y, mask = build_tensor(df, FEATURE_COLS, "resistance_rate",
                              graph["n_nodes"], 12, start_year=2021)
    X, y, mask = X.to(device), y.to(device), mask.to(device)

    X_mean = X.mean(dim=(0, 1), keepdim=True)
    X_std = X.std(dim=(0, 1), keepdim=True) + 1e-6
    X = (X - X_mean) / X_std

    model = SimpleLSTM(X.shape[2]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_loss = np.inf
    patience, wait = 30, 0

    for epoch in range(300):
        model.train()
        opt.zero_grad()
        preds = torch.sigmoid(model(X))
        loss = (((preds - y) ** 2) * mask).sum() / (mask.sum() + 1e-8)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d} | Loss {loss.item():.5f}")

        if loss.item() < best_loss - 1e-6:
            best_loss = loss.item()
            wait = 0
            torch.save(model.state_dict(), "results/models/lstm.pt")
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stop at {epoch}")
                break

    model.load_state_dict(torch.load("results/models/lstm.pt", weights_only=True))
    model.eval()
    with torch.no_grad():
        preds = torch.sigmoid(model(X))

    y_true, y_pred = [], []
    for t in [8, 9, 10, 11]:
        m = mask[:, t].bool()
        y_true.append(y[m, t].cpu().numpy())
        y_pred.append(preds[m, t].cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    print(f"LSTM  R2={r2:.4f}  MAE={mae:.4f}")

    Path("results/tables").mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"model": "LSTM", "r2": r2, "mae": mae}]).to_csv(
        "results/tables/lstm_results.csv", index=False)


if __name__ == "__main__":
    main()