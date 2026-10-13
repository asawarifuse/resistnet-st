"""
ResistNet-ST ST-GNN Training v2 (district x pathogen nodes)
"""

import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.st_gnn import STGNN
from models.data_loader_v2 import load_and_build_tensors


def train():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    X, y, mask, meta, edge_index, edge_weight, cols = load_and_build_tensors()
    X, y, mask = X.to(device), y.to(device), mask.to(device)
    edge_index, edge_weight = edge_index.to(device), edge_weight.to(device)

    X_mean = X.mean(dim=(0, 1), keepdim=True)
    X_std = X.std(dim=(0, 1), keepdim=True) + 1e-6
    X = (X - X_mean) / X_std

    model = STGNN(n_features=X.shape[2], gnn_hidden=128, lstm_hidden=256, dropout=0.25).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    targets = y

    epochs = 500
    patience = 50
    best_loss = np.inf
    wait = 0

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(X, edge_index, edge_weight)
        preds = torch.sigmoid(logits)
        loss = (((preds - targets) ** 2) * mask).sum() / (mask.sum() + 1e-8)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d} | Loss {loss.item():.5f}")

        if loss.item() < best_loss - 1e-6:
            best_loss = loss.item()
            wait = 0
            torch.save(model.state_dict(), "results/models/st_gnn.pt")
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stop at epoch {epoch}")
                break

    print("Best loss:", best_loss)


if __name__ == "__main__":
    Path("results/models").mkdir(parents=True, exist_ok=True)
    train()