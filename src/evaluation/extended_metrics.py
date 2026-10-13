"""
ResistNet-ST Extended Evaluation
Lead-time + calibration (ECE).
"""

import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.st_gnn import STGNN
from models.data_loader_v2 import load_and_build_tensors


def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        avg_conf = y_prob[mask].mean()
        avg_acc = (y_true[mask] >= 0.70).mean()  # high-risk threshold
        ece += (mask.sum() / len(y_true)) * abs(avg_conf - avg_acc)
    return ece


def lead_time_analysis(y_true_all, y_pred_all, mask_all, threshold=0.70, horizon=4):
    """For each node, check if prediction crosses threshold `horizon` quarters
    before ground truth does. Returns median lead time in quarters."""
    lead_times = []
    n_nodes, T = y_true_all.shape
    for n in range(n_nodes):
        first_true = None
        first_pred = None
        for t in range(T):
            if not mask_all[n, t]:
                continue
            if first_true is None and y_true_all[n, t] >= threshold:
                first_true = t
            if first_pred is None and y_pred_all[n, t] >= threshold:
                first_pred = t
            if first_true is not None and first_pred is not None:
                break
        if first_true is not None and first_pred is not None and first_pred <= first_true:
            lead_times.append(first_true - first_pred)
    if not lead_times:
        return {"median_lead_q": 0, "n_matched": 0}
    return {
        "median_lead_q": float(np.median(lead_times)),
        "mean_lead_q": float(np.mean(lead_times)),
        "n_matched": len(lead_times),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y, mask, meta, ei, ew, cols = load_and_build_tensors()
    X, y, mask = X.to(device), y.to(device), mask.to(device)
    ei, ew = ei.to(device), ew.to(device)

    X_mean = X.mean(dim=(0, 1), keepdim=True)
    X_std = X.std(dim=(0, 1), keepdim=True) + 1e-6
    X = (X - X_mean) / X_std

    model = STGNN(n_features=X.shape[2], gnn_hidden=128, lstm_hidden=256, dropout=0.25).to(device)
    model.load_state_dict(torch.load("results/models/st_gnn.pt", weights_only=True))
    model.eval()

    with torch.no_grad():
        preds = torch.sigmoid(model(X, ei, ew)).cpu().numpy()

    y_np = y.cpu().numpy()
    m_np = mask.cpu().numpy()

    # Lead time (whole series, all timesteps with mask)
    lt = lead_time_analysis(y_np, preds, m_np, threshold=0.70)
    print("Lead-time analysis:", lt)

    # Calibration (test only, binary high-risk)
    y_true, y_prob = [], []
    for t in [8, 9, 10, 11]:
        m = m_np[:, t].astype(bool)
        y_true.append(y_np[m, t])
        y_prob.append(preds[m, t])
    y_true = np.concatenate(y_true)
    y_prob = np.concatenate(y_prob)

    ece = expected_calibration_error(y_true, y_prob, n_bins=10)
    print(f"ECE: {ece:.4f}")

    Path("results/tables").mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "median_lead_q": lt["median_lead_q"],
        "mean_lead_q": lt.get("mean_lead_q", 0),
        "n_matched": lt["n_matched"],
        "ece": ece,
    }]).to_csv("results/tables/extended_metrics.csv", index=False)
    print("Saved: results/tables/extended_metrics.csv")


if __name__ == "__main__":
    main()