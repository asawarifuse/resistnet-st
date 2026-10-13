"""
ResistNet-ST Figures
"""

import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.st_gnn import STGNN
from models.data_loader_v2 import load_and_build_tensors


FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)
sns.set_style("whitegrid")


def fig_pred_vs_actual(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.2, s=8)
    ax.plot([0, 1], [0, 1], "r--", lw=1)
    ax.set_xlabel("True resistance")
    ax.set_ylabel("Predicted resistance")
    ax.set_title("Predicted vs Actual (2023)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_pred_vs_actual.png", dpi=150)
    plt.close()


def fig_results_bar(df_results):
    fig, ax = plt.subplots(figsize=(7, 4))
    df_results = df_results.sort_values("r2")
    ax.barh(df_results["model"], df_results["r2"], color="#3a7")
    ax.set_xlabel("R²")
    ax.set_title("Model comparison (test 2023)")
    ax.axvline(0, color="k", lw=0.5)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_results_bar.png", dpi=150)
    plt.close()


def fig_ablation(df_abl):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(df_abl["variant"], df_abl["r2"], color="#c84")
    ax.set_xlabel("R²")
    ax.set_title("Ablation study")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_ablation.png", dpi=150)
    plt.close()


def fig_dml(df_dml):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(df_dml["group"], df_dml["effect"], color="#57a")
    for i, row in df_dml.iterrows():
        if not np.isnan(row["ci_lo"]):
            ax.plot([row["ci_lo"], row["ci_hi"]], [i, i], "k-", lw=2)
    ax.set_xlabel("ATE (consumption → resistance)")
    ax.set_title("DML causal effect")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_dml.png", dpi=150)
    plt.close()


def fig_spatial_map(df):
    """District-level mean resistance on lat/lon."""
    d = df.groupby("district_id").agg(
        lat=("lat", "first"), lon=("lon", "first"),
        res=("resistance_rate", "mean"),
    ).reset_index()
    fig, ax = plt.subplots(figsize=(7, 7))
    sc = ax.scatter(d["lon"], d["lat"], c=d["res"], cmap="YlOrRd", s=15)
    plt.colorbar(sc, label="Mean resistance")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Spatial distribution of resistance")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_spatial_map.png", dpi=150)
    plt.close()


def fig_lead_time(lead_q_median, n_matched):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Median lead time"], [lead_q_median], color="#a35")
    ax.set_ylabel("Quarters ahead")
    ax.set_title(f"Lead time (n={n_matched} districts)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_lead_time.png", dpi=150)
    plt.close()


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

    yt, yp = [], []
    for t in [8, 9, 10, 11]:
        mm = m_np[:, t].astype(bool)
        yt.append(y_np[mm, t])
        yp.append(preds[mm, t])
    yt = np.concatenate(yt)
    yp = np.concatenate(yp)

    fig_pred_vs_actual(yt, yp)

    if Path("results/tables/main_results.csv").exists():
        fig_results_bar(pd.read_csv("results/tables/main_results.csv"))
    if Path("results/tables/ablation.csv").exists():
        fig_ablation(pd.read_csv("results/tables/ablation.csv"))
    if Path("results/tables/dml_results.csv").exists():
        fig_dml(pd.read_csv("results/tables/dml_results.csv"))

    df = pd.read_parquet("data/processed/features.parquet")
    fig_spatial_map(df)

    # Lead time
    from evaluation.extended_metrics import lead_time_analysis
    lt = lead_time_analysis(y_np, preds, m_np, threshold=0.70)
    fig_lead_time(lt["median_lead_q"], lt["n_matched"])

    print("Figures saved to", FIG_DIR)


if __name__ == "__main__":
    main()