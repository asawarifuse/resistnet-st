"""
ResistNet-ST Data Loader
Builds (n_nodes, T, F) tensors from features.parquet.
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path


FEAT_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "features.parquet"
GRAPH_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "district_graph.pt"

FEATURE_COLS = [
    "res_lag1", "res_lag2", "res_lag4", "res_roll3", "res_roll4",
    "res_velocity", "res_acceleration", "res_seasonal", "res_trend",
    "res_volatility",
    "cons_current", "cons_lag1", "cons_lag2", "cons_roll4", "cons_velocity",
    "access_share", "watch_share", "reserve_share", "cons_anomaly",
    "population_log", "density", "urban", "literacy", "sex_ratio",
    "age_60plus", "age_0_14", "reproductive_women", "male_elderly",
    "median_age", "beds", "doctors", "labs", "sanitation", "gdp",
    "lat", "lon", "state_id",
    "quarter", "year", "time_index", "monsoon", "winter",
    "pathogen_id", "antibiotic_id",
]


def load_all():
    df = pd.read_parquet(FEAT_PATH)
    graph = torch.load(GRAPH_PATH, weights_only=False)

    for c in FEATURE_COLS:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    return df, graph


def build_tensor(df, feature_cols, target_col, n_nodes, n_quarters, start_year=2021):
    """Aggregate to (n_nodes, T, F). Target = 1 if >=30% of pairs exceed 0.70."""
    df = df.copy()
    df["year_offset"] = (df["year"] - start_year) * 4 + (df["quarter"] - 1)

    X = np.zeros((n_nodes, n_quarters, len(feature_cols)), dtype=np.float32)
    y = np.zeros((n_nodes, n_quarters), dtype=np.float32)
    mask = np.zeros((n_nodes, n_quarters), dtype=np.float32)

    # Feature aggregation: mean
    feat_grouped = df.groupby(["district_id", "year_offset"])[feature_cols].mean().reset_index()

    # Target: fraction of pairs with resistance >= 0.70
    def frac_high(s):
        v = s.dropna().values
        if len(v) == 0:
            return np.nan
        return float((v >= 0.70).mean())

    tgt_grouped = df.groupby(["district_id", "year_offset"])[target_col].apply(frac_high).reset_index()
    tgt_grouped.columns = ["district_id", "year_offset", "frac_high"]

    grouped = feat_grouped.merge(tgt_grouped, on=["district_id", "year_offset"])

    for _, row in grouped.iterrows():
        d = int(row["district_id"])
        t = int(row["year_offset"])
        if 0 <= t < n_quarters and 0 <= d < n_nodes:
            X[d, t, :] = row[feature_cols].values.astype(np.float32)
            y[d, t] = 1.0 if row["frac_high"] >= 0.30 else 0.0
            mask[d, t] = 1.0

    return torch.tensor(X), torch.tensor(y), torch.tensor(mask)


if __name__ == "__main__":
    df, graph = load_all()
    print("Features rows:", len(df))
    print("Feature count:", len(FEATURE_COLS))
    print("Nodes:", graph["n_nodes"])

    X, y, mask = build_tensor(df, FEATURE_COLS, "resistance_rate",
                              graph["n_nodes"], 12, start_year=2021)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Mask coverage:", mask.mean().item())
    print("y mean (all):", (y * mask).sum().item() / mask.sum().item())
    print("y high_risk frac (all):", (y * mask).sum().item() / mask.sum().item())
    print("y high_risk frac (test 8:12):",
          (y[:, 8:12] * mask[:, 8:12]).sum().item() / mask[:, 8:12].sum().item())