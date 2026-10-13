"""
ResistNet-ST Data Loader v2 (leakage-free)
Nodes = district x pathogen (766 x 5 = 3830).
Antibiotic resistance features are SHIFTED by 1 quarter; target is current quarter.
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path

FEAT_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "features.parquet"
GRAPH_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "district_graph.pt"

PATHOGENS = ["E_coli", "K_pneumoniae", "A_baumannii", "P_aeruginosa", "S_aureus"]
ANTIBIOTICS = [
    "Ampicillin", "Amoxicillin_clavulanate", "Piperacillin_tazobactam",
    "Cefotaxime", "Ceftazidime", "Ceftriaxone", "Cefepime",
    "Meropenem", "Imipenem", "Ciprofloxacin", "Levofloxacin",
    "Gentamicin", "Amikacin", "Trimethoprim_sulfamethoxazole",
]

STATIC_COLS = [
    "population_log", "density", "urban", "literacy", "sex_ratio",
    "age_60plus", "age_0_14", "reproductive_women", "male_elderly",
    "median_age", "beds", "doctors", "labs", "sanitation", "gdp",
    "lat", "lon", "state_id",
]

TEMP_COLS = ["quarter", "year", "time_index", "monsoon", "winter"]

CONS_COLS = ["cons_current", "cons_lag1", "cons_lag2", "cons_roll4",
             "cons_velocity", "access_share", "watch_share",
             "reserve_share", "cons_anomaly"]


def load_and_build_tensors(start_year=2021, n_quarters=12):
    df = pd.read_parquet(FEAT_PATH)
    graph = torch.load(GRAPH_PATH, weights_only=False)

    df["pathogen"] = df["pathogen"].astype(object)
    df["antibiotic"] = df["antibiotic"].astype(object)

    n_districts = graph["n_nodes"]
    n_pathogens = len(PATHOGENS)
    n_nodes = n_districts * n_pathogens

    df["year_offset"] = (df["year"] - start_year) * 4 + (df["quarter"] - 1)

    piv = df.pivot_table(
        index=["district_id", "pathogen", "year_offset"],
        columns="antibiotic",
        values="resistance_rate",
        aggfunc="mean",
    ).reset_index()

    for a in ANTIBIOTICS:
        if a not in piv.columns:
            piv[a] = np.nan

    # District static
    dstat = df.drop_duplicates("district_id")[
        ["district_id"] + STATIC_COLS
    ].reset_index(drop=True)

    # District-quarter consumption
    dcons = df.drop_duplicates(["district_id", "year_offset"])[
        ["district_id", "year_offset"] + CONS_COLS
    ].reset_index(drop=True)

    # Global temporal
    tt = df.drop_duplicates("year_offset")[
        ["year_offset", "quarter", "year", "time_index", "monsoon", "winter"]
    ].reset_index(drop=True)

    piv = piv.merge(dstat, on="district_id", how="left")
    piv = piv.merge(dcons, on=["district_id", "year_offset"], how="left")
    piv = piv.merge(tt, on="year_offset", how="left")

    path_idx = {p: i for i, p in enumerate(PATHOGENS)}
    piv["node_id"] = piv["district_id"] * n_pathogens + piv["pathogen"].map(path_idx)

    piv = piv.sort_values(["node_id", "year_offset"]).reset_index(drop=True)

    # Target = current-quarter mean of antibiotic resistances
    piv["_target"] = piv[ANTIBIOTICS].mean(axis=1, skipna=True)

    # Leakage-free: shift antibiotic resistances by 1 quarter
    piv[ANTIBIOTICS] = piv.groupby("node_id", observed=True)[ANTIBIOTICS].shift(1)

    feature_cols = ANTIBIOTICS + STATIC_COLS + CONS_COLS + TEMP_COLS
    F_dim = len(feature_cols)

    X = np.zeros((n_nodes, n_quarters, F_dim), dtype=np.float32)
    y = np.zeros((n_nodes, n_quarters), dtype=np.float32)
    mask = np.zeros((n_nodes, n_quarters), dtype=np.float32)

    for _, r in piv.iterrows():
        nid = int(r["node_id"])
        t = int(r["year_offset"])
        if 0 <= t < n_quarters:
            X[nid, t, :] = r[feature_cols].values.astype(np.float32)
            tv = r["_target"]
            if not pd.isna(tv):
                y[nid, t] = float(tv)
                mask[nid, t] = 1.0

    X = np.nan_to_num(X, nan=0.0)

    edge_index, edge_weight = build_st_edges(graph, n_districts, n_pathogens)

    return (torch.tensor(X), torch.tensor(y), torch.tensor(mask),
            None, edge_index, edge_weight, feature_cols)


def build_st_edges(graph, n_districts, n_pathogens):
    district_ei = graph["edge_index"]
    district_ew = graph["edge_weight"]

    src, dst, w = [], [], []

    # Same district, cross-pathogen (star)
    for d in range(n_districts):
        base = d * n_pathogens
        for p1 in range(n_pathogens):
            for p2 in range(n_pathogens):
                if p1 == p2:
                    continue
                src.append(base + p1)
                dst.append(base + p2)
                w.append(0.5)

    # Spatial edges within same pathogen
    for p in range(n_pathogens):
        for k in range(district_ei.shape[1]):
            u = int(district_ei[0, k])
            v = int(district_ei[1, k])
            ww = float(district_ew[k])
            src.append(u * n_pathogens + p)
            dst.append(v * n_pathogens + p)
            w.append(ww)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = torch.tensor(w, dtype=torch.float)
    return edge_index, edge_weight


if __name__ == "__main__":
    X, y, mask, meta, ei, ew, cols = load_and_build_tensors()
    print("X:", X.shape)
    print("y:", y.shape)
    print("edges:", ei.shape[1])
    print("features:", len(cols))
    print("mask coverage:", mask.mean().item())