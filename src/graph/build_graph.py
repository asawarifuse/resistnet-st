"""
ResistNet-ST Graph Construction
Build KNN + state-edge graph for 766 districts.
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from scipy.spatial.distance import cdist


DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "synthetic" / "resistnet_st_dataset.csv"
OUT_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "district_graph.pt"


def load_districts():
    df = pd.read_csv(DATA_PATH, usecols=["district_id", "state_id", "lat", "lon"])
    return df.drop_duplicates("district_id").sort_values("district_id").reset_index(drop=True)


def build_knn_edges(coords, k=5):
    """K-nearest neighbors based on geographic distance."""
    dist = cdist(coords, coords, metric="euclidean")
    np.fill_diagonal(dist, np.inf)
    knn = np.argsort(dist, axis=1)[:, :k]

    src, dst, w = [], [], []
    for i in range(len(coords)):
        for j in knn[i]:
            src.append(i)
            dst.append(j)
            w.append(1.0 / (1.0 + dist[i, j]))
    return src, dst, w


def add_state_edges(df, src, dst, w):
    """Add edges between districts in the same state (sampled to avoid blowup)."""
    rng = np.random.default_rng(42)
    for state, grp in df.groupby("state_id"):
        ids = grp["district_id"].values
        if len(ids) < 2:
            continue
        # Sample up to 5 pairs per state
        n_pairs = min(5, len(ids) * (len(ids) - 1) // 2)
        for _ in range(n_pairs):
            a, b = rng.choice(ids, 2, replace=False)
            src.append(int(a))
            dst.append(int(b))
            w.append(0.5)
            src.append(int(b))
            dst.append(int(a))
            w.append(0.5)
    return src, dst, w


def make_undirected(src, dst, w, n_nodes):
    """Add reverse edges and deduplicate."""
    edge_set = {}
    for s, d, ww in zip(src, dst, w):
        edge_set[(s, d)] = max(edge_set.get((s, d), 0), ww)
        edge_set[(d, s)] = max(edge_set.get((d, s), 0), ww)
    s = [k[0] for k in edge_set]
    d = [k[1] for k in edge_set]
    ww = [edge_set[k] for k in edge_set]
    return torch.tensor([s, d], dtype=torch.long), torch.tensor(ww, dtype=torch.float)


def main():
    df = load_districts()
    n_nodes = len(df)
    coords = df[["lat", "lon"]].values

    src, dst, w = build_knn_edges(coords, k=5)
    src, dst, w = add_state_edges(df, src, dst, w)

    edge_index, edge_weight = make_undirected(src, dst, w, n_nodes)

    # Connectivity check
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    G.add_edges_from(edge_index.t().tolist())
    n_components = nx.number_connected_components(G)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "n_nodes": n_nodes,
        "district_ids": df["district_id"].values,
    }, OUT_PATH)

    print(f"Nodes: {n_nodes}")
    print(f"Edges (directed): {edge_index.shape[1]}")
    print(f"Connected components: {n_components}")
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()