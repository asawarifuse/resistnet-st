"""
ResistNet-ST Dataset Generator
Synthetic AMR dataset for 766 Indian districts with spatial autocorrelation.
"""

import numpy as np
import pandas as pd
import yaml
import torch
from pathlib import Path


CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


PATHOGENS = ["E_coli", "K_pneumoniae", "A_baumannii", "P_aeruginosa", "S_aureus"]

ANTIBIOTICS = [
    "Ampicillin", "Amoxicillin_clavulanate", "Piperacillin_tazobactam",
    "Cefotaxime", "Ceftazidime", "Ceftriaxone", "Cefepime",
    "Meropenem", "Imipenem", "Ciprofloxacin", "Levofloxacin",
    "Gentamicin", "Amikacin", "Trimethoprim_sulfamethoxazole",
]

ICMR_ANCHORS = {
    ("E_coli", "Meropenem"): 0.37,
    ("E_coli", "Cefotaxime"): 0.65,
    ("E_coli", "Ceftriaxone"): 0.65,
    ("E_coli", "Ciprofloxacin"): 0.55,
    ("E_coli", "Ampicillin"): 0.75,
    ("E_coli", "Trimethoprim_sulfamethoxazole"): 0.60,
    ("K_pneumoniae", "Piperacillin_tazobactam"): 0.74,
    ("K_pneumoniae", "Meropenem"): 0.45,
    ("K_pneumoniae", "Imipenem"): 0.45,
    ("K_pneumoniae", "Cefotaxime"): 0.70,
    ("A_baumannii", "Meropenem"): 0.72,
    ("A_baumannii", "Imipenem"): 0.72,
    ("P_aeruginosa", "Meropenem"): 0.40,
    ("P_aeruginosa", "Imipenem"): 0.40,
    ("P_aeruginosa", "Ciprofloxacin"): 0.35,
    ("S_aureus", "Levofloxacin"): 0.40,
    ("S_aureus", "Ciprofloxacin"): 0.40,
}

DEFAULT_RESISTANCE = 0.40


AGE_SEX_COEFFICIENTS = {
    ("E_coli", "Ciprofloxacin"): [("male_60plus", 0.12)],
    ("E_coli", "Ampicillin"): [("female_18_40", 0.10)],
    ("E_coli", "Trimethoprim_sulfamethoxazole"): [("female_18_40", 0.08)],
    ("K_pneumoniae", "Cefotaxime"): [("male_under18", 0.18)],
    ("K_pneumoniae", "Meropenem"): [("male_60plus", 0.10)],
    ("S_aureus", "Levofloxacin"): [("age_85plus", 0.15)],
    ("A_baumannii", "Meropenem"): [("male_15_50", 0.12)],
    ("P_aeruginosa", "Ciprofloxacin"): [("male_60plus", 0.10)],
}


def generate_district_scaffold(n_districts=766, seed=42):
    rng = np.random.default_rng(seed)
    state_ids = rng.integers(1, 37, size=n_districts)
    lat = rng.uniform(8.0, 35.0, size=n_districts)
    lon = rng.uniform(68.0, 97.0, size=n_districts)

    population = rng.lognormal(mean=13.0, sigma=0.8, size=n_districts).astype(int)
    population = np.clip(population, 50_000, 20_000_000)
    population_log = np.log10(population)

    density = rng.lognormal(mean=5.5, sigma=0.9, size=n_districts)
    urban = rng.beta(2, 3, size=n_districts)
    literacy = rng.normal(0.72, 0.10, size=n_districts).clip(0.40, 0.98)
    sex_ratio = rng.normal(940, 40, size=n_districts).clip(800, 1100)
    age_60plus = rng.normal(0.09, 0.03, size=n_districts).clip(0.03, 0.20)
    age_0_14 = rng.normal(0.28, 0.05, size=n_districts).clip(0.15, 0.40)
    reproductive_women = rng.normal(0.22, 0.03, size=n_districts).clip(0.15, 0.30)
    male_elderly = (age_60plus * 0.48 * rng.normal(1.0, 0.1, n_districts)).clip(0.01, 0.12)
    median_age = rng.normal(28, 4, size=n_districts).clip(20, 40)

    beds = rng.lognormal(mean=0.8, sigma=0.5, size=n_districts)
    doctors = rng.lognormal(mean=0.3, sigma=0.5, size=n_districts)
    labs = rng.integers(1, 25, size=n_districts)
    sanitation = rng.beta(5, 3, size=n_districts)
    gdp = rng.lognormal(mean=10.5, sigma=0.6, size=n_districts)

    district_id = np.arange(n_districts)

    return pd.DataFrame({
        "district_id": district_id,
        "district_name": [f"D{d:04d}" for d in district_id],
        "state_id": state_ids,
        "lat": lat, "lon": lon,
        "population": population, "population_log": population_log,
        "density": density, "urban": urban, "literacy": literacy,
        "sex_ratio": sex_ratio, "age_60plus": age_60plus,
        "age_0_14": age_0_14, "reproductive_women": reproductive_women,
        "male_elderly": male_elderly, "median_age": median_age,
        "beds": beds, "doctors": doctors, "labs": labs,
        "sanitation": sanitation, "gdp": gdp,
    })


def generate_base_resistance(seed=42):
    rows = []
    rng = np.random.default_rng(seed)
    for p in PATHOGENS:
        for a in ANTIBIOTICS:
            base = ICMR_ANCHORS.get((p, a), DEFAULT_RESISTANCE + rng.normal(0, 0.05))
            base = float(np.clip(base, 0.05, 0.90))
            rows.append({"pathogen": p, "antibiotic": a, "base_resistance": base})
    return pd.DataFrame(rows)


def apply_district_adjustment(base_table, districts, seed=42):
    """District effects with spatial autocorrelation via graph smoothing."""
    rng = np.random.default_rng(seed)
    n_d = len(districts)

    beds_z = (districts["beds"] - districts["beds"].mean()) / districts["beds"].std()
    san_centered = districts["sanitation"] - 0.5
    lit_centered = districts["literacy"] - 0.72

    independent_noise = rng.normal(0, 0.03, n_d)

    graph_path = Path(__file__).resolve().parents[2] / "data" / "processed" / "district_graph.pt"
    if graph_path.exists():
        g = torch.load(graph_path, weights_only=False)
        ei = g["edge_index"].numpy()
        ew = g["edge_weight"].numpy()
        neigh = np.zeros(n_d)
        wsum = np.zeros(n_d)
        for k in range(ei.shape[1]):
            u, v = ei[0, k], ei[1, k]
            neigh[v] += ew[k] * independent_noise[u]
            wsum[v] += ew[k]
        wsum = np.where(wsum > 0, wsum, 1.0)
        smoothed = neigh / wsum
        district_offset = (
            0.05 * districts["urban"].values
            + 0.03 * beds_z.values
            - 0.04 * san_centered.values
            - 0.03 * lit_centered.values
            + 0.5 * independent_noise
            + 0.5 * smoothed
        )
    else:
        district_offset = (
            0.05 * districts["urban"].values
            + 0.03 * beds_z.values
            - 0.04 * san_centered.values
            - 0.03 * lit_centered.values
            + independent_noise
        )

    base = base_table["base_resistance"].values
    adjusted = base[:, None] + district_offset[None, :]
    return np.clip(adjusted, 0.02, 0.95)


def apply_age_sex_adjustment(adjusted, base_table, districts):
    adj = adjusted.copy()
    male_elderly = districts["male_elderly"].values
    reproductive_women = districts["reproductive_women"].values
    age_0_14 = districts["age_0_14"].values
    age_60plus = districts["age_60plus"].values

    for i, row in base_table.iterrows():
        coeffs = AGE_SEX_COEFFICIENTS.get((row["pathogen"], row["antibiotic"]))
        if not coeffs:
            continue
        for group, delta in coeffs:
            if group == "male_60plus":
                adj[i, :] += delta * (male_elderly / male_elderly.mean())
            elif group == "female_18_40":
                adj[i, :] += delta * (reproductive_women / reproductive_women.mean())
            elif group == "male_under18":
                adj[i, :] += delta * (age_0_14 / age_0_14.mean())
            elif group == "age_85plus":
                adj[i, :] += delta * (age_60plus / age_60plus.mean())
            elif group == "male_15_50":
                adj[i, :] += delta * 0.5

    return np.clip(adj, 0.02, 0.95)


def apply_temporal_trend(adjusted, base_table, n_quarters=12, start_year=2021, seed=42):
    rng = np.random.default_rng(seed)
    n_pairs, n_districts = adjusted.shape
    out = np.zeros((n_pairs, n_districts, n_quarters))

    growth = 1.0 + 0.015 * (np.arange(n_quarters) / 4.0)
    seasonal = np.array([1.00, 1.01, 1.02, 1.03])
    seasonal = np.tile(seasonal, 3)[:n_quarters]
    district_noise = rng.normal(0, 0.01, (n_districts, n_quarters)).cumsum(axis=1) * 0.02

    for t in range(n_quarters):
        out[:, :, t] = adjusted * growth[t] * seasonal[t] + district_noise[:, t][None, :]

    return np.clip(out, 0.02, 0.95)


def generate_consumption(n_districts, n_quarters, districts, seed=42):
    rng = np.random.default_rng(seed)
    access_share = np.clip(rng.normal(0.40, 0.05, n_districts), 0.25, 0.55)
    watch_share = np.clip(rng.normal(0.45, 0.05, n_districts), 0.30, 0.60)
    reserve_share = np.clip(1.0 - access_share - watch_share, 0.02, 0.15)

    base_per_capita = rng.lognormal(mean=0.0, sigma=0.3, size=n_districts)
    urban_factor = 1.0 + 0.5 * districts["urban"].values
    pop_factor = districts["population_log"].values / districts["population_log"].mean()

    consumption = np.zeros((n_districts, n_quarters, 4))
    for t in range(n_quarters):
        growth = 1.0 + 0.02 * t
        noise = rng.normal(0, 0.03, n_districts)
        pc = base_per_capita * urban_factor * pop_factor * growth + noise
        consumption[:, t, 0] = pc * access_share
        consumption[:, t, 1] = pc * watch_share
        consumption[:, t, 2] = pc * reserve_share
        consumption[:, t, 3] = pc

    return consumption


def apply_consumption_coupling(panel, consumption, beta=0.15, noise_sd=0.04, seed=42):
    rng = np.random.default_rng(seed)
    _, n_districts, n_q = panel.shape
    out = panel.copy()

    pc = consumption[:, :, 3]
    mu, sd = pc.mean(), pc.std()
    pc_z = (pc - mu) / (sd + 1e-8)

    for t in range(2, n_q):
        effect = beta * pc_z[:, t - 2] + rng.normal(0, noise_sd, n_districts)
        out[:, :, t] += effect[None, :]

    return np.clip(out, 0.02, 0.95)


def inject_noise(panel, n_districts, n_quarters, seed=42):
    rng = np.random.default_rng(seed)
    n_pairs, n_d, n_q = panel.shape
    out = panel.copy()

    out += rng.normal(0, 0.10, out.shape)

    mask = rng.random(out.shape) < 0.10
    for i in range(n_pairs):
        for d in range(n_d):
            for t in range(n_q - 1, 0, -1):
                if mask[i, d, t]:
                    out[i, d, t] = out[i, d, t - 1]

    n_breaks = 3
    for year in range(3):
        break_districts = rng.choice(n_d, n_breaks, replace=False)
        t_start = year * 4
        t_end = min(t_start + 4, n_q)
        out[:, break_districts, t_start:t_end] += 0.15

    return np.clip(out, 0.02, 0.98)


def inject_missing(df, missing_rate=0.08, seed=42):
    rng = np.random.default_rng(seed)
    mask = rng.random(len(df)) < missing_rate
    df.loc[mask, "resistance_rate"] = np.nan
    return df


def build_long_dataframe(panel, consumption, base_table, districts, cfg):
    n_pairs, n_d, n_q = panel.shape
    start_year = cfg["data"]["start_year"]

    records = []
    for i, row in base_table.iterrows():
        p = row["pathogen"]
        a = row["antibiotic"]
        for d in range(n_d):
            for t in range(n_q):
                year = start_year + t // 4
                quarter = (t % 4) + 1
                records.append({
                    "district_id": d,
                    "pathogen": p,
                    "antibiotic": a,
                    "year": year,
                    "quarter": quarter,
                    "time_index": t,
                    "resistance_rate": panel[i, d, t],
                    "consumption_access": consumption[d, t, 0],
                    "consumption_watch": consumption[d, t, 1],
                    "consumption_reserve": consumption[d, t, 2],
                    "consumption_per_capita": consumption[d, t, 3],
                })
    df = pd.DataFrame(records)
    df = df.merge(districts, on="district_id", how="left")
    df["high_risk"] = (df["resistance_rate"] >= 0.70).astype(int)
    return df


def save_dataset(df, path="data/synthetic/resistnet_st_dataset.csv"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved {len(df):,} records to {path}")


if __name__ == "__main__":
    cfg = load_config()
    n_q = cfg["data"]["n_quarters"]
    n_d = cfg["data"]["n_districts"]

    districts = generate_district_scaffold(n_d)
    base_table = generate_base_resistance()
    adjusted = apply_district_adjustment(base_table, districts)
    adjusted = apply_age_sex_adjustment(adjusted, base_table, districts)
    panel = apply_temporal_trend(adjusted, base_table, n_quarters=n_q)
    consumption = generate_consumption(n_d, n_q, districts)
    panel = apply_consumption_coupling(panel, consumption)
    panel = inject_noise(panel, n_d, n_q)

    df = build_long_dataframe(panel, consumption, base_table, districts, cfg)
    df = inject_missing(df)
    save_dataset(df)

    print("\nShape:", df.shape)
    print("High-risk proportion:", round(df["high_risk"].mean(), 3))
    print("Resistance mean:", round(df["resistance_rate"].mean(), 3))