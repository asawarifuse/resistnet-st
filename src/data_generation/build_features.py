"""
ResistNet-ST Feature Engineering
Build 55+ features from synthetic dataset.
"""

import numpy as np
import pandas as pd
from pathlib import Path


DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "synthetic" / "resistnet_st_dataset.csv"
OUT_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "features.parquet"


def add_resistance_features(df):
    # Convert arrow-backed strings to plain object (avoids pandas 3.0 groupby bug)
    df["pathogen"] = df["pathogen"].astype(object)
    df["antibiotic"] = df["antibiotic"].astype(object)
    df["district_name"] = df["district_name"].astype(object)

    df = df.sort_values(["district_id", "pathogen", "antibiotic", "time_index"]).copy()
    g = df.groupby(["district_id", "pathogen", "antibiotic"], observed=True)["resistance_rate"]

    df["res_current"] = g.shift(0)
    df["res_lag1"] = g.shift(1)
    df["res_lag2"] = g.shift(2)
    df["res_lag4"] = g.shift(4)
    df["res_roll3"] = g.transform(lambda s: s.shift(1).rolling(3).mean())
    df["res_roll4"] = g.transform(lambda s: s.shift(1).rolling(4).mean())
    df["res_velocity"] = df["res_lag1"] - df["res_lag2"]
    df["res_acceleration"] = df["res_lag1"] - 2 * df["res_lag2"] + g.shift(3)
    df["res_seasonal"] = df["time_index"] % 4
    df["res_trend"] = df["time_index"]
    df["res_volatility"] = g.transform(lambda s: s.shift(1).rolling(4).std())
    df["res_percentile"] = g.transform(lambda s: s.expanding().rank(pct=True))

    # Consumption features
    dg = df.groupby("district_id", observed=True)["consumption_per_capita"]
    df["cons_current"] = df["consumption_per_capita"]
    df["cons_lag1"] = dg.shift(1)
    df["cons_lag2"] = dg.shift(2)
    df["cons_roll4"] = dg.transform(lambda s: s.shift(1).rolling(4).mean())
    df["cons_velocity"] = df["cons_lag1"] - df["cons_lag2"]
    df["access_share"] = df["consumption_access"] / (df["consumption_per_capita"] + 1e-8)
    df["watch_share"] = df["consumption_watch"] / (df["consumption_per_capita"] + 1e-8)
    df["reserve_share"] = df["consumption_reserve"] / (df["consumption_per_capita"] + 1e-8)
    df["cons_anomaly"] = dg.transform(lambda s: (s - s.mean()) / (s.std() + 1e-8))

    # Temporal
    df["monsoon"] = (df["quarter"] == 3).astype(int)
    df["winter"] = (df["quarter"] == 4).astype(int)

    # Pathogen-antibiotic encoding
    df["pathogen_id"] = df["pathogen"].astype("category").cat.codes
    df["antibiotic_id"] = df["antibiotic"].astype("category").cat.codes
    return df


def main():
    df = pd.read_csv(DATA_PATH)
    df = add_resistance_features(df)
    df = df.dropna(subset=["res_lag4", "res_roll4"]).reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)

    feature_cols = [c for c in df.columns if c not in
                    ["district_name", "pathogen", "antibiotic", "resistance_rate", "high_risk"]]
    print(f"Rows: {len(df):,}")
    print(f"Feature columns: {len(feature_cols)}")
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()