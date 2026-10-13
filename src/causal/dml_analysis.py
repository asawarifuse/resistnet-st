"""
ResistNet-ST DML Causal Analysis (manual, no econml)
Partial-linear DML with cross-fitting.
Treatment: consumption lag-2 (z-scored)
Outcome: resistance_rate
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold


FEAT_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "features.parquet"
OUT_PATH = Path(__file__).resolve().parents[2] / "results" / "tables" / "dml_results.csv"


def cross_fit_residuals(y, T, W, n_splits=3, seed=42):
    """Cross-fitted residuals for DML (partialling out W)."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_res = np.zeros_like(y, dtype=float)
    t_res = np.zeros_like(T, dtype=float)

    for train_idx, test_idx in kf.split(W):
        my = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=seed)
        mt = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=seed)
        my.fit(W[train_idx], y[train_idx])
        mt.fit(W[train_idx], T[train_idx])
        y_res[test_idx] = y[test_idx] - my.predict(W[test_idx])
        t_res[test_idx] = T[test_idx] - mt.predict(W[test_idx])

    return y_res, t_res


def dml_ate(y_res, t_res):
    """ATE = coefficient from OLS of y_res ~ t_res."""
    num = (t_res * y_res).sum()
    den = (t_res ** 2).sum() + 1e-12
    return num / den


def bootstrap_ci(y_res, t_res, n_boot=500, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_res)
    ests = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        ests.append(dml_ate(y_res[idx], t_res[idx]))
    lo, hi = np.percentile(ests, [2.5, 97.5])
    return lo, hi


def main():
    df = pd.read_parquet(FEAT_PATH).dropna(subset=["resistance_rate", "cons_lag2"])

    T = df["cons_lag2"].values.astype(float)
    T = (T - T.mean()) / (T.std() + 1e-8)

    Y = df["resistance_rate"].values.astype(float)

    W = df[[
        "population_log", "density", "urban", "literacy",
        "age_60plus", "sanitation", "gdp",
        "quarter", "monsoon", "winter",
    ]].fillna(0).values.astype(float)

    print(f"n = {len(df):,}")
    print("Cross-fitting residuals...")
    y_res, t_res = cross_fit_residuals(Y, T, W)

    ate = dml_ate(y_res, t_res)
    lo, hi = bootstrap_ci(y_res, t_res)
    print(f"\nATE = {ate:.4f}  (95% CI: {lo:.4f}, {hi:.4f})")

    # Heterogeneous by urban tertile
    urban = df["urban"].values
    tertiles = np.quantile(urban, [0.33, 0.67])
    groups = np.digitize(urban, tertiles)

    rows = [{
        "group": "ATE (all)",
        "effect": ate,
        "ci_lo": lo,
        "ci_hi": hi,
        "n": len(df),
    }]
    for g, name in zip([0, 1, 2], ["low_urban", "mid_urban", "high_urban"]):
        mask = groups == g
        if mask.sum() < 50:
            continue
        eff = dml_ate(y_res[mask], t_res[mask])
        rows.append({
            "group": name,
            "effect": eff,
            "ci_lo": np.nan,
            "ci_hi": np.nan,
            "n": int(mask.sum()),
        })

    out = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print("\n" + out.to_string(index=False))
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()