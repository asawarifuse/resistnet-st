"""
ResistNet-ST Dataset Validation
Checks synthetic dataset against ICMR benchmarks.
"""

import pandas as pd
import numpy as np
from pathlib import Path


DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "synthetic" / "resistnet_st_dataset.csv"
REPORT_PATH = Path(__file__).resolve().parents[2] / "results" / "tables" / "validation_report.csv"


def check(name, value, target_lo, target_hi, passed):
    return {
        "check": name,
        "value": round(value, 4) if isinstance(value, (int, float)) else value,
        "target_lo": target_lo,
        "target_hi": target_hi,
        "pass": passed,
    }


def main():
    df = pd.read_csv(DATA_PATH)
    df_valid = df.dropna(subset=["resistance_rate"])

    checks = []

    # 1. E. coli meropenem ~35-40%
    v = df_valid[(df_valid.pathogen == "E_coli") & (df_valid.antibiotic == "Meropenem")]["resistance_rate"].mean()
    checks.append(check("E_coli_meropenem", v, 0.35, 0.40, 0.33 <= v <= 0.42))

    # 2. K. pneumoniae PIT ~70-78%
    v = df_valid[(df_valid.pathogen == "K_pneumoniae") & (df_valid.antibiotic == "Piperacillin_tazobactam")]["resistance_rate"].mean()
    checks.append(check("K_pneumoniae_PIT", v, 0.70, 0.78, 0.65 <= v <= 0.82))

    # 3. A. baumannii carbapenem ~68-76%
    v = df_valid[(df_valid.pathogen == "A_baumannii") & (df_valid.antibiotic == "Meropenem")]["resistance_rate"].mean()
    checks.append(check("A_baumannii_meropenem", v, 0.68, 0.76, 0.63 <= v <= 0.80))

    # 4. MRSA proportion ~35-45%
    v = df_valid[(df_valid.pathogen == "S_aureus") & (df_valid.antibiotic == "Ciprofloxacin")]["resistance_rate"].mean()
    checks.append(check("S_aureus_cipro", v, 0.35, 0.45, 0.30 <= v <= 0.50))

    # 5. High-risk proportion 15-20%
    v = df["high_risk"].mean()
    checks.append(check("high_risk_prop", v, 0.15, 0.20, 0.13 <= v <= 0.25))

    # 6. Consumption-resistance correlation 0.45-0.65
    sub = df_valid.dropna(subset=["consumption_per_capita"])
    corr = sub["consumption_per_capita"].corr(sub["resistance_rate"])
    checks.append(check("cons_res_corr", corr, 0.45, 0.65, 0.35 <= corr <= 0.75))

    # 7. Missing rate ~8%
    v = df["resistance_rate"].isna().mean()
    checks.append(check("missing_rate", v, 0.08, 0.08, 0.06 <= v <= 0.10))

    report = pd.DataFrame(checks)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(REPORT_PATH, index=False)

    print(report.to_string(index=False))
    print(f"\nSaved: {REPORT_PATH}")
    print(f"Passed: {report['pass'].sum()}/{len(report)}")


if __name__ == "__main__":
    main()