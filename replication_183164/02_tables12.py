"""Replicate Tables 1 and 2 of Antman, Duncan, Trejo (JEP 2023).

Table 1: OLS of completed years of schooling on race_cat4 / race_cat16 + age FEs.
Table 2: OLS of log annual earnings on race_cat4 / race_cat16 + age + state FEs,
         separately for men and women.

All regressions are weighted (aweight=perwt, which statsmodels WLS implements
with weights∝perwt after normalization), with HC1 robust SEs (Stata's
`vce(robust)` on aweight-weighted OLS ≈ WLS with HC1 in statsmodels).
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path

from utils import OUT

BASE_RACE4 = 3          # US-born non-Hispanic whites
BASE_RACE16 = 15        # US-born non-Hispanic whites
BASE_AGE = 35
BASE_STATE = 6          # California


def build_design(df, race_col, include_state_fe):
    X_parts = [pd.Series(1.0, index=df.index, name="const")]
    # race dummies (drop base)
    race_dum = pd.get_dummies(df[race_col], prefix=race_col, drop_first=False)
    base_map = {"race_cat4": f"race_cat4_{BASE_RACE4}", "race_cat16": f"race_cat16_{BASE_RACE16}"}
    base_col = base_map[race_col]
    if base_col in race_dum.columns:
        race_dum = race_dum.drop(columns=[base_col])
    X_parts.append(race_dum.astype(float))
    # age dummies (base = 35)
    age_dum = pd.get_dummies(df["age"], prefix="age", drop_first=False)
    if f"age_{BASE_AGE}" in age_dum.columns:
        age_dum = age_dum.drop(columns=[f"age_{BASE_AGE}"])
    X_parts.append(age_dum.astype(float))
    if include_state_fe:
        st_dum = pd.get_dummies(df["statefip"], prefix="state", drop_first=False)
        if f"state_{BASE_STATE}" in st_dum.columns:
            st_dum = st_dum.drop(columns=[f"state_{BASE_STATE}"])
        X_parts.append(st_dum.astype(float))
    X = pd.concat(X_parts, axis=1)
    return X


def run_weighted_ols(df, y_col, race_col, include_state_fe):
    """Weighted OLS matching Stata's ``regress y ... [aweight=perwt], vce(robust)``.

    Stata's aweight rescales weights so the mean weight is 1 and then computes
    HC1 (robust) SEs.  We mirror that by normalizing perwt and using WLS with
    cov_type='HC1'.
    """
    d = df.dropna(subset=[y_col]).copy()
    w = d["perwt"].astype(float).to_numpy()
    w = w * len(w) / w.sum()
    y = d[y_col].astype(float).to_numpy()
    X = build_design(d, race_col, include_state_fe)
    model = sm.WLS(y, X.to_numpy(), weights=w)
    res = model.fit(cov_type="HC1")
    res.exog_names = list(X.columns)
    return res, d, w


def coef_row(res, label):
    idx = res.exog_names.index(label)
    return res.params[idx], res.bse[idx], res.nobs


def format_row(name, beta, se):
    return f"{name:<35s} {beta: .4f}  ({se:.4f})"


def main():
    df = pd.read_parquet(OUT / "census_2019.parquet")
    for c in ["race_cat4", "race_cat16", "age", "statefip", "sex"]:
        df[c] = df[c].astype(int)
    print(f"[02] N={len(df):,}")
    print(f"[02] mean edyrs = {np.average(df['edyrs'], weights=df['perwt']):.4f}")

    # ---- Table 1, col 1: race_cat4 ---- #
    r1c1, _, _ = run_weighted_ols(df, "edyrs", "race_cat4", include_state_fe=False)
    r1c2, _, _ = run_weighted_ols(df, "edyrs", "race_cat16", include_state_fe=False)

    print("\n=== Table 1: Years of schooling (age FE only) ===")
    print("Paper col 1 values shown alongside replicated values.")
    PAPER_T1 = {
        "race_cat4_1": (-3.162, 0.016),   # Foreign-born Hispanics
        "race_cat4_2": (-1.113, 0.011),   # US-born Hispanics
        "race_cat4_4": (-0.850, 0.010),   # US-born Blacks
    }
    print(f"  N: paper=1,211,621  replication={int(r1c1.nobs):,}")
    print(f"  R2: paper=0.1195    replication={r1c1.rsquared:.4f}")
    for k, (pb, pse) in PAPER_T1.items():
        b, se, _ = coef_row(r1c1, k)
        print(f"  {k:<16s} paper {pb:+.3f} ({pse:.3f})   repl {b:+.4f} ({se:.4f})")

    PAPER_T1_C2 = {
        "race_cat16_1":  (-4.016, 0.021, "FB Mexican"),
        "race_cat16_2":  (-1.332, 0.044, "FB Puerto Rican"),
        "race_cat16_3":  (-0.979, 0.047, "FB Cuban"),
        "race_cat16_4":  (-4.225, 0.045, "FB Central American"),
        "race_cat16_5":  (-0.287, 0.035, "FB South American"),
        "race_cat16_6":  (-1.911, 0.058, "FB Dominican"),
        "race_cat16_7":  (-1.889, 0.107, "FB Other Hispanic"),
        "race_cat16_8":  (-1.284, 0.013, "USB Mexican"),
        "race_cat16_9":  (-1.089, 0.031, "USB Puerto Rican"),
        "race_cat16_10": ( 0.064, 0.062, "USB Cuban"),
        "race_cat16_11": (-0.893, 0.060, "USB Central American"),
        "race_cat16_12": ( 0.254, 0.060, "USB South American"),
        "race_cat16_13": (-0.614, 0.080, "USB Dominican"),
        "race_cat16_14": (-1.004, 0.039, "USB Other Hispanic"),
        "race_cat16_16": (-0.851, 0.010, "USB Black"),
    }
    print("\nTable 1 col 2 (race_cat16):")
    for k, (pb, pse, nm) in PAPER_T1_C2.items():
        b, se, _ = coef_row(r1c2, k)
        print(f"  {nm:<24s} paper {pb:+.3f} ({pse:.3f})   repl {b:+.4f} ({se:.4f})")

    # ---- Table 2: log earnings ---- #
    print("\n=== Table 2: log(annual earnings), by sex ===")
    T2_PAPER = {
        ("men", "race_cat4_1"): (-0.495, 0.005),
        ("men", "race_cat4_2"): (-0.296, 0.006),
        ("men", "race_cat4_4"): (-0.471, 0.007),
        ("women", "race_cat4_1"): (-0.520, 0.007),
        ("women", "race_cat4_2"): (-0.189, 0.007),
        ("women", "race_cat4_4"): (-0.171, 0.006),
    }
    for sex_code, label in [(1, "men"), (2, "women")]:
        sub = df[df["sex"] == sex_code]
        r4, _, _ = run_weighted_ols(sub, "ln_annualearnings", "race_cat4", include_state_fe=True)
        r16, _, _ = run_weighted_ols(sub, "ln_annualearnings", "race_cat16", include_state_fe=True)
        print(f"\n-- {label.upper()} N={int(r4.nobs):,}  R2={r4.rsquared:.4f} --")
        for k in ["race_cat4_1", "race_cat4_2", "race_cat4_4"]:
            pb, pse = T2_PAPER[(label, k)]
            b, se, _ = coef_row(r4, k)
            print(f"  {k:<15s} paper {pb:+.3f} ({pse:.3f})   repl {b:+.4f} ({se:.4f})")

        # also print national-origin breakdown for men, women
        T2_C24_PAPER_MEN = {
            "race_cat16_1":  (-0.539, 0.007, "FB Mexican"),
            "race_cat16_2":  (-0.394, 0.025, "FB Puerto Rican"),
            "race_cat16_3":  (-0.357, 0.021, "FB Cuban"),
            "race_cat16_4":  (-0.566, 0.011, "FB Central American"),
            "race_cat16_5":  (-0.278, 0.016, "FB South American"),
            "race_cat16_6":  (-0.504, 0.026, "FB Dominican"),
            "race_cat16_7":  (-0.453, 0.042, "FB Other Hispanic"),
            "race_cat16_8":  (-0.325, 0.008, "USB Mexican"),
            "race_cat16_9":  (-0.319, 0.017, "USB Puerto Rican"),
            "race_cat16_10": (-0.053, 0.042, "USB Cuban"),
            "race_cat16_11": (-0.249, 0.025, "USB Central American"),
            "race_cat16_12": (-0.062, 0.029, "USB South American"),
            "race_cat16_13": (-0.356, 0.044, "USB Dominican"),
            "race_cat16_14": (-0.287, 0.023, "USB Other Hispanic"),
            "race_cat16_16": (-0.470, 0.007, "USB Black"),
        }
        T2_C24_PAPER_WOMEN = {
            "race_cat16_1":  (-0.594, 0.010, "FB Mexican"),
            "race_cat16_2":  (-0.359, 0.026, "FB Puerto Rican"),
            "race_cat16_3":  (-0.303, 0.024, "FB Cuban"),
            "race_cat16_4":  (-0.610, 0.017, "FB Central American"),
            "race_cat16_5":  (-0.327, 0.017, "FB South American"),
            "race_cat16_6":  (-0.555, 0.027, "FB Dominican"),
            "race_cat16_7":  (-0.375, 0.048, "FB Other Hispanic"),
            "race_cat16_8":  (-0.229, 0.009, "USB Mexican"),
            "race_cat16_9":  (-0.204, 0.018, "USB Puerto Rican"),
            "race_cat16_10": ( 0.119, 0.029, "USB Cuban"),
            "race_cat16_11": (-0.126, 0.027, "USB Central American"),
            "race_cat16_12": ( 0.047, 0.029, "USB South American"),
            "race_cat16_13": (-0.208, 0.043, "USB Dominican"),
            "race_cat16_14": (-0.176, 0.026, "USB Other Hispanic"),
            "race_cat16_16": (-0.170, 0.006, "USB Black"),
        }
        paper_tab = T2_C24_PAPER_MEN if label == "men" else T2_C24_PAPER_WOMEN
        print(f"\n  -- national origin ({label}) --")
        for k, (pb, pse, nm) in paper_tab.items():
            b, se, _ = coef_row(r16, k)
            print(f"  {nm:<24s} paper {pb:+.3f} ({pse:.3f})   repl {b:+.4f} ({se:.4f})")

    print("\n[02] done.")


if __name__ == "__main__":
    main()
