"""Reproduce the key tables from Gonzales, Leon-Ciliotta & Martinez (AEJ:Applied 2022).
Focus: Tables 1 (panels A & B), 2, 3, 4, 5.

For Table 2 col 5 (runoff interaction), the paper uses the long-run sample but
includes fine_a × 1(Runoff); we follow the paper's specification exactly.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from utils import feols_hdfe, make_group_ids, iterative_singleton_drop, OUT


def row(res, label, extra=""):
    parts = []
    for i, x in enumerate(res["xvars"]):
        parts.append(f"{x}={res['beta'][i]:+.4f}[{res['se'][i]:.4f}]")
    coefs = "  ".join(parts)
    return f"  {label}: {coefs}  N={res['N']}  D={res['districts']}  ybar={res['ymean']:.3f}{extra}"


def table1(df):
    print("\n============ Table 1: Marginal Effect of Fine on Turnout ============")
    print("-- Panel A: turnout on fine_a (S/100) --")
    # Col 1: baseline
    r1 = feols_hdfe(df, "turnout", ["fine_a"], ["ubigeo", "dpc_id"], "province_id", "electores_01")
    print(row(r1, "(1) baseline            "))
    # Col 2: election FE only (use full_national since singleton drop pattern differs)
    full = pd.read_parquet(OUT / "full_national.parquet")
    r2 = feols_hdfe(full, "turnout", ["fine_a"], ["ubigeo", "date_id"], "province_id", "electores_01")
    print(row(r2, "(2) Election FE         "))
    # Col 3: election × province FE
    r3 = feols_hdfe(full, "turnout", ["fine_a"], ["ubigeo", "dp_id"], "province_id", "electores_01")
    print(row(r3, "(3) Elec x Prov FE      "))
    # Col 4: unweighted, baseline FE
    r4 = feols_hdfe(df, "turnout", ["fine_a"], ["ubigeo", "dpc_id"], "province_id", weight=None)
    print(row(r4, "(4) no weights          "))
    # Col 5: poverty-share interactions (c.non_extreme#i.date c.extreme#i.date)
    # Build election-specific shares
    df5 = df.copy()
    date_dummies = pd.get_dummies(df5["date_id"], prefix="dt")
    ne_cols = []
    ex_cols = []
    for c in date_dummies.columns:
        df5[f"ne_{c}"] = df5["non_extreme"] * date_dummies[c].astype(float)
        df5[f"ex_{c}"] = df5["extreme"] * date_dummies[c].astype(float)
        ne_cols.append(f"ne_{c}")
        ex_cols.append(f"ex_{c}")
    # drop the omitted category (pick one date as reference by dropping its interactions)
    drop_ref = [ne_cols[0], ex_cols[0]]
    covars = ["fine_a"] + [c for c in ne_cols + ex_cols if c not in drop_ref]
    r5 = feols_hdfe(df5, "turnout", covars, ["ubigeo", "dpc_id"], "province_id", "electores_01")
    print(row(r5, "(5) poverty shares      "))
    # Col 6: education shares
    df6 = df.dropna(subset=["share_primaria", "share_sec", "share_univ"]).copy()
    # Singleton drop again on restricted sample
    df6 = iterative_singleton_drop(df6, ["ubigeo", "dpc_id"])
    for col in ["ubigeo", "dpc_id"]:
        df6[col] = pd.Categorical(df6[col]).codes.astype(int)
    r6 = feols_hdfe(df6, "turnout", ["fine_a", "share_primaria", "share_sec", "share_univ"],
                    ["ubigeo", "dpc_id"], "province_id", "electores_01")
    print(row(r6, "(6) education shares    "))
    # Col 7: polling stations
    r7 = feols_hdfe(df, "turnout", ["fine_a", "ln_local"], ["ubigeo", "dpc_id"], "province_id", "electores_01")
    print(row(r7, "(7) polling stations    "))

    print("\n-- Panel B: ln turnout on ln fine --")
    rb1 = feols_hdfe(df, "ln_turnout", ["ln_fine"], ["ubigeo", "dpc_id"], "province_id", "electores_01")
    print(row(rb1, "(1) baseline            "))
    rb2 = feols_hdfe(full, "ln_turnout", ["ln_fine"], ["ubigeo", "date_id"], "province_id", "electores_01")
    print(row(rb2, "(2) Election FE         "))
    rb3 = feols_hdfe(full, "ln_turnout", ["ln_fine"], ["ubigeo", "dp_id"], "province_id", "electores_01")
    print(row(rb3, "(3) Elec x Prov FE      "))
    rb4 = feols_hdfe(df, "ln_turnout", ["ln_fine"], ["ubigeo", "dpc_id"], "province_id", weight=None)
    print(row(rb4, "(4) no weights          "))
    rb5 = feols_hdfe(df5, "ln_turnout", ["ln_fine"] + covars[1:], ["ubigeo", "dpc_id"], "province_id", "electores_01")
    print(row(rb5, "(5) poverty shares      "))
    rb6 = feols_hdfe(df6, "ln_turnout", ["ln_fine", "share_primaria", "share_sec", "share_univ"],
                     ["ubigeo", "dpc_id"], "province_id", "electores_01")
    print(row(rb6, "(6) education shares    "))
    rb7 = feols_hdfe(df, "ln_turnout", ["ln_fine", "ln_local"], ["ubigeo", "dpc_id"], "province_id", "electores_01")
    print(row(rb7, "(7) polling stations    "))

    print("\nPaper Panel A (1): 0.049 [0.008]; (2): 0.073 [0.012]; (3): 0.046 [0.009];")
    print("                   (4): 0.062 [0.010]; (5): 0.035 [0.011]; (6): 0.061 [0.011]; (7): 0.046 [0.009]")
    print("Paper Panel B (1): 0.030 [0.005]; (2): 0.040 [0.006]; (3): 0.028 [0.006];")
    print("                   (4): 0.037 [0.006]; (5): 0.023 [0.007]; (6): 0.037 [0.007]; (7): 0.029 [0.006]")
    return {"r1": r1, "rb1": rb1}


def table2(df):
    print("\n============ Table 2: Heterogeneous Effects ============")
    # Col 1: fine_a + fine_2016 (long-run interaction)
    r1 = feols_hdfe(df, "turnout", ["fine_a", "fine_2016"], ["ubigeo", "dpc_id"], "province_id", "electores_01")
    print(row(r1, "(1) long-run fine_a,fine_2016"))
    # Col 2: fine_a + fine_r2 (runoff)
    r2 = feols_hdfe(df, "turnout", ["fine_a", "fine_r2"], ["ubigeo", "dpc_id"], "province_id", "electores_01")
    print(row(r2, "(2) runoff       fine_a,fine_r2"))
    # Col 3: fine_a + fine_any_poor
    r3 = feols_hdfe(df, "turnout", ["fine_a", "fine_any_poor"], ["ubigeo", "dpc_id"], "province_id", "electores_01")
    print(row(r3, "(3) poverty      fine_a,fine_any_poor"))
    # Col 4-6: same in ln (uses ln_turnout, ln_fine, lfine_2016, lfine_r2, lfine_any_poor)
    r4 = feols_hdfe(df, "ln_turnout", ["ln_fine", "lfine_2016"], ["ubigeo", "dpc_id"], "province_id", "electores_01")
    print(row(r4, "(4) ln long-run              "))
    r5 = feols_hdfe(df, "ln_turnout", ["ln_fine", "lfine_r2"], ["ubigeo", "dpc_id"], "province_id", "electores_01")
    print(row(r5, "(5) ln runoff                "))
    r6 = feols_hdfe(df, "ln_turnout", ["ln_fine", "lfine_any_poor"], ["ubigeo", "dpc_id"], "province_id", "electores_01")
    print(row(r6, "(6) ln poverty               "))

    print("\nPaper: (1) a=0.020 [.008], b=0.051 [.005];  (2) a=0.039 [.009], b=0.019 [.004]")
    print("       (3) a=-0.022 [.022], b=0.072 [.021]")
    print("       (4) a=0.011 [.005], b=0.038 [.003];  (5) a=0.023 [.006], b=0.014 [.003]")
    print("       (6) a=-0.021 [.022], b=0.058 [.025]")


def table3(df):
    print("\n============ Table 3: Registration by Age (ln_electores on ln_fine, generals only) ============")
    # Sample: runoff==0 & year!=2006 → general elections in 2001, 2011, 2016 (3 elections)
    d = df[(df["runoff"] == 0) & (df["year"] != 2006)].copy()
    d = iterative_singleton_drop(d, ["ubigeo", "dpc_id"])
    for col in ["ubigeo", "dpc_id"]:
        d[col] = pd.Categorical(d[col]).codes.astype(int)
    print(f"  sample N={len(d)}, districts={d['ubigeo'].nunique()}  (paper N=5,076, D=1,692)")
    ages = ["ln_electores", "ln_entre_18_20", "ln_entre_21_29", "ln_entre_30_35",
            "ln_entre_36_50", "ln_entre_51_75", "ln_mayores_75"]
    paper_vals = [(-0.045, 0.019), (-0.276, 0.043), (-0.055, 0.020), (-0.031, 0.022),
                  (-0.021, 0.020), (-0.017, 0.024), (-0.057, 0.051)]
    for y, pv in zip(ages, paper_vals):
        r = feols_hdfe(d, y, ["ln_fine"], ["ubigeo", "dpc_id"], "province_id", "electores_01")
        print(row(r, f"{y:<18}") + f"  paper: {pv[0]:+.3f} [{pv[1]:.3f}]")


def table4(df):
    print("\n============ Table 4: Registration & Votes ============")
    # Cols 1-2: ln_electores on ln_fine (runoff==0 only, all years)
    d = df[df["runoff"] == 0].copy()
    d = iterative_singleton_drop(d, ["ubigeo", "dpc_id"])
    for col in ["ubigeo", "dpc_id"]:
        d[col] = pd.Categorical(d[col]).codes.astype(int)
    r1 = feols_hdfe(d, "ln_electores", ["ln_fine"], ["ubigeo", "dpc_id"], "province_id", "electores_01")
    print(row(r1, "(1) ln_electores on ln_fine (runoff==0)"))
    r2 = feols_hdfe(d, "ln_electores", ["ln_fine", "lfine_2016"], ["ubigeo", "dpc_id"], "province_id", "electores_01")
    print(row(r2, "(2) + lfine_2016                       "))
    # Cols 3-4: ln_votos_emitidos on ln_fine (full sample)
    r3 = feols_hdfe(df, "ln_votos_emitidos", ["ln_fine"], ["ubigeo", "dpc_id"], "province_id", "electores_01")
    print(row(r3, "(3) ln_votes on ln_fine               "))
    r4 = feols_hdfe(df, "ln_votos_emitidos", ["ln_fine", "lfine_2016"], ["ubigeo", "dpc_id"], "province_id", "electores_01")
    print(row(r4, "(4) + lfine_2016                       "))
    print("Paper: (1) -0.046 [.015]; (2) a=-0.035 [.012], b=-0.022 [.009]")
    print("       (3) -0.016 [.016]; (4) a=-0.024 [.014], b=+0.015 [.009]")


def table5(df):
    print("\n============ Table 5: Fine effect on Invalid/Blank votes ============")
    d = df[df["runoff"] == 0].copy()
    d = iterative_singleton_drop(d, ["ubigeo", "dpc_id"])
    for col in ["ubigeo", "dpc_id"]:
        d[col] = pd.Categorical(d[col]).codes.astype(int)
    r1 = feols_hdfe(d, "turnout", ["fine_a"], ["ubigeo", "dpc_id"], "province_id", "electores_01")
    print(row(r1, "(1) turnout on fine_a        "))
    r2 = feols_hdfe(d, "turnout", ["fine_a", "fine_2016"], ["ubigeo", "dpc_id"], "province_id", "electores_01")
    print(row(r2, "(2) + fine_2016              "))
    r3 = feols_hdfe(d, "spoiled_elec", ["fine_a"], ["ubigeo", "dpc_id"], "province_id", "electores_01")
    print(row(r3, "(3) spoiled on fine_a        "))
    r4 = feols_hdfe(d, "spoiled_elec", ["fine_a", "fine_2016"], ["ubigeo", "dpc_id"], "province_id", "electores_01")
    print(row(r4, "(4) + fine_2016              "))
    print("Paper Table 5 (first-round only):")
    print("  (1) turnout: 0.043 [.008];  (2) fine_a=0.017 [.009], fine_2016=0.045 [.005]")
    print("  (3) spoiled: 0.037 [.007];  (4) fine_a=0.022 [.008], fine_2016=0.026 [.005]")


def main():
    df = pd.read_parquet(OUT / "main_sample.parquet")
    table1(df)
    table2(df)
    table3(df)
    table4(df)
    table5(df)


if __name__ == "__main__":
    main()
