"""Tables 3, 4, 5 — balance test, decision quality, by case type.

Table 3 (balance test): regresses every case/bench characteristic on the
treatment, with all controls minus that dependent variable. Reported in
paper as 10 columns, all ~0 and insignificant.

Table 4 (decision quality):
  col 1 Case Delay       = -0.883** [0.385]
  col 2 Merit            =  0.213*** [0.0402]
  col 3 Correct          =  0.189*** [0.0522]
  col 4 Process Followed =  0.407*** [0.127]
  (all with bench+case controls, N=8446)

Table 5 (by case type, Panel A & B):
  Panel A col 1 Constitutional (FE only) = -0.270 [0.0304] N=6094
  Panel A col 2 Constitutional (controls) = -0.258 [0.0271] N=6094
  Panel A col 3 Criminal (FE only) =  0.261 [1.365] N=2368
  Panel A col 4 Criminal (controls) =  1.733 [1.938] N=2368
  Panel B col 1 Human-rights = -0.229 [0.0460] N=3428
  Panel B col 2 Land         = -0.307 [0.0471] N=2650
  Panel B col 3 Non-Islamic  =  0.060 [1.268] N=2146
  Panel B col 4 Islamic      =  0.0242 [0.188] N=222
"""

import numpy as np
import pandas as pd
import pyhdfe
import statsmodels.api as sm

from utils import (
    load_team1, main_sample_mask,
    CONTROLS_CASE, CONTROLS_JUDGE, CONTROLS_DISTRICT, CONTROLS_BENCH,
)


def run(df, y_name, treat, controls):
    y = df[y_name].to_numpy(dtype=float)
    X = df[[treat] + list(controls)].to_numpy(dtype=float)
    fe = df["yeardecisionXdistrict_encode"].to_numpy().reshape(-1, 1)
    cluster = df["districtXbench_enco"].to_numpy()
    algo = pyhdfe.create(fe, drop_singletons=False)
    yr = algo.residualize(y.reshape(-1, 1))[:, 0]
    Xr = algo.residualize(X)
    res = sm.OLS(yr, Xr).fit(cov_type="cluster", cov_kwds={"groups": cluster})
    return res.params[0], res.bse[0], len(y)


def main():
    df = load_team1()
    sub = df.loc[main_sample_mask(df)].copy()
    assert len(sub) == 8446

    print("\n===== Table 3: Balance test =====")
    print("Retirements_2010 x Post2010 on case/bench characteristics\n")
    t3_specs = [
        ("constitutional", "Constitutional case"),
        ("criminal", "Criminal case"),
        ("pagesjudgenum", "No. Pages"),
        ("benchchiefjustice", "CJ on case"),
        ("lawyer_number", "No. Lawyers"),
        ("judge_number", "No. Judges on case"),
        ("TotalJudges_bench", "No. Judges on bench"),
        ("Criminal_count_bench", "No. Criminal cases on bench"),
        ("Land_count_per_bench", "No. Land cases on bench"),
        ("Human_count_per_bench", "No. HR cases on bench"),
    ]
    t3_pub = [
        -0.00336, -0.00834, -0.130, -0.00925, -0.132, 0.0238,
        -0.159, 1.031, 0.488, 0.719,
    ]
    for (dep, label), pub in zip(t3_specs, t3_pub):
        ctl = [c for c in CONTROLS_CASE + CONTROLS_DISTRICT + CONTROLS_JUDGE + CONTROLS_BENCH if c != dep]
        c, se, n = run(sub, dep, "Retirements_2010_Post2010", ctl)
        print(f"  {label:<30} coef {c:+.4f}  se {se:.4f}  pub {pub:+.4f}")

    print("\n===== Table 4: Decision quality =====")
    t4 = [
        ("caselag", "Case Delay", -0.883, 0.385),
        ("Merit", "Merit", 0.213, 0.0402),
        ("correct", "Correct Decisions", 0.189, 0.0522),
        ("Process_Followed", "Process Followed", 0.407, 0.127),
    ]
    for dep, label, pub_c, pub_se in t4:
        if dep not in sub.columns:
            candidates = [c for c in sub.columns if dep.lower().replace("_", "") in c.lower().replace("_", "")]
            print(f"  {label}: column '{dep}' not found, candidates: {candidates[:5]}")
            continue
        c, se, n = run(sub, dep, "Retirements_2010_Post2010",
                       CONTROLS_BENCH + CONTROLS_CASE)
        print(f"  {label:<20} coef {c:+.4f}  se {se:.4f}  pub {pub_c:+.4f} [{pub_se:.4f}]  N={n}")

    print("\n===== Table 5 Panel A: Constitutional vs Criminal =====")
    # Constitutional cases
    con = sub[sub["constitutional"] == 1]
    # FE only
    c1, se1, n1 = run(con, "StateWins", "Retirements_2010_Post2010", [])
    print(f"  (1) Constitutional (FE only)  coef {c1:+.4f}  se {se1:.4f}  pub -0.270 [0.0304]  N={n1} (pub 6094)")
    # controls
    c2, se2, n2 = run(con, "StateWins", "Retirements_2010_Post2010",
                      CONTROLS_BENCH + CONTROLS_CASE)
    print(f"  (2) Constitutional (ctrls)    coef {c2:+.4f}  se {se2:.4f}  pub -0.258 [0.0271]  N={n2}")

    cri = sub[sub["criminal"] == 1]
    c3, se3, n3 = run(cri, "StateWins", "Retirements_2010_Post2010", [])
    print(f"  (3) Criminal (FE only)        coef {c3:+.4f}  se {se3:.4f}  pub +0.261 [1.365]  N={n3} (pub 2368)")
    c4, se4, n4 = run(cri, "StateWins", "Retirements_2010_Post2010",
                      CONTROLS_BENCH + CONTROLS_CASE)
    print(f"  (4) Criminal (ctrls)          coef {c4:+.4f}  se {se4:.4f}  pub +1.733 [1.938]  N={n4}")

    print("\n===== Table 5 Panel B: disaggregated =====")
    # Human rights: HR==1 dummy
    hr = sub[sub["HR"] == 1]
    c, se, n = run(hr, "StateWins", "Retirements_2010_Post2010",
                   CONTROLS_BENCH + CONTROLS_CASE)
    print(f"  Human-Rights     coef {c:+.4f}  se {se:.4f}  pub -0.229 [0.0460]  N={n} (pub 3428)")
    land = sub[sub["land_case"] == 1]
    c, se, n = run(land, "StateWins", "Retirements_2010_Post2010",
                   CONTROLS_BENCH + CONTROLS_CASE)
    print(f"  Land             coef {c:+.4f}  se {se:.4f}  pub -0.307 [0.0471]  N={n} (pub 2650)")


if __name__ == "__main__":
    main()
