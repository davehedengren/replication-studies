"""Phase 4 — robustness checks for the baseline result
  StateWins = alpha * Retirements_2010_Post2010 + district-year FE + ctrls.

Baseline (Table 2, col 2): coef = -0.210, SE 0.0383.
We re-run the baseline under a series of sensitivity variations and report
them in a single table.
"""

import numpy as np
import pandas as pd
import pyhdfe
import statsmodels.api as sm

from utils import (
    load_team1, main_sample_mask,
    CONTROLS_CASE, CONTROLS_JUDGE, CONTROLS_DISTRICT, CONTROLS_BENCH,
)

RNG = np.random.default_rng(42)


def run(df, y_name="StateWins", treat="Retirements_2010_Post2010",
        controls=None, cluster_col="districtXbench_enco"):
    y = df[y_name].to_numpy(dtype=float)
    cols = [treat] + (controls or [])
    X = df[cols].to_numpy(dtype=float)
    fe = df["yeardecisionXdistrict_encode"].to_numpy().reshape(-1, 1)
    cluster = df[cluster_col].to_numpy()
    algo = pyhdfe.create(fe, drop_singletons=False)
    yr = algo.residualize(y.reshape(-1, 1))[:, 0]
    Xr = algo.residualize(X)
    res = sm.OLS(yr, Xr).fit(cov_type="cluster", cov_kwds={"groups": cluster})
    return res.params[0], res.bse[0], len(y)


def main():
    df = load_team1()
    sub = df.loc[main_sample_mask(df)].copy()
    assert len(sub) == 8446

    controls = CONTROLS_BENCH + CONTROLS_CASE

    rows = []

    def add(label, coef, se, n):
        rows.append((label, coef, se, n, "***" if abs(coef / se) > 2.58 else
                     ("**" if abs(coef / se) > 1.96 else
                      ("*" if abs(coef / se) > 1.645 else ""))))

    # 0 — Baseline
    c, se, n = run(sub, controls=controls)
    add("00 Baseline (Table 2 col 2)", c, se, n)

    # 1 — Using Appointments instead of Retirements
    c, se, n = run(sub, treat="Appointments_2010_Post2010", controls=controls)
    add("01 Alt treatment: Appointments_2010", c, se, n)

    # 2 — District-level clustering
    c, se, n = run(sub, controls=controls, cluster_col="distric_encode")
    add("02 Cluster at district level", c, se, n)

    # 3 — Judge-level clustering
    c, se, n = run(sub, controls=controls, cluster_col="judgename")
    add("03 Cluster at judge level", c, se, n)

    # 4 — Post-reform only sub-sample
    post = sub[sub.yeardecision >= 2010].copy()
    c, se, n = run(post, controls=controls)
    add("04 Post-2010 only", c, se, n)

    # 5 — Drop principal-bench districts (capital districts: islhc, lrhc, etc.)
    # Use district encode to drop Islamabad (capital) if present
    # keep only districts that appear pre- and post-reform
    sub5 = sub.copy()
    if "isprincipal" in sub.columns:
        sub5 = sub5[sub5.isprincipal == 0]
    else:
        # drop by name-based filter on distric_encode string form
        caps = sub5["distric_encode"].mode().iloc[0]  # fallback
        sub5 = sub5[sub5.distric_encode != caps]
    c, se, n = run(sub5, controls=controls)
    add("05 Drop most-frequent district", c, se, n)

    # 6 — Drop property bench
    if "Specialization_bench" in sub.columns:
        sub6 = sub[sub.Specialization_bench != sub.Specialization_bench.mode().iloc[0]]
        c, se, n = run(sub6, controls=controls)
        add("06 Drop most-common bench type", c, se, n)

    # 7 — Winsorize page count (top 1%) and rerun
    sub7 = sub.copy()
    cap = sub7.pagesjudgenum.quantile(0.99)
    sub7["pagesjudgenum"] = sub7.pagesjudgenum.clip(upper=cap)
    c, se, n = run(sub7, controls=controls)
    add("07 Winsorize pagesjudgenum at p99", c, se, n)

    # 8 — Leave-one-district-out min/max
    dists = sub.distric_encode.unique()
    loo = []
    for d in dists:
        tmp = sub[sub.distric_encode != d]
        try:
            c, se, n = run(tmp, controls=controls)
            loo.append((d, c, se, n))
        except Exception:
            pass
    loo_coefs = np.array([r[1] for r in loo])
    add(f"08 LOO district min ({loo_coefs.min():.4f})", loo_coefs.min(), float("nan"), -1)
    add(f"08 LOO district max ({loo_coefs.max():.4f})", loo_coefs.max(), float("nan"), -1)

    # 9 — Alternative outcome (Merit) as placebo-plus
    c, se, n = run(sub, y_name="Merit", controls=controls)
    add("09 Outcome = Merit (expected +)", c, se, n)

    # 10 — Placebo: shuffle treatment, repeat 500 times, report 5/95 pct of coefs
    placebo_coefs = []
    treat_col = sub["Retirements_2010_Post2010"].to_numpy()
    for _ in range(500):
        shuffled = RNG.permutation(treat_col)
        tmp = sub.copy()
        tmp["Retirements_2010_Post2010"] = shuffled
        c, _, _ = run(tmp, controls=controls)
        placebo_coefs.append(c)
    placebo_coefs = np.array(placebo_coefs)
    p = (np.abs(placebo_coefs) >= abs(rows[0][1])).mean()
    add(f"10 Placebo (shuffle) p95 pct = {np.percentile(placebo_coefs, 95):.4f}",
        p, float("nan"), 500)
    print(f"    Placebo two-sided p-value: {p:.4f}")

    # 11 — Sample from 2000-2019 only (drops early sparse years)
    sub11 = sub[sub.yeardecision >= 2000]
    c, se, n = run(sub11, controls=controls)
    add("11 Restrict to 2000-2019", c, se, n)

    # 12 — Drop 2010 reform year itself (donut)
    sub12 = sub[sub.yeardecision != 2010]
    c, se, n = run(sub12, controls=controls)
    add("12 Drop reform year (2010)", c, se, n)

    print("\nRobustness results")
    print("=" * 72)
    print(f"{'spec':<40} {'coef':>10} {'se':>10} {'N':>8}")
    for label, c, se, n, sig in rows:
        print(f"{label:<40} {c:>10.4f} {se:>10.4f} {n:>8}  {sig}")

    pd.DataFrame(rows, columns=["spec", "coef", "se", "n", "sig"]).to_csv(
        "replication_148001/output/robustness.csv", index=False
    )


if __name__ == "__main__":
    main()
