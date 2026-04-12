"""Robustness checks for the LDW-CPS analysis (Imbens & Xu 2025).

Anchor: OLS regression-adjusted ATT of training on re78 in LDW-CPS,
which the paper reports at 1066 (627). All robustness checks compare
against that baseline.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

from utils import COVAR_LDW, OUT_DIR, load_lalonde, load_trimmed, reg_est, om_reg_est

warnings.filterwarnings("ignore")

Y = "re78"
T = "treat"
SEED = 1234


def ols_att(d, covar=COVAR_LDW):
    return reg_est(d, Y, T, covar)


def main():
    lal = load_lalonde()
    trim = load_trimmed()
    d_full = lal["ldw_cps"]
    d_trim = trim["ldw_cps_trim"]

    rows = []
    base_full = ols_att(d_full)
    base_trim = ols_att(d_trim)
    rows.append(("Baseline: OLS reg adj, LDW-CPS full",       *base_full))
    rows.append(("Baseline: OLS reg adj, LDW-CPS trimmed",    *base_trim))

    # 1. Drop pre-treatment earnings controls (no re74/re75/u74/u75)
    covar_noy = [v for v in COVAR_LDW if v not in ("re74", "re75", "u74", "u75")]
    rows.append(("1. Drop pre-earnings controls (full)",
                 *reg_est(d_full, Y, T, covar_noy)))
    rows.append(("1. Drop pre-earnings controls (trim)",
                 *reg_est(d_trim, Y, T, covar_noy)))

    # 2. Drop 1974 earnings only (mimics "LDW no74" in paper)
    covar_no74 = [v for v in COVAR_LDW if v not in ("re74", "u74")]
    rows.append(("2. Drop re74/u74 (full)",
                 *reg_est(d_full, Y, T, covar_no74)))

    # 3. Log(1+re78) outcome
    d = d_full.copy()
    d["log_re78"] = np.log1p(d[Y])
    rows.append(("3. Outcome = log(1+re78), full",
                 *reg_est(d, "log_re78", T, COVAR_LDW)))

    # 4. Winsorize re78 at 99th percentile
    d = d_full.copy()
    cap = np.percentile(d[Y], 99)
    d[Y] = d[Y].clip(upper=cap)
    rows.append(("4. Winsorize re78 at 99th pct (full)",
                 *reg_est(d, Y, T, COVAR_LDW)))

    # 5. Subgroup: black only
    rows.append(("5. Subgroup: black==1 (full)",
                 *reg_est(d_full[d_full["black"] == 1], Y, T, COVAR_LDW)))
    # 6. Subgroup: black==0 (full)
    try:
        rows.append(("6. Subgroup: black==0 (full)",
                     *reg_est(d_full[d_full["black"] == 0], Y, T, COVAR_LDW)))
    except Exception as e:
        rows.append(("6. Subgroup: black==0 (full)", np.nan, np.nan))

    # 7. Subgroup: nodegree==1 (less-educated)
    rows.append(("7. Subgroup: nodegree==1 (full)",
                 *reg_est(d_full[d_full["nodegree"] == 1], Y, T, COVAR_LDW)))

    # 8. Placebo outcome: use re75 as outcome (should be ~0 within exp)
    rows.append(("8. Placebo outcome re75, LDW only",
                 *reg_est(lal["ldw"], "re75", T, [c for c in COVAR_LDW if c != "re75" and c != "u75"])))

    # 9. Permutation test: randomize treatment within sample, recompute ATT
    rng = np.random.default_rng(SEED)
    d = d_full.copy()
    perm_est = np.empty(500)
    tvals = d[T].values.copy()
    for b in range(500):
        perm = rng.permutation(tvals)
        d2 = d.copy()
        d2[T] = perm
        est, _ = reg_est(d2, Y, T, COVAR_LDW)
        perm_est[b] = est
    p = (np.abs(perm_est) >= abs(base_full[0])).mean()
    print(f"\n[9] Permutation null: mean={perm_est.mean():.1f}, sd={perm_est.std():.1f}, "
          f"|perm|>=|obs| p = {p:.3f}")
    rows.append(("9. Permutation two-sided p (full, 500 reps)", p, 0.0))

    # 10. Cluster SE by education level (alt SE)
    d = d_full[[Y, T] + COVAR_LDW + ["education"]].dropna()
    Xm = sm.add_constant(d[[T] + COVAR_LDW].astype(float).values)
    mod = sm.OLS(d[Y].astype(float).values, Xm).fit(
        cov_type="cluster", cov_kwds={"groups": d["education"].values})
    rows.append(("10. Cluster SE by education (full)",
                 float(mod.params[1]), float(mod.bse[1])))

    # 11. Bootstrap SE (case resampling, 500 reps)
    d = d_full.reset_index(drop=True)
    boots = np.empty(500)
    n = len(d)
    for b in range(500):
        idx = rng.integers(0, n, n)
        try:
            boots[b], _ = reg_est(d.iloc[idx], Y, T, COVAR_LDW)
        except Exception:
            boots[b] = np.nan
    rows.append(("11. Bootstrap SE (500 reps, full)",
                 float(np.nanmean(boots)), float(np.nanstd(boots, ddof=1))))

    # 12. om.reg variant (matches paper row 4)
    rows.append(("12. Outcome-model imputation, full",
                 *om_reg_est(d_full, Y, T, COVAR_LDW)))
    rows.append(("12. Outcome-model imputation, trimmed",
                 *om_reg_est(d_trim, Y, T, COVAR_LDW)))

    out = pd.DataFrame(rows, columns=["check", "estimate", "se"])
    pd.set_option("display.width", 150)
    pd.set_option("display.max_colwidth", 60)
    print("\n" + "=" * 78)
    print("Robustness results (baseline = 1066 (627) full; 1554 (811) trimmed)")
    print("=" * 78)
    print(out.to_string(index=False,
          formatters={"estimate": lambda x: f"{x:10.3f}",
                      "se":       lambda x: f"({x:8.3f})"}))
    out.to_csv(OUT_DIR / "out_robustness.csv", index=False)


if __name__ == "__main__":
    main()
