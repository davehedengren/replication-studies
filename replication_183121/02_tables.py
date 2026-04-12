"""Replicate Table 1 (main) and footnote-4 lag specifications."""
import pandas as pd
from utils import (
    load_data, run_areg,
    SPEC_CASES, SPEC_EXCESS,
    SPEC_CASES_7D, SPEC_EXCESS_7D,
    SPEC_CASES_14D, SPEC_EXCESS_14D,
    OUTPUT_DIR,
)

df = load_data()

SPECS = {
    "T1_col1_cases_deaths":       SPEC_CASES,
    "T1_col2_excess":             SPEC_EXCESS,
    "fn4_7day_lag_cases_deaths":  SPEC_CASES_7D,
    "fn4_7day_lag_excess":        SPEC_EXCESS_7D,
    "fn4_14day_lag_cases_deaths": SPEC_CASES_14D,
    "fn4_14day_lag_excess":       SPEC_EXCESS_14D,
}

rows = []
for label, covid_vars in SPECS.items():
    res = run_areg(df, covid_vars)
    for v in covid_vars:
        rows.append({
            "spec": label,
            "var": v,
            "coef": float(res.params[v]),
            "se": float(res.std_errors[v]),
            "t": float(res.tstats[v]),
            "p": float(res.pvalues[v]),
            "nobs": int(res.nobs),
            "r2": float(res.rsquared),
        })
    print(f"\n=== {label} ===")
    for v in covid_vars:
        b = res.params[v]; se = res.std_errors[v]; p = res.pvalues[v]
        stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
        print(f"  {v:30s}  {b:+.4f}{stars:3s}  ({se:.4f})")
    print(f"  N={int(res.nobs)}  R2={res.rsquared:.4f}")

out = pd.DataFrame(rows)
out.to_csv(OUTPUT_DIR / "table1_replication.csv", index=False)

print("\n\n====== PUBLISHED vs REPLICATION (Table 1) ======")
print(f"{'Variable':<22}{'Pub β':>10}{'Pub SE':>10}{'Rep β':>10}{'Rep SE':>10}")

pub = {
    "stdSTCasesT":    (-0.075, 0.044),
    "stdSTDeathsT":   ( 0.051, 0.050),
    "stdSTExLowEstT": (-0.056, 0.083),
}
rep = {r["var"]: (r["coef"], r["se"]) for r in rows if r["spec"].startswith("T1_")}
for v, (pb, psb) in pub.items():
    rb, rsb = rep[v]
    print(f"{v:<22}{pb:>10.3f}{psb:>10.3f}{rb:>10.3f}{rsb:>10.3f}")
print("Published N=1,000 for both columns; Adj R² = 0.042, 0.043")
