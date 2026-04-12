"""Robustness checks for Harrell et al. (2023)."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from utils import load_data, CONTROLS, OUTCOME, OUTPUT_DIR

df = load_data()
rng = np.random.default_rng(20260411)


def fit(data, covid_vars, cluster_col="EmailPairID",
        include_week=True, include_day=True, include_state=True,
        extra_regs=None):
    rhs = list(covid_vars) + CONTROLS
    blocks = [data[rhs].astype(float).reset_index(drop=True)]
    if include_week:
        wk = pd.get_dummies(data["Weeksent"], prefix="wk", drop_first=True).astype(float)
        blocks.append(wk.reset_index(drop=True))
    if include_day:
        dy = pd.get_dummies(data["Daysent"], prefix="dy", drop_first=True).astype(float)
        blocks.append(dy.reset_index(drop=True))
    if include_state:
        st = pd.get_dummies(data["State"], prefix="st", drop_first=True).astype(float)
        blocks.append(st.reset_index(drop=True))
    if extra_regs is not None:
        blocks.append(extra_regs.reset_index(drop=True))
    X = pd.concat(blocks, axis=1)
    X = sm.add_constant(X)
    y = data[OUTCOME].astype(float).reset_index(drop=True)
    if cluster_col is None:
        return sm.OLS(y, X).fit(cov_type="HC1")
    return sm.OLS(y, X).fit(
        cov_type="cluster",
        cov_kwds={"groups": data[cluster_col].astype(int).reset_index(drop=True).values},
    )


def main_coefs(res, covid_vars):
    return {v: (float(res.params[v]), float(res.bse[v]), float(res.pvalues[v]))
            for v in covid_vars}


def fmt(d):
    parts = []
    for v, (b, se, p) in d.items():
        stars = "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else ""
        parts.append(f"{v.replace('stdST','').replace('T',''):9s}={b:+.4f}{stars}(SE {se:.4f})")
    return "  ".join(parts)


# ======================================================================
print("\n======= Robustness checks =======\n")
results = []
SPECS = {
    "col1 (cases+deaths)": ["stdSTCasesT", "stdSTDeathsT"],
    "col2 (excess)":        ["stdSTExLowEstT"],
}

def add(label, covid_vars, d):
    results.append({
        "check": label, "spec": " / ".join(covid_vars),
        **{v: f"{d[v][0]:+.4f} (SE {d[v][1]:.4f}, p={d[v][2]:.3f})" for v in covid_vars}
    })


# 1. Baseline
for name, cv in SPECS.items():
    res = fit(df, cv)
    d = main_coefs(res, cv)
    print(f"[1] Baseline {name:25s} {fmt(d)}")
    add(f"1. Baseline ({name})", cv, d)

# 2. Robust HC1 SEs (no clustering)
for name, cv in SPECS.items():
    res = fit(df, cv, cluster_col=None)
    d = main_coefs(res, cv)
    print(f"[2] HC1 {name:31s} {fmt(d)}")
    add(f"2. HC1 robust SE ({name})", cv, d)

# 3. Cluster at State level
for name, cv in SPECS.items():
    res = fit(df, cv, cluster_col="State")
    d = main_coefs(res, cv)
    print(f"[3] State-cluster {name:22s} {fmt(d)}")
    add(f"3. Cluster at State ({name})", cv, d)

# 4. Drop NY+NJ (epicenter of early COVID)
dfd = df[~df["State"].isin([df["State"].mode()[0], 34])].copy()  # modal state & NJ-ish
# safer: drop two largest-cases states
tops = df.groupby("State")["STCasesT"].sum().nlargest(2).index.tolist()
dfd = df[~df["State"].isin(tops)].copy()
for name, cv in SPECS.items():
    res = fit(dfd, cv)
    d = main_coefs(res, cv)
    print(f"[4] Drop top-2 COVID states {name:14s} {fmt(d)}  N={len(dfd)}")
    add(f"4. Drop 2 top COVID states ({name})", cv, d)

# 5. Drop early weeks (≤ week 6) — before COVID really hit
dfe = df[df["Weeksent"] >= 7].copy()
for name, cv in SPECS.items():
    res = fit(dfe, cv)
    d = main_coefs(res, cv)
    print(f"[5] Weeks >= 7 {name:24s} {fmt(d)}  N={len(dfe)}")
    add(f"5. Weeks >= 7 ({name})", cv, d)

# 6. No fixed effects at all (just controls)
for name, cv in SPECS.items():
    res = fit(df, cv, include_week=False, include_day=False, include_state=False)
    d = main_coefs(res, cv)
    print(f"[6] No FE {name:29s} {fmt(d)}")
    add(f"6. Drop all FEs ({name})", cv, d)

# 7. Only State FE (drop week/day)
for name, cv in SPECS.items():
    res = fit(df, cv, include_week=False, include_day=False)
    d = main_coefs(res, cv)
    print(f"[7] State FE only {name:21s} {fmt(d)}")
    add(f"7. State FE only ({name})", cv, d)

# 8. No demographic controls
def fit_noctrl(data, covid_vars):
    blocks = [data[list(covid_vars)].astype(float).reset_index(drop=True)]
    wk = pd.get_dummies(data["Weeksent"], prefix="wk", drop_first=True).astype(float)
    dy = pd.get_dummies(data["Daysent"], prefix="dy", drop_first=True).astype(float)
    st = pd.get_dummies(data["State"], prefix="st", drop_first=True).astype(float)
    X = pd.concat([b.reset_index(drop=True) for b in [blocks[0], wk, dy, st]], axis=1)
    X = sm.add_constant(X)
    y = data[OUTCOME].astype(float).reset_index(drop=True)
    return sm.OLS(y, X).fit(cov_type="cluster",
                             cov_kwds={"groups": data["EmailPairID"].astype(int).values})
for name, cv in SPECS.items():
    res = fit_noctrl(df, cv)
    d = main_coefs(res, cv)
    print(f"[8] No demographic controls {name:11s} {fmt(d)}")
    add(f"8. No controls ({name})", cv, d)

# 9. Permutation (placebo) test — shuffle COVID intensity across state-days, 500 draws
print("\n[9] Permutation test (500 draws) on stdSTCasesT coef:")
rng = np.random.default_rng(20260411)
df_perm = df.copy()
orig_coef = None
coefs_perm = []
for i in range(500):
    df_perm["stdSTCasesT_perm"] = rng.permutation(df["stdSTCasesT"].values)
    df_perm["stdSTDeathsT_perm"] = rng.permutation(df["stdSTDeathsT"].values)
    res = fit(df_perm, ["stdSTCasesT_perm", "stdSTDeathsT_perm"])
    coefs_perm.append(float(res.params["stdSTCasesT_perm"]))
coefs_perm = np.array(coefs_perm)
actual_b = -0.0780
p_perm = (np.abs(coefs_perm) >= np.abs(actual_b)).mean()
print(f"   Permutation 2-sided p-value for |coef| ≥ {actual_b:+.4f}: {p_perm:.3f}")
print(f"   Permutation mean: {coefs_perm.mean():+.4f}  sd: {coefs_perm.std():.4f}")
add("9. Permutation p-value (cases)",
    ["stdSTCasesT"],
    {"stdSTCasesT": (actual_b, float(coefs_perm.std()), float(p_perm))})

# 10. Leave-one-week-out sensitivity
print("\n[10] Leave-one-week-out for stdSTCasesT coef (col1 spec):")
loo = []
for wk in sorted(df["Weeksent"].unique()):
    subs = df[df["Weeksent"] != wk]
    if subs["Weeksent"].nunique() < 2:
        continue
    res = fit(subs, ["stdSTCasesT", "stdSTDeathsT"])
    b = float(res.params["stdSTCasesT"])
    se = float(res.bse["stdSTCasesT"])
    loo.append((wk, b, se, len(subs)))
    print(f"   drop wk {wk:>4.1f}:  β={b:+.4f}  SE={se:.4f}  N={len(subs)}")
low = min(x[1] for x in loo); high = max(x[1] for x in loo)
add("10. LOO week range (cases)", ["stdSTCasesT"],
    {"stdSTCasesT": (low, high - low, 0.0)})

# 11. Alt: use RAW (un-standardized) COVID variables (coef will be scaled)
print("\n[11] Raw (un-standardized) COVID variables:")
for name, rawv, stdname in [("col1-cases", "STCasesT", "stdSTCasesT"),
                             ("col1-deaths", "STDeathsT", "stdSTDeathsT"),
                             ("col2-excess", "STExLowEstT", "stdSTExLowEstT")]:
    pass
# Just do the paired specs:
res = fit(df, ["STCasesT", "STDeathsT"])
b1, b2 = res.params["STCasesT"], res.params["STDeathsT"]
# Re-standardize ex post to confirm match
print(f"   raw cases β={b1:.2e}  × sd({df['STCasesT'].std():.0f}) = {b1*df['STCasesT'].std():+.4f}")
print(f"   raw deaths β={b2:.2e}  × sd({df['STDeathsT'].std():.0f}) = {b2*df['STDeathsT'].std():+.4f}")
add("11. Raw-var cases β × SD", ["STCasesT"],
    {"STCasesT": (float(b1*df['STCasesT'].std()), float(res.bse['STCasesT']*df['STCasesT'].std()),
                 float(res.pvalues['STCasesT']))})

# 12. Placebo outcome: shuffle Posoutcome, re-estimate
print("\n[12] Placebo outcome (shuffled Posoutcome):")
df_plac = df.copy()
df_plac["Posoutcome"] = rng.permutation(df["Posoutcome"].values)
res = fit(df_plac, ["stdSTCasesT", "stdSTDeathsT"])
d = main_coefs(res, ["stdSTCasesT", "stdSTDeathsT"])
print(f"   Placebo: {fmt(d)}")
add("12. Placebo outcome", ["stdSTCasesT", "stdSTDeathsT"], d)

# Save
out = pd.DataFrame(results)
out.to_csv(OUTPUT_DIR / "robustness_summary.csv", index=False)
print(f"\nSaved robustness_summary.csv ({len(out)} rows)")
