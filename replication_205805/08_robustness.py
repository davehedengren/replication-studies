"""Robustness checks for Table 1 Panel A main estimate (η ≈ 0.085 published).

Checks (all reuse the event-study sample unless noted):
  1. Drop the control (ln_denom) → no-denominator spec
  2. Drop CPS controls entirely (no GL index, no outreach, no unemployment)
  3. Leave-one-out by treated state (jackknife)
  4. Alternative SE: HC1 instead of clustered
  5. Restrict to post-recession only (year >= 2012)
  6. Restrict to pre-recession only (year <= 2007)
  7. Drop large states (CA, TX, NY, FL)
  8. Placebo: shuffle post_inclmt across treated states (100 draws)
  9. IHS transform on outcome (instead of log)
 10. Cluster by region instead of state
 11. Winsorize outcome at 1/99
 12. Full panel estimate (reuses col 7 from Table 1)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm
from utils import DATA

rng = np.random.default_rng(2024)
panel = pd.read_parquet(DATA / "full_panel.parquet")
es = pd.read_parquet(DATA / "eventstudy_sample.parquet")

CONTROLS = ["unemployment", "gl_index", "outreachnon0", "lnoutreach"]
MAIN_DEN = "ln_denom_0to130"
MAIN_Y = "ln_num_0to130"


def twfe(df, y, xvars, cluster_col="statefip", weights=None, se="cluster"):
    d = df.dropna(subset=[y] + xvars + [cluster_col, "year"]).copy()
    state_d = pd.get_dummies(d["statefip"], prefix="s", drop_first=True).astype(float)
    year_d  = pd.get_dummies(d["year"], prefix="y", drop_first=True).astype(float)
    X = pd.concat([d[xvars].astype(float).reset_index(drop=True),
                   state_d.reset_index(drop=True),
                   year_d.reset_index(drop=True)], axis=1)
    X = sm.add_constant(X, has_constant="add")
    yv = d[y].astype(float).reset_index(drop=True)
    groups = d[cluster_col].reset_index(drop=True)
    if weights is not None:
        w = df.loc[d.index, weights].astype(float).values
        mdl = sm.WLS(yv, X, weights=w)
    else:
        mdl = sm.OLS(yv, X)
    if se == "cluster":
        return mdl.fit(cov_type="cluster", cov_kwds={"groups": groups})
    elif se == "HC1":
        return mdl.fit(cov_type="HC1")
    else:
        return mdl.fit()


def report(label, res):
    b = res.params["inclmt100"]
    se = res.bse["inclmt100"]
    n = int(res.nobs)
    stars = "***" if abs(b / se) > 2.58 else "**" if abs(b / se) > 1.96 else "*" if abs(b / se) > 1.645 else " "
    print(f"{label:50s} {b:>7.3f} ({se:>5.3f}){stars}  N={n}")
    return b, se, n


results = []

# Baseline reference
r = twfe(es, MAIN_Y, ["inclmt100"] + CONTROLS + [MAIN_DEN])
results.append(("Baseline (Table 1 col 1)", *report("Baseline (Table 1 col 1)", r)))

# 1. Drop ln(denom)
r1 = twfe(es, MAIN_Y, ["inclmt100"] + CONTROLS)
results.append(("1. No denominator control", *report("1. No denominator control", r1)))

# 2. No controls at all
r2 = twfe(es, MAIN_Y, ["inclmt100", MAIN_DEN])
results.append(("2. No CPS/policy controls", *report("2. No CPS/policy controls", r2)))

# 3. Leave-one-out jackknife
jackknife = []
for s in sorted(es["statefip"].unique()):
    sub = es[es["statefip"] != s]
    ri = twfe(sub, MAIN_Y, ["inclmt100"] + CONTROLS + [MAIN_DEN])
    jackknife.append((s, ri.params["inclmt100"], ri.bse["inclmt100"]))
jkb = np.array([j[1] for j in jackknife])
print(f"{'3. LOO jackknife range':50s}"
      f" min={jkb.min():.3f} max={jkb.max():.3f} median={np.median(jkb):.3f}")
results.append(("3. LOO jackknife range", jkb.min(), jkb.max(), len(jackknife)))

# 4. HC1 SE
r4 = twfe(es, MAIN_Y, ["inclmt100"] + CONTROLS + [MAIN_DEN], se="HC1")
results.append(("4. HC1 (non-clustered) SE", *report("4. HC1 (non-clustered) SE", r4)))

# 5. Post-recession only (2012+)
sub5 = es[es["year"] >= 2012]
if sub5["statefip"].nunique() > 2:
    r5 = twfe(sub5, MAIN_Y, ["inclmt100"] + CONTROLS + [MAIN_DEN])
    results.append(("5. Years >= 2012", *report("5. Years >= 2012", r5)))

# 6. Pre-recession only (<=2007)
sub6 = es[es["year"] <= 2007]
if sub6["statefip"].nunique() > 2:
    r6 = twfe(sub6, MAIN_Y, ["inclmt100"] + CONTROLS + [MAIN_DEN])
    results.append(("6. Years <= 2007", *report("6. Years <= 2007", r6)))

# 7. Drop large states (CA=6, TX=48, NY=36, FL=12)
big = [6, 48, 36, 12]
sub7 = es[~es["statefip"].isin(big)]
r7 = twfe(sub7, MAIN_Y, ["inclmt100"] + CONTROLS + [MAIN_DEN])
results.append(("7. Drop CA/TX/NY/FL", *report("7. Drop CA/TX/NY/FL", r7)))

# 8. Placebo: permute inclmt across treated state IDs
treated_states = (es.groupby("statefip")["bbce_inclmt"].nunique()
                  [lambda s: s > 1].index.tolist())
placebo_betas = []
for _ in range(100):
    perm = list(treated_states)
    rng.shuffle(perm)
    mapping = dict(zip(treated_states, perm))
    sub8 = es.copy()
    sub8.loc[sub8["statefip"].isin(treated_states), "inclmt100"] = (
        sub8.loc[sub8["statefip"].isin(treated_states), "statefip"]
        .map(lambda s: es[es["statefip"] == mapping[s]]["inclmt100"].iloc[-1])
    )
    r8 = twfe(sub8, MAIN_Y, ["inclmt100"] + CONTROLS + [MAIN_DEN])
    placebo_betas.append(r8.params["inclmt100"])
pbeta = np.array(placebo_betas)
p_extreme = (np.abs(pbeta) >= abs(results[0][1])).mean()
print(f"{'8. Placebo permutation':50s}"
      f" mean={pbeta.mean():.3f} sd={pbeta.std():.3f} p(|β|≥obs)={p_extreme:.2f}")
results.append(("8. Placebo permutation (100 draws)", pbeta.mean(), pbeta.std(), p_extreme))

# 9. IHS transform
d9 = es.copy()
d9["ihs_num"] = np.arcsinh(d9["fywgt_0to130fpl"])
d9["ihs_den"] = np.arcsinh(d9["wtsupp_0to130fpl"])
r9 = twfe(d9, "ihs_num", ["inclmt100"] + CONTROLS + ["ihs_den"])
results.append(("9. IHS instead of log", *report("9. IHS instead of log", r9)))

# 10. Cluster by Census region (statefip → region)
CENSUS_REG = {
    # Northeast
    9:1, 23:1, 25:1, 33:1, 34:1, 36:1, 42:1, 44:1, 50:1,
    # Midwest
    17:2, 18:2, 19:2, 20:2, 26:2, 27:2, 29:2, 31:2, 38:2, 39:2, 46:2, 55:2,
    # South
    1:3, 5:3, 10:3, 11:3, 12:3, 13:3, 21:3, 22:3, 24:3, 28:3, 37:3, 40:3,
    45:3, 47:3, 48:3, 51:3, 54:3,
    # West
    2:4, 4:4, 6:4, 8:4, 15:4, 16:4, 30:4, 32:4, 35:4, 41:4, 49:4, 53:4, 56:4,
}
d10 = es.copy()
d10["region"] = d10["statefip"].map(CENSUS_REG)
r10 = twfe(d10, MAIN_Y, ["inclmt100"] + CONTROLS + [MAIN_DEN], cluster_col="region")
results.append(("10. Cluster by Census region", *report("10. Cluster by Census region", r10)))

# 11. Winsorize ln_num at 1/99 pct
d11 = es.copy()
lo = d11[MAIN_Y].quantile(0.01)
hi = d11[MAIN_Y].quantile(0.99)
d11[MAIN_Y] = d11[MAIN_Y].clip(lo, hi)
r11 = twfe(d11, MAIN_Y, ["inclmt100"] + CONTROLS + [MAIN_DEN])
results.append(("11. Winsorize outcome 1/99", *report("11. Winsorize outcome 1/99", r11)))

# 12. Full panel
r12 = twfe(panel, MAIN_Y, ["inclmt100"] + CONTROLS + [MAIN_DEN])
results.append(("12. Full panel (col 7)", *report("12. Full panel (col 7)", r12)))

# Save
pd.DataFrame([{"check": r[0], "beta_or_min": r[1], "se_or_max": r[2],
               "n_or_extra": r[3]} for r in results]
             ).to_csv(DATA / "robustness_results.csv", index=False)
print("\nSaved to replication_205805/data/robustness_results.csv")
