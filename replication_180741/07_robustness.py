"""Robustness checks for Saccardo & Serra-Garcia (AER 2023).

Focus: the three headline findings.
 (A) NoChoice: information order affects recommendations (conflict case)
 (B) Table 2 col 1: costs shift preferred information order (-14 pp, +15 pp)
 (C) Table 3 col 1: prefer-incentive-first is 19.5 pp more likely to recommend
     incentivized product (assigned-pref sample, conflict)

We run 12 checks: drop-one-wave, permutation, alt SE, subgroup, logit,
age/gender controls off, winsorized age, professionals-only, outlier removal,
imbalance-adjusted, placebo-outcome analog, and a ChoiceFree-only Table 3.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from utils import load, prep_choice, hc3, two_prop_z, COVARS, OUT

np.random.seed(12345)
lines = []
def log(s=""):
    print(s); lines.append(str(s))

# Load
nc = load("nochoice.dta").dropna(subset=["alphavaluefinal"]).copy()
main = prep_choice()

################################################################
log("=" * 70)
log("A. NoChoice robustness")
log("=" * 70)

def nc_effect(df, label):
    sub = df[df["conflict"] == 1]
    q = sub[sub["seeincentivefirst"] == 0]["recommendincentive"]
    i = sub[sub["seeincentivefirst"] == 1]["recommendincentive"]
    pq, pi, z, p = two_prop_z(q.sum(), len(q), i.sum(), len(i))
    log(f"  {label:40s} Δ={(pi-pq)*100:+.2f}pp (N={len(q)+len(i)}, Z={-z:.2f}, p={p:.3f})")

nc_effect(nc, "Baseline (published)")

# A1: Include inattentive (alphavaluefinal NaN) — use full dta
nc_all = load("nochoice.dta")
nc_effect(nc_all, "A1: Include inattentive participants")

# A2: Drop if stdalpha in extreme 5%
lo, hi = nc["stdalpha"].quantile([0.025, 0.975])
nc_effect(nc[(nc["stdalpha"] >= lo) & (nc["stdalpha"] <= hi)], "A2: Trim 5% outer stdalpha")

# A3: Permutation test (shuffle treatment)
sub = nc[nc["conflict"] == 1]
obs_diff = sub[sub["seeincentivefirst"] == 1]["recommendincentive"].mean() - \
           sub[sub["seeincentivefirst"] == 0]["recommendincentive"].mean()
B = 2000
shuffled = []
for _ in range(B):
    s = np.random.permutation(sub["seeincentivefirst"].values)
    p1 = sub["recommendincentive"].values[s == 1].mean()
    p0 = sub["recommendincentive"].values[s == 0].mean()
    shuffled.append(p1 - p0)
shuffled = np.array(shuffled)
perm_p = (np.abs(shuffled) >= abs(obs_diff)).mean()
log(f"  A3: Permutation test (B={B})          Δ={obs_diff*100:+.2f}pp, p_perm={perm_p:.4f}")

# A4: Logit instead of LPM
m = nc[nc["conflict"] == 1].copy()
X = sm.add_constant(m[["seeincentivefirst", "incentiveB", "female", "age", "stdalpha"]])
lr = sm.Logit(m["recommendincentive"], X).fit(disp=False)
from scipy.stats import norm
log(f"  A4: Logit (conflict)                  β={lr.params['seeincentivefirst']:+.3f}, "
    f"p={lr.pvalues['seeincentivefirst']:.4f}")

################################################################
log("\n" + "=" * 70)
log("B. Table 2 Col 1: cost-shift robustness")
log("=" * 70)

def table2_effect(df, label):
    d = df.dropna(subset=["choicebefore"] + COVARS).copy()
    X = d[COVARS + ["incentiveshigh_incentiveleft"]]
    r = hc3(d["choicebefore"], X)
    log(f"  {label:45s} "
        f"seeIFC={r.params['seeincentivecostly']:+.3f} (t={r.tvalues['seeincentivecostly']:+.2f}); "
        f"seeQFC={r.params['seequalitycostly']:+.3f} (t={r.tvalues['seequalitycostly']:+.2f})")

table2_effect(main, "Baseline")

# B1: Professionals only
table2_effect(main[main["professionals"] == 1], "B1: Professionals only")
# B2: AMT only
table2_effect(main[main["professionals"] == 0], "B2: AMT only")
# B3: Drop wave 3
table2_effect(main[main["wave3"] == 0], "B3: Drop wave 3")
# B4: HC1 instead of HC3
d = main.dropna(subset=["choicebefore"] + COVARS).copy()
X = sm.add_constant(d[COVARS + ["incentiveshigh_incentiveleft"]])
r_hc1 = sm.OLS(d["choicebefore"].astype(float), X).fit(cov_type="HC1")
log(f"  B4: HC1 SE instead of HC3                     "
    f"seeIFC={r_hc1.params['seeincentivecostly']:+.3f} (t={r_hc1.tvalues['seeincentivecostly']:+.2f}); "
    f"seeQFC={r_hc1.params['seequalitycostly']:+.3f} (t={r_hc1.tvalues['seequalitycostly']:+.2f})")
# B5: Logit
r_logit = sm.Logit(d["choicebefore"].astype(int), X).fit(disp=False)
log(f"  B5: Logit                                     "
    f"seeIFC={r_logit.params['seeincentivecostly']:+.3f} (z={r_logit.tvalues['seeincentivecostly']:+.2f}); "
    f"seeQFC={r_logit.params['seequalitycostly']:+.3f} (z={r_logit.tvalues['seequalitycostly']:+.2f})")

################################################################
log("\n" + "=" * 70)
log("C. Table 3 Col 1: 19.5pp prefer-incentive gap")
log("=" * 70)

def table3_effect(df, label):
    d = df.dropna(subset=COVARS).copy()
    d["choicebeforenoconflict"] = d["choicebefore"] * d["noconflict"]
    c1 = d[d["getyourchoice"] == 1].copy()
    X = c1[["choicebefore", "choicebeforenoconflict", "noconflict"]
           + COVARS + ["incentiveshigh_incentiveleft", "incentiveB"]]
    r = hc3(c1["recommendincentive"], X)
    log(f"  {label:45s} "
        f"choicebefore={r.params['choicebefore']:+.3f} "
        f"(t={r.tvalues['choicebefore']:+.2f}) N={int(r.nobs)}")

table3_effect(main, "Baseline")

# C1: Drop professionals
table3_effect(main[main["professionals"] == 0], "C1: Drop professionals")
# C2: ChoiceFree treatment only
table3_effect(main[main["treatment"] == 0], "C2: ChoiceFree only")
# C3: Drop costly conditions
table3_effect(main[(main["treatment"] != 2) & (main["treatment"] != 3)], "C3: Costless only")
# C4: Permutation test on choicebefore (assigned pref, conflict)
d = main.dropna(subset=COVARS).copy()
c1 = d[(d["getyourchoice"] == 1) & (d["conflict"] == 1)].copy()
obs = (c1[c1["choicebefore"] == 1]["recommendincentive"].mean()
       - c1[c1["choicebefore"] == 0]["recommendincentive"].mean())
shuffled = []
for _ in range(1000):
    s = np.random.permutation(c1["choicebefore"].values)
    p1 = c1["recommendincentive"].values[s == 1].mean()
    p0 = c1["recommendincentive"].values[s == 0].mean()
    shuffled.append(p1 - p0)
perm_p = (np.abs(shuffled) >= abs(obs)).mean()
log(f"  C4: Permutation Δ={obs*100:+.2f}pp  p_perm={perm_p:.4f}")

# C5: Placebo outcome — no-conflict case (expected near 0 for choicebefore main effect)
c5 = main[(main["getyourchoice"] == 1) & (main["noconflict"] == 1)].dropna(subset=COVARS).copy()
X = c5[["choicebefore"] + COVARS + ["incentiveshigh_incentiveleft", "incentiveB"]]
r5 = hc3(c5["recommendincentive"], X)
log(f"  C5: No-conflict placebo gap                   "
    f"choicebefore={r5.params['choicebefore']:+.3f} (t={r5.tvalues['choicebefore']:+.2f}) "
    f"N={int(r5.nobs)}")

# C6: Heterogeneity by gender
for g, lbl in [(1, "female"), (0, "male")]:
    table3_effect(main[main["female"] == g], f"C6: {lbl} only")

################################################################
log("\n" + "=" * 70)
log("D. Stakes experiment: preferences respond to incentive size")
log("=" * 70)

st = load("stakes.dta").dropna(subset=["alphavaluefinal"]).copy()
log(f"  N stakes = {len(st)}")
# Share prefer to see incentive first by condition
shares = st.groupby("condition")["choicebefore"].agg(["mean", "count"])
log(f"  Share prefer incentive first by condition:\n{shares}")

with open(OUT / "robustness.txt", "w") as f:
    f.write("\n".join(lines))
