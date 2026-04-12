"""Section 7: Table 3 — Recommendations by Assignment x Preference.

Published (pages 30-33):
- Assigned preference gap (conflict): ~19-20 pp between prefer-incentive vs prefer-quality
- Not-assigned preference gap: ~0
- 9.8 pp gap if you got your preferred order vs 23.5 pp main gap (for ChoiceFree only)
- "9 percentage points less likely" when moral types get quality first (t=3.05, p=0.002)
"""
from utils import prep_choice, hc3, COVARS, OUT
import pandas as pd

main = prep_choice()
m = main.dropna(subset=COVARS).copy()

base_cols = COVARS + ["incentiveshigh_incentiveleft", "incentiveB"]

# Build interaction variables
m["choicebeforenoconflict"] = m["choicebefore"] * m["noconflict"]
m["choicebeforenotgetyourchoice"] = m["choicebefore"] * (1 - m["getyourchoice"])
m["notgetyourchoicenoconflict"] = (1 - m["getyourchoice"]) * m["noconflict"]
m["notgetyourchoice"] = 1 - m["getyourchoice"]

# Col 1: assigned preference
c1 = m[m["getyourchoice"] == 1].copy()
X1 = c1[["choicebefore", "choicebeforenoconflict", "noconflict"] + base_cols]
r1 = hc3(c1["recommendincentive"], X1)

# Col 2: not assigned preference
c2 = m[m["getyourchoice"] == 0].copy()
X2 = c2[["choicebefore", "choicebeforenoconflict", "noconflict"] + base_cols]
r2 = hc3(c2["recommendincentive"], X2)

# Col 3: both, with interactions
X3 = m[["choicebefore", "choicebeforenoconflict", "noconflict",
        "notgetyourchoice", "choicebeforenotgetyourchoice",
        "notgetyourchoicenoconflict"] + base_cols]
r3 = hc3(m["recommendincentive"], X3)

for label, r in [("Col 1 (assigned pref)", r1),
                 ("Col 2 (not assigned)", r2),
                 ("Col 3 (both)", r3)]:
    print(f"\n=== Table 3 {label} (N={int(r.nobs)}) ===")
    print(r.summary().tables[1])

# Page 30: "9.8 percentage point more likely ... t=3.66"
# reg recommend ~ getbefore + incentiveB + incentiveBgetbefore + covariates
# sample: choicebefore==1, conflict==1
sub = m[(m["choicebefore"] == 1) & (m["conflict"] == 1)].copy()
sub["incentiveBgetbefore"] = sub["incentiveB"] * sub["getbefore"]
X = sub[["getbefore", "incentiveB", "incentiveBgetbefore"] + COVARS + ["incentiveshigh_incentiveleft"]]
r_pref_inc = hc3(sub["recommendincentive"], X)
print("\nPage 30: Prefer incentive-first x get preference (conflict):")
print(r_pref_inc.summary().tables[1].as_text().split("\n")[:6])
gb = r_pref_inc.params["getbefore"]
print(f"  getbefore coef = {gb:.4f}, t = {r_pref_inc.tvalues['getbefore']:.2f}")

# Page 32: "9 pp less, t=3.05 p=0.002" for moral types (choicebefore==0)
sub0 = m[(m["choicebefore"] == 0) & (m["conflict"] == 1)].copy()
sub0["incentiveBgetbefore"] = sub0["incentiveB"] * sub0["getbefore"]
X0 = sub0[["getbefore", "incentiveB", "incentiveBgetbefore"] + COVARS + ["incentiveshigh_incentiveleft"]]
r_pref_q = hc3(sub0["recommendincentive"], X0)
gb0 = r_pref_q.params["getbefore"]
print(f"\nPage 32: moral types getbefore coef = {gb0:.4f}, t = {r_pref_q.tvalues['getbefore']:.2f}")

# Page 33: ChoiceFree only, assigned pref, 23.5 pp gap
sub_cf = m[(m["treatment"] == 0) & (m["getyourchoice"] == 1)].copy()
Xcf = sub_cf[["choicebefore", "choicebeforenoconflict", "noconflict"] + base_cols]
Xcf = Xcf.drop(columns=["professionalsfree", "seeincentivecostly", "seequalitycostly"])
r_cf = hc3(sub_cf["recommendincentive"], Xcf)
print(f"\nPage 33: ChoiceFree-only gap (choicebefore coef) = "
      f"{r_cf.params['choicebefore']:.4f} "
      f"95%CI=[{r_cf.conf_int().loc['choicebefore'].values}]")

with open(OUT / "table3_results.txt", "w") as f:
    for label, r in [("Col1", r1), ("Col2", r2), ("Col3", r3)]:
        f.write(f"\n{label} N={int(r.nobs)}:\n")
        for k in r.params.index:
            f.write(f"  {k:35s} {r.params[k]:+.4f} ({r.bse[k]:.4f})\n")
    f.write(f"\nPage 30 getbefore (prefer incentive): "
            f"{gb:+.4f} t={r_pref_inc.tvalues['getbefore']:.2f}\n")
    f.write(f"Page 32 getbefore (prefer quality): "
            f"{gb0:+.4f} t={r_pref_q.tvalues['getbefore']:.2f}\n")
    f.write(f"Page 33 ChoiceFree gap: {r_cf.params['choicebefore']:+.4f}\n")
