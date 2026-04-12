"""Section 7: Table 4 — Belief updating by preference and assignment.

Outcome: log-odds belief. No-constant OLS with HC3.
Regressors: `bad` = log-likelihood ratio for conflict signal,
            `good` = log-likelihood ratio for no-conflict signal.

Published (page 34):
- Assigned pref, incentive-first: beta_C = 0.267, beta_NC = 0.324
- Assigned pref, quality-first: beta_C = 0.346, beta_NC = 0.444
- Differences in beta_C: t=2.45 (p=0.014); in beta_NC: t=2.19 (p=0.029)
- Full sample equality test (F=5.57, p=0.018); excl-wrong (F=12.06, p<0.001)
"""
from utils import prep_choice, OUT
import statsmodels.api as sm
import pandas as pd

m = prep_choice()

def run(df, label):
    X = df[["bad", "good"]].astype(float)
    y = df["logitbelief"].astype(float)
    # no-constant OLS; HC3
    r = sm.OLS(y, X).fit(cov_type="HC3")
    print(f"{label} N={int(r.nobs)}: "
          f"beta_C={r.params['bad']:.3f} (SE {r.bse['bad']:.3f}), "
          f"beta_NC={r.params['good']:.3f} (SE {r.bse['good']:.3f})")
    return r

# Panel A / B structure
subs = {
    "All assigned pref": m[m["getyourchoice"] == 1],
    "  choicebefore=0 (quality first)": m[(m["getyourchoice"] == 1) & (m["choicebefore"] == 0)],
    "  choicebefore=1 (incentive first)": m[(m["getyourchoice"] == 1) & (m["choicebefore"] == 1)],
    "All not assigned": m[m["getyourchoice"] == 0],
    "  choicebefore=0": m[(m["getyourchoice"] == 0) & (m["choicebefore"] == 0)],
    "  choicebefore=1": m[(m["getyourchoice"] == 0) & (m["choicebefore"] == 1)],
    "Assigned, excl wrong": m[(m["getyourchoice"] == 1) & (m["updatewrong"] == 0)],
    "  choicebefore=0 excl wrong": m[(m["getyourchoice"] == 1) & (m["updatewrong"] == 0) & (m["choicebefore"] == 0)],
    "  choicebefore=1 excl wrong": m[(m["getyourchoice"] == 1) & (m["updatewrong"] == 0) & (m["choicebefore"] == 1)],
}
results = {}
for label, df in subs.items():
    df2 = df.dropna(subset=["logitbelief", "bad", "good"])
    if len(df2) > 0:
        results[label] = run(df2, label)

# Test beta_C = beta_NC in assigned-pref, all (paper: F=5.57, p=0.018)
r = results["All assigned pref"]
t = (r.params["good"] - r.params["bad"]) / \
    ((r.cov_params().loc["good", "good"] + r.cov_params().loc["bad", "bad"]
      - 2 * r.cov_params().loc["good", "bad"]) ** 0.5)
print(f"\nAssigned pref (all) good=bad: t={t:.3f}, F={t**2:.3f}")

r = results["Assigned, excl wrong"]
t = (r.params["good"] - r.params["bad"]) / \
    ((r.cov_params().loc["good", "good"] + r.cov_params().loc["bad", "bad"]
      - 2 * r.cov_params().loc["good", "bad"]) ** 0.5)
print(f"Assigned (excl wrong) good=bad: F={t**2:.3f}")

# Interaction tests (paper: t=2.45, 2.19 for assigned-pref x choice for bad, good)
for label, filt in [("Assigned pref (all)", m["getyourchoice"] == 1),
                    ("Assigned (excl wrong)", (m["getyourchoice"] == 1) & (m["updatewrong"] == 0))]:
    d = m[filt].dropna(subset=["logitbelief", "good", "bad", "goodchoicebefore", "badchoicebefore"])
    X = d[["good", "bad", "goodchoicebefore", "badchoicebefore"]].astype(float)
    y = d["logitbelief"].astype(float)
    rr = sm.OLS(y, X).fit(cov_type="HC3")
    print(f"\n{label} interaction (N={int(rr.nobs)}):")
    for k in ["goodchoicebefore", "badchoicebefore"]:
        print(f"  {k}: {rr.params[k]:+.3f} (SE {rr.bse[k]:.3f}) t={rr.tvalues[k]:+.2f} p={rr.pvalues[k]:.3f}")

with open(OUT / "table4_results.txt", "w") as f:
    for label, r in results.items():
        f.write(f"{label} N={int(r.nobs)}: "
                f"beta_C={r.params['bad']:.3f} (SE {r.bse['bad']:.3f}), "
                f"beta_NC={r.params['good']:.3f} (SE {r.bse['good']:.3f})\n")
