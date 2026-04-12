"""Section 5: NoChoice experiment. Replicate the core information-order effect.

Published (page 22):
- Conflict: See Incentive First -> 79% recommend incentivized; Quality First -> 62%
  Z = 2.69, p = 0.007, N = 213
- No conflict: Incentive First -> 89%, Quality First -> 86% (N=86, p=0.685)
- 16% of the time advisors recommend A even when signal=$2 and incentive is B
"""
import numpy as np
import pandas as pd
from utils import load, hc3, two_prop_z, OUT

df = load("nochoice.dta").dropna(subset=["alphavaluefinal"]).copy()
print("N =", len(df))

# Proportion tests (paper p.22)
for c in [1, 0]:
    sub = df[df["conflict"] == c]
    q = sub[sub["seeincentivefirst"] == 0]["recommendincentive"]
    i = sub[sub["seeincentivefirst"] == 1]["recommendincentive"]
    pq, pi, z, pv = two_prop_z(q.sum(), len(q), i.sum(), len(i))
    print(f"conflict={c}: quality_first p={pq:.3f} (n={len(q)}); "
          f"incentive_first p={pi:.3f} (n={len(i)}); "
          f"Z={-z:.3f}, p={pv:.4f}, N={len(q)+len(i)}")

# 16% case: recommend A when incentive=B and signal=$2 (no conflict)
sub = df[(df["incentiveB"] == 1) & (df["noconflict"] == 1)]
notrec = 1 - sub["recommendincentive"]
print(f"Recommend A when incentive=B & signal=$2: {notrec.mean():.3f}, N={len(sub)}")

# Appendix C.1 regression (Table C.1): conflict sample, HC3
m = df[(df["missingalpha"] == 0) & (df["conflict"] == 1)].copy()
cols = ["seeincentivefirst", "noconflict", "incentiveB", "female", "age", "stdalpha"]
X = m[cols].copy()
X = X.drop(columns=["noconflict"])  # collinear in conflict==1 subset
res = hc3(m["recommendincentive"], X)
print("\nTable C.1 col 1 (conflict only):")
print(res.summary().tables[1])

with open(OUT / "nochoice_results.txt", "w") as f:
    for c in [1, 0]:
        sub = df[df["conflict"] == c]
        q = sub[sub["seeincentivefirst"] == 0]["recommendincentive"]
        i = sub[sub["seeincentivefirst"] == 1]["recommendincentive"]
        pq, pi, z, pv = two_prop_z(q.sum(), len(q), i.sum(), len(i))
        f.write(f"conflict={c}: quality={pq:.4f} (n={len(q)}); "
                f"incentive={pi:.4f} (n={len(i)}); Z={-z:.4f}; p={pv:.4f}\n")
    f.write(f"\nSee Incentive First coefficient (conflict, HC3): "
            f"{res.params['seeincentivefirst']:.4f} "
            f"(SE {res.bse['seeincentivefirst']:.4f})\n")
