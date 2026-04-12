"""Reproduce Table 1 sample sizes (experimental design outline)."""
from utils import load, OUT

rows = []

# NoChoice
df = load("nochoice.dta").dropna(subset=["alphavaluefinal"])
for v, lbl in [(1, "NoChoice-SeeIncentiveFirst"), (0, "NoChoice-AssessQualityFirst")]:
    rows.append((lbl, int((df["seeincentivefirst"] == v).sum())))

# Choice (main)
cx = load("choice_experiments.dta")
cx = cx[~((cx["study"] != 1) & cx["alphavaluefinal"].isna())]
main = cx[(cx["Highx10"] == 0) & (cx["Highx100"] == 0)]
labels = {0: "ChoiceFree", 1: "ChoiceFree-Professionals",
          2: "IncentiveFirstCostly", 3: "QualityFirstCostly"}
for t, lbl in labels.items():
    rows.append((lbl, int((main["treatment"] == t).sum())))

# High stakes
rows.append(("ChoiceFree-HighStakes10fold", int((cx["Highx10"] == 1).sum())))
rows.append(("ChoiceFree-HighStakes100fold", int((cx["Highx100"] == 1).sum())))

# Stakes
st = load("stakes.dta").dropna(subset=["alphavaluefinal"])
for t in sorted(st["condition"].dropna().unique()):
    rows.append((f"Stakes-{t}", int((st["condition"] == t).sum())))

# Information Architect
ia = load("InformationArchitect.dta").dropna(subset=["alphavaluefinal"])
for v, lbl in [(1, "IA-Advisor"), (0, "IA-Client")]:
    rows.append((lbl, int((ia["IAAdvisor"] == v).sum())))

with open(OUT / "table1_sample_sizes.txt", "w") as f:
    total = 0
    for lbl, n in rows:
        f.write(f"{lbl:35s} {n:6d}\n")
        total += n
    f.write(f"{'TOTAL':35s} {total:6d}\n")
    print(f"Total N = {total}")

for lbl, n in rows:
    print(f"  {lbl:35s} {n:6d}")
