"""Data audit for Saccardo & Serra-Garcia (AER 2023).

Checks coverage, distributions, consistency, missingness, and duplicates in
the three main datasets: nochoice, choice_experiments, and stakes.
"""
from utils import load, OUT
import pandas as pd
import numpy as np

log_lines = []
def log(s=""):
    print(s); log_lines.append(str(s))

for name, file in [("NoChoice", "nochoice.dta"),
                   ("Choice", "choice_experiments.dta"),
                   ("Stakes", "stakes.dta"),
                   ("IA", "InformationArchitect.dta")]:
    log(f"\n=== {name} ({file}) ===")
    df = load(file)
    log(f"rows={len(df)}, cols={len(df.columns)}")
    log(f"duplicated id: {df['id'].duplicated().sum() if 'id' in df.columns else 'no id col'}")
    for c in ["recommendincentive", "choicebefore", "conflict", "getbefore",
              "getyourchoice", "incentiveB", "female", "age", "alphavaluefinal",
              "logitbelief", "treatment", "condition"]:
        if c in df.columns:
            n_miss = df[c].isna().sum()
            if df[c].dtype.kind in "biufc":
                log(f"  {c:22s} miss={n_miss:5d} mean={df[c].mean():.3f} "
                    f"min={df[c].min():.3f} max={df[c].max():.3f}")
            else:
                log(f"  {c:22s} miss={n_miss:5d} unique={df[c].nunique()}")

# Consistency checks on Choice
c = load("choice_experiments.dta")
c = c[~((c["study"] != 1) & c["alphavaluefinal"].isna())].copy()
m = c[(c["Highx10"] == 0) & (c["Highx100"] == 0)].copy()

log("\n=== Consistency checks ===")

# noconflict + conflict == 1 always
bad = ((m["noconflict"] + m["conflict"]) != 1).sum()
log(f"noconflict + conflict != 1: {bad} rows")

# getyourchoice implies getbefore == choicebefore
g = m[m["getyourchoice"] == 1]
mismatch = (g["getbefore"] != g["choicebefore"]).sum()
log(f"getyourchoice==1 but getbefore != choicebefore: {mismatch} rows")

# beliefs should be 0..100 in raw; belief in [0,1] range
for col in ["belief", "beliefbin", "logitbelief"]:
    if col in m.columns:
        log(f"  {col}: min={m[col].min():.3f} max={m[col].max():.3f} "
            f"mean={m[col].mean():.3f}")

# Age plausibility
for col in ["age"]:
    if col in m.columns:
        out_of_range = ((m[col] < 18) | (m[col] > 100)).sum()
        log(f"  age out of [18,100]: {out_of_range}")

# Missingness by treatment (systematic?)
log("\nLogit belief missingness by treatment (choice main):")
log(m.groupby("treatment").apply(
    lambda d: pd.Series({"n": len(d), "miss_logit": d["logitbelief"].isna().sum(),
                         "miss_rec": d["recommendincentive"].isna().sum()})
).to_string())

# Sample composition: female / age by treatment
log("\nFemale share and age by treatment:")
log(m.groupby("treatment")[["female", "age"]].mean().to_string())

# Balance check: choicebefore by wave
log("\nchoicebefore by wave:")
log(m.groupby(["wave1", "wave2", "wave3"])["choicebefore"].agg(["mean", "count"]).to_string())

# NoChoice: balance across arms
n = load("nochoice.dta").dropna(subset=["alphavaluefinal"])
log("\nNoChoice balance (female, age, stdalpha) by seeincentivefirst:")
log(n.groupby("seeincentivefirst")[["female", "age", "stdalpha"]].mean().to_string())

with open(OUT / "data_audit.txt", "w") as f:
    f.write("\n".join(log_lines))
