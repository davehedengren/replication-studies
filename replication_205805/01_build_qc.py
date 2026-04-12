"""Build QC state-year aggregates of weighted SNAP enrollment counts by FPL band.

The source QC .dta files are wide — one row per household with up to 16 person
slots (AGE1..AGE16 etc.). The Stata code reshapes long, drops rows where all of
(AGE, SEX, REL) are missing, then `collapse (rawsum) fywgt`. That is equivalent
to: for each household, K = # of person slots with any of age/sex/rel filled,
and the contribution to the weighted sum is K * fywgt. We replicate by counting
non-missing AGE* columns per household row as K.
"""
from __future__ import annotations
import re
import pandas as pd
import numpy as np
from utils import qc_file_paths, DATA

GROUPS = {
    "0to50fpl":    lambda tp: (tp >= 0) & (tp <= 50),
    "50to115fpl":  lambda tp: (tp >= 50) & (tp <= 115),
    "0to130fpl":   lambda tp: (tp >= 0) & (tp <= 130),
    "100to130fpl": lambda tp: (tp >= 100) & (tp <= 130),
    "more130fpl":  lambda tp: (tp > 130),
}


def load_one(path) -> pd.DataFrame:
    df = pd.read_stata(str(path), convert_categoricals=False)
    df.columns = [c.lower() for c in df.columns]
    needed = ["state", "yrmonth", "fywgt", "tpov"]
    for c in needed:
        if c not in df.columns:
            raise RuntimeError(f"{path.name}: missing {c}")
    age_cols = [c for c in df.columns if re.fullmatch(r"age\d+", c)]
    df["k_persons"] = df[age_cols].notna().sum(axis=1).clip(lower=1)
    out = df[needed + ["k_persons"]].copy()
    return out


def main():
    frames = []
    for p in qc_file_paths():
        print(f"load {p.name}", flush=True)
        df = load_one(p)
        df = df.dropna(subset=["state", "yrmonth", "tpov", "fywgt"])
        ym = df["yrmonth"].astype(float)
        df.loc[ym < 10000, "yrmonth"] = ym[ym < 10000] + 190000
        df["year"] = (df["yrmonth"] // 100).astype(int)
        df["pwgt"] = df["fywgt"] * df["k_persons"]
        frames.append(df[["state", "year", "tpov", "pwgt"]])

    qc = pd.concat(frames, ignore_index=True)
    print(f"QC total person-equivalent rows: {len(qc):,}")
    qc = qc[(qc["year"] >= 1996) & (qc["year"] <= 2016)].copy()

    total = (qc.groupby(["state", "year"], as_index=False)["pwgt"].sum()
             .rename(columns={"pwgt": "fywgt_total"}))
    merged = total
    for gname, cond in GROUPS.items():
        mask = cond(qc["tpov"])
        g = (qc.loc[mask].groupby(["state", "year"], as_index=False)["pwgt"].sum()
             .rename(columns={"pwgt": f"fywgt_{gname}"}))
        merged = merged.merge(g, on=["state", "year"], how="left")
    merged = merged.fillna(0)
    merged.rename(columns={"state": "statefip"}, inplace=True)
    merged["statefip"] = merged["statefip"].astype(int)
    merged.to_parquet(DATA / "qc_stateyear.parquet")
    print(merged.head())
    print(f"QC state-year rows: {len(merged):,}")
    print(merged[["statefip", "year", "fywgt_0to130fpl", "fywgt_50to115fpl"]].describe())


if __name__ == "__main__":
    main()
