"""Build CPS ASEC state-year denominators: weighted person counts by FPL band.

Mirrors meanstest/code/build/cps.do. We use ftotval (family total income) and
cutoff (federal poverty threshold) to assign each person to an FPL band.
Weights: asecwt (divided by 2 in 2014 per IPUMS instructions).
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from utils import CPS_DTA, DATA

NEEDED = ["year", "statefip", "asecflag", "asecwt", "ftotval",
          "cutoff", "age", "sex", "serial", "famunit"]


def main():
    print("reading CPS .dta (chunked)...", flush=True)
    reader = pd.read_stata(str(CPS_DTA), convert_categoricals=False,
                           columns=NEEDED, chunksize=500_000)
    parts = []
    total_rows = 0
    for chunk in reader:
        chunk = chunk[chunk["asecflag"] == 1]
        total_rows += len(chunk)
        parts.append(chunk)
        print(f"  chunk rows kept: {len(chunk):,} (running {total_rows:,})", flush=True)
    cps = pd.concat(parts, ignore_index=True)
    print(f"CPS ASEC rows: {len(cps):,}")

    cps["wtsupp"] = cps["asecwt"].astype(float)
    cps.loc[cps["year"] == 2014, "wtsupp"] /= 2

    cps = cps.dropna(subset=["ftotval", "cutoff"])
    cps = cps[cps["ftotval"] >= 0]
    cps["fpl"] = cps["ftotval"] / cps["cutoff"] * 100.0

    groups = {
        "0to50fpl":    (cps["fpl"] >= 0) & (cps["fpl"] <= 50),
        "0to100fpl":   (cps["fpl"] >= 0) & (cps["fpl"] <= 100),
        "0to130fpl":   (cps["fpl"] >= 0) & (cps["fpl"] <= 130),
        "100to130fpl": (cps["fpl"] >= 100) & (cps["fpl"] <= 130),
        "50to115fpl":  (cps["fpl"] >= 50) & (cps["fpl"] <= 115),
        "50to130fpl":  (cps["fpl"] >= 50) & (cps["fpl"] <= 130),
    }
    # Per-group wtsupp sums by state-year
    out = (cps.groupby(["statefip", "year"], as_index=False)["wtsupp"].sum()
           .rename(columns={"wtsupp": "pop"}))
    for gname, mask in groups.items():
        g = (cps.loc[mask].groupby(["statefip", "year"], as_index=False)["wtsupp"].sum()
             .rename(columns={"wtsupp": f"wtsupp_{gname}"}))
        out = out.merge(g, on=["statefip", "year"], how="left")
    out = out.fillna(0)
    out["statefip"] = out["statefip"].astype(int)
    out.to_parquet(DATA / "cps_stateyear.parquet")
    print(out.head())
    print(f"CPS state-year rows: {len(out):,}")
    print(out[["wtsupp_0to130fpl", "wtsupp_50to115fpl", "pop"]].describe())


if __name__ == "__main__":
    main()
