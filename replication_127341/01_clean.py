"""Build the gravity-regression dataset for equation (20).

Mirrors replication/code/stata/203_estimation_elasticities.do:
  - Start from SEL_CC (district-to-district credit card expenditures).
  - Merge in yod2 (= chi, commuting flow relative to pre-pandemic) and
    ipat from SEL_parms_od.
  - Merge in bilateral distances from KOR_commute_obs.
  - Restrict to 2020 observations.
"""
import numpy as np
import pandas as pd
from utils import OUT, load_sel_cc, load_sel_parms, load_commute


def main():
    cc = load_sel_cc()
    parms = load_sel_parms()[["RCo", "RCd", "statadate", "yod2", "ipat"]]
    merged = cc.merge(parms, on=["RCo", "RCd", "statadate"], how="inner")

    commute = load_commute("KOR")[["RCo", "RCd", "dist"]]
    merged = merged.merge(commute, on=["RCo", "RCd"], how="inner")

    merged["lns"] = np.log(merged["eod"])
    merged["lndist"] = np.log(merged["dist"] + 1)
    merged["yod"] = merged["yod2"]
    merged["year"] = merged["statadate"].dt.year

    out = OUT / "sel_gravity.parquet"
    merged.to_parquet(out)

    print(f"Rows total:       {len(merged):,}")
    print(f"Rows year==2020:  {(merged['year'] == 2020).sum():,}")
    print(f"Unique RCo:       {merged['RCo'].nunique()}")
    print(f"Unique RCd:       {merged['RCd'].nunique()}")
    print(f"Date range:       {merged['statadate'].min().date()} "
          f"to {merged['statadate'].max().date()}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
