"""Phase 3 data audit for Sampson (2023) R&D efficiency replication.

Checks coverage, distributions, panel balance, and duplicates in the two input
datasets (ANBERD and STAN) that feed the R&D efficiency calculation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils import RAW, BASELINE_COUS, MANUF_INDS, DROP_SUFFIX, compute_rd_intensity


def main():
    print("=" * 72)
    print("DATA AUDIT — ANBERD + STAN inputs for equation (46)")
    print("=" * 72)

    anberd = pd.read_csv(RAW / "ANBERD_REV4.csv", low_memory=False)
    print("\n1. ANBERD_REV4.csv raw")
    print(f"   rows = {len(anberd):,}")
    print(f"   countries (COU codes) = {anberd['COU'].nunique()}")
    print(f"   industries (IND codes) = {anberd['IND'].nunique()}")
    print(f"   years = {anberd['TIME'].min()}..{anberd['TIME'].max()}")
    print(f"   CUR values = {anberd['CUR'].unique().tolist()}")
    print(f"   any Value NaN / negatives = "
          f"{anberd['Value'].isna().sum()} / {(anberd['Value'] < 0).sum()}")

    stan = pd.read_csv(RAW / "STANI4_2016.csv", low_memory=False)
    print("\n2. STANI4_2016.csv raw")
    print(f"   rows = {len(stan):,}")
    print(f"   countries = {stan['LOCATION'].nunique()}")
    print(f"   industries = {stan['IND'].nunique()}")
    print(f"   variables = {stan['VAR'].nunique()} ({list(stan['VAR'].unique()[:5])}...)")
    print(f"   years = {stan['TIME'].min()}..{stan['TIME'].max()}")
    va = stan[stan["VAR"] == "VALU"]
    print(f"   VALU rows = {len(va):,}")
    print(f"   VALU Value NaN = {va['Value'].isna().sum()}")

    # After filters
    m = compute_rd_intensity()
    print("\n3. Merged R&D intensity table (after filters)")
    print(f"   rows = {len(m):,}")
    print(f"   countries (3-letter cou) = {m['cou'].nunique()}")
    print(f"   industries = {m['ind'].nunique()} "
          f"(expected {len(MANUF_INDS)} manufacturing)")
    print(f"   years = {m['year'].min()}..{m['year'].max()}")
    print(f"   rd_intensity min/median/max = "
          f"{m['rd_intensity'].min():.2e} / "
          f"{m['rd_intensity'].median():.2e} / "
          f"{m['rd_intensity'].max():.2e}")
    print(f"   any non-positive intensity = {(m['rd_intensity'] <= 0).sum()} "
          "(should be 0)")

    # Panel balance for 2010-14 baseline sample
    s = m[m["year"].between(2010, 2014)].copy()
    s["cou_obs_man"] = s.groupby(["cou", "year"])["cou"].transform("count")
    balance = s[s["cou_obs_man"] >= 14].groupby("cou")["cou_obs_man"].agg(["min", "max", "count"])
    print(f"\n4. Baseline sample (2010-14, >=14 industries per cou-year)")
    print(f"   countries making it into baseline = {balance.shape[0]}")
    missing = set(BASELINE_COUS) - set(balance.index)
    extra = set(balance.index) - set(BASELINE_COUS)
    print(f"   BASELINE_COUS missing from data = {sorted(missing) or 'none'}")
    print(f"   data countries not in BASELINE_COUS = {sorted(extra) or 'none'}")

    # Check that 25 baseline countries form the expected panel
    base = s[s["cou"].isin(BASELINE_COUS) & (s["cou_obs_man"] >= 14)]
    print(f"\n5. Panel structure for 25 OECD baseline")
    print(f"   rows = {len(base):,}   "
          f"(expect roughly 25 * 5 * 20 = 2500 max, "
          f"will be less due to missing industry-years)")
    pn = base.groupby("cou").size().rename("obs")
    print(f"   obs per country: min={pn.min()}, median={int(pn.median())}, max={pn.max()}")
    # Countries with fewest obs
    print("\n   fewest-obs countries in baseline:")
    for cou, n in pn.sort_values().head(5).items():
        print(f"     {cou}: {n}")

    # Duplicate check on (cou, ind, year)
    dup = m.duplicated(subset=["cou", "ind", "year"]).sum()
    print(f"\n6. Duplicate (cou, ind, year) triples in merged data = {dup} (should be 0)")

    # Check that duplicate-reporter rules were applied
    raw_cous = set(anberd["COU"].unique())
    kept = {c for c in raw_cous if c not in DROP_SUFFIX}
    overlap = sorted(c for c in kept if c.endswith("_PF") or c.endswith("_MA"))
    print(f"\n7. After duplicate-reporter drop rules, COU codes still carrying "
          f"_PF/_MA suffix = {len(overlap)}")
    print(f"   (these are non-overlap countries like {overlap[:5]})")

    # Outliers in log R&D intensity
    q = m["lrd_intensity"].quantile([0.01, 0.05, 0.5, 0.95, 0.99]).to_dict()
    print(f"\n8. log R&D intensity distribution:")
    for p, v in q.items():
        print(f"   p{int(p*100):02d} = {v:.3f}")


if __name__ == "__main__":
    main()
