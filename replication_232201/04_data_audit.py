"""Data audit for Imbens & Xu (2025) replication datasets.

Checks coverage, distributions, missingness, duplicates, sub-sample
consistency, and overlap diagnostics across the LaLonde data (NSW/DW +
CPS/PSID controls) and the IRS lottery data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils import COVAR_IRS, COVAR_LDW, load_irs, load_lalonde, load_trimmed


def summary(df, name, cols):
    print(f"\n[{name}] n={len(df)}, cols={len(df.columns)}")
    print(df[cols].describe().T[["count", "mean", "std", "min", "max"]])
    nulls = df[cols].isna().sum()
    if nulls.any():
        print(" missing:", dict(nulls[nulls > 0]))
    else:
        print(" missing: none")


def main():
    lal = load_lalonde()
    trim = load_trimmed()
    irs = load_irs()

    print("=" * 70)
    print("1. LaLonde datasets — coverage and missingness")
    print("=" * 70)
    for key in ("nsw", "ldw", "cps1", "psid1", "ldw_cps", "ldw_psid"):
        cols = [c for c in COVAR_LDW + ["re78"] if c in lal[key].columns]
        summary(lal[key], key, cols)

    # Consistency: ldw_cps = ldw_tr + cps1 (treat counts)
    assert len(lal["ldw_cps"]) == len(lal["ldw_tr"]) + len(lal["cps1"])
    assert len(lal["ldw_psid"]) == len(lal["ldw_tr"]) + len(lal["psid1"])
    print("\nSample-construction identities (treated experimental + nonexp controls) pass.")

    # Duplicates
    print("\n-- Duplicate rows (full-row match) --")
    for key in ("ldw", "cps1", "psid1"):
        n_dup = lal[key].duplicated().sum()
        print(f"  {key}: {n_dup} full-row duplicates")

    print("\n-- Key indicator sanity --")
    for key in ("cps1", "psid1", "ldw"):
        d = lal[key]
        if "re74" in d.columns:
            agree = ((d["re74"] == 0) == (d["u74"] == 1)).all()
            print(f"  {key}: u74 == (re74==0)?  {agree}")
        agree = ((d["re75"] == 0) == (d["u75"] == 1)).all()
        print(f"  {key}: u75 == (re75==0)?  {agree}")

    print("\n-- Outcome bounds --")
    for key in ("ldw", "cps1", "psid1"):
        d = lal[key]
        print(f"  {key}: re78 min={d['re78'].min():.0f} "
              f"max={d['re78'].max():.0f} mean={d['re78'].mean():.0f}")

    print("\n-- Covariate means, treated vs CPS controls (standardized differences) --")
    tr = lal["ldw_tr"]
    co = lal["cps1"]
    for v in COVAR_LDW:
        mt, mc = tr[v].mean(), co[v].mean()
        s = np.sqrt((tr[v].var() + co[v].var()) / 2)
        sd = (mt - mc) / s if s > 0 else np.nan
        flag = "  <-- imbalance" if abs(sd) > 0.25 else ""
        print(f"  {v:10s} tr={mt:10.2f}  co={mc:10.2f}  std.diff={sd:+.2f}{flag}")

    print("\n-- Trimmed sample size reductions (GRF propensity overlap) --")
    print(f"  LDW-CPS: full n={len(lal['ldw_cps'])} -> trimmed n={len(trim['ldw_cps_trim'])} "
          f"({len(trim['ldw_cps_trim'])/len(lal['ldw_cps']):.1%})")
    print(f"  LDW-PSID: full n={len(lal['ldw_psid'])} -> trimmed n={len(trim['ldw_psid_trim'])} "
          f"({len(trim['ldw_psid_trim'])/len(lal['ldw_psid']):.1%})")

    print("\n" + "=" * 70)
    print("2. IRS lottery — coverage and consistency")
    print("=" * 70)
    print(f"n = {len(irs)}, winners={int(irs['winner'].sum())}, "
          f"big={int(irs['bigwinner'].sum())}")

    # winner/bigwinner cross-tab
    print("\nwinner x bigwinner table:")
    print(pd.crosstab(irs["winner"], irs["bigwinner"]))

    cols_check = COVAR_IRS + ["yearn.avg", "xearn.avg"]
    summary(irs, "IRS", cols_check)

    # Pre/post earnings should be non-negative
    for v in cols_check:
        if "earn" in v:
            neg = (irs[v] < 0).sum()
            print(f"  {v}: negative values = {neg}")

    # Panel: years 1-7 should have same N
    ns = [irs[f"yearn.{k}"].notna().sum() for k in range(1, 8)]
    print(f"\npost-earnings N per year: {ns}")

    print("\nData audit complete.")


if __name__ == "__main__":
    main()
