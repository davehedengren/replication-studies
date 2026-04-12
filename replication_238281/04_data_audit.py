"""Data audit: sanity checks on the constructed panel."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PANEL = Path(__file__).resolve().parent / "panel.parquet"
panel = pd.read_parquet(PANEL)

print("=" * 60)
print("REPLICATION 238281 — DATA AUDIT")
print("=" * 60)

# 1. shape
print(f"\n[1] Panel shape: {panel.shape}")
print(f"    Paper Table 1 N: 239,173 — match: {panel.shape[0] == 239173}")

# 2. unique villages
nv = panel["village"].nunique()
print(f"\n[2] Unique villages: {nv} (paper: 99 used after attrition)")
# (We get 98; one of the 99 listed is one of the 7 dropped no-payout villages.)

# 3. months covered
print(f"\n[3] Month indices: {sorted(panel['month_idx'].unique())}")
print(f"    => calendar range: 2022-{panel['month_idx'].min()-12} thru "
      f"2023-{panel['month_idx'].max()-12}")

# 4. treatment cell sizes
print("\n[4] Treatment cell sizes (HH-month obs):")
print(panel["treatment"].value_counts().to_string())

# 5. extreme values
print("\n[5] tot_consumption summary:")
print(panel["tot_consumption"].describe().to_string())
print(f"    rows with > 1000 L in a month: {(panel['tot_consumption'] > 1000).sum()}")
print(f"    rows with negative litres   : {(panel['tot_consumption'] < 0).sum()}")

# 6. sub-treatment shares within Discount / Exchangeable
for trt in ["Discount", "Exchangeable"]:
    sub = panel.loc[panel["treatment"] == trt, "subtreatment"].value_counts()
    print(f"\n[6] {trt} sub-treatment cells: {sub.to_dict()}")

# 7. control mean (replicates paper Table 1 footer)
cm_b = panel.loc[panel["Control"] == 1, "bought"].mean()
cm_l = panel.loc[panel["Control"] == 1, "tot_consumption"].mean()
print(f"\n[7] Control means: any={cm_b:.4f}  litres={cm_l:.4f}")
print(f"    Paper:         any=0.012   litres=2.818")

# 8. unique (HH x month) — should equal panel rows
hhm = panel[["village", "only_digits_scard_id", "month_idx"]].drop_duplicates().shape[0]
print(f"\n[8] Unique HH-village-month rows: {hhm} (panel rows: {panel.shape[0]})")

# 9. balance: average HH per village
hh_per_v = panel.groupby("village")["only_digits_scard_id"].nunique()
print(f"\n[9] HH per village — mean: {hh_per_v.mean():.1f}, "
      f"median: {hh_per_v.median()}, range: [{hh_per_v.min()}, {hh_per_v.max()}]")

# 10. months per HH
mo_per_hh = panel.groupby(["village", "only_digits_scard_id"])["month_idx"].nunique()
print(f"\n[10] Months observed per HH — mean: {mo_per_hh.mean():.2f}, "
      f"median: {mo_per_hh.median()}, range: [{mo_per_hh.min()}, {mo_per_hh.max()}]")

# 11. share of HH never ordering
ever_bought = panel.groupby(["village", "only_digits_scard_id"])["bought"].max()
print(f"\n[11] Share of HH ever ordering: {ever_bought.mean():.3f}")
print(f"     By group:")
trt_by_hh = panel.groupby(["village", "only_digits_scard_id"])["treatment"].first()
for grp, grp_hh in trt_by_hh.groupby(trt_by_hh):
    sub = ever_bought.loc[grp_hh.index]
    print(f"       {grp:15s}  N={len(sub):6d}  ever ordered: {sub.mean():.3f}")
