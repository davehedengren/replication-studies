"""Data audit for the Table 1 analysis panel."""
import numpy as np
import pandas as pd

from utils import OUT, build_regression_panel, load_merged_panel

PANEL = build_regression_panel()
RAW = load_merged_panel()

print("=" * 70)
print("COVERAGE")
print("=" * 70)
print(f"Universe (merged HERD+IPEDS, sector in {{1,2}}): {len(RAW):,} uni-year rows")
print(f"Unique uni_IDs in universe                    : {RAW.uni_ID.nunique():,}")
print(f"Year range                                    : {RAW.year.min():.0f}-{RAW.year.max():.0f}")

print(f"\nFinal Table 1 panel (unis in 2015 w/ yy,xx,zz) : {len(PANEL):,}")
print(f"  Public                                       : {(PANEL.sector==1).sum()}")
print(f"  Private non-profit                           : {(PANEL.sector==2).sum()}")
print(f"  Liberal arts colleges                        : {PANEL.flag_lib_art_col.sum()}")

print("\n" + "=" * 70)
print("DISTRIBUTIONS OF KEY REGRESSION VARIABLES")
print("=" * 70)
for c in ["yy", "xx", "zz", "placebo_xx", "fte_wts",
          "init_uni_size", "init_fac_stud_ratio", "init_life_sci_uni"]:
    s = PANEL[c].dropna()
    print(f"{c:25s} n={len(s):5d} mean={s.mean():12.3f} sd={s.std():12.3f} "
          f"min={s.min():12.3f} p50={s.median():12.3f} max={s.max():12.3f}")

print("\n" + "=" * 70)
print("TOP OUTLIERS IN yy (change in net tuition per FTE, USD)")
print("=" * 70)
top = PANEL.nlargest(10, "yy")[["uni_ID", "sector", "flag_lib_art_col", "yy", "xx"]]
print(top.to_string(index=False))
print("\nBOTTOM:")
bot = PANEL.nsmallest(10, "yy")[["uni_ID", "sector", "flag_lib_art_col", "yy", "xx"]]
print(bot.to_string(index=False))

print("\n" + "=" * 70)
print("LOGICAL CONSISTENCY")
print("=" * 70)
print(f"zz (instrument shares) sum to: {PANEL['zz'].sum():.6f}  (should be 1.0)")
print(f"Share of unis with zz > 0 (had lifesci fed R&D in pre-period): "
      f"{(PANEL['zz'] > 0).mean():.3f}")
print(f"yy has negatives (tuition fell): {(PANEL['yy'] < 0).sum()} unis")
print(f"xx has negatives (research fell): {(PANEL['xx'] < 0).sum()} unis")
print(f"Negative fte_wts                 : {(PANEL['fte_wts'] < 0).sum()}")

print("\n" + "=" * 70)
print("MISSING DATA BY PERIOD (Dt)")
print("=" * 70)
df = RAW.copy()
from utils import assign_Dt
df["Dt"] = assign_Dt(df["year"])
df = df.dropna(subset=["Dt"]).copy()
df["Dt"] = df["Dt"].astype(int)
for v in ["nettuition01", "research01", "RD_LifeSci_Federal",
          "ft_faculty_per_100fte", "studserv01", "fte_count"]:
    miss = df.groupby("Dt")[v].apply(lambda s: s.isna().mean())
    print(f"{v:25s}  by Dt: {dict(miss.round(3))}")

print("\n" + "=" * 70)
print("PANEL BALANCE (Dt coverage per uni)")
print("=" * 70)
# After the collapse, how many unis have all 4 periods?
Dt_presence = (
    df.dropna(subset=["fte_count"])
      .groupby(["uni_ID", "Dt"])
      .size().unstack(fill_value=0) > 0
)
print(f"Unis with all 4 Dt bins (0,1,2,3): {Dt_presence.all(axis=1).sum()}")
print(f"Unis with Dt 0 and Dt 1 (yy/xx feasible): "
      f"{(Dt_presence[[0, 1]].all(axis=1)).sum()}")
print(f"Unis with Dt 3 (pre-period controls feasible): "
      f"{Dt_presence[3].sum()}")

print("\n" + "=" * 70)
print("DUPLICATES")
print("=" * 70)
dupes = PANEL.duplicated(subset=["uni_ID"]).sum()
print(f"Duplicate uni_IDs in final panel: {dupes}")

print("\n" + "=" * 70)
print("LIBERAL-ARTS WEIGHT VS SAMPLE SHARE")
print("=" * 70)
g = PANEL.groupby("flag_lib_art_col")["fte_wts"].agg(["count", "sum", "mean"])
g["share_of_n"] = g["count"] / g["count"].sum()
g["share_of_fte"] = g["sum"] / g["sum"].sum()
print(g.round(4))

# Save a summary parquet
PANEL.describe().to_csv(OUT / "audit_describe.csv")
print(f"\nSaved describe() to {OUT / 'audit_describe.csv'}")
