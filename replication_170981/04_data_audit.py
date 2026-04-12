"""Data audit for Leonardi & Moretti (2022) — neighborhood panel."""
import numpy as np
import pandas as pd
from utils import load_raw_neighborhood, build_analysis_frame

raw = load_raw_neighborhood()
df = build_analysis_frame()

print("=" * 70)
print("COVERAGE")
print("=" * 70)
print(f"Rows (neighborhoods): {len(df)}")
print(f"Unique zona180: {df['zona180'].nunique()}")
print(f"Duplicate zone IDs: {df['zona180'].duplicated().sum()}")
years = [2000, 2002, 2004, 2006, 2008, 2010, 2012]
print(f"Years in restaurant count panel: {years}")

print("\n" + "=" * 70)
print("MISSINGNESS (key variables)")
print("=" * 70)
key_vars = [
    "lat2000", "lat2004", "lat2012",
    "lat2000_d", "lat2004_d", "lat2012_d",
    "lat2000_a", "lat2004_a", "lat2012_a",
    "day_pop", "prezzo2004Abitaz", "prezzo2004Negozi",
    "prezzo2004", "cucina2004", "mmichelin2004",
    "metro_dum", "univ", "attraction_dum", "d1204_r",
]
miss = pd.DataFrame({
    "n_missing": df[key_vars].isna().sum(),
    "pct_missing": df[key_vars].isna().mean() * 100,
})
print(miss.round(1))

print("\n" + "=" * 70)
print("RAW ESTABLISHMENT COUNTS (pre-per-capita)")
print("=" * 70)
raw_counts = pd.DataFrame({
    "year": years,
    "restaurants_sum": [raw[f"lat{y}"].sum() for y in years],
    "retail_sum": [raw[f"lat{y}_d"].sum() for y in years],
    "food_retail_sum": [raw[f"lat{y}_a"].sum() for y in years],
})
print(raw_counts.to_string(index=False))
# These should match the hard-coded weights in replication_submit.do (lines 116-126)
# rest: 5887 (2000), 5995 (2002), 6057 (2004), 6117 (2006), 6178 (2008), 6628 (2010), 6931 (2012)
expected_rest = {2000: 5887, 2002: 5995, 2004: 6057, 2006: 6117, 2008: 6178, 2010: 6628, 2012: 6931}
print("\nExpected (from do-file comments):", expected_rest)
mismatches = [y for y in years if int(raw[f"lat{y}"].sum()) != expected_rest[y]]
print(f"Mismatches vs do-file weights: {mismatches}")

print("\n" + "=" * 70)
print("DISTRIBUTION CHECKS (post-transformation per-capita counts)")
print("=" * 70)
print(df[["lat2000", "lat2004", "lat2012", "day_pop"]].describe().round(2))

print("\n" + "=" * 70)
print("OUTLIER SCAN (top-5 extreme values)")
print("=" * 70)
for v in ["d1204_r", "d1204_d", "d1204_a", "day_pop"]:
    tops = df.nlargest(5, v)[["zona180", v]]
    bots = df.nsmallest(5, v)[["zona180", v]]
    print(f"\n{v}:")
    print("  top-5    :", list(zip(tops.zona180.astype(int), tops[v].round(3))))
    print("  bottom-5 :", list(zip(bots.zona180.astype(int), bots[v].round(3))))

print("\n" + "=" * 70)
print("LOGICAL CONSISTENCY")
print("=" * 70)
# After per-capita scaling, do counts stay non-negative?
neg_counts = (df[[f"lat{y}" for y in years]] < 0).sum().sum()
print(f"Negative per-capita restaurant counts: {neg_counts}")
# d1204_r = log(lat2012) - log(lat2004). Should match reconstruction:
recon = np.log(df["lat2012"]) - np.log(df["lat2004"])
max_diff = (recon - df["d1204_r"]).abs().max()
print(f"Max |d1204_r - log(2012)+log(2004)|: {max_diff:.2e}")

# Restaurants that went to zero (infinite log growth)?
inf_2012 = np.isinf(np.log(df["lat2012"])).sum()
inf_2004 = np.isinf(np.log(df["lat2004"])).sum()
print(f"Zones with lat2004 == 0: {(df['lat2004'] == 0).sum()}")
print(f"Zones with lat2012 == 0: {(df['lat2012'] == 0).sum()}")
print(f"Zones with d1204_r == -inf: {np.isinf(df['d1204_r']).sum()}")

print("\n" + "=" * 70)
print("SHARE VARIABLES IN [0,1]?")
print("=" * 70)
for v in ["mmichelin2004", "methnic2004", "msitdown2004"]:
    if v in df.columns:
        s = df[v].dropna()
        print(f"  {v}: min={s.min():.3f} max={s.max():.3f} mean={s.mean():.3f}")

print("\n" + "=" * 70)
print("PANEL BALANCE (years with any missing count per zone)")
print("=" * 70)
for prefix in ["lat", "lat", "lat"]:
    pass
for suffix, label in [("", "restaurants"), ("_d", "retail"), ("_a", "food_retail")]:
    cols = [f"lat{y}{suffix}" for y in years]
    any_miss = df[cols].isna().any(axis=1).sum()
    print(f"  {label}: zones with any missing-year: {any_miss}/180")
