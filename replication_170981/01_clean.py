"""Load data_final.dta, apply post-save transformations, and print sample checks."""
import pandas as pd
from utils import build_analysis_frame

df = build_analysis_frame()
print(f"Neighborhoods (180 expected): {len(df)}")

print("\nTable 1 sample (non-missing d1204_r):")
non_missing = df["d1204_r"].notna().sum()
print(f"  d1204_r non-missing: {non_missing}")
print(f"  Non-missing with full covariates:")
cols = [
    "d1204_r",
    "prezzo2004Abitaz",
    "prezzo2004Negozi",
    "metro_dum",
    "univ",
    "attraction_dum",
    "day_pop",
    "lat2004",
    "prezzo2004",
    "cucina2004",
    "mmichelin2004",
]
print(f"  N={df[cols].dropna().shape[0]} (paper reports 140)")

print("\nSummary of growth variable:")
print(df["d1204_r"].describe())

df.to_pickle("replication_170981/clean.pkl")
print("\nSaved cleaned dataframe to replication_170981/clean.pkl")
