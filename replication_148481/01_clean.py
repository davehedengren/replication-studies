"""Clean Experiment 1 data from the raw CSV.  Replicates Experiment1RepKit/1_data_clean.do.

Target: N=446 workers, 4460 worker-round observations."""

import pandas as pd
from utils import clean_wide, clean_long, OUT

wide = clean_wide()
long, num_ind = clean_long()

print("=" * 60)
print("PHASE 2 — DATA CLEANING")
print("=" * 60)
print(f"Final wide-form N (workers)           : {len(wide)}")
print(f"Final long-form N (worker-rounds)     : {len(long)}")
print(f"Unique workers in long                : {long['id'].nunique()}")
print(f"Gift-type distribution (wide):\n{wide['gift_type'].value_counts().sort_index()}")
print(f"Order distribution (wide)             : {dict(wide['order'].value_counts().sort_index())}")
print(f"Phase distribution                    : {dict(wide['phase'].value_counts().sort_index())}")

wide.to_parquet(OUT / "wide.parquet", index=False)
long.to_parquet(OUT / "long.parquet", index=False)
print(f"\nWrote {OUT/'wide.parquet'} and {OUT/'long.parquet'}")
