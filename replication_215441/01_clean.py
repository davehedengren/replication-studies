"""Phase 2: sample construction sanity checks.

Verifies we can reproduce the same observation counts the paper reports
from the cleaned long-format file before touching the estimators.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from utils import load_long, collapse_one_per_margin, build_bma_covariates

df = load_long()
print(f"Long-format rows (full):               {len(df)}")
print(f"Unique papers:                         {df['paper_id'].nunique()}  (published 57)")
print(f"Unique (paper, margin, ue_measure):    {df.groupby(['paper_id','pbd_vs_rr','ue_measure']).ngroups}")

print("\nBy margin (all rows, including Hunt outlier):")
print(df["pbd_vs_rr"].value_counts())

# For AK parametric (Table 1 descriptives), the R script uses the FULL long file
# split by margin — Hunt is kept for the baseline RR column.
rr = df[df["pbd_vs_rr"] == "RR"]
pbd = df[df["pbd_vs_rr"] == "PBD"]
print(f"\nRR rows used in AK baseline:           {len(rr)}  (published 42)")
print(f"PBD rows used in AK baseline:          {len(pbd)}  (published 49)")

# Mean elasticities — Table 1 header and stats.csv lines 27-28
print(f"\nMean RR elasticity:   {rr['elasticity'].mean():.5f}  (published 0.43135)")
print(f"Mean PBD elasticity:  {pbd['elasticity'].mean():.5f}  (published 0.46239)")

# BMA script drops Hunt outlier before collapsing
df_hunt = load_long(drop_hunt=True)
coll = collapse_one_per_margin(df_hunt)
print(f"\nOne-per-margin rows (Hunt dropped):    {len(coll)}  (expected {len(df_hunt) - 19})")

bma = build_bma_covariates(coll)
print(f"BMA design matrix (drop_na):           {bma.shape}")
