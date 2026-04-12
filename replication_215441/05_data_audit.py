"""Phase 3: data audit on clean_review_estimates_long.csv and bma_input.csv."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from utils import load_long, collapse_one_per_margin, build_bma_covariates, BMA_COVARS, CLEAN

df = load_long()
print("=" * 70)
print("DATA AUDIT")
print("=" * 70)

# ---- 1. Coverage ----
print("\n[1] Coverage")
print(f"  rows:                {len(df)}")
print(f"  unique papers:       {df['paper_id'].nunique()}")
print(f"  unique reforms:      {df['reform_id'].nunique()}")
print(f"  unique countries:    {df['sample_country'].nunique()}")
print(f"  year range:          {df['year'].min():.0f}–{df['year'].max():.0f}")
print(f"  sample_year range:   {df['sample_year'].min():.0f}–{df['sample_year'].max():.0f}")
print(f"  include=='Yes':      {(df['include']=='Yes').sum()}")

# ---- 2. Key variable distributions ----
print("\n[2] Distributions of analysis variables")
for col in ["elasticity", "se", "tstat", "mean_pbd", "mean_rr",
            "country_tax_wedge", "ue_deviation", "impact_factor_z"]:
    s = pd.to_numeric(df[col], errors="coerce")
    print(f"  {col:22s} n={s.notna().sum():3d}  "
          f"mean={s.mean():7.3f}  sd={s.std():7.3f}  "
          f"min={s.min():7.3f}  max={s.max():7.3f}")

# ---- 3. Outliers by IQR ----
print("\n[3] Elasticity outliers (outside 1.5*IQR)")
e = df["elasticity"]
q1, q3 = e.quantile(0.25), e.quantile(0.75)
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
flagged = df[(e < lo) | (e > hi)][["authors", "pbd_vs_rr", "elasticity", "se"]]
print(f"  bounds: [{lo:.3f}, {hi:.3f}]")
print(flagged.to_string(index=False))

# ---- 4. Top-10 largest SEs (AK gives them low weight) ----
print("\n[4] 10 largest standard errors (noisiest estimates)")
top_se = df.nlargest(10, "se")[["authors", "pbd_vs_rr", "elasticity", "se", "tstat"]]
print(top_se.to_string(index=False))

# ---- 5. Logical consistency ----
print("\n[5] Logical consistency checks")
print(f"  rows with se <= 0:                           {(df['se'] <= 0).sum()}")
print(f"  rows with |tstat| vs elasticity/se mismatch: "
      f"{((df['tstat'] - df['elasticity']/df['se']).abs() > 1e-3).sum()}")
print(f"  rows with PBD_indicator!={{0,1}}:              "
      f"{(~df['PBD_indicator'].isin([0,1])).sum()}")
print(f"  rows with pbd_vs_rr/ PBD_indicator mismatch: "
      f"{(((df['pbd_vs_rr']=='PBD').astype(int)) != df['PBD_indicator']).sum()}")
print(f"  rows with mean_rr outside [0,1]:             "
      f"{(~df['mean_rr'].between(0, 1)).sum()}")

# ---- 6. Duplicate structure ----
print("\n[6] Duplicates and collapsing")
dupes = df.groupby(["paper_id", "pbd_vs_rr", "ue_measure"]).size()
print(f"  groups:                 {len(dupes)}")
print(f"  groups with 1 row:      {(dupes == 1).sum()}")
print(f"  groups with 2 rows:     {(dupes == 2).sum()}")
print(f"  groups with >2 rows:    {(dupes > 2).sum()}")
print(f"  max group size:         {dupes.max()}")
print(f"  rows after collapsing (no Hunt):   {len(collapse_one_per_margin(load_long(drop_hunt=True)))}")

# ---- 7. Hunt outlier impact ----
print("\n[7] Hunt (1995) outlier")
hunt = df[df["authors"].str.contains("Hunt", na=False)]
print(hunt[["authors", "pbd_vs_rr", "elasticity", "se", "tstat"]].to_string(index=False))
rr = df[df["pbd_vs_rr"] == "RR"]
print(f"  RR mean with Hunt:     {rr['elasticity'].mean():.4f}")
print(f"  RR mean without Hunt:  {rr[rr['elasticity'] >= -2]['elasticity'].mean():.4f}")

# ---- 8. Missingness in BMA design ----
print("\n[8] Missingness by BMA covariate (after collapse)")
coll = collapse_one_per_margin(load_long(drop_hunt=True))
renamed = build_bma_covariates.__globals__["build_bma_covariates"]
raw_coll = collapse_one_per_margin(load_long(drop_hunt=True)).copy()
raw_coll["YearsTo2023"] = 2023 - raw_coll["year"]
raw_coll["RR"] = (raw_coll["pbd_vs_rr"] != "PBD").astype(int)
raw_coll["NonemploymentAsOutcome"] = pd.to_numeric(raw_coll["ue_measure_nonemp"], errors="coerce")
raw_coll = raw_coll.rename(columns={
    "se": "SE", "PBD_indicator": "PBD", "macro_treatment": "MacroTreatment",
    "design_RDD_RKD": "RDD", "admin": "Admin", "hazard": "HazardModel",
    "us_country": "USA", "country_tax_wedge": "TaxWedge",
    "ue_deviation": "RelativeUnemp", "impact_factor_z": "ImpactFactorZ",
    "mean_pbd": "BaselinePBD", "mean_rr": "BaselineRR",
})
for c in ["SE", "PBD", "MacroTreatment", "Admin", "NonemploymentAsOutcome",
          "HazardModel", "YearsTo2023", "RelativeUnemp", "USA", "TaxWedge",
          "ImpactFactorZ", "BaselinePBD", "BaselineRR"]:
    if c in raw_coll.columns:
        print(f"  {c:22s} missing = {raw_coll[c].isna().sum()}")
print(f"  BMA rows after dropna: {len(build_bma_covariates(coll))}")

# ---- 9. Country coverage ----
print("\n[9] Country breakdown (top 10)")
print(df["sample_country"].value_counts().head(10).to_string())

# ---- 10. Research design ----
print("\n[10] Research design breakdown")
print(df["research_design"].value_counts().to_string())
print(f"\n  design_DID:       {df['design_DID'].sum()}")
print(f"  design_RDD_RKD:   {df['design_RDD_RKD'].sum()}")
