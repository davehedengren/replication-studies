"""
04_data_audit.py — Comprehensive data audit for both datasets.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import *

def audit_section(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")

# ─────────────────────────────────────────────────────────────────────
# STATE DATA
# ─────────────────────────────────────────────────────────────────────
audit_section("STATE DATA AUDIT")
sdf = pd.read_stata(STATE_DTA, convert_categoricals=False)
sdf["q_fld"] = q_fld_to_int(sdf["q_fld"])

# Coverage
audit_section("1. Coverage")
print(f"Observations: {len(sdf)}")
print(f"States: {sorted(sdf['state'].unique())} (n={sdf['state'].nunique()})")
print(f"Quarter range: {sdf['q_fld'].min()} to {sdf['q_fld'].max()}")
n_quarters = sdf['q_fld'].nunique()
print(f"Unique quarters: {n_quarters}")
print(f"Expected: 9 states x {n_quarters} quarters = {9 * n_quarters}")
print(f"Actual: {len(sdf)}")
print(f"Balanced panel: {len(sdf) == 9 * n_quarters}")

# Variable completeness
audit_section("2. Variable Completeness")
for col in sdf.columns:
    n_miss = sdf[col].isna().sum()
    if n_miss > 0:
        print(f"  {col}: {n_miss} missing ({100*n_miss/len(sdf):.1f}%)")
    else:
        print(f"  {col}: complete")

# Distributions
audit_section("3. Distributions & Plausibility")
for var in ["allapps", "DIonly", "SSItotal", "SSDItotal", "ue", "wapop", "unin"]:
    if var in sdf.columns:
        s = sdf[var]
        print(f"\n  {var}:")
        print(f"    mean={s.mean():.4f}, sd={s.std():.4f}, min={s.min():.4f}, max={s.max():.4f}")
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        outliers = ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()
        print(f"    Q1={q1:.4f}, Q3={q3:.4f}, IQR={iqr:.4f}, outliers={outliers}")
        if var in ["allapps", "DIonly", "SSItotal", "SSDItotal"]:
            neg = (s < 0).sum()
            print(f"    Negative values: {neg}")

# Logical consistency
audit_section("4. Logical Consistency")
# allapps should ≈ DIonly + SSItotal (since SSDI total includes concurrent)
sdf["check_sum"] = sdf["DIonly"] + sdf["SSItotal"]
diff = (sdf["allapps"] - sdf["check_sum"]).abs()
print(f"allapps vs DIonly + SSItotal:")
print(f"  Max absolute diff: {diff.max():.6f}")
print(f"  Mean absolute diff: {diff.mean():.6f}")
# SSDItotal should be >= DIonly (SSDI total = DI only + concurrent)
violations = (sdf["SSDItotal"] < sdf["DIonly"]).sum()
print(f"SSDItotal < DIonly violations: {violations}")

# Panel balance
audit_section("5. Panel Balance")
panel_counts = sdf.groupby("state")["q_fld"].count()
print("Obs per state:")
print(panel_counts.to_string())
imbalanced = panel_counts[panel_counts != panel_counts.max()]
if len(imbalanced) > 0:
    print(f"Imbalanced states: {list(imbalanced.index)}")
else:
    print("All states have equal observations — balanced panel.")

# Duplicates
audit_section("6. Duplicates")
dupes = sdf.duplicated(subset=["state", "q_fld"]).sum()
print(f"Duplicate state-quarter pairs: {dupes}")

# ─────────────────────────────────────────────────────────────────────
# COUNTY DATA
# ─────────────────────────────────────────────────────────────────────
audit_section("COUNTY DATA AUDIT")
cdf = pd.read_stata(COUNTY_DTA, convert_categoricals=False)
cdf["q_fld"] = q_fld_to_int(cdf["q_fld"])

audit_section("1. Coverage")
print(f"Observations (raw): {len(cdf)}")
print(f"States: {sorted(cdf['state'].unique())} (n={cdf['state'].nunique()})")
print(f"Quarter range: {cdf['q_fld'].min()} to {cdf['q_fld'].max()}")
n_q = cdf['q_fld'].nunique()
n_c = cdf['county'].nunique()
print(f"Unique quarters: {n_q}")
print(f"Unique counties: {n_c}")
print(f"Expected if balanced: {n_c * n_q}")
print(f"Actual: {len(cdf)}")
print(f"Balanced: {len(cdf) == n_c * n_q}")

# MA counties
ma_counties = cdf.loc[cdf["state"] == "MA", "county"].unique()
print(f"\nMA counties: {len(ma_counties)}")
for s in sorted(cdf["state"].unique()):
    nc = cdf.loc[cdf["state"] == s, "county"].nunique()
    print(f"  {s}: {nc} counties")

# Variable completeness
audit_section("2. Variable Completeness")
for col in cdf.columns:
    n_miss = cdf[col].isna().sum()
    if n_miss > 0:
        print(f"  {col}: {n_miss} missing ({100*n_miss/len(cdf):.1f}%)")

# Suppressed cells check
audit_section("3. Suppressed Cells")
cdf["anymissing"] = ((cdf["DIonly"] == 0) | (cdf["SSIonly"] == 0) | (cdf["concurrent"] == 0)).astype(int)
n_affected = cdf["anymissing"].sum()
print(f"Obs with any zero in DIonly/SSIonly/concurrent: {n_affected}")
problem_counties = cdf.loc[cdf["anymissing"] == 1, "county"].unique()
print(f"Counties affected: {len(problem_counties)}")

# After dropping
cdf_clean = cdf[~cdf["county"].isin(DROP_COUNTIES)].copy()
print(f"\nAfter dropping {len(DROP_COUNTIES)} counties:")
print(f"  Remaining obs: {len(cdf_clean)}")
print(f"  Remaining counties: {cdf_clean['county'].nunique()}")

# Distributions (clean data)
audit_section("4. Distributions (after drops)")
for var in ["allapps", "DIonly", "SSItotal", "SSDItotal", "ue", "wapop", "nohi05"]:
    if var in cdf_clean.columns:
        s = cdf_clean[var].dropna()
        print(f"\n  {var}:")
        print(f"    mean={s.mean():.4f}, sd={s.std():.4f}, min={s.min():.4f}, max={s.max():.4f}")
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        outliers = ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()
        print(f"    Q1={q1:.4f}, Q3={q3:.4f}, IQR={iqr:.4f}, outliers={outliers}")

# Logical consistency
audit_section("5. Logical Consistency (county)")
cdf_clean["check_sum"] = cdf_clean["DIonly"] + cdf_clean["SSItotal"]
diff = (cdf_clean["allapps"] - cdf_clean["check_sum"]).abs()
print(f"allapps vs DIonly + SSItotal:")
print(f"  Max absolute diff: {diff.max():.6f}")
print(f"  Mean absolute diff: {diff.mean():.6f}")
# SSIonly + concurrent should = SSItotal
cdf_clean["ssi_check"] = cdf_clean["SSIonly"] + cdf_clean["concurrent"]
ssi_diff = (cdf_clean["SSItotal"] - cdf_clean["ssi_check"]).abs()
print(f"SSItotal vs SSIonly + concurrent:")
print(f"  Max absolute diff: {ssi_diff.max():.6f}")

# Missing data patterns
audit_section("6. Missing Data by Treatment Status")
cdf_clean["MA"] = (cdf_clean["state"] == "MA").astype(int)
for var in ["allapps", "DIonly", "SSItotal", "SSDItotal", "ue", "nohi05"]:
    if var in cdf_clean.columns:
        for ma in [1, 0]:
            sub = cdf_clean[cdf_clean["MA"] == ma]
            n_miss = sub[var].isna().sum()
            label = "MA" if ma else "Other"
            print(f"  {var} missing [{label}]: {n_miss}/{len(sub)}")

# Time-to-filing variables
audit_section("7. Time-to-Filing Variables")
time_cols = [c for c in cdf_clean.columns if "mntime" in c.lower() or "mtime" in c.lower()]
print(f"Time columns found: {time_cols}")
for var in time_cols:
    s = cdf_clean[var].dropna()
    n_miss = cdf_clean[var].isna().sum()
    print(f"\n  {var}:")
    print(f"    N non-missing: {len(s)}, missing: {n_miss}")
    if len(s) > 0:
        print(f"    mean={s.mean():.2f}, sd={s.std():.2f}, min={s.min():.2f}, max={s.max():.2f}")

# Panel balance
audit_section("8. Panel Balance (county)")
obs_per_county = cdf_clean.groupby("county")["q_fld"].count()
print(f"Obs per county: min={obs_per_county.min()}, max={obs_per_county.max()}, "
      f"median={obs_per_county.median()}")
if obs_per_county.min() != obs_per_county.max():
    short_counties = obs_per_county[obs_per_county < obs_per_county.max()]
    print(f"Counties with fewer than max obs: {len(short_counties)}")

# Duplicates
audit_section("9. Duplicates")
dupes = cdf_clean.duplicated(subset=["county", "q_fld"]).sum()
print(f"Duplicate county-quarter pairs: {dupes}")

# nohi05 distribution
audit_section("10. Uninsurance Rate Distribution (nohi05)")
s = cdf_clean.drop_duplicates("county")["nohi05"]
print(f"  N counties: {len(s)}")
print(f"  mean={s.mean():.4f}, sd={s.std():.4f}")
print(f"  min={s.min():.4f}, max={s.max():.4f}")
print(f"  Threshold for lowHI: 0.12")
print(f"  Counties with nohi05 >= 0.12: {(s >= 0.12).sum()}")
print(f"  Counties with nohi05 < 0.12: {(s < 0.12).sum()}")

print("\n\nData audit complete.")
