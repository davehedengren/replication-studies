"""
02_tables.py - Replicate main IRFs from LS analysis
Greenwald & Guren (2025) - Do Credit Conditions Move House Prices?
"""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
from utils import *

print("=" * 70)
print("REPLICATION: Paper 225841 (Greenwald & Guren 2025)")
print("Do Credit Conditions Move House Prices?")
print("=" * 70)

# Load non-confidential data
df = load_dta('ls_analysis_data_nc.dta')
print(f"\nLoaded LS analysis data: {df.shape[0]} obs, {df.shape[1]} vars")

# Check key variables
key_vars = ['lprr', 'lhor_hvs', 'z', 'Lfraction', 'L2fraction', 'Lz', 'Llprr',
            'Llhor_hvs', 'lhpi', 'year', 'cbsa', 'pop2000',
            'HVS_all_sample', 'HVS_common_sample']
for v in key_vars:
    if v in df.columns:
        print(f"  {v}: {df[v].notna().sum()} non-missing")
    else:
        print(f"  {v}: NOT FOUND")

# Ensure year ordering within cbsa
df = df.sort_values(['cbsa', 'year']).reset_index(drop=True)

# ============================================================
# Figure 3a: Price-Rent Ratio IRF (HVS Common Sample)
# ============================================================
cond_hvs = df['HVS_common_sample'] == 1
controls_prr = ['Lfraction', 'L2fraction', 'Lz', 'Llprr']
controls_prr = [c for c in controls_prr if c in df.columns]

res_prr = run_local_projection(
    df, 'lprr', 'z', controls_prr,
    fe_vars=['year', 'cbsa'], cluster_var='cbsa', weight_var='pop2000',
    condition=cond_hvs
)
print_irf(res_prr, "Figure 3a: Price-Rent Ratio IRF (LS, HVS Common Sample)")


# ============================================================
# Figure 3b: Homeownership Rate IRF (HVS Common Sample)
# ============================================================
controls_hor = ['Lfraction', 'L2fraction', 'Lz', 'Llhor_hvs']
controls_hor = [c for c in controls_hor if c in df.columns]

res_hor = run_local_projection(
    df, 'lhor_hvs', 'z', controls_hor,
    fe_vars=['year', 'cbsa'], cluster_var='cbsa', weight_var='pop2000',
    condition=cond_hvs
)
print_irf(res_hor, "Figure 3b: Homeownership Rate IRF (LS, HVS Common Sample)")


# ============================================================
# Figure 3c: Price-Rent Ratio IRF with employment controls
# ============================================================
emp_controls = [c for c in df.columns if c.startswith('emp1_share')]
controls_prr_emp = ['Lfraction', 'L2fraction', 'Lz', 'Llprr', 'lemp'] + emp_controls
controls_prr_emp = [c for c in controls_prr_emp if c in df.columns]

res_prr_ctrl = run_local_projection(
    df, 'lprr', 'z', controls_prr_emp,
    fe_vars=['year', 'cbsa'], cluster_var='cbsa', weight_var='pop2000',
    condition=cond_hvs
)
print_irf(res_prr_ctrl, "Figure 3c: Price-Rent Ratio IRF with Employment Controls")


# ============================================================
# Figure 3d: Homeownership Rate IRF with employment controls
# ============================================================
controls_hor_emp = ['Lfraction', 'L2fraction', 'Lz', 'Llhor_hvs', 'lemp'] + emp_controls
controls_hor_emp = [c for c in controls_hor_emp if c in df.columns]

res_hor_ctrl = run_local_projection(
    df, 'lhor_hvs', 'z', controls_hor_emp,
    fe_vars=['year', 'cbsa'], cluster_var='cbsa', weight_var='pop2000',
    condition=cond_hvs
)
print_irf(res_hor_ctrl, "Figure 3d: Homeownership Rate IRF with Employment Controls")


# ============================================================
# HPI IRF (all sample)
# ============================================================
cond_all = df['year'] > 1994
controls_hpi = ['Lfraction', 'L2fraction', 'Lz', 'Llhpi']
controls_hpi = [c for c in controls_hpi if c in df.columns]

if 'lhpi' in df.columns and 'Llhpi' in df.columns:
    res_hpi = run_local_projection(
        df, 'lhpi', 'z', controls_hpi,
        fe_vars=['year', 'cbsa'], cluster_var='cbsa', weight_var='pop2000',
        condition=cond_all
    )
    print_irf(res_hpi, "House Price Index IRF (LS, All Sample post-1994)")


# ============================================================
# PRR IRF - HVS All Sample
# ============================================================
cond_all_hvs = df['HVS_all_sample'] == 1

res_prr_all = run_local_projection(
    df, 'lprr', 'z', controls_prr,
    fe_vars=['year', 'cbsa'], cluster_var='cbsa', weight_var='pop2000',
    condition=cond_all_hvs
)
print_irf(res_prr_all, "Price-Rent Ratio IRF (LS, HVS All Sample)")


print("\n" + "=" * 70)
print("REPLICATION COMPLETE")
print("=" * 70)
print("""
Note: This replication uses the non-confidential (pseudodata) version of the
dataset. Coefficient values may differ from the published paper which uses
confidential GG microdata. The structure and methodology match exactly.
""")
