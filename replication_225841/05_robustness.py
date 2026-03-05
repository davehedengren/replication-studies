"""
05_robustness.py - Robustness checks for paper 225841
Greenwald & Guren (2025) - Do Credit Conditions Move House Prices?
"""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
from utils import *

print("=" * 70)
print("ROBUSTNESS CHECKS: Paper 225841 (Greenwald & Guren 2025)")
print("=" * 70)

df = load_dta('ls_analysis_data_nc.dta')
df = df.sort_values(['cbsa', 'year']).reset_index(drop=True)

# ============================================================
# 1. DIFFERENT SAMPLES
# ============================================================
print("\n1. SAMPLE SENSITIVITY")
print("-" * 50)

controls = ['Lfraction', 'L2fraction', 'Lz', 'Llhor_hvs']
controls = [c for c in controls if c in df.columns]

for sample_name, sample_var in [('HVS Common', 'HVS_common_sample'),
                                  ('HVS All', 'HVS_all_sample'),
                                  ('HVS All Unbalanced', 'HVS_horall_sample')]:
    if sample_var in df.columns:
        cond = df[sample_var] == 1
        r = run_local_projection(df, 'lhor_hvs', 'z', controls,
                                  ['year', 'cbsa'], 'cbsa', 'pop2000', condition=cond, max_h=2)
        if r['b']:
            print(f"{sample_name}: h=0 coef={r['b'][0]:.4f} (SE={r['se'][0]:.4f}), N={r['n'][0]}")


# ============================================================
# 2. WITH vs WITHOUT EMPLOYMENT CONTROLS (HOR)
# ============================================================
print("\n\n2. EMPLOYMENT CONTROLS SENSITIVITY")
print("-" * 50)

cond_hvs = df['HVS_common_sample'] == 1
emp_controls = [c for c in df.columns if c.startswith('emp1_share')]

controls_base = ['Lfraction', 'L2fraction', 'Lz', 'Llhor_hvs']
controls_base = [c for c in controls_base if c in df.columns]
controls_emp = controls_base + ['lemp'] + emp_controls
controls_emp = [c for c in controls_emp if c in df.columns]

r_base = run_local_projection(df, 'lhor_hvs', 'z', controls_base,
                               ['year', 'cbsa'], 'cbsa', 'pop2000', condition=cond_hvs, max_h=2)
r_emp = run_local_projection(df, 'lhor_hvs', 'z', controls_emp,
                              ['year', 'cbsa'], 'cbsa', 'pop2000', condition=cond_hvs, max_h=2)

print(f"{'Horizon':<10} {'No emp ctrl':>15} {'With emp ctrl':>15}")
for h in range(min(len(r_base['b']), len(r_emp['b']))):
    print(f"h={r_base['h'][h]:<8} {r_base['b'][h]:>12.4f}   {r_emp['b'][h]:>12.4f}")


# ============================================================
# 3. DIFFERENT OUTCOMES (PRR, HOR, HPI)
# ============================================================
print("\n\n3. DIFFERENT OUTCOME VARIABLES")
print("-" * 50)

for yvar, lag_var, label in [('lprr', 'Llprr', 'Price-Rent Ratio'),
                               ('lhor_hvs', 'Llhor_hvs', 'Homeownership Rate'),
                               ('lhpi', 'Llhpi', 'House Price Index')]:
    if yvar in df.columns and lag_var in df.columns:
        ctrl = ['Lfraction', 'L2fraction', 'Lz', lag_var]
        ctrl = [c for c in ctrl if c in df.columns]
        r = run_local_projection(df, yvar, 'z', ctrl,
                                  ['year', 'cbsa'], 'cbsa', 'pop2000', condition=cond_hvs, max_h=2)
        if r['b']:
            print(f"{label}: h=0 coef={r['b'][0]:.4f} (SE={r['se'][0]:.4f})")


# ============================================================
# 4. EXTENDED SAMPLE PERIOD
# ============================================================
print("\n\n4. EXTENDED SAMPLE PERIOD (HPI)")
print("-" * 50)

ctrl_hpi = ['Lfraction', 'L2fraction', 'Lz', 'Llhpi']
ctrl_hpi = [c for c in ctrl_hpi if c in df.columns]

for start_yr in [1990, 1994, 1998]:
    cond = df['year'] > start_yr
    r = run_local_projection(df, 'lhpi', 'z', ctrl_hpi,
                              ['year', 'cbsa'], 'cbsa', 'pop2000', condition=cond, max_h=2)
    if r['b']:
        print(f"Post-{start_yr}: h=0 coef={r['b'][0]:.4f} (SE={r['se'][0]:.4f}), N={r['n'][0]}")


# ============================================================
# 5. UNWEIGHTED vs WEIGHTED
# ============================================================
print("\n\n5. WEIGHTED vs UNWEIGHTED")
print("-" * 50)

r_w = run_local_projection(df, 'lhor_hvs', 'z', controls_base,
                            ['year', 'cbsa'], 'cbsa', 'pop2000', condition=cond_hvs, max_h=2)
r_uw = run_local_projection(df, 'lhor_hvs', 'z', controls_base,
                             ['year', 'cbsa'], 'cbsa', None, condition=cond_hvs, max_h=2)

print(f"{'Horizon':<10} {'Weighted':>12} {'Unweighted':>12}")
for h in range(min(len(r_w['b']), len(r_uw['b']))):
    print(f"h={r_w['h'][h]:<8} {r_w['b'][h]:>10.4f}   {r_uw['b'][h]:>10.4f}")


print("\n\n" + "=" * 70)
print("ROBUSTNESS SUMMARY")
print("=" * 70)
print("""
Note: All results use non-confidential pseudodata. Coefficient magnitudes
may not match the published paper but methodological implementation matches.

Key robustness dimensions tested:
1. Sample sensitivity (different HVS/GG sample definitions)
2. Employment controls (with/without industry employment shares)
3. Different outcome variables (price-rent, homeownership, HPI)
4. Extended sample periods
5. Weighted vs unweighted regressions
""")
