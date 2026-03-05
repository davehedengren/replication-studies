"""
04_data_audit.py — Data quality audit for nber-ces-clean.dta

Checks: coverage, distributions, logical consistency, missing data,
panel balance, duplicates, and coding anomalies.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from utils import load_clean_data, OUT_DIR

print("=" * 70)
print("DATA AUDIT")
print("=" * 70)

df = load_clean_data()

# ══════════════════════════════════════════════════════════════════════
# 1. COVERAGE
# ══════════════════════════════════════════════════════════════════════
print("\n1. COVERAGE")
print(f"   Total observations: {len(df)}")
print(f"   Industries (sic87dd): {df['sic87dd'].nunique()}")
print(f"   Year range: {int(df['year'].min())}-{int(df['year'].max())}")
print(f"   Years: {sorted(df['year'].unique().astype(int))[:5]}...{sorted(df['year'].unique().astype(int))[-5:]}")

# Analysis sample
asub = df[(df['year'] >= 1980) & (df['year'] <= 2009)]
print(f"\n   Analysis sample (1980-2009):")
print(f"   Obs: {len(asub)}, Industries: {asub['sic87dd'].nunique()}, Years: 30")
print(f"   Expected obs: {asub['sic87dd'].nunique()} x 30 = {asub['sic87dd'].nunique() * 30}")

# Computer sector breakdown
comp = df[df['comp_broad'] == 1]['sic87dd'].nunique()
noncomp = df[df['comp_broad'] == 0]['sic87dd'].nunique()
print(f"   Computer sector: {comp} industries")
print(f"   Non-computer: {noncomp} industries")
print(f"   SIC 34-38 total: {df[(df['sic87dd'] >= 3400) & (df['sic87dd'] <= 3899)]['sic87dd'].nunique()}")
print(f"   SIC 34-38 non-comp: {df[(df['sic87dd'] >= 3400) & (df['sic87dd'] <= 3899) & (df['comp_broad'] == 0)]['sic87dd'].nunique()}")

# ══════════════════════════════════════════════════════════════════════
# 2. PANEL BALANCE
# ══════════════════════════════════════════════════════════════════════
print("\n2. PANEL BALANCE")
obs_per_ind = df.groupby('sic87dd')['year'].count()
obs_per_year = df.groupby('year')['sic87dd'].count()
print(f"   Obs per industry: min={obs_per_ind.min()}, max={obs_per_ind.max()}, "
      f"mean={obs_per_ind.mean():.1f}")
print(f"   Obs per year: min={obs_per_year.min()}, max={obs_per_year.max()}")

expected_years = len(df['year'].unique())
unbal = obs_per_ind[obs_per_ind != expected_years]
if len(unbal) > 0:
    print(f"   WARNING: {len(unbal)} industries have fewer than {expected_years} years")
else:
    print(f"   Panel is perfectly balanced: {expected_years} years for all {len(obs_per_ind)} industries")

# Check for duplicates
dupes = df.duplicated(subset=['sic87dd', 'year'], keep=False)
print(f"   Duplicate (sic87dd, year) pairs: {dupes.sum()}")

# ══════════════════════════════════════════════════════════════════════
# 3. VARIABLE COMPLETENESS
# ══════════════════════════════════════════════════════════════════════
print("\n3. VARIABLE COMPLETENESS")
key_vars = ['emp', 'prodemp', 'prodhrs', 'nom_pay', 'nom_prodpay', 'nom_vship',
            'nom_matcost', 'nom_invest', 'nom_energy', 'nom_cap',
            'real_pay', 'real_prodpay', 'real_vship', 'real_matcost', 'real_invest', 'real_energy',
            'piship', 'pimat', 'piinv', 'pien', 'tfp', 'dtfp']

for v in key_vars:
    if v in df.columns:
        n_miss = df[v].isna().sum()
        n_zero = (df[v] == 0).sum()
        n_neg = (df[v] < 0).sum() if df[v].dtype in ['float64', 'float32', 'int64'] else 0
        flag = ""
        if n_miss > 0:
            flag += f" MISSING={n_miss}"
        if n_neg > 0:
            flag += f" NEGATIVE={n_neg}"
        if flag:
            print(f"   {v:20s}: {flag}")

# CI variables
ci_vars = ['nom_compinvest1977', 'nom_compinvest1982', 'nom_compinvest1987',
           'nom_compinvest_cm1992', 'nom_compinvest_cm2002', 'nom_compinvest_cm2007',
           'smtshare_1988', 'smtshare_1993']
print(f"\n   Computer investment / SMT variables (time-invariant, counted per industry):")
for v in ci_vars:
    if v in df.columns:
        per_ind = df.groupby('sic87dd')[v].first()
        n_miss = per_ind.isna().sum()
        print(f"   {v:35s}: {per_ind.notna().sum()} non-missing, {n_miss} missing")

# ══════════════════════════════════════════════════════════════════════
# 4. DISTRIBUTIONS & SUMMARY STATS
# ══════════════════════════════════════════════════════════════════════
print("\n4. DISTRIBUTIONS (analysis sample 1980-2009)")
asub = df[(df['year'] >= 1980) & (df['year'] <= 2009)].copy()

for v in ['emp', 'nom_vship', 'real_vship', 'nom_pay', 'real_pay', 'nom_invest', 'tfp']:
    if v in asub.columns:
        s = asub[v].describe()
        print(f"\n   {v}:")
        print(f"     mean={s['mean']:.2f}, std={s['std']:.2f}, min={s['min']:.2f}, "
              f"max={s['max']:.2f}, median={s['50%']:.2f}")

        # Top 10 by employment
        top10 = asub[asub['year'] == 2000].nlargest(10, v)
        if len(top10) > 0:
            print(f"     Top 5 in 2000: {top10['sic87dd'].values[:5].tolist()}")

# ══════════════════════════════════════════════════════════════════════
# 5. LOGICAL CONSISTENCY
# ══════════════════════════════════════════════════════════════════════
print("\n5. LOGICAL CONSISTENCY")

# Production pay <= total pay
viol = asub[asub['nom_prodpay'] > asub['nom_pay']]
print(f"   nom_prodpay > nom_pay: {len(viol)} violations")
if len(viol) > 0:
    print(f"     Industries: {viol['sic87dd'].unique()[:5]}")

# Real prod pay <= real total pay (one known violation)
viol_real = asub[asub['real_prodpay'] > asub['real_pay']]
print(f"   real_prodpay > real_pay: {len(viol_real)} violations")
if len(viol_real) > 0:
    for _, r in viol_real.iterrows():
        print(f"     sic87dd={int(r['sic87dd'])}, year={int(r['year'])}, "
              f"prodpay={r['real_prodpay']:.2f}, pay={r['real_pay']:.2f}")

# Materials cost < shipments
viol_mat = asub[asub['nom_matcost'] >= asub['nom_vship']]
print(f"   nom_matcost >= nom_vship (negative value added): {len(viol_mat)} obs")

# Price deflators should be 1 in 2007
for v in ['piship', 'pimat', 'piinv', 'pien']:
    sub2007 = asub[asub['year'] == 2007]
    max_dev = (sub2007[v] - 1).abs().max()
    print(f"   {v} in 2007: max deviation from 1 = {max_dev:.6f}")

# All positive
for v in ['emp', 'nom_vship', 'real_vship', 'nom_pay', 'nom_invest']:
    n_nonpos = (asub[v] <= 0).sum()
    if n_nonpos > 0:
        print(f"   WARNING: {v} has {n_nonpos} non-positive values")
    else:
        print(f"   {v}: all positive")

# ══════════════════════════════════════════════════════════════════════
# 6. MISSING DATA PATTERNS
# ══════════════════════════════════════════════════════════════════════
print("\n6. MISSING DATA PATTERNS BY COMPUTER SECTOR")
asub_comp = asub[asub['comp_broad'] == 1]
asub_noncomp = asub[asub['comp_broad'] == 0]

for v in ['nom_compinvest_cm1992', 'nom_compinvest_cm2007', 'smtshare_1988']:
    if v in asub.columns:
        miss_comp = asub_comp.groupby('sic87dd')[v].first().isna().sum()
        miss_noncomp = asub_noncomp.groupby('sic87dd')[v].first().isna().sum()
        print(f"   {v}: computer sector missing={miss_comp}, non-computer missing={miss_noncomp}")

# ══════════════════════════════════════════════════════════════════════
# 7. OUTLIERS
# ══════════════════════════════════════════════════════════════════════
print("\n7. OUTLIER ANALYSIS (CI rates before standardization)")
ci_raw = pd.read_pickle(os.path.join(OUT_DIR, 'ci_measures_raw.pkl'))
for v in ['cimean', 'ci7782', 'ci8792', 'ci0207']:
    q1 = ci_raw[v].quantile(0.25)
    q3 = ci_raw[v].quantile(0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    outliers = ci_raw[(ci_raw[v] < low) | (ci_raw[v] > high)]
    print(f"   {v}: IQR=[{q1:.2f}, {q3:.2f}], fence=[{low:.2f}, {high:.2f}], "
          f"outliers={len(outliers)} ({len(outliers[outliers['comp_broad']==1])} in comp sector)")

# Employment concentration
print("\n   Employment concentration (analysis sample):")
asub2000 = asub[asub['year'] == 2000].copy()
asub2000['emp_share'] = asub2000['emp'] / asub2000['emp'].sum()
asub2000 = asub2000.sort_values('emp_share', ascending=False)
top5_share = asub2000.head(5)['emp_share'].sum()
top10_share = asub2000.head(10)['emp_share'].sum()
print(f"   Top 5 industries: {top5_share:.1%} of employment")
print(f"   Top 10 industries: {top10_share:.1%} of employment")

# ══════════════════════════════════════════════════════════════════════
# 8. TIME TRENDS
# ══════════════════════════════════════════════════════════════════════
print("\n8. AGGREGATE TIME TRENDS")
agg = asub.groupby('year').agg({
    'emp': 'sum',
    'nom_vship': 'sum',
    'real_vship': 'sum',
    'nom_pay': 'sum',
}).reset_index()

for v in ['emp', 'real_vship']:
    v1980 = agg.loc[agg['year'] == 1980, v].values[0]
    v2009 = agg.loc[agg['year'] == 2009, v].values[0]
    print(f"   {v}: 1980={v1980:.0f}, 2009={v2009:.0f}, change={100*(v2009/v1980-1):.1f}%")

print("\n" + "=" * 70)
print("DATA AUDIT COMPLETE")
print("=" * 70)
