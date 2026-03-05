"""
01_clean.py — Build the analysis-ready dataset from nber-ces-clean.dta

Translates the data-preparation portions of main-text.do and appendix-core.do:
1. Load the pre-built clean dataset
2. Compute employment weights (avg emp share 1980-2009)
3. Define computer investment rates (ci1977–ci2007, cimean, ci7782, ci8792, ci0207)
4. Define SMT technology usage measure
5. Standardize IT measures to zero weighted mean, unit weighted SD
6. Compute log outcomes (laborprod, real_vship, nom_vship, emp, pay, etc.)
7. Save analysis-ready dataset
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from utils import load_clean_data, compute_emp_weights, OUT_DIR, CI_WEIGHTS

print("=" * 70)
print("PHASE 1: Building analysis-ready dataset")
print("=" * 70)

# ── 1. Load clean data ──
df = load_clean_data()
print(f"\nLoaded nber-ces-clean.dta: {len(df)} obs, {df['sic87dd'].nunique()} industries, "
      f"years {df['year'].min()}-{df['year'].max()}")

# ── 2. Compute employment weights (1980-2009) ──
wt = compute_emp_weights(df, 1980, 2009)
print(f"Employment weights: {len(wt)} industries, sum = {wt['wt'].sum():.4f}")

# ── 3. Define computer investment rates ──
# Need years 1977, 1982, 1987, 1992 from the clean data + CM data for 2002/2007
ci_data = df.loc[df['year'].isin([1977, 1982, 1987, 1992]),
                 ['sic87dd', 'year', 'comp_broad', 'nom_invest',
                  'nom_compinvest1977', 'nom_compinvest1982', 'nom_compinvest1987',
                  'nom_compinvest_cm1992',
                  'nom_compinvest_cm2002', 'nom_compinvest_cm2007',
                  'nom_invest_cm2002', 'nom_invest_cm2007',
                  'smtshare_1988', 'smtshare_1993']].copy()

# Reshape to get nom_invest by year
invest_wide = ci_data.pivot_table(index='sic87dd', columns='year', values='nom_invest')
invest_wide.columns = [f'nom_invest{int(c)}' for c in invest_wide.columns]

# Get one row per industry (these are time-invariant in the clean data)
ci_static = ci_data.groupby('sic87dd').first().reset_index()
ci_static = ci_static[['sic87dd', 'comp_broad',
                        'nom_compinvest1977', 'nom_compinvest1982', 'nom_compinvest1987',
                        'nom_compinvest_cm1992',
                        'nom_compinvest_cm2002', 'nom_compinvest_cm2007',
                        'nom_invest_cm2002', 'nom_invest_cm2007',
                        'smtshare_1988', 'smtshare_1993']]

ci_static = ci_static.merge(invest_wide.reset_index(), on='sic87dd')
ci_static = ci_static.merge(wt, on='sic87dd')

# Rename CM variables to match Stata code
ci_static.rename(columns={
    'nom_compinvest_cm1992': 'nom_compinvest1992',
    'nom_compinvest_cm2002': 'nom_compinvest2002',
    'nom_compinvest_cm2007': 'nom_compinvest2007',
    'nom_invest_cm2002': 'nom_invest2002',
    'nom_invest_cm2007': 'nom_invest2007',
}, inplace=True)

# Compute CI rates: 100 * (comp_invest / total_invest) for each year
for y in [1977, 1982, 1987, 1992, 2002, 2007]:
    ci_static[f'ci{y}'] = 100.0 * (ci_static[f'nom_compinvest{y}'] / ci_static[f'nom_invest{y}'])

# Verify no missing for 1977, 1982, 1987, 2002
for y in [1977, 1982, 1987, 2002]:
    assert ci_static[f'ci{y}'].notna().all(), f"ci{y} has missing values!"

# Interpolate missing 1992 values
miss92 = ci_static['ci1992'].isna()
ci_static.loc[miss92, 'ci1992'] = (
    (2/3) * ci_static.loc[miss92, 'ci1987'] +
    (1/3) * ci_static.loc[miss92, 'ci2002']
)
print(f"Interpolated {miss92.sum()} missing ci1992 values")

# Extrapolate missing 2007 values
miss07 = ci_static['ci2007'].isna()
nonmiss07 = ~miss07
ci2002_mean = np.average(ci_static.loc[nonmiss07, 'ci2002'], weights=ci_static.loc[nonmiss07, 'wt'])
ci2007_mean = np.average(ci_static.loc[nonmiss07, 'ci2007'], weights=ci_static.loc[nonmiss07, 'wt'])
ci_static.loc[miss07, 'ci2007'] = ci_static.loc[miss07, 'ci2002'] * (ci2007_mean / ci2002_mean)
print(f"Extrapolated {miss07.sum()} missing ci2007 values")

# Preferred measure: weighted average
ci_static['cimean'] = sum(CI_WEIGHTS[y] * ci_static[f'ci{y}'] for y in CI_WEIGHTS)

# Vintage averages
ci_static['ci7782'] = (ci_static['ci1977'] + ci_static['ci1982']) / 2
ci_static['ci8792'] = (ci_static['ci1987'] + ci_static['ci1992']) / 2
ci_static['ci0207'] = (ci_static['ci2002'] + ci_static['ci2007']) / 2

# Verify all CI in [0, 100]
for v in ['cimean', 'ci7782', 'ci8792', 'ci0207'] + [f'ci{y}' for y in [1977,1982,1987,1992,2002,2007]]:
    assert (ci_static[v] >= 0).all() and (ci_static[v] <= 100).all(), f"{v} out of [0,100]"

# SMT measure: mean of 1988 and 1993
ci_static['smt'] = (ci_static['smtshare_1988'] + ci_static['smtshare_1993']) / 2

# ── 4. Standardize IT measures ──
# CI measures: standardize to zero weighted mean, unit weighted SD across all industries
ci_measures_std = ci_static[['sic87dd', 'wt', 'comp_broad',
                              'cimean', 'ci7782', 'ci8792', 'ci0207',
                              'ci1977', 'ci1982', 'ci1987', 'ci1992', 'ci2002', 'ci2007',
                              'smt']].copy()

for v in ['cimean', 'ci7782', 'ci8792', 'ci0207',
          'ci1977', 'ci1982', 'ci1987', 'ci1992', 'ci2002', 'ci2007']:
    wmean = np.average(ci_measures_std[v], weights=ci_measures_std['wt'])
    wvar = np.average((ci_measures_std[v] - wmean)**2, weights=ci_measures_std['wt'])
    wsd = np.sqrt(wvar)
    ci_measures_std[v] = (ci_measures_std[v] - wmean) / wsd

# SMT: standardize among industries where it's defined
smt_mask = ci_measures_std['smt'].notna()
if smt_mask.any():
    wmean_smt = np.average(ci_measures_std.loc[smt_mask, 'smt'],
                           weights=ci_measures_std.loc[smt_mask, 'wt'])
    wvar_smt = np.average((ci_measures_std.loc[smt_mask, 'smt'] - wmean_smt)**2,
                          weights=ci_measures_std.loc[smt_mask, 'wt'])
    wsd_smt = np.sqrt(wvar_smt)
    ci_measures_std.loc[smt_mask, 'smt'] = (ci_measures_std.loc[smt_mask, 'smt'] - wmean_smt) / wsd_smt

print(f"\nStandardized IT measures (weighted mean ≈ 0, weighted SD ≈ 1):")
for v in ['cimean', 'ci7782', 'ci8792', 'ci0207', 'smt']:
    mask = ci_measures_std[v].notna()
    wm = np.average(ci_measures_std.loc[mask, v], weights=ci_measures_std.loc[mask, 'wt'])
    wv = np.average((ci_measures_std.loc[mask, v] - wm)**2, weights=ci_measures_std.loc[mask, 'wt'])
    print(f"  {v}: mean={wm:.6f}, sd={np.sqrt(wv):.6f}, N={mask.sum()}")

# ── 5. Prepare outcomes ──
outcomes = df.loc[(df['year'] >= 1980) & (df['year'] <= 2009),
                  ['sic87dd', 'year', 'comp_broad', 'real_vship', 'nom_vship', 'emp',
                   'real_pay', 'real_prodpay', 'piship', 'nom_vadd', 'tfp']].copy()

outcomes.rename(columns={'real_pay': 'pay', 'real_prodpay': 'prodpay'}, inplace=True)

# Labor productivity
outcomes['laborprod'] = outcomes['real_vship'] / outcomes['emp']

# Handle one case where prodpay > pay (sic87dd 3263, year 1995)
mask_fix = (outcomes['sic87dd'] == 3263) & (outcomes['year'] == 1995)
outcomes.loc[mask_fix, 'prodpay'] = outcomes.loc[mask_fix, 'pay']

outcomes['nonprodpay'] = outcomes['pay'] - outcomes['prodpay']

# Log outcomes (100 * log)
for v in ['laborprod', 'real_vship', 'nom_vship', 'emp', 'pay', 'prodpay', 'nonprodpay',
          'piship', 'nom_vadd', 'tfp']:
    outcomes[f'log_{v}'] = 100.0 * np.log(outcomes[v])

print(f"\nOutcomes: {len(outcomes)} obs, {outcomes['sic87dd'].nunique()} industries")

# ── 6. Merge and save ──
analysis = outcomes.merge(
    ci_measures_std[['sic87dd', 'wt', 'cimean', 'ci7782', 'ci8792', 'ci0207',
                      'ci1977', 'ci1982', 'ci1987', 'ci1992', 'ci2002', 'ci2007', 'smt']],
    on='sic87dd', how='left'
)

# Verify
print(f"\nFinal analysis dataset: {len(analysis)} obs")
print(f"Industries: {analysis['sic87dd'].nunique()}")
print(f"Years: {analysis['year'].min()}-{analysis['year'].max()}")
print(f"Weight sum: {analysis.groupby('year')['wt'].sum().mean():.4f}")

# Key sample sizes from the paper
n_all = analysis['sic87dd'].nunique()
n_nocomp = analysis.loc[analysis['comp_broad'] == 0, 'sic87dd'].nunique()
n_sic3438 = analysis.loc[(analysis['sic87dd'] >= 3400) & (analysis['sic87dd'] <= 3899), 'sic87dd'].nunique()
n_sic3438_nocomp = analysis.loc[(analysis['sic87dd'] >= 3400) & (analysis['sic87dd'] <= 3899) &
                                 (analysis['comp_broad'] == 0), 'sic87dd'].nunique()

print(f"\nSample sizes (should match paper):")
print(f"  All manufacturing: {n_all} (paper: 387)")
print(f"  Excl. computer sector: {n_nocomp} (paper: 359)")
print(f"  SIC 34-38: {n_sic3438} (paper: 148)")
print(f"  SIC 34-38 excl. computer: {n_sic3438_nocomp} (paper: 120)")

# Save
analysis.to_pickle(os.path.join(OUT_DIR, 'analysis_data.pkl'))
# Also save the un-standardized CI data for Table A4
ci_static.to_pickle(os.path.join(OUT_DIR, 'ci_measures_raw.pkl'))

print("\nSaved analysis_data.pkl and ci_measures_raw.pkl")
print("=" * 70)
