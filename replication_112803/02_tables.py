"""
02_tables.py — Replicate Table A4: Changes-on-Changes regressions

Translates appendix-changes-on-changes.do:
- Compute annualized changes in CI rates and log outcomes across 5 periods
- Run pooled regressions: dlog_Y = delta_t + beta * d_CI + e
- Employment-weighted, clustered SEs by industry
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import statsmodels.api as sm
from utils import load_clean_data, OUT_DIR

print("=" * 70)
print("TABLE A4: Changes-on-Changes Regressions")
print("=" * 70)

# ── 1. Load data and build the changes-on-changes dataset ──
# Following appendix-changes-on-changes.do closely
# It uses its own employment weights: 1977-2007
df = load_clean_data()

# Employment weights for this spec: average emp share over 1977-2007
sub_wt = df.loc[(df['year'] >= 1977) & (df['year'] <= 2007), ['sic87dd', 'year', 'emp']].copy()
totemp = sub_wt.groupby('year')['emp'].transform('sum')
sub_wt['empsh'] = sub_wt['emp'] / totemp
wt = sub_wt.groupby('sic87dd')['empsh'].mean().reset_index()
wt.columns = ['sic87dd', 'wt']

# Get the needed variables at benchmark years
years_needed = [1977, 1982, 1987, 1992, 2002, 2007]
bench = df.loc[df['year'].isin(years_needed),
               ['sic87dd', 'year', 'comp_broad', 'nom_invest',
                'nom_compinvest1977', 'nom_compinvest1982', 'nom_compinvest1987',
                'nom_compinvest_cm1992', 'nom_compinvest_cm2002', 'nom_compinvest_cm2007',
                'nom_invest_cm2002', 'nom_invest_cm2007',
                'real_vship', 'nom_vship', 'emp', 'real_pay', 'real_prodpay']].copy()

bench.rename(columns={
    'real_pay': 'pay',
    'real_prodpay': 'prodpay',
}, inplace=True)

# Reshape time-varying vars to wide (one row per industry)
vals = ['nom_invest', 'real_vship', 'nom_vship', 'emp', 'pay', 'prodpay']
wide = bench.pivot_table(index='sic87dd', columns='year', values=vals)
wide.columns = [f'{v}{int(y)}' for v, y in wide.columns]
wide = wide.reset_index()

# Get comp_broad and time-invariant vars (take first non-null per industry)
static_cols = ['sic87dd', 'comp_broad',
               'nom_compinvest1977', 'nom_compinvest1982', 'nom_compinvest1987',
               'nom_compinvest_cm1992', 'nom_compinvest_cm2002', 'nom_compinvest_cm2007',
               'nom_invest_cm2002', 'nom_invest_cm2007']
static = bench.groupby('sic87dd')[static_cols[1:]].first().reset_index()
wide = wide.merge(static, on='sic87dd')
wide = wide.merge(wt, on='sic87dd')

# CI rates: use NBER-CES nom_invest for 77/82/87/92, CM invest for 02/07
for y in [1977, 1982, 1987]:
    wide[f'ci{y}'] = 100.0 * (wide[f'nom_compinvest{y}'] / wide[f'nom_invest{y}'])
wide['ci1992'] = 100.0 * (wide['nom_compinvest_cm1992'] / wide['nom_invest1992'])
wide['ci2002'] = 100.0 * (wide['nom_compinvest_cm2002'] / wide['nom_invest_cm2002'])
wide['ci2007'] = 100.0 * (wide['nom_compinvest_cm2007'] / wide['nom_invest_cm2007'])

# Impute missing
miss92 = wide['ci1992'].isna()
wide.loc[miss92, 'ci1992'] = (2/3) * wide.loc[miss92, 'ci1987'] + (1/3) * wide.loc[miss92, 'ci2002']

miss07 = wide['ci2007'].isna()
nonmiss07 = ~miss07
ci2002_mean = np.average(wide.loc[nonmiss07, 'ci2002'], weights=wide.loc[nonmiss07, 'wt'])
ci2007_mean = np.average(wide.loc[nonmiss07, 'ci2007'], weights=wide.loc[nonmiss07, 'wt'])
wide.loc[miss07, 'ci2007'] = wide.loc[miss07, 'ci2002'] * (ci2007_mean / ci2002_mean)

# Labor productivity and non-prod pay
for y in years_needed:
    wide[f'laborprod{y}'] = wide[f'real_vship{y}'] / wide[f'emp{y}']
    wide[f'nonprodpay{y}'] = wide[f'pay{y}'] - wide[f'prodpay{y}']

# Annualized changes
periods = [('1977', '1982'), ('1982', '1987'), ('1987', '1992'), ('1992', '2002'), ('2002', '2007')]

rows = []
for start, end in periods:
    s, e = int(start), int(end)
    span = e - s
    period_label = f'{start}_{end}'

    row = wide[['sic87dd', 'comp_broad', 'wt']].copy()
    row['period'] = period_label

    # Annualized change in CI rate (not logged)
    row['d_ci'] = (wide[f'ci{e}'] - wide[f'ci{s}']) / span

    # Annualized log changes in outcomes
    for v in ['laborprod', 'real_vship', 'nom_vship', 'emp', 'pay', 'prodpay', 'nonprodpay']:
        row[f'dlog_{v}'] = 100.0 * (np.log(wide[f'{v}{e}']) - np.log(wide[f'{v}{s}'])) / span

    rows.append(row)

panel = pd.concat(rows, ignore_index=True)

# Period dummies
for start, end in periods:
    panel[f't_{start}_{end}'] = (panel['period'] == f'{start}_{end}').astype(float)

time_cols = [f't_{s}_{e}' for s, e in periods]

print(f"\nPanel: {len(panel)} obs, {panel['sic87dd'].nunique()} industries, "
      f"{len(periods)} periods")

# ── 2. Run regressions ──
# Published values from Table A4
published = {
    ('laborprod', False): {'coef': -0.39, 'se': 0.20, 'n_inds': 387, 'N': 1935},
    ('laborprod', True):  {'coef': -0.14, 'se': 0.09, 'n_inds': 359, 'N': 1795},
    ('real_vship', True): {'coef': -0.12, 'se': 0.17, 'n_inds': 359, 'N': 1795},
    ('nom_vship', True):  {'coef': -0.02, 'se': 0.18, 'n_inds': 359, 'N': 1795},
    ('emp', True):        {'coef': 0.02, 'se': 0.14, 'n_inds': 359, 'N': 1795},
    ('pay', True):        {'coef': 0.09, 'se': 0.15, 'n_inds': 359, 'N': 1795},
    ('prodpay', True):    {'coef': -0.13, 'se': 0.16, 'n_inds': 359, 'N': 1795},
    ('nonprodpay', True): {'coef': 0.33, 'se': 0.15, 'n_inds': 359, 'N': 1795},
}

print(f"\n{'Dep Var':<15} {'ExclComp':>8} {'beta':>8} {'pub_b':>8} {'SE':>8} {'pub_SE':>8} {'N':>6} {'n_ind':>6} {'Match':>6}")
print("-" * 85)

for depvar in ['laborprod', 'real_vship', 'nom_vship', 'emp', 'pay', 'prodpay', 'nonprodpay']:
    for excl_comp in [False, True]:
        # Only run including computer sector for labor productivity
        if depvar != 'laborprod' and not excl_comp:
            continue

        key = (depvar, excl_comp)
        if key not in published:
            continue

        if excl_comp:
            data = panel.loc[panel['comp_broad'] == 0].copy()
        else:
            data = panel.copy()

        y = data[f'dlog_{depvar}'].values
        X = data[['d_ci'] + time_cols].values.astype(float)
        w = data['wt'].values
        clusters = data['sic87dd'].values

        # WLS, no constant (time dummies span the space)
        model = sm.WLS(y, X, weights=w)
        result = model.fit(cov_type='cluster', cov_kwds={'groups': clusters}, use_t=True)

        beta = result.params[0]
        se = result.bse[0]
        n_obs = len(y)
        n_inds = data['sic87dd'].nunique()

        pub = published[key]
        match_b = "OK" if abs(beta - pub['coef']) < 0.015 else f"DIFF({beta - pub['coef']:+.3f})"
        match_se = "OK" if abs(se - pub['se']) < 0.015 else f"DIFF"

        print(f"{depvar:<15} {'Yes' if excl_comp else 'No':>8} {beta:>8.2f} {pub['coef']:>8.2f} "
              f"{se:>8.2f} {pub['se']:>8.2f} {n_obs:>6} {n_inds:>6} {match_b:>6}")

print("\n" + "=" * 70)
