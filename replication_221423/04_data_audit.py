"""
04_data_audit.py — Data audit for 221423-V1.

Paper: "Income Inequality in the Nordic Countries"
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from utils import *

print('=' * 70)
print('04_data_audit.py — Data audit')
print('=' * 70)

# ══════════════════════════════════════════════════════════════════════
# 1. Data completeness
# ══════════════════════════════════════════════════════════════════════
print('\n1. Data Completeness')
print('=' * 70)

# Table 1 data
pop = pd.read_stata(os.path.join(TABLE1_DIR, 'population.dta'))
prod = pd.read_stata(os.path.join(TABLE1_DIR, 'productivity.dta'))
labor = pd.read_stata(os.path.join(TABLE1_DIR, 'labor.dta'))
fertility = pd.read_csv(os.path.join(TABLE1_DIR, 'fertility.csv'))
foreign = pd.read_csv(os.path.join(TABLE1_DIR, 'foreign.csv'), header=2)
bli = pd.read_csv(os.path.join(TABLE1_DIR, 'better_life_index.csv'))

print('  Table 1 sources:')
for name, df in [('population.dta', pop), ('productivity.dta', prod),
                 ('labor.dta', labor), ('fertility.csv', fertility),
                 ('foreign.csv', foreign), ('better_life_index.csv', bli)]:
    print(f'    {name}: {df.shape[0]} rows, {df.shape[1]} cols, '
          f'{df.isnull().sum().sum()} missing values')

# Table 3 data
gini = pd.read_csv(os.path.join(TABLE3_DIR, 'OECD_income_gini_poverty.csv'))
print(f'\n  Table 3 source:')
print(f'    OECD_income_gini_poverty.csv: {gini.shape[0]} rows')
print(f'    Measures: {sorted(gini["MEASURE"].unique())}')
print(f'    Years: {sorted(gini["TIME"].unique())}')
gini_2019 = gini[gini.TIME == 2019]
print(f'    Countries with 2019 data: {gini_2019["Country"].nunique()}')

# Figure 3 data
adt = pd.read_csv(os.path.join(FIGURE3_DIR, 'OECD-AIAS-ICTWSS-CSV.csv'))
lbf = pd.read_csv(os.path.join(FIGURE3_DIR, 'OECD-Labor-Force.csv'))
print(f'\n  Figure 3 sources:')
print(f'    OECD-AIAS-ICTWSS-CSV.csv: {adt.shape[0]} rows, {adt.shape[1]} cols')
print(f'    OECD-Labor-Force.csv: {lbf.shape[0]} rows')

# ══════════════════════════════════════════════════════════════════════
# 2. Missing data patterns
# ══════════════════════════════════════════════════════════════════════
print('\n2. Missing Data and Data Gaps')
print('=' * 70)

# PIAAC availability
piaac_dir = os.path.join(DATA_DIR, 'piaac_raw_PUFs')
piaac_files = [f for f in os.listdir(piaac_dir) if f.endswith('.csv')]
print(f'  PIAAC PUFs available: {len(piaac_files)}/30')
print(f'    Present: {sorted(piaac_files)}')

required_piaac = ['prgdnkp1.csv', 'prgfinp1.csv', 'prgnorp1.csv', 'prgswe01.csv',
                  'prgusap1.csv', 'prggbrp1.csv']
for f in required_piaac:
    status = 'PRESENT' if f in piaac_files else 'MISSING'
    print(f'    {f}: {status}')

# WVS/EVS availability
wvs_path = os.path.join(TABLE1_DIR, 'Joint_EVS_WVS_2017_2022.dta')
print(f'\n  WVS/EVS data: {"PRESENT" if os.path.exists(wvs_path) else "MISSING"}')

# PISA availability
pisa_path = os.path.join(DATA_DIR, 'pisa_2018_raw')
print(f'  PISA 2018 data: {"PRESENT" if os.path.exists(pisa_path) else "MISSING"}')

# ══════════════════════════════════════════════════════════════════════
# 3. ICTWSS data coverage for Figure 3
# ══════════════════════════════════════════════════════════════════════
print('\n3. ICTWSS Data Coverage')
print('=' * 70)

adt.loc[adt.country == 'United States of America', 'country'] = 'United States'
countries_fig3 = ['Norway', 'Sweden', 'Denmark', 'Finland', 'United Kingdom',
                  'United States', 'Germany', 'France', 'Spain', 'Portugal']

for var in ['UD', 'UD_hist', 'UnadjCov', 'AdjCov_hist']:
    print(f'\n  {var} (1980-2019):')
    for c in countries_fig3:
        d = adt[(adt.country == c) & (adt.year >= 1980) & (adt.year <= 2019)]
        valid = d[var].replace(-88, np.nan).replace(-99, np.nan).dropna()
        n_valid = len(valid)
        yr_range = f'{int(valid.iloc[0])}-{int(valid.iloc[-1])}' if n_valid > 0 else 'none'
        coverage = f'{n_valid}/40 years'
        print(f'    {c:<20}: {coverage} ({yr_range if n_valid > 0 else "no data"})')

# ══════════════════════════════════════════════════════════════════════
# 4. OECD Gini data distributions
# ══════════════════════════════════════════════════════════════════════
print('\n4. Gini Data Distributions')
print('=' * 70)

gini_vals = gini[(gini.MEASURE == 'GINI') & (gini.TIME == 2019)]
ginib_vals = gini[(gini.MEASURE == 'GINIB') & (gini.TIME == 2019)]

print(f'  Gini disposable income (2019): N={len(gini_vals)}')
print(f'    Mean={gini_vals["Value"].mean():.3f}, Median={gini_vals["Value"].median():.3f}')
print(f'    Min={gini_vals["Value"].min():.3f} ({gini_vals.loc[gini_vals["Value"].idxmin(), "Country"]})')
print(f'    Max={gini_vals["Value"].max():.3f} ({gini_vals.loc[gini_vals["Value"].idxmax(), "Country"]})')

print(f'\n  Gini market income (2019): N={len(ginib_vals)}')
print(f'    Mean={ginib_vals["Value"].mean():.3f}, Median={ginib_vals["Value"].median():.3f}')
print(f'    Min={ginib_vals["Value"].min():.3f} ({ginib_vals.loc[ginib_vals["Value"].idxmin(), "Country"]})')
print(f'    Max={ginib_vals["Value"].max():.3f} ({ginib_vals.loc[ginib_vals["Value"].idxmax(), "Country"]})')

# ══════════════════════════════════════════════════════════════════════
# 5. Population data validation
# ══════════════════════════════════════════════════════════════════════
print('\n5. Population Data Validation')
print('=' * 70)

pop_raw = pd.read_csv(os.path.join(TABLE1_DIR, 'population.csv'))
print(f'  Raw population CSV: {pop_raw.shape}')
print(f'    Countries: {pop_raw["REF_AREA"].unique().tolist()}')
print(f'    Measures: {pop_raw["MEASURE"].unique().tolist() if "MEASURE" in pop_raw.columns else "N/A"}')

nordic_pop = pd.read_stata(os.path.join(DATA_DIR, 'nordic_population.dta'))
print(f'\n  Nordic populations (from DTA):')
for _, row in nordic_pop.iterrows():
    print(f'    {row["country"]}: {row["POP"]:,.0f}')
print(f'    Total: {nordic_pop["POP"].sum():,.0f}')

# ══════════════════════════════════════════════════════════════════════
# 6. Table coverage summary
# ══════════════════════════════════════════════════════════════════════
print('\n6. Replication Coverage Summary')
print('=' * 70)

tables = [
    ('Table 1', 'Demographics & Economy', '15/16 rows (Social Trust needs WVS+PIAAC)', 'Partial'),
    ('Table 3', 'Gini coefficients', '3/5 columns (earnings Gini needs PIAAC)', 'Partial'),
    ('Tables 4-14', 'PIAAC-based analysis', 'All need PIAAC data (0/30 CSVs for key countries)', 'Not replicable'),
    ('Figure 2', 'Kleven tax/subsidy scatter', 'All data provided', 'Full'),
    ('Figure 3', 'Union density & coverage', 'All data provided', 'Full'),
    ('Figure 5', 'Norwegian micro-data', 'Restricted access, no code provided', 'Not replicable'),
]

for name, desc, coverage, status in tables:
    print(f'  {name:<15} {desc:<35} {status}')
    print(f'    {" "*15} {coverage}')

print('\n' + '=' * 70)
print('04_data_audit.py — DONE')
print('=' * 70)
