"""
01_clean.py — Load and validate all datasets for 221423-V1.

Paper: "Income Inequality in the Nordic Countries"
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from utils import *

print('=' * 70)
print('01_clean.py — Data loading and validation')
print('=' * 70)

# ══════════════════════════════════════════════════════════════════════
# 1. Table 1 data: Demographics & Economy
# ══════════════════════════════════════════════════════════════════════
print('\n1. Table 1 Data')
print('=' * 70)

# 1a. Population
pop_csv = pd.read_csv(os.path.join(TABLE1_DIR, 'population.csv'))
print(f'  population.csv: {pop_csv.shape}')
pop_dta = pd.read_stata(os.path.join(TABLE1_DIR, 'population.dta'))
print(f'  population.dta: {pop_dta.shape}, columns: {list(pop_dta.columns)}')

# 1b. Fertility
fertility = pd.read_csv(os.path.join(TABLE1_DIR, 'fertility.csv'))
print(f'  fertility.csv: {fertility.shape}')

# 1c. Foreign born
foreign = pd.read_csv(os.path.join(TABLE1_DIR, 'foreign.csv'), header=2)
print(f'  foreign.csv: {foreign.shape}')

# 1d. Foreign origin 2018
foreign_origin = pd.read_excel(os.path.join(TABLE1_DIR, 'foreign_origin_2018.XLSX'),
                                header=0, skiprows=range(0, 39), nrows=40)
print(f'  foreign_origin_2018.xlsx: {foreign_origin.shape}')

# 1e. Better life index
bli = pd.read_csv(os.path.join(TABLE1_DIR, 'better_life_index.csv'))
print(f'  better_life_index.csv: {bli.shape}')

# 1f. Productivity
prod_csv = pd.read_csv(os.path.join(TABLE1_DIR, 'productivity.csv'))
print(f'  productivity.csv: {prod_csv.shape}')
prod_dta = pd.read_stata(os.path.join(TABLE1_DIR, 'productivity.dta'))
print(f'  productivity.dta: {prod_dta.shape}, columns: {list(prod_dta.columns)}')

# 1g. Labor
labor_csv = pd.read_csv(os.path.join(TABLE1_DIR, 'labor.csv'))
print(f'  labor.csv: {labor_csv.shape}')
labor_dta = pd.read_stata(os.path.join(TABLE1_DIR, 'labor.dta'))
print(f'  labor.dta: {labor_dta.shape}, columns: {list(labor_dta.columns)}')

# 1h. Public sector employment
public = pd.read_excel(os.path.join(TABLE1_DIR, 'public.xlsx'),
                        header=0, skiprows=range(0, 33), nrows=38)
print(f'  public.xlsx: {public.shape}')


# ══════════════════════════════════════════════════════════════════════
# 2. Figure 2 data: Kleven tax/subsidy
# ══════════════════════════════════════════════════════════════════════
print('\n2. Figure 2 Data (Kleven)')
print('=' * 70)

kleven_tax = pd.read_excel(os.path.join(FIGURE2_DIR, 'kleven_jep2014_exceldata.xlsx'),
                            sheet_name='Figure 4a', header=1)
print(f'  Figure 4a: {kleven_tax.shape}, columns: {list(kleven_tax.columns[:5])}')

kleven_sub = pd.read_excel(os.path.join(FIGURE2_DIR, 'kleven_jep2014_exceldata.xlsx'),
                            sheet_name='Figure 5a', header=0)
print(f'  Figure 5a: {kleven_sub.shape}, columns: {list(kleven_sub.columns[:5])}')


# ══════════════════════════════════════════════════════════════════════
# 3. Figure 3 data: Union density & bargaining coverage
# ══════════════════════════════════════════════════════════════════════
print('\n3. Figure 3 Data (OECD-AIAS-ICTWSS)')
print('=' * 70)

ictwss = pd.read_csv(os.path.join(FIGURE3_DIR, 'OECD-AIAS-ICTWSS-CSV.csv'))
print(f'  OECD-AIAS-ICTWSS-CSV.csv: {ictwss.shape}')
print(f'    Countries: {ictwss["country"].nunique()}, Years: {ictwss["year"].min()}-{ictwss["year"].max()}')

labor_force = pd.read_csv(os.path.join(FIGURE3_DIR, 'OECD-Labor-Force.csv'))
print(f'  OECD-Labor-Force.csv: {labor_force.shape}')


# ══════════════════════════════════════════════════════════════════════
# 4. Table 3 data: Gini coefficients
# ══════════════════════════════════════════════════════════════════════
print('\n4. Table 3 Data (OECD Gini/Poverty)')
print('=' * 70)

gini = pd.read_csv(os.path.join(TABLE3_DIR, 'OECD_income_gini_poverty.csv'))
print(f'  OECD_income_gini_poverty.csv: {gini.shape}')
print(f'    Measures: {sorted(gini["MEASURE"].unique())}')
print(f'    Columns: {list(gini.columns[:6])}')
print(f'    Countries: {gini["Country"].nunique()}, Years: {sorted(gini["TIME"].unique())}')

# Nordic population for weighted averages
nordic_pop = pd.read_stata(os.path.join(DATA_DIR, 'nordic_population.dta'))
print(f'\n  nordic_population.dta: {nordic_pop.shape}')
print(f'    {nordic_pop[["ref_area", "country", "POP"]].to_string(index=False)}')


# ══════════════════════════════════════════════════════════════════════
# 5. Check for PIAAC data
# ══════════════════════════════════════════════════════════════════════
print('\n5. PIAAC Data Status')
print('=' * 70)

piaac_dir = os.path.join(DATA_DIR, 'piaac_raw_PUFs')
piaac_files = [f for f in os.listdir(piaac_dir) if f.endswith('.csv')]
print(f'  PIAAC CSVs present: {len(piaac_files)}/30')
print(f'    Files: {sorted(piaac_files)}')

# Check for pre-built analysis data
for fname in ['piaac_raw_data.dta', 'piaac_all_individuals.dta', 'piaac_all_workers.dta']:
    fpath = os.path.join(DATA_DIR, fname)
    if os.path.exists(fpath):
        print(f'  {fname}: EXISTS')
    else:
        print(f'  {fname}: NOT FOUND (requires PIAAC download + cleaning)')

# Check for PISA data
pisa_path = os.path.join(DATA_DIR, 'pisa_2018_raw')
if os.path.exists(pisa_path):
    print(f'  PISA 2018 data: EXISTS')
else:
    print(f'  PISA 2018 data: NOT FOUND')

# Check for WVS data
wvs_path = os.path.join(TABLE1_DIR, 'Joint_EVS_WVS_2017_2022.dta')
if os.path.exists(wvs_path):
    print(f'  WVS/EVS data: EXISTS')
else:
    print(f'  WVS/EVS data: NOT FOUND (needed for Table 1 social trust)')


# ══════════════════════════════════════════════════════════════════════
# 6. Published outputs (for comparison)
# ══════════════════════════════════════════════════════════════════════
print('\n6. Published Outputs Available')
print('=' * 70)

for fname in sorted(os.listdir(OUTPUTS_DIR)):
    fpath = os.path.join(OUTPUTS_DIR, fname)
    print(f'  {fname}: {os.path.getsize(fpath):,} bytes')


print('\n' + '=' * 70)
print('01_clean.py — DONE')
print('=' * 70)
