"""
02_tables.py — Replicate Table 1 and Table 3 for 221423-V1.

Paper: "Income Inequality in the Nordic Countries"
Table 1: Demographics & Economy (all rows except Social Trust which needs WVS+PIAAC)
Table 3: Gini coefficients (columns 1-3 only; columns 4-5 need PIAAC)
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from utils import *

print('=' * 70)
print('02_tables.py — Table replication')
print('=' * 70)

# ══════════════════════════════════════════════════════════════════════
# Table 1: Demographics & Economy
# ══════════════════════════════════════════════════════════════════════
print('\nTABLE 1: Demographics & Economy')
print('=' * 70)

# --- Load pre-cleaned DTA files ---
pop = pd.read_stata(os.path.join(TABLE1_DIR, 'population.dta'))
prod = pd.read_stata(os.path.join(TABLE1_DIR, 'productivity.dta'))
labor = pd.read_stata(os.path.join(TABLE1_DIR, 'labor.dta'))
nordic_pop = pd.read_stata(os.path.join(DATA_DIR, 'nordic_population.dta'))

# Start building table from population data
table1 = pop[['ref_area', 'country', 'cntry', 'population_millions',
              'working_age_pop_millions', 'dependency_ratio_tens']].copy()

# --- Row 4: Fertility ---
fertility = pd.read_csv(os.path.join(TABLE1_DIR, 'fertility.csv'))
fert = fertility[['REF_AREA', 'OBS_VALUE']].rename(
    columns={'REF_AREA': 'ref_area', 'OBS_VALUE': 'fertility'})
fert.loc[fert.ref_area == 'OECD_REP', 'ref_area'] = 'OECD'
table1 = table1.merge(fert, on='ref_area', how='left')

# --- Row 5: Foreign born residents ---
foreign = pd.read_csv(os.path.join(TABLE1_DIR, 'foreign.csv'), header=2)
foreign.columns = ['country', 'foreign_born_pct']
# Compute OECD average
oecd_avg = foreign['foreign_born_pct'].mean()
oecd_row = pd.DataFrame({'country': ['OECD'], 'foreign_born_pct': [oecd_avg]})
foreign = pd.concat([foreign, oecd_row], ignore_index=True)
table1 = table1.merge(foreign, on='country', how='left')

# --- Row 6: Non-western immigrants ---
foreign_origin = pd.read_excel(os.path.join(TABLE1_DIR, 'foreign_origin_2018.XLSX'),
                                header=0, skiprows=range(0, 39), nrows=40)
foreign_origin = foreign_origin.rename(columns={foreign_origin.columns[0]: 'country'})
foreign_origin.loc[foreign_origin.country == 'OECD total (33)', 'country'] = 'OECD'
# Compute non-western as Africa + Asia + Latin America
for col in ['Africa', 'Asia', 'LatinAmerica', 'Latin America']:
    if col in foreign_origin.columns:
        break
# Find the right columns
cols = foreign_origin.columns.tolist()
# The Excel has columns like: country, Total, Europe, Africa, Asia, LatinAmerica, ...
# Need to identify Africa, Asia, Latin America columns
africa_col = [c for c in cols if 'africa' in c.lower() or 'Africa' in c]
asia_col = [c for c in cols if 'asia' in c.lower() or 'Asia' in c]
latam_col = [c for c in cols if 'latin' in c.lower() or 'Latin' in c]

if africa_col and asia_col and latam_col:
    foreign_origin['foreign_non_western'] = (
        pd.to_numeric(foreign_origin[africa_col[0]], errors='coerce') +
        pd.to_numeric(foreign_origin[asia_col[0]], errors='coerce') +
        pd.to_numeric(foreign_origin[latam_col[0]], errors='coerce')
    )
else:
    # Try numeric columns
    print(f'  Foreign origin columns: {cols}')
    foreign_origin['foreign_non_western'] = np.nan

fo = foreign_origin[['country', 'foreign_non_western']].dropna()
table1 = table1.merge(fo, on='country', how='left')

# --- Row 8: Life satisfaction (skip row 7 Social Trust - needs WVS) ---
bli = pd.read_csv(os.path.join(TABLE1_DIR, 'better_life_index.csv'))
life = bli[['LOCATION', 'OBS_VALUE']].rename(
    columns={'LOCATION': 'ref_area', 'OBS_VALUE': 'life_satisfy'})
table1 = table1.merge(life, on='ref_area', how='left')

# --- Rows 9-13: Productivity ---
table1 = table1.merge(prod, on='ref_area', how='left')

# --- Rows 14-15: Labor market ---
# Labor DTA has EMP and LF for individual countries, need OECD average
lab = labor.copy()
# OECD average from Stata code: collapse (mean) EMP LF, then gen ref_area = "OECD"
lab_countries = lab[lab.ref_area != 'OECD'].copy()
oecd_lab = pd.DataFrame({
    'ref_area': ['OECD'],
    'EMP': [lab_countries['EMP'].mean()],
    'LF': [lab_countries['LF'].mean()]
})
lab = pd.concat([lab[lab.ref_area.isin(['DNK', 'FIN', 'NOR', 'SWE', 'GBR', 'USA'])],
                 oecd_lab], ignore_index=True)

table1 = table1.merge(lab, on='ref_area', how='left')
# employment_rate = EMP / working_age_pop_millions / 10^6
# But working_age_pop is already in millions, so EMP must be in absolute numbers
# From the DTA: labor.dta has EMP and LF which are raw values with unit_mult applied
# The Stata code: gen employment_rate = EMP / working_age_pop_millions / 10^6
table1['employment_rate'] = table1['EMP'] / (table1['working_age_pop_millions'] * 1e6)
table1['labor_force_participation'] = table1['LF'] / (table1['working_age_pop_millions'] * 1e6)

# --- Row 16: Public sector employment ---
public = pd.read_excel(os.path.join(TABLE1_DIR, 'public.xlsx'),
                        header=0, skiprows=range(0, 33), nrows=38)
# Columns: [country_name, ISO_code, 2019_value, 2021_value]
# Stata reads B:D = [ISO_code, 2019, 2021], drops C (2019), uses D (2021)
pub_cols = public.columns.tolist()
pub = public[[pub_cols[1], pub_cols[3]]].copy()
pub.columns = ['ref_area', 'public_employment']
pub['public_employment'] = pd.to_numeric(pub['public_employment'], errors='coerce') / 100
table1 = table1.merge(pub, on='ref_area', how='left')

# --- Nordic Average (population-weighted) ---
nordic_mask = table1.cntry.isin([208, 246, 578, 752])
nordic = table1[nordic_mask].copy()
pop_weights = nordic['population_millions'].values
total_pop = pop_weights.sum()

nordic_avg = {}
for col in ['population_millions', 'working_age_pop_millions', 'dependency_ratio_tens',
            'fertility', 'foreign_born_pct', 'foreign_non_western', 'life_satisfy',
            'gdp_billions', 'gdp_thousands_per_capita', 'gdp_per_hour',
            'hour_per_capita', 'hour_per_worker',
            'labor_force_participation', 'employment_rate', 'public_employment']:
    if col in nordic.columns and nordic[col].notna().any():
        nordic_avg[col] = np.average(nordic[col].dropna(), weights=pop_weights[:len(nordic[col].dropna())])

nordic_avg['ref_area'] = 'NORDIC'
nordic_avg['country'] = 'Nordic Average'
nordic_avg['cntry'] = 0

nordic_row = pd.DataFrame([nordic_avg])
table1 = pd.concat([nordic_row, table1], ignore_index=True)
table1 = table1.sort_values('cntry').reset_index(drop=True)

# --- Display results ---
display_cols = {
    'Population (millions)': 'population_millions',
    'Working-age pop (millions)': 'working_age_pop_millions',
    'Old-age dependency ratio': 'dependency_ratio_tens',
    'Fertility rate': 'fertility',
    'Foreign born (%)': 'foreign_born_pct',
    'Non-western immigrants (%)': 'foreign_non_western',
    'Life satisfaction': 'life_satisfy',
    'GDP (billions)': 'gdp_billions',
    'GDP per capita (thousands)': 'gdp_thousands_per_capita',
    'GDP per hour': 'gdp_per_hour',
    'Hours per capita': 'hour_per_capita',
    'Hours per worker': 'hour_per_worker',
    'Labor force participation': 'labor_force_participation',
    'Employment rate': 'employment_rate',
    'Public sector employment': 'public_employment',
}

# Published values for comparison
published = {
    'Nordic Average': {'population_millions': 7.4, 'working_age_pop_millions': 4.2,
                       'dependency_ratio_tens': 35, 'fertility': 1.6,
                       'foreign_born_pct': 14, 'foreign_non_western': 46,
                       'life_satisfy': 7.5,
                       'gdp_billions': 384, 'gdp_thousands_per_capita': 53,
                       'gdp_per_hour': 72, 'hour_per_capita': 727, 'hour_per_worker': 1446,
                       'labor_force_participation': 0.91, 'employment_rate': 0.85,
                       'public_employment': 0.29},
    'Denmark': {'population_millions': 5.8, 'working_age_pop_millions': 3.4,
                'dependency_ratio_tens': 34, 'fertility': 1.7,
                'foreign_born_pct': 10, 'foreign_non_western': 43,
                'life_satisfy': 7.5,
                'gdp_billions': 306, 'gdp_thousands_per_capita': 53,
                'gdp_per_hour': 74, 'hour_per_capita': 709, 'hour_per_worker': 1371,
                'labor_force_participation': 0.89, 'employment_rate': 0.85,
                'public_employment': 0.28},
    'United States': {'population_millions': 328.3, 'working_age_pop_millions': 192.6,
                      'dependency_ratio_tens': 28, 'fertility': 1.7,
                      'foreign_born_pct': 14, 'foreign_non_western': 87,
                      'life_satisfy': 7.0,
                      'gdp_billions': 20137, 'gdp_thousands_per_capita': 61,
                      'gdp_per_hour': 72, 'hour_per_capita': 842, 'hour_per_worker': 1742,
                      'labor_force_participation': 0.84, 'employment_rate': 0.81,
                      'public_employment': 0.15},
    'OECD': {'population_millions': 35.9, 'working_age_pop_millions': 21.1,
             'dependency_ratio_tens': 29, 'fertility': 1.6,
             'foreign_born_pct': 14, 'foreign_non_western': 65,
             'life_satisfy': 6.7,
             'gdp_billions': 1548, 'gdp_thousands_per_capita': 43,
             'gdp_per_hour': 52, 'hour_per_capita': 825, 'hour_per_worker': 1742,
             'labor_force_participation': 0.79, 'employment_rate': 0.74,
             'public_employment': 0.19},
}

countries_to_show = ['Nordic Average', 'Denmark', 'Finland', 'Norway', 'Sweden',
                     'United Kingdom', 'United States', 'OECD']

print(f'\n{"Variable":<30}', end='')
for c in countries_to_show:
    label = c[:8] if len(c) > 8 else c
    print(f'{label:>10}', end='')
print()
print('-' * (30 + 10 * len(countries_to_show)))

for label, col in display_cols.items():
    print(f'{label:<30}', end='')
    for c in countries_to_show:
        row = table1[table1.country == c]
        if len(row) == 0:
            print(f'{"N/A":>10}', end='')
            continue
        val = row[col].values[0] if col in row.columns else np.nan
        if pd.isna(val):
            print(f'{"N/A":>10}', end='')
        elif col in ['labor_force_participation', 'employment_rate', 'public_employment']:
            print(f'{val:>10.2f}', end='')
        elif col in ['fertility', 'life_satisfy', 'population_millions', 'working_age_pop_millions']:
            print(f'{val:>10.1f}', end='')
        else:
            print(f'{val:>10.0f}', end='')
    print()

# Comparison with published values
print('\nComparison with published Table 1:')
print('-' * 70)
matches = 0
mismatches = 0
for c_name, pub_vals in published.items():
    row = table1[table1.country == c_name]
    if len(row) == 0:
        continue
    for col, pub_val in pub_vals.items():
        if col not in row.columns:
            continue
        repl_val = row[col].values[0]
        if pd.isna(repl_val):
            continue
        # Round to same precision as published
        if col in ['labor_force_participation', 'employment_rate', 'public_employment']:
            match = abs(round(repl_val, 2) - pub_val) < 0.015
        elif col in ['fertility', 'life_satisfy', 'population_millions', 'working_age_pop_millions']:
            match = abs(round(repl_val, 1) - pub_val) < 0.15
        else:
            match = abs(round(repl_val) - pub_val) <= 1
        if match:
            matches += 1
        else:
            mismatches += 1
            print(f'  MISMATCH: {c_name} / {col}: repl={repl_val:.4f}, pub={pub_val}')

print(f'\n  Matches: {matches}, Mismatches: {mismatches}')
print(f'  Note: Social trust row skipped (requires WVS/EVS + PIAAC data)')

# Save Table 1
table1.to_parquet(os.path.join(OUTPUT_DIR, 'table1.parquet'), index=False)


# ══════════════════════════════════════════════════════════════════════
# Table 3: Gini Coefficients (columns 1-3)
# ══════════════════════════════════════════════════════════════════════
print('\n\nTABLE 3: Gini Coefficients')
print('=' * 70)

gini = pd.read_csv(os.path.join(TABLE3_DIR, 'OECD_income_gini_poverty.csv'))

# Keep only GINI and GINIB measures, year 2019
gini = gini[gini['MEASURE'].isin(['GINI', 'GINIB'])]
gini = gini[gini['TIME'] == 2019]
gini = gini[['Country', 'MEASURE', 'Value']].copy()

# Reshape: one row per country with GINI and GINIB columns
gini_wide = gini.pivot_table(index='Country', columns='MEASURE', values='Value').reset_index()

# Compute OECD average (equal-weight mean of all countries)
oecd_gini = gini_wide[['GINI', 'GINIB']].mean()
oecd_row = pd.DataFrame({'Country': ['OECD Average'], 'GINI': [oecd_gini['GINI']],
                          'GINIB': [oecd_gini['GINIB']]})
gini_wide = pd.concat([gini_wide, oecd_row], ignore_index=True)

# Nordic average (population-weighted)
nordic_names = {'Denmark': 5814461, 'Finland': 5521605, 'Norway': 5347893, 'Sweden': 10278888}
nordic_gini = gini_wide[gini_wide.Country.isin(nordic_names.keys())].copy()
nordic_gini['POP'] = nordic_gini.Country.map(nordic_names)
nordic_wt_gini = np.average(nordic_gini['GINI'], weights=nordic_gini['POP'])
nordic_wt_ginib = np.average(nordic_gini['GINIB'], weights=nordic_gini['POP'])
nordic_row = pd.DataFrame({'Country': ['Nordic Countries'], 'GINI': [nordic_wt_gini],
                            'GINIB': [nordic_wt_ginib]})
gini_wide = pd.concat([gini_wide, nordic_row], ignore_index=True)

# Compute difference
gini_wide['gini_diff'] = gini_wide['GINI'] - gini_wide['GINIB']

# Keep relevant countries
keep_countries = ['Nordic Countries', 'Denmark', 'Finland', 'Norway', 'Sweden',
                  'United Kingdom', 'United States', 'OECD Average']
table3 = gini_wide[gini_wide.Country.isin(keep_countries)].copy()

# Order
order_map = {c: i for i, c in enumerate(keep_countries)}
table3['order'] = table3.Country.map(order_map)
table3 = table3.sort_values('order').reset_index(drop=True)

# Display
print(f'\n{"Country":<20} {"Gini disp":>12} {"Gini mkt":>12} {"Difference":>12}')
print('-' * 56)
for _, row in table3.iterrows():
    print(f'{row["Country"]:<20} {row["GINI"]:>12.2f} {row["GINIB"]:>12.2f} {row["gini_diff"]:>12.2f}')

# Published comparison
pub_table3 = {
    'Nordic Countries': (0.27, 0.39, -0.12),
    'Denmark': (0.27, 0.40, -0.13),
    'Finland': (0.28, 0.43, -0.15),
    'Norway': (0.27, 0.39, -0.11),
    'Sweden': (0.27, 0.36, -0.09),
    'United Kingdom': (0.36, 0.45, -0.09),
    'United States': (0.39, 0.47, -0.08),
    'OECD Average': (0.31, 0.41, -0.10),
}

print('\nComparison with published Table 3:')
print('-' * 70)
print(f'{"Country":<20} {"Pub GINI":>10} {"Repl GINI":>10} {"Pub GINIB":>10} {"Repl GINIB":>10} {"Match?":>8}')
t3_matches = 0
t3_mismatches = 0
for country, (pub_gini, pub_ginib, pub_diff) in pub_table3.items():
    row = table3[table3.Country == country]
    if len(row) == 0:
        continue
    r_gini = round(row['GINI'].values[0], 2)
    r_ginib = round(row['GINIB'].values[0], 2)
    r_diff = round(row['gini_diff'].values[0], 2)
    match_g = abs(r_gini - pub_gini) < 0.015
    match_gb = abs(r_ginib - pub_ginib) < 0.015
    match_d = abs(r_diff - pub_diff) < 0.015
    all_match = match_g and match_gb and match_d
    status = '✓' if all_match else 'MISMATCH'
    if all_match:
        t3_matches += 1
    else:
        t3_mismatches += 1
    print(f'{country:<20} {pub_gini:>10.2f} {r_gini:>10.2f} {pub_ginib:>10.2f} {r_ginib:>10.2f} {status:>8}')

print(f'\n  Matches: {t3_matches}/8, Mismatches: {t3_mismatches}')
print(f'  Note: Columns 4-5 (Gini earnings, Var log earnings) require PIAAC data — not replicated')

table3.to_parquet(os.path.join(OUTPUT_DIR, 'table3.parquet'), index=False)

print('\n' + '=' * 70)
print('02_tables.py — DONE')
print('=' * 70)
