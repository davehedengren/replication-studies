"""
01_build.py - Build the full data pipeline for paper 199083
Kantor & Whalley - Moonshot: Public R&D and Growth

Replicates 18 Stata build scripts to produce 3.compile.dta equivalent.
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
import os, sys, gc, time, warnings
warnings.filterwarnings('ignore')

PROJ = '/Users/davehedengren/code/replication_studies/199083-V1'
RAW = f'{PROJ}/data/raw'
PROC = f'{PROJ}/data/process'

def ensure_dir(path):
    os.makedirs(os.path.dirname(path) if '.' in os.path.basename(path) else path, exist_ok=True)

for d in ['manufacturing', 'patents', 'spending', 'final']:
    ensure_dir(f'{PROC}/{d}/')

def load_dta(path, **kwargs):
    return pd.read_stata(path, convert_categoricals=False, **kwargs)

# ==============================================================================
# PHASE 1: PATENT SIMILARITY (Script 0, Part A only)
# Creates county x SIC similarity scores from pre-1958 patents
# ==============================================================================
print("=" * 70)
print("PHASE 1: PATENT SIMILARITY (Script 0 Part A)")
print("=" * 70)
t0 = time.time()

# Load 1.2GB patent similarity data
print("Loading nasa_regression_data_new_var_extended.dta (1.2GB)...")
sim_cols_stemmed = [
    'cos_sim_stemmed_1958_8_19', 'cos_sim_stemmed_1959_11_3', 'cos_sim_stemmed_1959_9_8',
    'cos_sim_stemmed_1961_12_29', 'cos_sim_stemmed_1961_4_25', 'cos_sim_stemmed_1962_12_5',
    'cos_sim_stemmed_1965_1_27', 'cos_sim_stemmed_1967_3_2', 'cos_sim_stemmed_1969_6_19',
    'cos_sim_stemmed_1969_6_23', 'cos_sim_stemmed_1970_3_26', 'cos_sim_stemmed_1971_7_1',
    'cos_sim_stemmed_1973_12_20', 'cos_sim_stemmed_1974_10_15', 'cos_sim_stemmed_1975_11_15',
    'cos_sim_stemmed_1980_8_6', 'cos_sim_stemmed_1982_10_15', 'cos_sim_stemmed_1983_7_15i',
    'cos_sim_stemmed_1983_7_15ii', 'cos_sim_stemmed_1984_11_15', 'cos_sim_stemmed_1984_12_15',
    'cos_sim_stemmed_1985_12_15', 'cos_sim_stemmed_1985_12_15b', 'cos_sim_stemmed_1985_8_15',
    'cos_sim_stemmed_1986_3_15', 'cos_sim_stemmed_1987_6_15', 'cos_sim_stemmed_1988_9_15',
    'cos_sim_stemmed_1991_8_8',
]
sim_cols_unstemmed = [c.replace('stemmed', 'unstemmed') for c in sim_cols_stemmed]
era_cols_stemmed = [
    'cos_sim_stemmed_intel_1963_era', 'cos_sim_stemmed_intel_1967_era',
    'cos_sim_stemmed_intel_1972_era', 'cos_sim_stemmed_intel_1977_era',
    'cos_sim_stemmed_intel_1982_era', 'cos_sim_stemmed_intel_1987_era',
    'cos_sim_stemmed_intel_1992_era',
]
era_cols_unstemmed = [c.replace('stemmed', 'unstemmed') for c in era_cols_stemmed]
lim_cols = [
    'cos_sim_stemmed_1962_12_5', 'cos_sim_stemmed_1965_1_27', 'cos_sim_stemmed_1967_3_2',
    'cos_sim_stemmed_1969_6_19', 'cos_sim_stemmed_1970_3_26', 'cos_sim_stemmed_1971_7_1',
    'cos_sim_stemmed_1973_12_20', 'cos_sim_stemmed_1975_11_15', 'cos_sim_stemmed_1983_7_15i',
    'cos_sim_stemmed_1983_7_15ii', 'cos_sim_stemmed_1985_12_15', 'cos_sim_stemmed_1985_12_15b',
    'cos_sim_stemmed_1988_9_15',
]

needed_cols = list(set(
    sim_cols_stemmed + sim_cols_unstemmed + era_cols_stemmed + era_cols_unstemmed + lim_cols +
    ['grant_year', 'inv_fip', 'inv_fips', 'sic2_1972',
     'cos_sim_stemmed_intel_all', 'cos_sim_unstemmed_intel_all']
))

# Try to load with column selection
try:
    pat = load_dta(f'{RAW}/patents/arthur_intel_similar/nasa_regression_data_new_var_extended.dta',
                   columns=needed_cols)
except Exception:
    # Fallback: load all columns
    pat = load_dta(f'{RAW}/patents/arthur_intel_similar/nasa_regression_data_new_var_extended.dta')
    # Keep only needed cols that exist
    existing = [c for c in needed_cols if c in pat.columns]
    pat = pat[existing]

print(f"  Loaded: {len(pat)} patents, {pat.shape[1]} columns")

# Keep pre-1958 patents
pat = pat[pat['grant_year'] < 1958].copy()
print(f"  Pre-1958: {len(pat)} patents")

# Compute patent-level similarity scores
# post_sv2_stemmed = average of 28 NASA document similarities (stemmed, v2)
existing_stemmed = [c for c in sim_cols_stemmed if c in pat.columns]
pat['post_sv2_stemmed'] = pat[existing_stemmed].mean(axis=1)

existing_unstemmed = [c for c in sim_cols_unstemmed if c in pat.columns]
if existing_unstemmed:
    pat['post_sv2_unstemmed'] = pat[existing_unstemmed].mean(axis=1)

existing_era_s = [c for c in era_cols_stemmed if c in pat.columns]
if existing_era_s:
    pat['post_s_stemmed'] = pat[existing_era_s].mean(axis=1)

existing_era_u = [c for c in era_cols_unstemmed if c in pat.columns]
if existing_era_u:
    pat['post_s_unstemmed'] = pat[existing_era_u].mean(axis=1)

existing_lim = [c for c in lim_cols if c in pat.columns]
if existing_lim:
    pat['post_sv2_lim_stemmed'] = pat[existing_lim].mean(axis=1)

# Get county and SIC identifiers
fips_col = 'inv_fip' if 'inv_fip' in pat.columns else 'inv_fips'
sic_col = 'sic2_1972' if 'sic2_1972' in pat.columns else 'sic2'
pat = pat.rename(columns={fips_col: 'fips', sic_col: 'sic2'})
pat['fips'] = pd.to_numeric(pat['fips'], errors='coerce')
pat['sic2'] = pd.to_numeric(pat['sic2'], errors='coerce')
pat = pat.dropna(subset=['fips', 'sic2'])
pat['fips'] = pat['fips'].astype(int)
pat['sic2'] = pat['sic2'].astype(int)

# Aggregate to county x SIC level
print("  Aggregating to county x SIC level...")
agg = pat.groupby(['fips', 'sic2']).agg(
    mstot_patent58=('post_sv2_stemmed', 'size'),
    mspost_sv2_stemmed=('post_sv2_stemmed', 'median'),
    mspost_sv2_unstemmed=('post_sv2_unstemmed', 'median') if 'post_sv2_unstemmed' in pat.columns else ('post_sv2_stemmed', 'median'),
    mspost_s_stemmed=('post_s_stemmed', 'mean') if 'post_s_stemmed' in pat.columns else ('post_sv2_stemmed', 'mean'),
    mspost_s_unstemmed=('post_s_unstemmed', 'mean') if 'post_s_unstemmed' in pat.columns else ('post_sv2_stemmed', 'mean'),
    mspost_sv2_lim_stemmed=('post_sv2_lim_stemmed', 'mean') if 'post_sv2_lim_stemmed' in pat.columns else ('post_sv2_stemmed', 'mean'),
).reset_index()

if 'cos_sim_stemmed_intel_all' in pat.columns:
    agg2 = pat.groupby(['fips', 'sic2']).agg(
        mscos_sim_stemmed_all=('cos_sim_stemmed_intel_all', 'mean'),
    ).reset_index()
    agg = agg.merge(agg2, on=['fips', 'sic2'], how='left')

if 'cos_sim_unstemmed_intel_all' in pat.columns:
    agg3 = pat.groupby(['fips', 'sic2']).agg(
        mscos_sim_unstemmed_all=('cos_sim_unstemmed_intel_all', 'mean'),
    ).reset_index()
    agg = agg.merge(agg3, on=['fips', 'sic2'], how='left')

# Percentiles
pct = pat.groupby(['fips', 'sic2'])['post_sv2_stemmed'].quantile([0.75, 0.90]).unstack()
pct.columns = ['p75spost_sv2_stemmed', 'p90spost_sv2_stemmed']
pct = pct.reset_index()
agg = agg.merge(pct, on=['fips', 'sic2'], how='left')

# Create above-median indicator
median_val = agg['mspost_sv2_stemmed'].median()
agg['amspost_sv2_stemmed'] = (agg['mspost_sv2_stemmed'] >= median_val).astype(int)
print(f"  Above-median space capability: {agg['amspost_sv2_stemmed'].sum()} / {len(agg)} county-SIC cells")

# Merge with blank matrix to get all county-SIC combinations
try:
    blank = load_dta(f'{RAW}/locations/county_sic2_blank_matrix.dta')
    blank = blank.rename(columns={'sic': 'sic2'} if 'sic' in blank.columns else {})
    blank['fips'] = pd.to_numeric(blank['fips'], errors='coerce').astype('Int64')
    blank['sic2'] = pd.to_numeric(blank['sic2'], errors='coerce').astype('Int64')
    blank = blank.dropna(subset=['fips', 'sic2'])
    sim = blank.merge(agg, on=['fips', 'sic2'], how='left')
    sim['mstot_patent58'] = sim['mstot_patent58'].fillna(0)
except Exception as e:
    print(f"  Warning: Could not load blank matrix: {e}")
    sim = agg.copy()

sim = sim.rename(columns={'sic2': 'sic'})
sim.to_pickle(f'{PROC}/patents/0.patent_similarity.pkl')
print(f"  Saved patent similarity: {len(sim)} rows, {sim.shape[1]} cols")

del pat; gc.collect()
print(f"  Phase 1 complete: {time.time()-t0:.0f}s")

# ==============================================================================
# PHASE 2: MANUFACTURING PANEL (Scripts 1-3)
# ==============================================================================
print("\n" + "=" * 70)
print("PHASE 2: MANUFACTURING PANEL")
print("=" * 70)
t0 = time.time()

# Load crosswalk data
icpsr_fips = load_dta(f'{RAW}/geo_cross_walk/icpsr_to_fips_county.dta')
icpsr_fips['state'] = pd.to_numeric(icpsr_fips['state'], errors='coerce')
icpsr_fips['county'] = pd.to_numeric(icpsr_fips['county'], errors='coerce')
icpsr_fips['fips'] = pd.to_numeric(icpsr_fips['fips'], errors='coerce')

# Load 1958 county-SIC list (defines the panel universe)
fips_sic_list = load_dta(f'{RAW}/manufacturing/1958_man_counties_in_carl_taylor_data_fips_sic_list.dta')
fips_sic_list['fips'] = pd.to_numeric(fips_sic_list['fips'], errors='coerce')
fips_sic_list['sic'] = pd.to_numeric(fips_sic_list['sic'], errors='coerce')
fips_sic_list = fips_sic_list.dropna(subset=['fips', 'sic'])
fips_sic_list = fips_sic_list[['fips', 'sic']].drop_duplicates()

def read_county_year(filepath, year, var_map, icpsr_merge=True, fips_formula=None, state_is_fips=False):
    """Read a county manufacturing census file and standardize."""
    df = load_dta(filepath)
    # Destring everything and ensure float64 to avoid int8 overflow
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce').astype('float64')

    if fips_formula == '1972':
        df = df.dropna(subset=['state', 'county'])
        df['fips'] = (1000 * df['state'] + df['county'] / 10).round().astype(int)
    elif fips_formula == '1977':
        df = df.dropna(subset=['state', 'county'])
        df['fips'] = (1000 * df['state'] + df['county'] / 10).round().astype(int)
    elif fips_formula == '1982':
        df = df.dropna(subset=['fipsst', 'fipsct'])
        df['fips'] = (1000 * df['fipsst'] + df['fipsct']).round().astype(int)
    elif fips_formula == '1987_1992':
        df['st'] = pd.to_numeric(df.get('st', df.get('ST', pd.Series())), errors='coerce')
        df['county_num'] = pd.to_numeric(df.get('county', df.get('COUNTY', pd.Series())), errors='coerce')
        df = df.dropna(subset=['st', 'county_num'])
        df['fips'] = (1000 * df['st'] + df['county_num']).round().astype(int)
    elif icpsr_merge:
        # Need ICPSR to FIPS crosswalk
        if 'icpsr_county_codes' in df.columns:
            df = df.rename(columns={'icpsr_county_codes': 'county'})
            state_col = 'icpsr_state_codes' if 'icpsr_state_codes' in df.columns else 'icpsr_state'
            if state_is_fips:
                # State codes are FIPS → convert to ICPSR via state crosswalk
                state_cw = load_dta(f'{RAW}/geo_cross_walk/icpsr_to_fips_state.dta')
                state_cw['state_fips'] = pd.to_numeric(state_cw['state_fips'], errors='coerce')
                state_cw['state'] = pd.to_numeric(state_cw['state'], errors='coerce')
                df = df.rename(columns={state_col: 'state_fips'})
                df['state_fips'] = pd.to_numeric(df['state_fips'], errors='coerce')
                df = df.merge(state_cw[['state_fips', 'state']].drop_duplicates(),
                              on='state_fips', how='inner')
            else:
                df = df.rename(columns={state_col: 'state'})
        df['state'] = pd.to_numeric(df['state'], errors='coerce')
        df['county'] = pd.to_numeric(df['county'], errors='coerce')
        df = df.merge(icpsr_fips[['state', 'county', 'fips']].drop_duplicates(),
                      on=['state', 'county'], how='inner')

    # Get SIC code
    for sic_col in ['industry_no', 'sic_code', 'Indcode', 'classcode', 'sickey', 'sic']:
        if sic_col in df.columns:
            df['sic'] = pd.to_numeric(df[sic_col], errors='coerce')
            break

    df = df.dropna(subset=['fips', 'sic'])
    df['fips'] = df['fips'].round().astype(int)
    df['sic'] = df['sic'].round().astype(int)

    # Merge with 1958 panel universe
    df = df.merge(fips_sic_list, on=['fips', 'sic'], how='inner')

    # Apply variable mapping
    result = pd.DataFrame({'year': year, 'fips': df['fips'], 'sic': df['sic']})
    for target, source in var_map.items():
        if source in df.columns:
            result[target] = pd.to_numeric(df[source], errors='coerce')
        else:
            result[target] = np.nan

    return result

# Read each census year
print("Reading county manufacturing data...")
years_data = []

# 1947
try:
    d = read_county_year(
        f'{RAW}/manufacturing/census_by_geography/data_from_pdfs/county_cleaned/1947county clean and fixed.dta',
        1947, {
            'mval_add': 'value_added_by_manufacture',
            'memp': 'number_of_employees__avg_for_the',
            'mpayroll': 'all_employees_total_wages',
            'mestab': 'number_of_establishments',
            'minvest': 'expenditures_for_new_plant_and_e',
            'memp_pw': 'number_of_production_workers__av',
            'mpayroll_pw': 'total_wages_of_production_worker',
        })
    years_data.append(d)
    print(f"  1947: {len(d)} obs")
except Exception as e:
    print(f"  1947: ERROR - {e}")

# 1954
try:
    d = read_county_year(
        f'{RAW}/manufacturing/census_by_geography/data_from_pdfs/county_cleaned/1954county clean and fixed.dta',
        1954, {
            'mval_add': 'value_added_by_manufacture___1_0',
            'memp': 'number_of_employees',
            'mpayroll': 'payroll_of_employees___1_000_',
            'mestab': 'number_of_establishments',
            'minvest': 'capital_expenditures__new__1_000',
            'memp_pw': 'number_of_production_workers',
            'mpayroll_pw': 'wages_of_production_workers___1_',
        })
    years_data.append(d)
    print(f"  1954: {len(d)} obs")
except Exception as e:
    print(f"  1954: ERROR - {e}")

# 1958
try:
    d = read_county_year(
        f'{RAW}/manufacturing/census_by_geography/data_from_pdfs/county_cleaned/1958 clean and fixed county.dta',
        1958, {
            'mval_add': 'value_added_by_manufacture___1_0',
            'memp': 'number_of_employees',
            'mpayroll': 'payroll_of_employees___1_000_',
            'mestab': 'number_of_establishments',
            'minvest': 'capital_expenditures__new__1_000',
            'memp_pw': 'number_of_production_workers',
            'mpayroll_pw': 'wages_of_production_workers___1_',
        }, state_is_fips=True)
    years_data.append(d)
    print(f"  1958: {len(d)} obs")
except Exception as e:
    print(f"  1958: ERROR - {e}")

# 1963
try:
    d = read_county_year(
        f'{RAW}/manufacturing/census_by_geography/data_from_pdfs/county_cleaned/1963 clean and fixed county.dta',
        1963, {
            'mval_add': 'value_added_by_manufacture___1_0',
            'memp': 'number_of_employees',
            'mpayroll': 'payroll_of_employees___1_000_',
            'mestab': 'number_of_establishments',
            'minvest': 'capital_expenditures__new__1_000',
            'memp_pw': 'number_of_production_workers',
            'mpayroll_pw': 'wages_of_production_workers___1_',
        }, state_is_fips=True)
    years_data.append(d)
    print(f"  1963: {len(d)} obs")
except Exception as e:
    print(f"  1963: ERROR - {e}")

# 1967
try:
    d = read_county_year(
        f'{RAW}/manufacturing/census_by_geography/data_from_pdfs/county_cleaned/1967 clean and fixed county.dta',
        1967, {
            'mval_add': 'value_added_by_manufacture___mil',
            'memp': 'number_of_employees__1000',
            'mpayroll': 'payroll_of_employees___mil_',
            'mestab': 'number_of_establishments',
            'minvest': 'capital_expenditures__new__mil_',
            'memp_pw': 'number_of_production_workers__10',
            'mpayroll_pw': 'wages_of_production_workers___mi',
            'mval_tot': 'value_of_shipments__mil_',
        }, state_is_fips=True)
    years_data.append(d)
    print(f"  1967: {len(d)} obs")
except Exception as e:
    print(f"  1967: ERROR - {e}")

# 1972
try:
    d = read_county_year(
        f'{RAW}/manufacturing/census_by_geography/data_from_pdfs/county_cleaned/1972 clean and fixed county.dta',
        1972, {
            'mval_add': 'value_added',
            'memp': 'emp',
            'mpayroll': 'payroll',
            'mestab': 'estabs',
            'minvest': 'capital_inv',
            'memp_pw': 'emp_pw',
            'mpayroll_pw': 'payroll_pw',
            'mval_tot': 'value_shipments',
        }, icpsr_merge=False, fips_formula='1972')
    years_data.append(d)
    print(f"  1972: {len(d)} obs")
except Exception as e:
    print(f"  1972: ERROR - {e}")

# 1977
try:
    d = read_county_year(
        f'{RAW}/manufacturing/census_by_geography/data_from_pdfs/county_cleaned/1977 clean and fixed county.dta',
        1977, {
            'mval_add': 'value_added',
            'memp': 'emp',
            'mpayroll': 'payroll',
            'mestab': 'estabs',
            'minvest': 'capital_inv',
            'memp_pw': 'emp_pw',
            'mpayroll_pw': 'payroll_pw',
            'mval_tot': 'value_shipments',
        }, icpsr_merge=False, fips_formula='1977')
    years_data.append(d)
    print(f"  1977: {len(d)} obs")
except Exception as e:
    print(f"  1977: ERROR - {e}")

# 1982
try:
    d = read_county_year(
        f'{RAW}/manufacturing/census_by_geography/data_from_electronic_sources/stata/county_all/1982_data.dta',
        1982, {
            'mval_add': 'valueadd',
            'memp': 'emp',
            'mpayroll': 'payroll',
            'mestab': 'totalest',
            'minvest': 'newcapital',
            'memp_pw': 'prodwork',
            'mpayroll_pw': 'wages',
            'mval_tot': 'valueship',
        }, icpsr_merge=False, fips_formula='1982')
    years_data.append(d)
    print(f"  1982: {len(d)} obs")
except Exception as e:
    print(f"  1982: ERROR - {e}")

# 1987
try:
    df87 = load_dta(f'{RAW}/manufacturing/census_by_geography/data_from_electronic_sources/stata/county_all/1987_data.dta')
    for c in df87.columns:
        df87[c] = pd.to_numeric(df87[c], errors='coerce').astype('float64')
    df87['sic'] = df87.get('sic', pd.Series(dtype=float))
    df87 = df87.dropna(subset=['st', 'county'])
    df87['fips'] = (1000 * df87['st'] + df87['county']).round().astype(int)
    df87 = df87.merge(fips_sic_list, on=['fips', 'sic'], how='inner')
    d = pd.DataFrame({
        'year': 1987, 'fips': df87['fips'], 'sic': df87['sic'],
        'mval_add': df87.get('valadd'), 'memp': df87.get('employe'),
        'mpayroll': df87.get('payroll'), 'mestab': df87.get('estab'),
        'minvest': df87.get('capexp'), 'memp_pw': df87.get('workers'),
        'mpayroll_pw': df87.get('wages'), 'mval_tot': df87.get('shipmts'),
    })
    years_data.append(d)
    print(f"  1987: {len(d)} obs")
except Exception as e:
    print(f"  1987: ERROR - {e}")

# 1992
try:
    df92 = load_dta(f'{RAW}/manufacturing/census_by_geography/data_from_electronic_sources/stata/county_all/1992_data.dta')
    for c in df92.columns:
        df92[c] = pd.to_numeric(df92[c], errors='coerce').astype('float64')
    if 'sickey' in df92.columns:
        df92['sic'] = pd.to_numeric(df92['sickey'], errors='coerce')
    df92 = df92.dropna(subset=['st', 'county'])
    df92['fips'] = (1000 * df92['st'] + df92['county']).round().astype(int)
    df92 = df92.merge(fips_sic_list, on=['fips', 'sic'], how='inner')
    d = pd.DataFrame({
        'year': 1992, 'fips': df92['fips'], 'sic': df92['sic'],
        'mval_add': df92.get('valadd'), 'memp': df92.get('emp'),
        'mpayroll': df92.get('pay'), 'mestab': df92.get('estab'),
        'minvest': df92.get('capexp'), 'memp_pw': df92.get('workers'),
        'mpayroll_pw': df92.get('wages'), 'mval_tot': df92.get('value'),
    })
    years_data.append(d)
    print(f"  1992: {len(d)} obs")
except Exception as e:
    print(f"  1992: ERROR - {e}")

# Append all years
mfg = pd.concat(years_data, ignore_index=True)
mfg = mfg.groupby(['fips', 'year', 'sic']).sum(min_count=1).reset_index()

# Replace zeros with missing (withheld for confidentiality)
for v in ['mval_add', 'memp', 'mpayroll', 'mestab', 'mval_tot', 'memp_pw', 'mpayroll_pw']:
    if v in mfg.columns:
        mfg.loc[mfg[v] <= 0, v] = np.nan
mfg.loc[mfg['mval_add'].isna(), 'minvest'] = np.nan

# Standardize units to nominal dollars
# Pre-1963: values in $1000s
for v in ['mval_add', 'mpayroll', 'mval_tot', 'mpayroll_pw', 'minvest']:
    if v in mfg.columns:
        mask = (mfg['year'] <= 1963) & (mfg['year'] > 1939)
        mfg.loc[mask, v] = mfg.loc[mask, v] * 1000

# 1967, 1972, 1977, 1987: values in millions
for v in ['mval_add', 'mpayroll', 'mval_tot', 'mpayroll_pw', 'minvest']:
    if v in mfg.columns:
        mask = mfg['year'].isin([1967, 1972, 1977, 1987])
        mfg.loc[mask, v] = mfg.loc[mask, v] * 1_000_000

# 1982: values in $100,000s
for v in ['mval_add', 'mpayroll', 'mval_tot', 'mpayroll_pw', 'minvest']:
    if v in mfg.columns:
        mask = mfg['year'] == 1982
        mfg.loc[mask, v] = mfg.loc[mask, v] * 100_000

# 1992+: values in $1000s
for v in ['mval_add', 'mpayroll', 'mval_tot', 'mpayroll_pw', 'minvest']:
    if v in mfg.columns:
        mask = mfg['year'] >= 1992
        mfg.loc[mask, v] = mfg.loc[mask, v] * 1000

# Employment: 1967/72/77/87 in thousands, 1982 in hundreds
for v in ['memp', 'memp_pw']:
    if v in mfg.columns:
        mask = mfg['year'].isin([1967, 1972, 1977, 1987])
        mfg.loc[mask, v] = mfg.loc[mask, v] * 1000
        mask = mfg['year'] == 1982
        mfg.loc[mask, v] = mfg.loc[mask, v] * 100

# Drop extreme outliers
for v, thresh in [('mval_add', 4e18), ('mpayroll', 4e18), ('minvest', 9.1e19)]:
    if v in mfg.columns:
        mfg = mfg[~((mfg[v] >= thresh) & mfg[v].notna())]

# Non-production worker share
mfg['npw_share'] = (mfg['memp'] - mfg['memp_pw']) / mfg['memp']
mfg['lincome'] = mfg['mpayroll'] / mfg['memp']

print(f"  County manufacturing panel: {len(mfg)} obs across {sorted(mfg['year'].unique())}")

# ---- MSA Fill-In (Script 2): Allocate MSA data to missing counties ----
print("  MSA Fill-In: allocating MSA data to missing county-SIC cells...")

# Step 1: Compute allocation fractions from 1958+1982 avg employment
alloc_base = mfg[mfg['year'].isin([1958, 1982])][['fips', 'sic', 'memp']].copy()
alloc_base['memp'] = alloc_base['memp'].fillna(0)
alloc_base = alloc_base.groupby(['fips', 'sic'])['memp'].sum().reset_index()
alloc_base['avg_memp'] = alloc_base['memp'] / 2
alloc_base = alloc_base.merge(fips_sic_list, on=['fips', 'sic'], how='inner')
alloc_base['avg_memp'] = alloc_base['avg_memp'].fillna(0)

# Map counties to MSAs
msa_cw = load_dta(f'{RAW}/geo_cross_walk/county_to_MSA_1963_missing_added_no_duplicates_fips_sort.dta')
msa_cw['fips'] = pd.to_numeric(msa_cw['fips'], errors='coerce')
msa_cw['msa_code'] = pd.to_numeric(msa_cw['msa_code'], errors='coerce')
alloc_base = alloc_base.merge(msa_cw[['fips', 'msa_code']].drop_duplicates(), on='fips', how='inner')

# MSA total employment by SIC
msa_tot = alloc_base.groupby(['msa_code', 'sic'])['avg_memp'].sum().reset_index()
msa_tot = msa_tot.rename(columns={'avg_memp': 'msa_avg_memp'})
alloc_base = alloc_base.merge(msa_tot, on=['msa_code', 'sic'], how='left')
alloc_base['allocation_frac'] = np.where(alloc_base['msa_avg_memp'] > 0,
                                          alloc_base['avg_memp'] / alloc_base['msa_avg_memp'], 0)
alloc = alloc_base[['fips', 'sic', 'msa_code', 'allocation_frac']].drop_duplicates()

# Step 2: Read MSA-level data for 1963, 1967, 1972, 1977
msa_var_maps = {
    1963: {
        'file': f'{RAW}/manufacturing/census_by_geography/data_from_pdfs/MSA_cleaned/fixed_final_files/1963msa c&f_edit_2.dta',
        'emp': 'total_number_of_employees', 'payroll': 'all_employees__payroll___1_000_',
        'emp_pw': 'total_number_of_production_worke', 'payroll_pw': 'wages_of_production_workers___1_',
        'va': 'value_added_by_manufacture__adju', 'invest': 'capital_expenditures__new___1_00',
        'estab': 'total_number_of_establishments', 'shipments': None,
        'units_dollar': 1000, 'units_emp': 1,
    },
    1967: {
        'file': f'{RAW}/manufacturing/census_by_geography/data_from_pdfs/MSA_cleaned/fixed_final_files/1967msa c&f_edit.dta',
        'emp': 'number_of_all_employees__1_000_', 'payroll': 'all_employees__payroll__million_',
        'emp_pw': 'number_of_producton_workers__1_0', 'payroll_pw': 'wages_of_productin_workers__mill',
        'va': 'value_added_by_manufacture__mill', 'invest': 'capital_expenditures__new__milli',
        'estab': 'total_number_of_establishments', 'shipments': 'value_of_shipments__million_doll',
        'units_dollar': 1_000_000, 'units_emp': 1000,
    },
    1972: {
        'file': f'{RAW}/manufacturing/census_by_geography/data_from_pdfs/MSA_cleaned/fixed_final_files/1972msa c&f_edit.dta',
        'emp': 'number_of_all_employees__1_000_', 'payroll': 'all_employees__payroll__million_',
        'emp_pw': 'number_of_production_workers__1_', 'payroll_pw': 'production_workers_wages__millio',
        'va': 'value_added_by_manufacture__mill', 'invest': 'capital_expenditures__new__milli',
        'estab': 'total_number_of_establishment', 'shipments': 'value_of_shipments__million_doll',
        'units_dollar': 1_000_000, 'units_emp': 1000,
    },
    1977: {
        'file': f'{RAW}/manufacturing/census_by_geography/data_from_pdfs/MSA_cleaned/fixed_final_files/1977msa c&f_edit.dta',
        'emp': 'number_of_all_employees__1_000_', 'payroll': 'all_employees__payroll__million_',
        'emp_pw': 'number_of_production_workers__1_', 'payroll_pw': 'wages_of_production_workers__mil',
        'va': 'value_added_by_manufacture__mill', 'invest': 'new_capital_expenditures__millio',
        'estab': 'total_number_of_establishments', 'shipments': 'value_of_shipments__million_doll',
        'units_dollar': 1_000_000, 'units_emp': 1000,
    },
}

msa_fill_data = []
for yr, info in msa_var_maps.items():
    try:
        msadf = load_dta(info['file'])

        # Get SIC and MSA code
        msadf['sic'] = pd.to_numeric(msadf['industry_no'], errors='coerce')
        msadf['msa_code'] = pd.to_numeric(msadf['smsa1960_code'], errors='coerce')
        msadf = msadf.dropna(subset=['sic', 'msa_code'])
        msadf['sic'] = msadf['sic'].astype(int)
        msadf = msadf[(msadf['sic'] >= 20) & (msadf['sic'] <= 39)]

        # Extract and convert variables
        for col_name in ['emp', 'payroll', 'emp_pw', 'payroll_pw', 'va', 'invest', 'estab', 'shipments']:
            src = info[col_name]
            if src and src in msadf.columns:
                msadf[col_name] = pd.to_numeric(msadf[src], errors='coerce')
            else:
                msadf[col_name] = np.nan

        msadf = msadf[['msa_code', 'sic', 'emp', 'payroll', 'emp_pw', 'payroll_pw', 'va', 'invest', 'estab', 'shipments']].drop_duplicates()
        msadf = msadf.groupby(['msa_code', 'sic']).first().reset_index()

        # Find missing county-SIC cells for this year
        existing = mfg[mfg['year'] == yr][['fips', 'sic']].drop_duplicates()
        all_cells = fips_sic_list.copy()
        missing = all_cells.merge(existing, on=['fips', 'sic'], how='left', indicator=True)
        missing = missing[missing['_merge'] == 'left_only'].drop(columns='_merge')

        # Merge missing cells with allocation factors and MSA data
        missing = missing.merge(alloc[['fips', 'sic', 'msa_code', 'allocation_frac']], on=['fips', 'sic'], how='inner')
        missing = missing.merge(msadf, on=['msa_code', 'sic'], how='inner')

        # Allocate MSA values to counties
        fill = pd.DataFrame({
            'year': yr, 'fips': missing['fips'], 'sic': missing['sic'],
            'mval_add': missing['allocation_frac'] * missing['va'] * info['units_dollar'],
            'memp': missing['allocation_frac'] * missing['emp'] * info['units_emp'],
            'mpayroll': missing['allocation_frac'] * missing['payroll'] * info['units_dollar'],
            'mestab': missing['allocation_frac'] * missing['estab'],
            'mval_tot': missing['allocation_frac'] * missing['shipments'] * info['units_dollar'] if info['shipments'] else np.nan,
            'minvest': missing['allocation_frac'] * missing['invest'] * info['units_dollar'],
            'memp_pw': missing['allocation_frac'] * missing['emp_pw'] * info['units_emp'],
            'mpayroll_pw': missing['allocation_frac'] * missing['payroll_pw'] * info['units_dollar'],
        })
        # Replace zeros/negatives with NaN
        for v in ['mval_add', 'memp', 'mpayroll', 'mestab', 'mval_tot', 'minvest', 'memp_pw', 'mpayroll_pw']:
            if v in fill.columns:
                fill.loc[fill[v] <= 0, v] = np.nan
        msa_fill_data.append(fill)
        print(f"    {yr}: filled {len(fill)} county-SIC cells from MSA data")
    except Exception as e:
        print(f"    {yr}: MSA fill-in ERROR - {e}")

if msa_fill_data:
    msa_fill = pd.concat(msa_fill_data, ignore_index=True)
    mfg = pd.concat([mfg, msa_fill], ignore_index=True)
    mfg = mfg.drop_duplicates(subset=['fips', 'year', 'sic'], keep='first')
    mfg['npw_share'] = (mfg['memp'] - mfg['memp_pw']) / mfg['memp']
    mfg['lincome'] = mfg['mpayroll'] / mfg['memp']
    print(f"  After MSA fill-in: {len(mfg)} obs")

# ---- Manufacturing script 3: combine and filter ----
print("  Combining and filtering manufacturing panel...")

# Merge with patent similarity (for mstot_patent58 filter)
mfg = mfg.rename(columns={'sic': 'sic2'})
mfg = mfg.merge(sim[['fips', 'sic', 'mstot_patent58']].rename(columns={'sic': 'sic2'}),
                 on=['fips', 'sic2'], how='inner')
mfg = mfg.rename(columns={'sic2': 'sic'})
print(f"    After sim merge: {len(mfg)} obs")

# Merge MSA codes
msa_cw = load_dta(f'{RAW}/geo_cross_walk/county_to_MSA_1963_missing_added_no_duplicates_fips_sort.dta')
msa_cw['fips'] = pd.to_numeric(msa_cw['fips'], errors='coerce')
msa_cw['msa_code'] = pd.to_numeric(msa_cw['msa_code'], errors='coerce')
mfg = mfg.merge(msa_cw[['fips', 'msa_code']].drop_duplicates(), on='fips', how='left')
print(f"    After MSA merge: {len(mfg)} obs, MSA notna: {mfg['msa_code'].notna().sum()}")

# Merge price index and create real variables
try:
    ppi = load_dta(f'{RAW}/price_index/producer_price_index_divide_by.dta')
    ppi['year'] = pd.to_numeric(ppi['year'], errors='coerce')
    mfg = mfg.merge(ppi, on='year', how='left')
    if 'ppindex_58base' in mfg.columns:
        mfg['r58_mval_add'] = mfg['mval_add'] / mfg['ppindex_58base']
        mfg['r58_mpayroll'] = mfg['mpayroll'] / mfg['ppindex_58base']
        mfg['r58_minvest'] = mfg['minvest'] / mfg['ppindex_58base']
        mfg['r58_mpayroll_pw'] = mfg['mpayroll_pw'] / mfg['ppindex_58base']
except Exception as e:
    print(f"  Warning: Price index merge failed: {e}")

# Drop missing value added or employment
mfg = mfg[(mfg['mval_add'].notna()) & (mfg['mval_add'] > 0) &
           (mfg['memp'].notna()) & (mfg['memp'] > 0)]
print(f"    After drop missing va/emp: {len(mfg)} obs")

# Drop post-1992
mfg = mfg[mfg['year'] <= 1992]
print(f"    After drop post-1992: {len(mfg)} obs")

# Drop if MSA missing
mfg = mfg[mfg['msa_code'].notna()]
print(f"    After drop MSA missing: {len(mfg)} obs")

# Observation counter per fips-sic
obs_count = mfg.groupby(['fips', 'sic']).size().reset_index(name='sone')
mfg = mfg.merge(obs_count, on=['fips', 'sic'], how='left')
print(f"    sone distribution: {mfg['sone'].describe()}")
mfg = mfg[mfg['sone'] > 7]
print(f"    After sone > 7: {len(mfg)} obs")

# Require NASA-relevant sectors reporting in 1958
nasa_58 = mfg[(mfg['sic'].isin([36, 37])) & (mfg['year'] == 1958)].groupby('fips').size().reset_index(name='snone58')
mfg = mfg.merge(nasa_58, on='fips', how='left')
print(f"    NASA 58 notna: {mfg['snone58'].notna().sum()} / {len(mfg)}")
mfg = mfg[mfg['snone58'] > 0]
print(f"    After nasa_58: {len(mfg)} obs")

# Positive pre-1958 patent count
mfg = mfg[mfg['mstot_patent58'] > 0]
print(f"    After patent > 0: {len(mfg)} obs")

# Drop singleton states (SD=38, ND=46, WY=56)
mfg['state_fips'] = mfg['fips'] // 1000
mfg = mfg[~mfg['state_fips'].isin([38, 46, 56])]
print(f"    After drop singleton states: {len(mfg)} obs")

# Drop singleton sic-year cells
sic_year_count = mfg.groupby(['sic', 'year']).size().reset_index(name='sy_count')
mfg = mfg.merge(sic_year_count, on=['sic', 'year'], how='left')
mfg = mfg[mfg['sy_count'] > 1]
print(f"    After drop singleton sic-year: {len(mfg)} obs")

# Keep only manufacturing SIC codes (20-39)
mfg = mfg[(mfg['sic'] >= 20) & (mfg['sic'] <= 39)]
print(f"    After SIC 20-39: {len(mfg)} obs")

mfg = mfg.drop(columns=['sone', 'snone58', 'sy_count'], errors='ignore')
mfg = mfg.sort_values(['fips', 'year', 'sic']).reset_index(drop=True)
print(f"  Final manufacturing panel: {len(mfg)} obs, {mfg['fips'].nunique()} counties, {sorted(mfg['year'].unique())}")

mfg.to_pickle(f'{PROC}/3.all_filled_in_clean_2digit.pkl')
print(f"  Phase 2 complete: {time.time()-t0:.0f}s")

# ==============================================================================
# PHASE 3: COMPILE 1 - Capital Stock + TFP
# ==============================================================================
print("\n" + "=" * 70)
print("PHASE 3: CAPITAL STOCK + TFP (Compile 1)")
print("=" * 70)
t0 = time.time()

df = mfg.copy()
df = df.sort_values(['fips', 'sic', 'year']).reset_index(drop=True)

# Rename investment variable
df['cap_inv_new'] = df['minvest']

# Perpetual inventory method: 8% depreciation
# 1958 is initial year
# K_1958 = I_1958 / (delta + g), where g = investment growth rate
df = df.sort_values(['fips', 'sic', 'year'])
df['cap_inv_lag'] = df.groupby(['fips', 'sic'])['cap_inv_new'].shift(1)
df['cap_inv_lead'] = df.groupby(['fips', 'sic'])['cap_inv_new'].shift(-1)

# Growth rate of investment
df['g_cap'] = (df['cap_inv_new'] - df['cap_inv_lag']) / (100 * df['cap_inv_lag'])

# Initial capital (1958)
df['capital'] = np.nan
mask_58 = df['year'] == 1958
df.loc[mask_58, 'capital'] = df.loc[mask_58, 'cap_inv_new'] / (0.08 + df.loc[mask_58, 'g_cap'])
# Fallback: if growth rate makes initial capital negative or missing
mask_58_miss = mask_58 & df['capital'].isna()
df.loc[mask_58_miss, 'capital'] = df.loc[mask_58_miss, 'cap_inv_new'] / 0.08

# Forward: K_t = I_t + (1-delta)^gap * K_{t-1}
# Year gaps: 58->63 (5yr, 2.5 midpoint), 63->67 (4yr, 2), etc.
year_gaps = {1963: 2.5, 1967: 2.0, 1972: 2.5, 1977: 2.5, 1982: 2.5, 1987: 2.5, 1992: 2.5}
delta = 0.08

for yr, gap in year_gaps.items():
    df = df.sort_values(['fips', 'sic', 'year'])
    prev_cap = df[df['year'] == yr - (5 if gap == 2.5 else 4)].set_index(['fips', 'sic'])['capital']
    if prev_cap.empty:
        continue
    mask = df['year'] == yr
    for idx in df[mask].index:
        key = (df.loc[idx, 'fips'], df.loc[idx, 'sic'])
        if key in prev_cap.index and pd.notna(prev_cap[key]):
            k_prev = prev_cap[key]
        else:
            # Impute using SIC-level capital-employment ratio
            ratio = df[df['year'] == yr - (5 if gap == 2.5 else 4)].groupby('sic').apply(
                lambda x: x['memp'].sum() / x['capital'].sum() if x['capital'].sum() > 0 else np.nan
            )
            sic_val = df.loc[idx, 'sic']
            if sic_val in ratio.index and pd.notna(ratio[sic_val]):
                k_prev = df.loc[idx, 'memp'] / ratio[sic_val]
            else:
                continue
        inv = df.loc[idx, 'cap_inv_new']
        if pd.isna(inv):
            inv = 0
        df.loc[idx, 'capital'] = inv + ((1 - delta) ** gap) * k_prev

# Backward: 1954, 1947
# K_1954 = (K_1958 - I_1958_lead) / (1-delta)^2
for yr, gap in [(1954, 2.0), (1947, 3.0)]:
    next_yr = 1958 if yr == 1954 else 1954
    df = df.sort_values(['fips', 'sic', 'year'])
    next_cap = df[df['year'] == next_yr].set_index(['fips', 'sic'])['capital']
    next_inv = df[df['year'] == next_yr].set_index(['fips', 'sic'])['cap_inv_new']
    mask = df['year'] == yr
    for idx in df[mask].index:
        key = (df.loc[idx, 'fips'], df.loc[idx, 'sic'])
        if key in next_cap.index and pd.notna(next_cap[key]):
            k_next = next_cap[key]
            i_next = next_inv.get(key, 0)
            if pd.isna(i_next): i_next = 0
            k = (k_next - i_next) / ((1 - delta) ** gap)
            if k < 0: k = 0
            df.loc[idx, 'capital'] = k

# Interpolate remaining missing with average of neighbors
df = df.sort_values(['fips', 'sic', 'year'])
df['cap_prev'] = df.groupby(['fips', 'sic'])['capital'].shift(1)
df['cap_next'] = df.groupby(['fips', 'sic'])['capital'].shift(-1)
mask = df['capital'].isna()
df.loc[mask, 'capital'] = (df.loc[mask, 'cap_prev'] + df.loc[mask, 'cap_next']) / 2

# Replace zero capital with 1
df.loc[df['capital'] == 0, 'capital'] = 1
df.loc[df['capital'].isna(), 'capital'] = 1

# Generate log variables
df['lcapital'] = np.log(df['capital'])
df['lemp'] = np.log(df['memp'])
df['lva'] = np.log(df['mval_add'])

# Solow residual TFP: reg lva lemp lcapital, predict resid
valid = df[['lva', 'lemp', 'lcapital']].dropna()
X_tfp = sm.add_constant(df.loc[valid.index, ['lemp', 'lcapital']])
y_tfp = df.loc[valid.index, 'lva']
tfp_model = sm.OLS(y_tfp, X_tfp).fit()
df['ltfp'] = np.nan
df.loc[valid.index, 'ltfp'] = tfp_model.resid

print(f"  TFP regression: lva = {tfp_model.params['const']:.3f} + {tfp_model.params['lemp']:.3f}*lemp + {tfp_model.params['lcapital']:.3f}*lcapital")
print(f"  Capital stock: {df['capital'].notna().sum()} non-missing, mean={df['capital'].mean():.0f}")
df = df.drop(columns=['cap_inv_new', 'cap_inv_lag', 'cap_inv_lead', 'g_cap', 'cap_prev', 'cap_next'], errors='ignore')
df.to_pickle(f'{PROC}/final/1.compile.pkl')
print(f"  Phase 3 complete: {time.time()-t0:.0f}s")

# ==============================================================================
# PHASE 4: PATENT COMPILE (Scripts 1-5, 8 - patent counts by type)
# ==============================================================================
print("\n" + "=" * 70)
print("PHASE 4: PATENT LISTS + LOCATIONS + COMPILE")
print("=" * 70)
t0 = time.time()

# --- Patent year file from historical masterfile ---
print("  Loading historical masterfile...")
hm = load_dta(f'{RAW}/patents/historical_masterfile/historical_masterfile.dta',
              columns=['patent', 'disp_dt', 'disp_ty'])
hm['grantyear'] = pd.to_datetime(hm['disp_dt'], errors='coerce').dt.year
hm = hm[hm['disp_ty'] == 'ISS']
hm['patent'] = pd.to_numeric(hm['patent'], errors='coerce')
hm = hm.dropna(subset=['patent', 'grantyear'])
hm = hm[['patent', 'grantyear']].drop_duplicates()
print(f"  Historical masterfile: {len(hm)} patents")

# --- NASA patent lists ---
print("  Building NASA patent list...")
# Fleming assignee list
try:
    nasa_asg_list = load_dta(f'{RAW}/patents/fleming_patents/assignee_lists_by_gov_type/nasa_assignee_list_final.dta')
    fleming = load_dta(f'{RAW}/patents/fleming_patents/uspto.govt.reliance.metadata.dta',
                       columns=['patno', 'assignee', 'nasa', 'dod', 'usg', 'asg_usgovt'])
    fleming['patno'] = pd.to_numeric(fleming['patno'], errors='coerce')
    # NASA assignee patents (pre-1976)
    nasa_asg = fleming.merge(nasa_asg_list, on='assignee', how='inner')
    nasa_asg = nasa_asg[['patno']].rename(columns={'patno': 'patent'}).drop_duplicates()
    nasa_asg = nasa_asg.merge(hm, on='patent', how='inner')
    nasa_asg_pre76 = nasa_asg[nasa_asg['grantyear'] < 1976]['patent'].unique()
    print(f"    NASA assignee (Fleming): {len(nasa_asg_pre76)}")
except Exception as e:
    nasa_asg_pre76 = np.array([])
    print(f"    NASA assignee (Fleming): ERROR - {e}")

# PatentsView NASA assignee (post-1976)
try:
    pv_asg = load_dta(f'{RAW}/patents/USPTO_patentsview/patent_assignee.dta')
    nasa_pv_list = load_dta(f'{RAW}/patents/USPTO_patentsview/assignee_nasa_list.dta')
    id_col = 'assignee_id' if 'assignee_id' in nasa_pv_list.columns else 'id'
    pv_nasa = pv_asg.merge(nasa_pv_list, left_on='id', right_on=id_col, how='inner')
    pv_nasa['patent'] = pd.to_numeric(pv_nasa['patent_id'], errors='coerce')
    nasa_asg_post76 = pv_nasa['patent'].dropna().unique()
    print(f"    NASA assignee (PatentsView): {len(nasa_asg_post76)}")
except Exception as e:
    nasa_asg_post76 = np.array([])
    print(f"    NASA assignee (PatentsView): ERROR - {e}")

nasa_patents = np.union1d(nasa_asg_pre76, nasa_asg_post76)

# NASA funded patents
try:
    # Fleming funded
    nasa_fund_pre = fleming[fleming['nasa'] > 0]['patno'].values
    nasa_fund_pre = np.setdiff1d(nasa_fund_pre, nasa_asg_pre76)
    # PatentsView funded
    pv_gov = load_dta(f'{RAW}/patents/USPTO_patentsview/patent_govintorg.dta')
    nasa_gov_list = load_dta(f'{RAW}/patents/USPTO_patentsview/government_organization_nasa_list.dta')
    pv_nasa_fund = pv_gov.merge(nasa_gov_list, on='organization_id', how='inner')
    pv_nasa_fund['patent'] = pd.to_numeric(pv_nasa_fund['patent_id'], errors='coerce')
    nasa_fund_post = pv_nasa_fund['patent'].dropna().unique()
    nasa_funded = np.union1d(nasa_fund_pre, nasa_fund_post)
except:
    nasa_funded = np.array([])

all_nasa = np.union1d(nasa_patents, nasa_funded)
print(f"  Total NASA patents: {len(all_nasa)}")

# --- Army + Navy patent lists (similar logic, abbreviated) ---
print("  Building Army/Navy/Gov patent lists...")
army_patents = np.array([])
navy_patents = np.array([])
gov_patents = np.array([])
mil_text_patents = np.array([])

try:
    army_list = load_dta(f'{RAW}/patents/fleming_patents/assignee_lists_by_gov_type/army_assignee_list.dta')
    army_asg = fleming.merge(army_list, on='assignee', how='inner')
    army_pre = army_asg['patno'].values
    # PatentsView
    army_pv_list = load_dta(f'{RAW}/patents/USPTO_patentsview/assignee_army_list.dta')
    army_post = pv_asg.merge(army_pv_list, on='id', how='inner')['patent_id'].values
    army_post = pd.to_numeric(pd.Series(army_post), errors='coerce').dropna().values
    army_patents = np.union1d(army_pre, army_post)
    # Funded
    army_gov_list = load_dta(f'{RAW}/patents/USPTO_patentsview/government_organization_army_list.dta')
    army_fund = pv_gov.merge(army_gov_list, on='organization_id', how='inner')
    army_fund_pat = pd.to_numeric(army_fund['patent_id'], errors='coerce').dropna().values
    army_dod = fleming[fleming['dod'] > 0]['patno'].values
    army_patents = np.union1d(army_patents, np.union1d(army_fund_pat, army_dod))
    print(f"    Army: {len(army_patents)}")
except Exception as e:
    print(f"    Army: ERROR - {e}")

try:
    navy_list = load_dta(f'{RAW}/patents/fleming_patents/assignee_lists_by_gov_type/navy_assignee_list.dta')
    navy_asg = fleming.merge(navy_list, on='assignee', how='inner')
    navy_pre = navy_asg['patno'].values
    navy_pv_list = load_dta(f'{RAW}/patents/USPTO_patentsview/assignee_navy_list.dta')
    navy_post = pv_asg.merge(navy_pv_list, on='id', how='inner')['patent_id'].values
    navy_post = pd.to_numeric(pd.Series(navy_post), errors='coerce').dropna().values
    navy_patents = np.union1d(navy_pre, navy_post)
    navy_gov_list = load_dta(f'{RAW}/patents/USPTO_patentsview/government_organization_navy_list.dta')
    navy_fund = pv_gov.merge(navy_gov_list, on='organization_id', how='inner')
    navy_fund_pat = pd.to_numeric(navy_fund['patent_id'], errors='coerce').dropna().values
    navy_dod = fleming[fleming['dod'] > 0]['patno'].values
    navy_patents = np.union1d(navy_patents, np.union1d(navy_fund_pat, navy_dod))
    print(f"    Navy: {len(navy_patents)}")
except Exception as e:
    print(f"    Navy: ERROR - {e}")

try:
    gov_asg_list = load_dta(f'{RAW}/patents/USPTO_patentsview/assignee_gov_list.dta')
    gov_post = pv_asg.merge(gov_asg_list, on='id', how='inner')['patent_id'].values
    gov_post = pd.to_numeric(pd.Series(gov_post), errors='coerce').dropna().values
    gov_pre = fleming[fleming['asg_usgovt'] == 1]['patno'].values
    gov_all = load_dta(f'{RAW}/patents/USPTO_patentsview/government_organization.dta')
    gov_fund = pv_gov.merge(gov_all, on='organization_id', how='inner')
    gov_fund_pat = pd.to_numeric(gov_fund['patent_id'], errors='coerce').dropna().values
    gov_usg = fleming[fleming['usg'] > 0]['patno'].values
    gov_patents = np.union1d(np.union1d(gov_pre, gov_post), np.union1d(gov_fund_pat, gov_usg))
    print(f"    Gov: {len(gov_patents)}")
except Exception as e:
    print(f"    Gov: ERROR - {e}")

# Military-similar patents (top 10% of military similarity)
try:
    mil_sim = load_dta(f'{RAW}/patents/arthur_other_similar/both_patents_cos_sim.dta',
                       columns=['patent', 'cos_sim_military'])
    p90 = mil_sim['cos_sim_military'].quantile(0.90)
    mil_text_patents = mil_sim[mil_sim['cos_sim_military'] >= p90]['patent'].unique()
    print(f"    Military-similar: {len(mil_text_patents)} (>= p90={p90:.4f})")
except Exception as e:
    print(f"    Military-similar: ERROR - {e}")

# --- Patent locations ---
print("  Loading patent locations (disambiguated)...")
try:
    pat_loc = load_dta(f'{RAW}/patents/disambiguated/patent_inventors_county_application_date.dta',
                       columns=['patent', 'fips', 'invcount'])
    pat_loc = pat_loc.rename(columns={'patent': 'patent_number'})
    pat_loc['patent_number'] = pd.to_numeric(pat_loc['patent_number'], errors='coerce')
    pat_loc['fips'] = pd.to_numeric(pat_loc['fips'], errors='coerce')
    pat_loc = pat_loc.dropna(subset=['patent_number', 'fips'])
    # Fractional assignment: each inventor-patent gets equal weight
    pat_loc['invcount'] = pd.to_numeric(pat_loc['invcount'], errors='coerce').fillna(1)
    pat_loc['frac_inv'] = 1.0 / pat_loc['invcount']
    weight_col = 'frac_inv'
    # Merge in SIC2 from patent-SIC crosswalk
    sic_cw = load_dta(f'{RAW}/patents/sic_cross_walk/sic2_patent_number_cw_unique.dta',
                      columns=['patent', 'sic2_1972'])
    sic_cw = sic_cw.rename(columns={'patent': 'patent_number', 'sic2_1972': 'sic2'})
    sic_cw['patent_number'] = pd.to_numeric(sic_cw['patent_number'], errors='coerce')
    sic_cw['sic2'] = pd.to_numeric(sic_cw['sic2'], errors='coerce')
    sic_cw = sic_cw.dropna().drop_duplicates(subset='patent_number')
    pat_loc = pat_loc.merge(sic_cw, on='patent_number', how='inner')
    print(f"    Patent locations: {len(pat_loc)} inventor-patent records")
except Exception as e:
    print(f"    Patent locations: ERROR - {e}")
    pat_loc = pd.DataFrame()

# --- Build patent counts at county x SIC x year ---
print("  Building patent counts by type at county x SIC x year...")
if len(pat_loc) > 0:
    pl = pat_loc.copy()
    pl['patent'] = pl['patent_number']
    pl['fips'] = pl['fips'].astype(int)

    # Merge with grant year
    pl = pl.merge(hm.rename(columns={'patent': 'patent_number'}), on='patent_number', how='inner')

    # Flag patent types
    pl['is_nasa'] = pl['patent_number'].isin(all_nasa).astype(int)
    pl['is_army'] = pl['patent_number'].isin(army_patents).astype(int)
    pl['is_navy'] = pl['patent_number'].isin(navy_patents).astype(int)
    pl['is_gov'] = pl['patent_number'].isin(gov_patents).astype(int)
    pl['is_mil_text'] = pl['patent_number'].isin(mil_text_patents).astype(int)

    # Fractional counts
    for col in ['is_nasa', 'is_army', 'is_navy', 'is_gov', 'is_mil_text']:
        pl[f'f_{col}'] = pl[col] * pl['frac_inv']
    pl['frac_total'] = pl['frac_inv']

    # Map grant years to census years
    year_bins = [1932, 1937, 1942, 1947, 1954, 1958, 1963, 1967, 1972, 1977, 1982, 1987, 1992, 1997, 2002, 2007, 2012]

    # Build cumulative and 5-year counts
    if 'sic2' in pl.columns:
        grp_cols = ['fips', 'sic2']
    else:
        grp_cols = ['fips']

    # Annual fractional counts
    annual = pl.groupby(grp_cols + ['grantyear']).agg(
        msfpatent=('frac_total', 'sum'),
        msfnasa_patent=('f_is_nasa', 'sum'),
        msfarmy_patent=('f_is_army', 'sum'),
        msfnavy_patent=('f_is_navy', 'sum'),
        msfgov_patent=('f_is_gov', 'sum'),
        msfmil_text_patent=('f_is_mil_text', 'sum'),
    ).reset_index()

    # Cumulative counts up to each census year, then 5-year changes
    patent_counts = []
    for cy in year_bins:
        cum = annual[annual['grantyear'] <= cy].groupby(grp_cols).agg(
            smsfpatent=('msfpatent', 'sum'),
            smsfnasa_patent=('msfnasa_patent', 'sum'),
            smsfarmy_patent=('msfarmy_patent', 'sum'),
            smsfnavy_patent=('msfnavy_patent', 'sum'),
            smsfgov_patent=('msfgov_patent', 'sum'),
            smsfmil_text_patent=('msfmil_text_patent', 'sum'),
        ).reset_index()
        cum['year'] = cy
        patent_counts.append(cum)

    pat_panel = pd.concat(patent_counts, ignore_index=True)
    pat_panel = pat_panel[pat_panel['year'].isin(year_bins)]
    pat_panel = pat_panel.sort_values(grp_cols + ['year']).reset_index(drop=True)

    # 5-year changes (difference in cumulative)
    for v in ['smsfpatent', 'smsfnasa_patent', 'smsfarmy_patent', 'smsfnavy_patent', 'smsfgov_patent', 'smsfmil_text_patent']:
        pat_panel[f'{v}5'] = pat_panel.groupby(grp_cols)[v].diff()
        pat_panel[f'{v}5'] = pat_panel[f'{v}5'].fillna(pat_panel[v])

    # Keep census years only
    census_years = [1947, 1954, 1958, 1963, 1967, 1972, 1977, 1982, 1987, 1992]
    pat_panel = pat_panel[pat_panel['year'].isin(census_years)]

    if 'sic2' in pat_panel.columns:
        pat_panel = pat_panel.rename(columns={'sic2': 'sic'})

    pat_panel.to_pickle(f'{PROC}/patents/8.patents_compile_sic.pkl')
    print(f"    Patent compile: {len(pat_panel)} obs")
else:
    pat_panel = pd.DataFrame()
    print("    Patent compile: SKIPPED (no location data)")

try:
    del pl, annual
except NameError:
    pass
gc.collect()
print(f"  Phase 4 complete: {time.time()-t0:.0f}s")

# ==============================================================================
# PHASE 5: NASA SPENDING
# ==============================================================================
print("\n" + "=" * 70)
print("PHASE 5: NASA CONTRACTOR SPENDING")
print("=" * 70)
t0 = time.time()

try:
    # Load contractor data
    contractor = load_dta(f'{RAW}/nasa_contractor/1963-1997 dataset_final_Oct 2021.dta')
    crsp = load_dta(f'{RAW}/nasa_contractor/CRSP_company_list.dta')

    # Get SIC codes from CRSP via LPERMNO
    crsp['LPERMNO'] = pd.to_numeric(crsp['LPERMNO'], errors='coerce')
    crsp['sic'] = pd.to_numeric(crsp['sic'], errors='coerce')
    crsp['sic2'] = (crsp['sic'] // 100).astype('Int64')
    contractor['LPERMNO'] = pd.to_numeric(contractor.get('LPERMNO', contractor.get('lpermno')), errors='coerce')
    contractor = contractor.merge(crsp[['LPERMNO', 'sic2']].dropna().drop_duplicates(),
                                   on='LPERMNO', how='left')

    # Ensure we have fips, year, sic2, spending
    spend_col = 'netvalueofawards'
    contractor[spend_col] = pd.to_numeric(contractor[spend_col], errors='coerce') * 1000
    fips_col = 'countyfips' if 'countyfips' in contractor.columns else 'fips'
    contractor['fips'] = pd.to_numeric(contractor[fips_col], errors='coerce')
    contractor['year'] = pd.to_numeric(contractor['year'], errors='coerce')
    contractor['sic2'] = pd.to_numeric(contractor['sic2'], errors='coerce')

    # Map to census years
    cyear_map = {}
    for y in range(1963, 1998):
        if y <= 1963: cy = 1963
        elif y <= 1967: cy = 1967
        elif y <= 1972: cy = 1972
        elif y <= 1977: cy = 1977
        elif y <= 1982: cy = 1982
        elif y <= 1987: cy = 1987
        elif y <= 1992: cy = 1992
        else: cy = 1992
        cyear_map[y] = cy
    contractor['cyear'] = contractor['year'].map(cyear_map)

    # Collapse to census year x county x SIC
    spending = contractor.groupby(['cyear', 'fips', 'sic2']).agg(
        nasa_contractor_spend=(spend_col, 'sum')
    ).reset_index()
    spending = spending.rename(columns={'cyear': 'year', 'sic2': 'sic'})

    # Deflate to real 1958 dollars
    try:
        ppi = load_dta(f'{RAW}/price_index/producer_price_index_divide_by.dta')
        ppi['year'] = pd.to_numeric(ppi['year'], errors='coerce')
        spending = spending.merge(ppi, on='year', how='left')
        if 'ppindex_58base' in spending.columns:
            spending['r58_s2_nasa_contractor_spend'] = spending['nasa_contractor_spend'] / spending['ppindex_58base']
    except:
        spending['r58_s2_nasa_contractor_spend'] = spending['nasa_contractor_spend']

    # Normalize to millions
    spending['r58_s2_nasa_contractor_spend'] = spending['r58_s2_nasa_contractor_spend'] / 1_000_000

    # Arc-sinh transformation
    spending['arc_r58_s2_nasa_contractor_spend'] = np.log(
        spending['r58_s2_nasa_contractor_spend'] +
        np.sqrt(spending['r58_s2_nasa_contractor_spend']**2 + 1))

    # Indicator
    spending['ar58_s2_nasa_contractor_spend'] = (spending['r58_s2_nasa_contractor_spend'] > 0).astype(int)

    # Keep manufacturing SIC codes
    spending = spending[(spending['sic'] >= 20) & (spending['sic'] <= 39)]

    spending.to_pickle(f'{PROC}/spending/1.nasa_contractor_spending.pkl')
    print(f"  NASA spending: {len(spending)} obs")
except Exception as e:
    print(f"  NASA spending: ERROR - {e}")
    spending = pd.DataFrame()

print(f"  Phase 5 complete: {time.time()-t0:.0f}s")

# ==============================================================================
# PHASE 6: COMPILE 2 + 3 (Final Dataset)
# ==============================================================================
print("\n" + "=" * 70)
print("PHASE 6: FINAL COMPILATION (Compile 2+3)")
print("=" * 70)
t0 = time.time()

# Load compile 1 output
df = pd.read_pickle(f'{PROC}/final/1.compile.pkl')
print(f"  Starting: {len(df)} obs")

# Merge NASA spending
if len(spending) > 0:
    df = df.merge(spending[['fips', 'sic', 'year', 'r58_s2_nasa_contractor_spend',
                             'arc_r58_s2_nasa_contractor_spend', 'ar58_s2_nasa_contractor_spend']],
                  on=['fips', 'sic', 'year'], how='left')
    for c in ['r58_s2_nasa_contractor_spend', 'arc_r58_s2_nasa_contractor_spend']:
        df[c] = df[c].fillna(0)
    df['ar58_s2_nasa_contractor_spend'] = (df['r58_s2_nasa_contractor_spend'] > 0).astype(int)

# Merge military spending (raw data, no build needed)
try:
    mil_spend = load_dta(f'{RAW}/spending/county_real_military_47_06.dta')
    mil_spend['fips'] = pd.to_numeric(mil_spend['fips'], errors='coerce')
    mil_spend['year'] = pd.to_numeric(mil_spend['year'], errors='coerce')
    df = df.merge(mil_spend[['fips', 'year', 'r58_military_spend', 'military_spend']].drop_duplicates(),
                  on=['fips', 'year'], how='left')
    df['r58_military_spend'] = df['r58_military_spend'].fillna(0)
    df['military_spend'] = df['military_spend'].fillna(0)
    print(f"  Merged military spending")
except Exception as e:
    print(f"  Military spending: {e}")
    df['r58_military_spend'] = 0

# Merge census data (synthetic)
try:
    census = load_dta(f'{RAW}/other/1.census_full_synthetic.dta')
    cyear_map = {1947: 1950, 1954: 1950, 1958: 1960, 1963: 1960, 1967: 1970,
                 1972: 1970, 1977: 1980, 1982: 1980, 1987: 1990, 1992: 1990}
    df['cyear'] = df['year'].map(cyear_map)
    census['cyear'] = pd.to_numeric(census['cyear'], errors='coerce')
    census['fips'] = pd.to_numeric(census['fips'], errors='coerce')
    df = df.merge(census.drop_duplicates(subset=['cyear', 'fips']),
                  on=['cyear', 'fips'], how='left')
    # Rename synthetic vars
    for old, new in [('totpop_syn', 'totpop'), ('hs_grad_per_syn', 'hs_grad_per'),
                     ('col_grad_per_syn', 'col_grad_per'), ('school_years_syn', 'school_years')]:
        if old in df.columns:
            df = df.rename(columns={old: new})
    print(f"  Merged census data")
except Exception as e:
    print(f"  Census data: {e}")

# Merge IBM mainframes
try:
    ibm = load_dta(f'{RAW}/ibm_mainframes/synthetic_IBMinstallations1961_county_level.dta')
    ibm['fips'] = pd.to_numeric(ibm['fips'], errors='coerce')
    df = df.merge(ibm.drop_duplicates(subset=['fips']), on='fips', how='left')
    mainframe_col = [c for c in df.columns if 'mainframe' in c.lower() or 'sythetic' in c.lower()]
    if mainframe_col:
        df[mainframe_col[0]] = df[mainframe_col[0]].fillna(0)
    print(f"  Merged IBM mainframes")
except Exception as e:
    print(f"  IBM mainframes: {e}")

# Merge scientists
try:
    sci = load_dta(f'{RAW}/scientists/county_nasa_1962.dta')
    sci['fips'] = pd.to_numeric(sci['fips'], errors='coerce')
    df = df.merge(sci.drop_duplicates(subset=['fips']), on='fips', how='left')
    if 'tot_scientist62' in df.columns:
        df['tot_scientist62'] = df['tot_scientist62'].fillna(0)
    print(f"  Merged scientists")
except Exception as e:
    print(f"  Scientists: {e}")

# Merge patent similarity (full data with amspost_sv2_stemmed)
sim = pd.read_pickle(f'{PROC}/patents/0.patent_similarity.pkl')
df = df.merge(sim.rename(columns={'sic': 'sic2'}),
              left_on=['fips', 'sic'], right_on=['fips', 'sic2'], how='left')

# Merge patent compile data
if len(pat_panel) > 0:
    df = df.merge(pat_panel[['fips', 'sic', 'year',
                              'smsfpatent5', 'smsfnasa_patent5', 'smsfarmy_patent5',
                              'smsfnavy_patent5', 'smsfgov_patent5', 'smsfmil_text_patent5']],
                  on=['fips', 'sic', 'year'], how='left')
    for c in ['smsfpatent5', 'smsfnasa_patent5', 'smsfarmy_patent5', 'smsfnavy_patent5',
              'smsfgov_patent5', 'smsfmil_text_patent5']:
        if c in df.columns:
            df[c] = df[c].fillna(0)

# Merge lat/long
try:
    ll = load_dta(f'{RAW}/locations/county_lat_long.dta')
    ll['fips'] = pd.to_numeric(ll['fips'], errors='coerce')
    df = df.merge(ll.drop_duplicates(subset=['fips']), on='fips', how='left')
except:
    pass

# Merge state codes
try:
    state_cw = load_dta(f'{RAW}/geo_cross_walk/icpsr_to_fips_county.dta')
    state_cw['fips'] = pd.to_numeric(state_cw['fips'], errors='coerce')
    state_cw['state_fips'] = pd.to_numeric(state_cw['state_fips'], errors='coerce')
    df = df.merge(state_cw[['fips', 'state_fips']].drop_duplicates(subset=['fips']),
                  on='fips', how='left')
except:
    df['state_fips'] = df['fips'] // 1000

print(f"  After merges: {len(df)} obs")

# ---- COMPILE 3: Create treatment variables ----
print("  Creating treatment variables...")

# NASA industry indicator
df['nasa_ind'] = ((df['sic'] == 36) | (df['sic'] == 37)).astype(int)

# Income
df['income'] = df['mpayroll'] / df['memp']

# Handle missing non-production worker share
df['miss_npw'] = df['npw_share'].isna().astype(int)
df['npw_share'] = df['npw_share'].fillna(0)

# Drop if above-median space capability is missing
df = df.dropna(subset=['amspost_sv2_stemmed'])
print(f"  After dropping missing space capability: {len(df)} obs")

# Normalize contractor spending
if 'r58_s2_nasa_contractor_spend' not in df.columns:
    df['r58_s2_nasa_contractor_spend'] = 0
df['r58_s2_nasa_contractor_spend'] = df['r58_s2_nasa_contractor_spend'].fillna(0)
df['arc_r58_s2_nasa_contractor_spend'] = np.log(
    df['r58_s2_nasa_contractor_spend'] +
    np.sqrt(df['r58_s2_nasa_contractor_spend']**2 + 1))

# Arc-sinh patent variables
for v in ['smsfnasa_patent5', 'smsfpatent5', 'smsfgov_patent5',
          'smsfarmy_patent5', 'smsfnavy_patent5', 'smsfmil_text_patent5']:
    if v in df.columns:
        df[v] = df[v].fillna(0)
        df[f'arc_{v}'] = np.log(df[v] + np.sqrt(df[v]**2 + 1))

# Non-gov patents
if 'smsfpatent5' in df.columns and 'smsfgov_patent5' in df.columns:
    df['smsfngov_patent5'] = df['smsfpatent5'] - df['smsfgov_patent5']
    df['arc_smsfngov_patent5'] = np.log(df['smsfngov_patent5'] + np.sqrt(df['smsfngov_patent5']**2 + 1))

# Non-NASA patents
if 'smsfpatent5' in df.columns and 'smsfnasa_patent5' in df.columns:
    df['smsfnnasa_patent5'] = df['smsfpatent5'] - df['smsfnasa_patent5']
    df['arc_smsfnnasa_patent5'] = np.log(df['smsfnnasa_patent5'] + np.sqrt(df['smsfnnasa_patent5']**2 + 1))

# Create above-median × era interactions (the key treatment variables)
space_race = df['year'].isin([1963, 1967, 1972])
post_space_race = df['year'].isin([1977, 1982, 1987, 1992])

for var in ['mspost_sv2_stemmed']:
    avar = f'a{var}'
    if avar not in df.columns:
        # Create above-median indicator
        med = df[var].median()
        df[avar] = (df[var] >= med).astype(int)

    # Space race and post-space race interactions
    df[f'{avar}_sr'] = 0
    df.loc[space_race, f'{avar}_sr'] = df.loc[space_race, avar]
    df[f'{avar}_psr'] = 0
    df.loc[post_space_race, f'{avar}_psr'] = df.loc[post_space_race, avar]

    # NASA industry interactions
    df[f'n{avar}_sr'] = 0
    df.loc[space_race & (df['nasa_ind'] == 1), f'n{avar}_sr'] = df.loc[space_race & (df['nasa_ind'] == 1), avar]
    df[f'n{avar}_psr'] = 0
    df.loc[post_space_race & (df['nasa_ind'] == 1), f'n{avar}_psr'] = df.loc[post_space_race & (df['nasa_ind'] == 1), avar]

    # Year-specific interactions
    for yr in [1947, 1954, 1958, 1963, 1967, 1972, 1977, 1982, 1987, 1992]:
        df[f'{avar}_{yr}'] = ((df[avar] == 1) & (df['year'] == yr)).astype(int)
        df[f'n{avar}_{yr}'] = ((df[avar] == 1) & (df['year'] == yr) & (df['nasa_ind'] == 1)).astype(int)

# HS grad percentage at 1958
if 'hs_grad_per' in df.columns:
    hs58 = df[df['year'] == 1958].groupby('fips')['hs_grad_per'].first()
    df['shs_grad_per58'] = df['fips'].map(hs58)

# Value added share
df['tot_va'] = df.groupby(['fips', 'year'])['mval_add'].transform('sum')
df['va_share'] = df['mval_add'] / df['tot_va']

# TFP growth
df = df.sort_values(['sic', 'fips', 'year'])
df['ltfp_solow'] = df['ltfp']
df['ltfp_lag'] = df.groupby(['sic', 'fips'])['ltfp_solow'].shift(1)
df['gr_ltfp'] = (df['ltfp_solow'] - df['ltfp_lag']) / df['ltfp_lag'].abs()

# Mean NASA industry space capability
df['mnamspost_sv2_stemmed_sr'] = df.groupby(['fips', 'year'])['namspost_sv2_stemmed_sr'].transform('mean')
df['mnamspost_sv2_stemmed_psr'] = df.groupby(['fips', 'year'])['namspost_sv2_stemmed_psr'].transform('mean')

# Rename ltfp_solow to ltfp for consistency
df['ltfp'] = df['ltfp_solow']

# Final sort
df = df.sort_values(['fips', 'year', 'sic']).reset_index(drop=True)

# Save
df.to_pickle(f'{PROC}/final/3.compile.pkl')
print(f"\n  FINAL DATASET: {len(df)} obs, {df['fips'].nunique()} counties, "
      f"{df['sic'].nunique()} industries, {sorted(df['year'].unique())}")
print(f"  Key treatment vars: amspost_sv2_stemmed_sr mean={df['amspost_sv2_stemmed_sr'].mean():.3f}, "
      f"namspost_sv2_stemmed_sr mean={df['namspost_sv2_stemmed_sr'].mean():.3f}")
print(f"  Phase 6 complete: {time.time()-t0:.0f}s")

print("\n" + "=" * 70)
print("BUILD COMPLETE")
print("=" * 70)
