"""
01_clean.py - Data cleaning and merging for Davis (2014) replication.

Replicates the data import, reshape, merge, and unit conversion from allcode.do.
Produces a single cross-sectional dataset (temp1.csv) used by all subsequent scripts.
"""
import numpy as np
import pandas as pd
from utils import (DATA_DIR, OUTPUT_DIR, load_wb_data, load_eia_data,
                   drop_non_countries, rename_countries,
                   WB_FILES, EIA_FILES, EIA_REGIONS,
                   KG_OIL_TO_GJ, GJ_PER_GALLON_GAS, GJ_PER_GALLON_DIE,
                   GALLONS_PER_LITER)

# ============================================================
# 1. Load World Bank data
# ============================================================

# --- Gasoline prices (y1960-y2012) ---
gasprice_raw = load_wb_data('gasprice', WB_FILES['gasprice'], 2012)
# Forward-fill 2012: use 2011 if 2012 missing, then 2010
gasprice_raw['y2012'] = gasprice_raw['y2012'].fillna(gasprice_raw['y2011'])
gasprice_raw['y2012'] = gasprice_raw['y2012'].fillna(gasprice_raw['y2010'])
# Reshape to long
gasprice = gasprice_raw.melt(id_vars=['country', 'countrycode'],
                              var_name='year_str', value_name='gasprice')
gasprice['year'] = gasprice['year_str'].str.replace('y', '').astype(int)
gasprice = gasprice[['country', 'countrycode', 'year', 'gasprice']]

# --- Diesel prices (y1960-y2012) ---
dieprice_raw = load_wb_data('dieprice', WB_FILES['dieprice'], 2012)
dieprice_raw['y2012'] = dieprice_raw['y2012'].fillna(dieprice_raw['y2011'])
dieprice_raw['y2012'] = dieprice_raw['y2012'].fillna(dieprice_raw['y2010'])
dieprice = dieprice_raw.melt(id_vars=['country', 'countrycode'],
                              var_name='year_str', value_name='dieprice')
dieprice['year'] = dieprice['year_str'].str.replace('y', '').astype(int)
dieprice = dieprice[['country', 'countrycode', 'year', 'dieprice']]

# --- Gasoline consumption per capita (y1960-y2010) ---
gascon_raw = load_wb_data('gascon', WB_FILES['gascon'], 2010)
gascon_raw['y2010'] = gascon_raw['y2010'].fillna(gascon_raw['y2009'])
gascon_raw['y2010'] = gascon_raw['y2010'].fillna(gascon_raw['y2008'])
gascon = gascon_raw.melt(id_vars=['country', 'countrycode'],
                          var_name='year_str', value_name='gascon')
gascon['year'] = gascon['year_str'].str.replace('y', '').astype(int)
gascon = gascon[['country', 'countrycode', 'year', 'gascon']]

# --- Diesel consumption per capita (y1960-y2010) ---
diecon_raw = load_wb_data('diecon', WB_FILES['diecon'], 2010)
diecon_raw['y2010'] = diecon_raw['y2010'].fillna(diecon_raw['y2009'])
diecon_raw['y2010'] = diecon_raw['y2010'].fillna(diecon_raw['y2008'])
diecon = diecon_raw.melt(id_vars=['country', 'countrycode'],
                          var_name='year_str', value_name='diecon')
diecon['year'] = diecon['year_str'].str.replace('y', '').astype(int)
diecon = diecon[['country', 'countrycode', 'year', 'diecon']]

# --- Motor vehicles per 1000 (y1960-y2010, keep 2010 only) ---
vehicles_raw = load_wb_data('vehicles', WB_FILES['vehicles'], 2010)
vehicles = vehicles_raw.melt(id_vars=['country', 'countrycode'],
                              var_name='year_str', value_name='vehicles')
vehicles['year'] = vehicles['year_str'].str.replace('y', '').astype(int)
vehicles = vehicles[vehicles['year'] == 2010][['countrycode', 'vehicles']]
vehicles = vehicles.groupby('countrycode')['vehicles'].mean().reset_index()

# --- Population (y1960-y2011, keep 2010 only) ---
pop_raw = load_wb_data('pop', WB_FILES['pop'], 2011)
pop = pop_raw.melt(id_vars=['country', 'countrycode'],
                    var_name='year_str', value_name='pop')
pop['year'] = pop['year_str'].str.replace('y', '').astype(int)
pop = pop[pop['year'] == 2010][['countrycode', 'pop']]
pop = pop.groupby('countrycode')['pop'].mean().reset_index()

# --- GDP per capita (y1960-y2011, keep 2010 only) ---
gdp_raw = load_wb_data('gdp', WB_FILES['gdp'], 2011)
gdp = gdp_raw.melt(id_vars=['country', 'countrycode'],
                    var_name='year_str', value_name='gdp')
gdp['year'] = gdp['year_str'].str.replace('y', '').astype(int)
gdp = gdp[gdp['year'] == 2010][['countrycode', 'gdp']]
gdp = gdp.groupby('countrycode')['gdp'].mean().reset_index()

# ============================================================
# 2. Merge WB data
# ============================================================
# Start with gascon, merge diecon, gasprice, dieprice on (countrycode, year)
merged = gascon.merge(diecon[['countrycode', 'year', 'diecon']],
                      on=['countrycode', 'year'], how='outer')
merged = merged.merge(gasprice[['countrycode', 'year', 'gasprice']],
                      on=['countrycode', 'year'], how='outer')
merged = merged.merge(dieprice[['countrycode', 'year', 'dieprice']],
                      on=['countrycode', 'year'], how='outer')

# Fill country from any non-null source
if merged['country'].isna().any():
    country_map = gascon.dropna(subset=['country']).drop_duplicates('countrycode').set_index('countrycode')['country']
    merged['country'] = merged['country'].fillna(merged['countrycode'].map(country_map))

# Merge cross-sectional vars
merged = merged.merge(vehicles, on='countrycode', how='left')
merged = merged.merge(pop, on='countrycode', how='left')
merged = merged.merge(gdp, on='countrycode', how='left')

# ============================================================
# 3. Drop non-countries
# ============================================================
merged = drop_non_countries(merged)

# ============================================================
# 4. Rename countries for EIA matching
# ============================================================
merged = rename_countries(merged)

# ============================================================
# 5. Load and merge EIA data
# ============================================================
def process_eia(filepath, varname):
    df = load_eia_data(filepath)
    df_long = df.melt(id_vars=['country'], var_name='year_str', value_name=varname)
    df_long['year'] = df_long['year_str'].str.replace('y', '').astype(int)
    df_long = df_long[df_long['year'] == 2010]
    # Drop "--" values
    df_long[varname] = pd.to_numeric(df_long[varname], errors='coerce')
    df_long = df_long.dropna(subset=[varname])
    df_long = df_long.groupby('country')[varname].mean().reset_index()
    # Drop regional aggregates
    df_long = df_long[~df_long['country'].isin(EIA_REGIONS)].reset_index(drop=True)
    return df_long

eia_dfs = {}
for varname, filepath in EIA_FILES.items():
    eia_dfs[varname] = process_eia(filepath, varname)

# Sequential inner merges on country (same as Stata keep if _merge==3)
for varname in ['exportscrude', 'importscrude', 'importsrefined', 'exportsrefined']:
    merged = merged.merge(eia_dfs[varname], on='country', how='inner')

# Calculate net exports
merged['netexports'] = (merged['exportscrude'] + merged['exportsrefined']
                        - merged['importscrude'] - merged['importsrefined'])

# Drop trade columns
merged = merged.drop(columns=['exportscrude', 'exportsrefined',
                               'importscrude', 'importsrefined'])

# Post-EIA merge rename
merged['country'] = merged['country'].replace('United Arab Emirates', 'U.A.E.')

# ============================================================
# 6. Convert units to gallons
# ============================================================
# gascon/diecon: kg oil equivalent per capita -> gallons per capita per year
merged['gascon'] = merged['gascon'] * KG_OIL_TO_GJ / GJ_PER_GALLON_GAS
merged['diecon'] = merged['diecon'] * KG_OIL_TO_GJ / GJ_PER_GALLON_DIE

# gasprice/dieprice: $/liter -> $/gallon
merged['gasprice'] = merged['gasprice'] / GALLONS_PER_LITER
merged['dieprice'] = merged['dieprice'] / GALLONS_PER_LITER

# ============================================================
# 7. Keep latest available year only and collapse
# ============================================================
# Gascon/diecon: keep only 2010; gasprice/dieprice: keep only 2012; pop: keep only 2010
merged.loc[merged['year'] != 2010, 'gascon'] = np.nan
merged.loc[merged['year'] != 2010, 'diecon'] = np.nan
merged.loc[merged['year'] != 2012, 'gasprice'] = np.nan
merged.loc[merged['year'] != 2012, 'dieprice'] = np.nan
merged.loc[merged['year'] != 2010, 'pop'] = np.nan

# Collapse by country (take mean of non-null values)
temp1 = merged.groupby('country').agg({
    'netexports': 'mean',
    'gdp': 'mean',
    'pop': 'mean',
    'gascon': 'mean',
    'diecon': 'mean',
    'gasprice': 'mean',
    'dieprice': 'mean',
}).reset_index()

# ============================================================
# 8. Report summary stats
# ============================================================
print("=" * 60)
print("DATA SUMMARY")
print("=" * 60)
valid_gas = temp1[temp1['gasprice'].notna() & (temp1['gasprice'] > 0)]
valid_die = temp1[temp1['dieprice'].notna() & (temp1['dieprice'] > 0)]

print(f"\nCountries with gasoline price data: {len(valid_gas)}")
print(f"Mean gasoline price ($/gallon): ${valid_gas['gasprice'].mean():.2f}")
print(f"  [Paper reports $5.26]")

print(f"\nCountries with diesel price data: {len(valid_die)}")
print(f"Mean diesel price ($/gallon): ${valid_die['dieprice'].mean():.2f}")
print(f"  [Paper reports $4.12]")

# Venezuela check
ven = temp1[temp1['country'] == 'Venezuela']
print(f"\nVenezuela gasoline price: ${ven['gasprice'].values[0]:.2f}")
print(f"Venezuela diesel price: ${ven['dieprice'].values[0]:.2f}")

# US check
us = temp1[temp1['country'] == 'United States']
print(f"\nUnited States:")
print(us.to_string(index=False))

print(f"\nTotal observations in final dataset: {len(temp1)}")

# Save
temp1.to_csv(f'{OUTPUT_DIR}/temp1.csv', index=False)
print(f"\nSaved temp1.csv with {len(temp1)} countries")
