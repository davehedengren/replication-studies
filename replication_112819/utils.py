"""
Utility functions for replication of:
"The Economic Cost of Global Fuel Subsidies" (Davis, 2014)
AER Papers and Proceedings
"""
import os
import numpy as np
import pandas as pd

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', '112819-V1', 'Data-and-Code')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Key parameters
SPOT_GAS_PRICE = 2.817       # $/gallon, NY Harbor Nov 2012
SPOT_DIE_PRICE = 3.009       # $/gallon, NY Harbor No.2 Heating Oil Nov 2012
TRANSPORT_COST_LITER = 0.2   # $/liter for net importers
GALLONS_PER_LITER = 0.264172
TRANSPORT_COST_GALLON = TRANSPORT_COST_LITER / GALLONS_PER_LITER
EXTERNAL_COST = 1.11         # $/gallon (Parry et al. 2007)
DEMAND_ELASTICITY = -0.6     # Baseline long-run elasticity

# Unit conversion constants
KG_OIL_TO_GJ = 0.04187
GJ_PER_GALLON_GAS = 0.1318
GJ_PER_GALLON_DIE = 0.1465

# World Bank CSV files
WB_FILES = {
    'gasprice': 'EP.PMP.SGAS.CD_Indicator_MetaData_en_EXCEL.csv',
    'dieprice': 'EP.PMP.DESL.CD_Indicator_MetaData_en_EXCEL.csv',
    'gascon': 'IS.ROD.SGAS.PC_Indicator_MetaData_en_EXCEL.csv',
    'diecon': 'IS.ROD.DESL.PC_Indicator_MetaData_en_EXCEL.csv',
    'vehicles': 'IS.VEH.NVEH.P3_Indicator_MetaData_en_EXCEL.csv',
    'pop': 'SP.POP.TOTL_Indicator_MetaData_en_EXCEL.csv',
    'gdp': 'NY.GDP.PCAP.CD_Indicator_MetaData_en_EXCEL.csv',
}

# EIA CSV files
EIA_FILES = {
    'exportscrude': 'Exports_of_Crude_Oil_including_Lease_Condensate_-Thousand_Barrels_Per_Day-.csv',
    'importscrude': 'Imports_of_Crude_Oil_including_Lease_Condensate_-Thousand_Barrels_Per_Day-.csv',
    'exportsrefined': 'Total_Exports_of_Refined_Petroleum_Products_-Thousand_Barrels_Per_Day-.csv',
    'importsrefined': 'Total_Imports_of_Refined_Petroleum_Products_-Thousand_Barrels_Per_Day-.csv',
}

# Non-country entities to drop from WB data
NON_COUNTRIES = [
    'High', 'Low', 'Middle', 'Upper', 'East Asia', 'Sub-Saharan',
    'Europe', 'Latin America', 'Heavily indebted', 'Least developed',
    'Pacific island small states', 'Small states', 'Other small states',
    'Caribbean small states', 'Arab World', 'North America', 'Euro area',
    'Not classified', 'World', 'South Asia', 'OECD members',
]

# EIA regional aggregates to drop
EIA_REGIONS = [
    'North America', 'Central & South America', 'Europe', 'Eurasia',
    'Middle East', 'Africa', 'Asia & Oceania', 'Antarctica', 'World',
]

# Country name mappings (WB -> EIA compatible)
COUNTRY_RENAMES = {
    'Egypt, Arab Rep.': 'Egypt',
    'Iran, Islamic Rep.': 'Iran',
    'Venezuela, RB': 'Venezuela',
    'Russian Federation': 'Russia',
    'Brunei Darussalam': 'Brunei',
    'Hong Kong SAR, China': 'Hong Kong',
    'Yemen, Rep.': 'Yemen',
    'Virgin Islands (U.S.)': 'Virgin Islands,  U.S.',
    'Syrian Arab Republic': 'Syria',
    'Slovak Republic': 'Slovakia',
    'Lao PDR': 'Laos',
    'Kyrgyz Republic': 'Kyrgyzstan',
    'Korea, Dem. Rep.': 'Korea, North',
    'Korea, Rep.': 'Korea, South',
    'Myanmar': 'Burma (Myanmar)',
}


def load_wb_data(varname, filepath, year_end):
    """Load a World Bank CSV file (no header, columns: country, countrycode, y1960-yXXXX)."""
    fp = os.path.join(DATA_DIR, filepath)
    df = pd.read_csv(fp, header=None)

    # First two columns are country and countrycode
    # Remaining columns are years starting from 1960
    n_year_cols = df.shape[1] - 2
    year_start = 1960
    years = list(range(year_start, year_start + n_year_cols))

    col_names = ['country', 'countrycode'] + [f'y{y}' for y in years]
    df.columns = col_names

    return df


def load_eia_data(filepath):
    """Load an EIA CSV file (no header, columns: country, y1984-y2010)."""
    fp = os.path.join(DATA_DIR, filepath)
    df = pd.read_csv(fp, header=None)

    n_year_cols = df.shape[1] - 1
    year_start = 1984
    years = list(range(year_start, year_start + n_year_cols))

    col_names = ['country'] + [f'y{y}' for y in years]
    df.columns = col_names

    return df


def drop_non_countries(df):
    """Drop non-country entities from WB data."""
    mask = pd.Series(True, index=df.index)
    for prefix in NON_COUNTRIES:
        if prefix in ['Pacific island small states', 'Small states',
                       'Other small states', 'Caribbean small states',
                       'Arab World', 'North America', 'Euro area',
                       'Not classified', 'World', 'South Asia', 'OECD members']:
            mask = mask & (df['country'] != prefix)
        else:
            mask = mask & ~df['country'].str.startswith(prefix, na=False)
    return df[mask].reset_index(drop=True)


def rename_countries(df, col='country'):
    """Apply country name mappings."""
    for old, new in COUNTRY_RENAMES.items():
        df[col] = df[col].replace(old, new)
    return df
