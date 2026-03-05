"""
Shared utilities for the Geruso & Spears (2026) replication.

"The Likelihood of Persistently Low Global Fertility"
Journal of Economic Perspectives, 40(1): 3-26.
openICPSR 239496-V1
"""

import os
import numpy as np
import pandas as pd

DATA_DIR = '239496-V1/source_data'
OUTPUT_DIR = 'replication_239496/output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Countries in the HFD CCF.xlsx that are NOT sub-national
HFD_COUNTRIES_MAIN = [
    'AUT', 'BLR', 'BEL', 'BGR', 'CAN', 'CHL', 'HRV', 'CZE', 'DNK', 'EST',
    'FIN', 'FRATNP', 'DEUTNP', 'HUN', 'ISL', 'IRL', 'ITA', 'JPN', 'LTU',
    'NLD', 'NOR', 'POL', 'PRT', 'KOR', 'RUS', 'SVK', 'SVN', 'ESP', 'SWE',
    'CHE', 'TWN', 'GBR_NP', 'UKR', 'USA'
]

# Sub-national codes to exclude from main analysis
HFD_SUBNATIONAL = ['DEUTE', 'DEUTW', 'GBRTENW', 'GBR_NIR', 'GBR_SCO']

# All HFD country columns
HFD_ALL_CODES = [
    'AUT', 'BLR', 'BEL', 'BGR', 'CAN', 'CHL', 'HRV', 'CZE', 'DNK', 'EST',
    'FIN', 'FRATNP', 'DEUTNP', 'DEUTE', 'DEUTW', 'HUN', 'ISL', 'IRL', 'ITA',
    'JPN', 'LTU', 'NLD', 'NOR', 'POL', 'PRT', 'KOR', 'RUS', 'SVK', 'SVN',
    'ESP', 'SWE', 'CHE', 'TWN', 'GBR_NP', 'GBRTENW', 'GBR_NIR', 'GBR_SCO',
    'UKR', 'USA'
]

# ISO3 code mapping: HFD code -> WPP ISO3 code
HFD_TO_ISO3 = {c: c[:3] for c in HFD_ALL_CODES}
HFD_TO_ISO3['FRATNP'] = 'FRA'
HFD_TO_ISO3['DEUTNP'] = 'DEU'
HFD_TO_ISO3['DEUTE'] = 'DEU'
HFD_TO_ISO3['DEUTW'] = 'DEU'
HFD_TO_ISO3['GBR_NP'] = 'GBR'
HFD_TO_ISO3['GBRTENW'] = 'GBR'
HFD_TO_ISO3['GBR_NIR'] = 'GBR'
HFD_TO_ISO3['GBR_SCO'] = 'GBR'

# Country name mapping for labels
HFD_NAMES = {
    'AUT': 'Austria', 'BLR': 'Belarus', 'BEL': 'Belgium', 'BGR': 'Bulgaria',
    'CAN': 'Canada', 'CHL': 'Chile', 'HRV': 'Croatia', 'CZE': 'Czechia',
    'DNK': 'Denmark', 'EST': 'Estonia', 'FIN': 'Finland',
    'FRATNP': 'France', 'DEUTNP': 'Germany',
    'HUN': 'Hungary', 'ISL': 'Iceland', 'IRL': 'Ireland', 'ITA': 'Italy',
    'JPN': 'Japan', 'LTU': 'Lithuania', 'NLD': 'Netherlands', 'NOR': 'Norway',
    'POL': 'Poland', 'PRT': 'Portugal', 'KOR': 'South Korea', 'RUS': 'Russia',
    'SVK': 'Slovakia', 'SVN': 'Slovenia', 'ESP': 'Spain', 'SWE': 'Sweden',
    'CHE': 'Switzerland', 'TWN': 'Taiwan', 'GBR_NP': 'United Kingdom',
    'UKR': 'Ukraine', 'USA': 'United States'
}

# Territories/non-UN members to exclude from UN member state counts
NON_UN_TERRITORIES = [
    'American Samoa', 'Anguilla', 'Aruba', 'Bermuda',
    'Bonaire, Sint Eustatius and Saba', 'British Virgin Islands',
    'Cayman Islands', 'China, Hong Kong SAR', 'China, Macao SAR',
    'China, Taiwan Province of China', 'Cook Islands', 'Curaçao',
    'Falkland Islands (Malvinas)', 'Faroe Islands', 'French Guiana',
    'French Polynesia', 'Gibraltar', 'Greenland', 'Guadeloupe', 'Guam',
    'Guernsey', 'Holy See', 'Isle of Man', 'Jersey',
    'Kosovo (under UNSC res. 1244)', 'Martinique', 'Mayotte', 'Montserrat',
    'Niue', 'Northern Mariana Islands', 'New Caledonia', 'Puerto Rico',
    'Réunion', 'Saint Barthélemy', 'Saint Helena',
    'Saint Martin (French part)', 'Saint Pierre and Miquelon',
    'Sint Maarten (Dutch part)', 'State of Palestine', 'Tokelau',
    'Turks and Caicos Islands', 'United States Virgin Islands',
    'Wallis and Futuna Islands', 'Western Sahara'
]


def load_wpp(sheet='Estimates'):
    """Load WPP2024 compact indicators."""
    path = os.path.join(DATA_DIR, 'WPP2024_GEN_F01_DEMOGRAPHIC_INDICATORS_COMPACT.xlsx')
    df = pd.read_excel(path, sheet_name=sheet, header=16)
    # Standardize column names
    rename = {
        'Region, subregion, country or area *': 'Region',
        'ISO3 Alpha-code': 'ISO3',
        'Year': 'year',
        'Type': 'Type',
        'Total Population, as of 1 January (thousands)': 'pop_thousands',
        'Total Fertility Rate (live births per woman)': 'tfr',
        'Births (thousands)': 'births_thousands',
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # Convert numeric
    for col in ['year', 'pop_thousands', 'tfr', 'births_thousands']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def load_ccf():
    """Load HFD completed cohort fertility, reshaped to long format."""
    path = os.path.join(DATA_DIR, 'CCF.xlsx')
    df = pd.read_excel(path, sheet_name='Completed cohort fertility', header=2)
    # Reshape wide to long
    long = df.melt(id_vars='COHORT', var_name='country', value_name='ccf')
    long = long.rename(columns={'COHORT': 'cohort'})
    long.loc[long['ccf'] == 0, 'ccf'] = np.nan
    return long


def load_parity():
    """Load HFD cohort childlessness, reshaped to long format."""
    path = os.path.join(DATA_DIR, 'PARITY.xlsx')
    df = pd.read_excel(path, sheet_name='Cohort childlessness', header=2)
    long = df.melt(id_vars='COHORT', var_name='country', value_name='childlessness')
    long = long.rename(columns={'COHORT': 'cohort'})
    long.loc[long['childlessness'] == 0, 'childlessness'] = np.nan
    return long


def load_romania_ccf():
    """Load Romania CCF from Ghetau (1997) Table 6."""
    path = os.path.join(DATA_DIR, 'romania_ccf_ghetau_tabel6.dta')
    return pd.read_stata(path)


def load_births_history():
    """Load historical world births (Spears et al. 2024)."""
    path = os.path.join(DATA_DIR, 'plos_one', 'population_history.csv')
    return pd.read_csv(path)


def load_births_projections():
    """Load projected world births (Spears et al. 2024)."""
    path = os.path.join(DATA_DIR, 'plos_one', 'main_output.csv')
    return pd.read_csv(path)
