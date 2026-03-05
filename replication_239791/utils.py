"""
Shared utilities for the Family Institutions & Global Fertility Transition
replication study (239791-V1).

Paper: Gobbi, Hannusch, and Rossi (2026). "Family Institutions and the Global
       Fertility Transition." Journal of Economic Perspectives 40(1): 47-70.
"""

import os
import numpy as np
import pandas as pd

DATA_DIR = '239791-V1/data'
OUTPUT_DIR = 'replication_239791/output'
ANALYSIS_DATA_PATH = 'replication_239791/output/analysis_data.parquet'
ANALYSIS_DATA_FIG4_PATH = 'replication_239791/output/analysis_data_fig4.parquet'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# WDI year columns
YEAR_COLS = [str(y) for y in range(1960, 2025)]


def load_wdi_csv(filepath):
    """Load a WDI CSV and return DataFrame with Country Code and year columns.

    Handles multiple WDI formats (TFR/GDP have Country Name first,
    ChildMortality/MaternalMortality have Series Name first,
    SecEnrollment has Country Name then Series Name).
    """
    df = pd.read_csv(filepath)
    # Identify Country Code column position
    for col in df.columns:
        if col.strip() == 'Country Code':
            cc_col = col
            break
    else:
        raise ValueError(f"No 'Country Code' column in {filepath}")

    # Keep Country Code + year columns that exist
    keep_cols = [cc_col]
    for y in YEAR_COLS:
        if y in df.columns:
            keep_cols.append(y)

    out = df[keep_cols].copy()
    out = out.rename(columns={cc_col: 'CountryCode'})

    # Convert year columns to numeric
    for y in YEAR_COLS:
        if y in out.columns:
            out[y] = pd.to_numeric(out[y], errors='coerce')

    return out


def get_region_mapping():
    """Get ISO3 -> World Bank region mapping from geopandas naturalearth."""
    try:
        import geopandas as gpd
        # Try loading naturalearth data
        try:
            world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        except Exception:
            world = gpd.read_file(
                'https://naciscdn.org/naturalearth/110m/cultural/'
                'ne_110m_admin_0_countries.zip'
            )
        # The naturalearth_lowres may not have region_wb but has continent
        if 'region_wb' in world.columns:
            mapping = dict(zip(world['iso_a3'], world['region_wb']))
        else:
            # Fall back to manual mapping from continent
            mapping = _manual_region_mapping()
        return mapping
    except Exception:
        return _manual_region_mapping()


def get_region_mapping_medium():
    """Get ISO3 -> World Bank region mapping from naturalearth medium scale.

    This matches the R code's ne_countries(scale='medium') using iso_a3_eh.
    """
    try:
        import geopandas as gpd
        url = ('https://naciscdn.org/naturalearth/50m/cultural/'
               'ne_50m_admin_0_countries.zip')
        world = gpd.read_file(url)
        if 'region_wb' in world.columns and 'iso_a3_eh' in world.columns:
            mapping = {}
            for _, row in world.iterrows():
                iso = row.get('iso_a3_eh', row.get('iso_a3', ''))
                region = row.get('region_wb', '')
                if isinstance(iso, str) and len(iso) == 3 and iso != '-99':
                    mapping[iso] = region
            return mapping
    except Exception:
        pass
    return _manual_region_mapping()


def _manual_region_mapping():
    """Comprehensive WB region mapping as fallback."""
    SSA = [
        'AGO', 'BDI', 'BEN', 'BFA', 'BWA', 'CAF', 'CIV', 'CMR', 'COD',
        'COG', 'COM', 'CPV', 'DJI', 'ERI', 'ETH', 'GAB', 'GHA', 'GIN',
        'GMB', 'GNB', 'GNQ', 'KEN', 'LBR', 'LSO', 'MDG', 'MLI', 'MOZ',
        'MRT', 'MUS', 'MWI', 'NAM', 'NER', 'NGA', 'RWA', 'SDN', 'SEN',
        'SLE', 'SOM', 'SSD', 'STP', 'SWZ', 'SYC', 'TCD', 'TGO', 'TZA',
        'UGA', 'ZAF', 'ZMB', 'ZWE',
    ]
    EAP = [
        'AUS', 'BRN', 'CHN', 'FJI', 'FSM', 'HKG', 'IDN', 'JPN', 'KHM',
        'KIR', 'KOR', 'LAO', 'MAC', 'MHL', 'MMR', 'MNG', 'MYS', 'NCL',
        'NZL', 'PHL', 'PLW', 'PNG', 'PRK', 'PYF', 'SGP', 'SLB', 'THA',
        'TLS', 'TON', 'TUV', 'TWN', 'VNM', 'VUT', 'WSM',
    ]
    ECA = [
        'ALB', 'AND', 'ARM', 'AUT', 'AZE', 'BEL', 'BGR', 'BIH', 'BLR',
        'CHE', 'CYP', 'CZE', 'DEU', 'DNK', 'ESP', 'EST', 'FIN', 'FRA',
        'FRO', 'GBR', 'GEO', 'GRC', 'GRL', 'HRV', 'HUN', 'IMN', 'IRL',
        'ISL', 'ITA', 'KAZ', 'KGZ', 'LIE', 'LTU', 'LUX', 'LVA', 'MDA',
        'MKD', 'MNE', 'NLD', 'NOR', 'POL', 'PRT', 'ROU', 'RUS', 'SMR',
        'SRB', 'SVK', 'SVN', 'SWE', 'TJK', 'TKM', 'TUR', 'UKR', 'UZB',
        'XKX',
    ]
    LAC = [
        'ABW', 'ARG', 'ATG', 'BHS', 'BLZ', 'BOL', 'BRA', 'BRB', 'CHL',
        'COL', 'CRI', 'CUB', 'CUW', 'DMA', 'DOM', 'ECU', 'GRD', 'GTM',
        'GUY', 'HND', 'HTI', 'JAM', 'KNA', 'LCA', 'MEX', 'NIC', 'PAN',
        'PER', 'PRY', 'SLV', 'SUR', 'SXM', 'TCA', 'TTO', 'URY', 'VCT',
        'VEN', 'VGB', 'VIR',
    ]
    MENA = [
        'ARE', 'BHR', 'DZA', 'EGY', 'IRN', 'IRQ', 'ISR', 'JOR', 'KWT',
        'LBN', 'LBY', 'MAR', 'MLT', 'OMN', 'PSE', 'QAT', 'SAU', 'SYR',
        'TUN', 'YEM',
    ]
    NA_ = ['BMU', 'CAN', 'USA']
    SA = ['AFG', 'BGD', 'BTN', 'IND', 'LKA', 'MDV', 'NPL', 'PAK']

    mapping = {}
    for code in SSA:
        mapping[code] = 'Sub-Saharan Africa'
    for code in EAP:
        mapping[code] = 'East Asia & Pacific'
    for code in ECA:
        mapping[code] = 'Europe & Central Asia'
    for code in LAC:
        mapping[code] = 'Latin America & Caribbean'
    for code in MENA:
        mapping[code] = 'Middle East & North Africa'
    for code in NA_:
        mapping[code] = 'North America'
    for code in SA:
        mapping[code] = 'South Asia'
    return mapping


# List of WDI aggregate codes to exclude (not real countries)
WDI_AGGREGATES = {
    'AFE', 'AFW', 'ARB', 'CEB', 'CSS', 'EAP', 'EAR', 'EAS', 'ECA', 'ECS',
    'EMU', 'EUU', 'FCS', 'HIC', 'HPC', 'IBD', 'IBT', 'IDA', 'IDB', 'IDX',
    'INX', 'LAC', 'LCN', 'LDC', 'LIC', 'LMC', 'LMY', 'LTE', 'MIC', 'MNA',
    'MEA', 'NAC', 'OED', 'OSS', 'PRE', 'PSS', 'PST', 'SAS', 'SSA', 'SSF',
    'SST', 'TEA', 'TEC', 'TLA', 'TMN', 'TSA', 'TSS', 'UMC', 'WLD',
}


def format_stars(pval):
    """Return significance stars."""
    if pval < 0.01:
        return '***'
    elif pval < 0.05:
        return '**'
    elif pval < 0.10:
        return '*'
    return ''
