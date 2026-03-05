"""
01_clean.py — Data Cleaning (translates 2_clean_data.R to Python)

Merges ILO labor force data with WBL childcare laws data.
Input:  227802-V1/data/raw/ (5 CSV files)
Output: replication_227802/output/analysis_data.parquet
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from utils import RAW_DATA_DIR, CLEAN_DATA_PATH, OUTPUT_DIR, REGION_MAP


def clean_lfpr(raw_dir):
    """Clean labor force participation rate data (EAP_2WAP)."""
    df = pd.read_csv(os.path.join(raw_dir, 'EAP_2WAP_SEX_AGE_RT_A.csv'))
    df.columns = [c.lower() for c in df.columns]

    # Filter: female, aggregate age bands
    df['sex'] = df['sex'].str[-1:].str.lower()
    df['classif1'] = df['classif1'].str.lower()
    df = df[(df['classif1'].str.contains('aggregate')) & (df['sex'] == 'f')]

    # Extract age group
    df['age'] = df['classif1'].str[-5:].str.replace('-', '_')

    # Pivot to wide
    df = df[['ref_area', 'time', 'sex', 'age', 'obs_value']]
    df['col_name'] = 'lfpr_' + df['sex'] + '_' + df['age']
    pivot = df.pivot_table(index=['ref_area', 'time'], columns='col_name',
                           values='obs_value', aggfunc='first').reset_index()
    return pivot


def clean_lfpr_hhchild(raw_dir):
    """Clean LFPR of women in couple households with children <6 (GED_2LFP)."""
    df = pd.read_csv(os.path.join(raw_dir, 'GED_2LFP_SEX_RT_A.csv'))
    df.columns = [c.lower() for c in df.columns]
    df['sex'] = df['sex'].str[-1:].str.lower()
    df = df[df['sex'] == 'f']
    df = df.rename(columns={'obs_value': 'lfpr_hhchild_f'})
    return df[['ref_area', 'time', 'lfpr_hhchild_f']]


def clean_emp2pop(raw_dir):
    """Clean employment-to-population ratio (EMP_2WAP)."""
    df = pd.read_csv(os.path.join(raw_dir, 'EMP_2WAP_SEX_AGE_RT_A.csv'))
    df.columns = [c.lower() for c in df.columns]
    df['sex'] = df['sex'].str[-1:].str.lower()
    df['classif1'] = df['classif1'].str.lower()
    df = df[(df['classif1'].str.contains('ythadult')) & (df['sex'] == 'f')]
    df['age'] = df['classif1'].str[-5:].str.replace('-', '_')
    df = df[['ref_area', 'time', 'sex', 'age', 'obs_value']]
    df['col_name'] = 'emp2pop_' + df['sex'] + '_' + df['age']
    pivot = df.pivot_table(index=['ref_area', 'time'], columns='col_name',
                           values='obs_value', aggfunc='first').reset_index()
    return pivot


def clean_unemployment(raw_dir):
    """Clean unemployment rate (UNE_2EAP)."""
    df = pd.read_csv(os.path.join(raw_dir, 'UNE_2EAP_SEX_AGE_RT_A.csv'))
    df.columns = [c.lower() for c in df.columns]
    df['sex'] = df['sex'].str[-1:].str.lower()
    df['classif1'] = df['classif1'].str.lower()
    df = df[(df['classif1'].str.contains('ythadult')) & (df['sex'] == 'f')]
    df['age'] = df['classif1'].str[-5:].str.replace('-', '_')
    df = df[['ref_area', 'time', 'sex', 'age', 'obs_value']]
    df['col_name'] = 'unemp_' + df['sex'] + '_' + df['age']
    pivot = df.pivot_table(index=['ref_area', 'time'], columns='col_name',
                           values='obs_value', aggfunc='first').reset_index()
    return pivot


def merge_all(raw_dir):
    """Merge all ILO datasets with WBL childcare laws data."""
    print("Loading LFPR data...")
    lfpr = clean_lfpr(raw_dir)
    print(f"  LFPR shape: {lfpr.shape}")

    print("Loading LFPR household/child data...")
    hhchild = clean_lfpr_hhchild(raw_dir)
    print(f"  HHchild shape: {hhchild.shape}")

    print("Loading employment-to-population data...")
    emp = clean_emp2pop(raw_dir)
    print(f"  Emp2pop shape: {emp.shape}")

    print("Loading unemployment data...")
    unemp = clean_unemployment(raw_dir)
    print(f"  Unemployment shape: {unemp.shape}")

    # Merge ILO datasets
    print("Merging ILO datasets...")
    iloest = lfpr.merge(hhchild, on=['ref_area', 'time'], how='left')
    iloest = iloest.merge(emp, on=['ref_area', 'time'], how='left')
    iloest = iloest.merge(unemp, on=['ref_area', 'time'], how='left')
    iloest = iloest.rename(columns={'ref_area': 'id'})
    print(f"  Merged ILO shape: {iloest.shape}")

    # Load WBL data
    print("Loading WBL childcare laws data...")
    wbl = pd.read_csv(os.path.join(raw_dir, 'WBL_childcare_2024.csv'))
    wbl = wbl.rename(columns={'wbcodev2': 'id'})
    wbl['wbl_data'] = 1
    print(f"  WBL shape: {wbl.shape}")

    # Merge
    df = iloest.merge(wbl, on='id', how='left')
    df = df[df['wbl_data'] == 1]
    df = df.drop(columns=['wbl_data'])

    # Region abbreviations
    df['region_abbr'] = df['region'].map(REGION_MAP)

    # Rename time to year and create treatment variable
    df = df.rename(columns={'time': 'year'})
    df['treat'] = np.where(
        df['year'] >= df['enactment'], 1,
        np.where(df['year'] < df['enactment'], 0,
                 np.where(df['has_childcare_law'] == 0, 0, np.nan))
    )

    print(f"  Final merged shape: {df.shape}")
    return df


def main():
    print("=" * 60)
    print("01_clean.py: Data Cleaning")
    print("=" * 60)

    df = merge_all(RAW_DATA_DIR)

    # Create dimension-specific treatment variables (from table1-2.R)
    df['enactment_av'] = np.where(df['availability'] == 1, df['enactment'], np.nan)
    df['enforcement_av'] = np.where(df['availability'] == 1, df['enforcement'], np.nan)
    df['enactment_af'] = np.where(df['affordability'] == 1, df['enactment'], np.nan)
    df['enforcement_af'] = np.where(df['affordability'] == 1, df['enforcement'], np.nan)
    df['enactment_qua'] = np.where(df['quality'] == 1, df['enactment'], np.nan)
    df['enforcement_qua'] = np.where(df['quality'] == 1, df['enforcement'], np.nan)

    # Exclude Ukraine (matching R code)
    n_ukraine = (df['economy'] == 'Ukraine').sum()
    df = df[df['economy'] != 'Ukraine']
    print(f"\nExcluded Ukraine: {n_ukraine} obs")

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_parquet(CLEAN_DATA_PATH, index=False)
    print(f"\nCleaned data saved: {CLEAN_DATA_PATH}")
    print(f"Final shape: {df.shape}")
    print(f"Countries: {df['id'].nunique()}")
    print(f"Years: {df['year'].min()} - {df['year'].max()}")
    print(f"Countries with childcare law: {df.groupby('id')['has_childcare_law'].first().sum()}")

    # Key outcome variable coverage
    print("\n--- Outcome Variable Coverage ---")
    for var in ['lfpr_f_25_54', 'lfpr_hhchild_f', 'emp2pop_f_yge25', 'unemp_f_yge25']:
        if var in df.columns:
            n = df[var].notna().sum()
            countries_with = df[df[var].notna()]['id'].nunique()
            print(f"  {var:25s}  N={n:6d}  countries={countries_with:3d}")

    return df


if __name__ == '__main__':
    main()
