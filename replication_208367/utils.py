"""
utils.py — Paths, data loaders, and helpers for Andreolli & Surico (2025) replication.

Paper: "Shock Sizes and the Marginal Propensity to Consume"
Data: Italian SHIW (Survey of Household Income and Wealth) 2010, 2012, 2016
"""

import os
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), '208367-V1', 'Data')
STORICO_DIR = os.path.join(DATA_DIR, 'storico_stata')
IND10_DIR = os.path.join(DATA_DIR, 'ind10_stata')
IND12_DIR = os.path.join(DATA_DIR, 'ind12_stata')
IND16_DIR = os.path.join(DATA_DIR, 'ind16_stata')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Data loaders ───────────────────────────────────────────────────────────

def load_storico(name):
    """Load a .dta file from the storico_stata directory."""
    return pd.read_stata(os.path.join(STORICO_DIR, f'{name}.dta'))


def load_ind(year, name):
    """Load a .dta file from the ind{year}_stata directory."""
    dirmap = {2010: IND10_DIR, 2012: IND12_DIR, 2016: IND16_DIR}
    return pd.read_stata(os.path.join(dirmap[year], f'{name}.dta'))


def load_price_data():
    """Load the price Excel files for CPI comparison."""
    ne = os.path.join(DATA_DIR, 'ItalyUSPricesNonEssentials.xls')
    food = os.path.join(DATA_DIR, 'PriceFoodatHomeandAway.xls')
    ne_q = pd.read_excel(ne, sheet_name='Quarterly')
    ne_m = pd.read_excel(ne, sheet_name='Monthly')
    food_m = pd.read_excel(food, sheet_name='Monthly')
    return ne_q, ne_m, food_m


# ── Helper functions ───────────────────────────────────────────────────────

def winsor(s, lower=1, upper=99):
    """Winsorize a Series at given percentiles."""
    lo = np.nanpercentile(s.dropna(), lower)
    hi = np.nanpercentile(s.dropna(), upper)
    return s.clip(lo, hi)


def xtile(s, percentiles):
    """
    Stata-like xtile: assign values to quantile bins.
    percentiles should be like [10, 20, ..., 90] for deciles.
    Returns 1..len(percentiles)+1.
    """
    cuts = np.nanpercentile(s.dropna(), percentiles)
    return np.searchsorted(cuts, s, side='right') + 1


def demean(df, cols, condition=None):
    """
    Demean columns within a condition (boolean mask).
    Returns new column names with _demean suffix.
    """
    if condition is None:
        condition = pd.Series(True, index=df.index)
    result = {}
    for col in cols:
        vals = df.loc[condition, col].astype(float)
        m = vals.mean()
        new = df[col].astype(float).copy()
        new.loc[condition] = vals - m
        new.loc[~condition] = np.nan
        result[f'{col}_demean'] = new
    return pd.DataFrame(result, index=df.index)


def build_main_panel():
    """
    Replicate f01_MergeDatasets.do + f02_ProcessRawData.do:
    Merge SHIW storico data with MPC questions from 2010/2012/2016 supplements,
    then process variables.
    """
    print('  Loading storico files...')
    # Step 1: Merge storico files (f01 lines 18-43)
    rper = load_storico('rper')
    comp = load_storico('comp')
    rfam = load_storico('rfam')
    cons = load_storico('cons')
    ricf = load_storico('ricf')
    peso = load_storico('peso')
    fami = load_storico('fami')
    defl = load_storico('defl')

    # Merge: rper is individual-level (anno, nquest, nord)
    df = rper.copy()
    df = df.merge(comp, on=['anno', 'nquest', 'nord'], how='left', suffixes=('', '_comp'))
    df = df.merge(rfam, on=['anno', 'nquest'], how='left', suffixes=('', '_rfam'))
    df = df.merge(cons, on=['anno', 'nquest'], how='left', suffixes=('', '_cons'))
    df = df.merge(ricf, on=['anno', 'nquest'], how='left', suffixes=('', '_ricf'))
    df = df.merge(peso, on=['anno', 'nquest'], how='left', suffixes=('', '_peso'))
    df = df.merge(fami, on=['anno', 'nquest'], how='left', suffixes=('', '_fami'))
    # Deflator merges on anno only; drop rows where only defl exists
    df = df.merge(defl, on='anno', how='left', suffixes=('', '_defl'))
    print(f'    Panel after storico merge: {len(df):,} rows, years {df.anno.min()}-{df.anno.max()}')

    # Step 2: Add 2010 MPC questions (f01 lines 52-134)
    print('  Adding 2010 MPC supplement...')
    q10e = load_ind(2010, 'q10e')
    q10e['anno'] = 2010
    q10e = q10e.rename(columns={'riscons2': 'mpc_2010'})
    q10c1 = load_ind(2010, 'q10c1')
    q10c2 = load_ind(2010, 'q10c2')
    q10g = load_ind(2010, 'q10g')

    mpc10 = q10e[['nquest', 'anno', 'mpc_2010']].copy()
    # Merge supplementary 2010 files to get financial literacy, risk aversion, etc.
    for extra in [q10c1, q10c2, q10g]:
        cols = [c for c in extra.columns if c != 'nquest']
        mpc10 = mpc10.merge(extra[['nquest'] + cols], on='nquest', how='left', suffixes=('', '_dup'))
        # Drop any duplicate columns
        mpc10 = mpc10[[c for c in mpc10.columns if not c.endswith('_dup')]]

    # Financial literacy
    if 'qint' in mpc10.columns:
        mpc10['lit1_2010'] = (mpc10['qint'] == 2).astype(float)
    else:
        mpc10['lit1_2010'] = np.nan
    if 'qrisk' in mpc10.columns:
        mpc10['lit2_2010'] = (mpc10['qrisk'] == 1).astype(float)
    else:
        mpc10['lit2_2010'] = np.nan
    if 'qmutuo' in mpc10.columns:
        mpc10['lit3_2010'] = (mpc10['qmutuo'] == 2).astype(float)
    else:
        mpc10['lit3_2010'] = np.nan
    mpc10['totlit_2010'] = mpc10['lit1_2010'] + mpc10['lit2_2010'] + mpc10['lit3_2010']

    # Comprehension
    if 'comprens' in mpc10.columns:
        mpc10['comprens_2010'] = mpc10['comprens']

    # Risk aversion
    if 'risfin' in mpc10.columns:
        for i in range(1, 5):
            mpc10[f'riskaver_{i}_2010'] = (mpc10['risfin'] == i).astype(float)

    # Keep only needed columns
    keep_cols = ['anno', 'nquest', 'mpc_2010', 'totlit_2010', 'lit1_2010', 'lit2_2010', 'lit3_2010']
    keep_cols += [c for c in mpc10.columns if c.startswith('comprens_2010') or c.startswith('riskaver_')]
    keep_cols += [c for c in mpc10.columns if c.startswith('discf_')]
    mpc10 = mpc10[[c for c in keep_cols if c in mpc10.columns]]

    df = df.merge(mpc10, on=['anno', 'nquest'], how='left', suffixes=('', '_mpc10'))

    # Step 3: Add 2012 MPC questions (f01 lines 139-227)
    print('  Adding 2012 MPC supplement...')
    q12e = load_ind(2012, 'q12e')
    q12e['anno'] = 2012
    q12e = q12e.rename(columns={'ereditp2': 'mpc_dur_2012', 'ereditp3': 'mpc_nondur_2012'})
    q12e['mpc_2012'] = q12e['mpc_dur_2012'] + q12e['mpc_nondur_2012']

    # Eating out share
    if 'jconsalc' in q12e.columns and 'jconsalf' in q12e.columns:
        q12e['eatathome'] = q12e['jconsalc']
        q12e['eatout'] = q12e['jconsalf']
        q12e['eatoutshare'] = q12e['eatout'] / (q12e['eatout'] + q12e['eatathome']) * 100

    # Merge additional 2012 datasets
    for name in ['q12c1', 'q12c2', 'q12d', 'q12f', 'q12g']:
        extra = load_ind(2012, name)
        cols = [c for c in extra.columns if c != 'nquest']
        q12e = q12e.merge(extra[['nquest'] + cols], on='nquest', how='left', suffixes=('', '_dup'))
        q12e = q12e[[c for c in q12e.columns if not c.endswith('_dup')]]

    keep_cols_12 = ['anno', 'nquest', 'mpc_dur_2012', 'mpc_nondur_2012', 'mpc_2012',
                    'eatathome', 'eatout', 'eatoutshare']
    if 'comprens' in q12e.columns:
        keep_cols_12.append('comprens')
    mpc12 = q12e[[c for c in keep_cols_12 if c in q12e.columns]]

    df = df.merge(mpc12, on=['anno', 'nquest'], how='left', suffixes=('', '_mpc12'))

    # Step 4: Add 2016 MPC questions (f01 lines 231-262)
    print('  Adding 2016 MPC supplement...')
    q16e = load_ind(2016, 'q16e')
    q16e['anno'] = 2016
    if 'riscons2' in q16e.columns:
        q16e = q16e.rename(columns={'riscons2': 'mpc_2016'})
    mpc16 = q16e[['anno', 'nquest', 'mpc_2016']].copy() if 'mpc_2016' in q16e.columns else q16e[['anno', 'nquest']].copy()

    df = df.merge(mpc16, on=['anno', 'nquest'], how='left', suffixes=('', '_mpc16'))

    # ── f02 Processing ──────────────────────────────────────────────────────
    print('  Processing variables (f02)...')

    # Keep head of household only (nord == 1)
    df['nord'] = pd.to_numeric(df['nord'], errors='coerce')
    df = df[df['nord'] == 1].copy()

    # Demographics
    if 'staciv' in df.columns:
        df['married'] = (df['staciv'] == 1).astype(float)
    if 'eta' in df.columns:
        df = df.rename(columns={'eta': 'age'})
    elif 'age' not in df.columns:
        df['age'] = np.nan

    # Education
    if 'studio' in df.columns:
        edu_map = {1: 0, 2: 5, 3: 8, 4: 13, 5: 17, 6: 20}
        df['educ'] = df['studio'].map(edu_map)

    # Housing
    if 'godab' in df.columns:
        df['hown'] = df['godab'].isin([1, 3]).astype(float)

    # Employment
    if 'nonoc' in df.columns:
        df['unempl'] = df['nonoc'].isin([1, 5]).astype(float)
        df['retired'] = (df['nonoc'] == 4).astype(float)

    # Gender
    if 'sesso' in df.columns:
        df['male'] = (df['sesso'] == 1).astype(float)
        df.loc[df['sesso'].isna(), 'male'] = np.nan

    # South
    if 'ireg' in df.columns:
        df['south'] = (df['ireg'] >= 14).astype(float)

    # Cash-on-hand
    if 'af' in df.columns and 'y' in df.columns:
        df['cash'] = df['af'] + df['y']

    # Deflated variables (thousands of euros, then deflate)
    for var in ['c', 'y', 'af', 'cash']:
        if var in df.columns and 'defl' in df.columns:
            df[var] = df[var] / 1000  # to thousands
            df[f'r_{var}'] = df[var] / df['defl']
            df[f'l_{var}'] = np.log(df[f'r_{var}'].clip(lower=1e-10)) * 100
            df[f'l1_{var}'] = np.log(1 + df[f'r_{var}'].clip(lower=0)) * 100

    # Drop negative income and assets
    if 'y' in df.columns:
        df = df[df['y'] >= 0].copy()
    if 'af' in df.columns:
        df = df[df['af'] >= 0].copy()

    # MPC variables: unify and convert to fractions
    df['mpc'] = np.nan
    if 'mpc_2010' in df.columns:
        df.loc[df['anno'] == 2010, 'mpc'] = df.loc[df['anno'] == 2010, 'mpc_2010']
    if 'mpc_2012' in df.columns:
        df.loc[df['anno'] == 2012, 'mpc'] = df.loc[df['anno'] == 2012, 'mpc_2012']
    if 'mpc_2016' in df.columns:
        df.loc[df['anno'] == 2016, 'mpc'] = df.loc[df['anno'] == 2016, 'mpc_2016']

    # Convert MPC from percentage to fraction
    for col in ['mpc', 'mpc_2010', 'mpc_2012', 'mpc_2016', 'mpc_dur_2012', 'mpc_nondur_2012']:
        if col in df.columns:
            df[col] = df[col] / 100

    # Extensive margin
    df['mpc0'] = np.where(df['mpc'].notna(), (df['mpc'] == 0).astype(float), np.nan)
    df['mpc1'] = np.where(df['mpc'].notna(), (df['mpc'] == 1).astype(float), np.nan)

    # Eating out share as fraction
    if 'eatoutshare' in df.columns:
        df['eatoutshare'] = df['eatoutshare'] / 100

    # City size dummies
    if 'acom4c' in df.columns:
        for i in range(1, 5):
            df[f'acomd{i}'] = (df['acom4c'] == i).astype(float)

    # Quintile dummies
    if 'cash' in df.columns:
        for yr in df['anno'].unique():
            mask = df['anno'] == yr
            cash_yr = df.loc[mask, 'cash']
            if cash_yr.notna().sum() > 0:
                df.loc[mask, 'qcash'] = xtile(cash_yr, list(range(20, 81, 20)))
                df.loc[mask, 'q10cash'] = xtile(cash_yr, list(range(10, 91, 10)))
                df.loc[mask, 'q10y'] = xtile(df.loc[mask, 'y'], list(range(10, 91, 10)))
                df.loc[mask, 'q10af'] = xtile(df.loc[mask, 'af'], list(range(10, 91, 10)))

    # Decile and percentile dummies for q10cash (used in regressions)
    if 'q10cash' in df.columns:
        for i in range(1, 11):
            df[f'q10cash_d{i}'] = (df['q10cash'] == i).astype(float)

    # Age dummies
    if 'age' in df.columns:
        df['age30'] = (df['age'] <= 30).astype(float)
        df['age45'] = ((df['age'] > 30) & (df['age'] <= 45)).astype(float)
        df['age60'] = ((df['age'] > 45) & (df['age'] <= 60)).astype(float)

    # Financial liabilities / indebtedness
    if 'pf' in df.columns:
        df['indebt'] = (df['pf'] > 0).astype(float)

    # Durables share
    if 'cd' in df.columns and 'c' in df.columns:
        df['cdur_shr'] = df['cd'] / df['c'] * 100
        df.loc[df['c'] < 0, 'cdur_shr'] = np.nan
        df['cdur_shr'] = winsor(df['cdur_shr'], 1, 99)
        df['cdur_shr_dum'] = np.where(
            df['cdur_shr'].notna(),
            (df['cdur_shr'] > 0).astype(float),
            np.nan
        )

    # Keep only years with MPC questions
    df = df[df['anno'].isin([2010, 2012, 2016])].copy()
    df = df.sort_values(['anno', 'nquest']).reset_index(drop=True)

    print(f'    Final panel: {len(df):,} rows')
    print(f'    Years: {sorted(df.anno.unique())}')
    print(f'    Unique households: {df.nquest.nunique():,}')

    return df


def create_panel_variables(df):
    """
    Replicate f02 panel operations: create lead/lag MPC variables
    by merging across years within the same household (nquest).
    """
    print('  Creating panel lead/lag variables...')

    # Pivot to wide format by year for key variables
    years = sorted(df['anno'].unique())

    # For each household, get their data in each year
    hh_data = {}
    for yr in years:
        yr_df = df[df['anno'] == yr][['nquest', 'mpc', 'mpc0', 'mpc1', 'cash',
                                       'l_cash', 'l1_cash', 'l_c', 'l_y', 'l_af',
                                       'unempl', 'q10cash']
                                      + [c for c in df.columns if c.startswith('mpc_2')]
                                      + [c for c in df.columns if c == 'eatoutshare']
                                      ].copy()
        yr_df = yr_df.drop_duplicates('nquest')
        hh_data[yr] = yr_df.set_index('nquest')

    # For 2010 observations: add 2012 MPC responses (forward 2 years)
    if 2010 in hh_data and 2012 in hh_data:
        data_2012 = hh_data[2012]
        # mpc_2012_in2010: the 2012 MPC response matched to the 2010 observation
        map_2012 = data_2012[['mpc_2012']].rename(columns={'mpc_2012': 'mpc_2012_in2010'})
        df = df.merge(map_2012, left_on='nquest', right_index=True, how='left', suffixes=('', '_f2'))
        # mpc0, mpc1 from 2012 matched to 2010
        if 'mpc0' in data_2012.columns:
            map_mpc0_12 = data_2012[['mpc0']].rename(columns={'mpc0': 'mpc0_2012_in2010'})
            df = df.merge(map_mpc0_12, left_on='nquest', right_index=True, how='left', suffixes=('', '_f2b'))
            map_mpc1_12 = data_2012[['mpc1']].rename(columns={'mpc1': 'mpc1_2012_in2010'})
            df = df.merge(map_mpc1_12, left_on='nquest', right_index=True, how='left', suffixes=('', '_f2c'))
        # mpc_dur, mpc_nondur from 2012 matched to 2010
        if 'mpc_dur_2012' in data_2012.columns:
            map_dur = data_2012[['mpc_dur_2012']].rename(columns={'mpc_dur_2012': 'mpc_dur_2012_in2010'})
            df = df.merge(map_dur, left_on='nquest', right_index=True, how='left', suffixes=('', '_f2d'))
            map_nondur = data_2012[['mpc_nondur_2012']].rename(columns={'mpc_nondur_2012': 'mpc_nondur_2012_in2010'})
            df = df.merge(map_nondur, left_on='nquest', right_index=True, how='left', suffixes=('', '_f2e'))

        # eatoutshare from 2012 matched to 2010
        if 'eatoutshare' in data_2012.columns:
            map_eat = data_2012[['eatoutshare']].rename(columns={'eatoutshare': 'eatoutshare_2012_in2010'})
            df = df.merge(map_eat, left_on='nquest', right_index=True, how='left', suffixes=('', '_f2f'))

        # Log cash/consumption changes
        for var in ['cash', 'c', 'y', 'af']:
            lvar = f'l_{var}'
            if lvar in data_2012.columns:
                fwd = data_2012[[lvar]].rename(columns={lvar: f'l_{var}_2012_in2010'})
                df = df.merge(fwd, left_on='nquest', right_index=True, how='left', suffixes=('', f'_chg{var}'))
                df[f'Dl_{var}_1210'] = df[f'l_{var}_2012_in2010'] - df[f'l_{var}']
                df.loc[df['anno'] != 2010, f'Dl_{var}_1210'] = np.nan

    # Difference in MPC between small and large shock (in 2010 sample)
    if 'mpc_2012_in2010' in df.columns and 'mpc_2010' in df.columns:
        df['dmpc_1012_in2010'] = df['mpc_2010'] - df['mpc_2012_in2010']
        df.loc[df['anno'] != 2010, 'dmpc_1012_in2010'] = np.nan

    # Mark sample for regressions (2010 observations with both MPC responses)
    df['marksample10'] = 0
    mask = (df['anno'] == 2010) & df['dmpc_1012_in2010'].notna() & (df['cash'] > 0) & df['cash'].notna() & df['q10cash'].notna()
    df.loc[mask, 'marksample10'] = 1

    # Percentile dummies for sample (computed on estimation sample year)
    if 'cash' in df.columns:
        mask10 = (df['anno'] == 2010) & (df['mpc'].notna())
        if mask10.sum() > 0:
            df.loc[mask10, 'q100cash'] = xtile(df.loc[mask10, 'cash'], list(range(1, 100)))
            for i in range(1, 101):
                df[f'q100cash_d{i}'] = (df['q100cash'] == i).astype(float)
                df.loc[df['q100cash'].isna(), f'q100cash_d{i}'] = np.nan

    print(f'    marksample10 = 1: {(df["marksample10"] == 1).sum():,} observations')

    return df


def load_analysis_data():
    """Load the processed analysis dataset."""
    path = os.path.join(OUTPUT_DIR, 'analysis_data.parquet')
    return pd.read_parquet(path)
