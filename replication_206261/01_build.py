"""
01_build.py - Build data pipeline for paper 206261
Burchardi, Chaney, Hassan, Tarquinio, Terry -
  Immigration, Innovation, and Growth

Translates ~20 Stata/R scripts into a single Python pipeline.
Phases:
  1. Parse IPUMS census data → immigration/ancestry by (county, country, period)
  2. Population data
  3. Patent outcomes
  4. Native wages, dynamism, QCEW outcomes
  5. Push-pull instrument components
  6. Instrument construction (3-stage)
  7. Final dataset assembly
"""
import pandas as pd
import numpy as np
import os, gc, sys
import warnings
warnings.filterwarnings('ignore')

from utils import (PROJ, INPUT, OUTPUT, TEMP, CODE,
                   VAR_YR, PERIODS, PERIOD_A, PERIOD_AP1,
                   load_dta, read_ipums_fwf, period_to_year, year_to_period,
                   run_reghdfe, run_reghdfe_resid, winsorize,
                   ED_CUTOFF1, ED_CUTOFF2, ED_CUTOFF3, ED_CUTOFF4)

os.makedirs(OUTPUT, exist_ok=True)
os.makedirs(TEMP, exist_ok=True)

# ========================================================================
# IPUMS COLUMN SPECIFICATIONS (from .do files)
# ========================================================================
# Format: list of (name, start_col, end_col) - 1-indexed as in Stata
IPUMS_SPECS = {
    1880: {
        'file': 'usa_00023.dat',
        'cols': [
            ('year', 1, 4), ('statefip', 27, 28), ('county', 29, 32),
            ('perwt', 38, 47), ('age', 48, 50),
            ('bpld', 54, 58), ('bpld_mom', 73, 77), ('bpld_pop', 78, 82),
        ],
        'divides': {'perwt': 100},
    },
    1900: {
        'file': 'usa_00020.dat',
        'cols': [
            ('year', 1, 4), ('statefip', 27, 28), ('county', 29, 32),
            ('perwt', 38, 47), ('age', 48, 50),
            ('bpld', 54, 58), ('yrimmig', 61, 64),
        ],
        'divides': {'perwt': 100},
    },
    1910: {
        'file': 'usa_00017.dat',
        'cols': [
            ('year', 1, 4), ('statefip', 27, 28), ('county', 29, 32),
            ('perwt', 38, 47), ('age', 48, 50),
            ('bpld', 54, 58), ('yrimmig', 61, 64),
        ],
        'divides': {'perwt': 100},
    },
    1920: {
        'file': 'usa_00019.dat',
        'cols': [
            ('year', 1, 4), ('statefip', 27, 28), ('county', 29, 32),
            ('perwt', 38, 47), ('age', 48, 50),
            ('bpld', 54, 58), ('yrimmig', 61, 64),
        ],
        'divides': {'perwt': 100},
    },
    1930: {
        'file': 'usa_00018.dat',
        'cols': [
            ('year', 1, 4), ('statefip', 27, 28), ('county', 29, 32),
            ('perwt', 38, 47), ('age', 48, 50),
            ('bpld', 54, 58), ('yrimmig', 60, 63),
        ],
        'divides': {'perwt': 100},
    },
    1970: {
        'file': 'usa_00024.dat',
        'cols': [
            ('year', 1, 4), ('statefip', 27, 28), ('county', 29, 32),
            ('cntygp97', 36, 40),
            ('perwt', 46, 55), ('age', 56, 58),
            ('bpld', 62, 66), ('yrimmig', 68, 71), ('educ', 73, 74),
        ],
        'divides': {'perwt': 100},
    },
    1980: {
        'file': 'usa_00025.dat',
        'cols': [
            ('year', 1, 4), ('statefip', 27, 28), ('county', 29, 32),
            ('cntygp98', 36, 38),
            ('perwt', 44, 53), ('age', 54, 56),
            ('bpld', 60, 64), ('ancestr1', 65, 67),
            ('yrimmig', 80, 83), ('educ', 91, 92),
        ],
        'divides': {'perwt': 100},
    },
    1990: {
        'file': 'usa_00028.dat',
        'cols': [
            ('year', 1, 4), ('statefip', 27, 28), ('county', 29, 32),
            ('puma', 37, 41),
            ('perwt', 47, 56), ('age', 57, 59),
            ('bpld', 63, 67), ('ancestr1', 68, 70),
            ('yrimmig', 83, 86), ('educ', 94, 95),
        ],
        'divides': {'perwt': 100},
    },
    2000: {
        'file': 'usa_00031.dat',
        'cols': [
            ('year', 1, 4), ('statefip', 27, 28), ('county', 29, 32),
            ('puma', 36, 40),
            ('perwt', 46, 55), ('age', 56, 58),
            ('bpld', 62, 66), ('ancestr1', 67, 69),
            ('yrimmig', 82, 85), ('educ', 95, 96),
        ],
        'divides': {'perwt': 100},
    },
    2010: {
        'file': 'usa_00030.dat',
        'cols': [
            ('year', 1, 4), ('statefip', 44, 45), ('county', 46, 49),
            ('puma', 53, 57),
            ('perwt', 63, 72), ('age', 73, 75),
            ('bpld', 79, 83), ('ancestr1', 84, 86),
            ('yrimmig', 99, 102), ('educ', 112, 113),
        ],
        'divides': {'perwt': 100},
    },
}

# Previous census for each year
PREV_CENSUS = {
    1880: None, 1900: 1880, 1910: 1900, 1920: 1910, 1930: 1920,
    1970: 1930, 1980: 1970, 1990: 1980, 2000: 1990, 2010: 2000,
}

# Period mapping for immigration bins (var_yr=5)
def get_period_bins(year):
    """Get immigration year bins for sub-period aggregation."""
    midyear = year - 5
    prev = PREV_CENSUS[year]
    if year == 1880:
        return None
    elif year == 1970:
        # Special: 1935,1945,1950,1955,1960,1965,1970
        return [(1930, 1935, 1935), (1935, 1945, 1945), (1945, 1950, 1950),
                (1950, 1955, 1955), (1955, 1960, 1960), (1960, 1965, 1965),
                (1965, 1970, 1970)]
    elif year == 1900:
        return [(1880, 1885, 1885), (1885, 1890, 1890),
                (1890, 1895, 1895), (1895, 1900, 1900)]
    else:
        return [(prev, midyear, midyear), (midyear, year, year)]


# ========================================================================
# PHASE 1: IPUMS CENSUS DATA → Immigration & Ancestry
# ========================================================================
def phase1_ipums():
    print("=" * 70)
    print("PHASE 1: PROCESSING IPUMS CENSUS DATA")
    print("=" * 70)

    bpld_all = []
    ancestry_all = []
    education_all = []
    pop_all = []

    for census_year in [1880, 1900, 1910, 1920, 1930, 1970, 1980, 1990, 2000, 2010]:
        print(f"\n--- Processing {census_year} ---")
        spec = IPUMS_SPECS[census_year]

        # Parse fixed-width file
        filepath = f'{INPUT}/IPUMS/{census_year}/{spec["file"]}'
        colspecs = [(c[1], c[2]) for c in spec['cols']]
        colnames = [c[0] for c in spec['cols']]

        print(f"  Reading {filepath}...")
        df = read_ipums_fwf(filepath, colspecs, colnames)
        print(f"  Raw rows: {len(df):,}")

        # Apply divisions
        for col, divisor in spec.get('divides', {}).items():
            df[col] = df[col] / divisor

        # Create geographic identifier and transition to 1990 counties
        if census_year <= 1930:
            df['county_id'] = df['statefip'] * 10000 + df['county']
            trans_file = f'{INPUT}/EMIT/Transitions/Transition{census_year}to1990.dta'
            trans = load_dta(trans_file)
            trans_col = f'county_{census_year}'
            trans_key = f'transition_{census_year}to1990'
            trans = trans.rename(columns={trans_col: 'county_id', trans_key: 'trans_weight'})
            df = df.merge(trans[['county_id', 'state_county_code_1990', 'trans_weight']],
                         on='county_id', how='inner')
            df['count'] = df['perwt'] * df['trans_weight']
        elif census_year in [1970, 1980]:
            # Two-step: county group → county_YEAR → 1990 county
            if census_year == 1970:
                df['countygroup_1970'] = df['cntygp97']
                grp_key = 'countygroup_1970'
            else:
                df['countygroup_1980'] = df['statefip'] * 10000 + df['cntygp98']
                grp_key = 'countygroup_1980'
            # Step 1: county group → intermediate county
            trans_grp = load_dta(f'{INPUT}/EMIT/Transitions/TransitionGroup{census_year}to1990.dta')
            df = df.merge(trans_grp, on=grp_key, how='inner')
            trans_key = f'transition_{census_year}to1990'
            df['count'] = df['perwt'] * df[trans_key]
            # Collapse to intermediate county level
            cty_col = f'county_{census_year}'
            keep_cols = [c for c in ['bpld', 'yrimmig', 'ancestr1', 'educ', 'age'] if c in df.columns]
            df = df.groupby([cty_col] + keep_cols, dropna=False).agg({'count': 'sum'}).reset_index()
            # Step 2: intermediate county → 1990 county
            trans2 = load_dta(f'{INPUT}/EMIT/Transitions/Transition{census_year}to1990.dta')
            trans2_key = f'transition_{census_year}to1990'
            df = df.merge(trans2[[cty_col, 'state_county_code_1990', trans2_key]],
                         on=cty_col, how='inner')
            df['count'] = df['count'] * df[trans2_key]
            df = df.groupby(['state_county_code_1990'] + keep_cols, dropna=False).agg({'count': 'sum'}).reset_index()
        else:  # 1990, 2000, 2010
            cg_col = f'countygroup_{census_year}'
            df[cg_col] = df['statefip'] * 10000 + df['puma']
            # Reshape wide transition matrix to long
            trans_file = f'{INPUT}/EMIT/Original/TransitionMatrix_CountyGroup_{census_year}.dta'
            print(f"  Reshaping wide transition matrix for {census_year}...")
            trans_wide = load_dta(trans_file)
            id_col = cg_col
            val_cols = [c for c in trans_wide.columns if c.startswith('state_county_')]
            trans_long = trans_wide.melt(id_vars=[id_col], value_vars=val_cols,
                                        var_name='scc_str', value_name='trans_weight')
            trans_long = trans_long[trans_long['trans_weight'] > 0]
            trans_long['state_county_code_1990'] = trans_long['scc_str'].str.replace('state_county_', '').astype(float)
            trans_long = trans_long[[id_col, 'state_county_code_1990', 'trans_weight']]
            print(f"  Transition: {len(trans_long)} non-zero entries")

            df = df.merge(trans_long, on=cg_col, how='inner')
            df['count'] = df['perwt'] * df['trans_weight']
            keep_cols = [c for c in ['bpld', 'yrimmig', 'ancestr1', 'educ', 'age'] if c in df.columns]
            df = df.groupby(['state_county_code_1990'] + keep_cols, dropna=False).agg({'count': 'sum'}).reset_index()

        if 'state_county_code_1990' not in df.columns:
            print(f"  WARNING: No state_county_code_1990 for {census_year}, skipping")
            continue

        print(f"  After transition: {len(df):,} rows, {df['state_county_code_1990'].nunique()} counties")

        # --- Population ---
        pop = df.groupby('state_county_code_1990')['count'].sum().reset_index()
        pop.columns = ['state_county_code_1990', f'pop_IPUMS_{census_year}']
        pop_all.append(pop)

        # --- Ancestry (1980+ only) ---
        if census_year >= 1980 and 'ancestr1' in df.columns:
            anc = df.groupby(['state_county_code_1990', 'ancestr1'])['count'].sum().reset_index()
            # Map ancestr1 to country_code_1990
            tc = load_dta(f'{INPUT}/EMIT/Transitions/TransitionCountries.dta')
            anc = anc.merge(tc, on='ancestr1', how='inner')
            anc['count'] = anc['count'] * anc['transitionAncto1990']
            anc = anc.groupby(['state_county_code_1990', 'country_code_1990'])['count'].sum().reset_index()
            anc = anc.rename(columns={'count': f'ancestry_{census_year}'})
            # Mid-year ancestry (filter on recent arrivals or young)
            midyear = census_year - 5
            if 'yrimmig' in df.columns:
                anc_mid = df[(df['yrimmig'] > midyear) | (df['age'] < 5)].copy()
                if len(anc_mid) > 0:
                    anc_mid = anc_mid.groupby(['state_county_code_1990', 'ancestr1'])['count'].sum().reset_index()
                    anc_mid = anc_mid.merge(tc, on='ancestr1', how='inner')
                    anc_mid['count'] = anc_mid['count'] * anc_mid['transitionAncto1990']
                    anc_mid = anc_mid.groupby(['state_county_code_1990', 'country_code_1990'])['count'].sum().reset_index()
                    anc_mid = anc_mid.rename(columns={'count': f'ancestry_{midyear}'})
                    anc = anc.merge(anc_mid, on=['state_county_code_1990', 'country_code_1990'], how='outer')
            ancestry_all.append(anc)

        # --- Birthplace / Immigration ---
        if census_year == 1880:
            # 1880: use bpld, assign missing to parents
            imm = df[['state_county_code_1990', 'bpld', 'count']].copy()
            # Drop ambiguous codes
            bad_codes = [12092, 19900, 21090, 26092, 29900, 30000, 30090, 30091, 49900, 59900, 60000, 60099]
            imm = imm[~imm['bpld'].isin(bad_codes)]
            imm = imm[(imm['bpld'] >= 10000) & (imm['bpld'] < 90000)]
            imm = imm.dropna(subset=['bpld'])
        else:
            # Keep only new immigrants since previous census
            prev = PREV_CENSUS[census_year]
            if 'yrimmig' in df.columns:
                imm = df[df['yrimmig'] > prev].copy() if prev else df.copy()
            else:
                imm = df.copy()

        # Map bpld to country_code_1990
        bpld_trans_file = f'{INPUT}/EMIT/Transitions/TransitionBPLD{census_year}to1990.dta'
        if os.path.exists(bpld_trans_file):
            bpld_trans = load_dta(bpld_trans_file)
            bpld_col = f'bpld_code_{census_year}'
            bpld_trans_key = f'transition_{census_year}to1990'
            if bpld_col not in bpld_trans.columns:
                # Try alternate naming
                bpld_cols = [c for c in bpld_trans.columns if 'bpld' in c.lower()]
                if bpld_cols:
                    bpld_trans = bpld_trans.rename(columns={bpld_cols[0]: 'bpld'})
                else:
                    bpld_trans['bpld'] = bpld_trans.iloc[:, 0]
            else:
                bpld_trans = bpld_trans.rename(columns={bpld_col: 'bpld'})

            if bpld_trans_key not in bpld_trans.columns:
                trans_cols = [c for c in bpld_trans.columns if 'transition' in c.lower()]
                if trans_cols:
                    bpld_trans = bpld_trans.rename(columns={trans_cols[0]: 'bpld_trans'})
                else:
                    bpld_trans['bpld_trans'] = 1.0
            else:
                bpld_trans = bpld_trans.rename(columns={bpld_trans_key: 'bpld_trans'})

            imm = imm.merge(bpld_trans[['bpld', 'country_code_1990', 'bpld_trans']].drop_duplicates(),
                           on='bpld', how='inner')
            imm['count'] = imm['count'] * imm['bpld_trans']
        else:
            print(f"  WARNING: No BPLD transition for {census_year}")
            continue

        # Build sub-period immigration (var_yr=5)
        period_bins = get_period_bins(census_year)

        if census_year == 1880:
            # Simple collapse
            imm_agg = imm.groupby(['state_county_code_1990', 'country_code_1990'])['count'].sum().reset_index()
            imm_agg = imm_agg.rename(columns={'count': 'immigrants_1880'})
            bpld_all.append(imm_agg)
        elif period_bins:
            # Create period bins
            imm_periods = []
            for lo, hi, period_yr in period_bins:
                sub = imm[(imm['yrimmig'] > lo) & (imm['yrimmig'] <= hi)].copy()
                if len(sub) > 0:
                    sub_agg = sub.groupby(['state_county_code_1990', 'country_code_1990'])['count'].sum().reset_index()
                    sub_agg = sub_agg.rename(columns={'count': f'immigrants_{period_yr}'})
                    imm_periods.append(sub_agg)

            if imm_periods:
                result = imm_periods[0]
                for ip in imm_periods[1:]:
                    result = result.merge(ip, on=['state_county_code_1990', 'country_code_1990'], how='outer')
                bpld_all.append(result)
        else:
            # Simple collapse
            imm_agg = imm.groupby(['state_county_code_1990', 'country_code_1990'])['count'].sum().reset_index()
            imm_agg = imm_agg.rename(columns={'count': f'immigrants_{census_year}'})
            bpld_all.append(imm_agg)

        # --- Education data (1980+ only) ---
        if census_year >= 1980 and 'educ' in imm.columns:
            edu = imm[imm['age'] >= 25].copy()
            if len(edu) > 0:
                midyear = census_year - 5
                # Assign period
                if 'yrimmig' in edu.columns:
                    edu['period'] = np.where(edu['yrimmig'] > midyear, census_year, midyear)
                else:
                    edu['period'] = census_year

                # Education years mapping
                educ_map = {0: 0, 1: 2, 2: 6.5, 3: 9, 4: 10, 5: 11, 6: 12,
                           7: 13, 8: 14, 9: 15, 10: 16, 11: 19}
                college_map = {7: 1, 8: 2, 9: 3, 10: 4, 11: 7}

                edu['educ_years'] = edu['educ'].map(educ_map).fillna(0) * edu['count']
                edu['college_years'] = edu['educ'].map(college_map).fillna(0) * edu['count']
                edu['imm_25o'] = edu['count']

                edu_agg = edu.groupby(['state_county_code_1990', 'country_code_1990', 'period']).agg({
                    'educ_years': 'sum', 'college_years': 'sum', 'imm_25o': 'sum'
                }).reset_index()

                education_all.append(edu_agg)

        del df, imm
        gc.collect()
        print(f"  Done with {census_year}")

    # --- Merge all immigration data ---
    print("\n  Merging all immigration periods...")
    if bpld_all:
        bpld_merged = bpld_all[0]
        for bp in bpld_all[1:]:
            bpld_merged = bpld_merged.merge(bp, on=['state_county_code_1990', 'country_code_1990'], how='outer')
        bpld_merged = bpld_merged.fillna(0)
        bpld_merged.to_pickle(f'{TEMP}/BPLDData.pkl')
        print(f"  Immigration data: {bpld_merged.shape}")
    else:
        print("  WARNING: No immigration data built")
        bpld_merged = pd.DataFrame()

    # --- Merge ancestry ---
    if ancestry_all:
        anc_merged = ancestry_all[0]
        for a in ancestry_all[1:]:
            anc_merged = anc_merged.merge(a, on=['state_county_code_1990', 'country_code_1990'], how='outer')
        anc_merged = anc_merged.fillna(0)
        anc_merged.to_pickle(f'{TEMP}/AncestryData.pkl')
        print(f"  Ancestry data: {anc_merged.shape}")
    else:
        anc_merged = pd.DataFrame()

    # --- Merge education ---
    if education_all:
        edu_merged = pd.concat(education_all, ignore_index=True)
        edu_merged.to_pickle(f'{TEMP}/EducationData.pkl')
        print(f"  Education data: {edu_merged.shape}")

    # --- Population ---
    if pop_all:
        pop_merged = pop_all[0]
        for p in pop_all[1:]:
            pop_merged = pop_merged.merge(p, on='state_county_code_1990', how='outer')
        pop_merged.to_pickle(f'{TEMP}/IPUMSPopData.pkl')
        print(f"  Population data: {pop_merged.shape}")

    # --- Final assembly: merge with EMITid ---
    print("\n  Building InstrumentData...")
    emit = load_dta(f'{INPUT}/EMIT/EMITid.dta')
    emit = emit.rename(columns={'country_code1990': 'country_code_1990'})

    if len(bpld_merged) > 0:
        emit = emit.merge(bpld_merged, on=['state_county_code_1990', 'country_code_1990'], how='left')
    if len(anc_merged) > 0:
        emit = emit.merge(anc_merged, on=['state_county_code_1990', 'country_code_1990'], how='left')

    # Fill missing with 0
    imm_cols = [c for c in emit.columns if c.startswith('immigrants_') or c.startswith('ancestry_')]
    emit[imm_cols] = emit[imm_cols].fillna(0)

    # Divide by 1000 (units: thousands)
    for c in imm_cols:
        emit[c] = emit[c] / 1000

    # Drop empty state names
    if 'state_name_1990' in emit.columns:
        emit = emit[emit['state_name_1990'] != '']

    emit.to_pickle(f'{OUTPUT}/InstrumentData.pkl')
    print(f"  InstrumentData: {emit.shape}")
    print("PHASE 1 COMPLETE")
    return emit


# ========================================================================
# PHASE 2: POPULATION DATA
# ========================================================================
def phase2_population():
    print("\n" + "=" * 70)
    print("PHASE 2: POPULATION DATA")
    print("=" * 70)

    pop = load_dta(f'{INPUT}/Population Data/county_population.dta')
    # Convert to float to avoid int8 overflow
    for c in pop.columns:
        pop[c] = pd.to_numeric(pop[c], errors='coerce').astype('float64')
    pop = pop[pop['county_fips'] != 0].copy()
    pop['fipsstcty'] = pop['state_fips'] * 1000 + pop['county_fips']

    # Transition matrices for different reporting vintages
    pop_result = pd.DataFrame()

    # For each population year, map to 1990 counties
    for year in [1970, 1975, 1980, 1985, 1990, 1995, 2000, 2005]:
        pop_col = f'pop{year}' if year % 10 == 0 else f'pop{year}'
        if pop_col not in pop.columns:
            # Try alternate
            pop_col = f'pop{year}'
            if pop_col not in pop.columns:
                print(f"  Skipping {year}: no column {pop_col}")
                continue

        if year <= 1985:
            # 1970-1985: reported as 1980 counties → transition to 1990
            trans = load_dta(f'{INPUT}/EMIT/Transitions/Transition1980to1990.dta')
            trans_key = [c for c in trans.columns if 'transition' in c.lower()][0]
            cty_col = [c for c in trans.columns if 'county' in c.lower() and '1990' not in c.lower()][0]
            sub = pop[['fipsstcty', pop_col]].copy()
            sub = sub.rename(columns={'fipsstcty': cty_col})
            sub[cty_col] = sub[cty_col] * 10
            sub = sub.merge(trans, on=cty_col, how='inner')
            sub[f'population_{year}'] = sub[pop_col] * sub[trans_key]
            sub = sub.groupby('state_county_code_1990')[f'population_{year}'].sum().reset_index()
        elif year <= 1995:
            # 1990-1995: reported as 2000 counties → transition to 1990
            trans_file = f'{INPUT}/EMIT/Transitions/Transition1980to1990.dta'
            # Actually need 2000→1990 if available
            # Use fipsstcty * 10 as state_county_code_1990 directly for 1990 counties
            sub = pop[['fipsstcty', pop_col]].copy()
            sub['state_county_code_1990'] = sub['fipsstcty'] * 10
            sub = sub.groupby('state_county_code_1990')[pop_col].sum().reset_index()
            sub = sub.rename(columns={pop_col: f'population_{year}'})
        else:
            # 2000+: reported as 2010 counties
            sub = pop[['fipsstcty', pop_col]].copy()
            sub['state_county_code_1990'] = sub['fipsstcty'] * 10
            sub = sub.groupby('state_county_code_1990')[pop_col].sum().reset_index()
            sub = sub.rename(columns={pop_col: f'population_{year}'})

        if len(pop_result) == 0:
            pop_result = sub
        else:
            pop_result = pop_result.merge(sub, on='state_county_code_1990', how='outer')

    # 2010 from ACS
    acs_file = f'{INPUT}/Population Data/ACS_10_5YR_B01003_with_ann.csv'
    if os.path.exists(acs_file):
        try:
            acs = pd.read_csv(acs_file, skiprows=1)
            # Extract county FIPS from GEO.id2
            id_col = [c for c in acs.columns if 'id2' in c.lower() or 'geo' in c.lower()][0] if any('id2' in c.lower() or 'geo' in c.lower() for c in acs.columns) else acs.columns[1]
            pop_col_acs = [c for c in acs.columns if 'estimate' in c.lower() or 'total' in c.lower()]
            if not pop_col_acs:
                pop_col_acs = [acs.columns[3]] if len(acs.columns) > 3 else [acs.columns[-1]]
            acs['state_county_code_1990'] = pd.to_numeric(acs.iloc[:, 1], errors='coerce') * 10
            acs['population_2010'] = pd.to_numeric(acs[pop_col_acs[0]], errors='coerce')
            acs = acs[['state_county_code_1990', 'population_2010']].dropna()
            pop_result = pop_result.merge(acs, on='state_county_code_1990', how='outer')
        except Exception as e:
            print(f"  Warning: Could not read ACS: {e}")

    # Divide by 1000
    pop_cols = [c for c in pop_result.columns if c.startswith('population_')]
    for c in pop_cols:
        pop_result[c] = pop_result[c] / 1000

    pop_result.to_pickle(f'{OUTPUT}/PopulationData.pkl')
    print(f"  Population data: {pop_result.shape}")

    # Population growth (5-year differences)
    pop_growth_records = []
    for t in range(PERIOD_A, PERIODS + 1):
        yr = period_to_year(t)
        yr_prev = yr - 5
        col_now = f'population_{yr}'
        col_prev = f'population_{yr_prev}'
        if col_now in pop_result.columns and col_prev in pop_result.columns:
            g = pop_result[['state_county_code_1990', col_now, col_prev]].copy()
            g['deltapop'] = g[col_now] - g[col_prev]
            g['time'] = t
            pop_growth_records.append(g[['state_county_code_1990', 'time', 'deltapop']])

    if pop_growth_records:
        pop_growth = pd.concat(pop_growth_records, ignore_index=True)
        pop_growth.to_pickle(f'{OUTPUT}/PopulationGrowth.pkl')
        print(f"  Population growth: {pop_growth.shape}")

    print("PHASE 2 COMPLETE")
    return pop_result


# ========================================================================
# PHASE 3: PATENT DATA
# ========================================================================
def phase3_patents(pop_data):
    print("\n" + "=" * 70)
    print("PHASE 3: PATENT DATA")
    print("=" * 70)

    # Read patent application dates
    print("  Reading patent applications...")
    app = pd.read_csv(f'{INPUT}/Patents/g_application.tsv', sep='\t',
                      usecols=['patent_id', 'filing_date'], dtype={'patent_id': str})
    app['appyear'] = pd.to_datetime(app['filing_date'], errors='coerce').dt.year
    app = app.dropna(subset=['appyear'])
    app['appyear'] = app['appyear'].astype(int)
    print(f"  Applications: {len(app):,}")

    # Read patent grant info (utility patents)
    print("  Reading patent grants...")
    pat = pd.read_csv(f'{INPUT}/Patents/g_patent.tsv', sep='\t',
                      usecols=['patent_id', 'patent_type', 'patent_date'],
                      dtype={'patent_id': str})
    pat = pat[pat['patent_type'] == 'utility']
    pat['grant_year'] = pd.to_datetime(pat['patent_date'], errors='coerce').dt.year
    print(f"  Utility patents: {len(pat):,}")

    # Merge app year
    pat = pat.merge(app[['patent_id', 'appyear']], on='patent_id', how='inner')
    pat = pat[(pat['appyear'] >= 1975) & (pat['appyear'] <= 2018)]
    print(f"  After year filter: {len(pat):,}")

    # Read assignee locations
    print("  Reading assignees...")
    assignee = pd.read_csv(f'{INPUT}/Patents/g_assignee_disambiguated.tsv', sep='\t',
                           dtype={'patent_id': str}, low_memory=False)
    # Keep corporate assignees (type 2 or 3)
    if 'assignee_type' in assignee.columns:
        assignee = assignee[assignee['assignee_type'].isin([2, 3, '2', '3'])]

    # Read locations
    print("  Reading locations...")
    locs = pd.read_csv(f'{INPUT}/Patents/g_location_disambiguated.tsv', sep='\t')
    locs = locs.rename(columns={'location_id': 'disambig_assignee_location_id'})

    # Merge assignee → location
    loc_col = [c for c in assignee.columns if 'location' in c.lower()][0] if any('location' in c.lower() for c in assignee.columns) else None
    if loc_col:
        assignee = assignee.merge(locs[['disambig_assignee_location_id', 'state_fips', 'county_fips']].drop_duplicates(),
                                  left_on=loc_col, right_on='disambig_assignee_location_id', how='inner')
    else:
        print("  WARNING: No location column in assignee data")

    # Create county code
    assignee['state_fips'] = pd.to_numeric(assignee['state_fips'], errors='coerce')
    assignee['county_fips'] = pd.to_numeric(assignee['county_fips'], errors='coerce')
    assignee = assignee.dropna(subset=['state_fips', 'county_fips'])
    assignee['state_county_code_1990'] = (assignee['state_fips'] * 1000 + assignee['county_fips']) * 10

    # Keep first assignee per patent
    assignee = assignee.drop_duplicates(subset='patent_id', keep='first')

    # Merge with patent data
    pat = pat.merge(assignee[['patent_id', 'state_county_code_1990']], on='patent_id', how='inner')
    print(f"  Patents with location: {len(pat):,}")

    # Read citations
    print("  Reading citations (this may take a while)...")
    try:
        cit = pd.read_csv(f'{INPUT}/Patents/g_us_patent_citation.tsv', sep='\t',
                          usecols=['patent_id', 'citation_patent_id'],
                          dtype={'patent_id': str, 'citation_patent_id': str})
        cit_counts = cit.groupby('patent_id').size().reset_index(name='citations')
        pat = pat.merge(cit_counts, on='patent_id', how='left')
        pat['citations'] = pat['citations'].fillna(0)
        del cit
        gc.collect()
    except Exception as e:
        print(f"  WARNING: Could not read citations: {e}")
        pat['citations'] = 1

    # Read NBER categories for citation adjustment
    try:
        nber = pd.read_csv(f'{INPUT}/Patents/nber.tsv', sep='\t',
                          dtype={'patent_id': str})
        if 'category' in nber.columns:
            pat = pat.merge(nber[['patent_id', 'category']], on='patent_id', how='left')
            pat['category'] = pat['category'].fillna(6)
            # Adjust citations by grant-year x category mean
            means = pat.groupby(['grant_year', 'category'])['citations'].transform('mean')
            pat['citations_adj'] = pat['citations'] / means.clip(lower=0.01)
        else:
            pat['citations_adj'] = pat['citations']
    except:
        pat['citations_adj'] = pat['citations']

    # Collapse to county-year
    pat['npat_a'] = 1
    pat['npat_aN'] = pat['citations_adj']
    county_year = pat.groupby(['state_county_code_1990', 'appyear']).agg({
        'npat_a': 'sum', 'npat_aN': 'sum'
    }).reset_index()

    # Aggregate to 5-year periods
    print("  Building 5-year patent flows...")
    period_data = []
    for t in range(PERIOD_A, PERIODS + 1):
        yr = period_to_year(t)
        yr_prev = yr - 5

        # Flow: patents filed in (yr_prev, yr]
        flow = county_year[(county_year['appyear'] > yr_prev) & (county_year['appyear'] <= yr)]
        flow_agg = flow.groupby('state_county_code_1990').agg({'npat_a': 'sum', 'npat_aN': 'sum'}).reset_index()
        flow_agg['time'] = t
        flow_agg = flow_agg.rename(columns={'npat_a': 'PatentFlow_a', 'npat_aN': 'PatentFlow_aN'})

        # Cumulative stock
        cum = county_year[county_year['appyear'] <= yr]
        cum_agg = cum.groupby('state_county_code_1990').agg({'npat_a': 'sum', 'npat_aN': 'sum'}).reset_index()
        cum_agg = cum_agg.rename(columns={'npat_a': 'PatentNum_a', 'npat_aN': 'PatentNum_aN'})

        merged = flow_agg.merge(cum_agg, on='state_county_code_1990', how='outer')
        merged['time'] = t
        period_data.append(merged)

    patents = pd.concat(period_data, ignore_index=True).fillna(0)

    # Per capita (per 100k using 1970 population)
    if pop_data is not None and 'population_1970' in pop_data.columns:
        pop70 = pop_data[['state_county_code_1990', 'population_1970']].copy()
        pop70['pop100k'] = pop70['population_1970'] / 100  # thousands → 100k
        patents = patents.merge(pop70[['state_county_code_1990', 'pop100k']], on='state_county_code_1990', how='left')
        patents['pop100k'] = patents['pop100k'].clip(lower=0.001)

        for v in ['a', 'aN']:
            patents[f'deltaPatpc70_{v}'] = patents[f'PatentNum_{v}'] / patents['pop100k']

        # First difference of per-capita flow
        patents = patents.sort_values(['state_county_code_1990', 'time'])
        for v in ['a', 'aN']:
            patents[f'deltaPatpc70_{v}_lag'] = patents.groupby('state_county_code_1990')[f'deltaPatpc70_{v}'].shift(1)
            patents[f'diffPV_{v}'] = patents[f'deltaPatpc70_{v}'] - patents[f'deltaPatpc70_{v}_lag']

        # Winsorize
        for v in ['a', 'aN']:
            if f'diffPV_{v}' in patents.columns:
                patents[f'diffPV_{v}_W'] = winsorize(patents[f'diffPV_{v}'].dropna())
                # Fill back
                patents[f'diffPV_{v}_W'] = patents[f'diffPV_{v}'].clip(
                    patents[f'diffPV_{v}'].quantile(0.01),
                    patents[f'diffPV_{v}'].quantile(0.99)
                )

    # IHS transformation
    for v in ['a', 'aN']:
        col = f'deltaPatpc70_{v}'
        if col in patents.columns:
            patents[f'IHS_{v}_'] = np.arcsinh(patents[col])

    patents.to_pickle(f'{OUTPUT}/PatentsData.pkl')
    print(f"  Patent data: {patents.shape}")
    print("PHASE 3 COMPLETE")
    return patents


# ========================================================================
# PHASE 4: QCEW WAGES & DYNAMISM
# ========================================================================
def phase4_outcomes():
    print("\n" + "=" * 70)
    print("PHASE 4: QCEW WAGES & DYNAMISM")
    print("=" * 70)

    # --- PCE Price Index ---
    pce = pd.read_csv(f'{INPUT}/Index/PCEPI.csv')
    # Normalize to 2010
    pce_col = [c for c in pce.columns if 'pce' in c.lower() or 'index' in c.lower() or 'value' in c.lower()]
    if pce_col:
        pce_vals = pce[pce_col[0]]
    else:
        pce_vals = pce.iloc[:, 1]
    year_col = [c for c in pce.columns if 'year' in c.lower() or 'date' in c.lower()][0] if any('year' in c.lower() or 'date' in c.lower() for c in pce.columns) else pce.columns[0]

    # --- QCEW Data ---
    print("  Processing QCEW wages...")
    qcew_records = []
    for year in [1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010]:
        qfile = f'{INPUT}/QCEW Data/QCEWAnnualAvg{year}.csv'
        if not os.path.exists(qfile):
            continue
        try:
            q = pd.read_csv(qfile)
            # Filter: county level (5-digit FIPS), total covered (own_code=0), all industries
            if 'area_fips' in q.columns:
                q['fips'] = pd.to_numeric(q['area_fips'], errors='coerce')
            else:
                fips_col = [c for c in q.columns if 'fips' in c.lower() or 'area' in c.lower()][0]
                q['fips'] = pd.to_numeric(q[fips_col], errors='coerce')

            # Keep county-level total covered
            q = q[q['fips'].between(1001, 56999)]  # valid county FIPS
            if 'own_code' in q.columns:
                q['own_code'] = pd.to_numeric(q['own_code'], errors='coerce')
                q = q[q['own_code'] == 0]  # Total covered
            if 'agglvl_code' in q.columns:
                q['agglvl_code'] = pd.to_numeric(q['agglvl_code'], errors='coerce')
                q = q[q['agglvl_code'].isin([70, 71, 72, 73, 74, 75, 76, 77, 78])]  # County level

            q['wage'] = pd.to_numeric(q['avg_annual_pay'], errors='coerce') if 'avg_annual_pay' in q.columns else np.nan
            q = q.dropna(subset=['fips', 'wage'])
            q['state_county_code_1990'] = q['fips'] * 10
            q['year'] = year

            qcew_records.append(q[['state_county_code_1990', 'year', 'wage']].drop_duplicates(
                subset=['state_county_code_1990'], keep='first'))
            print(f"    {year}: {len(q)} rows → {len(qcew_records[-1])} counties")
        except Exception as e:
            print(f"  Warning: QCEW {year}: {e}")

    if qcew_records:
        qcew = pd.concat(qcew_records, ignore_index=True)
        # Map to time periods
        qcew['time'] = qcew['year'].apply(year_to_period)
        # 5-year difference / 100
        qcew = qcew.sort_values(['state_county_code_1990', 'time'])
        qcew['wage_lag'] = qcew.groupby('state_county_code_1990')['wage'].shift(1)
        qcew['delta_avg_wage_adj'] = (qcew['wage'] - qcew['wage_lag']) / 100
        qcew = qcew[qcew['time'] >= PERIOD_AP1]
        qcew.to_pickle(f'{OUTPUT}/QCEWData.pkl')
        print(f"  QCEW data: {qcew.shape}")

    # --- Business Dynamism ---
    print("  Processing dynamism data...")
    bds_file = f'{INPUT}/Dynamism Data/BDS_msa_7714.dta'
    if os.path.exists(bds_file):
        bds = load_dta(bds_file)
        print(f"  BDS data: {bds.shape}, columns: {list(bds.columns)[:10]}")
        bds.to_pickle(f'{OUTPUT}/DynStats.pkl')
    else:
        print("  No BDS file found")

    # --- County dynamism from CBP ---
    dyn_dir = f'{INPUT}/Dynamism Data/County'
    if os.path.exists(dyn_dir):
        dyn_files = [f for f in os.listdir(dyn_dir) if f.endswith('.csv') or f.endswith('.dta')]
        print(f"  Found {len(dyn_files)} county dynamism files")

    print("PHASE 4 COMPLETE")


# ========================================================================
# PHASE 5: PUSH-PULL INSTRUMENT COMPONENTS
# ========================================================================
def phase5_instruments(instrument_data=None):
    print("\n" + "=" * 70)
    print("PHASE 5: PUSH-PULL INSTRUMENT COMPONENTS")
    print("=" * 70)

    if instrument_data is None:
        instrument_data = pd.read_pickle(f'{OUTPUT}/InstrumentData.pkl')

    df = instrument_data.copy()
    print(f"  InstrumentData: {df.shape}")

    # Identify European countries
    is_euro = df['continent'] == 'Europe'
    print(f"  European obs: {is_euro.sum():,}, Non-European: {(~is_euro).sum():,}")

    # Get all immigrant columns
    imm_cols = sorted([c for c in df.columns if c.startswith('immigrants_')])
    print(f"  Immigration periods: {imm_cols}")

    # For each period, compute aggregates
    # immigration_d_t = total immigrants to county d
    # immigration_o_t = total immigrants from country o to US
    # immigration_o_dv_t = immigrants from o to division r(d)
    # immigration_o_ndv_t = immigration_o_t - immigration_o_dv_t (push factor)
    # immigration_Euro_d_t = European immigrants to county d
    # immigration_Euro_t = total European immigrants to US
    # immigration_Euro_dv_t = European immigrants to division r(d)

    push_pull_vars = []
    for imm_col in imm_cols:
        period_yr = imm_col.replace('immigrants_', '')
        print(f"  Computing push-pull for period {period_yr}...")

        # Total from country o to US
        imm_o = df.groupby('country_code_1990')[imm_col].sum()
        df[f'immigration_o_{period_yr}'] = df['country_code_1990'].map(imm_o)

        # Total from country o to division r(d)
        imm_o_dv = df.groupby(['country_code_1990', 'division'])[imm_col].sum()
        df[f'immigration_o_dv_{period_yr}'] = df.set_index(['country_code_1990', 'division']).index.map(
            lambda x: imm_o_dv.get(x, 0) if x in imm_o_dv.index else 0
        )
        # Simpler approach
        imm_o_dv_df = df.groupby(['country_code_1990', 'division'])[imm_col].sum().reset_index()
        imm_o_dv_df = imm_o_dv_df.rename(columns={imm_col: f'immigration_o_dv_{period_yr}'})
        df = df.drop(columns=[f'immigration_o_dv_{period_yr}'], errors='ignore')
        df = df.merge(imm_o_dv_df, on=['country_code_1990', 'division'], how='left')
        df[f'immigration_o_dv_{period_yr}'] = df[f'immigration_o_dv_{period_yr}'].fillna(0)

        # Push factor: leave out own division
        df[f'immigration_o_ndv_{period_yr}'] = df[f'immigration_o_{period_yr}'] - df[f'immigration_o_dv_{period_yr}']

        # European immigrants to county d
        euro_mask = df['continent'] == 'Europe'
        euro_d = df[euro_mask].groupby('state_county_code_1990')[imm_col].sum().reset_index()
        euro_d = euro_d.rename(columns={imm_col: f'immigration_Euro_d_{period_yr}'})
        df = df.merge(euro_d, on='state_county_code_1990', how='left')
        df[f'immigration_Euro_d_{period_yr}'] = df[f'immigration_Euro_d_{period_yr}'].fillna(0)

        # Total European to US
        euro_total = df[euro_mask][imm_col].sum()
        df[f'immigration_Euro_{period_yr}'] = euro_total

        # European to division
        euro_dv = df[euro_mask].groupby('division')[imm_col].sum().reset_index()
        euro_dv = euro_dv.rename(columns={imm_col: f'immigration_Euro_dv_{period_yr}'})
        df = df.merge(euro_dv, on='division', how='left')
        df[f'immigration_Euro_dv_{period_yr}'] = df[f'immigration_Euro_dv_{period_yr}'].fillna(0)

        # Non-European to county d
        noneuro_d = df[~euro_mask].groupby('state_county_code_1990')[imm_col].sum().reset_index()
        noneuro_d = noneuro_d.rename(columns={imm_col: f'immigration_nEuro_d_{period_yr}'})
        df = df.merge(noneuro_d, on='state_county_code_1990', how='left')
        df[f'immigration_nEuro_d_{period_yr}'] = df[f'immigration_nEuro_d_{period_yr}'].fillna(0)

        # Push x Pull: immi_Euro_ndv_X_o = push * (Euro_d / Euro_total)
        euro_share_d = df[f'immigration_Euro_d_{period_yr}'] / df[f'immigration_Euro_{period_yr}'].clip(lower=1e-10)
        df[f'immi_Euro_ndv_X_o_{period_yr}'] = df[f'immigration_o_ndv_{period_yr}'] * euro_share_d

        # Set to NaN for European countries (they are instruments, not endogenous)
        df.loc[df['continent'] == 'Europe', f'immi_Euro_ndv_X_o_{period_yr}'] = np.nan

        # Regional adjustment: share of European immigration going to division
        immi_dv_shareEuro = df[f'immigration_Euro_dv_{period_yr}'] / df[f'immigration_Euro_{period_yr}'].clip(lower=1e-10)
        df[f'immi_dv_shareEuro_{period_yr}'] = immi_dv_shareEuro

        push_pull_vars.append(f'immi_Euro_ndv_X_o_{period_yr}')

    # Time-series orthogonalization
    print("\n  Time-orthogonalizing push-pull variables...")
    for i, v in enumerate(push_pull_vars):
        if i > 0:
            prior = push_pull_vars[:i]
            mask = df[v].notna()
            if mask.sum() > len(prior):
                X = df.loc[mask, prior].fillna(0).values
                y = df.loc[mask, v].values
                try:
                    from numpy.linalg import lstsq
                    beta, _, _, _ = lstsq(X, y, rcond=None)
                    df.loc[mask, v] = y - X @ beta
                except:
                    pass

    # Regional adjustment
    for imm_col in imm_cols:
        period_yr = imm_col.replace('immigrants_', '')
        v = f'immi_Euro_ndv_X_o_{period_yr}'
        share_col = f'immi_dv_shareEuro_{period_yr}'
        if v in df.columns and share_col in df.columns:
            ra_factor = df[share_col] / (1 - df[share_col]).clip(lower=1e-10)
            df[f'{v}_RA'] = df[v] * ra_factor

    # FE orthogonalization (reghdfe style)
    print("  FE-orthogonalizing instruments...")
    for imm_col in imm_cols:
        period_yr = imm_col.replace('immigrants_', '')
        ra_col = f'immi_Euro_ndv_X_o_{period_yr}_RA'
        if ra_col not in df.columns:
            continue
        try:
            resid = run_reghdfe_resid(
                df, ra_col, ['dist', 'distance_lat'],
                ['region_country_code', 'county_continent_code'],
                'country_code_1990'
            )
            df[f'immi_Euro_ndv_X_o_RAorth_{period_yr}'] = resid
        except Exception as e:
            print(f"  Warning: FE orth failed for {period_yr}: {e}")
            df[f'immi_Euro_ndv_X_o_RAorth_{period_yr}'] = df[ra_col]

    # Orthogonalize ancestry variables too
    anc_cols = [c for c in df.columns if c.startswith('ancestry_')]
    for ac in anc_cols:
        try:
            resid = run_reghdfe_resid(
                df, ac, ['dist', 'distance_lat'],
                ['region_country_code', 'county_continent_code'],
                'country_code_1990'
            )
            year_str = ac.replace('ancestry_', '')
            df[f'ancestry_orth_{year_str}'] = resid
        except Exception as e:
            df[f'ancestry_orth_{ac.replace("ancestry_", "")}'] = df[ac]

    # Drop European countries from analysis sample
    df_nonEuro = df[df['continent'] != 'Europe'].copy()
    print(f"  Non-European sample: {df_nonEuro.shape}")

    df_nonEuro.to_pickle(f'{OUTPUT}/IIGData1.pkl')
    print("PHASE 5 COMPLETE")
    return df_nonEuro


# ========================================================================
# PHASE 6: INSTRUMENT CONSTRUCTION (3-STAGE)
# ========================================================================
def phase6_instruments(iig_data=None):
    print("\n" + "=" * 70)
    print("PHASE 6: INSTRUMENT CONSTRUCTION (3-STAGE)")
    print("=" * 70)

    if iig_data is None:
        iig_data = pd.read_pickle(f'{OUTPUT}/IIGData1.pkl')

    df = iig_data.copy()

    # Get orthogonalized instrument columns
    orth_cols = sorted([c for c in df.columns if 'RAorth' in c])
    anc_orth_cols = sorted([c for c in df.columns if c.startswith('ancestry_orth_')])

    if not orth_cols or not anc_orth_cols:
        print("  WARNING: Missing orthogonalized variables. Using raw instruments.")
        # Fallback: use raw push-pull variables
        orth_cols = sorted([c for c in df.columns if c.startswith('immi_Euro_ndv_X_o_') and '_RA' not in c and 'orth' not in c])
        anc_orth_cols = sorted([c for c in df.columns if c.startswith('ancestry_') and 'orth' not in c])

    print(f"  Instrument cols: {len(orth_cols)}")
    print(f"  Ancestry cols: {len(anc_orth_cols)}")

    # ---- STAGE 0: Predict Ancestry ----
    print("\n  Stage 0: Predicting ancestry from push-pull instruments...")

    # For each ancestry year, regress on all instruments × division
    divisions = sorted(df['division'].dropna().unique())
    print(f"  Divisions: {divisions}")

    # Create division dummies
    div_dummies = pd.get_dummies(df['division'], prefix='div', drop_first=True)
    for c in div_dummies.columns:
        df[c] = div_dummies[c].values

    ancestry_hat_cols = []
    for anc_col in anc_orth_cols:
        year_str = anc_col.replace('ancestry_orth_', '')
        print(f"    Predicting ancestry_{year_str}...")

        # Create instrument × division interactions
        interaction_cols = []
        for oc in orth_cols:
            for dc in div_dummies.columns:
                icol = f'{oc}_x_{dc}'
                df[icol] = df[oc].fillna(0) * df[dc]
                interaction_cols.append(icol)

        # Run regression: ancestry_orth = f(instrument × division)
        mask = df[anc_col].notna()
        if mask.sum() < len(interaction_cols) + 10:
            print(f"    Skipping {anc_col}: too few obs")
            continue

        y = df.loc[mask, anc_col].values
        X = df.loc[mask, interaction_cols].fillna(0).values

        try:
            from numpy.linalg import lstsq
            beta, _, _, _ = lstsq(X, y, rcond=None)
            hat_col = f'ancestry_hat_{year_str}'
            df[hat_col] = 0.0
            df.loc[mask, hat_col] = X @ beta
            ancestry_hat_cols.append(hat_col)
        except Exception as e:
            print(f"    Warning: Stage 0 failed for {anc_col}: {e}")

        # Clean up interaction cols
        df = df.drop(columns=interaction_cols)

    print(f"  Predicted ancestry columns: {len(ancestry_hat_cols)}")

    # ---- Build triple interaction: Ahat_X_immihat_o_ndv ----
    print("\n  Building triple interactions (predicted ancestry × push × regional adj)...")

    # For each period t (PERIOD_AP1 to PERIODS), create:
    # Ahat_X_immihat_o_ndv_t = immigration_o_ndv_t * regional_adj_t * ancestry_hat_{year(t-1)}
    imm_cols = sorted([c for c in df.columns if c.startswith('immigrants_')])
    triple_cols = []

    for t in range(PERIOD_AP1, PERIODS + 1):
        yr = period_to_year(t)
        yr_prev = yr - 5

        # Find the relevant immigration push column
        push_col = f'immigration_o_ndv_{yr}'
        if push_col not in df.columns:
            # Try alternate period naming
            alt_push = [c for c in df.columns if c.startswith('immigration_o_ndv_') and str(yr) in c]
            if alt_push:
                push_col = alt_push[0]
            else:
                continue

        # Regional adjustment
        share_col = f'immi_dv_shareEuro_{yr}'
        if share_col in df.columns:
            ra_factor = df[share_col] / (1 - df[share_col]).clip(lower=1e-10)
        else:
            ra_factor = 1.0

        # Predicted ancestry from prior period
        hat_col = f'ancestry_hat_{yr_prev}'
        if hat_col not in df.columns:
            # Try with year directly
            hat_col = f'ancestry_hat_{yr}'
            if hat_col not in df.columns:
                # Use first available
                if ancestry_hat_cols:
                    hat_col = ancestry_hat_cols[-1]
                else:
                    continue

        triple_name = f'Ahat_X_immihat_o_ndv_{t}'
        df[triple_name] = df[push_col] * ra_factor * df.get(hat_col, 0)
        triple_cols.append(triple_name)

    print(f"  Triple interaction columns: {len(triple_cols)}")

    # ---- STAGE 1: Predict Immigration ----
    print("\n  Stage 1: Predicting bilateral immigration...")

    # Need to reshape to panel: for each period t, stack the data
    # Immigration outcome: immigrants from country o to county d in period t
    panel_records = []
    for imm_col in imm_cols:
        period_yr = imm_col.replace('immigrants_', '')
        t = year_to_period(int(period_yr))
        if t < PERIOD_AP1:
            continue

        sub = df[['state_county_code_1990', 'country_code_1990', imm_col,
                  'dist', 'distance_lat', 'region_country_code', 'county_continent_code'] +
                 [c for c in triple_cols if c in df.columns]].copy()
        sub['immigrants_'] = sub[imm_col]
        sub['time'] = t
        panel_records.append(sub)

    if not panel_records:
        print("  WARNING: No panel data for Stage 1")
        return None

    panel = pd.concat(panel_records, ignore_index=True)
    print(f"  Stage 1 panel: {panel.shape}")

    # Run reghdfe: immigrants_ ~ triple_cols + dist + distance_lat, absorb(region_country, county_continent, time)
    xvars_s1 = [c for c in triple_cols if c in panel.columns] + ['dist', 'distance_lat']
    xvars_s1 = [c for c in xvars_s1 if c in panel.columns]

    res_s1 = run_reghdfe(panel, 'immigrants_', xvars_s1,
                         ['region_country_code', 'county_continent_code', 'time'],
                         'country_code_1990')
    print(f"  Stage 1: N={res_s1['n']}, R2={res_s1.get('r2', 0):.4f}")

    # Get predicted values from just the triple interaction terms
    triple_idx = [xvars_s1.index(c) for c in triple_cols if c in xvars_s1 and c in res_s1['xvars']]
    if triple_idx:
        # Compute predicted bilateral immigration
        all_vars = ['immigrants_'] + xvars_s1 + ['region_country_code', 'county_continent_code', 'time', 'country_code_1990', 'state_county_code_1990']
        all_vars = list(dict.fromkeys(all_vars))
        sub = panel[all_vars].dropna().copy()

        # Simple approach: just use fitted values from the triple interactions
        X_triple = sub[[c for c in triple_cols if c in sub.columns]].fillna(0).values
        beta_triple = np.array([res_s1['b'][res_s1['xvars'].index(c)] for c in triple_cols
                               if c in res_s1['xvars']])
        if len(beta_triple) == X_triple.shape[1]:
            sub['IhatM1_o_d'] = X_triple @ beta_triple
        else:
            sub['IhatM1_o_d'] = res_s1.get('fitted', np.zeros(len(sub)))

        # Aggregate to county-time level: IhatM1_d = sum over countries
        IhatM1 = sub.groupby(['state_county_code_1990', 'time'])['IhatM1_o_d'].sum().reset_index()
        IhatM1 = IhatM1.rename(columns={'IhatM1_o_d': 'IhatM1_d'})

        # Also get actual non-European immigration at county-time level
        nEuro_imm = sub.groupby(['state_county_code_1990', 'time'])['immigrants_'].sum().reset_index()
        nEuro_imm = nEuro_imm.rename(columns={'immigrants_': 'immigration_nEuro_d_'})

        instruments = IhatM1.merge(nEuro_imm, on=['state_county_code_1990', 'time'], how='outer')

        # IHS transformations
        instruments['IhatM1_d_IHS'] = np.arcsinh(instruments['IhatM1_d'])
        instruments['immigration_nEuro_d_IHS'] = np.arcsinh(instruments['immigration_nEuro_d_'])

        # 10-year versions
        instruments = instruments.sort_values(['state_county_code_1990', 'time'])
        instruments['IhatM1_d_lag2'] = instruments.groupby('state_county_code_1990')['IhatM1_d'].shift(2)
        instruments['imm_lag2'] = instruments.groupby('state_county_code_1990')['immigration_nEuro_d_'].shift(2)
        instruments['IhatM1_d10y'] = instruments['IhatM1_d'] + instruments['IhatM1_d_lag2'].fillna(0)
        instruments['immigration_nEuro_d_10y'] = instruments['immigration_nEuro_d_'] + instruments['imm_lag2'].fillna(0)

        instruments.to_pickle(f'{OUTPUT}/IIGData_Instruments_1.pkl')
        print(f"  Instruments: {instruments.shape}")

        # Save Stage 1 data for Table 2
        panel.to_pickle(f'{OUTPUT}/IIGData_Stage1.pkl')

        print("PHASE 6 COMPLETE")
        return instruments
    else:
        print("  WARNING: No triple interactions in regression")
        return None


# ========================================================================
# PHASE 7: FINAL DATASET ASSEMBLY
# ========================================================================
def phase7_final(instruments=None, patents=None):
    print("\n" + "=" * 70)
    print("PHASE 7: FINAL DATASET ASSEMBLY")
    print("=" * 70)

    if instruments is None:
        instruments = pd.read_pickle(f'{OUTPUT}/IIGData_Instruments_1.pkl')

    base = instruments.copy()
    print(f"  Base (instruments): {base.shape}")

    # Merge patents
    if patents is None and os.path.exists(f'{OUTPUT}/PatentsData.pkl'):
        patents = pd.read_pickle(f'{OUTPUT}/PatentsData.pkl')
    if patents is not None:
        patent_cols = ['state_county_code_1990', 'time'] + \
                      [c for c in patents.columns if c.startswith('diffPV_') or c.startswith('IHS_') or c.startswith('deltaPat')]
        patent_cols = list(dict.fromkeys(patent_cols))
        patent_cols = [c for c in patent_cols if c in patents.columns]
        base = base.merge(patents[patent_cols], on=['state_county_code_1990', 'time'], how='left')
        print(f"  After patents merge: {base.shape}")

    # Merge population growth
    if os.path.exists(f'{OUTPUT}/PopulationGrowth.pkl'):
        popg = pd.read_pickle(f'{OUTPUT}/PopulationGrowth.pkl')
        base = base.merge(popg, on=['state_county_code_1990', 'time'], how='left')

    # Merge QCEW (collapse to county-time first to avoid dups from industry/ownership codes)
    if os.path.exists(f'{OUTPUT}/QCEWData.pkl'):
        qcew = pd.read_pickle(f'{OUTPUT}/QCEWData.pkl')
        qcew_cty = qcew.groupby(['state_county_code_1990', 'time']).agg({
            'delta_avg_wage_adj': 'mean', 'wage': 'mean'
        }).reset_index()
        base = base.merge(qcew_cty[['state_county_code_1990', 'time', 'delta_avg_wage_adj']],
                         on=['state_county_code_1990', 'time'], how='left')

    # Merge population density
    if os.path.exists(f'{OUTPUT}/PopulationData.pkl'):
        popd = pd.read_pickle(f'{OUTPUT}/PopulationData.pkl')
        # Create density for relevant years
        base = base.merge(popd[['state_county_code_1990']].drop_duplicates(),
                         on='state_county_code_1990', how='left')

    # Create state factor
    base['state_code'] = (base['state_county_code_1990'] / 10000).astype(int)
    base['state_factor'] = base['state_code']

    # Add 25+ immigration variable from education data
    if os.path.exists(f'{TEMP}/EducationData.pkl'):
        edu = pd.read_pickle(f'{TEMP}/EducationData.pkl')
        # Collapse to county-time: drop European countries
        # Get continent info from EMITid
        emit = load_dta(f'{INPUT}/EMIT/EMITid.dta')
        emit = emit.rename(columns={'country_code1990': 'country_code_1990'})
        euro_countries = emit[emit['continent'] == 'Europe']['country_code_1990'].unique()
        edu = edu[~edu['country_code_1990'].isin(euro_countries)]

        edu['time'] = edu['period'].apply(year_to_period)
        edu_cty = edu.groupby(['state_county_code_1990', 'time']).agg({
            'imm_25o': 'sum', 'educ_years': 'sum', 'college_years': 'sum'
        }).reset_index()
        edu_cty['educ_years_avg'] = edu_cty['educ_years'] / edu_cty['imm_25o'].clip(lower=1)
        edu_cty['college_years_avg'] = edu_cty['college_years'] / edu_cty['imm_25o'].clip(lower=1)
        edu_cty['imm_25o'] = edu_cty['imm_25o'] / 1000  # Convert to thousands
        edu_cty = edu_cty.rename(columns={'imm_25o': 'immigration_25o_'})

        # Demean education variables
        edu_cty['educ_years_dem'] = edu_cty['educ_years_avg'] - edu_cty['educ_years_avg'].mean()
        edu_cty['college_years_dem'] = edu_cty['college_years_avg'] - edu_cty['college_years_avg'].mean()

        # 10-year versions
        edu_cty = edu_cty.sort_values(['state_county_code_1990', 'time'])
        edu_cty['immigration_25o_lag'] = edu_cty.groupby('state_county_code_1990')['immigration_25o_'].shift(1)
        edu_cty['immigration_25o_10y'] = edu_cty['immigration_25o_'] + edu_cty['immigration_25o_lag'].fillna(0)

        base = base.merge(edu_cty[['state_county_code_1990', 'time', 'immigration_25o_',
                                   'educ_years_dem', 'college_years_dem', 'immigration_25o_10y']],
                         on=['state_county_code_1990', 'time'], how='left')

    # Filter: keep t >= 20 and non-missing immigration
    base = base[base['time'] >= PERIOD_AP1].copy()
    base = base.dropna(subset=['immigration_nEuro_d_'])
    print(f"  After filters (t>=20, non-missing imm): {base.shape}")

    # Create state name mapping
    state_map = {1:'AL',2:'AK',4:'AZ',5:'AR',6:'CA',8:'CO',9:'CT',10:'DE',11:'DC',12:'FL',
                 13:'GA',15:'HI',16:'ID',17:'IL',18:'IN',19:'IA',20:'KS',21:'KY',22:'LA',
                 23:'ME',24:'MD',25:'MA',26:'MI',27:'MN',28:'MS',29:'MO',30:'MT',31:'NE',
                 32:'NV',33:'NH',34:'NJ',35:'NM',36:'NY',37:'NC',38:'ND',39:'OH',40:'OK',
                 41:'OR',42:'PA',44:'RI',45:'SC',46:'SD',47:'TN',48:'TX',49:'UT',50:'VT',
                 51:'VA',53:'WA',54:'WV',55:'WI',56:'WY'}
    base['state_name'] = base['state_code'].map(state_map)

    base.to_pickle(f'{OUTPUT}/IIGFinalData.pkl')
    print(f"\n  FINAL DATASET: {base.shape}")
    print(f"  Counties: {base['state_county_code_1990'].nunique()}")
    print(f"  Time periods: {sorted(base['time'].unique())}")
    print(f"  States: {base['state_code'].nunique()}")

    print("PHASE 7 COMPLETE")
    return base


# ========================================================================
# MAIN
# ========================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("BUILDING DATASET FOR PAPER 206261")
    print("Immigration, Innovation, and Growth")
    print("=" * 70)

    # Phase 1: IPUMS
    instrument_data = phase1_ipums()

    # Phase 2: Population
    pop_data = phase2_population()

    # Phase 3: Patents
    patents = phase3_patents(pop_data)

    # Phase 4: QCEW, Dynamism
    phase4_outcomes()

    # Phase 5: Push-pull components
    iig_data = phase5_instruments(instrument_data)

    # Phase 6: 3-stage instruments
    instruments = phase6_instruments(iig_data)

    # Phase 7: Final assembly
    if instruments is not None:
        final = phase7_final(instruments, patents)
    else:
        print("\nWARNING: Instrument construction failed. Building partial dataset.")

    print("\n" + "=" * 70)
    print("BUILD COMPLETE")
    print("=" * 70)
