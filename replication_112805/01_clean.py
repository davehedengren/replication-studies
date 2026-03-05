"""
01_clean.py - Data cleaning and VAM estimation
Replicates the Stata .do file: creates covariates, estimates VAMs, builds analysis dataset.
"""
import sys
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (load_data, create_sample_indicators, create_year_vars,
                   create_school_cohort_fe, create_test_scores, create_imputed_scores,
                   create_year_test_scores, create_peer_scores,
                   estimate_vam_ar, estimate_vam_fe, estimate_vam_mix,
                   GRADES, MODELS, ESTIMATORS, SAMPLES, OUTCOMES, SCHOOL_TYPES,
                   OUTPUT_DIR)

print("=" * 70)
print("PHASE 1: DATA CLEANING AND VARIABLE CONSTRUCTION")
print("=" * 70)

# Load data
df = load_data()
print(f"Loaded data: {df.shape[0]} obs, {df.shape[1]} vars")

# Create sample indicators
df = create_sample_indicators(df)
print(f"Sample (grades 4-8, miss_02==0): {df['sample'].sum()}")
print(f"On-margin sample: {df['onmargin_sample'].sum()}")

# Create year variables
df = create_year_vars(df)

# Create test scores
df = create_test_scores(df)

# Create imputed scores with missing indicators
df = create_imputed_scores(df)

# Create yearly test scores
df = create_year_test_scores(df)

# Create school-by-cohort FE
df = create_school_cohort_fe(df)

# Create peer prior test scores
df = create_peer_scores(df)

print("\n" + "=" * 70)
print("PHASE 2: VALUE-ADDED MODEL ESTIMATION")
print("=" * 70)

# Estimate VAMs for each grade, model, estimator, and sample period
# Only test outcome (average of math + reading) and Models 1 & 2

vam_results = {}  # (model, estimator, sample, outcome, grade) -> Series

for t in OUTCOMES:
    for g in GRADES:
        lag = g - 1
        school_col = f'school_{g}'
        outcome_col = f'{t}_{g}'

        # Base sample: not on-margin
        base_mask = (df['onmargin_sample'] != 1)

        # Model 1 covariates: none (levels only)
        mod1_covs = []

        # Model 2 covariates: lagged scores polynomial + missing flags
        mod2_covs = [f'math_{lag}_imp', f'read_{lag}_imp',
                     f'math_{lag}_imp_sq', f'read_{lag}_imp_sq',
                     f'math_{lag}_imp_cub', f'read_{lag}_imp_cub',
                     f'math_{lag}_miss', f'read_{lag}_miss']

        for m, covs in [(1, mod1_covs), (2, mod2_covs)]:
            # --- Sample: 2002 only ---
            mask_02 = base_mask & (df[f'year_{g}'] == 2002)

            # Average residual (ar)
            print(f"  Model {m}, AR, 02, {t}, grade {g}...", end=" ")
            vam = estimate_vam_ar(df, outcome_col, covs, school_col, mask_02)
            vam_results[(m, 'ar', '02', t, g)] = vam
            print(f"N_schools={len(vam)}")

            # Mixed effects (mix)
            print(f"  Model {m}, MIX, 02, {t}, grade {g}...", end=" ")
            vam = estimate_vam_mix(df, outcome_col, covs, school_col, mask_02)
            vam_results[(m, 'mix', '02', t, g)] = vam
            print(f"N_schools={len(vam)}")

            # Fixed effects (FE)
            print(f"  Model {m}, FE, 02, {t}, grade {g}...", end=" ")
            vam = estimate_vam_fe(df, outcome_col, covs, school_col, mask_02)
            vam_results[(m, 'FE', '02', t, g)] = vam
            print(f"N_schools={len(vam)}")

            # --- Sample: 2-year (2001-2002) ---
            mask_2yr = base_mask & (df[f'year_{g}'] >= 2001) & (df[f'year_{g}'] <= 2002)

            # For multi-year samples, include year dummies
            year_dummies_2yr = []
            if mask_2yr.any():
                sub_years = df.loc[mask_2yr, f'year_{g}'].dropna().unique()
                if len(sub_years) > 1:
                    for yr in sorted(sub_years)[1:]:  # drop first for reference
                        dname = f'_yr_{int(yr)}'
                        df[dname] = (df[f'year_{g}'] == yr).astype(float)
                        year_dummies_2yr.append(dname)

            covs_2yr = covs + year_dummies_2yr

            print(f"  Model {m}, AR, 2yr, {t}, grade {g}...", end=" ")
            vam = estimate_vam_ar(df, outcome_col, covs_2yr, school_col, mask_2yr)
            vam_results[(m, 'ar', '2yr', t, g)] = vam
            print(f"N_schools={len(vam)}")

            print(f"  Model {m}, MIX, 2yr, {t}, grade {g}...", end=" ")
            vam = estimate_vam_mix(df, outcome_col, covs_2yr, school_col, mask_2yr)
            vam_results[(m, 'mix', '2yr', t, g)] = vam
            print(f"N_schools={len(vam)}")

            print(f"  Model {m}, FE, 2yr, {t}, grade {g}...", end=" ")
            vam = estimate_vam_fe(df, outcome_col, covs_2yr, school_col, mask_2yr)
            vam_results[(m, 'FE', '2yr', t, g)] = vam
            print(f"N_schools={len(vam)}")

            # --- Sample: all years (<=2002) ---
            mask_all = base_mask & (df[f'year_{g}'] <= 2002)

            year_dummies_all = []
            if mask_all.any():
                sub_years = df.loc[mask_all, f'year_{g}'].dropna().unique()
                if len(sub_years) > 1:
                    for yr in sorted(sub_years)[1:]:
                        dname = f'_yr_all_{int(yr)}'
                        df[dname] = (df[f'year_{g}'] == yr).astype(float)
                        year_dummies_all.append(dname)

            covs_all = covs + year_dummies_all

            print(f"  Model {m}, AR, all, {t}, grade {g}...", end=" ")
            vam = estimate_vam_ar(df, outcome_col, covs_all, school_col, mask_all)
            vam_results[(m, 'ar', 'all', t, g)] = vam
            print(f"N_schools={len(vam)}")

            print(f"  Model {m}, MIX, all, {t}, grade {g}...", end=" ")
            vam = estimate_vam_mix(df, outcome_col, covs_all, school_col, mask_all)
            vam_results[(m, 'mix', 'all', t, g)] = vam
            print(f"N_schools={len(vam)}")

            print(f"  Model {m}, FE, all, {t}, grade {g}...", end=" ")
            vam = estimate_vam_fe(df, outcome_col, covs_all, school_col, mask_all)
            vam_results[(m, 'FE', 'all', t, g)] = vam
            print(f"N_schools={len(vam)}")

print(f"\nTotal VAM estimates computed: {len(vam_results)}")

# === CREATE SCHOOL-LEVEL VA DATASETS ===
print("\n" + "=" * 70)
print("PHASE 3: CREATE SCHOOL-LEVEL VA DATASETS")
print("=" * 70)

va_by_grade = {}
for g in GRADES:
    school_col = f'school_{g}'
    va_df = pd.DataFrame()

    for key, vam in vam_results.items():
        m, e, s, t, grade = key
        if grade != g:
            continue
        col_name = f'mod{m}{e}_{s}_{t}_{g}'
        if len(vam) > 0:
            va_df[col_name] = vam

    if len(va_df) > 0:
        va_df.index.name = 'school'
        va_by_grade[g] = va_df
        print(f"Grade {g}: {len(va_df)} schools, {len(va_df.columns)} VAM measures")

# Merge all grades into one school-level dataset
va_schl = None
for g in GRADES:
    if g in va_by_grade:
        temp = va_by_grade[g].copy()
        temp.index.name = 'school'
        if va_schl is None:
            va_schl = temp
        else:
            va_schl = va_schl.join(temp, how='outer')

# Apply Stata filters (drop school==0, certain school ranges)
if va_schl is not None:
    va_schl = va_schl[va_schl.index != 0]
    va_schl = va_schl[~((va_schl.index > 5999) & (va_schl.index < 8000))]
    va_schl = va_schl[~((va_schl.index > 5999) & (va_schl.index != 8482))]

    # Set to missing for certain school/grade combos
    for col in va_schl.columns:
        if col.endswith('_4') or col.endswith('_5'):
            va_schl.loc[va_schl.index > 5000, col] = np.nan
        if col.endswith('_6') or col.endswith('_7') or col.endswith('_8'):
            va_schl.loc[(va_schl.index > 4000) & (va_schl.index < 5000), col] = np.nan

    print(f"\nCombined school VA dataset: {len(va_schl)} schools, {len(va_schl.columns)} columns")

# === RESTRICT TO ANALYSIS SAMPLE AND MERGE VA ===
print("\n" + "=" * 70)
print("PHASE 4: MERGE VA TO STUDENT DATA")
print("=" * 70)

# Keep analysis sample
analysis = df[df['sample'] == 1].copy()
print(f"Analysis sample: {len(analysis)} students")

# Create school variables for merging
analysis['school_as'] = analysis['schl_d20']
for c in [1, 2, 3]:
    analysis[f'school_ch{c}'] = analysis[f'choice{c}_schl']
analysis['school_hm'] = analysis['home0203']

# Merge VA for each school type
for schl_type in SCHOOL_TYPES:
    school_col = f'school_{schl_type}'
    if va_schl is None:
        continue

    for m in MODELS:
        for e in ESTIMATORS:
            for s in SAMPLES:
                for t in OUTCOMES:
                    # Current year VA
                    va_col_name = f'{schl_type}_mod{m}{e}_{s}_{t}'
                    analysis[va_col_name] = np.nan

                    # Lead VA (for 2004 outcomes)
                    lead_col_name = f'{schl_type}lead_mod{m}{e}_{s}_{t}'
                    analysis[lead_col_name] = np.nan

                    for g in range(4, 8):  # 4-7 for current, lead is g+1
                        lead = g + 1
                        va_grade_col = f'mod{m}{e}_{s}_{t}_{g}'
                        va_lead_col = f'mod{m}{e}_{s}_{t}_{lead}'

                        if va_schl is not None:
                            mask = analysis['future_grd'] == g
                            if va_grade_col in va_schl.columns:
                                va_map = va_schl[va_grade_col]
                                analysis.loc[mask, va_col_name] = analysis.loc[mask, school_col].map(va_map)
                            if va_lead_col in va_schl.columns:
                                analysis.loc[mask, lead_col_name] = analysis.loc[mask, school_col].map(va_map)

                    # Grade 8 for current year
                    g = 8
                    va_grade_col = f'mod{m}{e}_{s}_{t}_{g}'
                    if va_schl is not None and va_grade_col in va_schl.columns:
                        mask = analysis['future_grd'] == 8
                        va_map = va_schl[va_grade_col]
                        analysis.loc[mask, va_col_name] = analysis.loc[mask, school_col].map(va_map)

# Create alternative counterfactual VA
for m in MODELS:
    for e in ESTIMATORS:
        for s in SAMPLES:
            for t in OUTCOMES:
                ch2_col = f'ch2_mod{m}{e}_{s}_{t}'
                ch3_col = f'ch3_mod{m}{e}_{s}_{t}'
                hm_col = f'hm_mod{m}{e}_{s}_{t}'
                alt_col = f'alt_mod{m}{e}_{s}_{t}'

                if all(c in analysis.columns for c in [ch2_col, ch3_col, hm_col]):
                    analysis[alt_col] = (
                        analysis['margin2'] * analysis[ch2_col] +
                        ((1 - analysis['margin2']) *
                         (analysis['margin3'] * analysis[ch3_col] +
                          (1 - analysis['margin3']) * analysis[hm_col]))
                    )

# Create grade-home FE for Table A1
analysis['grade_home'] = analysis.groupby(['home0203', 'future_grd']).ngroup()

print(f"Final analysis dataset: {len(analysis)} obs, {len(analysis.columns)} columns")

# Check VA coverage
for e in ESTIMATORS:
    for s in SAMPLES:
        col = f'as_mod2{e}_{s}_test'
        if col in analysis.columns:
            valid = analysis[col].notna().sum()
            print(f"  {col}: {valid} valid ({valid/len(analysis)*100:.1f}%)")

# Save
output_path = os.path.join(OUTPUT_DIR, 'analysis_data.pkl')
analysis.to_pickle(output_path)
print(f"\nSaved analysis dataset to {output_path}")

# Also save VA school data
if va_schl is not None:
    va_path = os.path.join(OUTPUT_DIR, 'va_school.pkl')
    va_schl.to_pickle(va_path)
    print(f"Saved school VA data to {va_path}")

print("\n=== CLEANING COMPLETE ===")
