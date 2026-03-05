"""
01_clean.py — Data Cleaning (translates clean_bl_PPS.do + clean_fu_PPS.do)

Builds the analysis dataset from raw survey data.
Pipeline: raw_data -> baseline wide -> follow-up wide -> merge -> reshape long -> analysis_data

Key variables constructed:
  - wtp_std: WTP normalized by within-good SD (full sample)
  - girl: child is female (varies by good—older vs younger child)
  - mom: respondent is female
  - momXgirl: mother × daughter interaction
  - adult_wtp_std: adult good WTP (benchmark control)
  - childgood, humcap, health, hypo, toys_bin: good category indicators
  - _G*: good fixed effects
  - young: good is for younger reference child
  - fu: follow-up wave indicator
  - strat: randomization stratum
  - hhid: household ID
"""

import sys, os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category=UnicodeWarning)

sys.path.insert(0, os.path.dirname(__file__))
from utils import RAW_DATA_DIR, OUTPUT_DIR, ANALYSIS_DATA_PATH


def recode_yes_no(series):
    """Convert Stata-style '01 Yes'/'02 No' to 1/0."""
    s = series.astype(str)
    result = pd.Series(np.nan, index=series.index)
    result[s.str.contains('01 Yes|^1$|^1\\.0$', na=False)] = 1.0
    result[s.str.contains('02 No|^0$|^0\\.0$', na=False)] = 0.0
    result[s.str.contains('-888', na=False)] = 0.0  # "Does Not Know" -> No
    return result


def extract_wtp_baseline(df, var_prefix, letters, low_price, price_step,
                         bonus_var=None, bonus_amt_var=None, bonus_is_add=False,
                         check_var=None, first_q_var=None):
    """
    Extract WTP from BDM price list (baseline).
    Letters are in order a,b,c,... (highest to lowest price).
    low_price: lowest price on the list.
    price_step: increment between prices.
    """
    n_items = len(letters)

    # Recode price list responses
    for letter in letters:
        col = f'{var_prefix}_{letter}'
        if col in df.columns:
            df[col] = recode_yes_no(df[col])

    # Initialize WTP
    if first_q_var and first_q_var in df.columns:
        # Only defined when first question is answered
        wtp = pd.Series(np.nan, index=df.index)
        has_data = df[first_q_var].notna() & (df[first_q_var] != '')
        wtp[has_data] = 0.0
    else:
        wtp = pd.Series(0.0, index=df.index)

    # Walk from lowest to highest price; last "yes" wins
    price = low_price
    for letter in reversed(letters):
        col = f'{var_prefix}_{letter}'
        if col in df.columns:
            yes = (df[col] == 1.0)
            wtp = np.where(yes, price, wtp)
        price += price_step

    wtp = pd.Series(wtp, index=df.index, dtype=float)

    # Bonus adjustment
    if bonus_var and bonus_var in df.columns:
        df[bonus_var] = recode_yes_no(df[bonus_var])
        bonus_yes = (df[bonus_var] == 1.0)
        if bonus_is_add:
            # Add fixed amount (deworm, shoes)
            wtp = np.where(bonus_yes, wtp + bonus_amt_var, wtp)
        elif bonus_amt_var and isinstance(bonus_amt_var, str) and bonus_amt_var in df.columns:
            # Replace with bonus amount column
            bonus_val = pd.to_numeric(df[bonus_amt_var], errors='coerce')
            wtp = np.where(bonus_yes & bonus_val.notna(), bonus_val, wtp)

    wtp = pd.Series(wtp, index=df.index, dtype=float)

    # Check variable: if WTP is 0, use check value
    if check_var and check_var in df.columns:
        check = pd.to_numeric(df[check_var], errors='coerce')
        wtp = np.where((wtp == 0) & check.notna() & (check >= 0), check, wtp)

    return pd.Series(wtp, index=df.index, dtype=float)


def clean_baseline():
    """Clean baseline survey data (translates clean_bl_PPS.do)."""
    print("Loading baseline data...")

    # Load randomized data - only keep what we need
    rand = pd.read_stata(os.path.join(RAW_DATA_DIR, 'randomized_data.dta'),
                         convert_categoricals=False)
    # Columns that exist in the data
    rand_keep = ['pid', 'strat', 'd04_childgender_', 'a06_class', 'schoolcode']
    rand = rand[[c for c in rand_keep if c in rand.columns]].copy()

    # Load survey data
    survey = pd.read_stata(os.path.join(RAW_DATA_DIR, 'product_pricing_hh_final.dta'),
                           convert_categoricals=False)

    df = rand.merge(survey, on='pid', how='inner')
    print(f"  Merged baseline: {len(df)} obs")

    # Drop ineligible
    if 's0q9_capability' in df.columns:
        df = df[df['s0q9_capability'].astype(str).str.contains('01 Yes', na=False)]

    # Drop based on consent
    if 's0q24_consent' in df.columns:
        consent = df['s0q24_consent'].astype(str)
        module2 = df.get('_m_pps_hh_2_dem1_c', pd.Series(0, index=df.index))
        # Drop if consent=Yes but module2 completed (means they dropped out)
        df = df[~((consent == '01 Yes') & (module2 == 1))]
        # Drop if consent=No
        df = df[consent != '02 No']
        # Drop if consent blank and module2 completed
        df = df[~((consent == '') & (module2 == 1))]

    print(f"  After eligibility: {len(df)} obs")

    # Parent gender
    df['dad'] = (df['s0q4_r_sex'].astype(str).str.contains('01 Male', na=False)).astype(float)
    df.loc[(df['dad'] == 0) & ~df['s0q4_r_sex'].astype(str).str.contains('02 Female', na=False), 'dad'] = np.nan
    df['mom'] = 1 - df['dad']

    # ---- WTP EXTRACTION ----

    # Cups: 3600 down to 300, step 300, 12 items (a-l)
    df['cup_wtp'] = extract_wtp_baseline(
        df, 's4q3', list('abcdefghijkl'),
        low_price=300, price_step=300,
        bonus_var='s4q3_maxcup_granular', bonus_amt_var='s4q3_maxcup_100',
        check_var='s4q4_highpri')

    # Tests: 6000 down to 500, step 500, 12 items (a-l)
    df['test_wtp'] = extract_wtp_baseline(
        df, 's4q13', list('abcdefghijkl'),
        low_price=500, price_step=500,
        bonus_var='s4q3_maxtest_bonus', bonus_amt_var='s4q3_maxtest_100',
        check_var='s4q14_highpri')

    # Test WTP corrections (matching Stata)
    df.loc[df['pid'] == 4332, 'test_wtp'] = np.nan

    # Sieve: 2000 down to 200, step 200, 10 items (a-j)
    df['sieve_wtp'] = extract_wtp_baseline(
        df, 's4q37', list('abcdefghij'),
        low_price=200, price_step=200,
        bonus_var='s4_max_wtp_sieve', bonus_amt_var='s4_max100',
        check_var='s4q38_highpri')

    # Toy: 10000 down to 1000, step 1000, 10 items (a-j)
    df['toy_wtp'] = extract_wtp_baseline(
        df, 's4q43', list('abcdefghij'),
        low_price=1000, price_step=1000,
        bonus_var='s4_max_wtp_toy', bonus_amt_var='s4_max500',
        check_var='s4q44_highpri')
    df.loc[df['toy_wtp'] < 0, 'toy_wtp'] = np.nan

    # Deworming: 4000 down to 500, step 500, 8 items (a-h)
    df['deworm_wtp'] = extract_wtp_baseline(
        df, 's10q1', list('abcdefgh'),
        low_price=500, price_step=500,
        bonus_var='s10q1_maxplus_tabs', bonus_amt_var=200, bonus_is_add=True,
        check_var='s10q2_wtptabs', first_q_var='s10q1_a')

    # Shoes price list A (older child): 2500 down to 100, step 200, 13 items (a-m)
    df['shoes_a_wtp'] = extract_wtp_baseline(
        df, 's10q4', list('abcdefghijklm'),
        low_price=100, price_step=200,
        bonus_var='s10q4_x', bonus_amt_var=100, bonus_is_add=True,
        check_var='s10q5_wtpshoes', first_q_var='s10q4_a')

    # Shoes price list B (younger child): 2000 down to 200, step 200, 10 items (n-w)
    df['shoes_b_wtp'] = extract_wtp_baseline(
        df, 's10q4', list('nopqrstuvw'),
        low_price=200, price_step=200,
        bonus_var='s10q4_x', bonus_amt_var=100, bonus_is_add=True,
        check_var='s10q5_wtpshoes', first_q_var='s10q4_n')

    # Combine shoes: A if available, else B
    df['shoes_wtp'] = df['shoes_a_wtp'].copy()
    df.loc[df['shoes_wtp'].isna(), 'shoes_wtp'] = df.loc[df['shoes_wtp'].isna(), 'shoes_b_wtp']
    df['shoes_mkt'] = np.where(df['shoes_a_wtp'].notna(), 2500,
                               np.where(df['shoes_b_wtp'].notna(), 2000, np.nan))
    df['older_shoes'] = df['shoes_b_wtp'].isna() & df['shoes_a_wtp'].notna()

    # Set shoes missing for ineligible PIDs
    inelig_shoes = [2029, 2084, 2274, 2544, 2746, 2805, 2952, 3185, 3198,
                    3361, 3541, 3667, 3740, 3771, 3864, 3902, 3979, 3995,
                    4533, 4656, 4983, 5049]
    df.loc[df['pid'].isin(inelig_shoes), 'shoes_wtp'] = np.nan

    # Drop if test WTP is missing
    n_before = len(df)
    df = df[df['test_wtp'].notna()].copy()
    print(f"  Dropped {n_before - len(df)} obs with missing test WTP. N={len(df)}")

    # ---- OTHER VARIABLES ----

    # Daughter (baseline): matches Stata: daughter = 1 if "1 female" OR missing
    df['daughter'] = ((df['d04_childgender_'].astype(str) == '1 female') |
                      df['d04_childgender_'].isna() |
                      (df['d04_childgender_'].astype(str).str.strip() == '')).astype(float)

    # Shoes reference child gender (younger child gender)
    if 's10_genderchosen' in df.columns:
        df['shoes_daughter'] = (df['s10_genderchosen'].astype(str) == '01 Female').astype(float)
        df.loc[df['s10_genderchosen'].isna() | (df['s10_genderchosen'].astype(str).str.strip() == ''),
               'shoes_daughter'] = np.nan
    else:
        df['shoes_daughter'] = np.nan

    # s10_totalkids_s: needed for shoes drop condition
    if 's10_totalkids_s' in df.columns:
        df['has_s10_totalkids_s'] = df['s10_totalkids_s'].notna()
    else:
        df['has_s10_totalkids_s'] = False

    # Compensation
    if 's0_comp' in df.columns:
        df['comp'] = pd.to_numeric(df['s0_comp'], errors='coerce')
    elif 's0q25_compensation1' in df.columns:
        df['comp'] = pd.to_numeric(df['s0q25_compensation1'], errors='coerce')
    else:
        df['comp'] = 8000.0

    df['hhid'] = df['pid']

    # Enumerator ID
    df['enumerator_id'] = df.get('enumerator_id', pd.Series(np.nan, index=df.index))

    # lovemore_resp: respondent thinks they care more about children
    if 's9q6_care' in df.columns:
        df['Blovemore_resp'] = (df['s9q6_care'].astype(str).str.contains('You', na=False)).astype(float)
        df.loc[df['s9q6_care'].astype(str).isin(['', 'nan']), 'Blovemore_resp'] = np.nan
    else:
        df['Blovemore_resp'] = np.nan

    # Stratum
    if 'strat' not in df.columns:
        df['strat'] = 1  # fallback

    print(f"  Baseline cleaned: {len(df)} obs, {df['mom'].notna().sum()} with parent gender")
    return df


def clean_followup():
    """Clean follow-up survey data (translates clean_fu_PPS.do)."""
    print("\nLoading follow-up data...")

    # Load randomized data
    rand = pd.read_stata(os.path.join(RAW_DATA_DIR, 'randomized_data.dta'),
                         convert_categoricals=False)
    rand_keep = ['pid', 'strat', 'd04_childgender_', 'a06_class', 'schoolcode']
    rand = rand[[c for c in rand_keep if c in rand.columns]].copy()

    # Load follow-up survey
    fu = pd.read_stata(os.path.join(RAW_DATA_DIR, 'product_pricing_fu.dta'),
                       convert_categoricals=False)

    fu = rand.merge(fu, on='pid', how='inner')
    print(f"  Merged follow-up: {len(fu)} obs")

    # Drop ineligible
    if 's0q9_capability' in fu.columns:
        fu = fu[fu['s0q9_capability'].astype(str).str.contains('01 Yes', na=False)]
    if 's0q24_consent' in fu.columns:
        fu = fu[fu['s0q24_consent'].astype(str) != '02 No']
        module2 = fu.get('_m_fu2_dem1_c', pd.Series(0, index=fu.index))
        fu = fu[~((fu['s0q24_consent'].astype(str) == '') & (module2 == 1))]

    # Drop specific incomplete surveys
    fu = fu[~((fu['pid'] == 3190) & (fu.get('_m_fu8_pploop1_c', pd.Series(0, index=fu.index)) == 1))]
    fu = fu[~fu['pid'].isin([3279, 4721])]

    # Parent gender
    fu['dad'] = (fu['s0q4_r_sex'].astype(str).str.contains('01 Male', na=False)).astype(float)
    fu.loc[(fu['dad'] == 0) & ~fu['s0q4_r_sex'].astype(str).str.contains('02 Female', na=False), 'dad'] = np.nan
    fu['mom'] = 1 - fu['dad']

    # ---- WTP EXTRACTION ----
    # For follow-up, use pre-computed pdwtp variables with bonus adjustments

    # Data corrections (matching Stata)
    if 's4_poster_1600' in fu.columns:
        fu.loc[fu['pid'] == 5391, 's4_poster_1600'] = '01 Yes'
    if 's10_poster_pdwtp' in fu.columns:
        fu.loc[fu['pid'] == 5391, 's10_poster_pdwtp'] = 1600
    if 's4_candy_200' in fu.columns:
        fu.loc[fu['pid'] == 2163, 's4_candy_200'] = '01 Yes'
    if 's4_candy_pdwtp' in fu.columns:
        fu.loc[fu['pid'] == 2163, 's4_candy_pdwtp'] = 200
    if 's4_poster_200' in fu.columns:
        fu.loc[fu['pid'] == 5405, 's4_poster_200'] = '01 Yes'
    if 's10_poster_pdwtp' in fu.columns:
        fu.loc[fu['pid'] == 5405, 's10_poster_pdwtp'] = 200

    # Good configurations: (pdwtp_var, bonus_var, bonus_amt_var, lo_wtp_var)
    fu_goods = {
        'poster': ('s10_poster_pdwtp', 's10_poster_bonus', 's10_poster_bonusamt', 's10_wtpposter'),
        'shoes': ('s10q4_pdwtp', 's10q4_x', 's4_shoes_plus_bonus', 's10q5_wtpshoes_a'),
        'plainball': ('s4_plainball_pdwtp', 's4_plainball_bonus', 's4_plainball_bonusamt', 's4_plainball_highpri'),
        'fancyball': ('s4_fancyball_pdwtp', None, None, None),  # no bonus
        'candy': ('s4_candy_pdwtp', 's4_candy_bonus', 's4_candy_bonusamt', 's4_candy_highpri'),
        'jerry': ('s4_jerry_pdwtp', 's4_jerry_bonus', 's4_jerry_bonusamt', 's4_jerry_highpri'),
        'workbook': ('s4_wb_pdwtp', 's4_wb_bonus', 's4_wb_bonusamt', 's4_wb_highpri'),
        'test': ('s4q13_pdwtp', 's4q3_maxtest_bonus', 's4q3_maxtest_100', 's4q14_highpri'),
        'deworm': ('s10q1_pdwtp', 's10q1_maxplus_tabs', 's10_maxtabs_200', 's10q2_wtptabs'),
        'toy': ('s4q43_pdwtp', 's4_max_wtp_toy', 's4_maxtoy_500', 's4q44_highpri'),
    }

    for good, (pdwtp, bonus, bonus_amt, lo_wtp) in fu_goods.items():
        if pdwtp in fu.columns:
            wtp = pd.to_numeric(fu[pdwtp], errors='coerce').copy()
        else:
            wtp = pd.Series(np.nan, index=fu.index)

        # Bonus adjustment
        if bonus and bonus in fu.columns and good != 'fancyball':
            bonus_yes = fu[bonus].astype(str).str.contains('01 Yes', na=False)
            if bonus_amt and bonus_amt in fu.columns:
                bonus_val = pd.to_numeric(fu[bonus_amt], errors='coerce')
                wtp = np.where(bonus_yes & bonus_val.notna(), bonus_val, wtp)
            wtp = pd.Series(wtp, index=fu.index, dtype=float)

        fu[f'F{good}_wtp'] = pd.Series(wtp, index=fu.index, dtype=float)

    # Special: fancyball WTP = fancyball + plainball (Stata line 326-327)
    fu['Ffancyball_wtp'] = fu['Ffancyball_wtp'] + fu.get('Fplainball_wtp', 0)

    # Rename workbook
    fu.rename(columns={'Fworkbook_wtp': 'Fworkbook_wtp'}, inplace=True)

    # Shoes: handle two price lists (older kid >5 uses list A=2500, younger <=5 uses list B=2000)
    if 'mob_nrcage' in fu.columns:
        fu['F_older_shoes'] = (fu['mob_nrcage'] > 5) & fu['mob_nrcage'].notna()
    else:
        fu['F_older_shoes'] = True
    fu['Fshoes_mkt'] = np.where(fu['F_older_shoes'], 2500, 2000)

    # Follow-up reference child genders
    # New reference child (younger)
    if 's1bq2_sex1' in fu.columns:
        fu['Ndaughter'] = (fu['s1bq2_sex1'].astype(str) == '02 Female').astype(float)
    else:
        fu['Ndaughter'] = np.nan

    # Old reference child (from FU survey)
    if 's1bq7_sex2' in fu.columns:
        fu['Fdaughter'] = (fu['s1bq7_sex2'].astype(str) == '02 Female').astype(float)
    else:
        fu['Fdaughter'] = np.nan

    fu['hhid'] = fu['pid']

    # lovemore_resp: respondent thinks they care more about children
    if 's9q6_care' in fu.columns:
        fu['Flovemore_resp'] = (fu['s9q6_care'].astype(str).str.contains('You', na=False)).astype(float)
        fu.loc[fu['s9q6_care'].astype(str).isin(['', 'nan']), 'Flovemore_resp'] = np.nan
    else:
        fu['Flovemore_resp'] = np.nan

    print(f"  Follow-up cleaned: {len(fu)} obs")
    return fu


def build_long_dataset(bl, fu):
    """Reshape from wide to long (one row per household-good-wave)."""
    print("\nBuilding long dataset...")

    # Merge BL and FU at household level
    bl_prefix = bl.copy()
    fu_prefix = fu.copy()

    # Get common IDs
    bl_ids = set(bl_prefix['hhid'].unique())
    fu_ids = set(fu_prefix['hhid'].unique())
    fu_only = fu_ids - bl_ids  # FU-only households (should be dropped)

    print(f"  BL: {len(bl_ids)} HH, FU: {len(fu_ids)} HH, overlap: {len(bl_ids & fu_ids)}")

    # Good numbering and properties (matching Stata tempgood 1-16)
    GOOD_MAP = {
        1:  ('test',       'Btest_wtp',       6000,  True,  True,  False, True,  False, False),
        2:  ('deworm',     'Bdeworm_wtp',     4000,  True,  True,  True,  True,  False, False),
        3:  ('shoes',      'Bshoes_wtp',      None,  True,  True,  True,  True,  True,  False),
        4:  ('toy',        'Btoy_wtp',        10000, True,  False, False, True,  False, True),
        5:  ('sieve',      'Bsieve_wtp',      2000,  False, False, False, False, False, False),
        6:  ('cup',        'Bcup_wtp',        3600,  False, False, False, False, False, False),
        7:  ('Fposter',    'Fposter_wtp',     2000,  False, False, False, False, False, False),
        8:  ('Fshoes',     'Fshoes_wtp',      None,  True,  True,  True,  False, True,  False),
        9:  ('Fplainball', 'Fplainball_wtp',  1500,  True,  False, False, False, False, False),
        10: ('Ffancyball', 'Ffancyball_wtp',  2500,  True,  False, False, False, False, True),
        11: ('Fcandy',     'Fcandy_wtp',      3000,  True,  False, False, False, True,  False),
        12: ('Fjerry',     'Fjerry_wtp',      4000,  False, False, False, False, False, False),
        13: ('Fworkbook',  'Fworkbook_wtp',   4500,  True,  True,  False, False, False, False),
        14: ('Ftest',      'Ftest_wtp',       6000,  True,  True,  False, True,  False, False),
        15: ('Fdeworm',    'Fdeworm_wtp',     4000,  True,  True,  True,  True,  True,  False),
        16: ('Ftoy',       'Ftoy_wtp',        10000, True,  False, False, True,  True,  True),
    }
    # Columns: good_name, wtp_col, mkt_price, childgood, humcap, health, hypo, Nchildgood(young), toys_bin

    rows = []

    for _, hh in bl_prefix.iterrows():
        hhid = hh['hhid']
        strat = hh.get('strat', np.nan)
        bl_mom = hh['mom']
        bl_daughter = hh.get('daughter', np.nan)
        shoes_daughter = hh.get('shoes_daughter', np.nan)
        bl_comp = hh.get('comp', 8000)
        has_s10 = hh.get('has_s10_totalkids_s', False)
        older_shoes = hh.get('older_shoes', True)

        # Check if this HH is in FU sample
        fu_row = fu_prefix[fu_prefix['hhid'] == hhid]
        in_fu = len(fu_row) > 0
        if in_fu:
            fu_row = fu_row.iloc[0]

        for tg in range(1, 17):
            good_name, wtp_col_name, mkt_price, is_child, is_hc, is_health, is_hypo, is_young, is_toybin = GOOD_MAP[tg]
            is_fu = (tg >= 7)

            # Skip FU goods if not in FU sample
            if is_fu and not in_fu:
                continue

            # Get WTP
            if not is_fu:
                # Baseline good
                bl_wtp_col = good_name.replace('F', '') + '_wtp'
                wtp_raw = hh.get(bl_wtp_col, np.nan)
            else:
                # Follow-up good
                fu_wtp_col = f'F{good_name.replace("F", "")}_wtp'
                wtp_raw = fu_row.get(fu_wtp_col, np.nan)

            if pd.isna(wtp_raw):
                # Drop condition for shoes: if has_s10_totalkids_s missing and wtp missing
                if tg == 3 and not has_s10:
                    continue
                # Skip if wtp is missing (will be dropped)
                continue

            # Drop BL deworm for low-compensation HH
            if tg == 2 and bl_comp == 8000:
                continue

            # Market price for shoes
            if tg == 3:  # BL shoes
                mkt_price = 2500 if older_shoes else 2000
            elif tg == 8:  # FU shoes
                if in_fu:
                    mkt_price = fu_row.get('Fshoes_mkt', 2000)
                else:
                    mkt_price = 2000

            # Girl variable depends on which child
            if good_name in ['test', 'deworm', 'toy', 'Fworkbook', 'Ftest', 'Fplainball', 'Ffancyball']:
                # Old reference child goods
                girl_val = bl_daughter
            elif good_name == 'shoes':
                # Shoes: uses shoes reference child (younger child at BL)
                girl_val = shoes_daughter
            elif good_name in ['Fshoes', 'Fdeworm', 'Fcandy', 'Ftoy']:
                # New reference child goods
                girl_val = fu_row.get('Ndaughter', np.nan) if in_fu else np.nan
            else:
                # Adult goods: no child gender
                girl_val = np.nan

            rows.append({
                'hhid': hhid,
                'strat': strat,
                'mom': bl_mom if not is_fu else fu_row.get('mom', bl_mom),
                'good': good_name,
                'tempgood': tg,
                'wtp_raw': wtp_raw,
                'mkt_price': mkt_price,
                'fu': int(is_fu),
                'girl': girl_val,
                'childgood': int(is_child),
                'humcap': int(is_hc),
                'health': int(is_health),
                'hypo': int(is_hypo),
                'young': int(is_young),
                'toys_bin': int(is_toybin),
            })

    long = pd.DataFrame(rows)
    print(f"  Long dataset: {len(long)} obs, {long['hhid'].nunique()} households")

    # ---- NORMALIZE WTP ----

    # Cap WTP at market price, then divide by market price
    long['wtp_capped'] = np.minimum(long['wtp_raw'], long['mkt_price'])
    long['wtp'] = long['wtp_capped'] / long['mkt_price']

    # Special handling: shoes bonus can push WTP above max
    # For shoes with bonus (+100), cap at max then normalize
    for tg in [3, 8]:
        mask = long['tempgood'] == tg
        long.loc[mask, 'wtp'] = long.loc[mask, 'wtp_raw'] / long.loc[mask, 'mkt_price']
        # Cap at market price as fraction
        cap = long.loc[mask, 'mkt_price']
        long.loc[mask, 'wtp'] = np.minimum(long.loc[mask, 'wtp_raw'], cap) / cap

    # wtp_std: WTP (as fraction of market price) / SD(WTP for that good, full sample)
    long['wtp_std'] = np.nan
    for tg in long['tempgood'].unique():
        mask = long['tempgood'] == tg
        sd = long.loc[mask, 'wtp'].std()
        if sd > 0 and not pd.isna(sd):
            long.loc[mask, 'wtp_std'] = long.loc[mask, 'wtp'] / sd

    # ---- ADULT WTP CONTROL ----
    # BL: cup WTP in SD units -> spread to all goods in HH
    # FU: average of jerry + poster WTP in SD units

    cup_wtp = long.loc[long['good'] == 'cup', ['hhid', 'wtp_std']].rename(
        columns={'wtp_std': 'wtp_std_cup'}).drop_duplicates('hhid')

    jerry_wtp = long.loc[long['good'] == 'Fjerry', ['hhid', 'wtp_std']].rename(
        columns={'wtp_std': 'wtp_std_jerry'}).drop_duplicates('hhid')
    poster_wtp = long.loc[long['good'] == 'Fposter', ['hhid', 'wtp_std']].rename(
        columns={'wtp_std': 'wtp_std_poster'}).drop_duplicates('hhid')

    long = long.merge(cup_wtp, on='hhid', how='left')
    long = long.merge(jerry_wtp, on='hhid', how='left')
    long = long.merge(poster_wtp, on='hhid', how='left')

    long['adult_wtp_std'] = np.where(
        long['fu'] == 0,
        long['wtp_std_cup'],
        (long['wtp_std_jerry'].fillna(0) + long['wtp_std_poster'].fillna(0)) / 2)
    # Handle missing FU adult WTP
    long.loc[(long['fu'] == 1) & long['wtp_std_jerry'].isna() & long['wtp_std_poster'].isna(),
             'adult_wtp_std'] = np.nan

    long['adult_wtp_stdXfu'] = long['adult_wtp_std'] * long['fu']

    long.drop(columns=['wtp_std_cup', 'wtp_std_jerry', 'wtp_std_poster'], errors='ignore', inplace=True)

    # ---- INTERACTION TERMS ----
    long['momXgirl'] = long['mom'] * long['girl']

    # ---- LOVEMORE_MOM_HHMED ----
    # Construct HH-level indicator: both parents agree mom loves children more
    # Build HH-level from BL and FU lovemore_resp
    hh_lovemore = bl_prefix[['hhid', 'mom', 'Blovemore_resp']].copy()
    fu_lovemore = fu_prefix[['hhid', 'mom', 'Flovemore_resp']].copy()
    hh_lovemore = hh_lovemore.merge(fu_lovemore[['hhid', 'Flovemore_resp']].drop_duplicates('hhid'),
                                     on='hhid', how='left')

    # lovemore_mom: 1 if both parents agree mom loves more
    # = 1 if (mom==1 & lovemore_resp==1) or (mom==0 & lovemore_resp==0)
    # Use BL lovemore_resp for this
    hh_lovemore['lovemore_mom'] = np.where(
        hh_lovemore['Blovemore_resp'].isna(), np.nan,
        np.where(
            (hh_lovemore['mom'] == 1) & (hh_lovemore['Blovemore_resp'] == 1), 1.0,
            np.where(
                (hh_lovemore['mom'] == 0) & (hh_lovemore['Blovemore_resp'] == 0), 1.0,
                0.0)))

    # HH-level: requires both BL and FU non-missing
    hh_lovemore['both_waves'] = hh_lovemore['Blovemore_resp'].notna() & hh_lovemore['Flovemore_resp'].notna()
    hh_lovemore.loc[~hh_lovemore['both_waves'], 'lovemore_mom'] = np.nan

    # Average at HH level across parents
    hh_agg = hh_lovemore.groupby('hhid')['lovemore_mom'].mean().reset_index()
    hh_agg.rename(columns={'lovemore_mom': 'lovemore_mom_HH'}, inplace=True)

    # lovemore_mom_HHmed = 1 if HH average == 1 (both parents agree)
    hh_agg['lovemore_mom_HHmed'] = np.where(
        hh_agg['lovemore_mom_HH'].isna(), np.nan,
        (hh_agg['lovemore_mom_HH'] == 1).astype(float))

    long = long.merge(hh_agg[['hhid', 'lovemore_mom_HHmed']], on='hhid', how='left')

    # ---- GOOD FIXED EFFECTS ----
    good_dummies = pd.get_dummies(long['good'], prefix='_G', dtype=float)
    good_dummy_cols = sorted(good_dummies.columns.tolist())
    if len(good_dummy_cols) > 0:
        good_dummies = good_dummies.drop(columns=[good_dummy_cols[0]])
    long = pd.concat([long.reset_index(drop=True), good_dummies.reset_index(drop=True)], axis=1)

    # Drop sieve (matching paper_statistics.do: sieve dropped from analysis)
    long = long[long['good'] != 'sieve'].copy()

    print(f"  Final long dataset: {len(long)} obs")

    return long


def main():
    print("=" * 60)
    print("01_clean.py: Data Cleaning")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    bl = clean_baseline()
    fu = clean_followup()
    long = build_long_dataset(bl, fu)

    # Save
    long.to_parquet(ANALYSIS_DATA_PATH, index=False)
    print(f"\nSaved: {ANALYSIS_DATA_PATH}")
    print(f"Shape: {long.shape}")
    print(f"Households: {long['hhid'].nunique()}")
    print(f"Goods: {sorted(long['good'].unique())}")

    # Summary
    print(f"\n--- Variable Coverage ---")
    for var in ['wtp_std', 'girl', 'mom', 'momXgirl', 'adult_wtp_std', 'fu', 'young']:
        n = long[var].notna().sum()
        print(f"  {var:20s}  N={n:6d}  ({100*n/len(long):.1f}%)")

    # Key sample: childgood & toys_bin==0
    main_sample = long[(long['childgood'] == 1) & (long['toys_bin'] == 0)]
    main_with_vars = main_sample.dropna(subset=['wtp_std', 'girl', 'mom', 'momXgirl'])
    print(f"\nMain analysis sample (childgood, no toys_bin): {len(main_sample)} obs")
    print(f"  With all key vars: {len(main_with_vars)} obs")
    print(f"  Published N: 6,673")

    # Father-son mean of wtp_dadboydm (for Table 1 comparison)
    for g in sorted(long['good'].unique()):
        mask = long['good'] == g
        n_obs = mask.sum()
        wtp_mean = long.loc[mask, 'wtp'].mean()
        wtp_sd = long.loc[mask, 'wtp'].std()
        print(f"  {g:12s}  N={n_obs:5d}  mean_wtp={wtp_mean:.4f}  sd={wtp_sd:.4f}")

    return long


if __name__ == '__main__':
    main()
