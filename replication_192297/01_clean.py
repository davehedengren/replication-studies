"""
01_clean.py — Translates 01_Cleaning.do for Bryan, Karlan & Osman (2024).

Produces:
  - 16-ABA-All-Merged.parquet (main analysis dataset)
  - 17-ABA-penalty-clean.parquet (penalty data for Tables 3-4)

Then validates against the Stata-produced .dta files.
"""

import sys, os, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
sys.path.insert(0, os.path.dirname(__file__))
from utils import RAW_DATA_DIR, CLEAN_DATA_DIR, OUTPUT_DIR, winsor

np.random.seed(1129)


# ================================================================
# (1) BASELINE CLEANING
# ================================================================
def clean_baseline():
    """Translate baseline cleaning from 01_Cleaning.do lines 18-243."""
    print("\n--- (1) Baseline Cleaning ---")
    bl = pd.read_stata(os.path.join(RAW_DATA_DIR, '10-ABA-baseline.dta'),
                       convert_categoricals=False)
    print(f"  Raw baseline: {len(bl)} obs, {bl.shape[1]} vars")

    # Age
    if 'b_date_visit' in bl.columns:
        bl['b_year'] = pd.to_datetime(bl['b_date_visit']).dt.year
    else:
        bl['b_year'] = 2016
    bl['b_age'] = bl['b_year'] - bl['b_birth_year']

    # Education dummies (1-16 mapping)
    edu_names = ['b_edu_noschooling', 'b_edu_litcertif', 'b_edu_enrolelem', 'b_edu_gradelem',
                 'b_edu_enrolprep', 'b_edu_gradprep', 'b_edu_enrolsec', 'b_edu_gradsec',
                 'b_edu_enrolvoc', 'b_edu_gradvoc', 'b_edu_enrolpostsec', 'b_edu_gradpostsec',
                 'b_edu_enrolcolg', 'b_edu_gradcolg', 'b_edu_enrolpostgrad', 'b_edu_gradpostgrad']
    for k, name in enumerate(edu_names, start=1):
        bl[name] = np.where(bl['b_education'] == k, k, 0).astype(float)

    # Premises dummies (1-6 + other=8)
    prem_names = ['b_premises_parthome', 'b_premises_office', 'b_premises_shop',
                  'b_premises_warehouse', 'b_premises_workshop', 'b_premises_agriland']
    for k, name in enumerate(prem_names, start=1):
        bl[name] = np.where(bl['b_premises_type'] == k, k, 0).astype(float)
    bl['b_premises_other'] = (bl['b_premises_type'] == 8).astype(float)

    # Premises ownership dummies
    own_names = ['b_premisesown_own', 'b_premisesown_rentold', 'b_premisesown_rentnew', 'b_premisesown_other']
    for k, name in enumerate(own_names, start=1):
        if name not in bl.columns:
            bl[name] = np.where(bl.get('b_premises_ownership', pd.Series(np.nan)) == k, k, 0).astype(float)

    # Experience
    bl['b_experience'] = bl['b_year'] - bl.get('b_start_year', pd.Series(np.nan, index=bl.index))

    # Work hours
    bl['b_workhours_week'] = bl.get('b_hours', 0) * bl.get('b_days', 0)
    bl.loc[bl['b_workhours_week'] == 168, 'b_workhours_week'] = np.nan

    # Branches
    bl['b_hasbranch'] = (bl.get('b_branches', 0) >= 1).astype(float)
    bl.loc[bl.get('b_branches', pd.Series(0)) == 0, 'b_hasbranch'] = 0

    # Staff: set to missing on refusal, zeros for sub-categories if no staff
    bl.loc[bl.get('b_staff_yesno', pd.Series()) == 2, 'b_staff_yesno'] = np.nan
    for v in ['b_full_yesno', 'b_temp_yesno', 'b_regular_parttime', 'b_unpaid_yesno', 'b_intern_yesno']:
        if v in bl.columns:
            bl.loc[bl['b_staff_yesno'] == 0, v] = 0

    for v in ['b_num_employ_full', 'b_num_employ_full_family', 'b_employeesalary_full', 'b_employeebonus_full']:
        if v in bl.columns and 'b_full_yesno' in bl.columns:
            bl.loc[bl['b_full_yesno'] == 0, v] = 0

    if 'b_regular_parttime' in bl.columns and 'b_num_employ_part' in bl.columns:
        bl.loc[bl['b_regular_parttime'] == 0, 'b_num_employ_part'] = 0

    for prefix, flag in [('temp', 'b_temp_yesno'), ('intern', 'b_intern_yesno'), ('unpaid', 'b_unpaid_yesno')]:
        for suffix in ['', '_family']:
            num_col = f'b_num_employ_{prefix}{suffix}'
            if num_col in bl.columns and flag in bl.columns:
                bl.loc[bl[flag] == 0, num_col] = 0
        for suffix in ['salary', 'bonus']:
            sal_col = f'b_employee{suffix}_{prefix}'
            emp_col = f'b_empoyee{suffix}_{prefix}'  # note Stata typo
            for c in [sal_col, emp_col]:
                if c in bl.columns and flag in bl.columns:
                    bl.loc[bl[flag] == 0, c] = 0

    # Total employees
    emp_cols = ['b_num_employ_full', 'b_num_employ_part', 'b_num_employ_temp',
                'b_num_employ_intern', 'b_num_employ_unpaid']
    existing_emp = [c for c in emp_cols if c in bl.columns]
    bl['b_nemployees_main'] = bl[existing_emp].sum(axis=1)
    bl.loc[bl['b_staff_yesno'].isna(), 'b_nemployees_main'] = np.nan

    # Wage bill
    for c in ['b_employeesalary_full', 'b_employeebonus_full']:
        if c not in bl.columns:
            bl[c] = 0
    bl['b_wagebill_full'] = bl['b_employeesalary_full'].fillna(0) + bl['b_employeebonus_full'].fillna(0)
    bl['b_wagebill_part'] = 0.5 * bl['b_wagebill_full']

    for prefix in ['temp', 'intern', 'unpaid']:
        sal = f'b_employeesalary_{prefix}'
        bon = f'b_empoyeebonus_{prefix}'
        if sal not in bl.columns: bl[sal] = 0
        if bon not in bl.columns: bl[bon] = 0
        bl[f'b_wagebill_{prefix}'] = bl[sal].fillna(0) + bl[bon].fillna(0)
    if 'b_wagebill_temp' in bl.columns:
        bl.loc[bl['b_wagebill_temp'] == -9998, 'b_wagebill_temp'] = np.nan

    # Replace special codes (996=don't know, 998/999=refused) with 0 (Stata line 100-102)
    for wb in ['b_wagebill_full', 'b_wagebill_part', 'b_wagebill_temp', 'b_wagebill_intern', 'b_wagebill_unpaid']:
        if wb in bl.columns:
            bl.loc[bl[wb].isin([996, 998, 999]), wb] = 0

    wb_cols = [c for c in bl.columns if c.startswith('b_wagebill_') and c != 'b_wagebill_main']
    bl['b_wagebill_main'] = bl[wb_cols].fillna(0).sum(axis=1)

    # Digit span
    r_cols = [c for c in bl.columns if c.startswith('b_r') and c.endswith('_answer')]
    if r_cols:
        bl['digitspan'] = bl[r_cols].fillna(0).sum(axis=1)
    else:
        bl['digitspan'] = np.nan

    # Other credit
    loan_cols = [c for c in bl.columns if any(c.startswith(p) for p in
                 ['b_loan_size_bank', 'b_loan_size_mfi', 'b_loan_size_informal', 'b_loan_size_familyfriends'])]
    bl['othercredit2'] = bl[loan_cols].fillna(0).sum(axis=1) if loan_cols else 0
    bl.loc[bl['othercredit2'] == 1015000, 'othercredit2'] = 15000
    bl['b_othercredit'] = winsor(bl['othercredit2'], p=0.01)

    # Raven test
    raven_answers = {1: 6, 2: 5, 3: 3, 4: 7, 5: 7, 6: 3, 7: 7, 8: 3, 9: 1, 10: 5}
    for i, correct in raven_answers.items():
        col = f'b_raven_test_{i}'
        bl[f'raven{i}_correct'] = (bl.get(col, pd.Series()) == correct).astype(float) if col in bl.columns else 0
    raven_cols = [f'raven{i}_correct' for i in range(1, 11)]
    bl['raven_score'] = bl[raven_cols].sum(axis=1)

    # Financial literacy
    fl_answers = {1: 2, 2: 1, 3: 2, 4: 1, 5: 3, 6: 2, 7: 4, 8: 2, 9: 2, 10: 4,
                  11: 2, 12: 4, 13: 2, 14: 1, 15: 2, 16: 4}
    for i, correct in fl_answers.items():
        col = f'b_fl_{i}'
        bl[f'fl{i}_correct'] = (bl.get(col, pd.Series()) == correct).astype(float) if col in bl.columns else 0
    fl_cols = [f'fl{i}_correct' for i in range(1, 17)]
    bl['fl_score'] = bl[fl_cols].sum(axis=1)

    # Sales dummies
    bl['b_sales_better'] = (bl.get('b_sales', pd.Series()) == 1).astype(float)
    bl['b_sales_average'] = (bl.get('b_sales', pd.Series()) == 2).astype(float)
    bl['b_sales_worse'] = (bl.get('b_sales', pd.Series()) == 3).astype(float)

    # Seasonal sales
    bl['b_seasonal_sales'] = (bl.get('b_nature_sales', pd.Series()) == 2).astype(float)
    bl.loc[bl.get('b_nature_sales', pd.Series()).isin([3, np.nan]), 'b_seasonal_sales'] = np.nan

    # Set survey coding issues to missing
    for v in ['b_lastmonth_sales', 'b_beflastmonth_sales', 'b_highest_profits', 'b_lowest_profits']:
        if v in bl.columns:
            bl.loc[bl[v].isin([-997, 9998, 9999, 996]), v] = np.nan

    # Bookkeeping dummies
    bl['b_bookeep_manual'] = (bl.get('b_fin_computerize', pd.Series()) == 1).astype(float)
    bl['b_bookeep_computer'] = (bl.get('b_fin_computerize', pd.Series()) == 2).astype(float)
    bl['b_bookeep_both'] = (bl.get('b_fin_computerize', pd.Series()) == 3).astype(float)
    bl.loc[bl.get('b_fin_computerize', pd.Series()) == 3, 'b_bookeep_manual'] = 1
    bl.loc[bl.get('b_fin_computerize', pd.Series()) == 3, 'b_bookeep_computer'] = 1

    # Merge treatment
    treat = pd.read_stata(os.path.join(RAW_DATA_DIR, '00-ABA-treatment.dta'),
                          convert_categoricals=False)
    bl = bl.merge(treat, on='client_key', how='left', suffixes=('', '_treat'))

    print(f"  Baseline cleaned: {len(bl)} obs")
    return bl


# ================================================================
# (2) MIDLINE/ENDLINE CLEANING
# ================================================================
def clean_midline_endline(bl):
    """Translate midline/endline cleaning from 01_Cleaning.do lines 245-443."""
    print("\n--- (2) Midline/Endline Cleaning ---")

    # Load endline
    el = pd.read_stata(os.path.join(RAW_DATA_DIR, '12-ABA-endline.dta'),
                       convert_categoricals=False)
    el['round'] = 2

    # Load midline
    ml = pd.read_stata(os.path.join(RAW_DATA_DIR, '11-ABA-midline.dta'),
                       convert_categoricals=False)
    ml['round'] = 1

    # Append (force=True equivalent)
    me = pd.concat([el, ml], ignore_index=True, sort=False)
    me.loc[me['round'].isna(), 'round'] = 1

    # Employee harmonization
    for x in range(1, 6):
        emp_subcats = [f'e_num_employ_full_{x}', f'e_num_employ_part_{x}',
                       f'e_num_employ_temp_{x}', f'e_num_employ_intern_{x}',
                       f'e_num_employ_unpaid_{x}']
        existing = [c for c in emp_subcats if c in me.columns]
        if existing:
            me[f'm_nemployees_{x}'] = me.loc[me['round'] == 1, existing].fillna(0).sum(axis=1)
        else:
            me[f'm_nemployees_{x}'] = np.nan

    for x in range(1, 5):
        ne_col = f'e_nemployees_{x}'
        mn_col = f'm_nemployees_{x}'
        np_col = 'e_number_projects'
        if ne_col not in me.columns:
            me[ne_col] = np.nan
        # Replace 0 if no business x in round 2
        me.loc[(me[ne_col].isna()) & (me.get(np_col, 99) < x) & (me['round'] == 2), ne_col] = 0
        # Use midline calculation for round 1
        if mn_col in me.columns:
            me.loc[me['round'] == 1, ne_col] = me.loc[me['round'] == 1, mn_col]
        # Set coding errors to missing
        me.loc[me[ne_col].isin([-998, -997, -98, -97, -94, 999, 997, 998]), ne_col] = np.nan

    # Total employees across all businesses
    ne_cols = [f'e_nemployees_{x}' for x in range(1, 6) if f'e_nemployees_{x}' in me.columns]
    me['e_nemployees_all'] = me[ne_cols].fillna(0).sum(axis=1)

    # Has business
    me['e_hasbusiness'] = np.where(me.get('e_number_projects', 0) >= 1, 1, 0).astype(float)
    me.loc[me.get('e_number_projects', pd.Series()).isna(), 'e_hasbusiness'] = np.nan

    # Wage bill construction (simplified - sum across businesses)
    wb_cols = [c for c in me.columns if c.startswith('e_wagebill_') and c[11:].isdigit()]
    if not wb_cols:
        me['e_wagebill_all'] = np.nan
    else:
        me['e_wagebill_all'] = me[wb_cols].fillna(0).sum(axis=1)

    # Clean financial variables: set refusals/DK to missing, 0 if no business
    for x in range(1, 5):
        for v in [f'e_lastmonth_expenses_{x}', f'e_lastmonth_profits_{x}',
                  f'e_beflastmonth_expenses_{x}', f'e_beflastmonth_profits_{x}',
                  f'e_lastmonth_amount_{x}']:
            if v in me.columns:
                me.loc[me[v].isin([-998, -997, 998, 997, 999]) & (me['round'] == 2), v] = np.nan
                me.loc[me.get('e_number_projects', 99) < x, v] = 0

    for x in range(1, 6):
        for v in [f'e_amount_{x}', f'e_lastmonth_sales_{x}', f'e_beflastmonth_sales_{x}',
                  f'e_monthly_profits_{x}', f'e_highest_profits_{x}', f'e_lowest_profits_{x}',
                  f'e_monthly_revenue_{x}']:
            if v in me.columns:
                me.loc[me[v].isin([-998, -997, 998, 997, 999, 666, -988]), v] = np.nan
                me.loc[me.get('e_number_projects', 99) < x, v] = 0

    # Harmonize profits, revenues, expenses across rounds
    for x in range(1, 5):
        lp = f'e_lastmonth_profits_{x}'
        bp = f'e_beflastmonth_profits_{x}'
        le = f'e_lastmonth_expenses_{x}'
        be = f'e_beflastmonth_expenses_{x}'
        ls = f'e_lastmonth_sales_{x}'
        bs = f'e_beflastmonth_sales_{x}'
        mp = f'e_monthly_profits_{x}'
        mr = f'e_monthly_revenue_{x}'
        mx = f'e_monthly_expenses_{x}'

        # Round 2: average of last month and before last month
        for new, a, b in [(f'e_month_profit_{x}', lp, bp),
                          (f'e_month_exp_{x}', le, be),
                          (f'e_month_rev_{x}', ls, bs)]:
            cols = [c for c in [a, b] if c in me.columns]
            if cols:
                me[new] = me.loc[me['round'] == 2, cols].mean(axis=1)
            else:
                me[new] = np.nan

        # Round 1: use monthly reported values
        if mp in me.columns:
            me.loc[me['round'] == 1, f'e_month_profit_{x}'] = me.loc[me['round'] == 1, mp]
        if mr in me.columns:
            me.loc[me['round'] == 1, f'e_month_rev_{x}'] = me.loc[me['round'] == 1, mr]
        if mx in me.columns:
            me.loc[me['round'] == 1, f'e_month_exp_{x}'] = me.loc[me['round'] == 1, mx]

    for x in [5]:
        me[f'e_month_profit_{x}'] = np.nan
        me[f'e_month_rev_{x}'] = np.nan
        me[f'e_month_exp_{x}'] = np.nan
        mp = f'e_monthly_profits_{x}'
        mr = f'e_monthly_revenue_{x}'
        mx = f'e_monthly_expenses_{x}'
        if mp in me.columns:
            me.loc[me['round'] == 1, f'e_month_profit_{x}'] = me.loc[me['round'] == 1, mp]
        if mr in me.columns:
            me.loc[me['round'] == 1, f'e_month_rev_{x}'] = me.loc[me['round'] == 1, mr]
        if mx in me.columns:
            me.loc[me['round'] == 1, f'e_month_exp_{x}'] = me.loc[me['round'] == 1, mx]

    # Create _main variables (Stata line 353: gen e_month_profit_main=e_month_profit_1)
    if 'e_month_profit_1' in me.columns:
        me['e_month_profit_main'] = me['e_month_profit_1']
    if 'e_month_rev_1' in me.columns:
        me['e_month_rev_main'] = me['e_month_rev_1']
    if 'e_month_exp_1' in me.columns:
        me['e_month_exp_main'] = me['e_month_exp_1']

    # Aggregate across businesses (Stata: rowtotal(e_month_profit_*))
    # NOTE: Stata's wildcard matches _main too, so _1 is double-counted.
    # This appears to be a bug in the original code, but we replicate it exactly.
    prof_cols = [c for c in me.columns if c.startswith('e_month_profit_') and c != 'e_month_profit_all']
    me['e_month_profit_all'] = me[prof_cols].fillna(0).sum(axis=1)
    me.loc[me.get('e_month_profit_1', pd.Series()).isna(), 'e_month_profit_all'] = np.nan

    # Revenue: Stata uses rowtotal(e_month_rev_?) — single-char wildcard, matches 1-5 only
    rev_cols = [f'e_month_rev_{x}' for x in range(1, 6) if f'e_month_rev_{x}' in me.columns]
    me['e_month_rev_all'] = me[rev_cols].fillna(0).sum(axis=1)
    me.loc[me.get('e_month_rev_1', pd.Series()).isna(), 'e_month_rev_all'] = np.nan

    # Expense: Stata uses rowtotal(e_month_exp_?) — single-char wildcard, matches 1-5 only
    exp_cols = [f'e_month_exp_{x}' for x in range(1, 6) if f'e_month_exp_{x}' in me.columns]
    me['e_month_exp_all'] = me[exp_cols].fillna(0).sum(axis=1)
    me.loc[me.get('e_month_exp_1', pd.Series()).isna(), 'e_month_exp_all'] = np.nan

    # Household expenditure
    weekly_items = ['e_bread_expenses', 'e_vegetables_expenses', 'e_fruits_expenses',
                    'e_chicken_expenses', 'e_dairy_expenses', 'e_tea_expenses', 'e_inkind_lastweek_expenses']
    monthly_items = ['e_water_expenses', 'e_medical_expenses', 'e_personal_care_expenses',
                     'e_transportation_expenses', 'e_communication_expenses', 'e_entertainment_expenses',
                     'e_education_expenses', 'e_pocketmoney_expenses', 'e_rent_expenses',
                     'e_inkind_monthly_expenses', 'e_other_services_expenses']
    yearly_items = ['e_clothing_expenses', 'e_phone_expenses', 'e_education_expenses_yearly',
                    'e_other_large_expenses', 'e_inkind_various_expenses']

    # Set coding errors to missing
    for v in weekly_items + monthly_items + yearly_items:
        if v in me.columns:
            me.loc[me[v].isin([988, 998, -997, -998, 997]), v] = np.nan

    w_cols = [c for c in weekly_items if c in me.columns]
    m_cols = [c for c in monthly_items if c in me.columns]
    y_cols = [c for c in yearly_items if c in me.columns]

    me['e_weekly_cons'] = me[w_cols].fillna(0).sum(axis=1) if w_cols else 0
    me['e_monthly_cons'] = me[m_cols].fillna(0).sum(axis=1) if m_cols else 0
    me['e_yearly_cons'] = me[y_cols].fillna(0).sum(axis=1) if y_cols else 0

    # Short survey replacements
    for long_v, short_v in [('e_weekly_cons', 'e_hh_weekexpenses_ss'),
                            ('e_monthly_cons', 'e_hh_monthexpenses_ss'),
                            ('e_yearly_cons', 'e_hh_yearexpenses_ss')]:
        if short_v in me.columns:
            me.loc[me[short_v].notna(), long_v] = me.loc[me[short_v].notna(), short_v]

    me['e_hh_exp'] = me['e_weekly_cons'] * 4 + me['e_monthly_cons'] + me['e_yearly_cons'] / 12

    # Asset values
    for x in range(1, 5):
        asset_types = ['equip', 'vehicule', 'land', 'realestate', 'livestock', 'other']
        for at in asset_types:
            val_col = f'e_assetval_{at}_{x}'
            has_col = f'e_assets_{at}_{x}'
            if val_col in me.columns and has_col in me.columns:
                me[has_col] = pd.to_numeric(me[has_col], errors='coerce')
                me[val_col] = pd.to_numeric(me[val_col], errors='coerce')
                me.loc[(me[has_col] == 0) & (me['round'] == 2), val_col] = 0
                me.loc[(me.get('e_number_projects', 99) < x) & (me['round'] == 2), val_col] = 0

        av_cols = [f'e_assetval_{at}_{x}' for at in asset_types if f'e_assetval_{at}_{x}' in me.columns]
        me[f'e_totvalassets_{x}'] = me[av_cols].fillna(0).sum(axis=1) if av_cols else 0

    ta_cols = [f'e_totvalassets_{x}' for x in range(1, 5) if f'e_totvalassets_{x}' in me.columns]
    me['e_totvalassets_all'] = me[ta_cols].fillna(0).sum(axis=1) if ta_cols else 0

    # Psychometric: set 6 to missing
    psy_cols = [c for c in me.columns if c.startswith('e_psy')]
    for c in psy_cols:
        me.loc[me[c] == 6, c] = np.nan

    # Survey took flag
    me['e_tooksurvey'] = 1  # all rows in survey took it

    print(f"  Midline/endline cleaned: {len(me)} obs")
    return me


# ================================================================
# (3) ADMIN DATA CLEANING
# ================================================================
def clean_admin():
    """Translate admin cleaning from 01_Cleaning.do lines 445-617."""
    print("\n--- (3) Admin Cleaning ---")
    admin = pd.read_stata(os.path.join(RAW_DATA_DIR, '13-ABA-admin.dta'),
                          convert_categoricals=False)
    print(f"  Raw admin: {len(admin)} obs")

    # Reshape wide: one row per client with all loans
    admin = admin.sort_values(['client_key', 'b_date_visit'], na_position='last')

    # Pivot from long (one row per loan) to wide (one row per client)
    # First, figure out loannum per client
    admin['loannum'] = admin.groupby('client_key').cumcount() + 1

    # Extract key loan-level variables
    loan_vars = ['loan_id', 'disbdate', 'repaydate', 'loansize', 'ninstal', 'rate',
                 'lastpayment', 'totallate', 'totalpenalty', 'default']

    # Pivot to wide
    wide_parts = []
    for var in loan_vars:
        if var in admin.columns:
            piv = admin.pivot_table(index='client_key', columns='loannum', values=var, aggfunc='first')
            piv.columns = [f'{var}_{int(c)}' for c in piv.columns]
            wide_parts.append(piv)

    # Keep non-loan-level variables (one per client)
    client_vars = ['client_key', 'random_date', 'edm', 'treatment', 'loid_baseline',
                   'branch_id', 'client_id', 'random_cohort', 'nloans']
    client_vars = [c for c in client_vars if c in admin.columns]
    client_df = admin.drop_duplicates('client_key')[client_vars].set_index('client_key')

    wide = pd.concat([client_df] + wide_parts, axis=1).reset_index()
    print(f"  Wide admin: {len(wide)} obs")

    # Ensure date columns are datetime
    wide['random_date'] = pd.to_datetime(wide['random_date'], errors='coerce')
    for x in range(1, 8):
        for dcol in [f'disbdate_{x}', f'repaydate_{x}']:
            if dcol in wide.columns:
                wide[dcol] = pd.to_datetime(wide[dcol], errors='coerce')

    # Identify experiment loan (first loan after randomization, within 270 days)
    wide['exploan'] = 0
    for x in range(1, 8):
        disb_col = f'disbdate_{x}'
        next_col = f'disbdate_{x+1}' if x < 7 else None
        if disb_col not in wide.columns:
            continue
        mask = (wide[disb_col] >= wide['random_date']) & wide[disb_col].notna()
        if next_col and next_col in wide.columns:
            mask = mask & ((wide[next_col] < wide['random_date']) | wide[next_col].isna())
        elif x == 1:
            pass  # first loan after randomization
        wide.loc[mask & (wide['exploan'] == 0), 'exploan'] = x

    # exploan=0 if most recent loan is before randomization
    if 'disbdate_1' in wide.columns:
        wide.loc[wide['random_date'] > wide['disbdate_1'], 'exploan'] = 0

    # Drop if experiment loan > 270 days after randomization
    wide['exploan_disbdate'] = pd.NaT
    for x in range(1, 8):
        disb_col = f'disbdate_{x}'
        if disb_col in wide.columns:
            wide.loc[wide['exploan'] == x, 'exploan_disbdate'] = wide.loc[wide['exploan'] == x, disb_col]

    wide['exp_loan_diff'] = (pd.to_datetime(wide['exploan_disbdate']) - pd.to_datetime(wide['random_date'])).dt.days
    wide.loc[wide['exp_loan_diff'] > 270, 'exploan'] = 0
    wide.loc[wide['exploan'] == 0, 'exploan_disbdate'] = np.nan

    # Extract experiment loan variables
    for var in ['loansize', 'repaydate', 'ninstal', 'rate', 'totallate', 'totalpenalty', 'default', 'lastpayment']:
        wide[f'exploan_{var}'] = np.nan
        for x in range(1, 8):
            col = f'{var}_{x}'
            if col in wide.columns:
                wide.loc[wide['exploan'] == x, f'exploan_{var}'] = wide.loc[wide['exploan'] == x, col]
        wide.loc[wide['exploan'] == 0, f'exploan_{var}'] = 0

    wide.rename(columns={'exploan_loansize': 'exploan_size'}, inplace=True)

    # Takeup
    wide['took_loan'] = (wide['exploan'] != 0).astype(float)

    # Perfect repayment
    wide['perfect_repayment'] = (wide['exploan_rate'] == 1).astype(float)
    wide.loc[wide['exploan'] == 0, 'perfect_repayment'] = 1
    wide.loc[wide['exploan'] == 0, 'exploan_totallate'] = 0
    wide.loc[wide['exploan'] == 0, 'exploan_totalpenalty'] = 0

    # Loan term
    wide['exploan_repaydate'] = pd.to_datetime(wide['exploan_repaydate'], errors='coerce')
    wide['exploan_termd'] = (wide['exploan_repaydate'] - pd.to_datetime(wide['exploan_disbdate'])).dt.days
    wide['exploan_termm'] = np.floor(pd.to_numeric(wide['exploan_termd'], errors='coerce') / 30)
    wide.loc[wide['exploan'] == 0, 'exploan_termm'] = 0

    # Post-1 loan within 3 months
    # (simplified: whether took another loan within 3 months of experiment loan ending)
    wide['ad_post1_3months'] = 0  # placeholder

    # Add prefix
    rename_dict = {}
    for c in wide.columns:
        if c not in ['client_key', 'branch_id', 'client_id', 'edm', 'random_date', 'random_cohort']:
            rename_dict[c] = f'ad_{c}'
    wide = wide.rename(columns=rename_dict)

    # Restore key names (these shouldn't have ad_ prefix)
    for old, new in [('ad_treatment', 'treatment'), ('ad_loid_baseline', 'loid_baseline'),
                     ('ad_took_loan', 'took_loan'), ('ad_perfect_repayment', 'perfect_repayment'),
                     ('ad_nloans', 'nloans')]:
        if old in wide.columns:
            wide.rename(columns={old: new}, inplace=True)

    print(f"  Admin cleaned: {len(wide)} obs")
    return wide


# ================================================================
# (4) LOAN OFFICER CLEANING (simplified)
# ================================================================
def clean_loan_officer():
    """Translate loan officer cleaning from 01_Cleaning.do lines 619-694."""
    print("\n--- (4) Loan Officer Cleaning ---")
    lo = pd.read_stata(os.path.join(RAW_DATA_DIR, '14-ABA-loan-officer.dta'),
                       convert_categoricals=False)
    print(f"  Raw loan officer: {len(lo)} obs, {lo.shape[1]} vars")

    # The LO data needs reshaping from wide (one row per LO) to long (one row per client)
    # Identify client columns
    client_cols = [c for c in lo.columns if c.startswith('client') and c != 'client_key']

    # Reshape: each LO has client_1_pl through client_N_pl
    rows = []
    for _, row in lo.iterrows():
        loid = row.get('loid_baseline', np.nan)
        for c in client_cols:
            if pd.notna(row[c]) and row[c] != 0:
                rows.append({'loid_baseline': loid, 'client_key': int(row[c])})

    lo_long = pd.DataFrame(rows)

    # Add prefix
    lo_long = lo_long.drop_duplicates('client_key')
    print(f"  Loan officer cleaned: {len(lo_long)} obs")
    return lo_long


# ================================================================
# (5) OUTSTANDING DEBT AT MIDLINE/ENDLINE
# ================================================================
def compute_outstanding(admin_raw):
    """Compute outstanding20 and outstanding30 from admin data."""
    print("\n--- (5) Outstanding Debt ---")

    # Reshape to installment level
    admin = admin_raw.copy()

    # Find installment-level columns
    mature_cols = [c for c in admin.columns if c.startswith('mature') and c[6:].isdigit()]
    max_inst = max(int(c[6:]) for c in mature_cols) if mature_cols else 0

    rows = []
    for _, loan in admin.iterrows():
        for i in range(1, max_inst + 1):
            mat_col = f'mature{i}'
            pay_col = f'payment{i}'
            amt_col = f'amount{i}'
            if mat_col in admin.columns and pd.notna(loan.get(mat_col)):
                payment = loan.get(pay_col)
                if pd.isna(payment):
                    continue  # not yet due
                rows.append({
                    'client_key': loan['client_key'],
                    'loan_id': loan.get('loan_id', np.nan),
                    'installment': i,
                    'mature': loan[mat_col],
                    'payment': payment,
                    'amount': loan.get(amt_col, 0),
                    'loansize': loan.get('loansize', 0),
                    'ninstal': loan.get('ninstal', 0),
                    'duration': loan.get('duration', 0),
                    'lastpayment': loan.get('lastpayment', np.nan),
                    'treatment': loan.get('treatment', np.nan),
                    'edm': loan.get('edm', np.nan),
                    'loid_baseline': loan.get('loid_baseline', np.nan),
                })

    if not rows:
        print("  WARNING: No installment data found")
        return pd.DataFrame(columns=['client_key', 'outstanding20', 'outstanding30'])

    inst = pd.DataFrame(rows)

    # Drop problem loans
    bad_loans = [63568220150514, 3676720160727, 32773920141012, 611443320141209]
    inst = inst[~inst['loan_id'].isin(bad_loans)]

    # Drop if payment before May 2015
    inst['payment'] = pd.to_datetime(inst['payment'], errors='coerce')
    inst = inst[inst['payment'] >= pd.Timestamp('2015-05-01')]

    # Total due
    inst['amountpaid'] = inst.groupby('loan_id')['amount'].cumsum()

    # Calculate total due per loan
    inst['mduration'] = np.floor(inst['duration'] / 30)
    inst.loc[inst['mduration'] < 0, 'mduration'] = inst.loc[inst['mduration'] < 0, 'ninstal']
    inst['interest'] = (1.42 * inst['mduration']) / 100
    inst['totdue'] = inst['loansize'] * (1 + inst['interest'])

    # Monthly
    inst['payment_m'] = inst['payment'].dt.to_period('M')

    # Keep last installment per loan-month
    inst = inst.sort_values(['loan_id', 'payment_m', 'installment'])
    inst = inst.drop_duplicates(subset=['loan_id', 'payment_m'], keep='last')

    # Outstanding = total due - cumulative paid
    inst['outstanding_m'] = inst['totdue'] - inst['amountpaid']

    # Months since edm
    inst['edm'] = pd.to_datetime(inst['edm'], errors='coerce')
    inst['edm_period'] = inst['edm'].dt.to_period('M')
    inst['msinceedm'] = (inst['payment_m'] - inst['edm_period']).apply(lambda x: x.n if pd.notna(x) else np.nan)

    # Drop duplicate client-months
    inst = inst.sort_values(['client_key', 'payment_m'])
    inst = inst.drop_duplicates(subset=['client_key', 'payment_m'], keep='last')

    # Outstanding at 20 and 30 months
    out20 = inst[inst['msinceedm'] == 20].groupby('client_key')['outstanding_m'].max().reset_index()
    out20.columns = ['client_key', 'outstanding20']

    out30 = inst[inst['msinceedm'] == 30].groupby('client_key')['outstanding_m'].max().reset_index()
    out30.columns = ['client_key', 'outstanding30']

    result = out20.merge(out30, on='client_key', how='outer')
    result['outstanding20'] = result['outstanding20'].fillna(0)
    result['outstanding30'] = result['outstanding30'].fillna(0)

    print(f"  Outstanding debt: {len(result)} clients")
    return result


# ================================================================
# (6) PENALTY DATA
# ================================================================
def compute_penalty(admin_raw):
    """Compute penalty data from 01_Cleaning.do lines 958-1048."""
    print("\n--- (6) Penalty Data ---")
    admin = admin_raw.copy()

    # Same reshape as outstanding
    mature_cols = [c for c in admin.columns if c.startswith('mature') and c[6:].isdigit()]
    max_inst = max(int(c[6:]) for c in mature_cols) if mature_cols else 0

    rows = []
    for _, loan in admin.iterrows():
        for i in range(1, max_inst + 1):
            mat_col = f'mature{i}'
            pay_col = f'payment{i}'
            amt_col = f'amount{i}'
            late_col = f'late{i}'
            pen_col = f'penalty{i}'
            if mat_col in admin.columns and pd.notna(loan.get(mat_col)):
                payment = loan.get(pay_col)
                if pd.isna(payment):
                    continue
                rows.append({
                    'client_key': loan['client_key'],
                    'loan_id': loan.get('loan_id', np.nan),
                    'installment': i,
                    'mature': loan[mat_col],
                    'payment': payment,
                    'amount': loan.get(amt_col, 0),
                    'late': loan.get(late_col, 0),
                    'penalty': loan.get(pen_col, 0),
                    'loansize': loan.get('loansize', 0),
                    'treatment': loan.get('treatment', np.nan),
                    'loid_baseline': loan.get('loid_baseline', np.nan),
                    'edm': loan.get('edm', np.nan),
                    'loannum': loan.get('loannum', np.nan) if 'loannum' in admin.columns else np.nan,
                    'disbdate': loan.get('disbdate', np.nan),
                    'random_date': loan.get('random_date', np.nan),
                })

    if not rows:
        return pd.DataFrame(columns=['client_key', 'totpenalty24', 'value_late24', 'loid_baseline', 'treatment'])

    inst = pd.DataFrame(rows)

    # Drop problem loans
    bad_loans = [63568220150514, 3676720160727, 32773920141012, 611443320141209]
    inst = inst[~inst['loan_id'].isin(bad_loans)]

    # Fix late
    inst['mature'] = pd.to_datetime(inst['mature'], errors='coerce')
    inst['payment'] = pd.to_datetime(inst['payment'], errors='coerce')
    inst.loc[inst['late'].isna() & (inst['payment'] <= inst['mature']), 'late'] = 0
    inst.loc[inst['late'].isna() & (inst['payment'] > inst['mature']),
             'late'] = (inst['payment'] - inst['mature']).dt.days

    # Filter: keep only 24 months from edm
    inst['edm'] = pd.to_datetime(inst['edm'], errors='coerce')
    inst['payment_m'] = inst['payment'].dt.to_period('M')
    inst['edm_m'] = inst['edm'].dt.to_period('M')
    inst['month24'] = inst['edm_m'] + 24

    inst = inst[inst['payment_m'] >= inst['edm_m']]
    inst = inst[inst['payment_m'] <= inst['month24']]

    # Late flag
    inst['late_f'] = (inst['late'] > 0).astype(float)

    # Aggregates by loan
    loan_agg = inst.groupby('loan_id').agg(
        client_key=('client_key', 'first'),
        treatment=('treatment', 'first'),
        loid_baseline=('loid_baseline', 'first'),
        totlateinst=('late_f', 'sum'),
        totpenalty24=('penalty', 'sum'),
        value_late_l=('amount', lambda x: (x * inst.loc[x.index, 'late']).sum()),
    ).reset_index()

    # Aggregate by client
    client_pen = loan_agg.groupby('client_key').agg(
        totlateinst=('totlateinst', 'sum'),
        totpenalty24=('totpenalty24', 'sum'),
        value_late_l=('value_late_l', 'sum'),
        treatment=('treatment', 'mean'),
        loid_baseline=('loid_baseline', 'mean'),
    ).reset_index()

    # Annualized value of late payments
    client_pen['value_late24'] = client_pen['value_late_l'] / (365 * 2)

    print(f"  Penalty data: {len(client_pen)} clients")
    return client_pen


# ================================================================
# (7) MERGE ALL
# ================================================================
def merge_all(bl, me, admin_clean, lo, outstanding):
    """Merge all datasets into final analysis file. Creates balanced 3-round panel."""
    print("\n--- (7) Merging All ---")

    # Baseline × midline/endline (1:m merge, keep all)
    merged = bl.merge(me, on='client_key', how='outer', suffixes=('', '_me'))
    print(f"  After BL × ME merge: {len(merged)} obs")

    # × Admin (m:1)
    admin_keep = [c for c in admin_clean.columns if c.startswith('ad_') or c.startswith('exploan') or c in
                  ['client_key', 'treatment', 'loid_baseline', 'took_loan', 'perfect_repayment',
                   'random_date', 'edm', 'random_cohort', 'nloans']]
    admin_keep = list(set(admin_keep))
    admin_subset = admin_clean[[c for c in admin_keep if c in admin_clean.columns]].copy()
    merged = merged.merge(admin_subset, on='client_key', how='left', suffixes=('', '_admin'))
    print(f"  After admin merge: {len(merged)} obs")

    # × Outstanding (m:1)
    merged = merged.merge(outstanding, on='client_key', how='left')
    merged['outstanding20'] = merged['outstanding20'].fillna(0)
    merged['outstanding30'] = merged['outstanding30'].fillna(0)

    # --- Panel structure (Stata: xtset + tsfill, full) ---
    # unique = 1 for the "latest" round per client
    merged['round'] = merged.get('round', pd.Series(np.nan, index=merged.index))
    max_round = merged.groupby('client_key')['round'].transform('max')
    merged['unique'] = 0
    merged.loc[merged['round'] == 2, 'unique'] = 1
    merged.loc[(max_round == 1) & (merged['round'] == 1), 'unique'] = 1
    merged.loc[max_round.isna(), 'unique'] = 1

    # Set round=0 for baseline-only observations (missing round)
    merged['round'] = merged['round'].fillna(0).astype(int)

    # tsfill, full: create balanced panel with rounds 0, 1, 2 for every client
    all_clients = merged['client_key'].unique()
    full_idx = pd.MultiIndex.from_product([all_clients, [0, 1, 2]], names=['client_key', 'round'])
    full_panel = pd.DataFrame(index=full_idx).reset_index()
    merged = full_panel.merge(merged, on=['client_key', 'round'], how='left')
    print(f"  After tsfill: {len(merged)} obs ({len(all_clients)} clients × 3 rounds)")

    # Attrition
    merged['attrition1'] = np.where((merged['round'] == 1) & merged.get('e_tooksurvey', pd.Series()).isna(), 1, 0)
    merged['attrition2'] = np.where((merged['round'] == 2) & merged.get('e_tooksurvey', pd.Series()).isna(), 1, 0)
    a1 = merged.groupby('client_key')['attrition1'].transform('max')
    a2 = merged.groupby('client_key')['attrition2'].transform('max')
    merged['a_both'] = ((a1 == 1) & (a2 == 1)).astype(int)
    # Recode round for a_both clients: swap 0↔1
    swap_mask = merged['a_both'] == 1
    merged.loc[swap_mask & (merged['round'] == 0), 'round'] = -1
    merged.loc[swap_mask & (merged['round'] == 1), 'round'] = 0
    merged.loc[swap_mask & (merged['round'] == -1), 'round'] = 1

    # Fill treatment/loid across rounds for same client
    for v in ['treatment', 'loid_baseline']:
        if v in merged.columns:
            max_val = merged.groupby('client_key')[v].transform('max')
            merged[v] = merged[v].fillna(max_val)

    # Fill took_loan across rounds
    if 'took_loan' in merged.columns:
        merged['took_loan'] = merged.groupby('client_key')['took_loan'].transform('max')
    if 'perfect_repayment' in merged.columns:
        merged['perfect_repayment'] = merged.groupby('client_key')['perfect_repayment'].transform('max')

    # --- Derived variables computed BEFORE filling BL vars (matching Stata order) ---

    # Recompute unique after tsfill
    max_round_post = merged.groupby('client_key')['round'].transform('max')
    merged['unique'] = 0
    merged.loc[merged['round'] == 2, 'unique'] = 1
    merged.loc[(max_round_post == 1) & (merged['round'] == 1), 'unique'] = 1
    merged.loc[max_round_post == 0, 'unique'] = 1

    # Registration
    if 'b_registration' in merged.columns:
        merged['registration'] = (merged['b_registration'] == 1).astype(float)
        merged.loc[merged['b_staff_yesno'].isna(), 'registration'] = np.nan
        merged.loc[merged['b_registration'] == 3, 'registration'] = np.nan

    # has_other_loan
    if 'b_othercredit' in merged.columns:
        merged['b_has_other_loan'] = (merged['b_othercredit'] > 0).astype(float)

    # Total wagebill (Stata line 843: gen b_wagebill_all = full + part + temp + intern)
    # Stata '+' propagates missing: if any part is NaN, sum is NaN
    wb_parts = ['b_wagebill_full', 'b_wagebill_part', 'b_wagebill_temp', 'b_wagebill_intern']
    existing_wb = [c for c in wb_parts if c in merged.columns]
    if existing_wb:
        merged['b_wagebill_all'] = merged[existing_wb].sum(axis=1, min_count=1)  # NaN if all missing
        merged.loc[(merged['b_staff_yesno'] == 0) & merged['b_wagebill_all'].isna(), 'b_wagebill_all'] = 0

    # Fill experience
    if 'b_experience' in merged.columns and 'b_start_year' in merged.columns:
        merged.loc[merged['b_experience'].isna(), 'b_experience'] = 2016 - merged.loc[merged['b_experience'].isna(), 'b_start_year']

    # Winsorize baseline financial vars (BEFORE filling BL vars across rounds)
    for v in ['b_monthly_profits', 'b_monthly_expenses', 'b_monthly_revenue', 'b_wagebill_all']:
        if v in merged.columns:
            merged[f'{v}_w'] = winsor(merged[v], p=0.01)

    # --- NOW fill baseline and admin variables across rounds ---
    bl_vars = [c for c in merged.columns if c.startswith('b_') or c.startswith('ad_') or
               c.startswith('exploan') or c in ['random_date', 'edm', 'random_cohort',
               'nloans', 'digitspan', 'raven_score', 'fl_score', 'othercredit2',
               'registration', 'years_of_ed', 'prop_invest']]
    for v in bl_vars:
        if v in merged.columns:
            filled = merged.groupby('client_key')[v].transform('first')
            merged[v] = merged[v].fillna(filled)

    # Also fill b_size_lastloan specifically (Stata: egen max)
    if 'b_size_lastloan' in merged.columns:
        merged['b_size_lastloan'] = merged.groupby('client_key')['b_size_lastloan'].transform('max')

    # Years of education (Stata recode: unmapped values stay unchanged, so 5→5)
    if 'b_education' in merged.columns:
        ed_map = {1: 0, 2: 3, 3: 4, 4: 5, 5: 5, 6: 8, 7: 10, 8: 12, 9: 10, 10: 12,
                  11: 13, 12: 14, 13: 14, 14: 16, 15: 17, 16: 18}
        merged['years_of_ed'] = merged['b_education'].map(ed_map)

    # Prop invest
    if 'b_risk_investchoice' in merged.columns:
        merged['prop_invest'] = merged['b_risk_investchoice'] / 10000

    # Missing indicators and mean-filled versions (Stata: gen mis_X=X==.; gen X_f=X; replace X_f=mean if X==.)
    # IMPORTANT: Stata regressions use _w (with missing), NOT _f. So keep _w as-is.
    for v in ['b_monthly_profits_w', 'b_monthly_revenue_w', 'b_monthly_expenses_w',
              'b_wagebill_all_w', 'b_hh_exp_w', 'b_num_employ_full']:
        if v not in merged.columns and v.replace('_w', '') in merged.columns:
            base = v.replace('_w', '')
            merged[v] = winsor(merged[base], p=0.01) if '_w' in v else merged[base]

        if v in merged.columns:
            merged[f'mis_{v}'] = merged[v].isna().astype(float)
            # Create mean-filled version as _f, keep _w untouched
            mean_val = merged[v].mean()
            merged[f'{v}_f'] = merged[v].fillna(mean_val)

    # b_hh_exp_w
    if 'b_hh_exp' in merged.columns and 'b_hh_exp_w' not in merged.columns:
        merged['b_hh_exp_w'] = winsor(merged['b_hh_exp'], p=0.01)
        merged['mis_b_hh_exp_w'] = merged['b_hh_exp_w'].isna().astype(float)
        merged['b_hh_exp_w_f'] = merged['b_hh_exp_w'].fillna(merged['b_hh_exp_w'].mean())

    # Loan installment size
    if 'ad_exploan_size' in merged.columns and 'ad_exploan_ninstal' in merged.columns:
        merged['ad_exploan_instalsiz'] = merged['ad_exploan_size'] / merged['ad_exploan_ninstal']
        merged.loc[merged.get('ad_exploan', pd.Series()) == 0, 'ad_exploan_instalsiz'] = 0

    # long_term
    merged['long_term'] = (merged['outstanding30'] > 0).astype(float)

    # loid_bas (alias used in Table 4)
    if 'loid_baseline' in merged.columns:
        merged['loid_bas'] = merged['loid_baseline']

    # e_profits_all (aggregate)
    prof_cols = [f'e_month_profit_{x}' for x in range(1, 6) if f'e_month_profit_{x}' in merged.columns]
    if prof_cols:
        merged['e_profits_all'] = merged[prof_cols].fillna(0).sum(axis=1)
        merged.loc[merged.get('e_month_profit_1', pd.Series()).isna(), 'e_profits_all'] = np.nan

    # b_gender fill from endline (Stata: egen max_egender=max(e_gender), by(client_key))
    if 'b_gender' in merged.columns and 'e_gender' in merged.columns:
        max_egender = merged.groupby('client_key')['e_gender'].transform('max')
        merged.loc[merged['b_gender'].isna(), 'b_gender'] = max_egender[merged['b_gender'].isna()]

    # b_age fill (Stata: egen max_eyear=max(e_birth_year), by(client_key); replace b_age=2016-max_eyear if b_age==.)
    if 'b_age' in merged.columns and 'e_birth_year' in merged.columns:
        max_eyear = merged.groupby('client_key')['e_birth_year'].transform('max')
        merged.loc[merged['b_age'].isna(), 'b_age'] = 2016 - max_eyear[merged['b_age'].isna()]

    print(f"  Final merged: {len(merged)} obs, {merged.shape[1]} vars")
    return merged


# ================================================================
# MAIN
# ================================================================
def main():
    print("=" * 60)
    print("01_clean.py: Data Cleaning for 192297-V1")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Clean baseline
    bl = clean_baseline()

    # Step 2: Clean midline/endline
    me = clean_midline_endline(bl)

    # Step 3: Clean admin
    admin_raw = pd.read_stata(os.path.join(RAW_DATA_DIR, '13-ABA-admin.dta'),
                              convert_categoricals=False)
    admin_clean = clean_admin()

    # Step 4: Loan officer (simplified)
    lo = clean_loan_officer()

    # Step 5: Outstanding debt
    outstanding = compute_outstanding(admin_raw)

    # Step 6: Penalty data
    penalty = compute_penalty(admin_raw)

    # Step 7: Merge all
    merged = merge_all(bl, me, admin_clean, lo, outstanding)

    # Save
    merged.to_parquet(os.path.join(OUTPUT_DIR, '16-ABA-All-Merged.parquet'), index=False)
    penalty.to_parquet(os.path.join(OUTPUT_DIR, '17-ABA-penalty-clean.parquet'), index=False)

    print(f"\nSaved merged: {merged.shape}")
    print(f"Saved penalty: {penalty.shape}")

    # ================================================================
    # VALIDATION: Compare against Stata-cleaned data
    # ================================================================
    print("\n" + "=" * 60)
    print("VALIDATION: Comparing to Stata-cleaned data")
    print("=" * 60)

    stata_merged = pd.read_stata(os.path.join(CLEAN_DATA_DIR, '16-ABA-All-Merged.dta'),
                                 convert_categoricals=False)
    stata_penalty = pd.read_stata(os.path.join(CLEAN_DATA_DIR, '17-ABA-penalty-clean.dta'),
                                  convert_categoricals=False)

    print(f"\n  Shape: Ours={merged.shape}, Stata={stata_merged.shape}")
    print(f"  Penalty: Ours={penalty.shape}, Stata={stata_penalty.shape}")

    # Compare key variables on unique==1 sample
    ours_uniq = merged[merged['unique'] == 1].drop_duplicates('client_key')
    stata_uniq = stata_merged[stata_merged.get('unique', pd.Series()) == 1].copy()
    if 'client_key' in stata_uniq.columns:
        stata_uniq = stata_uniq.drop_duplicates('client_key')

    print(f"\n  Unique==1: Ours={len(ours_uniq)}, Stata={len(stata_uniq)}")

    # Compare key vars
    key_vars = ['b_age', 'b_gender', 'years_of_ed', 'registration', 'b_experience',
                'b_monthly_profits_w', 'b_monthly_revenue_w', 'b_monthly_expenses_w',
                'b_wagebill_all_w', 'b_staff_yesno', 'b_num_employ_full',
                'b_size_lastloan', 'b_has_other_loan', 'b_othercredit',
                'treatment', 'took_loan']

    for v in key_vars:
        if v in ours_uniq.columns and v in stata_uniq.columns:
            our_mean = ours_uniq[v].mean()
            stata_mean = stata_uniq[v].mean()
            diff = our_mean - stata_mean if pd.notna(stata_mean) else np.nan
            match = 'OK' if abs(diff) < 0.01 else 'DIFF'
            print(f"    {v:30s}  Ours={our_mean:10.3f}  Stata={stata_mean:10.3f}  Diff={diff:+8.3f}  [{match}]")
        elif v in stata_uniq.columns:
            print(f"    {v:30s}  MISSING in our data")

    return merged, penalty


if __name__ == '__main__':
    main()
