"""
01_clean.py — Data Cleaning (translates clean.do to Python)

Input:  228101-V1/data/raw/gender_gaps_ghana.dta
Output: replication/output/gender_gaps_ghana_clean.parquet

Translates the Stata cleaning pipeline exactly, including:
- Variable renaming and dummy generation
- Product categorization via split/reshape logic
- Business network variables (help, meetings, collaborations)
- Supplier and client variables
- Employment variables
- Z-score index construction (business_network_index, friendrel, female)
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from utils import zindex, DATA_PATH, CLEAN_DATA_PATH


def load_raw_data():
    """Load raw Stata data with numeric codes (not category labels)."""
    df = pd.read_stata(DATA_PATH, convert_categoricals=False)
    print(f"Raw data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def clean_demographics(df):
    """Rename survey variables and generate demographic dummies."""
    # Rename variables (matching clean.do lines 14-19)
    df = df.rename(columns={
        'a5': 'age',
        'a16': 'years_operation',
        'a17': 'monthly_revenues',
        'b3': 'education',
        'b4': 'marital_status',
        'b8': 'children',
    })

    # Female dummy (clean.do lines 22-23)
    # gender column is string: "Male", "Female", or ""
    df['female'] = np.where(df['gender'] == 'Female', 1.0,
                   np.where(df['gender'] == 'Male', 0.0, np.nan))

    # College degree dummy (education==27) (clean.do lines 26-27)
    df['collegedegree'] = np.where(
        df['education'].isna() | (df['education'] == -666), np.nan,
        np.where(df['education'] == 27, 1.0, 0.0)
    )

    # Married dummy (marital_status 3 or 4) (clean.do lines 58-60)
    df['married'] = np.where(
        df['marital_status'].isna(), np.nan,
        np.where((df['marital_status'] == 3) | (df['marital_status'] == 4), 1.0, 0.0)
    )

    return df


def categorize_products(df):
    """
    Replicate Stata's split + reshape long + product categorization.
    b2 is a space-delimited string of product codes.
    """
    # Parse product codes from b2 (clean.do lines 30-50)
    product_categories = {
        'product_aginput': range(1, 8),       # 1-7
        'product_livestock': range(8, 15),     # 8-14
        'product_crop': range(15, 50),         # 15-49
        'product_cocoa': range(50, 52),        # 50-51
        'product_forestry': range(53, 56),     # 53-55
        'product_fish': range(56, 58),         # 56-57
        'product_processed': range(58, 64),    # 58-63
    }

    # Initialize product dummies
    for cat in list(product_categories.keys()) + ['product_other']:
        df[cat] = 0.0

    for idx, row in df.iterrows():
        b2_val = row['b2']
        if pd.isna(b2_val) or b2_val.strip() == '':
            # If b2 is empty, all product dummies should be NaN (clean.do lines 53-56)
            for cat in list(product_categories.keys()) + ['product_other']:
                df.at[idx, cat] = np.nan
            continue

        codes = b2_val.strip().split()
        for code_str in codes:
            try:
                code = int(code_str)
            except ValueError:
                continue

            if code == -666:
                df.at[idx, 'product_other'] = 1.0
                continue

            for cat, rng in product_categories.items():
                if code in rng:
                    df.at[idx, cat] = 1.0
                    break

    return df


def clean_business_characteristics(df):
    """Generate business characteristic dummies."""
    # Business primary income (clean.do lines 62-64)
    df['business_primary_inc'] = np.where(df['b11b'] == 1, 1.0, 0.0)
    df.loc[df['b11b'].isna(), 'business_primary_inc'] = np.nan

    # Work for pay (clean.do lines 67-68)
    pay_cols = ['b11c_1', 'b11c_2', 'b11c_3', 'b11c_4', 'b11c_5']
    df['work_for_pay'] = 0.0
    for col in pay_cols:
        df['work_for_pay'] = np.where(df[col] == 1, 1.0, df['work_for_pay'])
    df.loc[df['b11c'] == '', 'work_for_pay'] = np.nan

    # Business at home (clean.do lines 71-72)
    df['business_at_home'] = np.where(
        (df['b13'] == 1) | (df['b13'] == 2), 1.0, 0.0
    )
    df.loc[df['b13'].isna() | (df['b13'] == -666), 'business_at_home'] = np.nan

    # Export dummy (clean.do lines 243-244)
    df['export'] = np.where(df['d1'] == 1, 1.0, 0.0)
    df.loc[df['d1'].isna(), 'export'] = np.nan

    return df


def clean_business_help(df):
    """Clean business help network variables (clean.do lines 74-114)."""
    # Any business help
    df['any_business_help'] = np.where(df['b26'] == 1, 1.0, 0.0)
    df.loc[df['b26'].isna(), 'any_business_help'] = np.nan

    # Number of people helped (business_help_n)
    # Logic: 0 if b26==2 (no help), then count via b26c chain
    df['business_help_n'] = np.nan
    df.loc[df['b26'] == 2, 'business_help_n'] = 0
    df.loc[df['b26c'] == 2, 'business_help_n'] = 1    # stopped after 1st
    df.loc[df['b26c1'] == 2, 'business_help_n'] = 2   # stopped after 2nd
    df.loc[df['b26c2'] == 2, 'business_help_n'] = 3   # stopped after 3rd
    df.loc[df['b26c3'] == 2, 'business_help_n'] = 4   # stopped after 4th
    df.loc[df['b26c3'] == 1, 'business_help_n'] = 5   # continued to 5th

    # Women count among help recipients
    help_gender_cols = ['b26a', 'b26a_1', 'b26a_2', 'b26a_3', 'b26a_4']
    df['business_help_women'] = 0.0
    for col in help_gender_cols:
        df['business_help_women'] += (df[col] == 2).astype(float)
    df.loc[df['business_help_n'].isna() | (df['business_help_n'] == 0),
           'business_help_women'] = np.nan
    df['share_business_help_women'] = df['business_help_women'] / df['business_help_n']

    # Friends, relatives, friend/relative, association counts
    help_rel_cols = ['b26b', 'b26b_1', 'b26b_2', 'b26b_3', 'b26b_4']
    for var, code_check in [
        ('business_help_friends', lambda x: x == 1),
        ('business_help_relatives', lambda x: x == 2),
        ('business_help_friendrel', lambda x: (x == 1) | (x == 2)),
        ('business_help_assoc', lambda x: x == 3),
    ]:
        df[var] = 0.0
        for col in help_rel_cols:
            df[var] += code_check(df[col]).astype(float)
        df.loc[df['business_help_n'].isna() | (df['business_help_n'] == 0), var] = np.nan

    # Share variables
    df['share_business_help_friends'] = df['business_help_friends'] / df['business_help_n']
    df['share_business_help_relatives'] = df['business_help_relatives'] / df['business_help_n']
    df['share_business_help_friendrel'] = df['business_help_friendrel'] / df['business_help_n']
    df['share_business_help_assoc'] = df['business_help_assoc'] / df['business_help_n']

    return df


def clean_meetings(df):
    """Clean business meeting/association variables (clean.do lines 116-135)."""
    # Business association member
    df['business_association'] = np.where(df['b27'] == 1, 1.0, 0.0)
    df.loc[df['b27'].isna(), 'business_association'] = np.nan

    # Meet other business owners
    df['meet_other_business'] = np.where(df['b28'] == 1, 1.0, 0.0)
    df.loc[df['b28'].isna(), 'meet_other_business'] = np.nan

    # Meeting frequency
    df['meet_other_business_freq'] = df['b28a']

    # Number of business owners they talk to
    df['meet_other_business_n'] = df['b28b'].copy()
    df.loc[df['meet_other_business'] == 0, 'meet_other_business_n'] = 0

    # Women and friend/relative counts in meetings
    df['meet_other_business_women'] = df['b28b_1'].copy()
    df['meet_other_business_friendrel'] = df['b28b_2'].copy()

    # Share variables
    df['share_meet_business_women'] = (
        df['meet_other_business_women'] / df['meet_other_business_n']
    )
    df['share_meet_business_friendrel'] = (
        df['meet_other_business_friendrel'] / df['meet_other_business_n']
    )

    return df


def clean_collaborations(df):
    """Total collaborations (clean.do line 139)."""
    collab_cols = ['b30a', 'b30b', 'b30c', 'b30d', 'b30e', 'b30f', 'b30g']
    # Stata: egen total_collab = rowtotal(b30a-b30g), missing
    # With 'missing' option: returns NaN only if ALL are NaN
    df['total_collab'] = df[collab_cols].sum(axis=1, skipna=True)
    all_missing = df[collab_cols].isna().all(axis=1)
    df.loc[all_missing, 'total_collab'] = np.nan
    return df


def clean_suppliers(df):
    """Clean supplier variables (clean.do lines 141-171)."""
    # Number of suppliers
    df['suppliers_n'] = df['b32a'].copy()

    # Number of important suppliers (count non-empty names)
    supplier_name_cols = ['b32b1', 'b32b1_2', 'b32b1_3']
    df['suppliers_impt_n'] = 0
    for col in supplier_name_cols:
        df['suppliers_impt_n'] += (df[col] != '').astype(int)

    # Important suppliers who are women (gender==2 means Female)
    supplier_gender_cols = ['b32b2', 'b32b2_2', 'b32b2_3']
    df['suppliers_impt_women'] = 0.0
    for col in supplier_gender_cols:
        df['suppliers_impt_women'] += (df[col] == 2).astype(float)
    # Set to NaN if all gender fields are NaN
    all_gender_na = True
    for col in supplier_gender_cols:
        all_gender_na = all_gender_na & df[col].isna()
    df.loc[all_gender_na, 'suppliers_impt_women'] = np.nan

    df['share_suppliers_impt_women'] = df['suppliers_impt_women'] / df['suppliers_impt_n']
    df['any_female_suppliers'] = (
        (df['suppliers_impt_women'] > 0) & df['suppliers_impt_women'].notna()
    ).astype(float)

    # Friends/relatives as suppliers (clean.do lines 166-171)
    # Stata: destring b32b5_2, replace -- needed because it's stored as string
    # Then for each x in {b32b5, b32b5_2, b32b5_3}: check x_1==1 | x_2==1
    # x_1 = friend indicator, x_2 = relative indicator

    # First, destring b32b5_2 (convert from string to numeric)
    df['b32b5_2'] = pd.to_numeric(df['b32b5_2'], errors='coerce')

    df['suppliers_impt_friendrel'] = 0.0
    # Supplier 1: b32b5_1 (friend) or b32b5_2 (relative)
    df['suppliers_impt_friendrel'] += (
        (df['b32b5_1'] == 1) | (df['b32b5_2'] == 1)
    ).astype(float)
    # Supplier 2: b32b5_2_1 (friend) or b32b5_2_2 (relative)
    df['suppliers_impt_friendrel'] += (
        (df['b32b5_2_1'] == 1) | (df['b32b5_2_2'] == 1)
    ).astype(float)
    # Supplier 3: b32b5_3_1 (friend) or b32b5_3_2 (relative)
    df['suppliers_impt_friendrel'] += (
        (df['b32b5_3_1'] == 1) | (df['b32b5_3_2'] == 1)
    ).astype(float)

    df['share_suppliers_impt_friendrel'] = (
        df['suppliers_impt_friendrel'] / df['suppliers_impt_n']
    )

    return df


def clean_clients(df):
    """Clean client variables (clean.do lines 173-204)."""
    # Number of clients
    df['clients_n'] = df['b32c'].copy()

    # Number of important clients
    client_name_cols = ['b32d1', 'b32d1_2', 'b32d1_3']
    df['clients_impt_n'] = 0
    for col in client_name_cols:
        df['clients_impt_n'] += (df[col] != '').astype(int)

    # Important clients who are women
    client_gender_cols = ['b32d2', 'b32d2_2', 'b32d2_3']
    df['clients_impt_women'] = 0.0
    for col in client_gender_cols:
        df['clients_impt_women'] += (df[col] == 2).astype(float)
    all_gender_na = True
    for col in client_gender_cols:
        all_gender_na = all_gender_na & df[col].isna()
    df.loc[all_gender_na, 'clients_impt_women'] = np.nan

    df['share_clients_impt_women'] = df['clients_impt_women'] / df['clients_impt_n']
    df['any_female_clients'] = (
        (df['clients_impt_women'] > 0) & df['clients_impt_women'].notna()
    ).astype(float)

    # Friends/relatives as clients (clean.do lines 198-204)
    df['b32d5_2'] = pd.to_numeric(df['b32d5_2'], errors='coerce')

    df['clients_impt_friendrel'] = 0.0
    # Client 1: b32d5_1 (friend) or b32d5_2 (relative)
    df['clients_impt_friendrel'] += (
        (df['b32d5_1'] == 1) | (df['b32d5_2'] == 1)
    ).astype(float)
    # Client 2: b32d5_2_1 (friend) or b32d5_2_2 (relative)
    df['clients_impt_friendrel'] += (
        (df['b32d5_2_1'] == 1) | (df['b32d5_2_2'] == 1)
    ).astype(float)
    # Client 3: b32d5_3_1 (friend) or b32d5_3_2 (relative)
    df['clients_impt_friendrel'] += (
        (df['b32d5_3_1'] == 1) | (df['b32d5_3_2'] == 1)
    ).astype(float)

    df['share_clients_impt_friendrel'] = (
        df['clients_impt_friendrel'] / df['clients_impt_n']
    )

    return df


def clean_employment(df):
    """Clean employment and hours variables (clean.do lines 206-236)."""
    df['hours_business'] = df['c1'].copy()
    df['hours_salary'] = df['c3a'].copy()
    df['hours_childcare'] = df['c3b'].copy()

    df['workers_wage'] = df['c4a'].copy()
    df['workers_casual'] = df['c4b'].copy()
    df['workers_unpaid'] = df['c4c'].copy()

    # Stata: egen workers_total = rowtotal(workers_wage workers_casual workers_unpaid)
    # rowtotal without 'missing' treats NaN as 0
    worker_cols = ['workers_wage', 'workers_casual', 'workers_unpaid']
    df['workers_total'] = df[worker_cols].fillna(0).sum(axis=1)

    df['workers_female'] = df['c5a'].copy()
    df['workers_female_share'] = df['workers_female'] / df['workers_total']
    # Replace with NaN if share > 1 (clean.do line 236)
    df.loc[
        (df['workers_female_share'] > 1) & df['workers_female_share'].notna(),
        'workers_female_share'
    ] = np.nan

    return df


def clean_sales_profits(df):
    """Clean sales and profit variables (clean.do lines 247-251)."""
    df['sales_lastmonth'] = df['g4'].copy()
    df['logsales_lastmonth'] = np.log(df['sales_lastmonth'])
    df['profits_lastmonth'] = df['g6'].copy()
    # NOTE: output.do uses 'profit' but clean.do creates 'profits_lastmonth'.
    # This is a bug in the original code. We create 'profit' as alias.
    df['profit'] = df['profits_lastmonth'].copy()
    return df


def generate_indices(df):
    """Generate z-score network indices (clean.do lines 253-289)."""
    # Business network index
    bni_vars = [
        'business_help_n', 'meet_other_business', 'meet_other_business_n',
        'business_association', 'total_collab', 'suppliers_n', 'clients_n'
    ]
    df['business_network_index'] = zindex(df, bni_vars, gen_name='business_network_index')

    # Friends/relatives networking index
    fri_vars = [
        'share_business_help_friendrel', 'share_meet_business_friendrel',
        'share_suppliers_impt_friendrel'
    ]
    df['business_network_friendrel_index'] = zindex(
        df, fri_vars, gen_name='business_network_friendrel_index'
    )

    # Female networking index
    fem_vars = [
        'share_business_help_women', 'share_meet_business_women',
        'share_suppliers_impt_women', 'share_clients_impt_women'
    ]
    df['business_network_female_index'] = zindex(
        df, fem_vars, gen_name='business_network_female_index'
    )

    return df


def main():
    print("=" * 60)
    print("01_clean.py: Data Cleaning")
    print("=" * 60)

    df = load_raw_data()

    print("\nStep 1: Demographics...")
    df = clean_demographics(df)

    print("Step 2: Product categorization...")
    df = categorize_products(df)

    print("Step 3: Business characteristics...")
    df = clean_business_characteristics(df)

    print("Step 4: Business help networks...")
    df = clean_business_help(df)

    print("Step 5: Meetings and associations...")
    df = clean_meetings(df)

    print("Step 6: Collaborations...")
    df = clean_collaborations(df)

    print("Step 7: Suppliers...")
    df = clean_suppliers(df)

    print("Step 8: Clients...")
    df = clean_clients(df)

    print("Step 9: Employment...")
    df = clean_employment(df)

    print("Step 10: Sales and profits...")
    df = clean_sales_profits(df)

    print("Step 11: Z-score indices...")
    df = generate_indices(df)

    # Save
    os.makedirs(os.path.dirname(CLEAN_DATA_PATH), exist_ok=True)
    df.to_parquet(CLEAN_DATA_PATH, index=False)
    print(f"\nCleaned data saved: {CLEAN_DATA_PATH}")
    print(f"Final shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # Summary stats for key variables
    print("\n--- Key Variable Summary ---")
    key_vars = ['female', 'profit', 'age', 'years_operation',
                'business_network_index', 'business_network_female_index',
                'business_network_friendrel_index']
    for v in key_vars:
        if v in df.columns:
            n_valid = df[v].notna().sum()
            m = df[v].mean()
            s = df[v].std()
            print(f"  {v:40s}  N={n_valid:5d}  mean={m:10.3f}  sd={s:10.3f}")

    return df


if __name__ == '__main__':
    main()
