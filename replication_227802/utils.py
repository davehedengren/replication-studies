"""
Shared utilities for the Childcare Laws replication study (227802-V1).

Paper: "Filling the Gaps: Childcare Laws for Women's Economic Empowerment"
Authors: S Anukriti, Lelys Dinarte-Diaz, Maria Montoya-Aguirre, Alena Sakhonchik
"""

import numpy as np
import pandas as pd

RAW_DATA_DIR = '227802-V1/data/raw'
CLEAN_DATA_PATH = 'replication_227802/output/analysis_data.parquet'
OUTPUT_DIR = 'replication_227802/output'

OUTCOME_VARS = ['lfpr_f_25_54', 'lfpr_hhchild_f', 'emp2pop_f_yge25', 'unemp_f_yge25']
OUTCOME_LABELS = {
    'lfpr_f_25_54': 'FLFP 25-54 (All)',
    'lfpr_hhchild_f': 'FLFP 25-54 w. children <6',
    'emp2pop_f_yge25': 'Employment to pop.',
    'unemp_f_yge25': 'Unemp. rate',
}

TREATMENT_VARS = ['enactment', 'enforcement', 'enactment_av', 'enactment_af', 'enactment_qua']
TREATMENT_LABELS = {
    'enactment': 'Law enactment',
    'enforcement': 'Law enforcement',
    'enactment_av': 'Availability',
    'enactment_af': 'Affordability',
    'enactment_qua': 'Quality',
}

REGION_MAP = {
    'East Asia & Pacific': 'EAP',
    'Europe & Central Asia': 'ECA',
    'High income: OECD': 'HIC',
    'Latin America & Caribbean': 'LAC',
    'Middle East & North Africa': 'MENA',
    'South Asia': 'SAS',
    'Sub-Saharan Africa': 'SSA',
}


def prepare_estimation_sample(df, y_var, treat_time):
    """
    Prepare estimation sample matching the R code logic:
    1. Remove years where ALL obs of y_var are missing
    2. Remove countries where ANY obs of y_var is missing
    3. Set treat=0 for never-treated, treat=1 if year >= treatment time
    4. Keep only never-treated + treated with treatment between (first_year+2) and 2020
    """
    foo = df.copy()

    # Remove years where all y_var values are NaN
    year_valid = foo.groupby('year')[y_var].apply(lambda x: not x.isna().all())
    valid_years = year_valid[year_valid].index
    foo = foo[foo['year'].isin(valid_years)]

    # Remove countries where any y_var is missing
    country_valid = foo.groupby('id')[y_var].apply(lambda x: not x.isna().any())
    valid_countries = country_valid[country_valid].index
    foo = foo[foo['id'].isin(valid_countries)]

    # First treatment year
    first_treat = foo['year'].min()

    # Set treatment variable
    foo['treat'] = np.where(
        foo[treat_time].isna(), 0,
        np.where(foo['year'] >= foo[treat_time], 1, 0)
    )

    # Keep never-treated + treated with treatment in (first_treat+2, 2020)
    valid_treat_range = range(int(first_treat + 2), 2021)
    foo = foo[
        foo[treat_time].isna() |
        foo[treat_time].isin(valid_treat_range)
    ]

    return foo


def create_sub_experiment(dataset, time_col, group_col, adoption_col,
                          focal_time, kappa_pre, kappa_post):
    """
    Create a sub-experiment dataset (translates R create_sub_exp function).
    Includes treated units adopting at focal_time and clean controls.
    """
    dt = dataset.copy()

    min_time = dt[time_col].min()
    max_time = dt[time_col].max()

    # Keep treated at focal_time + clean controls (adopt after focal_time + kappa_post) + never-treated
    mask = (
        (dt[adoption_col] == focal_time) |
        (dt[adoption_col] > focal_time + kappa_post) |
        dt[adoption_col].isna()
    )
    dt = dt[mask]

    # Limit to event window
    time_range = range(int(focal_time - kappa_pre), int(focal_time + kappa_post + 1))
    dt = dt[dt[time_col].isin(time_range)]

    # Add variables
    dt = dt.copy()
    dt['treat'] = np.where(dt[adoption_col] == focal_time, 1, 0)
    dt['event_time'] = dt[time_col] - focal_time
    dt['post'] = np.where(dt[time_col] >= focal_time, 1, 0)
    dt['feasible'] = int(focal_time - kappa_pre >= min_time and
                         focal_time + kappa_post <= max_time)
    dt['sub_exp'] = focal_time

    return dt


def create_stack(dataset, time_col, group_col, adoption_col, kappa_pre, kappa_post):
    """
    Create stacked dataset from all sub-experiments (translates R create_stack).
    """
    events = dataset[dataset[adoption_col].notna()][adoption_col].unique()
    events = sorted(events)

    stacked = []
    for focal_time in events:
        sub = create_sub_experiment(
            dataset, time_col, group_col, adoption_col,
            focal_time, kappa_pre, kappa_post
        )
        stacked.append(sub)

    return pd.concat(stacked, ignore_index=True)


def weights_stacked_did(dataset, treated_var, event_time_var, subexp_var):
    """
    Compute stacked DiD weights (translates R weights_stackedDiD).
    """
    dt = dataset.copy()

    # Stack-level counts by event_time
    stack_counts = dt.groupby(event_time_var).agg(
        stack_treat_n=(treated_var, 'sum'),
        stack_n=(treated_var, 'count'),
    ).reset_index()
    stack_counts['stack_control_n'] = stack_counts['stack_n'] - stack_counts['stack_treat_n']

    dt = dt.merge(stack_counts, on=event_time_var, how='left')

    # Sub-experiment level counts
    sub_counts = dt.groupby([subexp_var, event_time_var]).agg(
        sub_n=(treated_var, 'count'),
        sub_treat_n=(treated_var, 'sum'),
    ).reset_index()
    sub_counts['sub_control_n'] = sub_counts['sub_n'] - sub_counts['sub_treat_n']

    dt = dt.merge(sub_counts, on=[subexp_var, event_time_var], how='left')

    # Shares
    dt['sub_share'] = dt['sub_n'] / dt['stack_n']
    dt['sub_treat_share'] = dt['sub_treat_n'] / dt['stack_treat_n']
    dt['sub_control_share'] = dt['sub_control_n'] / dt['stack_control_n']

    # Weights
    dt['stack_weight'] = np.where(
        dt[treated_var] == 1, 1.0,
        dt['sub_treat_share'] / dt['sub_control_share']
    )

    return dt
