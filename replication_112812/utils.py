"""
Replication of: "Benefits of Neuroeconomic Modeling: New Policy Interventions
and Predictors of Preference"
Authors: Ian Krajbich, Bastiaan Oud, and Ernst Fehr (2014, AER P&P)

Shared paths, variable lists, and helper functions.
"""
import os
import numpy as np
import pandas as pd

# === PATHS ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', '112812-V1', 'P2014_1135_data')
STUDY1_DTA = os.path.join(DATA_DIR, 'krajbich_data_study1.dta')
STUDY2_CSV = os.path.join(DATA_DIR, 'fig4_data.csv')
OUTPUT_DIR = BASE_DIR

# === CONSTANTS ===
HUMAN_CHOOSER = 1
FIRST_BLOCK = 1

# Block type labels
BLOCK_LABELS = {1: 'first block', 2: 'pre-intervention', 3: 'intervention', 4: 'post-intervention'}

# Published key statistics
PUBLISHED = {
    'mean_rt_low_stakes': 1.65,      # seconds
    'mean_rt_high_stakes': 1.11,     # seconds
    'avg_missed_trials': 44,         # trials
    'avg_money_left': 20.10,         # CHF per subject
    'n_subjects_study1': 49,
    'n_probabilistic_ug': 16,        # out of 18
    'n_subjects_study2': 18,
}


def load_study1():
    """Load Study 1 (time-constrained food choice) data."""
    df = pd.read_stata(STUDY1_DTA, convert_categoricals=False)
    return df


def load_study2():
    """Load Study 2 (ultimatum game) data."""
    df = pd.read_csv(STUDY2_CSV)
    return df


def create_fourblocktype(df):
    """Create the four-block-type variable matching Stata code."""
    df = df.copy()
    df['fourblocktype'] = np.nan
    df.loc[df['blocktype'] == FIRST_BLOCK, 'fourblocktype'] = 1
    df.loc[df['nonint_pre_int'] == 1, 'fourblocktype'] = 2
    df.loc[df['interventionblock'] == 1, 'fourblocktype'] = 3
    df.loc[df['nonint_post_int'] == 1, 'fourblocktype'] = 4
    return df


def demean_choice_surplus(df):
    """
    De-mean choice surplus using individual means from blocks 2-5 (non-first blocks).
    Matches the Stata code in the .do file.
    """
    df = df.copy()

    # Compute subject-level mean choicevalue for each block
    block_means = df.groupby(['subject', 'fourblocktype'])['choicevalue'].mean().reset_index()
    block_means.columns = ['subject', 'fourblocktype', 'mean_choicevalue']

    # Compute individual means from blocks 2-5 (blocksequence != 1)
    blocks_2345 = df[df['blocksequence'] != 1].groupby('subject')['choicevalue'].mean().reset_index()
    blocks_2345.columns = ['subject', 'meanchoicevalue2345']

    # Merge and de-mean
    block_means = block_means.merge(blocks_2345, on='subject', how='left')
    block_means['demeaned_choicevalue'] = block_means['mean_choicevalue'] - block_means['meanchoicevalue2345']

    return block_means
