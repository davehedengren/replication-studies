"""
Utility functions and shared paths for replication of:
Bednar & Gicheva (2014), "Are Female Supervisors More Female-Friendly?"
AER Papers & Proceedings, 104(5), 370-375.
"""

import os
import numpy as np
import pandas as pd

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', '112798-V1', 'Gicheva_Data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Data files
MAIN_DATA = os.path.join(DATA_DIR, 'p_and_p_data.dta')
EADA_DATA = os.path.join(DATA_DIR, 'p_and_p_EADAdata.dta')

# Constants from Stata code
NUM_ADS = 137       # Multi-school ADs
NUM_ADS_ALL = 433   # All ADs

# Key variables
OUTCOME_VAR = 'fsoc2'  # Share of female coaches (4 sports: basketball, soccer, softball, volleyball)


def load_main_data():
    """Load the main panel dataset."""
    df = pd.read_stata(MAIN_DATA, convert_categoricals=False)
    return df


def load_eada_data():
    """Load the EADA expenditure dataset."""
    df = pd.read_stata(EADA_DATA, convert_categoricals=False)
    return df


def get_ad_dum_cols(df, prefix='ad_dum', exclude_prefix='ad_dum_all'):
    """Get AD dummy column names (multi-school ADs only)."""
    return [c for c in df.columns if c.startswith(prefix) and not c.startswith(exclude_prefix)]


def get_ad_dum_all_cols(df):
    """Get all AD dummy column names."""
    return [c for c in df.columns if c.startswith('ad_dum_all')]


def get_year_dum_cols(df):
    """Get year dummy column names."""
    return [c for c in df.columns if c.startswith('year_dum')]


def get_school_dummies(df):
    """Create school fixed effects (dummies for i.school_id)."""
    dummies = pd.get_dummies(df['school_id'], prefix='school', drop_first=True).astype(float)
    return dummies


def print_separator(title):
    """Print a formatted section separator."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")
