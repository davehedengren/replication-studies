"""
Replication of: Entrepreneurial Innovation: Killer Apps in the iPhone Ecosystem
Authors: Pai-Ling Yin, Jason P. Davis, Yulia Muzyrya
Paper ID: 112783
"""

import os
import pandas as pd
import numpy as np

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', '112783-V1', 'P2014_1175_data')
DATA_FILE = os.path.join(DATA_DIR, 'yindavismuzyryadata.dta')

# Variable lists
DEPVAR = 'killerappgros'

# Regressors (order matches Stata code)
REGRESSORS = ['noupdates', 'countapp', 'numverapp', 'avdeltatime',
              'avprice', 'avsize', 'lnnumcomapp', 'scoreapp']

# Display names for tables (maps code var -> paper label)
VAR_LABELS = {
    'countapp': 'App Order',
    'numverapp': 'Number of Versions',
    'noupdates': 'No Updates',
    'avdeltatime': 'Time Between Versions',
    'avprice': 'Price',
    'avsize': 'Size',
    'numcomapp': 'Number of Comments',
    'lnnumcomapp': 'Log Number of Comments',
    'scoreapp': 'Score',
}

# Table 1 variables (raw, not log-transformed)
TABLE1_VARS = ['countapp', 'numverapp', 'noupdates', 'avdeltatime',
               'avprice', 'avsize', 'numcomapp', 'scoreapp']

# Cohort dummies used in Stata (drops cohort2 and cohort18 as reference)
COHORT_DUMMIES = [f'cohort{i}' for i in range(1, 39) if i not in (2, 18)]

# Category dummies for non-games model (drops cat2 and cat5 as reference)
CAT_DUMMIES_NONGAMES = [f'cat{i}' for i in range(1, 19) if i not in (2, 5)]


def load_data():
    """Load the dataset."""
    df = pd.read_stata(DATA_FILE, convert_categoricals=False)
    return df
