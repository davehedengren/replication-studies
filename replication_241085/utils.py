"""
utils.py — Shared paths and loaders for Prager (2026) replication.

Paper: "Antitrust Enforcement in Labor Markets" (JEP 2026)
Replication package: openICPSR 241085-V1
"""

import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', '241085-V1',
                        'Schubert Stansbury Taska - Occ Transitions Public Data Set (Jan 2021)')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_transitions():
    """Load occupation transition data from Schubert, Stansbury, Taska (2024)."""
    path = os.path.join(DATA_DIR, 'occupation_transitions_public_data_set.dta')
    df = pd.read_stata(path, convert_categoricals=False)
    return df


def load_soc_hierarchy():
    """Load SOC code hierarchy/crosswalk."""
    path = os.path.join(DATA_DIR, 'soc_hierarchy.dta')
    df = pd.read_stata(path, convert_categoricals=False)
    return df
