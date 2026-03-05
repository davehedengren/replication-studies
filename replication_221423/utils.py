"""
utils.py — Paths and helpers for 221423-V1.

Paper: "Income Inequality in the Nordic Countries: Myths, Facts, and Lessons"
Authors: Mogstad, Salvanes, Torsvik (2025), AER
Original Language: Stata + R + Python
Replication Language: Python (pandas, matplotlib)
"""

import os
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), '221423-V1', 'data and code', 'data')
CODE_DIR = os.path.join(os.path.dirname(BASE_DIR), '221423-V1', 'data and code', 'code')
OUTPUTS_DIR = os.path.join(os.path.dirname(BASE_DIR), '221423-V1', 'data and code', 'outputs')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

TABLE1_DIR = os.path.join(DATA_DIR, 'table1')
TABLE3_DIR = os.path.join(DATA_DIR, 'table3')
FIGURE2_DIR = os.path.join(DATA_DIR, 'figure2')
FIGURE3_DIR = os.path.join(DATA_DIR, 'figure3')

# ── Country codes ──────────────────────────────────────────────────────
NORDIC_CODES = {'DNK': 208, 'FIN': 246, 'NOR': 578, 'SWE': 752}
REPORT_CODES = {'DNK': 'Denmark', 'FIN': 'Finland', 'NOR': 'Norway', 'SWE': 'Sweden',
                'GBR': 'United Kingdom', 'USA': 'United States', 'OECD': 'OECD'}

COUNTRY_ORDER = ['Nordic Average', 'Denmark', 'Finland', 'Norway', 'Sweden',
                 'United Kingdom', 'United States', 'OECD Average']
