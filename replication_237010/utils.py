"""
utils.py — Shared paths, loaders, and helpers for Gertler, Huckfeldt, Trigari (2025).

Paper: "Temporary Layoffs, Loss-of-Recall, and Cyclical Unemployment Dynamics"
Replication package: openICPSR 237010-V1
"""

import os
import re
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', '237010-V1')
CPS_RESULTS = os.path.join(DATA_DIR, 'empirics_CPS', 'results')
SIPP_TABLES = os.path.join(DATA_DIR, 'empirics_SIPP', 'tables')
SIPP_DATA = os.path.join(DATA_DIR, 'empirics_SIPP', 'final_data')
MODEL_DIR = os.path.join(DATA_DIR, 'model')
FINAL_OUTPUT = os.path.join(DATA_DIR, 'final_output')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════
# SED FILE PARSER
# ══════════════════════════════════════════════════════════════════════

def parse_sed_file(filepath):
    """Parse a sed substitution file into a dict of {varname: value_str}."""
    result = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            m = re.match(r's/\\<(\w+)\\>/\s*(.*?)\s*/g', line)
            if m:
                key = m.group(1)
                val = m.group(2).strip()
                result[key] = val
    return result


def sed_val(d, key, dtype=float):
    """Extract a numeric value from a parsed sed dict."""
    raw = d.get(key, None)
    if raw is None:
        return None
    # Remove parentheses (used for SEs) and commas
    raw = raw.replace('(', '').replace(')', '').replace(',', '').replace('%', '').replace('~', '').strip()
    try:
        return dtype(raw)
    except (ValueError, TypeError):
        return raw


# ══════════════════════════════════════════════════════════════════════
# DATA LOADERS
# ══════════════════════════════════════════════════════════════════════

def load_transition_probabilities():
    """Load monthly transition probabilities (CPS, 1978-2019)."""
    return pd.read_csv(os.path.join(CPS_RESULTS, 'transitionProbabilities.csv'))


def load_quarterly_transitions():
    """Load quarterly transition probabilities."""
    return pd.read_csv(os.path.join(CPS_RESULTS, 'quarterlyTransitionProbabilities.csv'))


def load_ght_stats():
    """Load GHT summary statistics (means, std, correlations)."""
    return pd.read_csv(os.path.join(CPS_RESULTS, 'GHT_stats.csv'))


def load_for_matlab():
    """Load data formatted for MATLAB (monthly 2019-2021)."""
    return pd.read_csv(os.path.join(CPS_RESULTS, 'forMatlabAdj.csv'))


def load_for_raw_plot():
    """Load raw (non-seasonally-adjusted) monthly data."""
    return pd.read_csv(os.path.join(CPS_RESULTS, 'forRawPlot.csv'))


def load_for_adjusted_plot():
    """Load seasonally adjusted monthly data."""
    return pd.read_csv(os.path.join(CPS_RESULTS, 'forAdjustedPlot.csv'))


def load_sipp_hazard():
    """Load SIPP exit hazard rates by duration."""
    return pd.read_csv(os.path.join(SIPP_DATA, 'hazard.csv'))


def load_sipp_hazard_panel():
    """Load SIPP hazard panel data."""
    return pd.read_csv(os.path.join(SIPP_DATA, 'hazard_panel.csv'))


def load_sipp_hazard_srefmon():
    """Load SIPP hazard by spell reference month."""
    path = os.path.join(SIPP_DATA, 'hazard_srefmonA.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def load_published_stats():
    """Load all pre-computed statistics from sed files."""
    stats = {}
    # CPS stats
    cps_path = os.path.join(CPS_RESULTS, 'stats.txt')
    if os.path.exists(cps_path):
        stats.update(parse_sed_file(cps_path))
    # Regression stats
    reg_path = os.path.join(CPS_RESULTS, 'regsFlows.txt')
    if os.path.exists(reg_path):
        stats.update(parse_sed_file(reg_path))
    # SIPP stats
    sipp_path = os.path.join(SIPP_TABLES, 'stats.txt')
    if os.path.exists(sipp_path):
        stats.update(parse_sed_file(sipp_path))
    # SIPP vs CPS comparison
    comp_path = os.path.join(SIPP_TABLES, 'SIPP_v_CPS.txt')
    if os.path.exists(comp_path):
        stats.update(parse_sed_file(comp_path))
    # Model estimates
    model_path = os.path.join(MODEL_DIR, 'logfiles', 'outputIS0_NC0.out')
    if os.path.exists(model_path):
        stats.update(parse_sed_file(model_path))
    # Pandemic estimates
    pandemic_path = os.path.join(MODEL_DIR, 'logfiles', 'pandemic_outputIS0_NC0.out')
    if os.path.exists(pandemic_path):
        stats.update(parse_sed_file(pandemic_path))
    return stats


# ══════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════

def fmt_p(val):
    """Format a probability or small number."""
    if val is None:
        return '---'
    if abs(val) < 0.001:
        return f'{val:.4f}'
    return f'{val:.3f}'
