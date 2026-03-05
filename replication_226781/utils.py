"""
utils.py — Paths, data loaders, and helpers for 226781-V1.

Paper: "Trade, Value Added, and Productivity Linkages"
Authors: de Soyres & Gaillard (2025), AEJ: Macroeconomics
"""

import os
import numpy as np
import pandas as pd
import pyreadr

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), '226781-V1')
EMPIRICS_DIR = os.path.join(DATA_DIR, 'empirics_replication', 'datasets')
TRADE_DIR = os.path.join(EMPIRICS_DIR, 'create_data_TRADE_COMOVEMENT')
PROFIT_DIR = os.path.join(TRADE_DIR, 'profit_comovement')
EIM_DIR = os.path.join(EMPIRICS_DIR, 'create_EM_IM_margins')
MARKUP_DIR = os.path.join(EMPIRICS_DIR, 'create_data_MARKUPS_corrTOTGDP')
MODEL_DIR = os.path.join(DATA_DIR, 'model_replication', 'code')
FIG_DIR = os.path.join(DATA_DIR, 'figures_tables')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

COUNTRY_LIST = [
    "ARG", "AUS", "AUT", "BEL", "BRA", "CAN", "CHE", "CHL", "CHN", "DEU",
    "DNK", "ESP", "FRA", "GBR", "GRC", "IND", "IRL", "ITA", "JPN", "KOR",
    "MEX", "NLD", "NOR", "NZL", "POL", "PRT", "SWE", "THA", "TUR", "USA"
]

EXCLUDE_COUNTRIES = ["VNM", "ZAF", "ISR", "IDN", "FIN"]


# ── Data loaders ───────────────────────────────────────────────────────
def load_rdata(filepath, key=None):
    """Load an RData file and return the DataFrame."""
    result = pyreadr.read_r(filepath)
    if key:
        return result[key]
    return list(result.values())[0]


def load_tcp_10():
    """Main trade-comovement dataset: 10-year windows, country pairs."""
    return load_rdata(os.path.join(TRADE_DIR, 'TCP_10_third.RData'), 'TCP_10_third')


def load_tcp_profit():
    """Profit comovement dataset."""
    return load_rdata(os.path.join(PROFIT_DIR, 'TCP_PROFIT.RData'), 'TCP_PROFIT')


def load_eim():
    """Extensive/Intensive margin dataset."""
    return load_rdata(os.path.join(EIM_DIR, 'EIM_SITC_2.RData'), 'EIM_SITC_2')


def load_markup_tot():
    """Markup-ToT-GDP correlation dataset."""
    return load_rdata(os.path.join(MARKUP_DIR, 'data_TOTGDP_markup.RData'), 'data_TOTGDP_markup')


def load_trade_va():
    """Johnson-Noguera value-added trade dataset."""
    return pd.read_stata(os.path.join(TRADE_DIR, 'VAdataset.dta'))


def load_sr_data():
    """Solow residual correlation data."""
    return pd.read_csv(os.path.join(TRADE_DIR, 'SR3_TCP_final_data_full.csv'))


def load_pwt():
    """Penn World Table 10.0."""
    return pd.read_stata(os.path.join(TRADE_DIR, 'pwt100.dta'))


# ── Regression helpers ────────────────────────────────────────────────
def feols(df, yvar, xvars, fe_var, cluster_var):
    """Fixed-effects OLS with clustered SEs (equivalent to R's fixest::feols).

    Uses linearmodels.AbsorbingLS.
    """
    from linearmodels.iv import AbsorbingLS

    d = df.copy()
    all_vars = [yvar] + xvars + [fe_var, cluster_var]
    d = d.dropna(subset=[v for v in all_vars if v in d.columns])

    y = d[yvar]
    X = d[xvars].astype(float)
    absorb = pd.DataFrame({fe_var: pd.Categorical(d[fe_var].astype(str))})
    clusters = d[cluster_var]

    model = AbsorbingLS(y, X, absorb=absorb, drop_absorbed=True)
    result = model.fit(cov_type='clustered', clusters=clusters)

    return {
        'params': result.params,
        'se': result.std_errors,
        'pvalues': result.pvalues,
        'nobs': int(result.nobs),
        'r2': result.rsquared,
        'r2_within': result.rsquared_within if hasattr(result, 'rsquared_within') else None,
        'result': result,
    }


def format_coef(beta, se, pval):
    """Format coefficient with significance stars."""
    stars = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.1 else ''))
    return f'{beta:.4f}{stars} ({se:.4f})'
