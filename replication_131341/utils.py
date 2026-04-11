"""Shared paths, variable lists, and helper functions for paper 131341."""
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parent
PKG = ROOT.parent / "131341-V1"
DATA = PKG / "data"
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)

EXPERIMENT_DTA = DATA / "cleandata_experiment.dta"

# Baseline controls from A_Master_Replication.do:
#   female age age_sq college northeast midwest south west i.occupation_1 high_conf
# Note: northeast/midwest/south/west sum to 1 for every respondent, so we drop
# "west" as the omitted category to avoid collinearity with the constant.
# Stata's `reg` silently drops a collinear regressor; we do it explicitly.
BASE_CONTROLS = [
    "female", "age", "age_sq", "college",
    "northeast", "midwest", "south",
    "high_conf",
]
# i.occupation_1 => dummies for occupation_1 (one omitted)
OCC_VAR = "occupation_1"

# Treatment & instrument-related regressors (all four always included):
INFO_REGS = ["Del_unempincr", "unempincr_alt", "Del_unempbase", "unempbase_alt"]

# First-stage / manipulation outcomes (Table 2 Panel A)
PANEL_A_OUTCOMES = [
    "unemp_nextrecession",
    "z_manip_1",
    "z_manip_2",
    "z_manip_3",
    "z_manip_ind",
]
# Forecast demand outcomes (Table 2 Panels B/C)
PANEL_BC_OUTCOMES = [
    "forecast_recession",
    "forecast_govspending",
    "forecast_iratebonds",
    "forecast_inflation",
    "othermacro",
    "noforecast",
]


def load_experiment():
    df = pd.read_stata(EXPERIMENT_DTA, convert_categoricals=False)
    return df


def build_design(df, extra=None, drop_na=None):
    """Return (y placeholder, X) with control matrix.

    Expands occupation_1 to dummies dropping the first category (Stata default).
    HC1 robust SEs are applied separately. extra: list of numeric cols to prepend
    before the base controls (treatment/instrument regressors).
    """
    extra = extra or []
    if drop_na is None:
        drop_na = list(extra) + BASE_CONTROLS + [OCC_VAR]
    d = df.dropna(subset=drop_na).copy()
    # occupation dummies — drop first to avoid collinearity with constant
    occ = pd.get_dummies(d[OCC_VAR].astype(int), prefix="occ", drop_first=True).astype(float)
    X = pd.concat(
        [d[extra].astype(float).reset_index(drop=True),
         d[BASE_CONTROLS].astype(float).reset_index(drop=True),
         occ.reset_index(drop=True)],
        axis=1,
    )
    X = sm.add_constant(X, has_constant="add")
    return d.reset_index(drop=True), X


def ols_hc1(y, X):
    return sm.OLS(y, X, missing="drop").fit(cov_type="HC1")


def iv_2sls(df, outcome, endog, instrument, exog_controls):
    """Run 2SLS via linearmodels with HC1 (robust) SEs."""
    from linearmodels.iv import IV2SLS
    d = df.dropna(subset=[outcome, endog, instrument] + exog_controls).copy()
    occ = pd.get_dummies(d[OCC_VAR].astype(int), prefix="occ", drop_first=True).astype(float)
    occ = occ.loc[:, occ.sum(axis=0) > 0]  # drop empty occupation cells
    exog = pd.concat(
        [d[exog_controls].astype(float).reset_index(drop=True),
         occ.reset_index(drop=True)],
        axis=1,
    )
    # drop any exog columns with zero variance (e.g., a collinear control)
    nz = exog.std() > 0
    exog = exog.loc[:, nz]
    exog = sm.add_constant(exog, has_constant="add")
    y = d[outcome].astype(float).reset_index(drop=True)
    endog_s = d[[endog]].astype(float).reset_index(drop=True)
    instr_s = d[[instrument]].astype(float).reset_index(drop=True)
    mod = IV2SLS(dependent=y, exog=exog, endog=endog_s, instruments=instr_s)
    res = mod.fit(cov_type="robust", debiased=True)
    return res


def fmt(x, d=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return f"{x:.{d}f}"
