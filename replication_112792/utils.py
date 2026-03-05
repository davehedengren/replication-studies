"""
Shared utilities for replication of paper 112792:
"Disability Insurance and Health Insurance Reform: Evidence from Massachusetts"
Maestas, Mullen, and Strand (2014)
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

# ── paths ──
BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "..", "112792-V1", "P2014_1138_data")
STATE_DTA = os.path.join(DATA, "MA_state_data.dta")
COUNTY_DTA = os.path.join(DATA, "MA_county_data.dta")
OUT = os.path.join(BASE, "output")
os.makedirs(OUT, exist_ok=True)

# ── variable lists ──
APP_VARS_STATE = ["allapps", "DIonly", "SSItotal", "SSDItotal"]
APP_VARS_COUNTY = ["allapps", "DIonly", "SSItotal", "SSDItotal"]
TIME_VARS = ["DIonly_mntime", "SSItotal_mntime"]

# Counties to drop (any zero in DIonly/SSIonly/concurrent → suppressed cells)
DROP_COUNTIES = [
    23015, 23021, 23023, 25007, 25019, 33003, 33007, 33019,
    34019, 36041, 36049, 36095, 36097, 36099, 36123,
    42023, 42053, 42057, 42067, 42093, 42099, 42105, 42113, 42119,
    50001, 50005, 50009, 50013, 50015, 50017,
]


def q_fld_to_int(series):
    """Convert q_fld (datetime from Stata quarterly) to Stata quarter number.
    Stata quarterly: integer = (year-1960)*4 + (quarter-1)."""
    dt = pd.to_datetime(series)
    return (dt.dt.year - 1960) * 4 + (dt.dt.quarter - 1)


def load_state_data():
    """Load state-level data and create analysis variables."""
    df = pd.read_stata(STATE_DTA, convert_categoricals=False)
    # Convert q_fld datetime to Stata quarter integer
    df["q_fld"] = q_fld_to_int(df["q_fld"])
    # Stata: gen MA = state=="MA"
    df["MA"] = (df["state"] == "MA").astype(int)
    # quarter numeric (Stata qofd(dofq(q_fld)))
    df["qnum"] = df["q_fld"].astype(int)
    # Post-period indicators (fiscal years: FY2007=q187-190, FY2008=q191-194, FY2009=q195-198)
    df["post1"] = ((df["qnum"] >= 187) & (df["qnum"] <= 190)).astype(int)
    df["post2"] = ((df["qnum"] >= 191) & (df["qnum"] <= 194)).astype(int)
    df["post3"] = ((df["qnum"] >= 195) & (df["qnum"] <= 198)).astype(int)
    # Interaction terms
    df["MAXpost1"] = df["MA"] * df["post1"]
    df["MAXpost2"] = df["MA"] * df["post2"]
    df["MAXpost3"] = df["MA"] * df["post3"]
    # State dummies (drop MA = state 2 per Stata code: stnum1 stnum3-stnum9)
    states = sorted(df["state"].unique())
    for i, s in enumerate(states, 1):
        df[f"stnum{i}"] = (df["state"] == s).astype(int)
    return df


def load_county_data():
    """Load county-level data and create analysis variables."""
    df = pd.read_stata(COUNTY_DTA, convert_categoricals=False)
    # Convert q_fld datetime to Stata quarter integer
    df["q_fld"] = q_fld_to_int(df["q_fld"])
    # Drop counties with suppressed cells
    df = df[~df["county"].isin(DROP_COUNTIES)].copy()
    df["MA"] = (df["state"] == "MA").astype(int)
    # Post-period indicators (county data only goes to 2008q3, so 2 post periods)
    df["post1"] = ((df["q_fld"] >= 187) & (df["q_fld"] <= 190)).astype(int)
    df["post2"] = ((df["q_fld"] >= 191) & (df["q_fld"] <= 194)).astype(int)
    df["MAXpost1"] = df["MA"] * df["post1"]
    df["MAXpost2"] = df["MA"] * df["post2"]
    # Low health insurance indicator: nohi05 >= 0.12
    df["lowHI"] = (df["nohi05"] >= 0.12).astype(int)
    # Quarter dummies
    q_vals = sorted(df["q_fld"].unique())
    for i, q in enumerate(q_vals, 1):
        df[f"qnum{i}"] = (df["q_fld"] == q).astype(int)
    return df, q_vals


def run_wls(df, depvar, xvars, weight_col="wapop", cluster_col="state"):
    """
    Run WLS with analytic weights and clustered SEs.
    Stata: reg y x [aw=wapop], cluster(state)
    Analytic weights in Stata = WLS with weights = wapop.
    """
    subset = df.dropna(subset=[depvar] + xvars + [weight_col]).copy()
    y = subset[depvar].astype(float)
    X = subset[xvars].astype(float)
    X = sm.add_constant(X)
    w = subset[weight_col].astype(float)

    mod = sm.WLS(y, X, weights=w)
    # Cluster at state level
    groups = subset[cluster_col]
    res = mod.fit(cov_type="cluster", cov_kwds={"groups": groups})
    return res, subset


def print_did_results(res, interaction_vars, label=""):
    """Print DiD coefficients nicely."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    for v in interaction_vars:
        coef = res.params[v]
        se = res.bse[v]
        pval = res.pvalues[v]
        stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
        print(f"  {v:12s}  coef={coef:9.4f}  se=({se:.4f})  p={pval:.3f} {stars}")
    print(f"  N = {int(res.nobs)}")
