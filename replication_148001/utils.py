"""Shared helpers for the Mehmood (2021) replication.

Paper: "The impact of Presidential appointment of judges: Montesquieu or
the Federalists?" (Sultan Mehmood, AMSE WP 2021-18 / HAL-03161933).
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "148001-V1" / "REPLICATION" / "Input"
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)

DATA1 = INPUT / "DataTeam1.dta"
DATA2 = INPUT / "DataTeam2.dta"

# Control sets (from Master.do globals)
CONTROLS_CASE = [
    "constitutional", "criminal", "land_case",
    "pagesjudgenum", "benchchiefjustice", "lawyer_number", "judge_number",
]
CONTROLS_JUDGE = [
    "TenureatDecision", "Gender", "Plot_Alloted",
    "ElevatedtoSupremeCourt", "Former_LowerCourtJudge",
    "Former_MemberBarAssociation", "Prior_Political_Office",
    "Former_Lawyer", "Vote_share", "Muslim",
    "Punjabi", "Sindhi", "Balochi", "Pashtun", "Others",
]
CONTROLS_DISTRICT = [
    "Area_sqkm", "Population", "Density_persqkm",
    "Agri_production", "avg_mean",
]
CONTROLS_BENCH = [
    "TotalJudges_bench", "Criminal_count_bench",
    "Land_count_per_bench", "Human_count_per_bench",
]


def load_team1():
    return pd.read_stata(DATA1, convert_categoricals=False)


def fit_ols_cluster(y, X, cluster, absorb_col=None):
    """Run OLS with district-bench clustered SEs, optionally absorbing a FE.

    Uses pyhdfe if absorb_col is provided, otherwise plain statsmodels OLS
    (user must already have added dummies). Returns (coef, se, n, r2).
    """
    import statsmodels.api as sm

    if absorb_col is not None:
        import pyhdfe
        algo = pyhdfe.create(absorb_col.values.reshape(-1, 1), drop_singletons=False)
        yX = np.column_stack([y, X])
        yXr = algo.residualize(yX)
        yr = yXr[:, 0]
        Xr = yXr[:, 1:]
        # fit without constant; then correct df for absorbed groups
        model = sm.OLS(yr, Xr)
        res = model.fit(
            cov_type="cluster",
            cov_kwds={"groups": cluster.values},
        )
        return res
    else:
        X = sm.add_constant(X, has_constant="add")
        res = sm.OLS(y, X).fit(
            cov_type="cluster", cov_kwds={"groups": cluster.values}
        )
        return res


def main_sample_mask(df):
    """Return boolean mask matching the Stata `e(sample)` for Table 2 col 2."""
    needed = (
        ["StateWins", "Retirements_2010_Post2010", "yeardecisionXdistrict_encode",
         "districtXbench_enco"]
        + CONTROLS_CASE + CONTROLS_JUDGE + CONTROLS_DISTRICT
    )
    return df[needed].notna().all(axis=1)
