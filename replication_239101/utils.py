"""Shared helpers for replication of Evdokimov & Garfagnini,
"Cognitive Ability and Perceived Disagreement in Learning"."""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = ROOT / "239101-V1" / "R_project" / "data" / "data_cognition.csv"

P_PRIOR = 0.5
Q_ORANGE = 2 / 3
Q_PURPLE = 1 / 3
N_PERIODS = 30


def bayes_posterior(n_orange: np.ndarray, n: np.ndarray,
                    p: float = P_PRIOR, qO: float = Q_ORANGE,
                    qP: float = Q_PURPLE) -> np.ndarray:
    """Posterior probability of the Orange urn given nO orange balls in n draws."""
    num = p * qO ** n_orange * (1 - qO) ** (n - n_orange)
    denom = num + (1 - p) * qP ** n_orange * (1 - qP) ** (n - n_orange)
    return num / denom


def load_raw() -> pd.DataFrame:
    return pd.read_csv(DATA_CSV)


def build_analysis_df() -> pd.DataFrame:
    """Replicate the variable construction in analysis_manuscript.R."""
    d = load_raw().copy()

    d["baseline"] = (d["informed"] == 0).astype(int)
    d["informed_top"] = ((d["informed"] == 1) & (d["partner_type"] == "Top")).astype(int)
    d["informed_bottom"] = ((d["informed"] == 1) & (d["partner_type"] == "Bottom")).astype(int)
    d["informed_own"] = (d["informed"] == 2).astype(int)

    d["treatment"] = np.select(
        [d["baseline"] == 1, d["informed_top"] == 1, d["informed_bottom"] == 1],
        [1, 2, 3],
        default=4,
    )
    name_map = {1: "Baseline", 2: "InformedTop", 3: "InformedBottom", 4: "InformedOwn"}
    d["treatment_name"] = d["treatment"].map(name_map)
    d["main_treatments"] = (
        (d["baseline"] == 1) | (d["informed_bottom"] == 1) | (d["informed_top"] == 1)
    ).astype(int)

    d = d.sort_values(["id", "period"]).reset_index(drop=True)
    d["cumOrangeBalls"] = d.groupby("id")["orange"].cumsum()

    d["bayesianBeliefs"] = bayes_posterior(
        d["cumOrangeBalls"].to_numpy(), d["period"].to_numpy()
    )
    d["bayesianBeliefsTruth"] = np.where(
        d["urn"] == 1, d["bayesianBeliefs"], 1 - d["bayesianBeliefs"]
    )
    d["dist_bayes"] = (d["guess1"] - d["bayesianBeliefs"]).abs()
    d["guess1_truth"] = np.where(d["urn"] == 1, d["guess1"], 1 - d["guess1"])

    d["pdis"] = (d["guess1"] - d["guess2"]).abs()
    d["pdis_dum"] = (d["pdis"] > 0).astype(int)

    dir_pdis = np.where(
        d["guess1"] > 0.5,
        d["guess1"] - d["guess2"],
        np.where(d["guess1"] < 0.5, d["guess2"] - d["guess1"], np.nan),
    )
    d["dir_pdis"] = dir_pdis

    # Partner match: within (session, group, period), role 1 ↔ role 2.
    partner = d[["session", "group", "period", "role", "guess1", "test_score"]].copy()
    partner["role"] = np.where(partner["role"] == 1, 2, 1)
    partner = partner.rename(columns={"guess1": "guess_partner", "test_score": "p_test_score"})
    d = d.merge(partner, on=["session", "group", "period", "role"], how="left")

    d["adis"] = (d["guess1"] - d["guess_partner"]).abs()
    d["ts_gap"] = (d["test_score"] - d["p_test_score"]).abs()

    subj = (
        d.groupby("id")
        .agg(
            mean_pdis=("pdis", "mean"),
            mean_adis=("adis", "mean"),
            mean_dirpdis=("dir_pdis", "mean"),
        )
        .reset_index()
    )
    d = d.merge(subj, on="id", how="left")

    d["payoff_plus_fee"] = d["payoff"] + 0.05
    d["hourly_wage"] = d["payoff_plus_fee"] * 3600 / d["total_time"]

    return d


def cluster_ols(formula: str, data: pd.DataFrame, cluster_col: str = "id"):
    """Fit OLS with cluster-robust SE by `cluster_col`.
    Returns (result, n_obs, n_clusters).
    """
    from patsy import dmatrices

    y, X = dmatrices(formula, data=data, return_type="dataframe")
    use = y.index
    clusters = data.loc[use, cluster_col]
    model = sm.OLS(y, X)
    res = model.fit(cov_type="cluster", cov_kwds={"groups": clusters.values})
    return res, int(res.nobs), int(clusters.nunique())


def fmt_coef(res, name: str) -> str:
    c = res.params[name]
    se = res.bse[name]
    p = res.pvalues[name]
    return f"{c: .4f} (SE {se:.4f}, p={p:.4g})"
