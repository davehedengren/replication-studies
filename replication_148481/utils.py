"""Shared paths and helpers for the DellaVigna, List, Malmendier, Rao (2022)
replication of Experiment 1."""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EXP1 = ROOT / "148481-V1" / "Experiment1RepKit"
RAW_CSV = EXP1 / "insheetdata" / "Gift ExchangeFinalDataFromGS copy.csv"
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)

GIFT_TYPES = ["neu", "pos", "neg", "knd"]


def load_raw():
    return pd.read_csv(RAW_CSV)


def clean_wide():
    """Translate 1_data_clean.do through the wide save.

    Returns (wide, num_ind) — mimics dropping pilot + missing-output rows until
    N=446 workers remain.
    """
    df = load_raw()
    # First three chars of gift_type (pos/neg/neu/in-)
    df["gift_type"] = df["gift_type"].astype(str).str.slice(0, 3)
    df.loc[df["gift_type"] == "in-", "gift_type"] = "knd"

    df["date2"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce")
    df["phase"] = np.where(
        df["date2"] < "2013-10-27", 0,
        np.where(df["date2"] < "2014-11-22", 1, 2),
    )
    df = df[df["phase"] != 0].copy()

    # Destring the s*_output columns so non-numeric ("") becomes NaN, drop any NA.
    for i in range(1, 11):
        col = f"s{i}_output"
        df[f"output{i}"] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[f"output{i}" for i in range(1, 11)]).copy()
    df = df.reset_index(drop=True)
    df["id"] = np.arange(1, len(df) + 1)

    df["order1"] = (df["order"] == 1).astype(int)
    df["order2"] = (df["order"] == 2).astype(int)
    for g in GIFT_TYPES:
        df[f"d_{g}_gift"] = (df["gift_type"] == g).astype(int)
    df["match_f"] = (df["match_order"] == "match first").astype(int)
    df["match_l"] = (df["match_order"] == "match last").astype(int)

    # Demographics for balance
    for col in ["employed", "donated", "volunteered"]:
        df[col] = df[col].astype(str).str.strip().str.title()
    df["dempl"] = df["employed"].isin(["Yes", "Yes (Self)"]).astype(int)
    df["female"] = (df["gender"] == "f").astype(int)
    df["age_num"] = pd.to_numeric(df["age"], errors="coerce")
    df["age_num"] = df["age_num"].fillna(33)
    df["donate"] = (df["donated"] == "Yes").astype(int)
    df["dvolunteer"] = (df["volunteered"] == "Yes").astype(int)

    return df


def clean_long():
    """Replicates 1_data_clean.do reshape and generated long-panel variables."""
    wide = clean_wide()
    num_ind = len(wide)
    id_cols = [
        "id", "order", "gift_type", "charity_order", "match_f", "match_l",
        "order1", "order2", "d_neu_gift", "d_pos_gift", "d_neg_gift", "d_knd_gift",
        "group", "phase", "date2", "female", "age_num", "dempl", "donate",
        "dvolunteer",
    ]
    long = wide.melt(
        id_vars=id_cols,
        value_vars=[f"output{i}" for i in range(1, 11)],
        var_name="time",
        value_name="output",
    )
    long["time"] = long["time"].str.replace("output", "").astype(int)
    long = long.dropna(subset=["output"]).copy()
    long = long.sort_values(["order", "time"]).reset_index(drop=True)

    long["lunch"] = (long["time"] > 4).astype(int)
    long["TR"] = ((long["time"] == 1) | (long["time"] == 5)).astype(int)
    long["d_firm"] = (
        ((long["time"] == 4) & (long["order"] == 1))
        | ((long["time"] == 6) & (long["order"] == 2))
    ).astype(int)
    long["d_charity"] = ((long["TR"] == 0) & (long["d_firm"] == 0)).astype(int)
    long["room"] = (long["time"] > 8).astype(int)

    # matchgift_on: whether the match is active in round 9 or 10
    long["matchgift_on"] = np.nan
    cond_on = (
        ((long["time"] == 9) & (long["match_f"] == 1))
        | ((long["time"] == 10) & (long["match_l"] == 1))
    )
    cond_off = (
        ((long["time"] == 10) & (long["match_f"] == 1))
        | ((long["time"] == 9) & (long["match_l"] == 1))
    )
    long.loc[cond_on, "matchgift_on"] = 1
    long.loc[cond_off, "matchgift_on"] = 0

    long["alpha_pos"] = ((long["time"] > 8) & (long["gift_type"] == "pos")).astype(int)
    long["alpha_neg"] = ((long["time"] > 8) & (long["gift_type"] == "neg")).astype(int)

    match = (
        (long["time"] == 2)
        | (long["time"] == 8)
        | ((long["time"] == 9) & (long["match_f"] == 1))
        | ((long["time"] == 10) & (long["match_l"] == 1))
    )
    long["match"] = match.astype(int)

    # Per-round price to firm / worker (not needed for reduced form but replicated)
    long["Pf"] = 0.3
    long.loc[long["TR"] == 1, "Pf"] = 0.0
    long.loc[long["match"] == 1, "Pf"] = 0.6

    long["Pw"] = 0.0
    long.loc[(long["time"] < 5) & (long["order"] == 1), "Pw"] = 0.1
    long.loc[(long["time"] >= 5) & (long["time"] <= 6) & (long["order"] == 1), "Pw"] = 0.2
    long.loc[(long["time"] >= 5) & (long["time"] <= 8) & (long["order"] == 2), "Pw"] = 0.1
    long.loc[((long["time"] == 1) | (long["time"] == 4)) & (long["order"] == 2), "Pw"] = 0.2

    long["lnoutput"] = np.log(long["output"])
    # time dummies
    for t in range(1, 11):
        long[f"d_time{t}"] = (long["time"] == t).astype(int)

    return long, num_ind


def cluster_se_ols(y, X, groups):
    """OLS with cluster-robust SE (Stata cluster() — G/(G-1)*(N-1)/(N-k) multiplier).

    Returns dict with 'params', 'se', 'tvals', 'pvals', 'ci_low', 'ci_high',
    'N', 'G', 'r2'.
    """
    import statsmodels.api as sm
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    mod = sm.OLS(y, X)
    res = mod.fit(cov_type="cluster", cov_kwds={"groups": groups})
    return res


def ibn_regression(values, categories, groups):
    """Replicates Stata `reg y ibn.type, nocons cluster(group)` — a set of
    group-mean estimates with cluster-robust SE and 95% CIs."""
    import statsmodels.api as sm
    cats = sorted(pd.unique(categories))
    X = np.column_stack([(categories == c).astype(float) for c in cats])
    res = sm.OLS(values, X).fit(cov_type="cluster", cov_kwds={"groups": groups})
    out = pd.DataFrame({
        "type": cats,
        "mean": res.params,
        "se": res.bse,
    })
    # 95% CI using normal approx (matches Stata's parmest default with robust SEs)
    out["ci_low"] = out["mean"] - 1.96 * out["se"]
    out["ci_high"] = out["mean"] + 1.96 * out["se"]
    return out, res


def cluster_ttest(values, group_labels, cluster_ids):
    """Two-sample comparison with cluster-robust SEs — replicates Stata's
    cltest ... by(type).  Uses an OLS with an intercept and a group dummy and
    reads off the p-value for the dummy."""
    import statsmodels.api as sm
    group_labels = np.asarray(group_labels)
    vals = np.asarray(values, dtype=float)
    cats = np.unique(group_labels)
    if len(cats) != 2:
        raise ValueError("need exactly two groups")
    X = np.column_stack([np.ones_like(vals), (group_labels == cats[1]).astype(float)])
    res = sm.OLS(vals, X).fit(cov_type="cluster", cov_kwds={"groups": cluster_ids})
    return res.pvalues[1], res
