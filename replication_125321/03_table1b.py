"""Replicate Table 1 Panel B: Event-study DiD around automation deadlines.
Equation (2): P_ict = sum_tau gamma_tau * Auto_tau + beta*W + alpha_i + month_t."""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from utils import OUT, WEATHER_VARS

EVENT_VARS = [
    "treat_m712_before2", "treat_m56_before2", "treat_m34_before2",
    "treat_m12_after2", "treat_m34_after2", "treat_m56_after2", "treat_m712_after2",
]

PUB_B = {
    "(1) Deadline, month FE": {
        "treat_m712_before2": -8.5,
        "treat_m56_before2":   6.8,
        "treat_m34_before2":  -6.4,
        "treat_m12_after2":   60.3,
        "treat_m34_after2":   45.0,
        "treat_m56_after2":   28.1,
        "treat_m712_after2":  40.0,
        "n": 176426, "r2": 0.34,
    },
    "(2) Deadline, year-month FE": {
        "treat_m712_before2": -17.2,
        "treat_m56_before2":  -19.2,
        "treat_m34_before2":  -12.0,
        "treat_m12_after2":    31.4,
        "treat_m34_after2":    33.6,
        "treat_m56_after2":    22.2,
        "treat_m712_after2":    9.8,
        "n": 176426, "r2": 0.35,
    },
    "(3) +Matching, month FE": {
        "treat_m712_before2": -10.7,
        "treat_m56_before2":   10.5,
        "treat_m34_before2":   -2.8,
        "treat_m12_after2":   66.5,
        "treat_m34_after2":   47.2,
        "treat_m56_after2":   33.4,
        "treat_m712_after2":  42.9,
        "n": 186499, "r2": 0.33,
    },
    "(4) +Matching, year-month FE": {
        "treat_m712_before2": -10.8,
        "treat_m56_before2":   -2.2,
        "treat_m34_before2":   -5.2,
        "treat_m12_after2":   45.6,
        "treat_m34_after2":   32.5,
        "treat_m56_after2":   29.0,
        "treat_m712_after2":  15.8,
        "n": 186499, "r2": 0.34,
    },
    "(5) +Matching, log PM10, year-month FE": {
        "treat_m712_before2": -0.13,
        "treat_m56_before2":   0.02,
        "treat_m34_before2":  -0.03,
        "treat_m12_after2":    0.24,
        "treat_m34_after2":    0.32,
        "treat_m56_after2":    0.29,
        "treat_m712_after2":   0.24,
        "n": 186469, "r2": 0.38,
    },
}


def build_event_vars(df):
    df = df.copy()
    df["treat"] = (df["auto_date"] == 19359).astype(int)
    df["m12_before2"] = ((df.year == 2012) & df.month.between(11, 12)).astype(int)
    df["m34_before2"] = ((df.year == 2012) & df.month.between(9, 10)).astype(int)
    df["m56_before2"] = ((df.year == 2012) & df.month.between(7, 8)).astype(int)
    df["m712_before2"] = ((df.year == 2012) & df.month.between(1, 6)).astype(int)
    df["m12_after2"] = ((df.year == 2013) & df.month.between(1, 2)).astype(int)
    df["m34_after2"] = ((df.year == 2013) & df.month.between(3, 4)).astype(int)
    df["m56_after2"] = ((df.year == 2013) & df.month.between(5, 6)).astype(int)
    df["m712_after2"] = ((df.year == 2013) & df.month.between(7, 12)).astype(int)
    for v in ["12", "34", "56", "712"]:
        df[f"treat_m{v}_before2"] = df["treat"] * df[f"m{v}_before2"]
        df[f"treat_m{v}_after2"] = df["treat"] * df[f"m{v}_after2"]
    df["year_month"] = df["year"] * 100 + df["month"]
    return df


def absorb(df, y, xs, absorb_cols):
    """Two-way within transform: iteratively subtract group means for each FE
    dimension until convergence. Returns demeaned y and X."""
    sub = df[[y] + xs + absorb_cols].dropna().copy()
    for _ in range(50):
        maxchg = 0.0
        for g in absorb_cols:
            for c in [y] + xs:
                m = sub.groupby(g)[c].transform("mean")
                newc = sub[c] - m
                maxchg = max(maxchg, (newc - sub[c]).abs().max())
                sub[c] = newc
        if maxchg < 1e-8:
            break
    return sub


def cluster_se(model_res, cluster):
    """Cluster-robust SE (CR1) — Stata-style DOF correction."""
    u = model_res.resid.to_numpy()
    X = model_res.model.exog
    N, K = X.shape
    G = cluster.nunique()
    bread = np.linalg.inv(X.T @ X)
    meat = np.zeros((K, K))
    for g, idx in cluster.groupby(cluster).groups.items():
        Xg = X[cluster.index.get_indexer(idx)]
        ug = u[cluster.index.get_indexer(idx)]
        Xu = Xg.T @ ug
        meat += np.outer(Xu, Xu)
    dof_adj = (G / (G - 1)) * ((N - 1) / (N - K))
    V = dof_adj * bread @ meat @ bread
    return np.sqrt(np.diag(V))


def run_event(df, y, absorb_cols, label):
    d = build_event_vars(df)
    controls = EVENT_VARS + WEATHER_VARS
    sub = d[[y] + controls + absorb_cols + ["code_city"]].dropna().copy()
    # Within-demean on absorb dimensions
    d2 = sub[[y] + controls + absorb_cols].copy()
    for _ in range(60):
        prev = d2[[y] + controls].to_numpy().copy()
        for g in absorb_cols:
            means = d2.groupby(g)[[y] + controls].transform("mean")
            d2[[y] + controls] = d2[[y] + controls] - means
        if np.abs(d2[[y] + controls].to_numpy() - prev).max() < 1e-8:
            break
    Y = d2[y].to_numpy()
    X = d2[controls].to_numpy()
    # No constant (absorbed). Degrees-of-freedom adjustment for absorbed FE handled ad-hoc
    model = sm.OLS(Y, X)
    res = model.fit()
    # cluster SE
    resid = Y - X @ res.params
    groups = sub["code_city"].reset_index(drop=True)
    N, K = X.shape
    G = groups.nunique()
    bread = np.linalg.inv(X.T @ X)
    meat = np.zeros((K, K))
    for _, idx in groups.groupby(groups).groups.items():
        ix = idx.values
        Xg = X[ix]
        ug = resid[ix]
        s = Xg.T @ ug
        meat += np.outer(s, s)
    dof_adj = (G / (G - 1)) * ((N - 1) / (N - K))
    V = dof_adj * bread @ meat @ bread
    ses = np.sqrt(np.diag(V))

    # Compute R2 against the original (un-demeaned) y: use fitted values including FE
    coefs = dict(zip(controls, res.params))
    ses_map = dict(zip(controls, ses))

    total_ss = ((sub[y] - sub[y].mean()) ** 2).sum()
    resid_ss = (resid ** 2).sum()
    r2_within = 1 - resid_ss / ((Y - Y.mean()) ** 2).sum()

    return {
        "label": label,
        "coefs": coefs,
        "ses": ses_map,
        "n": N,
        "r2_within": r2_within,
    }


def print_compare(results):
    col_labels = [r["label"] for r in results]
    print(f"{'var':<22} " + " ".join(f"{c[:18]:>22}" for c in col_labels))
    for v in EVENT_VARS:
        row = f"{v:<22} "
        for r in results:
            pub = PUB_B[r["label"]][v]
            est = r["coefs"][v]
            se = r["ses"][v]
            tag = "OK" if abs(est - pub) < max(2.0, 0.2 * abs(pub)) else "!"
            row += f" {est:7.2f} ({se:5.2f})[{tag}]"
        print(row)
    row = f"{'N':<22} "
    for r in results:
        pub = PUB_B[r["label"]]["n"]
        row += f" {r['n']:>18,d}     "
    print(row)


def main():
    print("=== Table 1 Panel B: Event-Study ===")

    # Cols 1-2: deadline sample from station_day_1116
    sd = pd.read_parquet(OUT / "station_day_rd.parquet")
    ddl = sd[(sd.year.isin([2012, 2013])) & (sd.auto_date.isin([19359, 19724]))].copy()
    print(f"Deadline sample rows: {len(ddl):,}")
    c1 = run_event(ddl, "pm10", ["pm10_n", "month"], "(1) Deadline, month FE")
    c2 = run_event(ddl, "pm10", ["pm10_n", "year_month"], "(2) Deadline, year-month FE")

    # Cols 3-5: matched DiD sample
    did = pd.read_parquet(OUT / "did_ddl_match.parquet")
    did["month"] = pd.to_datetime(did["date"], unit="D", origin="1960-01-01").dt.month
    did["year"] = pd.to_datetime(did["date"], unit="D", origin="1960-01-01").dt.year
    print(f"Matched DiD sample rows: {len(did):,}")
    c3 = run_event(did, "pm10", ["pm10_n", "month"], "(3) +Matching, month FE")
    c4 = run_event(did, "pm10", ["pm10_n", "year_month"], "(4) +Matching, year-month FE")
    c5 = run_event(did, "l_pm10", ["pm10_n", "year_month"], "(5) +Matching, log PM10, year-month FE")

    results = [c1, c2, c3, c4, c5]
    print_compare(results)

    # Save
    rows = []
    for r in results:
        for v in EVENT_VARS:
            rows.append({"col": r["label"], "var": v, "coef": r["coefs"][v],
                         "se": r["ses"][v], "n": r["n"]})
    pd.DataFrame(rows).to_csv(OUT / "table1b_replicated.csv", index=False)


if __name__ == "__main__":
    main()
