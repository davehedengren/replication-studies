"""Shared paths and helpers for 125561 replication (Gonzales, Leon-Ciliotta, Martinez 2022)."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
PKG = REPO / "125561-V1"
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)

# CLEAN data is in the zip extracted with 'null' prefix at the package root
ELECTIONS_DTA = PKG / "nullElections.dta"
TABLE6_DTA = PKG / "nullTable6.dta"
TABLEC2_DTA = PKG / "nullTableC2.dta"
GOOGLE_DTA = PKG / "nullgoogle_trends.dta"
ELECCIONES2016_DTA = PKG / "nullelecciones_2016full.dta"
FIGUREA2_DTA = PKG / "nullFigureA2.dta"

# Keep only the 8 national presidential elections
# (paper Table 1 col 2 uses these 8 → 14,040 district-elections, 1,755 districts)
NATIONAL_DATES = [
    "2001-04-08",  # 2001 general
    "2001-06-03",  # 2001 runoff
    "2006-04-09",  # 2006 general
    "2006-06-04",  # 2006 runoff
    "2011-04-10",  # 2011 general
    "2011-06-05",  # 2011 runoff
    "2016-04-10",  # 2016 general
    "2016-06-05",  # 2016 runoff
]


def load_elections():
    df = pd.read_stata(ELECTIONS_DTA, convert_categoricals=False)
    df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")
    df = df[df["date_str"].isin(NATIONAL_DATES)].copy()
    df["date_id"] = pd.Categorical(df["date_str"]).codes.astype(int)
    return df.reset_index(drop=True)


def make_group_ids(df, cols, name):
    """Integer group id from multiple columns."""
    key = df[cols].astype(str).agg("_".join, axis=1)
    df[name] = pd.Categorical(key).codes.astype(int)
    return df


def iterative_singleton_drop(df, group_cols):
    """Drop observations that form singletons in any of the listed groups,
    iterating until a fixed point (matches reghdfe's default)."""
    while True:
        n0 = len(df)
        for g in group_cols:
            vc = df[g].value_counts()
            df = df[df[g].isin(vc[vc >= 2].index)]
        if len(df) == n0:
            break
    return df.reset_index(drop=True)


def absorb_within(df, yvar, xvars, absorb, weight=None, tol=1e-10, max_iter=200):
    """Partial out high-dimensional FEs via alternating group means (Gauss-Seidel).

    absorb : list of columns to sweep (each integer-like group id).
    Returns residualized y and X as numpy arrays, plus the sample mask
    (after any singleton drops we did not perform).
    """
    df = df.dropna(subset=[yvar] + xvars + absorb).copy()
    if weight is not None:
        df = df.dropna(subset=[weight])
        w = df[weight].to_numpy(dtype=float)
    else:
        w = np.ones(len(df))

    # Start with the raw series; iteratively subtract weighted group means.
    def demean(series):
        v = series.to_numpy(dtype=float).copy()
        for _ in range(max_iter):
            max_delta = 0.0
            for g in absorb:
                gid = df[g].to_numpy()
                # weighted group means
                # sum w*v per group / sum w per group
                num = np.bincount(gid, weights=w * v)
                den = np.bincount(gid, weights=w)
                den_safe = np.where(den == 0, 1.0, den)
                means = num / den_safe
                delta = means[gid]
                v -= delta
                md = float(np.max(np.abs(delta)))
                if md > max_delta:
                    max_delta = md
            if max_delta < tol:
                break
        return v

    y_res = demean(df[yvar])
    X_res = np.column_stack([demean(df[x]) for x in xvars])
    return df, y_res, X_res, w


def feols_hdfe(df, yvar, xvars, absorb, cluster, weight=None):
    """Stata-style reghdfe: absorb FEs, WLS, cluster-robust SE.

    Reports a 'Stata-style' cluster SE: (M/(M-1))*((N-1)/(N-K)) adjustment,
    where K is the *total* number of parameters including absorbed FEs.
    """
    used, y_res, X_res, w = absorb_within(df, yvar, xvars, absorb, weight=weight)
    cluster_vals = used[cluster].to_numpy()

    # WLS
    sw = np.sqrt(w)
    Xw = X_res * sw[:, None]
    yw = y_res * sw
    XtX = Xw.T @ Xw
    XtX_inv = np.linalg.inv(XtX)
    beta = XtX_inv @ (Xw.T @ yw)
    resid = yw - Xw @ beta

    # Cluster-robust SE (Liang-Zeger)
    clusters = np.unique(cluster_vals)
    M = len(clusters)
    # reghdfe subtracts absorbed FE levels that are nested within clusters.
    # Here district FE (ubigeo) and date×province×cat FE are both nested in
    # province (cluster), so they should count as 0 added parameters.
    # Non-nested FEs contribute (levels - 1) each after the first.
    K_absorbed = 0
    for i, g in enumerate(absorb):
        # Is this FE nested in cluster? (each level maps to a single cluster)
        nested = (used.groupby(g)[cluster].nunique() == 1).all()
        if not nested:
            K_absorbed += int(used[g].nunique()) - (1 if i > 0 else 0)
    K = X_res.shape[1] + K_absorbed
    N = len(used)
    meat = np.zeros((X_res.shape[1], X_res.shape[1]))
    for c in clusters:
        mask = cluster_vals == c
        u = (Xw[mask].T @ resid[mask]).reshape(-1, 1)
        meat += u @ u.T
    dof = (M / max(M - 1, 1)) * ((N - 1) / max(N - K, 1))
    vcov = dof * XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(vcov))

    # Within R² = 1 - RSS_w / TSS_w (on weighted, demeaned scale)
    rss = float(np.sum(resid ** 2))
    ybar_w = float(np.sum(w * y_res) / np.sum(w))
    tss = float(np.sum(w * (y_res - ybar_w) ** 2))
    r2_within = 1.0 - rss / tss if tss > 0 else np.nan

    return {
        "N": N,
        "districts": int(used["ubigeo"].nunique()),
        "beta": beta,
        "se": se,
        "r2_within": r2_within,
        "ymean": float(np.sum(w * used[yvar].to_numpy()) / np.sum(w)),
        "xvars": xvars,
    }


def fmt(res, name=None):
    xs = res["xvars"]
    lines = []
    for i, x in enumerate(xs):
        lines.append(f"  {x}: {res['beta'][i]:.4f}  (SE={res['se'][i]:.4f})")
    hdr = name if name else ""
    return f"{hdr}\n" + "\n".join(lines) + f"\n  N={res['N']}  districts={res['districts']}  ybar={res['ymean']:.4f}  R2w={res['r2_within']:.4f}"
