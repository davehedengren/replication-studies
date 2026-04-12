"""Shared paths and helpers for the Alon/Capelle/Matsuda (2025) replication."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PKG = ROOT.parent / "226201-V1" / "empirical"
RAW = PKG / "data raw"
PROC = PKG / "data proc"
OUT_ORIG = PKG / "output"
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)


def load_merged_panel():
    """Load the pre-built IPEDS+HERD panel supplied in the replication package."""
    df = pd.read_stata(PROC / "merged_HERD_IPEDS_data.dta", convert_categoricals=False)
    df = df[df["sector"].isin([1, 2])].copy()
    return df


def load_libart_flag():
    df = pd.read_excel(RAW / "Flag_Liberal_Arts_College.xlsx", sheet_name="LibArtFlag")
    df.columns = [c.strip() for c in df.columns]
    df["flag_lib_art_col"] = df["flag_lib_art_col"].fillna(0).astype(int)
    return df[["unitid", "flag_lib_art_col"]].drop_duplicates("unitid")


def assign_Dt(year):
    """Match the Stata Dt assignment (line 170-176 of NIH regression do file)."""
    out = np.full(len(year), np.nan)
    y = np.asarray(year)
    out[(y >= 1993) & (y <= 1997)] = 0
    out[(y >= 2004) & (y <= 2008)] = 1
    out[(y >= 1998) & (y <= 2003)] = 2
    out[(y >= 1988) & (y <= 1992)] = 3
    return out


def aw_cluster_iv(y, X_exog, X_endog, Z_instr, weights, cluster):
    """
    Analytic weighted 2SLS with cluster-robust (CR1) SEs.

    y:       (n,) outcome
    X_exog:  (n, kX) exogenous regressors incl. constant
    X_endog: (n, kE) endogenous regressors
    Z_instr: (n, kZ) excluded instruments (kZ >= kE)
    weights: (n,) analytic weights
    cluster: (n,) cluster ids

    Returns dict with keys: beta, se, names, n, df_resid, resid, R2.

    Matches Stata `ivregress 2sls ... [aw=wts], robust cluster(id)`:
    analytic weights + Stata small-sample cluster correction
    (G/(G-1)) * ((N-1)/(N-K)).
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    X_exog = np.asarray(X_exog, dtype=float)
    X_endog = np.asarray(X_endog, dtype=float)
    Z_instr = np.asarray(Z_instr, dtype=float)
    w = np.asarray(weights, dtype=float).reshape(-1)
    w = w * len(w) / w.sum()  # Stata aw normalizes so sum(w)=N
    sw = np.sqrt(w)
    cluster = np.asarray(cluster)

    Z_full = np.column_stack([Z_instr, X_exog])
    X_full = np.column_stack([X_endog, X_exog])
    yw = y * sw
    Xw = X_full * sw[:, None]
    Zw = Z_full * sw[:, None]

    # First stage: project endog onto Z_full (weighted)
    ZtZ = Zw.T @ Zw
    Pi = np.linalg.solve(ZtZ, Zw.T @ Xw)
    Xhat_w = Zw @ Pi

    # Second stage
    XtX_hat = Xhat_w.T @ Xhat_w
    beta = np.linalg.solve(XtX_hat, Xhat_w.T @ yw)

    # Residuals (use actual X, not Xhat — standard 2SLS)
    resid = y - X_full @ beta
    n = len(y)
    k = X_full.shape[1]

    # Cluster-robust variance (CR1). Build meat in the weighted "Xhat" space.
    # For weighted 2SLS with robust/cluster, bread = (X̂' W X̂)^{-1}
    # meat  = Σ_g (X̂_g' W_g u_g)(X̂_g' W_g u_g)'
    bread = np.linalg.inv(XtX_hat)
    u_w = resid * sw  # weighted residuals
    Xhat_w_scaled = Xhat_w  # already sqrt-w'd
    score = Xhat_w_scaled * u_w[:, None]
    df = pd.DataFrame(score)
    df["_g"] = cluster
    S = df.groupby("_g").sum().values
    meat = S.T @ S
    G = len(np.unique(cluster))
    # Stata's `ivregress 2sls, robust cluster()` applies NO small-sample
    # correction (unlike regress, which applies G/(G-1)*(N-1)/(N-K)).
    # See Stata help for ivregress; matches the published SEs to 5 decimals.
    V = bread @ meat @ bread
    se = np.sqrt(np.diag(V))

    # R^2 (weighted, matches Stata post-2sls e(r2))
    y_mean = (w * y).sum() / w.sum()
    ss_res = (w * resid ** 2).sum()
    ss_tot = (w * (y - y_mean) ** 2).sum()
    r2 = 1 - ss_res / ss_tot

    return {
        "beta": beta,
        "se": se,
        "n": n,
        "df_resid": n - k,
        "resid": resid,
        "R2": r2,
        "V": V,
    }


def build_regression_panel():
    """Replicate the Stata sample construction through `collapse/reshape`.

    Returns the wide panel that feeds the Table 1 IV specs, with columns:
      uni_ID, sector, geo_state, flag_lib_art_col, yy, xx, zz, fte_wts,
      init_life_sci_uni, init_uni_size, init_fac_stud_ratio, placebo_xx.
    """
    df = load_merged_panel()
    flag = load_libart_flag()
    df = df.merge(flag, on="unitid", how="left")
    df["flag_lib_art_col"] = df["flag_lib_art_col"].fillna(0).astype(int)

    df["tuition"] = df["nettuition01"]
    df["research"] = df["research01"]
    df["stud_amenities"] = df["studserv01"]
    df["fs_ratio"] = df["ft_faculty_per_100fte"] * 10

    # Non-collapsible characteristics as of year==2015, 1 row per uni.
    # In the Stata code (`preserve ... keep if year==2015`) this snapshot is
    # taken BEFORE the Dt filter — so it must come before we drop rows.
    nc = df[df["year"] == 2015].drop_duplicates("uni_ID")[
        ["uni_ID", "sector", "geo_state", "flag_lib_art_col"]
    ]

    df["Dt"] = assign_Dt(df["year"])
    df = df.dropna(subset=["Dt"])
    df["Dt"] = df["Dt"].astype(int)

    # `collapse (sum)` in Stata with missings: for each (uni_ID, Dt) group,
    # sum ignores NaN (default), and mean also ignores NaN. We replicate that.
    agg = (
        df.groupby(["uni_ID", "Dt"])
        .agg(
            avg_tuit_pc=("nettuition01", lambda s: (s / df.loc[s.index, "fte_count"]).mean()),
            avg_fte=("fte_count", "mean"),
            fs_ratio=("fs_ratio", "mean"),
            tot_tuit=("tuition", "sum"),
            tot_research=("research", "sum"),
            RD_LifeSci_Federal=("RD_LifeSci_Federal", "sum"),
            RD_LifeSci_All=("RD_LifeSci_All", "sum"),
            tot_fte=("fte_count", "sum"),
            tot_stud_amenities=("stud_amenities", "sum"),
        )
        .reset_index()
    )

    wide = agg.pivot(index="uni_ID", columns="Dt")
    wide.columns = [f"{v}{int(d)}" for v, d in wide.columns]
    wide = wide.reset_index()

    wide = wide.merge(nc, on="uni_ID", how="inner")  # keep only unis present in 2015

    wide["yy"] = (wide["tot_tuit1"] / wide["tot_fte1"]) - (
        wide["tot_tuit0"] / wide["tot_fte0"]
    )
    wide["xx"] = (wide["tot_research1"] / wide["tot_fte1"]) - (
        wide["tot_research0"] / wide["tot_fte0"]
    )
    wide["placebo_xx"] = (
        wide["tot_stud_amenities1"] / wide["tot_fte1"]
    ) - (wide["tot_stud_amenities0"] / wide["tot_fte0"])
    wide["fte_wts"] = wide["avg_fte0"]

    agg_lifesci_fed = wide["RD_LifeSci_Federal0"].sum()
    wide["zz"] = wide["RD_LifeSci_Federal0"] / agg_lifesci_fed

    # Pre-period characteristics
    wide["init_life_sci_uni"] = np.where(
        wide["RD_LifeSci_All3"].isna(),
        np.nan,
        (wide["RD_LifeSci_All3"] > 0).astype(float),
    )
    wide["init_uni_size"] = wide["tot_fte3"] / 1000
    wide["init_fac_stud_ratio"] = wide["fs_ratio3"]

    return wide


def dummies_from(series, name, drop_first=False):
    """Expand a categorical into 0/1 columns, like Stata `i.var`."""
    return pd.get_dummies(series, prefix=name, drop_first=drop_first).astype(float)
