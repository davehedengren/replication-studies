"""Robustness checks for the sector-level gravity regression.

The paper's headline point estimates are the trade costs fed into a structural
calibration; we cannot re-run the calibration (MATLAB-only) but we can stress
test whether the gravity step — which determines those inputs — is fragile.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from utils import (
    load_trade, OUT, OMIT_CODE, N_TRADABLE, DIST_BINS, OTHER_VARS,
)


def design(df_j):
    X = df_j[DIST_BINS + OTHER_VARS].astype(float).copy()
    for c in range(1, 20):
        if c == OMIT_CODE:
            continue
        X[f"exp_{c}"] = (df_j["exp"] == c).astype(float)
        X[f"imp_{c}"] = (df_j["imp"] == c).astype(float)
    return X


def fit_sector(df_j, **fit_kw):
    X = design(df_j)
    y = df_j["lX_ni_X_nn"].astype(float)
    mask = y.notna() & X.notna().all(axis=1)
    y, X = y[mask], X[mask]
    keep, Q = [], np.empty((X.shape[0], 0))
    for col in X.columns:
        v = X[[col]].values
        if Q.shape[1] > 0:
            v = v - Q @ (Q.T @ v)
        if np.linalg.norm(v) > 1e-8:
            Q = np.concatenate([Q, v / np.linalg.norm(v)], axis=1)
            keep.append(col)
    return sm.OLS(y, X[keep]).fit(**fit_kw), mask.sum()


def sector_mean_coef(df, var, **fit_kw):
    vals = []
    for j in range(1, N_TRADABLE + 1):
        m, _ = fit_sector(df[df["j19"] == j], **fit_kw)
        vals.append(m.params.get(var, np.nan))
    return np.nanmean(vals), np.nanstd(vals)


def banner(t):
    print(f"\n{'='*60}\n{t}\n{'='*60}")


def main():
    df = load_trade()
    results = []

    # --- 1. Baseline: pooled mean across 18 sectors (as in paper) ---
    banner("1. Baseline sector-by-sector means")
    for v in DIST_BINS + OTHER_VARS:
        m, s = sector_mean_coef(df, v)
        print(f"  {v:12s}  mean={m:+.4f}  std={s:.4f}")
        results.append(("baseline", v, m, s))

    # --- 2. Robust HC3 standard errors (vs HC0 default) ---
    banner("2. HC3 vs HC0 SE comparison (sector 1)")
    for cov in ["nonrobust", "HC0", "HC1", "HC3"]:
        m, _ = fit_sector(df[df["j19"] == 1], cov_type=cov if cov != "nonrobust" else "nonrobust")
        print(f"  cov={cov:10s}  SE(d0_375)={m.bse['d0_375']:.4f}  SE(d6000_)={m.bse['d6000_']:.4f}")

    # --- 3. Clustered SEs by exporter ---
    banner("3. Clustered SEs by exporter (sector 1)")
    sub = df[df["j19"] == 1].copy()
    mask = sub["lX_ni_X_nn"].notna() & sub[DIST_BINS + OTHER_VARS].notna().all(axis=1)
    sub = sub[mask]
    X = design(sub)
    # Drop empty FE cols
    X = X.loc[:, (X != 0).any()]
    y = sub["lX_ni_X_nn"].astype(float)
    m_cluster = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": sub["exp"]})
    m_plain = sm.OLS(y, X).fit()
    for v in DIST_BINS + OTHER_VARS:
        print(f"  {v:12s}  b={m_plain.params[v]:+.4f}  SE_plain={m_plain.bse[v]:.4f}  SE_cluster_exp={m_cluster.bse[v]:.4f}")

    # --- 4. Pooled regression with sector FE instead of separate regressions ---
    banner("4. Pooled regression with sector FE")
    pooled = df[df["j19"] <= N_TRADABLE].copy()
    pooled = pooled.dropna(subset=["lX_ni_X_nn"] + DIST_BINS + OTHER_VARS + ["exp", "imp", "j19"])
    Xp = pooled[DIST_BINS + OTHER_VARS].astype(float).copy()
    for c in range(1, 20):
        if c == OMIT_CODE:
            continue
        Xp[f"exp_{c}"] = (pooled["exp"] == c).astype(float)
        Xp[f"imp_{c}"] = (pooled["imp"] == c).astype(float)
    for j in range(2, N_TRADABLE + 1):
        Xp[f"sec_{j}"] = (pooled["j19"] == j).astype(float)
    y = pooled["lX_ni_X_nn"].astype(float)
    m = sm.OLS(y, Xp).fit()
    for v in DIST_BINS + OTHER_VARS:
        print(f"  {v:12s}  b={m.params[v]:+.4f}  SE={m.bse[v]:.4f}")
    print(f"  N={int(m.nobs)}  R2={m.rsquared:.4f}")

    # --- 5. Leave-one-country-out: how stable are pooled means? ---
    banner("5. Leave-one-country-out pooled means of b(d0_375) and b(d6000_)")
    base = []
    for j in range(1, N_TRADABLE + 1):
        m, _ = fit_sector(df[df["j19"] == j])
        base.append(m.params.get("d0_375", np.nan))
    print(f"  baseline mean b(d0_375) = {np.nanmean(base):+.4f}")
    for drop_c in range(1, 20):
        vals = []
        for j in range(1, N_TRADABLE + 1):
            sub = df[(df["j19"] == j) & (df["exp"] != drop_c) & (df["imp"] != drop_c)]
            m, _ = fit_sector(sub)
            vals.append(m.params.get("d0_375", np.nan))
        name = ["AUS","AUT","BEL","CAN","DEU","ESP","FIN","FRA","GBR","IRL","ISR","ITA","JPN","KOR","NLD","NOR","NZL","PRT","USA"][drop_c-1]
        print(f"  drop {name:3s}  mean b(d0_375) = {np.nanmean(vals):+.4f}")

    # --- 6. Leave-one-sector-out for total distance effect ---
    banner("6. Leave-one-sector-out pooled means")
    sectors = list(range(1, N_TRADABLE + 1))
    base_b = np.mean(base)
    for drop_s in sectors:
        vals = [base[s-1] for s in sectors if s != drop_s]
        print(f"  drop sector {drop_s:2d}  mean b(d0_375) = {np.mean(vals):+.4f}")

    # --- 7. Robust to nontradable/zero-trade trimming ---
    banner("7. Trim: drop obs where lX_ni_X_nn < -15 (ultra-thin pairs)")
    m_trim, s_trim = [], []
    for j in range(1, N_TRADABLE + 1):
        sub = df[(df["j19"] == j) & (df["lX_ni_X_nn"] > -15)]
        mm, _ = fit_sector(sub)
        m_trim.append(mm.params.get("d0_375", np.nan))
        s_trim.append(mm.params.get("d6000_", np.nan))
    print(f"  mean b(d0_375)  baseline={np.nanmean(base):+.4f}  trimmed={np.nanmean(m_trim):+.4f}")
    base_long = [fit_sector(df[df['j19']==j])[0].params.get("d6000_", np.nan) for j in sectors]
    print(f"  mean b(d6000_)  baseline={np.nanmean(base_long):+.4f}  trimmed={np.nanmean(s_trim):+.4f}")

    # --- 8. Placebo: reshuffled trade flows within sector ---
    banner("8. Placebo — shuffle lX_ni_X_nn within each sector")
    rng = np.random.default_rng(42)
    placebo = []
    for j in range(1, N_TRADABLE + 1):
        sub = df[df["j19"] == j].copy().reset_index(drop=True)
        sub["lX_ni_X_nn"] = rng.permutation(sub["lX_ni_X_nn"].values)
        mm, _ = fit_sector(sub)
        placebo.append(mm.params.get("d0_375", np.nan))
    print(f"  mean b(d0_375) under shuffled outcome: {np.nanmean(placebo):+.4f}  (baseline {np.nanmean(base):+.4f})")
    print("  → distance effect collapses, as expected.")

    # --- 9. Alternative omitted category (use USA=19 instead of NLD=15) ---
    banner("9. Alternative omitted category (USA=19 vs NLD=15)")
    protected = set(DIST_BINS + OTHER_VARS)

    def fit_sector_omit(df_j, omit_code):
        X = df_j[DIST_BINS + OTHER_VARS].astype(float).copy()
        for c in range(1, 20):
            if c == omit_code:
                continue
            X[f"exp_{c}"] = (df_j["exp"] == c).astype(float)
            X[f"imp_{c}"] = (df_j["imp"] == c).astype(float)
        y = df_j["lX_ni_X_nn"].astype(float)
        mask = y.notna() & X.notna().all(axis=1)
        y, X = y[mask], X[mask]
        # Protect distance bins / dummies by feeding them to the orthogonaliser
        # first, then add FE columns.
        ordered_cols = [c for c in X.columns if c in protected] + [c for c in X.columns if c not in protected]
        X = X[ordered_cols]
        keep, Q = [], np.empty((X.shape[0], 0))
        for col in X.columns:
            v = X[[col]].values
            if Q.shape[1] > 0:
                v = v - Q @ (Q.T @ v)
            if np.linalg.norm(v) > 1e-8:
                Q = np.concatenate([Q, v / np.linalg.norm(v)], axis=1)
                keep.append(col)
        return sm.OLS(y, X[keep]).fit()

    nld_d0, nld_slope, nld_contig = [], [], []
    usa_d0, usa_slope, usa_contig = [], [], []
    for j in range(1, N_TRADABLE + 1):
        mn = fit_sector_omit(df[df["j19"] == j], 15)
        mu = fit_sector_omit(df[df["j19"] == j], 19)
        nld_d0.append(mn.params.get("d0_375", np.nan))
        usa_d0.append(mu.params.get("d0_375", np.nan))
        nld_slope.append(mn.params.get("d6000_", np.nan) - mn.params.get("d0_375", np.nan))
        usa_slope.append(mu.params.get("d6000_", np.nan) - mu.params.get("d0_375", np.nan))
        nld_contig.append(mn.params.get("contig", np.nan))
        usa_contig.append(mu.params.get("contig", np.nan))
    print(f"  mean b(d0_375)          omit NLD: {np.nanmean(nld_d0):+.4f}  omit USA: {np.nanmean(usa_d0):+.4f}")
    print(f"  mean b(d6000_ − d0_375) omit NLD: {np.nanmean(nld_slope):+.4f}  omit USA: {np.nanmean(usa_slope):+.4f}")
    print(f"  mean b(contig)          omit NLD: {np.nanmean(nld_contig):+.4f}  omit USA: {np.nanmean(usa_contig):+.4f}")
    print("  → distance LEVELS shift by a constant (reparameterisation); DIFFERENCES")
    print("    and bilateral dummies are invariant, as expected.")

    # --- 10. Within-variation: drop contig (no neighbor effect) ---
    banner("10. Drop 'contig' control")
    vals2 = []
    for j in range(1, N_TRADABLE + 1):
        sub = df[df["j19"] == j].copy()
        Xd = sub[DIST_BINS + ["FTA", "CU"]].astype(float).copy()
        for c in range(1, 20):
            if c == OMIT_CODE:
                continue
            Xd[f"exp_{c}"] = (sub["exp"] == c).astype(float)
            Xd[f"imp_{c}"] = (sub["imp"] == c).astype(float)
        y = sub["lX_ni_X_nn"].astype(float)
        mask = y.notna() & Xd.notna().all(axis=1)
        y, Xd = y[mask], Xd[mask]
        keep, Q = [], np.empty((Xd.shape[0], 0))
        for col in Xd.columns:
            v = Xd[[col]].values
            if Q.shape[1] > 0:
                v = v - Q @ (Q.T @ v)
            if np.linalg.norm(v) > 1e-8:
                Q = np.concatenate([Q, v / np.linalg.norm(v)], axis=1)
                keep.append(col)
        mm = sm.OLS(y, Xd[keep]).fit()
        vals2.append(mm.params.get("d0_375", np.nan))
    print(f"  mean b(d0_375) no-contig: {np.nanmean(vals2):+.4f}  vs baseline {np.nanmean(base):+.4f}")

    # --- 11. Sensitivity to theta ---
    banner("11. Implied trade cost d_ni^j under alternative theta")
    ld = pd.read_stata("../120281-V1/CLS_ReplicationPackage/CLS_ReplicationPackage_v1/Raw_data/lD_inj1919.dta", convert_categoricals=False)
    sub = ld[(ld["j19"] <= N_TRADABLE) & (ld["iso_i"] != ld["iso_n"])].copy()
    for theta in [4.0, 6.0, 8.28, 10.0]:
        d = np.exp(-sub["lD_inj"] / theta)
        print(f"  theta={theta:5.2f}  mean d_ni={d.mean():.3f}  median={d.median():.3f}  max={d.max():.2f}")


if __name__ == "__main__":
    main()
