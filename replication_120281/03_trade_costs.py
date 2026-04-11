"""Construct model-implied bilateral log trade cost -theta*log(d_ni^j) from
my gravity coefficients and compare to the shipped lD_inj1919.dta.

Stata: gen lD_inj = v1*d0_375 + v2*d375_750 + ... + v6*d6000 + v7*contig
        + v8*FTA + v9*CU - theta_ex_ij + res

theta_ex_ij = log(S_nj_0/S_ij_0) (exporter-side adjustment using fixed effects).
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from utils import load_trade, OUT, OMIT_CODE, N_TRADABLE, DIST_BINS, OTHER_VARS, RAW


def fit_sector(df_j):
    X = df_j[DIST_BINS + OTHER_VARS].astype(float).copy()
    for c in range(1, 20):
        if c == OMIT_CODE:
            continue
        X[f"exp_{c}"] = (df_j["exp"] == c).astype(float)
        X[f"imp_{c}"] = (df_j["imp"] == c).astype(float)
    y = df_j["lX_ni_X_nn"].astype(float)
    mask = y.notna() & X.notna().all(axis=1)
    y, X = y[mask], X[mask]
    # Drop dependent cols
    keep, Q = [], np.empty((X.shape[0], 0))
    for col in X.columns:
        v = X[[col]].values
        if Q.shape[1] > 0:
            v = v - Q @ (Q.T @ v)
        nrm = np.linalg.norm(v)
        if nrm > 1e-8:
            Q = np.concatenate([Q, v / nrm], axis=1)
            keep.append(col)
    X_k = X[keep]
    model = sm.OLS(y, X_k).fit()
    resid = y - model.predict(X_k)
    return model, X_k.index, resid


def main():
    df = load_trade()
    rows = []
    for j in range(1, N_TRADABLE + 1):
        sub = df[df["j19"] == j].copy().reset_index(drop=True)
        m, idx, resid = fit_sector(sub)
        # Distance contribution
        dist_lin = sum(m.params.get(v, 0.0) * sub.loc[idx, v] for v in DIST_BINS + OTHER_VARS)
        # Stata code: theta_ex_ij = log(S_nj_0/S_ij_0)
        #             where S_nj_0 = exp(-F_imp), S_ij_0 = exp(+F_exp)
        #             → theta_ex_ij = -F_imp(i) - F_exp(i)  (evaluated at exporter)
        # lD_inj = dist_lin + resid - theta_ex_ij
        #        = dist_lin + resid + F_imp(i) + F_exp(i)
        imp_fe = {OMIT_CODE: 0.0}
        exp_fe = {OMIT_CODE: 0.0}
        for c in range(1, 20):
            if c == OMIT_CODE:
                continue
            imp_fe[c] = m.params.get(f"imp_{c}", 0.0)
            exp_fe[c] = m.params.get(f"exp_{c}", 0.0)
        theta_ex = sub.loc[idx, "exp"].apply(
            lambda c: -imp_fe[int(c)] - exp_fe[int(c)]
        )
        lD_inj = dist_lin + resid - theta_ex
        # Stata: replace lD_inj=0 if lD_inj>0 (cap above at 0)
        lD_inj = lD_inj.clip(upper=0.0)
        out = pd.DataFrame({
            "iso_i": sub.loc[idx, "iso_i"].values,
            "iso_n": sub.loc[idx, "iso_n"].values,
            "j": j,
            "lD_inj_py": lD_inj.values,
        })
        rows.append(out)
    computed = pd.concat(rows, ignore_index=True)
    computed.to_csv(OUT / "lD_inj_python.csv", index=False)

    # Compare to shipped
    ld = pd.read_stata(RAW / "lD_inj1919.dta", convert_categoricals=False)
    # Filter to sectors 1..18, off-diagonal
    ld = ld[(ld["j19"] <= N_TRADABLE) & (ld["iso_i"] != ld["iso_n"])].copy()
    ld = ld[["iso_i", "iso_n", "j19", "lD_inj"]].rename(columns={"j19": "j"})
    merged = computed.merge(ld, on=["iso_i", "iso_n", "j"], how="inner")
    # lD_inj (shipped) has 1000 placeholder for j==19 and 0 for diagonal — both filtered.
    diff = merged["lD_inj_py"] - merged["lD_inj"]
    print(f"Merged rows: {len(merged)}")
    print(f"Shipped lD_inj stats:  mean={merged['lD_inj'].mean():.4f}  std={merged['lD_inj'].std():.4f}")
    print(f"Python  lD_inj stats:  mean={merged['lD_inj_py'].mean():.4f}  std={merged['lD_inj_py'].std():.4f}")
    print(f"Max abs diff: {diff.abs().max():.3e}")
    print(f"Mean abs diff: {diff.abs().mean():.3e}")
    print(f"Correlation: {merged['lD_inj_py'].corr(merged['lD_inj']):.6f}")

    # Distribution of implied trade cost d_ni^j = exp(-lD_inj/theta), theta=8.28
    theta = 8.28
    merged["d_py"] = np.exp(-merged["lD_inj_py"] / theta)
    merged["d_stata"] = np.exp(-merged["lD_inj"] / theta)
    print(f"\nImplied d_ni^j summary (theta=8.28):")
    print(merged[["d_py", "d_stata"]].describe().to_string())


if __name__ == "__main__":
    main()
