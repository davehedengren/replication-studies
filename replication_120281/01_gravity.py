"""Replicates the sector-level gravity regression (Raw_data/gravity1919_rep.do).

Estimates, for each tradable sector j = 1..18:

    log(X_ni^j / X_nn^j) = D_bin + contig + FTA + CU + exporter_FE + importer_FE + e

with NLD (code 15) as the omitted category for both FE sets, and no constant.
These are Appendix A / Section 5.1 trade-cost estimates that feed the structural
calibration.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from utils import (
    load_trade, OUT, COUNTRIES, OMIT_CODE, N_TRADABLE, DIST_BINS, OTHER_VARS,
)


def build_design(df_j):
    X = df_j[DIST_BINS + OTHER_VARS].astype(float).copy()
    # exporter FE (exp = 1..19, omit OMIT_CODE=15)
    for c in range(1, 20):
        if c == OMIT_CODE:
            continue
        X[f"exp_{c}"] = (df_j["exp"] == c).astype(float)
    # importer FE (imp = 1..19, omit OMIT_CODE=15)
    for c in range(1, 20):
        if c == OMIT_CODE:
            continue
        X[f"imp_{c}"] = (df_j["imp"] == c).astype(float)
    return X


def run_one_sector(df, j):
    sub = df[df["j19"] == j].copy()
    sub = sub.dropna(subset=["lX_ni_X_nn", "exp", "imp"] + DIST_BINS + OTHER_VARS)
    y = sub["lX_ni_X_nn"].astype(float)
    X = build_design(sub)
    # Drop perfectly-collinear columns (e.g., a country with no within-sample
    # observations as importer for this sector). statsmodels will otherwise
    # return NaNs; we use rank-revealing removal.
    keep = []
    R = X.values
    used = np.zeros(R.shape[1], dtype=bool)
    # Simple forward-selection: drop columns that are linearly dependent
    # on already-kept columns.
    Q = np.empty((R.shape[0], 0))
    for i, col in enumerate(X.columns):
        v = R[:, i:i + 1]
        if Q.shape[1] > 0:
            proj = Q @ (Q.T @ v)
            resid = v - proj
        else:
            resid = v
        norm = np.linalg.norm(resid)
        if norm > 1e-8:
            Q = np.concatenate([Q, resid / norm], axis=1)
            keep.append(col)
    X_k = X[keep]
    model = sm.OLS(y, X_k).fit()  # no constant; exp/imp FE absorb it
    return model, sub, keep


def main():
    df = load_trade()
    summary_rows = []
    coef_rows = []
    for j in range(1, N_TRADABLE + 1):
        m, sub, keep = run_one_sector(df, j)
        row = {
            "sector": j,
            "N": int(m.nobs),
            "R2": m.rsquared,
            "RMSE": float(np.sqrt(m.mse_resid)),
        }
        for v in DIST_BINS + OTHER_VARS:
            row[f"b_{v}"] = m.params.get(v, np.nan)
            row[f"se_{v}"] = m.bse.get(v, np.nan)
        summary_rows.append(row)
        for name in DIST_BINS + OTHER_VARS:
            coef_rows.append({
                "sector": j,
                "var": name,
                "coef": m.params.get(name, np.nan),
                "se": m.bse.get(name, np.nan),
                "t": m.tvalues.get(name, np.nan),
            })

    summary = pd.DataFrame(summary_rows)
    coefs = pd.DataFrame(coef_rows)
    summary.to_csv(OUT / "gravity_summary.csv", index=False)
    coefs.to_csv(OUT / "gravity_coefs.csv", index=False)

    print("\n=== Gravity Regression: Per-Sector Summary ===")
    print(summary[["sector", "N", "R2", "b_d0_375", "b_d6000_", "b_contig", "b_FTA"]].to_string(index=False))

    print("\n=== Pooled means of coefficients across 18 sectors ===")
    means = summary[[c for c in summary.columns if c.startswith("b_")]].mean()
    for k, v in means.items():
        print(f"  {k}: {v:.4f}")

    # Compare our fitted D_bin*distance contributions to the shipped trade costs
    # via lD_inj1919.dta, at least via correlation.
    try:
        ld = pd.read_stata(
            "../120281-V1/CLS_ReplicationPackage/CLS_ReplicationPackage_v1/Raw_data/lD_inj1919.dta",
            convert_categoricals=False,
        )
        print(f"\nlD_inj1919.dta rows: {len(ld)}; non-null lD_inj: {ld['lD_inj'].notna().sum()}")
        print(ld["lD_inj"].describe())
    except Exception as e:
        print(f"(could not load lD_inj1919.dta: {e})")


if __name__ == "__main__":
    main()
