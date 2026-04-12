"""Phase 4: robustness checks on the AK descriptive estimates.

The paper's main story rests on the AK-corrected latent mean theta_bar and the
spread tau. Each check below re-estimates the t-dist, one-cutoff AK model on
RR and PBD and reports how theta, tau, and beta_p move.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from utils import load_long, OUT
from importlib import import_module
ak = import_module("02_ak_descriptives")


def fit_row(label, X, sigma):
    fit = ak.ak_fit(X, sigma, cutoffs=(1.96,), symmetric=False, model="t")
    lam, tau, df, beta = fit.x
    return {
        "check": label, "n": len(X), "mean_eps": X.mean(),
        "beta_p": beta, "theta": lam, "tau": tau, "df": df,
    }


def run_checks_for(margin):
    df = load_long()
    sub = df[df["pbd_vs_rr"] == margin].copy()
    X = sub["elasticity"].values
    s = sub["se"].values

    rows = [fit_row("baseline", X, s)]

    # 1. Drop Hunt-like outlier (<-2 or >2)
    m = np.abs(X) <= 2
    rows.append(fit_row("drop |elas|>2", X[m], s[m]))

    # 2. Drop extreme-SE studies (top 10% SE)
    q90 = np.quantile(s, 0.90)
    m = s <= q90
    rows.append(fit_row("drop top 10% SE", X[m], s[m]))

    # 3. Winsorize elasticity at 1st/99th pct
    lo, hi = np.quantile(X, [0.01, 0.99])
    Xw = np.clip(X, lo, hi)
    rows.append(fit_row("winsorize 1/99", Xw, s))

    # 4. USA-only
    m = sub["us_country"].values == 1
    if m.sum() >= 10:
        rows.append(fit_row("USA only", X[m], s[m]))

    # 5. Non-USA only
    m = sub["us_country"].values == 0
    if m.sum() >= 10:
        rows.append(fit_row("non-USA only", X[m], s[m]))

    # 6. Post-2000 studies
    m = sub["year"].values >= 2000
    rows.append(fit_row("year>=2000", X[m], s[m]))

    # 7. Admin data only
    m = sub["admin"].values == 1
    if m.sum() >= 10:
        rows.append(fit_row("admin-data only", X[m], s[m]))

    # 8. Quasi-experimental only (DID or RDD or RKD)
    m = sub["research_design"].isin(["DID", "RDD", "RKD"]).values
    rows.append(fit_row("quasi-exp only", X[m], s[m]))

    # 9. Leave-one-paper-out — worst-case swing in theta_hat
    papers = sub["paper_id"].unique()
    thetas = []
    for p in papers:
        mm = sub["paper_id"].values != p
        if mm.sum() < 5:
            continue
        fit = ak.ak_fit(X[mm], s[mm], cutoffs=(1.96,), symmetric=False, model="t")
        thetas.append((p, fit.x[0]))
    thetas.sort(key=lambda t: t[1])
    if thetas:
        rows.append({"check": f"LOO min theta (drop paper {thetas[0][0]})",
                     "n": len(X) - int((sub['paper_id']==thetas[0][0]).sum()),
                     "mean_eps": np.nan, "beta_p": np.nan,
                     "theta": thetas[0][1], "tau": np.nan, "df": np.nan})
        rows.append({"check": f"LOO max theta (drop paper {thetas[-1][0]})",
                     "n": len(X) - int((sub['paper_id']==thetas[-1][0]).sum()),
                     "mean_eps": np.nan, "beta_p": np.nan,
                     "theta": thetas[-1][1], "tau": np.nan, "df": np.nan})

    # 10. Normal distribution (no df free) — already in Table 2 but kept for context
    fit = ak.ak_fit(X, s, cutoffs=(1.96,), symmetric=False, model="normal")
    lam, tau, beta = fit.x
    rows.append({"check": "Normal dist", "n": len(X),
                 "mean_eps": X.mean(), "beta_p": beta, "theta": lam,
                 "tau": tau, "df": np.nan})

    return pd.DataFrame(rows)


def main():
    for margin in ["RR", "PBD"]:
        print(f"\n=== {margin} robustness ===")
        out = run_checks_for(margin)
        print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        out.to_csv(OUT / f"robustness_{margin}.csv", index=False)


if __name__ == "__main__":
    main()
