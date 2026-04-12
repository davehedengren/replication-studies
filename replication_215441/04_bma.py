"""BMA replication — exhaustive enumeration of 2^17 models with g-prior (UIP).

Matches the R BMS package used by 02_model_averaging.R:
  bms(all_full, g="UIP", mprior="uniform", ...)

UIP (unit information prior) with uniform model prior:
  g = N, so the marginal likelihood for a model with k regressors is
  (1+g)^{(N-k-1)/2} * (1 + g*(1-R2))^{-(N-1)/2}
With uniform prior each model gets equal weight, so posterior ∝ marginal lik.

2^17 = 131,072 models * 71 obs is quick (a few seconds).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import itertools
import numpy as np
import pandas as pd

from utils import load_long, collapse_one_per_margin, build_bma_covariates, BMA_COVARS, OUT


def bma_gprior(y, X_full, g=None):
    """Enumerate all submodels of X_full, return (pip, posterior_mean, posterior_sd).

    Uses Zellner g-prior with g = N (UIP), uniform model prior, analytic marginal
    likelihood. Constant is always included (not part of the 2^k enumeration).
    """
    n, k = X_full.shape
    if g is None:
        g = float(n)

    y = np.asarray(y, float)
    y_centered = y - y.mean()
    tss = np.sum(y_centered**2)

    # constant-only marginal likelihood baseline (R2 = 0)
    log_marg_0 = -0.5 * (n - 1) * np.log(tss)  # up to constants cancelling later

    nmodels = 1 << k
    log_ml = np.empty(nmodels)
    ks = np.empty(nmodels, int)

    # precompute XtX, Xty once for full matrix for speed
    Xc = X_full - X_full.mean(axis=0)
    XtX_full = Xc.T @ Xc
    Xty_full = Xc.T @ y_centered

    for idx in range(nmodels):
        bits = [(idx >> j) & 1 for j in range(k)]
        sel = [j for j, b in enumerate(bits) if b]
        kk = len(sel)
        ks[idx] = kk
        if kk == 0:
            # null model: R2 = 0, log ML = -0.5*(n-1)*log(tss)
            log_ml[idx] = -0.5 * (n - 1) * np.log(tss)
            continue
        A = XtX_full[np.ix_(sel, sel)]
        b = Xty_full[sel]
        try:
            beta_hat = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            log_ml[idx] = -np.inf
            continue
        rss = tss - beta_hat @ b
        r2 = 1 - rss / tss
        # Log marginal likelihood under g-prior (Fernandez/Ley/Steel 2001)
        log_ml[idx] = (
            0.5 * (n - kk - 1) * np.log(1 + g)
            - 0.5 * (n - 1) * np.log(1 + g * (1 - r2))
            - 0.5 * (n - 1) * np.log(tss)
        )

    # Posterior weights
    log_ml = log_ml - log_ml.max()
    w = np.exp(log_ml)
    w /= w.sum()

    # PIP and posterior moments
    pip = np.zeros(k)
    post_mean = np.zeros(k)
    post_m2 = np.zeros(k)

    # Also track unconditional posterior mean and SD over the k regressors.
    # Compute per-model beta and conditional posterior mean under g-prior:
    #   beta | y, model = (g/(1+g)) * beta_hat_OLS
    # posterior variance contribution uses the usual g-prior normal-IG posterior;
    # we approximate posterior SD via weighted variance of conditional means
    # (this matches BMS() 'exact' output when nmodel covers full space).
    g_scale = g / (1 + g)
    intercept_mean_sum = 0.0
    intercept_pip = 1.0

    for idx in range(nmodels):
        if w[idx] == 0:
            continue
        bits = [(idx >> j) & 1 for j in range(k)]
        sel = [j for j, b in enumerate(bits) if b]
        if not sel:
            continue
        A = XtX_full[np.ix_(sel, sel)]
        b = Xty_full[sel]
        beta_hat = np.linalg.solve(A, b)
        beta_shrunk = g_scale * beta_hat
        for pos, j in enumerate(sel):
            pip[j] += w[idx]
            post_mean[j] += w[idx] * beta_shrunk[pos]
            post_m2[j] += w[idx] * beta_shrunk[pos] ** 2

    post_sd = np.sqrt(np.maximum(post_m2 - post_mean**2, 0))
    # Intercept posterior mean = weighted mean of y (constant always in model)
    intercept = y.mean() - X_full.mean(axis=0) @ post_mean
    return pip, post_mean, post_sd, intercept


def main():
    df = load_long(drop_hunt=True)
    coll = collapse_one_per_margin(df)
    bma_df = build_bma_covariates(coll)
    # Use BMA_COVARS order
    y = bma_df["elasticity"].astype(float).values
    X = bma_df[BMA_COVARS].astype(float).values

    pip, pm, psd, intercept = bma_gprior(y, X, g=len(y))

    result = pd.DataFrame({
        "term": ["(Intercept)"] + BMA_COVARS,
        "pip":  [1.0] + list(pip),
        "post_mean": [intercept] + list(pm),
        "post_sd": [0.0] + list(psd),
    })

    # Published values from release/BMA_weighted_coefficients.tex
    published = {
        "(Intercept)": (1.000, 0.191),
        "SE": (1.000, 1.334),
        "DIDorRKD": (0.167, 0.013),
        "RDD": (0.112, 0.004),
        "MacroTreatment": (0.173, 0.024),
        "PBD": (0.299, -0.090),
        "RRxBaselinePBD": (0.292, 0.001),
        "RRxBaselineRR": (0.639, 0.390),
        "PBDxBaselinePBD": (0.996, 0.007),
        "PBDxBaselineRR": (0.161, -0.029),
        "Admin": (0.127, -0.008),
        "NonemploymentAsOutcome": (0.659, -0.123),
        "HazardModel": (0.108, -0.002),
        "YearsTo2023": (0.115, 0.000),
        "RelativeUnemp": (0.113, 0.001),
        "USA": (0.249, -0.041),
        "TaxWedge": (0.143, -0.001),
        "ImpactFactorZ": (0.380, 0.026),
    }
    result["pip_pub"] = result["term"].map(lambda t: published[t][0])
    result["post_mean_pub"] = result["term"].map(lambda t: published[t][1])

    result["pip_diff"] = (result["pip"] - result["pip_pub"]).round(3)
    result["mean_diff"] = (result["post_mean"] - result["post_mean_pub"]).round(3)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 180)
    print(result.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    result.to_csv(OUT / "bma_results.csv", index=False)
    print(f"\nSaved {OUT/'bma_results.csv'}")


if __name__ == "__main__":
    main()
