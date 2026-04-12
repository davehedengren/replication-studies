"""Re-estimate the Pareto-lognormal (EMG in log) parameters from SCF 1989 & 2016.

The paper assumes log y = x1 + x2 where x1 ~ N(mu_y, sigma^2), x2 ~ Exp(alpha),
so log y is EMG-distributed with params (mu, sigma, lambda=alpha). MLE is fit
using the SCF tabulated (income, count) data for ages 40-59, conditional on
y > $10k (the paper drops very low incomes as in their Excel routine).
"""
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_scf_tabulated, DATA, OUT


def emg_logpdf(z, mu, sigma, lam):
    # EMG(mu, sigma^2, lambda) pdf for log-income z:
    # f(z) = lam * exp(lam*(mu - z) + lam^2 sigma^2 / 2) * Phi((z - mu)/sigma - lam*sigma)
    sigma = abs(sigma)
    lam = abs(lam)
    a = np.log(lam) + lam * (mu - z) + 0.5 * lam * lam * sigma * sigma
    z_arg = (z - mu) / sigma - lam * sigma
    b = norm.logcdf(z_arg)
    return a + b


def emg_logsf_trunc(z_trunc, mu, sigma, lam):
    # log P(Z >= z_trunc) for EMG(mu,sigma,lam); used for left-truncation at log(10k).
    # Use P(Z >= t) = 1 - CDF(t). CDF(t) = Phi((t-mu)/sigma) - exp(lam*(mu-t)+lam^2*sigma^2/2)*Phi((t-mu)/sigma - lam*sigma)
    sigma = abs(sigma)
    lam = abs(lam)
    t = z_trunc
    phi1 = norm.cdf((t - mu) / sigma)
    phi2 = norm.cdf((t - mu) / sigma - lam * sigma)
    cdf = phi1 - np.exp(lam * (mu - t) + 0.5 * lam * lam * sigma * sigma) * phi2
    sf = 1.0 - cdf
    return np.log(max(sf, 1e-300))


def neg_loglik(params, z, w):
    # Untruncated EMG NLL (matching Cai & Heathcote's Excel Solver routine,
    # which fits the marginal EMG density without a left-truncation correction).
    mu, log_sigma, log_lam = params
    sigma = np.exp(log_sigma)
    lam = np.exp(log_lam)
    ll = emg_logpdf(z, mu, sigma, lam)
    return -np.sum(w * ll)


def fit_emg(tab, y_min=10_000.0):
    # Drop rows with income <= y_min (matches the 'mean(log y | y>10k)' sample
    # annotated in the SCF_AER_submit.xlsx '2016' / '1989' sheets).
    tab = tab[tab["income"] > y_min].copy()
    z = np.log(tab["income"].values)
    w = tab["count"].values
    w = w / w.sum()
    mu0 = np.sum(w * z)
    s0 = np.sqrt(np.sum(w * (z - mu0) ** 2))
    x0 = np.array([mu0, np.log(0.7 * s0), np.log(2.0)])
    best = None
    for log_lam0 in (np.log(1.5), np.log(2.0), np.log(3.0)):
        x0[2] = log_lam0
        res = minimize(
            neg_loglik,
            x0,
            args=(z, w),
            method="Nelder-Mead",
            options={"xatol": 1e-9, "fatol": 1e-11, "maxiter": 50000, "maxfev": 50000},
        )
        if best is None or res.fun < best.fun:
            best = res
    mu, log_sigma, log_lam = best.x
    sigma2 = np.exp(log_sigma) ** 2
    lam = np.exp(log_lam)
    return dict(mu=mu, sigma2=sigma2, alpha=lam, var_log_y=sigma2 + 1.0 / lam ** 2, nll=best.fun)


def paper_published_ml():
    """Values Cai & Heathcote report in the '2016' and '1989' sheets (ML column)."""
    return {
        2016: dict(mu=10.554835, sigma2=0.547499, alpha=1.667315, var_log_y=0.907219),
        1989: dict(mu=10.744489, sigma2=0.478096, alpha=2.401270, var_log_y=0.651523),
    }


def main():
    print("=" * 70)
    print("03_scf_params.py — EMG fit of SCF log income, 1989 vs 2016")
    print("=" * 70)

    pub = paper_published_ml()
    rows = []
    for year in (1989, 2016):
        tab = load_scf_tabulated(year)
        fit = fit_emg(tab, y_min=10_000.0)
        p = pub[year]
        print(f"\n{year} (N_bins = {len(tab)})")
        print(f"  mu (Python MLE):        {fit['mu']:.6f}   paper sheet (ML): {p['mu']:.6f}")
        print(f"  sigma^2 (Python MLE):   {fit['sigma2']:.6f}   paper sheet (ML): {p['sigma2']:.6f}")
        print(f"  alpha   (Python MLE):   {fit['alpha']:.6f}   paper sheet (ML): {p['alpha']:.6f}")
        print(f"  var log y:              {fit['var_log_y']:.6f}   paper sheet (ML): {p['var_log_y']:.6f}")
        rows.append({
            "year": year,
            "mu_paper": p["mu"], "mu_repl": fit["mu"],
            "sigma2_paper": p["sigma2"], "sigma2_repl": fit["sigma2"],
            "alpha_paper": p["alpha"], "alpha_repl": fit["alpha"],
            "varlogy_paper": p["var_log_y"], "varlogy_repl": fit["var_log_y"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "table5_scf_params.csv", index=False)
    print(f"\nSaved {OUT / 'table5_scf_params.csv'}")

    # Paper Table 5 reports: 2016 sigma2=0.548, alpha=1.67; 1989 sigma2=0.478, alpha=2.40
    print("\nPaper Table 5 (rounded):  2016 sigma^2=0.548, alpha=1.67;  1989 sigma^2=0.478, alpha=2.40")


if __name__ == "__main__":
    main()
