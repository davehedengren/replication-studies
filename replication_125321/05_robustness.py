"""Robustness checks for the main PM10 RD estimate (Table 1 Panel A, col 2).
Baseline: residualized PM10 on running variable T, triangle kernel, p=1 q=2,
Calonico MSE-optimal BW, city-clustered SEs — published 34.9 ug/m3 (5.8)."""

import numpy as np
import pandas as pd
from rdrobust import rdrobust
from utils import OUT, WEATHER_VARS, residualize

BASELINE = 34.9
SE_BASE = 5.8


def rd(df, y="resid_pm10_smw", x="T", cluster_col="code_city", **kw):
    sub = df[[y, x, cluster_col]].dropna()
    params = dict(c=0, p=1, q=2, kernel="tri", stdvars=True, masspoints="off")
    params.update(kw)
    res = rdrobust(y=sub[y], x=sub[x], cluster=sub[cluster_col], **params)
    return {
        "coef": float(res.coef.iloc[0, 0]),
        "se": float(res.se.iloc[0, 0]),
        "n": int(res.N_h[0] + res.N_h[1]),
        "bw": float(res.bws.iloc[0, 0]),
    }


def line(label, r, base=BASELINE):
    diff = r["coef"] - base
    pct = 100 * diff / base
    print(f"{label:<52} coef={r['coef']:7.2f}  se={r['se']:5.2f}  "
          f"n={r['n']:>7,d}  bw={r['bw']:6.1f}  (Δ={diff:+.1f}, {pct:+.0f}%)")


def main():
    print("=" * 90)
    print("ROBUSTNESS CHECKS — PM10 RD estimate (baseline 34.9 ug/m3)")
    print("=" * 90)

    sd = pd.read_parquet(OUT / "station_day_rd.parquet")
    sd["resid_pm10_smw"] = residualize(
        sd, "pm10", fe_cols=["pm10_n", "month"], controls=WEATHER_VARS
    )
    sd["l_pm10"] = np.log(sd["pm10"].replace(0, np.nan))

    print("\n--- Baseline reproduction ---")
    line("Baseline (pub 34.9)", rd(sd))

    print("\n--- 1. Alt kernel (uniform) ---")
    line("Uniform kernel", rd(sd, kernel="uni"))
    print("\n--- 2. Alt kernel (Epanechnikov) ---")
    line("Epanechnikov kernel", rd(sd, kernel="epa"))

    print("\n--- 3. Alt polynomial order ---")
    line("Local quadratic (p=2)", rd(sd, p=2, q=3))

    print("\n--- 4. Fixed narrower/wider bandwidths ---")
    line("h=60 days",  rd(sd, h=60))
    line("h=180 days", rd(sd, h=180))
    line("h=365 days", rd(sd, h=365))

    print("\n--- 5. Placebo: pre-period only, cutoff at -180 ---")
    # Use only t in [-360,0], fake cutoff at T=-180
    pre = sd[(sd["T"] >= -360) & (sd["T"] < 0)].copy()
    pre["T_fake"] = pre["T"] + 180  # new cutoff at 0
    line("Placebo (pre-period, fake cut -180)",
         rd(pre, x="T_fake"))

    print("\n--- 6. Placebo: post-period only, cutoff at +180 ---")
    post = sd[(sd["T"] >= 0) & (sd["T"] <= 360)].copy()
    post["T_fake"] = post["T"] - 180
    line("Placebo (post-period, fake cut +180)",
         rd(post, x="T_fake"))

    print("\n--- 7. Winsorize PM10 at 1%/99% before residualizing ---")
    sd2 = sd.copy()
    lo, hi = sd2["pm10"].quantile([.01, .99])
    sd2["pm10"] = sd2["pm10"].clip(lo, hi)
    sd2["resid_pm10_smw"] = residualize(
        sd2, "pm10", fe_cols=["pm10_n", "month"], controls=WEATHER_VARS
    )
    line("Winsorized 1/99", rd(sd2))

    print("\n--- 8. Log PM10 as outcome (residualized) ---")
    sd3 = sd.copy()
    sd3["resid_lpm10"] = residualize(
        sd3, "l_pm10", fe_cols=["pm10_n", "month"], controls=WEATHER_VARS
    )
    r = rd(sd3, y="resid_lpm10")
    print(f"{'Log PM10, residualized':<52} coef={r['coef']:+.4f}  "
          f"se={r['se']:.4f}  n={r['n']:,d}  bw={r['bw']:.1f}  "
          f"(≈{r['coef']*100:+.1f}%)")

    print("\n--- 9. Drop Wave-2 and estimate on Wave-1 only ---")
    line("Wave 1 only", rd(sd[sd.phase == 1]))
    print("--- 10. Drop Wave-1 and estimate on Wave-2 only ---")
    line("Wave 2 only", rd(sd[sd.phase == 2]))

    print("\n--- 11. Leave-one-wave-out: drop all Wave-1 deadline stations ---")
    line("Drop auto_date==19359", rd(sd[sd.auto_date != 19359]))
    print("--- 12. Leave-one-wave-out: drop all Wave-2 deadline stations ---")
    line("Drop auto_date==19724", rd(sd[sd.auto_date != 19724]))

    print("\n--- 13. Alternative cluster: station-level ---")
    sub = sd.dropna(subset=["resid_pm10_smw", "T"])
    res = rdrobust(
        y=sub["resid_pm10_smw"], x=sub["T"], c=0, p=1, q=2, kernel="tri",
        cluster=sub["pm10_n"], masspoints="off", stdvars=True,
    )
    r = {"coef": float(res.coef.iloc[0, 0]), "se": float(res.se.iloc[0, 0]),
         "n": int(res.N_h[0] + res.N_h[1]), "bw": float(res.bws.iloc[0, 0])}
    line("Cluster at station (not city)", r)

    print("\n--- 14. Drop 5% most polluted cities (pre-auto mean) ---")
    city_pre_mean = (sd[sd.date < sd.auto_date]
                     .groupby("code_city")["pm10"].mean())
    cutoff = city_pre_mean.quantile(0.95)
    keep = city_pre_mean[city_pre_mean <= cutoff].index
    sd_drop = sd[sd.code_city.isin(keep)]
    line("Drop top-5% polluted cities", rd(sd_drop))

    print("\n--- 15. Parametric OLS within h=109 (no bias correction) ---")
    bw = 109
    sub = sd[(sd["T"].abs() <= bw)].dropna(subset=["resid_pm10_smw", "T"]).copy()
    sub["I"] = (sub["T"] >= 0).astype(int)
    sub["T_pos"] = sub["T"] * sub["I"]
    w = (1 - sub["T"].abs() / bw).to_numpy()
    X = np.column_stack([np.ones(len(sub)), sub["I"].to_numpy(),
                          sub["T"].to_numpy(), sub["T_pos"].to_numpy()])
    Y = sub["resid_pm10_smw"].to_numpy()
    Xw = X * np.sqrt(w)[:, None]
    Yw = Y * np.sqrt(w)
    beta, *_ = np.linalg.lstsq(Xw, Yw, rcond=None)
    resid = Y - X @ beta
    # City-clustered SEs (CR1 Stata-style)
    groups = sub["code_city"].to_numpy()
    uniq = np.unique(groups)
    K = X.shape[1]; N = len(sub); G = len(uniq)
    XtWX_inv = np.linalg.inv(Xw.T @ Xw)
    meat = np.zeros((K, K))
    Xwu = Xw * (resid * np.sqrt(w))[:, None]
    for g in uniq:
        mask_g = groups == g
        s = Xwu[mask_g].sum(axis=0)
        meat += np.outer(s, s)
    dof = (G / (G - 1)) * ((N - 1) / (N - K))
    V = dof * XtWX_inv @ meat @ XtWX_inv
    se_I = np.sqrt(V[1, 1])
    print(f"Parametric OLS (tri-weighted, h=109):       coef={beta[1]:7.2f}  "
          f"se={se_I:5.2f}  n={N:,d}")


if __name__ == "__main__":
    main()
