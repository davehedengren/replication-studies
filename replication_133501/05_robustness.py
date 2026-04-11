"""Robustness checks for Huh & Reif (2021).

We focus on the three headline estimates:
  1. ΔMVA (full sample)       — motor vehicle fatalities per 100k
  2. ΔMVA (female)             — paper emphasizes a 4.46 effect
  3. ΔPoisoning (female)       — paper's signature "surprising" finding (0.747)
"""
import numpy as np
import pandas as pd

from utils import (
    OUT, load_mortality, mean_before, rd_mse_opt, rd_ols_fixed_bw, tri_weights,
)


def rd_custom_bw(df, y, h):
    """rdrobust with user-specified common bandwidth, covs(firstmonth)."""
    from rdrobust import rdrobust
    r = rdrobust(
        y=df[y].values.astype(float),
        x=df["agemo_mda"].values.astype(float),
        c=0, p=1, kernel="triangular",
        covs=df[["firstmonth"]].values.astype(float),
        h=h, b=h * 1.5, all=True,
    )
    coef = np.asarray(r.coef.values).flatten()
    ci = np.asarray(r.ci.values)
    return {
        "conv": float(coef[0]),
        "ci": (float(ci[2, 0]), float(ci[2, 1])),
    }


def rd_nocovs(df, y):
    est = rd_mse_opt(df, y, covs=False)
    return {"conv": est["conv"], "ci": est["robust_ci"]}


def rd_drop_zero(df, y):
    """Drop the agemo_mda==0 month entirely instead of using the firstmonth
    dummy (alternative handling of the Dong 2015 measurement error)."""
    sub = df[df.agemo_mda != 0].copy()
    return rd_nocovs(sub, y)


def rd_uniform_kernel(df, y, bw=13):
    import statsmodels.api as sm
    mask = df.agemo_mda.between(-bw + 1, bw - 1)
    sub = df.loc[mask]
    X = np.column_stack([
        np.ones(len(sub)),
        sub.post.values,
        sub.agemo_mda.values,
        (sub.post.values * sub.agemo_mda.values),
        sub.firstmonth.values,
    ])
    m = sm.OLS(sub[y].values, X).fit(cov_type="HC1")
    return {
        "conv": float(m.params[1]),
        "ci": tuple(m.conf_int()[1]),
    }


def rd_poly(df, y, p):
    """MSE-optimal rdrobust with polynomial order p."""
    from rdrobust import rdrobust
    r = rdrobust(
        y=df[y].values.astype(float),
        x=df["agemo_mda"].values.astype(float),
        c=0, p=p, kernel="triangular",
        covs=df[["firstmonth"]].values.astype(float),
        all=True,
    )
    coef = np.asarray(r.coef.values).flatten()
    ci = np.asarray(r.ci.values)
    return {"conv": float(coef[0]), "ci": (float(ci[2, 0]), float(ci[2, 1]))}


def placebo_cutoffs(df, y, cutoffs=None):
    """RD at fake cutoffs on either side of the true cutoff."""
    from rdrobust import rdrobust
    if cutoffs is None:
        cutoffs = list(range(-36, -11)) + list(range(12, 37))
    results = []
    for c in cutoffs:
        sub = df.copy()
        if c < 0:
            sub = sub[sub.agemo_mda < 0]
        else:
            sub = sub[sub.agemo_mda > 0]
        if sub[y].isna().any():
            continue
        try:
            r = rdrobust(
                y=sub[y].values.astype(float),
                x=sub.agemo_mda.values.astype(float),
                c=c, p=1, kernel="triangular", all=True,
            )
            coef = float(np.asarray(r.coef.values).flatten()[0])
            results.append((c, coef))
        except Exception:
            pass
    return results


# ------------------------------------------------------------------
OUTCOMES = [
    ("none", "cod_MVA", "MVA (full)", 4.92),
    ("Female", "cod_MVA", "MVA (female)", 4.46),
    ("Female", "cod_sa_poisoning", "Poisoning (female)", 0.747),
    ("none", "cod_any", "All-cause (full)", 5.84),
]

print("=" * 92)
print("ROBUSTNESS CHECKS")
print("=" * 92)

rows = []
for scenario, y, label, baseline in OUTCOMES:
    df = load_mortality(scenario)
    print(f"\n--- {label} (scenario={scenario}, paper={baseline:.3f}) ---")

    checks = {
        "01 baseline (MSE-opt, firstmonth covs)": rd_mse_opt(df, y, covs=True),
        "02 no firstmonth covariate": rd_nocovs(df, y),
        "03 drop agemo==0 month": rd_drop_zero(df, y),
        "04 OLS bw=13 (Stata reg)": None,   # filled below
        "05 OLS bw=12": None,
        "06 OLS bw=24": None,
        "07 fixed h=8  (rdrobust)": rd_custom_bw(df, y, 8),
        "08 fixed h=12 (rdrobust)": rd_custom_bw(df, y, 12),
        "09 fixed h=24 (rdrobust)": rd_custom_bw(df, y, 24),
        "10 quadratic p=2":        rd_poly(df, y, 2),
        "11 cubic p=3":            rd_poly(df, y, 3),
        "12 uniform kernel bw=13": rd_uniform_kernel(df, y, bw=13),
    }

    for bw_label, bw_val in [("04 OLS bw=13 (Stata reg)", 13),
                             ("05 OLS bw=12", 12),
                             ("06 OLS bw=24", 24)]:
        est = rd_ols_fixed_bw(df, y, bw=bw_val)
        checks[bw_label] = {"conv": est["beta"], "ci": est["ci"]}

    for name, est in checks.items():
        if est is None:
            continue
        conv = est.get("conv") if "conv" in est else est["beta"]
        ci = est["ci"] if "ci" in est else est.get("robust_ci")
        sign_ok = " " if conv * baseline > 0 else "!"
        mag = abs(conv / baseline - 1) * 100 if baseline != 0 else 0
        rows.append({
            "scenario": scenario, "outcome": y, "check": name,
            "conv": conv, "ci_lo": ci[0], "ci_hi": ci[1], "pct_diff": mag,
        })
        print(f"  {name:<40} {conv:>9.3f}  [{ci[0]:>8.3f}, {ci[1]:>8.3f}]  "
              f"Δ={mag:>5.1f}% {sign_ok}")

    # Placebos
    plcbs = placebo_cutoffs(df, y)
    ests = np.array([c for _, c in plcbs])
    baseline_est = rd_mse_opt(df, y, covs=True)["conv"]
    rank = int((np.abs(ests) >= abs(baseline_est)).sum())
    pseudo_p = rank / len(ests)
    print(f"  13 placebo cutoffs: n={len(ests)}, median={np.median(ests):.3f}, "
          f"|effect|≥|baseline| in {rank}/{len(ests)} "
          f"(pseudo p={pseudo_p:.3f})")
    rows.append({
        "scenario": scenario, "outcome": y,
        "check": "13 placebo pseudo-p",
        "conv": pseudo_p, "ci_lo": np.nan, "ci_hi": np.nan,
        "pct_diff": np.nan,
    })

pd.DataFrame(rows).to_csv(OUT / "robustness.csv", index=False)
print("\nSaved robustness.csv")
