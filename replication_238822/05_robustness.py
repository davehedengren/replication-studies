"""Phase 4: robustness checks for the main result (Table 1, column 1).

The headline result is that digitization caused a Rs.6.57 fall in tax collected
per acre — a 47% decline relative to the control mean (Rs.14.2). All checks
below test the same TWFE OLS specification with district + year FE and cluster
SE at the district level, modifying the sample, definition, or functional form.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from utils import load, OUT


def drop_singletons(df, unit, time):
    while True:
        n0 = len(df)
        cu = df.groupby(unit).size()
        ct = df.groupby(time).size()
        df = df[df[unit].isin(cu[cu > 1].index) & df[time].isin(ct[ct > 1].index)]
        if len(df) == n0:
            return df


def twfe(df, y, x, unit, time, cluster):
    cols = list(dict.fromkeys([y, x, unit, time, cluster]))
    d = df[cols].dropna().copy()
    d = drop_singletons(d, unit, time)
    if len(d) == 0:
        return np.nan, np.nan, 0, np.nan
    ud = pd.get_dummies(d[unit], prefix="u", drop_first=True).astype(float)
    td = pd.get_dummies(d[time], prefix="t", drop_first=True).astype(float)
    X = pd.concat([pd.Series(1.0, index=d.index, name="const"),
                   d[[x]].astype(float), ud, td], axis=1)
    try:
        res = sm.OLS(d[y].astype(float).values, X.values).fit(
            cov_type="cluster", cov_kwds={"groups": d[cluster].values}
        )
        idx = list(X.columns).index(x)
        return float(res.params[idx]), float(res.bse[idx]), int(res.nobs), float(res.pvalues[idx])
    except Exception:
        return np.nan, np.nan, len(d), np.nan


def main():
    out = []

    def log(msg=""):
        print(msg)
        out.append(str(msg))

    log("======================================================================")
    log(" ROBUSTNESS — Table 1 col 1 baseline: tax_acres = -6.57 (SE 3.69)")
    log("======================================================================")
    tax = load("Tax_District_Yr.dta")

    b, se, n, p = twfe(tax, "tax_acres", "digitization", "district", "fin_yr", "district")
    log(f"\n[Baseline]                          beta={b:7.3f}  se={se:.3f}  N={n}  p={p:.3f}")

    # 1. Alternative timing: 12% threshold (uses digitization_12pc)
    log("\n[1] Alternative threshold: 12% villages digitized triggers treatment")
    b, se, n, p = twfe(tax, "tax_acres", "digitization_12pc", "district", "fin_yr", "district")
    log(f"    digitization_12pc                beta={b:7.3f}  se={se:.3f}  N={n}  p={p:.3f}")

    # 2. Drop largest tax outlier districts (top 3 districts by mean tax_acres)
    big = tax.groupby("district")["tax_acres"].mean().nlargest(3).index
    sub = tax[~tax["district"].isin(big)]
    b, se, n, p = twfe(sub, "tax_acres", "digitization", "district", "fin_yr", "district")
    log(f"\n[2] Drop top-3 highest-mean tax districts beta={b:7.3f}  se={se:.3f}  N={n}  p={p:.3f}")

    # 3. Drop smallest tax outlier districts (bottom 3)
    small = tax.groupby("district")["tax_acres"].mean().nsmallest(3).index
    sub = tax[~tax["district"].isin(small)]
    b, se, n, p = twfe(sub, "tax_acres", "digitization", "district", "fin_yr", "district")
    log(f"[3] Drop bottom-3 districts          beta={b:7.3f}  se={se:.3f}  N={n}  p={p:.3f}")

    # 4. Winsorize tax_acres at 1% and 99%
    lo, hi = tax["tax_acres"].quantile([0.01, 0.99])
    w = tax.copy()
    w["tax_acres"] = w["tax_acres"].clip(lo, hi)
    b, se, n, p = twfe(w, "tax_acres", "digitization", "district", "fin_yr", "district")
    log(f"[4] Winsorize tax_acres at 1/99 pct  beta={b:7.3f}  se={se:.3f}  N={n}  p={p:.3f}")

    # 5. Trim top 5%
    cutoff = tax["tax_acres"].quantile(0.95)
    sub = tax[tax["tax_acres"] <= cutoff]
    b, se, n, p = twfe(sub, "tax_acres", "digitization", "district", "fin_yr", "district")
    log(f"[5] Trim top 5% of tax_acres         beta={b:7.3f}  se={se:.3f}  N={n}  p={p:.3f}")

    # 6. Log(1+y) transform
    w = tax.copy()
    w["lny"] = np.log1p(w["tax_acres"].clip(lower=0))
    b, se, n, p = twfe(w, "lny", "digitization", "district", "fin_yr", "district")
    log(f"[6] log(1+tax_acres) outcome         beta={b:7.3f}  se={se:.3f}  N={n}  p={p:.3f}")

    # 7. Drop phase 1 (most homogenous treatment cohort)
    sub = tax[tax["phase"] != 1.0]
    b, se, n, p = twfe(sub, "tax_acres", "digitization", "district", "fin_yr", "district")
    log(f"\n[7] Drop phase 1 districts           beta={b:7.3f}  se={se:.3f}  N={n}  p={p:.3f}")

    # 8. Drop phase 2 districts
    sub = tax[tax["phase"] != 2.0]
    b, se, n, p = twfe(sub, "tax_acres", "digitization", "district", "fin_yr", "district")
    log(f"[8] Drop phase 2 districts           beta={b:7.3f}  se={se:.3f}  N={n}  p={p:.3f}")

    # 9. Restrict to FY <= 2012 (only first treatment year visible)
    sub = tax[tax["fin_yr"] <= 2012]
    b, se, n, p = twfe(sub, "tax_acres", "digitization", "district", "fin_yr", "district")
    log(f"[9] Restrict to fin_yr <= 2012       beta={b:7.3f}  se={se:.3f}  N={n}  p={p:.3f}")

    # 10. Leave-one-district-out (LOO): show min, max, and mean coefficient
    b, se, n, p = twfe(tax, "tax_acres", "digitization", "district", "fin_yr", "district")
    base = b
    loo = []
    for d in tax["district"].unique():
        sub = tax[tax["district"] != d]
        bi, _, _, _ = twfe(sub, "tax_acres", "digitization", "district", "fin_yr", "district")
        loo.append((int(d), bi))
    loo_df = pd.DataFrame(loo, columns=["dropped_district", "beta"])
    log(f"\n[10] Leave-one-district-out (36 reps): "
        f"min={loo_df['beta'].min():.3f}  max={loo_df['beta'].max():.3f}  "
        f"mean={loo_df['beta'].mean():.3f}  median={loo_df['beta'].median():.3f}")
    log(f"     all 36 LOO betas remain negative? "
        f"{(loo_df['beta'] < 0).sum()}/{len(loo_df)} (baseline {base:.3f})")
    most_inf = loo_df.loc[loo_df['beta'].abs().idxmin(), 'dropped_district']
    log(f"     district whose removal weakens effect most: {int(most_inf)}")

    # 11. Placebo permutation: shuffle phase across districts 500 times
    rng = np.random.default_rng(125)
    phase_map = tax[["district", "phase"]].drop_duplicates().set_index("district")["phase"]
    phases = phase_map.values
    districts = phase_map.index.values
    base_b, _, _, _ = twfe(tax, "tax_acres", "digitization", "district", "fin_yr", "district")
    null = []
    for r in range(500):
        shuffled = rng.permutation(phases)
        # construct placebo digitization based on shuffled phase
        pmap = dict(zip(districts, shuffled))
        sim = tax.copy()
        sim_phase = sim["district"].map(pmap)
        # original digitization rule: phase 1 treated FY>=2012, phase 2 treated FY>=2013
        sim["digi_p"] = ((sim_phase == 1.0) & (sim["fin_yr"] >= 2012)).astype(int) | \
                        ((sim_phase == 2.0) & (sim["fin_yr"] >= 2013)).astype(int)
        bi, _, _, _ = twfe(sim, "tax_acres", "digi_p", "district", "fin_yr", "district")
        null.append(bi)
    null = np.array(null)
    pval = float(np.mean(np.abs(null) >= abs(base_b)))
    log(f"\n[11] Permutation placebo (500 reps shuffling phase across districts):")
    log(f"     baseline beta={base_b:.3f},  null mean={null.mean():.3f},  "
        f"std={null.std():.3f},  two-sided p={pval:.3f}")

    # 12. Heterogeneity: split sample by pre-period tax level (above/below median)
    pre_means = tax[tax["fin_yr"] <= 2011].groupby("district")["tax_acres"].mean()
    high = pre_means[pre_means >= pre_means.median()].index
    sub_h = tax[tax["district"].isin(high)]
    sub_l = tax[~tax["district"].isin(high)]
    bH, seH, nH, _ = twfe(sub_h, "tax_acres", "digitization", "district", "fin_yr", "district")
    bL, seL, nL, _ = twfe(sub_l, "tax_acres", "digitization", "district", "fin_yr", "district")
    log(f"\n[12] Heterogeneity by pre-period tax level:")
    log(f"     above median pre-tax  beta={bH:7.3f}  se={seH:.3f}  N={nH}")
    log(f"     below median pre-tax  beta={bL:7.3f}  se={seL:.3f}  N={nL}")

    (OUT / "05_robustness.txt").write_text("\n".join(out))


if __name__ == "__main__":
    main()
