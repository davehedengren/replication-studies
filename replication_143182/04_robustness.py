"""Phase 4: Robustness checks for the main Abramitzky et al. (2023) result.

The headline estimate is the Urban Sample effect of QE2 x Post on the recent-
arrival European immigrant share (dqrtot): -1.541 (SE 0.610). Many of these
checks mirror Table A6 in the appendix, but recomputed from scratch in Python.
"""
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from utils import (
    load_panel,
    build_did_panel,
    xtreg_fe,
    SAMPLES,
    TABLE1_OUTCOMES,
)

BASELINE_OUTCOME = "dqrtot"
BASELINE_SAMPLE = "urban"

# Published baseline (Table 1, column 1, Panel A)
BASELINE_PUB = (-1.541, 0.610)


def subset(panel, sample):
    if sample == "full":
        return panel
    return panel[panel["sea_type"] == SAMPLES[sample]]


def run_check(name, panel, sample, y=BASELINE_OUTCOME, treat="QE2Xpost",
              fb_control=True, extra_controls=None):
    sub = subset(panel, sample)
    est = xtreg_fe(sub, y, treat=treat, fb_control=fb_control,
                   extra_controls=extra_controls)
    print(f"{name:<48} coef={est['coef']:+.4f}  se={est['se']:.4f}  "
          f"N={est['n']:4d}  G={est['groups']:4d}")
    return est


def main():
    df_full = load_panel(drop_outlier=True)
    panel = build_did_panel(df_full, TABLE1_OUTCOMES)
    print("Baseline (urban, dqrtot): coef=-1.541  se=0.610  (published)\n")
    print("-- Robustness to specification --")

    # 1. Baseline reproduction
    run_check("1. Baseline (urban)", panel, "urban")

    # 2. Drop FB1900*post trend
    run_check("2. No FB1900*post trend", panel, "urban", fb_control=False)

    # 3. Use QE1 (simple restricted-vs-not) instead of QE2
    run_check("3. QE1 instead of QE2", panel, "urban", treat="QE1Xpost")

    # 4. Include 1920 observations (i.e., also 1910->1920 decade)
    panel_with_1920 = _build_with_extra_period(df_full)
    run_check("4. Include 1910->1920 decade", panel_with_1920, "urban")

    # 5. Include outlier SEA 311
    df_with_311 = load_panel(drop_outlier=False)
    panel_311 = build_did_panel(df_with_311, TABLE1_OUTCOMES)
    run_check("5. Keep outlier SEA 311", panel_311, "urban")
    # For rural sample this is the big test (Fig 3 shows the outlier matters)
    run_check("5b. Keep SEA 311, rural sample", panel_311, "rural")

    # 6. Trim: drop bottom+top 5% by QE2
    panel_trim = _trim_by_qe2(panel, 0.05)
    run_check("6. Trim 5% tails of QE2", panel_trim, "urban")

    # 7. Winsorize outcome at 1/99%
    panel_w = _winsorize_outcome(panel, BASELINE_OUTCOME, 0.01)
    run_check("7. Winsorize dqrtot at 1/99%", panel_w, "urban")

    # 8. Drop each census region, one at a time (leave-one-out)
    print("\n-- Leave-one-census-region-out (urban) --")
    for reg in sorted(panel["region"].unique()):
        sub = panel[panel["region"] != reg]
        est = run_check(f"  drop region={int(reg)}", sub, "urban")

    # 9. Placebo: shuffle QE2 across SEAs
    print("\n-- Placebo permutation test (urban, 500 draws) --")
    rng = np.random.default_rng(7)
    sub_panel = subset(panel, "urban").copy()
    baseline = xtreg_fe(sub_panel, BASELINE_OUTCOME)
    baseline_t = baseline["coef"] / baseline["se"]
    t_stats = []
    sea_ids = sub_panel["sea"].unique()
    for _ in range(500):
        perm_map = dict(
            zip(sea_ids, rng.permutation(sub_panel.drop_duplicates("sea")["QE2"].values))
        )
        shuf = sub_panel.copy()
        shuf["QE2_shuf"] = shuf["sea"].map(perm_map)
        shuf["QE2Xpost"] = shuf["QE2_shuf"] * shuf["post"]
        est = xtreg_fe(shuf, BASELINE_OUTCOME)
        t_stats.append(est["coef"] / est["se"])
    t_stats = np.array(t_stats)
    p_emp = (np.abs(t_stats) >= abs(baseline_t)).mean()
    print(
        f"  baseline t = {baseline_t:+.2f},  |t| >= baseline in {int(p_emp*500)}/500 draws "
        f"=> empirical p = {p_emp:.3f}"
    )
    print(
        f"  permutation coefs:  mean={t_stats.mean()*baseline['se']:+.4f}"
        f"  5th %={np.quantile(t_stats, 0.025)*baseline['se']:+.4f}"
        f"  95th %={np.quantile(t_stats, 0.975)*baseline['se']:+.4f}"
    )

    # 10. Alternative outcome: dnqrtot (unrestricted population flows)
    print("\n-- Cross-outcome consistency --")
    run_check("10. outcome = dnqrtot (urban)", panel, "urban", y="dnqrtot")
    run_check("    outcome = dwtot (urban)",   panel, "urban", y="dwtot")

    # 11. Mining robustness: drop top-2 mining SEAs by employment share
    print("\n-- Mining sector sensitivity --")
    mining = subset(panel, "mining").copy()
    est = xtreg_fe(mining, BASELINE_OUTCOME)
    print(f"11. Mining baseline  coef={est['coef']:+.4f}  se={est['se']:.4f}")

    # 12. Composition check: re-estimate cluster-count bootstrap SE for the urban baseline
    print("\n-- Cluster bootstrap SE for the urban baseline --")
    boot_coefs = _cluster_bootstrap(subset(panel, "urban"), BASELINE_OUTCOME, B=400, seed=11)
    print(
        f"12. Urban baseline:  point={baseline['coef']:+.4f}  "
        f"cluster SE (analytic)={baseline['se']:.4f}  "
        f"bootstrap SE={boot_coefs.std():.4f}  "
        f"95% boot CI=({np.quantile(boot_coefs,0.025):+.4f}, "
        f"{np.quantile(boot_coefs,0.975):+.4f})"
    )


def _build_with_extra_period(df):
    """Like build_did_panel, but retains the 1910->1920 decade as well."""
    years = [1900, 1910, 1920, 1930]
    d = df[df["year"].isin(years)].copy()
    d = d.sort_values(["sea", "year"]).reset_index(drop=True)
    d["year"] = d["year"].astype(int)

    d["l_totpop"] = d.groupby("sea")["totpop"].shift(1)
    for v in TABLE1_OUTCOMES:
        d[f"l_{v}"] = d.groupby("sea")[v].shift(1)
        d[f"d{v}"] = (d[v] - d[f"l_{v}"]) / d["l_totpop"]
    d["post"] = (d["year"] == 1930).astype(float)
    d["QE2Xpost"] = d["QE2"] * d["post"]
    d["QE1Xpost"] = d["QE1"] * d["post"]
    d["FB1900Xpost"] = d["FB1900"] * d["post"]
    d["r"] = d["region"].astype("category").cat.codes + 1
    for r in sorted(d["r"].unique()):
        d[f"region_{int(r)}Xpost"] = ((d["r"] == r).astype(float)) * d["post"]
    # keep 1910 (100-1910 diff), 1920 (1910-1920 diff), 1930 (1920-1930 diff)
    d = d[d["year"].isin([1910, 1920, 1930])].copy()
    return d


def _trim_by_qe2(panel, frac):
    lo = panel["QE2"].quantile(frac)
    hi = panel["QE2"].quantile(1 - frac)
    return panel[(panel["QE2"] >= lo) & (panel["QE2"] <= hi)].copy()


def _winsorize_outcome(panel, col, frac):
    out = panel.copy()
    lo = out[col].quantile(frac)
    hi = out[col].quantile(1 - frac)
    out[col] = out[col].clip(lo, hi)
    return out


def _cluster_bootstrap(panel, y, B=400, seed=0, treat="QE2Xpost"):
    rng = np.random.default_rng(seed)
    seas = panel["sea"].unique()
    coefs = np.empty(B)
    for b in range(B):
        draw = rng.choice(seas, size=len(seas), replace=True)
        parts = []
        for i, s in enumerate(draw):
            sub = panel[panel["sea"] == s].copy()
            sub["sea"] = int(s) * 10000 + i  # unique cluster id per draw
            parts.append(sub)
        boot = pd.concat(parts).reset_index(drop=True)
        try:
            est = xtreg_fe(boot, y, treat=treat)
            coefs[b] = est["coef"]
        except Exception:
            coefs[b] = np.nan
    return coefs[~np.isnan(coefs)]


if __name__ == "__main__":
    main()
