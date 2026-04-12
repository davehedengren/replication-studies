"""Robustness checks for replication 172903.

The paper is a review/descriptive article: there are no regression tables to
perturb. We instead stress-test each figure's construction choices.

Checks:
 1. Fig 1: alternative year filtering (keep 1966; restrict 2000–2020).
 2. Fig 1: qualitative growth split pre-2010 vs post-2010.
 3. Fig 3 time: alternative wave selection (2012 only; 2013 only; pooled mean).
 4. Fig 3 risk: same alternative wave selection.
 5. Fig 3/4: alternative histogram bins (even-width deciles).
 6. Fig 4 dictator: show raw 0-6 count distribution (no normalisation).
 7. Fig 4B: OLS age-trend test on proportion shared.
 8. Fig 4B: OLS age-trend test on P(share exactly half).
 9. Fig 4B: exclude the [6.5,10) bucket (bucket is plotted at x=6.5 in paper).
10. Fig 4B: bootstrap SEs for the by-age means (small-sample sanity check).
11. Fig 3A zero-mass: decompose 0-bar contribution by source wave.
12. Fig 1: check release-vs-paper 8-paper discrepancy by year.
"""
import numpy as np
import pandas as pd
from statsmodels.formula.api import ols

from utils import load_papers, load_experimental, first_nonmissing, OUT

RNG = np.random.default_rng(20210520)


def fig1_checks():
    papers = load_papers()
    lines = ["--- Fig 1 robustness ---"]
    lines.append(f"(1a) keep 1966, drop nothing: N = {len(papers)}")
    lines.append(f"(1b) drop 1966 (paper default): N = {(papers['year']!=1966).sum()}")
    lines.append(f"(1c) restrict to 2000..2020 : N = {((papers['year']>=2000)&(papers['year']<=2020)).sum()}")
    lines.append(f"(appendix A.5 claim         : N = 257)")
    yr = papers.loc[papers["year"] != 1966, "year"]
    lines.append("")
    lines.append(f"(2) pre-2010 count = {(yr < 2010).sum()}, "
                 f"post-2010 count = {(yr >= 2010).sum()}")
    lines.append(f"    2000-2009 share = {(yr<2010).mean():.3f}, "
                 f"2010-2020 share = {(yr>=2010).mean():.3f}")
    return "\n".join(lines)


def wave_selection_checks():
    exp = load_experimental()
    lines = ["--- Wave selection robustness (Fig 3) ---"]
    time_paper = first_nonmissing(exp, ["child_2010_all", "child_2012_all", "child_2013_all"])
    risk_paper = first_nonmissing(exp, ["child_2010_risk_all", "child_2012_risk_all", "child_2013_risk_all"])

    for name, tcol, rcol in [
        ("2010 only", "child_2010_all", "child_2010_risk_all"),
        ("2012 only", "child_2012_all", "child_2012_risk_all"),
        ("2013 only", "child_2013_all", "child_2013_risk_all"),
    ]:
        t = exp[tcol].dropna()
        r = exp[rcol].dropna()
        lines.append(f"{name}: time N={len(t)}, mean={t.mean():.4f}, "
                     f"P(=0)={(t==0).mean():.4f}, P(=1)={(t==1).mean():.4f}")
        lines.append(f"{'':10}risk N={len(r)}, mean={r.mean():.4f}, "
                     f"P(=0)={(r==0).mean():.4f}, P(=1)={(r==1).mean():.4f}")

    t = time_paper.dropna()
    r = risk_paper.dropna()
    lines.append(f"paper    : time N={len(t)}, mean={t.mean():.4f}, "
                 f"P(=0)={(t==0).mean():.4f}, P(=1)={(t==1).mean():.4f}")
    lines.append(f"{'':10}risk N={len(r)}, mean={r.mean():.4f}, "
                 f"P(=0)={(r==0).mean():.4f}, P(=1)={(r==1).mean():.4f}")

    # Pooled mean across any non-missing waves (alt selection)
    tmean = exp[["child_2010_all","child_2012_all","child_2013_all"]].mean(axis=1).dropna()
    rmean = exp[["child_2010_risk_all","child_2012_risk_all","child_2013_risk_all"]].mean(axis=1).dropna()
    lines.append(f"pooled   : time N={len(tmean)}, mean={tmean.mean():.4f}, "
                 f"P(=0)={(tmean==0).mean():.4f}, P(=1)={(tmean==1).mean():.4f}")
    lines.append(f"{'':10}risk N={len(rmean)}, mean={rmean.mean():.4f}, "
                 f"P(=0)={(rmean==0).mean():.4f}, P(=1)={(rmean==1).mean():.4f}")
    return "\n".join(lines)


def alt_bin_hist():
    exp = load_experimental()
    lines = ["--- Alternative histogram bins (even-width deciles) ---"]
    time = first_nonmissing(exp, ["child_2010_all", "child_2012_all", "child_2013_all"]).dropna()
    risk = first_nonmissing(exp, ["child_2010_risk_all", "child_2012_risk_all", "child_2013_risk_all"]).dropna()
    d2012 = (exp["dictator_giving2012"] / 6).dropna()
    bins = np.arange(0, 1.11, 0.1)
    for name, s in [("time", time), ("risk", risk), ("dictator", d2012)]:
        counts, _ = np.histogram(s, bins=bins)
        pct = counts / counts.sum() * 100
        dist = ", ".join(f"[{bins[i]:.1f},{bins[i+1]:.1f}):{pct[i]:.1f}" for i in range(len(pct)))
        lines.append(f"{name:9s} N={len(s)}: {dist}")
    return "\n".join(lines)


def dictator_raw_counts():
    exp = load_experimental()
    d = exp["dictator_giving2012"].dropna()
    counts = d.value_counts().sort_index()
    pct = counts / len(d) * 100
    lines = ["--- Fig 4A raw 0-6 distribution (no proportion binning) ---"]
    for k in sorted(counts.index):
        lines.append(f"  shared {int(k)}/6: n={int(counts[k])}, pct={pct[k]:.2f}")
    lines.append(f"  total N = {len(d)}")
    lines.append(f"  mean stickers shared = {d.mean():.4f}")
    return "\n".join(lines)


def age_trend_tests():
    exp = load_experimental()
    d = exp["dictator_giving2012"] / 6
    age = exp["age_in_month_at_exp_2012"] / 12
    half = (d == 0.5).astype(float)
    half[d.isna()] = np.nan
    df = pd.DataFrame({"d": d, "age": age, "half": half}).dropna()
    lines = ["--- Age-trend OLS on Fig 4B outcomes ---"]
    for outcome in ["d", "half"]:
        m = ols(f"{outcome} ~ age", data=df).fit(cov_type="HC1")
        lines.append(f"  {outcome:4s} ~ age : beta={m.params['age']:.4f}, "
                     f"se={m.bse['age']:.4f}, t={m.tvalues['age']:.2f}, "
                     f"p={m.pvalues['age']:.4f}, N={int(m.nobs)}")
    # Also linear trend on just 4 to 6.5 window
    sub = df[df["age"] < 6.5]
    for outcome in ["d", "half"]:
        m = ols(f"{outcome} ~ age", data=sub).fit(cov_type="HC1")
        lines.append(f"  [4,6.5) {outcome:4s} ~ age : beta={m.params['age']:.4f}, "
                     f"se={m.bse['age']:.4f}, p={m.pvalues['age']:.4f}, N={int(m.nobs)}")
    return "\n".join(lines)


def bootstrap_fig4b():
    exp = load_experimental()
    d = exp["dictator_giving2012"] / 6
    age = exp["age_in_month_at_exp_2012"] / 12
    half = (d == 0.5).astype(float)
    half[d.isna()] = np.nan

    bins = [4, 4.5, 5, 5.5, 6, 6.5, 10]
    labels = [4, 4.5, 5, 5.5, 6, 6.5]
    age_cat = pd.cut(age, bins=bins, right=False, labels=labels, ordered=False)
    df = pd.DataFrame({"d": d, "h": half, "age_cat": age_cat}).dropna()

    lines = ["--- Bootstrap (1000 reps) SEs for Fig 4B by age bin ---"]
    lines.append(f"{'age':>5} {'mean_d':>8} {'se_d_anal':>10} {'se_d_boot':>10} "
                 f"{'mean_h':>8} {'se_h_anal':>10} {'se_h_boot':>10} {'N':>5}")
    for a in labels:
        sub = df[df["age_cat"] == a]
        if len(sub) == 0:
            continue
        n = len(sub)
        se_d_anal = sub["d"].std() / np.sqrt(n)
        se_h_anal = sub["h"].std() / np.sqrt(n)
        boot_d, boot_h = [], []
        idx = np.arange(n)
        for _ in range(1000):
            pick = RNG.choice(idx, size=n, replace=True)
            boot_d.append(sub["d"].values[pick].mean())
            boot_h.append(sub["h"].values[pick].mean())
        lines.append(f"{a:>5} {sub['d'].mean():>8.4f} {se_d_anal:>10.4f} "
                     f"{np.std(boot_d):>10.4f} {sub['h'].mean():>8.4f} "
                     f"{se_h_anal:>10.4f} {np.std(boot_h):>10.4f} {n:>5d}")
    return "\n".join(lines)


def exclude_overage_bin():
    exp = load_experimental()
    d = exp["dictator_giving2012"] / 6
    age = exp["age_in_month_at_exp_2012"] / 12
    half = (d == 0.5).astype(float)
    half[d.isna()] = np.nan

    df = pd.DataFrame({"d": d, "h": half, "age": age}).dropna()
    sub = df[df["age"] < 6.5]
    labels = [4, 4.5, 5, 5.5, 6]
    sub["age_cat"] = pd.cut(sub["age"], bins=[4, 4.5, 5, 5.5, 6, 6.5], right=False, labels=labels, ordered=False)
    lines = ["--- Fig 4B excluding 6.5+ bucket (which spans [6.5,10) in paper) ---"]
    lines.append(f"{'age':>5} {'mean_d':>8} {'mean_h':>8} {'N':>5}")
    for a in labels:
        g = sub[sub["age_cat"] == a]
        lines.append(f"{a:>5} {g['d'].mean():>8.4f} {g['h'].mean():>8.4f} {len(g):>5d}")
    return "\n".join(lines)


def zero_mass_by_wave():
    exp = load_experimental()
    lines = ["--- Fig 3A zero-mass decomposition by source wave ---"]
    for col in ["child_2010_all", "child_2012_all", "child_2013_all"]:
        s = exp[col].dropna()
        lines.append(f"  {col}: N={len(s)}, P(time=0)={(s==0).mean():.4f}")
    return "\n".join(lines)


def main():
    blocks = [
        fig1_checks(),
        wave_selection_checks(),
        alt_bin_hist(),
        dictator_raw_counts(),
        age_trend_tests(),
        bootstrap_fig4b(),
        exclude_overage_bin(),
        zero_mass_by_wave(),
    ]
    out = "\n\n".join(blocks)
    (OUT / "robustness.txt").write_text(out)
    print(out)
    print()
    print("wrote:", OUT / "robustness.txt")


if __name__ == "__main__":
    main()
