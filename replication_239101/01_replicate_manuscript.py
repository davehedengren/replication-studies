"""Replicate core empirical results from analysis_manuscript.R.

Reproduces the introductory statistics, Tables 1-4, and the Kolmogorov-Smirnov
tests reported on pp. 10-20 of the manuscript.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import build_analysis_df, cluster_ols

OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_DIR.mkdir(exist_ok=True)

d = build_analysis_df()
main = d[d["informed_own"] == 0].copy()   # Baseline + InformedTop + InformedBottom
main_p1 = main[main["period"] == 1].copy()

lines: list[str] = []

def log(msg: str = ""):
    print(msg)
    lines.append(msg)


# ----------------------------------------------------------------------------
log("=" * 72)
log("INTRODUCTION — page 5")
log("=" * 72)

sub = d[(d["informed_own"] == 1) & (d["period"] == 1) & (d["test_score"] < 4)]
pct = sub["guess_own_score"].mean() * 100
log(f"Avg guess about own score, low-scorers (test_score<4): {pct:.1f}%  [paper: 71.5%]")


# ----------------------------------------------------------------------------
log("")
log("=" * 72)
log("SECTION II — page 10: sample sizes, payoffs, test scores")
log("=" * 72)

n_main = main["id"].nunique()
log(f"Number of subjects in main treatments: {n_main}  [paper: ~672]")

counts = main_p1["treatment_name"].value_counts().to_dict()
log("Participants per treatment (period 1): " + str(counts))

med_wage = main_p1["hourly_wage"].median()
log(f"Median hourly wage: ${med_wage:.2f}")

med_mins = (main_p1["total_time"].median()) / 60
log(f"Median total time: {med_mins:.0f} minutes")

med_ts = main_p1["test_score"].median()
log(f"Median test score (main): {med_ts}")

for t in ["Baseline", "InformedTop", "InformedBottom"]:
    m = main[main["treatment_name"] == t]["test_score"].median()
    log(f"  {t}: median test_score = {m}")


# ----------------------------------------------------------------------------
log("")
log("=" * 72)
log("TABLE 1 — First-order belief accuracy")
log("=" * 72)

r1, n1, g1 = cluster_ols(
    "dist_bayes ~ informed_top + informed_bottom + test_score", main, cluster_col="id"
)
log(f"Col 1 (dist from Bayes):  N={n1}, clusters={g1}")
for k in ["Intercept", "informed_top", "informed_bottom", "test_score"]:
    log(f"  {k:16s}  b={r1.params[k]: .4f}  SE={r1.bse[k]:.4f}  p={r1.pvalues[k]:.4g}")

r2, n2, g2 = cluster_ols(
    "guess1_truth ~ informed_top + informed_bottom + test_score", main, cluster_col="id"
)
log(f"Col 2 (normalized guess1): N={n2}, clusters={g2}")
for k in ["Intercept", "informed_top", "informed_bottom", "test_score"]:
    log(f"  {k:16s}  b={r2.params[k]: .4f}  SE={r2.bse[k]:.4f}  p={r2.pvalues[k]:.4g}")

log(f"\np10: test_score coef on dist_bayes: p = {r1.pvalues['test_score']:.3g}"
    f"  [paper: P<0.001]")
log(f"p10: test_score coef on normalized g1: p = {r2.pvalues['test_score']:.3g}"
    f"  [paper: P<0.001]")
smallest = min(r1.pvalues["informed_top"], r1.pvalues["informed_bottom"],
               r2.pvalues["informed_top"], r2.pvalues["informed_bottom"])
log(f"p10: smallest treatment p-value across both cols: {smallest:.3f}  [paper: 0.493]")


# ----------------------------------------------------------------------------
log("")
log("=" * 72)
log("FOOTNOTE 12 — share of first-order guesses at 0.5")
log("=" * 72)
pct_half = (main["guess1"] == 0.5).mean() * 100
log(f"% of first-order beliefs equal to 0.5: {pct_half:.1f}%")

dummy05 = (main["guess1"] == 0.5).astype(int)
all_half = main.groupby("id")["guess1"].apply(lambda s: (s == 0.5).all())
n_non = all_half.sum()
log(f"# subjects always reporting guess1=0.5 (main sample): {n_non} / {main['id'].nunique()}")

share_both_half = ((main["guess1"] == 0.5) & (main["guess2"] == 0.5)).sum() / (
    main["guess1"] == 0.5).sum()
log(f"Share where guess1=0.5 & guess2=0.5: {share_both_half*100:.1f}%  [paper: 71.4%]")

log("")
for t in ["Baseline", "InformedTop", "InformedBottom"]:
    sub = main[main["treatment_name"] == t]
    diff = (sub["guess1"] != sub["guess2"]).mean() * 100
    log(f"  Share guess1 ≠ guess2 in {t}: {diff:.1f}%")


# ----------------------------------------------------------------------------
log("")
log("=" * 72)
log("TABLE 2 — Mean perceived disagreement & direction by treatment")
log("=" * 72)

for t in ["Baseline", "InformedTop", "InformedBottom"]:
    sub = main_p1[main_p1["treatment_name"] == t]
    log(f"  {t:15s}  pdis={sub['mean_pdis'].mean():.4f}"
        f"  dirpdis={sub['mean_dirpdis'].mean():.4f}")

log("\nKS tests, period=1, subject-level means:")
base_pdis = main_p1[main_p1["baseline"] == 1]["mean_pdis"].dropna()
bot_pdis = main_p1[main_p1["informed_bottom"] == 1]["mean_pdis"].dropna()
top_pdis = main_p1[main_p1["informed_top"] == 1]["mean_pdis"].dropna()

base_dir = main_p1[main_p1["baseline"] == 1]["mean_dirpdis"].dropna()
bot_dir = main_p1[main_p1["informed_bottom"] == 1]["mean_dirpdis"].dropna()
top_dir = main_p1[main_p1["informed_top"] == 1]["mean_dirpdis"].dropna()

ks_pd_bb = stats.ks_2samp(base_pdis, bot_pdis).pvalue
ks_dir_bb = stats.ks_2samp(base_dir, bot_dir).pvalue
ks_pd_bt = stats.ks_2samp(base_pdis, top_pdis).pvalue
ks_dir_bt = stats.ks_2samp(base_dir, top_dir).pvalue

log(f"  Base vs Bottom — pdis:   KS p = {ks_pd_bb:.4g}")
log(f"  Base vs Bottom — dirpdis: KS p = {ks_dir_bb:.4g}")
log(f"  Larger of two p-vals (paper: < 0.01): {max(ks_pd_bb, ks_dir_bb):.4g}")
log(f"  Base vs Top    — pdis:   KS p = {ks_pd_bt:.4g}")
log(f"  Base vs Top    — dirpdis: KS p = {ks_dir_bt:.4g}")
log(f"  Smallest of two p-vals (paper: 0.917): {min(ks_pd_bt, ks_dir_bt):.4g}")


# ----------------------------------------------------------------------------
log("")
log("=" * 72)
log("TABLE 3 — Treatment effects on disagreement measures")
log("=" * 72)

r3a, n3a, _ = cluster_ols(
    "pdis ~ informed_top + informed_bottom + test_score", main)
r3b, n3b, _ = cluster_ols(
    "dir_pdis ~ informed_top + informed_bottom + test_score", main)
r3c, n3c, _ = cluster_ols(
    "adis ~ informed_top + informed_bottom + test_score + ts_gap", main)
# column 4: adis - pdis on the same sample (informed != 2 already == main)
main = main.assign(adis_m_pdis=lambda df: df["adis"] - df["pdis"])
r3d, n3d, _ = cluster_ols(
    "adis_m_pdis ~ informed_top + informed_bottom + test_score + ts_gap", main)

tbl3 = []
for name, res, n in [
    ("Perceived dis.", r3a, n3a),
    ("Direction", r3b, n3b),
    ("Actual dis.", r3c, n3c),
    ("Actual − perceived", r3d, n3d),
]:
    row = {"col": name, "N": n}
    for k in ["informed_top", "informed_bottom", "test_score"]:
        if k in res.params.index:
            row[k] = f"{res.params[k]:.4f}"
            row[k + "_p"] = f"{res.pvalues[k]:.4g}"
    tbl3.append(row)

for row in tbl3:
    log(str(row))

smallest_top = min(r3a.pvalues["informed_top"], r3b.pvalues["informed_top"],
                   r3c.pvalues["informed_top"], r3d.pvalues["informed_top"])
log(f"\np15: smallest p-value for InformedTop across Table 3: {smallest_top:.3g}"
    f"  [paper: 0.696]")

p_bot_pdis = r3a.pvalues["informed_bottom"]
log(f"p15: InformedBottom on pdis, p = {p_bot_pdis:.3g}  [paper: <0.05]")

# Simple ATE without test score
r_ate, *_ = cluster_ols("pdis ~ informed_top + informed_bottom", main)
ate_bot = r_ate.params["informed_bottom"] * 100
intercept = r_ate.params["Intercept"]
log(f"ATE of InformedBottom on perceived disagreement: {ate_bot:.1f} pp  [paper: 3.4 pp]")
pct_increase = r_ate.params["informed_bottom"] / intercept * 100
log(f"Relative increase vs Baseline: {pct_increase:.1f}%  [paper: ~30%]")

# KS test of pdis vs adis within each treatment (p16)
log("\nKS: perceived vs actual disagreement by treatment (p16, paper: all <0.001)")
for t in ["Baseline", "InformedTop", "InformedBottom"]:
    sub = main_p1[main_p1["treatment_name"] == t]
    p = stats.ks_2samp(sub["mean_pdis"].dropna(), sub["mean_adis"].dropna()).pvalue
    log(f"  {t:15s}  p = {p:.3e}")


# ----------------------------------------------------------------------------
log("")
log("=" * 72)
log("TABLE 4 — Perceived disagreement by cognitive type")
log("=" * 72)

main = main.assign(high_crt=(main["test_score"] >= 4).astype(int),
                   last15=(main["period"] >= 15).astype(int))
main["high_x_last15"] = main["high_crt"] * main["last15"]

r4a, n4a, _ = cluster_ols("pdis ~ high_crt", main)
r4b, n4b, _ = cluster_ols("pdis ~ high_crt + last15 + high_x_last15", main)
r4c, n4c, _ = cluster_ols("dir_pdis ~ high_crt", main)
r4d, n4d, _ = cluster_ols("dir_pdis ~ high_crt + last15 + high_x_last15", main)

log(f"Col 1 pdis ~ HighCRT:  b={r4a.params['high_crt']:.4f}"
    f"  p={r4a.pvalues['high_crt']:.3g}")
log(f"  abs pp = {abs(r4a.params['high_crt'])*100:.2f}  [paper: 4.85]")
log(f"Col 3 dir_pdis ~ HighCRT: b={r4c.params['high_crt']:.4f}"
    f"  p={r4c.pvalues['high_crt']:.3g}  [paper: 2.5 pp, p<0.05]")


# ----------------------------------------------------------------------------
log("")
log("=" * 72)
log("KS tests between cognitive types within treatment (pp.19-20)")
log("=" * 72)

def ks_by_type(df_, var):
    lo = df_[df_["test_score"] < 4][var].dropna()
    hi = df_[df_["test_score"] >= 4][var].dropna()
    return stats.ks_2samp(lo, hi).pvalue

for t, col in [("Baseline", "baseline"),
               ("InformedTop", "informed_top"),
               ("InformedBottom", "informed_bottom")]:
    sub = main_p1[main_p1[col] == 1]
    p_pd = ks_by_type(sub, "mean_pdis")
    p_dir = ks_by_type(sub, "mean_dirpdis")
    log(f"  {t:15s}  pdis KS p={p_pd:.3g}   dirpdis KS p={p_dir:.3g}")


# ----------------------------------------------------------------------------
(OUT_DIR / "01_replicate_manuscript.log").write_text("\n".join(lines))
print(f"\nWrote {OUT_DIR/'01_replicate_manuscript.log'}")
