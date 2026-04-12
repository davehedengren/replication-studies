"""Replicate the bar-chart / mean comparisons that feed Figure 2 and the
paper's narrative on piece-rate, consequential-work, match-return, and gift
treatment effects.

Reference targets from DellaVigna, List, Malmendier, Rao (2022):
 - Piece rate 0c -> 20c : ~ +4 envelopes (+12 percent).
 - Envelopes used vs not used (round 5 vs 6 in Order A) : +3.5 envelopes (+10 percent).
 - High vs low return match (rounds 7 vs 8) : +0.6 envelopes (+1.7 percent), not significant.
 - Gift vs control (rounds 9 & 10, no controls):
     Positive  : +0.45 envelopes, not sig.
     Negative  : ~0, not sig.
     In-kind   : -1.15 envelopes, not sig.
"""

import numpy as np
import pandas as pd
from utils import clean_long, ibn_regression, cluster_ttest, OUT

long, _ = clean_long()


def section(title):
    bar = "=" * 70
    print(f"\n{bar}\n{title}\n{bar}")


def fmt_row(row):
    return f"  type={int(row['type']):<2}  mean={row['mean']:7.3f}  se={row['se']:.3f}  [{row['ci_low']:7.3f}, {row['ci_high']:7.3f}]"


# ---------------------------------------------------------------------------
# Panel A : Piece rate comparison
# Uses round 7 (order A) / 3 (order B) for 0c, round 3(A)/7(B) for 10c, 6(A)/4(B) for 20c.
# ---------------------------------------------------------------------------
section("FIGURE 2a — Piece rate comparison (0c vs 10c vs 20c)")

d = long.copy()
d["type"] = np.nan
d.loc[((d["time"] == 7) & (d["order"] == 1)) | ((d["time"] == 3) & (d["order"] == 2)), "type"] = 1
d.loc[((d["time"] == 3) & (d["order"] == 1)) | ((d["time"] == 7) & (d["order"] == 2)), "type"] = 3
d.loc[((d["time"] == 6) & (d["order"] == 1)) | ((d["time"] == 4) & (d["order"] == 2)), "type"] = 5
sub = d.dropna(subset=["type"]).copy()
tbl, _ = ibn_regression(sub["output"], sub["type"], sub["group"])
for _, row in tbl.iterrows():
    print(fmt_row(row))
p_10_vs_0, _ = cluster_ttest(sub[sub["type"].isin([1, 3])]["output"],
                              sub[sub["type"].isin([1, 3])]["type"],
                              sub[sub["type"].isin([1, 3])]["group"])
p_20_vs_0, _ = cluster_ttest(sub[sub["type"].isin([1, 5])]["output"],
                              sub[sub["type"].isin([1, 5])]["type"],
                              sub[sub["type"].isin([1, 5])]["group"])
diff_20_0 = tbl.loc[tbl["type"] == 5, "mean"].iloc[0] - tbl.loc[tbl["type"] == 1, "mean"].iloc[0]
pct_20_0 = 100 * diff_20_0 / tbl.loc[tbl["type"] == 1, "mean"].iloc[0]
print(f"  20c - 0c diff = {diff_20_0:+.3f} envelopes ({pct_20_0:+.1f}%)")
print(f"  p-values: 10c=0c -> {p_10_vs_0:.3f}   20c=0c -> {p_20_vs_0:.3f}")

# ---------------------------------------------------------------------------
# Panel B : Envelopes used vs not used  (round 5 vs round 6)
# Round 5 = training at 20c (letters discarded), round 6 = real work at 20c for firm.
# Only Order A has this comparison (round 5 is training in order A only).
# Stata code does it pooled: type=1 if time==5, type=3 if time==6.
# ---------------------------------------------------------------------------
section("FIGURE 2b — Envelopes used vs not used (round 5 vs round 6)")

d = long.copy()
d["type"] = np.nan
d.loc[d["time"] == 5, "type"] = 1
d.loc[d["time"] == 6, "type"] = 3
sub = d.dropna(subset=["type"]).copy()
tbl, _ = ibn_regression(sub["output"], sub["type"], sub["group"])
for _, row in tbl.iterrows():
    print(fmt_row(row))
p_used, _ = cluster_ttest(sub["output"], sub["type"], sub["group"])
diff = tbl.loc[tbl["type"] == 3, "mean"].iloc[0] - tbl.loc[tbl["type"] == 1, "mean"].iloc[0]
pct = 100 * diff / tbl.loc[tbl["type"] == 1, "mean"].iloc[0]
print(f"  used - not-used diff = {diff:+.3f} envelopes ({pct:+.1f}%)")
print(f"  p-value: {p_used:.3f}")

# ---------------------------------------------------------------------------
# Panel C : High vs low return (match vs no-match, rounds 7 vs 8)
# Cross-session, both orders: type=1 if time==7, type=3 if time==8.
# ---------------------------------------------------------------------------
section("FIGURE 2c — High vs low return (round 7 low-return vs round 8 match)")

d = long.copy()
d["type"] = np.nan
d.loc[d["time"] == 7, "type"] = 1
d.loc[d["time"] == 8, "type"] = 3
sub = d.dropna(subset=["type"]).copy()
tbl, _ = ibn_regression(sub["output"], sub["type"], sub["group"])
for _, row in tbl.iterrows():
    print(fmt_row(row))
p_ret, _ = cluster_ttest(sub["output"], sub["type"], sub["group"])
diff = tbl.loc[tbl["type"] == 3, "mean"].iloc[0] - tbl.loc[tbl["type"] == 1, "mean"].iloc[0]
pct = 100 * diff / tbl.loc[tbl["type"] == 1, "mean"].iloc[0]
print(f"  match - no-match diff = {diff:+.3f} envelopes ({pct:+.1f}%)")
print(f"  p-value: {p_ret:.3f}")

# ---------------------------------------------------------------------------
# Panel D : Gift effects (rounds 9 and 10)  no-controls specification.
# ---------------------------------------------------------------------------
section("FIGURE 2d — Gift effects in rounds 9 & 10 (no controls)")

d = long[(long["time"] == 9) | (long["time"] == 10)].copy()
d["type"] = np.nan
d.loc[d["d_neu_gift"] == 1, "type"] = 1
d.loc[d["d_pos_gift"] == 1, "type"] = 3
d.loc[d["d_neg_gift"] == 1, "type"] = 5
d.loc[d["d_knd_gift"] == 1, "type"] = 7
sub = d.dropna(subset=["type"]).copy()
tbl, _ = ibn_regression(sub["output"], sub["type"], sub["group"])
for _, row in tbl.iterrows():
    print(fmt_row(row))
ctrl_mean = tbl.loc[tbl["type"] == 1, "mean"].iloc[0]
for treat_type, label in [(3, "Positive"), (5, "Negative"), (7, "In-kind")]:
    diff = tbl.loc[tbl["type"] == treat_type, "mean"].iloc[0] - ctrl_mean
    p_val, _ = cluster_ttest(
        sub[sub["type"].isin([1, treat_type])]["output"],
        sub[sub["type"].isin([1, treat_type])]["type"],
        sub[sub["type"].isin([1, treat_type])]["group"],
    )
    print(f"  {label:9s} - Control = {diff:+.3f} envelopes   p={p_val:.3f}")

# Save a tidy CSV of the gift-effects means for the writeup.
tbl.to_csv(OUT / "fig2d_gift_effects_means.csv", index=False)
print(f"\nWrote {OUT/'fig2d_gift_effects_means.csv'}")
