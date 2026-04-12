"""Data audit for the Evdokimov–Garfagnini cognition dataset."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import build_analysis_df, load_raw

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)

lines: list[str] = []
def log(msg=""):
    print(msg)
    lines.append(msg)

raw = load_raw()
d = build_analysis_df()

log("=" * 72)
log("COVERAGE")
log("=" * 72)
log(f"Total observations:      {len(d):,}")
log(f"Unique subjects:         {d['id'].nunique()}")
log(f"Unique sessions:         {d['session'].nunique()}")
log(f"Periods per subject:     min={d.groupby('id')['period'].count().min()}  "
    f"max={d.groupby('id')['period'].count().max()}")
log(f"Periods cover 1..30:     {sorted(d['period'].unique())[:5]}..."
    f"{sorted(d['period'].unique())[-5:]}")
log(f"oTree version mix (new): {d[d['period']==1]['new'].value_counts().to_dict()}")
log(f"Treatment counts (p=1):  {d[d['period']==1]['treatment_name'].value_counts().to_dict()}")

per_subj_rows = d.groupby("id").size()
log(f"All subjects have 30 rows? {(per_subj_rows == 30).all()}")


log("")
log("=" * 72)
log("VARIABLE COMPLETENESS")
log("=" * 72)
miss = d.isna().sum().sort_values(ascending=False)
for c, n in miss.items():
    if n > 0:
        log(f"  {c:22s}  {n:6d}  ({n/len(d)*100:.2f}%)")
log("  (guess_own_score/guess_partner_score missing outside InformedOwn — expected)")
log("  (partner_type missing in Baseline — expected)")
log("  (dir_pdis missing when guess1 == 0.5 — by construction)")


log("")
log("=" * 72)
log("DISTRIBUTIONAL SUMMARY — key numeric variables")
log("=" * 72)
desc_cols = ["guess1", "guess2", "test_score", "payoff", "total_time",
             "rt1", "rt2", "pdis", "adis", "dir_pdis", "bayesianBeliefs"]
desc = d[desc_cols].describe(percentiles=[.01, .05, .5, .95, .99]).T
log(desc.round(4).to_string())


log("")
log("=" * 72)
log("PLAUSIBILITY CHECKS")
log("=" * 72)
log(f"guess1 in [0,1]?  {((d['guess1']>=0)&(d['guess1']<=1)).all()}")
log(f"guess2 in [0,1]?  {((d['guess2']>=0)&(d['guess2']<=1)).all()}")
log(f"test_score in 0..7?  {d['test_score'].between(0, 7).all()}")
log(f"orange in {{0,1}}?  {set(d['orange'].unique()) <= {0, 1}}")
log(f"urn in {{0,1}}?  {set(d['urn'].unique()) <= {0, 1}}")
log(f"pdis in [0,1]?  {((d['pdis']>=0)&(d['pdis']<=1)).all()}")
# Payoff check — binarized scoring rule gives [0,1] per paid decision
log(f"payoff range: [{d['payoff'].min():.3f}, {d['payoff'].max():.3f}]")
log(f"total_time > 0 throughout? {(d['total_time']>0).all()}")


log("")
log("=" * 72)
log("LOGICAL CONSISTENCY")
log("=" * 72)
# Cumulative orange balls should equal #periods that drew an orange
check = d.groupby("id").apply(
    lambda df: (df["cumOrangeBalls"] == df["orange"].cumsum()).all()
).all()
log(f"cumOrangeBalls matches running sum of orange for every subject: {check}")

# Bayesian beliefs bounded in [0,1]
log(f"bayesianBeliefs in [0,1]: "
    f"{((d['bayesianBeliefs']>=0)&(d['bayesianBeliefs']<=1)).all()}")

# Share of orange balls conditional on urn ~ qO/qP
orange_given_urn = d.groupby("urn")["orange"].mean().to_dict()
log(f"P(orange | urn=1) = {orange_given_urn.get(1,0):.3f}   [design: 2/3 ≈ 0.667]")
log(f"P(orange | urn=0) = {orange_given_urn.get(0,0):.3f}   [design: 1/3 ≈ 0.333]")


log("")
log("=" * 72)
log("PARTNER MATCHING")
log("=" * 72)
# Groups in main treatments should have exactly 2 members per period
sizes = d[d["main_treatments"] == 1].groupby(["session", "group", "period"]).size()
log(f"Group-period sizes in main treatments: "
    f"min={sizes.min()} max={sizes.max()} mean={sizes.mean():.3f}")
log(f"All size-2 pairs: {(sizes == 2).all()}")

own = d[d["informed_own"] == 1]
if len(own):
    sizes_own = own.groupby(["session", "group", "period"]).size()
    log(f"InformedOwn group-period sizes: min={sizes_own.min()} "
        f"max={sizes_own.max()} mean={sizes_own.mean():.3f}")

# Partner beliefs should be in [0,1] where matched
log(f"guess_partner NA rate: {d['guess_partner'].isna().mean()*100:.2f}%")


log("")
log("=" * 72)
log("DUPLICATES")
log("=" * 72)
dup_id_period = d.duplicated(subset=["id", "period"]).sum()
log(f"Duplicate (id, period) rows: {dup_id_period}")
dup_full = d.duplicated().sum()
log(f"Fully duplicated rows: {dup_full}")


log("")
log("=" * 72)
log("MISSING DATA BY TREATMENT")
log("=" * 72)
for t in ["Baseline", "InformedTop", "InformedBottom", "InformedOwn"]:
    sub = d[d["treatment_name"] == t]
    log(f"  {t:15s}  N_rows={len(sub):5d}  "
        f"rt1_na%={sub['rt1'].isna().mean()*100:.2f}  "
        f"rt2_na%={sub['rt2'].isna().mean()*100:.2f}")


log("")
log("=" * 72)
log("OUTLIER SCAN")
log("=" * 72)
for col in ["total_time", "rt1", "rt2"]:
    s = d[col].dropna()
    iqr = s.quantile(0.75) - s.quantile(0.25)
    hi = s.quantile(0.75) + 1.5 * iqr
    log(f"  {col}: P50={s.median():.0f} P95={s.quantile(.95):.0f} "
        f"max={s.max():.0f} n>P75+1.5IQR={int((s > hi).sum())}")

non_updaters = d.groupby("id")["guess1"].apply(lambda g: (g == 0.5).all()).sum()
log(f"Non-updaters (all 30 periods guess1=0.5): {non_updaters} of {d['id'].nunique()}")


(OUT / "02_data_audit.log").write_text("\n".join(lines))
print(f"\nWrote {OUT/'02_data_audit.log'}")
