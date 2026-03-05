"""
04_data_audit.py - Data quality audit for Callen & Long (2015)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import *

print("=" * 70)
print("DATA AUDIT: Callen & Long (2015)")
print("=" * 70)

# ── Load data ──────────────────────────────────────────────────────────────
df_vd = load_vote_diff()
df_pc = load_pc()

# ═══════════════════════════════════════════════════════════════════════════
# 1. COVERAGE
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("1. COVERAGE")
print("=" * 70)

print(f"\n--- master_vote_difference_native ---")
print(f"Total observations: {len(df_vd)}")
print(f"Unique polling centers: {df_vd['pc'].nunique()}")
n_ps = df_vd.groupby(['pc', 'pollingstation']).ngroups
print(f"Unique polling substations: {n_ps}")
print(f"Unique candidates (cand_id_s): {df_vd['cand_id_s'].nunique()}")
print(f"Unique provinces (provid398): {df_vd['provid398'].nunique()}")
print(f"Province IDs: {sorted(df_vd['provid398'].unique())}")

print(f"\n--- master_pc_native ---")
print(f"Total polling centers: {len(df_pc)}")
print(f"Treatment: {int(df_pc['treat_lib'].sum())}, Control: {int((1-df_pc['treat_lib']).sum())}")

print(f"\n--- Variable completeness (vote_diff sample) ---")
vd = df_vd[df_vd['vote_diff'].notna()].copy()
for col in ['picture_votes_1st', 'votes', 'vote_diff', 'cand_data', 'cand_id_s',
            'treat_lib', 'strata_group', 'pashtun', 'tajik']:
    n_valid = vd[col].notna().sum()
    pct = n_valid / len(vd) * 100
    print(f"  {col:<25} {n_valid:>8} / {len(vd)} ({pct:.1f}%)")

# ═══════════════════════════════════════════════════════════════════════════
# 2. DISTRIBUTIONS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("2. DISTRIBUTIONS")
print("=" * 70)

print("\n--- Key outcome variables ---")
for var in ['votes', 'picture_votes_1st', 'vote_diff', 'abs_diff']:
    col = df_vd[var].dropna()
    print(f"\n  {var}:")
    print(f"    N={len(col)}, Mean={col.mean():.3f}, SD={col.std():.3f}")
    print(f"    Min={col.min():.1f}, P1={col.quantile(0.01):.1f}, P5={col.quantile(0.05):.1f}, "
          f"Median={col.median():.1f}, P95={col.quantile(0.95):.1f}, P99={col.quantile(0.99):.1f}, Max={col.max():.1f}")

    # IQR-based outliers
    q1 = col.quantile(0.25)
    q3 = col.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    n_outliers = ((col < lower) | (col > upper)).sum()
    print(f"    IQR outliers (1.5x): {n_outliers} ({n_outliers/len(col)*100:.1f}%)")

print("\n--- Top 10 extreme vote_diff values ---")
vd_top = df_vd[df_vd['vote_diff'].notna()].nlargest(10, 'vote_diff')[['pc', 'pollingstation', 'cand_id_s', 'vote_diff', 'votes', 'picture_votes_1st']]
print(vd_top.to_string(index=False))

print("\n--- Bottom 10 extreme vote_diff values ---")
vd_bot = df_vd[df_vd['vote_diff'].notna()].nsmallest(10, 'vote_diff')[['pc', 'pollingstation', 'cand_id_s', 'vote_diff', 'votes', 'picture_votes_1st']]
print(vd_bot.to_string(index=False))

# Connection variable frequencies
print("\n--- Connection variable frequencies ---")
for var in CONNECTION_VARS + ['cand_data', 'powerful']:
    n1 = (df_vd[var] == 1).sum()
    pct = n1 / len(df_vd) * 100
    n_cands = df_vd.loc[df_vd[var] == 1, 'cand_id_s'].nunique()
    print(f"  {var:<25} N=1: {n1:>8} ({pct:.2f}%), unique candidates: {n_cands}")

# ═══════════════════════════════════════════════════════════════════════════
# 3. LOGICAL CONSISTENCY
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("3. LOGICAL CONSISTENCY")
print("=" * 70)

# vote_diff should equal votes - picture_votes_1st
vd_check = df_vd[df_vd['vote_diff'].notna()].copy()
computed_diff = vd_check['votes'] - vd_check['picture_votes_1st']
mismatch = (vd_check['vote_diff'] - computed_diff).abs() > 0.01
print(f"\nvote_diff vs (votes - picture_votes_1st):")
print(f"  Mismatches: {mismatch.sum()} / {len(vd_check)}")

# abs_diff should equal abs(vote_diff)
abs_check = (vd_check['abs_diff'] - vd_check['vote_diff'].abs()).abs() > 0.01
print(f"abs_diff vs abs(vote_diff):")
print(f"  Mismatches: {abs_check.sum()} / {len(vd_check)}")

# Votes should be non-negative
neg_votes = (df_vd['votes'] < 0).sum()
print(f"Negative votes: {neg_votes}")

neg_pic = (df_vd['picture_votes_1st'] < 0).sum()
print(f"Negative picture_votes_1st: {neg_pic}")

# Connection dummies should be binary
for var in CONNECTION_VARS + ['cand_data', 'powerful', 'treat_lib']:
    vals = df_vd[var].dropna().unique()
    is_binary = set(vals).issubset({0, 1, 0.0, 1.0})
    print(f"  {var} is binary: {is_binary} (unique values: {sorted(vals)[:5]})")

# deo_fullsample should be subset of peo_fullsample (DEO connected means PEO connected too)
# Actually, per the paper, PEO is provincial elections officer, DEO is district + provincial
deo_not_peo = ((df_vd['deo_fullsample'] == 1) & (df_vd['peo_fullsample'] == 0)).sum()
print(f"\ndeo_fullsample=1 but peo_fullsample=0: {deo_not_peo}")
print(f"  (DEO connection includes provincial connection, so this should be most)")

# Treatment consistency: treat_lib should be constant within pc
treat_var_in_pc = df_vd.groupby('pc')['treat_lib'].nunique()
inconsistent = (treat_var_in_pc > 1).sum()
print(f"PCs with varying treat_lib: {inconsistent}")

# ═══════════════════════════════════════════════════════════════════════════
# 4. MISSING DATA PATTERNS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("4. MISSING DATA PATTERNS")
print("=" * 70)

# Missing vote_diff by treatment
for treat_val in [0, 1]:
    sub = df_vd[df_vd['treat_lib'] == treat_val]
    n_total = len(sub)
    n_vd = sub['vote_diff'].notna().sum()
    print(f"Treatment={treat_val}: vote_diff available {n_vd}/{n_total} ({n_vd/n_total*100:.2f}%)")

# Missing picture data by treatment
for treat_val in [0, 1]:
    sub = df_vd[df_vd['treat_lib'] == treat_val]
    n_pic = sub['picture_votes_1st'].notna().sum()
    print(f"Treatment={treat_val}: picture_votes_1st available {n_pic}/{len(sub)} ({n_pic/len(sub)*100:.2f}%)")

# This is the key attrition finding: treatment increases availability of picture data
# Test significance
pic_avail = df_vd.copy()
pic_avail['has_picture'] = pic_avail['picture_votes_1st'].notna().astype(int)
pic_by_treat = pic_avail.groupby('treat_lib')['has_picture'].mean()
print(f"\nProportion with picture data: Control={pic_by_treat[0]:.4f}, Treatment={pic_by_treat[1]:.4f}")
print(f"  Difference: {pic_by_treat[1] - pic_by_treat[0]:.4f}")

# Missing controls in PC data
print("\n--- Missing data in PC dataset ---")
for var in df_pc.columns:
    n_miss = df_pc[var].isna().sum()
    if n_miss > 0:
        print(f"  {var}: {n_miss} missing ({n_miss/len(df_pc)*100:.1f}%)")

# Strata_group missing
print(f"\nstrata_group missing in PC data: {df_pc['strata_group'].isna().sum()}")
print(f"  (These are the 21 Kabul PCs added late without baseline survey)")

# ═══════════════════════════════════════════════════════════════════════════
# 5. PANEL BALANCE AND STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("5. PANEL BALANCE AND STRUCTURE")
print("=" * 70)

# Distribution of polling substations per polling center
ps_per_pc = df_vd.groupby('pc')['pollingstation'].nunique()
print(f"\nPolling substations per PC:")
print(f"  Mean: {ps_per_pc.mean():.1f}, Median: {ps_per_pc.median():.0f}, "
      f"Min: {ps_per_pc.min()}, Max: {ps_per_pc.max()}")

# Candidates per polling substation
cands_per_ps = df_vd.groupby(['pc', 'pollingstation'])['cand_id_s'].nunique()
print(f"\nCandidates per polling substation:")
print(f"  Mean: {cands_per_ps.mean():.1f}, Median: {cands_per_ps.median():.0f}, "
      f"Min: {cands_per_ps.min()}, Max: {cands_per_ps.max()}")

# Votes per polling substation (after_total)
ps_totals = df_vd.groupby(['pc', 'pollingstation'])['after_total'].first()
print(f"\nTotal votes per polling substation (after_total):")
print(f"  Mean: {ps_totals.mean():.1f}, Median: {ps_totals.median():.0f}, "
      f"Min: {ps_totals.min():.0f}, Max: {ps_totals.max():.0f}")

# Province sizes
prov_sizes = df_vd.groupby('provid398')['pc'].nunique()
print(f"\nPolling centers per province:")
print(f"  Mean: {prov_sizes.mean():.1f}, Min: {prov_sizes.min()}, Max: {prov_sizes.max()}")

# ═══════════════════════════════════════════════════════════════════════════
# 6. DUPLICATES AND ANOMALIES
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("6. DUPLICATES AND ANOMALIES")
print("=" * 70)

# Check for duplicate candidate-PS observations
dupes = df_vd.duplicated(subset=['pc', 'pollingstation', 'cand_id_s'], keep=False)
print(f"Duplicate (pc, pollingstation, cand_id_s) rows: {dupes.sum()}")

# Zero-vote polling substations
zero_after = df_vd.groupby(['pc', 'pollingstation'])['after_total'].first()
n_zero = (zero_after == 0).sum()
print(f"Polling substations with 0 total after-votes: {n_zero}")

# Extreme vote_diff = -800 (noted as outlier in code)
n_extreme = (df_vd['vote_diff'] == -800).sum()
print(f"vote_diff = -800 (extreme outlier): {n_extreme} observations")
if n_extreme > 0:
    extreme_obs = df_vd[df_vd['vote_diff'] == -800][['pc', 'pollingstation', 'cand_id_s', 'votes', 'picture_votes_1st', 'vote_diff']]
    print(extreme_obs.to_string(index=False))

# Candidates with very high absolute vote changes
high_change = df_vd[df_vd['abs_diff'] > 100].groupby('cand_id_s').size().sort_values(ascending=False)
print(f"\nCandidates with |vote_diff| > 100 at any substation: {len(high_change)}")
if len(high_change) > 0:
    print(f"  Top 5: {dict(high_change.head())}")

# Strata group distribution
print(f"\n--- Strata group distribution ---")
strata_sizes = df_vd.groupby('strata_group')['pc'].nunique()
print(f"Number of strata: {len(strata_sizes)}")
print(f"PCs per stratum: Mean={strata_sizes.mean():.1f}, Min={strata_sizes.min()}, Max={strata_sizes.max()}")

print("\n" + "=" * 70)
print("DATA AUDIT COMPLETE")
print("=" * 70)
