"""
01_clean.py - Data loading and sample construction verification
Callen & Long (2015) - Institutional Corruption and Election Fraud
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import *

print("=" * 70)
print("PHASE 1: DATA LOADING AND SAMPLE VERIFICATION")
print("=" * 70)

# ── Load datasets ──────────────────────────────────────────────────────────
df_vd = load_vote_diff()
df_pc = load_pc()

print(f"\n--- master_vote_difference_native ---")
print(f"Shape: {df_vd.shape}")
print(f"Total observations: {len(df_vd)}")

print(f"\n--- master_pc_native ---")
print(f"Shape: {df_pc.shape}")
print(f"Total polling centers: {len(df_pc)}")

# ── Verify key sample sizes from paper ─────────────────────────────────────
print("\n" + "=" * 70)
print("SAMPLE SIZE VERIFICATION")
print("=" * 70)

# Table 2/3: drop if vote_diff==., then N=48,008 (after outlier removal)
vd_nonnull = df_vd[df_vd['vote_diff'].notna()].copy()
print(f"\nObservations with non-missing vote_diff: {len(vd_nonnull)}")
print(f"  Published: 48,018 (before outlier trim)")

# Table 3 uses insample (drop 5 highest + 5 lowest vote_diff)
vd_sorted = vd_nonnull.sort_values('vote_diff').reset_index(drop=True)
vd_sorted['ascending'] = range(1, len(vd_sorted) + 1)
vd_sorted = vd_sorted.sort_values('vote_diff', ascending=False).reset_index(drop=True)
vd_sorted['descending'] = range(1, len(vd_sorted) + 1)
insample = vd_sorted[(vd_sorted['ascending'] > 5) & (vd_sorted['descending'] > 5)]
print(f"Insample (drop 5 top/bottom vote_diff): {len(insample)}")
print(f"  Published Table 3: 48,008")

# Number of candidates in Table 3
n_cands = insample['cand_id_s'].nunique()
print(f"Number unique candidates (insample): {n_cands}")
print(f"  Published Table 3: 1,783-1,784")

# Polling stations
n_pc = vd_nonnull.groupby(['pc', 'pollingstation']).ngroups
print(f"Number of polling stations (with vote_diff): {n_pc}")

# Table 2 summary stats verification
print("\n--- Table 2: Summary Statistics (insample) ---")
for var in ['peo_fullsample', 'deo_fullsample', 'karzai_fullsample',
            'gov_fullsample', 'incumbent', 'vote_diff']:
    m = insample[var].mean()
    sd = insample[var].std()
    n = insample[var].notna().sum()
    print(f"  {var}: mean={m:.3f}, sd={sd:.3f}, n={n}")

# Share diff (Table 2 last row)
vd2 = vd_nonnull.copy()
vd2['picture_total'] = vd2.groupby(['pc', 'pollingstation'])['picture_votes_1st'].transform('sum')
vd2['before_share'] = vd2['picture_votes_1st'] / vd2['picture_total']
vd2['after_share'] = vd2['votes'] / vd2['picture_total']
vd2['share_diff'] = vd2['after_share'] - vd2['before_share']

# Outliers based on share_diff (Table 2 uses share_diff outliers)
vd2_sorted = vd2.sort_values('share_diff').reset_index(drop=True)
vd2_sorted['ascending'] = range(1, len(vd2_sorted) + 1)
vd2_sorted = vd2_sorted.sort_values('share_diff', ascending=False).reset_index(drop=True)
vd2_sorted['descending'] = range(1, len(vd2_sorted) + 1)
insample2 = vd2_sorted[(vd2_sorted['ascending'] > 5) & (vd2_sorted['descending'] > 5)]
share_diff1 = insample2['share_diff'] * 100
print(f"\n  share_diff*100: mean={share_diff1.mean():.3f}, sd={share_diff1.std():.3f}, n={len(share_diff1)}")
print(f"  Published: mean=0.570, sd=14.287, n=48,008")

# Table 4: Polling center level, need strata_group
print("\n--- Table 4: Polling Center Level ---")
pc_valid = df_pc[df_pc['strata_group'].notna()].copy()
print(f"PCs with strata_group: {len(pc_valid)}")
n_treat = pc_valid['treat_lib'].sum()
n_control = len(pc_valid) - n_treat
print(f"  Treatment: {int(n_treat)}, Control: {int(n_control)}")
print(f"  Published: Treatment ~227-238, Control ~223-233")

# Table 7: Candidate-PS level, votes outcome
print("\n--- Table 7: Vote totals sample ---")
print(f"Total candidate-PS obs with votes: {df_vd['votes'].notna().sum()}")
print(f"  Published Table 7: 386,855 (col 1)")

# Treatment/control breakdown
treat_breakdown = df_vd.groupby('treat_lib').size()
print(f"By treatment: {dict(treat_breakdown)}")

# Number of polling centers
n_pcs = df_vd['pc'].nunique()
print(f"Unique polling centers: {n_pcs}")
print(f"  Published: 461-471")

# Table 1 sample: polling stations with both picture and scrape data
print("\n--- Table 1: Aggregation Patterns ---")
t1 = df_vd[df_vd['vote_diff'].notna()].copy()
t1['pos_cand'] = (t1['pos_diff'] > 0).astype(int)
t1['neg_cand'] = (t1['neg_diff'] > 0).astype(int)
t1['pos_votes'] = t1['vote_diff'].where(t1['vote_diff'] > 0, 0)
t1['neg_votes'] = t1['vote_diff'].where(t1['vote_diff'] < 0, 0)

# Collapse to polling station level
station_id = t1.groupby(['pc', 'pollingstation']).ngroup()
t1['n_stations'] = station_id
t1_coll = t1.groupby('n_stations').agg(
    vote_diff=('vote_diff', 'sum'),
    pos_votes=('pos_votes', 'sum'),
    neg_votes=('neg_votes', 'sum'),
).reset_index()
t1_coll['abs_neg'] = t1_coll['neg_votes'].abs()
n_stations = len(t1_coll)

# Patterns
no_fraud = ((t1_coll['pos_votes'] == t1_coll['abs_neg']) & (t1_coll['abs_neg'] == 0)).sum()
add_only = ((t1_coll['pos_votes'] > 0) & (t1_coll['abs_neg'] == 0)).sum()
sub_only = ((t1_coll['pos_votes'] == 0) & (t1_coll['abs_neg'] > 0)).sum()
add_sub_equal = ((t1_coll['pos_votes'] == t1_coll['abs_neg']) & (t1_coll['abs_neg'] > 0)).sum()
add_more = ((t1_coll['pos_votes'] > t1_coll['abs_neg']) &
            (t1_coll['abs_neg'] > 0) & (t1_coll['pos_votes'] > 0)).sum()
sub_more = ((t1_coll['abs_neg'] > t1_coll['pos_votes']) &
            (t1_coll['pos_votes'] > 0)).sum()

print(f"Total polling substations: {n_stations}")
print(f"  Published: 346")  # Note: paper says 341 for this table

patterns = {
    'No fraud': (no_fraud, 74, 21.4),
    'Adding votes only': (add_only, 70, 20.2),
    'Subtracting votes only': (sub_only, 15, 4.3),
    'Adding and subtracting equally': (add_sub_equal, 15, 4.3),
    'Adding more than subtracting': (add_more, 127, 36.7),
    'Subtracting more than adding': (sub_more, 45, 13.01),
}
for name, (py_val, pub_n, pub_pct) in patterns.items():
    py_pct = py_val / n_stations * 100
    print(f"  {name}: N={py_val} ({py_pct:.1f}%), Published: N={pub_n} ({pub_pct}%)")

print("\n" + "=" * 70)
print("DATA LOADING AND VERIFICATION COMPLETE")
print("=" * 70)
