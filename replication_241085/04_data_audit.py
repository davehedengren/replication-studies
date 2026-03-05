"""
04_data_audit.py — Data quality checks for Prager (2026).
"""

import numpy as np
import pandas as pd
from utils import OUTPUT_DIR, load_transitions, load_soc_hierarchy

print("=" * 60)
print("04_data_audit.py — Data quality audit")
print("=" * 60)

trans = load_transitions()
soc = load_soc_hierarchy()

# ══════════════════════════════════════════════════════════════════════
# 1. COVERAGE
# ══════════════════════════════════════════════════════════════════════

print("\n── 1. Coverage ──\n")

print(f"  Transition pairs: {len(trans)}")
print(f"  Unique source occupations (soc1): {trans['soc1'].nunique()}")
print(f"  Unique destination occupations (soc2): {trans['soc2'].nunique()}")
all_socs = set(trans['soc1'].unique()) | set(trans['soc2'].unique())
print(f"  Unique occupations (union): {len(all_socs)}")
print(f"  SOC hierarchy entries: {len(soc)}")

# Coverage: what fraction of possible pairs are observed?
n_soc1 = trans['soc1'].nunique()
n_soc2 = trans['soc2'].nunique()
possible = n_soc1 * n_soc2
print(f"  Possible pairs (soc1 × soc2): {possible:,}")
print(f"  Observed pairs: {len(trans):,}")
print(f"  Coverage: {100 * len(trans) / possible:.1f}%")

# ══════════════════════════════════════════════════════════════════════
# 2. DISTRIBUTIONS
# ══════════════════════════════════════════════════════════════════════

print("\n── 2. Distributions ──\n")

# Transition share
ts = trans['transition_share']
print(f"  Transition share: mean={ts.mean():.6f}, median={ts.median():.6f}, "
      f"sd={ts.std():.6f}")
print(f"    Range: [{ts.min():.6f}, {ts.max():.6f}]")
print(f"    Top 5 transition shares:")
top5 = trans.nlargest(5, 'transition_share')
for _, row in top5.iterrows():
    print(f"      {row['soc1_name']} -> {row['soc2_name']}: {row['transition_share']:.4f}")

# Total observations
obs = trans['total_obs']
print(f"\n  Total obs per pair: mean={obs.mean():.0f}, median={obs.median():.0f}")
print(f"    Range: [{obs.min():.0f}, {obs.max():.0f}]")

# ══════════════════════════════════════════════════════════════════════
# 3. LOGICAL CONSISTENCY
# ══════════════════════════════════════════════════════════════════════

print("\n── 3. Logical consistency ──\n")

# Transition shares in [0, 1]
bad_shares = ((ts < 0) | (ts > 1)).sum()
print(f"  Transition shares outside [0,1]: {bad_shares}")

# Self-transitions
self_trans = trans[trans['soc1'] == trans['soc2']]
print(f"  Self-transition pairs (soc1 == soc2): {len(self_trans)}")

# Symmetry: a->b exists iff b->a exists?
pairs_ab = set(zip(trans['soc1'], trans['soc2']))
pairs_ba = set(zip(trans['soc2'], trans['soc1']))
asymmetric = len(pairs_ab.symmetric_difference(pairs_ba))
print(f"  Asymmetric pairs (in a->b but not b->a, or vice versa): {asymmetric}")

# For symmetric pairs, how correlated are the shares?
sym_merge = trans.merge(
    trans.rename(columns={'soc1': 'soc2', 'soc2': 'soc1',
                          'transition_share': 'share_reverse'})[['soc1', 'soc2', 'share_reverse']],
    on=['soc1', 'soc2'], how='inner'
)
corr = sym_merge['transition_share'].corr(sym_merge['share_reverse'])
print(f"  Correlation(share_ab, share_ba) for symmetric pairs: {corr:.3f}")

# ══════════════════════════════════════════════════════════════════════
# 4. MISSING DATA
# ══════════════════════════════════════════════════════════════════════

print("\n── 4. Missing data ──\n")

for col in trans.columns:
    n_miss = trans[col].isna().sum()
    if n_miss > 0:
        print(f"  {col}: {n_miss} missing ({100 * n_miss / len(trans):.1f}%)")
if trans.isna().sum().sum() == 0:
    print(f"  No missing values in any column")

# ══════════════════════════════════════════════════════════════════════
# 5. DUPLICATES
# ══════════════════════════════════════════════════════════════════════

print("\n── 5. Duplicates ──\n")

dupes = trans.duplicated(subset=['soc1', 'soc2']).sum()
print(f"  Duplicate (soc1, soc2) pairs: {dupes}")

soc_dupes = soc.duplicated(subset=['detailedoccupation']).sum()
print(f"  Duplicate SOC codes in hierarchy: {soc_dupes}")

# ══════════════════════════════════════════════════════════════════════
# 6. SOC HIERARCHY COVERAGE
# ══════════════════════════════════════════════════════════════════════

print("\n── 6. SOC hierarchy coverage ──\n")

soc_in_trans = all_socs
soc_in_hier = set(soc['detailedoccupation'])
in_trans_not_hier = soc_in_trans - soc_in_hier
in_hier_not_trans = soc_in_hier - soc_in_trans
print(f"  SOC codes in transitions but not hierarchy: {len(in_trans_not_hier)}")
print(f"  SOC codes in hierarchy but not transitions: {len(in_hier_not_trans)}")

# Major group distribution of transition data
trans_soc1_major = trans['soc1'].str[:2].value_counts().sort_index()
print(f"\n  Transition source occupations by major group (first 2 digits):")
for mg, count in trans_soc1_major.items():
    name = soc[soc['majorgroup'].str[:2] == mg]['majorname'].iloc[0] if len(soc[soc['majorgroup'].str[:2] == mg]) > 0 else 'Unknown'
    # Truncate long names
    name = name[:50] if len(name) > 50 else name
    print(f"    {mg}: {count:>6,} pairs  ({name})")

print("\n" + "=" * 60)
print("04_data_audit.py — DONE")
print("=" * 60)
