"""
01_clean.py — Load and validate data for Prager (2026).

Loads Schubert-Stansbury-Taska occupation transition data and SOC hierarchy.
"""

import numpy as np
import pandas as pd
from utils import OUTPUT_DIR, load_transitions, load_soc_hierarchy

print("=" * 60)
print("01_clean.py — Data loading and validation")
print("=" * 60)

# ══════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════

trans = load_transitions()
soc = load_soc_hierarchy()

print(f"\n  Transition data: {len(trans)} rows, {len(trans.columns)} columns")
print(f"  Columns: {list(trans.columns)}")
print(f"  Unique soc1: {trans['soc1'].nunique()}")
print(f"  Unique soc2: {trans['soc2'].nunique()}")

print(f"\n  SOC hierarchy: {len(soc)} rows, {len(soc.columns)} columns")
print(f"  Columns: {list(soc.columns)}")

# ══════════════════════════════════════════════════════════════════════
# BASIC VALIDATION
# ══════════════════════════════════════════════════════════════════════

print("\n── Validation ──\n")

# Transition shares should be in [0, 1]
print(f"  Transition share range: [{trans['transition_share'].min():.6f}, "
      f"{trans['transition_share'].max():.6f}]")
assert trans['transition_share'].min() >= 0
assert trans['transition_share'].max() <= 1

# Total obs should be positive
print(f"  Total obs range: [{trans['total_obs'].min()}, {trans['total_obs'].max()}]")
assert trans['total_obs'].min() > 0

# No missing transition shares
missing = trans['transition_share'].isna().sum()
print(f"  Missing transition shares: {missing}")

# Check symmetry: are all (a,b) pairs also present as (b,a)?
pairs_ab = set(zip(trans['soc1'], trans['soc2']))
pairs_ba = set(zip(trans['soc2'], trans['soc1']))
only_ab = pairs_ab - pairs_ba
only_ba = pairs_ba - pairs_ab
print(f"  Pairs in a->b only: {len(only_ab)}")
print(f"  Pairs in b->a only: {len(only_ba)}")

# Transition share distribution
print(f"\n  Transition share quantiles:")
for q in [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
    print(f"    {q:.0%}: {trans['transition_share'].quantile(q):.6f}")

# SOC hierarchy check
print(f"\n  SOC detailed occupations: {soc['detailedoccupation'].nunique()}")
print(f"  SOC major groups: {soc['majorgroup'].nunique()}")

# Save parquet
trans.to_parquet(f"{OUTPUT_DIR}/transitions.parquet", index=False)
soc.to_parquet(f"{OUTPUT_DIR}/soc_hierarchy.parquet", index=False)
print(f"\n  Saved parquet files to {OUTPUT_DIR}")

print("\n" + "=" * 60)
print("01_clean.py — DONE")
print("=" * 60)
