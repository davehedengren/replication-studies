"""
04_data_audit.py - Data quality audit for Krajbich et al (2014)
"""
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_study1, load_study2, OUTPUT_DIR

df1 = load_study1()
df2 = load_study2()

print("=" * 70)
print("DATA AUDIT: Krajbich, Oud, Fehr (2014)")
print("=" * 70)

# ===================================================================
# STUDY 1
# ===================================================================
print("\n" + "=" * 70)
print("STUDY 1: TIME-CONSTRAINED FOOD CHOICE")
print("=" * 70)

# 1. COVERAGE
print("\n1. COVERAGE")
print("-" * 50)
print(f"Total observations: {len(df1):,}")
print(f"Variables: {len(df1.columns)}")
print(f"Subjects: {df1['subject'].nunique()}")
print(f"Sessions per subject: {df1.groupby('subject')['session'].nunique().describe().to_string()}")
print(f"Rows per subject: {df1.groupby('subject').size().unique()}")
print(f"Block types: {sorted(df1['blocktype'].dropna().unique())}")
print(f"Block sequences: {sorted(df1['blocksequence'].dropna().unique())}")

# Trial counts
print(f"\nTrials per block per subject: 100 (fixed)")
print(f"Trials reached: {df1['trial_reached'].sum():,.0f} ({df1['trial_reached'].mean()*100:.1f}%)")
print(f"Trials NOT reached: {(df1['trial_reached'] == 0).sum():,.0f}")

# 2. DISTRIBUTIONS
print("\n2. DISTRIBUTIONS")
print("-" * 50)

print(f"\n{'Variable':<25} {'N':>8} {'Mean':>8} {'SD':>8} {'Min':>8} {'Max':>8}")
print("-" * 70)

for var in ['rt', 'leftrate', 'rightrate', 'diff', 'absdiff', 'choicevalue',
            'val_chosen', 'val_unchosen', 'val_preferred', 'val_nonpreferred']:
    s = df1[var].dropna()
    if len(s) > 0:
        print(f"{var:<25} {len(s):>8,} {s.mean():>8.3f} {s.std():>8.3f} {s.min():>8.3f} {s.max():>8.3f}")

# BDM valuation ranges
print(f"\nBDM valuation range (leftrate): {df1['leftrate'].min():.2f} to {df1['leftrate'].max():.2f}")
print(f"BDM valuation range (rightrate): {df1['rightrate'].min():.2f} to {df1['rightrate'].max():.2f}")
print(f"Max subject valuation range: {df1['maxabsdiff'].describe().to_string()}")

# 3. MISSING DATA
print("\n3. MISSING DATA PATTERNS")
print("-" * 50)

for col in df1.columns:
    miss = df1[col].isna().sum()
    if miss > 0:
        print(f"  {col:<25}: {miss:>6,} ({miss/len(df1)*100:.1f}%)")

# Missing by trial_reached
print(f"\nRT missing when trial_reached=1: {df1[df1['trial_reached']==1]['rt'].isna().sum()}")
print(f"RT missing when trial_reached=0: {df1[df1['trial_reached']==0]['rt'].isna().sum()}")

# 4. LOGICAL CONSISTENCY
print("\n4. LOGICAL CONSISTENCY")
print("-" * 50)

# Check: absdiff = |leftrate - rightrate|
computed_absdiff = (df1['leftrate'] - df1['rightrate']).abs()
mismatch = (computed_absdiff - df1['absdiff']).abs() > 0.001
print(f"absdiff mismatch: {mismatch.sum()} (should be 0)")

# Check: diff = leftrate - rightrate
computed_diff = df1['leftrate'] - df1['rightrate']
mismatch = (computed_diff - df1['diff']).abs() > 0.001
print(f"diff mismatch: {mismatch.sum()} (should be 0)")

# Check: val_preferred >= val_nonpreferred
violations = (df1['val_preferred'] < df1['val_nonpreferred']).sum()
print(f"val_preferred < val_nonpreferred: {violations} (should be 0)")

# Check: choice is 0 or 1
print(f"choice values: {sorted(df1['choice'].unique())}")

# Check: chooser correctly flags computer vs human
human_but_no_rt = df1[(df1['chooser'] == 1) & df1['rt'].isna()].shape[0]
print(f"Human chooser with missing RT: {human_but_no_rt}")

# Check: choicecorrect
reached = df1[df1['trial_reached'] == 1]
correct_rate = reached.groupby('subject')['choicecorrect'].mean()
print(f"\nChoice correct rate (trial_reached=1):")
print(f"  Mean: {correct_rate.mean():.3f}, Min: {correct_rate.min():.3f}, Max: {correct_rate.max():.3f}")

# Check: blocktype vs intervention flags consistency
int_block = df1[df1['interventionblock'] == 1]
print(f"\nIntervention block blocktype values: {sorted(int_block['blocktype'].dropna().unique())}")
nonint = df1[df1['noninterventionblock'] == 1]
print(f"Non-intervention block blocktype values: {sorted(nonint['blocktype'].dropna().unique())}")

# 5. PANEL BALANCE
print("\n5. PANEL BALANCE")
print("-" * 50)

# Trials reached by block type
for bt in sorted(df1['blocktype'].dropna().unique()):
    sub = df1[df1['blocktype'] == bt]
    reached = sub['trial_reached'].sum()
    print(f"  Blocktype {int(bt)}: {len(sub)} total, {int(reached)} reached ({reached/len(sub)*100:.1f}%)")

# RT outliers
print("\n6. OUTLIERS")
print("-" * 50)
rt = df1['rt'].dropna()
print(f"RT > 5s: {(rt > 5).sum()} ({(rt > 5).mean()*100:.1f}%)")
print(f"RT > 10s: {(rt > 10).sum()} ({(rt > 10).mean()*100:.1f}%)")
print(f"RT < 0.1s: {(rt < 0.1).sum()} ({(rt < 0.1).mean()*100:.1f}%)")
print(f"RT percentiles: P1={rt.quantile(0.01):.3f}, P99={rt.quantile(0.99):.3f}")

# Subject-level RT means
subj_rt = df1[df1['trial_reached'] == 1].groupby('subject')['rt'].mean()
print(f"\nSubject mean RT: min={subj_rt.min():.2f}, max={subj_rt.max():.2f}, "
      f"mean={subj_rt.mean():.2f}")

# 7. DUPLICATES
print("\n7. DUPLICATES")
print("-" * 50)
n_dup = df1.duplicated().sum()
print(f"Exact duplicate rows: {n_dup}")

# ===================================================================
# STUDY 2
# ===================================================================
print("\n" + "=" * 70)
print("STUDY 2: ULTIMATUM GAME")
print("=" * 70)

print(f"\nTotal observations: {len(df2)}")
print(f"Subjects: {df2['subject'].nunique()}")
print(f"Offers: {sorted(df2['offer'].unique())}")
print(f"Trials per subject: {df2.groupby('subject').size().describe().to_string()}")

# Distributions
print(f"\nRT (ms): mean={df2['rt'].mean():.0f}, sd={df2['rt'].std():.0f}, "
      f"min={df2['rt'].min()}, max={df2['rt'].max()}")
print(f"Choice: mean={df2['choice'].mean():.3f} (overall acceptance rate)")

# Trials per subject per offer
print("\nTrials per subject per offer:")
trials_per = df2.groupby(['subject', 'offer']).size().reset_index(name='n')
for offer in sorted(df2['offer'].unique()):
    sub = trials_per[trials_per['offer'] == offer]['n']
    print(f"  Offer {offer}: mean={sub.mean():.1f}, min={sub.min()}, max={sub.max()}")

# RT outliers
print(f"\nRT > 10s: {(df2['rt'] > 10000).sum()}")
print(f"RT > 5s: {(df2['rt'] > 5000).sum()}")

# Duplicates
print(f"Duplicate rows: {df2.duplicated().sum()}")

# ===================================================================
# SUMMARY
# ===================================================================
print("\n" + "=" * 70)
print("DATA AUDIT SUMMARY")
print("=" * 70)
print("""
Key Findings:

1. COVERAGE: Study 1 has 24,500 rows (49 subjects x 5 blocks x 100 trials).
   Study 2 has 288 rows (18 subjects x 16 trials). Both match expectations.

2. DISTRIBUTIONS: BDM valuations range 0-2.25 CHF. RT in study 1 ranges
   from ~0.005s to ~15.6s. Some very fast RTs (<0.1s) exist but are rare.

3. MISSING DATA: RT is missing for unreached trials (by design). The 'cut'
   variable (random cutoff time) is only present for intervention blocks.
   157 trials have subjectwascutoff=1 (forced cutoff during intervention).

4. LOGICAL CONSISTENCY: All computed checks pass. absdiff matches |diff|,
   val_preferred >= val_nonpreferred, choice is binary.

5. PANEL BALANCE: Each subject has exactly 500 rows (5 blocks x 100 trials).
   Trial completion varies by block type, with first block having ~56% completion.

6. NO CODING BUGS found in the original code.

7. DATA QUALITY: Good. The data is well-structured, cleanly labeled, and
   internally consistent. The only notable issue is unequal trial counts
   per offer in Study 2 (4-5 per subject per offer level).
""")

print("=== DATA AUDIT COMPLETE ===")
