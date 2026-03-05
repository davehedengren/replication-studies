"""
01_clean.py - Data cleaning and variable construction for Krajbich et al (2014)
Study 1: Time-constrained food choice (49 subjects, binary choices with BDM valuations)
Study 2: Ultimatum game (18 subjects, 16 rounds each)
"""
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_study1, load_study2, create_fourblocktype, demean_choice_surplus, OUTPUT_DIR

print("=" * 70)
print("DATA CLEANING: Krajbich, Oud, Fehr (2014)")
print("=" * 70)

# ===================================================================
# STUDY 1: Time-constrained food choice
# ===================================================================
print("\n--- STUDY 1: Time-Constrained Food Choice ---")
df1 = load_study1()
print(f"Raw data: {len(df1)} rows, {len(df1.columns)} columns")
print(f"Subjects: {df1['subject'].nunique()}")

# Key filters
human = df1[df1['chooser'] == 1]  # Human chooser (not computer)
first_block_human = human[human['blocktype'] == 1]
reached = human[human['trial_reached'] == 1]

print(f"Human-chosen trials: {len(human)}")
print(f"First block, human: {len(first_block_human)}")
print(f"Trials reached (human): {len(reached)}")
print(f"Trials NOT reached (human): {len(human[human['trial_reached'] == 0])}")

# Create fourblocktype
df1 = create_fourblocktype(df1)
print(f"\nFourblocktype distribution:")
print(df1['fourblocktype'].value_counts().sort_index())

# Compute key statistics matching paper claims
print("\n--- KEY STATISTICS ---")

# "easy" variable: 1 = high stakes (large value diff), 0 = low stakes
# RT by stakes type (first block, human chooser, trial reached)
fb_reached = df1[(df1['chooser'] == 1) & (df1['blocktype'] == 1) & (df1['trial_reached'] == 1)]
low_stakes = fb_reached[fb_reached['easy'] == 0]
high_stakes = fb_reached[fb_reached['easy'] == 1]

print(f"Mean RT low stakes: {low_stakes['rt'].mean():.2f} s (published: 1.65 s)")
print(f"Mean RT high stakes: {high_stakes['rt'].mean():.2f} s (published: 1.11 s)")

# Missed trials: in first block, 100 trials total, count how many were NOT reached
fb_all = df1[df1['blocktype'] == 1]  # All first-block rows (100 per subject)
missed_by_subj = 100 - fb_all.groupby('subject')['trial_reached'].sum()
print(f"Mean missed trials per subject (first block): {missed_by_subj.mean():.1f} (published: ~44)")
print(f"Mean trials reached (first block): {(100 - missed_by_subj).mean():.1f}")

# Money left on table: for unreached first-block trials, compute potential surplus
# Paper says "$20.10 left on the table per subject if all trials had been paid"
# This is the sum of (val_preferred - val_nonpreferred)/2 for missed trials
# (expected surplus from optimal choice)
fb_unreached = fb_all[fb_all['trial_reached'] == 0]
money_left = fb_unreached.groupby('subject').apply(
    lambda x: ((x['val_preferred'] - x['val_nonpreferred']) / 2).sum()
)
print(f"Mean expected surplus from missed trials: {money_left.mean():.2f} CHF (published: ~$20.10)")

# Alternative: sum of val_preferred for missed trials (max possible payoff)
money_max = fb_unreached.groupby('subject')['val_preferred'].sum()
print(f"Mean max payoff from missed trials: {money_max.mean():.2f} CHF")

# De-meaned choice surplus by block type (for Figure 3B)
# Only human chooser trials that were reached
human_reached = df1[(df1['chooser'] == 1) & (df1['trial_reached'] == 1)].copy()
human_reached = create_fourblocktype(human_reached)

dm_surplus = demean_choice_surplus(human_reached)
print("\nDe-meaned choice surplus by block type:")
for bt in [1, 2, 3, 4]:
    sub = dm_surplus[dm_surplus['fourblocktype'] == bt]
    label = {1: 'First', 2: 'Pre-intervention', 3: 'Intervention', 4: 'Post-intervention'}[bt]
    if len(sub) > 0:
        print(f"  {label}: mean={sub['demeaned_choicevalue'].mean():.4f}, "
              f"se={sub['demeaned_choicevalue'].std()/np.sqrt(len(sub)):.4f}, N={len(sub)}")

# Regression: RT on absdiff (first block, human, reached)
import statsmodels.api as sm
fb_data = fb_reached[['rt', 'absdiff']].dropna()
X = sm.add_constant(fb_data['absdiff'])
model = sm.OLS(fb_data['rt'], X).fit(cov_type='HC1')
print(f"\nRegression: RT = a + b * |value diff| (first block)")
print(f"  Intercept: {model.params['const']:.4f}")
print(f"  Slope (absdiff): {model.params['absdiff']:.4f} ({model.pvalues['absdiff']:.4f})")
print(f"  N: {model.nobs:.0f}, R²: {model.rsquared:.4f}")

# Save cleaned study 1 data
df1.to_pickle(os.path.join(OUTPUT_DIR, 'study1_clean.pkl'))
print(f"\nSaved study 1 data: {len(df1)} rows")

# ===================================================================
# STUDY 2: Ultimatum Game
# ===================================================================
print("\n--- STUDY 2: Ultimatum Game ---")
df2 = load_study2()
print(f"Raw data: {len(df2)} rows")
print(f"Subjects: {df2['subject'].nunique()}")
print(f"Offers: {sorted(df2['offer'].unique())}")

# Check for probabilistic choices (same offer both accepted and rejected)
print("\nProbabilistic choice check:")
n_probabilistic = 0
for subj in df2['subject'].unique():
    sub = df2[df2['subject'] == subj]
    is_prob = False
    for offer in sub['offer'].unique():
        choices = sub[sub['offer'] == offer]['choice']
        if choices.nunique() > 1:  # Both accept and reject for same offer
            is_prob = True
            break
    if is_prob:
        n_probabilistic += 1

print(f"Subjects with probabilistic choices: {n_probabilistic}/18 (published: 16/18)")

# RT by offer level
print("\nMean RT (ms) by offer level:")
for offer in sorted(df2['offer'].unique()):
    sub = df2[df2['offer'] == offer]
    print(f"  {offer} CHF: RT={sub['rt'].mean():.0f} ms, "
          f"p(accept)={sub['choice'].mean():.3f}, N={len(sub)}")

# Convert RT to seconds for consistency
df2['rt_sec'] = df2['rt'] / 1000.0

# Save
df2.to_pickle(os.path.join(OUTPUT_DIR, 'study2_clean.pkl'))
print(f"\nSaved study 2 data: {len(df2)} rows")

print("\n=== CLEANING COMPLETE ===")
