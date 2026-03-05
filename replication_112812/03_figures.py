"""
03_figures.py - Replicate Figures 3A, 3B, and 4
Krajbich, Oud, Fehr (2014) AER P&P
"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (create_fourblocktype, demean_choice_surplus,
                   HUMAN_CHOOSER, FIRST_BLOCK, BLOCK_LABELS, OUTPUT_DIR)

df1 = pd.read_pickle(os.path.join(OUTPUT_DIR, 'study1_clean.pkl'))
df2 = pd.read_pickle(os.path.join(OUTPUT_DIR, 'study2_clean.pkl'))

# ===================================================================
# FIGURE 3A: RT as function of |value difference| (first block)
# ===================================================================
print("Creating Figure 3A...")

fb = df1[(df1['chooser'] == HUMAN_CHOOSER) & (df1['blocktype'] == FIRST_BLOCK) &
         (df1['trial_reached'] == 1)].copy()

# Collapse by absdiff: mean RT and SE
collapsed = fb.groupby('absdiff').agg(
    rt_mean=('rt', 'mean'),
    rt_se=('rt', lambda x: x.std() / np.sqrt(len(x))),
    count=('rt', 'count')
).reset_index()

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(collapsed['absdiff'], collapsed['rt_mean'] * 1000, 'k-o', markersize=4, linewidth=1.5)
ax.fill_between(collapsed['absdiff'],
                (collapsed['rt_mean'] - collapsed['rt_se']) * 1000,
                (collapsed['rt_mean'] + collapsed['rt_se']) * 1000,
                alpha=0.2, color='gray')

# Add error bars like the paper
for _, row in collapsed.iterrows():
    ax.plot([row['absdiff'], row['absdiff']],
            [(row['rt_mean'] - row['rt_se']) * 1000, (row['rt_mean'] + row['rt_se']) * 1000],
            'k-', linewidth=0.8)

ax.set_xlabel('Absolute value difference [Swiss francs]', fontsize=12)
ax.set_ylabel('rt [ms]', fontsize=12)
ax.set_title('Figure 3A: Response Times as a Function of Valuation Difference\n(First Block Only)', fontsize=11)
ax.set_xlim(-0.1, 2.6)
ax.set_ylim(400, 2200)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig3a_rt_vs_absdiff.png'), dpi=150)
plt.close()
print("  Saved fig3a_rt_vs_absdiff.png")

# ===================================================================
# FIGURE 3B: De-meaned choice surplus by block type
# ===================================================================
print("Creating Figure 3B...")

human_reached = df1[(df1['chooser'] == HUMAN_CHOOSER) & (df1['trial_reached'] == 1)].copy()
human_reached = create_fourblocktype(human_reached)
dm = demean_choice_surplus(human_reached)

fig, ax = plt.subplots(figsize=(7, 5))

labels = ['First', 'Pre', 'Intervention', 'Post']
means = []
ses = []
for bt in [1, 2, 3, 4]:
    sub = dm[dm['fourblocktype'] == bt]['demeaned_choicevalue']
    means.append(sub.mean())
    ses.append(sub.std() / np.sqrt(len(sub)))

x = np.arange(len(labels))
colors = ['steelblue', 'steelblue', 'black', 'steelblue']
bars = ax.bar(x, means, yerr=ses, capsize=5, color=colors, width=0.6, alpha=0.8,
              edgecolor='black', linewidth=0.5)

ax.set_ylabel('De-meaned choice surplus (Swiss francs)', fontsize=11)
ax.set_xlabel('Block type', fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_title('Figure 3B: Individually De-Meaned Choice Surplus\nby Block Type', fontsize=11)
ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
ax.set_ylim(-0.06, 0.03)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig3b_choice_surplus.png'), dpi=150)
plt.close()
print("  Saved fig3b_choice_surplus.png")

# ===================================================================
# FIGURE 4: Choices and RTs for 4 subjects in Ultimatum Game
# ===================================================================
print("Creating Figure 4...")

rating_scale = [4, 6, 8, 10]

# Pick 4 subjects to display (paper shows subjects 1, and 3 others)
# The R code only shows subject 1, but the paper shows 4 panels
subjects = sorted(df2['subject'].unique())[:4]

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

for idx, subj in enumerate(subjects):
    ax = axes[idx // 2][idx % 2]
    sub = df2[df2['subject'] == subj]

    # Choice probability by offer
    choice_by_offer = sub.groupby('offer')['choice'].mean()
    rt_by_offer = sub.groupby('offer')['rt'].mean() / 1000  # Convert to seconds

    # Left axis: p(accept)
    ax.plot(choice_by_offer.index, choice_by_offer.values, 'ko', markersize=8)
    ax.set_ylabel('p(accept)', fontsize=10)
    ax.set_ylim(-0.05, 1.1)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

    # Right axis: RT
    ax2 = ax.twinx()
    ax2.plot(rt_by_offer.index, rt_by_offer.values, '-', color='gray', linewidth=2)
    ax2.set_ylabel('rt [s]', fontsize=10, color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')

    ax.set_xlabel('Offer [Swiss francs]', fontsize=10)
    ax.set_title(f'Subject {subj}', fontsize=11)
    ax.set_xlim(3, 11)

plt.suptitle('Figure 4: Choices and RTs for Four Subjects in the Ultimatum Game', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_ug_subjects.png'), dpi=150)
plt.close()
print("  Saved fig4_ug_subjects.png")

# ===================================================================
# ADDITIONAL: RT distribution and observation density (Figure 3A complement)
# ===================================================================
print("Creating additional diagnostic figure...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# RT histogram by stakes
low = fb[fb['easy'] == 0]['rt']
high = fb[fb['easy'] == 1]['rt']
ax1.hist(low, bins=50, alpha=0.5, label=f'Low stakes (mean={low.mean():.2f}s)', density=True, color='red')
ax1.hist(high, bins=50, alpha=0.5, label=f'High stakes (mean={high.mean():.2f}s)', density=True, color='blue')
ax1.set_xlabel('Response time (seconds)')
ax1.set_ylabel('Density')
ax1.set_title('RT Distribution by Stakes (First Block)')
ax1.legend()
ax1.set_xlim(0, 5)

# Observation density
total = len(fb)
density = collapsed.copy()
density['density'] = density['count'] / total
ax2.bar(density['absdiff'], density['density'], width=0.15, color='steelblue', alpha=0.7)
ax2.set_xlabel('Absolute value difference [Swiss francs]')
ax2.set_ylabel('Fraction of trials')
ax2.set_title('Trial Density by Value Difference (First Block)')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig_diagnostic_rt.png'), dpi=150)
plt.close()
print("  Saved fig_diagnostic_rt.png")

print("\n=== FIGURES COMPLETE ===")
