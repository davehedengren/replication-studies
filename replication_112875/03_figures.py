"""
03_figures.py - Replicate Figures A1-A3 from Callen & Long (2015)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import *
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

print("=" * 70)
print("FIGURE REPLICATION: Callen & Long (2015)")
print("=" * 70)

df = load_vote_diff()

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE A1: KDE of Vote Differences by Treatment for Each Connection Type
# ═══════════════════════════════════════════════════════════════════════════
print("\nFigure A1: KDE of vote differences by treatment/control for connection types")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Figure A1: Vote Difference Distributions by Treatment Status', fontsize=14)

panels = [
    ('peo_fullsample', 'Provincial Aggregator Connection (=1)'),
    ('deo_fullsample', 'Prov. + Dist. Aggregator Connection (=1)'),
    ('karzai_fullsample', 'Karzai Connection (=1)'),
    ('gov_fullsample', 'Government Connection (=1)'),
    ('incumbent', 'Incumbent (=1)'),
    ('cand_data', 'Investigated (=1)'),
]

for idx, (var, title) in enumerate(panels):
    ax = axes[idx // 3, idx % 3]
    # Exclude the extreme outlier at -800
    mask_base = (df['vote_diff'].notna()) & (df['vote_diff'] != -800) & (df[var] == 1)
    treat = df.loc[mask_base & (df['treat_lib'] == 1), 'vote_diff'].values
    ctrl = df.loc[mask_base & (df['treat_lib'] == 0), 'vote_diff'].values

    if len(treat) > 10 and len(ctrl) > 10:
        x_range = np.linspace(min(treat.min(), ctrl.min()) - 5,
                              max(treat.max(), ctrl.max()) + 5, 500)
        kde_t = gaussian_kde(treat)
        kde_c = gaussian_kde(ctrl)
        ax.plot(x_range, kde_t(x_range), label='Treatment', color='blue')
        ax.plot(x_range, kde_c(x_range), label='Control', color='red', linestyle='--')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Vote Difference During Aggregation')
    ax.set_ylabel('Density')
    if idx == 0:
        ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figureA1.png'), dpi=150, bbox_inches='tight')
print("  Saved figureA1.png")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE A2: Treatment Effects by Candidate Subsample (Lee Bounds)
# ═══════════════════════════════════════════════════════════════════════════
print("\nFigure A2: Treatment effects by candidate subsample with Lee bounds")

df2 = load_vote_diff()
df2['whole_sample'] = (~df2['picture_votes_1st'].isna() & ~df2['votes'].isna()).astype(int)
df2['votes_pic_sample'] = df2['votes'].where(df2['whole_sample'] == 1)
df2['votes_1st_pic_sample'] = df2['picture_votes_1st'].where(df2['whole_sample'] == 1)
df2['complete'] = 1

samples = [
    ('complete', 'Full Sample'),
    ('peo_fullsample', 'Prov. Aggr. Connection'),
    ('deo_fullsample', 'Dist. Aggr. Connection'),
    ('karzai_fullsample', 'Karzai Connection'),
    ('gov_fullsample', 'Government Service'),
    ('incumbent', 'Incumbent'),
    ('cand_data', 'Investigated'),
]

outcomes = [
    ('votes_1st_pic_sample', 'votes before'),
    ('votes_pic_sample', 'votes after'),
    ('vote_diff', 'votes diff'),
]

results = []
for sample_var, sample_label in samples:
    for outcome_var, outcome_label in outcomes:
        lb = lee_bounds(
            outcome=df2[outcome_var],
            treatment=df2['treat_lib'],
            sample_mask=df2[sample_var],
            cluster_var=df2['pc'],
            n_boot=0,
        )
        # Simple mean difference estimate
        mask = (df2[sample_var] == 1) & df2[outcome_var].notna()
        treat_m = df2.loc[mask & (df2['treat_lib'] == 1), outcome_var].mean()
        ctrl_m = df2.loc[mask & (df2['treat_lib'] == 0), outcome_var].mean()
        est = treat_m - ctrl_m

        results.append({
            'sample': sample_label,
            'outcome': outcome_label,
            'est': est,
            'lower': lb['lower_bound'],
            'upper': lb['upper_bound'],
        })

fig, ax = plt.subplots(figsize=(10, 12))
y_pos = []
y_labels = []
pos = 0
for sample_var, sample_label in reversed(samples):
    # Add a blank line as separator
    pos += 1
    y_labels.append(sample_label)
    y_pos.append(pos)
    pos += 0.5

    for outcome_var, outcome_label in outcomes:
        r = [r for r in results if r['sample'] == sample_label and r['outcome'] == outcome_label]
        if r:
            r = r[0]
            ax.scatter(r['est'], pos, color='red', s=30, zorder=5)
            ax.hlines(pos, r['lower'], r['upper'], colors='black', linewidths=1)
            y_labels.append(f"  {outcome_label}")
            y_pos.append(pos)
            pos += 0.5

ax.axvline(x=0, color='red', linestyle='-', alpha=0.5)
ax.set_yticks(y_pos)
ax.set_yticklabels(y_labels, fontsize=8)
ax.set_xlabel('Estimated Treatment Effect (Votes)')
ax.set_title('Figure A2: Treatment Effects by Candidate Subsample')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figureA2.png'), dpi=150, bbox_inches='tight')
print("  Saved figureA2.png")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE A3: Average Votes by Treatment for Each Connection Type
# ═══════════════════════════════════════════════════════════════════════════
print("\nFigure A3: Average votes in treatment vs control for each connection type")

df3 = load_vote_diff()

connection_types = [
    ('powerful', 'Most Connected'),
    ('peo_fullsample', 'Provincial\nAggregator\nConnection'),
    ('deo_fullsample', 'Provincial and\nDistrict\nAggregator\nConnection'),
    ('gov_fullsample', 'Government\nService'),
    ('karzai_fullsample', 'Karzai\nConnection'),
]

fig, ax = plt.subplots(figsize=(12, 6))
x_positions = np.arange(len(connection_types)) * 3

for i, (var, label) in enumerate(connection_types):
    p99 = df3.loc[df3[var] == 1, 'votes'].quantile(0.99)
    mask = (df3[var] == 1) & (df3['votes'] < p99)
    ctrl_mean = df3.loc[mask & (df3['treat_lib'] == 0), 'votes'].mean()
    treat_mean = df3.loc[mask & (df3['treat_lib'] == 1), 'votes'].mean()

    # Cluster-robust CIs
    for j, (grp_mean, grp_val) in enumerate([(ctrl_mean, 0), (treat_mean, 1)]):
        subdf = df3.loc[mask & (df3['treat_lib'] == grp_val)]
        y = subdf['votes'].values
        cl = subdf['pc'].values
        n = len(y)
        unique_cl = np.unique(cl)
        G = len(unique_cl)
        grand_mean = y.mean()

        # Cluster-robust SE of the mean
        cluster_sums = np.array([np.sum(y[cl == c] - grand_mean) for c in unique_cl])
        se = np.sqrt(G / (G - 1) * np.sum(cluster_sums ** 2) / n ** 2)

        color = '#1f77b4' if grp_val == 0 else '#ff7f0e'
        xpos = x_positions[i] + j
        ax.bar(xpos, grp_mean, width=0.8, color=color, alpha=0.8)
        ax.errorbar(xpos, grp_mean, yerr=1.96 * se, fmt='none', color='red', capsize=3)

ax.set_xticks(x_positions + 0.5)
ax.set_xticklabels([label for _, label in connection_types], fontsize=8)
ax.set_ylabel('Average Votes')
ax.set_title('Figure A3: Average Votes by Treatment Status and Connection Type')
ax.legend(['Control', 'Treatment'], loc='upper right')
ax.set_ylim(0, 32)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figureA3.png'), dpi=150, bbox_inches='tight')
print("  Saved figureA3.png")

print("\n" + "=" * 70)
print("FIGURE REPLICATION COMPLETE")
print("=" * 70)
