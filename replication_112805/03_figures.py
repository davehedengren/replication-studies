"""
03_figures.py - Diagnostic and summary figures
The paper has no main figures, but we create:
1. Distribution of school VAM estimates across specifications
2. Coefficient comparison chart (replicated vs published)
3. VAM scatter: assigned school vs home school
"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import OUTPUT_DIR

df = pd.read_pickle(os.path.join(OUTPUT_DIR, 'analysis_data.pkl'))
va_schl = pd.read_pickle(os.path.join(OUTPUT_DIR, 'va_school.pkl'))
margin = df[df['onmargin'] == 1].copy()

# ===================================================================
# Figure 1: Distribution of School VAM Estimates
# ===================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, (s, label) in enumerate([('02', '2002 Only'), ('2yr', '2001-2002'), ('all', '1998-2002')]):
    ax = axes[i]
    for e, color, ename in [('ar', 'blue', 'Avg Residual'),
                             ('mix', 'red', 'Mixed Effects'),
                             ('FE', 'green', 'Fixed Effects')]:
        col = f'mod2{e}_{s}_test_5'
        if col in va_schl.columns:
            vals = va_schl[col].dropna()
            ax.hist(vals, bins=15, alpha=0.4, color=color, label=ename, density=True)
    ax.set_title(f'Model 2 VAM - {label}')
    ax.set_xlabel('School Effect (SD units)')
    ax.set_ylabel('Density')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig1_vam_distributions.png'), dpi=150)
plt.close()
print("Saved fig1_vam_distributions.png")

# ===================================================================
# Figure 2: Coefficient Comparison (Replicated vs Published)
# ===================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Replicated and published coefficients for Table 1 Panel A
specs = ['M1\n2002', 'M1\n01-02', 'M1\n98-02', 'M2\n2002', 'M2\n01-02', 'M2\n98-02']
replicated = [0.015, 0.026, 0.010, 0.474, 0.760, 0.898]
published = [0.025, 0.034, 0.014, 0.531, 0.807, 0.966]
rep_se = [0.067, 0.062, 0.068, 0.194, 0.216, 0.325]
pub_se = [0.077, 0.070, 0.072, 0.208, 0.236, 0.342]

x = np.arange(len(specs))
width = 0.35

bars1 = ax.bar(x - width/2, replicated, width, label='Replicated', color='steelblue',
               yerr=[1.96*s for s in rep_se], capsize=4, alpha=0.8)
bars2 = ax.bar(x + width/2, published, width, label='Published', color='coral',
               yerr=[1.96*s for s in pub_se], capsize=4, alpha=0.8)

ax.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Unbiased (coef=1)')
ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
ax.set_xlabel('VAM Specification')
ax.set_ylabel('2SLS Coefficient on School VA')
ax.set_title('Table 1 Panel A: Replicated vs Published Coefficients')
ax.set_xticks(x)
ax.set_xticklabels(specs)
ax.legend()
ax.set_ylim(-0.5, 2.0)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig2_coef_comparison.png'), dpi=150)
plt.close()
print("Saved fig2_coef_comparison.png")

# ===================================================================
# Figure 3: VAM of Assigned School vs Home School
# ===================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for i, (e, ename) in enumerate([('ar', 'Average Residual'), ('mix', 'Mixed Effects')]):
    ax = axes[i]
    va_as = margin[f'as_mod2{e}_all_test']
    va_hm = margin[f'hm_mod2{e}_all_test']

    valid = va_as.notna() & va_hm.notna()
    ax.scatter(va_hm[valid], va_as[valid], alpha=0.1, s=5, color='steelblue')

    # 45-degree line
    lims = [min(va_hm[valid].min(), va_as[valid].min()),
            max(va_hm[valid].max(), va_as[valid].max())]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='45-degree line')

    # Color by lottery status
    won = valid & (margin['lottery'] == 1)
    lost = valid & (margin['lottery'] == 0)
    ax.scatter(va_hm[won], va_as[won], alpha=0.15, s=8, color='green', label='Won Lottery')
    ax.scatter(va_hm[lost], va_as[lost], alpha=0.15, s=8, color='red', label='Lost Lottery')

    ax.set_xlabel('Home School VAM')
    ax.set_ylabel('Assigned School VAM')
    ax.set_title(f'Model 2 ({ename}) - All Years')
    ax.legend(fontsize=8, markerscale=3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig3_vam_scatter.png'), dpi=150)
plt.close()
print("Saved fig3_vam_scatter.png")

# ===================================================================
# Figure 4: SD of School Effects by Specification
# ===================================================================
fig, ax = plt.subplots(figsize=(10, 5))

sd_data = []
for m in [1, 2]:
    for s in ['02', '2yr', 'all']:
        for e in ['ar', 'mix', 'FE']:
            col = f'as_mod{m}{e}_{s}_test'
            if col in margin.columns:
                sd = margin[col].std()
                sd_data.append({'model': m, 'sample': s, 'estimator': e, 'sd': sd})

sd_df = pd.DataFrame(sd_data)
if not sd_df.empty:
    for e, color in [('ar', 'blue'), ('mix', 'red'), ('FE', 'green')]:
        sub = sd_df[sd_df['estimator'] == e]
        labels = [f"M{r['model']}\n{r['sample']}" for _, r in sub.iterrows()]
        ax.bar([f"M{r['model']} {r['sample']}" for _, r in sub.iterrows()],
               sub['sd'], alpha=0.5, label=e, color=color)

ax.set_ylabel('SD of School Effects')
ax.set_title('Standard Deviation of VAM Estimates by Specification')
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_sd_effects.png'), dpi=150)
plt.close()
print("Saved fig4_sd_effects.png")

print("\n=== FIGURES COMPLETE ===")
