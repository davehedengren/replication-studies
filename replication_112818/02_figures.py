"""Replicate Figures 1 and 2 from paper 112818.

Figure 1: US Economic Policy Uncertainty and Government Activity
  - Left axis: Policy uncertainty (norm_news), Pages of regulation (norm_reg)
  - Right axis: Government share of GDP (govsharegdp)

Figure 2: US Economic Policy Uncertainty and Political Polarization
  - Left axis: Polarization of roll-calls (diff_std), SD of district pres vote
    (sd_dshare_std), Voter perception of party diff (seediff_std)
  - Right axis: Policy uncertainty (norm_news)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import OUTPUT_DIR
import os

df = pd.read_csv(os.path.join(OUTPUT_DIR, "cleaned.csv"))

# ============================================================
# Figure 1: US Economic Policy Uncertainty and Government Activity
# ============================================================
fig, ax1 = plt.subplots(figsize=(10, 6))

# Left axis: policy uncertainty and regulation pages (both normalized to mean=100)
ax1.plot(df['year'], df['norm_news'], 'b-s', linewidth=2, markersize=5,
         label='Policy uncertainty')
ax1.plot(df['year'], df['norm_reg'], 'r-^', linewidth=2, markersize=5,
         linestyle='dashdot', label='Pages of regulation')
ax1.set_xlabel('year')
ax1.set_ylabel('')
ax1.set_xlim(1950, 2012)
ax1.set_xticks(range(1950, 2011, 10))
ax1.set_ylim(0, 250)

# Right axis: government share of GDP
ax2 = ax1.twinx()
ax2.plot(df['year'], df['govsharegdp'], 'g-+', linewidth=2, markersize=7,
         linestyle='dashed', label='Government share of GDP')
ax2.set_ylabel('Government share of GDP')
ax2.set_ylim(20, 38)

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower center',
           bbox_to_anchor=(0.5, -0.15), ncol=2)

plt.title('Figure 1: US Economic Policy Uncertainty and Government Activity')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figure1.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Figure 1 saved.")

# Print key values for verification
print("\nFigure 1 key values:")
print(f"  norm_news 1950: {df.loc[df['year']==1950, 'norm_news'].values[0]:.2f}")
print(f"  norm_news 2012: {df.loc[df['year']==2012, 'norm_news'].values[0]:.2f}")
print(f"  norm_reg 1950: {df.loc[df['year']==1950, 'norm_reg'].values[0]:.2f}")
print(f"  norm_reg 2012: {df.loc[df['year']==2012, 'norm_reg'].values[0]:.2f}")
print(f"  gov 1950: {df.loc[df['year']==1950, 'govsharegdp'].values[0]:.1f}")
print(f"  gov 2012: {df.loc[df['year']==2012, 'govsharegdp'].values[0]:.1f}")

# ============================================================
# Figure 2: US Economic Policy Uncertainty and Political Polarization
# ============================================================
fig, ax1 = plt.subplots(figsize=(10, 6))

# Left axis: polarization measures (all standardized to mean=0, sd=1)
ax1.plot(df['year'], df['diff_std'], 'r-^', linewidth=2, markersize=5,
         linestyle='dashdot', label='Polarization of roll-calls')
ax1.plot(df['year'], df['sd_dshare_std'], 'g-+', linewidth=2, markersize=7,
         linestyle='dashed', label='SD of district pres. vote')
ax1.plot(df['year'], df['seediff_std'], color='orange', marker='o', linewidth=2,
         markersize=5, linestyle='dashed', label='Voter perception of party diff')
ax1.set_xlabel('year')
ax1.set_ylabel('')
ax1.set_xlim(1950, 2012)
ax1.set_xticks(range(1950, 2011, 10))

# Right axis: policy uncertainty
ax2 = ax1.twinx()
ax2.plot(df['year'], df['norm_news'], 'b-s', linewidth=2, markersize=5,
         label='Policy uncertainty')
ax2.set_ylabel('')

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower center',
           bbox_to_anchor=(0.5, -0.15), ncol=2)

plt.title('Figure 2: US Economic Policy Uncertainty and Political Polarization')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figure2.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\nFigure 2 saved.")

# Print key values for verification
print("\nFigure 2 key values:")
print(f"  diff_std range: {df['diff_std'].min():.4f} to {df['diff_std'].max():.4f}")
print(f"  sd_dshare_std range: {df['sd_dshare_std'].min():.4f} to {df['sd_dshare_std'].max():.4f}")
print(f"  seediff_std range: {df['seediff_std'].min():.4f} to {df['seediff_std'].max():.4f}")
print(f"  diff_std non-missing: {df['diff_std'].notna().sum()}")
print(f"  sd_dshare_std non-missing: {df['sd_dshare_std'].notna().sum()}")
print(f"  seediff_std non-missing: {df['seediff_std'].notna().sum()}")

# Correlations
print("\nCorrelations with norm_news:")
for col in ['norm_reg', 'govsharegdp', 'diff_std', 'sd_dshare_std', 'seediff_std']:
    valid = df[['norm_news', col]].dropna()
    if len(valid) > 2:
        r = valid['norm_news'].corr(valid[col])
        print(f"  {col}: r={r:.4f} (n={len(valid)})")
