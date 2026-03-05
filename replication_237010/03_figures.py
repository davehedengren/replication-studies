"""
03_figures.py — Reproduce main figures from pre-computed data for GHT (2025).

Paper: "Temporary Layoffs, Loss-of-Recall, and Cyclical Unemployment Dynamics"
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from utils import *

print('=' * 70)
print('03_figures.py — Figure replication')
print('=' * 70)

# ══════════════════════════════════════════════════════════════════════
# FIGURE 1: TL unemployment and JL-from-TL, 1979-2019
# ══════════════════════════════════════════════════════════════════════
print('\n-- Figure 1: TL unemployment and JL-from-TL, 1979-2019 --')

adj = load_for_adjusted_plot()
adj['date'] = pd.to_datetime(adj[['year', 'month']].assign(day=1))

# Pre-COVID period
adj_pre = adj[adj.year <= 2019].copy()

fig, ax = plt.subplots(figsize=(12, 5))

# Column names: trate (TL rate), prate (JL rate), urate (total), JL_from_TL
ax.plot(adj_pre['date'], adj_pre['trate'] * 100, color='tab:blue',
        linewidth=1.5, label='TL unemployment ($u^{TL}$)')

if 'JL_from_TL' in adj_pre.columns:
    jl_from_tl = adj_pre['trate'] + adj_pre['JL_from_TL']
    ax.plot(adj_pre['date'], jl_from_tl * 100, color='tab:orange',
            linewidth=1.5, label='$u^{TL}$ + $u^{JL-from-TL}$')

ax.set_ylabel('Percent')
ax.set_xlabel('')
ax.legend(loc='upper right')
ax.set_title('Figure 1: TL unemployment and JL-from-TL, 1979-2019')
ax.grid(True, alpha=0.3)

# Shade recessions
for start, end in [('1980-01', '1980-07'), ('1981-07', '1982-11'),
                    ('1990-07', '1991-03'), ('2001-03', '2001-11'),
                    ('2007-12', '2009-06')]:
    ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.1, color='gray')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figure_1_adjusted_unemployment.png'), dpi=150)
plt.close()
print('  Saved figure_1_adjusted_unemployment.png')

# ══════════════════════════════════════════════════════════════════════
# FIGURE 2: TL unemployment and JL-from-TL, COVID period (2020-2022)
# ══════════════════════════════════════════════════════════════════════
print('\n-- Figure 2: TL and JL-from-TL, COVID period --')

matlab = load_for_matlab()
matlab['date'] = pd.to_datetime(matlab[['year', 'month']].assign(day=1))

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(matlab['date'], matlab['Trate_adj'] * 100, color='tab:blue',
        linewidth=2, label='TL unemployment ($u^{TL}$)')
if 'PfromT' in matlab.columns:
    jl_from_tl_covid = matlab['Trate_adj'] + matlab['PfromT']
    ax.plot(matlab['date'], jl_from_tl_covid * 100, color='tab:orange',
            linewidth=2, label='$u^{TL}$ + $u^{JL-from-TL}$')

ax.set_ylabel('Percent')
ax.legend(loc='upper right')
ax.set_title('Figure 2: TL unemployment and JL-from-TL, 2020-2022')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figure_2_covid_unemployment.png'), dpi=150)
plt.close()
print('  Saved figure_2_covid_unemployment.png')

# ══════════════════════════════════════════════════════════════════════
# FIGURE A.5: SIPP Exit Hazards by Duration
# ══════════════════════════════════════════════════════════════════════
print('\n-- Figure A.5: SIPP exit hazards by duration --')

haz = load_sipp_hazard()
tl = haz[haz.rwkesr2 == 3].copy()
ps = haz[haz.rwkesr2 == 4].copy()

fig, axes = plt.subplots(2, 3, figsize=(14, 8))

# Panel A: Employment prob (TL)
ax = axes[0, 0]
ax.bar(tl.duration, tl.pE, color='tab:blue', alpha=0.7)
ax.set_title('A. Pr(E) — TL')
ax.set_xlabel('Duration (months)')
ax.set_ylabel('Probability')
ax.set_ylim(0, 0.8)

# Panel B: Employment prob (PS)
ax = axes[0, 1]
ax.bar(ps.duration, ps.pE, color='tab:orange', alpha=0.7)
ax.set_title('B. Pr(E) — PS')
ax.set_xlabel('Duration (months)')
ax.set_ylim(0, 0.8)

# Panel C: Recall prob (TL)
ax = axes[0, 2]
ax.bar(tl.duration, tl.pR, color='tab:blue', alpha=0.7)
ax.set_title('C. Recall — TL')
ax.set_xlabel('Duration (months)')
ax.set_ylim(0, 0.5)

# Panel D: New job prob (TL)
ax = axes[1, 0]
ax.bar(tl.duration, tl.pN, color='tab:blue', alpha=0.7)
ax.set_title('D. New job — TL')
ax.set_xlabel('Duration (months)')
ax.set_ylim(0, 0.5)

# Panel E: Recall prob (PS)
ax = axes[1, 1]
ax.bar(ps.duration, ps.pR, color='tab:orange', alpha=0.7)
ax.set_title('E. Recall — PS')
ax.set_xlabel('Duration (months)')
ax.set_ylim(0, 0.05)
# Add vertical dashed line at duration 4 (SIPP measurement limit)
ax.axvline(x=4.5, color='gray', linestyle='--', alpha=0.7)

# Panel F: New job prob (PS)
ax = axes[1, 2]
ax.bar(ps.duration, ps.pN, color='tab:orange', alpha=0.7)
ax.set_title('F. New job — PS')
ax.set_xlabel('Duration (months)')
ax.set_ylim(0, 0.5)
ax.axvline(x=4.5, color='gray', linestyle='--', alpha=0.7)

fig.suptitle('Figure A.5: Recall and new-job-finding hazard, TL and PS separators', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figure_A5_sipp_hazards.png'), dpi=150, bbox_inches='tight')
plt.close()
print('  Saved figure_A5_sipp_hazards.png')

# ══════════════════════════════════════════════════════════════════════
# FIGURE: Transition probabilities time series
# ══════════════════════════════════════════════════════════════════════
print('\n-- Figure: Key transition probabilities over time --')

tp = load_transition_probabilities()
tp['date'] = pd.to_datetime(tp[['year', 'month']].assign(day=1))
tp_pre = tp[tp.year <= 2019].copy()

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

panels = [
    (axes[0, 0], 'ctflow_et', 'E -> TL', 'tab:red'),
    (axes[0, 1], 'ctflow_te', 'TL -> E', 'tab:green'),
    (axes[1, 0], 'ctflow_ep', 'E -> JL', 'tab:red'),
    (axes[1, 1], 'ctflow_pe', 'JL -> E', 'tab:green'),
]

for ax, col, title, color in panels:
    ax.plot(tp_pre['date'], tp_pre[col], color=color, linewidth=0.7, alpha=0.8)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    # Shade recessions
    for start, end in [('1980-01', '1980-07'), ('1981-07', '1982-11'),
                        ('1990-07', '1991-03'), ('2001-03', '2001-11'),
                        ('2007-12', '2009-06')]:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.1, color='gray')

fig.suptitle('Key Transition Probabilities, 1978-2019')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figure_transition_probs.png'), dpi=150)
plt.close()
print('  Saved figure_transition_probs.png')

# ══════════════════════════════════════════════════════════════════════
# FIGURE: COVID-period transition probabilities
# ══════════════════════════════════════════════════════════════════════
print('\n-- Figure: COVID-period transition probabilities --')

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

panels_covid = [
    (axes[0, 0], 'pET', 'E -> TL'),
    (axes[0, 1], 'pTE', 'TL -> E'),
    (axes[1, 0], 'pEP', 'E -> JL'),
    (axes[1, 1], 'pPE', 'JL -> E'),
]

for ax, col, title in panels_covid:
    if col in matlab.columns:
        ax.plot(matlab['date'], matlab[col], 'o-', linewidth=1.5, markersize=4)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

fig.suptitle('Transition Probabilities During COVID, 2019-2021')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figure_covid_transitions.png'), dpi=150)
plt.close()
print('  Saved figure_covid_transitions.png')

# ══════════════════════════════════════════════════════════════════════
# FIGURE: Unemployment stocks during COVID
# ══════════════════════════════════════════════════════════════════════
print('\n-- Figure: Unemployment stocks during COVID --')

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

if 'Trate_adj' in matlab.columns:
    axes[0].plot(matlab['date'], matlab['Trate_adj'] * 100, 'o-', color='tab:blue')
    axes[0].set_title('TL Unemployment (%)')
    axes[0].grid(True, alpha=0.3)

if 'Prate_adj' in matlab.columns:
    axes[1].plot(matlab['date'], matlab['Prate_adj'] * 100, 'o-', color='tab:red')
    axes[1].set_title('JL Unemployment (%)')
    axes[1].grid(True, alpha=0.3)

if 'Urate_adj' in matlab.columns:
    axes[2].plot(matlab['date'], matlab['Urate_adj'] * 100, 'o-', color='tab:purple')
    axes[2].set_title('Total Unemployment (%)')
    axes[2].grid(True, alpha=0.3)

fig.suptitle('Unemployment Stocks During COVID')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figure_covid_stocks.png'), dpi=150)
plt.close()
print('  Saved figure_covid_stocks.png')

print('\n' + '=' * 70)
print('03_figures.py — DONE')
print('=' * 70)
