"""
03_figures.py - Reproduce Figures 1-6 from Charness & Levin (2005)
"""
import sys
sys.path.insert(0, '.')
from utils import *
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

df = load_data_fast()

# ══════════════════════════════════════════════════════════════════════
# FIGURE 1: Starting-Error Rates (histograms by individual)
# Treatment 1 Phase II + Treatment 2 Phase II
# ══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 4))

# Compute individual starting-error rates for T1 Phase II and T2 Phase II
t12 = df[df['treatment'].isin([1, 2]) & (df['phase'] == 'II')]
ind_start_rates = t12.groupby('Individual')['starting_error'].mean()

bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
bin_labels = ['0-10%', '10-20%', '20-30%', '30-40%', '40-50%',
              '50-60%', '60-70%', '70-80%', '80-90%', '90-100%']

# T1
t1_rates = t12[t12['treatment'] == 1].groupby('Individual')['starting_error'].mean()
t2_rates = t12[t12['treatment'] == 2].groupby('Individual')['starting_error'].mean()

t1_hist, _ = np.histogram(t1_rates, bins=bins)
t2_hist, _ = np.histogram(t2_rates, bins=bins)

x = np.arange(len(bin_labels))
width = 0.35
ax.bar(x - width/2, t1_hist / len(t1_rates) * 100, width, label='Treatment 1', color='darkred')
ax.bar(x + width/2, t2_hist / len(t2_rates) * 100, width, label='Treatment 2', color='lightgreen')
ax.set_ylabel('% of population')
ax.set_title('Figure 1: Starting-Error Rates (Phase II)')
ax.set_xticks(x)
ax.set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=8)
ax.legend()
ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figure1_starting_errors.png'), dpi=150)
plt.close()
print("Figure 1 saved.")

# ══════════════════════════════════════════════════════════════════════
# FIGURE 2: Switching-Error Rates after Right draw (Phase I-II)
# ══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))

t12_data = df[df['treatment'].isin([1, 2]) & df['phase'].isin(['I', 'II'])]

# Compute individual switching error rates after Right draws
right_draws = t12_data[t12_data['Left1'] == 0]
ind_right_err = right_draws.groupby('Individual')['switching_error'].mean()

bins_small = [0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.01]
bin_labels_small = ['0-2%', '2-5%', '5-10%', '10-15%', '15-20%', '20-30%', '30-50%', '50-100%']

# Split by treatment
for t, color in [(1, 'blue'), (2, 'orange')]:
    rates = right_draws[right_draws['treatment'] == t].groupby('Individual')['switching_error'].mean()
    hist, _ = np.histogram(rates, bins=bins_small)
    ax.bar(np.arange(len(bin_labels_small)) + (t - 1.5) * 0.3, hist / len(rates) * 100, 0.28,
           label=f'Treatment {t}', alpha=0.8)

ax.set_ylabel('% of population')
ax.set_title('Figure 2: Switching-Error Rates after Right Draw (Phases I-II)')
ax.set_xticks(np.arange(len(bin_labels_small)))
ax.set_xticklabels(bin_labels_small, rotation=45, ha='right', fontsize=8)
ax.legend()
ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figure2_right_switching.png'), dpi=150)
plt.close()
print("Figure 2 saved.")

# ══════════════════════════════════════════════════════════════════════
# FIGURE 3: Switching-Error Rates after Left Black draw (Phase I-II)
# ══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))

bins_med = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
bin_labels_med = ['0-10%', '10-20%', '20-30%', '30-40%', '40-50%',
                  '50-60%', '60-70%', '70-80%', '80-90%', '90-100%']

left_black = t12_data[(t12_data['Left1'] == 1) & (t12_data['first_draw_black'] == 1)]

# Phase I and Phase II separately
for phase, color, label in [('I', 'darkred', 'Phase I'), ('II', 'lightgreen', 'Phase II')]:
    sub = left_black[left_black['phase'] == phase]
    rates = sub.groupby('Individual')['switching_error'].mean()
    hist, _ = np.histogram(rates, bins=bins_med)
    offset = -0.2 if phase == 'I' else 0.2
    ax.bar(np.arange(len(bin_labels_med)) + offset, hist / len(rates.dropna()) * 100, 0.35,
           label=label, alpha=0.8)

ax.set_ylabel('% of population')
ax.set_title('Figure 3: Switching-Error Rates after Left Black (T1+T2)')
ax.set_xticks(np.arange(len(bin_labels_med)))
ax.set_xticklabels(bin_labels_med, rotation=45, ha='right', fontsize=8)
ax.legend()
ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figure3_left_black.png'), dpi=150)
plt.close()
print("Figure 3 saved.")

# ══════════════════════════════════════════════════════════════════════
# FIGURE 4: Switching-Error Rates after Left White draw (Phase I-II)
# ══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))

left_white = t12_data[(t12_data['Left1'] == 1) & (t12_data['first_draw_black'] == 0)]

for phase, color, label in [('I', 'blue', 'Phase I'), ('II', 'orange', 'Phase II')]:
    sub = left_white[left_white['phase'] == phase]
    rates = sub.groupby('Individual')['switching_error'].mean()
    hist, _ = np.histogram(rates.dropna(), bins=bins_med)
    offset = -0.2 if phase == 'I' else 0.2
    ax.bar(np.arange(len(bin_labels_med)) + offset, hist / len(rates.dropna()) * 100, 0.35,
           label=label, alpha=0.8)

ax.set_ylabel('% of population')
ax.set_title('Figure 4: Switching-Error Rates after Left White (T1+T2)')
ax.set_xticks(np.arange(len(bin_labels_med)))
ax.set_xticklabels(bin_labels_med, rotation=45, ha='right', fontsize=8)
ax.legend()
ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figure4_left_white.png'), dpi=150)
plt.close()
print("Figure 4 saved.")

# ══════════════════════════════════════════════════════════════════════
# FIGURE 5: Treatment 3 Switching-Error Rates
# ══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))

t3 = df[df['treatment'] == 3]

for fav_val, label, offset in [(1, 'Favorable', -0.15), (0, 'Unfavorable', 0.15)]:
    sub = t3[t3['first_draw_favorable'] == fav_val]
    rates = sub.groupby('Individual')['switching_error'].mean()
    hist, _ = np.histogram(rates.dropna(), bins=bins_med)
    ax.bar(np.arange(len(bin_labels_med)) + offset, hist / len(rates.dropna()) * 100, 0.28,
           label=label, alpha=0.8)

ax.set_ylabel('% of population')
ax.set_title('Figure 5: Treatment 3 Switching-Error Rates')
ax.set_xticks(np.arange(len(bin_labels_med)))
ax.set_xticklabels(bin_labels_med, rotation=45, ha='right', fontsize=8)
ax.legend()
ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figure5_t3_switching.png'), dpi=150)
plt.close()
print("Figure 5 saved.")

# ══════════════════════════════════════════════════════════════════════
# FIGURE 6: Cost and Frequency of Errors
# ══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 6))

# Recompute error types as in 02_tables.py
from itertools import product

def compute_eu_second(treatment, state, urn, phase):
    urns = T1_URNS if treatment == 1 else T2_URNS
    b, w = urns[(state, urn)]
    p_black = b / (b + w) if (b + w) > 0 else 0
    if phase in ('I', 'II'):
        return p_black * (PAYOFF_LEFT_BLACK if urn == 'Left' else PAYOFF_RIGHT_BLACK)
    else:
        return p_black * (PAYOFF_RIGHT_BLACK if urn == 'Left' else PAYOFF_LEFT_BLACK)

def cost_switch_error(treatment, phase, first_left, first_black):
    urns = T1_URNS if treatment == 1 else T2_URNS
    first_urn = 'Left' if first_left else 'Right'
    b_up, w_up = urns[('Up', first_urn)]
    b_down, w_down = urns[('Down', first_urn)]
    p_b_up = b_up / (b_up + w_up) if (b_up + w_up) > 0 else 0
    p_b_down = b_down / (b_down + w_down) if (b_down + w_down) > 0 else 0
    if first_black:
        p_d_up, p_d_down = p_b_up, p_b_down
    else:
        p_d_up, p_d_down = 1 - p_b_up, 1 - p_b_down
    denom = 0.5 * p_d_up + 0.5 * p_d_down
    if denom == 0:
        return 0
    p_up = 0.5 * p_d_up / denom
    eu_l = p_up * compute_eu_second(treatment, 'Up', 'Left', phase) + \
           (1 - p_up) * compute_eu_second(treatment, 'Down', 'Left', phase)
    eu_r = p_up * compute_eu_second(treatment, 'Up', 'Right', phase) + \
           (1 - p_up) * compute_eu_second(treatment, 'Down', 'Right', phase)
    return abs(eu_l - eu_r)

costs = []
freqs = []

for t in [1, 2]:
    sub = df[df['treatment'] == t]
    for phase in ['I', 'II', 'III']:
        for left_val in [0, 1]:
            for black_val in [0, 1]:
                mask = (sub['phase'] == phase) & (sub['Left1'] == left_val) & \
                       (sub['first_draw_black'] == black_val)
                obs = sub[mask]
                if len(obs) > 0:
                    freq = obs['switching_error'].mean()
                    cost = cost_switch_error(t, phase, bool(left_val), bool(black_val))
                    costs.append(cost)
                    freqs.append(freq)

# T3
t3 = df[df['treatment'] == 3]
for phase in ['I', 'II']:
    for fav_val in [0, 1]:
        mask = (t3['phase'] == phase) & (t3['first_draw_favorable'] == fav_val)
        obs = t3[mask]
        if len(obs) > 0:
            freq = obs['switching_error'].mean()
            cost = cost_switch_error(2, phase if phase == 'I' else 'III', True, bool(fav_val))
            costs.append(cost)
            freqs.append(freq)

ax.scatter(costs, freqs, c='navy', marker='D', s=40, zorder=5)
ax.set_xlabel('Cost of error')
ax.set_ylabel('Frequency of error')
ax.set_title('Figure 6: Cost and Frequency of Errors')
ax.set_xlim(0, 0.65)
ax.set_ylim(0, 0.75)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figure6_cost_frequency.png'), dpi=150)
plt.close()
print("Figure 6 saved.")

print("\nAll figures saved to output/")
