"""
03_figures.py – Diagnostic figures (no figures in original replication package)
Banerjee, Duflo & Hornbeck (2014)
"""

import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')
from utils import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def figure_1_renewal_by_treatment():
    """Bar chart of loan renewal rates by treatment status."""
    loans = load_loans()
    treat_df = load_treatment()
    loans = loans.merge(treat_df[['center_id', 'village_id', 'treatment']], on='center_id', how='left')
    eligible = loans[loans['eligible'] == 1].dropna(subset=['renewal_first'])

    treat_rate = eligible[eligible['treatment'] == 1]['renewal_first'].mean()
    ctrl_rate = eligible[eligible['treatment'] == 0]['renewal_first'].mean()

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(['Control', 'Treatment'], [ctrl_rate, treat_rate],
                  color=['#888888', '#333333'], width=0.5)
    ax.set_ylabel('Loan Renewal Rate')
    ax.set_title('First Loan Renewal by Treatment Status')
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, [ctrl_rate, treat_rate]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Figure1_renewal.pdf'))
    plt.close()
    print("Figure 1 saved.")


def figure_2_coefficient_plot():
    """Coefficient plot for Table 3 interaction terms."""
    # Published coefficients and SEs from table3.csv
    rows = [
        ('Chronic disease', -0.016, 0.013),
        ('Family chronic', -0.010, 0.013),
        ('Health scale', -0.002, 0.015),
        ('Poor health', -0.027, 0.013),
        ('Consult symptoms', -0.007, 0.015),
        ('Smoke/drink', -0.015, 0.015),
        ('Plan baby', -0.014, 0.017),
        ('Females 17-24', -0.002, 0.014),
        ('Overnight (pred)', -0.005, 0.014),
        ('Health exp (pred)', -0.019, 0.015),
        ('Pregnancy (pred)', -0.009, 0.016),
        ('Overnight (end)', 0.013, 0.015),
        ('Health exp (end)', -0.010, 0.014),
        ('New baby (end)', 0.017, 0.013),
        ('Consumption', 0.032, 0.017),
        ('Financial status', -0.012, 0.017),
        ('Own business', -0.026, 0.018),
    ]

    labels = [r[0] for r in rows]
    coefs = [r[1] for r in rows]
    ci95 = [1.96 * r[2] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 8))
    y_pos = np.arange(len(labels))
    ax.errorbar(coefs, y_pos, xerr=ci95, fmt='o', color='black',
                capsize=3, markersize=5, linewidth=1)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Interaction Coefficient (Treatment × Normalized Characteristic)')
    ax.set_title('Adverse Selection Tests (Table 3)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Figure2_coefplot.pdf'))
    plt.close()
    print("Figure 2 saved.")


def figure_3_baseline_balance():
    """Baseline balance visualization (Table 1 differences)."""
    rows = [
        ('Serious health events', 0.141, 0.098),
        ('Total health expense', 422.764, 445.329),
        ('Hospitalization expense', 93.747, 96.487),
        ('Consumption', -774.012, 2164.823),
        ('Insurance available', -0.019, 0.020),
        ('Insurance owned', 0.002, 0.002),
    ]

    # Normalize by SE for visualization (t-statistics)
    labels = [r[0] for r in rows]
    t_stats = [r[1]/r[2] for r in rows]

    fig, ax = plt.subplots(figsize=(7, 5))
    y_pos = np.arange(len(labels))
    colors = ['#333333' if abs(t) >= 1.96 else '#888888' for t in t_stats]
    ax.barh(y_pos, t_stats, color=colors, height=0.5)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.axvline(-1.96, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axvline(1.96, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('t-statistic (Treatment - Control)')
    ax.set_title('Baseline Balance (Table 1)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Figure3_balance.pdf'))
    plt.close()
    print("Figure 3 saved.")


def main():
    figure_1_renewal_by_treatment()
    figure_2_coefficient_plot()
    figure_3_baseline_balance()
    print(f"\nAll figures saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
