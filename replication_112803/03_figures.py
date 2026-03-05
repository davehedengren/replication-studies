"""
03_figures.py — Replicate Figures 1-3 (main text) and appendix figures

Core regression: log_Y = ind_FE + time_FE + sum(beta_t * IT_j * 1{year==t}) + e
Weighted by employment shares, clustered SEs by industry.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import run_it_year_regression, OUT_DIR, DATA_DIR

print("=" * 70)
print("FIGURES: Main text and appendix regressions")
print("=" * 70)

# ── Load data ──
analysis = pd.read_pickle(os.path.join(OUT_DIR, 'analysis_data.pkl'))

# Load published estimates for comparison
pub_main = pd.read_stata(os.path.join(DATA_DIR, 'out', 'main-text', 'dta', 'estimates.dta'),
                          convert_categoricals=False)
pub_app = pd.read_stata(os.path.join(DATA_DIR, 'out', 'appendix', 'dta', 'estimates.dta'),
                         convert_categoricals=False)

# ── Helper to compare results ──
def compare_results(results_df, pub_df, itvar, depvar, drop_comp=None, restrict_sic3438=None, label=""):
    """Compare replicated coefficients to published values."""
    mask = (pub_df['ITvar'].str.strip() == itvar) & (pub_df['depvar'].str.strip() == depvar)
    if drop_comp is not None and 'drop_comp' in pub_df.columns:
        mask &= (pub_df['drop_comp'] == drop_comp)
    if restrict_sic3438 is not None and 'restrict_to_sic_3438' in pub_df.columns:
        mask &= (pub_df['restrict_to_sic_3438'] == restrict_sic3438)

    pub = pub_df.loc[mask, ['year', 'beta', 'se']].copy()
    pub.columns = ['year', 'pub_beta', 'pub_se']

    merged = results_df.merge(pub, on='year', how='left')
    merged['diff_beta'] = merged['beta'] - merged['pub_beta']
    merged['diff_se'] = merged['se'] - merged['pub_se']

    max_beta_diff = merged['diff_beta'].abs().max()
    max_se_diff = merged['diff_se'].abs().max()
    mean_beta_diff = merged['diff_beta'].abs().mean()

    status = "MATCH" if max_beta_diff < 0.01 else ("CLOSE" if max_beta_diff < 0.1 else "DIFFERS")
    print(f"  {label}: max|diff_beta|={max_beta_diff:.4f}, max|diff_se|={max_se_diff:.4f} [{status}]")
    return merged

# ── Run all main-text specifications ──
specs = []

# Figure 1A: laborprod with cimean, all mfg vs excl comp
for drop_comp in [0, 1]:
    if drop_comp == 0:
        data = analysis.copy()
        label = "Fig1A all_mfg"
    else:
        data = analysis[analysis['comp_broad'] == 0].copy()
        label = "Fig1A excl_comp"

    res, _ = run_it_year_regression(data, 'log_laborprod', 'cimean')
    res['ITvar'] = 'cimean'
    res['depvar'] = 'laborprod'
    res['drop_comp'] = drop_comp
    res['restrict_to_sic_3438'] = 0
    specs.append(res)
    compare_results(res, pub_main, 'cimean', 'laborprod', drop_comp=drop_comp,
                    restrict_sic3438=0, label=label)

# Figure 1B specs: laborprod, excl comp, SIC 34-38
for itvar in ['cimean', 'ci8792', 'smt']:
    if itvar == 'smt':
        data = analysis[(analysis['comp_broad'] == 0) &
                        (analysis['sic87dd'] >= 3400) & (analysis['sic87dd'] <= 3899) &
                        analysis['smt'].notna()].copy()
    else:
        data = analysis[(analysis['comp_broad'] == 0) &
                        (analysis['sic87dd'] >= 3400) & (analysis['sic87dd'] <= 3899)].copy()

    res, _ = run_it_year_regression(data, 'log_laborprod', itvar)
    res['ITvar'] = itvar
    res['depvar'] = 'laborprod'
    res['drop_comp'] = 1
    res['restrict_to_sic_3438'] = 1
    specs.append(res)
    compare_results(res, pub_main, itvar, 'laborprod', drop_comp=1,
                    restrict_sic3438=1, label=f"Fig1B {itvar}")

# Figures 2A, 3A: cimean, excl comp, all mfg — real/nom shipments, emp, pay
for depvar in ['real_vship', 'nom_vship', 'emp', 'pay']:
    data = analysis[analysis['comp_broad'] == 0].copy()
    res, _ = run_it_year_regression(data, f'log_{depvar}', 'cimean')
    res['ITvar'] = 'cimean'
    res['depvar'] = depvar
    res['drop_comp'] = 1
    res['restrict_to_sic_3438'] = 0
    specs.append(res)
    compare_results(res, pub_main, 'cimean', depvar, drop_comp=1,
                    restrict_sic3438=0, label=f"cimean/{depvar}")

# Figures 2B, 3B: smt, excl comp, SIC 34-38
for depvar in ['real_vship', 'nom_vship', 'emp', 'pay']:
    data = analysis[(analysis['comp_broad'] == 0) &
                    (analysis['sic87dd'] >= 3400) & (analysis['sic87dd'] <= 3899) &
                    analysis['smt'].notna()].copy()
    res, _ = run_it_year_regression(data, f'log_{depvar}', 'smt')
    res['ITvar'] = 'smt'
    res['depvar'] = depvar
    res['drop_comp'] = 1
    res['restrict_to_sic_3438'] = 1
    specs.append(res)
    compare_results(res, pub_main, 'smt', depvar, drop_comp=1,
                    restrict_sic3438=1, label=f"smt/{depvar}/sic3438")

# ── Appendix specs (excl comp, various IT measures) ──
print("\nAppendix regressions:")

# Appendix vintage figures: ci7782, ci8792, ci0207 for laborprod, real_vship, nom_vship, emp
for itvar in ['cimean', 'ci7782', 'ci8792', 'ci0207']:
    data = analysis[analysis['comp_broad'] == 0].copy()
    for depvar in ['laborprod', 'real_vship', 'nom_vship', 'emp']:
        res, _ = run_it_year_regression(data, f'log_{depvar}', itvar)
        res['ITvar'] = itvar
        res['depvar'] = depvar
        compare_results(res, pub_app, itvar, depvar, label=f"App {itvar}/{depvar}")

# Appendix cimean and smt for additional outcomes
for itvar in ['cimean', 'smt']:
    if itvar == 'smt':
        data = analysis[(analysis['comp_broad'] == 0) &
                        (analysis['sic87dd'] >= 3400) & (analysis['sic87dd'] <= 3899) &
                        analysis['smt'].notna()].copy()
    else:
        data = analysis[analysis['comp_broad'] == 0].copy()

    for depvar in ['piship', 'nom_vadd', 'tfp', 'prodpay', 'nonprodpay']:
        res, _ = run_it_year_regression(data, f'log_{depvar}', itvar)
        res['ITvar'] = itvar
        res['depvar'] = depvar
        compare_results(res, pub_app, itvar, depvar, label=f"App {itvar}/{depvar}")

# ── Generate Figures ──
print("\nGenerating figures...")
all_specs = pd.concat(specs, ignore_index=True)

# Figure 1A
fig, ax = plt.subplots(figsize=(8, 6))
for dc, style, color, lbl in [(0, '-o', 'navy', 'All Mfg Industries'),
                                (1, '-^', 'maroon', 'Excluding Computer-Producing')]:
    sub = all_specs[(all_specs['ITvar'] == 'cimean') & (all_specs['depvar'] == 'laborprod') &
                    (all_specs['drop_comp'] == dc) & (all_specs['restrict_to_sic_3438'] == 0)]
    ax.plot(sub['year'], sub['beta'], style, color=color, markersize=4, label=lbl)
    ax.plot(sub['year'], sub['beta_low'], '--', color=color, alpha=0.5)
    ax.plot(sub['year'], sub['beta_high'], '--', color=color, alpha=0.5)
ax.set_ylabel('Coef. on IT Measure x Year Dummy')
ax.set_xlabel('')
ax.legend(fontsize=9)
ax.set_title('Figure 1A: IT Intensity and Log Real Shipments per Worker')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'Figure_1A.png'), dpi=150)
plt.close()

# Figure 1B
fig, ax = plt.subplots(figsize=(8, 6))
for itvar, style, color, lbl in [('cimean', '-o', 'navy', '1977-2007 Comp Investments'),
                                   ('ci8792', '-^', 'maroon', '1987/1992 Comp Investments'),
                                   ('smt', '-s', 'green', '1988/1993 Technology Usage')]:
    sub = all_specs[(all_specs['ITvar'] == itvar) & (all_specs['depvar'] == 'laborprod') &
                    (all_specs['drop_comp'] == 1) & (all_specs['restrict_to_sic_3438'] == 1)]
    ax.plot(sub['year'], sub['beta'], style, color=color, markersize=4, label=lbl)
ax.set_ylabel('Coef. on IT Measure x Year Dummy')
ax.legend(fontsize=9)
ax.set_title('Figure 1B: IT Intensity and Log Real Shipments per Worker (SIC 34-38)')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'Figure_1B.png'), dpi=150)
plt.close()

# Figure 2A
fig, ax = plt.subplots(figsize=(8, 6))
for dv, style, color, lbl in [('real_vship', '-o', 'navy', 'Real Shipments'),
                                ('nom_vship', '-^', 'maroon', 'Nominal Shipments')]:
    sub = all_specs[(all_specs['ITvar'] == 'cimean') & (all_specs['depvar'] == dv) &
                    (all_specs['drop_comp'] == 1) & (all_specs['restrict_to_sic_3438'] == 0)]
    ax.plot(sub['year'], sub['beta'], style, color=color, markersize=4, label=lbl)
    ax.plot(sub['year'], sub['beta_low'], '--', color=color, alpha=0.5)
    ax.plot(sub['year'], sub['beta_high'], '--', color=color, alpha=0.5)
ax.set_ylabel('Coef. on IT Measure x Year Dummy')
ax.legend(fontsize=9)
ax.set_title('Figure 2A: IT Intensity and Log Shipments (excl. computer sector)')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'Figure_2A.png'), dpi=150)
plt.close()

# Figure 3A
fig, ax = plt.subplots(figsize=(8, 6))
for dv, style, color, lbl in [('emp', '-o', 'navy', 'Employment'),
                                ('pay', '-^', 'maroon', 'Real Wage Bill')]:
    sub = all_specs[(all_specs['ITvar'] == 'cimean') & (all_specs['depvar'] == dv) &
                    (all_specs['drop_comp'] == 1) & (all_specs['restrict_to_sic_3438'] == 0)]
    ax.plot(sub['year'], sub['beta'], style, color=color, markersize=4, label=lbl)
    ax.plot(sub['year'], sub['beta_low'], '--', color=color, alpha=0.5)
    ax.plot(sub['year'], sub['beta_high'], '--', color=color, alpha=0.5)
ax.set_ylabel('Coef. on IT Measure x Year Dummy')
ax.legend(fontsize=9)
ax.set_title('Figure 3A: IT Intensity, Employment, and Wage Bill (excl. computer sector)')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'Figure_3A.png'), dpi=150)
plt.close()

print("Figures saved to output/")
print("=" * 70)
