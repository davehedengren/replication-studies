"""
03_figures.py — Reproduce Figure 1 from Danieli et al. (2025).

Figure 1: Correlation-correlation scatter plot for the ADH application.
For each negative control (NC), plots |cor(outcome, NC)| vs |cor(IV, NC)|
after residualizing IV, outcome, and NCs on controls using WLS.

Paper: "Negative Control Falsification Tests for IV Designs"
Replication package: openICPSR 238658-V1
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from utils import OUTPUT_DIR, load_adh, load_adh_preperiod

print("=" * 60)
print("03_figures.py — Reproducing Figure 1")
print("=" * 60)


# ══════════════════════════════════════════════════════════════════════
# LOAD AND PREPARE ADH DATA (same as 02_tables.py)
# ══════════════════════════════════════════════════════════════════════

adh = load_adh()
adh_pre = load_adh_preperiod()

# Create year-2000 versions of controls and IV/outcome variables
for czone_grp in adh.groupby('czone'):
    czone, grp = czone_grp
    for var in ['l_shind_manuf_cbp', 'reg_midatl', 'reg_encen', 'reg_wncen',
                'reg_satl', 'reg_escen', 'reg_wscen', 'reg_mount', 'reg_pacif',
                'l_sh_popedu_c', 'l_sh_popfborn', 'l_sh_empl_f',
                'l_sh_routine33', 'l_task_outsource']:
        val_2000 = grp.loc[grp['yr'] == 2000, var].values
        if len(val_2000) > 0:
            adh.loc[grp.index, f'{var}_2000'] = val_2000[0]
    for var, newname in [('d_tradeotch_pw_lag', 'instrument2000'),
                         ('d_tradeusch_pw', 'exposure2000'),
                         ('d_sh_empl_mfg', 'outcome2000'),
                         ('d_tradeotch_pw_lag', 'instrument1990'),
                         ('d_tradeusch_pw', 'exposure1990'),
                         ('d_sh_empl_mfg', 'outcome1990')]:
        yr_val = 2000 if '2000' in newname else 1990
        val = grp.loc[grp['yr'] == yr_val, var].values
        if len(val) > 0:
            adh.loc[grp.index, newname] = val[0]

# Filter to 1990 cross-section
china_1990 = adh[adh['yr'] == 1990].copy().reset_index(drop=True)

# Remove columns matching R's logic
drop_cols = [c for c in china_1990.columns
             if c.startswith('d_') or c.startswith('relchg') or c.startswith('lnchg')]
drop_cols += ['l_tradeusch_pw', 'l_tradeotch_pw', 'czone', 'yr', 't2', 'city']
china_1990 = china_1990.drop(columns=[c for c in drop_cols if c in china_1990.columns])

# Add lagged outcomes from pre-period
china_1990_cz = adh[adh['yr'] == 1990][['czone']].reset_index(drop=True)
china_1990['czone_tmp'] = china_1990_cz['czone'].values
china_1990 = china_1990.sort_values('czone_tmp').reset_index(drop=True)

adh_pre_1970 = adh_pre[adh_pre['yr'] == 1970].sort_values('czone').reset_index(drop=True)
adh_pre_1980 = adh_pre[adh_pre['yr'] == 1980].sort_values('czone').reset_index(drop=True)

if len(adh_pre_1970) == len(china_1990):
    china_1990['outcome1970'] = adh_pre_1970['d_sh_empl_mfg'].values
    china_1990['outcome1980'] = adh_pre_1980['d_sh_empl_mfg'].values

china_1990 = china_1990.drop(columns=['czone_tmp'])

print(f"  china_1990: {china_1990.shape[0]} rows, {china_1990.shape[1]} columns")

# Define controls (14 variables, matching col_6 minus t2)
all_controls = ['l_shind_manuf_cbp_2000',
                'reg_midatl_2000', 'reg_encen_2000', 'reg_wncen_2000',
                'reg_satl_2000', 'reg_escen_2000', 'reg_wscen_2000',
                'reg_mount_2000', 'reg_pacif_2000',
                'l_sh_popedu_c_2000', 'l_sh_popfborn_2000', 'l_sh_empl_f_2000',
                'l_sh_routine33_2000', 'l_task_outsource_2000']

all_controls_old = ['l_shind_manuf_cbp',
                    'reg_midatl', 'reg_encen', 'reg_wncen',
                    'reg_satl', 'reg_escen', 'reg_wscen',
                    'reg_mount', 'reg_pacif',
                    'l_sh_popedu_c', 'l_sh_popfborn', 'l_sh_empl_f',
                    'l_sh_routine33', 'l_task_outsource']

variables_to_remove = ['timepwt48', 'instrument2000', 'outcome2000', 'statefip',
                       'instrument1990', 'exposure1990', 'exposure2000']

# NC selection: everything except removed vars + controls + old controls
exclude_set = set(variables_to_remove + all_controls + all_controls_old)
nc_cols = [c for c in china_1990.columns if c not in exclude_set]
nc_cols = [c for c in nc_cols if china_1990[c].notna().all()]

print(f"  Number of negative controls: {len(nc_cols)}")

IV = china_1990['instrument2000'].values
outcome = china_1990['outcome2000'].values
weights = china_1990['timepwt48'].values
cntrls = china_1990[all_controls].values


# ══════════════════════════════════════════════════════════════════════
# RESIDUALIZE IV, OUTCOME, AND NCs ON CONTROLS (WLS)
# ══════════════════════════════════════════════════════════════════════
# Matches R's get_resids_data with z_resid="lm", nc_resid="lm":
#   - Residualize Z (IV or outcome) on controls using WLS
#   - Residualize each NC on controls using WLS

print("\n── Residualizing variables on controls using WLS ──")

def residualize_on_controls(y, controls, w):
    """Residualize y on controls using WLS (matching R's lm with weights)."""
    X = sm.add_constant(controls)
    model = sm.WLS(y, X, weights=w).fit()
    return y - model.fittedvalues

# Residualize IV on controls
iv_resid = residualize_on_controls(IV, cntrls, weights)

# Residualize outcome on controls
outcome_resid = residualize_on_controls(outcome, cntrls, weights)

# Residualize each NC on controls
nc_resids = np.column_stack([
    residualize_on_controls(china_1990[nc].values, cntrls, weights)
    for nc in nc_cols
])

print(f"  Residualized IV, outcome, and {nc_resids.shape[1]} NCs")


# ══════════════════════════════════════════════════════════════════════
# COMPUTE CORRELATIONS
# ══════════════════════════════════════════════════════════════════════
# R code: cor(iv_resid$Z, iv_resid$NC) and cor(outcome_resid$Z, iv_resid$NC)
# These are standard (unweighted) Pearson correlations of the residualized variables.

cor_iv_nc = np.array([np.corrcoef(iv_resid, nc_resids[:, j])[0, 1]
                      for j in range(nc_resids.shape[1])])

cor_outcome_nc = np.array([np.corrcoef(outcome_resid, nc_resids[:, j])[0, 1]
                           for j in range(nc_resids.shape[1])])

print(f"\n  Correlation ranges:")
print(f"    |cor(IV, NC)|:      [{np.abs(cor_iv_nc).min():.4f}, {np.abs(cor_iv_nc).max():.4f}]")
print(f"    |cor(outcome, NC)|: [{np.abs(cor_outcome_nc).min():.4f}, {np.abs(cor_outcome_nc).max():.4f}]")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 1: CORRELATION-CORRELATION PLOT
# ══════════════════════════════════════════════════════════════════════
# R plot: x = |cor(IV, NC)|, y = |cor(outcome, NC)|
# Note: R code labels x-axis "Outcome ~ NC Correlation" and y-axis
# "IV ~ NC Correlation", but plots cor_iv_nc on x and cor_outcome_nc on y.
# We follow the user's specification: x = |cor(outcome, NC)|,
# y = |cor(IV, NC)|.

print("\n── Generating Figure 1: Cor-cor plot ──")

abs_cor_iv = np.abs(cor_iv_nc)
abs_cor_outcome = np.abs(cor_outcome_nc)

fig, ax = plt.subplots(figsize=(9, 7))

ax.scatter(abs_cor_outcome, abs_cor_iv, alpha=0.7, s=40, color='steelblue',
           edgecolors='white', linewidth=0.5, zorder=3)

# Label each point with NC name
for j, nc_name in enumerate(nc_cols):
    ax.annotate(nc_name, (abs_cor_outcome[j], abs_cor_iv[j]),
                fontsize=5.5, alpha=0.75, rotation=20,
                xytext=(3, 1), textcoords='offset points')

# 45-degree reference line
max_val = max(abs_cor_outcome.max(), abs_cor_iv.max()) * 1.05
ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.4, linewidth=1,
        label='45-degree line', zorder=1)

ax.set_xlabel('|cor(Outcome, NC)|', fontsize=12)
ax.set_ylabel('|cor(IV, NC)|', fontsize=12)
ax.set_title('Cor-cor of Outcome (year 2000) and IV\n'
             'Residualized by controls (spec. 6) using linear regression. '
             'Correlation in absolute values.',
             fontsize=10)
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()

outpath = f"{OUTPUT_DIR}/Figure1_cor_cor_plot.png"
fig.savefig(outpath, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {outpath}")


print("\n" + "=" * 60)
print("03_figures.py — DONE")
print("=" * 60)
