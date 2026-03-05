"""
02_tables.py — Reproduce main test results from Danieli et al. (2025).

Table 4: NC falsification test results for ADH and Deming
Table 5: NC falsification test results for Ashraf-Galor and Nunn-Qian
Table 2: Literature survey summary
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from utils import (OUTPUT_DIR, load_adh, load_adh_preperiod, load_deming,
                   load_ashraf_galor, load_nunn_qian, load_literature_survey,
                   nc_f_test, nc_bonferroni, nc_wald_test,
                   reduced_form_bonferroni, reduced_form_wald)

def fmt_p(p):
    """Format p-value for display."""
    if p < 0.001:
        return "<0.001"
    elif p < 0.01:
        return "<0.01"
    else:
        return f"{p:.3f}"

print("=" * 60)
print("02_tables.py — Reproducing tables")
print("=" * 60)


# ══════════════════════════════════════════════════════════════════════
# ADH APPLICATION (Table 4, Panel A)
# ══════════════════════════════════════════════════════════════════════

print("\n── ADH: NC Falsification Tests (Table 4, Panel A) ──\n")

adh = load_adh()
adh_pre = load_adh_preperiod()

# --- Build china_1990 matching R's ADH_Init.R ---

# Create year-2000 versions of controls and IV/exposure/outcome
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

# Remove columns matching R's logic: d_*, relchg*, lnchg*, plus identifiers
drop_cols = [c for c in china_1990.columns
             if c.startswith('d_') or c.startswith('relchg') or c.startswith('lnchg')]
drop_cols += ['l_tradeusch_pw', 'l_tradeotch_pw', 'czone', 'yr', 't2', 'city']
china_1990 = china_1990.drop(columns=[c for c in drop_cols if c in china_1990.columns])

# Add lagged outcomes from pre-period
china_1990_sorted = china_1990.sort_values('statefip').reset_index(drop=True)
adh_pre_1970 = adh_pre[adh_pre['yr'] == 1970].sort_values('czone').reset_index(drop=True)
adh_pre_1980 = adh_pre[adh_pre['yr'] == 1980].sort_values('czone').reset_index(drop=True)

# Sort by czone for matching (need to get czone from original data)
china_1990_cz = adh[adh['yr'] == 1990][['czone']].reset_index(drop=True)
china_1990['czone_tmp'] = china_1990_cz['czone'].values
china_1990 = china_1990.sort_values('czone_tmp').reset_index(drop=True)

if len(adh_pre_1970) == len(china_1990):
    china_1990['outcome1970'] = adh_pre_1970['d_sh_empl_mfg'].values
    china_1990['outcome1980'] = adh_pre_1980['d_sh_empl_mfg'].values

china_1990 = china_1990.drop(columns=['czone_tmp'])

print(f"  china_1990: {china_1990.shape[0]} rows, {china_1990.shape[1]} columns")

# --- Define controls (col_6 minus t2 = all_controls, 14 vars) ---
all_controls = ['l_shind_manuf_cbp_2000',
                'reg_midatl_2000', 'reg_encen_2000', 'reg_wncen_2000',
                'reg_satl_2000', 'reg_escen_2000', 'reg_wscen_2000',
                'reg_mount_2000', 'reg_pacif_2000',
                'l_sh_popedu_c_2000', 'l_sh_popfborn_2000', 'l_sh_empl_f_2000',
                'l_sh_routine33_2000', 'l_task_outsource_2000']

# Old (1990-period) versions of the same controls
all_controls_old = ['l_shind_manuf_cbp',
                    'reg_midatl', 'reg_encen', 'reg_wncen',
                    'reg_satl', 'reg_escen', 'reg_wscen',
                    'reg_mount', 'reg_pacif',
                    'l_sh_popedu_c', 'l_sh_popfborn', 'l_sh_empl_f',
                    'l_sh_routine33', 'l_task_outsource']

# --- Variables to remove (matching R's ADH_Run.R line 25) ---
variables_to_remove = ['timepwt48', 'instrument2000', 'outcome2000', 'statefip',
                       'instrument1990', 'exposure1990', 'exposure2000']

# --- NC selection: everything except removed vars + controls + old controls ---
exclude_set = set(variables_to_remove + all_controls + all_controls_old)
nc_cols = [c for c in china_1990.columns if c not in exclude_set]

# Remove any columns with NaN
nc_cols = [c for c in nc_cols if china_1990[c].notna().all()]

IV = china_1990['instrument2000'].values
weights = china_1990['timepwt48'].values
cluster = china_1990['statefip'].values
cntrls = china_1990[all_controls].values
NCs = china_1990[nc_cols].values

print(f"  Number of negative controls: {NCs.shape[1]}")
print(f"  (R code uses ~52 NCs)")

# --- F-test (Z ~ NC + controls) ---
f_pval, f_stat = nc_f_test(IV, NCs, cntrls, weights)
print(f"\n  Multiple NCO results:")
print(f"  F-test p-value: {fmt_p(f_pval)}")

# --- Bonferroni (NC_i ~ Z + controls, correct direction) ---
bonf_pval, bonf_pvals = nc_bonferroni(IV, NCs, cntrls, weights, cluster)
print(f"  Bonferroni p-value: {fmt_p(bonf_pval)}")

# --- Wald test (Z ~ NC + controls, cluster-robust) ---
wald_pval, wald_stat = nc_wald_test(IV, NCs, cntrls, weights, cluster)
print(f"  Wald test p-value: {fmt_p(wald_pval)}")

# --- Single NCO (lagged outcome 1970) ---
if 'outcome1970' in china_1990.columns:
    nc_1970 = china_1990[['outcome1970']].values
    bonf_lo, pvals_lo = nc_bonferroni(IV, nc_1970, cntrls, weights, cluster)
    print(f"\n  Single NCO (outcome 1970):")
    print(f"  Bonferroni p-value: {fmt_p(pvals_lo[0])}")
    print(f"  (R code reports ≈ 0.403)")

# --- ADH Table 3 replication (2SLS, pre-period) ---
print("\n  ADH pre-period 2SLS replication:")
pre_1970 = adh_pre[adh_pre['yr'] == 1970].copy()
if 'd_tradeusch_pw_future' in pre_1970.columns and 'd_tradeotch_pw_lag_future' in pre_1970.columns:
    y = pre_1970['d_sh_empl_mfg'].values
    endog = pre_1970['d_tradeusch_pw_future'].values
    iv = pre_1970['d_tradeotch_pw_lag_future'].values
    w = pre_1970['timepwt48'].values

    X_fs = sm.add_constant(iv)
    fs = sm.WLS(endog, X_fs, weights=w).fit()
    endog_hat = fs.fittedvalues

    X_ss = sm.add_constant(endog_hat)
    ss = sm.WLS(y, X_ss, weights=w).fit()
    ss_cl = ss.get_robustcov_results(cov_type='cluster',
                                      groups=pre_1970['statefip'].values)
    coef_2sls = ss_cl.params[1]
    se_2sls = ss_cl.bse[1]
    t_stat = coef_2sls / se_2sls
    p_2sls = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(y) - 2))
    print(f"  2SLS coef: {coef_2sls:.4f}, SE: {se_2sls:.4f}, p-value: {fmt_p(p_2sls)}")
    print(f"  (R code reports p ≈ 0.403)")


# ══════════════════════════════════════════════════════════════════════
# DEMING APPLICATION (Table 4, Panel B)
# ══════════════════════════════════════════════════════════════════════

print("\n\n── Deming: NC Falsification Tests (Table 4, Panel B) ──\n")

dem = load_deming()

# Construct VA variables (matching Deming_init.R)
model_str = "mod2"
prev_yr = "mix_all"

ch1_va_col = f'ch1_{model_str}{prev_yr}_test'
hm_va_col = f'hm_{model_str}{prev_yr}_test'

if ch1_va_col in dem.columns and hm_va_col in dem.columns:
    # lott_VA = home school VA when lottery==0, choice school VA when lottery==1
    dem['lott_VA'] = np.where(dem['lottery'] == 1, dem[ch1_va_col], dem[hm_va_col])
    dem['new_lott_VA'] = np.where(dem['lottery'] == 1, dem[ch1_va_col], 0)

# Drop missing lottery_FE
dem = dem[dem['lottery_FE'].notna()].copy()

# Control variables
ctrl_vars = ['math_2002_imp', 'read_2002_imp',
             'math_2002_imp_sq', 'read_2002_imp_sq',
             'math_2002_imp_cub', 'read_2002_imp_cub',
             'math_2002_miss', 'read_2002_miss']

# Create squared and cubic terms if not present
for base in ['math_2002_imp', 'read_2002_imp']:
    sq = f'{base}_sq'
    cub = f'{base}_cub'
    if sq not in dem.columns and base in dem.columns:
        dem[sq] = dem[base] ** 2
    if cub not in dem.columns and base in dem.columns:
        dem[cub] = dem[base] ** 3

# Select relevant columns and complete cases
outcome_col = 'testz2003'
iv_cols = ['lottery', 'lott_VA']
va_cols = [c for c in dem.columns if model_str in c and 'test' in c]

# NC candidates: prior test scores and other pre-treatment variables
nc_exclude = set([outcome_col, 'lottery_FE'] + iv_cols + ctrl_vars + va_cols +
                 ['new_lott_VA'])

# Keep columns that are potential NCs
all_needed = [outcome_col] + iv_cols + ctrl_vars + ['lottery_FE']
available = [c for c in all_needed if c in dem.columns]
dem_sub = dem[dem.columns.intersection(available + [c for c in dem.columns
             if c not in nc_exclude])].copy()

# Complete cases on key variables
key_vars = [c for c in available if c in dem_sub.columns]
dem_sub = dem_sub.dropna(subset=key_vars).reset_index(drop=True)

# Exclude lottery group 14
dem_sub = dem_sub[dem_sub['lottery_FE'] != 14].reset_index(drop=True)

# Lottery FE dummies
fe_dummies = pd.get_dummies(dem_sub['lottery_FE'], prefix='fe', drop_first=True)

# NC columns
nc_dem_cols = [c for c in dem_sub.columns if c not in nc_exclude and
               c != 'lottery_FE' and dem_sub[c].notna().all() and
               dem_sub[c].dtype in ['float64', 'float32', 'int64', 'int32']]

# Remove enrolled if present
nc_dem_cols = [c for c in nc_dem_cols if c != 'enrolled']

if nc_dem_cols and 'lottery' in dem_sub.columns:
    cntrls_dem = np.column_stack([dem_sub[ctrl_vars].values, fe_dummies.values])
    NCs_dem = dem_sub[nc_dem_cols].values

    print(f"  Sample: {len(dem_sub)} students")
    print(f"  Number of NCs: {NCs_dem.shape[1]}")
    print(f"  Lottery FE groups: {dem_sub['lottery_FE'].nunique()}")

    for iv_name in ['lottery', 'lott_VA']:
        if iv_name not in dem_sub.columns:
            continue
        IV_dem = dem_sub[iv_name].values

        f_p, f_s = nc_f_test(IV_dem, NCs_dem, cntrls_dem)
        print(f"\n  IV = {iv_name}:")
        print(f"    F-test p-value: {fmt_p(f_p)}")

        # Bonferroni: NC_i ~ Z + controls (same direction as ADH)
        bonf_p_dem, _ = nc_bonferroni(IV_dem, NCs_dem, cntrls_dem)
        print(f"    Bonferroni p-value: {fmt_p(bonf_p_dem)}")

    # Single NCO: testz2002 ~ lott_VA + controls
    if 'testz2002' in dem_sub.columns and 'lott_VA' in dem_sub.columns:
        y_nco = dem_sub['testz2002'].values
        X_nco = sm.add_constant(np.column_stack([dem_sub['lott_VA'].values, cntrls_dem]))
        m_nco = sm.OLS(y_nco, X_nco).fit()
        print(f"\n  Single NCO (testz2002 ~ lott_VA): p = {fmt_p(m_nco.pvalues[1])}")
else:
    print("  Deming data: insufficient columns for NC analysis")


# ══════════════════════════════════════════════════════════════════════
# ASHRAF-GALOR APPLICATION (Table 5, Panel A)
# ══════════════════════════════════════════════════════════════════════

print("\n\n── Ashraf-Galor: NC Falsification Tests (Table 5, Panel A) ──\n")

ag = load_ashraf_galor()

# Clean sample
ag_clean = ag[ag['cleanpd1500'] == 1].copy().reset_index(drop=True)

# Controls — complete cases on these (NOT on mdist_hgdp which is mostly NA)
ag_ctrl_cols = ['ln_yst', 'ln_arable', 'ln_abslat', 'ln_suitavg']
ag_clean = ag_clean.dropna(subset=['ln_pd1500'] + ag_ctrl_cols).reset_index(drop=True)
print(f"  Clean sample: {len(ag_clean)} countries")

Y_ag = ag_clean['ln_pd1500'].values
ag_controls = ag_clean[ag_ctrl_cols].values

# NC IVs (fake migration distances)
nc_vars = ['mdist_addis', 'mdist_london', 'mdist_tokyo', 'mdist_mexico']
nc_vars = [v for v in nc_vars if v in ag_clean.columns and ag_clean[v].notna().all()]
print(f"  NC IVs: {nc_vars}")

# IV for adjustment
IV_ag = ag_clean['mdist_hgdp'].values
IV_ag_sq = ag_clean.get('mdist_hgdp_sqr', ag_clean['mdist_hgdp'] ** 2 if 'mdist_hgdp' in ag_clean else None)
has_iv = ag_clean['mdist_hgdp'].notna()
n_with_iv = has_iv.sum()
print(f"  Countries with mdist_hgdp: {n_with_iv}")

# --- Without IV adjustment: outcome ~ NC + NC^2 + controls ---
print("\n  Without IV adjustment (reduced form):")
for nc_name in nc_vars:
    nc_i = ag_clean[nc_name].values
    nc_sq = nc_i ** 2
    X = sm.add_constant(np.column_stack([nc_i, nc_sq, ag_controls]))
    model = sm.OLS(Y_ag, X).fit(cov_type='HC3')
    p_lin = model.pvalues[1]
    p_sq = model.pvalues[2]
    print(f"    {nc_name}: linear p={fmt_p(p_lin)}, squared p={fmt_p(p_sq)}")

# --- With IV adjustment: outcome ~ NC + NC^2 + IV + IV^2 + controls ---
print(f"\n  With IV adjustment (controlling for mdist_hgdp, N={n_with_iv}):")
ag_iv = ag_clean[has_iv].reset_index(drop=True)
Y_ag_iv = ag_iv['ln_pd1500'].values
ag_ctrl_iv = ag_iv[ag_ctrl_cols].values
IV_vals = ag_iv['mdist_hgdp'].values
IV_sq_vals = IV_vals ** 2

for nc_name in nc_vars:
    if nc_name in ag_iv.columns and ag_iv[nc_name].notna().all():
        nc_i = ag_iv[nc_name].values
        nc_sq = nc_i ** 2
        X = sm.add_constant(np.column_stack([nc_i, nc_sq, IV_vals, IV_sq_vals, ag_ctrl_iv]))
        model = sm.OLS(Y_ag_iv, X).fit(cov_type='HC3')
        p_lin = model.pvalues[1]
        p_sq = model.pvalues[2]
        print(f"    {nc_name}: linear p={fmt_p(p_lin)}, squared p={fmt_p(p_sq)}")

# --- Joint Wald test (London + Tokyo + Mexico, IV-adjusted) ---
print(f"\n  Joint Wald test (London, Tokyo, Mexico) with IV adj (N={n_with_iv}):")
joint_ncs = ['mdist_london', 'mdist_tokyo', 'mdist_mexico']
joint_ncs = [v for v in joint_ncs if v in ag_iv.columns]
if joint_ncs:
    nc_cols_joint = []
    for nc_name in joint_ncs:
        nc_cols_joint.extend([ag_iv[nc_name].values, ag_iv[nc_name].values ** 2])
    NC_joint = np.column_stack(nc_cols_joint)
    iv_block = np.column_stack([IV_vals, IV_sq_vals])
    X_joint = sm.add_constant(np.column_stack([NC_joint, iv_block, ag_ctrl_iv]))
    model_joint = sm.OLS(Y_ag_iv, X_joint).fit(cov_type='HC3')

    # Wald on NC coefficients (indices 1 through 2*len(joint_ncs))
    n_nc_coefs = 2 * len(joint_ncs)
    nc_idx = list(range(1, n_nc_coefs + 1))
    beta_nc = model_joint.params[nc_idx]
    cov = model_joint.cov_params()
    if hasattr(cov, 'iloc'):
        vcov_nc = cov.iloc[nc_idx, nc_idx].values
    else:
        vcov_nc = cov[np.ix_(nc_idx, nc_idx)]
    wald_stat = beta_nc @ np.linalg.pinv(vcov_nc) @ beta_nc
    wald_p = 1 - stats.chi2.cdf(wald_stat, n_nc_coefs)

    # Bonferroni on individual coefficients
    indiv_pvals = [model_joint.pvalues[i] for i in nc_idx]
    bonf_joint = min(min(indiv_pvals) * n_nc_coefs, 1.0)

    print(f"    Wald p-value: {fmt_p(wald_p)}")
    print(f"    Bonferroni p-value: {fmt_p(bonf_joint)}")


# ══════════════════════════════════════════════════════════════════════
# NUNN-QIAN APPLICATION (Table 5, Panel B)
# ══════════════════════════════════════════════════════════════════════

print("\n\n── Nunn-Qian: NC Falsification Tests (Table 5, Panel B) ──\n")

nq = load_nunn_qian()

# Outcome (scaled by 1000 as in R code)
nq['intra_state_1000'] = nq['intra_state'] * 1000

# Identify NC IVs: lagged US crop production × fadum_avg
nciv_crops = ['Oranges', 'Grapes', 'Lettuce', 'Cotton_lint', 'Onions_dry',
              'Grapefruit', 'Cabbages', 'Watermelons', 'Carrots_turnips',
              'Peaches_nectarines']
nciv_cols = [f'l_USprod_{crop}' for crop in nciv_crops]
nciv_cols = [c for c in nciv_cols if c in nq.columns]

# Transform NCIVs: divide by 1000 and multiply by fadum_avg
for c in nciv_cols:
    nq[f'{c}_t'] = (nq[c] / 1000) * nq['fadum_avg']

nciv_transformed = [f'{c}_t' for c in nciv_cols]

# Controls
us_ctrl = [c for c in ['oil_fadum_avg', 'US_income_fadum_avg', 'US_democ_pres_fadum_avg']
           if c in nq.columns]
weather_ctrl = [c for c in nq.columns if c.startswith('all_Precip_') or c.startswith('all_Temp_')]
gdp_ctrl = [c for c in nq.columns if c.startswith('gdp_')]
usmil_ctrl = [c for c in nq.columns if c.startswith('usmil_')]
usec_ctrl = [c for c in nq.columns if c.startswith('usec_')]
cereal_ctrl = [c for c in nq.columns if c.startswith('rcereal_') or c.startswith('rimport_')]
baseline_controls = us_ctrl + weather_ctrl + gdp_ctrl + usmil_ctrl + usec_ctrl + cereal_ctrl

# Fixed effects: country + year×region
nq['year_f'] = nq['year'].astype(str)
if 'wb_region' in nq.columns:
    nq['year_region'] = nq['year_f'] + '_' + nq['wb_region'].astype(str)
elif 'region' in nq.columns:
    nq['year_region'] = nq['year_f'] + '_' + nq['region'].astype(str)

# Build control matrix with FE dummies
all_ctrl_vars = baseline_controls.copy()
needed_cols = ['intra_state_1000', 'instrument'] + nciv_transformed + all_ctrl_vars
if 'risocode' in nq.columns:
    needed_cols.append('risocode')
if 'year_region' in nq.columns:
    needed_cols.append('year_region')

# Complete cases
nq_sub = nq.dropna(subset=[c for c in needed_cols if c in nq.columns]).reset_index(drop=True)

# FE dummies
fe_parts = []
if 'risocode' in nq_sub.columns:
    country_fe = pd.get_dummies(nq_sub['risocode'], prefix='cfe', drop_first=True)
    fe_parts.append(country_fe)
if 'year_region' in nq_sub.columns:
    yr_fe = pd.get_dummies(nq_sub['year_region'], prefix='yrfe', drop_first=True)
    fe_parts.append(yr_fe)

available_ctrls = [c for c in all_ctrl_vars if c in nq_sub.columns and nq_sub[c].notna().all()]
ctrl_vals = [nq_sub[available_ctrls].values] + [fe.values for fe in fe_parts]
cntrls_nq = np.column_stack(ctrl_vals) if ctrl_vals else None

Y_nq = nq_sub['intra_state_1000'].values
NCs_nq = nq_sub[nciv_transformed].values
cluster_nq = nq_sub['risocode'].values if 'risocode' in nq_sub.columns else None

print(f"  Sample: {len(nq_sub)} obs")
print(f"  NCIVs: {len(nciv_transformed)}")
print(f"  Controls + FE: {cntrls_nq.shape[1] if cntrls_nq is not None else 0}")

# --- Without IV adjustment ---
print("\n  Without IV adjustment:")

# F-test: outcome ~ NCIVs + controls
f_p_nq, _ = nc_f_test(Y_nq, NCs_nq, cntrls_nq)
print(f"    F-test p-value: {fmt_p(f_p_nq)}")

# Bonferroni: outcome ~ NCIV_i + controls (reduced form)
bonf_p_nq, bonf_pvals_nq = reduced_form_bonferroni(Y_nq, NCs_nq, cntrls_nq,
                                                      cluster=cluster_nq)
print(f"    Bonferroni p-value: {fmt_p(bonf_p_nq)}")

# Wald: joint test on all NCIVs
wald_p_nq, _ = reduced_form_wald(Y_nq, NCs_nq, cntrls_nq, cluster=cluster_nq)
print(f"    Wald p-value: {fmt_p(wald_p_nq)}")

# --- With IV adjustment (add instrument to controls) ---
print("\n  With IV adjustment:")
iv_nq = nq_sub['instrument'].values.reshape(-1, 1)
cntrls_nq_adj = np.column_stack([iv_nq, cntrls_nq]) if cntrls_nq is not None else iv_nq

f_p_nq2, _ = nc_f_test(Y_nq, NCs_nq, cntrls_nq_adj)
print(f"    F-test p-value: {fmt_p(f_p_nq2)}")

bonf_p_nq2, _ = reduced_form_bonferroni(Y_nq, NCs_nq, cntrls_nq_adj, cluster=cluster_nq)
print(f"    Bonferroni p-value: {fmt_p(bonf_p_nq2)}")

wald_p_nq2, _ = reduced_form_wald(Y_nq, NCs_nq, cntrls_nq_adj, cluster=cluster_nq)
print(f"    Wald p-value: {fmt_p(wald_p_nq2)}")

# Individual NCIV p-values (highlight Grapes)
print("\n  Individual NCIV p-values (without IV adj):")
for i, nc_name in enumerate(nciv_transformed):
    print(f"    {nc_name}: p = {fmt_p(bonf_pvals_nq[i])}")


# ══════════════════════════════════════════════════════════════════════
# LITERATURE SURVEY (Table 2)
# ══════════════════════════════════════════════════════════════════════

print("\n\n── Literature Survey (Table 2) ──\n")

lit = load_literature_survey()
lit.columns = lit.columns.str.strip()

if 'is_falsification' in lit.columns:
    frac_fals = lit['is_falsification'].mean()
    print(f"  Papers using any falsification test: {frac_fals:.0%} "
          f"({lit['is_falsification'].sum()}/{len(lit)})")
    print(f"  (Paper reports: 51%)")

if 'journal' in lit.columns:
    print(f"\n  By journal:")
    for journal in sorted(lit['journal'].unique()):
        sub = lit[lit['journal'] == journal]
        if 'is_falsification' in sub.columns:
            frac = sub['is_falsification'].mean()
            print(f"    {journal}: {frac:.0%} ({sub['is_falsification'].sum()}/{len(sub)})")

# Types of falsification tests
for pattern, label in [('nco', 'NCO'), ('nci', 'NCI'), ('nc_', 'NC'),
                        ('placebo', 'Placebo')]:
    cols = [c for c in lit.columns if pattern in c.lower()]
    if cols:
        print(f"\n  {label} columns: {cols}")
        for c in cols:
            if lit[c].dtype in ['float64', 'int64']:
                print(f"    {c}: {lit[c].sum()}/{len(lit)} ({lit[c].mean():.0%})")


print("\n" + "=" * 60)
print("02_tables.py — DONE")
print("=" * 60)
