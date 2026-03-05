"""
05_robustness.py — Robustness checks for Danieli et al. (2025).

Paper: "Negative Control Falsification Tests for IV Designs"
Replication package: openICPSR 238658-V1

Eight robustness checks across the ADH, Ashraf-Galor, and Nunn-Qian applications:
  1. Alternative control specifications (ADH)
  2. Random NC subset stability (ADH)
  3. Single NC category tests (ADH)
  4. Without weights (ADH)
  5. HC3 vs cluster-robust SE (ADH)
  6. Alternative sample restriction (Ashraf-Galor)
  7. Individual NC IV tests (Nunn-Qian)
  8. Symmetric vs asymmetric Bonferroni (ADH)
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from utils import (OUTPUT_DIR, load_adh, load_adh_preperiod,
                   load_ashraf_galor, load_nunn_qian,
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


print("=" * 70)
print("05_robustness.py — Robustness checks")
print("=" * 70)


# ======================================================================
# ADH DATA SETUP (same as 02_tables.py)
# ======================================================================

print("\n-- Loading and preparing ADH data --\n")

adh = load_adh()
adh_pre = load_adh_preperiod()

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
china_1990_cz = adh[adh['yr'] == 1990][['czone']].reset_index(drop=True)
china_1990['czone_tmp'] = china_1990_cz['czone'].values
china_1990 = china_1990.sort_values('czone_tmp').reset_index(drop=True)

adh_pre_1970 = adh_pre[adh_pre['yr'] == 1970].sort_values('czone').reset_index(drop=True)
adh_pre_1980 = adh_pre[adh_pre['yr'] == 1980].sort_values('czone').reset_index(drop=True)

if len(adh_pre_1970) == len(china_1990):
    china_1990['outcome1970'] = adh_pre_1970['d_sh_empl_mfg'].values
    china_1990['outcome1980'] = adh_pre_1980['d_sh_empl_mfg'].values

china_1990 = china_1990.drop(columns=['czone_tmp'])

# Define control sets
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

IV = china_1990['instrument2000'].values
weights = china_1990['timepwt48'].values
cluster = china_1990['statefip'].values
cntrls = china_1990[all_controls].values
NCs = china_1990[nc_cols].values

print(f"  china_1990: {china_1990.shape[0]} rows, {china_1990.shape[1]} columns")
print(f"  Number of negative controls: {NCs.shape[1]}")

# Store all results for summary table
results = []


# ======================================================================
# CHECK 1: Alternative control specifications (ADH)
# ======================================================================

print("\n" + "=" * 70)
print("Check 1: Alternative control specifications (ADH)")
print("=" * 70)

# Column 3: regions only
col_3_controls = ['l_shind_manuf_cbp_2000',
                  'reg_midatl_2000', 'reg_encen_2000', 'reg_wncen_2000',
                  'reg_satl_2000', 'reg_escen_2000', 'reg_wscen_2000',
                  'reg_mount_2000', 'reg_pacif_2000']

# Column 4: regions + demographics
col_4_controls = col_3_controls + ['l_sh_popedu_c_2000', 'l_sh_popfborn_2000',
                                    'l_sh_empl_f_2000']

# Column 6: all controls (same as all_controls)
col_6_controls = col_4_controls + ['l_sh_routine33_2000', 'l_task_outsource_2000']

specs = [
    ('Col 3 (regions)', col_3_controls),
    ('Col 4 (regions+demo)', col_4_controls),
    ('Col 6 (all)', col_6_controls),
]

for spec_name, ctrl_list in specs:
    ctrl_vals = china_1990[ctrl_list].values

    f_p, _ = nc_f_test(IV, NCs, ctrl_vals, weights)
    bonf_p, _ = nc_bonferroni(IV, NCs, ctrl_vals, weights, cluster)
    wald_p, _ = nc_wald_test(IV, NCs, ctrl_vals, weights, cluster)

    print(f"\n  {spec_name} ({len(ctrl_list)} controls):")
    print(f"    F-test p:      {fmt_p(f_p)}")
    print(f"    Bonferroni p:  {fmt_p(bonf_p)}")
    print(f"    Wald p:        {fmt_p(wald_p)}")

    results.append({
        'Check': '1. Alt controls',
        'Spec': spec_name,
        'F-test': fmt_p(f_p),
        'Bonferroni': fmt_p(bonf_p),
        'Wald': fmt_p(wald_p),
    })


# ======================================================================
# CHECK 2: Random NC subset stability (ADH)
# ======================================================================

print("\n" + "=" * 70)
print("Check 2: Random NC subset stability (ADH)")
print("=" * 70)

np.random.seed(42)
n_nc = NCs.shape[1]
n_draws = 5
subset_size = n_nc // 2

print(f"  Drawing {n_draws} random subsets of {subset_size} NCs (50% of {n_nc})")

for draw in range(n_draws):
    idx = np.random.choice(n_nc, size=subset_size, replace=False)
    idx.sort()
    NC_sub = NCs[:, idx]

    f_p, _ = nc_f_test(IV, NC_sub, cntrls, weights)
    bonf_p, _ = nc_bonferroni(IV, NC_sub, cntrls, weights, cluster)
    wald_p, _ = nc_wald_test(IV, NC_sub, cntrls, weights, cluster)

    print(f"\n  Draw {draw + 1}:")
    print(f"    F-test p:      {fmt_p(f_p)}")
    print(f"    Bonferroni p:  {fmt_p(bonf_p)}")
    print(f"    Wald p:        {fmt_p(wald_p)}")

    results.append({
        'Check': '2. Random subset',
        'Spec': f'Draw {draw + 1}',
        'F-test': fmt_p(f_p),
        'Bonferroni': fmt_p(bonf_p),
        'Wald': fmt_p(wald_p),
    })


# ======================================================================
# CHECK 3: Single NC category tests (ADH)
# ======================================================================

print("\n" + "=" * 70)
print("Check 3: Single NC category tests (ADH)")
print("=" * 70)

nc_categories = {
    'Mfg employment':     [c for c in nc_cols if c.startswith('l_sh_empl_mfg')],
    'Non-mfg employment': [c for c in nc_cols if c.startswith('l_sh_empl_nmfg')],
    'Unemployment':       [c for c in nc_cols if c.startswith('l_sh_unempl')],
    'NILF':               [c for c in nc_cols if c.startswith('l_sh_nilf')],
    'Transfers':          [c for c in nc_cols if c.startswith('l_trans')],
    'Wages':              [c for c in nc_cols if c.startswith('l_avg_lnwkwage') or
                           c == 'l_sh_ssadiswkrs'],
    'Household income':   [c for c in nc_cols if c.startswith('l_avg_hhinc')],
    'Lagged outcomes':    [c for c in nc_cols if c in ('outcome1970', 'outcome1980')],
}

for cat_name, cat_cols in nc_categories.items():
    if not cat_cols:
        print(f"\n  {cat_name}: no matching NCs found")
        results.append({
            'Check': '3. Single category',
            'Spec': cat_name,
            'F-test': 'N/A',
            'Bonferroni': 'N/A',
            'Wald': 'N/A',
        })
        continue

    NC_cat = china_1990[cat_cols].values
    bonf_p, _ = nc_bonferroni(IV, NC_cat, cntrls, weights, cluster)

    print(f"\n  {cat_name} ({len(cat_cols)} NCs: {cat_cols}):")
    print(f"    Bonferroni p:  {fmt_p(bonf_p)}")

    results.append({
        'Check': '3. Single category',
        'Spec': f'{cat_name} (n={len(cat_cols)})',
        'F-test': '--',
        'Bonferroni': fmt_p(bonf_p),
        'Wald': '--',
    })


# ======================================================================
# CHECK 4: Without weights (ADH)
# ======================================================================

print("\n" + "=" * 70)
print("Check 4: Without weights (ADH) -- unweighted OLS")
print("=" * 70)

f_p_uw, _ = nc_f_test(IV, NCs, cntrls, weights=None)
bonf_p_uw, _ = nc_bonferroni(IV, NCs, cntrls, weights=None, cluster=cluster)
wald_p_uw, _ = nc_wald_test(IV, NCs, cntrls, weights=None, cluster=cluster)

print(f"\n  Unweighted:")
print(f"    F-test p:      {fmt_p(f_p_uw)}")
print(f"    Bonferroni p:  {fmt_p(bonf_p_uw)}")
print(f"    Wald p:        {fmt_p(wald_p_uw)}")

# Baseline (weighted) for comparison
f_p_w, _ = nc_f_test(IV, NCs, cntrls, weights)
bonf_p_w, _ = nc_bonferroni(IV, NCs, cntrls, weights, cluster)
wald_p_w, _ = nc_wald_test(IV, NCs, cntrls, weights, cluster)

print(f"\n  Weighted (baseline):")
print(f"    F-test p:      {fmt_p(f_p_w)}")
print(f"    Bonferroni p:  {fmt_p(bonf_p_w)}")
print(f"    Wald p:        {fmt_p(wald_p_w)}")

results.append({
    'Check': '4. No weights',
    'Spec': 'Unweighted',
    'F-test': fmt_p(f_p_uw),
    'Bonferroni': fmt_p(bonf_p_uw),
    'Wald': fmt_p(wald_p_uw),
})
results.append({
    'Check': '4. No weights',
    'Spec': 'Weighted (baseline)',
    'F-test': fmt_p(f_p_w),
    'Bonferroni': fmt_p(bonf_p_w),
    'Wald': fmt_p(wald_p_w),
})


# ======================================================================
# CHECK 5: HC3 vs cluster-robust SE (ADH)
# ======================================================================

print("\n" + "=" * 70)
print("Check 5: HC3 vs cluster-robust SE (ADH)")
print("=" * 70)

# HC3: run Bonferroni without cluster (nc_bonferroni defaults to model pvalues,
# which uses non-robust SEs). We need to manually compute HC3 p-values.
n_obs = len(IV)
p_nc = NCs.shape[1]
hc3_pvals = []

for i in range(p_nc):
    nc_i = NCs[:, i]
    X = sm.add_constant(np.column_stack([IV.reshape(-1, 1), cntrls]))
    model = sm.WLS(nc_i, X, weights=weights).fit()
    model_hc3 = model.get_robustcov_results(cov_type='HC3')
    hc3_pvals.append(model_hc3.pvalues[1])

min_hc3 = min(hc3_pvals)
bonf_hc3 = min(min_hc3 * p_nc, 1.0)

# Cluster-robust (baseline)
bonf_cl, _ = nc_bonferroni(IV, NCs, cntrls, weights, cluster)

print(f"\n  Bonferroni with HC3 SEs:           {fmt_p(bonf_hc3)}")
print(f"  Bonferroni with cluster-robust SEs: {fmt_p(bonf_cl)}")

results.append({
    'Check': '5. HC3 vs cluster',
    'Spec': 'HC3',
    'F-test': '--',
    'Bonferroni': fmt_p(bonf_hc3),
    'Wald': '--',
})
results.append({
    'Check': '5. HC3 vs cluster',
    'Spec': 'Cluster-robust (baseline)',
    'F-test': '--',
    'Bonferroni': fmt_p(bonf_cl),
    'Wald': '--',
})


# ======================================================================
# CHECK 6: Alternative sample restriction (Ashraf-Galor)
# ======================================================================

print("\n" + "=" * 70)
print("Check 6: Alternative sample restriction (Ashraf-Galor)")
print("=" * 70)

ag = load_ashraf_galor()

ag_ctrl_cols = ['ln_yst', 'ln_arable', 'ln_abslat', 'ln_suitavg']
nc_vars_ag = ['mdist_addis', 'mdist_london', 'mdist_tokyo', 'mdist_mexico']

# Baseline: cleanpd1500 == 1
ag_clean = ag[ag['cleanpd1500'] == 1].copy().reset_index(drop=True)
ag_clean = ag_clean.dropna(subset=['ln_pd1500'] + ag_ctrl_cols).reset_index(drop=True)

# Filter to NCs available in clean sample
nc_vars_clean = [v for v in nc_vars_ag if v in ag_clean.columns and ag_clean[v].notna().all()]

print(f"\n  Baseline (cleanpd1500==1): {len(ag_clean)} countries")
print(f"  NC IVs available: {nc_vars_clean}")

if nc_vars_clean:
    Y_ag = ag_clean['ln_pd1500'].values
    ag_ctrls = ag_clean[ag_ctrl_cols].values
    NCs_ag = ag_clean[nc_vars_clean].values

    # Add quadratic terms for NCs (matching the paper's specification)
    NCs_ag_sq = np.column_stack([NCs_ag, NCs_ag ** 2])

    bonf_base, pvals_base = reduced_form_bonferroni(Y_ag, NCs_ag_sq, ag_ctrls)
    wald_base, _ = reduced_form_wald(Y_ag, NCs_ag_sq, ag_ctrls)

    print(f"    Bonferroni p (reduced form): {fmt_p(bonf_base)}")
    print(f"    Wald p (reduced form):       {fmt_p(wald_base)}")

    results.append({
        'Check': '6. AG sample',
        'Spec': f'Baseline (N={len(ag_clean)})',
        'F-test': '--',
        'Bonferroni': fmt_p(bonf_base),
        'Wald': fmt_p(wald_base),
    })

# Alternative: all countries with non-missing controls and NC IVs
ag_alt = ag.dropna(subset=['ln_pd1500'] + ag_ctrl_cols + nc_vars_ag).reset_index(drop=True)

print(f"\n  Alternative (all non-missing): {len(ag_alt)} countries")

if len(ag_alt) > 0:
    Y_ag_alt = ag_alt['ln_pd1500'].values
    ag_ctrls_alt = ag_alt[ag_ctrl_cols].values
    nc_vars_alt = [v for v in nc_vars_ag if v in ag_alt.columns]
    NCs_ag_alt = ag_alt[nc_vars_alt].values
    NCs_ag_alt_sq = np.column_stack([NCs_ag_alt, NCs_ag_alt ** 2])

    bonf_alt, pvals_alt = reduced_form_bonferroni(Y_ag_alt, NCs_ag_alt_sq, ag_ctrls_alt)
    wald_alt, _ = reduced_form_wald(Y_ag_alt, NCs_ag_alt_sq, ag_ctrls_alt)

    print(f"    Bonferroni p (reduced form): {fmt_p(bonf_alt)}")
    print(f"    Wald p (reduced form):       {fmt_p(wald_alt)}")

    results.append({
        'Check': '6. AG sample',
        'Spec': f'All non-missing (N={len(ag_alt)})',
        'F-test': '--',
        'Bonferroni': fmt_p(bonf_alt),
        'Wald': fmt_p(wald_alt),
    })


# ======================================================================
# CHECK 7: Individual NC IV tests (Nunn-Qian)
# ======================================================================

print("\n" + "=" * 70)
print("Check 7: Individual NC IV tests (Nunn-Qian)")
print("=" * 70)

nq = load_nunn_qian()

# Outcome (scaled by 1000 as in R code)
nq['intra_state_1000'] = nq['intra_state'] * 1000

# Identify NC IVs: lagged US crop production x fadum_avg
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

# Fixed effects: country + year x region
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

print(f"\n  Sample: {len(nq_sub)} obs")
print(f"  NCIVs: {len(nciv_transformed)}")

# Run individual reduced-form tests for each NCIV
print(f"\n  Individual NCIV reduced-form p-values:")
indiv_pvals = []
for i, nciv_name in enumerate(nciv_transformed):
    nc_single = NCs_nq[:, i:i+1]
    bonf_single, pvals_single = reduced_form_bonferroni(
        Y_nq, nc_single, cntrls_nq, cluster=cluster_nq)
    pval_i = pvals_single[0]
    indiv_pvals.append((nciv_name, pval_i))
    print(f"    {nciv_name}: p = {fmt_p(pval_i)}")

# Sort by p-value to identify which crops drive marginal rejection
indiv_pvals_sorted = sorted(indiv_pvals, key=lambda x: x[1])

print(f"\n  Ranked by p-value (most significant first):")
for rank, (name, pval) in enumerate(indiv_pvals_sorted, 1):
    bonf_adj = min(pval * len(nciv_transformed), 1.0)
    flag = " <-- drives rejection" if bonf_adj < 0.10 else ""
    print(f"    {rank}. {name}: raw p = {fmt_p(pval)}, "
          f"Bonferroni-adj = {fmt_p(bonf_adj)}{flag}")

# Joint test for reference
bonf_joint_nq, _ = reduced_form_bonferroni(Y_nq, NCs_nq, cntrls_nq, cluster=cluster_nq)
print(f"\n  Joint Bonferroni (all {len(nciv_transformed)} NCIVs): {fmt_p(bonf_joint_nq)}")

results.append({
    'Check': '7. NQ individual',
    'Spec': f'Joint ({len(nciv_transformed)} NCIVs)',
    'F-test': '--',
    'Bonferroni': fmt_p(bonf_joint_nq),
    'Wald': '--',
})
# Add most significant crop
if indiv_pvals_sorted:
    top_name, top_pval = indiv_pvals_sorted[0]
    results.append({
        'Check': '7. NQ individual',
        'Spec': f'Top: {top_name}',
        'F-test': '--',
        'Bonferroni': fmt_p(min(top_pval * len(nciv_transformed), 1.0)),
        'Wald': '--',
    })


# ======================================================================
# CHECK 8: Symmetric vs asymmetric Bonferroni (ADH)
# ======================================================================

print("\n" + "=" * 70)
print("Check 8: Symmetric (2-sided) vs asymmetric (1-sided) Bonferroni (ADH)")
print("=" * 70)

# 2-sided: standard Bonferroni (baseline, already computed)
bonf_2sided, pvals_2sided = nc_bonferroni(IV, NCs, cntrls, weights, cluster)

# 1-sided: compute manually using 1-sided p-values
n_obs = len(IV)
pvals_1sided_pos = []  # test for positive coefficient
pvals_1sided_neg = []  # test for negative coefficient

for i in range(NCs.shape[1]):
    nc_i = NCs[:, i]
    X = sm.add_constant(np.column_stack([IV.reshape(-1, 1), cntrls]))
    k = X.shape[1]

    model = sm.WLS(nc_i, X, weights=weights).fit()
    model_cl = model.get_robustcov_results(
        cov_type='cluster', groups=cluster, use_t=True)
    coef_z = model_cl.params[1]
    se_z = model_cl.bse[1]
    t_stat = coef_z / se_z
    df = n_obs - k

    # 1-sided: P(T > t) for positive direction
    p_pos = 1 - stats.t.cdf(t_stat, df=df)
    # 1-sided: P(T < t) for negative direction
    p_neg = stats.t.cdf(t_stat, df=df)

    pvals_1sided_pos.append(p_pos)
    pvals_1sided_neg.append(p_neg)

# Asymmetric Bonferroni: use minimum of the 1-sided p-values across NCs
# For each NC, take the smaller 1-sided p-value (i.e., testing the direction
# that gives the most evidence against null)
pvals_1sided_min = [min(p_pos, p_neg) for p_pos, p_neg in
                    zip(pvals_1sided_pos, pvals_1sided_neg)]
min_1sided = min(pvals_1sided_min)
bonf_1sided = min(min_1sided * NCs.shape[1], 1.0)

print(f"\n  2-sided Bonferroni p:  {fmt_p(bonf_2sided)}")
print(f"  1-sided Bonferroni p:  {fmt_p(bonf_1sided)}")

# Also show which NC drove each
idx_2sided = np.argmin(pvals_2sided)
idx_1sided = np.argmin(pvals_1sided_min)
print(f"\n  Driving NC (2-sided): {nc_cols[idx_2sided]} "
      f"(raw p = {fmt_p(pvals_2sided[idx_2sided])})")
print(f"  Driving NC (1-sided): {nc_cols[idx_1sided]} "
      f"(raw p = {fmt_p(pvals_1sided_min[idx_1sided])})")

results.append({
    'Check': '8. Bonf direction',
    'Spec': '2-sided (baseline)',
    'F-test': '--',
    'Bonferroni': fmt_p(bonf_2sided),
    'Wald': '--',
})
results.append({
    'Check': '8. Bonf direction',
    'Spec': '1-sided (asymmetric)',
    'F-test': '--',
    'Bonferroni': fmt_p(bonf_1sided),
    'Wald': '--',
})


# ======================================================================
# SUMMARY TABLE
# ======================================================================

print("\n\n" + "=" * 70)
print("ROBUSTNESS CHECKS -- SUMMARY TABLE")
print("=" * 70)

df_results = pd.DataFrame(results)
print()
print(df_results.to_string(index=False))

# Save to CSV
out_path = f"{OUTPUT_DIR}/robustness_summary.csv"
df_results.to_csv(out_path, index=False)
print(f"\n  Saved: {out_path}")

print("\n" + "=" * 70)
print("05_robustness.py -- DONE")
print("=" * 70)
