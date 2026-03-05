"""
02_tables.py - Replicate Tables 1, A1, A2, A3
Main analysis: IV validation of school VAMs using lottery data.

Note: Public use data lacks demographics (gender, race, lunch status),
so we can only estimate Models 1 & 2 (no demographics). The paper's
Panel B (Models with demogs) cannot be replicated. Published results
will differ slightly due to scrambled IDs and removed demographics.
"""
import sys
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import bootstrap_2sls_fe, sd_of_school_effects, OUTPUT_DIR

# Load analysis data
df = pd.read_pickle(os.path.join(OUTPUT_DIR, 'analysis_data.pkl'))
print(f"Loaded analysis data: {len(df)} obs")

# Focus on on-margin sample for main analysis
margin = df[df['onmargin'] == 1].copy()
print(f"On-margin sample: {len(margin)}")

# Covariates for IV regression (controls for prior scores)
cov_cols = ['math_2002_imp', 'read_2002_imp',
            'math_2002_imp_sq', 'math_2002_imp_cub',
            'read_2002_imp_sq', 'read_2002_imp_cub',
            'math_2002_miss', 'read_2002_miss']

# ===================================================================
# TABLE A1: Descriptive Statistics
# ===================================================================
print("\n" + "=" * 70)
print("TABLE A1: DESCRIPTIVE STATISTICS")
print("=" * 70)

# Condition on testz2003 available and as_mod1mix_02_test available (like Stata code)
valid_mask = margin['testz2003'].notna() & margin['as_mod1mix_02_test'].notna()
non_lottery = df[(df['onmargin'] != 1) & df['testz2003'].notna() & df['as_mod1mix_02_test'].notna()].copy()
lottery_samp = margin[valid_mask].copy()

print(f"\nNon-lottery sample: {len(non_lottery)}")
print(f"Lottery sample: {len(lottery_samp)}")

# Published Table A1 values (columns 1 & 2 - means)
# Note: we can only compare math/reading scores (no demographics in public data)
print("\n{:<25} {:>12} {:>12} {:>15}".format("Variable", "Non-lottery", "Lottery", "Published Lot."))
print("-" * 70)

for var, pub_nonlot, pub_lot in [
    ('mathz2002', 0.074, -0.037),
    ('readz2002', 0.001, -0.070),
]:
    nl_mean = non_lottery[var].mean()
    l_mean = lottery_samp[var].mean()
    print(f"{var:<25} {nl_mean:>12.3f} {l_mean:>12.3f} {pub_lot:>15.3f}")

print(f"\n{'Sample Size':<25} {len(non_lottery):>12} {len(lottery_samp):>12}")
print(f"Published N:                    31,455       2,599")

# ===================================================================
# TABLE A3: Impact of Winning Lottery (ITT and IV)
# ===================================================================
print("\n" + "=" * 70)
print("TABLE A3: IMPACT OF WINNING THE LOTTERY")
print("=" * 70)

# Need on-margin sample with lottery_FE
t_a3 = margin[margin['lottery_FE'].notna()].copy()
print(f"Table A3 sample: {len(t_a3)}")

# ITT: xtreg outcome lottery covariates, fe vce(cluster lottery_FE)
print("\nPanel: ITT of Lottery on Enrollment")
print(f"{'Outcome':<20} {'Coef':>10} {'SE':>10} {'Published':>12}")
print("-" * 55)

for outcome, pub_coef, pub_se in [
    ('enrolled', 0.557, 0.036),
    ('enroll_home', -0.352, 0.040),
    ('attend_magnet', 0.198, 0.046),
]:
    sub = t_a3[['lottery', outcome, 'lottery_FE'] + cov_cols].dropna()
    if len(sub) < 20:
        print(f"{outcome:<20} insufficient data")
        continue

    # Demean by lottery_FE
    y = sub[outcome].astype(float)
    X_vars = ['lottery'] + cov_cols
    X = sub[X_vars].astype(float)

    for c in [outcome] + X_vars:
        gm = sub.groupby('lottery_FE')[c].transform('mean')
        sub[f'{c}_dm'] = sub[c].astype(float) - gm

    y_dm = sub[f'{outcome}_dm'].values
    X_dm = sub[[f'{c}_dm' for c in X_vars]].values

    model = sm.OLS(y_dm, X_dm).fit(cov_type='cluster',
                                     cov_kwds={'groups': sub['lottery_FE'].values})
    print(f"{outcome:<20} {model.params[0]:>10.3f} [{model.bse[0]:>7.3f}] {pub_coef:>8.3f} [{pub_se:>5.3f}]")

# IV: xtivreg2 outcome covariates (enrolled=lottery), fe cluster(lottery_FE)
print("\nPanel: IV Effect of Enrollment on School Characteristics and Achievement")
print(f"{'Outcome':<20} {'Coef':>10} {'SE':>10} {'Published':>12}")
print("-" * 55)

for outcome, pub_coef, pub_se in [
    ('pctmathz2002', 0.227, 0.058),
    ('pctreadz2002', -0.011, 0.011),
]:
    # Need to create pct variables first
    if outcome.startswith('pct'):
        base_var = outcome.replace('pct', '')
        t_a3[outcome] = t_a3.groupby('schl_d20')[base_var].transform('mean')

    sub = t_a3[[outcome, 'enrolled', 'lottery', 'lottery_FE'] + cov_cols].dropna()
    if len(sub) < 20:
        continue

    result = bootstrap_2sls_fe(
        y=sub[outcome], endog=sub['enrolled'],
        instrument=sub['lottery'], fe_groups=sub['lottery_FE'],
        covariates_df=sub[cov_cols],
        n_reps=100, cluster_var=sub['lottery_FE']
    )
    if result:
        print(f"{outcome:<20} {result['coef']:>10.3f} [{result['se']:>7.3f}] {pub_coef:>8.3f} [{pub_se:>5.3f}]")

# Achievement outcomes
for outcome, pub_coef, pub_se in [
    ('mathz2003', -0.037, 0.043),
    ('readz2003', -0.060, 0.047),
]:
    sub = t_a3[[outcome, 'enrolled', 'lottery', 'lottery_FE'] + cov_cols].dropna()
    if len(sub) < 20:
        continue

    result = bootstrap_2sls_fe(
        y=sub[outcome], endog=sub['enrolled'],
        instrument=sub['lottery'], fe_groups=sub['lottery_FE'],
        covariates_df=sub[cov_cols],
        n_reps=100, cluster_var=sub['lottery_FE']
    )
    if result:
        print(f"{outcome:<20} {result['coef']:>10.3f} [{result['se']:>7.3f}] {pub_coef:>8.3f} [{pub_se:>5.3f}]")


# ===================================================================
# TABLE 1: Main Results - Validating VAMs Using Lottery Data
# ===================================================================
print("\n" + "=" * 70)
print("TABLE 1: VALIDATING MODELS OF 'SCHOOL EFFECTS' USING LOTTERY DATA")
print("(Panel A only - Models 1 & 2, no demographics in public data)")
print("=" * 70)

# The paper uses Model 2 (gains) with AR estimator for the main results
# We replicate columns 4-6 (Model 2, no shrinkage, AR approach)
# Using average residual (ar) as the main approach
# Instrument: VAM of first choice school for winners, home school for losers

# Published Table 1 Panel A values (Model 2 = "prior scores (1 lag)")
published_table1 = {
    # (model, sample): (coef, se, pval_eq1, sd_effects)
    # Cols 1-3: Model 1 (no covariates, AR)
    (1, '02', 'ar'): (0.025, 0.077, 0.000, 0.431),
    (1, '2yr', 'ar'): (0.034, 0.070, 0.000, 0.417),
    (1, 'all', 'ar'): (0.014, 0.072, 0.000, 0.397),
    # Cols 4-6: Model 2 (prior scores, AR)
    (2, '02', 'ar'): (0.531, 0.208, 0.024, 0.110),
    (2, '2yr', 'ar'): (0.807, 0.236, 0.413, 0.096),
    (2, 'all', 'ar'): (0.966, 0.342, 0.920, 0.073),
}

# Table 1 uses the "hm" (home school) counterfactual by default
# VA = VA of assigned school (as)
# lott_VA = VA of home school if lottery loser, VA of first choice if lottery winner

print(f"\n{'Col':<5} {'Model':<8} {'Years':<12} {'Coef':>8} {'SE':>8} {'p(=1)':>8} {'SD':>8} | {'Pub Coef':>10} {'Pub SE':>10} {'Pub p(=1)':>10}")
print("-" * 110)

table1_results = {}
col_num = 0

for m in [1, 2]:
    for s in ['02', '2yr', 'all']:
        col_num += 1
        e = 'ar'  # average residual approach
        t = 'test'

        va_col = f'as_mod{m}{e}_{s}_{t}'
        hm_va_col = f'hm_mod{m}{e}_{s}_{t}'
        ch1_va_col = f'ch1_mod{m}{e}_{s}_{t}'

        # Create VA and instrument
        sub = margin.copy()
        sub['VA'] = sub[va_col]
        sub['lott_VA'] = np.where(sub['lottery'] == 0, sub[hm_va_col], sub[ch1_va_col])

        # Filter to valid observations
        keep_cols = ['testz2003', 'VA', 'lott_VA', 'lottery_FE'] + cov_cols
        sub = sub[keep_cols].dropna()

        if len(sub) < 50:
            print(f"{col_num:<5} M{m:<7} {s:<12} {'N/A':>8}")
            continue

        # SD of school effects
        sd_eff = sub['VA'].std()

        # Bootstrap 2SLS with FE
        result = bootstrap_2sls_fe(
            y=sub['testz2003'], endog=sub['VA'],
            instrument=sub['lott_VA'], fe_groups=sub['lottery_FE'],
            covariates_df=sub[cov_cols],
            n_reps=100, cluster_var=sub['lottery_FE']
        )

        if result:
            pub = published_table1.get((m, s, e), (None, None, None, None))
            table1_results[(m, s, e)] = result

            pub_str = f"{pub[0]:>10.3f} {pub[1]:>10.3f} {pub[2]:>10.3f}" if pub[0] else "N/A"
            print(f"{col_num:<5} M{m:<7} {s:<12} {result['coef']:>8.3f} [{result['se']:>5.3f}] {result['pval_eq1']:>8.3f} {sd_eff:>8.3f} | {pub_str}")

# Also try with mix (random effects) estimator for comparison
print("\n--- Same with Mixed Effects (mix) estimator ---")
col_num = 0
for m in [2]:
    for s in ['02', '2yr', 'all']:
        col_num += 1
        e = 'mix'
        t = 'test'

        va_col = f'as_mod{m}{e}_{s}_{t}'
        hm_va_col = f'hm_mod{m}{e}_{s}_{t}'
        ch1_va_col = f'ch1_mod{m}{e}_{s}_{t}'

        sub = margin.copy()
        sub['VA'] = sub[va_col]
        sub['lott_VA'] = np.where(sub['lottery'] == 0, sub[hm_va_col], sub[ch1_va_col])

        keep_cols = ['testz2003', 'VA', 'lott_VA', 'lottery_FE'] + cov_cols
        sub = sub[keep_cols].dropna()

        if len(sub) < 50:
            continue

        sd_eff = sub['VA'].std()

        result = bootstrap_2sls_fe(
            y=sub['testz2003'], endog=sub['VA'],
            instrument=sub['lott_VA'], fe_groups=sub['lottery_FE'],
            covariates_df=sub[cov_cols],
            n_reps=100, cluster_var=sub['lottery_FE']
        )

        if result:
            print(f"  M{m} {e} {s}: coef={result['coef']:.3f} [{result['se']:.3f}], p(=1)={result['pval_eq1']:.3f}, SD={sd_eff:.3f}, N={result['n']}")


# ===================================================================
# TABLE A2: Persistence of School Effects (2004 Outcomes)
# ===================================================================
print("\n" + "=" * 70)
print("TABLE A2: PERSISTENCE OF SCHOOL EFFECTS (2004 Outcomes)")
print("=" * 70)

# Published Table A2 uses Model 2 with demogs + prior scores, which we can't replicate
# We use Model 2 with prior scores only
# Outcome: testz2004, restrict to future_grd != 8

sub_a2 = margin[(margin['future_grd'] != 8)].copy()
print(f"Table A2 sample (excl grade 8): {len(sub_a2)}")

for m in [2]:
    for e in ['ar', 'mix']:
        for s in ['02', 'all']:
            va_col = f'as_mod{m}{e}_{s}_test'
            hm_va_col = f'hm_mod{m}{e}_{s}_test'
            ch1_va_col = f'ch1_mod{m}{e}_{s}_test'
            lead_va_col = f'aslead_mod{m}{e}_{s}_test'
            lead_hm_col = f'hmlead_mod{m}{e}_{s}_test'
            lead_ch1_col = f'ch1lead_mod{m}{e}_{s}_test'

            sub = sub_a2.copy()
            sub['leadVA'] = sub.get(lead_va_col, np.nan)
            sub['lott_leadVA'] = np.where(
                sub['lottery'] == 0,
                sub.get(lead_hm_col, np.nan),
                sub.get(lead_ch1_col, np.nan)
            )

            keep = ['testz2004', 'leadVA', 'lott_leadVA', 'lottery_FE'] + cov_cols
            sub = sub[[c for c in keep if c in sub.columns]].dropna()

            if len(sub) < 50:
                continue

            result = bootstrap_2sls_fe(
                y=sub['testz2004'], endog=sub['leadVA'],
                instrument=sub['lott_leadVA'], fe_groups=sub['lottery_FE'],
                covariates_df=sub[cov_cols],
                n_reps=100, cluster_var=sub['lottery_FE']
            )

            if result:
                print(f"  M{m} {e} {s}: coef={result['coef']:.3f} [{result['se']:.3f}], p(=1)={result['pval_eq1']:.3f}, N={result['n']}")


# ===================================================================
# SUMMARY COMPARISON
# ===================================================================
print("\n" + "=" * 70)
print("SUMMARY: REPLICATION vs PUBLISHED (Table 1 Panel A)")
print("=" * 70)

print(f"\n{'Spec':<20} {'Replicated':>12} {'Published':>12} {'Difference':>12}")
print("-" * 60)

for key in [(1, '02', 'ar'), (1, '2yr', 'ar'), (1, 'all', 'ar'),
            (2, '02', 'ar'), (2, '2yr', 'ar'), (2, 'all', 'ar')]:
    if key in table1_results:
        rep = table1_results[key]['coef']
        pub = published_table1[key][0]
        diff = rep - pub
        m, s, e = key
        label = f"M{m} {s}"
        print(f"{label:<20} {rep:>12.3f} {pub:>12.3f} {diff:>12.3f}")

print("\nNote: Differences expected due to scrambled school IDs and")
print("removal of demographic variables in the public-use dataset.")
print("The paper's author notes results will differ slightly.")

print("\n=== TABLES COMPLETE ===")
