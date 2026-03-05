"""
05_robustness.py - Robustness checks for Altonji, Kahn, Speer (2014).
"""
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '.')
from utils import (load_data, WEIGHT, first_stage, wcov,
                   build_design_matrix, run_wls_cluster)

print("=" * 70)
print("PHASE 4: ROBUSTNESS CHECKS")
print("=" * 70)

df = load_data()
w = df[WEIGHT].values
groups = df['clust_var'].values
y = df['lnearnings'].values

# Baseline results (Column 1: beta^m with grad controls)
X_base = build_design_matrix(df, betam_var='majorbeta_hgc_wtime', grad_controls=True)
res_base = run_wls_cluster(y, X_base, w, groups)
bm_var = 'majorbeta_hgc_wtime'
base_sigma93 = res_base.params[bm_var]
base_delta03 = res_base.params[f'{bm_var}_x_s03']
base_delta09 = res_base.params[f'{bm_var}_x_s09']

print(f"\nBaseline (Column 1):")
print(f"  sigma_93={base_sigma93:.4f}, delta_03={base_delta03:.4f}, delta_0911={base_delta09:.4f}")
print(f"  Published: sigma_93=0.136, delta_03=0.032, delta_0911=0.018")


def run_robustness(df_sub, label, betam='majorbeta_hgc_wtime', grad=True):
    """Run regression on a subsample and report key coefficients."""
    w_sub = df_sub[WEIGHT].values
    g_sub = df_sub['clust_var'].values
    y_sub = df_sub['lnearnings'].values

    try:
        X_sub = build_design_matrix(df_sub, betam_var=betam, grad_controls=grad)
        res = run_wls_cluster(y_sub, X_sub, w_sub, g_sub)
        s93 = res.params[betam]
        d03 = res.params[f'{betam}_x_s03']
        d09 = res.params[f'{betam}_x_s09']
        se93 = res.bse[betam]
        se03 = res.bse[f'{betam}_x_s03']
        se09 = res.bse[f'{betam}_x_s09']
        n = len(df_sub)
        print(f"\n  {label}:")
        print(f"    N={n:,d}, sigma_93={s93:.4f} ({se93:.4f}), "
              f"delta_03={d03:.4f} ({se03:.4f}), delta_0911={d09:.4f} ({se09:.4f})")
        print(f"    Diff from baseline: sigma_93={s93-base_sigma93:+.4f}, "
              f"delta_03={d03-base_delta03:+.4f}, delta_0911={d09-base_delta09:+.4f}")
        return {'sigma93': s93, 'delta03': d03, 'delta09': d09, 'n': n}
    except Exception as e:
        print(f"\n  {label}: FAILED - {e}")
        return None


# =====================================================================
# 1. Drop one survey at a time
# =====================================================================
print("\n" + "-" * 50)
print("1. Leave-One-Survey-Out")
print("-" * 50)
for sv in [1993, 2003, 2009]:
    df_sub = df[df['survey'] != sv].copy()
    run_robustness(df_sub, f"Drop survey {sv}")

# =====================================================================
# 2. Restrict to males only
# =====================================================================
print("\n" + "-" * 50)
print("2. Subgroup by Gender")
print("-" * 50)
run_robustness(df[df['male'] == 1].copy(), "Males only")
run_robustness(df[df['male'] == 0].copy(), "Females only")

# =====================================================================
# 3. Drop extreme majors (top/bottom 3 by beta^m)
# =====================================================================
print("\n" + "-" * 50)
print("3. Drop Outlier Majors")
print("-" * 50)
major_betas = df.groupby('major')['majorbeta_hgc_wtime'].first()
top3 = major_betas.nlargest(3).index.tolist()
bot3 = major_betas.nsmallest(3).index.tolist()
print(f"  Top 3 majors dropped: {top3}")
print(f"  Bottom 3 majors dropped: {bot3}")
df_no_extremes = df[~df['major'].isin(top3 + bot3)].copy()
run_robustness(df_no_extremes, "Drop top/bottom 3 majors")

# =====================================================================
# 4. Winsorize earnings at 1st and 99th percentiles
# =====================================================================
print("\n" + "-" * 50)
print("4. Winsorize Log Earnings (1st/99th percentile)")
print("-" * 50)
p1 = np.percentile(df['lnearnings'], 1)
p99 = np.percentile(df['lnearnings'], 99)
df_wins = df.copy()
df_wins['lnearnings'] = df_wins['lnearnings'].clip(p1, p99)
y_wins = df_wins['lnearnings'].values
X_wins = build_design_matrix(df_wins, betam_var=bm_var, grad_controls=True)
res_wins = run_wls_cluster(y_wins, X_wins, w, groups)
s93_w = res_wins.params[bm_var]
d03_w = res_wins.params[f'{bm_var}_x_s03']
d09_w = res_wins.params[f'{bm_var}_x_s09']
print(f"\n  Winsorized at [{p1:.3f}, {p99:.3f}]:")
print(f"    sigma_93={s93_w:.4f}, delta_03={d03_w:.4f}, delta_0911={d09_w:.4f}")
print(f"    Diff from baseline: sigma_93={s93_w-base_sigma93:+.4f}, "
      f"delta_03={d03_w-base_delta03:+.4f}, delta_0911={d09_w-base_delta09:+.4f}")

# =====================================================================
# 5. Alternative standard errors: cluster at major level only
# =====================================================================
print("\n" + "-" * 50)
print("5. Alternative Clustering: Major Level Only (vs Major x Year)")
print("-" * 50)
# Get major-level cluster variable
major_groups = df['major'].values
res_major_clust = run_wls_cluster(y, X_base, w, major_groups)
s93_mc = res_major_clust.params[bm_var]
se93_mc = res_major_clust.bse[bm_var]
se03_mc = res_major_clust.bse[f'{bm_var}_x_s03']
se09_mc = res_major_clust.bse[f'{bm_var}_x_s09']
print(f"  Major-level clustering (51 clusters):")
print(f"    sigma_93 SE: major_cluster={se93_mc:.4f} vs major_x_year={res_base.bse[bm_var]:.4f}")
print(f"    delta_03 SE: major_cluster={se03_mc:.4f} vs major_x_year={res_base.bse[f'{bm_var}_x_s03']:.4f}")
print(f"    delta_09 SE: major_cluster={se09_mc:.4f} vs major_x_year={res_base.bse[f'{bm_var}_x_s09']:.4f}")

# =====================================================================
# 6. No grad controls (already in Table 1 but include for completeness)
# =====================================================================
print("\n" + "-" * 50)
print("6. No Grad Controls (Column 2 of Table 1)")
print("-" * 50)
run_robustness(df, "No grad controls", betam='majorbeta_none', grad=False)

# =====================================================================
# 7. Restrict to younger workers (age 25-40 vs 40-55)
# =====================================================================
print("\n" + "-" * 50)
print("7. Age Subgroups")
print("-" * 50)
run_robustness(df[df['age'] <= 40].copy(), "Age 25-40 (younger)")
run_robustness(df[df['age'] > 40].copy(), "Age 41-55 (older)")

# =====================================================================
# 8. Placebo test: shuffle majors
# =====================================================================
print("\n" + "-" * 50)
print("8. Placebo Test: Shuffle Major Assignments")
print("-" * 50)
np.random.seed(42)
n_permutations = 100
placebo_results = []
# Get unique majors and their beta^m values
major_betas_df = df.groupby('major')[bm_var].first().reset_index()
for i in range(n_permutations):
    # Shuffle the mapping from major to beta^m
    shuffled_betas = major_betas_df[bm_var].values.copy()
    np.random.shuffle(shuffled_betas)
    major_map = dict(zip(major_betas_df['major'], shuffled_betas))

    # Create shuffled beta^m
    df_shuf = df.copy()
    df_shuf[bm_var] = df_shuf['major'].map(major_map)

    X_shuf = build_design_matrix(df_shuf, betam_var=bm_var, grad_controls=True)
    try:
        res_shuf = run_wls_cluster(y, X_shuf, w, groups)
        placebo_results.append({
            'd03': res_shuf.params[f'{bm_var}_x_s03'],
            'd09': res_shuf.params[f'{bm_var}_x_s09'],
        })
    except:
        pass

if placebo_results:
    d03_plac = [r['d03'] for r in placebo_results]
    d09_plac = [r['d09'] for r in placebo_results]
    p_d03 = np.mean(np.abs(d03_plac) >= np.abs(base_delta03))
    p_d09 = np.mean(np.abs(d09_plac) >= np.abs(base_delta09))
    print(f"  {len(placebo_results)} permutations completed")
    print(f"  Actual delta_03={base_delta03:.4f}, placebo mean={np.mean(d03_plac):.4f}, "
          f"SD={np.std(d03_plac):.4f}, p-value={p_d03:.4f}")
    print(f"  Actual delta_09={base_delta09:.4f}, placebo mean={np.mean(d09_plac):.4f}, "
          f"SD={np.std(d09_plac):.4f}, p-value={p_d09:.4f}")

# =====================================================================
# 9. Alternative functional form: levels instead of logs
# =====================================================================
print("\n" + "-" * 50)
print("9. Levels Instead of Log Earnings")
print("-" * 50)
earn = df['earnings_annual'].astype(float)
y_levels = earn.values
X_levels = build_design_matrix(df, betam_var=bm_var, grad_controls=True)
res_levels = run_wls_cluster(y_levels, X_levels, w, groups)
s93_lev = res_levels.params[bm_var]
d03_lev = res_levels.params[f'{bm_var}_x_s03']
d09_lev = res_levels.params[f'{bm_var}_x_s09']
mean_earn = np.average(earn, weights=w)
print(f"  Mean earnings: ${mean_earn:,.0f}")
print(f"  sigma_93={s93_lev:,.0f}, delta_03={d03_lev:,.0f}, delta_0911={d09_lev:,.0f}")
print(f"  As % of mean: sigma_93={s93_lev/mean_earn*100:.1f}%, "
      f"delta_03={d03_lev/mean_earn*100:.1f}%, delta_0911={d09_lev/mean_earn*100:.1f}%")

# =====================================================================
# 10. HC1 vs HC3 standard errors (without clustering)
# =====================================================================
print("\n" + "-" * 50)
print("10. HC1 vs HC3 Standard Errors (no clustering)")
print("-" * 50)
from statsmodels.regression.linear_model import WLS
mod = WLS(y, X_base, weights=w)
res_hc1 = mod.fit(cov_type='HC1')
res_hc3 = mod.fit(cov_type='HC3')
print(f"  {'Variable':30s}  {'HC1':>10s}  {'HC3':>10s}  {'Cluster':>10s}")
for key in [bm_var, f'{bm_var}_x_s03', f'{bm_var}_x_s09']:
    print(f"  {key:30s}  {res_hc1.bse[key]:10.4f}  {res_hc3.bse[key]:10.4f}  {res_base.bse[key]:10.4f}")

# =====================================================================
# Summary
# =====================================================================
print("\n" + "=" * 70)
print("ROBUSTNESS SUMMARY")
print("=" * 70)
print("""
Main findings:
1. Leave-one-survey-out: Tests sensitivity to each data source
2. Gender subgroups: Tests whether trends differ by gender
3. Drop extreme majors: Tests whether results driven by outlier majors
4. Winsorization: Tests sensitivity to extreme earnings values
5. Alternative clustering: Compares SE levels under different cluster definitions
6. No grad controls: Already in paper (Table 1, Col 2)
7. Age subgroups: Tests whether trends differ for younger vs older workers
8. Placebo test: Tests statistical significance via permutation
9. Levels vs logs: Tests functional form sensitivity
10. HC1 vs HC3: Compares heteroskedasticity-robust SE estimators
""")

print("=" * 70)
print("Robustness checks complete.")
print("=" * 70)
