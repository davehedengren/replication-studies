"""
02_tables.py - Replicate Table 1 and Table 2.
Altonji, Kahn, Speer (2014) replication.
"""
import numpy as np
import sys
sys.path.insert(0, '.')
from utils import (load_data, WEIGHT, first_stage, wcov,
                   build_design_matrix, run_wls_cluster, print_comparison)

print("=" * 70)
print("PHASE 2: TABLE REPLICATION")
print("=" * 70)

# Load and prepare data
df = load_data()
w = df[WEIGHT].values
groups = df['clust_var'].values
y = df['lnearnings'].values

# =====================================================================
# First-stage: beta^m on A, R, M -> residuals v^m
# =====================================================================
resid_none, var_v_none, coefs_none, r2_none = first_stage(df, 'majorbeta_none')
resid_hgc, var_v_hgc, coefs_hgc, r2_hgc = first_stage(df, 'majorbeta_hgc_wtime')

df['all_T_none'] = resid_none
df['all_T_hgc'] = resid_hgc

# Covariances for variance decomposition
covAR = wcov(df['A'].values, df['R'].values, w)
covAM = wcov(df['A'].values, df['M'].values, w)
covRM = wcov(df['R'].values, df['M'].values, w)

print(f"\nFirst-stage R^2: none={r2_none:.4f} (pub=0.655), hgc={r2_hgc:.4f} (pub=0.685)")
print(f"Var(v^m): none={var_v_none:.4f} (pub=0.345), hgc={var_v_hgc:.4f} (pub=0.315)")
print(f"Covariances: AR={covAR:.4f} (0.565), AM={covAM:.4f} (-0.380), RM={covRM:.4f} (0.104)")

# =====================================================================
# TABLE 1: Log(Earnings) and Major Characteristics
# =====================================================================
print("\n" + "=" * 70)
print("TABLE 1: Log(Earnings) and Major Characteristics")
print("=" * 70)

# Published values from Table 1
# Format: (coef, SE)
published = {
    'col1': {  # beta^m, grad controls (_hgc_wtime)
        'beta_m': (0.136, 0.005),
        'beta_m_x_D03': (0.032, 0.008),
        'beta_m_x_D0911': (0.018, 0.006),
        'N': 897675, 'R2': 0.218,
    },
    'col2': {  # beta^m, no grad controls (_none)
        'beta_m': (0.139, 0.006),
        'beta_m_x_D03': (0.035, 0.008),
        'beta_m_x_D0911': (0.025, 0.006),
        'N': 897675, 'R2': 0.160,
    },
    'col3': {  # task decomposition, grad controls (_hgc_wtime)
        'A': (0.054, 0.007), 'A_x_D03': (0.021, 0.009), 'A_x_D0911': (-0.001, 0.007),
        'R': (0.068, 0.005), 'R_x_D03': (0.006, 0.008), 'R_x_D0911': (0.013, 0.006),
        'M': (-0.027, 0.006), 'M_x_D03': (-0.013, 0.009), 'M_x_D0911': (-0.013, 0.007),
        'vm': (0.131, 0.011), 'vm_x_D03': (0.030, 0.013), 'vm_x_D0911': (0.018, 0.011),
        'N': 897675, 'R2': 0.219,
    },
    'col4': {  # task decomposition, no grad controls (_none)
        'A': (0.060, 0.007), 'A_x_D03': (0.028, 0.010), 'A_x_D0911': (0.015, 0.007),
        'R': (0.062, 0.005), 'R_x_D03': (0.000, 0.008), 'R_x_D0911': (0.005, 0.006),
        'M': (-0.019, 0.005), 'M_x_D03': (-0.009, 0.009), 'M_x_D0911': (0.002, 0.006),
        'vm': (0.140, 0.009), 'vm_x_D03': (0.038, 0.010), 'vm_x_D0911': (0.031, 0.008),
        'N': 897675, 'R2': 0.160,
    },
}

results = {}

# --- Column 1: beta^m (hgc_wtime), grad controls ---
print("\n--- Column 1: beta^m (grad controls) ---")
X1 = build_design_matrix(df, betam_var='majorbeta_hgc_wtime', grad_controls=True)
res1 = run_wls_cluster(y, X1, w, groups)
results['col1'] = res1

bm_var = 'majorbeta_hgc_wtime'
print_comparison('beta^m (sigma_93)', res1.params[bm_var], 0.136,
                 res1.bse[bm_var], 0.005)
print_comparison('beta^m x D03', res1.params[f'{bm_var}_x_s03'], 0.032,
                 res1.bse[f'{bm_var}_x_s03'], 0.008)
print_comparison('beta^m x D0911', res1.params[f'{bm_var}_x_s09'], 0.018,
                 res1.bse[f'{bm_var}_x_s09'], 0.006)
print(f"  R-squared: {res1.rsquared:.4f}  (Published: 0.218)")
print(f"  N: {res1.nobs:.0f}")

# --- Column 2: beta^m (none), no grad controls ---
print("\n--- Column 2: beta^m (no grad controls) ---")
X2 = build_design_matrix(df, betam_var='majorbeta_none', grad_controls=False)
res2 = run_wls_cluster(y, X2, w, groups)
results['col2'] = res2

bm_var = 'majorbeta_none'
print_comparison('beta^m (sigma_93)', res2.params[bm_var], 0.139,
                 res2.bse[bm_var], 0.006)
print_comparison('beta^m x D03', res2.params[f'{bm_var}_x_s03'], 0.035,
                 res2.bse[f'{bm_var}_x_s03'], 0.008)
print_comparison('beta^m x D0911', res2.params[f'{bm_var}_x_s09'], 0.025,
                 res2.bse[f'{bm_var}_x_s09'], 0.006)
print(f"  R-squared: {res2.rsquared:.4f}  (Published: 0.160)")

# --- Column 3: task decomposition, grad controls ---
print("\n--- Column 3: A, R, M, v^m residuals (grad controls) ---")
X3 = build_design_matrix(df, grad_controls=True, use_tasks=True, resid_var='all_T_hgc')
res3 = run_wls_cluster(y, X3, w, groups)
results['col3'] = res3

for var, pub_key in [('A', 'A'), ('R', 'R'), ('M', 'M'), ('all_T_hgc', 'vm')]:
    pub = published['col3']
    print_comparison(f'{pub_key}', res3.params[var], pub[pub_key][0],
                     res3.bse[var], pub[pub_key][1])
    print_comparison(f'{pub_key} x D03', res3.params[f'{var}_x_s03'], pub[f'{pub_key}_x_D03'][0],
                     res3.bse[f'{var}_x_s03'], pub[f'{pub_key}_x_D03'][1])
    print_comparison(f'{pub_key} x D0911', res3.params[f'{var}_x_s09'], pub[f'{pub_key}_x_D0911'][0],
                     res3.bse[f'{var}_x_s09'], pub[f'{pub_key}_x_D0911'][1])
print(f"  R-squared: {res3.rsquared:.4f}  (Published: 0.219)")

# --- Column 4: task decomposition, no grad controls ---
print("\n--- Column 4: A, R, M, v^m residuals (no grad controls) ---")
X4 = build_design_matrix(df, grad_controls=False, use_tasks=True, resid_var='all_T_none')
res4 = run_wls_cluster(y, X4, w, groups)
results['col4'] = res4

for var, pub_key in [('A', 'A'), ('R', 'R'), ('M', 'M'), ('all_T_none', 'vm')]:
    pub = published['col4']
    print_comparison(f'{pub_key}', res4.params[var], pub[pub_key][0],
                     res4.bse[var], pub[pub_key][1])
    print_comparison(f'{pub_key} x D03', res4.params[f'{var}_x_s03'], pub[f'{pub_key}_x_D03'][0],
                     res4.bse[f'{var}_x_s03'], pub[f'{pub_key}_x_D03'][1])
    print_comparison(f'{pub_key} x D0911', res4.params[f'{var}_x_s09'], pub[f'{pub_key}_x_D0911'][0],
                     res4.bse[f'{var}_x_s09'], pub[f'{pub_key}_x_D0911'][1])
print(f"  R-squared: {res4.rsquared:.4f}  (Published: 0.160)")

# =====================================================================
# TABLE 2: Decomposing the Change in Variance in Returns to Majors
# =====================================================================
print("\n" + "=" * 70)
print("TABLE 2: Decomposing the Change in Variance in Returns to Majors")
print("=" * 70)


def variance_decomposition(res, resid_var_name, var_v, covAR, covAM, covRM, label):
    """Compute variance decomposition from regression coefficients."""
    b = res.params

    # Base year (1993) coefficients
    dA = b['A']
    dR = b['R']
    dM = b['M']
    dv = b[resid_var_name]

    # Changes for 2003
    DA03 = b[f'A_x_s03']
    DR03 = b[f'R_x_s03']
    DM03 = b[f'M_x_s03']
    Dv03 = b[f'{resid_var_name}_x_s03']

    # Changes for 2009-11
    DA09 = b[f'A_x_s09']
    DR09 = b[f'R_x_s09']
    DM09 = b[f'M_x_s09']
    Dv09 = b[f'{resid_var_name}_x_s09']

    def compute_var(da, dr, dm, dvv):
        """Compute Var(B^m_t) given task coefficients and v^m coefficient."""
        return (da**2 + dr**2 + dm**2
                + 2*da*dr*covAR + 2*da*dm*covAM + 2*dm*dr*covRM
                + dvv**2 * var_v)

    V93 = compute_var(dA, dR, dM, dv)
    V03_full = compute_var(dA + DA03, dR + DR03, dM + DM03, dv + Dv03)
    V09_full = compute_var(dA + DA09, dR + DR09, dM + DM09, dv + Dv09)

    pct_change_03 = (V03_full - V93) / V93
    pct_change_09 = (V09_full - V93) / V93

    # Partial contributions
    # Only A changes
    V03_onlyA = compute_var(dA + DA03, dR, dM, dv)
    V09_onlyA = compute_var(dA + DA09, dR, dM, dv)

    # Only R changes
    V03_onlyR = compute_var(dA, dR + DR03, dM, dv)
    V09_onlyR = compute_var(dA, dR + DR09, dM, dv)

    # Only M changes
    V03_onlyM = compute_var(dA, dR, dM + DM03, dv)
    V09_onlyM = compute_var(dA, dR, dM + DM09, dv)

    # All tasks change (but not v^m)
    V03_allT = compute_var(dA + DA03, dR + DR03, dM + DM03, dv)
    V09_allT = compute_var(dA + DA09, dR + DR09, dM + DM09, dv)

    # Only v^m changes
    V03_onlyV = compute_var(dA, dR, dM, dv + Dv03)
    V09_onlyV = compute_var(dA, dR, dM, dv + Dv09)

    def partial_pct(V_partial, V_base, total_change):
        if abs(total_change) < 1e-12:
            return 0.0
        return ((V_partial - V_base) / V_base) / total_change * 100

    # As fraction of total % change
    share_A_03 = partial_pct(V03_onlyA, V93, pct_change_03) if pct_change_03 != 0 else 0
    share_A_09 = partial_pct(V09_onlyA, V93, pct_change_09) if pct_change_09 != 0 else 0
    share_R_03 = partial_pct(V03_onlyR, V93, pct_change_03) if pct_change_03 != 0 else 0
    share_R_09 = partial_pct(V09_onlyR, V93, pct_change_09) if pct_change_09 != 0 else 0
    share_M_03 = partial_pct(V03_onlyM, V93, pct_change_03) if pct_change_03 != 0 else 0
    share_M_09 = partial_pct(V09_onlyM, V93, pct_change_09) if pct_change_09 != 0 else 0
    share_allT_03 = partial_pct(V03_allT, V93, pct_change_03) if pct_change_03 != 0 else 0
    share_allT_09 = partial_pct(V09_allT, V93, pct_change_09) if pct_change_09 != 0 else 0
    share_V_03 = partial_pct(V03_onlyV, V93, pct_change_03) if pct_change_03 != 0 else 0
    share_V_09 = partial_pct(V09_onlyV, V93, pct_change_09) if pct_change_09 != 0 else 0

    print(f"\n  --- {label} ---")
    print(f"  Var(beta^m_93): {V93:.6f}")
    print(f"  {'':30s}  {'1993-2003':>12s}  {'1993-2011':>12s}")
    print(f"  {'% Change':30s}  {pct_change_03*100:11.2f}%  {pct_change_09*100:11.2f}%")
    print(f"  {'Partial contribution (% of total change):':50s}")
    print(f"  {'  Abstract':30s}  {share_A_03:11.2f}%  {share_A_09:11.2f}%")
    print(f"  {'  Routine':30s}  {share_R_03:11.2f}%  {share_R_09:11.2f}%")
    print(f"  {'  Manual':30s}  {share_M_03:11.2f}%  {share_M_09:11.2f}%")
    print(f"  {'  Total (A+R+M+interactions)':30s}  {share_allT_03:11.2f}%  {share_allT_09:11.2f}%")
    print(f"  {'  Residual (v^m)':30s}  {share_V_03:11.2f}%  {share_V_09:11.2f}%")

    return {
        'V93': V93, 'pct_03': pct_change_03*100, 'pct_09': pct_change_09*100,
        'A_03': share_A_03, 'A_09': share_A_09,
        'R_03': share_R_03, 'R_09': share_R_09,
        'M_03': share_M_03, 'M_09': share_M_09,
        'total_03': share_allT_03, 'total_09': share_allT_09,
        'resid_03': share_V_03, 'resid_09': share_V_09,
    }


# Grad controls decomposition (Column 3)
decomp_grad = variance_decomposition(
    res3, 'all_T_hgc', var_v_hgc, covAR, covAM, covRM,
    "Grad Controls (Column 3)")

# No grad controls decomposition (Column 4)
decomp_nograd = variance_decomposition(
    res4, 'all_T_none', var_v_none, covAR, covAM, covRM,
    "No Grad Controls (Column 4)")

# Published Table 2 comparison
print("\n--- Comparison to Published Table 2 ---")
print(f"  {'':30s}  {'Grad Controls':>25s}  {'No Grad Controls':>25s}")
print(f"  {'':30s}  {'93-03':>12s} {'93-11':>12s}  {'93-03':>12s} {'93-11':>12s}")

pub_t2 = {
    'Var(beta_93)': (0.019, 0.019, 0.019, 0.019),
    '% Change': (54.89, 27.63, 60.38, 38.81),
    'Abstract': (46.05, -5.29, 56.78, 44.67),
    'Routine': (12.37, 52.00, 0.30, 11.95),
    'Manual': (11.69, 23.09, 6.04, -1.53),
    'Total': (73.36, 68.51, 64.82, 55.93),
    'Residual': (26.6, 31.5, 35.2, 44.1),
}

py_t2 = {
    'Var(beta_93)': (decomp_grad['V93'], decomp_grad['V93'],
                     decomp_nograd['V93'], decomp_nograd['V93']),
    '% Change': (decomp_grad['pct_03'], decomp_grad['pct_09'],
                 decomp_nograd['pct_03'], decomp_nograd['pct_09']),
    'Abstract': (decomp_grad['A_03'], decomp_grad['A_09'],
                 decomp_nograd['A_03'], decomp_nograd['A_09']),
    'Routine': (decomp_grad['R_03'], decomp_grad['R_09'],
                decomp_nograd['R_03'], decomp_nograd['R_09']),
    'Manual': (decomp_grad['M_03'], decomp_grad['M_09'],
               decomp_nograd['M_03'], decomp_nograd['M_09']),
    'Total': (decomp_grad['total_03'], decomp_grad['total_09'],
              decomp_nograd['total_03'], decomp_nograd['total_09']),
    'Residual': (decomp_grad['resid_03'], decomp_grad['resid_09'],
                 decomp_nograd['resid_03'], decomp_nograd['resid_09']),
}

for key in pub_t2:
    pub_vals = pub_t2[key]
    py_vals = py_t2[key]
    if key == 'Var(beta_93)':
        print(f"  {key:30s}  {py_vals[0]:12.4f} {'':12s}  {py_vals[2]:12.4f}")
        print(f"  {'  (Published)':30s}  {pub_vals[0]:12.4f} {'':12s}  {pub_vals[2]:12.4f}")
    else:
        fmt = '%' if key != 'Var(beta_93)' else ''
        print(f"  {key:30s}  {py_vals[0]:11.2f}{fmt} {py_vals[1]:11.2f}{fmt}  "
              f"{py_vals[2]:11.2f}{fmt} {py_vals[3]:11.2f}{fmt}")
        print(f"  {'  (Published)':30s}  {pub_vals[0]:11.2f}{fmt} {pub_vals[1]:11.2f}{fmt}  "
              f"{pub_vals[2]:11.2f}{fmt} {pub_vals[3]:11.2f}{fmt}")

print("\n" + "=" * 70)
print("Table replication complete.")
print("=" * 70)
