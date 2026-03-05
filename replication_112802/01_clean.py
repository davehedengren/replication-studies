"""
01_clean.py - Load data and verify against published summary statistics.
Altonji, Kahn, Speer (2014) replication.
"""
import numpy as np
import sys
sys.path.insert(0, '.')
from utils import load_data, WEIGHT, first_stage, wcov

print("=" * 70)
print("PHASE 1: DATA LOADING AND VERIFICATION")
print("=" * 70)

df = load_data()
w = df[WEIGHT].values

# Basic verification
print(f"\nSample size: {len(df):,d}  (Published: 897,675)")
print(f"Weight sum: {w.sum():,.2f}  (Published: 2,481,334.77)")

# Weighted means of key variables
print("\n--- Weighted Summary Statistics ---")
for var, pub_mean, pub_sd in [
    ('majorbeta_none', 0.0, 1.0),
    ('majorbeta_hgc_wtime', 0.0, 1.0),
    ('A', 0.0, 1.0),
    ('R', 0.0, 1.0),
    ('M', 0.0, 1.0),
    ('lnearnings', 10.975, 0.650),
    ('male', 0.566, 0.496),
    ('black', 0.068, 0.252),
    ('hispanic', 0.048, 0.215),
]:
    wmean = np.average(df[var], weights=w)
    wvar = np.average((df[var] - wmean)**2, weights=w)
    wsd = np.sqrt(wvar)
    print(f"  {var:25s}: mean={wmean:9.4f} (pub={pub_mean:.3f})  "
          f"sd={wsd:7.4f} (pub={pub_sd:.3f})")

# Weighted correlation matrix
print("\n--- Weighted Correlations (Table A2 / Stata log) ---")
vars_corr = ['majorbeta_none', 'majorbeta_hgc_wtime', 'A', 'R', 'M']
labels = ['beta_none', 'beta_hgc', 'A', 'R', 'M']
n_v = len(vars_corr)
corr = np.zeros((n_v, n_v))
for i in range(n_v):
    for j in range(n_v):
        xi = df[vars_corr[i]].values
        xj = df[vars_corr[j]].values
        mi = np.average(xi, weights=w)
        mj = np.average(xj, weights=w)
        cov_ij = np.average((xi - mi) * (xj - mj), weights=w)
        vi = np.average((xi - mi)**2, weights=w)
        vj = np.average((xj - mj)**2, weights=w)
        corr[i, j] = cov_ij / np.sqrt(vi * vj)

print(f"  {'':12s}", end='')
for l in labels:
    print(f"  {l:>10s}", end='')
print()
for i, l in enumerate(labels):
    print(f"  {l:12s}", end='')
    for j in range(n_v):
        print(f"  {corr[i,j]:10.4f}", end='')
    print()

# Published correlations from Stata log
print("\n  Published correlations:")
print("  beta_none x beta_hgc: 0.9310 (regression sample)")
print(f"  Python:               {corr[0,1]:.4f}")
print(f"  A x R:   Python={corr[2,3]:.4f}  Published=0.5649")
print(f"  A x M:   Python={corr[2,4]:.4f}  Published=-0.3804")
print(f"  R x M:   Python={corr[3,4]:.4f}  Published=0.1041")

# Observations by survey
print("\n--- Observations by Survey ---")
for sv in [1993, 2003, 2009]:
    mask = df['survey'] == sv
    n_obs = mask.sum()
    years = df.loc[mask, 'year'].unique()
    print(f"  Survey {sv}: {n_obs:,d} obs, years: {sorted(years)}")

# First-stage regressions
print("\n--- First-Stage Regressions: beta^m on A, R, M ---")
for betam, label in [('majorbeta_none', 'No grad controls'),
                      ('majorbeta_hgc_wtime', 'With grad controls')]:
    resid, var_v, coefs, r2 = first_stage(df, betam)
    print(f"\n  {label} ({betam}):")
    print(f"    A coef: {coefs['A']:.4f}   R coef: {coefs['R']:.4f}   M coef: {coefs['M']:.4f}")
    print(f"    R^2: {r2:.4f}   Var(v^m): {var_v:.4f}")

    if betam == 'majorbeta_none':
        print(f"    Published: A=0.478, R=0.417, M=-0.076, R2=0.655, Var=0.345")
    else:
        print(f"    Published: A=0.357, R=0.520, M=-0.209, R2=0.685, Var=0.315")

# Covariances (used in variance decomposition)
print("\n--- Task Measure Covariances (weighted) ---")
covAR = wcov(df['A'].values, df['R'].values, w)
covAM = wcov(df['A'].values, df['M'].values, w)
covRM = wcov(df['R'].values, df['M'].values, w)
print(f"  Cov(A,R): {covAR:.6f}  (Published: 0.5649)")
print(f"  Cov(A,M): {covAM:.6f}  (Published: -0.3804)")
print(f"  Cov(R,M): {covRM:.6f}  (Published: 0.1041)")

# Number of clusters
n_clust = df['clust_var'].nunique()
print(f"\n  Number of clusters (major x year): {n_clust}  (Published: 240)")
n_majors = df['major'].nunique()
print(f"  Number of majors: {n_majors}  (Published: 51)")

print("\n" + "=" * 70)
print("Data verification complete.")
print("=" * 70)
