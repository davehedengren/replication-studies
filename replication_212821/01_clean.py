"""
01_clean.py — Verify clean data for Horn & Loewenstein (2025) replication.

The replication package includes pre-cleaned .dta files. This script
validates them and saves as parquet for faster loading.
"""

import numpy as np
import pandas as pd
from utils import DATA_DIR, OUTPUT_DIR, STUDIES, load_study
import os

print("=" * 60)
print("01_clean.py — Verifying clean data")
print("=" * 60)

for study_num in [1, 2, 3, 4, 0]:
    cfg = STUDIES[study_num]
    df = load_study(study_num)
    print(f"\n  Study {study_num} ({cfg['name']}): {df.shape[0]} obs, {df.shape[1]} vars")

    # Basic validation
    max_r = cfg['max_rounds']

    # Check performance columns exist
    perf_cols = [f'r{r}_perf' for r in range(1, max_r + 1)]
    missing_perf = [c for c in perf_cols if c not in df.columns]
    if missing_perf:
        print(f"    WARNING: Missing perf columns: {missing_perf}")
    else:
        # Performance summary
        means = [df[f'r{r}_perf'].mean() for r in range(1, max_r + 1)]
        print(f"    Performance range: R1={means[0]:.2f} → R{max_r}={means[-1]:.2f}")

    # Check prediction diff columns
    for ahead in cfg['ahead']:
        max_t = max_r - ahead
        diff_cols = [f't{t}r{ahead}_perfdiff' for t in range(0, max_t + 1)
                     if f't{t}r{ahead}_perfdiff' in df.columns]
        n_diff = len(diff_cols)
        print(f"    Perf diff (R+{ahead}): {n_diff} time periods")

    # Check learning diff columns
    learn_cols = [c for c in df.columns if c.endswith('_learndiff')]
    print(f"    Learning diff vars: {len(learn_cols)}")

    # Study 4: check performer variable
    if study_num == 4:
        n_perf = (df['performer'] == 1).sum()
        n_pred = (df['performer'] == 0).sum()
        print(f"    Performers: {n_perf}, Predictors: {n_pred}")

    # Save as parquet
    out_path = os.path.join(OUTPUT_DIR, f'study{study_num}.parquet')
    df.to_parquet(out_path, index=False)
    print(f"    Saved: {out_path}")


# Quick validation: Table 1, Panel A, Col 1 (Study 1, R+1 perf, aggregate)
print("\n\n── Validation: Table 1, Panel A, Col 1 ──")
df1 = load_study(1)
# Reshape t{T}r1_perfdiff to long, drop T0
perfdiff_cols = [f't{t}r1_perfdiff' for t in range(0, 10) if f't{t}r1_perfdiff' in df1.columns]
long = df1[['id'] + perfdiff_cols].melt(id_vars='id', var_name='var', value_name='diff')
long['t'] = long['var'].str.extract(r't(\d+)').astype(int)
long = long[long['t'] > 0]  # drop T0

# Clustered mean (reg on constant, cl(id))
import statsmodels.api as sm
X = np.ones(len(long))
res = sm.OLS(long['diff'].values, X).fit(
    cov_type='cluster', cov_kwds={'groups': long['id'].values})
print(f"  Mean: {res.params[0]:.2f} (SE: {res.bse[0]:.2f})")
print(f"  Published: -0.42 (SE: 0.05)")

print("\n" + "=" * 60)
print("01_clean.py — DONE")
print("=" * 60)
