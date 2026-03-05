"""
02_tables.py — Reproduce Tables 1-2 and Appendix Tables A1-A5.

Horn & Loewenstein (2025) "Underestimating Learning by Doing"
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from utils import OUTPUT_DIR, STUDIES, load_study
import os

print("=" * 60)
print("02_tables.py — Reproducing all tables")
print("=" * 60)


def clustered_mean(df, diff_cols, id_col='id', drop_t0=True):
    """Compute aggregate mean with clustered SE (Panel A approach).

    Reshapes diff_cols to long, optionally drops T0, regresses on constant with cl(id).
    Note: Stata code drops T0 for performance but NOT for learning.
    """
    long = df[['id'] + diff_cols].melt(id_vars='id', var_name='var', value_name='diff')
    long = long.dropna(subset=['diff'])

    # Extract time period from variable name
    long['t'] = long['var'].str.extract(r't(\d+)').astype(int)
    if drop_t0:
        long = long[long['t'] > 0]  # Drop T0 (for performance only)

    if len(long) == 0:
        return None, None

    X = np.ones(len(long))
    res = sm.OLS(long['diff'].values, X).fit(
        cov_type='cluster', cov_kwds={'groups': long['id'].values})
    return res.params[0], res.bse[0]


def clustered_regression(df, diff_cols, id_col='id', drop_t0=True):
    """Regress prediction error on time period with clustered SE (Panel C)."""
    long = df[['id'] + diff_cols].melt(id_vars='id', var_name='var', value_name='diff')
    long = long.dropna(subset=['diff'])
    long['t'] = long['var'].str.extract(r't(\d+)').astype(int)
    if drop_t0:
        long = long[long['t'] > 0]  # Drop T0 (for performance only)

    if len(long) == 0:
        return None, None, None, None

    X = sm.add_constant(long['t'].values)
    res = sm.OLS(long['diff'].values, X).fit(
        cov_type='cluster', cov_kwds={'groups': long['id'].values})
    return res.params[1], res.bse[1], res.params[0], res.bse[0]  # coef, se, const, const_se


# ══════════════════════════════════════════════════════════════════════
# TABLE 1: Prediction errors for Studies 1-3
# ══════════════════════════════════════════════════════════════════════

print("\n── TABLE 1: Studies 1-3 ──\n")

for study_num in [1, 2, 3]:
    cfg = STUDIES[study_num]
    df = load_study(study_num)
    max_r = cfg['max_rounds']
    ahead_list = cfg['ahead']

    print(f"\n  Study {study_num}: {cfg['name']} (N={len(df)})")

    for ahead in ahead_list:
        max_t = max_r - ahead

        # Panel A: Aggregate mean with clustered SE
        diff_cols = [f't{t}r{ahead}_perfdiff' for t in range(0, max_t + 1)
                     if f't{t}r{ahead}_perfdiff' in df.columns]
        mean, se = clustered_mean(df, diff_cols)
        if mean is not None:
            print(f"    Panel A — Perf R+{ahead}: mean={mean:.2f} (SE={se:.2f})")

    # Learning aggregate
    learn_cols = [c for c in df.columns if c.endswith('_learndiff')]
    mean, se = clustered_mean(df, learn_cols, drop_t0=False)
    if mean is not None:
        print(f"    Panel A — Learning: mean={mean:.2f} (SE={se:.2f})")

    # Panel B: Individual time period means
    print(f"    Panel B:")
    for ahead in ahead_list:
        max_t = max_r - ahead
        for t in range(0, 10):
            col = f't{t}r{ahead}_perfdiff'
            if col in df.columns:
                vals = df[col].dropna()
                mean = vals.mean()
                se = vals.std(ddof=1) / np.sqrt(len(vals))
                print(f"      T{t} R+{ahead}: {mean:.2f} ({se:.2f})")

    # Learning by period
    for col in sorted(learn_cols):
        vals = df[col].dropna()
        t = int(col.split('_')[0][1:])
        mean = vals.mean()
        se = vals.std(ddof=1) / np.sqrt(len(vals))
        print(f"      T{t} Learn: {mean:.2f} ({se:.2f})")

    # Panel C: Regression on time period
    for ahead in ahead_list:
        max_t = max_r - ahead
        diff_cols = [f't{t}r{ahead}_perfdiff' for t in range(0, max_t + 1)
                     if f't{t}r{ahead}_perfdiff' in df.columns]
        coef, se, const, const_se = clustered_regression(df, diff_cols)
        if coef is not None:
            print(f"    Panel C — Perf R+{ahead}: slope={coef:.2f} ({se:.2f}), const={const:.2f}")

    coef, se, const, const_se = clustered_regression(df, learn_cols, drop_t0=False)
    if coef is not None:
        print(f"    Panel C — Learning: slope={coef:.2f} ({se:.2f}), const={const:.2f}")


# ══════════════════════════════════════════════════════════════════════
# TABLE 2: Performers vs Predictors (Study 4)
# ══════════════════════════════════════════════════════════════════════

print("\n\n── TABLE 2: Performers vs Predictors ──\n")

df4 = load_study(4)
performers = df4[df4['performer'] == 1].copy()
predictors = df4[df4['performer'] == 0].copy()

print(f"  Performers: N={len(performers)}")
print(f"  Predictors: N={len(predictors)}")

max_r = 10
ahead_list = [1, 4]

for label, sub in [("Performers", performers), ("Predictors", predictors)]:
    print(f"\n  {label}:")

    for ahead in ahead_list:
        max_t = max_r - ahead
        diff_cols = [f't{t}r{ahead}_perfdiff' for t in range(0, max_t + 1)
                     if f't{t}r{ahead}_perfdiff' in sub.columns]
        mean, se = clustered_mean(sub, diff_cols)
        if mean is not None:
            print(f"    Panel A — Perf R+{ahead}: mean={mean:.2f} (SE={se:.2f})")

    learn_cols = [c for c in sub.columns if c.endswith('_learndiff')]
    mean, se = clustered_mean(sub, learn_cols, drop_t0=False)
    if mean is not None:
        print(f"    Panel A — Learning: mean={mean:.2f} (SE={se:.2f})")

# Difference (Performers - Predictors): regression on performer indicator
print(f"\n  Difference (Performers - Predictors):")
for ahead in ahead_list:
    max_t = max_r - ahead
    diff_cols = [f't{t}r{ahead}_perfdiff' for t in range(0, max_t + 1)
                 if f't{t}r{ahead}_perfdiff' in df4.columns]

    long = df4[['id', 'performer'] + diff_cols].melt(
        id_vars=['id', 'performer'], var_name='var', value_name='diff')
    long = long.dropna(subset=['diff'])
    long['t'] = long['var'].str.extract(r't(\d+)').astype(int)
    long = long[long['t'] > 0]

    res = sm.OLS(long['diff'].values,
                 sm.add_constant(long['performer'].values)).fit(
        cov_type='cluster', cov_kwds={'groups': long['id'].values})
    coef = res.params[1]
    se = res.bse[1]
    print(f"    Perf R+{ahead}: diff={coef:.2f} (SE={se:.2f})")

# Learning difference
learn_cols = [c for c in df4.columns if c.endswith('_learndiff')]
long = df4[['id', 'performer'] + learn_cols].melt(
    id_vars=['id', 'performer'], var_name='var', value_name='diff')
long = long.dropna(subset=['diff'])

res = sm.OLS(long['diff'].values,
             sm.add_constant(long['performer'].values)).fit(
    cov_type='cluster', cov_kwds={'groups': long['id'].values})
print(f"    Learning: diff={res.params[1]:.2f} (SE={res.bse[1]:.2f})")


# ══════════════════════════════════════════════════════════════════════
# APPENDIX TABLES A1-A3: Detailed summary stats
# ══════════════════════════════════════════════════════════════════════

print("\n\n── APPENDIX TABLES A1-A3 ──\n")

for study_num in [1, 2, 3]:
    cfg = STUDIES[study_num]
    df = load_study(study_num)
    max_r = cfg['max_rounds']

    print(f"\n  Table A{study_num}: {cfg['name']} (N={len(df)})")
    print(f"  {'Round':<12} {'Actual':>8} {'Pred R+1':>10} {'Diff R+1':>10} {'Pred R+{}'.format(cfg['ahead'][1]):>10} {'Diff R+{}'.format(cfg['ahead'][1]):>10}")

    for r in range(1, max_r + 1):
        actual = df[f'r{r}_perf'].mean()
        actual_se = df[f'r{r}_perf'].std(ddof=1) / np.sqrt(len(df))

        # Prediction 1 round before: t = r-1
        t = r - 1
        pred1_col = f't{t}_pred_r{r}'
        if pred1_col in df.columns:
            pred1 = df[pred1_col].mean()
            diff1_col = f't{t}r1_perfdiff'
            diff1 = df[diff1_col].mean() if diff1_col in df.columns else np.nan
        else:
            pred1 = np.nan
            diff1 = np.nan

        # Prediction ahead[1] rounds before
        ahead2 = cfg['ahead'][1]
        t2 = r - ahead2
        if t2 >= 0:
            pred2_col = f't{t2}_pred_r{r}'
            if pred2_col in df.columns:
                pred2 = df[pred2_col].mean()
                diff2_col = f't{t2}r{ahead2}_perfdiff'
                diff2 = df[diff2_col].mean() if diff2_col in df.columns else np.nan
            else:
                pred2 = np.nan
                diff2 = np.nan
        else:
            pred2 = np.nan
            diff2 = np.nan

        print(f"  Round {r:<5} {actual:>8.2f} {pred1:>10.2f} {diff1:>10.2f} "
              f"{pred2:>10.2f} {diff2:>10.2f}" if not np.isnan(pred2)
              else f"  Round {r:<5} {actual:>8.2f} {pred1:>10.2f} {diff1:>10.2f}")


# ══════════════════════════════════════════════════════════════════════
# APPENDIX TABLE A5: Low-Pay Study 1
# ══════════════════════════════════════════════════════════════════════

print("\n\n── APPENDIX TABLE A5: Low-Pay Study 1 (N=342) ──\n")

df0 = load_study(0)
max_r = 10

for ahead in [1, 4]:
    max_t = max_r - ahead
    diff_cols = [f't{t}r{ahead}_perfdiff' for t in range(0, max_t + 1)
                 if f't{t}r{ahead}_perfdiff' in df0.columns]
    mean, se = clustered_mean(df0, diff_cols)
    if mean is not None:
        print(f"  Panel A — Perf R+{ahead}: mean={mean:.2f} (SE={se:.2f})")

learn_cols = [c for c in df0.columns if c.endswith('_learndiff')]
mean, se = clustered_mean(df0, learn_cols)
if mean is not None:
    print(f"  Panel A — Learning: mean={mean:.2f} (SE={se:.2f})")

# Panel C
for ahead in [1, 4]:
    max_t = max_r - ahead
    diff_cols = [f't{t}r{ahead}_perfdiff' for t in range(0, max_t + 1)
                 if f't{t}r{ahead}_perfdiff' in df0.columns]
    coef, se, const, const_se = clustered_regression(df0, diff_cols)
    if coef is not None:
        print(f"  Panel C — Perf R+{ahead}: slope={coef:.2f} ({se:.2f}), const={const:.2f}")

coef, se, const, const_se = clustered_regression(df0, learn_cols)
if coef is not None:
    print(f"  Panel C — Learning: slope={coef:.2f} ({se:.2f}), const={const:.2f}")


# ══════════════════════════════════════════════════════════════════════
# VALIDATION SUMMARY
# ══════════════════════════════════════════════════════════════════════

print("\n\n── VALIDATION SUMMARY ──\n")

# Compare key published values from Table 1
published = {
    'Table1_A_S1_R1': (-0.42, 0.05),
    'Table1_A_S1_R4': (-0.79, 0.08),
    'Table1_A_S1_Learn': (-0.58, 0.04),
    'Table1_A_S2_R1': (-0.38, 0.06),
    'Table1_A_S2_R3': (-0.62, 0.08),
    'Table1_A_S2_Learn': (-0.47, 0.04),
    'Table1_A_S3_R1': (-0.60, 0.06),
    'Table1_A_S3_R4': (-1.39, 0.11),
    'Table1_A_S3_Learn': (-1.05, 0.08),
}

results = {}
for study_num in [1, 2, 3]:
    cfg = STUDIES[study_num]
    df = load_study(study_num)
    max_r = cfg['max_rounds']

    for ahead in cfg['ahead']:
        max_t = max_r - ahead
        diff_cols = [f't{t}r{ahead}_perfdiff' for t in range(0, max_t + 1)
                     if f't{t}r{ahead}_perfdiff' in df.columns]
        mean, se = clustered_mean(df, diff_cols)
        key = f'Table1_A_S{study_num}_R{ahead}'
        results[key] = (round(mean, 2), round(se, 2))

    learn_cols = [c for c in df.columns if c.endswith('_learndiff')]
    mean, se = clustered_mean(df, learn_cols, drop_t0=False)
    key = f'Table1_A_S{study_num}_Learn'
    results[key] = (round(mean, 2), round(se, 2))

print(f"{'Key':<30} {'Published':>15} {'Replication':>15} {'Match':>8}")
print("-" * 70)
for key in published:
    pub = published[key]
    rep = results.get(key, (None, None))
    match = "YES" if pub == rep else "NO"
    print(f"{key:<30} {pub[0]:>7.2f} ({pub[1]:.2f}) {rep[0]:>7.2f} ({rep[1]:.2f}) {match:>8}")


print("\n" + "=" * 60)
print("02_tables.py — DONE")
print("=" * 60)
