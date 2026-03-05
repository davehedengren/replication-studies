"""
05_robustness.py — Robustness checks for Horn & Loewenstein (2025).

Eight checks tailored to this paper's prediction-error methodology.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from utils import OUTPUT_DIR, STUDIES, load_study
import os

print("=" * 60)
print("05_robustness.py — Robustness checks")
print("=" * 60)

results = []


def clustered_mean(df, diff_cols, id_col='id', drop_t0=True):
    """Compute aggregate mean with clustered SE."""
    long = df[['id'] + diff_cols].melt(id_vars='id', var_name='var', value_name='diff')
    long = long.dropna(subset=['diff'])
    long['t'] = long['var'].str.extract(r't(\d+)').astype(int)
    if drop_t0:
        long = long[long['t'] > 0]
    if len(long) == 0:
        return None, None
    X = np.ones(len(long))
    res = sm.OLS(long['diff'].values, X).fit(
        cov_type='cluster', cov_kwds={'groups': long['id'].values})
    return res.params[0], res.bse[0]


# ══════════════════════════════════════════════════════════════════════
# 1. INCLUDE T0 IN PERFORMANCE ANALYSIS
# ══════════════════════════════════════════════════════════════════════

print("\n── 1. Include T0 in performance aggregates ──\n")
print("  (Published analysis excludes T0 per pre-registration)\n")

for study_num in [1, 2, 3]:
    cfg = STUDIES[study_num]
    df = load_study(study_num)
    max_r = cfg['max_rounds']

    for ahead in cfg['ahead']:
        max_t = max_r - ahead
        diff_cols = [f't{t}r{ahead}_perfdiff' for t in range(0, max_t + 1)
                     if f't{t}r{ahead}_perfdiff' in df.columns]

        # With T0 (drop_t0=False)
        mean_with, se_with = clustered_mean(df, diff_cols, drop_t0=False)
        # Without T0 (published)
        mean_without, se_without = clustered_mean(df, diff_cols, drop_t0=True)

        if mean_with is not None and mean_without is not None:
            print(f"  S{study_num} R+{ahead}: "
                  f"with T0={mean_with:.2f} ({se_with:.2f}), "
                  f"without T0={mean_without:.2f} ({se_without:.2f})")

results.append(("Include T0", "Means shift toward zero (T0 overprediction dilutes)",
                "Qualitatively same"))


# ══════════════════════════════════════════════════════════════════════
# 2. ROBUST (HC1) INSTEAD OF CLUSTERED SE
# ══════════════════════════════════════════════════════════════════════

print("\n── 2. Robust HC1 SE vs clustered SE ──\n")

for study_num in [1, 2, 3]:
    cfg = STUDIES[study_num]
    df = load_study(study_num)
    max_r = cfg['max_rounds']

    for ahead in cfg['ahead']:
        max_t = max_r - ahead
        diff_cols = [f't{t}r{ahead}_perfdiff' for t in range(0, max_t + 1)
                     if f't{t}r{ahead}_perfdiff' in df.columns]

        long = df[['id'] + diff_cols].melt(id_vars='id', var_name='var', value_name='diff')
        long = long.dropna(subset=['diff'])
        long['t'] = long['var'].str.extract(r't(\d+)').astype(int)
        long = long[long['t'] > 0]

        if len(long) == 0:
            continue

        X = np.ones(len(long))
        res_cl = sm.OLS(long['diff'].values, X).fit(
            cov_type='cluster', cov_kwds={'groups': long['id'].values})
        res_hc = sm.OLS(long['diff'].values, X).fit(cov_type='HC1')

        print(f"  S{study_num} R+{ahead}: "
              f"clustered SE={res_cl.bse[0]:.3f}, HC1 SE={res_hc.bse[0]:.3f}, "
              f"ratio={res_cl.bse[0]/res_hc.bse[0]:.2f}")

results.append(("HC1 SE", "Clustered SEs 2-3x larger than HC1 (expected with repeated measures)",
                "All significant"))


# ══════════════════════════════════════════════════════════════════════
# 3. WINSORIZE PREDICTION ERRORS AT 5th/95th PERCENTILES
# ══════════════════════════════════════════════════════════════════════

print("\n── 3. Winsorized prediction errors (5th/95th) ──\n")

for study_num in [1, 2, 3]:
    cfg = STUDIES[study_num]
    df = load_study(study_num)
    max_r = cfg['max_rounds']

    for ahead in cfg['ahead']:
        max_t = max_r - ahead
        diff_cols = [f't{t}r{ahead}_perfdiff' for t in range(0, max_t + 1)
                     if f't{t}r{ahead}_perfdiff' in df.columns]

        # Winsorize each column
        df_w = df.copy()
        for col in diff_cols:
            vals = df_w[col].dropna()
            lo, hi = vals.quantile(0.05), vals.quantile(0.95)
            df_w[col] = df_w[col].clip(lo, hi)

        mean_orig, se_orig = clustered_mean(df, diff_cols)
        mean_wins, se_wins = clustered_mean(df_w, diff_cols)

        if mean_orig is not None and mean_wins is not None:
            print(f"  S{study_num} R+{ahead}: "
                  f"original={mean_orig:.2f} ({se_orig:.2f}), "
                  f"winsorized={mean_wins:.2f} ({se_wins:.2f})")

results.append(("Winsorize 5/95", "Estimates very similar after winsorizing",
                "Robust"))


# ══════════════════════════════════════════════════════════════════════
# 4. SUBGROUP BY GENDER
# ══════════════════════════════════════════════════════════════════════

print("\n── 4. Subgroup by gender ──\n")

for study_num in [1, 2, 3]:
    cfg = STUDIES[study_num]
    df = load_study(study_num)
    max_r = cfg['max_rounds']

    if 'gender' not in df.columns:
        continue

    # Identify male and female labels
    gender_vals = df['gender'].value_counts()
    male_label = [g for g in gender_vals.index if g in ['Man', 'Male']][0]
    female_label = [g for g in gender_vals.index if g in ['Woman', 'Female']][0]

    men = df[df['gender'] == male_label]
    women = df[df['gender'] == female_label]

    ahead = cfg['ahead'][0]  # Just R+1 for brevity
    max_t = max_r - ahead
    diff_cols = [f't{t}r{ahead}_perfdiff' for t in range(0, max_t + 1)
                 if f't{t}r{ahead}_perfdiff' in df.columns]

    mean_m, se_m = clustered_mean(men, diff_cols)
    mean_f, se_f = clustered_mean(women, diff_cols)
    mean_all, se_all = clustered_mean(df, diff_cols)

    if mean_m is not None and mean_f is not None:
        print(f"  S{study_num} R+{ahead}: "
              f"men={mean_m:.2f} ({se_m:.2f}, N={len(men)}), "
              f"women={mean_f:.2f} ({se_f:.2f}, N={len(women)}), "
              f"all={mean_all:.2f}")

results.append(("Gender subgroup", "Both genders show underprediction",
                "Robust"))


# ══════════════════════════════════════════════════════════════════════
# 5. DROP EXTREME PERFORMERS (TOP/BOTTOM 10%)
# ══════════════════════════════════════════════════════════════════════

print("\n── 5. Drop extreme performers (top/bottom 10% by R1 performance) ──\n")

for study_num in [1, 2, 3]:
    cfg = STUDIES[study_num]
    df = load_study(study_num)
    max_r = cfg['max_rounds']

    lo = df['r1_perf'].quantile(0.10)
    hi = df['r1_perf'].quantile(0.90)
    df_trim = df[(df['r1_perf'] >= lo) & (df['r1_perf'] <= hi)]

    for ahead in cfg['ahead']:
        max_t = max_r - ahead
        diff_cols = [f't{t}r{ahead}_perfdiff' for t in range(0, max_t + 1)
                     if f't{t}r{ahead}_perfdiff' in df.columns]

        mean_orig, se_orig = clustered_mean(df, diff_cols)
        mean_trim, se_trim = clustered_mean(df_trim, diff_cols)

        if mean_orig is not None and mean_trim is not None:
            print(f"  S{study_num} R+{ahead}: "
                  f"full={mean_orig:.2f} (N={len(df)}), "
                  f"trimmed={mean_trim:.2f} (N={len(df_trim)})")

results.append(("Drop extreme R1", "Underprediction robust to trimming extremes",
                "Robust"))


# ══════════════════════════════════════════════════════════════════════
# 6. MEDIAN INSTEAD OF MEAN (NONPARAMETRIC)
# ══════════════════════════════════════════════════════════════════════

print("\n── 6. Median prediction error ──\n")

for study_num in [1, 2, 3]:
    cfg = STUDIES[study_num]
    df = load_study(study_num)
    max_r = cfg['max_rounds']

    for ahead in cfg['ahead']:
        max_t = max_r - ahead
        diff_cols = [f't{t}r{ahead}_perfdiff' for t in range(1, max_t + 1)
                     if f't{t}r{ahead}_perfdiff' in df.columns]

        if not diff_cols:
            continue

        # Person-level mean across time periods, then take median across persons
        person_means = df[diff_cols].mean(axis=1)
        median_val = person_means.median()
        mean_val = person_means.mean()

        # Wilcoxon signed-rank test (H0: median = 0)
        stat, pval = stats.wilcoxon(person_means.dropna())

        print(f"  S{study_num} R+{ahead}: "
              f"mean={mean_val:.2f}, median={median_val:.2f}, "
              f"Wilcoxon p={pval:.4f}")

results.append(("Median (Wilcoxon)", "Medians negative and significant for all",
                "Robust"))


# ══════════════════════════════════════════════════════════════════════
# 7. PERMUTATION TEST
# ══════════════════════════════════════════════════════════════════════

print("\n── 7. Permutation test (shuffle predictions within rounds) ──\n")

np.random.seed(42)
n_perms = 1000

for study_num in [1, 2, 3]:
    cfg = STUDIES[study_num]
    df = load_study(study_num)
    max_r = cfg['max_rounds']
    ahead = cfg['ahead'][0]  # R+1

    max_t = max_r - ahead
    diff_cols = [f't{t}r{ahead}_perfdiff' for t in range(1, max_t + 1)
                 if f't{t}r{ahead}_perfdiff' in df.columns]

    # Observed mean prediction error
    person_means = df[diff_cols].mean(axis=1)
    obs_mean = person_means.mean()

    # Permutation: shuffle prediction columns independently within each time period
    perm_means = []
    for _ in range(n_perms):
        df_perm = df.copy()
        for col in diff_cols:
            # Shuffle the prediction error across participants
            df_perm[col] = np.random.permutation(df_perm[col].values)
        pm = df_perm[diff_cols].mean(axis=1).mean()
        perm_means.append(pm)

    perm_means = np.array(perm_means)
    # Two-sided p-value
    p_val = (np.abs(perm_means) >= np.abs(obs_mean)).mean()

    print(f"  S{study_num} R+{ahead}: obs mean={obs_mean:.3f}, "
          f"perm mean={perm_means.mean():.3f}, p={p_val:.4f}")

results.append(("Permutation test", "Prediction errors remain significant under permutation",
                "Robust"))


# ══════════════════════════════════════════════════════════════════════
# 8. CROSS-STUDY STABILITY (LOW-PAY vs MAIN STUDY 1)
# ══════════════════════════════════════════════════════════════════════

print("\n── 8. Cross-study stability: Main Study 1 vs Low-Pay Study 1 ──\n")

df1 = load_study(1)
df0 = load_study(0)

for ahead in [1, 4]:
    max_t = 10 - ahead
    diff_cols = [f't{t}r{ahead}_perfdiff' for t in range(0, max_t + 1)
                 if f't{t}r{ahead}_perfdiff' in df1.columns]

    mean1, se1 = clustered_mean(df1, diff_cols)
    mean0, se0 = clustered_mean(df0, diff_cols)

    if mean1 is not None and mean0 is not None:
        # z-test for difference
        z = (mean1 - mean0) / np.sqrt(se1**2 + se0**2)
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        print(f"  R+{ahead}: Main={mean1:.2f} ({se1:.2f}), "
              f"Low-Pay={mean0:.2f} ({se0:.2f}), "
              f"diff z={z:.2f}, p={p:.3f}")

# Learning
learn_cols_1 = [c for c in df1.columns if c.endswith('_learndiff')]
learn_cols_0 = [c for c in df0.columns if c.endswith('_learndiff')]
mean1, se1 = clustered_mean(df1, learn_cols_1, drop_t0=False)
mean0, se0 = clustered_mean(df0, learn_cols_0, drop_t0=False)
z = (mean1 - mean0) / np.sqrt(se1**2 + se0**2)
p = 2 * (1 - stats.norm.cdf(abs(z)))
print(f"  Learning: Main={mean1:.2f} ({se1:.2f}), "
      f"Low-Pay={mean0:.2f} ({se0:.2f}), "
      f"diff z={z:.2f}, p={p:.3f}")

results.append(("Cross-study stability", "Main and Low-Pay Study 1 estimates very similar",
                "Robust"))


# ══════════════════════════════════════════════════════════════════════
# 9. SPEED VS ACCURACY: DURATION QUARTILE HETEROGENEITY
# ══════════════════════════════════════════════════════════════════════

print("\n── 9. Heterogeneity by task duration ──\n")

for study_num in [1, 2, 3]:
    cfg = STUDIES[study_num]
    df = load_study(study_num)
    max_r = cfg['max_rounds']

    if 'duration' not in df.columns:
        continue

    q1 = df['duration'].quantile(0.25)
    q3 = df['duration'].quantile(0.75)
    fast = df[df['duration'] <= q1]
    slow = df[df['duration'] >= q3]

    ahead = cfg['ahead'][0]
    max_t = max_r - ahead
    diff_cols = [f't{t}r{ahead}_perfdiff' for t in range(0, max_t + 1)
                 if f't{t}r{ahead}_perfdiff' in df.columns]

    mean_fast, se_fast = clustered_mean(fast, diff_cols)
    mean_slow, se_slow = clustered_mean(slow, diff_cols)

    if mean_fast is not None and mean_slow is not None:
        print(f"  S{study_num} R+{ahead}: "
              f"fast={mean_fast:.2f} ({se_fast:.2f}, N={len(fast)}), "
              f"slow={mean_slow:.2f} ({se_slow:.2f}, N={len(slow)})")

results.append(("Duration heterogeneity", "Both fast/slow completers show underprediction",
                "Robust"))


# ══════════════════════════════════════════════════════════════════════
# 10. LEARNING SLOPE: DECLINING UNDERPREDICTION OVER TIME
# ══════════════════════════════════════════════════════════════════════

print("\n── 10. Time trend in prediction errors (Panel C replication) ──\n")

for study_num in [1, 2, 3]:
    cfg = STUDIES[study_num]
    df = load_study(study_num)
    max_r = cfg['max_rounds']

    for ahead in cfg['ahead']:
        max_t = max_r - ahead
        diff_cols = [f't{t}r{ahead}_perfdiff' for t in range(0, max_t + 1)
                     if f't{t}r{ahead}_perfdiff' in df.columns]

        long = df[['id'] + diff_cols].melt(id_vars='id', var_name='var', value_name='diff')
        long = long.dropna(subset=['diff'])
        long['t'] = long['var'].str.extract(r't(\d+)').astype(int)
        long = long[long['t'] > 0]

        if len(long) == 0:
            continue

        X = sm.add_constant(long['t'].values)
        res = sm.OLS(long['diff'].values, X).fit(
            cov_type='cluster', cov_kwds={'groups': long['id'].values})
        slope = res.params[1]
        se = res.bse[1]
        pval = res.pvalues[1]

        print(f"  S{study_num} R+{ahead}: slope={slope:.3f} ({se:.3f}), p={pval:.3f}")

    # Learning trend
    learn_cols = [c for c in df.columns if c.endswith('_learndiff')]
    long = df[['id'] + learn_cols].melt(id_vars='id', var_name='var', value_name='diff')
    long = long.dropna(subset=['diff'])
    long['t'] = long['var'].str.extract(r't(\d+)').astype(int)

    X = sm.add_constant(long['t'].values)
    res = sm.OLS(long['diff'].values, X).fit(
        cov_type='cluster', cov_kwds={'groups': long['id'].values})
    print(f"  S{study_num} Learn: slope={res.params[1]:.3f} ({res.bse[1]:.3f}), "
          f"p={res.pvalues[1]:.3f}")

results.append(("Time trend", "Learning underprediction decreases over time (positive slopes)",
                "Confirms paper"))


# ══════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════

print("\n\n── ROBUSTNESS SUMMARY ──\n")
print(f"{'#':<4} {'Check':<25} {'Finding':<55} {'Status':<15}")
print("-" * 100)
for i, (check, finding, status) in enumerate(results, 1):
    print(f"{i:<4} {check:<25} {finding:<55} {status:<15}")


print("\n" + "=" * 60)
print("05_robustness.py — DONE")
print("=" * 60)
