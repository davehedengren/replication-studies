"""
05_robustness.py – Robustness checks for Samaniego (2008) replication.

Baseline: reg turnover etcdlls87st c_dummies i_dummies, robust
Result: coef = -0.70, p = 0.006
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from utils import load_industry_data, load_did_data, run_areg_robust

df_ind = load_industry_data()
df_ind['i'] = range(1, 42)
df_did = load_did_data()

BASELINE_XVAR = 'etcdlls87st'
BASELINE_YVAR = 'turnover'

def stars(p):
    if p < 0.01: return '***'
    if p < 0.05: return '**'
    if p < 0.1: return '*'
    return ''

# Baseline result
baseline = run_areg_robust(df_did, BASELINE_YVAR, BASELINE_XVAR)
print("=" * 70)
print("ROBUSTNESS CHECKS")
print("=" * 70)
print(f"\nBaseline: coef={baseline['coef']:.3f}, p={baseline['pval']:.3f}, "
      f"N={baseline['N']}, R²={baseline['R2']:.3f}")

results = []

def log_result(name, res):
    s = stars(res['pval'])
    results.append({
        'check': name,
        'coef': res['coef'],
        'pval': res['pval'],
        'N': res['N'],
        'R2': res['R2'],
    })
    print(f"  {name:<50s}: coef={res['coef']:>7.3f}{s:<3s} (p={res['pval']:.3f}) "
          f"N={res['N']} R²={res['R2']:.3f}")


# ── 1. Alternative SE: cluster by industry ───────────────
print("\n1. ALTERNATIVE STANDARD ERRORS")
print("-" * 50)

# Cluster by industry (as in README's areg command)
temp = df_did[['turnover', BASELINE_XVAR, 'c', 'i']].dropna().copy()
c_dum = pd.get_dummies(temp['c'], prefix='c', drop_first=True).astype(float)
i_dum = pd.get_dummies(temp['i'], prefix='i', drop_first=True).astype(float)
X = pd.concat([temp[[BASELINE_XVAR]].reset_index(drop=True),
               c_dum.reset_index(drop=True),
               i_dum.reset_index(drop=True)], axis=1)
X = sm.add_constant(X)
y = temp['turnover'].reset_index(drop=True)
groups = temp['i'].reset_index(drop=True)

model_clust = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': groups})
log_result('Cluster by industry', {
    'coef': model_clust.params[BASELINE_XVAR],
    'pval': model_clust.pvalues[BASELINE_XVAR],
    'N': int(model_clust.nobs), 'R2': model_clust.rsquared})

# HC3 SEs
model_hc3 = sm.OLS(y, X).fit(cov_type='HC3')
log_result('HC3 robust SEs', {
    'coef': model_hc3.params[BASELINE_XVAR],
    'pval': model_hc3.pvalues[BASELINE_XVAR],
    'N': int(model_hc3.nobs), 'R2': model_hc3.rsquared})

# Cluster by country
groups_c = temp['c'].reset_index(drop=True)
model_clust_c = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': groups_c})
log_result('Cluster by country', {
    'coef': model_clust_c.params[BASELINE_XVAR],
    'pval': model_clust_c.pvalues[BASELINE_XVAR],
    'N': int(model_clust_c.nobs), 'R2': model_clust_c.rsquared})


# ── 2. Alternative interaction variable ──────────────────
print("\n2. ALTERNATIVE ISTC MEASURES")
print("-" * 50)

for label, xvar in [
    ('ISTC excl structures (DLLS)', 'etcdlls'),
    ('ISTC 1947-2000 (DLLS)', 'etcdlls00'),
    ('ISTC incl structures (WB)', 'etcwb87st'),
    ('ISTC excl structures (WB)', 'etcwb'),
]:
    res = run_areg_robust(df_did, 'turnover', xvar)
    log_result(label, res)


# ── 3. Drop outlier observations ─────────────────────────
print("\n3. DROP OUTLIER OBSERVATIONS")
print("-" * 50)

# Drop top/bottom 1% of turnover
q01, q99 = df_did['turnover'].quantile([0.01, 0.99])
trimmed = df_did[(df_did['turnover'] >= q01) & (df_did['turnover'] <= q99)]
res = run_areg_robust(trimmed, 'turnover', BASELINE_XVAR)
log_result('Drop top/bottom 1% turnover', res)

# Drop extreme outlier: Norway Finance (turnover=72.4)
no_outlier = df_did[~((df_did['c'] == 17) & (df_did['i'] == 28))]
res = run_areg_robust(no_outlier, 'turnover', BASELINE_XVAR)
log_result('Drop Norway Finance (max outlier)', res)


# ── 4. Drop one country at a time ────────────────────────
print("\n4. LEAVE-ONE-COUNTRY-OUT")
print("-" * 50)

country_names = {
    1: 'Belgium', 2: 'Czech Rep.', 3: 'Denmark', 4: 'Spain',
    5: 'Italy', 6: 'Latvia', 7: 'Lithuania', 8: 'Hungary',
    9: 'Netherl.', 10: 'Portugal', 11: 'Slovenia', 12: 'Slovakia',
    13: 'Finland', 14: 'Sweden', 15: 'UK', 16: 'Romania',
    17: 'Norway', 18: 'Switzerland'
}

for c in sorted(df_did['c'].unique()):
    sub = df_did[df_did['c'] != c]
    res = run_areg_robust(sub, 'turnover', BASELINE_XVAR)
    log_result(f'Drop {country_names[c]:<15s}', res)


# ── 5. Drop one industry at a time (most influential) ────
print("\n5. LEAVE-ONE-INDUSTRY-OUT (top 5 most influential)")
print("-" * 50)

ind_influence = []
for i in sorted(df_did['i'].unique()):
    sub = df_did[df_did['i'] != i]
    res = run_areg_robust(sub, 'turnover', BASELINE_XVAR)
    ind_influence.append((i, res['coef'], res['pval']))

# Sort by absolute change in coefficient
ind_influence.sort(key=lambda x: abs(x[1] - baseline['coef']), reverse=True)
for i, coef, pval in ind_influence[:5]:
    name = df_ind[df_ind['i'] == i]['Industry'].values[0]
    s = stars(pval)
    print(f"  Drop i={i:>2d} ({name:<35s}): coef={coef:>7.3f}{s:<3s} (p={pval:.3f}) "
          f"Δ={coef - baseline['coef']:+.3f}")


# ── 6. Winsorize outcomes ────────────────────────────────
print("\n6. WINSORIZED OUTCOMES")
print("-" * 50)

for pct in [1, 5]:
    temp = df_did.copy()
    lo, hi = temp['turnover'].quantile([pct/100, 1-pct/100])
    temp['turnover_w'] = temp['turnover'].clip(lo, hi)
    res = run_areg_robust(temp, 'turnover_w', BASELINE_XVAR)
    log_result(f'Winsorize turnover at {pct}%', res)


# ── 7. Placebo test: shuffle treatment ───────────────────
print("\n7. PLACEBO TEST (shuffle interaction variable)")
print("-" * 50)

np.random.seed(42)
n_perms = 1000
perm_coefs = []

temp = df_did[['turnover', BASELINE_XVAR, 'c', 'i']].dropna().copy()
for _ in range(n_perms):
    temp_perm = temp.copy()
    temp_perm[BASELINE_XVAR] = np.random.permutation(temp_perm[BASELINE_XVAR].values)
    res = run_areg_robust(temp_perm, 'turnover', BASELINE_XVAR)
    perm_coefs.append(res['coef'])

perm_coefs = np.array(perm_coefs)
perm_pval = np.mean(perm_coefs <= baseline['coef'])
print(f"  Baseline coef: {baseline['coef']:.3f}")
print(f"  Permutation distribution: mean={perm_coefs.mean():.3f}, sd={perm_coefs.std():.3f}")
print(f"  Permutation p-value (one-sided): {perm_pval:.3f}")
print(f"  Significant at 5%: {'YES' if perm_pval < 0.05 else 'NO'}")


# ── 8. Entry and exit separately ─────────────────────────
print("\n8. ENTRY AND EXIT SEPARATELY")
print("-" * 50)

for yvar in ['entry', 'exit']:
    res = run_areg_robust(df_did, yvar, BASELINE_XVAR)
    log_result(f'Dep var: {yvar}', res)


# ── 9. Only entry/exit as dependent (placebo: exit should also be neg)
print("\n9. PLACEBO OUTCOME: etcst (industry ISTC, should be absorbed by i FEs)")
print("-" * 50)
# regress etcst on the interaction - should be zero since etcst varies only by i
# and we have i FEs
res = run_areg_robust(df_did, 'etcst', BASELINE_XVAR)
log_result('Placebo: etcst as dependent var', res)


# ── 10. Restrict to non-zero obs only ────────────────────
print("\n10. RESTRICT TO NON-ZERO OUTCOMES")
print("-" * 50)

nonzero = df_did[df_did['turnover'] > 0]
res = run_areg_robust(nonzero, 'turnover', BASELINE_XVAR)
log_result('Drop zero-turnover obs', res)


# ── 11. Log transform ────────────────────────────────────
print("\n11. LOG TRANSFORMATION")
print("-" * 50)

temp = df_did.copy()
temp['log_turnover'] = np.log(temp['turnover'].clip(lower=0.01))
res = run_areg_robust(temp, 'log_turnover', BASELINE_XVAR)
log_result('Log(turnover) as dep var', res)


# ── 12. Using Spearman rank correlation ──────────────────
print("\n12. SPEARMAN RANK CORRELATION (cross-section)")
print("-" * 50)

# Re-compute Table 2 with Spearman correlations
# Using industry-level data
istc_map = dict(zip(df_ind['i'], df_ind['istc']))
means = df_did.groupby('i')[['turnover', 'entry', 'exit']].mean()
means['istc'] = means.index.map(istc_map)

for yvar in ['turnover', 'entry', 'exit']:
    valid = means[[yvar, 'istc']].dropna()
    r_spearman, p_spearman = stats.spearmanr(valid[yvar], valid['istc'])
    r_pearson, p_pearson = stats.pearsonr(valid[yvar], valid['istc'])
    print(f"  {yvar} × ISTC: Spearman r={r_spearman:.3f} (p={p_spearman:.3f}), "
          f"Pearson r={r_pearson:.3f} (p={p_pearson:.3f})")


# ── Summary ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("ROBUSTNESS SUMMARY")
print("=" * 70)

df_results = pd.DataFrame(results)
n_negative = (df_results['coef'] < 0).sum()
n_signif = (df_results['pval'] < 0.05).sum()
n_total = len(df_results)

print(f"\n  Baseline: coef = {baseline['coef']:.3f} (p = {baseline['pval']:.3f})")
print(f"  Checks with negative coefficient: {n_negative}/{n_total}")
print(f"  Checks significant at 5%: {n_signif}/{n_total}")
print(f"  Coefficient range: [{df_results['coef'].min():.3f}, {df_results['coef'].max():.3f}]")
print(f"  Result is {'ROBUST' if n_signif / n_total > 0.7 else 'FRAGILE'}")
