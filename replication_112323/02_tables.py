"""
02_tables.py – Replicate Tables 1-5 and 11 from Samaniego (2008).

Key regressions:
  Tables 4-5: reg y interaction c_dummies i_dummies, robust (HC1)
  Alternative: areg y interaction, absorb(c) cluster(i)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from utils import (load_industry_data, load_did_data,
                   run_areg_robust, run_areg_cluster,
                   OUTCOMES, DLLS_VARS, WB_VARS)

df_ind = load_industry_data()
df_did = load_did_data()

# ================================================================
# TABLE 1: Industry data
# ================================================================
print("=" * 70)
print("TABLE 1: Annual % rate of ISTC, turnover, entry, exit")
print("=" * 70)

# Construct regression-based indices: y_{j,c} = α_c + α_j + ε
# Index = α_j + α_c(median)
index_results = {}
for yvar in ['turnover', 'entry', 'exit']:
    temp = df_did[['c', 'i', yvar]].dropna().copy()
    c_dum = pd.get_dummies(temp['c'], prefix='c', drop_first=True).astype(float)
    i_dum = pd.get_dummies(temp['i'], prefix='i', drop_first=True).astype(float)
    X = pd.concat([c_dum.reset_index(drop=True),
                   i_dum.reset_index(drop=True)], axis=1)
    X = sm.add_constant(X)
    y = temp[yvar].reset_index(drop=True)
    model = sm.OLS(y, X).fit()

    # Industry FEs: constant = reference (c=1, i=1) baseline
    ind_fe = {}
    ind_fe[1] = model.params['const']
    for col in i_dum.columns:
        i_num = int(col.replace('i_', ''))
        ind_fe[i_num] = model.params['const'] + model.params[col]

    # Country FEs
    c_fe = {1: 0.0}
    for col in c_dum.columns:
        c_num = int(col.replace('c_', ''))
        c_fe[c_num] = model.params[col]

    median_c_val = np.median(list(c_fe.values()))
    for i_num in ind_fe:
        ind_fe[i_num] += median_c_val

    index_results[yvar] = ind_fe

# Build table mapping i-codes to industry names and ISTC
df_ind['i'] = range(1, 42)
table1 = df_ind[['i', 'Industry', 'istc']].copy()
for yvar in ['turnover', 'entry', 'exit']:
    table1[yvar] = table1['i'].map(index_results[yvar])

table1 = table1.sort_values('istc')
print(table1[['Industry', 'istc', 'turnover', 'entry', 'exit']].to_string(index=False))
print(f"\n{'Median value':<40s} {table1['istc'].median():.2f} "
      f"{table1['turnover'].median():.2f} {table1['entry'].median():.2f} "
      f"{table1['exit'].median():.2f}")

# Published Table 1 median: ISTC=4.02, Turnover=22.77, Entry=11.96, Exit=9.43
print("\nPublished medians: ISTC=4.02, Turnover=22.77, Entry=11.96, Exit=9.43")

# ================================================================
# TABLE 2: Correlations between turnover measures and ISTC
# ================================================================
print("\n" + "=" * 70)
print("TABLE 2: Correlations between turnover measures and ISTC")
print("=" * 70)

# Use the regression-based indices
corr_data = pd.DataFrame({
    'turnover': pd.Series(index_results['turnover']),
    'entry': pd.Series(index_results['entry']),
    'exit': pd.Series(index_results['exit']),
    'istc': table1.set_index('i')['istc'],
})

# Pairwise correlations with p-values
pairs = [
    ('turnover', 'entry'), ('turnover', 'exit'), ('turnover', 'istc'),
    ('entry', 'exit'), ('entry', 'istc'), ('exit', 'istc'),
]

print(f"\n{'Pair':<25s} {'Corr':>8s} {'P-value':>10s} {'Published':>10s}")
print("-" * 55)
published = {
    ('turnover', 'entry'):  (0.96, 0.000),
    ('turnover', 'exit'):   (0.85, 0.000),
    ('turnover', 'istc'):   (0.52, 0.000),
    ('entry', 'exit'):      (0.67, 0.000),
    ('entry', 'istc'):      (0.48, 0.002),
    ('exit', 'istc'):       (0.49, 0.001),
}
for a, b in pairs:
    valid = corr_data[[a, b]].dropna()
    r, p = stats.pearsonr(valid[a], valid[b])
    pub_r, pub_p = published[(a, b)]
    print(f"  {a} × {b:<12s} {r:8.2f}   ({p:.3f})     {pub_r:.2f} ({pub_p:.3f})")

# ================================================================
# TABLE 3: Cross-industry correlations by country
# ================================================================
print("\n" + "=" * 70)
print("TABLE 3: Cross-industry correlations by country")
print("=" * 70)

# For each country, correlate ISTC with turnover/entry/exit across industries
# ISTC is from the etcst variable (industry-level, already in DiD data)
# But we need the un-normalized ISTC. Use the istc from industry file.
istc_map = dict(zip(df_ind['i'], df_ind['istc']))

country_names = {
    1: 'Belgium', 2: 'Czech Rep.', 3: 'Denmark', 4: 'Spain',
    5: 'Italy', 6: 'Latvia', 7: 'Lithuania', 8: 'Hungary',
    9: 'Netherl.', 10: 'Portugal', 11: 'Slovenia', 12: 'Slovakia',
    13: 'Finland', 14: 'Sweden', 15: 'UK', 16: 'Romania',
    17: 'Norway', 18: 'Switzerland'
}

print(f"\n{'Country':<15s} {'Turnover':>10s} {'Entry':>10s} {'Exit':>10s}")
print("-" * 50)

# Published Table 3 values
pub_t3 = {
    'Belgium':     (0.70, 0.70, 0.47),
    'Czech Rep.':  (0.37, 0.27, 0.33),
    'Denmark':     (0.70, 0.65, 0.75),
    'Spain':       (0.39, 0.47, 0.03),
    'Italy':       (0.70, 0.67, 0.44),
    'Latvia':      (0.29, 0.23, 0.31),
    'Lithuania':   (0.34, 0.30, 0.12),
    'Hungary':     (0.28, 0.38, 0.05),
    'Netherl.':    (0.77, 0.73, 0.75),
    'Portugal':    (0.40, 0.37, 0.26),
    'Slovenia':    (0.25, 0.30, -0.05),
    'Slovakia':    (0.14, 0.16, 0.07),
    'Finland':     (0.58, 0.54, 0.57),
    'Sweden':      (0.57, 0.54, 0.44),
    'UK':          (0.49, 0.50, 0.35),
    'Romania':     (0.59, 0.55, 0.21),
    'Norway':      (0.59, 0.54, 0.43),
    'Switzerland': (0.75, 0.63, 0.73),
}

for c_code in sorted(country_names.keys()):
    c_name = country_names[c_code]
    sub = df_did[df_did['c'] == c_code].copy()
    sub['istc'] = sub['i'].map(istc_map)
    corrs = []
    for yvar in ['turnover', 'entry', 'exit']:
        valid = sub[['istc', yvar]].dropna()
        if len(valid) > 2:
            r, p = stats.pearsonr(valid['istc'], valid[yvar])
            stars = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
            corrs.append(f"{r:.2f}{stars}")
        else:
            corrs.append('N/A')

    pub = pub_t3.get(c_name, (None, None, None))
    print(f"  {c_name:<13s} {corrs[0]:>10s} {corrs[1]:>10s} {corrs[2]:>10s}"
          f"  | Pub: {pub[0]}, {pub[1]}, {pub[2]}")


# ================================================================
# TABLES 4-5: Diff-in-diff regressions
# ================================================================
print("\n" + "=" * 70)
print("TABLE 4: Effect on turnover of ISTC × DLLS entry costs")
print("Regression: reg y interaction c_dummies i_dummies, robust")
print("=" * 70)

def stars(p):
    if p < 0.01: return '***'
    if p < 0.05: return '**'
    if p < 0.1: return '*'
    return ''

# Table 4 - DLLS entry costs
table4_specs = [
    # (label, sample_filter, xvar, dep_vars)
    ('All industries, 1987-97', None, 'etcdlls87st', ['turnover', 'entry', 'exit']),
    ('All industries, 1947-2000', None, 'etcdlls00', ['turnover', 'entry', 'exit']),
    ('Manufacturing only, 1987-97', lambda d: d[d['manuf'] == 1], 'etcdlls87st', ['turnover']),
    ('Non-manuf only, 1987-97', lambda d: d[d['manuf'] == 0], 'etcdlls87st', ['turnover']),
]

# Published Table 4 values (coef, p-value, N, R²)
pub_t4 = {
    ('All industries, 1987-97', 'turnover'):     (-0.70, 0.007, 719, 0.63),
    ('All industries, 1987-97', 'entry'):         (-0.34, 0.042, 724, 0.62),
    ('All industries, 1987-97', 'exit'):          (-0.37, 0.005, 721, 0.49),
    ('All industries, 1947-2000', 'turnover'):    (-0.59, 0.016, 719, 0.63),
    ('All industries, 1947-2000', 'entry'):       (-0.26, 0.078, 724, 0.62),
    ('All industries, 1947-2000', 'exit'):        (-0.34, 0.012, 721, 0.49),
    ('Manufacturing only, 1987-97', 'turnover'):  (-1.00, 0.044, 283, 0.68),
    ('Non-manuf only, 1987-97', 'turnover'):      (-0.79, 0.014, 436, 0.58),
}

print(f"\n{'Spec':<35s} {'Y':>10s} {'Coef':>8s} {'P-val':>8s} {'N':>6s} {'R²':>6s}"
      f" | {'Pub Coef':>9s} {'Pub P':>8s} {'Pub N':>6s}")
print("-" * 110)

for label, filt, xvar, dep_vars in table4_specs:
    sub = filt(df_did) if filt else df_did
    for yvar in dep_vars:
        res = run_areg_robust(sub, yvar, xvar)
        pub = pub_t4.get((label, yvar), (None, None, None, None))
        s = stars(res['pval'])
        print(f"  {label:<33s} {yvar:>10s} {res['coef']:>7.2f}{s:<3s} ({res['pval']:.3f}) "
              f"{res['N']:>6d} {res['R2']:>5.2f}"
              f"  | {pub[0]:>8.2f}  ({pub[1]:.3f}) {pub[2]:>6d}")

# Table 5 - WB entry costs
print("\n" + "=" * 70)
print("TABLE 5: Effect on turnover of ISTC × WB entry costs")
print("=" * 70)

table5_specs = [
    ('All industries, 1987-97', None, 'etcwb87st', ['turnover', 'entry', 'exit']),
    ('All industries, 1947-2000', None, 'etcwb00', ['turnover', 'entry', 'exit']),
    ('Manufacturing only, 1987-97', None, 'etcwb87st', ['turnover']),
    ('Non-manuf only, 1987-97', None, 'etcwb87st', ['turnover']),
]

pub_t5 = {
    ('All industries, 1987-97', 'turnover'):     (-0.70, 0.002, 719, 0.63),
    ('All industries, 1987-97', 'entry'):         (-0.30, 0.058, 724, 0.62),
    ('All industries, 1987-97', 'exit'):          (-0.42, 0.000, 721, 0.49),
    ('All industries, 1947-2000', 'turnover'):    (-0.64, 0.003, 719, 0.63),
    ('All industries, 1947-2000', 'entry'):       (-0.28, 0.052, 724, 0.62),
    ('All industries, 1947-2000', 'exit'):        (-0.38, 0.000, 721, 0.49),
    ('Manufacturing only, 1987-97', 'turnover'):  (-0.70, 0.074, 283, 0.68),
    ('Non-manuf only, 1987-97', 'turnover'):      (-0.71, 0.017, 436, 0.58),
}

print(f"\n{'Spec':<35s} {'Y':>10s} {'Coef':>8s} {'P-val':>8s} {'N':>6s} {'R²':>6s}"
      f" | {'Pub Coef':>9s} {'Pub P':>8s} {'Pub N':>6s}")
print("-" * 110)

for label, filt, xvar, dep_vars in table5_specs:
    sub = df_did.copy()
    if 'Manufacturing' in label:
        sub = sub[sub['manuf'] == 1]
    elif 'Non-manuf' in label:
        sub = sub[sub['manuf'] == 0]
    for yvar in dep_vars:
        res = run_areg_robust(sub, yvar, xvar)
        pub = pub_t5.get((label, yvar), (None, None, None, None))
        s = stars(res['pval'])
        print(f"  {label:<33s} {yvar:>10s} {res['coef']:>7.2f}{s:<3s} ({res['pval']:.3f}) "
              f"{res['N']:>6d} {res['R2']:>5.2f}"
              f"  | {pub[0]:>8.2f}  ({pub[1]:.3f}) {pub[2]:>6d}")


# ================================================================
# Also try areg with cluster(i) approach from README
# ================================================================
print("\n" + "=" * 70)
print("ALTERNATIVE: areg y x, absorb(c) cluster(i)")
print("(per README command - country FE only, clustered by industry)")
print("=" * 70)

for xvar_label, xvar in [('DLLS 87-97 st', 'etcdlls87st'), ('WB 87-97 st', 'etcwb87st')]:
    for yvar in ['turnover', 'entry', 'exit']:
        res = run_areg_cluster(df_did, yvar, xvar)
        s = stars(res['pval'])
        print(f"  {xvar_label:<15s} {yvar:>10s}: coef={res['coef']:>7.3f}{s:<3s} "
              f"(p={res['pval']:.3f}) N={res['N']} R²={res['R2']:.2f}")

# ================================================================
# TABLE 11: Correlations between lumpiness, turnover, ISTC, TFP
# ================================================================
print("\n" + "=" * 70)
print("TABLE 11: Correlations between lumpiness, turnover, and ISTC")
print("(Using industry-level data)")
print("=" * 70)

# Use industry-level data: lumpy, turnover (from indices), istc
# Paper also uses TFP from Jorgenson et al - not available in data
corr11_data = pd.DataFrame({
    'turnover': table1.set_index('i')['turnover'],
    'istc': table1.set_index('i')['istc'],
    'lumpy': df_ind.set_index('i')['lumpy'],
    'lump500': df_ind.set_index('i')['lump1a500'],
    'lump250': df_ind.set_index('i')['lump1a250'],
})

pub_t11 = {
    ('lumpy', 'turnover', 'All'):  (0.43, 0.002),
    ('lumpy', 'istc', 'All'):      (0.40, 0.005),
    ('lump500', 'turnover', '≤500'): (0.30, 0.028),
    ('lump500', 'istc', '≤500'):   (0.36, 0.007),
    ('lump250', 'turnover', '≤250'): (0.25, 0.100),
    ('lump250', 'istc', '≤250'):   (0.32, 0.041),
}

for size_label, lumpy_var in [('All', 'lumpy'), ('≤500', 'lump500'), ('≤250', 'lump250')]:
    print(f"\n  Size: {size_label}")
    for other_var in ['turnover', 'istc']:
        valid = corr11_data[[lumpy_var, other_var]].dropna()
        r, p = stats.pearsonr(valid[lumpy_var], valid[other_var])
        pub_key = (lumpy_var if size_label == 'All' else lumpy_var, other_var, size_label)
        pub = pub_t11.get(pub_key, (None, None))
        s = stars(p)
        print(f"    Lumpy × {other_var:<10s}: r={r:.2f}{s:<3s} (p={p:.3f})"
              f"  | Pub: r={pub[0]}, p={pub[1]}")

print("\n" + "=" * 70)
print("REPLICATION COMPLETE")
print("=" * 70)
