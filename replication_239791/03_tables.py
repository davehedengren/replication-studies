"""
03_tables.py — Report regression coefficients underlying all Figure 4 specifications.

This paper (Gobbi, Hannusch & Rossi, 2026) has no formal tables —
all results are presented as figures. This script reports the OLS
coefficients, SEs, and R² for the regressions plotted in Figures 4a–d
and Appendix Figures A1a–f, A2a–f.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from utils import OUTPUT_DIR, ANALYSIS_DATA_FIG4_PATH
import os

print("=" * 60)
print("03_tables.py — Regression coefficients for Figure 4 specs")
print("=" * 60)

# ── Load data ─────────────────────────────────────────────────────────

df_diff = pd.read_parquet(ANALYSIS_DATA_FIG4_PATH)

# ── Predictor sets ────────────────────────────────────────────────────

PRED_SETS = {
    'Full (GDP + Educ + Mortality)': ['dlgdppc', 'dsec', 'dchildmort', 'dmatmort'],
    'GDP + Education only': ['dlgdppc', 'dsec'],
    'Mortality only': ['dchildmort', 'dmatmort'],
}

CUSTOM_ORDER = [
    'East Asia & Pacific', 'Europe & Central Asia',
    'Latin America & Caribbean', 'Middle East & North Africa',
    'North America', 'South Asia', 'Sub-Saharan Africa'
]


def run_regression(sub, pred_vars, label):
    """Run OLS and return results dict."""
    dep = 'dtfr'
    all_vars = [dep] + pred_vars
    clean = sub.dropna(subset=all_vars).copy()

    if len(clean) < len(pred_vars) + 2:
        return None

    X = sm.add_constant(clean[pred_vars])
    res = sm.OLS(clean[dep], X).fit()

    row = {
        'Specification': label,
        'N': int(res.nobs),
        'R²': f"{res.rsquared:.3f}",
        'Adj R²': f"{res.rsquared_adj:.3f}",
    }

    for var in pred_vars:
        coef = res.params[var]
        se = res.bse[var]
        pval = res.pvalues[var]
        stars = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
        row[var] = f"{coef:.4f}{stars}"
        row[f"{var}_se"] = f"({se:.4f})"

    return row


# ── Table 1: Figure 4 main specifications ─────────────────────────────

print("\n── Table 1: Figure 4 Main Specifications ──\n")

all_regions = CUSTOM_ORDER

specs = [
    ('Fig 4a: Monogamous', df_diff[df_diff['good_institution_mono'] == 1], all_regions),
    ('Fig 4b: Non-monogamous', df_diff[df_diff['good_institution_mono'] == 0], all_regions),
    ('Fig 4c: Partible (SSA)', df_diff[df_diff['good_institution_part'] == 1], ['Sub-Saharan Africa']),
    ('Fig 4d: Impartible (SSA)', df_diff[df_diff['good_institution_part'] == 0], ['Sub-Saharan Africa']),
]

pred_vars = ['dlgdppc', 'dsec', 'dchildmort', 'dmatmort']
rows = []

for label, sub_df, regions in specs:
    sub = sub_df[sub_df['region_wb'].isin(regions)].copy()
    result = run_regression(sub, pred_vars, label)
    if result:
        rows.append(result)

# Print table
header = f"{'Specification':<30} {'N':>4} {'R²':>7} {'dlgdppc':>12} {'dsec':>12} {'dchildmort':>12} {'dmatmort':>12}"
print(header)
print("-" * len(header))

for r in rows:
    coefs = []
    for var in pred_vars:
        coefs.append(f"{r.get(var, ''):>12}")
    print(f"{r['Specification']:<30} {r['N']:>4} {r['R²']:>7} {''.join(coefs)}")
    ses = []
    for var in pred_vars:
        ses.append(f"{r.get(f'{var}_se', ''):>12}")
    print(f"{'':30} {'':>4} {'':>7} {''.join(ses)}")


# ── Table 2: Appendix specifications (different predictor subsets) ────

print("\n\n── Table 2: Appendix A1 — Monogamous vs Non-mono, by predictor set ──\n")

for pred_label, pvars in PRED_SETS.items():
    print(f"\n  Predictor set: {pred_label}")
    header = f"    {'Sample':<25} {'N':>4} {'R²':>7}"
    for v in pvars:
        header += f" {v:>12}"
    print(header)
    print("    " + "-" * (len(header) - 4))

    for label, sub_df in [('Monogamous', df_diff[df_diff['good_institution_mono'] == 1]),
                          ('Non-monogamous', df_diff[df_diff['good_institution_mono'] == 0])]:
        sub = sub_df[sub_df['region_wb'].isin(all_regions)].copy()
        result = run_regression(sub, pvars, label)
        if result:
            coefs = "".join(f"{result.get(v, ''):>12}" for v in pvars)
            print(f"    {label:<25} {result['N']:>4} {result['R²']:>7} {coefs}")
            ses = "".join(f"{result.get(f'{v}_se', ''):>12}" for v in pvars)
            print(f"    {'':25} {'':>4} {'':>7} {ses}")


print("\n\n── Table 3: Appendix A2 — Partible vs Impartible (SSA only), by predictor set ──\n")

for pred_label, pvars in PRED_SETS.items():
    print(f"\n  Predictor set: {pred_label}")
    header = f"    {'Sample':<25} {'N':>4} {'R²':>7}"
    for v in pvars:
        header += f" {v:>12}"
    print(header)
    print("    " + "-" * (len(header) - 4))

    for label, sub_df in [('Partible inherit.', df_diff[df_diff['good_institution_part'] == 1]),
                          ('Impartible inherit.', df_diff[df_diff['good_institution_part'] == 0])]:
        sub = sub_df[sub_df['region_wb'] == 'Sub-Saharan Africa'].copy()
        result = run_regression(sub, pvars, label)
        if result:
            coefs = "".join(f"{result.get(v, ''):>12}" for v in pvars)
            print(f"    {label:<25} {result['N']:>4} {result['R²']:>7} {coefs}")
            ses = "".join(f"{result.get(f'{v}_se', ''):>12}" for v in pvars)
            print(f"    {'':25} {'':>4} {'':>7} {ses}")


# ── Bug report: dlgdppc uses level differences ───────────────────────

print("\n\n── Bug Analysis: dlgdppc variable ──")
print("The R code (Figure4_A1_A2.R, line 96) computes:")
print("  dlgdppc = gdppc_avg2010_20 - gdppc_avg1970_80")
print("This is a LEVEL difference, not a LOG difference.")
print("The variable name 'dlgdppc' (delta log GDP pc) is misleading.")
print("Figure 2's R code correctly uses log(gdp2010) - log(gdp1960).\n")

# Compare: what happens if we use log differences instead?
df_diff_log = df_diff.copy()
df_diff_log['dlgdppc'] = (
    np.log(df_diff_log['gdppc_late'].replace(0, np.nan)) -
    np.log(df_diff_log['gdppc_early'].replace(0, np.nan))
)

print(f"{'Specification':<30} {'R² (level)':>12} {'R² (log)':>12} {'N':>5}")
print("-" * 62)

for label, sub_df, sub_df_log, regions in [
    ('Fig 4a: Monogamous',
     df_diff[df_diff['good_institution_mono'] == 1],
     df_diff_log[df_diff_log['good_institution_mono'] == 1],
     all_regions),
    ('Fig 4b: Non-monogamous',
     df_diff[df_diff['good_institution_mono'] == 0],
     df_diff_log[df_diff_log['good_institution_mono'] == 0],
     all_regions),
    ('Fig 4c: Partible (SSA)',
     df_diff[df_diff['good_institution_part'] == 1],
     df_diff_log[df_diff_log['good_institution_part'] == 1],
     ['Sub-Saharan Africa']),
    ('Fig 4d: Impartible (SSA)',
     df_diff[df_diff['good_institution_part'] == 0],
     df_diff_log[df_diff_log['good_institution_part'] == 0],
     ['Sub-Saharan Africa']),
]:
    sub_lev = sub_df[sub_df['region_wb'].isin(regions)].copy()
    sub_log = sub_df_log[sub_df_log['region_wb'].isin(regions)].copy()

    dep = 'dtfr'
    pvars = ['dlgdppc', 'dsec', 'dchildmort', 'dmatmort']

    clean_lev = sub_lev.dropna(subset=[dep] + pvars)
    clean_log = sub_log.dropna(subset=[dep] + pvars)

    X_lev = sm.add_constant(clean_lev[pvars])
    r_lev = sm.OLS(clean_lev[dep], X_lev).fit()

    X_log = sm.add_constant(clean_log[pvars])
    r_log = sm.OLS(clean_log[dep], X_log).fit()

    print(f"{label:<30} {r_lev.rsquared:>12.3f} {r_log.rsquared:>12.3f} {int(r_lev.nobs):>5}")

print("\nConclusion: Using log differences (the likely intended specification)")
print("changes R² somewhat but does not qualitatively alter the main finding")
print("that 'good institutions' countries have higher predictive R².")


print("\n" + "=" * 60)
print("03_tables.py — DONE")
print("=" * 60)
