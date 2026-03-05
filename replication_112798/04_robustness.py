"""
Robustness checks for Bednar & Gicheva (2014) replication.
"Are Female Supervisors More Female-Friendly?"

The main finding: AD gender (female_ad) is NOT related to female-friendliness FE.
We test whether this null result is robust.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from utils import (load_main_data, load_eada_data, get_ad_dum_cols,
                   get_ad_dum_all_cols, get_year_dum_cols, get_school_dummies,
                   print_separator, OUTPUT_DIR)
import os


def drop_collinear(X):
    """Drop perfectly collinear columns from design matrix."""
    _, R = np.linalg.qr(X.values.astype(float))
    independent = np.abs(np.diag(R)) > 1e-10
    return X[X.columns[independent]]


def prepare_data():
    """Run the base regressions and prepare data for robustness checks."""
    df = load_main_data()
    y = df['fsoc2'].values.astype(float)
    year_cols = get_year_dum_cols(df)
    school_dums = get_school_dummies(df)
    ad_cols = get_ad_dum_cols(df)
    ad_all_cols = get_ad_dum_all_cols(df)

    # Model with all 433 ADs
    X3 = pd.concat([df[ad_all_cols].astype(float), df[year_cols].astype(float), school_dums], axis=1)
    X3 = sm.add_constant(X3)
    X3 = drop_collinear(X3)
    model3 = sm.OLS(y, X3).fit()

    fe_all = np.zeros(len(df))
    fe_all_se = np.zeros(len(df))
    for col in ad_all_cols:
        if col in model3.params.index:
            fe_all += model3.params[col] * df[col].values.astype(float)
            fe_all_se += model3.bse[col] * df[col].values.astype(float)
    fe_all[fe_all == 0] = np.nan
    fe_all_se[fe_all_se == 0] = np.nan
    df['fe_all'] = fe_all
    df['fe_all_se'] = fe_all_se
    df['weight_all'] = 1 / fe_all_se**2

    mask = df['ADunique'] == 1
    mean_fa = df.loc[mask, 'fe_all'].mean()
    df['friendly_all0'] = df['fe_all'] - mean_fa

    # Also model with 137 multi-school ADs
    X2 = pd.concat([df[ad_cols].astype(float), df[year_cols].astype(float), school_dums], axis=1)
    X2 = sm.add_constant(X2)
    X2 = drop_collinear(X2)
    model2 = sm.OLS(y, X2).fit()

    fem_friendly = np.zeros(len(df))
    fem_fr_se = np.zeros(len(df))
    for col in ad_cols:
        if col in model2.params.index:
            fem_friendly += model2.params[col] * df[col].values.astype(float)
            fem_fr_se += model2.bse[col] * df[col].values.astype(float)
    fem_friendly[fem_friendly == 0] = np.nan
    fem_fr_se[fem_fr_se == 0] = np.nan
    df['fem_friendly'] = fem_friendly
    df['fem_fr_se'] = fem_fr_se
    df['weight'] = 1 / fem_fr_se**2

    mean_ff = df.loc[mask, 'fem_friendly'].mean()
    df['friendly0'] = df['fem_friendly'] - mean_ff

    return df


def check1_unweighted_ols(df):
    """Check 1: Unweighted OLS (no precision weighting)."""
    print_separator("CHECK 1: Unweighted OLS")
    mask = df['ADunique'] == 1

    # All ADs
    sub = df.loc[mask].dropna(subset=['friendly_all0']).copy()
    X = sm.add_constant(sub[['female_ad']].astype(float))
    y = sub['friendly_all0'].values.astype(float)
    m = sm.OLS(y, X).fit()
    print(f"All ADs (unweighted): coef={m.params['female_ad']:.4f}, se={m.bse['female_ad']:.4f}, p={m.pvalues['female_ad']:.4f}, N={int(m.nobs)}")

    # Multi-school
    sub2 = df.loc[mask].dropna(subset=['friendly0']).copy()
    X2 = sm.add_constant(sub2[['female_ad']].astype(float))
    y2 = sub2['friendly0'].values.astype(float)
    m2 = sm.OLS(y2, X2).fit()
    print(f"Multi-school (unweighted): coef={m2.params['female_ad']:.4f}, se={m2.bse['female_ad']:.4f}, p={m2.pvalues['female_ad']:.4f}, N={int(m2.nobs)}")


def check2_robust_se(df):
    """Check 2: HC1 robust standard errors."""
    print_separator("CHECK 2: Robust Standard Errors (HC1)")
    mask = df['ADunique'] == 1

    sub = df.loc[mask].dropna(subset=['friendly_all0', 'weight_all']).copy()
    X = sm.add_constant(sub[['female_ad']].astype(float))
    y = sub['friendly_all0'].values.astype(float)
    w = sub['weight_all'].values.astype(float)
    m = sm.WLS(y, X, weights=w).fit(cov_type='HC1')
    print(f"All ADs (WLS+HC1): coef={m.params['female_ad']:.4f}, se={m.bse['female_ad']:.4f}, p={m.pvalues['female_ad']:.4f}")

    sub2 = df.loc[mask].dropna(subset=['friendly0', 'weight']).copy()
    X2 = sm.add_constant(sub2[['female_ad']].astype(float))
    y2 = sub2['friendly0'].values.astype(float)
    w2 = sub2['weight'].values.astype(float)
    m2 = sm.WLS(y2, X2, weights=w2).fit(cov_type='HC1')
    print(f"Multi-school (WLS+HC1): coef={m2.params['female_ad']:.4f}, se={m2.bse['female_ad']:.4f}, p={m2.pvalues['female_ad']:.4f}")


def check3_drop_interim(df):
    """Check 3: Drop interim ADs from FE estimation."""
    print_separator("CHECK 3: Exclude Interim ADs")
    df_no_interim = df[df['interim'] == 0].copy().reset_index(drop=True)
    print(f"Dropping {(df['interim']==1).sum()} interim observations, keeping {len(df_no_interim)}")

    # Re-estimate with all AD dummies on non-interim sample
    y = df_no_interim['fsoc2'].values.astype(float)
    year_cols = get_year_dum_cols(df_no_interim)
    school_dums = get_school_dummies(df_no_interim)
    ad_all_cols = get_ad_dum_all_cols(df_no_interim)

    # Only keep dummies with variation
    active_cols = [c for c in ad_all_cols if df_no_interim[c].sum() > 0]
    X = pd.concat([df_no_interim[active_cols].astype(float),
                    df_no_interim[year_cols].astype(float), school_dums], axis=1)
    X = sm.add_constant(X)
    X = drop_collinear(X)
    model = sm.OLS(y, X).fit()

    fe_all = np.zeros(len(df_no_interim))
    for col in active_cols:
        if col in model.params.index:
            fe_all += model.params[col] * df_no_interim[col].values.astype(float)
    fe_all[fe_all == 0] = np.nan
    df_no_interim['fe_all'] = fe_all

    mask = df_no_interim['ADunique'] == 1
    mean_fa = df_no_interim.loc[mask, 'fe_all'].dropna().mean()
    df_no_interim['friendly_all0'] = df_no_interim['fe_all'] - mean_fa

    sub = df_no_interim.loc[mask].dropna(subset=['friendly_all0']).copy()
    X_reg = sm.add_constant(sub[['female_ad']].astype(float))
    y_reg = sub['friendly_all0'].values.astype(float)
    m = sm.OLS(y_reg, X_reg).fit()
    print(f"All ADs (no interim, unweighted): coef={m.params['female_ad']:.4f}, se={m.bse['female_ad']:.4f}, p={m.pvalues['female_ad']:.4f}, N={int(m.nobs)}")


def check4_permutation_test(df):
    """Check 4: Permutation test - shuffle female_ad and compute p-value."""
    print_separator("CHECK 4: Permutation Test")
    mask = df['ADunique'] == 1

    sub = df.loc[mask].dropna(subset=['friendly_all0', 'weight_all']).copy()
    X = sm.add_constant(sub[['female_ad']].astype(float))
    y = sub['friendly_all0'].values.astype(float)
    w = sub['weight_all'].values.astype(float)

    # Actual coefficient
    m_actual = sm.WLS(y, X, weights=w).fit()
    actual_coef = m_actual.params['female_ad']
    print(f"Actual coefficient: {actual_coef:.4f}")

    # Permutation
    np.random.seed(42)
    n_perms = 1000
    perm_coefs = []
    female_ad_vals = sub['female_ad'].values.copy()
    for _ in range(n_perms):
        np.random.shuffle(female_ad_vals)
        X_perm = sm.add_constant(pd.DataFrame({'female_ad': female_ad_vals}))
        m_perm = sm.WLS(y, X_perm, weights=w).fit()
        perm_coefs.append(m_perm.params['female_ad'])

    perm_coefs = np.array(perm_coefs)
    p_two_sided = (np.abs(perm_coefs) >= np.abs(actual_coef)).mean()
    print(f"Permutation p-value (two-sided, {n_perms} permutations): {p_two_sided:.4f}")
    print(f"Permutation distribution: mean={perm_coefs.mean():.4f}, sd={perm_coefs.std():.4f}")


def check5_winsorize_fe(df):
    """Check 5: Winsorize extreme FE values at 5th/95th percentile."""
    print_separator("CHECK 5: Winsorize FEs at 5/95 Percentiles")
    mask = df['ADunique'] == 1

    sub = df.loc[mask].dropna(subset=['friendly_all0', 'weight_all']).copy()

    p5 = sub['friendly_all0'].quantile(0.05)
    p95 = sub['friendly_all0'].quantile(0.95)
    sub['friendly_wins'] = sub['friendly_all0'].clip(p5, p95)

    X = sm.add_constant(sub[['female_ad']].astype(float))
    y = sub['friendly_wins'].values.astype(float)
    w = sub['weight_all'].values.astype(float)
    m = sm.WLS(y, X, weights=w).fit()
    print(f"Winsorized (5/95): coef={m.params['female_ad']:.4f}, se={m.bse['female_ad']:.4f}, p={m.pvalues['female_ad']:.4f}")

    # Also try trimming
    sub_trim = sub[(sub['friendly_all0'] >= p5) & (sub['friendly_all0'] <= p95)].copy()
    X_t = sm.add_constant(sub_trim[['female_ad']].astype(float))
    y_t = sub_trim['friendly_wins'].values.astype(float)
    w_t = sub_trim['weight_all'].values.astype(float)
    m_t = sm.WLS(y_t, X_t, weights=w_t).fit()
    print(f"Trimmed (5/95): coef={m_t.params['female_ad']:.4f}, se={m_t.bse['female_ad']:.4f}, p={m_t.pvalues['female_ad']:.4f}, N={int(m_t.nobs)}")


def check6_alternative_outcome(df):
    """Check 6: Use number of female coaches instead of share."""
    print_separator("CHECK 6: Alternative Outcome (fsoc2 as-is, check mean by AD gender)")
    mask = df['ADunique'] == 1

    # Simple comparison of mean fsoc2 by AD gender
    sub = df.loc[mask].copy()
    male_mean = sub.loc[sub['female_ad']==0, 'fsoc2'].mean()
    female_mean = sub.loc[sub['female_ad']==1, 'fsoc2'].mean()
    t_stat, p_val = stats.ttest_ind(
        sub.loc[sub['female_ad']==0, 'fsoc2'].dropna(),
        sub.loc[sub['female_ad']==1, 'fsoc2'].dropna()
    )
    print(f"Mean fsoc2 under male AD: {male_mean:.4f}")
    print(f"Mean fsoc2 under female AD: {female_mean:.4f}")
    print(f"t-test: t={t_stat:.4f}, p={p_val:.4f}")


def check7_subgroup_by_school_size(df):
    """Check 7: Heterogeneity by school panel length."""
    print_separator("CHECK 7: Heterogeneity by School Panel Length")
    mask = df['ADunique'] == 1

    # Split schools by median panel length
    school_years = df.groupby('school_id').size()
    median_len = school_years.median()
    long_schools = school_years[school_years >= median_len].index
    short_schools = school_years[school_years < median_len].index

    for label, schools in [("Long panel", long_schools), ("Short panel", short_schools)]:
        sub = df.loc[mask & df['school_id'].isin(schools)].dropna(subset=['friendly_all0', 'weight_all']).copy()
        if len(sub) < 10:
            print(f"{label}: Too few observations ({len(sub)})")
            continue
        X = sm.add_constant(sub[['female_ad']].astype(float))
        y = sub['friendly_all0'].values.astype(float)
        w = sub['weight_all'].values.astype(float)
        m = sm.WLS(y, X, weights=w).fit()
        print(f"{label} (N={int(m.nobs)}): coef={m.params['female_ad']:.4f}, se={m.bse['female_ad']:.4f}, p={m.pvalues['female_ad']:.4f}")


def check8_leave_one_sport_out(df):
    """Check 8: Since fsoc2 is share across 4 sports, check if result holds concept-wise.
    We can't decompose fsoc2 further without raw data, but we can check sensitivity
    to extreme values of the FE."""
    print_separator("CHECK 8: Sensitivity to Extreme FE Values")
    mask = df['ADunique'] == 1
    sub = df.loc[mask].dropna(subset=['friendly_all0', 'weight_all']).copy()

    # Drop top and bottom 10% of FE values
    for pct in [5, 10, 15]:
        low = sub['friendly_all0'].quantile(pct/100)
        high = sub['friendly_all0'].quantile(1 - pct/100)
        trimmed = sub[(sub['friendly_all0'] >= low) & (sub['friendly_all0'] <= high)].copy()
        X = sm.add_constant(trimmed[['female_ad']].astype(float))
        y = trimmed['friendly_all0'].values.astype(float)
        w = trimmed['weight_all'].values.astype(float)
        m = sm.WLS(y, X, weights=w).fit()
        print(f"Drop {pct}% tails (N={int(m.nobs)}): coef={m.params['female_ad']:.4f}, se={m.bse['female_ad']:.4f}, p={m.pvalues['female_ad']:.4f}")


def check9_expenditure_robust_se(df):
    """Check 9: Table 3 expenditure regressions with robust SEs."""
    print_separator("CHECK 9: Expenditure Regressions with HC1 SEs")
    eada = load_eada_data()
    year_cols = [c for c in eada.columns if c.startswith('year_dum')]

    sub = eada.dropna(subset=['exp_rev_w', 'friendly_all', 'female_ad', 'weight_all']).copy()
    td = pd.get_dummies(sub['team'], prefix='team', drop_first=True).astype(float)
    X = pd.concat([sub[['friendly_all', 'female_ad'] + year_cols].astype(float).reset_index(drop=True),
                    td.reset_index(drop=True)], axis=1)
    X = sm.add_constant(X)
    y = sub['exp_rev_w'].values.astype(float)
    w = sub['weight_all'].values.astype(float)

    m_default = sm.WLS(y, X, weights=w).fit()
    m_hc1 = sm.WLS(y, X, weights=w).fit(cov_type='HC1')

    print(f"Women's exp ~ friendly_all + female_ad:")
    print(f"  friendly_all: coef={m_default.params['friendly_all']:.4f}, default_se={m_default.bse['friendly_all']:.4f}, HC1_se={m_hc1.bse['friendly_all']:.4f}")
    print(f"  female_ad: coef={m_default.params['female_ad']:.4f}, default_se={m_default.bse['female_ad']:.4f}, HC1_se={m_hc1.bse['female_ad']:.4f}")


def check10_shrinkage_sensitivity(df):
    """Check 10: Sensitivity of Table 2 results to shrinkage approach."""
    print_separator("CHECK 10: Shrinkage Sensitivity")
    mask = df['ADunique'] == 1

    sub = df.loc[mask].dropna(subset=['friendly_all0', 'fe_all_se', 'weight_all']).copy()

    # No shrinkage (original Table 2 result)
    X = sm.add_constant(sub[['female_ad']].astype(float))
    y = sub['friendly_all0'].values.astype(float)
    w = sub['weight_all'].values.astype(float)
    m_noshrink = sm.WLS(y, X, weights=w).fit()
    print(f"No shrinkage (Table 2): coef={m_noshrink.params['female_ad']:.4f}, se={m_noshrink.bse['female_ad']:.4f}, p={m_noshrink.pvalues['female_ad']:.4f}")

    # EB shrinkage
    var_ols_all = sub['friendly_all0'].var(ddof=1)
    avg_var_all = (sub['fe_all_se']**2).mean()
    var_fe_all = var_ols_all - avg_var_all
    sub['friendly_shrunk'] = sub['friendly_all0'] * var_fe_all / (var_fe_all + sub['fe_all_se']**2)

    y_s = sub['friendly_shrunk'].values.astype(float)
    m_shrink = sm.WLS(y_s, X, weights=w).fit()
    print(f"EB shrinkage: coef={m_shrink.params['female_ad']:.4f}, se={m_shrink.bse['female_ad']:.4f}, p={m_shrink.pvalues['female_ad']:.4f}")

    # Alternative: 50% shrinkage
    sub['friendly_half'] = sub['friendly_all0'] * 0.5
    y_h = sub['friendly_half'].values.astype(float)
    m_half = sm.WLS(y_h, X, weights=w).fit()
    print(f"50% shrinkage: coef={m_half.params['female_ad']:.4f}, se={m_half.bse['female_ad']:.4f}, p={m_half.pvalues['female_ad']:.4f}")


if __name__ == '__main__':
    print_separator("ROBUSTNESS CHECKS")
    df = prepare_data()

    # Store fe_all_se for check10
    check1_unweighted_ols(df)
    check2_robust_se(df)
    check3_drop_interim(df)
    check4_permutation_test(df)
    check5_winsorize_fe(df)
    check6_alternative_outcome(df)
    check7_subgroup_by_school_size(df)
    check8_leave_one_sport_out(df)
    check9_expenditure_robust_se(df)
    check10_shrinkage_sensitivity(df)

    print_separator("ROBUSTNESS CHECKS COMPLETE")
