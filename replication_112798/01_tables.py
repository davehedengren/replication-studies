"""
Replicate Tables 1, 2, and 3 from Bednar & Gicheva (2014).
"Are Female Supervisors More Female-Friendly?"

Table 1: FE regressions of female coach share on AD dummies + year + school FE
Table 2: Regress normalized AD FE on AD gender (OLS + quantile)
Table 3: Expenditure regressions with shrunk FE measures
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from utils import (load_main_data, load_eada_data, get_ad_dum_cols,
                   get_ad_dum_all_cols, get_year_dum_cols, get_school_dummies,
                   print_separator, NUM_ADS, NUM_ADS_ALL, OUTPUT_DIR)
import os

np.set_printoptions(precision=6)
pd.set_option('display.float_format', '{:.4f}'.format)


def drop_collinear(X):
    """Drop perfectly collinear columns from design matrix."""
    # Use numpy rank to find independent columns
    _, R = np.linalg.qr(X.values.astype(float))
    independent = np.abs(np.diag(R)) > 1e-10
    kept_cols = X.columns[independent]
    dropped = X.columns[~independent]
    if len(dropped) > 0:
        print(f"  Dropped {len(dropped)} collinear columns")
    return X[kept_cols], kept_cols, dropped


def run_table1(df):
    """
    Table 1: Female Coach Share Regressions

    Three models:
    (1) fsoc2 ~ year_dummies + school_FE  (no AD effects)
    (2) fsoc2 ~ ad_dum1-137 + year_dummies + school_FE  (137 multi-school ADs)
    (3) fsoc2 ~ ad_dum_all1-433 + year_dummies + school_FE  (all 433 ADs)
    """
    print_separator("TABLE 1: Female Coach Share Regressions")

    y = df['fsoc2'].values.astype(float)
    year_cols = get_year_dum_cols(df)
    school_dums = get_school_dummies(df)
    ad_cols = get_ad_dum_cols(df)
    ad_all_cols = get_ad_dum_all_cols(df)

    # Model 1: No AD effects
    print("Model 1: No AD effects...")
    X1 = pd.concat([df[year_cols].astype(float), school_dums], axis=1)
    X1 = sm.add_constant(X1)
    X1, kept1, _ = drop_collinear(X1)
    model1 = sm.OLS(y, X1).fit()

    # Model 2: Multi-school AD dummies (137)
    print("Model 2: 137 multi-school AD dummies...")
    X2 = pd.concat([df[ad_cols].astype(float), df[year_cols].astype(float), school_dums], axis=1)
    X2 = sm.add_constant(X2)
    X2, kept2, dropped2 = drop_collinear(X2)
    model2 = sm.OLS(y, X2).fit()

    # F-test for AD dummies in model 2
    ad_in_model2 = [c for c in ad_cols if c in kept2]
    r_matrix2 = np.zeros((len(ad_in_model2), X2.shape[1]))
    for i, col in enumerate(ad_in_model2):
        col_idx = list(X2.columns).index(col)
        r_matrix2[i, col_idx] = 1
    f_test2 = model2.f_test(r_matrix2)

    # Model 3: All AD dummies (433)
    print("Model 3: 433 all AD dummies...")
    X3 = pd.concat([df[ad_all_cols].astype(float), df[year_cols].astype(float), school_dums], axis=1)
    X3 = sm.add_constant(X3)
    X3, kept3, dropped3 = drop_collinear(X3)
    model3 = sm.OLS(y, X3).fit()

    # F-test for AD dummies in model 3
    ad_in_model3 = [c for c in ad_all_cols if c in kept3]
    r_matrix3 = np.zeros((len(ad_in_model3), X3.shape[1]))
    for i, col in enumerate(ad_in_model3):
        col_idx = list(X3.columns).index(col)
        r_matrix3[i, col_idx] = 1
    f_test3 = model3.f_test(r_matrix3)

    # Extract AD fixed effects for model 2 (137 multi-school ADs)
    # For each obs, FE = sum of (coef * dummy) for all AD dummies the obs belongs to
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

    # Extract AD fixed effects for model 3 (all 433 ADs)
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

    # Print results
    print(f"\n{'':40s} (1)          (2)          (3)")
    print(f"{'':40s} No AD FE   137 ADs FE   433 ADs FE")
    print("-" * 75)
    print(f"{'Constant':40s} {model1.params['const']:10.4f}   {model2.params['const']:10.4f}   {model3.params['const']:10.4f}")
    print(f"{'':40s} ({model1.bse['const']:10.4f})  ({model2.bse['const']:10.4f})  ({model3.bse['const']:10.4f})")
    print(f"{'N':40s} {model1.nobs:10.0f}   {model2.nobs:10.0f}   {model3.nobs:10.0f}")
    print(f"{'Adj. R-sq':40s} {model1.rsquared_adj:10.4f}   {model2.rsquared_adj:10.4f}   {model3.rsquared_adj:10.4f}")
    print(f"{'F stat (AD)':40s}              {float(f_test2.fvalue):10.4f}   {float(f_test3.fvalue):10.4f}")
    print(f"{'p-value (AD)':40s}              {float(f_test2.pvalue):10.6f}   {float(f_test3.pvalue):10.6f}")
    print(f"{'AD dummies in model':40s}              {len(ad_in_model2):10d}   {len(ad_in_model3):10d}")

    # Count non-NaN FEs
    mask = df['ADunique'] == 1
    n_ff = df.loc[mask, 'fem_friendly'].notna().sum()
    n_fa = df.loc[mask, 'fe_all'].notna().sum()
    print(f"\n  ADs with estimated FE (multi-school): {n_ff}")
    print(f"  ADs with estimated FE (all): {n_fa}")

    return df, model2, model3


def normalize_fes(df):
    """Normalize the FEs to sum to 0 (subtract mean among unique ADs)."""
    mask = df['ADunique'] == 1

    mean_ff = df.loc[mask, 'fem_friendly'].mean()
    df['friendly0'] = df['fem_friendly'] - mean_ff
    var_ols = df.loc[mask, 'fem_friendly'].var(ddof=1)  # Stata r(Var) uses N-1
    df['var_ols'] = var_ols

    mean_fa = df.loc[mask, 'fe_all'].mean()
    df['friendly_all0'] = df['fe_all'] - mean_fa
    var_ols_all = df.loc[mask, 'fe_all'].var(ddof=1)  # Stata r(Var) uses N-1
    df['var_ols_all'] = var_ols_all

    # Correlation between the two FE measures
    unique_df = df.loc[mask].dropna(subset=['friendly0', 'friendly_all0'])
    corr = unique_df['friendly0'].corr(unique_df['friendly_all0'])
    print(f"\nCorrelation between friendly0 and friendly_all0 (unique ADs): {corr:.4f}")
    print(f"  var_ols (multi-school): {var_ols:.6f}")
    print(f"  var_ols_all (all): {var_ols_all:.6f}")

    return df


def run_table2(df):
    """
    Table 2: Is Female-Friendliness Related to AD Gender?

    Panel A: 137 multi-school ADs
    Panel B: All 433 ADs

    Each panel: OLS, Q25, Q50, Q75 of normalized FE on female_ad
    Weighted by inverse variance
    """
    print_separator("TABLE 2: Female-Friendliness and AD Gender")

    mask = df['ADunique'] == 1

    # Panel A: Multi-school ADs (137)
    print("Panel A: Multi-school ADs (friendly0 on female_ad)")
    sub_a = df.loc[mask].dropna(subset=['friendly0', 'weight']).copy()

    y_a = sub_a['friendly0'].values.astype(float)
    X_a = sm.add_constant(sub_a[['female_ad']].astype(float))
    w_a = sub_a['weight'].values.astype(float)

    # OLS weighted
    ols_a = sm.WLS(y_a, X_a, weights=w_a).fit()

    # Quantile regressions (unweighted - statsmodels QuantReg doesn't support aweights well)
    qr25_a = sm.QuantReg(y_a, X_a).fit(q=0.25)
    qr50_a = sm.QuantReg(y_a, X_a).fit(q=0.50)
    qr75_a = sm.QuantReg(y_a, X_a).fit(q=0.75)

    print(f"{'':15s} {'OLS':>10s} {'Q25':>10s} {'Q50':>10s} {'Q75':>10s}")
    print("-" * 55)
    print(f"{'female_ad':15s} {ols_a.params['female_ad']:10.4f} {qr25_a.params['female_ad']:10.4f} {qr50_a.params['female_ad']:10.4f} {qr75_a.params['female_ad']:10.4f}")
    print(f"{'':15s} ({ols_a.bse['female_ad']:10.4f}) ({qr25_a.bse['female_ad']:10.4f}) ({qr50_a.bse['female_ad']:10.4f}) ({qr75_a.bse['female_ad']:10.4f})")
    print(f"{'Constant':15s} {ols_a.params['const']:10.4f} {qr25_a.params['const']:10.4f} {qr50_a.params['const']:10.4f} {qr75_a.params['const']:10.4f}")
    print(f"{'':15s} ({ols_a.bse['const']:10.4f}) ({qr25_a.bse['const']:10.4f}) ({qr50_a.bse['const']:10.4f}) ({qr75_a.bse['const']:10.4f})")
    print(f"{'N':15s} {ols_a.nobs:10.0f} {qr25_a.nobs:10.0f} {qr50_a.nobs:10.0f} {qr75_a.nobs:10.0f}")
    print(f"{'R-squared':15s} {ols_a.rsquared:10.4f}")

    # Panel B: All ADs (433)
    print(f"\nPanel B: All ADs (friendly_all0 on female_ad)")
    sub_b = df.loc[mask].dropna(subset=['friendly_all0', 'weight_all']).copy()

    y_b = sub_b['friendly_all0'].values.astype(float)
    X_b = sm.add_constant(sub_b[['female_ad']].astype(float))
    w_b = sub_b['weight_all'].values.astype(float)

    # OLS weighted
    ols_b = sm.WLS(y_b, X_b, weights=w_b).fit()

    # Quantile regressions
    qr25_b = sm.QuantReg(y_b, X_b).fit(q=0.25)
    qr50_b = sm.QuantReg(y_b, X_b).fit(q=0.50)
    qr75_b = sm.QuantReg(y_b, X_b).fit(q=0.75)

    print(f"{'':15s} {'OLS':>10s} {'Q25':>10s} {'Q50':>10s} {'Q75':>10s}")
    print("-" * 55)
    print(f"{'female_ad':15s} {ols_b.params['female_ad']:10.4f} {qr25_b.params['female_ad']:10.4f} {qr50_b.params['female_ad']:10.4f} {qr75_b.params['female_ad']:10.4f}")
    print(f"{'':15s} ({ols_b.bse['female_ad']:10.4f}) ({qr25_b.bse['female_ad']:10.4f}) ({qr50_b.bse['female_ad']:10.4f}) ({qr75_b.bse['female_ad']:10.4f})")
    print(f"{'Constant':15s} {ols_b.params['const']:10.4f} {qr25_b.params['const']:10.4f} {qr50_b.params['const']:10.4f} {qr75_b.params['const']:10.4f}")
    print(f"{'':15s} ({ols_b.bse['const']:10.4f}) ({qr25_b.bse['const']:10.4f}) ({qr50_b.bse['const']:10.4f}) ({qr75_b.bse['const']:10.4f})")
    print(f"{'N':15s} {ols_b.nobs:10.0f} {qr25_b.nobs:10.0f} {qr50_b.nobs:10.0f} {qr75_b.nobs:10.0f}")
    print(f"{'R-squared':15s} {ols_b.rsquared:10.4f}")

    return df


def shrink_fes(df):
    """
    Empirical Bayes shrinkage of estimated fixed effects.

    Stata code:
      gen fem_fr_var = fem_fr_se^2
      sum fem_fr_var if ADunique  -> avg_var
      var_fe = var_ols - avg_var
      friendly1 = friendly0 * var_fe / (var_fe + fem_fr_var)
    """
    print_separator("SHRINKAGE OF FIXED EFFECTS")

    mask = df['ADunique'] == 1

    # Multi-school ADs
    df['fem_fr_var'] = df['fem_fr_se']**2
    avg_var = df.loc[mask, 'fem_fr_var'].mean()
    var_ols = df['var_ols'].iloc[0]
    var_fe = var_ols - avg_var
    df['friendly1'] = df['friendly0'] * var_fe / (var_fe + df['fem_fr_var'])

    print(f"Multi-school ADs:")
    print(f"  var_ols = {var_ols:.6f}")
    print(f"  avg_var (mean estimation variance) = {avg_var:.6f}")
    print(f"  var_fe (signal variance) = {var_fe:.6f}")

    # All ADs
    df['fe_var'] = df['fe_all_se']**2
    avg_var_all = df.loc[mask, 'fe_var'].mean()
    var_ols_all = df['var_ols_all'].iloc[0]
    var_fe_all = var_ols_all - avg_var_all
    df['friendly_all_shrunk'] = df['friendly_all0'] * var_fe_all / (var_fe_all + df['fe_var'])

    print(f"\nAll ADs:")
    print(f"  var_ols_all = {var_ols_all:.6f}")
    print(f"  avg_var_all = {avg_var_all:.6f}")
    print(f"  var_fe_all = {var_fe_all:.6f}")

    return df


def run_table3():
    """
    Table 3: Expenditure Regressions

    Uses pre-constructed EADA data with shrunk FE measures.
    Stata code ordering: n1 n3 n2 n4 (women-all, men-all, women-multi, men-multi)
    """
    print_separator("TABLE 3: Expenditure Regressions")

    eada = load_eada_data()

    year_cols = [c for c in eada.columns if c.startswith('year_dum')]

    def run_wls(data, yvar, friendly_var, weight_var):
        sub = data.dropna(subset=[yvar, friendly_var, 'female_ad', weight_var]).copy()
        td = pd.get_dummies(sub['team'], prefix='team', drop_first=True).astype(float)
        X = pd.concat([sub[[friendly_var, 'female_ad'] + year_cols].astype(float).reset_index(drop=True),
                        td.reset_index(drop=True)], axis=1)
        X = sm.add_constant(X)
        y = sub[yvar].values.astype(float)
        w = sub[weight_var].values.astype(float)
        m = sm.WLS(y, X, weights=w).fit()
        return m, friendly_var

    m1, _ = run_wls(eada, 'exp_rev_w', 'friendly_all', 'weight_all')
    m2, _ = run_wls(eada, 'exp_rev_w', 'friendly1', 'weight')
    m3, _ = run_wls(eada, 'exp_rev_m', 'friendly_all', 'weight_all')
    m4, _ = run_wls(eada, 'exp_rev_m', 'friendly1', 'weight')

    # Print (paper order: n1, n3, n2, n4)
    print(f"{'':20s} {'Women(all)':>12s} {'Men(all)':>12s} {'Women(multi)':>12s} {'Men(multi)':>12s}")
    print("-" * 70)

    print(f"{'friendly_all':20s} {m1.params['friendly_all']:12.4f} {m3.params['friendly_all']:12.4f}")
    print(f"{'':20s} ({m1.bse['friendly_all']:12.4f}) ({m3.bse['friendly_all']:12.4f})")
    print(f"{'friendly1':20s} {'':12s} {'':12s} {m2.params['friendly1']:12.4f} {m4.params['friendly1']:12.4f}")
    print(f"{'':20s} {'':12s} {'':12s} ({m2.bse['friendly1']:12.4f}) ({m4.bse['friendly1']:12.4f})")
    print(f"{'female_ad':20s} {m1.params['female_ad']:12.4f} {m3.params['female_ad']:12.4f} {m2.params['female_ad']:12.4f} {m4.params['female_ad']:12.4f}")
    print(f"{'':20s} ({m1.bse['female_ad']:12.4f}) ({m3.bse['female_ad']:12.4f}) ({m2.bse['female_ad']:12.4f}) ({m4.bse['female_ad']:12.4f})")
    print(f"{'N':20s} {m1.nobs:12.0f} {m3.nobs:12.0f} {m2.nobs:12.0f} {m4.nobs:12.0f}")
    print(f"{'R-squared':20s} {m1.rsquared:12.4f} {m3.rsquared:12.4f} {m2.rsquared:12.4f} {m4.rsquared:12.4f}")

    return {'n1': m1, 'n2': m2, 'n3': m3, 'n4': m4}


if __name__ == '__main__':
    df = load_main_data()

    # Table 1
    df, model2, model3 = run_table1(df)

    # Normalize FEs
    df = normalize_fes(df)

    # Table 2
    df = run_table2(df)

    # Shrink FEs
    df = shrink_fes(df)

    # Table 3 (uses pre-built EADA data)
    results = run_table3()

    print_separator("REPLICATION COMPLETE")
