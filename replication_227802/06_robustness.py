"""
06_robustness.py — Robustness Checks for Childcare Laws Study

Checks:
1. Alternative treatment timing (enforcement instead of enactment)
2. Restricting to post-1991 enactments only
3. Dropping OECD countries
4. Dropping small islands / micro-states
5. Placebo test: random treatment assignment
6. Alternative outcome: male LFPR as placebo
7. Shorter pre/post windows for event study
8. Winsorizing outlier countries
9. Region-by-region heterogeneity
10. Leave-one-region-out sensitivity
"""

import sys, os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from utils import (CLEAN_DATA_PATH, OUTPUT_DIR, OUTCOME_VARS, OUTCOME_LABELS,
                   prepare_estimation_sample, create_stack, weights_stacked_did)


def section(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def run_twfe(df, y_var, treat_time):
    """Quick TWFE DiD regression."""
    foo = prepare_estimation_sample(df, y_var, treat_time).reset_index(drop=True)
    if len(foo) == 0 or foo['treat'].sum() == 0:
        return None

    y = foo[y_var].values
    treat = foo['treat'].values.astype(float)

    country_dummies = pd.get_dummies(foo['id'], prefix='c', drop_first=True, dtype=float)
    year_dummies = pd.get_dummies(foo['year'], prefix='y', drop_first=True, dtype=float)

    X = pd.concat([
        pd.DataFrame({'treat': treat}),
        country_dummies,
        year_dummies
    ], axis=1)

    model = sm.OLS(y, sm.add_constant(X))
    groups = foo['id'].values
    try:
        results = model.fit(cov_type='cluster', cov_kwds={'groups': groups})
    except Exception:
        results = model.fit(cov_type='HC1')

    return {
        'estimate': results.params['treat'],
        'se': results.bse['treat'],
        'pval': results.pvalues['treat'],
        'n': len(foo),
        'n_countries': foo['id'].nunique(),
    }


def check1_enforcement(df):
    """Check 1: Use enforcement year instead of enactment."""
    section("CHECK 1: ENFORCEMENT TIMING (instead of enactment)")
    print("Using enforcement year as treatment timing instead of enactment.\n")

    print(f"{'Outcome':25s} {'Enactment Est':>14s} {'Enforcement Est':>16s} {'Enf. p-val':>12s}")
    print("-" * 70)

    for y_var in OUTCOME_VARS:
        res_enact = run_twfe(df, y_var, 'enactment')
        res_enforce = run_twfe(df, y_var, 'enforcement')
        if res_enact and res_enforce:
            print(f"{OUTCOME_LABELS[y_var]:25s} {res_enact['estimate']:14.3f} "
                  f"{res_enforce['estimate']:16.3f} {res_enforce['pval']:12.3f}")


def check2_post1991(df):
    """Check 2: Restrict to countries that enacted after 1991."""
    section("CHECK 2: POST-1991 ENACTMENTS ONLY")

    df2 = df.copy()
    # Keep never-treated + those with enactment >= 1991
    df2 = df2[df2['enactment'].isna() | (df2['enactment'] >= 1991)]
    n_countries = df2['id'].nunique()
    print(f"Sample restricted to never-treated + post-1991 enactments: {n_countries} countries\n")

    print(f"{'Outcome':25s} {'Full Sample':>14s} {'Post-1991':>14s} {'p-val':>10s}")
    print("-" * 65)

    for y_var in OUTCOME_VARS:
        res_full = run_twfe(df, y_var, 'enactment')
        res_post = run_twfe(df2, y_var, 'enactment')
        if res_full and res_post:
            print(f"{OUTCOME_LABELS[y_var]:25s} {res_full['estimate']:14.3f} "
                  f"{res_post['estimate']:14.3f} {res_post['pval']:10.3f}")


def check3_drop_oecd(df):
    """Check 3: Drop high-income OECD countries."""
    section("CHECK 3: DROP HIGH-INCOME OECD COUNTRIES")

    df3 = df[df['region'] != 'High income: OECD'].copy()
    n_dropped = df['id'].nunique() - df3['id'].nunique()
    print(f"Dropped {n_dropped} OECD countries. Remaining: {df3['id'].nunique()}\n")

    print(f"{'Outcome':25s} {'Full Sample':>14s} {'Non-OECD':>14s} {'p-val':>10s}")
    print("-" * 65)

    for y_var in OUTCOME_VARS:
        res_full = run_twfe(df, y_var, 'enactment')
        res_nooecd = run_twfe(df3, y_var, 'enactment')
        if res_full and res_nooecd:
            print(f"{OUTCOME_LABELS[y_var]:25s} {res_full['estimate']:14.3f} "
                  f"{res_nooecd['estimate']:14.3f} {res_nooecd['pval']:10.3f}")


def check4_drop_small(df):
    """Check 4: Drop micro-states (countries with very few year observations)."""
    section("CHECK 4: DROP MICRO-STATES")

    # Identify countries with very short panels (< 15 years of data for main outcome)
    obs_count = df.groupby('id')['lfpr_f_25_54'].apply(lambda x: x.notna().sum())
    keep_ids = obs_count[obs_count >= 15].index
    df4 = df[df['id'].isin(keep_ids)].copy()
    n_dropped = df['id'].nunique() - df4['id'].nunique()
    print(f"Dropped {n_dropped} countries with <15 years of LFPR data. "
          f"Remaining: {df4['id'].nunique()}\n")

    print(f"{'Outcome':25s} {'Full Sample':>14s} {'No Micro':>14s} {'p-val':>10s}")
    print("-" * 65)

    for y_var in OUTCOME_VARS:
        res_full = run_twfe(df, y_var, 'enactment')
        res_nomicro = run_twfe(df4, y_var, 'enactment')
        if res_full and res_nomicro:
            print(f"{OUTCOME_LABELS[y_var]:25s} {res_full['estimate']:14.3f} "
                  f"{res_nomicro['estimate']:14.3f} {res_nomicro['pval']:10.3f}")


def check5_placebo(df, n_iter=200):
    """Check 5: Placebo test with random treatment assignment."""
    section("CHECK 5: PLACEBO TEST (random treatment assignment)")
    print(f"Running {n_iter} iterations with randomly shuffled treatment status.\n")

    y_var = 'lfpr_f_25_54'
    real_res = run_twfe(df, y_var, 'enactment')
    if not real_res:
        print("Cannot run: main estimate failed.")
        return

    real_est = real_res['estimate']
    placebo_ests = []

    rng = np.random.RandomState(42)
    foo = prepare_estimation_sample(df, y_var, 'enactment').reset_index(drop=True)

    for i in range(n_iter):
        foo_perm = foo.copy()
        # Shuffle treatment across countries (preserve within-country correlation)
        country_ids = foo_perm['id'].unique()
        treat_map = dict(zip(country_ids, rng.permutation(
            foo_perm.groupby('id')['enactment'].first().values)))
        foo_perm['perm_enactment'] = foo_perm['id'].map(treat_map)
        foo_perm['treat'] = np.where(
            pd.isna(foo_perm['perm_enactment']), 0,
            np.where(foo_perm['year'] >= foo_perm['perm_enactment'], 1, 0)
        )

        y = foo_perm[y_var].values
        treat = foo_perm['treat'].values.astype(float)
        country_dummies = pd.get_dummies(foo_perm['id'], prefix='c', drop_first=True, dtype=float)
        year_dummies = pd.get_dummies(foo_perm['year'], prefix='y', drop_first=True, dtype=float)

        X = pd.concat([pd.DataFrame({'treat': treat}), country_dummies, year_dummies], axis=1)
        try:
            model = sm.OLS(y, sm.add_constant(X))
            results = model.fit()
            placebo_ests.append(results.params['treat'])
        except Exception:
            pass

    placebo_arr = np.array(placebo_ests)
    pval_perm = (np.abs(placebo_arr) >= np.abs(real_est)).mean()

    print(f"Real estimate: {real_est:.3f}")
    print(f"Placebo distribution: mean={placebo_arr.mean():.3f}, "
          f"sd={placebo_arr.std():.3f}")
    print(f"Placebo range: [{placebo_arr.min():.3f}, {placebo_arr.max():.3f}]")
    print(f"Permutation p-value: {pval_perm:.3f}")
    print(f"  (fraction of |placebo| >= |real|)")


def check6_male_placebo(df):
    """Check 6: Male LFPR as placebo outcome."""
    section("CHECK 6: MALE LFPR AS PLACEBO OUTCOME")

    male_vars = [c for c in df.columns if c.startswith('lfpr_') and '_m_' in c.lower()]
    # Try to find male 25-54 LFPR
    if 'lfpr_f_25_54' in df.columns:
        # The data may have male counterparts; check for them
        # The ILO data was filtered to females only, so we note this
        pass

    print("Note: The cleaned data only includes female labor outcomes (per the R code).")
    print("Male LFPR data was not merged in the cleaning step.")
    print("We instead check: Does treatment affect FEMALE UNEMPLOYMENT (expected: negative/null)?")
    print()

    res_unemp = run_twfe(df, 'unemp_f_yge25', 'enactment')
    if res_unemp:
        print(f"Female unemployment effect: {res_unemp['estimate']:.3f} "
              f"(SE={res_unemp['se']:.3f}, p={res_unemp['pval']:.3f})")
        print("A null result on unemployment alongside positive LFPR effect suggests")
        print("labor force entrants are finding employment, not just searching.")


def check7_shorter_windows(df):
    """Check 7: Shorter event study windows."""
    section("CHECK 7: SHORTER EVENT STUDY WINDOWS")

    y_var = 'lfpr_f_25_54'
    df_clean = df[df[y_var].notna()].copy()

    for kpre, kpost in [(4, 5), (3, 3)]:
        print(f"\n--- Window: {kpre} pre, {kpost} post ---")
        try:
            df_stack = create_stack(df_clean, 'year', 'id', 'enactment', kpre, kpost)
            df_stack = weights_stacked_did(df_stack, 'treat', 'event_time', 'sub_exp')

            event_times = sorted(df_stack['event_time'].unique())
            ref_time = -1

            y = df_stack[y_var].values
            weights = df_stack['stack_weight'].values

            et_treat_cols = {}
            for et in event_times:
                if et == ref_time:
                    continue
                col_name = f'et_{et}_treat'
                et_treat_cols[col_name] = ((df_stack['event_time'] == et) &
                                            (df_stack['treat'] == 1)).astype(float).values

            treat_fe = df_stack['treat'].astype(float).values
            et_dummies = pd.get_dummies(df_stack['event_time'], prefix='et',
                                         drop_first=True, dtype=float)

            X_df = pd.DataFrame(et_treat_cols)
            X_df['treat_fe'] = treat_fe
            for col in et_dummies.columns:
                X_df[col] = et_dummies[col].values
            X_df = sm.add_constant(X_df)

            model = sm.WLS(y, X_df, weights=weights)
            groups = df_stack['id'].values
            try:
                results = model.fit(cov_type='cluster', cov_kwds={'groups': groups})
            except Exception:
                results = model.fit(cov_type='HC1')

            # Extract post-treatment coefficients
            post_names = [f'et_{et}_treat' for et in range(0, kpost + 1)
                         if f'et_{et}_treat' in results.params.index]
            post_coefs = [results.params[n] for n in post_names]
            att = np.mean(post_coefs) if post_coefs else np.nan

            print(f"  N stacked obs: {len(df_stack)}")
            print(f"  ATT (avg post): {att:.3f}")
            for name in post_names:
                print(f"    {name}: {results.params[name]:.3f} "
                      f"(SE={results.bse[name]:.3f})")
        except Exception as e:
            print(f"  Error: {e}")


def check8_winsorize(df):
    """Check 8: Winsorize extreme LFPR values at 1st/99th percentile."""
    section("CHECK 8: WINSORIZED OUTCOMES")

    y_var = 'lfpr_f_25_54'
    df8 = df.copy()
    p01 = df8[y_var].quantile(0.01)
    p99 = df8[y_var].quantile(0.99)
    df8[y_var] = df8[y_var].clip(p01, p99)
    print(f"Winsorized {y_var} at [{p01:.1f}, {p99:.1f}]\n")

    res_orig = run_twfe(df, y_var, 'enactment')
    res_wins = run_twfe(df8, y_var, 'enactment')

    if res_orig and res_wins:
        print(f"{'':20s} {'Original':>12s} {'Winsorized':>12s}")
        print(f"{'Estimate':20s} {res_orig['estimate']:12.3f} {res_wins['estimate']:12.3f}")
        print(f"{'SE':20s} {res_orig['se']:12.3f} {res_wins['se']:12.3f}")
        print(f"{'p-value':20s} {res_orig['pval']:12.3f} {res_wins['pval']:12.3f}")


def check9_by_region(df):
    """Check 9: Region-by-region heterogeneity."""
    section("CHECK 9: HETEROGENEITY BY REGION")

    y_var = 'lfpr_f_25_54'
    print(f"{'Region':30s} {'Est':>10s} {'SE':>10s} {'p-val':>10s} {'N':>8s} {'Countries':>10s}")
    print("-" * 80)

    for region in sorted(df['region'].dropna().unique()):
        df_region = df[df['region'] == region].copy()
        res = run_twfe(df_region, y_var, 'enactment')
        if res:
            print(f"{region:30s} {res['estimate']:10.3f} {res['se']:10.3f} "
                  f"{res['pval']:10.3f} {res['n']:8d} {res['n_countries']:10d}")
        else:
            print(f"{region:30s} {'N/A':>10s}")


def check10_leave_one_region_out(df):
    """Check 10: Leave-one-region-out sensitivity."""
    section("CHECK 10: LEAVE-ONE-REGION-OUT")

    y_var = 'lfpr_f_25_54'
    res_full = run_twfe(df, y_var, 'enactment')
    if res_full:
        print(f"Full sample: {res_full['estimate']:.3f} (SE={res_full['se']:.3f})\n")

    print(f"{'Dropped Region':30s} {'Est':>10s} {'SE':>10s} {'p-val':>10s} {'Countries':>10s}")
    print("-" * 72)

    for region in sorted(df['region'].dropna().unique()):
        df_drop = df[df['region'] != region].copy()
        res = run_twfe(df_drop, y_var, 'enactment')
        if res:
            print(f"{region:30s} {res['estimate']:10.3f} {res['se']:10.3f} "
                  f"{res['pval']:10.3f} {res['n_countries']:10d}")


def main():
    print("=" * 80)
    print("06_robustness.py: Robustness Checks")
    print("=" * 80)

    df = pd.read_parquet(CLEAN_DATA_PATH)
    print(f"Data loaded: {len(df)} observations, {df['id'].nunique()} countries")

    results = {}

    check1_enforcement(df)
    check2_post1991(df)
    check3_drop_oecd(df)
    check4_drop_small(df)
    check5_placebo(df, n_iter=200)
    check6_male_placebo(df)
    check7_shorter_windows(df)
    check8_winsorize(df)
    check9_by_region(df)
    check10_leave_one_region_out(df)

    section("ROBUSTNESS SUMMARY")
    print("All robustness checks completed. See output above for details.")
    print("Key findings should be summarized in the writeup.")


if __name__ == '__main__':
    main()
