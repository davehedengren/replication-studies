"""
03_tables.py — Tables 1, 2, and A1

Table 1: Effects of law enactment/enforcement on women's labor outcomes
Table 2: Effects of law dimensions (availability, affordability, quality)
Table A1: Summary statistics

Uses the Synthetic Difference-in-Differences (synthdid) estimator from
Arkhangelsky et al. (2021), implemented from scratch in Python.
"""

import sys, os
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(__file__))
from utils import (CLEAN_DATA_PATH, OUTPUT_DIR, OUTCOME_VARS, OUTCOME_LABELS,
                   TREATMENT_VARS, TREATMENT_LABELS, prepare_estimation_sample)

# Published synthetic DiD estimates from Table 1 (for comparison)
PUBLISHED_TABLE1 = {
    ('enactment', 'lfpr_f_25_54'): (1.41, 0.060),
    ('enactment', 'lfpr_hhchild_f'): (1.584, 0.018),
    ('enactment', 'emp2pop_f_yge25'): (1.071, 0.111),
    ('enactment', 'unemp_f_yge25'): (-0.35, 0.408),
    ('enforcement', 'lfpr_f_25_54'): (1.384, 0.064),
    ('enforcement', 'lfpr_hhchild_f'): (1.333, 0.041),
    ('enforcement', 'emp2pop_f_yge25'): (0.989, 0.145),
    ('enforcement', 'unemp_f_yge25'): (-0.314, 0.462),
}

PUBLISHED_TABLE2 = {
    ('enactment_av', 'lfpr_f_25_54'): (1.410, 0.060),
    ('enactment_af', 'lfpr_f_25_54'): (2.690, 0.001),
    ('enactment_qua', 'lfpr_f_25_54'): (1.848, 0.038),
    ('enactment_av', 'lfpr_hhchild_f'): (1.584, 0.018),
    ('enactment_af', 'lfpr_hhchild_f'): (2.245, 0.007),
    ('enactment_qua', 'lfpr_hhchild_f'): (0.579, 0.519),
    ('enactment_av', 'emp2pop_f_yge25'): (1.071, 0.111),
    ('enactment_af', 'emp2pop_f_yge25'): (1.541, 0.008),
    ('enactment_qua', 'emp2pop_f_yge25'): (1.614, 0.021),
    ('enactment_av', 'unemp_f_yge25'): (-0.350, 0.408),
    ('enactment_af', 'unemp_f_yge25'): (-0.214, 0.722),
    ('enactment_qua', 'unemp_f_yge25'): (-0.513, 0.388),
}


# ---------------------------------------------------------------------------
# Synthetic Difference-in-Differences implementation
# ---------------------------------------------------------------------------

def _build_panel_matrix(df, y_var, treat_time):
    """
    Build a balanced panel matrix Y[units x time] from the estimation sample.

    Returns:
        Y_co: (N_co x T) array of control unit outcomes
        Y_tr: (N_tr x T) array of treated unit outcomes
        T_pre: number of pre-treatment periods
        T_post: number of post-treatment periods
        control_ids: list of control unit ids
        treated_ids: list of treated unit ids
        years: sorted list of years
    """
    # Identify control vs treated units
    # Control units: never treated (treat_time is NaN for all their obs)
    unit_treat = df.groupby('id')[treat_time].first()
    control_ids = sorted(unit_treat[unit_treat.isna()].index.tolist())
    treated_ids = sorted(unit_treat[unit_treat.notna()].index.tolist())

    years = sorted(df['year'].unique())

    # Build panel matrices
    pivot = df.pivot_table(index='id', columns='year', values=y_var)
    pivot = pivot[years]  # ensure column order

    Y_co = pivot.loc[control_ids].values  # N_co x T
    Y_tr = pivot.loc[treated_ids].values  # N_tr x T

    # Determine T_pre: the earliest treatment time among treated units
    # All periods before the earliest treatment are pre-treatment
    treat_times = unit_treat[unit_treat.notna()].values
    earliest_treat = int(min(treat_times))
    T_pre = sum(1 for y in years if y < earliest_treat)
    T_post = len(years) - T_pre

    return Y_co, Y_tr, T_pre, T_post, control_ids, treated_ids, years


def _estimate_sigma(Y_co_pre):
    """
    Estimate noise level sigma from first differences of control pre-treatment outcomes.
    sigma^2 = var(first differences across time for control units)
    """
    if Y_co_pre.shape[1] < 2:
        return 1.0
    # First differences across time: Y_{i,t} - Y_{i,t-1}
    diffs = np.diff(Y_co_pre, axis=1)
    sigma_sq = np.var(diffs, ddof=1) if diffs.size > 1 else 1.0
    if sigma_sq <= 0 or np.isnan(sigma_sq):
        return 1.0
    return np.sqrt(sigma_sq)


def _compute_zeta(N_tr, T_post, sigma):
    """Regularization parameter: zeta = (N_tr * T_post)^(1/4) * sigma."""
    return (N_tr * T_post) ** 0.25 * sigma


def _solve_unit_weights(Y_co_pre, Y_tr_pre_avg, zeta):
    """
    Solve for unit weights omega that minimize:
        ||Y_co_pre' * omega - Y_tr_pre_avg||^2 + zeta^2 * ||omega||^2
    subject to: omega >= 0, sum(omega) = 1

    Y_co_pre: (N_co x T_pre)
    Y_tr_pre_avg: (T_pre,) — average of treated units' pre-treatment outcomes
    """
    N_co = Y_co_pre.shape[0]

    if N_co == 0:
        return np.array([])

    # Objective: || omega' Y_co_pre - Y_tr_pre_avg ||^2 + zeta^2 * ||omega||^2
    # omega is (N_co,), Y_co_pre is (N_co x T_pre), so omega @ Y_co_pre = (T_pre,)
    def objective(omega):
        residual = omega @ Y_co_pre - Y_tr_pre_avg  # (T_pre,)
        return np.sum(residual ** 2) + zeta ** 2 * np.sum(omega ** 2)

    def gradient(omega):
        residual = omega @ Y_co_pre - Y_tr_pre_avg  # (T_pre,)
        return 2 * (Y_co_pre @ residual) + 2 * zeta ** 2 * omega

    # Initial: uniform weights
    omega0 = np.ones(N_co) / N_co

    # Constraints: sum(omega) = 1
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    bounds = [(0, None)] * N_co

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = minimize(objective, omega0, jac=gradient, method='SLSQP',
                          bounds=bounds, constraints=constraints,
                          options={'maxiter': 1000, 'ftol': 1e-12})

    if result.success:
        omega = result.x
    else:
        # Try again without gradient
        result2 = minimize(objective, omega0, method='SLSQP',
                           bounds=bounds, constraints=constraints,
                           options={'maxiter': 2000, 'ftol': 1e-12})
        if result2.success:
            omega = result2.x
        else:
            # Fallback to uniform
            omega = np.ones(N_co) / N_co

    # Clip tiny negatives from numerical noise
    omega = np.maximum(omega, 0)
    omega = omega / omega.sum()
    return omega


def _solve_time_weights(Y_co_pre, Y_co_post_avg, zeta):
    """
    Solve for time weights lambda that minimize:
        ||Y_co_pre @ lambda - Y_co_post_avg||^2 + zeta^2 * ||lambda||^2
    subject to: lambda >= 0, sum(lambda) = 1

    Y_co_pre: (N_co x T_pre)
    Y_co_post_avg: (N_co,) — average of control units' post-treatment outcomes
    """
    T_pre = Y_co_pre.shape[1]

    if T_pre == 0:
        return np.array([])

    # Objective: || Y_co_pre @ lam - Y_co_post_avg ||^2 + zeta^2 * ||lam||^2
    def objective(lam):
        residual = Y_co_pre @ lam - Y_co_post_avg
        return np.sum(residual ** 2) + zeta ** 2 * np.sum(lam ** 2)

    def gradient(lam):
        residual = Y_co_pre @ lam - Y_co_post_avg  # (N_co,)
        return 2 * (Y_co_pre.T @ residual) + 2 * zeta ** 2 * lam

    lam0 = np.ones(T_pre) / T_pre

    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    bounds = [(0, None)] * T_pre

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = minimize(objective, lam0, jac=gradient, method='SLSQP',
                          bounds=bounds, constraints=constraints,
                          options={'maxiter': 1000, 'ftol': 1e-12})

    if result.success:
        lam = result.x
    else:
        result2 = minimize(objective, lam0, method='SLSQP',
                           bounds=bounds, constraints=constraints,
                           options={'maxiter': 2000, 'ftol': 1e-12})
        if result2.success:
            lam = result2.x
        else:
            lam = np.ones(T_pre) / T_pre

    lam = np.maximum(lam, 0)
    lam = lam / lam.sum()
    return lam


def _synthdid_estimate(Y_co, Y_tr, T_pre, T_post, omega, lam):
    """
    Compute the synthdid point estimate given weights.

    tau = weighted_post_diff - weighted_pre_diff
    where:
        weighted_post_diff = (1/T_post) * sum_t(post) [ Y_tr_avg_t - omega' Y_co_t ]
        weighted_pre_diff  = sum_t(pre) lam_t * [ Y_tr_avg_t - omega' Y_co_t ]
    """
    N_tr = Y_tr.shape[0]

    # Y_tr averaged across treated units at each time
    Y_tr_avg = Y_tr.mean(axis=0)  # (T,)

    # omega-weighted control at each time
    Y_co_weighted = omega @ Y_co  # (T,)

    # Difference at each time
    diff = Y_tr_avg - Y_co_weighted  # (T,)

    # Post-treatment average difference
    post_diff = np.mean(diff[T_pre:])

    # Pre-treatment lambda-weighted difference
    pre_diff = lam @ diff[:T_pre]

    tau = post_diff - pre_diff
    return tau


def _synthdid_core(Y_co, Y_tr, T_pre, T_post):
    """
    Core synthdid computation. Returns (tau, omega, lam).
    """
    N_co = Y_co.shape[0]
    N_tr = Y_tr.shape[0]

    Y_co_pre = Y_co[:, :T_pre]   # N_co x T_pre
    Y_tr_pre = Y_tr[:, :T_pre]   # N_tr x T_pre

    # Average treated pre-treatment outcomes
    Y_tr_pre_avg = Y_tr_pre.mean(axis=0)  # (T_pre,)

    # Average control post-treatment outcomes
    Y_co_post_avg = Y_co[:, T_pre:].mean(axis=1)  # (N_co,)

    # Estimate sigma and zeta
    sigma = _estimate_sigma(Y_co_pre)
    zeta = _compute_zeta(N_tr, T_post, sigma)

    # Solve for weights
    omega = _solve_unit_weights(Y_co_pre, Y_tr_pre_avg, zeta)
    lam = _solve_time_weights(Y_co_pre, Y_co_post_avg, zeta)

    # Compute estimate
    tau = _synthdid_estimate(Y_co, Y_tr, T_pre, T_post, omega, lam)

    return tau, omega, lam


def _placebo_se(Y_co, Y_tr, T_pre, T_post):
    """
    Placebo SE: for each control unit, pretend it's the only treated unit,
    compute synthdid on the remaining controls vs that unit.
    SE = sd(placebo_estimates).
    """
    N_co = Y_co.shape[0]
    if N_co < 3:
        return np.nan

    placebo_taus = []
    for i in range(N_co):
        # Unit i becomes "treated", rest are controls
        Y_tr_placebo = Y_co[i:i+1, :]  # 1 x T
        Y_co_placebo = np.delete(Y_co, i, axis=0)  # (N_co-1) x T

        try:
            tau_p, _, _ = _synthdid_core(Y_co_placebo, Y_tr_placebo, T_pre, T_post)
            if np.isfinite(tau_p):
                placebo_taus.append(tau_p)
        except Exception:
            continue

    if len(placebo_taus) < 3:
        return np.nan

    return np.std(placebo_taus, ddof=1)


def run_synthdid(df, y_var, treat_time):
    """
    Run Synthetic Difference-in-Differences estimator.

    Takes the DataFrame from prepare_estimation_sample(), constructs a balanced
    panel, computes unit and time weights, and returns the synthdid estimate
    with placebo standard errors.

    Returns dict with: estimate, se, pval, n, control_mean
    """
    foo = prepare_estimation_sample(df, y_var, treat_time).reset_index(drop=True)

    if len(foo) == 0 or foo['treat'].sum() == 0:
        return None

    # Build balanced panel matrix
    try:
        Y_co, Y_tr, T_pre, T_post, control_ids, treated_ids, years = \
            _build_panel_matrix(foo, y_var, treat_time)
    except Exception as e:
        print(f"  Panel construction failed for {y_var}/{treat_time}: {e}")
        return None

    N_co, N_tr = Y_co.shape[0], Y_tr.shape[0]

    if N_co < 2 or N_tr < 1 or T_pre < 2 or T_post < 1:
        print(f"  Insufficient data: N_co={N_co}, N_tr={N_tr}, T_pre={T_pre}, T_post={T_post}")
        return None

    # Check for NaN in panel
    if np.any(np.isnan(Y_co)) or np.any(np.isnan(Y_tr)):
        print(f"  NaN values in panel for {y_var}/{treat_time}")
        return None

    # Run synthdid
    try:
        tau, omega, lam = _synthdid_core(Y_co, Y_tr, T_pre, T_post)
    except Exception as e:
        print(f"  Synthdid estimation failed for {y_var}/{treat_time}: {e}")
        return None

    # Placebo SE
    se = _placebo_se(Y_co, Y_tr, T_pre, T_post)

    # P-value from normal approximation
    if np.isfinite(se) and se > 0:
        z = tau / se
        pval = 2 * (1 - stats.norm.cdf(abs(z)))
    else:
        pval = np.nan

    # Control mean
    control_mean = foo[foo[treat_time].isna()][y_var].mean()

    return {
        'estimate': tau,
        'se': se,
        'pval': pval,
        'n': len(foo),
        'control_mean': control_mean,
    }


def run_table(df, treatments, published_ref, table_name):
    """Run synthdid regressions for a set of treatments and display results."""
    print(f"\n{'='*120}")
    print(f"  {table_name}")
    print(f"{'='*120}")

    header = f"{'Treatment':20s}"
    for y_var in OUTCOME_VARS:
        header += f" {OUTCOME_LABELS[y_var]:>20s}"
    print(header)
    print("-" * 120)

    all_results = {}
    for treat in treatments:
        row_est = f"{TREATMENT_LABELS[treat]:20s}"
        row_pval = f"{'':20s}"
        row_pub = f"{'  (published)':20s}"

        for y_var in OUTCOME_VARS:
            res = run_synthdid(df, y_var, treat)
            if res:
                stars = '***' if res['pval'] < 0.01 else '**' if res['pval'] < 0.05 else '*' if res['pval'] < 0.1 else ''
                row_est += f" {res['estimate']:17.3f}{stars:3s}"
                pval_str = f"[{res['pval']:.3f}]"
                row_pval += f" {pval_str:>20s}"
                all_results[(treat, y_var)] = res

                # Published comparison
                pub = published_ref.get((treat, y_var))
                if pub:
                    row_pub += f" {pub[0]:17.3f}   "
                else:
                    row_pub += f" {'':>20s}"
            else:
                row_est += f" {'N/A':>20s}"
                row_pval += f" {'':>20s}"
                row_pub += f" {'':>20s}"

        print(row_est)
        print(row_pval)
        print(row_pub)
        print()

    # Observations and control means
    print("-" * 120)
    for treat in treatments[:1]:
        n_row = f"{'Observations':20s}"
        m_row = f"{'Never-treated mean':20s}"
        for y_var in OUTCOME_VARS:
            res = all_results.get((treat, y_var))
            if res:
                n_row += f" {res['n']:>20,d}"
                m_row += f" {res['control_mean']:20.2f}"
            else:
                n_row += f" {'':>20s}"
                m_row += f" {'':>20s}"
        print(n_row)
        print(m_row)

    return all_results


def table_a1(df):
    """Summary statistics (Table A1)."""
    print(f"\n{'='*100}")
    print(f"  TABLE A1: Summary Statistics")
    print(f"{'='*100}")

    # Use the estimation sample for LFPR 25-54 enactment
    foo = prepare_estimation_sample(df, 'lfpr_f_25_54', 'enactment')

    vars_to_summarize = [
        ('has_childcare_law', 'Has a childcare law'),
        ('availability', 'Law regulates availability'),
        ('affordability', 'Law regulates affordability'),
        ('quality', 'Law regulates quality'),
        ('lfpr_f_25_54', 'FLFP 25-54'),
        ('lfpr_hhchild_f', 'FLFP 25-54 w. children <6'),
        ('emp2pop_f_yge25', '% of women working (25+)'),
        ('unemp_f_yge25', 'Female unemployment rate (25+)'),
    ]

    print(f"{'Variable':45s} {'Obs':>8s} {'Mean':>10s} {'SD':>10s} "
          f"{'Median':>10s} {'Min':>10s} {'Max':>10s}")
    print("-" * 100)

    rows = []
    for var, label in vars_to_summarize:
        if var not in foo.columns:
            continue
        col = foo[var].dropna()
        row = {
            'variable': label,
            'obs': len(col),
            'mean': col.mean(),
            'sd': col.std(),
            'median': col.median(),
            'min': col.min(),
            'max': col.max(),
        }
        rows.append(row)
        print(f"{label:45s} {len(col):8d} {col.mean():10.2f} {col.std():10.2f} "
              f"{col.median():10.2f} {col.min():10.2f} {col.max():10.2f}")

    return pd.DataFrame(rows)


def main():
    print("=" * 60)
    print("03_tables.py: Tables 1, 2, and A1")
    print("=" * 60)

    df = pd.read_parquet(CLEAN_DATA_PATH)
    print(f"Data loaded: {len(df)} observations, {df['id'].nunique()} countries")

    print("\nUsing Synthetic Difference-in-Differences (Arkhangelsky et al. 2021).")
    print("Published synthdid estimates shown in parentheses for comparison.\n")

    # Table 1
    t1_results = run_table(
        df, ['enactment', 'enforcement'], PUBLISHED_TABLE1,
        "TABLE 1: Childcare Laws and Women's Labor Market Outcomes (Synthetic DiD)"
    )

    # Table 2
    t2_results = run_table(
        df, ['enactment_av', 'enactment_af', 'enactment_qua'], PUBLISHED_TABLE2,
        "TABLE 2: Childcare Law Attributes (Synthetic DiD)"
    )

    # Table A1
    stats_df = table_a1(df)

    # Save
    with open(os.path.join(OUTPUT_DIR, 'tables.txt'), 'w') as f:
        f.write("See console output for formatted tables\n")
        f.write("Synthetic DiD results (Arkhangelsky et al. 2021)\n")
    stats_df.to_csv(os.path.join(OUTPUT_DIR, 'table_a1.csv'), index=False)
    print("\nSaved: tables.txt, table_a1.csv")

    return t1_results, t2_results


if __name__ == '__main__':
    main()
