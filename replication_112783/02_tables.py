"""
02_tables.py - Replicate Tables 1 and 2 from paper 112783.

Table 1: Descriptive statistics by killer/non-killer x game/non-game
Table 2: Probit regressions with marginal effects
  - Model 1: Games (cat5=1), cohort FE, robust clustered SEs
  - Model 2: Non-Games (cat5=0), cohort + category FE, robust clustered SEs
"""

import sys
sys.path.insert(0, '.')
from utils import *
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Probit
import warnings
warnings.filterwarnings('ignore')

df = load_data()

# ============================================================
# TABLE 1: Descriptive Statistics
# ============================================================
print("=" * 80)
print("TABLE 1: DESCRIPTIVE STATISTICS FOR KILLER & NON-KILLER, GAME & NON-GAME APPS")
print("=" * 80)

groups = {
    'Games Killer': df[(df['cat5'] == 1) & (df['killerappgros'] == 1)],
    'Games Non-Killer': df[(df['cat5'] == 1) & (df['killerappgros'] == 0)],
    'Non-Games Killer': df[(df['cat5'] == 0) & (df['killerappgros'] == 1)],
    'Non-Games Non-Killer': df[(df['cat5'] == 0) & (df['killerappgros'] == 0)],
}

header = f"{'Variable':<25}"
for g in groups:
    header += f"{g:>20}"
print(header)
print("-" * 105)

for var in TABLE1_VARS:
    label = VAR_LABELS.get(var, var)
    means = ""
    sds = ""
    for g in groups:
        means += f"{groups[g][var].mean():>20.2f}"
        sds += f"({'%.2f' % groups[g][var].std():>18})"
    print(f"{label:<25}{means}")
    print(f"{'':25}{sds}")

print("-" * 105)
obs_row = f"{'Observations':<25}"
for g in groups:
    obs_row += f"{len(groups[g]):>20}"
print(obs_row)

# Published Table 1 values for comparison
print("\n--- Comparison with Published Table 1 ---")
pub_means = {
    'Games Killer': {'countapp': 4.67, 'numverapp': 2.36, 'noupdates': 0.58,
                     'avdeltatime': 2.90, 'avprice': 2.13, 'avsize': 34.60,
                     'numcomapp': 330.96, 'scoreapp': 4.16},
    'Games Non-Killer': {'countapp': 2.30, 'numverapp': 1.45, 'noupdates': 0.74,
                         'avdeltatime': 3.59, 'avprice': 1.00, 'avsize': 12.10,
                         'numcomapp': 8.56, 'scoreapp': 2.35},
    'Non-Games Killer': {'countapp': 2.62, 'numverapp': 3.42, 'noupdates': 0.45,
                         'avdeltatime': 7.24, 'avprice': 15.73, 'avsize': 24.44,
                         'numcomapp': 197.38, 'scoreapp': 3.81},
    'Non-Games Non-Killer': {'countapp': 4.46, 'numverapp': 1.65, 'noupdates': 0.69,
                              'avdeltatime': 6.34, 'avprice': 1.97, 'avsize': 13.33,
                              'numcomapp': 6.98, 'scoreapp': 1.90},
}
pub_ns = {'Games Killer': 535, 'Games Non-Killer': 7148,
           'Non-Games Killer': 922, 'Non-Games Non-Killer': 27744}

print(f"{'Variable':<25}{'Group':<22}{'Published':>12}{'Replicated':>12}{'Diff':>10}")
any_diff = False
for g in groups:
    for var in TABLE1_VARS:
        label = VAR_LABELS.get(var, var)
        pub = pub_means[g][var]
        rep = groups[g][var].mean()
        diff = rep - pub
        if abs(diff) > 0.02:
            any_diff = True
            print(f"{label:<25}{g:<22}{pub:>12.2f}{rep:>12.2f}{diff:>10.2f} ***")

if not any_diff:
    print("All means match within 0.02 tolerance.")

for g in groups:
    n_pub = pub_ns[g]
    n_rep = len(groups[g])
    match = "MATCH" if n_pub == n_rep else f"MISMATCH (pub={n_pub})"
    print(f"  N {g}: {n_rep} - {match}")


# ============================================================
# TABLE 2: Probit Regressions
# ============================================================
print("\n" + "=" * 80)
print("TABLE 2: REGRESSION RESULTS FOR GAME & NON-GAME APPS")
print("=" * 80)

def run_probit(df_sub, xvars, cluster_var='cohort'):
    """Run probit with robust clustered SEs, return results and marginal effects."""
    y = df_sub[DEPVAR].astype(float)
    X = sm.add_constant(df_sub[xvars].astype(float))

    model = Probit(y, X)
    res = model.fit(cov_type='cluster', cov_kwds={'groups': df_sub[cluster_var].astype(int)},
                    disp=0, maxiter=300)

    # Average marginal effects (Stata's margeff uses MEM but AME is closer to published)
    mfx = res.get_margeff(at='overall', method='dydx')

    return res, mfx

# Model 1: Games (cat5=1)
games = df[df['cat5'] == 1].copy()
games_xvars = REGRESSORS + COHORT_DUMMIES
res_games, mfx_games = run_probit(games, games_xvars)

# Model 2: Non-Games (cat5=0)
nongames = df[df['cat5'] == 0].copy()
nongames_xvars = REGRESSORS + COHORT_DUMMIES + CAT_DUMMIES_NONGAMES
res_nongames, mfx_nongames = run_probit(nongames, nongames_xvars)

# Display results
print(f"\n{'Variable':<25}{'Games':>12}{'Games':>12}{'Non-Games':>12}{'Non-Games':>12}")
print(f"{'':25}{'Probit':>12}{'Marg.Eff.':>12}{'Probit':>12}{'Marg.Eff.':>12}")
print("-" * 73)

# Published Table 2 values
pub_t2 = {
    'countapp':    (0.0219, 0.00504, 0.00133, 0.000300, 0.00119, 0.00105, 0.0000613, 0.0000536),
    'numverapp':   (0.0126, 0.0295, 0.000768, 0.00180, 0.0224, 0.0113, 0.00116, 0.000588),
    'noupdates':   (0.317, 0.127, 0.0184, 0.00854, 0.108, 0.0688, 0.00546, 0.00372),
    'avdeltatime': (-0.00513, 0.00441, -0.000312, 0.000265, -0.00407, 0.00154, -0.000210, 0.0000809),
    'avprice':     (0.140, 0.0351, 0.00854, 0.00216, 0.00800, 0.00279, 0.000413, 0.000139),
    'avsize':      (0.00445, 0.00133, 0.000271, 0.0000771, 0.00132, 0.000259, 0.0000684, 0.0000132),
    'lnnumcomapp': (0.673, 0.0315, 0.0409, 0.00129, 0.533, 0.0179, 0.0275, 0.00106),
    'scoreapp':    (0.103, 0.0340, 0.00629, 0.00199, 0.0683, 0.0134, 0.00353, 0.000690),
}
pub_const = {'games': (-3.610, 0.196), 'nongames': (-3.179, 0.167)}

# Map from Stata ordering to display ordering matching paper
display_order = ['countapp', 'numverapp', 'noupdates', 'avdeltatime',
                 'avprice', 'avsize', 'lnnumcomapp', 'scoreapp']

for var in display_order:
    label = VAR_LABELS.get(var, var)

    # Games probit
    g_coef = res_games.params[var]
    g_se = res_games.bse[var]

    # Games marginal effects
    mfx_idx = REGRESSORS.index(var)
    g_mfx = mfx_games.margeff[mfx_idx]
    g_mfx_se = mfx_games.margeff_se[mfx_idx]

    # Non-games probit
    ng_coef = res_nongames.params[var]
    ng_se = res_nongames.bse[var]

    # Non-games marginal effects
    ng_mfx = mfx_nongames.margeff[mfx_idx]
    ng_mfx_se = mfx_nongames.margeff_se[mfx_idx]

    print(f"{label:<25}{g_coef:>12.4f}{g_mfx:>12.6f}{ng_coef:>12.4f}{ng_mfx:>12.7f}")
    print(f"{'':25}({g_se:>10.4f})({g_mfx_se:>10.6f})({ng_se:>10.4f})({ng_mfx_se:>10.7f})")

# Constant
print(f"{'Constant':<25}{res_games.params['const']:>12.4f}{'':>12}{res_nongames.params['const']:>12.4f}")
print(f"{'':25}({res_games.bse['const']:>10.4f}){'':>12}({res_nongames.bse['const']:>10.4f})")

print("-" * 73)
print(f"{'Cohort Fixed Effects':<25}{'Yes':>12}{'Yes':>12}{'Yes':>12}{'Yes':>12}")
print(f"{'Category Fixed Effects':<25}{'No':>12}{'No':>12}{'Yes':>12}{'Yes':>12}")
print(f"{'Observations':<25}{res_games.nobs:>12.0f}{res_games.nobs:>12.0f}{res_nongames.nobs:>12.0f}{res_nongames.nobs:>12.0f}")
print(f"{'Pseudo R-squared':<25}{res_games.prsquared:>12.4f}{res_games.prsquared:>12.4f}{res_nongames.prsquared:>12.4f}{res_nongames.prsquared:>12.4f}")

# Detailed comparison with published values
print("\n--- Detailed Comparison with Published Table 2 ---")
print(f"{'Variable':<25}{'Col':<8}{'Published':>12}{'Replicated':>12}{'Diff':>12}{'Match?':>8}")

for var in display_order:
    label = VAR_LABELS.get(var, var)
    pub = pub_t2[var]
    # Games probit coef, SE
    g_coef = res_games.params[var]
    g_se = res_games.bse[var]
    mfx_idx = REGRESSORS.index(var)
    g_mfx = mfx_games.margeff[mfx_idx]
    g_mfx_se = mfx_games.margeff_se[mfx_idx]
    ng_coef = res_nongames.params[var]
    ng_se = res_nongames.bse[var]
    ng_mfx = mfx_nongames.margeff[mfx_idx]
    ng_mfx_se = mfx_nongames.margeff_se[mfx_idx]

    pairs = [
        ('G coef', pub[0], g_coef),
        ('G SE', pub[1], g_se),
        ('G mfx', pub[2], g_mfx),
        ('G mfx SE', pub[3], g_mfx_se),
        ('NG coef', pub[4], ng_coef),
        ('NG SE', pub[5], ng_se),
        ('NG mfx', pub[6], ng_mfx),
        ('NG mfx SE', pub[7], ng_mfx_se),
    ]
    for colname, p, r in pairs:
        diff = r - p
        tol = max(abs(p) * 0.05, 0.001)  # 5% or 0.001 absolute
        match = "OK" if abs(diff) < tol else "DIFF"
        if match == "DIFF":
            print(f"{label:<25}{colname:<8}{p:>12.6f}{r:>12.6f}{diff:>12.6f}{match:>8}")

# Constants
for name, res, pub in [('Games', res_games, pub_const['games']),
                        ('Non-Games', res_nongames, pub_const['nongames'])]:
    c = res.params['const']
    se = res.bse['const']
    diff_c = c - pub[0]
    diff_se = se - pub[1]
    if abs(diff_c) > 0.01:
        print(f"{'Constant':<25}{name + ' coef':<8}{pub[0]:>12.4f}{c:>12.4f}{diff_c:>12.4f}{'DIFF':>8}")
    if abs(diff_se) > 0.01:
        print(f"{'Constant':<25}{name + ' SE':<8}{pub[1]:>12.4f}{se:>12.4f}{diff_se:>12.4f}{'DIFF':>8}")

print(f"\nGames Pseudo R2: {res_games.prsquared:.4f} (published: 0.4953)")
print(f"Non-Games Pseudo R2: {res_nongames.prsquared:.4f} (published: 0.3512)")
print("Done.")
