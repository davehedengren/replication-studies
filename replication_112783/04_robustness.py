"""
04_robustness.py - Robustness checks for paper 112783.
10 checks tailored to probit models of killer app status.
"""

import sys
sys.path.insert(0, '.')
from utils import *
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Probit, Logit
import warnings
warnings.filterwarnings('ignore')

df = load_data()


def run_probit(df_sub, xvars, cluster_var='cohort', label=""):
    """Run probit and return key results."""
    y = df_sub[DEPVAR].astype(float)
    X = sm.add_constant(df_sub[xvars].astype(float))
    try:
        model = Probit(y, X)
        res = model.fit(cov_type='cluster', cov_kwds={'groups': df_sub[cluster_var].astype(int)},
                        disp=0, maxiter=300)
        return res
    except Exception as e:
        print(f"  [{label}] Failed: {e}")
        return None


def print_key_results(res, label, vars_to_show=None):
    """Print key coefficient results."""
    if res is None:
        print(f"  {label}: FAILED")
        return
    if vars_to_show is None:
        vars_to_show = REGRESSORS
    print(f"\n  {label} (N={res.nobs:.0f}, Pseudo R2={res.prsquared:.4f})")
    print(f"  {'Variable':<25}{'Coef':>10}{'SE':>10}{'P-val':>8}{'Sig':>6}")
    for v in vars_to_show:
        if v in res.params.index:
            c = res.params[v]
            se = res.bse[v]
            p = res.pvalues[v]
            sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
            print(f"  {VAR_LABELS.get(v, v):<25}{c:>10.4f}{se:>10.4f}{p:>8.3f}{sig:>6}")


# Baseline results for comparison
print("=" * 70)
print("ROBUSTNESS CHECKS: Paper 112783")
print("=" * 70)

# Baseline
games = df[df['cat5'] == 1].copy()
nongames = df[df['cat5'] == 0].copy()
games_xvars = REGRESSORS + COHORT_DUMMIES
nongames_xvars = REGRESSORS + COHORT_DUMMIES + CAT_DUMMIES_NONGAMES

print("\n--- BASELINE ---")
res_g = run_probit(games, games_xvars, label="Baseline Games")
res_ng = run_probit(nongames, nongames_xvars, label="Baseline Non-Games")
print_key_results(res_g, "Baseline: Games")
print_key_results(res_ng, "Baseline: Non-Games")


# ============================================================
# CHECK 1: Drop duplicate observations
# ============================================================
print("\n\n" + "=" * 70)
print("CHECK 1: Drop exact duplicate observations")
print("=" * 70)
df_nodup = df.drop_duplicates()
print(f"Dropped {len(df) - len(df_nodup)} exact duplicates ({len(df)} -> {len(df_nodup)})")
g1 = df_nodup[df_nodup['cat5'] == 1].copy()
ng1 = df_nodup[df_nodup['cat5'] == 0].copy()
res_g1 = run_probit(g1, games_xvars, label="NoDup Games")
res_ng1 = run_probit(ng1, nongames_xvars, label="NoDup Non-Games")
print_key_results(res_g1, "No Duplicates: Games")
print_key_results(res_ng1, "No Duplicates: Non-Games")


# ============================================================
# CHECK 2: Drop extreme outliers (top 1% countapp, avprice, avsize)
# ============================================================
print("\n\n" + "=" * 70)
print("CHECK 2: Drop extreme outliers (top 1% of countapp, avprice, avsize)")
print("=" * 70)
q99_count = df['countapp'].quantile(0.99)
q99_price = df['avprice'].quantile(0.99)
q99_size = df['avsize'].quantile(0.99)
df_trim = df[(df['countapp'] <= q99_count) & (df['avprice'] <= q99_price) & (df['avsize'] <= q99_size)]
print(f"Trimmed: {len(df)} -> {len(df_trim)} (dropped {len(df)-len(df_trim)})")
g2 = df_trim[df_trim['cat5'] == 1].copy()
ng2 = df_trim[df_trim['cat5'] == 0].copy()
res_g2 = run_probit(g2, games_xvars, label="Trim Games")
res_ng2 = run_probit(ng2, nongames_xvars, label="Trim Non-Games")
print_key_results(res_g2, "Trimmed Outliers: Games")
print_key_results(res_ng2, "Trimmed Outliers: Non-Games")


# ============================================================
# CHECK 3: Logit instead of Probit
# ============================================================
print("\n\n" + "=" * 70)
print("CHECK 3: Logit instead of Probit")
print("=" * 70)
for subset_name, sub, xvars in [("Games", games, games_xvars), ("Non-Games", nongames, nongames_xvars)]:
    y = sub[DEPVAR].astype(float)
    X = sm.add_constant(sub[xvars].astype(float))
    model = Logit(y, X)
    res = model.fit(cov_type='cluster', cov_kwds={'groups': sub['cohort'].astype(int)}, disp=0, maxiter=300)
    print(f"\n  Logit: {subset_name} (N={res.nobs:.0f}, Pseudo R2={res.prsquared:.4f})")
    print(f"  {'Variable':<25}{'Coef':>10}{'SE':>10}{'P-val':>8}{'Sig':>6}")
    for v in REGRESSORS:
        c = res.params[v]
        se = res.bse[v]
        p = res.pvalues[v]
        sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
        print(f"  {VAR_LABELS.get(v, v):<25}{c:>10.4f}{se:>10.4f}{p:>8.3f}{sig:>6}")


# ============================================================
# CHECK 4: Drop apps with zero comments (potential missing data)
# ============================================================
print("\n\n" + "=" * 70)
print("CHECK 4: Restrict to apps with at least 1 comment")
print("=" * 70)
df_com = df[df['numcomapp'] > 0]
print(f"Restricted: {len(df)} -> {len(df_com)} (dropped {len(df)-len(df_com)} zero-comment apps)")
g4 = df_com[df_com['cat5'] == 1].copy()
ng4 = df_com[df_com['cat5'] == 0].copy()
res_g4 = run_probit(g4, games_xvars, label="HasComments Games")
res_ng4 = run_probit(ng4, nongames_xvars, label="HasComments Non-Games")
print_key_results(res_g4, "Has Comments: Games")
print_key_results(res_ng4, "Has Comments: Non-Games")


# ============================================================
# CHECK 5: Alternative clustering (no clustering / robust only)
# ============================================================
print("\n\n" + "=" * 70)
print("CHECK 5: Robust SEs without clustering (HC1)")
print("=" * 70)
for subset_name, sub, xvars in [("Games", games, games_xvars), ("Non-Games", nongames, nongames_xvars)]:
    y = sub[DEPVAR].astype(float)
    X = sm.add_constant(sub[xvars].astype(float))
    model = Probit(y, X)
    res = model.fit(cov_type='HC1', disp=0, maxiter=300)
    print(f"\n  HC1 Robust: {subset_name} (N={res.nobs:.0f}, Pseudo R2={res.prsquared:.4f})")
    print(f"  {'Variable':<25}{'Coef':>10}{'SE':>10}{'P-val':>8}{'Sig':>6}")
    for v in REGRESSORS:
        c = res.params[v]
        se = res.bse[v]
        p = res.pvalues[v]
        sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
        print(f"  {VAR_LABELS.get(v, v):<25}{c:>10.4f}{se:>10.4f}{p:>8.3f}{sig:>6}")


# ============================================================
# CHECK 6: Drop largest cohort (cohort 18)
# ============================================================
print("\n\n" + "=" * 70)
print("CHECK 6: Drop largest cohort (cohort 18)")
print("=" * 70)
df_no18 = df[df['cohort'] != 18]
print(f"Dropped cohort 18: {len(df)} -> {len(df_no18)}")
g6 = df_no18[df_no18['cat5'] == 1].copy()
ng6 = df_no18[df_no18['cat5'] == 0].copy()
# Adjust cohort dummies (remove cohort18 which is already dropped as reference, keep list same)
res_g6 = run_probit(g6, games_xvars, label="NoCoh18 Games")
res_ng6 = run_probit(ng6, nongames_xvars, label="NoCoh18 Non-Games")
print_key_results(res_g6, "Drop Cohort 18: Games")
print_key_results(res_ng6, "Drop Cohort 18: Non-Games")


# ============================================================
# CHECK 7: Restrict to early cohorts only (cohorts 1-19)
# ============================================================
print("\n\n" + "=" * 70)
print("CHECK 7: Restrict to early cohorts (1-19)")
print("=" * 70)
df_early = df[df['cohort'] <= 19]
print(f"Early cohorts: {len(df)} -> {len(df_early)}")
g7 = df_early[df_early['cat5'] == 1].copy()
ng7 = df_early[df_early['cat5'] == 0].copy()
early_cohorts = [f'cohort{i}' for i in range(1, 20) if i not in (2, 18)]
res_g7 = run_probit(g7, REGRESSORS + early_cohorts, label="Early Games")
res_ng7 = run_probit(ng7, REGRESSORS + early_cohorts + CAT_DUMMIES_NONGAMES, label="Early Non-Games")
print_key_results(res_g7, "Early Cohorts: Games")
print_key_results(res_ng7, "Early Cohorts: Non-Games")


# ============================================================
# CHECK 8: Restrict to late cohorts only (cohorts 20-38)
# ============================================================
print("\n\n" + "=" * 70)
print("CHECK 8: Restrict to late cohorts (20-38)")
print("=" * 70)
df_late = df[df['cohort'] >= 20]
print(f"Late cohorts: {len(df)} -> {len(df_late)}")
g8 = df_late[df_late['cat5'] == 1].copy()
ng8 = df_late[df_late['cat5'] == 0].copy()
late_cohorts = [f'cohort{i}' for i in range(20, 39)]
res_g8 = run_probit(g8, REGRESSORS + late_cohorts, label="Late Games")
res_ng8 = run_probit(ng8, REGRESSORS + late_cohorts + CAT_DUMMIES_NONGAMES, label="Late Non-Games")
print_key_results(res_g8, "Late Cohorts: Games")
print_key_results(res_ng8, "Late Cohorts: Non-Games")


# ============================================================
# CHECK 9: Winsorize continuous variables at 1st/99th percentile
# ============================================================
print("\n\n" + "=" * 70)
print("CHECK 9: Winsorize continuous variables at 1st/99th percentile")
print("=" * 70)
df_w = df.copy()
for v in ['countapp', 'numverapp', 'avdeltatime', 'avprice', 'avsize', 'lnnumcomapp']:
    lo, hi = df_w[v].quantile(0.01), df_w[v].quantile(0.99)
    df_w[v] = df_w[v].clip(lo, hi)
    print(f"  {v}: clipped to [{lo:.2f}, {hi:.2f}]")
g9 = df_w[df_w['cat5'] == 1].copy()
ng9 = df_w[df_w['cat5'] == 0].copy()
res_g9 = run_probit(g9, games_xvars, label="Winsor Games")
res_ng9 = run_probit(ng9, nongames_xvars, label="Winsor Non-Games")
print_key_results(res_g9, "Winsorized: Games")
print_key_results(res_ng9, "Winsorized: Non-Games")


# ============================================================
# CHECK 10: Placebo test - shuffle killerappgros within category
# ============================================================
print("\n\n" + "=" * 70)
print("CHECK 10: Placebo test - shuffle killer status within game/non-game")
print("=" * 70)
np.random.seed(42)
n_perm = 100
sig_count_g = {v: 0 for v in REGRESSORS}
sig_count_ng = {v: 0 for v in REGRESSORS}

for i in range(n_perm):
    df_p = df.copy()
    # Shuffle within games
    mask_g = df_p['cat5'] == 1
    df_p.loc[mask_g, 'killerappgros'] = np.random.permutation(df_p.loc[mask_g, 'killerappgros'].values)
    mask_ng = df_p['cat5'] == 0
    df_p.loc[mask_ng, 'killerappgros'] = np.random.permutation(df_p.loc[mask_ng, 'killerappgros'].values)

    gp = df_p[df_p['cat5'] == 1].copy()
    ngp = df_p[df_p['cat5'] == 0].copy()
    try:
        rg = run_probit(gp, games_xvars, label=f"Perm{i} Games")
        if rg is not None:
            for v in REGRESSORS:
                if rg.pvalues[v] < 0.05:
                    sig_count_g[v] += 1
    except:
        pass
    try:
        rng = run_probit(ngp, nongames_xvars, label=f"Perm{i} NG")
        if rng is not None:
            for v in REGRESSORS:
                if rng.pvalues[v] < 0.05:
                    sig_count_ng[v] += 1
    except:
        pass

print(f"\n  Fraction significant (p<0.05) in {n_perm} permutations:")
print(f"  {'Variable':<25}{'Games':>10}{'Non-Games':>12}")
for v in REGRESSORS:
    print(f"  {VAR_LABELS.get(v, v):<25}{sig_count_g[v]/n_perm:>10.2f}{sig_count_ng[v]/n_perm:>12.2f}")
print(f"  Expected under null: ~0.05")


# ============================================================
# SUMMARY
# ============================================================
print("\n\n" + "=" * 70)
print("ROBUSTNESS SUMMARY")
print("=" * 70)
print("""
Key findings across robustness checks:

Games model key results (App Order +, No Updates +):
- Check 1 (drop duplicates): [see above]
- Check 2 (drop outliers): [see above]
- Check 3 (logit): [see above]
- Check 4 (drop zero-comment apps): [see above]
- Check 5 (HC1 SEs): [see above]
- Check 6 (drop cohort 18): [see above]
- Check 7 (early cohorts): [see above]
- Check 8 (late cohorts): [see above]
- Check 9 (winsorized): [see above]
- Check 10 (placebo): rejection rates above

Non-Games model key results (Number of Versions +, Time Between Versions -):
- Same checks applied above.

See detailed output above for full coefficient comparisons.
""")
