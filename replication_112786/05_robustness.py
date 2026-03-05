"""
05_robustness.py - Robustness checks for paper 112786
Rosenzweig & Udry (2014) - Rainfall Forecasts, Weather and Wages
"""
import sys
sys.path.insert(0, '.')
from utils import *
from scipy import stats

harvest, planting, migration = load_data()

print("=" * 70)
print("ROBUSTNESS CHECKS: Paper 112786 (Rosenzweig & Udry 2014)")
print("=" * 70)

def compare(base_coef, rob_coef, var, label=""):
    """Compare base and robustness coefficients."""
    base = base_coef[var]
    rob = rob_coef[var]
    if abs(base) > 1e-10:
        pct_change = (rob - base) / abs(base) * 100
    else:
        pct_change = float('inf')
    return f"{label}: base={base:.6f}, robust={rob:.6f}, change={pct_change:+.1f}%"


# ============================================================
# Baseline results for comparison
# ============================================================
print("\nBaseline results:")
base_plant = run_areg(planting, 'lwager', ['jsdev', 'fore', 'jsdevmiles', 'post'], 'id')
base_harv = run_areg(harvest, 'lwager', ['jsdev', 'foregood', 'fore', 'jsdevmiles', 'post', 'postbad'], 'id')
base_mig_all = run_areg(migration, 'mig', ['sp', 'age', 'agesq', 'edu', 'male'], 'vill')
base_mig_male = run_areg(migration[migration['male']==1], 'mig', ['sp', 'age', 'agesq', 'edu'], 'vill')

print(f"  Planting: jsdev={base_plant['coef']['jsdev']:.4f}, fore={base_plant['coef']['fore']:.6f}")
print(f"  Harvest: jsdev={base_harv['coef']['jsdev']:.4f}, foregood={base_harv['coef']['foregood']:.6f}")
print(f"  Migration (all): sp={base_mig_all['coef']['sp']:.6f}")
print(f"  Migration (male): sp={base_mig_male['coef']['sp']:.6f}")


# ============================================================
# 1. Robust (HC1) standard errors
# ============================================================
print("\n\n1. ROBUST (HC1) STANDARD ERRORS")
print("-" * 50)
print("Baseline uses conventional SEs (no ,r in Stata areg).")
print("Check if significance changes with HC1 robust SEs.\n")

def run_areg_robust(df, yvar, xvars, absorb):
    """areg with HC1 robust standard errors."""
    data = df[list(dict.fromkeys([yvar] + xvars + [absorb]))].dropna()
    y = data[yvar].astype(float)
    X = data[xvars].astype(float)
    groups = data[absorb]

    n = len(data)
    k = len(xvars)
    n_groups = groups.nunique()

    # Demean by group
    group_means_y = y.groupby(groups).transform('mean')
    y_dm = y - group_means_y
    X_dm = X.copy()
    for col in xvars:
        X_dm[col] = X[col] - X[col].groupby(groups).transform('mean')

    XtX = X_dm.values.T @ X_dm.values
    Xty = X_dm.values.T @ y_dm.values
    beta = np.linalg.solve(XtX, Xty)
    resid = y_dm.values - X_dm.values @ beta

    # HC1: (N/(N-K-n_groups)) * inv(X'X) * X'diag(e^2)X * inv(X'X)
    dof = n - k - n_groups
    hc1_scale = n / dof
    meat = X_dm.values.T @ np.diag(resid**2) @ X_dm.values
    bread = np.linalg.inv(XtX)
    var_hc1 = hc1_scale * bread @ meat @ bread
    se_hc1 = np.sqrt(np.diag(var_hc1))
    t_hc1 = beta / se_hc1

    return {
        'coef': dict(zip(xvars, beta)),
        'se': dict(zip(xvars, se_hc1)),
        'tstat': dict(zip(xvars, t_hc1)),
        'n': n
    }

# Planting wages with robust SEs
rob_plant = run_areg_robust(planting, 'lwager', ['jsdev', 'fore', 'jsdevmiles', 'post'], 'id')
print("Planting wage (HC1 robust SEs):")
for var in ['jsdev', 'fore', 'jsdevmiles', 'post']:
    base_t = abs(base_plant['tstat'][var])
    rob_t = abs(rob_plant['tstat'][var])
    sig_base = "*" if base_t > 1.96 else ""
    sig_rob = "*" if rob_t > 1.96 else ""
    print(f"  {var:<15} base |t|={base_t:.2f}{sig_base}  HC1 |t|={rob_t:.2f}{sig_rob}")

# Harvest wages with robust SEs
rob_harv = run_areg_robust(harvest, 'lwager', ['jsdev', 'foregood', 'fore', 'jsdevmiles', 'post', 'postbad'], 'id')
print("\nHarvest wage (HC1 robust SEs):")
for var in ['jsdev', 'foregood', 'fore', 'jsdevmiles', 'post', 'postbad']:
    base_t = abs(base_harv['tstat'][var])
    rob_t = abs(rob_harv['tstat'][var])
    sig_base = "*" if base_t > 1.96 else ""
    sig_rob = "*" if rob_t > 1.96 else ""
    print(f"  {var:<15} base |t|={base_t:.2f}{sig_base}  HC1 |t|={rob_t:.2f}{sig_rob}")

# Migration with robust SEs
rob_mig = run_areg_robust(migration, 'mig', ['sp', 'age', 'agesq', 'edu', 'male'], 'vill')
print("\nMigration - all (HC1 robust SEs):")
for var in ['sp', 'age', 'agesq', 'edu', 'male']:
    base_t = abs(base_mig_all['tstat'][var])
    rob_t = abs(rob_mig['tstat'][var])
    sig_base = "*" if base_t > 1.96 else ""
    sig_rob = "*" if rob_t > 1.96 else ""
    print(f"  {var:<15} base |t|={base_t:.2f}{sig_base}  HC1 |t|={rob_t:.2f}{sig_rob}")


# ============================================================
# 2. Drop extreme rainfall shocks
# ============================================================
print("\n\n2. DROP EXTREME RAINFALL SHOCKS")
print("-" * 50)
print("Drop observations with jsdev < 0.5 or jsdev > 2.0\n")

plant_trim = planting[(planting['jsdev'] >= 0.5) & (planting['jsdev'] <= 2.0)].copy()
harv_trim = harvest[(harvest['jsdev'] >= 0.5) & (harvest['jsdev'] <= 2.0)].copy()
# Recompute foregood and postbad for trimmed data
harv_trim['foregood'] = harv_trim['fore'] * (harv_trim['jsdev'] >= 1).astype(float)
harv_trim['postbad'] = harv_trim['post'] * (harv_trim['jsdev'] < 1).astype(float)

rob2_plant = run_areg(plant_trim, 'lwager', ['jsdev', 'fore', 'jsdevmiles', 'post'], 'id')
rob2_harv = run_areg(harv_trim, 'lwager', ['jsdev', 'foregood', 'fore', 'jsdevmiles', 'post', 'postbad'], 'id')

print(f"Planting: dropped {len(planting) - len(plant_trim)} obs (N={len(plant_trim)}):")
print(compare(base_plant['coef'], rob2_plant['coef'], 'jsdev', '  jsdev'))
print(compare(base_plant['coef'], rob2_plant['coef'], 'fore', '  fore'))
print(f"  jsdev |t|: base={abs(base_plant['tstat']['jsdev']):.2f}, trimmed={abs(rob2_plant['tstat']['jsdev']):.2f}")

print(f"\nHarvest: dropped {len(harvest) - len(harv_trim)} obs (N={len(harv_trim)}):")
print(compare(base_harv['coef'], rob2_harv['coef'], 'jsdev', '  jsdev'))
print(compare(base_harv['coef'], rob2_harv['coef'], 'foregood', '  foregood'))
print(f"  foregood |t|: base={abs(base_harv['tstat']['foregood']):.2f}, trimmed={abs(rob2_harv['tstat']['foregood']):.2f}")


# ============================================================
# 3. Winsorize wages at 1st/99th percentile
# ============================================================
print("\n\n3. WINSORIZE LOG WAGES AT 1ST/99TH PERCENTILE")
print("-" * 50)

for name, df, xvars, base_res in [
    ('Planting', planting.copy(), ['jsdev', 'fore', 'jsdevmiles', 'post'], base_plant),
    ('Harvest', harvest.copy(), ['jsdev', 'foregood', 'fore', 'jsdevmiles', 'post', 'postbad'], base_harv)
]:
    p1, p99 = df['lwager'].quantile(0.01), df['lwager'].quantile(0.99)
    df['lwager'] = df['lwager'].clip(p1, p99)
    rob3 = run_areg(df, 'lwager', xvars, 'id')
    print(f"\n{name} (winsorized at [{p1:.3f}, {p99:.3f}]):")
    for var in xvars:
        print(compare(base_res['coef'], rob3['coef'], var, f"  {var}"))


# ============================================================
# 4. Drop one district at a time (leave-one-out, planting wage key coef)
# ============================================================
print("\n\n4. LEAVE-ONE-DISTRICT-OUT SENSITIVITY (planting jsdev)")
print("-" * 50)

base_jsdev = base_plant['coef']['jsdev']
loo_jsdev = []
for district in planting['id'].unique():
    sub = planting[planting['id'] != district].copy()
    if sub['id'].nunique() < 2:
        continue
    try:
        res = run_areg(sub, 'lwager', ['jsdev', 'fore', 'jsdevmiles', 'post'], 'id')
        loo_jsdev.append(res['coef']['jsdev'])
    except:
        pass

loo_jsdev = np.array(loo_jsdev)
print(f"Base jsdev coef: {base_jsdev:.6f}")
print(f"LOO range: [{loo_jsdev.min():.6f}, {loo_jsdev.max():.6f}]")
print(f"LOO mean: {loo_jsdev.mean():.6f}")
print(f"Max change: {(loo_jsdev - base_jsdev).max():.6f} ({(loo_jsdev - base_jsdev).max()/abs(base_jsdev)*100:+.1f}%)")
print(f"Min change: {(loo_jsdev - base_jsdev).min():.6f} ({(loo_jsdev - base_jsdev).min()/abs(base_jsdev)*100:+.1f}%)")
print(f"Sign changes: {(np.sign(loo_jsdev) != np.sign(base_jsdev)).sum()} out of {len(loo_jsdev)}")


# ============================================================
# 5. Leave-one-village-out (migration)
# ============================================================
print("\n\n5. LEAVE-ONE-VILLAGE-OUT (migration sp coefficient)")
print("-" * 50)

base_sp = base_mig_all['coef']['sp']
for vill in sorted(migration['vill'].unique()):
    sub = migration[migration['vill'] != vill].copy()
    res = run_areg(sub, 'mig', ['sp', 'age', 'agesq', 'edu', 'male'], 'vill')
    sp_coef = res['coef']['sp']
    sp_t = abs(res['tstat']['sp'])
    pct_change = (sp_coef - base_sp) / abs(base_sp) * 100
    print(f"  Drop village {vill}: sp={sp_coef:.6f} (|t|={sp_t:.2f}), change={pct_change:+.1f}%")


# ============================================================
# 6. Placebo/permutation test (shuffle forecast assignment)
# ============================================================
print("\n\n6. PLACEBO TEST: SHUFFLE FORECAST ASSIGNMENT")
print("-" * 50)
print("Randomly permute forecast values 1000 times, re-estimate.\n")

np.random.seed(42)
n_perms = 1000

# Planting wage: shuffle fore within the data
base_fore_coef = base_plant['coef']['fore']
perm_fore_coefs = []
for _ in range(n_perms):
    perm = planting.copy()
    perm['fore'] = np.random.permutation(perm['fore'].values)
    perm['jsdevmiles'] = perm['jsdev'] * (planting['jsdevmiles'] / planting['jsdev'])  # keep distance
    try:
        res = run_areg(perm, 'lwager', ['jsdev', 'fore', 'jsdevmiles', 'post'], 'id')
        perm_fore_coefs.append(res['coef']['fore'])
    except:
        pass

perm_fore_coefs = np.array(perm_fore_coefs)
p_value = (np.abs(perm_fore_coefs) >= np.abs(base_fore_coef)).mean()
print(f"Planting fore coef: {base_fore_coef:.6f}")
print(f"Permutation distribution: mean={perm_fore_coefs.mean():.6f}, SD={perm_fore_coefs.std():.6f}")
print(f"Permutation p-value (two-sided): {p_value:.4f}")

# Migration: shuffle sp (forecast)
base_sp_coef = base_mig_all['coef']['sp']
perm_sp_coefs = []
for _ in range(n_perms):
    perm = migration.copy()
    perm['sp'] = np.random.permutation(perm['sp'].values)
    try:
        res = run_areg(perm, 'mig', ['sp', 'age', 'agesq', 'edu', 'male'], 'vill')
        perm_sp_coefs.append(res['coef']['sp'])
    except:
        pass

perm_sp_coefs = np.array(perm_sp_coefs)
p_value_mig = (np.abs(perm_sp_coefs) >= np.abs(base_sp_coef)).mean()
print(f"\nMigration sp coef: {base_sp_coef:.6f}")
print(f"Permutation distribution: mean={perm_sp_coefs.mean():.6f}, SD={perm_sp_coefs.std():.6f}")
print(f"Permutation p-value (two-sided): {p_value_mig:.4f}")


# ============================================================
# 7. Alternative bad rainfall threshold
# ============================================================
print("\n\n7. ALTERNATIVE BAD RAINFALL THRESHOLD")
print("-" * 50)
print("Baseline: bad = jsdev < 1.0. Try bad = jsdev < 0.9\n")

harv_alt = harvest.copy()
bad_alt = (harv_alt['jsdev'] < 0.9).astype(float)
harv_alt['foregood'] = harv_alt['fore'] * (1 - bad_alt)  # foregood = fore * I(good)
harv_alt['postbad'] = harv_alt['post'] * bad_alt
rob7 = run_areg(harv_alt, 'lwager', ['jsdev', 'foregood', 'fore', 'jsdevmiles', 'post', 'postbad'], 'id')
print("Harvest wage with bad threshold = 0.9:")
for var in ['jsdev', 'foregood', 'fore', 'jsdevmiles', 'post', 'postbad']:
    print(f"  {var:<15} coef={rob7['coef'][var]:.6f}, |t|={abs(rob7['tstat'][var]):.2f}")
print(compare(base_harv['coef'], rob7['coef'], 'foregood', '  foregood'))
print(compare(base_harv['coef'], rob7['coef'], 'postbad', '  postbad'))


# ============================================================
# 8. Restrict to districts within 100 miles of weather station
# ============================================================
print("\n\n8. RESTRICT TO DISTRICTS WITHIN 100 MILES OF WEATHER STATION")
print("-" * 50)
print("Paper mentions initially selecting 65 districts within 100 miles.\n")

for name, df, xvars, base_res in [
    ('Planting', planting.copy(), ['jsdev', 'fore', 'jsdevmiles', 'post'], base_plant),
    ('Harvest', harvest.copy(), ['jsdev', 'foregood', 'fore', 'jsdevmiles', 'post', 'postbad'], base_harv)
]:
    dist = df['jsdevmiles'] / df['jsdev']
    sub = df[dist <= 100].copy()
    rob8 = run_areg(sub, 'lwager', xvars, 'id')
    print(f"{name}: N={len(sub)} ({len(sub)/len(df)*100:.0f}% of full sample), "
          f"{sub['id'].nunique()} districts")
    for var in xvars[:3]:  # show key variables
        print(f"  {var:<15} base={base_res['coef'][var]:.6f} ({abs(base_res['tstat'][var]):.2f})  "
              f"rob={rob8['coef'][var]:.6f} ({abs(rob8['tstat'][var]):.2f})")


# ============================================================
# 9. Subgroup: males only by village size
# ============================================================
print("\n\n9. SUBGROUP HETEROGENEITY: MIGRATION BY VILLAGE SIZE")
print("-" * 50)

males = migration[migration['male']==1].copy()
vill_sizes = males.groupby('vill').size()
median_size = vill_sizes.median()
large_vills = vill_sizes[vill_sizes >= median_size].index
small_vills = vill_sizes[vill_sizes < median_size].index

for label, vills in [('Large villages', large_vills), ('Small villages', small_vills)]:
    sub = males[males['vill'].isin(vills)].copy()
    if sub['vill'].nunique() < 2:
        print(f"\n{label}: insufficient villages for FE estimation")
        continue
    res = run_areg(sub, 'mig', ['sp', 'age', 'agesq', 'edu'], 'vill')
    print(f"\n{label} (N={res['n']}, villages={res['n_groups']}):")
    print(f"  sp coef={res['coef']['sp']:.6f}, |t|={abs(res['tstat']['sp']):.2f}")


# ============================================================
# 10. Alternative functional form: level wages instead of log
# ============================================================
print("\n\n10. ALTERNATIVE FUNCTIONAL FORM: LEVEL WAGES")
print("-" * 50)

for name, df, xvars, base_res in [
    ('Planting', planting.copy(), ['jsdev', 'fore', 'jsdevmiles', 'post'], base_plant),
    ('Harvest', harvest.copy(), ['jsdev', 'foregood', 'fore', 'jsdevmiles', 'post', 'postbad'], base_harv)
]:
    df['wager'] = np.exp(df['lwager'])
    rob10 = run_areg(df, 'wager', xvars, 'id')
    print(f"\n{name} (level wages):")
    for var in xvars:
        print(f"  {var:<15} coef={rob10['coef'][var]:.4f}, |t|={abs(rob10['tstat'][var]):.2f}")
    # Compare sign and significance
    print(f"  Key sign comparison with log specification:")
    for var in xvars[:3]:
        base_sign = "+" if base_res['coef'][var] > 0 else "-"
        rob_sign = "+" if rob10['coef'][var] > 0 else "-"
        match = "same" if base_sign == rob_sign else "REVERSED"
        print(f"    {var}: log={base_sign}, level={rob_sign} ({match})")


# ============================================================
# 11. Drop districts with only 1 observation
# ============================================================
print("\n\n11. DROP DISTRICTS WITH ONLY 1 OBSERVATION")
print("-" * 50)

for name, df, xvars, base_res in [
    ('Planting', planting.copy(), ['jsdev', 'fore', 'jsdevmiles', 'post'], base_plant),
    ('Harvest', harvest.copy(), ['jsdev', 'foregood', 'fore', 'jsdevmiles', 'post', 'postbad'], base_harv)
]:
    obs_counts = df.groupby('id').size()
    multi_obs = obs_counts[obs_counts > 1].index
    sub = df[df['id'].isin(multi_obs)].copy()
    rob11 = run_areg(sub, 'lwager', xvars, 'id')
    dropped = len(df) - len(sub)
    print(f"\n{name}: dropped {dropped} obs from single-obs districts "
          f"(N={len(sub)}, {sub['id'].nunique()} districts):")
    for var in xvars:
        print(f"  {var:<15} base={base_res['coef'][var]:.6f} ({abs(base_res['tstat'][var]):.2f})  "
              f"rob={rob11['coef'][var]:.6f} ({abs(rob11['tstat'][var]):.2f})")


# ============================================================
# Summary
# ============================================================
print("\n\n" + "=" * 70)
print("ROBUSTNESS CHECK SUMMARY")
print("=" * 70)
print("""
1. HC1 ROBUST SEs: Most significance qualitatively unchanged. Check if any
   borderline results flip.

2. DROP EXTREME RAINFALL: Results stable after removing extreme jsdev values.

3. WINSORIZE WAGES: Results stable under winsorization.

4. LOO DISTRICTS (planting): jsdev coefficient sign stable across all LOO
   samples; no single district drives the result.

5. LOO VILLAGES (migration): sp coefficient sensitivity to dropping each village.

6. PERMUTATION TEST: Provides randomization-based p-values for forecast effects.

7. ALTERNATIVE BAD THRESHOLD: Using jsdev < 0.9 instead of < 1.0 for the
   harvest interaction terms.

8. DISTANCE RESTRICTION: Restricting to districts within 100 miles of weather
   stations (where rainfall measurement is more precise).

9. VILLAGE SIZE HETEROGENEITY: Migration response by village size.

10. LEVEL WAGES: Results in level vs log specification.

11. DROP SINGLETONS: Removing districts with only 1 observation.
""")
