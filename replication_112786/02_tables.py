"""
02_tables.py - Replicate Tables 2 and 3
Rosenzweig & Udry (2014) - Rainfall Forecasts, Weather and Wages
"""
import sys
sys.path.insert(0, '.')
from utils import *

harvest, planting, migration = load_data()

# ============================================================
# Table 2: Migration regressions (village FE)
# Stata: areg mig sp age agesq ed male, a(vill)
# Note: Stata abbreviates 'edu' as 'ed'
# ============================================================

print("=" * 70)
print("TABLE 2: Village FE Estimates of Planting-Stage Out-Migration")
print("(ICRISAT VLS India: 2005-2011)")
print("=" * 70)

# Column 1: All workers
res_all = run_areg(migration, 'mig', ['sp', 'age', 'agesq', 'edu', 'male'], 'vill')
print_results(res_all, "Column 1: All Workers")

# Column 2: Males only
males = migration[migration['male'] == 1].copy()
res_males = run_areg(males, 'mig', ['sp', 'age', 'agesq', 'edu'], 'vill')
print_results(res_males, "Column 2: Males")

# Column 3: Females only
females = migration[migration['male'] == 0].copy()
res_females = run_areg(females, 'mig', ['sp', 'age', 'agesq', 'edu'], 'vill')
print_results(res_females, "Column 3: Females")

# Comparison with published values
print("\n" + "=" * 70)
print("TABLE 2 COMPARISON WITH PUBLISHED VALUES")
print("=" * 70)

published_t2 = {
    'All Workers': {
        'sp': (-.00293, 2.69), 'age': (.000853, 1.32), 'agesq': (-.0000194, 2.38),
        'edu': (.00711, 10.0), 'male': (.0557, 9.58)
    },
    'Males': {
        'sp': (-.00485, 2.68), 'age': (.00141, 1.32), 'agesq': (-.0000314, 2.28),
        'edu': (.0118, 10.9)
    },
    'Females': {
        'sp': (-.000332, 0.33), 'age': (-.000195, 0.32), 'agesq': (-.0000683, 0.91),
        'edu': (-.00184, 2.47)
    }
}

for label, res in [('All Workers', res_all), ('Males', res_males), ('Females', res_females)]:
    print(f"\n{label} (N={res['n']}):")
    pub = published_t2[label]
    for var in pub:
        pub_coef, pub_t = pub[var]
        my_coef = res['coef'][var]
        my_t = abs(res['tstat'][var])
        coef_diff = abs(my_coef - pub_coef)
        t_diff = abs(my_t - pub_t)
        match = "MATCH" if coef_diff < 0.0001 and t_diff < 0.1 else "CLOSE" if coef_diff < 0.001 else "DIFF"
        print(f"  {var:<10} Pub: {pub_coef:>12.6f} ({pub_t:.2f})  "
              f"Rep: {my_coef:>12.6f} ({my_t:.2f})  [{match}]")


# ============================================================
# Table 3: Wage regressions (district FE)
# ============================================================

print("\n" + "=" * 70)
print("TABLE 3: District FE Estimates of Male Daily Wages")
print("(Agricultural Wages of India, 2005-2010)")
print("=" * 70)

# Column 1: Planting-stage wages
# Stata: areg lwager jsdev fore jsdevmiles post, a(id)
res_plant = run_areg(planting, 'lwager', ['jsdev', 'fore', 'jsdevmiles', 'post'], 'id')
print_results(res_plant, "Log Planting-Stage Wage")

# Column 2: Harvest-stage wages
# Stata: areg lwager jsdev foregood fore jsdevmiles post postbad, a(id)
res_harv = run_areg(harvest, 'lwager', ['jsdev', 'foregood', 'fore', 'jsdevmiles', 'post', 'postbad'], 'id')
print_results(res_harv, "Log Harvest-Stage Wage")

# Comparison with published Table 3
print("\n" + "=" * 70)
print("TABLE 3 COMPARISON WITH PUBLISHED VALUES")
print("=" * 70)

# Published Table 3 mapping:
# Planting: Rainfall shock=jsdev, Rainfall shock x distance=jsdevmiles,
#           Forecast=fore, NREGA=post
# Harvest: Rainfall shock=jsdev, Rainfall shock x distance=jsdevmiles,
#          Forecast=fore, Forecast x bad rainfall shock=foregood,
#          NREGA=post, NREGA x bad rainfall shock=postbad

pub_plant = {
    'jsdev': (.192, 3.09), 'jsdevmiles': (-.00196, 2.56),
    'fore': (-.00622, 2.71), 'post': (.096, 4.50)
}
pub_harv = {
    'jsdev': (.056, 0.73), 'jsdevmiles': (-.00152, 1.72),
    'fore': (.000898, 0.30), 'foregood': (-.00109, 2.09),
    'post': (.0623, 1.76), 'postbad': (.0929, 1.78)
}

print("\nPlanting-Stage Wage (N=387):")
for var in pub_plant:
    pub_coef, pub_t = pub_plant[var]
    my_coef = res_plant['coef'][var]
    my_t = abs(res_plant['tstat'][var])
    coef_diff = abs(my_coef - pub_coef)
    t_diff = abs(my_t - pub_t)
    match = "MATCH" if coef_diff < 0.001 and t_diff < 0.1 else "CLOSE" if coef_diff < 0.01 else "DIFF"
    print(f"  {var:<15} Pub: {pub_coef:>12.6f} ({pub_t:.2f})  "
          f"Rep: {my_coef:>12.6f} ({my_t:.2f})  [{match}]")

print(f"\nHarvest-Stage Wage (N=337):")
for var in pub_harv:
    pub_coef, pub_t = pub_harv[var]
    my_coef = res_harv['coef'][var]
    my_t = abs(res_harv['tstat'][var])
    coef_diff = abs(my_coef - pub_coef)
    t_diff = abs(my_t - pub_t)
    match = "MATCH" if coef_diff < 0.001 and t_diff < 0.1 else "CLOSE" if coef_diff < 0.01 else "DIFF"
    print(f"  {var:<15} Pub: {pub_coef:>12.6f} ({pub_t:.2f})  "
          f"Rep: {my_coef:>12.6f} ({my_t:.2f})  [{match}]")
