"""
02_tables.py - Reproduce Table 1, Table A1, Table A2, Table A3.
"""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
from utils import *

# ===================================================================
# TABLE 1: Summary Statistics
# ===================================================================
print("=" * 80)
print("TABLE 1: Characteristics of US and Indian Coal-Fired Power Plants")
print("=" * 80)

# --- US sample: apply regressioncode.do filters ---
us_raw = load_us_data()
us_raw = us_raw[us_raw['heatrate'].notna() &
                (us_raw['heatrate'] >= 6000) & (us_raw['heatrate'] <= 20000)].copy()
us_raw = us_raw[us_raw['w_vintage'].notna() & us_raw['vintage'].notna() &
                (us_raw['w_vintage'] >= 1910) & (us_raw['vintage'] >= 1910)].copy()
us_raw = us_raw[~((us_raw['combinedheatpower'] == 1) |
                   (us_raw['eiasectornumber'] > 3))].copy()
us_raw['capfactor_pct'] = us_raw['capfactor'] * 100
us_raw['age_cw'] = us_raw['w_age'] + 1  # capacity-weighted age corrected
us_raw['vintage_cw'] = us_raw['w_vintage']
# Filter to nameplate >= 25 and key vars non-missing
us_t1 = us_raw[
    (us_raw['nameplate'] >= 25) &
    us_raw['w_age'].notna() &
    us_raw['capfactor'].notna() &
    us_raw['private'].notna() &
    (us_raw['year'] < 2010)
].copy()

# --- India sample ---
india_raw = load_india_data()
india_state = india_raw[india_raw['private'] == 0].copy()
india_state['has_ophr'] = (india_state['age'].notna() &
                           india_state['heatrate'].notna() &
                           india_state['nameplate'].notna() &
                           india_state['capfactor'].notna() &
                           india_state['vintage'].notna())


def print_stats(data, var, label, pub_mean=None, pub_median=None, pub_sd=None,
                pub_n=None):
    vals = data[var].dropna()
    n = len(vals)
    if n == 0:
        print(f"  {label:30s}  {'—':>8s}  {'—':>8s}  {'—':>8s}  {0:>5d}", end="")
    else:
        m, med, sd = vals.mean(), vals.median(), vals.std()
        print(f"  {label:30s}  {m:8.2f}  {med:8.2f}  {sd:8.2f}  {n:5d}", end="")
    if pub_mean is not None and n > 0:
        print(f"   | pub: {pub_mean:8.2f}  {pub_median:8.2f}  {pub_sd:8.2f}  {pub_n:5d}", end="")
        pct_diff = (m - pub_mean) / abs(pub_mean) * 100 if pub_mean != 0 else 0
        print(f"  [{pct_diff:+.1f}%]", end="")
    print()


for yr in [1988, 2009]:
    print(f"\n{'─' * 80}")
    print(f"  YEAR = {yr}")
    print(f"{'─' * 80}")
    print(f"  {'Variable':30s}  {'Mean':>8s}  {'Median':>8s}  {'SD':>8s}  {'N':>5s}")

    # US
    us_yr = us_t1[us_t1['year'] == yr]
    print(f"\n  US plants")
    print_stats(us_yr, 'age_cw', 'Age',
                21.88 if yr == 1988 else 40.32,
                22 if yr == 1988 else 41,
                10.48 if yr == 1988 else 11.86,
                406)
    print_stats(us_yr, 'nameplate', 'Capacity',
                765.2 if yr == 1988 else 795.1,
                529.7 if yr == 1988 else 566.6,
                690.5 if yr == 1988 else 729.2,
                406)
    print_stats(us_yr, 'vintage_cw', 'Vintage',
                1967 if yr == 1988 else 1970,
                1967 if yr == 1988 else 1969,
                10.48 if yr == 1988 else 11.86,
                406)
    print_stats(us_yr, 'capfactor_pct', 'Capacity factor (%)',
                51.10 if yr == 1988 else 52.52,
                53.19 if yr == 1988 else 55.80,
                19.58 if yr == 1988 else 22.39,
                406)
    print_stats(us_yr, 'btucontent', 'Heat content of coal',
                11077 if yr == 1988 else 10314,
                11655 if yr == 1988 else 10545,
                1801 if yr == 1988 else 1806,
                406 if yr == 1988 else 404)
    print_stats(us_yr, 'aux', 'Auxiliary gen (%)',
                None if yr == 1988 else 7.97,
                None if yr == 1988 else 7.32,
                None if yr == 1988 else 3.13,
                0 if yr == 1988 else 334)
    print_stats(us_yr, 'heatrate', 'Heat rate',
                11010 if yr == 1988 else 11326,
                10589 if yr == 1988 else 10746,
                1508 if yr == 1988 else 1836,
                406)

    # India with OPHR
    ind_yr = india_state[(india_state['year'] == yr) & india_state['has_ophr']]
    print(f"\n  India plants with OPHR")
    pub_n_hr = 38 if yr == 1988 else 36
    print_stats(ind_yr, 'age', 'Age',
                10.32 if yr == 1988 else 22.66,
                10.91 if yr == 1988 else 22.42,
                5.48 if yr == 1988 else 10.04,
                pub_n_hr)
    print_stats(ind_yr, 'nameplate', 'Capacity',
                535.6 if yr == 1988 else 809.0,
                435 if yr == 1988 else 840,
                339.4 if yr == 1988 else 506.5,
                pub_n_hr)
    print_stats(ind_yr, 'vintage', 'Vintage',
                1979 if yr == 1988 else 1987,
                1978 if yr == 1988 else 1988,
                5.48 if yr == 1988 else 10.04,
                pub_n_hr)
    print_stats(ind_yr, 'capfactor', 'Capacity factor (%)',
                50.75 if yr == 1988 else 71.35,
                52.33 if yr == 1988 else 75.88,
                14.88 if yr == 1988 else 18.31,
                pub_n_hr)
    pub_n_btu = 37 if yr == 1988 else 29
    print_stats(ind_yr, 'btucontent', 'Heat content of coal',
                7289 if yr == 1988 else 6431,
                7488 if yr == 1988 else 6485,
                1039 if yr == 1988 else 664.7,
                pub_n_btu)
    pub_n_aux = 37 if yr == 1988 else 36
    print_stats(ind_yr, 'aux', 'Auxiliary gen (%)',
                10.71 if yr == 1988 else 10.32,
                10.48 if yr == 1988 else 9.45,
                1.82 if yr == 1988 else 2.68,
                pub_n_aux)
    print_stats(ind_yr, 'heatrate', 'Heat rate',
                12355 if yr == 1988 else 11615,
                11962 if yr == 1988 else 10917,
                2409 if yr == 1988 else 2233,
                pub_n_hr)

    # India without OPHR
    ind_no = india_state[(india_state['year'] == yr) & ~india_state['has_ophr']]
    print(f"\n  India plants without OPHR")
    pub_n_no = 5 if yr == 1988 else 20
    print_stats(ind_no, 'age', 'Age',
                18.8 if yr == 1988 else 25.61,
                22 if yr == 1988 else 28.5,
                6.76 if yr == 1988 else 13.31,
                pub_n_no)
    print_stats(ind_no, 'nameplate', 'Capacity',
                154 if yr == 1988 else 615.5,
                200 if yr == 1988 else 385,
                101.4 if yr == 1988 else 502.3,
                pub_n_no)


# ===================================================================
# TABLE A1: Regression Results
# ===================================================================
print("\n" + "=" * 80)
print("TABLE A1: Regression-Based Average Treatment Effects")
print("=" * 80)

df = build_regression_sample()

# Model 1: constant india effect
year_dummies = [f'year{y}' for y in range(1989, 2010)]
controls = ['nameplate', 'nameplatesq', 'age', 'agesq', 'agecube']
x1 = ['india'] + controls + ['private', 'elec_private'] + year_dummies
model1, reg_df1 = run_ols_clustered(df, 'log_heatrate', x1, 'plantcode')

print(f"\nModel 1 (constant treatment effect):")
print(f"  N = {int(model1.nobs)}")
print(f"  Adj R² = {model1.rsquared_adj:.4f}")
print(f"  India coeff = {model1.params.get('india', 0):.4f} "
      f"(SE = {model1.bse.get('india', 0):.4f})")
print(f"  Published: India ≈ 0.094 (9.4% avg treatment effect)")

# Model 2: year-specific india effects
india_year = [f'indiayear{y}' for y in range(1988, 2010)]
x2 = india_year + controls + ['private', 'elec_private'] + year_dummies
model2, reg_df2 = run_ols_clustered(df, 'log_heatrate', x2, 'plantcode')

print(f"\nModel 2 (year-specific treatment effects):")
print(f"  N = {int(model2.nobs)}")
print(f"  Adj R² = {model2.rsquared_adj:.4f}")

# Extract year-specific coefficients
print(f"\n  Year-by-year treatment effects (published avg: 13.7% for 1988-91, "
      f"8.0% post-1997):")
ate_vals = {}
for y in range(1988, 2010):
    col = f'indiayear{y}'
    if col in model2.params.index:
        b = model2.params[col]
        se = model2.bse[col]
        if abs(b) > 1e-10:
            ate_vals[y] = (b, se)
            sig = '***' if model2.pvalues[col] < 0.01 else (
                '**' if model2.pvalues[col] < 0.05 else (
                '*' if model2.pvalues[col] < 0.10 else ''))
            print(f"    {y}: {b:.4f} ({se:.4f}){sig}")

# Compute period averages
early = [v[0] for y, v in ate_vals.items() if 1988 <= y <= 1991]
late = [v[0] for y, v in ate_vals.items() if y >= 1997]
all_ate = [v[0] for v in ate_vals.values()]
print(f"\n  Period averages:")
print(f"    1988-1991: {np.mean(early):.4f} (published: 0.137)")
print(f"    1997-2009: {np.mean(late):.4f} (published: 0.080)")
print(f"    Overall:   {np.mean(all_ate):.4f} (published: 0.094)")


# ===================================================================
# TABLE A2: NN Matching - Heat Rate
# ===================================================================
print("\n" + "=" * 80)
print("TABLE A2: Nearest Neighbor Matching - Heat Rate")
print("=" * 80)

mhr = build_matching_sample_heatrate()
print(f"\n  {'Year':>6s}  {'SATT':>8s}  {'SE':>8s}  {'N_India':>8s}  {'N_US':>8s}")

match_hr = {}
for y in range(1988, 2010):
    dy = mhr[(mhr['year'] == y)]
    n_ind = len(dy[dy['india'] == 1])
    n_us = len(dy[dy['india'] == 0])
    if n_ind > 0 and n_us > 0:
        satt, se = nn_match('log_heatrate', dy, ['age', 'np', 'private'], M=5)
        match_hr[y] = (satt, se)
        print(f"  {y:6d}  {satt:8.4f}  {se:8.4f}  {n_ind:8d}  {n_us:8d}")

early_m = [v[0] for y, v in match_hr.items() if 1988 <= y <= 1991]
late_m = [v[0] for y, v in match_hr.items() if y >= 1997]
all_m = [v[0] for v in match_hr.values()]
print(f"\n  Period averages:")
print(f"    1988-1991: {np.mean(early_m):.4f} (published: 0.152)")
print(f"    1997-2009: {np.mean(late_m):.4f} (published: 0.089)")
print(f"    Overall:   {np.mean(all_m):.4f} (published: 0.104)")


# ===================================================================
# TABLE A3: NN Matching - Auxiliary Generation
# ===================================================================
print("\n" + "=" * 80)
print("TABLE A3: Nearest Neighbor Matching - Auxiliary Generation")
print("=" * 80)

maux = build_matching_sample_aux()
print(f"\n  {'Year':>6s}  {'SATT':>8s}  {'SE':>8s}  {'N_India':>8s}  {'N_US':>8s}")

match_aux = {}
for y in range(1988, 2010):
    dy = maux[(maux['year'] == y)]
    # Also filter to non-missing heatrate (as in nnmatch_aux.do)
    dy = dy[dy['heatrate'].notna()]
    n_ind = len(dy[dy['india'] == 1])
    n_us = len(dy[dy['india'] == 0])
    if n_ind > 0 and n_us > 0:
        satt, se = nn_match('aux', dy, ['age', 'np', 'private'], M=5)
        match_aux[y] = (satt, se)
        print(f"  {y:6d}  {satt:8.4f}  {se:8.4f}  {n_ind:8d}  {n_us:8d}")

late_aux = [v[0] for y, v in match_aux.items() if y >= 1997]
print(f"\n  1997-2009 average: {np.mean(late_aux):.4f} (published: 3.48)")

print("\nDone.")
