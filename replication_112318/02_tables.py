"""
02_tables.py - Replicate Tables 2-5 from Urquiola (2005).
"""
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
sys.path.insert(0, '.')
from utils import *

# ============================================================
# TABLE 2: District availability and between level differences
# ============================================================
print("=" * 70)
print("TABLE 2: District availability and between level differences")
print("=" * 70)

ma = load_ma()
# Drop Miami and one other as in Stata code
ma_t2 = ma[~ma['ma9dn'].isin([5000, 5990])].copy()

# Column 1: Logit dumm on region dummies (robust)
print("\n--- Column 1: Logit (Dependent: Dummy for between-level differences) ---")
from statsmodels.discrete.discrete_model import Logit
y_col = 'dumm'
x_cols_1 = ['rN', 'rS', 'rW']
data1 = ma_t2[[y_col] + x_cols_1].dropna()
logit1 = Logit(data1[y_col], sm.add_constant(data1[x_cols_1])).fit(
    cov_type='HC1', disp=0)
print(f"  N = {int(logit1.nobs)}, Pseudo R² = {logit1.prsquared:.3f}")
print(f"  Published: N=331, Pseudo R²=0.102")
for v in x_cols_1:
    print(f"  {v}: {logit1.params[v]:.1f} ({logit1.bse[v]:.2f})")
print(f"  Published: rN=0.8***, rS=-1.1***, rW=0.7**")

# Column 2: Logit with full controls (robust)
print("\n--- Column 2: Logit (full controls) ---")
x_cols_2 = ['rblkn9g', 'rhspn9g', 'riih9g', 'rcldn9g', 'rcthn9g', 'rppn9c',
            'mdny9g', 'rwwh9g', 'rooh9g', 'n9c', 'dnty9c', 'rimn9g', 'n9d',
            'rN', 'rS', 'rW']
data2 = ma_t2[[y_col] + x_cols_2].dropna()
logit2 = Logit(data2[y_col], sm.add_constant(data2[x_cols_2])).fit(
    cov_type='HC1', disp=0)
print(f"  N = {int(logit2.nobs)}, Pseudo R² = {logit2.prsquared:.3f}")
print(f"  Published: N=331, Pseudo R²=0.340")
for v in x_cols_2[:5]:
    print(f"  {v}: {logit2.params[v]:.1f} ({logit2.bse[v]:.1f})")

# Column 3: OLS ratio of primary to secondary districts
print("\n--- Column 3: OLS (Dependent: Ratio primary/secondary districts) ---")
y_col3 = 'diff2'
x_cols_3 = ['rblkn9g', 'rhspn9g', 'riih9g', 'rcldn9g', 'rcthn9g', 'rppn9c',
            'mdny9g', 'rwwh9g', 'rooh9g', 'n9c', 'dnty9c', 'rimn9g', 'n9d',
            'rN', 'rS', 'rW', 'dumm']
data3 = ma_t2[[y_col3] + x_cols_3].dropna()
ols3 = sm.OLS(data3[y_col3], sm.add_constant(data3[x_cols_3])).fit(cov_type='HC1')
print(f"  N = {int(ols3.nobs)}, R² = {ols3.rsquared:.3f}")
print(f"  Published: N=331, R²=0.390")

# Column 4: OLS number of districts
print("\n--- Column 4: OLS (Dependent: No. of districts) ---")
y_col4 = 'n9d'
x_cols_4 = ['rblkn9g', 'rhspn9g', 'riih9g', 'rcldn9g', 'rcthn9g', 'rppn9c',
            'mdny9g', 'rwwh9g', 'rooh9g', 'n9c', 'dnty9c', 'rimn9g',
            'rN', 'rS', 'rW']
data4 = ma_t2[[y_col4] + x_cols_4].dropna()
ols4 = sm.OLS(data4[y_col4], sm.add_constant(data4[x_cols_4])).fit(cov_type='HC1')
print(f"  N = {int(ols4.nobs)}, R² = {ols4.rsquared:.3f}")
print(f"  Published: N=331, R²=0.566")

# ============================================================
# TABLE 3: District-level sorting
# ============================================================
print("\n" + "=" * 70)
print("TABLE 3: Does district availability affect district level peer groups?")
print("=" * 70)

# Panel A - Race
print("\n--- Panel A: Race ---")
dr = load_district_race()
drs = load_district_race_stacked()

controls = ['lnm9f', 'e9d', 'riin9c', 'rcldn9c', 'rcth1n9c', 'rppn9c',
            'mdy9c', 'rwwh9c', 'rooh9c', 'n9c', 'n9c2', 'n9c3']

# Col 1: log districts, no controls
m1 = run_ols(dr, 'yrc1e9s', ['ln9d'], cluster_col='ma9dn')
print(f"\n  Col 1 (log): ln9d = {m1.params['ln9d']:.1f} ({m1.bse['ln9d']:.1f}), "
      f"N={int(m1.nobs)}, R²={m1.rsquared:.3f}")
print(f"  Published:   ln9d = -7.1 (1.1), N=5,555, R²=0.023")

# Col 2: log districts + controls
m2 = run_ols(dr, 'yrc1e9s', ['ln9d'] + controls, cluster_col='ma9dn')
print(f"  Col 2 (log): ln9d = {m2.params['ln9d']:.1f} ({m2.bse['ln9d']:.1f}), "
      f"N={int(m2.nobs)}, R²={m2.rsquared:.3f}")
print(f"  Published:   ln9d = -1.9 (3.0), N=5,555, R²=0.070")

# Col 1 levels: n9d, no controls
m1b = run_ols(dr, 'yrc1e9s', ['n9d'], cluster_col='ma9dn')
print(f"  Col 1 (lvl): n9d = {m1b.params['n9d']:.2f} ({m1b.bse['n9d']:.2f}), "
      f"N={int(m1b.nobs)}, R²={m1b.rsquared:.3f}")
print(f"  Published:   n9d = -0.11 (0.03), N=5,555, R²=0.014")

# Col 2 levels
m2b = run_ols(dr, 'yrc1e9s', ['n9d', 'nm9f'] + ['e9d', 'riin9c', 'rcldn9c', 'rcth1n9c',
              'rppn9c', 'mdy9c', 'rwwh9c', 'rooh9c', 'n9c', 'n9c2', 'n9c3'],
              cluster_col='ma9dn')
print(f"  Col 2 (lvl): n9d = {m2b.params['n9d']:.2f} ({m2b.bse['n9d']:.2f}), "
      f"N={int(m2b.nobs)}, R²={m2b.rsquared:.3f}")
print(f"  Published:   n9d = -0.02 (0.06), N=5,555, R²=0.071")

# Level-specific data (cols 3-5)
# Col 3: OLS, no controls
m3 = run_ols(drs, 'yrc1e9s', ['lnlvl'], cluster_col='ma9dn')
print(f"\n  Col 3 (log): lnlvl = {m3.params['lnlvl']:.1f} ({m3.bse['lnlvl']:.1f}), "
      f"N={int(m3.nobs)}, R²={m3.rsquared:.3f}")
print(f"  Published:   lnlvl = -8.8 (1.1), N=9,452, R²=0.030")

# Col 4: with controls
ctrl_s = ['lnlvlmf', 'riin9c', 'rcldn9c', 'rcth1n9c', 'rppn9c',
          'mdy9c', 'rwwh9c', 'rooh9c', 'n9c', 'n9c2', 'n9c3']
m4 = run_ols(drs, 'yrc1e9s', ['lnlvl'] + ctrl_s, cluster_col='ma9dn')
print(f"  Col 4 (log): lnlvl = {m4.params['lnlvl']:.1f} ({m4.bse['lnlvl']:.1f}), "
      f"N={int(m4.nobs)}, R²={m4.rsquared:.3f}")
print(f"  Published:   lnlvl = -9.9 (1.9), N=9,452, R²=0.049")

# Col 5: areg with MA fixed effects
m5 = run_areg(drs, 'yrc1e9s', ['lnlvl', 'lnlvlmf'], 'ma9dn', cluster_col='ma9dn')
print(f"  Col 5 (log): lnlvl = {m5.params['lnlvl']:.1f} ({m5.bse['lnlvl']:.1f}), "
      f"N={int(m5.nobs)}")
print(f"  Published:   lnlvl = -10.2 (2.7), N=9,452, R²=0.164")

# Level regressions
m3b = run_ols(drs, 'yrc1e9s', ['nlvl'], cluster_col='ma9dn')
print(f"\n  Col 3 (lvl): nlvl = {m3b.params['nlvl']:.1f} ({m3b.bse['nlvl']:.1f}), "
      f"N={int(m3b.nobs)}")
print(f"  Published:   nlvl = -0.2 (0.0), N=9,452")

m4b = run_ols(drs, 'yrc1e9s', ['nlvl', 'nlvlmf'] + ['riin9c', 'rcldn9c', 'rcth1n9c',
              'rppn9c', 'mdy9c', 'rwwh9c', 'rooh9c', 'n9c', 'n9c2', 'n9c3'],
              cluster_col='ma9dn')
print(f"  Col 4 (lvl): nlvl = {m4b.params['nlvl']:.1f} ({m4b.bse['nlvl']:.1f}), "
      f"N={int(m4b.nobs)}")
print(f"  Published:   nlvl = -0.2 (0.1), N=9,452")

m5b = run_areg(drs, 'yrc1e9s', ['nlvl', 'nlvlmf'], 'ma9dn', cluster_col='ma9dn')
print(f"  Col 5 (lvl): nlvl = {m5b.params['nlvl']:.1f} ({m5b.bse['nlvl']:.1f}), "
      f"N={int(m5b.nobs)}")
print(f"  Published:   nlvl = -0.1 (0.1), N=9,452")

# Panel B - Education
print("\n--- Panel B: Education ---")
de = load_district_education()
des = load_district_edu_stacked()

m1e = run_ols(de, 'yede9s', ['ln9d'], cluster_col='ma9dn')
print(f"\n  Col 1 (log): ln9d = {m1e.params['ln9d']:.1f} ({m1e.bse['ln9d']:.1f}), "
      f"N={int(m1e.nobs)}, R²={m1e.rsquared:.3f}")
print(f"  Published:   ln9d = -2.1 (0.3), N=5,554, R²=0.026")

m2e = run_ols(de, 'yede9s', ['ln9d'] + controls, cluster_col='ma9dn')
print(f"  Col 2 (log): ln9d = {m2e.params['ln9d']:.1f} ({m2e.bse['ln9d']:.1f}), "
      f"N={int(m2e.nobs)}, R²={m2e.rsquared:.3f}")
print(f"  Published:   ln9d = -2.9 (0.5), N=5,554, R²=0.073")

m3e = run_ols(des, 'yede9s', ['lnlvl'], cluster_col='ma9dn')
print(f"\n  Col 3 (log): lnlvl = {m3e.params['lnlvl']:.1f} ({m3e.bse['lnlvl']:.1f}), "
      f"N={int(m3e.nobs)}, R²={m3e.rsquared:.3f}")
print(f"  Published:   lnlvl = -2.1 (0.3), N=9,458, R²=0.029")

m4e = run_ols(des, 'yede9s', ['lnlvl'] + ctrl_s, cluster_col='ma9dn')
print(f"  Col 4 (log): lnlvl = {m4e.params['lnlvl']:.1f} ({m4e.bse['lnlvl']:.1f}), "
      f"N={int(m4e.nobs)}, R²={m4e.rsquared:.3f}")
print(f"  Published:   lnlvl = -2.6 (0.3), N=9,458, R²=0.069")

m5e = run_areg(des, 'yede9s', ['lnlvl', 'lnlvlmf'], 'ma9dn', cluster_col='ma9dn')
print(f"  Col 5 (log): lnlvl = {m5e.params['lnlvl']:.1f} ({m5e.bse['lnlvl']:.1f}), "
      f"N={int(m5e.nobs)}")
print(f"  Published:   lnlvl = -6.6 (0.7), N=9,458, R²=0.142")

# Level regressions for education
m1eb = run_ols(de, 'yede9s', ['n9d'], cluster_col='ma9dn')
print(f"\n  Col 1 (lvl): n9d = {m1eb.params['n9d']:.2f} ({m1eb.bse['n9d']:.2f}), "
      f"N={int(m1eb.nobs)}")
print(f"  Published:   n9d = -0.03 (0.01), N=5,554")

m3eb = run_ols(des, 'yede9s', ['nlvl'], cluster_col='ma9dn')
print(f"  Col 3 (lvl): nlvl = {m3eb.params['nlvl']:.2f} ({m3eb.bse['nlvl']:.2f}), "
      f"N={int(m3eb.nobs)}")
print(f"  Published:   nlvl = -0.05 (0.01), N=9,458")

m5eb = run_areg(des, 'yede9s', ['nlvl', 'nlvlmf'], 'ma9dn', cluster_col='ma9dn')
print(f"  Col 5 (lvl): nlvl = {m5eb.params['nlvl']:.2f} ({m5eb.bse['nlvl']:.2f}), "
      f"N={int(m5eb.nobs)}")
print(f"  Published:   nlvl = -0.08 (0.02), N=9,458")

# ============================================================
# TABLE 4: School-level sorting (Race only)
# ============================================================
print("\n" + "=" * 70)
print("TABLE 4: Does district availability affect school level peer groups?")
print("=" * 70)

sc = load_schools()
scs = load_schools_stacked()

# Cross-sectional (cols 1-3)
# Col 1: ln9d only
m41 = run_ols(sc, 'ysrc9f', ['ln9d'], cluster_col='ma9dn')
print(f"\n  Col 1: ln9d = {m41.params['ln9d']:.1f} ({m41.bse['ln9d']:.1f}), "
      f"N={int(m41.nobs)}, R²={m41.rsquared:.3f}")
print(f"  Published: ln9d = -6.1 (1.3), N=48,075, R²=0.018")

# Col 2: ln9d + lnm9f
m42 = run_ols(sc, 'ysrc9f', ['ln9d', 'lnm9f'], cluster_col='ma9dn')
print(f"  Col 2: ln9d = {m42.params['ln9d']:.1f} ({m42.bse['ln9d']:.1f}), "
      f"N={int(m42.nobs)}, R²={m42.rsquared:.3f}")
print(f"  Published: ln9d = -3.1 (1.4), N=48,075, R²=0.022")

# Col 3: ln9d + lnm9f + controls
ctrl_sch = ['riin9c', 'rcldn9c', 'rcth1n9c', 'rppn9c', 'mdy9c', 'rwwh9c', 'rooh9c',
            'n9c', 'n9c2', 'n9c3']
m43 = run_ols(sc, 'ysrc9f', ['ln9d', 'lnm9f'] + ctrl_sch, cluster_col='ma9dn')
print(f"  Col 3: ln9d = {m43.params['ln9d']:.1f} ({m43.bse['ln9d']:.1f}), "
      f"N={int(m43.nobs)}, R²={m43.rsquared:.3f}")
print(f"  Published: ln9d = -1.6 (1.3), N=48,075, R²=0.046")

# Level regressions
m41b = run_ols(sc, 'ysrc9f', ['n9d'], cluster_col='ma9dn')
print(f"\n  Col 1 (lvl): n9d = {m41b.params['n9d']:.1f} ({m41b.bse['n9d']:.1f}), "
      f"N={int(m41b.nobs)}")
print(f"  Published:   n9d = -0.2 (0.0), N=48,075")

m42b = run_ols(sc, 'ysrc9f', ['n9d', 'nm9f'], cluster_col='ma9dn')
print(f"  Col 2 (lvl): n9d = {m42b.params['n9d']:.1f} ({m42b.bse['n9d']:.1f}), "
      f"N={int(m42b.nobs)}")
print(f"  Published:   n9d = -0.1 (0.0), N=48,075")

# Level-specific (cols 4-6)
m44 = run_ols(scs, 'ysrc9f', ['lnlvl'], cluster_col='ma9dn')
print(f"\n  Col 4 (log): lnlvl = {m44.params['lnlvl']:.1f} ({m44.bse['lnlvl']:.1f}), "
      f"N={int(m44.nobs)}, R²={m44.rsquared:.3f}")
print(f"  Published:   lnlvl = -6.4 (1.3), N=50,224, R²=0.019")

m45 = run_ols(scs, 'ysrc9f', ['lnlvl', 'lnlvlmf'], cluster_col='ma9dn')
print(f"  Col 5 (log): lnlvl = {m45.params['lnlvl']:.1f} ({m45.bse['lnlvl']:.1f}), "
      f"N={int(m45.nobs)}, R²={m45.rsquared:.3f}")
print(f"  Published:   lnlvl = -5.3 (1.3), N=50,224, R²=0.020")

m46 = run_areg(scs, 'ysrc9f', ['lnlvl', 'lnlvlmf'], 'ma9dn', cluster_col='ma9dn')
print(f"  Col 6 (log): lnlvl = {m46.params['lnlvl']:.1f} ({m46.bse['lnlvl']:.1f}), "
      f"N={int(m46.nobs)}")
print(f"  Published:   lnlvl = -8.2 (1.6), N=50,224, R²=0.085")

# Level regressions
m44b = run_ols(scs, 'ysrc9f', ['nlvl'], cluster_col='ma9dn')
print(f"\n  Col 4 (lvl): nlvl = {m44b.params['nlvl']:.1f} ({m44b.bse['nlvl']:.1f}), "
      f"N={int(m44b.nobs)}")
print(f"  Published:   nlvl = -0.2 (0.1), N=50,224")

m46b = run_areg(scs, 'ysrc9f', ['nlvl', 'nlvlmf'], 'ma9dn', cluster_col='ma9dn')
print(f"  Col 6 (lvl): nlvl = {m46b.params['nlvl']:.1f} ({m46b.bse['nlvl']:.1f}), "
      f"N={int(m46b.nobs)}")
print(f"  Published:   nlvl = -0.1 (0.0), N=50,224")

# ============================================================
# TABLE 5: Private enrollment
# ============================================================
print("\n" + "=" * 70)
print("TABLE 5: Does district availability affect private enrollment?")
print("=" * 70)

ma = load_ma()

# Cross-sectional (cols 1-2)
# Col 1: ln9d, robust
m51 = run_ols(ma, 'rve9s', ['ln9d'], robust=True)
print(f"\n  Col 1: ln9d = {m51.params['ln9d']:.1f} ({m51.bse['ln9d']:.1f}), "
      f"N={int(m51.nobs)}, R²={m51.rsquared:.3f}")
print(f"  Published: ln9d = 1.1 (0.2), N=291, R²=0.070")

# Col 2: ln9d + controls
ctrl_ma = ['lnnm9f', 'rb1n9c', 'rh1n9c', 'riin9c', 'rcldn9c', 'rcth1n9c',
           'rppn9c', 'mdy9c', 'rwwh9c', 'rooh9c', 'n9c', 'n9c2']
m52 = run_ols(ma, 'rve9s', ['ln9d'] + ctrl_ma, robust=True)
print(f"  Col 2: ln9d = {m52.params['ln9d']:.1f} ({m52.bse['ln9d']:.1f}), "
      f"N={int(m52.nobs)}, R²={m52.rsquared:.3f}")
print(f"  Published: ln9d = -0.2 (0.3), N=291, R²=0.458")

# Level-specific (cols 3-6)
mp = load_ma_private()

m53 = run_ols(mp, 'rve9s', ['lnlvl'], robust=True)
print(f"\n  Col 3: lnlvl = {m53.params['lnlvl']:.1f} ({m53.bse['lnlvl']:.1f}), "
      f"N={int(m53.nobs)}, R²={m53.rsquared:.3f}")
print(f"  Published: lnlvl = 1.3 (0.2), N=582, R²=0.067")

m54 = run_ols(mp, 'rve9s', ['lnlvl', 'lnlvlf'] + ['rb1n9c', 'rh1n9c', 'riin9c',
              'rcldn9c', 'rcth1n9c', 'rppn9c', 'mdy9c', 'rwwh9c', 'rooh9c',
              'n9c', 'n9c2'], robust=True)
print(f"  Col 4: lnlvl = {m54.params['lnlvl']:.1f} ({m54.bse['lnlvl']:.1f}), "
      f"N={int(m54.nobs)}, R²={m54.rsquared:.3f}")
print(f"  Published: lnlvl = -0.9 (0.2), N=582, R²=0.570")

m55 = run_ols(mp, 'rve9s', ['lnlvl', 'zsec', 'lnlvlf'] + ['rb1n9c', 'rh1n9c', 'riin9c',
              'rcldn9c', 'rcth1n9c', 'rppn9c', 'mdy9c', 'rwwh9c', 'rooh9c',
              'n9c', 'n9c2'], robust=True)
print(f"  Col 5: lnlvl = {m55.params['lnlvl']:.1f} ({m55.bse['lnlvl']:.1f}), "
      f"zsec = {m55.params['zsec']:.1f} ({m55.bse['zsec']:.1f}), "
      f"N={int(m55.nobs)}, R²={m55.rsquared:.3f}")
print(f"  Published: lnlvl = -0.0 (0.3), zsec = -5.8 (0.7), N=582, R²=0.618")

# Col 6: areg with MA fixed effects
m56 = run_areg(mp, 'rve9s', ['lnlvl', 'zsec', 'lnlvlf'] + ['rb1n9c', 'rh1n9c', 'riin9c',
               'rcldn9c', 'rcth1n9c', 'rppn9c', 'mdy9c', 'rwwh9c', 'rooh9c',
               'n9c', 'n9c2'], 'ma9dn')
print(f"  Col 6: lnlvl = {m56.params['lnlvl']:.1f} ({m56.bse['lnlvl']:.1f}), "
      f"N={int(m56.nobs)}")
print(f"  Published: lnlvl = -1.0 (0.5), N=582, R²=0.952")

# Level regressions Table 5
m51b = run_ols(ma, 'rve9s', ['n9d'], robust=True)
print(f"\n  Col 1 (lvl): n9d = {m51b.params['n9d']:.2f} ({m51b.bse['n9d']:.2f}), "
      f"N={int(m51b.nobs)}")
print(f"  Published:   n9d = 0.07 (0.01), N=291")

m53b = run_ols(mp, 'rve9s', ['nlvl'], robust=True)
print(f"  Col 3 (lvl): nlvl = {m53b.params['nlvl']:.2f} ({m53b.bse['nlvl']:.2f}), "
      f"N={int(m53b.nobs)}")
print(f"  Published:   nlvl = 0.09 (0.01), N=582")

m56b = run_areg(mp, 'rve9s', ['nlvl', 'zsec', 'nlvlf'] + ['rb1n9c', 'rh1n9c', 'riin9c',
                'rcldn9c', 'rcth1n9c', 'rppn9c', 'mdy9c', 'rwwh9c', 'rooh9c',
                'n9c', 'n9c2'], 'ma9dn')
print(f"  Col 6 (lvl): nlvl = {m56b.params['nlvl']:.2f} ({m56b.bse['nlvl']:.2f}), "
      f"N={int(m56b.nobs)}")
print(f"  Published:   nlvl = -0.03 (0.01), N=582")

print("\n[DONE] Tables replication complete.")
