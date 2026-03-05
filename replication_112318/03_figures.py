"""
03_figures.py - Replicate key figures from Urquiola (2005).
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
sys.path.insert(0, '.')
from utils import *

ma = load_ma()
dr = load_district_race()
sc = load_schools()

# ============================================================
# FIGURE 1: School and school district availability in MAs
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel A: Schools in all MAs (primary vs secondary)
ax = axes[0, 0]
ma['e9d_100k'] = ma['e9d'] / 100000
ax.scatter(ma['e9d_100k'], ma['prmm9f'], marker='o', facecolors='none',
           edgecolors='black', s=15, label='Primary', alpha=0.7)
ax.scatter(ma['e9d_100k'], ma['secm9f'], marker='^', facecolors='none',
           edgecolors='black', s=15, label='Secondary', alpha=0.7)
# Fit lines
for y, ls in [('prmm9f', '-'), ('secm9f', '--')]:
    mask = ma[['e9d_100k', y]].dropna()
    z = np.polyfit(mask['e9d_100k'], mask[y], 1)
    x_fit = np.linspace(0, 14, 100)
    ax.plot(x_fit, np.polyval(z, x_fit), 'k' + ls, lw=1)
ax.set_xlabel('Total enrollment (hundreds of thousands)')
ax.set_ylabel('Schools')
ax.set_title('Panel A: Schools in all MAs')
ax.set_xlim(0, 14)

# Panel B: Districts in all MAs
ax = axes[0, 1]
ax.scatter(ma['e9d_100k'], ma['n9d'], marker='o', c='black', s=12, alpha=0.7)
z = np.polyfit(ma[['e9d_100k', 'n9d']].dropna()['e9d_100k'],
               ma[['e9d_100k', 'n9d']].dropna()['n9d'], 1)
x_fit = np.linspace(0, 14, 100)
ax.plot(x_fit, np.polyval(z, x_fit), 'k-', lw=1)
ax.set_xlabel('Total enrollment (hundreds of thousands)')
ax.set_ylabel('Districts')
ax.set_title('Panel B: Districts in all MAs')
ax.set_xlim(0, 14)

# Panel C: Districts in selected MAs (with between-level differences)
ax = axes[1, 0]
mix = ma[ma['dumm'] == 1].copy()
ax.scatter(mix['e9d_100k'], mix['prm9d'], marker='o', facecolors='none',
           edgecolors='black', s=15, label='Primary')
ax.scatter(mix['e9d_100k'], mix['sec9d'], marker='^', facecolors='none',
           edgecolors='black', s=15, label='Secondary')
for y, ls in [('prm9d', '-'), ('sec9d', '--')]:
    mask = mix[['e9d_100k', y]].dropna()
    z = np.polyfit(mask['e9d_100k'], mask[y], 1)
    ax.plot(x_fit, np.polyval(z, x_fit), 'k' + ls, lw=1)
ax.set_xlabel('Total enrollment (hundreds of thousands)')
ax.set_ylabel('Districts')
ax.set_title('Panel C: Districts in selected MAs')
ax.set_xlim(0, 14)

# Panel D: Ratio of primary to secondary districts
ax = axes[1, 1]
vals = ma['diff2'].dropna()
vals = vals[vals > 0]
kde = gaussian_kde(vals, bw_method=0.14)
x_kde = np.linspace(0.5, 7, 200)
ax.plot(x_kde, kde(x_kde), 'k-', lw=1.5)
ax.set_xlabel('No. of primary/No. of secondary districts')
ax.set_ylabel('Density')
ax.set_title('Panel D: Ratio of primary to secondary dist.')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'figure1.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Figure 1 saved.")

# ============================================================
# FIGURE 3: Heterogeneity among MAs and districts
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel A: Race - Metropolitan areas
ax = axes[0, 0]
vals = ma['hmrc9f'].dropna()
kde = gaussian_kde(vals)
x_kde = np.linspace(0, 100, 200)
ax.plot(x_kde, kde(x_kde), 'k-', lw=1.5)
ax.set_xlabel('Absolute heterogeneity measure')
ax.set_ylabel('Density')
ax.set_title('Panel A: Race--Metropolitan areas')

# Panel B: Education - Metropolitan areas
ax = axes[0, 1]
vals = ma['hmede9s'].dropna()
kde = gaussian_kde(vals)
x_kde = np.linspace(50, 100, 200)
ax.plot(x_kde, kde(x_kde), 'k-', lw=1.5)
ax.set_xlabel('Absolute heterogeneity measure')
ax.set_ylabel('Density')
ax.set_title('Panel B: Education--Metropolitan areas')

# Panel C: Race - Districts
ax = axes[1, 0]
vals = dr['hrc9f'].dropna()
kde = gaussian_kde(vals)
x_kde = np.linspace(0, 100, 200)
ax.plot(x_kde, kde(x_kde), 'k-', lw=1.5)
ax.set_xlabel('Absolute heterogeneity measure')
ax.set_ylabel('Density')
ax.set_title('Panel C: Race--Districts')

# Panel D: Education - Districts
ax = axes[1, 1]
de = load_district_education()
vals = de['hede9s'].dropna()
kde = gaussian_kde(vals)
x_kde = np.linspace(0, 100, 200)
ax.plot(x_kde, kde(x_kde), 'k-', lw=1.5)
ax.set_xlabel('Absolute heterogeneity measure')
ax.set_ylabel('Density')
ax.set_title('Panel D: Education--Districts')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'figure3.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Figure 3 saved.")

# ============================================================
# FIGURE 4: Racial heterogeneity in schools and MAs
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel D: Schools relative to own district (kernel density)
ax = axes[1, 1]
vals = sc['vsrc9f'].dropna()
vals = vals[vals <= 300]
kde = gaussian_kde(vals)
x_kde = np.linspace(0, 300, 200)
ax.plot(x_kde, kde(x_kde), 'k-', lw=1.5)
ax.set_xlabel('Relative sorting measure')
ax.set_ylabel('Density')
ax.set_title('Panel D: Schools relative to own district')

# Panels A-C: Smoothed heterogeneity vs counts
from statsmodels.nonparametric.kernel_regression import KernelReg

# Panel A: vsrc9f vs n9f (schools in own district)
ax = axes[0, 0]
sub = sc[['vsrc9f', 'n9f']].dropna()
sub = sub[(sub['n9f'] > 0) & (sub['n9f'] <= 1000)]
kr = KernelReg(sub['vsrc9f'].values, sub['n9f'].values, 'c', bw=[0.10 * sub['n9f'].std()])
x_pred = np.linspace(sub['n9f'].min(), min(1000, sub['n9f'].max()), 200)
y_pred, _ = kr.fit(x_pred)
ax.plot(x_pred, y_pred, 'k-', lw=1.5)
ax.set_xlabel('Number of schools in own district')
ax.set_ylabel('Heterogeneity')
ax.set_title('Panel A: Schools relative to own district')

# Panel B: ysrc9f vs nm9f (schools in MA)
ax = axes[0, 1]
sub = sc[['ysrc9f', 'nm9f']].dropna()
sub = sub[(sub['nm9f'] > 0) & (sub['nm9f'] <= 2000)]
kr = KernelReg(sub['ysrc9f'].values, sub['nm9f'].values, 'c', bw=[0.10 * sub['nm9f'].std()])
x_pred = np.linspace(sub['nm9f'].min(), min(2000, sub['nm9f'].max()), 200)
y_pred, _ = kr.fit(x_pred)
ax.plot(x_pred, y_pred, 'k-', lw=1.5)
ax.set_xlabel('Number of schools in the MA')
ax.set_ylabel('Heterogeneity')
ax.set_title('Panel B: Schools relative to own MA')

# Panel C: ysrc9f vs n9d (districts in MA)
ax = axes[1, 0]
sub = sc[['ysrc9f', 'n9d']].dropna()
sub = sub[(sub['n9d'] > 0) & (sub['n9d'] <= 200)]
kr = KernelReg(sub['ysrc9f'].values, sub['n9d'].values, 'c', bw=[0.10 * sub['n9d'].std()])
x_pred = np.linspace(sub['n9d'].min(), min(200, sub['n9d'].max()), 200)
y_pred, _ = kr.fit(x_pred)
ax.plot(x_pred, y_pred, 'k-', lw=1.5)
ax.set_xlabel('Number of districts in the MA')
ax.set_ylabel('Heterogeneity')
ax.set_title('Panel C: Schools relative to own MA')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'figure4.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Figure 4 saved.")

# ============================================================
# FIGURE 5: Private enrollment
# ============================================================
mp = load_ma_private()
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel A: Density of private enrollment by level
ax = axes[0]
prv0 = mp.loc[mp['zsec'] == 0, 'rve9s'].dropna()
prv1 = mp.loc[mp['zsec'] == 1, 'rve9s'].dropna()
x_kde = np.linspace(0, 30, 200)
if len(prv0) > 1:
    kde0 = gaussian_kde(prv0)
    ax.plot(x_kde, kde0(x_kde), 'k-', lw=1.5, label='Primary')
if len(prv1) > 1:
    kde1 = gaussian_kde(prv1)
    ax.plot(x_kde, kde1(x_kde), 'k--', lw=1.5, label='Secondary')
ax.set_xlabel('Private enrollment rate')
ax.set_ylabel('Density')
ax.set_title('Panel A: Private enrollment')
ax.legend()

# Panel B: Private enrollment vs district availability
ax = axes[1]
for zsec_val, marker, label in [(0, 'o', 'Primary'), (1, '^', 'Secondary')]:
    sub = mp[mp['zsec'] == zsec_val][['rve9s', 'nlvl']].dropna()
    ax.scatter(sub['nlvl'], sub['rve9s'], marker=marker, facecolors='none',
               edgecolors='black', s=12, alpha=0.5, label=label)
ax.set_xlabel('Number of districts in the MA')
ax.set_ylabel('Private enrollment')
ax.set_title('Panel B: Private enroll. and dist. availability')
ax.legend()
ax.set_xlim(0, 200)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'figure5.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Figure 5 saved.")

print("\n[DONE] All figures saved.")
