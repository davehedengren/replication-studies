"""
03_figures.py – Replicate Figure 1 from Samaniego (2008).
Scatter plot of ISTC rate vs turnover rate across 41 industries.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import load_industry_data, load_did_data, OUTPUT_DIR

df_ind = load_industry_data()
df_did = load_did_data()

# Construct regression-based turnover indices (same as in 02_tables.py)
temp = df_did[['c', 'i', 'turnover']].dropna().copy()
c_dum = pd.get_dummies(temp['c'], prefix='c', drop_first=True).astype(float)
i_dum = pd.get_dummies(temp['i'], prefix='i', drop_first=True).astype(float)
X = pd.concat([c_dum.reset_index(drop=True), i_dum.reset_index(drop=True)], axis=1)
X = sm.add_constant(X)
y = temp['turnover'].reset_index(drop=True)
model = sm.OLS(y, X).fit()

ind_fe = {1: model.params['const']}
for col in i_dum.columns:
    i_num = int(col.replace('i_', ''))
    ind_fe[i_num] = model.params['const'] + model.params[col]

c_fe = {1: 0.0}
for col in c_dum.columns:
    c_num = int(col.replace('c_', ''))
    c_fe[c_num] = model.params[col]
median_c = np.median(list(c_fe.values()))
for i_num in ind_fe:
    ind_fe[i_num] += median_c

# Build plot data
df_ind['i'] = range(1, 42)
plot_data = df_ind[['i', 'Industry', 'istc']].copy()
plot_data['turnover'] = plot_data['i'].map(ind_fe)

# Industry labels (abbreviations matching Figure 1)
label_map = {
    'Oil and gas extraction': 'OILG', 'Real estate': 'REAL', 'Hotels': 'HOTL',
    'Arts, sports, amusement': 'ARTS', 'Education': 'EDUC', 'Other mining': 'OMIN',
    'Other services': 'OSRV', 'Utilities': 'UTIL', 'Waste disposal': 'WSTE',
    'Plastics': 'PLST', 'Restaurants': 'REST', 'Primary and fabricated metal prod.': 'FABM',
    'Retail Trade': 'RTIL', 'Transport Equip.': 'TRSP', 'Construction': 'CONS',
    'Insurance, trusts': 'INSU', 'Food products': 'FOOD', 'Electrical machinery': 'ELEC',
    'Leather': 'LTHR', 'Wood products': 'WOOD', 'Manuf n.e.c.': 'OMAN',
    'Textiles': 'TXTL', 'Petroleum and coal products': 'PETR', 'Land transport': 'LNDT',
    'Nonmetal products': 'NMTL', 'Chemicals': 'CHEM', 'General Machinery': 'MACH',
    'Healthcare': 'HLTH', 'Transport support': 'TSUP',
    'Computers and electronic prod.': 'COMP', 'Paper, printing, software': 'PAPR',
    'Wholesale Trade': 'WHOL', 'Water transport': 'WATR',
    'Technical Services': 'TCHS', 'Legal services': 'LGLS',
    'Finance (not insurance, trusts)': 'FNCE', 'Broadcasting': 'BRDC',
    'Information and data processing': 'INFO', 'Rental services': 'RENT',
    'Systems design': 'SYST', 'Air transport': 'AIRT',
}
plot_data['label'] = plot_data['Industry'].map(label_map)

# Figure 1: Scatter plot with OLS line
fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(plot_data['istc'], plot_data['turnover'], color='black', s=5, zorder=5)

# OLS fit line
slope, intercept, r_val, p_val, se = stats.linregress(plot_data['istc'], plot_data['turnover'])
x_line = np.linspace(plot_data['istc'].min(), plot_data['istc'].max(), 100)
ax.plot(x_line, intercept + slope * x_line, 'k-', linewidth=1)

# Labels
for _, row in plot_data.iterrows():
    ax.annotate(row['label'], (row['istc'], row['turnover']),
                fontsize=6, ha='center', va='bottom')

ax.set_xlabel('Rate of ISTC, %', fontsize=11)
ax.set_ylabel('Rate of turnover, %', fontsize=11)
ax.set_xlim(1, 9)
ax.set_ylim(10, 30)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figure1_istc_turnover.png'), dpi=150)
plt.close()

print(f"Figure 1 saved. OLS: turnover = {intercept:.2f} + {slope:.2f}*ISTC, R²={r_val**2:.2f}")
print(f"Published: correlation ≈ 0.50, R² ≈ 0.25")
print(f"Replicated: correlation = {r_val:.2f}, R² = {r_val**2:.2f}")
