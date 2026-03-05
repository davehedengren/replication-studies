"""
03_figures.py - Replicate figures from Davis (2014).

Figures:
  1. Figure 1: Gasoline consumption vs prices (scatter)
  2. Figure A1: Diesel consumption vs prices (scatter)
  3. Figure 2: Top 10 fuel subsidies (bar chart)
  4. Figure 3: Top 10 DWL (bar chart)
  5. Figure A3: Top 10 subsidies per capita (bar chart)
  6. Figure A4: Top 10 DWL per capita (bar chart)
  7. Figure A5: Top 10 DWL from pricing below SMC (bar chart)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import OUTPUT_DIR

df = pd.read_csv(f'{OUTPUT_DIR}/temp1.csv')
dwl = pd.read_csv(f'{OUTPUT_DIR}/dwl_private.csv')
smc = pd.read_csv(f'{OUTPUT_DIR}/dwl_smc.csv')

# ============================================================
# Figure 1: Gasoline consumption vs price
# ============================================================
fig1_df = df[(df['gasprice'] > 0) & df['gasprice'].notna() &
             df['gascon'].notna() & df['pop'].notna()].copy()
fig1_df.loc[:, 'country'] = fig1_df['country'].replace({'Korea, South': 'South Korea',
                                                         'Burma (Myanmar)': 'Myanmar'})

fig, ax = plt.subplots(figsize=(8, 6))
sizes = fig1_df['pop'] / 1e6  # scale for visibility
ax.scatter(fig1_df['gasprice'], fig1_df['gascon'], s=sizes,
           facecolors='none', edgecolors='black', linewidths=0.5, alpha=0.8)

# Label select countries
label_countries = ['United States', 'Saudi Arabia', 'Canada', 'Australia',
                   'Kuwait', 'Venezuela', 'U.A.E.', 'Japan', 'Mexico',
                   'Russia', 'Iran', 'Iraq', 'Indonesia', 'Egypt', 'Nigeria',
                   'United Kingdom', 'Germany', 'Netherlands', 'France',
                   'Italy', 'Turkey', 'Malaysia', 'Algeria', 'China', 'India',
                   'South Korea']
for _, r in fig1_df.iterrows():
    if r['country'] in label_countries:
        ax.annotate(r['country'], (r['gasprice'], r['gascon']),
                   fontsize=6, ha='left', va='bottom')

ax.set_xlabel('Gasoline Price ($ per gallon)', fontsize=10)
ax.set_ylabel('Gasoline Consumption (gallons/year/capita)', fontsize=10)
ax.set_xlim(0, 10)
ax.set_ylim(-25, 400)
ax.set_xticks(range(0, 11))
ax.set_yticks(range(0, 401, 100))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figure1_gas_consumption.png', dpi=150)
plt.close()
print("Saved figure1_gas_consumption.png")

# ============================================================
# Figure A1: Diesel consumption vs price
# ============================================================
figA1_df = fig1_df.copy()
figA1_df = figA1_df[figA1_df['dieprice'].notna() & figA1_df['diecon'].notna()]
# Drop Qatar and Luxembourg (unreasonably high diesel)
figA1_df = figA1_df[~figA1_df['country'].isin(['Qatar', 'Luxembourg'])]

fig, ax = plt.subplots(figsize=(8, 6))
sizes = figA1_df['pop'] / 1e6
ax.scatter(figA1_df['dieprice'], figA1_df['diecon'], s=sizes,
           facecolors='none', edgecolors='black', linewidths=0.5, alpha=0.8)

for _, r in figA1_df.iterrows():
    if r['country'] in label_countries:
        ax.annotate(r['country'], (r['dieprice'], r['diecon']),
                   fontsize=6, ha='left', va='bottom')

ax.set_xlabel('Diesel Price ($ per gallon)', fontsize=10)
ax.set_ylabel('Road-Sector Diesel Consumption (gallons/year/capita)', fontsize=10)
ax.set_xlim(0, 10)
ax.set_ylim(-15, 200)
ax.set_xticks(range(0, 11))
ax.set_yticks(range(0, 201, 50))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figureA1_diesel_consumption.png', dpi=150)
plt.close()
print("Saved figureA1_diesel_consumption.png")

# ============================================================
# Figure 2: Top 10 fuel subsidies (bar chart)
# ============================================================
from utils import SPOT_GAS_PRICE, SPOT_DIE_PRICE, TRANSPORT_COST_GALLON

sub_df = df.copy()
sub_df['spotgasprice'] = SPOT_GAS_PRICE
sub_df['spotdieprice'] = SPOT_DIE_PRICE
sub_df.loc[sub_df['netexports'] < 0, 'spotgasprice'] += TRANSPORT_COST_GALLON
sub_df.loc[sub_df['netexports'] < 0, 'spotdieprice'] += TRANSPORT_COST_GALLON
sub_df['GasSub'] = (sub_df['spotgasprice'] - sub_df['gasprice']) * sub_df['gascon'] * sub_df['pop'] / 1e9
sub_df['DieSub'] = (sub_df['spotdieprice'] - sub_df['dieprice']) * sub_df['diecon'] * sub_df['pop'] / 1e9
sub_df['TotSub'] = sub_df['GasSub'] + sub_df['DieSub']
top10_sub = sub_df.sort_values('TotSub', ascending=False).head(10)

fig, ax = plt.subplots(figsize=(8, 5))
y = range(len(top10_sub))
ax.barh(y, top10_sub['GasSub'], color='#CC6600', label='Gasoline')
ax.barh(y, top10_sub['DieSub'], left=top10_sub['GasSub'], color='black', label='Diesel')
ax.set_yticks(y)
ax.set_yticklabels(top10_sub['country'])
ax.invert_yaxis()
ax.set_xlabel('Subsidy Amount, Billions', fontsize=10)
ax.set_xlim(0, 25)
ax.legend(loc='lower right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figure2_subsidies.png', dpi=150)
plt.close()
print("Saved figure2_subsidies.png")

# ============================================================
# Figure 3: Top 10 DWL (bar chart)
# ============================================================
dwl_top10 = dwl.sort_values('DWLtot', ascending=False).head(10)

fig, ax = plt.subplots(figsize=(8, 5))
y = range(len(dwl_top10))
ax.barh(y, dwl_top10['DWLgas'], color='#CC6600', label='Gasoline')
ax.barh(y, dwl_top10['DWLdie'], left=dwl_top10['DWLgas'], color='black', label='Diesel')
ax.set_yticks(y)
ax.set_yticklabels(dwl_top10['country'])
ax.invert_yaxis()
ax.set_xlabel('Deadweight Loss, Billions', fontsize=10)
ax.set_xlim(0, 12.5)
ax.set_xticks(range(0, 13, 3))
ax.legend(loc='lower right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figure3_dwl.png', dpi=150)
plt.close()
print("Saved figure3_dwl.png")

# ============================================================
# Figure A3: Top 10 subsidies per capita
# ============================================================
sub_df['GasSubPC'] = sub_df['GasSub'] * 1e9 / sub_df['pop']
sub_df['DieSubPC'] = sub_df['DieSub'] * 1e9 / sub_df['pop']
sub_df['TotSubPC'] = sub_df['GasSubPC'] + sub_df['DieSubPC']
toppc = sub_df.sort_values('TotSubPC', ascending=False).head(10)

fig, ax = plt.subplots(figsize=(8, 5))
y = range(len(toppc))
ax.barh(y, toppc['GasSubPC'], color='#CC6600', label='Gasoline')
ax.barh(y, toppc['DieSubPC'], left=toppc['GasSubPC'], color='black', label='Diesel')
ax.set_yticks(y)
ax.set_yticklabels(toppc['country'])
ax.invert_yaxis()
ax.set_xlabel('Annual Subsidy Amount Per Capita', fontsize=10)
ax.set_xlim(0, 1000)
ax.set_xticks([0, 250, 500, 750, 1000])
ax.legend(loc='lower right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figureA3_subsidies_pc.png', dpi=150)
plt.close()
print("Saved figureA3_subsidies_pc.png")

# ============================================================
# Figure A4: Top 10 DWL per capita
# ============================================================
dwl['DWLgasPC'] = dwl['DWLgas'] * 1e9 / dwl['pop']
dwl['DWLdiePC'] = dwl['DWLdie'] * 1e9 / dwl['pop']
dwl['DWLtotPC'] = dwl['DWLgasPC'] + dwl['DWLdiePC']
dwlpc_top = dwl.sort_values('DWLtotPC', ascending=False).head(10)

fig, ax = plt.subplots(figsize=(8, 5))
y = range(len(dwlpc_top))
ax.barh(y, dwlpc_top['DWLgasPC'], color='#CC6600', label='Gasoline')
ax.barh(y, dwlpc_top['DWLdiePC'], left=dwlpc_top['DWLgasPC'], color='black', label='Diesel')
ax.set_yticks(y)
ax.set_yticklabels(dwlpc_top['country'])
ax.invert_yaxis()
ax.set_xlabel('Annual Deadweight Loss Per Capita', fontsize=10)
ax.set_xlim(0, 500)
ax.set_xticks([0, 100, 200, 300, 400, 500])
ax.legend(loc='lower right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figureA4_dwl_pc.png', dpi=150)
plt.close()
print("Saved figureA4_dwl_pc.png")

# ============================================================
# Figure A5: Top 10 DWL from pricing below SMC
# ============================================================
smc_top10 = smc.sort_values('DWLtot', ascending=False).head(10)

fig, ax = plt.subplots(figsize=(8, 5))
y = range(len(smc_top10))
ax.barh(y, smc_top10['DWLgas'], color='#CC6600', label='Gasoline')
ax.barh(y, smc_top10['DWLdie'], left=smc_top10['DWLgas'], color='black', label='Diesel')
ax.set_yticks(y)
ax.set_yticklabels(smc_top10['country'])
ax.invert_yaxis()
ax.set_xlabel('Deadweight Loss, Billions', fontsize=10)
ax.set_xlim(0, 20)
ax.set_xticks(range(0, 21, 4))
ax.legend(loc='lower right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figureA5_dwl_smc.png', dpi=150)
plt.close()
print("Saved figureA5_dwl_smc.png")

# ============================================================
# Gasoline price bar chart (countries with pop > 25M)
# ============================================================
bar_df = df[(df['gasprice'] > 0) & df['gasprice'].notna() & (df['pop'] > 25e6)].copy()
bar_df.loc[:, 'country'] = bar_df['country'].replace({'Korea, South': 'South Korea',
                                                       'Burma (Myanmar)': 'Myanmar'})
bar_df = bar_df.sort_values('gasprice')

fig, ax = plt.subplots(figsize=(5, 10))
ax.barh(range(len(bar_df)), bar_df['gasprice'], color='black')
ax.set_yticks(range(len(bar_df)))
ax.set_yticklabels(bar_df['country'], fontsize=7)
ax.set_xlabel('Gasoline Price Per Gallon', fontsize=10)
ax.set_xticks([2, 4, 6, 8, 10])
ax.set_xticklabels(['$2', '$4', '$6', '$8', '$10'])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figure_gas_prices_bar.png', dpi=150)
plt.close()
print("Saved figure_gas_prices_bar.png")

print("\nAll figures saved.")
