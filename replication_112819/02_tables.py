"""
02_tables.py - Subsidy and DWL calculations for Davis (2014) replication.

Replicates:
- Subsidy calculations (Section 1/Figure 2)
- DWL calculations (Section 2/Figure 3/Table A1)
- SMC calculations (Section 3/Figure A5)
- Key regression results
"""
import numpy as np
import pandas as pd
from utils import (OUTPUT_DIR, SPOT_GAS_PRICE, SPOT_DIE_PRICE,
                   TRANSPORT_COST_GALLON, EXTERNAL_COST, DEMAND_ELASTICITY)

df = pd.read_csv(f'{OUTPUT_DIR}/temp1.csv')

print("=" * 70)
print("SECTION 1: FUEL SUBSIDY CALCULATIONS")
print("=" * 70)

# ----- Subsidy calculation -----
df['spotgasprice'] = SPOT_GAS_PRICE
df['spotdieprice'] = SPOT_DIE_PRICE

# Add transport costs for net importers
df.loc[df['netexports'] < 0, 'spotgasprice'] += TRANSPORT_COST_GALLON
df.loc[df['netexports'] < 0, 'spotdieprice'] += TRANSPORT_COST_GALLON

# Calculate subsidy amounts (billions)
df['GasSub'] = (df['spotgasprice'] - df['gasprice']) * df['gascon'] * df['pop'] / 1e9
df['DieSub'] = (df['spotdieprice'] - df['dieprice']) * df['diecon'] * df['pop'] / 1e9
df['TotSub'] = df['GasSub'] + df['DieSub']

# Count subsidizing countries
valid = df[df['TotSub'].notna()]
n_sub = (valid['TotSub'] > 0).sum()
n_total = len(valid)
print(f"\nCountries that subsidize (total): {n_sub} of {n_total} = {n_sub/n_total:.2f}")

gas_sub = df[df['GasSub'].notna()]
n_gas_sub = (gas_sub['GasSub'] > 0).sum()
print(f"Countries that subsidize gasoline: {n_gas_sub}")
print(f"  [Paper: 24]")

die_sub = df[df['DieSub'].notna()]
n_die_sub = (die_sub['DieSub'] > 0).sum()
print(f"Countries that subsidize diesel: {n_die_sub}")
print(f"  [Paper: 35]")

# Total subsidies
gas_sub_pos = df.loc[df['GasSub'] > 0, 'GasSub'].sum()
die_sub_pos = df.loc[df['DieSub'] > 0, 'DieSub'].sum()
tot_sub_pos = df.loc[df['TotSub'] > 0, 'TotSub'].sum()
# Also count negative (taxes) for the full total
total_all = df['TotSub'].dropna()
total_sub_only = total_all[total_all > 0].sum()

print(f"\nTotal gasoline subsidies: ${gas_sub_pos:.1f} billion")
print(f"Total diesel subsidies: ${die_sub_pos:.1f} billion")
print(f"Total fuel subsidies (subsidy countries only): ${total_sub_only:.1f} billion")
print(f"  [Paper: ~$110 billion total, ~$55 billion each]")

# Top 10 subsidy countries
df_sub = df[df['TotSub'] > 0].sort_values('TotSub', ascending=False).head(10)
print(f"\nTop 10 subsidy countries:")
print(f"{'Country':<20} {'GasSub':>10} {'DieSub':>10} {'TotSub':>10}")
for _, r in df_sub.iterrows():
    print(f"{r['country']:<20} ${r['GasSub']:>9.1f} ${r['DieSub']:>9.1f} ${r['TotSub']:>9.1f}")

# Top 10 share
top10_sum = df_sub['TotSub'].sum()
print(f"\nTop 10 share of total: {top10_sum/total_sub_only:.2f}")
print(f"  [Paper: 0.90]")

# Saudi + Venezuela share
sa_ven = df[df['country'].isin(['Saudi Arabia', 'Venezuela'])]['TotSub'].sum()
print(f"Saudi Arabia + Venezuela share: {sa_ven/total_sub_only:.2f}")

# Per capita subsidies
df['GasSubPC'] = df['GasSub'] * 1e9 / df['pop']
df['DieSubPC'] = df['DieSub'] * 1e9 / df['pop']
df['TotSubPC'] = df['GasSubPC'] + df['DieSubPC']

df_subpc = df.sort_values('TotSubPC', ascending=False).head(10)
print(f"\nTop 10 subsidy per capita countries:")
print(f"{'Country':<20} {'GasSubPC':>10} {'DieSubPC':>10} {'TotSubPC':>10}")
for _, r in df_subpc.iterrows():
    print(f"{r['country']:<20} ${r['GasSubPC']:>9.0f} ${r['DieSubPC']:>9.0f} ${r['TotSubPC']:>9.0f}")

# US check
us = df[df['country'] == 'United States']
print(f"\nUnited States TotSub: ${us['TotSub'].values[0]:.1f} billion")
print(f"  [Paper: US does not subsidize]")

# Net exports of top 10
print(f"\nTop 10 subsidy countries - net oil exports:")
for _, r in df_sub.iterrows():
    print(f"  {r['country']:<20} {r['netexports']:>10.0f} thousand bbl/day")

print("\n" + "=" * 70)
print("SECTION 2: DEADWEIGHT LOSS CALCULATIONS (Private Cost)")
print("=" * 70)

# Reload for DWL section (fresh spot prices)
dwl = pd.read_csv(f'{OUTPUT_DIR}/temp1.csv')
dwl['spotgasprice'] = SPOT_GAS_PRICE
dwl['spotdieprice'] = SPOT_DIE_PRICE
dwl.loc[dwl['netexports'] < 0, 'spotgasprice'] += TRANSPORT_COST_GALLON
dwl.loc[dwl['netexports'] < 0, 'spotdieprice'] += TRANSPORT_COST_GALLON

e = DEMAND_ELASTICITY  # -0.6

# Total consumption (millions of gallons)
dwl['gas'] = dwl['gascon'] * dwl['pop'] / 1e6
dwl['die'] = dwl['diecon'] * dwl['pop'] / 1e6

# Scale parameter A = q / p^e
dwl['Agas'] = dwl['gas'] / (dwl['gasprice'] ** e)
dwl['Adie'] = dwl['die'] / (dwl['dieprice'] ** e)

# Predicted consumption at market prices
dwl['predictgas'] = dwl['Agas'] * dwl['spotgasprice'] ** e
dwl['predictdie'] = dwl['Adie'] * dwl['spotdieprice'] ** e

# DWL calculation (in billions)
# COST = (spot - price) * quantity
dwl['COSTgas'] = (dwl['spotgasprice'] - dwl['gasprice']) * dwl['gas']
dwl['COSTdie'] = (dwl['spotdieprice'] - dwl['dieprice']) * dwl['die']

# AREA = A/(1+e) * (spot^(1+e) - price^(1+e))
dwl['AREAgas'] = dwl['Agas'] / (1 + e) * (dwl['spotgasprice'] ** (1 + e) - dwl['gasprice'] ** (1 + e))
dwl['AREAdie'] = dwl['Adie'] / (1 + e) * (dwl['spotdieprice'] ** (1 + e) - dwl['dieprice'] ** (1 + e))

# DWL = (COST - AREA) / 1000 (converts millions to billions)
dwl['DWLgas'] = (dwl['COSTgas'] - dwl['AREAgas']) / 1000
dwl['DWLdie'] = (dwl['COSTdie'] - dwl['AREAdie']) / 1000

# Only count DWL for countries with prices below spot (subsidizers)
dwl.loc[dwl['gasprice'] > dwl['spotgasprice'], 'DWLgas'] = 0
dwl.loc[dwl['dieprice'] > dwl['spotdieprice'], 'DWLdie'] = 0
dwl['DWLtot'] = dwl['DWLgas'] + dwl['DWLdie']

# Report totals
total_dwl = dwl['DWLtot'].sum()
total_dwl_gas = dwl['DWLgas'].sum()
total_dwl_die = dwl['DWLdie'].sum()
print(f"\nTotal global DWL: ${total_dwl:.1f} billion")
print(f"  Gasoline DWL: ${total_dwl_gas:.1f} billion")
print(f"  Diesel DWL: ${total_dwl_die:.1f} billion")
print(f"  [Paper: $44B total, $20B gasoline, $24B diesel]")

# Saudi + Venezuela
sa_ven_dwl = dwl[dwl['country'].isin(['Saudi Arabia', 'Venezuela'])]['DWLtot'].sum()
print(f"\nSaudi Arabia + Venezuela DWL: ${sa_ven_dwl:.1f} billion")
print(f"  Share of total: {sa_ven_dwl/total_dwl:.2f}")
print(f"  [Paper: 50% of total]")

# Top 10
dwl_top10 = dwl.sort_values('DWLtot', ascending=False).head(10)
top10_dwl_sum = dwl_top10['DWLtot'].sum()
print(f"\nTop 10 DWL sum: ${top10_dwl_sum:.1f} billion")
print(f"  Share of total: {top10_dwl_sum/total_dwl:.2f}")

print(f"\nTop 10 DWL countries:")
print(f"{'Country':<20} {'DWLgas':>10} {'DWLdie':>10} {'DWLtot':>10}")
for _, r in dwl_top10.iterrows():
    print(f"{r['country']:<20} ${r['DWLgas']:>9.1f} ${r['DWLdie']:>9.1f} ${r['DWLtot']:>9.1f}")

# Table A1 - Top 10 gasoline
print(f"\n--- Table A1, Panel A: Gasoline ---")
gas_top = dwl[dwl['DWLgas'] > 0].sort_values('DWLgas', ascending=False).head(10)
print(f"{'Country':<15} {'Price':>8} {'Consump':>10} {'Predicted':>10} {'DWL':>8}")
for _, r in gas_top.iterrows():
    print(f"{r['country']:<15} ${r['gasprice']:>7.2f} {r['gas']:>10.0f} {r['predictgas']:>10.0f} {r['DWLgas']:>8.1f}")

print(f"\n--- Table A1, Panel B: Diesel ---")
die_top = dwl[dwl['DWLdie'] > 0].sort_values('DWLdie', ascending=False).head(10)
print(f"{'Country':<15} {'Price':>8} {'Consump':>10} {'Predicted':>10} {'DWL':>8}")
for _, r in die_top.iterrows():
    print(f"{r['country']:<15} ${r['dieprice']:>7.2f} {r['die']:>10.0f} {r['predictdie']:>10.0f} {r['DWLdie']:>8.1f}")

# DWL per capita
dwl['DWLgasPC'] = dwl['DWLgas'] * 1e9 / dwl['pop']
dwl['DWLdiePC'] = dwl['DWLdie'] * 1e9 / dwl['pop']
dwl['DWLtotPC'] = dwl['DWLgasPC'] + dwl['DWLdiePC']

dwlpc_top10 = dwl.sort_values('DWLtotPC', ascending=False).head(10)
print(f"\nTop 10 DWL per capita:")
print(f"{'Country':<20} {'DWLgasPC':>10} {'DWLdiePC':>10} {'DWLtotPC':>10}")
for _, r in dwlpc_top10.iterrows():
    print(f"{r['country']:<20} ${r['DWLgasPC']:>9.0f} ${r['DWLdiePC']:>9.0f} ${r['DWLtotPC']:>9.0f}")
print(f"  [Paper: Saudi Arabia ~$450 DWL per capita]")

# Overconsumption
gas_overcon = dwl[dwl['gasprice'] < dwl['spotgasprice']]
die_overcon = dwl[dwl['dieprice'] < dwl['spotdieprice']]
overcon_gas = gas_overcon['gas'].sum() - gas_overcon['predictgas'].sum()
overcon_die = die_overcon['die'].sum() - die_overcon['predictdie'].sum()
total_overcon = overcon_gas + overcon_die
print(f"\nOverconsumption (millions of gallons):")
print(f"  Gasoline: {overcon_gas:.0f}")
print(f"  Diesel: {overcon_die:.0f}")
print(f"  Total: {total_overcon:.0f}")
print(f"  Total (billions of gallons): {total_overcon/1000:.0f}")
print(f"  [Paper: 29 billion gallons]")

# Global market size
dwl['EXPgas'] = dwl['spotgasprice'] * dwl['gas']
dwl['EXPdie'] = dwl['spotdieprice'] * dwl['die']
dwl['EXPtot'] = dwl['EXPgas'] + dwl['EXPdie']
total_exp = dwl['EXPtot'].sum() / 1e6  # millions -> trillions
total_exp_gas = dwl['EXPgas'].sum() / 1e6
total_exp_die = dwl['EXPdie'].sum() / 1e6
print(f"\nGlobal market (trillions):")
print(f"  Gasoline: ${total_exp_gas:.2f}T")
print(f"  Diesel: ${total_exp_die:.2f}T")
print(f"  Total: ${total_exp:.2f}T")
print(f"  [Paper: $1.7 trillion]")

# DWL as share of market
print(f"  DWL as % of market: {total_dwl / (total_exp * 1000) * 100:.1f}%")
print(f"  [Paper: 4%]")

# US share of total consumption
us_row = dwl[dwl['country'] == 'United States']
total_gas = dwl['gas'].sum()
total_die = dwl['die'].sum()
total_tot = total_gas + total_die
us_gas = us_row['gas'].values[0]
us_die = us_row['die'].values[0]
us_tot = us_gas + us_die
print(f"\nUS share of total consumption: {us_tot/total_tot:.2f}")
print(f"  [Paper: ~30%]")

print("\n" + "=" * 70)
print("SECTION 3: SOCIAL MARGINAL COST (WITH EXTERNALITIES)")
print("=" * 70)

# Reload for SMC section
smc = pd.read_csv(f'{OUTPUT_DIR}/temp1.csv')
smc['spotgasprice'] = SPOT_GAS_PRICE
smc['spotdieprice'] = SPOT_DIE_PRICE
smc.loc[smc['netexports'] < 0, 'spotgasprice'] += TRANSPORT_COST_GALLON
smc.loc[smc['netexports'] < 0, 'spotdieprice'] += TRANSPORT_COST_GALLON

# Add external costs
smc['spotgasprice'] += EXTERNAL_COST
smc['spotdieprice'] += EXTERNAL_COST

smc['gas'] = smc['gascon'] * smc['pop'] / 1e6
smc['die'] = smc['diecon'] * smc['pop'] / 1e6
smc['Agas'] = smc['gas'] / (smc['gasprice'] ** e)
smc['Adie'] = smc['die'] / (smc['dieprice'] ** e)
smc['predictgas'] = smc['Agas'] * smc['spotgasprice'] ** e
smc['predictdie'] = smc['Adie'] * smc['spotdieprice'] ** e

smc['COSTgas'] = (smc['spotgasprice'] - smc['gasprice']) * smc['gas']
smc['COSTdie'] = (smc['spotdieprice'] - smc['dieprice']) * smc['die']
smc['AREAgas'] = smc['Agas'] / (1 + e) * (smc['spotgasprice'] ** (1 + e) - smc['gasprice'] ** (1 + e))
smc['AREAdie'] = smc['Adie'] / (1 + e) * (smc['spotdieprice'] ** (1 + e) - smc['dieprice'] ** (1 + e))
smc['DWLgas'] = (smc['COSTgas'] - smc['AREAgas']) / 1000
smc['DWLdie'] = (smc['COSTdie'] - smc['AREAdie']) / 1000

smc.loc[smc['gasprice'] > smc['spotgasprice'], 'DWLgas'] = 0
smc.loc[smc['dieprice'] > smc['spotdieprice'], 'DWLdie'] = 0
smc['DWLtot'] = smc['DWLgas'] + smc['DWLdie']

total_smc_dwl = smc['DWLtot'].sum()
print(f"\nTotal DWL (SMC): ${total_smc_dwl:.1f} billion")
print(f"  [Paper: $92 billion]")

# External cost of overconsumption
smc_gas_overcon = smc[smc['gasprice'] < smc['spotgasprice']]
smc_die_overcon = smc[smc['dieprice'] < smc['spotdieprice']]
overcon_total = (smc_gas_overcon['gas'].sum() - smc_gas_overcon['predictgas'].sum() +
                 smc_die_overcon['die'].sum() - smc_die_overcon['predictdie'].sum())
external_cost_total = overcon_total * EXTERNAL_COST / 1e6  # millions of gallons * $/gallon / 1e6 = trillions... no
# Actually: overconsumption in millions of gallons * $1.11/gallon = millions of dollars
# Convert to billions
external_cost_billions = total_overcon * EXTERNAL_COST / 1000  # using private-cost overconsumption
print(f"External costs of overconsumption: ${external_cost_billions:.1f} billion")
print(f"  [Paper: $32 billion]")

# Total economic cost
total_econ_cost = total_dwl + external_cost_billions
print(f"Total economic cost (DWL + externalities): ${total_econ_cost:.1f} billion")
print(f"  [Paper: $76 billion = $44B + $32B]")

# SMC top 10
smc_top10 = smc.sort_values('DWLtot', ascending=False).head(10)
print(f"\nTop 10 DWL (SMC) countries:")
print(f"{'Country':<20} {'DWLgas':>10} {'DWLdie':>10} {'DWLtot':>10}")
for _, r in smc_top10.iterrows():
    print(f"{r['country']:<20} ${r['DWLgas']:>9.1f} ${r['DWLdie']:>9.1f} ${r['DWLtot']:>9.1f}")

print("\n" + "=" * 70)
print("SECTION 4: SIMPLE REGRESSIONS")
print("=" * 70)

import statsmodels.api as sm

reg_df = pd.read_csv(f'{OUTPUT_DIR}/temp1.csv')
reg_df = reg_df[(reg_df['gasprice'] > 0) & reg_df['gasprice'].notna() &
                reg_df['gascon'].notna() & (reg_df['gascon'] > 0) &
                reg_df['pop'].notna()]

# Weighted regression: gascon on gasprice
X = sm.add_constant(reg_df['gasprice'])
mod = sm.WLS(reg_df['gascon'], X, weights=reg_df['pop']).fit()
print(f"\nWeighted OLS: gascon = a + b*gasprice")
print(f"  b = {mod.params['gasprice']:.3f}, SE = {mod.bse['gasprice']:.3f}")
print(f"  N = {mod.nobs:.0f}")

# Log-log: lncon = a + b*lnprice
reg_df['lncon'] = np.log(reg_df['gascon'])
reg_df['lnprice'] = np.log(reg_df['gasprice'])
reg_df['lngdp'] = np.log(reg_df['gdp'])

valid = reg_df.dropna(subset=['lncon', 'lnprice'])
X = sm.add_constant(valid['lnprice'])
mod2 = sm.OLS(valid['lncon'], X).fit()
print(f"\nOLS: ln(gascon) = a + b*ln(gasprice)")
print(f"  b (elasticity) = {mod2.params['lnprice']:.3f}")
print(f"  N = {mod2.nobs:.0f}")

# With GDP control
valid2 = reg_df.dropna(subset=['lncon', 'lnprice', 'lngdp'])
X = sm.add_constant(valid2[['lnprice', 'lngdp']])
mod3 = sm.OLS(valid2['lncon'], X).fit()
print(f"\nOLS: ln(gascon) = a + b*ln(gasprice) + c*ln(gdp)")
print(f"  b (elasticity) = {mod3.params['lnprice']:.3f}")
print(f"  c (income elast) = {mod3.params['lngdp']:.3f}")
print(f"  N = {mod3.nobs:.0f}")

# Save DWL data for figures
dwl.to_csv(f'{OUTPUT_DIR}/dwl_private.csv', index=False)
smc.to_csv(f'{OUTPUT_DIR}/dwl_smc.csv', index=False)
print(f"\nSaved dwl_private.csv and dwl_smc.csv")
