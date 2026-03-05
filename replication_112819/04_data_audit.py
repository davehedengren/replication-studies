"""
04_data_audit.py - Data quality audit for Davis (2014) replication.

Checks:
- Coverage: obs count, country count, variable completeness
- Distributions: summary stats, plausibility bounds, outliers
- Logical consistency: net exports, subsidy calculations
- Missing data patterns
- Cross-variable consistency
"""
import numpy as np
import pandas as pd
from utils import (OUTPUT_DIR, SPOT_GAS_PRICE, SPOT_DIE_PRICE,
                   TRANSPORT_COST_GALLON, GALLONS_PER_LITER)

df = pd.read_csv(f'{OUTPUT_DIR}/temp1.csv')

print("=" * 70)
print("DATA AUDIT: Davis (2014) Global Fuel Subsidies")
print("=" * 70)

# ============================================================
# 1. COVERAGE
# ============================================================
print("\n--- 1. COVERAGE ---")
print(f"Total countries in dataset: {len(df)}")
print(f"\nVariable completeness:")
for col in ['gasprice', 'dieprice', 'gascon', 'diecon', 'pop', 'gdp', 'netexports']:
    n_valid = df[col].notna().sum()
    pct = n_valid / len(df) * 100
    print(f"  {col:<15} {n_valid:>5} / {len(df)} ({pct:.0f}%)")

# Countries with complete data (all key vars)
complete = df.dropna(subset=['gasprice', 'dieprice', 'gascon', 'diecon', 'pop'])
print(f"\nCountries with all key variables: {len(complete)}")

# Countries with gas price + consumption (Figure 1 sample)
fig1 = df[(df['gasprice'] > 0) & df['gasprice'].notna() & df['gascon'].notna()]
print(f"Countries for Figure 1 (gas price + consumption): {len(fig1)}")
print(f"  [Paper: 128 countries]")

# ============================================================
# 2. DISTRIBUTIONS
# ============================================================
print("\n--- 2. DISTRIBUTIONS ---")
for col in ['gasprice', 'dieprice', 'gascon', 'diecon', 'pop', 'gdp', 'netexports']:
    valid = df[col].dropna()
    if len(valid) > 0:
        print(f"\n  {col}:")
        print(f"    N={len(valid)}, mean={valid.mean():.2f}, "
              f"sd={valid.std():.2f}, min={valid.min():.2f}, max={valid.max():.2f}")
        q1, q3 = valid.quantile(0.25), valid.quantile(0.75)
        iqr = q3 - q1
        n_outlier = ((valid < q1 - 1.5*iqr) | (valid > q3 + 1.5*iqr)).sum()
        print(f"    Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}, outliers={n_outlier}")

# ============================================================
# 3. PLAUSIBILITY CHECKS
# ============================================================
print("\n--- 3. PLAUSIBILITY CHECKS ---")

# Gas prices should be positive
neg_gas = df[df['gasprice'] < 0]
print(f"Negative gasoline prices: {len(neg_gas)}")
neg_die = df[df['dieprice'] < 0]
print(f"Negative diesel prices: {len(neg_die)}")

# Consumption should be positive
neg_gascon = df[df['gascon'] < 0]
print(f"Negative gasoline consumption: {len(neg_gascon)}")
neg_diecon = df[df['diecon'] < 0]
print(f"Negative diesel consumption: {len(neg_diecon)}")

# Population should be > 0
zero_pop = df[df['pop'] <= 0]
print(f"Zero/negative population: {len(zero_pop)}")

# Very high gas consumption (> 500 gal/capita = ~10 gal/week)
high_gascon = df[df['gascon'] > 500]
print(f"\nUnusually high gasoline consumption (>500 gal/cap/yr):")
if len(high_gascon) > 0:
    print(high_gascon[['country', 'gascon', 'pop']].to_string(index=False))
else:
    print("  None")

# Very low gas prices (< $0.25/gallon)
low_gas = df[(df['gasprice'] < 0.25) & df['gasprice'].notna()]
print(f"\nVery low gasoline prices (<$0.25/gal):")
for _, r in low_gas.iterrows():
    print(f"  {r['country']}: ${r['gasprice']:.2f}/gal")

# Very high diesel consumption (> 200 gal/capita)
high_diecon = df[df['diecon'] > 200]
print(f"\nUnusually high diesel consumption (>200 gal/cap/yr):")
if len(high_diecon) > 0:
    for _, r in high_diecon.iterrows():
        print(f"  {r['country']}: {r['diecon']:.0f} gal/cap/yr")
else:
    print("  None")

# Qatar and Luxembourg flagged in code for high diesel
for c in ['Qatar', 'Luxembourg']:
    row = df[df['country'] == c]
    if len(row) > 0:
        dc = row['diecon'].values[0]
        gc = row['gascon'].values[0]
        print(f"  {c}: gascon={gc:.0f}, diecon={dc:.0f} gal/cap/yr (flagged in code)")

# ============================================================
# 4. LOGICAL CONSISTENCY
# ============================================================
print("\n--- 4. LOGICAL CONSISTENCY ---")

# Price ordering: countries with diesel > gasoline (unusual in many markets)
both_prices = df[df['gasprice'].notna() & df['dieprice'].notna()]
die_gt_gas = both_prices[both_prices['dieprice'] > both_prices['gasprice']]
print(f"Countries where diesel > gasoline price: {len(die_gt_gas)} of {len(both_prices)}")

# Net exporters with transport cost applied (should be none - they're exporters)
# This checks the code logic
net_importers = df[df['netexports'] < 0]
net_exporters = df[df['netexports'] >= 0]
print(f"Net importers: {len(net_importers)}")
print(f"Net exporters: {len(net_exporters)}")
print(f"  (Importers get higher spot price due to transport costs)")

# US should not subsidize
us = df[df['country'] == 'United States']
spot_gas_us = SPOT_GAS_PRICE + TRANSPORT_COST_GALLON  # US is net importer
print(f"\nUS check:")
print(f"  Gas price: ${us['gasprice'].values[0]:.2f}, Spot: ${spot_gas_us:.2f}")
print(f"  US subsidizes gasoline? {us['gasprice'].values[0] < spot_gas_us}")

# ============================================================
# 5. MISSING DATA PATTERNS
# ============================================================
print("\n--- 5. MISSING DATA PATTERNS ---")

# Countries missing prices but having consumption
has_con_no_price = df[df['gascon'].notna() & df['gasprice'].isna()]
print(f"Countries with gas consumption but no price: {len(has_con_no_price)}")
if len(has_con_no_price) > 0:
    for _, r in has_con_no_price.head(5).iterrows():
        print(f"  {r['country']}")

has_price_no_con = df[df['gasprice'].notna() & df['gascon'].isna()]
print(f"Countries with gas price but no consumption: {len(has_price_no_con)}")
if len(has_price_no_con) > 0:
    for _, r in has_price_no_con.head(5).iterrows():
        print(f"  {r['country']}")

# Missing GDP
no_gdp = df[df['gdp'].isna()]
print(f"\nCountries missing GDP: {len(no_gdp)}")

# ============================================================
# 6. CROSS-VALIDATION
# ============================================================
print("\n--- 6. CROSS-VALIDATION ---")

# Compare gasoline prices to known values
known_prices = {
    'Venezuela': 0.09,
    'Saudi Arabia': 0.61,
    'United States': 3.67,  # approx from data
}
print("Gasoline price cross-validation ($/gallon):")
for country, expected in known_prices.items():
    row = df[df['country'] == country]
    if len(row) > 0:
        actual = row['gasprice'].values[0]
        diff = abs(actual - expected)
        print(f"  {country}: expected~${expected:.2f}, actual=${actual:.2f}, diff=${diff:.2f}")

# Population cross-validation (approximate 2010 values in millions)
known_pop = {
    'China': 1340e6,
    'India': 1234e6,
    'United States': 309e6,
    'Indonesia': 240e6,
    'Saudi Arabia': 27e6,
}
print("\nPopulation cross-validation:")
for country, expected in known_pop.items():
    row = df[df['country'] == country]
    if len(row) > 0:
        actual = row['pop'].values[0]
        pct_diff = (actual - expected) / expected * 100
        print(f"  {country}: expected~{expected/1e6:.0f}M, actual={actual/1e6:.0f}M ({pct_diff:+.1f}%)")

# ============================================================
# 7. SENSITIVITY OF KEY RESULTS TO DATA QUALITY
# ============================================================
print("\n--- 7. KEY RESULT SENSITIVITY ---")

# Check how sensitive total subsidies are to individual country data
sub_df = df.copy()
sub_df['spotgasprice'] = SPOT_GAS_PRICE
sub_df['spotdieprice'] = SPOT_DIE_PRICE
sub_df.loc[sub_df['netexports'] < 0, 'spotgasprice'] += TRANSPORT_COST_GALLON
sub_df.loc[sub_df['netexports'] < 0, 'spotdieprice'] += TRANSPORT_COST_GALLON
sub_df['GasSub'] = (sub_df['spotgasprice'] - sub_df['gasprice']) * sub_df['gascon'] * sub_df['pop'] / 1e9
sub_df['DieSub'] = (sub_df['spotdieprice'] - sub_df['dieprice']) * sub_df['diecon'] * sub_df['pop'] / 1e9
sub_df['TotSub'] = sub_df['GasSub'] + sub_df['DieSub']

total_sub = sub_df.loc[sub_df['TotSub'] > 0, 'TotSub'].sum()
print(f"Total subsidies: ${total_sub:.1f}B")

# Leave-one-out for top 5
top5 = sub_df.sort_values('TotSub', ascending=False).head(5)
for _, r in top5.iterrows():
    remaining = total_sub - r['TotSub']
    pct_change = (remaining - total_sub) / total_sub * 100
    print(f"  Without {r['country']}: ${remaining:.1f}B ({pct_change:+.1f}%)")

print("\n--- 8. UNIT CONVERSION VERIFICATION ---")
# Verify unit conversions match paper appendix example
# Saudi Arabia gasoline: price $0.61, consumption 5637 million gallons
sa = df[df['country'] == 'Saudi Arabia']
sa_gasprice = sa['gasprice'].values[0]
sa_gascon = sa['gascon'].values[0]
sa_pop = sa['pop'].values[0]
sa_gas_total = sa_gascon * sa_pop / 1e6  # millions of gallons
print(f"Saudi Arabia gasoline:")
print(f"  Price: ${sa_gasprice:.2f}/gal [Paper: $0.61]")
print(f"  Total consumption: {sa_gas_total:.0f} million gallons [Paper: 5,637]")
# DWL example from appendix
A = sa_gas_total / (sa_gasprice ** (-0.6))
predicted = A * 2.817 ** (-0.6)
dwl_ex = (2.817 - sa_gasprice) * sa_gas_total - A / (1 + (-0.6)) * (2.817 ** 0.4 - sa_gasprice ** 0.4)
dwl_ex_B = dwl_ex / 1000
print(f"  Scale parameter A: {A:.0f} [Paper: 4190]")
print(f"  Predicted consumption at market: {predicted:.0f} [Paper: 2241]")
print(f"  DWL: ${dwl_ex_B:.1f}B [Paper: $5.2B]")

print("\nData audit complete.")
