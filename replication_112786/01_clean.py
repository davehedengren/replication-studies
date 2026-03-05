"""
01_clean.py - Load and verify data for replication of paper 112786
Rosenzweig & Udry (2014) - Rainfall Forecasts, Weather and Wages
"""
import sys
sys.path.insert(0, '.')
from utils import *

print("=" * 60)
print("Data Loading and Verification")
print("=" * 60)

harvest, planting, migration = load_data()

# Verify shapes match paper
print(f"\nHarvest wage: {harvest.shape[0]} obs, {harvest['id'].nunique()} districts")
print(f"  Expected: 337 obs, 95 districts")
print(f"  Columns: {list(harvest.columns)}")

print(f"\nPlanting wage: {planting.shape[0]} obs, {planting['id'].nunique()} districts")
print(f"  Expected: 387 obs, 106 districts")
print(f"  Columns: {list(planting.columns)}")

print(f"\nMigration: {migration.shape[0]} obs, {migration['vill'].nunique()} villages")
print(f"  Expected: 6,501 obs")
print(f"  Males: {(migration['male']==1).sum()}, Females: {(migration['male']==0).sum()}")
print(f"  Expected: Males 3,507, Females 2,994")
print(f"  Columns: {list(migration.columns)}")

# Table 1 descriptive stats verification
print("\n" + "=" * 60)
print("Table 1 Verification: Descriptive Statistics")
print("=" * 60)

# ICRISAT VLS panel (migration data)
# Paper says males 15-39 migration rate = .157
print("\nICRISAT VLS India, 2005-2011:")
males = migration[migration['male'] == 1]
females = migration[migration['male'] == 0]
print(f"  Male migration rate:    mean={migration[migration['male']==1]['mig'].mean():.4f}  (paper: .157)")
print(f"  Female migration rate:  mean={migration[migration['male']==0]['mig'].mean():.4f}  (paper: .0467)")
print(f"  Age:                    mean={migration['age'].mean():.1f}, SD={migration['age'].std():.1f}  (paper: 33.3, 18.8)")
print(f"  Schooling:              mean={migration['edu'].mean():.2f}, SD={migration['edu'].std():.2f}  (paper: 5.37, 4.46)")
print(f"  Forecast:               mean={migration['sp'].mean():.1f}, SD={migration['sp'].std():.2f}  (paper: 96.5, 2.36)")

# Note: Table 1 reports stats for age 15-39 subgroup for migration rates
age_mask = (migration['age'] >= 15) & (migration['age'] <= 39)
males_1539 = migration[(migration['male'] == 1) & age_mask]
females_1539 = migration[(migration['male'] == 0) & age_mask]
print(f"\n  Males 15-39 migration:  mean={males_1539['mig'].mean():.4f}, SD={males_1539['mig'].std():.3f}  (paper: .157, .364)")
print(f"  Females 15-39 migration: mean={females_1539['mig'].mean():.4f}, SD={females_1539['mig'].std():.3f}  (paper: .0467, .211)")

# District wage data
print("\nDistrict Male Agricultural Daily Wages, 2005-2010:")
print(f"  Planting wage:  mean={np.exp(planting['lwager']).mean():.1f}, SD={np.exp(planting['lwager']).std():.1f}  (paper: 71.6, 32.2)")
print(f"  Harvest wage:   mean={np.exp(harvest['lwager']).mean():.1f}, SD={np.exp(harvest['lwager']).std():.1f}  (paper: 73.0, 29.4)")
print(f"  Rainfall shock (planting): mean={planting['jsdev'].mean():.2f}, SD={planting['jsdev'].std():.3f}  (paper: 1.04, .293)")
print(f"  Rainfall shock (harvest):  mean={harvest['jsdev'].mean():.2f}, SD={harvest['jsdev'].std():.3f}  (paper: 1.04, .293)")

# Bad rainfall shock indicator
harvest_bad = (harvest['jsdev'] < 1).mean()
planting_bad = (planting['jsdev'] < 1).mean()
print(f"  Bad rainfall shock (harvest): mean={harvest_bad:.3f}  (paper: .347)")
print(f"  Bad rainfall shock (planting): mean={planting_bad:.3f}  (paper: .347)")

# Distance to weather station
print(f"  Distance (planting):    mean={planting['jsdevmiles'].mean() / planting['jsdev'].mean():.1f}  (approx, paper: 82.1)")
print(f"  Forecast (planting):    mean={planting['fore'].mean():.1f}, SD={planting['fore'].std():.2f}  (paper: 95.1, 4.12)")
print(f"  Forecast (harvest):     mean={harvest['fore'].mean():.1f}, SD={harvest['fore'].std():.2f}  (paper: 95.1, 4.12)")
print(f"  NREGA (planting):       mean={planting['post'].mean():.3f}  (paper: .615)")
print(f"  NREGA (harvest):        mean={harvest['post'].mean():.3f}  (paper: .615)")

# Check NREGA in harvest
print(f"\n  Harvest NREGA mean: {harvest['post'].mean():.3f}")
print(f"  Harvest postbad mean: {harvest['postbad'].mean():.3f}")
print(f"  Harvest foregood mean: {harvest['foregood'].mean():.3f}")
