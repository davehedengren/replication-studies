"""
01_clean.py – Load and prepare data for Samaniego (2008) replication.
Constructs industry-level indices from the panel data (Table 1).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import statsmodels.api as sm
from utils import load_industry_data, load_did_data

# ── Load raw data ──────────────────────────────────────
df_ind = load_industry_data()
df_did = load_did_data()

print("=" * 60)
print("INDUSTRY VARIABLES (41 industries)")
print("=" * 60)
print(f"Shape: {df_ind.shape}")
print(f"Columns: {list(df_ind.columns)}")
print(df_ind[['Industry', 'istc', 'istceq', 'turnover', 'entry', 'exit', 'lumpy']].to_string())

print("\n" + "=" * 60)
print("DIFF-IN-DIFF PANEL (18 countries × 41 industries)")
print("=" * 60)
print(f"Shape: {df_did.shape}")
print(f"Countries: {df_did['c'].nunique()} (codes {df_did['c'].min()}-{df_did['c'].max()})")
print(f"Industries: {df_did['i'].nunique()} (codes {df_did['i'].min()}-{df_did['i'].max()})")
print(f"Manufacturing: {int(df_did['manuf'].sum())} obs, Non-manuf: {int((1-df_did['manuf']).sum())} obs")

# ── Missing data ───────────────────────────────────────
print("\n--- Missing data in DiD panel ---")
for col in ['entry', 'exit', 'turnover']:
    n_obs = df_did[col].notna().sum()
    n_miss = df_did[col].isna().sum()
    print(f"  {col}: {n_obs} obs, {n_miss} missing")

# ── Construct industry indices (Table 1) ───────────────
# Per paper: y_{j,c} = Σ α_c I_c + Σ α_j I_j + ε_{j,c}
# Index = α_j + α_c(median country)
# Paper says median country is Denmark

print("\n" + "=" * 60)
print("CONSTRUCTING INDUSTRY INDICES (Table 1)")
print("=" * 60)

indices = {}
for yvar in ['turnover', 'entry', 'exit']:
    temp = df_did[['c', 'i', yvar]].dropna().copy()
    c_dum = pd.get_dummies(temp['c'], prefix='c', drop_first=False).astype(float)
    i_dum = pd.get_dummies(temp['i'], prefix='i', drop_first=False).astype(float)

    # No constant since we have all dummies (no drop_first)
    # Actually for identification, drop one country and one industry
    # But we want all FEs. Use: drop one country as reference, keep all industries.
    # Then index = industry FE + median country FE
    # Better: use drop_first=True for countries (reference = country 1)
    # Keep all industry dummies

    c_dum_ref = pd.get_dummies(temp['c'], prefix='c', drop_first=True).astype(float)
    i_dum_all = pd.get_dummies(temp['i'], prefix='i', drop_first=True).astype(float)

    X = pd.concat([c_dum_ref.reset_index(drop=True),
                   i_dum_all.reset_index(drop=True)], axis=1)
    X = sm.add_constant(X)
    y = temp[yvar].reset_index(drop=True)

    model = sm.OLS(y, X).fit()

    # Extract industry effects
    # The constant absorbs the reference country (c=1) and reference industry (i=1)
    industry_effects = {}
    industry_effects[1] = model.params['const']  # reference industry
    for col in i_dum_all.columns:
        i_num = int(col.replace('i_', ''))
        industry_effects[i_num] = model.params['const'] + model.params[col]

    # Find median country effect
    # Country 1 is reference (effect = 0)
    country_effects = {1: 0.0}
    for col in c_dum_ref.columns:
        c_num = int(col.replace('c_', ''))
        country_effects[c_num] = model.params[col]

    # Median country FE
    c_vals = sorted(country_effects.values())
    median_c = np.median(c_vals)

    # Index = industry effect + median country effect
    for i_num in industry_effects:
        industry_effects[i_num] += median_c

    indices[yvar] = industry_effects

    # Find which country is median
    median_country = min(country_effects, key=lambda k: abs(country_effects[k] - median_c))
    print(f"  {yvar}: median country = {median_country} (effect = {country_effects[median_country]:.3f})")

# Build index table
df_ind_sorted = df_ind.sort_values('istc').reset_index(drop=True)
index_rows = []
for _, row in df_ind_sorted.iterrows():
    i_code = _ + 1  # industries are 1-indexed based on istc sort?
    # Actually we don't know the mapping from industry name to i code
    # Let me figure this out by matching

# Actually, I need to map i codes to industry names
# The industry variables file has 41 rows, and the diff-in-diff file has i=1..41
# Let's check if they correspond to the same ordering
print("\n--- Industry code mapping ---")
print(f"Industry file has {len(df_ind)} rows")
print(f"DiD file has industries 1-{df_did['i'].max()}")

# The industry data appears sorted differently. Let's try matching by creating
# indices from the regression and comparing to the industry file values
# to figure out the mapping.

# Build a comparison: for each i code, compute the index and see which industry matches
index_table = pd.DataFrame({
    'i': range(1, 42),
    'turnover_idx': [indices['turnover'].get(i, np.nan) for i in range(1, 42)],
    'entry_idx': [indices['entry'].get(i, np.nan) for i in range(1, 42)],
    'exit_idx': [indices['exit'].get(i, np.nan) for i in range(1, 42)],
})

# Also get ISTC from etcst in the did data (it's constant within industry)
istc_by_i = df_did.groupby('i')['etcst'].first().reset_index()
index_table = index_table.merge(istc_by_i, on='i', how='left')

# Match to industry names using the industry file
# The industry file rows may correspond to i=1..41 in order
df_ind['i'] = range(1, 42)
index_table = index_table.merge(df_ind[['i', 'Industry', 'istc', 'lumpy']], on='i', how='left')

print("\nConstructed indices vs published Table 1 (sample):")
index_table_sorted = index_table.sort_values('istc')
print(index_table_sorted[['Industry', 'istc', 'turnover_idx', 'entry_idx', 'exit_idx']].head(10).to_string())

# Verify industry count in manuf/non-manuf
manuf_industries = df_did[df_did['manuf'] == 1]['i'].unique()
nonmanuf_industries = df_did[df_did['manuf'] == 0]['i'].unique()
print(f"\nManufacturing industries: {len(manuf_industries)} (i={sorted(manuf_industries)})")
print(f"Non-manufacturing industries: {len(nonmanuf_industries)} (i={sorted(nonmanuf_industries)})")

# Expected from paper: 18 countries, manuf obs ~283 for turnover
manuf_turnover = df_did[(df_did['manuf'] == 1) & df_did['turnover'].notna()]
nonmanuf_turnover = df_did[(df_did['manuf'] == 0) & df_did['turnover'].notna()]
print(f"Manuf turnover obs: {len(manuf_turnover)}")
print(f"Non-manuf turnover obs: {len(nonmanuf_turnover)}")

print("\n" + "=" * 60)
print("DATA READY FOR ANALYSIS")
print("=" * 60)
