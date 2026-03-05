"""Load and clean data for paper 112818 replication.

Translates AEA.do: load figure1.dta, keep year>=1950, normalize news and
regulation to mean=100.
"""
import pandas as pd
import numpy as np
from utils import DATA_FILE, OUTPUT_DIR
import os

# Load data
df = pd.read_stata(DATA_FILE, convert_categoricals=False)
print(f"Raw data: {len(df)} obs, {df.columns.tolist()}")
print(f"Year range: {df['year'].min()} - {df['year'].max()}")
print()

# Filter year >= 1950 (matching Stata: keep if year>=1950)
df = df[df['year'] >= 1950].reset_index(drop=True)
print(f"After year>=1950 filter: {len(df)} obs")
print(f"Year range: {df['year'].min()} - {df['year'].max()}")
print()

# Normalize news to mean=100 (Stata: egen mnews=mean(news); gen norm_news=100*news/mnews)
mnews = df['news'].mean()
df['norm_news'] = 100 * df['news'] / mnews
print(f"News mean: {mnews:.4f}")
print(f"norm_news range: {df['norm_news'].min():.2f} - {df['norm_news'].max():.2f}")

# Normalize regulation to mean=100 (Stata: egen mreg=mean(regulation); gen norm_reg=100*reg/mreg)
# Note: Stata uses 'reg' not 'regulation' in the gen command - check which column exists
if 'regulation' in df.columns:
    mreg = df['regulation'].mean()
    df['norm_reg'] = 100 * df['regulation'] / mreg
    print(f"Regulation mean: {mreg:.4f}")
elif 'reg' in df.columns:
    mreg = df['reg'].mean()
    df['norm_reg'] = 100 * df['reg'] / mreg
    print(f"Reg mean: {mreg:.4f}")
print(f"norm_reg range: {df['norm_reg'].min():.2f} - {df['norm_reg'].max():.2f}")

print()
print("Variables available:")
for col in df.columns:
    n_valid = df[col].notna().sum()
    print(f"  {col}: {n_valid}/{len(df)} non-missing")

# Summary stats
print()
print("Summary statistics (post-filter):")
print(df.describe().round(4))

# Save cleaned data
out_path = os.path.join(OUTPUT_DIR, "cleaned.csv")
df.to_csv(out_path, index=False)
print(f"\nSaved cleaned data to {out_path}")
