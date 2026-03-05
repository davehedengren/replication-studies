"""
01_clean.py - Load and verify data for Paper 112799
Arceo-Gomez & Campos-Vazquez (2014)

Data is already clean from the replication package.
This script verifies sample sizes and variable distributions match Table 1.
"""

import sys
sys.path.insert(0, '.')
from replication_112799.utils import *

df = load_data()

print("=" * 70)
print("PHASE 1: DATA LOADING AND VERIFICATION")
print("=" * 70)

# ── Basic sample sizes ─────────────────────────────────────────────────────
print(f"\nTotal observations: {len(df)}")
print(f"  Published: 8,149  |  Ours: {len(df)}  |  Match: {len(df) == 8149}")

n_men = (df.sex == 0).sum()
n_women = (df.sex == 1).sum()
print(f"\nMen:   Published: 3,992  |  Ours: {n_men}  |  Match: {n_men == 3992}")
print(f"Women: Published: 4,157  |  Ours: {n_women}  |  Match: {n_women == 4157}")

# ── Table 1 verification: Descriptive statistics ───────────────────────────
print("\n" + "=" * 70)
print("TABLE 1 VERIFICATION: DESCRIPTIVE STATISTICS")
print("=" * 70)

# Published Table 1 values
table1_published = {
    'Men (share)':     {'All': 0.49, 'Men': None, 'Women': None},
    'Women (share)':   {'All': 0.51, 'Men': None, 'Women': None},
    'Business':        {'All': 0.71, 'Men': 0.70, 'Women': 0.73},
    'Engineering':     {'All': 0.29, 'Men': 0.30, 'Women': 0.27},
    'Public univ':     {'All': 0.62, 'Men': 0.64, 'Women': 0.61},
    'Private univ':    {'All': 0.38, 'Men': 0.36, 'Women': 0.39},
    'Married':         {'All': 0.27, 'Men': 0.29, 'Women': 0.26},
    'Age':             {'All': 24.5, 'Men': 24.6, 'Women': 24.4},
    'Scholarship':     {'All': 0.26, 'Men': 0.23, 'Women': 0.28},
    'Leadership':      {'All': 0.50, 'Men': 0.49, 'Women': 0.51},
    'Foreign lang':    {'All': 0.25, 'Men': 0.25, 'Women': 0.25},
    'Time avail':      {'All': 0.50, 'Men': 0.51, 'Women': 0.50},
}

# Compute our values
# ss_degree: 1=social science/business, 0=engineering
our_vals = {}
our_vals['Men (share)'] = {'All': (df.sex == 0).mean()}
our_vals['Women (share)'] = {'All': (df.sex == 1).mean()}

men = df[df.sex == 0]
women = df[df.sex == 1]

our_vals['Business'] = {
    'All': df.ss_degree.mean(),
    'Men': men.ss_degree.mean(),
    'Women': women.ss_degree.mean()
}
our_vals['Engineering'] = {
    'All': 1 - df.ss_degree.mean(),
    'Men': 1 - men.ss_degree.mean(),
    'Women': 1 - women.ss_degree.mean()
}
our_vals['Public univ'] = {
    'All': df.public_college.mean(),
    'Men': men.public_college.mean(),
    'Women': women.public_college.mean()
}
our_vals['Private univ'] = {
    'All': 1 - df.public_college.mean(),
    'Men': 1 - men.public_college.mean(),
    'Women': 1 - women.public_college.mean()
}
our_vals['Married'] = {
    'All': df.married.mean(),
    'Men': men.married.mean(),
    'Women': women.married.mean()
}
our_vals['Age'] = {
    'All': df.age.mean(),
    'Men': men.age.mean(),
    'Women': women.age.mean()
}
our_vals['Scholarship'] = {
    'All': df.scholarship.mean(),
    'Men': men.scholarship.mean(),
    'Women': women.scholarship.mean()
}
our_vals['Leadership'] = {
    'All': df.leadership.mean(),
    'Men': men.leadership.mean(),
    'Women': women.leadership.mean()
}
our_vals['Foreign lang'] = {
    'All': df.other_language.mean(),
    'Men': men.other_language.mean(),
    'Women': women.other_language.mean()
}
our_vals['Time avail'] = {
    'All': df.some_availab.mean(),
    'Men': men.some_availab.mean(),
    'Women': women.some_availab.mean()
}

print(f"\n{'Variable':<20} {'Col':<8} {'Published':>10} {'Ours':>10} {'Match':>6}")
print("-" * 60)
for var in table1_published:
    for col in ['All', 'Men', 'Women']:
        pub = table1_published[var].get(col)
        if pub is None:
            continue
        ours = our_vals[var].get(col)
        if ours is not None:
            match = abs(pub - round(ours, 2)) < 0.015
            print(f"{var:<20} {col:<8} {pub:>10.2f} {ours:>10.3f} {'OK' if match else 'DIFF':>6}")

# ── Variable distributions ─────────────────────────────────────────────────
print("\n" + "=" * 70)
print("VARIABLE DISTRIBUTIONS")
print("=" * 70)

print("\nPhoto distribution:")
for p in [1, 2, 3, 4]:
    n = (df.photo == p).sum()
    pct = n / len(df) * 100
    labels = {1: 'European', 2: 'Mestizo', 3: 'Indigenous', 4: 'No photo'}
    print(f"  Photo {p} ({labels[p]:>12}): {n:>5} ({pct:.1f}%)")

print(f"\nall8 sample: {(df.all8 == 1).sum()} ({(df.all8 == 1).mean()*100:.1f}%)")
print(f"  Published: 78.7%  |  Ours: {(df.all8 == 1).mean()*100:.1f}%")

print(f"\nUnique firms (id_offer): {df.id_offer.nunique()}")
print(f"CVs per firm: mean={df.groupby('id_offer').size().mean():.1f}, "
      f"median={df.groupby('id_offer').size().median():.0f}")

# ── Callback rates ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("CALLBACK RATES (Table 2 verification)")
print("=" * 70)

print(f"\nOverall callback: {df.callback.mean()*100:.2f}%")
print(f"Men callback:    {men.callback.mean()*100:.2f}%")
print(f"Women callback:  {women.callback.mean()*100:.2f}%")
print(f"  Published: Men 10.67%, Women 14.94%")

# ── Missing data check ─────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("MISSING DATA CHECK")
print("=" * 70)
for col in df.columns:
    n_miss = df[col].isna().sum()
    if n_miss > 0:
        print(f"  {col}: {n_miss} missing ({n_miss/len(df)*100:.1f}%)")
if df.isna().sum().sum() == 0:
    print("  No missing values in any variable.")

# ── Data types ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("DATA TYPES (note: Stata float32 may cause minor precision diffs)")
print("=" * 70)
for col in df.columns:
    print(f"  {col:<20} {str(df[col].dtype):<10} unique={df[col].nunique():<6} "
          f"min={df[col].min():<8} max={df[col].max()}")

print("\nData verification complete.")
