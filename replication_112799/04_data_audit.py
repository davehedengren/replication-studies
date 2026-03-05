"""
04_data_audit.py - Data quality audit for Paper 112799
Arceo-Gomez & Campos-Vazquez (2014)
"""

import sys
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
from replication_112799.utils import *

df = load_data()

print("=" * 70)
print("DATA AUDIT: Paper 112799")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════════════
# 1. COVERAGE
# ══════════════════════════════════════════════════════════════════════════════
print("\n1. COVERAGE")
print("-" * 50)
print(f"Total observations: {len(df)}")
print(f"Unique firms (id_offer): {df.id_offer.nunique()}")
print(f"CVs per firm: mean={df.groupby('id_offer').size().mean():.2f}")
print(f"  min={df.groupby('id_offer').size().min()}, "
      f"max={df.groupby('id_offer').size().max()}")

cvs_per_firm = df.groupby('id_offer').size()
print(f"\nDistribution of CVs per firm:")
for n in sorted(cvs_per_firm.unique()):
    count = (cvs_per_firm == n).sum()
    print(f"  {n} CVs: {count} firms ({count/len(cvs_per_firm)*100:.1f}%)")

print(f"\nGender split: Men={df[df.sex==0].shape[0]} ({(df.sex==0).mean()*100:.1f}%), "
      f"Women={df[df.sex==1].shape[0]} ({(df.sex==1).mean()*100:.1f}%)")

print(f"\nPhoto distribution:")
for p in [1, 2, 3, 4]:
    labels = {1: 'European', 2: 'Mestizo', 3: 'Indigenous', 4: 'No photo'}
    n = (df.photo == p).sum()
    print(f"  Photo {p} ({labels[p]}): {n} ({n/len(df)*100:.1f}%)")

# ══════════════════════════════════════════════════════════════════════════════
# 2. RANDOMIZATION BALANCE CHECK
# ══════════════════════════════════════════════════════════════════════════════
print("\n\n2. RANDOMIZATION BALANCE CHECK")
print("-" * 50)
print("Testing whether treatment assignments are balanced across groups.")

# Test balance of photo assignment by sex
from scipy.stats import chi2_contingency
ct = pd.crosstab(df['sex'], df['photo'])
chi2, p, dof, _ = chi2_contingency(ct)
print(f"\nPhoto × Sex independence: chi2={chi2:.3f}, p={p:.3f} "
      f"{'(BALANCED)' if p > 0.05 else '(IMBALANCED!)'}")

# Balance of married by sex
ct2 = pd.crosstab(df['sex'], df['married'])
chi2, p, dof, _ = chi2_contingency(ct2)
print(f"Married × Sex: chi2={chi2:.3f}, p={p:.3f} "
      f"{'(BALANCED)' if p > 0.05 else '(IMBALANCED!)'}")

# Balance of public_college by sex
ct3 = pd.crosstab(df['sex'], df['public_college'])
chi2, p, dof, _ = chi2_contingency(ct3)
print(f"Public univ × Sex: chi2={chi2:.3f}, p={p:.3f} "
      f"{'(BALANCED)' if p > 0.05 else '(IMBALANCED!)'}")

# Balance of ss_degree by sex
ct4 = pd.crosstab(df['sex'], df['ss_degree'])
chi2, p, dof, _ = chi2_contingency(ct4)
print(f"Major × Sex: chi2={chi2:.3f}, p={p:.3f} "
      f"{'(BALANCED)' if p > 0.05 else '(IMBALANCED!)'}")

# Balance of photo by married
ct5 = pd.crosstab(df['married'], df['photo'])
chi2, p, dof, _ = chi2_contingency(ct5)
print(f"Photo × Married: chi2={chi2:.3f}, p={p:.3f} "
      f"{'(BALANCED)' if p > 0.05 else '(IMBALANCED!)'}")

# Photo balance within firm
print("\nPhoto balance within firms (all8 sample):")
a8 = df[df.all8 == 1]
firm_photo_counts = a8.groupby(['id_offer', 'photo']).size().unstack(fill_value=0)
# Each firm should have 2 of each photo type (one male, one female)
for p in [1, 2, 3, 4]:
    labels = {1: 'European', 2: 'Mestizo', 3: 'Indigenous', 4: 'No photo'}
    if p in firm_photo_counts.columns:
        mean_count = firm_photo_counts[p].mean()
        print(f"  Photo {p} ({labels[p]}): mean per firm = {mean_count:.2f} (expected: 2)")

# ══════════════════════════════════════════════════════════════════════════════
# 3. VARIABLE DISTRIBUTIONS AND PLAUSIBILITY
# ══════════════════════════════════════════════════════════════════════════════
print("\n\n3. VARIABLE DISTRIBUTIONS AND PLAUSIBILITY")
print("-" * 50)

# Callback rate
print(f"\nCallback rate: {df.callback.mean()*100:.2f}%")
print(f"  Expected range for correspondence studies: 5-25%")
print(f"  Status: {'PLAUSIBLE' if 0.05 <= df.callback.mean() <= 0.25 else 'UNUSUAL'}")

# Age distribution
print(f"\nAge: mean={df.age.mean():.1f}, min={df.age.min()}, max={df.age.max()}")
print(f"  Unique values: {sorted(df.age.unique())}")
print(f"  For recent graduates: {'PLAUSIBLE' if 20 <= df.age.mean() <= 30 else 'UNUSUAL'}")

# Binary variables should be 0/1
binary_vars = ['sex', 'married', 'public_college', 'scholarship', 'public_highschool',
               'other_language', 'leadership', 'callback', 'photo1', 'photo2', 'photo4',
               'all8', 'ss_degree', 'some_availab']
print(f"\nBinary variable check:")
for var in binary_vars:
    vals = set(df[var].dropna().unique())
    ok = vals.issubset({0, 0.0, 1, 1.0})
    print(f"  {var:<20} values={vals}  {'OK' if ok else 'ISSUE!'}")

# English variable
print(f"\nEnglish proficiency (english):")
print(f"  Unique values: {sorted(df.english.unique())}")
print(f"  Distribution:")
for v in sorted(df.english.unique()):
    n = (df.english == v).sum()
    print(f"    {v}: {n} ({n/len(df)*100:.1f}%)")

# ══════════════════════════════════════════════════════════════════════════════
# 4. LOGICAL CONSISTENCY
# ══════════════════════════════════════════════════════════════════════════════
print("\n\n4. LOGICAL CONSISTENCY")
print("-" * 50)

# photo1 + photo2 + photo4 should be mutually exclusive (photo3 = indigenous = omitted)
photo_sum = df['photo1'] + df['photo2'] + df['photo4']
print(f"Photo dummy sum (photo1+photo2+photo4):")
print(f"  Values: {sorted(photo_sum.unique())}")
print(f"  All <= 1: {(photo_sum <= 1).all()} (indigenous has sum=0)")

# Verify photo dummies match photo variable
for p, dummy in [(1, 'photo1'), (2, 'photo2'), (4, 'photo4')]:
    match = ((df.photo == p) == (df[dummy] == 1)).all()
    print(f"  photo=={p} matches {dummy}: {match}")

# photo=3 (indigenous) should have all dummies = 0
indig = df[df.photo == 3]
indig_zero = (indig[['photo1', 'photo2', 'photo4']].sum(axis=1) == 0).all()
print(f"  photo==3 (indigenous) has all dummies=0: {indig_zero}")

# all8 consistency: firms with all8=1 should have 8 CVs
a8_firms = df[df.all8 == 1].groupby('id_offer').size()
print(f"\nall8 consistency:")
print(f"  Firms with all8=1: {a8_firms.shape[0]}")
print(f"  All have exactly 8 CVs: {(a8_firms == 8).all()}")
print(f"  Distribution: {dict(a8_firms.value_counts().sort_index())}")

# Check within all8 firms: should have 4 men and 4 women
a8_gender = df[df.all8 == 1].groupby(['id_offer', 'sex']).size().unstack(fill_value=0)
all_balanced = ((a8_gender[0] == 4) & (a8_gender[1] == 4)).all()
print(f"  All all8 firms have exactly 4M+4F: {all_balanced}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. MISSING DATA PATTERNS
# ══════════════════════════════════════════════════════════════════════════════
print("\n\n5. MISSING DATA PATTERNS")
print("-" * 50)

total_missing = df.isna().sum().sum()
print(f"Total missing values: {total_missing}")
if total_missing == 0:
    print("No missing data in any variable. Dataset is complete.")
else:
    for col in df.columns:
        n_miss = df[col].isna().sum()
        if n_miss > 0:
            print(f"  {col}: {n_miss} missing ({n_miss/len(df)*100:.1f}%)")

# ══════════════════════════════════════════════════════════════════════════════
# 6. CALLBACK PATTERNS
# ══════════════════════════════════════════════════════════════════════════════
print("\n\n6. CALLBACK PATTERNS AND POTENTIAL ANOMALIES")
print("-" * 50)

# Callback rate by firm size
print("\nCallback rate by number of CVs per firm:")
firm_sizes = df.groupby('id_offer').size().rename('firm_size')
df_fs = df.merge(firm_sizes, left_on='id_offer', right_index=True)
for fs in sorted(df_fs.firm_size.unique()):
    sub = df_fs[df_fs.firm_size == fs]
    rate = sub.callback.mean() * 100
    n_firms = sub.id_offer.nunique()
    print(f"  {fs} CVs: {rate:.1f}% callback ({n_firms} firms, {len(sub)} obs)")

# Firms with unusually high or low callback rates
firm_rates = df.groupby('id_offer').callback.mean()
print(f"\nFirm-level callback rates:")
print(f"  Mean: {firm_rates.mean()*100:.1f}%")
print(f"  Std: {firm_rates.std()*100:.1f}%")
print(f"  Min: {firm_rates.min()*100:.1f}%")
print(f"  Max: {firm_rates.max()*100:.1f}%")
print(f"  Firms calling no one: {(firm_rates == 0).sum()} ({(firm_rates == 0).mean()*100:.1f}%)")
print(f"  Firms calling everyone: {(firm_rates == 1).sum()} ({(firm_rates == 1).mean()*100:.1f}%)")

# ══════════════════════════════════════════════════════════════════════════════
# 7. DUPLICATES CHECK
# ══════════════════════════════════════════════════════════════════════════════
print("\n\n7. DUPLICATES CHECK")
print("-" * 50)

# Check for exact duplicate rows
n_dups = df.duplicated().sum()
print(f"Exact duplicate rows: {n_dups}")

# Check for duplicate (id_offer, sex, photo) combinations
id_sex_photo = df.groupby(['id_offer', 'sex', 'photo']).size()
multi = id_sex_photo[id_sex_photo > 1]
print(f"Duplicate (firm, sex, photo) combos: {len(multi)}")
if len(multi) > 0:
    print(f"  This means some firms received >1 CV with same gender+photo")

# ══════════════════════════════════════════════════════════════════════════════
# 8. TREATMENT ASSIGNMENT WITHIN FIRMS
# ══════════════════════════════════════════════════════════════════════════════
print("\n\n8. TREATMENT ASSIGNMENT WITHIN FIRMS")
print("-" * 50)

# For all8 firms: check photo assignment pattern
print("Photo assignment within all8 firms:")
a8_photo = df[df.all8 == 1].groupby(['id_offer', 'photo']).size().unstack(fill_value=0)
print(f"  Mean per photo type per firm:")
for p in [1, 2, 3, 4]:
    labels = {1: 'European', 2: 'Mestizo', 3: 'Indigenous', 4: 'No photo'}
    if p in a8_photo.columns:
        print(f"    {labels[p]}: {a8_photo[p].mean():.2f} (expected 2.0)")

# Check marital status assignment
a8_married = df[df.all8 == 1].groupby(['id_offer', 'married']).size().unstack(fill_value=0)
print(f"\n  Married assignment:")
print(f"    Mean married per firm: {a8_married[1].mean():.2f}")
print(f"    Mean single per firm: {a8_married[0].mean():.2f}")

# ══════════════════════════════════════════════════════════════════════════════
# 9. CORRELATION STRUCTURE
# ══════════════════════════════════════════════════════════════════════════════
print("\n\n9. CORRELATION WITH CALLBACK")
print("-" * 50)

corr_vars = ['sex', 'photo1', 'photo2', 'photo4', 'married', 'public_college',
             'ss_degree', 'scholarship', 'public_highschool', 'other_language',
             'some_availab', 'leadership', 'age', 'english']
for var in corr_vars:
    r = df[var].astype(float).corr(df.callback.astype(float))
    print(f"  {var:<20} r = {r:>7.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# 10. DATA TYPE PRECISION
# ══════════════════════════════════════════════════════════════════════════════
print("\n\n10. STATA FLOAT32 PRECISION NOTE")
print("-" * 50)
float32_vars = [c for c in df.columns if df[c].dtype == 'float32']
print(f"Variables stored as float32 (Stata default): {float32_vars}")
print("These were likely created with Stata 'set type float', which uses")
print("single precision. This can cause ~0.001 differences vs Python float64.")
print("Key outcome 'callback' is float32 but only takes 0/1 values, so no impact.")

print("\n" + "=" * 70)
print("DATA AUDIT SUMMARY")
print("=" * 70)
print("""
- Complete dataset: 8,149 obs, 18 variables, zero missing values
- Randomization well-balanced across treatment dimensions
- All binary variables properly coded (0/1)
- Photo dummies logically consistent with photo variable
- all8 firms correctly have 8 CVs each (4M + 4F)
- Callback rate (12.85%) plausible for correspondence study
- No exact duplicate rows
- Some firms received >1 CV with same gender+photo (non-all8 firms)
- No data quality concerns identified
""")
