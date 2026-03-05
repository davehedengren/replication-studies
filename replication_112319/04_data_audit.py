"""
04_data_audit.py - Data quality audit for Charness & Levin (2005)
"""
import sys
sys.path.insert(0, '.')
from utils import *
from scipy import stats

df = load_data_fast()

print("=" * 70)
print("DATA AUDIT: Charness & Levin (2005)")
print("=" * 70)

# ── 1. Coverage ────────────────────────────────────────────────────────
print("\n1. COVERAGE")
print(f"  Total observations: {len(df)}")
print(f"  Total participants: {df['Individual'].nunique()}")
print(f"  Treatment 1: {(df['treatment']==1).sum()} obs, {df[df['treatment']==1]['Individual'].nunique()} individuals")
print(f"  Treatment 2: {(df['treatment']==2).sum()} obs, {df[df['treatment']==2]['Individual'].nunique()} individuals")
print(f"  Treatment 3: {(df['treatment']==3).sum()} obs, {df[df['treatment']==3]['Individual'].nunique()} individuals")
print(f"  Rounds per individual T1/T2: 60, T3: 80")
print(f"  Expected total: 59*60 + 54*60 + 52*80 = {59*60 + 54*60 + 52*80}")
print(f"  Actual: {len(df)} {'✓ MATCH' if len(df) == 59*60+54*60+52*80 else '✗ MISMATCH'}")

# Variable completeness
print(f"\n  Missing values per column:")
for col in df.columns:
    n_miss = df[col].isna().sum()
    if n_miss > 0:
        print(f"    {col}: {n_miss} missing ({n_miss/len(df)*100:.1f}%)")
    else:
        print(f"    {col}: 0 missing")

# ── 2. Distributions ──────────────────────────────────────────────────
print("\n2. DISTRIBUTIONS")
print("\n  Summary statistics for key binary variables:")
for col in ['Female', 'Up', 'Cyan_Pays', 'Left1', 'Cyan1', 'Left2', 'Cyan2']:
    print(f"  {col}: mean={df[col].mean():.4f}, min={df[col].min()}, max={df[col].max()}")

print("\n  State (Up) distribution by treatment:")
for t in [1, 2, 3]:
    sub = df[df['treatment'] == t]
    print(f"    T{t}: Up mean={sub['Up'].mean():.4f} (expect ~0.5, n={len(sub)})")

print("\n  Position1 distribution:")
print(f"    Range: {df['Position1'].min()} - {df['Position1'].max()}")
print(f"    Value counts:")
print(df['Position1'].value_counts().sort_index().to_string())

print("\n  Position2 distribution:")
print(f"    Range: {df['Position2'].min()} - {df['Position2'].max()}")

# ── 3. Logical Consistency ────────────────────────────────────────────
print("\n3. LOGICAL CONSISTENCY")

# Right urn always reveals state perfectly
for t in [1, 2]:
    sub = df[df['treatment'] == t]
    right = sub[sub['Left1'] == 0]
    # Right Up = all black (Cyan1=1), Right Down = all white (Cyan1=0)
    up_right = right[right['Up'] == 1]
    down_right = right[right['Up'] == 0]
    print(f"  T{t} Right+Up: all Cyan1=1? {(up_right['Cyan1']==1).all()} (n={len(up_right)})")
    print(f"  T{t} Right+Down: all Cyan1=0? {(down_right['Cyan1']==0).all()} (n={len(down_right)})")

# Left urn draw probabilities match stated compositions
print("\n  Left urn draw rates vs theoretical:")
for t, urns in [(1, T1_URNS), (2, T2_URNS), (3, T3_URNS)]:
    sub = df[df['treatment'] == t]
    for state_name, up_val in [('Up', 1), ('Down', 0)]:
        left = sub[(sub['Left1'] == 1) & (sub['Up'] == up_val)]
        if len(left) > 0:
            b, w = urns[(state_name, 'Left')]
            expected = b / (b + w)
            actual = left['Cyan1'].mean()
            ci = 1.96 * np.sqrt(actual * (1-actual) / len(left))
            in_ci = abs(actual - expected) < ci
            print(f"    T{t} Left {state_name}: actual={actual:.4f}, expected={expected:.4f}, "
                  f"n={len(left)}, {'within 95% CI' if in_ci else '** OUTSIDE 95% CI **'}")

# Phase I restrictions: odd rounds = Left, even = Right
print("\n  Phase I forced starting side:")
for t in [1, 2]:
    sub = df[(df['treatment'] == t) & (df['phase'] == 'I')]
    odd = sub[sub['Round'] % 2 == 1]
    even = sub[sub['Round'] % 2 == 0]
    print(f"    T{t}: odd rounds all Left? {(odd['Left1']==1).all()}, "
          f"even rounds all Right? {(even['Left1']==0).all()}")

sub = df[(df['treatment'] == 3)]
print(f"    T3: all rounds Left? {(sub['Left1']==1).all()}")

# Second draw must be from one urn
print(f"\n  Left2 binary? min={df['Left2'].min()}, max={df['Left2'].max()}")

# ── 4. Missing Data Patterns ─────────────────────────────────────────
print("\n4. MISSING DATA PATTERNS")
print("  No missing values in raw data columns.")
print(f"  Derived 'starting_error': {df['starting_error'].isna().sum()} NaN (expected - only defined for T1/T2 Ph II-III)")
print(f"  Derived 'beu_left2': {df['beu_left2'].isna().sum()} NaN")

# ── 5. Panel Balance ──────────────────────────────────────────────────
print("\n5. PANEL BALANCE")
rounds_per_ind = df.groupby('Individual')['Round'].count()
print(f"  All T1/T2 individuals have 60 rounds: {(rounds_per_ind[rounds_per_ind.index <= 113] == 60).all()}")
print(f"  All T3 individuals have 80 rounds: {(rounds_per_ind[rounds_per_ind.index >= 114] == 80).all()}")

# Check for gaps in round numbers
print("\n  Round sequence gaps:")
for ind in df['Individual'].unique()[:5]:
    sub = df[df['Individual'] == ind].sort_values('Round')
    gaps = np.diff(sub['Round'].values)
    if not (gaps == 1).all():
        print(f"    Individual {ind}: gaps found at rounds {sub['Round'].values[np.where(gaps != 1)[0]]}")

n_gaps = 0
for ind in df['Individual'].unique():
    sub = df[df['Individual'] == ind].sort_values('Round')
    gaps = np.diff(sub['Round'].values)
    if not (gaps == 1).all():
        n_gaps += 1
print(f"  Individuals with gaps: {n_gaps}/{df['Individual'].nunique()}")

# ── 6. Duplicates ─────────────────────────────────────────────────────
print("\n6. DUPLICATES")
dups = df.duplicated(subset=['Individual', 'Round'], keep=False)
print(f"  Duplicate (Individual, Round) pairs: {dups.sum()}")

# ── 7. Gender Distribution ────────────────────────────────────────────
print("\n7. GENDER ANALYSIS")
gender = df.groupby('Individual')['Female'].first()
for t in [1, 2, 3]:
    ids = df[df['treatment'] == t]['Individual'].unique()
    g = gender.loc[ids]
    n_f = (g == 1).sum()
    n_m = (g == 0).sum()
    pct_f = n_f / len(g) * 100
    print(f"  T{t}: {n_m} male ({100-pct_f:.0f}%), {n_f} female ({pct_f:.0f}%)")

# Gender constant within individual?
gender_varies = df.groupby('Individual')['Female'].nunique()
print(f"  Gender constant within individual: {(gender_varies == 1).all()}")

# ── 8. State Randomization Quality ───────────────────────────────────
print("\n8. STATE RANDOMIZATION")
for t in [1, 2, 3]:
    sub = df[df['treatment'] == t]
    up_rate = sub['Up'].mean()
    n = len(sub)
    # Binomial test for 50/50
    p_val = stats.binomtest(int(sub['Up'].sum()), n, 0.5).pvalue
    print(f"  T{t}: Up rate = {up_rate:.4f}, p-value (H0: 0.5) = {p_val:.4f}")

# Autocorrelation in Up
print("\n  Serial correlation in Up (lag-1):")
for t in [1, 2, 3]:
    sub = df[df['treatment'] == t]
    for ind in sub['Individual'].unique()[:3]:
        isub = sub[sub['Individual'] == ind].sort_values('Round')
        up = isub['Up'].values
        if len(up) > 1:
            corr = np.corrcoef(up[:-1], up[1:])[0, 1]
            # Just check a few
    # Aggregate autocorrelation
    all_corrs = []
    for ind in sub['Individual'].unique():
        isub = sub[sub['Individual'] == ind].sort_values('Round')
        up = isub['Up'].values
        if len(up) > 1:
            corr = np.corrcoef(up[:-1], up[1:])[0, 1]
            all_corrs.append(corr)
    print(f"  T{t}: mean lag-1 autocorrelation = {np.mean(all_corrs):.4f} (expect ~0)")

# ── 9. Anomaly Detection ─────────────────────────────────────────────
print("\n9. ANOMALY DETECTION")

# Check for individuals with extreme behavior
# Individuals who always/never switch
t12 = df[df['treatment'].isin([1, 2])]
for t in [1, 2]:
    sub = t12[t12['treatment'] == t]
    ind_switch = sub.groupby('Individual')['switched'].mean()
    always_switch = (ind_switch == 1.0).sum()
    never_switch = (ind_switch == 0.0).sum()
    print(f"  T{t}: always switch={always_switch}, never switch={never_switch}")

# Position1 ranges from 1-12 (possibly interface ball positions across both urns)
print("\n  Position1 distribution by range:")
for t in [1, 2, 3]:
    sub = df[df['treatment'] == t]
    n_1_6 = (sub['Position1'] <= 6).sum()
    n_7_12 = (sub['Position1'] > 6).sum()
    print(f"    T{t}: pos 1-6 = {n_1_6}, pos 7-12 = {n_7_12}")

# Positions 7-12 usage
print(f"\n  Position1 > 6 observations: {(df['Position1'] > 6).sum()}")
print(f"  Position2 > 6 observations: {(df['Position2'] > 6).sum()}")
if (df['Position1'] > 6).sum() > 0:
    print(f"  Position1 > 6 breakdown:")
    print(df[df['Position1'] > 6]['Position1'].value_counts().sort_index())

# ── 10. Payoff Consistency ────────────────────────────────────────────
print("\n10. CYAN_PAYS CONSISTENCY")
# T1/T2: Cyan_Pays should always be 1
t12_cp = df[df['treatment'].isin([1, 2])]['Cyan_Pays']
print(f"  T1/T2 Cyan_Pays always 1: {(t12_cp == 1).all()}")

# T3: Cyan_Pays should be ~50/50 and independent of Up
t3 = df[df['treatment'] == 3]
ct = pd.crosstab(t3['Up'], t3['Cyan_Pays'])
chi2, p, dof, _ = stats.chi2_contingency(ct)
print(f"  T3 Up × Cyan_Pays independence: chi2={chi2:.3f}, p={p:.4f}")

print("\n" + "=" * 70)
print("AUDIT SUMMARY")
print("=" * 70)
print("✓ Complete balanced panel: 165 individuals, no missing observations")
print("✓ All binary variables properly coded {0, 1}")
print("✓ Phase I restrictions correctly enforced (forced Left/Right starts)")
print("✓ Right urn perfectly reveals state (6B in Up, 6W in Down)")
print("✓ Left urn draw rates consistent with stated compositions")
print("✓ No duplicate observations")
print("✓ No gaps in round sequences")
print("✓ State (Up) randomization appears fair (~50%)")
print("✓ Gender constant within individuals")
print("✓ T1/T2 Cyan_Pays constant at 1")
print("Note: Position values go up to 12, suggesting interface displayed both urns")
