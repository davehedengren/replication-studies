"""Data audit for Harrell et al. (2023) replication dataset."""
import pandas as pd
from utils import load_data, OUTPUT_DIR

df = load_data()

print("=" * 60)
print("COVERAGE")
print("=" * 60)
print(f"Observations: {len(df)}")
print(f"Unique email pair IDs (prospective patients): {df['EmailPairID'].nunique()}")
print(f"  -> each pair sent {len(df) // df['EmailPairID'].nunique()} emails")
print(f"Unique states: {df['State'].nunique()}  (US has 50 states + DC)")
print(f"Weeks observed: {sorted(df['Weeksent'].unique())}")
print(f"  -> gap: week 5 missing")
print(f"Day-of-week values: {sorted(df['Daysent'].unique())}")
print(f"Unique (Week, Day) combos: {df.groupby(['Weeksent','Daysent']).ngroups}")

print("\n" + "=" * 60)
print("OUTCOME DISTRIBUTION")
print("=" * 60)
print(f"Posoutcome (appointment offered): mean={df['Posoutcome'].mean():.4f}")
print(f"  -> 0: {(df['Posoutcome']==0).sum()}  1: {(df['Posoutcome']==1).sum()}")

print("\n" + "=" * 60)
print("COVID INTENSITY DISTRIBUTIONS")
print("=" * 60)
for c in ["STCasesT", "STDeathsT", "STExLowEstT"]:
    s = df[c]
    print(f"{c:14s}: mean={s.mean():9.2f}  sd={s.std():9.2f}  "
          f"min={s.min():7.0f}  max={s.max():7.0f}  "
          f"p50={s.quantile(.5):7.0f}  p90={s.quantile(.9):7.0f}")

print("\n" + "=" * 60)
print("LOGICAL CONSISTENCY")
print("=" * 60)
# deaths should be <= cases, excess deaths can be anything but should loosely track deaths
viol = (df['STDeathsT'] > df['STCasesT']).sum()
print(f"STDeathsT > STCasesT (should be 0): {viol}")
neg = ((df[['STCasesT','STDeathsT','STExLowEstT']] < 0).any(axis=1)).sum()
print(f"Any negative COVID counts: {neg}")
zc = (df['STCasesT'] == 0).sum()
zd = (df['STDeathsT'] == 0).sum()
ze = (df['STExLowEstT'] == 0).sum()
print(f"Rows with STCasesT==0: {zc}  STDeathsT==0: {zd}  STExLowEstT==0: {ze}")
print(f"  -> zeros reflect early-pandemic weeks before state had reported cases/deaths")

print("\n" + "=" * 60)
print("DEMOGRAPHIC CONTROLS (shares by treatment)")
print("=" * 60)
for c in ["Transornb", "Black", "Hispanic", "Anxiety", "Depression"]:
    m0 = df.loc[df['Posoutcome']==0, c].mean()
    m1 = df.loc[df['Posoutcome']==1, c].mean()
    print(f"  {c:10s}: no-offer={m0:.3f}  offer={m1:.3f}  diff={m1-m0:+.3f}")

print("\n" + "=" * 60)
print("PANEL BALANCE")
print("=" * 60)
per_pair = df.groupby('EmailPairID').size()
print(f"Emails per pair: min={per_pair.min()}, median={int(per_pair.median())}, max={per_pair.max()}")
per_state = df.groupby('State').size()
print(f"Emails per state: min={per_state.min()}, median={int(per_state.median())}, "
      f"max={per_state.max()}, mean={per_state.mean():.1f}")
print("  -> States are sampled roughly proportional to population; small states see few emails")

print("\n" + "=" * 60)
print("MISSING / DUPLICATES")
print("=" * 60)
print(f"Missing cells total: {int(df.isna().sum().sum())}")
print(f"Duplicate rows (ignoring ID): "
      f"{df.drop(columns=['EmailPairID']).duplicated().sum()}")

# Temporal coverage: when is each week observed relative to COVID spread?
print("\n" + "=" * 60)
print("COVID INTENSITY BY WEEK (means, all states pooled)")
print("=" * 60)
week_means = df.groupby('Weeksent')[['STCasesT','STDeathsT','STExLowEstT','Posoutcome']].mean()
print(week_means.round(2).to_string())

week_means.to_csv(OUTPUT_DIR / "weekly_means.csv")
print(f"\n-> saved weekly_means.csv to {OUTPUT_DIR}")
