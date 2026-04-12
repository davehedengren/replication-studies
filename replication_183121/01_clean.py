"""Load and validate dataset."""
from utils import load_data, CONTROLS, OUTCOME

df = load_data()
print(f"Observations: {len(df)}")
print(f"Columns: {len(df.columns)}")
print(f"\nOutcome mean (Posoutcome): {df[OUTCOME].mean():.4f}  [paper: 0.566]")
print(f"\nRaw COVID variables - summary:")
for c in ["Posoutcome", "STCasesT", "STDeathsT", "STExLowEstT"]:
    s = df[c]
    print(f"  {c}: mean={s.mean():.2f} sd={s.std():.2f} min={s.min()} max={s.max()}")

print("\nStd devs of raw COVID variables (paper Table 1 footer):")
print(f"  STCasesT    sd={df['STCasesT'].std():.1f}  [paper 11,121.9]")
print(f"  STDeathsT   sd={df['STDeathsT'].std():.1f}  [paper 489.4]")
print(f"  STExLowEstT sd={df['STExLowEstT'].std():.1f}  [paper 355.2]")

print("\nDemographic controls (shares):")
for c in CONTROLS:
    print(f"  {c}: {df[c].mean():.3f}")

print(f"\nUnique states: {df['State'].nunique()}")
print(f"Unique email pairs: {df['EmailPairID'].nunique()}")
print(f"Weeksent range: {df['Weeksent'].min()}..{df['Weeksent'].max()}")
print(f"Daysent range: {df['Daysent'].min()}..{df['Daysent'].max()}")
print("\nMissing values:", int(df.isna().sum().sum()))
