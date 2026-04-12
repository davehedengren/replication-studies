"""Load and clean Study 1 data. Mirrors clean_study1.do.

Saves a parquet file of the cleaned analysis sample and prints basic counts
to verify N matches the paper's 1,952 respondents."""

from utils import load_clean, OUTDIR

df = load_clean()

print(f"Cleaned sample size: {len(df)}")
print(f"Treatment breakdown: Before={ (df['Tr']==0).sum() }, After={ (df['Tr']==1).sum() }")
print(f"Biden voters: {(df['vote_trump']==0).sum()}, Trump voters: {(df['vote_trump']==1).sum()}")
print()
print("Demographics (means):")
print(df[["male", "age", "white", "college", "ln_income", "northeast", "midwest", "south", "west"]].mean().round(3))
print()
print("Paper-relevant outcomes:")
print(f"  Contain racism terms: {df['contain_racism_terms'].mean():.4f}")
print(f"  Contain labor terms:  {df['contain_labor_terms'].mean():.4f}")

df.to_parquet(OUTDIR / "study1_cleaned.parquet", index=False)
print(f"\nSaved to {OUTDIR/'study1_cleaned.parquet'}")
