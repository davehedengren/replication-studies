"""Data audit of base_sample_aej.dta and the regional mortality panel."""
import numpy as np
import pandas as pd
from utils import load_base_sample, table2_sample, DATA, OUT


def audit_base():
    df = load_base_sample()
    print("=== Base sample (post cleaning) ===")
    print(f"Rows: {len(df):,}")
    print(f"Unique individuals (identificator): {df['identificator'].nunique():,}")
    print(f"Years: {int(df['year'].min())}-{int(df['year'].max())}")
    print(f"Rounds: {sorted(df['round'].unique().tolist())}")
    print()

    print("Rows per round:")
    print(df["round"].value_counts().sort_index().to_string())
    print()

    # Panel balance
    obs_per_person = df.groupby("identificator").size()
    print(f"Obs per individual: mean={obs_per_person.mean():.2f}, "
          f"median={obs_per_person.median():.0f}, max={obs_per_person.max()}")
    print(f"% of individuals with ≥5 obs: {(obs_per_person >= 5).mean():.1%}")
    print()

    # Plausibility of shares (should be in [0,100])
    for col in ["share_vodka", "share_beer", "share_samogon", "share_dwine",
                "share_fwine", "share_other"]:
        s = df[col]
        print(f"{col}: min={s.min():.1f} max={s.max():.1f} "
              f"mean={s.mean():.2f} n_missing={s.isna().sum()}")
    print()

    # Shares should sum to ~100 (beer + vodka + samogon + dwine + fwine + other)
    share_sum = (df["share_beer"] + df["share_vodka"] + df["share_samogon"]
                 + df["share_dwine"] + df["share_fwine"] + df["share_other"])
    print(f"Sum of 6 shares: mean={share_sum.mean():.3f} "
          f"p1={share_sum.quantile(.01):.2f} p99={share_sum.quantile(.99):.2f}")
    print(f"% of rows with sum outside [99.5, 100.5]: "
          f"{((share_sum < 99.5) | (share_sum > 100.5)).mean():.2%}")
    print()

    # Alcohol intake outliers
    a = df["alcohol_intake"].dropna()
    print(f"alcohol_intake (grams ethanol): mean={a.mean():.1f}, "
          f"median={a.median():.1f}, p95={a.quantile(.95):.1f}, "
          f"p99={a.quantile(.99):.1f}, max={a.max():.1f}")
    print()

    # Treatment group counts in Table 2 sample
    t2 = table2_sample(df)
    print("=== Table 2 sample: cells by DiD group ===")
    print(pd.crosstab(t2["rural"], t2["gorbachev"], margins=True).to_string())
    print()
    # Post-treat cohort restriction: how many rural x gorbachev cells?
    ids_treated = t2.loc[(t2["rural"] == 1) & (t2["gorbachev"] == 1), "identificator"].nunique()
    ids_control_rural = t2.loc[(t2["rural"] == 1) & (t2["gorbachev"] == 0), "identificator"].nunique()
    ids_treat_urban = t2.loc[(t2["rural"] == 0) & (t2["gorbachev"] == 1), "identificator"].nunique()
    ids_control_urban = t2.loc[(t2["rural"] == 0) & (t2["gorbachev"] == 0), "identificator"].nunique()
    print("Unique individuals in each 2x2 cell (Table 2 sample):")
    print(f"  rural × during campaign   (treated):  {ids_treated:,}")
    print(f"  rural × other cohorts     (control):  {ids_control_rural:,}")
    print(f"  urban × during campaign:              {ids_treat_urban:,}")
    print(f"  urban × other cohorts:                {ids_control_urban:,}")
    print()

    # Missingness patterns on key controls
    print("=== Missingness on Table 2 controls (year>=2001, 18<=age<=65) ===")
    for c in ["alcohol_intake", "price_beer_to_vodka", "logincome",
              "univ_educ", "health_evaluation", "married", "wtself"]:
        pct = t2[c].isna().mean()
        print(f"  {c}: {pct:.2%} missing")


def audit_mortality():
    print("\n=== Regional mortality (5_year_89_12.dta) ===")
    m = pd.read_stata(DATA / "regional_mortality/5y_89_12.dta", convert_categoricals=False)
    print(f"Rows: {len(m):,}")
    print(f"Regions: {m['id'].nunique()}")
    print(f"Years: {int(m['year'].min())}-{int(m['year'].max())}")
    print(f"Age bins: {sorted(m['age'].unique().tolist())}")
    print(f"Groups (R=rural,U=urban): {sorted(m['group'].unique().tolist())}")
    print()

    zero = (m["alco_poisoning"] == 0).mean()
    print(f"Zero alcohol-poisoning cells: {zero:.2%}")
    print(f"Zero-total cells: {(m['total'] == 0).mean():.2%}")
    # These matter because log(0) becomes missing and drops from regressions
    print()

    # Panel balance
    n_years = m.groupby(["id", "age", "group"]).size()
    print(f"Panel cell balance: mean years/cell={n_years.mean():.1f}, "
          f"min={n_years.min()}, max={n_years.max()}")


def main():
    audit_base()
    audit_mortality()


if __name__ == "__main__":
    main()
