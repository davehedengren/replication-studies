"""Data audit of the California Prop 99 panel shipped in the replication package."""
from __future__ import annotations
import numpy as np
import pandas as pd
from utils import load_california, panel_matrices, HERE

OUT = HERE / "outputs"


def main():
    df = load_california()

    lines = []
    def log(msg=""):
        print(msg); lines.append(msg)

    log("=" * 60)
    log("DATA AUDIT: California Prop 99 (data/california_prop99.csv)")
    log("=" * 60)
    log(f"Rows: {len(df)}")
    log(f"Columns: {list(df.columns)}")
    log(f"Missing cells: {int(df.isna().sum().sum())}")

    n_states = df["State"].nunique()
    n_years = df["Year"].nunique()
    log(f"\nStates: {n_states}")
    log(f"Years: {n_years}  range [{df['Year'].min()}, {df['Year'].max()}]")
    log(f"Expected balanced obs = {n_states * n_years}; actual = {len(df)}")
    assert len(df) == n_states * n_years, "Panel not balanced"

    # Treatment pattern
    trt = df[df["treated"] == 1]
    log(f"\nTreated rows: {len(trt)}")
    log(f"Treated states: {sorted(trt['State'].unique())}")
    log(f"Treated year range: [{trt['Year'].min()}, {trt['Year'].max()}]")
    log(f"Untreated California rows: "
        f"{len(df[(df['State'] == 'California') & (df['treated'] == 0)])}")

    # Outcome summary stats
    out = df["PacksPerCapita"]
    log(f"\nPacksPerCapita summary:")
    log(f"  mean {out.mean():.2f}  sd {out.std():.2f}")
    log(f"  min  {out.min():.2f}  max {out.max():.2f}")
    log(f"  quartiles: {np.percentile(out, [25, 50, 75])}")

    # Top and bottom observations
    log(f"\nHighest 5 PacksPerCapita:")
    for _, r in df.nlargest(5, "PacksPerCapita").iterrows():
        log(f"  {r['State']:<15} {int(r['Year'])}  {r['PacksPerCapita']:.2f}")
    log(f"\nLowest 5 PacksPerCapita:")
    for _, r in df.nsmallest(5, "PacksPerCapita").iterrows():
        log(f"  {r['State']:<15} {int(r['Year'])}  {r['PacksPerCapita']:.2f}")

    # Duplicates
    dup = df.duplicated(subset=["State", "Year"]).sum()
    log(f"\nDuplicate (State, Year) rows: {int(dup)}")

    # State-level trend: decline from 1970 to 2000?
    trend = df.groupby("Year")["PacksPerCapita"].mean()
    log(f"\nAvg PacksPerCapita 1970: {trend.loc[1970]:.2f}")
    log(f"Avg PacksPerCapita 1988 (pre-treatment): {trend.loc[1988]:.2f}")
    log(f"Avg PacksPerCapita 2000: {trend.loc[2000]:.2f}")

    # California vs control pre-trends
    ca_pre = df[(df["State"] == "California") & (df["Year"] < 1989)]
    ctrl_pre = df[(df["State"] != "California") & (df["Year"] < 1989)]
    log(f"\nCalifornia pre-treatment mean: {ca_pre['PacksPerCapita'].mean():.2f}")
    log(f"Controls   pre-treatment mean: "
        f"{ctrl_pre.groupby('Year')['PacksPerCapita'].mean().mean():.2f}")

    # Notable "other treated" / donor pool exclusions
    # The Abadie et al. donor pool excludes 12 states that passed large taxes.
    # The package's data has 38 controls + California = 39 states.
    log(f"\nDonor pool size: {n_states - 1} states (Abadie et al. (2010) excluded "
        f"states that also passed large cigarette taxes).")

    # Panel matrix sanity
    setup = panel_matrices(df)
    Y = setup["Y"]
    log(f"\nPanel matrix shape: {Y.shape}  (N, T) = ({Y.shape[0]}, {Y.shape[1]})")
    log(f"  N0={setup['N0']}, T0={setup['T0']}")
    log(f"  treated unit: {setup['units'][-1]}")
    log(f"  pre-period mean for treated: {Y[-1, :setup['T0']].mean():.2f}")
    log(f"  post-period mean for treated: {Y[-1, setup['T0']:].mean():.2f}")
    log(f"  pre-period mean for controls: "
        f"{Y[:setup['N0'], :setup['T0']].mean():.2f}")
    log(f"  post-period mean for controls: "
        f"{Y[:setup['N0'], setup['T0']:].mean():.2f}")

    # Naive DID check
    naive_did = ((Y[-1, setup['T0']:].mean() - Y[-1, :setup['T0']].mean())
                 - (Y[:setup['N0'], setup['T0']:].mean()
                    - Y[:setup['N0'], :setup['T0']].mean()))
    log(f"\nNaive DID (California vs pooled controls): {naive_did:+.2f}")

    with open(OUT / "audit.txt", "w") as f:
        f.write("\n".join(lines))
    log(f"\nAudit report saved to {OUT}/audit.txt")


if __name__ == "__main__":
    main()
