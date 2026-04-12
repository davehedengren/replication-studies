"""Data audit for the only data in the paper: UN World Population Prospects 2019 TFR.

Also sanity-checks the model's parameter restrictions from SetParameters.m.
"""
import numpy as np
import pandas as pd
from utils import FERTILITY_DATA, FERTILITY_COLS, FERTILITY_YEARS, set_parameters, set_parameters_nber


def audit_fertility():
    print("=" * 72)
    print("FERTILITY DATA AUDIT")
    print("=" * 72)

    df = pd.DataFrame(FERTILITY_DATA, columns=FERTILITY_COLS, index=FERTILITY_YEARS)
    print(f"\nShape: {df.shape}")
    print(f"Year range: {df.index.min()}-{df.index.max()+4} (5-yr periods)")
    print(f"Countries/regions: {len(FERTILITY_COLS)}")

    print("\nSummary statistics:")
    print(df.describe().round(3))

    print("\nMissing values:")
    print(df.isna().sum().to_dict())

    print("\nPlausibility bounds (TFR should be in (0, 10)):")
    print(f"  min = {df.values.min():.3f}, max = {df.values.max():.3f}")
    assert df.values.min() > 0 and df.values.max() < 10, "Out-of-range TFR"

    print("\nMonotonicity check (1950→2015 change by country):")
    start = df.iloc[0]
    end = df.iloc[-1]
    change = (end - start).round(3)
    for name in FERTILITY_COLS:
        d = "↓" if change[name] < 0 else "↑"
        print(f"  {name:10s} {start[name]:5.2f} → {end[name]:5.2f}   {d} {change[name]:+.2f}")

    all_declining = (change < 0).all()
    print(f"\nAll regions show net decline 1950-2015: {all_declining}")

    # Spot-check values in paper text
    print("\n--- Text verification ---")
    # p.1: "1.3 for Italy and Spain"
    last = df.iloc[-1]
    print(f"  Italy 2015-20  = {last['Italy']:.3f}  (paper says 1.3)")
    print(f"  Spain 2015-20  = {last['Spain']:.3f}  (paper says 1.3)")
    print(f"  Germany 2015-20= {last['Germany']:.3f}  (paper says 1.6)")
    print(f"  Japan 2015-20  = {last['Japan']:.3f}  (paper says 1.4)")

    # High-income below 2 since when?
    hic_below_2 = df[df["HighInc"] < 2.0]
    if len(hic_below_2) > 0:
        first = hic_below_2.index[0]
        print(f"\n  High-income countries fell below 2.0 starting ~{first}-{first+4}")

    return df


def audit_parameters():
    print("\n\n" + "=" * 72)
    print("MODEL PARAMETER RESTRICTIONS (from SetParameters.m comments)")
    print("=" * 72)

    for label, p in [
        ("Current code baseline", set_parameters()),
        ("NBER Table 2 baseline", set_parameters_nber()),
    ]:
        print(f"\n--- {label} ---")
        gamma, lam, bbar, nu, delta, epsilon, rho = (
            p["gamma"], p["lam"], p["bbar"], p["nu"], p["delta"], p["epsilon"], p["rho"]
        )
        # Concavity of Hamiltonian: 1 + sqrt(gamma/lambda) - bbar*nu > 0
        concavity = 1 + np.sqrt(gamma / lam) - bbar * nu
        print(f"  Concavity   (>0): 1 + sqrt(γ/λ) - b̄ν = {concavity:+.4f}")

        # Maximum fertility at x=0: bbar - delta - 1/nu > 0
        maxfert = bbar - delta - 1 / nu
        print(f"  Max fert    (>0): b̄ - δ - 1/ν         = {maxfert:+.4f}")

        # Positive middle SS: rho/epsilon - (bbar-delta) > 0
        # (equivalent to nEqm<0 which is needed)
        pos_mid = rho / epsilon - (bbar - delta)
        print(f"  Pos mid SS  (>0): ρ/ε - (b̄-δ)          = {pos_mid:+.4f}")

        # Two real positive roots: nu*(bbar-delta) - (1+eps/lam) > 0
        two_real = nu * (bbar - delta) - (1 + epsilon / lam)
        print(f"  Two roots   (>0): ν(b̄-δ) - (1+ε/λ)     = {two_real:+.4f}")


def main():
    df = audit_fertility()
    audit_parameters()


if __name__ == "__main__":
    main()
