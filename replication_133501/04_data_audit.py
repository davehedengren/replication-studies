"""Data audit for the Huh & Reif replication package."""
import numpy as np
import pandas as pd

from utils import MORTALITY_OUTCOMES, load_addhealth, load_mortality

print("=" * 78)
print("DATA AUDIT — Huh & Reif (2021) replication package")
print("=" * 78)

# --- Mortality panel ------------------------------------------------------
for scen in ["none", "male", "female"]:
    df = load_mortality(scen)
    agemin, agemax = df["agemo_mda"].min(), df["agemo_mda"].max()
    n = len(df)
    dup = df["agemo_mda"].duplicated().sum()
    pre = df[(df.agemo_mda >= -12) & (df.agemo_mda <= -1)]
    post = df[(df.agemo_mda >= 1) & (df.agemo_mda <= 12)]
    print(f"\nmortality_{scen}.dta")
    print(f"  rows={n}, unique agemo={df.agemo_mda.nunique()}, dup_keys={dup}, "
          f"range=[{agemin:.0f},{agemax:.0f}] (expected [-48,47])")
    pop_pct = (df["pop"].std() / df["pop"].mean()) * 100
    print(f"  population mean={df['pop'].mean():.2e}  CV={pop_pct:.1f}%  "
          f"(flat pop ⇒ rates ∝ counts)")
    print(f"  cod_any rate pre-MDA  mean={pre['cod_any'].mean():.2f}  "
          f"post-MDA mean={post['cod_any'].mean():.2f}")
    print(f"  cod_MVA rate pre-MDA  mean={pre['cod_MVA'].mean():.2f}  "
          f"post-MDA mean={post['cod_MVA'].mean():.2f}")
    # Decomposition sanity check: internal + external ~ any
    discrep = (df["cod_internal"] + df["cod_external"] - df["cod_any"]).abs().max()
    print(f"  |internal+external - any|.max()={discrep:.3f}")
    # cod_sa subcategories sum to cod_sa
    sub = (df["cod_sa_firearms"] + df["cod_sa_poisoning"] + df["cod_sa_drowning"] + df["cod_sa_other"])
    discrep_sa = (sub - df["cod_sa"]).abs().max()
    print(f"  |firearms+poisoning+drowning+other - cod_sa|.max()={discrep_sa:.3f}")
    # At agemo_mda==0, the cod_MVA rate should be anomalously low (measurement error)
    row0 = df[df.agemo_mda == 0].iloc[0]
    row_m1 = df[df.agemo_mda == -1].iloc[0]
    row_p1 = df[df.agemo_mda == 1].iloc[0]
    print(f"  cod_MVA at agemo -1 / 0 / 1: {row_m1['cod_MVA']:.2f} / "
          f"{row0['cod_MVA']:.2f} / {row_p1['cod_MVA']:.2f}  "
          f"(expect dip at 0 — Dong 2015 measurement error)")

# --- Add Health -----------------------------------------------------------
for scen in ["none", "male", "female"]:
    df = load_addhealth(scen)
    print(f"\naddhealth_{scen}.dta")
    print(f"  rows={len(df)}  agemo range=[{df.agemo_mda.min():.0f},"
          f"{df.agemo_mda.max():.0f}]")
    for col in ["DriverLicense", "VehicleMiles_150", "VehicleMiles_265"]:
        print(f"  {col}: min={df[col].min():.2f} max={df[col].max():.2f} "
              f"mean={df[col].mean():.2f}  "
              f"pre-MDA mean={df[df.agemo_mda.between(-12,-1)][col].mean():.2f}")

# --- Male + Female ~ Full sample check -----------------------------------
print("\nConsistency: female rate + male rate ~= full rate × 2?")
m_none = load_mortality("none")
m_m = load_mortality("male")
m_f = load_mortality("female")
# The male+female weighted average should equal the "none" series.
# Population ratios: weights not equal, but avg of cod_any across sexes
# should be close to none value.
for col in ["cod_any", "cod_MVA", "cod_sa_poisoning"]:
    diff = ((m_m[col] + m_f[col]) / 2 - m_none[col]).abs().mean()
    print(f"  mean |avg(male,female) - none| for {col}: {diff:.4f}")

print("\nDONE.")
