"""Load the experiment dataset and verify it matches the published paper's sample.

Published values to hit: N=1008 full-time employees; mean of perceived unempl.
risk next recession ~ 32.98 (sd ~ 25.94); Pr(choose recession forecast) ~ 0.25.
"""
from utils import load_experiment, PANEL_BC_OUTCOMES, OUT
import pandas as pd

df = load_experiment()
print(f"N = {len(df)}")
assert len(df) == 1008, "sample size does not match published N=1008"

# Quick summary stats versus paper-reported values
summary = {
    "unemp_nextrecession_mean": df["unemp_nextrecession"].mean(),
    "unemp_nextrecession_sd": df["unemp_nextrecession"].std(),
    "forecast_recession_mean": df["forecast_recession"].mean(),
    "forecast_govspending_mean": df["forecast_govspending"].mean(),
    "forecast_iratebonds_mean": df["forecast_iratebonds"].mean(),
    "forecast_inflation_mean": df["forecast_inflation"].mean(),
    "noforecast_mean": df["noforecast"].mean(),
    "othermacro_mean": df["othermacro"].mean(),
    "high_conf_share": df["high_conf"].mean(),
    "tr_acs_share": df["tr_acs"].mean(),
    "female_share": df["female"].mean(),
    "age_mean": df["age"].mean(),
    "college_share": df["college"].mean(),
}
for k, v in summary.items():
    print(f"{k:32s} {v:8.4f}")

pd.Series(summary).to_csv(OUT / "01_summary.csv")
print("\nSaved:", OUT / "01_summary.csv")
