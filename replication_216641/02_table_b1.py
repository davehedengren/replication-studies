"""Table B.I: Data Description.

Reproduces the R code in main.R: for each country take the time-series mean of
Gross Debt, Revenue, CPI, and Center Target; then report average, min, max
across countries. Rounded to one decimal to match the paper.
"""
import os
import pandas as pd
from utils import OUT

PUBLISHED = {
    "Debt/GDP":      {"Average": 45.2, "Min": 13.4, "Max": 80.8},
    "Revenue/GDP":   {"Average": 32.9, "Min": 16.4, "Max": 56.1},
    "Expected CPI":  {"Average": 3.9,  "Min": 1.5,  "Max": 15.4},
    "CPI target":    {"Average": 3.2,  "Min": 1.5,  "Max": 8.2},
}

LABELS = {
    "Gross.Debt": "Debt/GDP",
    "Revenue": "Revenue/GDP",
    "CPI": "Expected CPI",
    "Center.Target": "CPI target",
}

def main():
    df = pd.read_csv(os.path.join(OUT, "panel.csv"))
    # Paper reports 20 inflation-targeting countries; the file includes 3
    # additional countries (Armenia, Romania, Guatemala) with no Revenue
    # coverage. Restrict to the 20-country sample actually used in Tables B.II/III.
    df = df.dropna(subset=["Revenue"])
    per_country = df.groupby("Country")[list(LABELS)].mean(numeric_only=True)
    stats = per_country.agg(["mean", "min", "max"]).round(1).rename(
        index={"mean": "Average", "min": "Min", "max": "Max"}
    )
    stats = stats.rename(columns=LABELS)

    print("Table B.I: Data Description")
    print("=" * 60)
    print(stats.to_string())
    print()
    print("Published vs. computed:")
    print(f"{'Variable':15s} {'Stat':10s} {'Pub':>8s} {'Mine':>8s} {'Diff':>8s}")
    for var, pub in PUBLISHED.items():
        for stat, pv in pub.items():
            mv = stats.loc[stat, var]
            print(f"{var:15s} {stat:10s} {pv:8.1f} {mv:8.1f} {mv - pv:+8.1f}")

    stats.to_csv(os.path.join(OUT, "table_b1.csv"))


if __name__ == "__main__":
    main()
