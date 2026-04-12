"""Data audit for the merged state-year panel.

Checks: coverage (states x years), variable distributions, plausibility bounds,
missing patterns, and basic sanity of enrollment counts vs. CPS denominators.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from utils import DATA

panel = pd.read_parquet(DATA / "full_panel.parquet")
es = pd.read_parquet(DATA / "eventstudy_sample.parquet")

print("=" * 72)
print("DATA AUDIT — full panel")
print("=" * 72)
print(f"rows: {len(panel):,}")
print(f"states: {panel['statefip'].nunique()}  years: {panel['year'].min()}-{panel['year'].max()}")
missing_cells = 51 * 21 - len(panel)
print(f"state-year cells missing: {missing_cells}")

# Balance: every state should have 21 years
years_per_state = panel.groupby("statefip").size()
print(f"years per state: min={years_per_state.min()} max={years_per_state.max()}")

# Summary stats
key = ["fywgt_0to130fpl", "fywgt_50to115fpl", "wtsupp_0to130fpl",
       "wtsupp_50to115fpl", "bbce_inclmt", "unemployment"]
print("\nSummary stats:")
print(panel[key].describe().round(2))

# Take-up rate implied: fywgt/wtsupp should be in [0,1] roughly
panel["takeup_0to130"] = panel["fywgt_0to130fpl"] / panel["wtsupp_0to130fpl"]
panel["takeup_50to115"] = panel["fywgt_50to115fpl"] / panel["wtsupp_50to115fpl"]
print("\nTake-up rates (implied):")
print(panel[["takeup_0to130", "takeup_50to115"]].describe().round(3))
too_high = (panel["takeup_0to130"] > 1.2).sum()
print(f"take-up > 1.2 (flag implausible): {too_high}")

# Policy distribution
print("\nbbce_inclmt distribution:")
print(panel["bbce_inclmt"].value_counts().sort_index())

print("\nhas_bbce counts by year:")
print(panel.groupby("year")["has_bbce"].sum())

# Events
panel_sorted = panel.sort_values(["statefip", "year"])
panel_sorted["prev"] = panel_sorted.groupby("statefip")["bbce_inclmt"].shift(1)
panel_sorted["is_change"] = ((panel_sorted["bbce_inclmt"] != panel_sorted["prev"])
                              & panel_sorted["prev"].notna()).astype(int)
event_counts = panel_sorted.groupby("statefip")["is_change"].sum()
print("\n# BBCE eligibility-limit changes per state:")
print(event_counts.value_counts().sort_index())

# Duplicates
dups = panel.duplicated(["statefip", "year"]).sum()
print(f"\nduplicate state-year rows: {dups}")

# Missing
print("\nMissing values per variable:")
print(panel.isna().sum().sort_values(ascending=False).head(15))

print("\n" + "=" * 72)
print("DATA AUDIT — event-study sample")
print("=" * 72)
print(f"rows: {len(es):,}  states: {es['statefip'].nunique()}")
# expected: 705, 45
print(f"years per state in ES:")
print(es.groupby("statefip").size().describe())

# Event-study states vs others
es_states = set(es["statefip"])
all_states = set(panel["statefip"])
print(f"\nstates NOT in ES: {sorted(all_states - es_states)}")

# Pre/post mean of inclmt among treated (ES states with event)
events = (panel_sorted[panel_sorted["is_change"] == 1]
          .groupby("statefip")[["year", "bbce_inclmt"]].first())
treated_in_es = events[events.index.isin(es_states)]
print(f"\ntreated states in ES: {len(treated_in_es)}")
print("post-event inclmt distribution among ES-treated:")
print(treated_in_es["bbce_inclmt"].value_counts().sort_index())
