"""Merge QC + CPS + policy + unemployment into the analysis panel, build
event-study sample with a 5-year balanced window around each state's BBCE
eligibility expansion, and save both the full-panel and event-study datasets.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from utils import DATA

qc = pd.read_parquet(DATA / "qc_stateyear.parquet")
cps = pd.read_parquet(DATA / "cps_stateyear.parquet")
pol = pd.read_parquet(DATA / "policy_stateyear.parquet")
unem = pd.read_parquet(DATA / "unemployment_stateyear.parquet")

# Restrict CPS to 50 states+DC (statefip <= 56, no territories)
cps = cps[cps["statefip"] <= 56].copy()
qc = qc[qc["statefip"] <= 56].copy()

panel = (cps.merge(qc, on=["statefip", "year"], how="inner")
              .merge(pol, on=["statefip", "year"], how="inner")
              .merge(unem, on=["statefip", "year"], how="left"))
panel = panel.sort_values(["statefip", "year"]).reset_index(drop=True)
print("panel shape:", panel.shape)
print("years:", panel["year"].min(), "-", panel["year"].max())
print("states:", panel["statefip"].nunique())

# Outreach controls:
#   outreachper0to130 = outreach_orig * 1000 / wtsupp_0to130fpl
#   outreachnon0     = 1 if >0 else 0
#   lnoutreach       = ln(outreachper0to130), 0 when outreach=0
panel["outreachper0to130"] = panel["outreach_orig"] * 1000.0 / panel["wtsupp_0to130fpl"]
panel["outreachnon0"] = (panel["outreachper0to130"] > 0).astype(float)
panel["lnoutreach"] = np.where(
    panel["outreachper0to130"] > 0,
    np.log(panel["outreachper0to130"].replace(0, np.nan)),
    0.0,
)

# Outcome variables
panel["ln_num_0to130"] = np.log(panel["fywgt_0to130fpl"])
panel["ln_denom_0to130"] = np.log(panel["wtsupp_0to130fpl"])
panel["ln_num_50to115"] = np.log(panel["fywgt_50to115fpl"])
panel["ln_denom_50to115"] = np.log(panel["wtsupp_50to115fpl"])

# Treatment regressor: inclmt / 100 (eligibility threshold as fraction of FPL)
panel["inclmt100"] = panel["bbce_inclmt"] / 100.0

# GL index control = index_nobbce (paper Column 1 uses the combined index)
panel["gl_index"] = panel["index_nobbce"]

# Save full panel
panel.to_parquet(DATA / "full_panel.parquet")

# -----------------------------------------------------------------------------
# Event-study sample construction (mirrors build_event in Stata)
# -----------------------------------------------------------------------------
# Restrict to 1996-2016
p = panel[(panel["year"] >= 1996) & (panel["year"] <= 2016)].copy()
p = p.sort_values(["statefip", "year"]).reset_index(drop=True)

# Detect BBCE eligibility expansion events: bbce_inclmt changes from year-1.
p["prev_inclmt"] = p.groupby("statefip")["bbce_inclmt"].shift(1)
p["new_bbce"] = ((p["bbce_inclmt"].notna()) & (p["prev_inclmt"].notna())
                  & (p["bbce_inclmt"] != p["prev_inclmt"])).astype(int)

# Count of events per state and event index
p["event_num"] = p.groupby("statefip")["new_bbce"].cumsum()
# States with exactly 1 event (drop states with 2+)
events_per_state = p.groupby("statefip")["new_bbce"].sum()
multi_event_states = events_per_state[events_per_state >= 2].index.tolist()
zero_event_states = events_per_state[events_per_state == 0].index.tolist()

# Treated: states with exactly one event and a 5-year balanced window around it.
treated_info = []
for s in events_per_state[events_per_state == 1].index:
    sub = p[p["statefip"] == s]
    ey = int(sub.loc[sub["new_bbce"] == 1, "year"].iloc[0])
    post_max = int(sub["year"].max())
    pre_min = int(sub["year"].min())
    # Balanced panel: need [ey-5, ey+5] all present
    if ey - 5 >= pre_min and ey + 5 <= post_max:
        treated_info.append({"statefip": s, "event_year": ey})
treated_df = pd.DataFrame(treated_info)
print(f"treated states w/ balanced 5-yr window: {len(treated_df)}")

# Build event-study sample: 45 states (as in paper)
#  - treated states: keep ±5 years around event
#  - untreated (no BBCE expansion ever): all years in 1996-2016? No — the
#    Stata code keeps untreated states with event_time set to -1 for every
#    observation. But the balanced panel filter in build_event restricts to
#    event_time in [-5,5]. For untreated states (event_time=-1 for all obs),
#    first_event==last_event==-1, so the "keep if first_event<=-5 or
#    (first_event==-1 & last_event==-1)" keeps them, then "keep if inrange(
#    event_time, -5, 5)" keeps them (since -1 is in range). So untreated
#    states contribute every year. But actually looking again, untreated
#    states have all years mapped to event_time=-1 but they're duplicated
#    per expansion. Let me re-read...
#
#  Actually the Stata logic expands each row 50x via "expand 50" and assigns
#  event_time relative to the single expansion event. For untreated states
#  (no row in the "regs" file that has new_bbce==1), the merge keeps using
#  the untreated state-year rows unchanged with event_time=-1. So untreated
#  states contribute each of their state-year observations exactly once with
#  event_time=-1, and inclmt is 130 (federal floor). Treated states
#  contribute 11 obs (event_time -5..+5) per their balanced window.
rows = []
for _, t in treated_df.iterrows():
    s = int(t["statefip"]); ey = int(t["event_year"])
    sub = p[(p["statefip"] == s) & (p["year"].between(ey - 5, ey + 5))]
    if len(sub) == 11:
        rows.append(sub)
    # else: not truly balanced on all years — skip
treated_obs = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

untreated = p[p["statefip"].isin(zero_event_states)].copy()

es = pd.concat([treated_obs, untreated], ignore_index=True)
print(f"event-study N: {len(es)}, states: {es['statefip'].nunique()}")

es.to_parquet(DATA / "eventstudy_sample.parquet")
