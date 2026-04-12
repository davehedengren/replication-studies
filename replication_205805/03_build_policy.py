"""Build SNAP Policy Database state-year panel + unemployment rates.

Mirrors meanstest/code/build/policy.do and unemploymentrate.do. Produces the
state-year policy variables needed for the main event-study regression:
bbce_inclmt, has_bbce, veh, cap, facerec, oapp, reportsimple, call, noShortCert,
index_nobbce, outreach_orig, and state unemployment.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from utils import POLICY_XLS, UNEMPL_XLS, DATA

# -----------------------------------------------------------------------------
# SNAP policy DB → state-year
# -----------------------------------------------------------------------------
pol = pd.read_excel(POLICY_XLS)
pol["year"] = (pol["yearmonth"] // 100).astype(int)
pol["month"] = (pol["yearmonth"] - pol["year"] * 100).astype(int)
pol = pol.rename(columns={"state_fips": "statefip"})

# Stata: gen veh = vehexclone + vehexclall; replace veh = 1 if veh == 2
pol["veh"] = pol["vehexclone"].fillna(0) + pol["vehexclall"].fillna(0)
pol["veh"] = np.where(pol["veh"] > 0, 1, pol["veh"]).astype(float)

# Certification share < 3 months flag
pol["noShortCert"] = ((pol["certearn0103"] < 0.2) | pol["certearn0103"].isna()).astype(int)

# Recode call/faceini/facerec/fingerprint/oapp where 2 -> 1
for c in ["call", "faceini", "facerec", "fingerprint", "oapp"]:
    pol[c] = pol[c].replace({2: 1})

# index_nobbce = (veh + cap + facerec + oapp + reportsimple + call + noShortCert)/7
index_cols_full = ["veh", "cap", "facerec", "oapp", "reportsimple", "call", "noShortCert"]
pol["index_nobbce"] = pol[index_cols_full].mean(axis=1, skipna=False)
missing_veh = pol["veh"].isna()
if missing_veh.any():
    alt_cols = ["cap", "facerec", "oapp", "reportsimple", "call", "noShortCert"]
    pol.loc[missing_veh, "index_nobbce"] = pol.loc[missing_veh, alt_cols].mean(axis=1)

# bbce_inclmt = 130 where bbce==0 (federal floor is 130)
pol.loc[pol["bbce"] == 0, "bbce_inclmt"] = 130.0

# collapse: (min) over months for the binary/structural vars; (mean) for outreach
min_cols = ["bbce_inclmt", "facerec", "oapp", "noShortCert", "index_nobbce",
            "reportsimple", "ebtissuance", "call", "cap", "veh", "bbce"]
mean_cols = ["outreach"]
grp = pol.groupby(["statefip", "year"], as_index=False)
agg_dict = {c: "min" for c in min_cols}
agg_dict.update({c: "mean" for c in mean_cols})
pol_sy = grp.agg(agg_dict).rename(columns={"bbce": "has_bbce", "outreach": "outreach_orig"})
pol_sy["statefip"] = pol_sy["statefip"].astype(int)
pol_sy.to_parquet(DATA / "policy_stateyear.parquet")
print("policy:", pol_sy.shape)
print(pol_sy[pol_sy["year"].isin([2000, 2005, 2010, 2015])].head())

# -----------------------------------------------------------------------------
# State-level unemployment rate
# -----------------------------------------------------------------------------
# Format: xls with columns [observation_date, <SERIES per state>]. Series code
# like "ALUR" (Alabama unemployment rate), monthly values 1976-2019.
unem_raw = pd.read_excel(UNEMPL_XLS, sheet_name="Monthly", header=0)
date_col = unem_raw.columns[0]
unem_raw[date_col] = pd.to_datetime(unem_raw[date_col])
unem_raw["year"] = unem_raw[date_col].dt.year

# State postal -> fips lookup via policy DB
pc_fips = pol[["state_pc", "statefip"]].drop_duplicates()
series_map = {}
for col in unem_raw.columns[1:-1]:  # skip date and year
    if not isinstance(col, str):
        continue
    pc = col[:2]
    row = pc_fips[pc_fips["state_pc"] == pc]
    if len(row):
        series_map[col] = int(row.iloc[0]["statefip"])

annual = unem_raw.groupby("year").mean(numeric_only=True)
rows = []
for col, fips in series_map.items():
    if col in annual.columns:
        for y, v in annual[col].items():
            rows.append({"statefip": fips, "year": int(y), "unemployment": float(v)})
unem = pd.DataFrame(rows)
unem = unem[(unem["year"] >= 1990) & (unem["year"] <= 2019)]
unem.to_parquet(DATA / "unemployment_stateyear.parquet")
print("unemployment:", unem.shape)
print(unem.head())
