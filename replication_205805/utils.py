"""Shared paths and helpers for replication of Anders & Rafkin (AEJ:Applied, 2025).

Paper 205805 — 'The Welfare Effects of Eligibility Expansions: Theory and Evidence from SNAP.'
"""
from __future__ import annotations
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "205805-V1" / "AR_Public_Repo_AEJ"
RAW = PKG / "raw"
GLDATA = RAW / "gldata"
QC_OLD = GLDATA / "qc"
QC_NEW = GLDATA / "qc_ar"
SNAP = RAW / "snap"
OUT = ROOT / "replication_205805"
DATA = OUT / "data"
DATA.mkdir(parents=True, exist_ok=True)

CPS_DTA = RAW / "cps_00043.dta"
POLICY_XLS = SNAP / "SNAP_Policy_Database.xlsx"
UNEMPL_XLS = RAW / "State_level_unemployment_rate.xls"
STATE_GEO = RAW / "state-geocodes-v2017.xlsx"

# FY files — each 'qcfyYYYY' spans Oct(YYYY-1)–Sep(YYYY). We want calendar years 1996–2016.
QC_FILES_OLD = [
    ("qcfy96.dta",), ("qcfy1997.dta",), ("qcfy1998.dta",), ("qcfy1999.dta",),
    ("qcfy2000.dta",), ("qcfy2001.dta",), ("qcfy2002.dta",), ("qcfy2003.dta",),
    ("qcfy2004.dta",), ("qcfy2005.dta",), ("qcfy2006.dta",), ("qcfy2007.dta",),
    ("qcfy2008.dta",), ("qcfy2009.dta",), ("qcfy2010.dta",), ("qcfy2011.dta",),
    ("qc_pub_fy2012.dta",), ("qc_pub_fy2013.dta",), ("qc_pub_fy2014.dta",),
]
QC_FILES_NEW = [("qc_pub_fy2015.dta",), ("qc_pub_fy2016.dta",), ("qc_pub_fy2017.dta",)]


def qc_file_paths():
    out = [QC_OLD / f for (f,) in QC_FILES_OLD]
    out += [QC_NEW / f for (f,) in QC_FILES_NEW]
    return out
