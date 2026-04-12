"""Shared paths and helpers for replication of Liu (2025/26) 'Partial Specialization
and Heterogeneous Task Assignments' (openICPSR 209643)."""
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "209643-V1" / "replication"
DATA = PKG / "data"
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)


def weighted_mean(x, w):
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    return np.sum(x[m] * w[m]) / np.sum(w[m])


def weighted_var(x, w, ddof=1):
    """Stata-style analytic-weight variance: uses ddof=1 on the effective N."""
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[m], w[m]
    W = w.sum()
    mu = np.sum(x * w) / W
    # Stata aweights: var = Σw(x-μ)² / (W - W/N) = N/(N-1) * Σw(x-μ)²/W
    N = len(x)
    ss = np.sum(w * (x - mu) ** 2)
    return ss / W * N / (N - ddof) if N > ddof else np.nan


def load_morg_pooled():
    df80 = pd.read_stata(DATA / "morg_cleaned_1980.dta", convert_categoricals=False)
    df00 = pd.read_stata(DATA / "morg_cleaned_2000.dta", convert_categoricals=False)
    df = pd.concat([df80, df00], ignore_index=True)
    return df


def apply_sample_filter(df):
    df = df[(df["age"] >= 21) & (df["age"] <= 60)].copy()
    df = df[df["mlr"].isin([1, 2])]
    df = df[df["ft"] == 1]
    df = df[df["hr_wage"].notna()]
    df = df[(df["hr_wage"] > 5) & (df["hr_wage"] < 1500)]
    df = df.rename(columns={"hr_wage": "wage"})
    df["young"] = (df["age"] <= 40).astype(int)
    df["male"] = (df["sex"] == 1).astype(int)
    return df


def build_edu(df):
    """Reproduce the 5-level edu construction in Table 1a.do, collapsed to 4 levels
    (edgtc merged into edclg by subtracting 1 from edcat5==5)."""
    # 1980 branch: uses _grdhi and grdcom
    pre = df["year"] <= 1991
    edu_pre = np.maximum(0, df["_grdhi"] - (df["grdcom"] == 2).astype(int))
    ed8 = (edu_pre <= 8).astype(int)
    ed9 = (edu_pre == 9).astype(int)
    ed10 = (edu_pre == 10).astype(int)
    ed11 = (edu_pre == 11).astype(int)
    edhsg = ((edu_pre == 12) & (df["grdcom"] == 1)).astype(int)
    edsmc = (((edu_pre >= 13) & (edu_pre <= 15)) | ((edu_pre == 12) & (df["grdcom"] == 2))).astype(int)
    edclg = ((edu_pre == 16) | (edu_pre == 17)).astype(int)
    edgtc = (edu_pre > 17).astype(int)

    # 2000 branch: uses grdatn; overrides for year 2000 or 2002 (only 2000 here)
    post = df["year"].isin([2000, 2002])
    g = df["grdatn"]
    ed8 = np.where(post, (g <= 34).astype(int), ed8)
    ed9 = np.where(post, (g == 35).astype(int), ed9)
    ed10 = np.where(post, (g == 36).astype(int), ed10)
    ed11 = np.where(post, (g == 37).astype(int), ed11)
    edhsg = np.where(post, ((g == 38) | (g == 39)).astype(int), edhsg)
    edsmc = np.where(post, ((g >= 40) & (g <= 42)).astype(int), edsmc)
    edclg = np.where(post, (g == 43).astype(int), edclg)
    edgtc = np.where(post, (g >= 44).astype(int), edgtc)

    edhsd = ed8 + ed9 + ed10 + ed11
    edcat5 = np.full(len(df), np.nan)
    edcat5 = np.where(edhsd == 1, 1, edcat5)
    edcat5 = np.where(edhsg == 1, 2, edcat5)
    edcat5 = np.where(edsmc == 1, 3, edcat5)
    edcat5 = np.where(edclg == 1, 4, edcat5)
    edcat5 = np.where(edgtc == 1, 5, edcat5)
    # collapse edgtc to edclg
    edcat5 = np.where(edcat5 == 5, 4, edcat5)
    df = df.copy()
    df["edu"] = edcat5
    df = df[df["edu"].notna()]
    return df


def merge_crosswalk(df):
    cw = pd.read_stata(DATA / "crosswalk.dta", convert_categoricals=False)
    # Stata file has year in {1980, 1990, 2000}. MORG has year in {1980, 2000}.
    cw = cw[cw["year"].isin([1980, 2000])].copy()
    df = df.merge(cw, on=["year", "occ"], how="left")
    df = df.rename(columns={"occ1990": "occ1990dd"})
    df = df.drop(columns=["occ"])
    return df


# ---- 20 / 30 / 40 occupation aggregators derived from Table 1a.do ----

def map_20occ(o):
    # executive, admin, managerial
    if 3 <= o <= 22: return 1
    # management related
    if 23 <= o <= 37: return 2
    # engineer
    if 44 <= o <= 59: return 3
    # math/computer
    if 64 <= o <= 68: return 3
    # life/phys/social sci
    if 69 <= o <= 83: return 3
    # social sci (misplaced codes)
    if o in (166, 167, 169, 173): return 3
    # architect
    if o == 43: return 5
    # community/social services
    if o in (163, 174, 176, 177): return 4
    # lawyers
    if o == 178: return 4
    # education/legal support
    if 154 <= o <= 159: return 5
    if o in (164, 165, 329, 387, 234, 468): return 5
    # arts/design/media
    if 183 <= o <= 199: return 5
    if o == 228: return 5
    # health diagnosing
    if 84 <= o <= 89: return 6
    # health assess / treat
    if 95 <= o <= 106: return 6
    # technicians
    if 203 <= o <= 235: return 7
    # financial sales
    if 243 <= o <= 258: return 8
    # retail sales
    if 274 <= o <= 283: return 9
    # administrative support
    if 303 <= o <= 389: return 10
    # housekeeping / cleaning
    if 405 <= o <= 408: return 11
    if 448 <= o <= 455: return 11
    # protective service
    if 415 <= o <= 427: return 12
    # food prep
    if 433 <= o <= 444: return 13
    if 457 <= o <= 458: return 13
    if 459 <= o <= 467: return 13
    if 469 <= o <= 472: return 13
    # health service (dental/nursing aides) → 6
    if 445 <= o <= 447: return 6
    # farm / extractive
    if 473 <= o <= 498: return 14
    if 614 <= o <= 617: return 14
    # mechanics
    if 503 <= o <= 549: return 15
    # construction
    if 558 <= o <= 599: return 16
    # precision production
    if 628 <= o <= 699: return 17
    # machine operators
    if 703 <= o <= 799: return 18
    # transport
    if 803 <= o <= 859: return 19
    # handlers/laborers
    if 860 <= o <= 889: return 20
    return np.nan


def map_30occ(o):
    if 3 <= o <= 22: return 1
    if 23 <= o <= 37: return 2
    if o == 43: return 3
    if 44 <= o <= 59: return 4
    if 64 <= o <= 68: return 5
    if 69 <= o <= 83: return 6
    if o in (166, 167, 169, 173): return 6
    if o in (163, 174, 176, 177): return 7
    if o == 178: return 8
    if 154 <= o <= 159: return 9
    if o in (164, 165, 329, 387, 234): return 9
    if 183 <= o <= 199: return 10
    if o == 228: return 10
    if 84 <= o <= 89: return 11
    if 95 <= o <= 106: return 12
    if 203 <= o <= 235: return 13
    if 243 <= o <= 258: return 14
    if 274 <= o <= 283: return 15
    if 303 <= o <= 389: return 16
    if 405 <= o <= 408: return 17
    if 415 <= o <= 427: return 18
    if 433 <= o <= 444: return 19
    if 445 <= o <= 447: return 20
    if 448 <= o <= 455: return 21
    if 457 <= o <= 458: return 22
    if 459 <= o <= 467: return 22
    if 469 <= o <= 472: return 22
    if o == 468: return 23
    if 473 <= o <= 498: return 24
    if 614 <= o <= 617: return 24
    if 503 <= o <= 549: return 25
    if 558 <= o <= 599: return 26
    if 628 <= o <= 699: return 27
    if 703 <= o <= 799: return 28
    if 803 <= o <= 859: return 29
    if 860 <= o <= 889: return 30
    return np.nan


def map_40occ(o):
    if 3 <= o <= 22: return 1
    if 23 <= o <= 37: return 2
    if o == 43: return 3
    if 44 <= o <= 59: return 4
    if 64 <= o <= 68: return 5
    if 69 <= o <= 83: return 6
    if 84 <= o <= 89: return 7
    if 95 <= o <= 106: return 8
    if 113 <= o <= 154: return 9
    if 155 <= o <= 163: return 10
    if o in (164, 165): return 11
    if 166 <= o <= 173: return 12
    if 174 <= o <= 176: return 13
    if o in (178, 179): return 14
    if 183 <= o <= 200: return 15
    if 203 <= o <= 208: return 16
    if 213 <= o <= 235: return 17
    if 243 <= o <= 256: return 18
    if 258 <= o <= 283: return 19
    if 303 <= o <= 315: return 20
    if 316 <= o <= 323: return 21
    if 326 <= o <= 336: return 22
    if 337 <= o <= 344: return 23
    if 345 <= o <= 357: return 24
    if 359 <= o <= 373: return 25
    if 375 <= o <= 389: return 26
    # 27..40 from second half of Table 1a.do (around lines 380+)
    if 405 <= o <= 408: return 27
    if 415 <= o <= 427: return 28
    if 433 <= o <= 444: return 29
    if 445 <= o <= 447: return 30
    if 448 <= o <= 455: return 31
    if 456 <= o <= 458: return 32
    if 459 <= o <= 467: return 32
    if 469 <= o <= 472: return 32
    if o == 468: return 33
    if 473 <= o <= 498: return 34
    if 614 <= o <= 617: return 34
    if 503 <= o <= 549: return 35
    if 558 <= o <= 599: return 36
    if 628 <= o <= 699: return 37
    if 703 <= o <= 799: return 38
    if 803 <= o <= 859: return 39
    if 860 <= o <= 889: return 40
    return np.nan
