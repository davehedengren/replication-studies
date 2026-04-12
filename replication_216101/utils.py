"""Shared helpers for replicating Moschini, Raveendranathan, Xu (AER 2024)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "216101-V1" / "empirical"
NLSY_CSV = PKG / "nlsy_hsls_psid" / "data_raw" / "nlsy97" / "NLSY97_extract.csv"
HSLS_DTA = PKG / "nlsy_hsls_psid" / "data_raw" / "hsls09" / "subset_hsls_17_student_pets_sr_v1_0.dta"
CPI_CSV = PKG / "nlsy_hsls_psid" / "data_raw" / "cpi" / "CPIAUCSL.csv"

NLSY_YEARS = [1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005,
              2006, 2007, 2008, 2009, 2010, 2011, 2013, 2015, 2017, 2019]

ED_STATUS_VARS = {
    1997: "R1201400", 1998: "R2560001", 1999: "R3881501", 2000: "R5460601",
    2001: "R7224201", 2002: "S1538001", 2003: "S2007701", 2004: "S3808501",
    2005: "S5408900", 2006: "S7509700", 2007: "T0013000", 2008: "T2015900",
    2009: "T3606200", 2010: "T5206600", 2011: "T6656400", 2013: "T8128800",
    2015: "U0008700", 2017: "U1845300", 2019: "U3443800",
}

CC_WEIGHT_VARS = {
    1997: "R1236101", 1998: "R2600301", 1999: "R3923701", 2000: "R5510600",
    2001: "R7274200", 2002: "S1598100", 2003: "S2067000", 2004: "S3861600",
    2005: "S5444200", 2006: "S7545500", 2007: "T0042100", 2008: "T2022500",
    2009: "T3613300", 2010: "T5213200", 2011: "T6665000", 2013: "T8135900",
    2015: "U0017100", 2017: "U1855400", 2019: "U3455500",
}

# Core cross-section identifiers and measures
ID = "R0000100"
AGE_1997 = "R1194100"
SEX = "R0536300"
ASVAB = "R9829600"
PARENT_INC_1997 = "R1204500"
RES_DAD_ED = "R1302600"
RES_MOM_ED = "R1302700"
YOUTH_BELIEF_1997 = "R0515100"
YOUTH_BELIEF_2001_A = "R6884400"
YOUTH_BELIEF_2001_B = "R6532700"
PARENT_BELIEF_1997 = "R0689000"


def load_nlsy() -> pd.DataFrame:
    df = pd.read_csv(NLSY_CSV)
    # negative codes (refusal, don't know, invalid skip, valid skip, noninterview)
    # are left as-is; downstream masks treat them explicitly.
    return df


def cpi_2016_deflator() -> dict[int, float]:
    cpi = pd.read_csv(CPI_CSV)
    cpi["year"] = pd.to_datetime(cpi["DATE"]).dt.year
    cpi = cpi.set_index("year")["CPIAUCSL_NBD19920101"]
    base = cpi.loc[2016]
    return (cpi / base).to_dict()


def weighted_mean_se(x: pd.Series, w: pd.Series | None = None):
    """Unweighted mean and SE of mean. Paper does not use NLSY weights."""
    x = pd.Series(x).dropna()
    n = len(x)
    if n == 0:
        return np.nan, np.nan, 0
    m = x.mean()
    se = x.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return m, se, n


def build_nlsy_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Derive all flags needed for Tables 1-4.

    Mirrors the Stata cleaning in NLSY97_Cleaning.do for the cross-section
    used in the beliefs tabulations. We follow the paper's convention of
    not using survey weights, but we do use cc_weight>0 for presence flags.
    """
    out = pd.DataFrame({"id": df[ID]})
    out["age_1997"] = df[AGE_1997]
    out["year_30"] = 1997 + 30 - out["age_1997"]
    out["year_25"] = 1997 + 25 - out["age_1997"]
    out["sex_female"] = (df[SEX] == 2).astype(int)
    out["asvab"] = np.where(df[ASVAB] >= 0, df[ASVAB] / 1000.0, np.nan)
    out["hh_gross_Y_parents"] = np.where(df[PARENT_INC_1997] >= 0,
                                         df[PARENT_INC_1997], np.nan)

    # Parental education: >=16 years & <95 => BA+
    dad = df[RES_DAD_ED]
    mom = df[RES_MOM_ED]
    dad_high = np.where(dad >= 0, ((dad >= 16) & (dad < 95)).astype(float), np.nan)
    mom_high = np.where(mom >= 0, ((mom >= 16) & (mom < 95)).astype(float), np.nan)
    both_missing = np.isnan(dad_high) & np.isnan(mom_high)
    high_par = np.fmax(np.nan_to_num(dad_high, nan=-1),
                       np.nan_to_num(mom_high, nan=-1))
    high_par = np.where(both_missing, np.nan, np.where(high_par < 0, 0, high_par))
    out["flag_highedrespar"] = high_par

    # Real parental income in 2016 dollars
    deflator = cpi_2016_deflator()[1997]  # CPI97/CPI16
    out["r_hh_gross_Y_parents"] = out["hh_gross_Y_parents"] / deflator
    out["r_log_hhinc"] = np.log(out["r_hh_gross_Y_parents"].where(out["r_hh_gross_Y_parents"] > 0))

    # Belief variables.  1997 question is only populated for those who were
    # still in grades 1-12 (==8 in ed status) in 1997 (i.e., born 1980-81).
    # The 2001 question is asked to those still in HS in 2001.
    r1997 = df[YOUTH_BELIEF_1997]
    r2001a = df[YOUTH_BELIEF_2001_A]
    r2001b = df[YOUTH_BELIEF_2001_B]
    ed97 = df[ED_STATUS_VARS[1997]]
    ed01 = df[ED_STATUS_VARS[2001]]

    pct = np.full(len(df), np.nan)
    age_b = np.full(len(df), np.nan)
    mask = (r1997 >= 0) & (ed97 == 8)
    pct[mask] = r1997[mask]
    age_b[mask] = out.loc[mask, "age_1997"]
    mask = (r2001a >= 0) & (ed01 == 8)
    pct[mask] = r2001a[mask]
    age_b[mask] = out.loc[mask, "age_1997"] + 4
    mask = (r2001b >= 0) & (ed01 == 8)
    pct[mask] = r2001b[mask]
    age_b[mask] = out.loc[mask, "age_1997"] + 4
    out["pct_chance_BA"] = pct
    out["age_youth_beliefs"] = age_b

    # Parent 1997 belief (asked only if child in HS grades in 1997)
    pb = df[PARENT_BELIEF_1997]
    out["parent_pct_chanceBA"] = np.where((pb >= 0) & (ed97 == 8), pb, np.nan)

    # Attainment / enrollment flags by age cutoffs
    ed_cols = {y: df[v] for y, v in ED_STATUS_VARS.items()}

    def _accum_flag(values_by_year, codes, start_year, cutoff):
        flag = np.zeros(len(df), dtype=float)
        # survey years strictly after birth-based cutoff year
        for y in NLSY_YEARS:
            if y == 1997:
                continue
            col = values_by_year[y]
            # For biannual years (2013, 2015, ...) the Stata code uses
            # "year_cutoff >= y-1" to catch off-year birthdays
            if y >= 2013:
                active = (out["year_30"] >= y - 1) if cutoff == 30 else (out["year_25"] >= y - 1)
            else:
                active = (out["year_30"] >= y) if cutoff == 30 else (out["year_25"] >= y)
            upd = active & (flag == 0) & col.isin(codes)
            flag = np.where(upd, 1, flag)
        return flag

    BA_codes = [6, 7, 11]
    ENR_BA_codes = [6, 7, 10, 11]
    HS_grad_codes = [3, 4, 5, 6, 7, 9, 10, 11]
    Enr2_codes = [5, 9]
    SomeColl_codes = [4]
    Missing_codes = [-3, -5]

    out["BA_by_30"] = _accum_flag(ed_cols, BA_codes, None, 30)
    out["EnrBA_by_30"] = _accum_flag(ed_cols, ENR_BA_codes, None, 30)
    out["HS_grad_by_30"] = _accum_flag(ed_cols, HS_grad_codes, None, 30)
    out["Enr2yr_by_30"] = _accum_flag(ed_cols, Enr2_codes, None, 30)
    out["Ambig_some_college_by_30"] = _accum_flag(ed_cols, SomeColl_codes, None, 30)

    out["BA_by_25"] = _accum_flag(ed_cols, BA_codes, None, 25)
    out["EnrBA_by_25"] = _accum_flag(ed_cols, ENR_BA_codes, None, 25)
    out["Enr2yr_by_25"] = _accum_flag(ed_cols, Enr2_codes, None, 25)
    out["Ambig_some_college_by_25"] = _accum_flag(ed_cols, SomeColl_codes, None, 25)

    # Missing education in year of age cutoff
    def _missing_in_cutoff_year(cutoff):
        mflag = np.zeros(len(df), dtype=int)
        for y in NLSY_YEARS:
            if y == 1997:
                continue
            col = ed_cols[y]
            if y >= 2013:
                match = (out[f"year_{cutoff}"] == y) | (out[f"year_{cutoff}"] == y - 1)
            else:
                match = (out[f"year_{cutoff}"] == y)
            mflag = np.where((mflag == 0) & match & col.isin(Missing_codes), 1, mflag)
        return mflag

    out["Missing_ed_flag30"] = _missing_in_cutoff_year(30)
    out["Missing_ed_flag25"] = _missing_in_cutoff_year(25)

    # Presence flags using cc_weight>0 in year_30 / year_25
    def _cc_present(cutoff):
        flag = np.zeros(len(df), dtype=int)
        for y, v in CC_WEIGHT_VARS.items():
            w = df[v]
            if y >= 2013:
                match = (out[f"year_{cutoff}"] == y) | (out[f"year_{cutoff}"] == y - 1)
            else:
                match = (out[f"year_{cutoff}"] == y)
            flag = np.where((flag == 0) & (w > 0) & match, 1, flag)
        return flag

    out["flag_cc_ID_present30"] = _cc_present(30)
    out["flag_cc_ID_present25"] = _cc_present(25)

    # Percent scale for EnrBA / BA flags
    for c in ["BA_by_30", "EnrBA_by_30", "HS_grad_by_30", "BA_by_25", "EnrBA_by_25"]:
        out[c + "_pct"] = 100 * out[c]

    # Baseline sample cleaning (NLSY97_Cleaning.do lines 703-713):
    # drop rows with missing ASVAB and those who didn't earn HS by 30.
    baseline_ok = out["asvab"].notna() & (out["HS_grad_by_30"] == 1)

    # Belief sample: cc-present at age 30, belief observed, HS grad by 30,
    # not ambiguous "some college" non-enrollee, education not missing at age 30.
    flag_unid = (
        (out["flag_cc_ID_present30"] > 0)
        & (out["Ambig_some_college_by_30"] == 1)
        & (out["Enr2yr_by_30"] == 0)
        & (out["EnrBA_by_30"] == 0)
    )
    sample = (
        baseline_ok
        & (out["flag_cc_ID_present30"] > 0)
        & out["pct_chance_BA"].notna()
        & ~flag_unid
        & (out["Missing_ed_flag30"] == 0)
    )
    out["flag_beliefs_sample"] = sample.astype(int)

    # ASVAB skill tercile computed on observed ASVAB (all respondents).
    q_asvab = pd.qcut(out["asvab"], 3, labels=[1, 2, 3])
    out["q_asvab"] = q_asvab
    # Paper uses "distribution of high school graduates" for terciles in
    # Tables 3/4. We follow the same convention: tercile is based on the
    # belief sample's skill distribution.
    hs_asvab = out.loc[out["flag_beliefs_sample"] == 1, "asvab"]
    cuts = hs_asvab.quantile([1 / 3, 2 / 3]).to_numpy()
    out["q_tabulations"] = np.where(
        out["asvab"].isna(), np.nan,
        np.where(out["asvab"] <= cuts[0], 1,
                 np.where(out["asvab"] <= cuts[1], 2, 3))
    )

    # Belief bins used in Table 1: [0,32], [33,66], [67,100]
    pb = out["pct_chance_BA"]
    out["belief_bin"] = np.where(pb.isna(), np.nan,
                                 np.where(pb <= 32, 1,
                                          np.where(pb <= 66, 2, 3)))
    return out
