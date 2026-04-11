"""Shared utilities for the Henry, Zhuravskaya & Guriev (AER 2022) replication.

Paper: "Checking and Sharing Alt-Facts"
Package: 140161-V1/

Key insight: the Stata do file extensively renames Qualtrics raw variable
numbers to questionnaire order (line 751+ of 1.infile_data.do). In the raw
CSV, "q34" is the "would you like to share alt-facts on Facebook?" yes/no,
and "q41" is the "would you like to share the fact-check on Facebook?" yes/no.
In the final analysis data these are called `want_share_fb` and
`want_share_facts` respectively (and flagged on line 79: `gen fake = 1 if q34==1`
and line 85: `replace fact = 1 if q41==1`).
"""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "140161-V1"
RAW = PKG / "original_data"
OUT = ROOT / "replication_140161" / "output"
OUT.mkdir(exist_ok=True, parents=True)

SURVEY_CSVS = {
    1: RAW / "Survey+1_May+27%2C+2019_01.02.csv",
    2: RAW / "Survey+2_May+27%2C+2019_01.03.csv",
    3: RAW / "Survey+3_May+27%2C+2019_01.03.csv",
}

TREATMENT_LABELS = {1: "Alt-Facts", 2: "Imposed Fact-Check", 3: "Voluntary Fact-Check"}


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def load_survey(i: int) -> pd.DataFrame:
    """Load one Wave 1 Qualtrics export and apply the do-file filters."""
    df = pd.read_csv(SURVEY_CSVS[i], low_memory=False)
    # Stata's `import delimited` strips spaces and parentheses from var names.
    df.columns = [c.lower().replace(" ", "").replace("(", "").replace(")", "")
                  for c in df.columns]
    # Rows 1-3 of the data frame (after header) are Qualtrics metadata; Stata
    # reads them as data and then `keep if n > 4`.
    df = df.iloc[3:].reset_index(drop=True)
    df = df[df["distributionchannel"] != "preview"].copy()
    df["durationinseconds"] = to_num(df["durationinseconds"])
    df = df[df["durationinseconds"] >= 250]
    df["gc"] = to_num(df["gc"])
    df = df[df["gc"] == 1]
    df["survey"] = i
    return df.reset_index(drop=True)


def build_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Construct the core Wave-1 analysis variables used in Tables 1-3."""
    out = pd.DataFrame(index=df.index)
    out["survey"] = df["survey"].astype(int)
    out["imposed"] = (out["survey"] == 2).astype(int)
    out["voluntary"] = (out["survey"] == 3).astype(int)

    # ---- main DVs (match the Stata infile pipeline) ----
    # Line 79-80 of 1.infile_data.do: gen fake = 1 if q34==1 / replace fake=0 if .
    # Then q34 → q29, and want_share_fb = 1 if q29==1. So want_share_fb == raw_q34==1.
    out["want_share_fb"] = (to_num(df.get("q34")) == 1).astype(int)
    # Line 84-85: replace fact = 1 if q41==1 (surveys 2 and 3 only).
    # Then q41 → q31, and want_share_facts = 1 if q31==1.
    q41 = to_num(df.get("q41")) if "q41" in df.columns else pd.Series(np.nan, index=df.index)
    fact = (q41 == 1).astype(int)
    fact.loc[out["survey"] == 1] = 0  # never asked in survey 1
    out["want_share_facts"] = fact.values

    # "See facts" (q28 in the raw, renamed q24 later, then `see_facts=q28` at
    # line 1005 is the RENAMED q28 i.e. raw q38). q38 in the raw CSV is the
    # branching question "Would you like to view fact-checking information?"
    # for the Voluntary Fact-Check treatment.
    see_facts = to_num(df.get("q38"))
    out["see_facts"] = (see_facts == 1).astype(int)  # 1 = view, 2 = don't, 3 = neutral
    out.loc[out["survey"] != 3, "see_facts"] = 0

    # ---- pre-treatment / demographic controls (approximations) ----
    # Raw Q2 = age (line 689: rename q2 age — pre-other renames).
    out["age"] = to_num(df.get("q2"))
    out["age_sqrd"] = out["age"] ** 2
    # Raw Q5 = gender (1 = male, 2 = female). Line 685 of 1.infile_data.do:
    #   gen male = 1 if q5==1  /  replace male = 0 if q5==2
    out["male_raw"] = to_num(df.get("q5"))
    out["male"] = (out["male_raw"] == 1).astype(int)
    # Raw Q4 = education (1-9 scale labelled on line 165).
    out["educ"] = to_num(df.get("q4"))
    # Raw Q7 = income (line 41 destringed along with q7).
    out["income"] = to_num(df.get("q7"))
    # Raw Q8 = religious (line 41 destringed).
    out["religion_q8"] = to_num(df.get("q8"))
    # Raw Q13 = FB frequency, Q14 = # friends, Q15 = share-often
    out["q13"] = to_num(df.get("q13"))
    out["q14"] = to_num(df.get("q14"))
    out["q15"] = to_num(df.get("q15"))
    # Raw Q17 = negative image of EU (integer scale)
    out["neg_image_ue"] = to_num(df.get("q17"))
    # Raw Q21 = 2nd-round 2017 vote (2 = MLP per the do-file label)
    out["vote_2nd"] = to_num(df.get("q21"))
    out["second_mlp"] = (out["vote_2nd"] == 2).astype(int)

    # Reasons to share (Q16_1..Q16_4) — Stata keeps as continuous 0-100 slider
    for k, name in [(1, "share_interest"), (2, "share_reciprocity"),
                    (3, "share_image"), (4, "share_influence")]:
        col = f"q16_{k}"
        out[name] = to_num(df.get(col))
    # Behavioral — donated money/blood/worked (Q18_*)
    for k, name in [(1, "gave_charity"), (2, "gave_homeless"),
                    (3, "donated_blood"), (4, "worked_charity")]:
        col = f"q18_{k}"
        out[name] = (to_num(df.get(col)) == 1).astype(int)
    out["altruism"] = out[["gave_charity", "gave_homeless",
                            "worked_charity", "donated_blood"]].mean(axis=1)
    # Self-reported reciprocity/image on Q19_*
    out["reciprocity"] = to_num(df.get("q19_1"))
    out["image"] = to_num(df.get("q19_2"))

    return out


def build_all() -> pd.DataFrame:
    pieces = [build_analysis(load_survey(i)) for i in (1, 2, 3)]
    return pd.concat(pieces, ignore_index=True)
