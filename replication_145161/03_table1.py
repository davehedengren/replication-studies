"""03_table1.py — build Table 1 (weekly hours in home production by housewives).

Replicates columns: USA 1920s (precomputed), USA 1965 + USA 2010 (MTUS),
SA 2000, SA 2010, Morocco 2011. Ghana & Sierra Leone are skipped (access-
restricted raw files not included).
"""
import numpy as np
import pandas as pd

from utils import DATA, OUT, read_dta, wmean

ACTS = ["cooking", "firewater", "cleaning", "laundry", "care", "domestic"]


def collapse_day_based(df):
    """For day-diary surveys: weighted mean per (day), then sum across days.

    This reproduces the Stata 'collapse X [pw=weight], by(day)' followed by
    'collapse (sum) X'. Net effect: weighted mean per diary day → sum over 7
    diary days → weekly hours.
    """
    if "day" not in df.columns:
        raise ValueError("need day column")
    df = df[df["day"].notna()].copy()
    out = {}
    for v in ACTS:
        if v not in df.columns:
            out[v] = 0.0
            continue
        per_day = []
        for _, d in df.groupby("day"):
            per_day.append(wmean(d[v], d["weight"]))
        out[v] = float(np.nansum(per_day))
    return out


# ------------------------------------------------------------------
# MTUS (US1965, US2010)
# ------------------------------------------------------------------
def build_mtus():
    m = read_dta(DATA / "Table1/MTUS/mtus_00004.dta")
    m = m[m["sample"].isin(["US1965", "US2010"])].copy()

    m["no_paid_mkt"] = (m["empstat"] == 4).astype("float64")
    m.loc[m["empstat"] < 1, "no_paid_mkt"] = np.nan

    m["no_education"] = (m["student"] != 1).astype("float64")
    m.loc[m["student"] < 0, "no_education"] = np.nan

    # Stata: married = (cohab==0|cohab==1). replace married=1 if cohab==.
    m["married"] = ((m["cohab"] == 0) | (m["cohab"] == 1)).astype("float64")
    m.loc[m["cohab"].isna(), "married"] = 1.0

    m["weight"] = m["propwt"]
    m = m[m["day"] >= 1].copy()
    m = m[m["age"] >= 0].copy()

    # rename
    ren = {
        "cooking": "cooking",
        "cleaning": "cleaning",
        "laundry": "laundry",
        "childcare": "care",
        "hhmanagement": "domestic",
    }
    for src, dst in ren.items():
        m[dst] = m[src].astype("float64") / 60.0
    m["firewater"] = 0.0

    results = {}
    n_rows = {}
    hhsize_rows = {}
    married_rows = {}
    sharehw_rows = {}

    for sample in ["US1965", "US2010"]:
        sub = m[m["sample"] == sample].copy()
        # share of housewives among women 15-59
        women = sub[(sub["sex"] == 2) & (sub["age"] >= 15) & (sub["age"] <= 59)].copy()
        hw_flag = ((women["no_education"] == 1) & (women["no_paid_mkt"] == 1)
                   & (women["married"] == 1)).astype(float)
        # published housewife share: weighted
        sharehw_rows[sample] = wmean(hw_flag, women["weight"])
        # married share among women 15-59
        married_rows[sample] = wmean(women["married"], women["weight"])

        hw = women[(women["no_education"] == 1) & (women["no_paid_mkt"] == 1)
                   & (women["married"] == 1)].copy()
        n_rows[sample] = len(hw)
        hhsize_rows[sample] = wmean(hw["hhldsize"], hw["weight"])
        results[sample] = collapse_day_based(hw)

    return results, n_rows, hhsize_rows, married_rows, sharehw_rows


# ------------------------------------------------------------------
# Morocco 2011
# ------------------------------------------------------------------
def build_morocco():
    c = read_dta(DATA / "Table1/Morocco/carnet_adulte_ENET2012.dta")
    men = read_dta(DATA / "Table1/Morocco/menage_ENET.dta")
    men = men[["ïn_mãnage", "taille_mãnage"]]
    df = c.merge(men, on="ïn_mãnage", how="left")

    # weight: string with comma decimal
    df["weight"] = df["coef_individu"].astype(str).str.replace(",", ".").astype(float)
    df["day"] = df["i050_q1"]

    df["cooking"] = df["dureeg51"].astype(float)
    df["firewater"] = df["dureeg56"].astype(float) + df["dureeg57"].astype(float)
    df["cleaning"] = df["dureeg52"].astype(float) + df["dureeg54"].astype(float)
    df["laundry"] = df["dureeg53"].astype(float)
    df["care"] = df[["dureeg61", "dureeg62", "dureeg63", "dureeg64"]].astype(float).sum(axis=1)
    df["domestic"] = df[["dureeg55", "dureeg58", "dureeg83", "dureeg84"]].astype(float).sum(axis=1)
    for v in ACTS:
        df[v] = df[v] / 60.0

    df["sex"] = df["sexe"]
    df["married"] = (df["etat_matrimonial"] == 2).astype(float)
    df["no_education"] = ((df["dureeg41"] == 0) & (df["dureeg42"] == 0)).astype(float)
    df["no_paid_mkt"] = ((df["type_activite"] == 3) & (df["dureeg3"] == 0)).astype(float)

    women = df[(df["sex"] == 2) & (df["age"] >= 15) & (df["age"] <= 59)].copy()
    # In Morocco, the Stata code sets no_education/no_paid_mkt to missing
    # when condition not met, so housewives require all flags==1.
    hw_flag = ((women["no_education"] == 1) & (women["no_paid_mkt"] == 1)
               & (women["married"] == 1)).astype(float)
    sharehw = wmean(hw_flag, women["weight"])
    married = wmean(women["married"], women["weight"])

    hw = women[(women["no_education"] == 1) & (women["no_paid_mkt"] == 1)
               & (women["married"] == 1)].copy()
    n = len(hw)
    hhsize = wmean(hw["taille_mãnage"], hw["weight"])
    hours = collapse_day_based(hw)
    return hours, n, hhsize, married, sharehw


# ------------------------------------------------------------------
# South Africa 2000
# ------------------------------------------------------------------
def build_sa2000():
    a = read_dta(DATA / "Table1/South Africa/2000/tus-2000-activall-v1.dta")
    p = read_dta(DATA / "Table1/South Africa/2000/tus-2000-person-v1.dta")
    mem = read_dta(DATA / "Table1/South Africa/2000/tus-2000-member-v1.dta")

    # tranche issue
    a.loc[(a["uqnr"] == 81740121901) & (a["tranche"] == "3"), "person"] = 2
    a["hhid"] = a["uqnr"].map(lambda x: f"{int(x):011d}")
    a["id"] = a["hhid"] + a["person"].astype(str)

    p = p.drop_duplicates()
    p.loc[(p["uqnr"] == 81740121901) & (p["tranche"] == "3"), "person"] = 2
    p["hhid"] = p["uqnr"].map(lambda x: f"{int(x):011d}")
    p["id"] = p["hhid"] + p["person"].astype(str)

    a = a.merge(p[["id", "child18h"]], on="id", how="left")
    a["child18h"] = pd.to_numeric(a["child18h"], errors="coerce")
    a["childinhouse"] = ((a["child18h"] > 0) & a["child18h"].notna()).astype(int)

    tp = a["timeper"].astype(str)
    act = pd.Series("Other", index=a.index)
    act = act.mask(tp.isin(["410", "230"]), "cooking")
    act = act.mask(tp.isin(["236", "250"]), "firewater")
    act = act.mask(tp.isin(["420", "210", "460"]), "cleaning")
    # care
    child_codes = ["511", "512", "521", "522", "531", "532", "671", "672"]
    act = act.mask(tp.isin(child_codes), "care")
    adult_codes = ["540", "550", "673"]
    act = act.mask(tp.isin(adult_codes), "care")
    either_codes = ["580", "590", "561", "562"]
    act = act.mask(tp.isin(either_codes), "care")  # regardless of child (both branches → care)
    act = act.mask(tp == "430", "laundry")
    act = act.mask(tp.isin(["260", "440", "330"]), "domestic")
    market_codes = ["111", "112", "114", "115", "130", "180", "188", "190"]
    act = act.mask(tp.isin(market_codes), "MARKET")
    edu_codes = ["710", "720", "780", "788", "790"]
    act = act.mask(tp.isin(edu_codes), "Education")
    a["activity"] = act

    a["min_split"] = pd.to_numeric(a["min"], errors="coerce") / 10.0

    # collapse sum by id, activity
    coll = a.groupby(["id", "activity"])["min_split"].sum().unstack(fill_value=0)
    coll.columns = [f"min_{c}" for c in coll.columns]
    coll = coll.reset_index()

    # Bring person-level fields
    pp = p.copy()
    pp["gender"] = pd.to_numeric(pp["gender"], errors="coerce")
    pp["age"] = pd.to_numeric(pp["age"], errors="coerce")
    pp["perwgt"] = pd.to_numeric(pp["perwgt"], errors="coerce")
    pp["married"] = (pp["mar1tals"] == "2").astype(float)
    pp = pp[pp["daydiary"] != "*"].copy()
    pp["daydiary"] = pd.to_numeric(pp["daydiary"], errors="coerce")

    df = coll.merge(pp[["id", "hhid", "gender", "age", "perwgt", "daydiary",
                         "mar1tals", "sourcein", "married"]], on="id", how="left")

    # hhsize from member file
    mem["hhid"] = mem["uqnr"].map(lambda x: f"{int(x):011d}")
    hs = mem.groupby("hhid").size().rename("hhsize").reset_index()
    df = df.merge(hs, on="hhid", how="left")

    df["sex"] = df["gender"]
    df["weight"] = df["perwgt"]
    df["day"] = df["daydiary"]

    df["no_education"] = (df.get("min_Education", 0) == 0).astype(float)
    df["no_paid_mkt"] = ((df.get("min_MARKET", 0) == 0)
                         & (~df["sourcein"].isin(["03", "04"]))).astype(float)

    for src, dst in [("min_cooking", "cooking"), ("min_firewater", "firewater"),
                      ("min_cleaning", "cleaning"), ("min_laundry", "laundry"),
                      ("min_care", "care"), ("min_domestic", "domestic")]:
        if src not in df.columns:
            df[dst] = 0.0
        else:
            df[dst] = df[src].astype(float) / 60.0

    women = df[(df["sex"] == 2) & (df["age"] >= 15) & (df["age"] <= 59)].copy()
    hw_flag = ((women["no_education"] == 1) & (women["no_paid_mkt"] == 1)
               & (women["married"] == 1)).astype(float)
    sharehw = wmean(hw_flag, women["weight"])
    married_share = wmean(women["married"], women["weight"])

    hw = women[(women["no_education"] == 1) & (women["no_paid_mkt"] == 1)
               & (women["married"] == 1)].copy()
    n = len(hw)
    hhsize = wmean(hw["hhsize"], hw["weight"])
    hours = collapse_day_based(hw)
    return hours, n, hhsize, married_share, sharehw


# ------------------------------------------------------------------
# South Africa 2010
# ------------------------------------------------------------------
def build_sa2010():
    a = read_dta(DATA / "Table1/South Africa/2010/tus-2010-activities-v1.2.dta")
    p = read_dta(DATA / "Table1/South Africa/2010/tus-2010-person-v1.2.dta")
    allp = read_dta(DATA / "Table1/South Africa/2010/tus-2010-allpersons-v1.2.dta")

    a["id"] = a["UQNO"].astype(str) + a["PERSONNO"].astype(str)

    pm = p[["UQNO", "PERSONNO", "Q27Child18HH", "Q45SourceIncome", "Q52DayDiary"]].copy()
    a = a.merge(pm, on=["UQNO", "PERSONNO"], how="left")
    a["childinhouse"] = ((a["Q27Child18HH"] > 0) & (a["Q27Child18HH"] <= 17)).astype(int)

    code = a["Activity_code"].astype("Int64")
    act = pd.Series("Other", index=a.index, dtype="object")
    act = act.mask(code.isin([410, 250]), "cooking")
    act = act.mask(code.isin([236, 250]), "firewater")
    act = act.mask(code.isin([420, 210, 340]), "cleaning")
    act = act.mask(code == 430, "laundry")
    care_codes = [511, 512, 521, 522, 531, 532, 671, 672, 540, 550, 673,
                  580, 590, 561, 562]
    act = act.mask(code.isin(care_codes), "care")
    dom_codes = [460, 260, 440, 330, 480, 490, 450, 488, 588]
    act = act.mask(code.isin(dom_codes), "domestic")
    mkt_codes = [111, 112, 114, 115, 130, 180, 188, 190]
    act = act.mask(code.isin(mkt_codes), "MARKET")
    edu_codes = [710, 720, 780, 788, 790]
    act = act.mask(code.isin(edu_codes), "Education")
    a["activity"] = act

    a["minutes"] = a["Timeper"].astype(float)

    coll = a.groupby(["id", "activity"])["minutes"].sum().unstack(fill_value=0)
    coll.columns = [f"min_{c}" for c in coll.columns]
    coll = coll.reset_index()

    # person-level fields
    p["id"] = p["UQNO"].astype(str) + p["PERSONNO"].astype(str)
    daymap = {1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday",
              5: "Friday", 6: "Saturday", 7: "Sunday"}
    p["daydiary"] = p["Q52DayDiary"].map(daymap)
    act_person = a.drop_duplicates("id")[["id", "Q117Age", "Weight", "Q116Gender",
                                           "Q23MaritalStatus"]]
    pp = p[["id", "Q45SourceIncome"]].merge(act_person, on="id", how="outer")

    df = coll.merge(pp, on="id", how="left")
    df["day"] = p.set_index("id").reindex(df["id"])["Q52DayDiary"].values

    allp["id"] = allp["UQNO"].astype(str) + allp["PERSONNO"].astype(str)
    numpers = allp.groupby("UQNO").size().rename("numpers").reset_index()
    # match by UQNO of the id
    df["UQNO"] = df["id"].str.slice(0, 18)
    df = df.merge(numpers, on="UQNO", how="left")

    df["sex"] = df["Q116Gender"]
    df["age"] = df["Q117Age"]
    df["weight"] = df["Weight"]

    df["no_paid_mkt"] = ((df.get("min_MARKET", 0) == 0)
                         & (df["Q45SourceIncome"] != 3)
                         & (df["Q45SourceIncome"] != 6)).astype(float)
    df["no_education"] = (df.get("min_Education", 0) == 0).astype(float)
    df["married"] = df["Q23MaritalStatus"].isin([1, 2]).astype(float)

    for src, dst in [("min_cooking", "cooking"), ("min_firewater", "firewater"),
                      ("min_cleaning", "cleaning"), ("min_laundry", "laundry"),
                      ("min_care", "care"), ("min_domestic", "domestic")]:
        if src not in df.columns:
            df[dst] = 0.0
        else:
            df[dst] = df[src].astype(float) / 60.0

    women = df[(df["sex"] == 2) & (df["age"] >= 15) & (df["age"] <= 59)].copy()
    hw_flag = ((women["no_education"] == 1) & (women["no_paid_mkt"] == 1)
               & (women["married"] == 1)).astype(float)
    sharehw = wmean(hw_flag, women["weight"])
    married_share = wmean(women["married"], women["weight"])

    hw = women[(women["no_education"] == 1) & (women["no_paid_mkt"] == 1)
               & (women["married"] == 1)].copy()
    n = len(hw)
    hhsize = wmean(hw["numpers"], hw["weight"])
    hours = collapse_day_based(hw)
    return hours, n, hhsize, married_share, sharehw


# ------------------------------------------------------------------
# USA 1920s (precomputed)
# ------------------------------------------------------------------
def build_usa1920s():
    df = read_dta(DATA / "Table1/usa_total.dta")
    row = df[df["country"] == 1.0].iloc[0]
    hours = {
        "cooking": float(row["unpaid_hh_cooking1"]),
        "firewater": float(row["unpaid_hh_firewater"]) if not pd.isna(row["unpaid_hh_firewater"]) else 0.0,
        "cleaning": float(row["unpaid_hh_cleaning1"]),
        "laundry": float(row["unpaid_hh_laundry1"]),
        "care": float(row["unpaid_hh_care1"]),
        "domestic": float(row["unpaid_hh_domestic1"]),
    }
    return hours, int(row["n1"]), float(row["hhsize1"]), np.nan, np.nan


# ------------------------------------------------------------------
# Assemble table
# ------------------------------------------------------------------
PUBLISHED = {
    "USA1920s": dict(cooking=25.1, firewater=1.5, cleaning=7.9, laundry=11.5,
                      care=3.6, domestic=1.7, total=51.3, n=619, hhsize=4.3),
    "USA1965":  dict(cooking=11.5, firewater=0.0, cleaning=14.4, laundry=7.0,
                      care=10.0, domestic=10.4, total=53.3, n=377, hhsize=3.9,
                      hw_share=0.37, married=0.79),
    "USA2010":  dict(cooking=7.0, firewater=0.0, cleaning=8.9, laundry=3.45,
                      care=15.7, domestic=10.6, total=45.7, n=987, hhsize=3.9,
                      hw_share=0.18, married=0.54),
    "SA2000":   dict(cooking=16.5, firewater=1.9, cleaning=13.1, laundry=6.4,
                      care=8.1, domestic=2.6, total=48.5, n=1900, hhsize=5.0,
                      hw_share=0.32, married=0.45),
    "SA2010":   dict(cooking=17.0, firewater=1.1, cleaning=11.9, laundry=5.4,
                      care=7.2, domestic=3.1, total=45.7, n=2581, hhsize=4.7,
                      hw_share=0.16, married=0.38),
    "Morocco2011": dict(cooking=23.6, firewater=0.5, cleaning=6.5, laundry=4.7,
                          care=7.2, domestic=3.1, total=45.7, n=3354, hhsize=5.0,
                          hw_share=0.44, married=0.69),
}


def main():
    # USA 1920s
    u1920, n_u1920, hh_u1920, _, _ = build_usa1920s()
    # MTUS
    mtus_res, mtus_n, mtus_hh, mtus_married, mtus_sharehw = build_mtus()
    # SA 2000, 2010
    sa00_h, sa00_n, sa00_hh, sa00_m, sa00_sh = build_sa2000()
    sa10_h, sa10_n, sa10_hh, sa10_m, sa10_sh = build_sa2010()
    # Morocco
    mo_h, mo_n, mo_hh, mo_m, mo_sh = build_morocco()

    cols = {
        "USA1920s": dict(cooking=u1920["cooking"], firewater=u1920["firewater"],
                          cleaning=u1920["cleaning"], laundry=u1920["laundry"],
                          care=u1920["care"], domestic=u1920["domestic"],
                          n=n_u1920, hhsize=hh_u1920, hw_share=np.nan, married=np.nan),
        "USA1965": dict(**mtus_res["US1965"], n=mtus_n["US1965"],
                         hhsize=mtus_hh["US1965"], hw_share=mtus_sharehw["US1965"],
                         married=mtus_married["US1965"]),
        "USA2010": dict(**mtus_res["US2010"], n=mtus_n["US2010"],
                         hhsize=mtus_hh["US2010"], hw_share=mtus_sharehw["US2010"],
                         married=mtus_married["US2010"]),
        "SA2000": dict(**sa00_h, n=sa00_n, hhsize=sa00_hh,
                        hw_share=sa00_sh, married=sa00_m),
        "SA2010": dict(**sa10_h, n=sa10_n, hhsize=sa10_hh,
                        hw_share=sa10_sh, married=sa10_m),
        "Morocco2011": dict(**mo_h, n=mo_n, hhsize=mo_hh,
                             hw_share=mo_sh, married=mo_m),
    }
    for k, v in cols.items():
        v["total"] = sum(v[a] for a in ACTS)

    tbl = pd.DataFrame(cols).T
    tbl = tbl[["total", "cooking", "firewater", "cleaning", "laundry", "care",
                "domestic", "n", "hhsize", "hw_share", "married"]]
    tbl.to_csv(OUT / "table1.csv", index_label="country")

    print("\n=========  Table 1 — replicated  =========")
    print(tbl.round(2).to_string())

    # Comparison
    rows = []
    for c in cols:
        pub = PUBLISHED[c]
        row = {"country": c}
        for a in ["total", "cooking", "firewater", "cleaning", "laundry", "care",
                   "domestic", "n", "hhsize", "hw_share", "married"]:
            rep = cols[c].get(a, np.nan)
            pv = pub.get(a, np.nan)
            row[f"{a}_rep"] = rep
            row[f"{a}_pub"] = pv
            row[f"{a}_diff"] = (rep - pv) if (isinstance(rep, (int, float)) and isinstance(pv, (int, float))) else np.nan
        rows.append(row)
    comp = pd.DataFrame(rows)
    comp.to_csv(OUT / "table1_comparison.csv", index=False)

    print("\n=========  Table 1 — comparison to paper  =========")
    disp = comp.set_index("country")
    for a in ["total", "cooking", "firewater", "cleaning", "laundry", "care",
               "domestic", "n", "hhsize", "hw_share", "married"]:
        print(f"\n-- {a} --")
        print(disp[[f"{a}_rep", f"{a}_pub", f"{a}_diff"]].round(2).to_string())


if __name__ == "__main__":
    main()
