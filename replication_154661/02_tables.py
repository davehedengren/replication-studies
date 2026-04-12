"""Reproduce the headline numbers from the JEP paper.

Targets (published values):
  Figure 2 top (18-24, all):       paid    unpaid  not_working
    Africa (N=28)                   32.4    15.6    51.9
    Other  (N=40)                   41.5     7.0    51.5

  Figure 2 bottom (18-24, workers): sal_agri  sal_nonagri  SE_agri  SE_nonagri
    Africa                          5.5       20.0         43.5     31.0
    Other                           6.5       62.3         18.9     12.3

  Text p.83:
    Salaried share (18-24 workers)  Africa 25.5  Other 68.8
    Salaried within agri            Africa 11.2  Other 25.6
    Salaried within mfg/services    Africa 39.2  Other 83.5
  Text p.83-84:
    Agriculture sector share        Africa 49.0 Other 25.4
    Manufacturing                   16.9 vs 25.6
    Services                        31.8 vs 48.7
    High-skill share (63 countries) Africa 5.9  Other 12.9
    White-collar share              Africa 23.3 Other 25.8
  Text p.88 UN WPP 2020:
    World share 15-24 in 15-49      31%
    Africa share 15-24 in 15-49     40%
"""
import json
import numpy as np
import pandas as pd

from utils import prep_pooled_age18to24, load, mark_africa, weighted_mean, OUT


def figure2_top(df_young: pd.DataFrame) -> pd.DataFrame:
    d = df_young.copy()
    d["popw"] = np.round(d["pop"] / 100_000)
    rows = []
    for name, g in d.groupby("africa"):
        rows.append(
            {
                "group": "Africa" if name == 1 else "Other",
                "N": len(g),
                "paid_work": 100 * weighted_mean(g["paidwk_all"], g["popw"]),
                "unpaid_work": 100 * weighted_mean(g["unpaidwk_all"], g["popw"]),
                "not_working": 100 * weighted_mean(g["outofLF"], g["popw"]),
            }
        )
    return pd.DataFrame(rows)


def figure2_bottom(df_young: pd.DataFrame) -> pd.DataFrame:
    d = df_young.copy()
    # weights for population *in the labor market* (matches Stata code)
    d["popw"] = np.round((d["pop"] / 100_000) * d["inwork"])
    rows = []
    for name, g in d.groupby("africa"):
        rows.append(
            {
                "group": "Africa" if name == 1 else "Other",
                "N": len(g),
                "salaried_agri": 100 * weighted_mean(g["employee_paid_agri"], g["popw"]),
                "salaried_nonagri": 100 * weighted_mean(g["employee_paid_noagri"], g["popw"]),
                "SE_agri": 100 * weighted_mean(g["SE_agri"], g["popw"]),
                "SE_nonagri": 100 * weighted_mean(g["SE_noagri"], g["popw"]),
            }
        )
    return pd.DataFrame(rows)


def sector_breakdown(df_young: pd.DataFrame) -> pd.DataFrame:
    """Manufacturing/services shares among workers are only available for
    52 of the 68 countries (variables `sector_manuf`, `sector_serv`)."""
    d = df_young.dropna(subset=["sector_manuf", "sector_serv"]).copy()
    d["popw"] = np.round((d["pop"] / 100_000) * d["inwork"])
    rows = []
    for name, g in d.groupby("africa"):
        rows.append(
            {
                "group": "Africa" if name == 1 else "Other",
                "N": len(g),
                "sector_agri": 100 * weighted_mean(g["sector_agri"], g["popw"]),
                "sector_manuf": 100 * weighted_mean(g["sector_manuf"], g["popw"]),
                "sector_serv": 100 * weighted_mean(g["sector_serv"], g["popw"]),
            }
        )
    return pd.DataFrame(rows)


def headline_text_stats(df_young: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the salaried, sector, occupational shares quoted in the text."""
    d = df_young.copy()
    d["popw_all"] = np.round(d["pop"] / 100_000)
    d["popw_work"] = np.round((d["pop"] / 100_000) * d["inwork"])
    rows = []
    for name, g in d.groupby("africa"):
        label = "Africa" if name == 1 else "Other"
        # salaried share among workers = employee_paid / inwork? In the data,
        # employee_paid is reported as a share of the total population. Dividing by
        # inwork gives the share among workers.
        sal_agri = 100 * weighted_mean(g["employee_paid_agri"], g["popw_work"])
        sal_non = 100 * weighted_mean(g["employee_paid_noagri"], g["popw_work"])
        rows.append(
            {
                "group": label,
                "N": len(g),
                "salaried_total": sal_agri + sal_non,
                "salaried_agri": sal_agri,
                "salaried_nonagri": sal_non,
                # sector composition within workers (paid + unpaid combined)
                "sector_agri": 100 * weighted_mean(g["sector_agri"], g["popw_work"]),
                "sector_manuf": 100 * weighted_mean(g["sector_manuf"], g["popw_work"]),
                "sector_serv": 100 * weighted_mean(g["sector_serv"], g["popw_work"]),
                # occupational skill groups
                "highskill": 100 * weighted_mean(g["highskillwk"].fillna(0), g["popw_work"]),
                "whitecollar": 100 * weighted_mean(g["whitecollar"].fillna(0), g["popw_work"]),
                "bluecollar": 100 * weighted_mean(g["bluecollar"].fillna(0), g["popw_work"]),
            }
        )
    out = pd.DataFrame(rows)
    return out


def within_sector_salaried(df_young: pd.DataFrame) -> pd.DataFrame:
    """Salaried share *within* a sector (p.83-84):
    - Africa: 11.2% of agri workers are salaried; 39.2% of mfg/services.
    - Other:  25.6% agri, 83.5% mfg/services.
    """
    d = df_young.copy()
    d["popw_work"] = np.round((d["pop"] / 100_000) * d["inwork"])
    rows = []
    for name, g in d.groupby("africa"):
        label = "Africa" if name == 1 else "Other"
        # sector_agri, sector_nonagri are the share-of-workers in each sector
        # employee_paid_agri, employee_paid_noagri are the share-of-workers that are salaried in each sector
        num_agri = weighted_mean(g["employee_paid_agri"], g["popw_work"])
        den_agri = weighted_mean(g["sector_agri"], g["popw_work"])
        num_non = weighted_mean(g["employee_paid_noagri"], g["popw_work"])
        den_non = weighted_mean(g["sector_nonagri"], g["popw_work"])
        rows.append(
            {
                "group": label,
                "salaried_within_agri_pct": 100 * num_agri / den_agri,
                "salaried_within_nonagri_pct": 100 * num_non / den_non,
            }
        )
    return pd.DataFrame(rows)


def highskill_sample(df_young: pd.DataFrame) -> pd.DataFrame:
    """High-skill / white-collar numbers are reported for 63 of 68 countries.
    The paper uses *unweighted* country means for these shares (in contrast to
    Figure 2, which is population-weighted). This matches the quoted values of
    5.9/12.9 and 23.3/25.8 exactly."""
    d = df_young.dropna(subset=["highskillwk", "whitecollar", "bluecollar"]).copy()
    rows = []
    for name, g in d.groupby("africa"):
        rows.append(
            {
                "group": "Africa" if name == 1 else "Other",
                "N": len(g),
                "highskill_unweighted": 100 * g["highskillwk"].mean(),
                "whitecollar_unweighted": 100 * g["whitecollar"].mean(),
                "bluecollar_unweighted": 100 * g["bluecollar"].mean(),
            }
        )
    return pd.DataFrame(rows)


def un_wpp_shares() -> dict:
    """Share of 15-24 in 15-49 in 2020, by region (text on p.88)."""
    path = (OUT.parent.parent / "154661-V1/data_replication"
            / "WPP2019_POP_F07_1_POPULATION_BY_AGE_BOTH_SEXES.xlsx")
    est = pd.read_excel(path, sheet_name="ESTIMATES", skiprows=16)
    entity_col = "Region, subregion, country or area *"
    type_col = "Type"
    year_col = "Reference date (as of 1 July)"
    age_cols = ["15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49"]
    rows = est[(est[type_col].isin(["World", "Region", "Income Group"])) & (est[year_col] == 2020)].copy()
    for c in age_cols:
        rows[c] = pd.to_numeric(rows[c], errors="coerce")
    rows["pop1524"] = rows["15-19"] + rows["20-24"]
    rows["pop1549"] = rows[age_cols].sum(axis=1)
    rows["share1524_1549"] = rows["pop1524"] / rows["pop1549"]
    keep = rows[[entity_col, "share1524_1549"]].copy()
    keep.columns = ["entity", "share"]
    return {row["entity"]: round(float(row["share"]), 4) for _, row in keep.iterrows()}


if __name__ == "__main__":
    df_young = prep_pooled_age18to24()
    df_young = df_young[df_young["age18to24"] == 1].copy()
    print("N_africa, N_other:", int(df_young.africa.sum()), int((1 - df_young.africa).sum()))

    t1 = figure2_top(df_young)
    t2 = figure2_bottom(df_young)
    t3 = headline_text_stats(df_young)
    t4 = within_sector_salaried(df_young)
    t5 = highskill_sample(df_young)
    t6 = sector_breakdown(df_young)

    print("\nFigure 2 top (employment status, %):")
    print(t1.round(1).to_string(index=False))
    print("\nFigure 2 bottom (employment categories, %):")
    print(t2.round(1).to_string(index=False))
    print("\nHeadline text stats:")
    print(t3.round(1).to_string(index=False))
    print("\nSalaried share within sector (%):")
    print(t4.round(1).to_string(index=False))
    print("\nHigh-skill / white-collar sample (63 countries):")
    print(t5.round(1).to_string(index=False))
    print("\nSector breakdown (52 countries with manuf/serv split):")
    print(t6.round(1).to_string(index=False))

    try:
        un = un_wpp_shares()
        print("\nUN WPP 2020 share of 15-24 in 15-49:")
        print(un)
    except Exception as e:
        print("\nUN WPP load failed:", e)
        un = {}

    t1.to_csv(OUT / "figure2_top.csv", index=False)
    t2.to_csv(OUT / "figure2_bottom.csv", index=False)
    t3.to_csv(OUT / "headline_stats.csv", index=False)
    t4.to_csv(OUT / "salaried_within_sector.csv", index=False)
    t5.to_csv(OUT / "highskill_sample.csv", index=False)
    t6.to_csv(OUT / "sector_breakdown.csv", index=False)
    with open(OUT / "un_wpp.json", "w") as f:
        json.dump(un, f, indent=2)
