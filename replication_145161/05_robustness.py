"""05_robustness.py — robustness checks for Dinkelman & Ngai (2022) replication."""
import numpy as np
import pandas as pd

from utils import DATA, OUT, read_dta, wmean, AFRICA11

ACTS = ["cooking", "firewater", "cleaning", "laundry", "care", "domestic"]


def collapse_day_based(df):
    out = {}
    if "day" not in df.columns or df.empty:
        return {a: np.nan for a in ACTS}
    for v in ACTS:
        if v not in df.columns:
            out[v] = 0.0
            continue
        per_day = []
        for _, d in df.groupby("day"):
            per_day.append(wmean(d[v], d["weight"]))
        out[v] = float(np.nansum(per_day))
    return out


# Use Morocco as a lab because it's fully in-package and fast.
def morocco_df():
    c = read_dta(DATA / "Table1/Morocco/carnet_adulte_ENET2012.dta")
    men = read_dta(DATA / "Table1/Morocco/menage_ENET.dta")[["ïn_mãnage", "taille_mãnage"]]
    df = c.merge(men, on="ïn_mãnage", how="left")
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
    return df


def fig1_shares():
    """Compute Fig1 share means for each country from the replication logic."""
    from importlib import import_module
    mod = import_module("02_figures") if False else None
    # Re-run build:
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    fig_module = __import__("02_figures")
    return fig_module.build_empshares()


def main():
    # -- Checks using Morocco (most detailed) --
    df = morocco_df()
    out = []
    def add(name, sub):
        hw = sub[(sub["sex"] == 2) & (sub["age"].between(15, 59))
                  & (sub["no_education"] == 1) & (sub["no_paid_mkt"] == 1)
                  & (sub["married"] == 1)].copy()
        hours = collapse_day_based(hw)
        total = sum(v for v in hours.values() if not np.isnan(v))
        out.append({"check": name, "n_hw": len(hw), "total": total, **hours})

    add("baseline (age 15-59)", df)
    add("age 18-64", df.assign(age=df["age"]).query("age>=18 & age<=64"))
    add("age 25-54", df.query("age>=25 & age<=54"))
    add("include unmarried women", df.assign(married=1.0))
    add("drop no-education restriction", df.assign(no_education=1.0))
    add("drop no-paid-mkt restriction", df.assign(no_paid_mkt=1.0))
    add("uniform weights", df.assign(weight=1.0))
    add("drop tranche 1 days", df[df["day"] != 1])
    add("weekdays only (d1-5)", df[df["day"].between(1, 5)])
    add("weekends only (d6-7)", df[df["day"].isin([6, 7])])
    # winsorize total minute components
    df_w = df.copy()
    for v in ACTS:
        q = df_w[v].quantile([0.01, 0.99])
        df_w[v] = df_w[v].clip(lower=q.iloc[0], upper=q.iloc[1])
    add("winsorize 1/99", df_w)

    morocco_results = pd.DataFrame(out).set_index("check")
    print("\n## Morocco robustness ##")
    print(morocco_results.round(2).to_string())
    morocco_results.to_csv(OUT / "robustness_morocco.csv")

    # -- Fig 1: drop individual countries (check single-country influence) --
    merged = fig1_shares()
    sub = merged[(merged["year"] >= 1970) & (merged["year"] <= 2010)
                  & merged["countrycode"].isin(AFRICA11)
                  & (merged["Variable"] == "EMP_F")]
    agg = (sub.groupby("countrycode")[["share_agric", "share_manuf", "share_services"]]
           .mean().round(3))
    print("\n## Fig 1 per-country mean female shares ##")
    print(agg.to_string())
    agg.to_csv(OUT / "robustness_fig1_per_country.csv")

    # drop-one averages
    drop_rows = []
    for cc in AFRICA11:
        s2 = sub[sub["countrycode"] != cc]
        drop_rows.append({
            "dropped": cc,
            "mean_agric": s2["share_agric"].mean(),
            "mean_manuf": s2["share_manuf"].mean(),
            "mean_serv": s2["share_services"].mean(),
        })
    drop_df = pd.DataFrame(drop_rows).set_index("dropped")
    print("\n## Fig 1 drop-one leave-out means ##")
    print(drop_df.round(3).to_string())
    drop_df.to_csv(OUT / "robustness_fig1_drop_one.csv")

    # -- Bridgman Fig 2B: alternative countries --
    br = read_dta(DATA / "Bridgman_etal/hh_mkt_hours_flfpr_by_country_year.dta")
    br_fem = br[br["sample"] == "female"].dropna(subset=["hh_hours"])
    print("\n## Bridgman female hh_hours all observations ##")
    brs = br_fem[["country", "year", "hh_hours", "mkt_hours", "flfpr", "rgdpe_pc"]]
    print(brs.to_string())
    brs.to_csv(OUT / "robustness_bridgman_all_obs.csv", index=False)


if __name__ == "__main__":
    main()
