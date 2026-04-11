"""02_figures.py — replicate Figures 1-5 of Dinkelman & Ngai (2022)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import DATA, OUT, read_dta, AFRICA11, FIG2B_COUNTRIES


# ------------------------------------------------------------------
# Figure 1: Female employment shares vs GDP per capita (11 African countries)
# ------------------------------------------------------------------
def build_empshares():
    gdcc = read_dta(DATA / "Global/GDCC/asd_oct_2013_yl.dta")
    gdcc = gdcc[gdcc["Variable"].isin(["EMP", "EMP_M", "EMP_F"])].copy()
    # Drop ISICcode=="70" (Dwellings)
    gdcc = gdcc[gdcc["ISICcode"] != "70"].copy()

    year_cols = [c for c in gdcc.columns if c.startswith("_") and c[1:].isdigit()]
    long = gdcc.melt(
        id_vars=["Country", "Countryname", "Variable", "ISICcode", "Sectorname"],
        value_vars=year_cols, var_name="year", value_name="val"
    )
    long["year"] = long["year"].str[1:].astype(int)

    # EMP is thousands of persons → multiply by 1000
    long.loc[long["Variable"] == "EMP", "val"] = long.loc[long["Variable"] == "EMP", "val"] * 1000.0

    # totemp = EMP value for (Country, Sector, year), broadcast to all Variables
    emp_rows = long[long["Variable"] == "EMP"][["Country", "Sectorname", "year", "val"]]
    emp_rows = emp_rows.rename(columns={"val": "totemp"})
    long = long.merge(emp_rows, on=["Country", "Sectorname", "year"], how="left")

    # n_ = share * totemp for EMP_M/EMP_F, else = val
    long["n"] = np.where(
        long["Variable"].isin(["EMP_M", "EMP_F"]),
        long["val"] * long["totemp"],
        long["val"],
    )

    long = long.dropna(subset=["n"])
    long = long.drop(columns=["val", "totemp"])

    # Pivot wide to Sectorname
    wide = long.pivot_table(
        index=["Country", "Countryname", "Variable", "year"],
        columns="Sectorname", values="n", aggfunc="first"
    ).reset_index()
    wide.columns.name = None

    for s in ["Agriculture", "Business services", "Construction", "Government services",
              "Manufacturing", "Mining", "Personal services", "Total Economy",
              "Trade services", "Transport services", "Utilities"]:
        if s not in wide.columns:
            wide[s] = np.nan

    # SUM = Total Economy (row with Sectorname == 'Total Economy'). Stata egen
    # rsum treats missing as 0, so we do the same via fillna(0).
    sum_col = wide["Total Economy"]
    manuf = wide[["Mining", "Manufacturing", "Construction"]].fillna(0).sum(axis=1)
    serv = wide[["Business services", "Government services", "Personal services",
                  "Trade services", "Transport services", "Utilities"]].fillna(0).sum(axis=1)
    wide["share_agric"] = wide["Agriculture"] / sum_col
    wide["share_manuf"] = manuf / sum_col
    wide["share_services"] = serv / sum_col

    wide = wide.dropna(subset=["share_agric", "share_manuf", "share_services"]).copy()
    wide = wide.rename(columns={"Country": "countrycode"})

    # Merge PWT
    pwt = read_dta(DATA / "Global/PWT/pwt91.dta")
    pwt["rgdpe_pc"] = pwt["rgdpe"] / pwt["pop"]
    pwt["lnrgdpe_pc"] = np.log(pwt["rgdpe_pc"])
    pwt = pwt[["countrycode", "year", "rgdpe", "pop", "rgdpe_pc", "lnrgdpe_pc"]]

    merged = wide.merge(pwt, on=["countrycode", "year"], how="inner")
    return merged


def figure1(merged):
    sub = merged[
        (merged["year"] >= 1970) & (merged["year"] <= 2010)
        & (merged["countrycode"].isin(AFRICA11))
        & (merged["Variable"] == "EMP_F")
    ].copy()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for ax, (var, title) in zip(
        axes,
        [("share_agric", "Agriculture"),
         ("share_manuf", "Manufacturing"),
         ("share_services", "Services")],
    ):
        for i, cc in enumerate(AFRICA11):
            d = sub[sub["countrycode"] == cc]
            marker = "x" if i % 2 == 0 else "o"
            face = "none" if marker == "o" else None
            ax.scatter(d["rgdpe_pc"], d[var], marker=marker,
                       facecolors=face, edgecolors=f"C{i % 10}",
                       s=28, alpha=0.85, label=cc)
        ax.set_xscale("log")
        ax.set_ylim(0, 1)
        ax.set_xlabel("Real GDP per capita (log)")
        ax.set_title(f"Female emp. share: {title}")
    axes[0].set_ylabel("Female employment share")
    fig.suptitle("Figure 1 — Female employment shares by sector, 11 African countries (1970-2010)")
    fig.tight_layout()
    fig.savefig(OUT / "figure1_female_empshares.png", dpi=150)
    plt.close(fig)
    print("Fig 1 observations:", sub.shape[0], "unique countries:", sub["countrycode"].nunique())
    return sub


# ------------------------------------------------------------------
# Figure 2 — Panel A: FLFPR scatter (ILO 2017). Panel B: hours (Bridgman).
# ------------------------------------------------------------------
def figure2():
    ilo = read_dta(DATA / "Global/ILO/ilo_flfpr_countryyear_graph.dta")
    ilo17 = ilo[(ilo["year"] == 2017) & ilo["lngdppc"].notna() & ilo["flfpr"].notna()].copy()

    br = read_dta(DATA / "Bridgman_etal/hh_mkt_hours_flfpr_by_country_year.dta")
    br = br[br["sample"] == "female"]
    rows = []
    for country, year in FIG2B_COUNTRIES:
        r = br[(br["country"] == country) & (br["year"] == year)]
        if len(r):
            rr = r.iloc[0]
            rows.append({
                "label": f"{country}-{year}",
                "hh_hours": rr["hh_hours"],
                "mkt_hours": rr["mkt_hours"],
            })
    b = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A
    ax = axes[0]
    africa = ilo17[ilo17["ilo_africa"] == 1]
    not_africa = ilo17[ilo17["ilo_africa"] != 1]
    ax.scatter(not_africa["lngdppc"], not_africa["flfpr"], marker="o",
               facecolors="none", edgecolors="grey", s=22, alpha=0.6, label="Rest of world")
    ax.scatter(africa["lngdppc"], africa["flfpr"], marker="o",
               facecolors="none", edgecolors="navy", s=38, alpha=0.9, label="Africa")
    north = africa[africa["north"] == 1]
    ax.scatter(north["lngdppc"], north["flfpr"], marker="o",
               facecolors="none", edgecolors="orange", s=48, alpha=0.95, label="N. Africa")
    for _, row in africa.iterrows():
        ax.annotate(str(row["country"])[:12], (row["lngdppc"], row["flfpr"]),
                    fontsize=5, alpha=0.6, xytext=(2, 1), textcoords="offset points")
    ax.set_xlabel("ln(Real GDP per capita)")
    ax.set_ylabel("Female LFPR")
    ax.set_title("Panel A. FLFPR vs GDP pc (2017, ILO)")
    ax.set_xlim(6, 11.5)
    ax.set_ylim(0, 100)
    ax.legend(loc="lower right", fontsize=8)

    # Panel B
    ax = axes[1]
    x = np.arange(len(b))
    width = 0.38
    ax.bar(x - width / 2, b["hh_hours"], width, label="Home hours")
    ax.bar(x + width / 2, b["mkt_hours"], width, label="Market hours")
    ax.set_xticks(x)
    ax.set_xticklabels(b["label"], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Weekly hours (females 15-64)")
    ax.set_title("Panel B. Weekly hours (Bridgman et al 2018)")
    ax.legend(fontsize=8)

    fig.suptitle("Figure 2 — Female labor force participation and time use")
    fig.tight_layout()
    fig.savefig(OUT / "figure2_flfp_and_hours.png", dpi=150)
    plt.close(fig)
    print("Fig 2 Panel B (hours):")
    print(b.to_string(index=False))
    return b


# ------------------------------------------------------------------
# Figure 3 — Stacked bars, women's market work composition, NA vs SSA, 1991 vs 2017.
# ------------------------------------------------------------------
def figure3():
    wb_classes = pd.read_excel(DATA / "Figure4/WB country classes.xls",
                                header=4, usecols="C:I")
    wb_classes.columns = ["countryname", "countrycode", "drop1", "region",
                          "incgroup", "bankstatus", "lending"]
    wb_classes = wb_classes.drop(columns=["drop1"])
    wb_classes = wb_classes.dropna(subset=["region", "countryname"])
    wb_classes = wb_classes[wb_classes["countryname"] != "x"]

    wdi = pd.read_csv(DATA / "Figure4/WDI_familyfarmworkers_Data.csv")
    wdi = wdi.dropna(subset=["Series Name"]).copy()
    year_cols = [c for c in wdi.columns if c.endswith("]")]
    def y(c): return int(c.split(" ")[0])
    wdi_long = wdi.melt(
        id_vars=["Country Name", "Country Code", "Series Name", "Series Code"],
        value_vars=year_cols, var_name="yr", value_name="val",
    )
    wdi_long["year"] = wdi_long["yr"].map(y)
    wdi_long["val"] = pd.to_numeric(wdi_long["val"], errors="coerce")

    def series_to_type(s):
        if "Contributing family" in s: return "familywork"
        if "Wage and salaried" in s: return "wage"
        if s.startswith("Employers"): return "employers"
        if "Self-employed" in s: return "selfemploy"
        if "Employment to population" in s: return "emppop"
        return None
    wdi_long["worktype"] = wdi_long["Series Name"].map(series_to_type)
    wdi_long = wdi_long.dropna(subset=["worktype"])
    wide = wdi_long.pivot_table(
        index=["Country Name", "Country Code", "year"],
        columns="worktype", values="val", aggfunc="first"
    ).reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={"Country Name": "countryname",
                                 "Country Code": "countrycode"})

    merged = wide.merge(wb_classes[["countryname", "region", "incgroup"]],
                        on="countryname", how="left")
    merged = merged[merged["region"].isin(["Sub-Saharan Africa",
                                            "Middle East & North Africa"])].copy()
    # Drop ME countries
    drops = ["Bahrain", "Iran, Islamic Rep.", "Iraq", "Israel", "Jordan", "Kuwait",
             "Lebanon", "Malta", "Oman", "Qatar", "Saudi Arabia",
             "Syrian Arab Republic", "United Arab Emirates", "West Bank and Gaza",
             "Yemen, Rep.", "Libya", "Algeria"]
    merged = merged[~merged["countryname"].isin(drops)]
    merged = merged[merged["incgroup"] != "High income"]
    merged.loc[merged["region"] == "Middle East & North Africa", "region"] = "North Africa"

    # Convert % -> share
    for c in ["familywork", "employers", "selfemploy", "wage", "emppop"]:
        merged[c] = merged[c] / 100.0
    merged["ownaccount"] = merged["selfemploy"] - merged["familywork"] - merged["employers"]
    for c in ["familywork", "employers", "ownaccount", "wage"]:
        merged[f"pf_{c}"] = merged[c] * merged["emppop"]
    merged["totemp"] = merged[["pf_wage", "pf_employers", "pf_ownaccount",
                                "pf_familywork"]].sum(axis=1)
    merged["notinlf"] = 1 - merged["totemp"]

    def reg_year(region, year):
        d = merged[(merged["region"] == region) & (merged["year"] == year)]
        out = d[["pf_wage", "pf_employers", "pf_ownaccount", "pf_familywork",
                  "notinlf"]].mean() * 100
        return out

    categories = ["pf_wage", "pf_employers", "pf_ownaccount", "pf_familywork", "notinlf"]
    labels = ["Wage/salary", "Employer", "Own account",
              "Contributing family worker", "Not in labor force"]
    colors = ["#1f3a68", "#8b0000", "#d2691e", "#2e7d32", "#e8a899"]

    bars = {
        "SSA 1991": reg_year("Sub-Saharan Africa", 1991),
        "SSA 2017": reg_year("Sub-Saharan Africa", 2017),
        "NA 1991": reg_year("North Africa", 1991),
        "NA 2017": reg_year("North Africa", 2017),
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(bars))
    bottoms = np.zeros(len(bars))
    for cat, lab, color in zip(categories, labels, colors):
        vals = np.array([bars[k][cat] for k in bars.keys()])
        ax.bar(x, vals, bottom=bottoms, label=lab, color=color)
        bottoms += vals
    ax.set_xticks(x)
    ax.set_xticklabels(bars.keys())
    ax.set_ylabel("% of female population")
    ax.set_title("Figure 3 — Women's market-work composition, SSA vs N Africa")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(OUT / "figure3_womens_market_work.png", dpi=150)
    plt.close(fig)
    print("Fig 3 values:")
    for k, v in bars.items():
        print(f"  {k}: {dict(v.round(1))}")
    return bars


# ------------------------------------------------------------------
# Figure 4 — Home-production substitute jobs, 5 countries.
# ------------------------------------------------------------------
def figure4():
    df = pd.read_csv(DATA / "Figure5/Women marketization.csv", header=None,
                     names=["country", "domestic", "alljobs", "service"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(df))
    w = 0.38
    ax.bar(x - w / 2, df["alljobs"], w, label="Share of all jobs")
    ax.bar(x + w / 2, df["service"], w, label="Share of service jobs")
    ax.set_xticks(x); ax.set_xticklabels(df["country"], rotation=20)
    ax.set_ylabel("Percent")
    ax.set_ylim(0, 100)
    ax.set_title("Figure 4 — Home-production substitute jobs, women")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "figure4_home_substitute_jobs.png", dpi=150)
    plt.close(fig)
    print("Fig 4 values:")
    print(df.to_string(index=False))
    return df


# ------------------------------------------------------------------
# Figure 5 — in this paper the same marketization data is reused (there are only
# 5 figures total; the marketization chart is the "final" figure). We verify by
# plotting the domestic-workers share variant as a second look.
# ------------------------------------------------------------------
def figure5():
    df = pd.read_csv(DATA / "Figure5/Women marketization.csv", header=None,
                     names=["country", "domestic", "alljobs", "service"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(df["country"], df["domestic"], color="#8b0000")
    ax.set_ylabel("% of female jobs")
    ax.set_title("Figure 5 — Domestic workers as a share of female employment")
    ax.set_ylim(0, max(df["domestic"]) * 1.25)
    for i, v in enumerate(df["domestic"]):
        ax.text(i, v + 0.3, f"{v:.1f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "figure5_domestic_workers.png", dpi=150)
    plt.close(fig)
    print("Fig 5 values:", df[["country", "domestic"]].to_dict(orient="records"))


if __name__ == "__main__":
    print("\n>> Building GDCC/PWT employment shares")
    merged = build_empshares()
    print("\n>> Figure 1")
    figure1(merged)
    print("\n>> Figure 2")
    figure2()
    print("\n>> Figure 3")
    figure3()
    print("\n>> Figure 4")
    figure4()
    print("\n>> Figure 5")
    figure5()
    print("\nAll figures written to", OUT)
