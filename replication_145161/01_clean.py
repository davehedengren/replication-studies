"""01_clean.py — load each dataset used in Dinkelman & Ngai (2022) and sanity-check."""
from utils import DATA, read_dta
import pandas as pd


def describe(name, df):
    print(f"\n=== {name} ===")
    print("shape:", df.shape)
    print("head:")
    print(df.head(3))


def main():
    # GDCC
    gdcc = read_dta(DATA / "Global/GDCC/asd_oct_2013_yl.dta")
    describe("GDCC (Timmer et al 2015)", gdcc)
    print("Variables:", gdcc["Variable"].unique().tolist())
    print("ISICs:", sorted(gdcc["ISICcode"].unique().tolist()))

    # PWT
    pwt = read_dta(DATA / "Global/PWT/pwt91.dta")
    describe("PWT 9.1", pwt[["countrycode", "country", "year", "rgdpe", "pop"]])
    print("years:", pwt["year"].min(), "-", pwt["year"].max())

    # ILO pre-made
    ilo_g = read_dta(DATA / "Global/ILO/ilo_flfpr_countryyear_graph.dta")
    describe("ILO FLFPR graph", ilo_g)
    print("africa 2017 rows:", ((ilo_g["ilo_africa"] == 1) & (ilo_g["year"] == 2017)).sum())

    # Bridgman
    br = read_dta(DATA / "Bridgman_etal/hh_mkt_hours_flfpr_by_country_year.dta")
    describe("Bridgman et al 2018", br)

    # WDI
    wdi = pd.read_csv(DATA / "Figure4/WDI_familyfarmworkers_Data.csv")
    describe("WDI family workers", wdi)
    print("Series:", wdi["Series Name"].dropna().unique().tolist())

    # Women marketization
    wm = pd.read_csv(DATA / "Figure5/Women marketization.csv", header=None,
                     names=["country", "domestic", "alljobs", "service"])
    describe("Women marketization", wm)

    # Table 1 data
    usa = read_dta(DATA / "Table1/usa_total.dta")
    describe("USA total (precomputed)", usa)

    mtus = read_dta(DATA / "Table1/MTUS/mtus_00004.dta")
    mtus_sub = mtus[mtus["sample"].isin(["US1965", "US2010"])]
    describe("MTUS US1965+US2010", mtus_sub[["sample", "sex", "age", "cohab",
                                              "empstat", "student", "hhldsize",
                                              "cooking", "cleaning", "laundry",
                                              "childcare", "hhmanagement", "propwt", "day"]])

    mor_c = read_dta(DATA / "Table1/Morocco/carnet_adulte_ENET2012.dta")
    describe("Morocco carnet", mor_c[["sexe", "age", "etat_matrimonial", "type_activite",
                                       "coef_individu", "dureeg51", "dureeg52"]])
    mor_m = read_dta(DATA / "Table1/Morocco/menage_ENET.dta")
    describe("Morocco menage", mor_m[["taille_mãnage"]])

    sa00 = read_dta(DATA / "Table1/South Africa/2000/tus-2000-activall-v1.dta")
    describe("SA 2000 activities", sa00[["uqnr", "person", "timeper", "min", "dmin"]])
    sa00p = read_dta(DATA / "Table1/South Africa/2000/tus-2000-person-v1.dta")
    describe("SA 2000 person", sa00p[["uqnr", "gender", "age", "perwgt", "daydiary",
                                       "mar1tals", "sourcein", "child18h"]])
    sa00m = read_dta(DATA / "Table1/South Africa/2000/tus-2000-member-v1.dta")
    describe("SA 2000 member", sa00m.iloc[:, :4])

    sa10a = read_dta(DATA / "Table1/South Africa/2010/tus-2010-activities-v1.2.dta")
    describe("SA 2010 activities", sa10a[["UQNO", "PERSONNO", "Q116Gender", "Q117Age",
                                           "Q23MaritalStatus", "Activity_code", "Timeper", "Weight"]])
    sa10p = read_dta(DATA / "Table1/South Africa/2010/tus-2010-person-v1.2.dta")
    describe("SA 2010 person", sa10p[["UQNO", "PERSONNO", "Q27Child18HH",
                                       "Q45SourceIncome", "Q52DayDiary"]])


if __name__ == "__main__":
    main()
