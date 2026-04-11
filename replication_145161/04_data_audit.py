"""04_data_audit.py — coverage, distributions, missingness, duplicates."""
import numpy as np
import pandas as pd

from utils import DATA, OUT, read_dta, AFRICA11


def audit_gdcc():
    print("\n### GDCC audit ###")
    df = read_dta(DATA / "Global/GDCC/asd_oct_2013_yl.dta")
    print("rows:", len(df))
    print("countries:", df["Country"].nunique())
    print("African 11 coverage:")
    for cc in AFRICA11:
        n = (df["Country"] == cc).sum()
        print(f"  {cc}: {n} rows")
    # per-country years present
    year_cols = [c for c in df.columns if c.startswith("_") and c[1:].isdigit()]
    f_rows = df[(df["Variable"] == "EMP_F") & (df["ISICcode"] == "TOT")]
    print("\nEMP_F Total Economy coverage by country:")
    for _, row in f_rows.iterrows():
        cc = row["Country"]
        if cc in AFRICA11:
            present = [c[1:] for c in year_cols if not pd.isna(row[c])]
            if present:
                print(f"  {cc}: {present[0]}-{present[-1]} ({len(present)} years)")


def audit_pwt():
    print("\n### PWT 9.1 audit ###")
    df = read_dta(DATA / "Global/PWT/pwt91.dta")
    print("rows:", len(df), "countries:", df["countrycode"].nunique(),
          "years:", df["year"].min(), "-", df["year"].max())
    missing = df[["rgdpe", "pop"]].isna().sum()
    print("missing:", missing.to_dict())
    dup = df.duplicated(subset=["countrycode", "year"]).sum()
    print("duplicates (ctry,year):", dup)


def audit_bridgman():
    print("\n### Bridgman audit ###")
    df = read_dta(DATA / "Bridgman_etal/hh_mkt_hours_flfpr_by_country_year.dta")
    print("rows:", len(df))
    print("countries:", df["country"].nunique())
    print("samples:", df["sample"].unique())
    print("missing hh_hours:", df["hh_hours"].isna().sum())
    print("missing mkt_hours:", df["mkt_hours"].isna().sum())
    fem = df[df["sample"] == "female"]
    print("female rows with non-NaN hh_hours:", fem["hh_hours"].notna().sum())


def audit_wdi():
    print("\n### WDI Figure 3 audit ###")
    df = pd.read_csv(DATA / "Figure4/WDI_familyfarmworkers_Data.csv")
    print("rows:", len(df), "countries:", df["Country Code"].nunique())
    print("series:", df["Series Name"].dropna().unique())


def audit_mtus():
    print("\n### MTUS audit ###")
    m = read_dta(DATA / "Table1/MTUS/mtus_00004.dta")
    sub = m[m["sample"].isin(["US1965", "US2010"])]
    print(sub.groupby("sample").size())
    print("cohab distribution:")
    print(sub.groupby("sample")["cohab"].value_counts(dropna=False))
    for s in ["US1965", "US2010"]:
        d = sub[sub["sample"] == s]
        print(f"\n{s}: women 15-59 =",
              ((d["sex"] == 2) & (d["age"].between(15, 59))).sum())
    print("empstat dist (US2010):")
    print(sub[sub["sample"] == "US2010"]["empstat"].value_counts(dropna=False))


def audit_morocco():
    print("\n### Morocco audit ###")
    c = read_dta(DATA / "Table1/Morocco/carnet_adulte_ENET2012.dta")
    print("carnet rows:", len(c), "unique households:", c["ïn_mãnage"].nunique())
    print("sexe distribution:", c["sexe"].value_counts().to_dict())
    print("etat_matrimonial:", c["etat_matrimonial"].value_counts().to_dict())
    print("type_activite:", c["type_activite"].value_counts().to_dict())
    w = c["coef_individu"].astype(str).str.replace(",", ".").astype(float)
    print(f"weight: mean={w.mean():.2f} min={w.min():.2f} max={w.max():.2f}")
    # Check total minutes sums
    dg = [f"dureeg{i}" for i in range(10)]
    s = c[dg].astype(float).sum(axis=1)
    print("sum of dureeg0..9: min/mean/max =",
          s.min(), s.mean(), s.max(), "(expect 1440)")


def audit_sa2000():
    print("\n### SA 2000 audit ###")
    a = read_dta(DATA / "Table1/South Africa/2000/tus-2000-activall-v1.dta")
    p = read_dta(DATA / "Table1/South Africa/2000/tus-2000-person-v1.dta")
    m = read_dta(DATA / "Table1/South Africa/2000/tus-2000-member-v1.dta")
    print("activities:", a.shape, "persons:", p.shape, "members:", m.shape)
    print("person uqnr distinct:", p["uqnr"].nunique())
    print("mar1tals dist:", p["mar1tals"].value_counts(dropna=False).to_dict())
    print("gender dist:", p["gender"].value_counts(dropna=False).to_dict())
    print("sourcein top 5:")
    print(p["sourcein"].value_counts().head())


def audit_sa2010():
    print("\n### SA 2010 audit ###")
    a = read_dta(DATA / "Table1/South Africa/2010/tus-2010-activities-v1.2.dta")
    p = read_dta(DATA / "Table1/South Africa/2010/tus-2010-person-v1.2.dta")
    allp = read_dta(DATA / "Table1/South Africa/2010/tus-2010-allpersons-v1.2.dta")
    print("activities:", a.shape, "persons:", p.shape, "allpersons:", allp.shape)
    print("marital dist:", p["Q23MaritalStatus"].value_counts(dropna=False).to_dict())
    print("gender dist:", p["Q116Gender"].value_counts(dropna=False).to_dict())
    # check totals
    t = a.groupby(["UQNO", "PERSONNO"])["Timeper"].sum()
    print("total minutes per person diary: min/mean/max =",
          t.min(), t.mean(), t.max())


def main():
    audit_gdcc()
    audit_pwt()
    audit_bridgman()
    audit_wdi()
    audit_mtus()
    audit_morocco()
    audit_sa2000()
    audit_sa2010()


if __name__ == "__main__":
    main()
