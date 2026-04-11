"""Shared paths and helpers for Cravino/Levchenko/Rojas (2021) replication."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
RAW = REPO / "140141-V1" / "data_macro" / "raw"
OUT = ROOT / "build"
OUT.mkdir(exist_ok=True)

EUKLEMS_XLSX = RAW / "EUKLEMS1970-2007.xlsx"
POP_XLSX = RAW / "Data_Extract_From_Population_estimates_and_projections.xlsx"
WDI_CTRL_XLSX = RAW / "WDI_macro_controls.xlsx"
WDI_SECT_XLS = RAW / "WDISectorData.xls"

# Countries used for Table 1 (from 02_euklems.do)
COUNTRIES = ["AUS", "AUT", "BEL", "CAN", "DNK", "ESP", "FIN", "FRA", "GER",
             "GRC", "IRL", "ITA", "JPN", "KOR", "LUX", "NLD", "PRT", "SWE",
             "UK", "USA"]
JP_KR = {"JPN", "KOR"}  # these sheets use a different cell range


def read_euklems_block(sheet: str, header_row: int):
    """Read 3 sector rows (Agr/Man/Ser) for years 1970-2007 from one sheet."""
    df = pd.read_excel(EUKLEMS_XLSX, sheet_name=sheet, header=header_row,
                       nrows=3, usecols="A:AN")
    df = df.rename(columns={df.columns[0]: "sector_name",
                            df.columns[1]: "code_col"})
    df = df.drop(columns=["code_col"])
    df = df.set_index("sector_name")
    df.columns = [int(c) for c in df.columns]
    df = df.T
    df.index.name = "year"
    df.columns = ["agr", "man", "ser"]
    return df.reset_index()


def load_euklems_va_hw() -> pd.DataFrame:
    """Return country-year panel of EUKLEMS VA and hours-worked shares."""
    out = []
    for ctry in COUNTRIES:
        hrow = 113 if ctry in JP_KR else 43
        va = read_euklems_block(f"VA {ctry}", hrow)
        va_tot = va[["agr", "man", "ser"]].sum(axis=1)
        va_out = pd.DataFrame({
            "code": ctry, "year": va["year"],
            "va_share_agr": va["agr"] / va_tot,
            "va_share_man": va["man"] / va_tot,
            "va_share_ser": va["ser"] / va_tot,
        })
        hw = read_euklems_block(f"H_EMP {ctry}", hrow)
        hw_tot = hw[["agr", "man", "ser"]].sum(axis=1)
        hw_out = pd.DataFrame({
            "code": ctry, "year": hw["year"],
            "hw_share_agr": hw["agr"] / hw_tot,
            "hw_share_man": hw["man"] / hw_tot,
            "hw_share_ser": hw["ser"] / hw_tot,
        })
        out.append(va_out.merge(hw_out, on=["code", "year"]))
    return pd.concat(out, ignore_index=True)


def load_wdi_population() -> pd.DataFrame:
    """Population shares by broad age group from WDI population projections."""
    df = pd.read_excel(POP_XLSX, sheet_name="Data")
    df = df.rename(columns={"Time": "year", "Country Code": "code",
                            "Series Name": "series", "Value": "value"})
    df = df[["year", "code", "series", "value"]].dropna(subset=["year"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    def age_group(s: str) -> int | None:
        if not isinstance(s, str):
            return None
        # Broad bands per 01_aux_macro.do
        young = ["00-04", "05-09", "10-14"]
        mid = ["15-19", "20-24", "25-29", "30-34", "35-39", "40-44",
               "45-49", "50-54", "55-59", "60-64"]
        old = ["65-69", "70-74", "75-79", "80+"]
        if any(b in s for b in young) and ("population" in s):
            return 0
        if any(b in s for b in mid) and ("population" in s):
            return 1
        if any(b in s for b in old) and ("population" in s):
            return 2
        return None

    df["grp"] = df["series"].apply(age_group)
    df = df.dropna(subset=["grp"])
    agg = (df.groupby(["code", "year", "grp"])["value"].sum()
             .unstack("grp")
             .rename(columns={0: "pop_014", 1: "pop_1564", 2: "pop_65plus"})
             .reset_index())
    agg["pop_tot"] = agg["pop_014"] + agg["pop_1564"] + agg["pop_65plus"]
    agg["share_65plus"] = agg["pop_65plus"] / agg["pop_tot"]
    return agg


def load_wdi_controls() -> pd.DataFrame:
    """Load WDI macro controls (trade, investment, gov, savings)."""
    df = pd.read_excel(WDI_CTRL_XLSX, sheet_name="Data")
    df = df.rename(columns={"Country Code": "code", "Time": "year"})
    # Find numeric columns by series name prefix
    rn = {}
    for c in df.columns:
        if isinstance(c, str):
            if c.startswith("Exports"): rn[c] = "exports"
            elif c.startswith("Imports"): rn[c] = "imports"
            elif c.startswith("Gross domestic savings"): rn[c] = "gross_savings"
            elif c.startswith("Gross capital formation"): rn[c] = "gross_capital"
            elif c.startswith("Trade (% of GDP)"): rn[c] = "trade"
            elif c.startswith("General government"): rn[c] = "government"
    df = df.rename(columns=rn)
    keep = ["code", "year"] + [v for v in
            ["exports", "imports", "gross_savings", "gross_capital",
             "trade", "government"] if v in df.columns]
    df = df[keep].copy()
    df = df.dropna(subset=["code"])
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    for v in keep[2:]:
        df[v] = pd.to_numeric(df[v], errors="coerce")
    # Stata recodes DEU->GER and GBR->UK
    df["code"] = df["code"].replace({"DEU": "GER", "GBR": "UK"})
    return df.dropna(subset=["year"]).reset_index(drop=True)


def load_maddison_gdp_pc() -> pd.DataFrame:
    """Maddison GDP per capita (1990 int. $) from WDISectorData.xls, sheet 'Maddison'."""
    # cellrange A5:AX225, firstrow → header on row index 4
    df = pd.read_excel(WDI_SECT_XLS, sheet_name="Maddison", header=4,
                       nrows=220)  # rows 5..224 ≈ 220 rows, drops 191/219 slice below
    df = df.rename(columns={df.columns[1]: "code"})
    df = df.drop(columns=[df.columns[0]])  # drop country_name (first col)
    # Stata drops rows 191-219 (0-indexed 190-218): these are aggregate rows
    # ("E7 GDP", "E15 GDP") that re-use the same country codes. Match Stata.
    df = df.reset_index(drop=True)
    df = df.drop(index=range(190, 219), errors="ignore").reset_index(drop=True)
    df = df.dropna(subset=["code"])
    df = df[df["code"].astype(str).str.strip() != ""].copy()
    df["code"] = df["code"].replace({"E15": "EU15"})
    # Reshape wide→long. All remaining columns are years.
    year_cols = [c for c in df.columns if c != "code"]
    long = df.melt(id_vars="code", value_vars=year_cols,
                   var_name="year", value_name="gdp_pc")
    long["year"] = pd.to_numeric(long["year"], errors="coerce")
    long["gdp_pc"] = pd.to_numeric(long["gdp_pc"], errors="coerce")
    long = long.dropna(subset=["year"])
    long["year"] = long["year"].astype(int)
    long = long[long["gdp_pc"] > 0].copy()
    long["log_gdp_pc"] = np.log(long["gdp_pc"])
    long["log_gdp_pc2"] = long["log_gdp_pc"] ** 2
    return long[["code", "year", "gdp_pc", "log_gdp_pc", "log_gdp_pc2"]]


def fmt_coef(b: float, se: float, p: float) -> str:
    stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
    return f"{b:7.3f}{stars:<3} ({se:.3f})"
