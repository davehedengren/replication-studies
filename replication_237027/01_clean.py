"""Port of 01_build_data.do — builds country-level dataset from raw GRD/VDEM/WB/ISORA.

Verifies that the rebuilt dataset matches the provided final df_ISORA_GRD_VDEM_WB.dta
to high precision.
"""
import numpy as np
import pandas as pd
from utils import RAW, OUT, FINAL_DTA, zscore


def build_grd():
    df = pd.read_csv(RAW / "UNUWIDERGRD_2023.csv")
    df = df[["iso", "year", "tax_inc_sc", "tax_ex_sc", "resourcetaxes"]]
    df = df[df["year"].between(2018, 2021)]
    df = df.rename(columns={"iso": "iso_code"})
    return df.groupby("iso_code", as_index=False)[["tax_inc_sc", "tax_ex_sc", "resourcetaxes"]].mean()


def build_vdem():
    df = pd.read_csv(RAW / "VDEM_short.csv")
    df = df[df["year"].between(2018, 2021)].rename(columns={
        "country_text_id": "iso_code",
        "v2clrspct_osp": "vdem_rigorous_impartial",
        "v2stcritrecadm_osp": "vdem_meritocratic_recruitment",
        "v2x_jucon": "vdem_executive_judicial",
        "v2xlg_legcon": "vdem_executive_legislative",
    })
    df["vdem_executive_legislative"] = pd.to_numeric(df["vdem_executive_legislative"], errors="coerce")
    cols = ["vdem_rigorous_impartial", "vdem_meritocratic_recruitment",
            "vdem_executive_judicial", "vdem_executive_legislative"]
    df = df.groupby(["iso_code", "year"], as_index=False)[cols].mean()
    df = df.groupby("iso_code", as_index=False)[cols].mean()
    for v in ["vdem_executive_judicial", "vdem_executive_legislative"]:
        df[v + "_z"] = zscore(df[v])
    df["vdem_executive"] = df[["vdem_executive_judicial_z", "vdem_executive_legislative_z"]].mean(axis=1)
    return df[["iso_code", "vdem_rigorous_impartial", "vdem_meritocratic_recruitment", "vdem_executive"]]


def build_wb_gdp():
    df = pd.read_csv(RAW / "worldbank_NY.GDP.PCAP.KD_20240808.csv")
    # Stata renamed numeric year columns to v1..vN; v63..v66 = 2018..2021
    df["gdp_pc"] = df[["2018", "2019", "2020", "2021"]].mean(axis=1)
    return df[["iso_code", "gdp_pc"]]


def _read_isora_sheet(file: str, sheet: str, var: str) -> pd.DataFrame:
    df = pd.read_excel(RAW / f"{file}.xlsx", sheet_name=sheet)
    keep = ["country_isora"] + [c for c in df.columns if c.startswith(var)]
    df = df[keep]
    df = df[df["country_isora"].notna()]
    df = df[df["country_isora"] != "Formula"]
    df = df[~df["country_isora"].astype(str).str.contains("Please see the tables referred", na=False)]
    long = df.melt(id_vars="country_isora", var_name="yvar", value_name=var)
    long["year"] = long["yvar"].str.extract(r"(\d{4})$").astype(int)
    long = long.drop(columns="yvar")
    long[var] = pd.to_numeric(long[var].replace("D", np.nan), errors="coerce")
    return long


def build_isora():
    cc = pd.read_excel(RAW / "countrycodes.xlsx")[
        ["country_isora", "iso_code", "income_group_wb"]
    ]
    years = pd.DataFrame({"year": [2018, 2019, 2020, 2021]})
    df = cc.merge(years, how="cross")

    # (var as it appears in the Excel column prefix, file, sheet, final column name)
    spec = [
        ("laborforce_per_FTE", "ISORA_14_Derived_indicators_Revenue_and_r", "D5_resource ratios_FTE", "laborforce_per_FTE"),
        ("population",         "ISORA_14_Derived_indicators_Revenue_and_r", "population_laborforce", "population"),
        ("staff_pct_master",   "ISORA_15_Derived_indicators_Staff",          "D10_academic_qualifications", "staff_pct_master"),
        ("staff_pct_20_more",  "ISORA_15_Derived_indicators_Staff",          "D14_length_more_10", "staff_pct_20_more"),
        ("e_payments_pct_number", "ISORA_12_Stakeholder_interactions_Return_", "A88_electronic_payments", "electronic_payments_pct_number"),
        ("e_payments_pct_value",  "ISORA_12_Stakeholder_interactions_Return_", "A88_electronic_payments", "electronic_payments_pct_value"),
    ]
    for raw_var, file, sheet, final_name in spec:
        sub = _read_isora_sheet(file, sheet, raw_var)
        if raw_var != final_name:
            sub = sub.rename(columns={raw_var: final_name})
        df = df.merge(sub, on=["country_isora", "year"], how="left")

    grp = df.groupby(["country_isora", "iso_code", "income_group_wb"], as_index=False, dropna=False).agg({
        "staff_pct_master": "mean",
        "staff_pct_20_more": "mean",
        "laborforce_per_FTE": "mean",
        "population": "mean",
        "electronic_payments_pct_number": "mean",
        "electronic_payments_pct_value": "mean",
    })

    # NOTE: z-scores in Stata are computed on the FULL ISORA roster (174 countries),
    # before the population / resource / tax filter that drops to 121 countries.
    # We do the same here so staff_index matches Stata exactly.
    for v in ["electronic_payments_pct_number", "staff_pct_20_more", "staff_pct_master"]:
        grp[v + "_z"] = zscore(grp[v])
    grp["z_score"] = grp[[
        "electronic_payments_pct_number_z", "staff_pct_20_more_z", "staff_pct_master_z"
    ]].mean(axis=1)
    grp["staff_index"] = grp[["staff_pct_20_more_z", "staff_pct_master_z"]].mean(axis=1)
    grp["fte_per_laborforce"] = 1.0 / grp["laborforce_per_FTE"]
    return grp


def main():
    print("Building GRD...")
    grd = build_grd()
    print("Building VDEM...")
    vdem = build_vdem()
    print("Building WB GDP...")
    wb = build_wb_gdp()
    print("Building ISORA...")
    isora = build_isora()

    # Save intermediate ISORA snapshot for the data audit script
    isora.to_parquet(OUT / "df_isora_python.parquet", index=False)

    print("Merging ISORA + GRD (left, drop unmatched)...")
    df = isora.merge(grd, on="iso_code", how="left", indicator=True)
    df = df[df["_merge"] == "both"].drop(columns="_merge")

    df["resource_share"] = df["resourcetaxes"] / df["tax_ex_sc"]
    df = df[(df["population"] > 1e6) & df["population"].notna()]
    df = df[(df["resource_share"] <= 1/3) | df["resource_share"].isna()]
    df = df[df["tax_ex_sc"].notna()]
    df = df.drop(columns="resource_share")

    df = df.merge(vdem, on="iso_code", how="inner")
    df = df.merge(wb, on="iso_code", how="inner")

    out = OUT / "df_ISORA_GRD_VDEM_WB_python.parquet"
    df.to_parquet(out, index=False)
    print(f"Wrote {out}  shape={df.shape}")

    # Compare to Stata final
    stata = pd.read_stata(FINAL_DTA, convert_categoricals=False)
    print(f"Stata final shape: {stata.shape}")
    print(f"Python      shape: {df.shape}")
    common = sorted(set(stata.columns) & set(df.columns))
    s_iso = set(stata.iso_code)
    p_iso = set(df.iso_code)
    print(f"ISO match: stata={len(s_iso)}, python={len(p_iso)}, both={len(s_iso & p_iso)}, only_stata={s_iso-p_iso}, only_python={p_iso-s_iso}")

    merged = stata[["iso_code"]].merge(df, on="iso_code", suffixes=("_s", "_p"), how="inner")
    merged = stata.merge(df, on="iso_code", suffixes=("_s", "_p"))
    print(f"\nMax abs diff per numeric column (where both non-null):")
    rows = []
    for c in common:
        if c == "iso_code":
            continue
        a = merged[c + "_s"]
        b = merged[c + "_p"]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            mask = a.notna() & b.notna()
            if mask.sum() == 0:
                d = np.nan
            else:
                d = float(np.max(np.abs(a[mask] - b[mask])))
            rows.append((c, mask.sum(), d))
    diffs = pd.DataFrame(rows, columns=["var", "n", "max_abs_diff"]).sort_values("max_abs_diff", ascending=False)
    print(diffs.to_string(index=False))


if __name__ == "__main__":
    main()
