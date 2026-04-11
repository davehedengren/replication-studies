"""Phase 2 step 1: build the macro country-year panel and save to parquet."""
from utils import (load_euklems_va_hw, load_wdi_population,
                   load_wdi_controls, load_maddison_gdp_pc, OUT)


def main():
    euk = load_euklems_va_hw()
    print(f"EUKLEMS VA+HW: {euk.shape}, countries={euk['code'].nunique()}")

    pop = load_wdi_population()
    print(f"WDI population: {pop.shape}, countries={pop['code'].nunique()}")

    wdi_ctrl = load_wdi_controls()
    print(f"WDI controls:   {wdi_ctrl.shape}")

    mad = load_maddison_gdp_pc()
    print(f"Maddison GDP:   {mad.shape}, countries={mad['code'].nunique()}")

    # Stata: merge EUKLEMS with age_wdi (renames UK→GBR temporarily in code)
    # age_wdi is keyed by code where WDI uses GBR. EUKLEMS uses UK.
    # Stata does: replace code = "GBR" if code == "UK"  then merge, then back to UK
    euk_m = euk.copy()
    euk_m["code"] = euk_m["code"].replace({"UK": "GBR"})
    pop_m = pop.rename(columns={})
    # Note: Stata also uses DEU→GER recoding in controls but WDI pop has DEU.
    # EUKLEMS uses GER. After UK->GBR, we still need DEU->GER match.
    pop_m["code"] = pop_m["code"].replace({"DEU": "GER"})
    merged = euk_m.merge(pop_m, on=["code", "year"], how="left")
    # Put code back to UK (matches EUKLEMS convention used in 11_macro.do)
    merged["code"] = merged["code"].replace({"GBR": "UK"})

    # Merge WDI controls (already recoded to GER/UK in loader)
    merged = merged.merge(wdi_ctrl, on=["code", "year"], how="left")

    # Merge Maddison GDP. Stata uses "GBR" key for this merge
    # (see 02_euklems.do line 104: replace code = "GBR" if code=="UK")
    # Inspect Maddison for codes
    mad_codes = set(mad["code"].unique())
    # Stata replaces UK→GBR before merging Maddison
    mm = merged.copy()
    mm["code"] = mm["code"].replace({"UK": "GBR"})
    # Some EUKLEMS labels (e.g. GER) may not match Maddison; try DEU
    if "GER" not in mad_codes and "DEU" in mad_codes:
        mm["code"] = mm["code"].replace({"GER": "DEU"})
    merged2 = mm.merge(mad, on=["code", "year"], how="left")
    # Keep only rows with GDP (Stata: keep if _merge==3)
    merged2 = merged2.dropna(subset=["gdp_pc"]).copy()
    # Put code labels back for consistency
    merged2["code"] = merged2["code"].replace({"GBR": "UK", "DEU": "GER"})

    out = merged2.reset_index(drop=True)
    path = OUT / "macro_panel.parquet"
    out.to_parquet(path)
    print(f"Saved {path} with shape {out.shape}")
    print(out[["code", "year"]].groupby("code").size())
    print("Sample:", out.head(3).to_string())


if __name__ == "__main__":
    main()
