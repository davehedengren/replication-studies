"""Phase 3: data audit of the main analysis datasets used in Tables 1-6."""

import numpy as np
import pandas as pd

from utils import load, OUT


def main():
    out = []

    def log(msg=""):
        print(msg)
        out.append(str(msg))

    log("======================================================================")
    log(" DATA AUDIT — Aman-Rana & Minaudier, Spillovers in State Capacity")
    log("======================================================================")

    # ----- 1. Tax_District_Yr (main panel) -----
    tax = load("Tax_District_Yr.dta")
    log("\n[1] Tax_District_Yr.dta — main TWFE sample")
    log(f"  rows={len(tax)}  districts={tax['district'].nunique()}")
    log(f"  fin_yr range: {int(tax['fin_yr'].min())}-{int(tax['fin_yr'].max())}")
    log(f"  expected balanced size: 36 districts x 8 years = 288")
    log(f"  observed: {len(tax)}  → unbalanced (gap rows = {288-len(tax)})")
    log("  obs per fin_yr:")
    for y, n in tax.groupby("fin_yr").size().items():
        log(f"    {int(y)}: {n}")
    log("  obs per district (panel balance):")
    pc = tax.groupby("district").size()
    log(f"    min={pc.min()}, max={pc.max()}, mean={pc.mean():.1f}")
    log(f"    districts with <8 years: {(pc<8).sum()} of {len(pc)}")

    # tax_acres distribution
    log("\n  tax_acres distribution:")
    log(tax["tax_acres"].describe().to_string())
    log(f"  share of zero/near-zero (<0.01) tax_acres: "
        f"{(tax['tax_acres']<0.01).mean()*100:.1f}%")
    log(f"  IQR-based outliers (>Q3+3*IQR): "
        f"{(tax['tax_acres'] > tax['tax_acres'].quantile(0.75) + 3*(tax['tax_acres'].quantile(0.75)-tax['tax_acres'].quantile(0.25))).sum()}")
    log(f"  top 5 tax_acres: {sorted(tax['tax_acres'].nlargest(5).round(2).tolist())}")

    # treatment timing
    log("\n  digitization assignment:")
    cross = tax.groupby(["fin_yr", "phase", "digitization"]).size().unstack(fill_value=0)
    log(cross.to_string())
    log("  → Phase 1 first treated FY2012; phase 2 first treated FY2012/2013; "
        "phase 3 never treated in sample.")

    # ----- 2. Stacked dataset -----
    stk = load("Tax_District_Yr_stacked.dta")
    log("\n[2] Tax_District_Yr_stacked.dta — stacked DID")
    log(f"  rows={len(stk)}  events={sorted(stk['event_h'].unique())}")
    log(f"  per event: {stk.groupby('event_h').size().to_dict()}")

    # ----- 3. Reported cultivated area -----
    rc = load("reportedcultivated.dta")
    log("\n[3] reportedcultivated.dta")
    log(f"  rows={len(rc)}  years={sorted(rc['fin_yr'].unique())}")
    log(f"  Actual_Area000Acres mean={rc['Actual_Area000Acres'].mean():.1f}, "
        f"median={rc['Actual_Area000Acres'].median():.1f}")
    log(f"  ln_area range: [{rc['ln_area'].min():.2f}, {rc['ln_area'].max():.2f}]")

    # ----- 4. NDVI -----
    n = load("NDVI_2006_2014.dta")
    log("\n[4] NDVI_2006_2014.dta")
    log(f"  rows={len(n)}  districts={n['district'].nunique()}  expected balanced 36*8=288")
    log(f"  NDVI_veg in [0,1]? min={n['NDVI_veg'].min():.3f}, "
        f"max={n['NDVI_veg'].max():.3f}, mean={n['NDVI_veg'].mean():.3f}")

    # ----- 5. HIES -----
    h = load("hh_level_data_hies.dta")
    log("\n[5] hh_level_data_hies.dta — household profit")
    log(f"  rows={len(h)}  cultivating sample={int((h['cultivatingsample']==1).sum())}")
    cs = h[h["cultivatingsample"] == 1]
    log(f"  profit_per_area_wins range: [{cs['profit_per_area_wins'].min():.1f}, "
        f"{cs['profit_per_area_wins'].max():.1f}]  mean={cs['profit_per_area_wins'].mean():.1f}")
    log(f"  HIES waves: {sorted(cs['fin_yr'].unique())}")
    log(f"  obs per wave: {cs.groupby('fin_yr').size().to_dict()}")

    # ----- 6. PSLM section EFGB -----
    e = load("secEFGB_3_25.dta")
    log("\n[6] secEFGB_3_25.dta — PSLM individual data")
    log(f"  rows={len(e)}  cols={e.shape[1]}")
    log(f"  ln_area non-missing: {e['ln_area'].notna().sum()}")
    log(f"  irrigation var (secF1_Q1_c4) values: "
        f"{e['secF1_Q1_c4'].value_counts(dropna=False).to_dict()}")
    log(f"  PSLM waves: {sorted(e['fin_yr'].dropna().unique().tolist())}")

    # ----- 7. Bureaucrat panel -----
    bu = load("Bureaucrat_district_Yr.dta")
    log("\n[7] Bureaucrat_district_Yr.dta")
    log(f"  rows={len(bu)}  unique bureaucrats={bu['unique_id'].nunique()}")
    log(f"  cum_tax range: [{bu['cum_tax'].min():.1f}, {bu['cum_tax'].max():.1f}]  "
        f"mean={bu['cum_tax'].mean():.1f}")
    log(f"  cum_tax > 100% obs: {(bu['cum_tax']>100).sum()}  "
        f"(possible if late payments)")
    log(f"  perf_dummy50 share: {bu['perf_dummy50'].mean():.2f}")
    log(f"  perf_dummy75 share: {bu['perf_dummy75'].mean():.2f}")

    # ----- 8. Revenue circles -----
    rv = load("revcircle_yr.dta")
    log("\n[8] revcircle_yr.dta")
    log(f"  rows={len(rv)}  rev circles={rv['id_SAR'].nunique()}  "
        f"districts={rv['district'].nunique()}")
    log(f"  share of computerization_id_dummy==1: {rv['computerization_id_dummy'].mean():.3f}")
    log(f"  rev circles per district: min={rv.groupby('district')['id_SAR'].nunique().min()}, "
        f"max={rv.groupby('district')['id_SAR'].nunique().max()}")
    log(f"  tax_acres distribution:")
    log(rv["tax_acres"].describe().to_string())

    # ----- 9. Treatment / control balance check -----
    log("\n[9] Pre-period (2006-2011) means by phase (T1=phase1, T2=phase2, control=phase3)")
    pre = tax[tax["fin_yr"] <= 2011]
    g = pre.groupby("phase")["tax_acres"].agg(["mean", "std", "count"])
    log(g.to_string())

    log("\n[10] Missing data patterns")
    for col in ["tax_acres", "max_cdemand2_PKR", "Actual_Area000Acres"]:
        miss = tax[col].isna().sum()
        log(f"  {col}: {miss} missing ({miss/len(tax)*100:.1f}%)")
    log("  Note: max_cdemand2_PKR has zeros which break log() in Table 4 col 2 — "
        "the Stata code does not pre-filter; statsmodels reports those rows as "
        "log(0)=-inf and the regression silently drops them.")

    (OUT / "04_data_audit.txt").write_text("\n".join(out))


if __name__ == "__main__":
    main()
