"""Phase 1/2: verify sample construction matches the published Stata pipeline.

The package ships pre-cleaned analysis datasets (Data/Analysis/*.dta) produced
by 01_cleaning_PSLM.py + 02_cleaning.do. We read those directly and print the
key shape/summary statistics that anchor the rest of the replication.
"""

from utils import load, OUT


def main():
    lines = []

    def log(msg):
        print(msg)
        lines.append(msg)

    log("=== District-year tax data (Tax_District_Yr.dta) ===")
    tax = load("Tax_District_Yr.dta")
    log(f"rows={len(tax)}  districts={tax['district'].nunique()}  years={sorted(tax['fin_yr'].unique())}")
    log(f"digitization==1 obs: {int(tax['digitization'].sum())}")
    log("obs per fin_yr:\n" + tax.groupby("fin_yr").size().to_string())
    log(f"tax_acres overall mean={tax['tax_acres'].mean():.3f}")
    log(f"tax_acres pre-2012 control mean={tax.loc[tax['fin_yr']<=2011, 'tax_acres'].mean():.3f}")
    log(f"phase breakdown: {tax['phase'].value_counts().to_dict()}")

    log("\n=== Stacked district-year data (Tax_District_Yr_stacked.dta) ===")
    stk = load("Tax_District_Yr_stacked.dta")
    log(f"rows={len(stk)}  events={sorted(stk['event_h'].unique())}")
    log(f"rows per event: {stk['event_h'].value_counts().to_dict()}")

    log("\n=== Reported cultivated area (reportedcultivated.dta) ===")
    rc = load("reportedcultivated.dta")
    log(f"rows={len(rc)}  years={sorted(rc['fin_yr'].unique())}")

    log("\n=== NDVI (NDVI_2006_2014.dta) ===")
    n = load("NDVI_2006_2014.dta")
    log(f"rows={len(n)}  districts={n['district'].nunique()}")

    log("\n=== HIES profit data (hh_level_data_hies.dta) ===")
    h = load("hh_level_data_hies.dta")
    log(f"rows={len(h)}  cultivating={int((h['cultivatingsample']==1).sum())}")

    log("\n=== PSLM section EFGB (secEFGB_3_25.dta) ===")
    e = load("secEFGB_3_25.dta")
    log(f"rows={len(e)}  cols={e.shape[1]}")

    log("\n=== Bureaucrat x district x year (Bureaucrat_district_Yr.dta) ===")
    b = load("Bureaucrat_district_Yr.dta")
    log(f"rows={len(b)}")
    log(f"cum_tax control mean = {b.loc[b['fin_yr']<=2011, 'cum_tax'].mean():.3f}")

    log("\n=== Revenue circle x year (revcircle_yr.dta) ===")
    r = load("revcircle_yr.dta")
    log(f"rows={len(r)}  rev circles={r['id_SAR'].nunique()}")

    (OUT / "01_clean.txt").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
