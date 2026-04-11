"""Data audit: coverage, plausibility, panel balance, and logical consistency
for the calibration dataset that drives Tables 1-3 and the misc cross-country
dataset that drives the elasticity regressions.
"""
import numpy as np
import pandas as pd

from utils import load_calib, TEMP, DATA, OUT


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    df = load_calib()
    misc = pd.read_stata(DATA / "misc" / "misc.dta", convert_categoricals=False)

    section("1. Coverage of calibration data (calib_all_1990_2010.dta)")
    calib_only = df[df["year_orig"].notna()].copy()
    print(f"  rows (country-years):     {len(calib_only)}")
    print(f"  distinct countries:       {calib_only['country'].nunique()}")
    print(f"  distinct sample strings:  {calib_only['sample'].nunique()}")
    print(f"  year_orig range:          {int(calib_only['year_orig'].min())}–{int(calib_only['year_orig'].max())}")
    print("\n  country-year observations:")
    cc = calib_only.groupby("country")["year_orig"].apply(lambda s: sorted(int(x) for x in s))
    for c, yrs in cc.items():
        print(f"    {c:<24} {yrs}")

    section("2. Coverage of macro panel (misc.dta)")
    print(f"  rows:                 {len(misc)}")
    print(f"  distinct countries:   {misc['country'].nunique()}")
    print(f"  year range:           {int(misc['year'].min())}–{int(misc['year'].max())}")
    print(f"  countries with l_y in 2000: {misc[(misc['year'] == 2000) & misc['l_y'].notna()]['country'].nunique()}")
    print(f"  Barro-Lee share1 non-null (2000): "
          f"{misc[(misc['year'] == 2000) & misc['share1_blee'].notna()]['country'].nunique()}")

    section("3. Plausibility bounds (baseline skill premium w5/w3, 2000)")
    base = df[(df["sample_micro"] == 1) & (df["year"] == 2000)].copy()
    wrat = np.exp(base["l_w5_dum_skti_secall"])
    print(f"  w5/w3 min / max:      {wrat.min():.3f} / {wrat.max():.3f}")
    print(f"  w5/w3 mean / median:  {wrat.mean():.3f} / {wrat.median():.3f}")
    # A log point is the mincerian return; common numeric range is 1.3-5.
    flag = base[(wrat < 1.2) | (wrat > 5.0)][["country"]].copy()
    flag["w5_over_w3"] = wrat[flag.index].values
    print(f"  outside [1.2, 5.0]:   {len(flag)} country-years")
    if len(flag):
        print(flag.to_string(index=False))

    section("4. Education share logical consistency (hours shares sum to 1)")
    for y in [2000]:
        sub = df[(df["sample_micro"] == 1) & (df["year"] == y)]
        tot = sum(sub[f"share{e}_hrs_secall"] for e in range(1, 6))
        print(f"  year {y}: sum of share1..5_hrs_secall – min={tot.min():.4f} max={tot.max():.4f}")
    for y in [2000]:
        sub = df[(df["sample_micro"] == 1) & (df["year"] == y)]
        tot_sec = sum(sub[f"share_hrs_sec{s}"] for s in range(1, 5))
        print(f"  year {y}: sum of share_hrs_sec1..4   – min={tot_sec.min():.4f} max={tot_sec.max():.4f}")

    # Within each sector, education shares should also sum ~1
    for s in range(1, 5):
        sub = df[(df["sample_micro"] == 1) & (df["year"] == 2000)]
        tot = sum(sub[f"share{e}_hrs_sec{s}"] for e in range(1, 6))
        print(f"  sector {s}: edu shares sum – min={tot.min():.4f} max={tot.max():.4f}")

    section("5. Shares in [0,1] check")
    base2 = df[(df["sample_micro"] == 1) & (df["year"] == 2000)]
    issues = 0
    for e in range(1, 6):
        for suf in ["hrs_secall", "bod_secall", "pop_secall"]:
            col = f"share{e}_{suf}"
            vals = base2[col]
            bad = ((vals < -1e-6) | (vals > 1 + 1e-6)).sum()
            if bad:
                print(f"  {col}: {bad} out-of-range")
                issues += 1
    print(f"  shares out of [0,1]: {issues} columns with issues")

    section("6. Relative skill efficiency range (irAQ53_hrs) — paper Table 1")
    # reload post-compute pieces via tab1_tab2 side effect would be cleaner,
    # but we can read directly from the saved csv from 01_tab1_tab2.py
    try:
        tab1 = pd.read_csv(OUT / "tab1_levels.csv")
        print(tab1.describe().loc[["min", "max", "mean"]].to_string())
    except FileNotFoundError:
        print("  (run 01_tab1_tab2.py first)")

    section("7. Missing patterns in calib 2000")
    cols = [c for c in ["l_w1_dum_skti_secall", "l_w5_dum_skti_secall",
                        "l_w1_dumse_skti_secall", "l_w5_dumse_skti_secall",
                        "l_w5_dum_skti_sec1", "l_w5_dum_skti_sec2",
                        "l_w5_dum_skti_sec3", "l_w5_dum_skti_sec4"] if c in df.columns]
    sub = df[(df["sample_micro"] == 1) & (df["year"] == 2000)]
    for c in cols:
        miss = sub[c].isna().sum()
        print(f"  {c:<30} missing {miss}/{len(sub)}")

    section("8. Panel balance across calib years")
    panel = calib_only.groupby("country")["year_orig"].nunique().sort_values()
    print(panel.to_string())

    section("9. Duplicates")
    dups = calib_only.duplicated(subset=["country", "year_orig"]).sum()
    print(f"  duplicate (country, year_orig) rows in calib: {dups}")
    dups_m = misc.duplicated(subset=["country", "year"]).sum()
    print(f"  duplicate (country, year) rows in misc:       {dups_m}")

    section("10. GDP dispersion in 12-country sample (log GDP p.w., 2000)")
    sub = df[(df["sample_micro"] == 1) & (df["year"] == 2000)][["country", "l_y"]]
    print(sub.to_string(index=False))
    print(f"  range of l_y: {sub['l_y'].max() - sub['l_y'].min():.3f} "
          f"(ratio of max/min GDP p.w. ≈ {np.exp(sub['l_y'].max() - sub['l_y'].min()):.1f}×)")


if __name__ == "__main__":
    main()
