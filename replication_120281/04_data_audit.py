"""Data audit for the replication package of Cai, Li & Santacreu (2022).

Audits coverage, distributions, and logical consistency of the raw inputs used
in the gravity regression and in the structural calibration (even though we do
not run the MATLAB calibration ourselves, we still want to understand the
quality of the inputs)."""
import numpy as np
import pandas as pd

from utils import load_trade, RAW, OUT, N_TRADABLE, COUNTRIES


def banner(t):
    print(f"\n{'='*60}\n{t}\n{'='*60}")


def main():
    df = load_trade()

    banner("1. Trade panel coverage")
    print(f"Observations:      {len(df)}")
    print(f"Exporters:         {df['iso_i'].nunique()}")
    print(f"Importers:         {df['iso_n'].nunique()}")
    print(f"Sectors (j19):     {df['j19'].nunique()}")
    print(f"Expected cells:    19*19*19 = {19*19*19}")
    cells = df.groupby("j19").size()
    print(f"Cells per sector:  min={cells.min()}  max={cells.max()}  (expect 19*18=342 off-diagonal pairs plus 19 own-own = 361)")
    # The dataset excludes own-own rows (domestic trade not a gravity obs)
    assert (cells == 342).all(), "Unequal panel across sectors"

    banner("2. Missingness in gravity inputs")
    key_vars = ["lX_ni_X_nn"] + [f"d{b}" for b in ["0_375","375_750","750_1500","1500_3000","3000_6000"]] + ["d6000_","contig","FTA","CU"]
    miss = df[key_vars].isna().sum()
    print("Missing counts by variable:")
    print(miss.to_string())
    print(f"\nNull lX_ni_X_nn by sector:")
    print(df.groupby("j19")["lX_ni_X_nn"].apply(lambda s: s.isna().sum()).to_string())
    print(f"\nA null lX implies zero bilateral trade (log of zero).")

    banner("3. Dependent variable distribution (log trade share)")
    d = df["lX_ni_X_nn"].dropna()
    print(d.describe().to_string())
    print(f"\nFraction below -15 (ultra-thin trade): {(d < -15).mean():.4f}")
    print(f"Fraction above 0 (more imports than domestic): {(d > 0).mean():.4f}")

    banner("4. Distance / bilateral dummies consistency")
    dist_bins = ["d0_375", "d375_750", "d750_1500", "d1500_3000", "d3000_6000", "d6000_"]
    pair = (df[df["j19"] == 1]
              [["iso_i", "iso_n"] + dist_bins + ["contig", "FTA", "CU"]]
              .drop_duplicates(subset=["iso_i", "iso_n"]))
    bin_sum = pair[dist_bins].sum(axis=1)
    print(f"Unique pairs (sector 1): {len(pair)}")
    print(f"Pairs where distance dummies sum to 1: {(bin_sum == 1).sum()}")
    nan_rows = pair[dist_bins].isna().any(axis=1).sum()
    print(f"Pairs with NaN distance bin: {nan_rows}")
    print(f"Contig fraction:  {pair['contig'].mean():.4f} ({int(pair['contig'].sum())}/{len(pair)})")
    print(f"FTA   fraction:   {pair['FTA'].mean():.4f}")
    print(f"CU    fraction:   {pair['CU'].mean():.4f}")

    banner("5. Productivity2005.dta coverage")
    p = pd.read_stata(RAW / "Productivity2005.dta", convert_categoricals=False)
    print(f"Shape: {p.shape}")
    print(f"Countries: {p['iso'].nunique()}")
    print(f"Sectors j: {sorted(p['j'].unique().tolist())}")
    print(p.groupby("iso").size().describe().to_string())
    print(f"Missing prod: {p['prod'].isna().sum()}")
    print("\nProductivity summary:")
    print(p["prod"].describe().to_string())
    print("\nLog-productivity dispersion by sector:")
    lp = p.assign(lprod=np.log(p["prod"])).groupby("j")["lprod"].agg(["mean", "std"])
    print(lp.to_string())

    banner("6. R&D intensity (s_ij_world)")
    s = pd.read_csv(RAW / "s_ij_world.txt", header=None, sep=",").values
    print(f"Shape: {s.shape}  (countries × sectors)")
    print(f"Range: [{s.min():.3e}, {s.max():.3e}]")
    print(f"Mean: {s.mean():.3e}   Median: {np.median(s):.3e}")
    # Total R&D by country
    s_df = pd.DataFrame(s, index=COUNTRIES)
    country_rd = s_df.sum(axis=1).sort_values(ascending=False)
    print("\nTotal R&D by country (sum across 19 sectors):")
    print(country_rd.to_string())

    banner("7. Trade cost summary (lD_inj1919.dta)")
    ld = pd.read_stata(RAW / "lD_inj1919.dta", convert_categoricals=False)
    # For tradable sectors, off-diagonal
    sub = ld[(ld["j19"] <= N_TRADABLE) & (ld["iso_i"] != ld["iso_n"])]
    print(f"Rows (tradable off-diag): {len(sub)}")
    print("lD_inj distribution:")
    print(sub["lD_inj"].describe().to_string())
    theta = 8.28
    d_ni = np.exp(-sub["lD_inj"] / theta)
    print(f"\nImplied d_ni (theta=8.28): mean={d_ni.mean():.3f} median={d_ni.median():.3f}")
    print(f"  25%={d_ni.quantile(.25):.3f}  75%={d_ni.quantile(.75):.3f}  max={d_ni.max():.3f}")
    # Placeholder check: j==19 row should have lD_inj = 1000
    sec19 = ld[ld["j19"] == 19]
    print(f"\nSector-19 (nontradable) rows: {len(sec19)}; lD_inj fillvalues (=1000): {(sec19['lD_inj'] == 1000).sum()}")
    diag = ld[ld["iso_i"] == ld["iso_n"]]
    print(f"Diagonal rows: {len(diag)}; lD_inj=0 frac: {(diag['lD_inj'] == 0).mean():.3f}")

    banner("8. GDP data (GDP_rawdata.dta)")
    g = pd.read_stata(RAW / "GDP_rawdata.dta", convert_categoricals=False)
    print(g.to_string(index=False))

    banner("9. Logical sanity checks")
    print("(a) FTA ⊆ Trade block? EU pairs should have FTA==1 in 2005.")
    eu = [
        ("DEU", "FRA"), ("FRA", "DEU"), ("DEU", "ITA"), ("BEL", "NLD"),
        ("ESP", "PRT"), ("FIN", "DEU"),
    ]
    for a, b in eu:
        r = df[(df["iso_i"] == a) & (df["iso_n"] == b) & (df["j19"] == 1)]
        if len(r):
            print(f"    {a}->{b}  FTA={int(r.iloc[0]['FTA'])}  CU={int(r.iloc[0]['CU'])}  contig={int(r.iloc[0]['contig'])}")
    print("(b) Euro-zone pairs should have CU==1. Note ISR/GBR/AUS are not in EZ.")
    print("(c) Distance: CAN<->USA should be < 375 miles in distance bin? (they are contiguous)")
    r = df[(df["iso_i"] == "CAN") & (df["iso_n"] == "USA") & (df["j19"] == 1)]
    if len(r):
        row = r.iloc[0]
        print(f"    CAN->USA distw={row['distw']:.0f}  contig={int(row['contig'])}  d1500_3000={int(row['d1500_3000'])}")

    banner("10. Correlation of own-sector trade cost across sectors")
    ld_tradable = ld[(ld["j19"] <= N_TRADABLE) & (ld["iso_i"] != ld["iso_n"])].copy()
    wide = ld_tradable.pivot_table(index=["iso_i","iso_n"], columns="j19", values="lD_inj")
    corr = wide.corr()
    print(f"Mean pairwise correlation of lD_inj across sectors: {corr.values[np.triu_indices_from(corr.values,1)].mean():.3f}")
    print(f"Min: {corr.values[np.triu_indices_from(corr.values,1)].min():.3f}")
    print(f"Max (excluding diagonal): {corr.values[np.triu_indices_from(corr.values,1)].max():.3f}")


if __name__ == "__main__":
    main()
