"""Phase 3: Data audit for Abramitzky et al. (2023) SEA panel.

Checks coverage, balance, distributions, and logical consistency of the
sea_panel_data.dta file used by Tables 1-3.
"""
import numpy as np
import pandas as pd

from utils import load_panel, TABLE1_OUTCOMES, SAMPLES


def section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main():
    df = load_panel(drop_outlier=False)

    section("1. Coverage")
    print(f"rows: {len(df):,}   SEAs: {df['sea'].nunique():,}   years: {sorted(df['year'].unique())}")
    print("\nrows per year:")
    print(df.groupby("year").size().to_string())

    census_years = [1900, 1910, 1920, 1930]
    panel = df[df["year"].isin(census_years)].copy()
    print(f"\ncensus-year rows: {len(panel):,}, unique SEAs: {panel['sea'].nunique()}")

    # Sample distribution
    print("\nSEAs by sea_type (1=urban, 2=mining, 3=rural):")
    print(panel.drop_duplicates("sea").groupby("sea_type").size().to_string())

    section("2. Panel balance")
    counts_per_sea = panel.groupby("sea")["year"].nunique()
    print("distribution of (census years observed per SEA):")
    print(counts_per_sea.value_counts().sort_index().to_string())
    unbalanced = counts_per_sea[counts_per_sea != 4]
    print(f"\nSEAs without all 4 census years: {len(unbalanced)}")
    if len(unbalanced):
        print(unbalanced.head(10).to_string())

    section("3. Exposure measures (QE1, QE2, FB1900)")
    stats = panel.drop_duplicates("sea")[["QE1", "QE2", "FB1900"]].describe(
        percentiles=[0.01, 0.5, 0.99]
    )
    print(stats.to_string())
    print("\ncorr(QE2, FB1900):", round(panel.drop_duplicates("sea")[["QE2", "FB1900"]].corr().iloc[0, 1], 3))
    print("corr(QE1, QE2):", round(panel.drop_duplicates("sea")[["QE1", "QE2"]].corr().iloc[0, 1], 3))

    # Sanity: QE2 and FB1900 non-negative
    neg_qe2 = (panel["QE2"] < 0).sum()
    neg_fb = (panel["FB1900"] < 0).sum()
    print(f"QE2 < 0 rows: {neg_qe2}   FB1900 < 0 rows: {neg_fb}")

    section("4. Sub-county population identities")
    # Check: qrtot + fbold + wtot + nwtot + fbnqr ~= totpop? They are working-age
    # male counts, so won't equal total. But qrtot + nqrtot should equal totpop (working-age males)
    # because nqrtot is defined as all unrestricted (natives + old FB + WH).
    panel2 = panel.dropna(subset=["qrtot", "nqrtot", "totpop"])
    lhs = panel2["qrtot"] + panel2["nqrtot"]
    diff_share = (lhs - panel2["totpop"]) / panel2["totpop"]
    print(
        "qrtot + nqrtot vs totpop   "
        f"mean diff share: {diff_share.mean():+.4f}  max |diff|: {diff_share.abs().max():.4f}"
    )
    # Check native-born white + native-born non-white + fbold + qrtot + fbnqr ~ totpop
    comp = (
        panel2["wtot"] + panel2["nwtot"] + panel2["fbold"] + panel2["qrtot"] + panel2["fbnqr"]
    )
    diff2 = (comp - panel2["totpop"]) / panel2["totpop"]
    print(
        "w + nw + fbold + qrtot + fbnqr vs totpop   "
        f"mean diff share: {diff2.mean():+.4f}  max |diff|: {diff2.abs().max():.4f}"
    )

    section("5. Outlier SEA 311 (North Dakota)")
    sea311 = panel[panel["sea"] == 311].drop_duplicates("sea")[["sea", "QE2", "FB1900", "stateicp"]]
    print(sea311.to_string())
    # How many sd above sample mean?
    m = panel.drop_duplicates("sea")["QE2"].mean()
    s = panel.drop_duplicates("sea")["QE2"].std()
    print(f"QE2 mean={m:.4f} sd={s:.4f}. SEA 311 QE2 z-score: {(sea311['QE2'].iloc[0] - m) / s:.2f}")

    section("6. Decadal-change outcomes (1920-1930, working-age males)")
    # Compute the d<var> actually used in Table 1, look at distribution
    df2 = load_panel(drop_outlier=True)
    sub = df2[df2["year"].isin([1920, 1930])].copy()
    sub["l_totpop"] = sub.groupby("sea")["totpop"].shift(1)
    rows = []
    for v in TABLE1_OUTCOMES:
        sub[f"l_{v}"] = sub.groupby("sea")[v].shift(1)
        sub[f"d{v}"] = (sub[v] - sub[f"l_{v}"]) / sub["l_totpop"]
        x = sub[f"d{v}"].dropna()
        rows.append(
            {
                "outcome": v,
                "mean": x.mean(),
                "std": x.std(),
                "p01": x.quantile(0.01),
                "p50": x.median(),
                "p99": x.quantile(0.99),
                "n_missing": int(sub[f"d{v}"].isna().sum()),
            }
        )
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    section("7. Missing data and duplicates")
    print("duplicate (sea, year) rows:", panel.duplicated(subset=["sea", "year"]).sum())
    for c in ["totpop", "QE2", "FB1900", "region", "sea_type"] + TABLE1_OUTCOMES:
        miss = panel[c].isna().sum()
        if miss:
            print(f"  {c}: {miss} missing")
    print("(no line => zero missing)")

    section("8. Sample sizes in the regression panel (should match paper)")
    from utils import build_did_panel
    reg = build_did_panel(load_panel(drop_outlier=True), TABLE1_OUTCOMES)
    print(f"full regression panel rows: {len(reg)}  (paper: 918)")
    for s, code in SAMPLES.items():
        sub = reg[reg["sea_type"] == code]
        n_sea = sub["sea"].nunique()
        print(f"  {s:<8} rows={len(sub):4d}  SEAs={n_sea:4d}")


if __name__ == "__main__":
    main()
