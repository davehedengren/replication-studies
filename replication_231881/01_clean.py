"""Phase 2: Translate 1_/2_/3_ data prep from Stata to Python.

Produces the cleaned weekly panel ebook2_2 used for Tables 2 and 3.
"""
import sys
import numpy as np
import pandas as pd
from utils import (
    OUT, PUBLISHER_NAMES, SWITCH_DATES,
    load_ebook2_0, load_sales, sales_rank_regression, stata_wkt_scalar,
)


def main():
    # ---- 1_ebooksbarg.do: sales rank regression + predict sales ----
    sales = load_sales()
    reg = sales_rank_regression(sales)
    ntitles = sales["idbn"].nunique()
    print("[sales-rank reg]")
    print(f"  log(Sales) = {reg.params['const']:.3f} + {reg.params['lsalesrank']:.3f}*log(Rank)")
    print(f"  (SE)      =  ({reg.bse['const']:.3f})  ({reg.bse['lsalesrank']:.3f})")
    print(f"  R^2 = {reg.rsquared:.3f}, N = {int(reg.nobs)}, titles = {ntitles}")

    df = load_ebook2_0()
    print(f"ebook2_0 loaded: {df.shape[0]:,} rows, {df.shape[1]} cols")

    # Predict sales per Stata code: Amazon = full, B&N = 1/4 of predicted
    a, b = reg.params["const"], reg.params["lsalesrank"]
    lsr = np.log(df["salesrank"].astype(float))
    df["sales"] = np.where(df["store"] == 1, np.exp(a + b * lsr),
                           0.25 * np.exp(a + b * lsr))
    # BHS alt: fixed coefficients
    df["salesbhs"] = np.where(df["store"] == 1,
                              np.exp(10.526 - 0.871 * lsr),
                              0.25 * np.exp(10.526 - 0.871 * lsr))

    # ---- 2_ebooksbarg.do: weekly aggregation ----
    # Weekly means over (ebookid, store, wkt)
    grp = ["ebookid", "store", "wkt"]
    wk = df.groupby(grp, as_index=False).agg(
        price=("price", "mean"),
        salesrank=("salesrank", "mean"),
        sales=("sales", "mean"),
        salesbhs=("salesbhs", "mean"),
        rating=("rating", "mean"),
        soldby=("soldby", "first"),
        idkindle=("idkindle", "first"),
        title=("title", "first"),
        author=("author", "first"),
    )
    # Stata scales *7 after the mean (daily → weekly total)
    wk["sales"] = wk["sales"] * 7
    wk["salesbhs"] = wk["salesbhs"] * 7

    # ebook-store id
    wk["ebookid2"] = wk["ebookid"].astype("int64") + 100000 * wk["store"].astype("int64")

    # lagged price (same ebookid2, wkt = previous wkt + 1)
    wk = wk.sort_values(["ebookid2", "wkt"]).reset_index(drop=True)
    wk["lagprice"] = wk.groupby("ebookid2")["price"].shift(1)
    wk["lag_wkt"] = wk.groupby("ebookid2")["wkt"].shift(1)
    wk.loc[wk["lag_wkt"] + 1 != wk["wkt"], "lagprice"] = np.nan
    wk = wk.drop(columns=["lag_wkt"])

    # agency dummy by publisher switch date
    wks = {k: stata_wkt_scalar(v) for k, v in SWITCH_DATES.items()}
    print("switch weeks:", wks)
    wk["agency"] = 0
    for sb, wkcut in wks.items():
        wk.loc[(wk["soldby"] == sb) & (wk["wkt"] >= wkcut), "agency"] = 1

    # drop books ever priced below 2.99 (min over ebookid+store)
    min_price = wk.groupby(["ebookid", "store"])["price"].transform("min")
    wk = wk[min_price >= 2.99].reset_index(drop=True)
    print(f"after min-price filter: {wk.shape[0]:,} rows")

    # Match 3_ebooksbarg.do sample filters
    # - drop wkt == 253 (used only for creating lagged prices)
    # - drop price missing, rating missing
    # - drop singleton observations (by ebookid2)
    wk = wk[wk["wkt"] != 253]
    wk = wk.dropna(subset=["price", "rating"])
    nobs_by_bk = wk.groupby("ebookid2")["wkt"].transform("size")
    wk = wk[nobs_by_bk > 1].reset_index(drop=True)
    print(f"analysis panel: {wk.shape[0]:,} rows, "
          f"{wk['ebookid2'].nunique():,} book-store series, "
          f"{wk['ebookid'].nunique():,} unique titles")

    wk["lprice"] = np.log(wk["price"])

    # Save
    out_path = OUT / "ebook2_2.parquet"
    wk.to_parquet(out_path, index=False)
    print(f"wrote {out_path}")

    # Summary of sample
    print("\n[coverage]")
    print(wk.groupby(["store", "soldby"]).size().unstack(fill_value=0))
    print("\n[agency period share by store]")
    print(wk.groupby(["store"])["agency"].mean().round(3))


if __name__ == "__main__":
    main()
