"""Phase 3: Data audit for the cleaned weekly panel."""
import numpy as np
import pandas as pd
from utils import OUT, PUBLISHER_NAMES, SWITCH_DATES, stata_wkt_scalar


def main():
    wk = pd.read_parquet(OUT / "ebook2_2.parquet")
    n = len(wk)
    print(f"\n[coverage] n={n:,} rows")
    print(f"  unique titles (ebookid):     {wk['ebookid'].nunique():,}")
    print(f"  unique title-stores:         {wk['ebookid2'].nunique():,}")
    print(f"  stores:                      {sorted(wk['store'].unique())}")
    print(f"  publishers (soldby):         {sorted(wk['soldby'].dropna().unique())}")
    print(f"  week range (Stata wkt):      {int(wk['wkt'].min())} → {int(wk['wkt'].max())}")
    print(f"  corresponds to ~nov 2014 → oct 2015 (≈{wk['wkt'].nunique()} weeks)")

    print(f"\n[agency switch weeks]")
    for sb, d in SWITCH_DATES.items():
        print(f"  soldby={sb} {PUBLISHER_NAMES[sb]:<22} switch={d} (wkt={stata_wkt_scalar(d)})")

    print("\n[observations per store × publisher × agency]")
    tab = wk.groupby(["store", "soldby", "agency"]).size().unstack(fill_value=0)
    print(tab)

    print("\n[distributions: price]")
    for (store, ag), sub in wk.groupby(["store", "agency"]):
        print(f"  store={store} agency={ag}: "
              f"mean={sub['price'].mean():.2f} median={sub['price'].median():.2f} "
              f"sd={sub['price'].std():.2f} "
              f"p1={sub['price'].quantile(.01):.2f} p99={sub['price'].quantile(.99):.2f}")

    print("\n[distributions: rating]")
    r = wk["rating"]
    print(f"  mean={r.mean():.2f} sd={r.std():.2f} min={r.min():.2f} max={r.max():.2f}")
    bad = ((r < 0) | (r > 5)).sum()
    print(f"  out-of-range ratings (<0 or >5): {bad}")

    print("\n[logical checks]")
    neg_price = (wk["price"] <= 0).sum()
    print(f"  price ≤ 0:                         {neg_price}")
    missing_lagprice = wk["lagprice"].isna().sum()
    print(f"  missing lagprice (ok at edges):    {missing_lagprice:,}")
    singleton = (wk.groupby("ebookid2")["wkt"].transform("size") == 1).sum()
    print(f"  singleton book-stores (should=0):  {singleton}")

    print("\n[panel balance]")
    counts = wk.groupby("ebookid2")["wkt"].size()
    print(f"  weeks/book-store: mean={counts.mean():.1f} median={int(counts.median())} "
          f"min={counts.min()} max={counts.max()}")
    print(f"  pct with ≥20 weeks: {(counts >= 20).mean()*100:.1f}%")

    print("\n[duplicates]")
    dup_key = wk.duplicated(["ebookid", "store", "wkt"]).sum()
    print(f"  duplicate (ebookid, store, wkt):   {dup_key}")

    print("\n[agency timing sanity]")
    # For each publisher, the share of pre-switch rows where agency==1 should be 0
    for sb in range(1, 6):
        wkcut = stata_wkt_scalar(SWITCH_DATES[sb])
        pre = wk[(wk["soldby"] == sb) & (wk["wkt"] < wkcut)]
        post = wk[(wk["soldby"] == sb) & (wk["wkt"] >= wkcut)]
        bad = 0 if len(pre) == 0 else (pre["agency"] == 1).sum()
        pct_post_agency = 100 * (post["agency"] == 1).mean() if len(post) else np.nan
        print(f"  soldby={sb} pre_rows_agency=1:{bad}  post_agency_share={pct_post_agency:.0f}%")

    # Missingness by group
    print("\n[missingness by store]")
    for s in (1, 2):
        sub = wk[wk["store"] == s]
        print(f"  store={s}: price_miss={sub['price'].isna().mean()*100:.2f}% "
              f"rating_miss={sub['rating'].isna().mean()*100:.2f}% "
              f"salesrank_miss={sub['salesrank'].isna().mean()*100:.2f}%")


if __name__ == "__main__":
    main()
