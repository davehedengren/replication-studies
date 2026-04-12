"""Phase 4: Robustness checks for the DiD specification (Table 3, col A)."""
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from utils import OUT, PUBLISHER_NAMES, SWITCH_DATES, stata_wkt_scalar


def did_fit(panel, include_rating=False, cluster_entity=True):
    df = panel.copy()
    for p in range(1, 6):
        df[f"ag_p{p}"] = ((df["agency"] == 1) & (df["soldby"] == p)).astype(float)
    regressors = [f"ag_p{p}" for p in range(1, 6) if df[f"ag_p{p}"].std() > 0]
    if include_rating and df["rating"].std() > 0:
        regressors.append("rating")
    df = df.set_index(["ebookid", "wkt"])
    mod = PanelOLS(df["lprice"], df[regressors], entity_effects=True, time_effects=True,
                   drop_absorbed=True, check_rank=False)
    if cluster_entity:
        return mod.fit(cov_type="clustered", cluster_entity=True)
    return mod.fit(cov_type="robust")


def agg_effect(res, panel):
    treated = panel[panel["agency"] == 1]
    if len(treated) == 0:
        return np.nan, np.nan
    w = np.array([(treated["soldby"] == p).sum() / len(treated) for p in range(1, 6)])
    keep = [f"ag_p{p}" for p in range(1, 6) if f"ag_p{p}" in res.params.index]
    if not keep:
        return np.nan, np.nan
    idx = [int(k[-1]) - 1 for k in keep]
    wk_ = w[idx]
    wk_ = wk_ / wk_.sum() if wk_.sum() > 0 else wk_
    b = np.array([res.params[k] for k in keep])
    V = res.cov.loc[keep, keep].values
    return float(wk_ @ b), float(np.sqrt(wk_ @ V @ wk_))


def run_both_stores(panel, label, **kw):
    out = {}
    for store in (1, 2):
        sub = panel[panel["store"] == store]
        if sub["agency"].sum() < 50:  # too few treated
            out[store] = (np.nan, np.nan, int(len(sub)))
            continue
        res = did_fit(sub, **kw)
        agg, se = agg_effect(res, sub)
        out[store] = (agg, se, int(res.nobs))
    print(f"{label:<60}  Amazon: {out[1][0]:+.3f} ({out[1][1]:.3f}) N={out[1][2]:,}  "
          f"B&N: {out[2][0]:+.3f} ({out[2][1]:.3f}) N={out[2][2]:,}")
    return {"label": label,
            "amz_agg": out[1][0], "amz_se": out[1][1], "amz_n": out[1][2],
            "bn_agg": out[2][0], "bn_se": out[2][1], "bn_n": out[2][2]}


def main():
    wk = pd.read_parquet(OUT / "ebook2_2.parquet")
    print(f"\nbaseline panel n={len(wk):,}")
    print("=" * 118)
    rows = []

    # 0 baseline (col A, no rating)
    rows.append(run_both_stores(wk, "[0] Baseline (no covariates)"))

    # 1 add rating covariate (col B)
    rows.append(run_both_stores(wk, "[1] + rating covariate", include_rating=True))

    # 2 robust SE instead of cluster
    rows.append(run_both_stores(wk, "[2] Heteroskedasticity-robust SE (no cluster)",
                                cluster_entity=False))

    # 3 winsorize price at [1,99]
    w3 = wk.copy()
    for s in (1, 2):
        m = w3["store"] == s
        lo = w3.loc[m, "price"].quantile(0.01)
        hi = w3.loc[m, "price"].quantile(0.99)
        w3.loc[m, "price"] = w3.loc[m, "price"].clip(lo, hi)
    w3["lprice"] = np.log(w3["price"])
    rows.append(run_both_stores(w3, "[3] Winsorize price at 1/99 pct"))

    # 4 drop 5% of titles with largest within-title price range
    rng = wk.groupby("ebookid2")["price"].agg(lambda x: x.max() - x.min())
    cut = rng.quantile(0.95)
    keep_ids = rng[rng <= cut].index
    w4 = wk[wk["ebookid2"].isin(keep_ids)]
    rows.append(run_both_stores(w4, "[4] Drop top-5% volatile titles"))

    # 5 balanced: book-stores with >=40 weeks
    counts = wk.groupby("ebookid2")["wkt"].size()
    keep = counts[counts >= 40].index
    w5 = wk[wk["ebookid2"].isin(keep)]
    rows.append(run_both_stores(w5, "[5] Balanced panel (≥40 weeks)"))

    # 6 drop ±4 weeks around each publisher's switch
    mask = np.ones(len(wk), dtype=bool)
    for sb, d in SWITCH_DATES.items():
        cutoff = stata_wkt_scalar(d)
        mask &= ~((wk["soldby"] == sb) & (wk["wkt"].between(cutoff - 4, cutoff + 3)))
    w6 = wk[mask]
    rows.append(run_both_stores(w6, "[6] Drop ±4 week window around switch"))

    # 7 placebo: shift agency 12 weeks earlier, keep pre-period only
    w7 = wk.copy()
    w7["agency"] = 0
    for sb, d in SWITCH_DATES.items():
        cut = stata_wkt_scalar(d) - 12
        w7.loc[(w7["soldby"] == sb) & (w7["wkt"] >= cut) & (w7["wkt"] < stata_wkt_scalar(d)),
               "agency"] = 1
    # restrict to pre-switch for this placebo
    pre_mask = np.ones(len(w7), dtype=bool)
    for sb, d in SWITCH_DATES.items():
        pre_mask &= ~((w7["soldby"] == sb) & (w7["wkt"] >= stata_wkt_scalar(d)))
    w7 = w7[pre_mask]
    rows.append(run_both_stores(w7, "[7] Placebo: 12wk-early pseudo-switch, pre-period only"))

    # 8 leave-one-publisher-out (drop a Big-5 publisher entirely)
    for drop_pub in range(1, 6):
        w8 = wk[wk["soldby"] != drop_pub]
        rows.append(run_both_stores(
            w8, f"[8.{drop_pub}] Drop {PUBLISHER_NAMES[drop_pub]}"))

    # 9 cluster at book-store (ebookid2) instead of title (ebookid)
    w9 = wk.copy()
    w9_idx = w9.set_index(["ebookid2", "wkt"])
    for p in range(1, 6):
        w9_idx[f"ag_p{p}"] = ((w9_idx["agency"] == 1) & (w9_idx["soldby"] == p)).astype(float)
    regressors = [f"ag_p{p}" for p in range(1, 6)]
    amz_sub = w9_idx[w9_idx["store"] == 1]
    bn_sub = w9_idx[w9_idx["store"] == 2]
    amz = PanelOLS(amz_sub["lprice"], amz_sub[regressors], entity_effects=True, time_effects=True,
                   drop_absorbed=True).fit(cov_type="clustered", cluster_entity=True)
    bn = PanelOLS(bn_sub["lprice"], bn_sub[regressors], entity_effects=True, time_effects=True,
                  drop_absorbed=True).fit(cov_type="clustered", cluster_entity=True)
    a_amz, se_amz = agg_effect(amz, wk[wk["store"] == 1])
    a_bn, se_bn = agg_effect(bn, wk[wk["store"] == 2])
    print(f"{'[9] Cluster at book-store (ebookid2) instead of title':<60}  "
          f"Amazon: {a_amz:+.3f} ({se_amz:.3f}) N={int(amz.nobs):,}  "
          f"B&N: {a_bn:+.3f} ({se_bn:.3f}) N={int(bn.nobs):,}")
    rows.append({"label": "[9] Cluster at book-store",
                 "amz_agg": a_amz, "amz_se": se_amz, "amz_n": int(amz.nobs),
                 "bn_agg": a_bn, "bn_se": se_bn, "bn_n": int(bn.nobs)})

    # 10 pooled across stores, include store × wkt FE and Amazon dummy interactions
    w10 = wk.copy()
    # Use (ebookid2, wkt) so FE absorbs title-store + week
    w10["wkt_store"] = w10["wkt"].astype(int) * 10 + w10["store"].astype(int)
    w10 = w10.set_index(["ebookid2", "wkt_store"])
    for p in range(1, 6):
        w10[f"ag_p{p}"] = ((w10["agency"] == 1) & (w10["soldby"] == p)).astype(float)
    regressors = [f"ag_p{p}" for p in range(1, 6)]
    pooled = PanelOLS(w10["lprice"], w10[regressors], entity_effects=True, time_effects=True,
                      drop_absorbed=True).fit(cov_type="clustered", cluster_entity=True)
    # weighted agg over both stores
    treated = wk[wk["agency"] == 1]
    weights = np.array([(treated["soldby"] == p).sum() / len(treated) for p in range(1, 6)])
    agg = float(weights @ np.array([pooled.params[f"ag_p{p}"] for p in range(1, 6)]))
    V = pooled.cov.loc[[f"ag_p{p}" for p in range(1, 6)],
                       [f"ag_p{p}" for p in range(1, 6)]].values
    agg_se = float(np.sqrt(weights @ V @ weights))
    print(f"{'[10] Pooled (book-store × store-week FE)':<60}  "
          f"pooled agg: {agg:+.3f} ({agg_se:.3f}) N={int(pooled.nobs):,}")
    rows.append({"label": "[10] Pooled w/ store×week FE",
                 "amz_agg": agg, "amz_se": agg_se, "amz_n": int(pooled.nobs),
                 "bn_agg": np.nan, "bn_se": np.nan, "bn_n": np.nan})

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT / "robustness.csv", index=False)
    print(f"\nwrote {OUT/'robustness.csv'}")


if __name__ == "__main__":
    main()
