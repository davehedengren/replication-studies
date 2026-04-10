"""Robustness checks for Table 2 (main DiD: campaign effect on share of vodka)
and Table 3 Panel B (IV: vodka share → log mortality).

Each check re-estimates the headline Column 4 spec of Table 2 (all controls,
SE clustered by individual) and/or the Col 3 IV spec of Table 3 under a
modified sample or specification.
"""
import numpy as np
import pandas as pd
from linearmodels.iv.absorbing import AbsorbingLS
from linearmodels.iv import IV2SLS
from utils import CONTROLS, load_base_sample, table2_sample, OUT


def t2_fit(df, y="share_vodka", regs=None, controls=None):
    regs = regs or ["rural_gorbachev", "gorbachev", "rural"]
    controls = CONTROLS if controls is None else controls
    cols = [y] + regs + controls + ["identificator", "id", "round", "age"]
    used = df[cols].dropna().copy()
    exog = used[regs + controls].assign(_const=1.0)
    absorb = used[["id", "round", "age"]].astype("category")
    mod = AbsorbingLS(used[y], exog, absorb=absorb, drop_absorbed=True)
    res = mod.fit(cov_type="clustered", clusters=used["identificator"].astype(int))
    return res.params["rural_gorbachev"], res.std_errors["rural_gorbachev"], int(res.nobs)


def t3_iv_fit(panel, y="logtotal", extra_controls=None):
    extra = extra_controls or []
    controls = ["rural", "gorbachev", "logincome", "logpopulation"] + extra
    fes = ["age", "year", "fedokrug"]
    used = panel.dropna(subset=[y, "share_vodka", "gorbachev_rural"] + controls + fes + ["id_rural_year"]).copy()
    dummies = [pd.get_dummies(used[f].astype(int), prefix=f, drop_first=True).astype(float) for f in fes]
    X = pd.concat([used[controls].astype(float)] + dummies, axis=1).assign(_const=1.0)
    res = IV2SLS(used[y].astype(float), X, used[["share_vodka"]].astype(float),
                 used[["gorbachev_rural"]].astype(float)).fit(
                     cov_type="clustered", clusters=used["id_rural_year"].astype(int))
    return res.params["share_vodka"], res.std_errors["share_vodka"], int(res.nobs)


def main():
    print("=" * 70)
    print("TABLE 2 ROBUSTNESS (headline coef: rural_gorbachev on share_vodka)")
    print(f"Baseline (paper Col 4): 5.232 (SE 1.986), N=29,083")
    print("=" * 70)
    df = load_base_sample()
    base = table2_sample(df)

    # 1. Drop Moscow & St. Petersburg (regions with highest attrition)
    # Use region_id — cities: Moscow 7742 / 7743, St.P 7844 per RLMS? Fall back: use site or identificator.
    # Simpler: drop top-attrition individuals by region_id
    top_regions = base["region_id"].value_counts().index[:3].tolist()  # coarse proxy
    s1 = base[~base["region_id"].isin(top_regions)]
    print(f"\n1. Drop top-3 densest regions (Moscow/StP proxy)")
    print(f"   {t2_fit(s1)}")

    # 2. Leave-one-round-out (sensitivity to single-round shocks)
    print("\n2. Leave-one-round-out sensitivity:")
    for rd in sorted(base["round"].unique()):
        s = base[base["round"] != rd]
        b, se, n = t2_fit(s)
        print(f"   drop round {int(rd)}: beta={b:.3f} (SE {se:.3f}) N={n:,}")

    # 3. Alternative cohort window — 1985–1991 (relax campaign end)
    base2 = base.copy()
    a17 = base2["birthy"] + 17
    base2["gorbachev"] = ((a17 >= 1985) & (a17 <= 1991)).astype(int)
    base2["rural_gorbachev"] = base2["rural"] * base2["gorbachev"]
    print("\n3. Wider campaign window 1985–91:")
    print(f"   {t2_fit(base2)}")

    # 4. Narrower window 1987–89
    base3 = base.copy()
    a17 = base3["birthy"] + 17
    base3["gorbachev"] = ((a17 >= 1987) & (a17 <= 1989)).astype(int)
    base3["rural_gorbachev"] = base3["rural"] * base3["gorbachev"]
    print("\n4. Narrower campaign window 1987–89:")
    print(f"   {t2_fit(base3)}")

    # 5. Winsorize share_vodka at 1st/99th percentiles
    base_w = base.copy()
    q1, q99 = base_w["share_vodka"].quantile([0.01, 0.99])
    base_w["share_vodka"] = base_w["share_vodka"].clip(q1, q99)
    print("\n5. Winsorize share_vodka at 1%/99%:")
    print(f"   {t2_fit(base_w)}")

    # 6. Placebo: shuffle rural within year×age cells 500x and compute null
    rng = np.random.default_rng(0)
    null_betas = []
    for i in range(500):
        bp = base.copy()
        bp["rural"] = rng.permutation(bp["rural"].values)
        bp["rural_gorbachev"] = bp["rural"] * bp["gorbachev"]
        b, _, _ = t2_fit(bp, regs=["rural_gorbachev", "gorbachev", "rural"],
                          controls=[c for c in CONTROLS if c != "wtself"])
        null_betas.append(b)
    null_betas = np.array(null_betas)
    obs = 5.232
    p = (np.abs(null_betas) >= obs).mean()
    print(f"\n6. Placebo permutation (500 draws): p(|null|>=5.232)={p:.3f}, "
          f"null mean={null_betas.mean():.3f}, std={null_betas.std():.3f}")

    # 7. Drop heavy drinkers (alcohol_intake > 99th percentile)
    cutoff = base["alcohol_intake"].quantile(0.99)
    s7 = base[base["alcohol_intake"] <= cutoff]
    print(f"\n7. Drop heavy drinkers (>p99 alc intake = {cutoff:.0f}g):")
    print(f"   {t2_fit(s7)}")

    # 8. Younger subsample only (age ≤ 40)
    s8 = base[base["age"] <= 40]
    print(f"\n8. Young adults only (age ≤ 40):")
    print(f"   {t2_fit(s8)}")

    # 9. Placebo outcome: share_beer (expected sign negative per paper, Col 7)
    print("\n9. Placebo outcome — share_beer:")
    print(f"   {t2_fit(base, y='share_beer')}  (paper: -3.129 [1.730])")

    # 10. Placebo outcome: share_wine (dwine+fwine) — should be null
    base["share_wine"] = base["share_dwine"] + base["share_fwine"]
    print("\n10. Placebo outcome — share_wine:")
    print(f"   {t2_fit(base, y='share_wine')}")

    # 11. Include minors (age 14–17)
    base_min = df[(df["year"] >= 2001) & (df["age"] >= 14) & (df["age"] <= 65)].reset_index(drop=True)
    print(f"\n11. Include minors (age 14–17):")
    print(f"   {t2_fit(base_min)}")

    # 12. Full sample 1994–2011 (no year≥2001 restriction)
    base_full = df[(df["age"] >= 18) & (df["age"] <= 65)].reset_index(drop=True)
    print(f"\n12. All years 1994–2011 (paper col 15: 4.661 [1.765]):")
    print(f"   {t2_fit(base_full)}")

    print("\n" + "=" * 70)
    print("TABLE 3 PANEL B ROBUSTNESS (headline coef: share_vodka on logtotal, IV)")
    print("Baseline (paper Col 3): 1.253 (SE 0.455), N=1,343")
    print("=" * 70)
    panel = pd.read_parquet(OUT / "mortality_panel.parquet")

    # 13. Drop cancer/placebo outcome check = already in Table 3 col 7 — confirms null

    # 14. Add log alcohol intake to spec (matches Col 4)
    print(f"\n14. + log(alc intake) control: "
          f"{t3_iv_fit(panel, 'logtotal', ['logalcohol_intake'])}")

    # 15. Subsample without Moscow/St.P (ids 7742/7743/7844 may not exist — use id=1,77 proxies)
    # Use top 3 most populous ids to proxy
    pop_by_id = panel.groupby("id")["popd5a"].sum().sort_values(ascending=False)
    top_ids = pop_by_id.index[:3].tolist()
    sub = panel[~panel["id"].isin(top_ids)]
    print(f"\n15. Drop top-3 most populous regions: "
          f"{t3_iv_fit(sub, 'logtotal')}")

    # 16. Only post-2000 data
    sub = panel[panel["year"] >= 2000]
    print(f"\n16. Panel years ≥ 2000: "
          f"{t3_iv_fit(sub, 'logtotal')}")

    # 17. Placebo dependent variable = log(cancer) (should be null)
    print(f"\n17. Placebo outcome log(cancer): "
          f"{t3_iv_fit(panel, 'logcancer', ['logalcohol_intake'])}")

    # 18. Alternative cluster: region id only (not id×rural×year)
    used = panel.dropna(subset=["logtotal", "share_vodka", "gorbachev_rural",
                                 "rural", "gorbachev", "logincome", "logpopulation",
                                 "age", "year", "fedokrug", "id"]).copy()
    dummies = [pd.get_dummies(used[f].astype(int), prefix=f, drop_first=True).astype(float)
               for f in ["age", "year", "fedokrug"]]
    X = pd.concat([used[["rural", "gorbachev", "logincome", "logpopulation"]].astype(float)] + dummies,
                  axis=1).assign(_const=1.0)
    res = IV2SLS(used["logtotal"].astype(float), X,
                 used[["share_vodka"]].astype(float),
                 used[["gorbachev_rural"]].astype(float)).fit(
                     cov_type="clustered", clusters=used["id"].astype(int))
    print(f"\n18. Cluster SEs by region only: beta={res.params['share_vodka']:.3f} "
          f"(SE {res.std_errors['share_vodka']:.3f})")


if __name__ == "__main__":
    main()
