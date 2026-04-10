"""Reproduce Table 3 Panel B (regional male mortality IV).

Translates Code/04_Mortality_campaign_IV_by_cause.do. Uses RLMS micro-data
(males + females) aggregated to region x year x 5-year age bin x rural, merged
to the 5-year regional mortality counts, and runs 2SLS of log mortality on
share of vodka, instrumented by I(became adolescent during campaign) x I(rural).
"""
import numpy as np
import pandas as pd
from linearmodels.iv import IV2SLS
from linearmodels.panel import PanelOLS
from utils import DATA, OUT


def build_micro_agg():
    male = pd.read_stata(DATA / "base_sample_aej.dta", convert_categoricals=False)
    fem = pd.read_stata(DATA / "female_sample_aej.dta", convert_categoricals=False)
    df = pd.concat([male, fem], ignore_index=True)

    # Age 22..65 and cohort window 1981..1996 (turned 17)
    df = df[(df["age"] >= 22) & (df["age"] <= 65)].copy()
    age17 = df["birthy"] + 17
    df = df[(age17 >= 1981) & (age17 <= 1996)].copy()

    # gorbachev is a "treatment intensity" coding: 1 during campaign, 2 after
    g = np.where((df["birthy"] + 17 >= 1986) & (df["birthy"] + 17 <= 1990), 1, 0)
    g = np.where(df["birthy"] + 17 > 1990, 2, g)
    df["gorbachev"] = g

    # urban_rlms: list of site IDs treated as urban (population ≥ 100k)
    urban_sites = {138, 141, 116, 45, 106, 66, 70, 9, 147, 92, 136, 71, 46,
                   105, 47, 84, 153, 146, 149, 155, 137, 154}
    df["urban_rlms"] = df["site"].isin(urban_sites).astype(int)

    df["alcohol_intake"] = df["alcohol_intake"] / 1000.0  # kg
    df["age"] = (df["age"] // 5) * 5

    share_cols = [c for c in df.columns if c.startswith("share_")]
    alco_cols = [c for c in df.columns if c.startswith("alcohol")]
    agg_cols = share_cols + alco_cols + ["logincome"]
    grp = df.groupby(["id", "urban_rlms", "year", "age"], dropna=False)
    out = grp[agg_cols].mean().reset_index()
    out["count"] = grp["share_vodka"].count().values
    out["rural"] = 1 - out["urban_rlms"]
    out = out.dropna(subset=["rural", "year", "age", "share_vodka"])
    return out


def build_panel():
    mort = pd.read_stata(DATA / "regional_mortality/5y_89_12.dta", convert_categoricals=False)
    mort = mort.rename(columns={"alco_poisoning": "alco_pois"})
    mort["birthstart"] = mort["year"] - mort["age"] - 4
    # Triangular kernel of treatment intensity
    gk = np.zeros(len(mort))
    bs17 = mort["birthstart"] + 17
    gk = np.where(bs17 == 1986, 1.0, gk)
    gk = np.where((bs17 == 1987) | (bs17 == 1985), 0.8, gk)
    gk = np.where((bs17 == 1988) | (bs17 == 1984), 0.6, gk)
    gk = np.where((bs17 == 1989) | (bs17 == 1983), 0.4, gk)
    gk = np.where((bs17 == 1990) | (bs17 == 1982), 0.2, gk)
    mort["gorbachev"] = gk
    mort = mort[(bs17 >= 1981) & (bs17 <= 1992)].copy()
    mort["rural"] = (mort["group"] == "R").astype(int)
    mort["gorbachev_rural"] = mort["gorbachev"] * mort["rural"]
    mort = mort[mort["year"] >= 1990].copy()

    alco = build_micro_agg()
    panel = mort.merge(
        alco.rename(columns={"urban_rlms": "urban_rlms_a"}),
        left_on=["id", "rural", "year", "age"],
        right_on=["id", alco["rural"].name if False else "rural", "year", "age"],
        how="left",
    )
    # simpler merge path
    panel = mort.merge(alco.drop(columns=["urban_rlms"]), on=["id", "rural", "year", "age"], how="left")
    panel = panel.dropna(subset=["share_vodka"])

    fedokrug = pd.read_stata(DATA / "id_fedokrug.dta", convert_categoricals=False)
    panel = panel.merge(fedokrug, on="id", how="left")
    panel.loc[(panel["id"] == 86) & (panel["fedokrug"].isna()), "fedokrug"] = 1005

    # Drop muslim regions per code comment
    panel = panel[~panel["id"].isin([7, 16])]
    panel = panel.dropna(subset=["total"])

    pop = panel.groupby(["id", "year"], as_index=False)["popd5a"].sum().rename(columns={"popd5a": "population"})
    panel = panel.merge(pop, on=["id", "year"])

    for col in ["total", "alco_pois", "ext_causes", "cancer"]:
        panel[f"log{col}"] = np.log(panel[col].replace(0, np.nan)) * 100
    panel["logpopulation"] = np.log(panel["population"]) * 100
    panel["logalcohol_intake"] = np.log(panel["alcohol_intake"]) * 100
    panel["logincome"] = panel["logincome"] * 100
    panel["id_rural_year"] = (panel["id"].astype(int) * 10000
                              + panel["rural"].astype(int) * 100
                              + (panel["year"].astype(int) - 1990))
    return panel


def iv_spec(df, y, endog, instr, controls, fes, cluster):
    used = df.dropna(subset=[y, endog, instr] + controls + fes + [cluster]).copy()
    # Build fixed-effect dummies for age, year, fedokrug
    dummies = []
    for f in fes:
        d = pd.get_dummies(used[f].astype(int), prefix=f, drop_first=True).astype(float)
        dummies.append(d)
    X = pd.concat([used[controls].astype(float)] + dummies, axis=1)
    X = X.assign(_const=1.0)
    dep = used[y].astype(float)
    endog_df = used[[endog]].astype(float)
    instr_df = used[[instr]].astype(float)
    res = IV2SLS(dep, X, endog_df, instr_df).fit(cov_type="clustered",
                                                  clusters=used[cluster].astype(int))
    return res, len(used)


def main():
    panel = build_panel()
    print(f"Panel N before IV: {len(panel):,}")
    panel.to_parquet(OUT / "mortality_panel.parquet", index=False)

    controls = ["rural", "gorbachev", "logincome", "logpopulation"]
    fes = ["age", "year", "fedokrug"]

    # Column 3: logtotal on share_vodka (IV, no logalcohol) → paper: 1.456***, SE 0.251
    res_tot, n_tot = iv_spec(panel, "logtotal", "share_vodka", "gorbachev_rural", controls, fes, "id_rural_year")
    # Column 4: + logalcohol → 1.454***, 0.200
    res4, n4 = iv_spec(panel, "logtotal", "share_vodka", "gorbachev_rural",
                        controls + ["logalcohol_intake"], fes, "id_rural_year")
    # Column 5: alco_pois → 1.253***, 0.455
    res5, n5 = iv_spec(panel, "logalco_pois", "share_vodka", "gorbachev_rural",
                        controls + ["logalcohol_intake"], fes, "id_rural_year")
    # Column 6: ext_causes → 1.271***, 0.473
    res6, n6 = iv_spec(panel, "logext_causes", "share_vodka", "gorbachev_rural",
                        controls + ["logalcohol_intake"], fes, "id_rural_year")
    # Column 7: cancer placebo → 3.836**, 1.532 (from paper)
    res7, n7 = iv_spec(panel, "logcancer", "share_vodka", "gorbachev_rural",
                        controls + ["logalcohol_intake"], fes, "id_rural_year")

    # Paper Table 3 Panel B column mapping (cols 1-2 are national OLS N=44):
    #   Col 3 logtotal IV (no alcohol ctrl): 1.253 [0.455], N=1,343
    #   Col 4 logtotal IV (+ alcohol ctrl):  1.271 [0.473], N=1,343
    #   Col 5 alco_pois IV:                   3.836 [1.532], N=1,327
    #   Col 6 ext_causes IV:                  1.230 [0.523], N=1,343
    #   Col 7 cancer IV (placebo):           -0.190 [1.225], N=1,273
    rows = []
    for name, res, n, paper_b, paper_se in [
        ("Col 3 logtotal (no alc)", res_tot, n_tot, 1.253, 0.455),
        ("Col 4 logtotal (+ alc)",   res4,    n4,    1.271, 0.473),
        ("Col 5 alco_poisoning",     res5,    n5,    3.836, 1.532),
        ("Col 6 ext_causes",         res6,    n6,    1.230, 0.523),
        ("Col 7 cancer (placebo)",   res7,    n7,   -0.190, 1.225),
    ]:
        rows.append({
            "name": name,
            "N": n,
            "beta_vodka": res.params["share_vodka"],
            "se_vodka": res.std_errors["share_vodka"],
            "paper_beta": paper_b,
            "paper_se": paper_se,
        })
    out = pd.DataFrame(rows)
    print("\n=== Replication of Table 3 Panel B ===")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    out.to_csv(OUT / "table3_replication.csv", index=False)


if __name__ == "__main__":
    main()
