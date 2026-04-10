"""Reproduce Table 2 of Kueng & Yakovlev (2020).

Replicates the Stata pipeline in Code/02_Gorbachev.do. The Stata code runs
`reghdfe y rural_gorbachev gorbachev rural [controls], absorb(id round age) vce(cluster identificator)`
on the sample `year>=2001 & 18<=age<=65`. Column 4 (all controls) defines the
e(sample) that all other columns then use.

We use linearmodels.AbsorbingLS for the HDFE equivalent and cluster SEs on
`identificator` (RLMS household/individual panel id).
"""
import numpy as np
import pandas as pd
from linearmodels.iv.absorbing import AbsorbingLS
from utils import CONTROLS, load_base_sample, table2_sample, OUT


def run_spec(df, y, controls, name):
    base_reg = ["rural_gorbachev", "gorbachev", "rural"]
    regs = base_reg + controls
    # Drop absorbed FE rows with NaN on any variable
    used = df[[y] + regs + ["identificator", "id", "round", "age"]].dropna()
    used = used.copy()
    # Use linearmodels AbsorbingLS
    exog = used[regs]
    exog = exog.assign(_const=1.0)
    dep = used[y]
    absorb = used[["id", "round", "age"]].astype("category")
    mod = AbsorbingLS(dep, exog, absorb=absorb, drop_absorbed=True)
    res = mod.fit(cov_type="clustered", clusters=used["identificator"].astype(int))
    return {
        "name": name,
        "N": int(res.nobs),
        "beta_DD": res.params["rural_gorbachev"],
        "se_DD": res.std_errors["rural_gorbachev"],
        "beta_D": res.params.get("gorbachev", np.nan),
        "se_D": res.std_errors.get("gorbachev", np.nan),
        "r2": res.rsquared,
    }


def main():
    df = load_base_sample()
    t2 = table2_sample(df)

    # Column 4 full specification first → e(sample) tracker
    full_vars = ["share_vodka"] + ["rural_gorbachev", "gorbachev", "rural"] + CONTROLS
    mask = t2[full_vars].notna().all(axis=1)
    t2["e_sample"] = mask.astype(int)
    sample_c4 = t2[t2["e_sample"] == 1].copy()
    print(f"Column 4 sample N = {len(sample_c4):,} (target 29,083)")

    specs = [
        ("Col 1 — only FEs", "share_vodka", []),
        ("Col 2 — + alcohol_intake", "share_vodka", ["alcohol_intake"]),
        ("Col 3 — + income/price", "share_vodka",
            ["alcohol_intake", "price_beer_to_vodka", "logincome", "logincome_missing"]),
        ("Col 4 — all controls", "share_vodka", CONTROLS),
    ]
    results = []
    for name, y, c in specs:
        # Cols 1–3 use the column-4 sample, col 4 uses its own
        sample = sample_c4 if "Col 4" not in name else t2
        r = run_spec(sample, y, c, name)
        results.append(r)

    # Column 5: log(alcohol_intake)*100, winsorized at 586 (95th percentile)
    t2c5 = t2.copy()
    t2c5["ln_alcohol_intake"] = np.log(t2c5["alcohol_intake"]) * 100
    t2c5["ln_alcohol_intake_"] = t2c5["ln_alcohol_intake"].clip(upper=586)
    c5_controls = [c for c in CONTROLS if c != "alcohol_intake"]
    r5 = run_spec(t2c5, "ln_alcohol_intake_", c5_controls, "Col 5 — log alcohol (winsor)")
    results.append(r5)

    # Column 7/8: share_beer, share_hard_alcohol (vodka + samogon + other hard)
    # Col 7 per paper: "Share of beer" with alcohol_intake + demographics as controls
    r7 = run_spec(t2, "share_beer", CONTROLS, "Col 7 — Share of beer")
    results.append(r7)

    t2["share_hard"] = t2["share_vodka"] + t2["share_samogon"] + t2["share_other"]
    r8 = run_spec(t2, "share_hard", CONTROLS, "Col 8 — Share of hard alcohol")
    results.append(r8)

    out = pd.DataFrame(results)
    print("\n=== Replication of Table 2 ===")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    paper = pd.DataFrame({
        "name": [
            "Col 1 — only FEs",
            "Col 2 — + alcohol_intake",
            "Col 3 — + income/price",
            "Col 4 — all controls",
            "Col 5 — log alcohol (winsor)",
            "Col 7 — Share of beer",
            "Col 8 — Share of hard alcohol",
        ],
        "paper_beta": [5.243, 5.049, 5.008, 5.232, 7.594, -3.129, 3.027],
        "paper_se":   [2.016, 2.009, 1.998, 1.986, 4.585,  1.730, 1.780],
        "paper_N":    [29083, 29083, 29083, 29083, 29083, 29083, 29083],
    })
    cmp = out.merge(paper, on="name")
    cmp["beta_diff"] = cmp["beta_DD"] - cmp["paper_beta"]
    print("\n=== Side-by-side ===")
    print(cmp[["name", "beta_DD", "paper_beta", "beta_diff", "se_DD", "paper_se", "N", "paper_N"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    cmp.to_csv(OUT / "table2_replication.csv", index=False)


if __name__ == "__main__":
    main()
