"""Replicate 02_calibrate/02_rd_efficiency.do: compute R&D efficiency b_s by country.

Equation (46) in Sampson (2023): b_s = (1/N_s) * sum_{t,j} log(RD_{jst} / RD_{js_bar t})
Reference country s_bar = Germany (DEU) in the .do file; normalisation sets b_USA = 0.

Outputs:
  - replication_182522/output/rd_efficiency.csv  (cou, rnd_eff, rnd_eff_m, rnd_eff_fix)
  - table of R&D efficiency values printed to stdout
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils import OUT, BASELINE_COUS, COU_NAMES, compute_rd_intensity


def main():
    m = compute_rd_intensity()
    print(f"Merged R&D intensity obs (all years, all countries): {len(m):,}")

    # Section B of 02_rd_efficiency.do: restrict to 2010-2014 country-years with
    # observations for >= 14 of the 20 manufacturing industries.
    m["cou_obs_man"] = m.groupby(["cou", "year"])["cou"].transform("count")
    s = m[(m["cou_obs_man"] >= 14) & (m["year"].between(2010, 2014))].copy()
    print(f"After 2010-14 filter with >=14 industries: {len(s):,} obs")

    # Compute Germany reference log R&D intensity within (ind, year)
    deu = (
        s[s["cou"] == "DEU"][["ind", "year", "lrd_intensity"]]
        .rename(columns={"lrd_intensity": "lrdint_deu"})
    )
    s = s.merge(deu, on=["ind", "year"], how="left")
    s["rel_lrdint"] = s["lrd_intensity"] - s["lrdint_deu"]

    # Drop observations where Germany reference is missing (can't form relative)
    s = s[s["rel_lrdint"].notna()].copy()

    # rnd_diff: mean across (ind, year) within country
    rnd_diff = s.groupby("cou")["rel_lrdint"].mean().rename("rnd_diff")
    # rnd_diff_m: median across (ind, year) within country
    rnd_diff_m = s.groupby("cou")["rel_lrdint"].median().rename("rnd_diff_m")

    out = pd.concat([rnd_diff, rnd_diff_m], axis=1).reset_index()

    # Restrict to baseline 25 OECD countries
    out = out[out["cou"].isin(BASELINE_COUS)].copy()

    # Normalise: subtract USA value so that b_USA = 0
    usa_mean = float(out.loc[out["cou"] == "USA", "rnd_diff"].iloc[0])
    usa_med = float(out.loc[out["cou"] == "USA", "rnd_diff_m"].iloc[0])
    out["rnd_eff"] = out["rnd_diff"] - usa_mean
    out["rnd_eff_m"] = out["rnd_diff_m"] - usa_med

    out["country"] = out["cou"].map(COU_NAMES)
    out = out.sort_values("rnd_eff_m", ascending=False).reset_index(drop=True)

    path = OUT / "rd_efficiency.csv"
    out[["cou", "country", "rnd_diff", "rnd_diff_m", "rnd_eff", "rnd_eff_m"]].to_csv(
        path, index=False
    )
    print(f"\nSaved {path}")

    print("\nR&D efficiency by country (25 OECD, normalised USA=0)")
    print("=" * 66)
    print(f"{'cou':<4}{'country':<20}{'rnd_eff (mean)':>18}{'rnd_eff_m (median)':>22}")
    print("-" * 66)
    for _, r in out.iterrows():
        print(f"{r['cou']:<4}{r['country']:<20}{r['rnd_eff']:>18.4f}{r['rnd_eff_m']:>22.4f}")

    # Summary stats for comparison with Figure 2 of the paper
    print("\nSummary stats on b_s (rnd_eff_m, USA=0):")
    d = out["rnd_eff_m"]
    print(f"  min = {d.min():.3f}   ({out.loc[d.idxmin(), 'cou']})")
    print(f"  max = {d.max():.3f}   ({out.loc[d.idxmax(), 'cou']})")
    print(f"  mean = {d.mean():.3f}")
    print(f"  sd = {d.std(ddof=1):.3f}")
    print(f"  N countries = {len(out)}")


if __name__ == "__main__":
    main()
